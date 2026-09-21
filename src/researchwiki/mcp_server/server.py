"""wiki MCP server（FastMCP）：把 wiki 的检索/读取/写入暴露成 MCP 工具。

启动方式::

    uv run python -m researchwiki.mcp_server            # stdio（MCP 客户端默认）
    uv run python -m researchwiki.mcp_server --transport http --port 8765

工具一览（描述里写清了"什么时候该用"，因为这是给模型看的）：
- wiki_search       检索已沉淀的笔记（回答前先查，避免重复研究）
- wiki_read         读一条笔记全文，merged/superseded 自动跟随重定向
- wiki_write        新增一条原子笔记（schema 校验 + 沙箱路径 + 写前备份）
- wiki_list_changes 最近变更列表，供客户端增量同步
- wiki_health       wiki 自检（笔记/页面/冲突计数 + 索引状态）

本模块只做"薄适配"：参数校验、写保护、索引进同步全在 service.WikiService，
所以逻辑测试直接打服务层，协议层另用 FastMCP 的 in-process Client 覆盖。
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from researchwiki import __version__
from researchwiki.mcp_server.service import WikiService, error_payload

logger = logging.getLogger("researchwiki.mcp_server")

DEFAULT_WIKI_ROOT = "wiki-data"
SERVER_NAME = "researchwiki"
SERVER_INSTRUCTIONS = (
    "研究 wiki 服务：一个跨会话累积的研究知识库。\n"
    "回答一个新问题前，先用 wiki_search 查一下 wiki 里是否已有沉淀的结论；"
    "命中后用 wiki_read 读全文（含来源 URL）。\n"
    "研究出新的、可复用的事实后，用 wiki_write 沉淀成原子笔记（一条笔记一个事实，"
    "尽量带 sources）。\n"
    "需要跟客户端本地缓存做增量同步时用 wiki_list_changes；想确认服务状态用 wiki_health。\n"
    "所有工具都返回 JSON：成功为 {\"ok\": true, ...}，失败为 "
    "{\"ok\": false, \"error\": {\"code\", \"message\", \"details\"?}}，"
    "不会抛协议异常——请检查 ok 字段而不是等报错。"
)

_DESC_SEARCH = (
    "在本地研究 wiki 里做混合检索（FTS5 关键词 + 向量语义 + RRF 融合，含置信度与新鲜度加权），"
    "返回最相关的原子笔记摘要。\n"
    "什么时候用：回答任何研究问题之前——wiki 里可能已经沉淀过答案，先查再决定要不要重新研究；"
    "也可以在需要为某个实体/主题收集既有结论时用。\n"
    "参数：query 为自然语言或关键词（中文可直接用，短词也能召回）；"
    "k 为返回条数（1..50，默认 5）。\n"
    "返回：{ok, query, k, count, results:[{note_id, title, snippet, score, match_type,"
    " redirected_from, redirect_note}]}。match_type 为 fts（关键词命中）/vector（语义命中）"
    "/both；redirected_from 非空表示命中了一条已并入他处的旧笔记，结果内容来自 note_id，"
    "redirect_note 里有人话说明。\n"
    "命中后请用 wiki_read 读全文再引用，不要只凭 snippet 下结论。"
)

_DESC_READ = (
    "读取一条原子笔记的全文（frontmatter 关键字段 + 正文 + 来源 URL）。\n"
    "什么时候用：wiki_search 命中后核实原文；已知笔记 ID 时直接读；"
    "引用 wiki 结论前确认其置信度 confidence、新鲜度 volatility 与 sources。\n"
    "参数：note_id 形如 N-0001（也接受 notes/N-0001.md 这种写法）。\n"
    "重定向：笔记被合并/取代时（status 为 merged/superseded）自动沿 redirect_to/"
    "superseded_by 跟到最终笔记，响应里 redirected=true、source_note 给出原始 ID、"
    "redirect_chain 给出完整来源链——引用时请引用最终 note_id，并可以提一句它由哪条合并而来。\n"
    "返回：{ok, requested_id, redirected, note:{note_id, title, status, confidence, volatility,"
    " entities, created, observed_at, reviewed_at, trace_id, sources, redirect_to,"
    " superseded_by, extra}, body, path}。\n"
    "失败：笔记不存在返回 {ok:false, error:{code:'not_found'}}（不是异常），"
    "ID 形态非法或越界返回 code='path_rejected'。"
)

_DESC_WRITE = (
    "新增一条原子笔记到 wiki（一条笔记只写一个事实/结论，正文自洽可独立阅读）。\n"
    "什么时候用：研究得出可复用的结论、需要跨会话记住的事实时。写之前先用 wiki_search "
    "查重，避免同一事实写多条；要修正已有笔记时先 wiki_read 再决定是否新增（本工具只新增，"
    "不覆盖既有笔记）。\n"
    "参数：body 正文（必填，Markdown，可用 [[entity:slug|显示名]] 双链）；title 标题（必填）；"
    "entities 实体 slug 列表，如 ['glm-5-3','上下文压缩']（不要传逗号分隔的字符串）；"
    "confidence 置信度 high/medium/low；volatility 时效性 stable（不衰减）/drifting（90 天半衰）"
    "/volatile（30 天半衰）；sources 来源 URL 字符串列表（强烈建议填，"
    "wiki 靠来源 URL 保证据链）。\n"
    "写保护：① frontmatter schema 校验（confidence/volatility 白名单、title 非空、"
    "entities 必须是字符串列表等，非法一律拒绝并给出原因）；② 路径限制在 wiki-data 沙箱内；"
    "③ 写前自动把可能被覆盖的原文件备份到 wiki-data/.backups/<时间戳>/，备份失败则拒绝写入。\n"
    "返回：{ok, note_id, title, status, created, path, backup:{dir, action, backup_file},"
    " indexed, warnings?}——note_id 是新建笔记的稳定 ID，后续引用/链接用它。\n"
    "失败：{ok:false, error:{code:'validation_failed'|'backup_failed'|'path_rejected',"
    " message, details}}，message 就是可读的拒绝原因（中文），照它改参数重试即可。"
)

_DESC_LIST_CHANGES = (
    "列出最近变更的笔记（按 frontmatter 的 created/reviewed_at 倒序，最新在前），"
    "用于 MCP 客户端做增量同步，或想快速了解「上次同步后 wiki 里多了什么」。\n"
    "什么时候用：客户端有本地缓存需要对齐；或想快速了解 wiki 里最近沉淀了什么。\n"
    "参数：since 为 ISO 时间字符串（如 2026-09-20T12:00:00+00:00），只返回变更时间"
    "严格晚于它的笔记（排他语义；传 None 返回全部）；limit 为返回条数上限（1..200，默认 50）。\n"
    "翻页：结果超过 limit 时——没给 since 就返回最新的一页（has_more 说明还有更早的没回）；"
    "给了 since 说明是增量同步，返回最早的一页并给出 next_since 游标，"
    "把 next_since 当下次的 since 继续拉，直到 has_more=false 就完整同步、不漏条目。\n"
    "返回：{ok, since, total, count, has_more, next_since, latest, changes:[{note_id, title,"
    " status, change_kind, confidence, volatility, entities, created, reviewed_at, updated,"
    " redirect_to, superseded_by, path}]}。change_kind 为 created（新建）/reviewed（复核更新）"
    "/status（合并或取代等状态变更）。"
)

_DESC_HEALTH = (
    "wiki 自检：返回笔记（总数/active/merged/superseded）、聚合页、实体、冲突台账的计数，"
    "以及检索索引的状态（是否存在、已索引条数、tokenizer、是否落后于 md 文件）。\n"
    "什么时候用：确认服务指向的 wiki 目录对不对、排查「检索不到刚写的笔记」这类问题、"
    "或汇报当前知识库规模时。\n"
    "返回：{ok, root, notes:{total, active, merged, superseded}, pages, entities,"
    " conflicts:{total, open, resolved}, index:{...}}。"
)


def resolve_wiki_root(
    root: str | Path | None = None, *, config: Mapping[str, Any] | None = None
) -> Path:
    """确定 wiki-data 根目录：显式参数 > 环境变量 > config [server].wiki_data_dir > 默认值。"""
    if root is not None:
        return Path(root)
    env_root = os.environ.get("RESEARCHWIKI_WIKI_DATA")
    if env_root:
        return Path(env_root)
    server_cfg = (config or {}).get("server")
    if isinstance(server_cfg, Mapping) and server_cfg.get("wiki_data_dir"):
        return Path(str(server_cfg["wiki_data_dir"]))
    return Path(DEFAULT_WIKI_ROOT)


def _call(fn: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """执行服务层方法：未预期异常记日志并转成结构化 internal 错误（不向客户端抛堆栈）。"""
    try:
        return fn(*args, **kwargs)  # type: ignore[no-any-return]
    except Exception as exc:  # noqa: BLE001 -- 服务器边界，任何异常都要变成结构化错误
        logger.exception("MCP 工具执行失败：%s", getattr(fn, "__name__", fn))
        return error_payload("internal", f"{type(exc).__name__}: {exc}")


def build_server(
    *, root: str | Path | None = None, config: Mapping[str, Any] | None = None
) -> FastMCP:
    """装配 wiki MCP server（root/config 可显式注入，测试用 tmp_path 隔离）。"""
    resolved_root = resolve_wiki_root(root, config=config)
    service = WikiService(resolved_root, config=config)
    mcp: FastMCP = FastMCP(
        name=SERVER_NAME,
        version=__version__,
        instructions=SERVER_INSTRUCTIONS,
        mask_error_details=True,  # 意外异常不回堆栈细节，详细原因留在服务端日志
    )

    @mcp.tool(
        name="wiki_search",
        description=_DESC_SEARCH,
        annotations={"readOnlyHint": True, "openWorldHint": False},
    )
    def wiki_search(query: str, k: int = 5) -> dict[str, Any]:
        """检索本地研究 wiki（FTS5 + 向量混合检索），返回相关笔记摘要列表。"""
        return _call(service.search, query, k)

    @mcp.tool(
        name="wiki_read",
        description=_DESC_READ,
        annotations={"readOnlyHint": True, "openWorldHint": False},
    )
    def wiki_read(note_id: str) -> dict[str, Any]:
        """读一条笔记全文；merged/superseded 自动跟随重定向并说明来源。"""
        return _call(service.read, note_id)

    @mcp.tool(
        name="wiki_write",
        description=_DESC_WRITE,
        annotations={
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": False,
        },
    )
    def wiki_write(
        body: str,
        *,
        title: str = "",
        entities: list[str] | None = None,
        confidence: str = "medium",
        volatility: str = "stable",
        sources: list[str] | None = None,
    ) -> dict[str, Any]:
        """新增一条原子笔记：schema 校验 + 沙箱路径限制 + 写前备份 + 索引同步。"""
        return _call(
            service.write,
            body,
            title=title,
            entities=entities,
            confidence=confidence,
            volatility=volatility,
            sources=sources,
        )

    @mcp.tool(
        name="wiki_list_changes",
        description=_DESC_LIST_CHANGES,
        annotations={"readOnlyHint": True, "openWorldHint": False},
    )
    def wiki_list_changes(since: str | None = None, limit: int = 50) -> dict[str, Any]:
        """列出最近变更的笔记（含新建与状态变更），供客户端增量同步。"""
        return _call(service.list_changes, since, limit)

    @mcp.tool(
        name="wiki_health",
        description=_DESC_HEALTH,
        annotations={"readOnlyHint": True, "openWorldHint": False},
    )
    def wiki_health() -> dict[str, Any]:
        """wiki 自检：笔记/页面/实体/冲突计数 + 索引状态。"""
        return _call(service.health)

    return mcp


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m researchwiki.mcp_server", description="wiki MCP server（FastMCP）"
    )
    parser.add_argument(
        "--root",
        default=None,
        help="wiki-data 目录（默认取 RESEARCHWIKI_WIKI_DATA 环境变量，再取 config.toml "
        "[server].wiki_data_dir，最后回落 wiki-data）",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="config.toml 路径（默认取 RESEARCHWIKI_CONFIG 环境变量，再回落 ./config.toml）",
    )
    parser.add_argument(
        "--transport",
        default="stdio",
        choices=["stdio", "http", "sse", "streamable-http"],
        help="MCP 传输方式，默认 stdio（Claude Code 等本地客户端用这个）",
    )
    parser.add_argument("--host", default="127.0.0.1", help="http/sse 传输的监听地址")
    parser.add_argument("--port", type=int, default=8765, help="http/sse 传输的监听端口")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """`python -m researchwiki.mcp_server` 的入口。"""
    args = _build_parser().parse_args(argv)
    # config.toml 的读取复用 server.main.load_config（含 RESEARCHWIKI_CONFIG 环境变量），
    # 这里按需 import，避免只为读一个配置就拉起 FastAPI/LLM 依赖栈
    from researchwiki.server.main import load_config

    config = load_config()
    server = build_server(root=args.root, config=config)
    resolved_root = resolve_wiki_root(args.root, config=config)
    print(
        f"[researchwiki] wiki MCP server 启动：root={resolved_root} transport={args.transport}",
        file=sys.stderr,
    )
    kwargs: dict[str, Any] = {}
    if args.transport != "stdio":
        kwargs = {"host": args.host, "port": args.port}
    # show_banner=False：stdio 传输下 stdout 只能走协议报文，横幅关掉最稳妥
    server.run(transport=args.transport, show_banner=False, **kwargs)
    return 0
