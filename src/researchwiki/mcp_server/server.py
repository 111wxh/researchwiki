"""wiki MCP server（FastMCP）：把 wiki 的检索/读取/写入暴露成 MCP 工具。

启动方式::

    uv run python -m researchwiki.mcp_server            # stdio（MCP 客户端默认）
    uv run python -m researchwiki.mcp_server --transport http --port 8765

工具一览（描述里写清了"什么时候该用"，因为这是给模型看的）：
- wiki_*（兼容层，五个）：
  - wiki_search       检索已沉淀的笔记（回答前先查，避免重复研究）
  - wiki_read         读一条笔记全文，merged/superseded 自动跟随重定向
  - wiki_write        新增一条原子笔记（schema 校验 + 沙箱路径 + 写前备份）
  - wiki_list_changes 最近变更列表，供客户端增量同步
  - wiki_health       wiki 自检（笔记/页面/冲突计数 + 索引状态）
- memory_*（记忆接口，九个）：把 wiki 当作 Agent 的外置时间感知记忆使用
  - memory_store      显式写入一条记忆（带 kind/importance）
  - memory_search     语义检索记忆（可按 kind 过滤）
  - memory_recall     动态召回入口（结果带 status/observed_at/kind 标注）
  - memory_update     修订既有记忆正文（必须留 reason，自动备份）
  - memory_supersede  用新内容替代旧记忆（新旧 ID 沿链可达）
  - memory_invalidate 判定记忆失效（自动生成墓碑笔记，保持链可达）
  - memory_timeline   单条记忆的版本史
  - memory_conflicts  冲突台账查询
  - memory_profile    User Memory 视图（kind=user 的 active 记忆）

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
    "研究 wiki 服务：一个跨会话累积的研究知识库，也是 Agent 的外置时间感知记忆。\n"
    "回答一个新问题前，先用 wiki_search 查一下 wiki 里是否已有沉淀的结论；"
    "命中后用 wiki_read 读全文（含来源 URL）。\n"
    "研究出新的、可复用的事实后，用 wiki_write 沉淀成原子笔记（一条笔记一个事实，"
    "尽量带 sources）。\n"
    "需要跟客户端本地缓存做增量同步时用 wiki_list_changes；想确认服务状态用 wiki_health。\n"
    "\n"
    "记忆接口 memory_*（推荐优先用）：\n"
    "- 主动记住一条事实/用户偏好/经验教训：memory_store"
    "（kind: knowledge 世界知识 / user 用户画像 / experience 经验教训，"
    "可标 importance 0.0–1.0）。\n"
    "- 检索记忆：memory_search（可按 kind 过滤）；或用 memory_recall 动态召回"
    "（结果带 status/observed_at/kind 标注，方便按记忆状态取舍）。\n"
    "- 修正已有记忆：小修正文用 memory_update（必须留 reason，自动备份原稿）；"
    "内容已过时且有新结论用 memory_supersede（新笔记替代，旧 ID 沿链可达新 ID）；"
    "判定失效且没有替代内容用 memory_invalidate（自动生成墓碑笔记，旧记忆沿链可达墓碑）。\n"
    "- 别直接改写或忽视过时记忆：修订/取代/失效都会留 reason 与备份，"
    "这是记忆系统的审计底线。\n"
    "- 查单条记忆的版本史用 memory_timeline；查冲突台账用 memory_conflicts；"
    "查用户画像（kind=user 的 active 记忆）用 memory_profile。\n"
    "\n"
    "何时仍用 wiki_*：通用的检索/读取/写入/增量同步/健康检查场景两者皆可；"
    "wiki_* 是稳定兼容层（签名不变），需要 kind/importance 等记忆语义时用 memory_*。\n"
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

_DESC_MEMORY_STORE = (
    "显式写入一条记忆（Agent/用户主动存一条要跨会话记住的内容）。\n"
    "什么时候用：用户告诉了你值得长期记住的信息（偏好、约束、背景、教训、新事实），"
    "或你自己得出了可复用的结论时。写之前先 memory_search 查重，避免同一事实存多条。\n"
    "参数：content 记忆正文（必填，一条记忆一个事实）；kind 记忆类型"
    " knowledge（世界知识）/ user（用户画像与偏好）/ experience（经验教训），默认 knowledge；"
    "importance 主观重要性 0.0–1.0（可选）；entities 相关实体 slug 列表；"
    "source_urls 来源 URL 列表（事实类记忆强烈建议填）；confidence high/medium/low；"
    "volatility stable/drifting/volatile（时效性）。\n"
    "标题不需要传：从 content 首行派生；要精确控制标题走 wiki_write。\n"
    "返回：{ok, note_id, kind, importance, title, path, backup, indexed, warnings?}。"
    "失败：{ok:false, error:{code:'invalid_kind'|'validation_failed'|...}}，"
    "message 是可读的中文原因，照它改参数重试即可。"
)

_DESC_MEMORY_SEARCH = (
    "在记忆库里做混合检索（FTS5 + 向量 + RRF 融合），可按记忆类型过滤。\n"
    "什么时候用：回答问题前查相关记忆；需要按类型取记忆时（如只看 user 偏好、"
    "只看 experience 教训）。\n"
    "参数：query 检索词（必填）；k 返回条数（1..50，默认 5）；"
    "kind 可选过滤 knowledge/user/experience，不传则不过滤。\n"
    "返回：{ok, query, k, count, results:[{note_id, title, snippet, score, match_type,"
    " redirected_from, redirect_note}]}。kind 非法返回 code='invalid_kind'。\n"
    "命中后建议用 wiki_read 或 memory_recall 读全文与记忆状态，不要只凭 snippet 下结论。"
)

_DESC_MEMORY_RECALL = (
    "动态召回入口：检索一批与 query 相关的记忆，并为每条标注记忆状态。\n"
    "什么时候用：会话开始恢复上下文、回答前召回相关记忆、需要根据记忆的"
    "时效（observed_at/status）与类型（kind）取舍内容时。\n"
    "参数：query 检索词（必填）；k 返回条数（默认 5）；kind 可选过滤。\n"
    "返回：{ok, mode:'passthrough', results:[{note_id, title, snippet, score, ...,"
    " status, observed_at, kind, importance}]}——status/observed_at/kind 告诉你这条记忆"
    "是否现役、何时观察到、属于哪类；merged/superseded 记忆自动跟随重定向到当前版本。\n"
    "注意：当前为结构化透传（P3 将升级为 budget-aware 动态召回，按 token 预算与"
    "重要性挑选记忆），请自行根据标注裁剪。"
)

_DESC_MEMORY_UPDATE = (
    "修订一条既有记忆的正文（不新增 ID，原地更新）。\n"
    "什么时候用：记忆内容基本正确但需要补充/修正细节时。如果旧记忆已整体过时、"
    "应改用 memory_supersede 或 memory_invalidate，而不是硬改。\n"
    "参数：note_id 记忆 ID（形如 N-0001，必须是 active 状态）；content 修订后的完整正文"
    "（覆盖原正文）；reason 修订原因（必填）。\n"
    "安全保障：写前自动备份原稿到 wiki-data/.backups/；reason 追加进 frontmatter 的"
    " update_reasons 列表（按修订顺序保留全部原因，含时间戳）并刷新 reviewed_at"
    "——禁止静默覆盖，审计留痕是硬要求。\n"
    "返回：{ok, note_id, reason, reviewed_at, backup, indexed, path}。"
    "失败：code='not_found'（记忆不存在）/ 'invalid_argument'（非 active，先跟随重定向）/"
    "'validation_failed'（content 或 reason 为空）。"
)

_DESC_MEMORY_SUPERSEDE = (
    "用新内容替代一条旧记忆：先写入新记忆，再把旧记忆标记为 superseded 并链到新 ID。\n"
    "什么时候用：旧记忆的结论已过时/被发现错误，你有了新的正确内容时"
    "（如「X 支持 128k」→「X 支持 200k」）。\n"
    "参数：note_id 旧记忆 ID（必须是 active）；new_content 新内容正文（必填）；"
    "reason 替代原因（必填）。\n"
    "语义：新记忆继承旧记忆的 entities/kind/confidence/volatility/importance；"
    "旧记忆 status=superseded、superseded_by=新 ID，之后读旧 ID 会自动跟随到新记忆；"
    "reason 记入旧记忆的 supersede_reason。\n"
    "返回：{ok, old_note_id, new_note_id, superseded_by, reason, backup, path}。"
    "失败：code='not_found' / 'invalid_argument' / 'validation_failed'。"
)

_DESC_MEMORY_INVALIDATE = (
    "判定一条记忆失效（没有替代内容，只是不再成立/不再适用）。\n"
    "什么时候用：发现某条记忆错了或失效、但又没有新内容可以替代时"
    "（例如「待确认」的传闻被证伪）。有替代内容请用 memory_supersede。\n"
    "参数：note_id 记忆 ID（必须是 active）；reason 失效原因（必填）。\n"
    "语义：自动生成一条墓碑笔记（kind=knowledge，正文=失效原因+时间戳，active 状态），"
    "旧记忆 status=superseded 且 superseded_by 指向墓碑——读旧 ID 会沿链到达墓碑，"
    "审计与「superseded 必须沿链可达 active」的不变量同时保住；"
    "reason 同时记入旧记忆的 invalidate_reason。\n"
    "返回：{ok, old_note_id, tombstone_id, superseded_by, reason, backup, path}。"
    "失败：code='not_found' / 'invalid_argument' / 'validation_failed'。"
)

_DESC_MEMORY_TIMELINE = (
    "查询单条记忆的完整版本史：沿 redirect 链收集所有版本（含更早被取代的版本），"
    "按旧 → 新输出有序事件列表。\n"
    "什么时候用：对某条记忆的演变有疑问时（它先后说过什么、何时被修订/取代/失效）；"
    "引用前核实记忆的来龙去脉。\n"
    "参数：note_id 记忆 ID（传链上任何一个版本的 ID 都行，会补齐整条链）。\n"
    "返回：{ok, note_id, current（当前 active 版本 ID）, direction:'oldest_first',"
    " count, events:[{note_id, title, kind, status, change_kind, created, reviewed_at,"
    " updated, superseded_by, redirect_to, path}]}（events 旧 → 新）。"
    "失败：code='not_found'（记忆不存在）/ 'redirect_cycle' / 'broken_redirect'。"
)

_DESC_MEMORY_CONFLICTS = (
    "查询冲突台账：同一实体的矛盾断言双方证据与裁决记录。\n"
    "什么时候用：回答前发现记忆之间可能矛盾、想看还有哪些未裁决的冲突、"
    "或汇报知识库一致性状态时。\n"
    "参数：status 过滤 open（默认，未裁决）/ resolved（已裁决）/ all（全部）。\n"
    "返回：{ok, status, count, conflicts:[{conflict_id, question, status, claim_a,"
    " claim_b, resolution, created, resolved_at}]}。\n"
    "发现新的矛盾时，当前版本可先用 wiki_write 记下双方证据并手工登记冲突文件；"
    "结构化的冲突登记工具在后续版本提供。"
)

_DESC_MEMORY_PROFILE = (
    "User Memory 视图：平铺列出所有 kind=user 的 active 记忆（用户画像、偏好、约束）。\n"
    "什么时候用：任务开始时快速了解用户是谁、有什么偏好与长期约束；"
    "个性化回复前核对已知偏好。\n"
    "参数：无。\n"
    "返回：{ok, count, memories:[{note_id, title, entities, confidence, importance,"
    " created, observed_at, snippet, path}]}（按 note_id 排序）。\n"
    "想补充用户画像用 memory_store(kind='user')；想看某条画像的完整正文用 wiki_read。"
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

    @mcp.tool(
        name="memory_store",
        description=_DESC_MEMORY_STORE,
        annotations={
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": False,
        },
    )
    def memory_store(
        content: str,
        *,
        kind: str = "knowledge",
        entities: list[str] | None = None,
        importance: float | None = None,
        source_urls: list[str] | None = None,
        confidence: str = "medium",
        volatility: str = "stable",
    ) -> dict[str, Any]:
        """显式写入一条记忆（kind/importance 可选），复用写保护三件套。"""
        return _call(
            service.store_memory,
            content,
            kind=kind,
            entities=entities,
            importance=importance,
            source_urls=source_urls,
            confidence=confidence,
            volatility=volatility,
        )

    @mcp.tool(
        name="memory_search",
        description=_DESC_MEMORY_SEARCH,
        annotations={"readOnlyHint": True, "openWorldHint": False},
    )
    def memory_search(query: str, k: int = 5, kind: str | None = None) -> dict[str, Any]:
        """语义检索记忆，可按 kind 过滤（knowledge/user/experience）。"""
        return _call(service.search, query, k, kind)

    @mcp.tool(
        name="memory_recall",
        description=_DESC_MEMORY_RECALL,
        annotations={"readOnlyHint": True, "openWorldHint": False},
    )
    def memory_recall(query: str, k: int = 5, kind: str | None = None) -> dict[str, Any]:
        """动态召回入口：检索 + 每条记忆标注 status/observed_at/kind。"""
        return _call(service.recall, query, k, kind)

    @mcp.tool(
        name="memory_update",
        description=_DESC_MEMORY_UPDATE,
        annotations={
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": False,
            "openWorldHint": False,
        },
    )
    def memory_update(note_id: str, content: str, reason: str) -> dict[str, Any]:
        """修订既有记忆正文：写前备份 + reason 落盘，禁止静默覆盖。"""
        return _call(service.update_memory, note_id, content, reason)

    @mcp.tool(
        name="memory_supersede",
        description=_DESC_MEMORY_SUPERSEDE,
        annotations={
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": False,
            "openWorldHint": False,
        },
    )
    def memory_supersede(note_id: str, new_content: str, reason: str) -> dict[str, Any]:
        """用新内容替代旧记忆：新笔记 + 旧笔记 superseded_by 链到新 ID。"""
        return _call(service.supersede_memory, note_id, new_content, reason)

    @mcp.tool(
        name="memory_invalidate",
        description=_DESC_MEMORY_INVALIDATE,
        annotations={
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": False,
            "openWorldHint": False,
        },
    )
    def memory_invalidate(note_id: str, reason: str) -> dict[str, Any]:
        """判定记忆失效：自动生成墓碑笔记，旧记忆沿链可达墓碑。"""
        return _call(service.invalidate_memory, note_id, reason)

    @mcp.tool(
        name="memory_timeline",
        description=_DESC_MEMORY_TIMELINE,
        annotations={"readOnlyHint": True, "openWorldHint": False},
    )
    def memory_timeline(note_id: str) -> dict[str, Any]:
        """单条记忆的版本史：redirect 链 + 变更记录，旧 → 新有序事件列表。"""
        return _call(service.timeline, note_id)

    @mcp.tool(
        name="memory_conflicts",
        description=_DESC_MEMORY_CONFLICTS,
        annotations={"readOnlyHint": True, "openWorldHint": False},
    )
    def memory_conflicts(status: str = "open") -> dict[str, Any]:
        """冲突台账查询：open/resolved/all 三种过滤。"""
        return _call(service.conflicts, status)

    @mcp.tool(
        name="memory_profile",
        description=_DESC_MEMORY_PROFILE,
        annotations={"readOnlyHint": True, "openWorldHint": False},
    )
    def memory_profile() -> dict[str, Any]:
        """User Memory 视图：列出 kind=user 的 active 记忆（平铺）。"""
        return _call(service.profile)

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
