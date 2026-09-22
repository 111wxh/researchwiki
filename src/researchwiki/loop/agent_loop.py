"""真实 agent loop：plan → act(tools) → observe → distill → report。

强模型驱动的主循环，产出与 ResearchRun（mock）同构的 UI Message Stream 事件：
start / reasoning-* / data-task / data-note / data-conflict / text-* / source-url / finish。
前端零改动；蒸馏与子 agent 优先用 cheap 档（未配置则复用 strong）。

熔断：max_steps（默认 12，计 act 循环的模型调用次数）与 token_budget
（默认 200_000，主循环/蒸馏/子 agent 的 input tokens 累计）任一超限即强制
收尾，给报告模型一条"预算耗尽，基于已有信息直接写报告"的指示。

state 约定：每次 run 落盘 wiki-data/runs/{时间戳}-{trace_id}/：
- research-plan.md：规划阶段产出的研究计划；
- state.md：滚动状态（阶段、步骤、token 消耗、来源数、关键发现），每阶段原子重写；
- report.md：最终报告（供阶段 3 Distiller 消费）；
- run-metrics.json：本次 run 的复用/成本/质量指标（PLAN §4.4，与 tokens.jsonl 可复算对账）。

Prior 注入（PLAN §4.2）：run 开始前对问题检索历史 Wiki 的 active notes（索引
落后先 rebuild），以"仅供核验"标签块注入 plan 步骤的 user 消息；Prior 的 URL
不进本轮 SourcePool，报告 [n] 编号只指向本轮 fresh 来源。

Formation 判定（PLAN v2 RQ1：什么时候应该记住）：蒸馏候选入库前过确定性策略
（wiki/formation.py，零模型调用）——正文过短 / knowledge 无来源 / importance
不足 / 与既有记忆近乎重复的候选拒绝入库并计数，其余按权重表赋 importance、把
判定理由写进笔记 extra。显式传入 [formation] 段才启用（server 通路），
``enabled = false`` 是逃生阀；未配置时照旧入库、零行为变化。
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from researchwiki.llm.accounting import TokenAccountant
from researchwiki.llm.provider import Message, Provider, TokenUsage, chunk_text
from researchwiki.loop.metrics import (
    RunMetrics,
    compute_citation_coverage,
    write_run_metrics,
)
from researchwiki.loop.registry import Tool, ToolRegistry
from researchwiki.loop.subagent import ResearchSubagent, SubagentResult
from researchwiki.tools import (
    DEFAULT_WIKI_DATA_ROOT,
    FetchResult,
    atomic_write_text,
    fetch_url,
    get_search_provider,
    safe_list,
    safe_read,
    safe_write,
)
from researchwiki.tools.search import SearchProvider

if TYPE_CHECKING:
    # 仅类型标注用：wiki 层在方法内延迟导入（见 AgentLoop._build_wiki_layer）——
    # wiki.store → loop.notes → loop/__init__ → loop.agent_loop 的包初始化顺序会让
    # 模块级「loop 导入 wiki」形成循环导入。
    from researchwiki.wiki.distiller import CandidateNote
    from researchwiki.wiki.embeddings import EmbeddingProvider
    from researchwiki.wiki.formation import FormationDecision
    from researchwiki.wiki.frontmatter import SourceRef
    from researchwiki.wiki.index import SearchIndex
    from researchwiki.wiki.prior import PriorContext
    from researchwiki.wiki.store import Note

# ---- 提示词 --------------------------------------------------------------


PLAN_SYSTEM = (
    "你是「自进化研究 Wiki」的研究 agent，负责针对用户问题做长程检索、阅读与综合。\n"
    "当前是规划阶段：请产出一份简明的研究计划，包括：\n"
    "1. 对问题的理解与研究目标；\n"
    "2. 任务拆分（3-5 个子任务，用「- 」列表，每项一句话）；\n"
    "3. 打算使用的检索词与信息源类型；\n"
    "4. 预期产出（带 [n] 引用的报告 + 原子笔记）。\n"
    "规划阶段不要调用任何工具，直接输出计划文本（300 字以内）。"
)

ACT_SYSTEM = (
    "你是「自进化研究 Wiki」的研究 agent，正在执行既定研究计划。可用工具：\n"
    "- web_search(query, max_results?)：网页检索，返回标题、URL、摘要；\n"
    "- fetch_url(url)：抓取网页正文（超长自动截断），原文快照自动落盘 sources/；\n"
    "- fs_read(path) / fs_write(path, content) / fs_list(path)：读写 wiki-data 沙箱内文件；\n"
    "- dispatch_research(topic, brief?)：派发一个独立上下文的子 agent 深入研究子问题，\n"
    "  返回 JSON（findings / notes / sources），适合需要多轮检索的分支任务。\n"
    "规则：\n"
    "- 每步调用一到多个工具推进研究；优先权威来源，同一 URL 不要重复 fetch；\n"
    "- 工具返回 {\"error\": ...} 时分析原因，调整参数或更换来源；\n"
    "- 信息足够回答问题时，直接回复一段简短的中文研究总结（不要再调用工具），\n"
    "  研究阶段即告结束，随后进入报告撰写；\n"
    "- 步数与 token 预算有限，效率优先。"
)

REPORT_REQUEST = "请基于以上研究过程与工具结果，撰写最终研究报告。"

# 蒸馏提示词已迁到 wiki/distiller.py（EXTRACT_SYSTEM，含 volatility / source_urls 契约），
# 由 Distiller.extract_notes 统一持有——loop 不再自带一份，避免两处措辞漂移。


_FORCED_REASONS = {
    "max_steps": "研究步数已达上限（max_steps）",
    "token_budget": "token 预算已耗尽（token_budget）",
}


def report_system(sources: list[dict[str, Any]], *, forced_reason: str = "") -> str:
    """报告阶段 system 提示词：附编号来源池，熔断时附加强制收尾指示。"""
    lines = [
        "你是「自进化研究 Wiki」的研究 agent，现在撰写最终研究报告。",
        "要求：",
        "- Markdown 格式，包含：背景与问题、核心发现、对本项目的启示、未决问题；",
        "- 行内引用使用 [n] 编号，n 对应下方来源列表的序号；",
        "- 只依据研究过程中实际获得的信息，不要编造来源或结论；",
        "- 只输出报告正文本身。",
    ]
    if sources:
        lines.append("来源列表：")
        lines.extend(f"[{s['n']}] {s['url']} — {s['title']}" for s in sources)
    if forced_reason:
        lines.append(f"注意：{forced_reason}，请基于已有信息直接完成报告，不要尝试新的调用。")
    return "\n".join(lines)


# ---- 运行上下文与来源池 ----------------------------------------------------


Tier = Literal["strong", "cheap"]


class TierRouter(Protocol):
    """ModelRouter 的结构化类型：按档位取 Provider（测试可注入替身）。"""

    def get(self, tier: Tier) -> Provider: ...


def pick_tier(llm_config: Mapping[str, Any] | None, preferred: Tier) -> Tier:
    """蒸馏 / 子 agent 用哪个档位：preferred（cheap）配置了 base_url 就用它；
    未配置但 strong 已配置则复用 strong；两者皆空（纯 mock）或无配置时沿用 preferred。
    """
    if not llm_config:
        return preferred

    def base_url(tier: str) -> str:
        cfg = llm_config.get(tier) or {}
        return str(cfg.get("base_url") or "").strip()

    if base_url(preferred):
        return preferred
    if base_url("strong"):
        return "strong"
    return preferred


@dataclass
class SourcePool:
    """本次 run 实际检索 / 抓取到的来源池（按 URL 去重）。

    报告的 [n] 引用编号与 source-url 事件都从这里生成，保证引用可溯。
    抓取成功时同时记下 content_hash（网页快照定位），蒸馏入库时写进笔记
    frontmatter.sources；只检索未抓取的来源 hash 为空串。
    """

    entries: list[dict[str, str]] = field(default_factory=list)
    hashes: dict[str, str] = field(default_factory=dict)
    _seen: set[str] = field(default_factory=set)

    def add(self, url: str, title: str = "", content_hash: str = "") -> bool:
        if not url or url in self._seen:
            if url and content_hash and not self.hashes.get(url):
                self.hashes[url] = content_hash
            return False
        self._seen.add(url)
        self.entries.append({"url": url, "title": title or url})
        if content_hash:
            self.hashes[url] = content_hash
        return True

    def numbered(self) -> list[dict[str, Any]]:
        return [{"n": i, **e} for i, e in enumerate(self.entries, start=1)]

    def source_refs(self) -> list[SourceRef]:
        """来源池 → SourceRef 列表（保序，带 content_hash），供蒸馏入库写 frontmatter。"""
        from researchwiki.wiki.frontmatter import SourceRef  # 延迟导入，见文件头 TYPE_CHECKING 说明

        return [
            SourceRef(url=e["url"], content_hash=self.hashes.get(e["url"], ""))
            for e in self.entries
        ]


@dataclass
class RunContext:
    """一次 run 内各工具共享的依赖与可变状态（单测全部可注入替身）。"""

    search_provider: SearchProvider
    sources_dir: Path
    wiki_root: Path
    fetch_transport: Any = None  # httpx.BaseTransport | None，测试注入 MockTransport
    fetch_timeout: float = 15.0
    max_fetch_chars: int = 8000
    source_pool: SourcePool = field(default_factory=SourcePool)
    task_sink: list[dict[str, Any]] = field(default_factory=list)
    task_counter: int = 0
    subagent_factory: Callable[[str, str], SubagentResult] | None = None
    # 本轮真实发生的 fresh 检索计数（run-metrics 用）；主循环与子 agent 的
    # search/fetch handler 共享同一个 ctx，两个 registry 的调用都计在这里
    search_calls: int = 0
    fetch_calls: int = 0

    def next_task_id(self) -> str:
        self.task_counter += 1
        return f"task-{self.task_counter}"

    def push_task(self, task_id: str, title: str, status: str, detail: str) -> None:
        """把一条 data-task 事件推入暂存队列，由主循环在工具执行后取走发出。"""
        self.task_sink.append(
            {
                "type": "data-task",
                "id": task_id,
                "data": {"taskId": task_id, "title": title, "status": status, "detail": detail},
            }
        )

    def add_source(self, url: str, title: str = "", content_hash: str = "") -> None:
        self.source_pool.add(url, title, content_hash=content_hash)

    def source_refs(self) -> list[SourceRef]:
        """来源池的 SourceRef 视图（蒸馏入库写 frontmatter.sources 用）。"""
        return self.source_pool.source_refs()


# ---- 具体工具接线 ----------------------------------------------------------


def _title_from_text(text: str, url: str) -> str:
    """从抓取正文猜标题：第一个非空行去掉 # 后截 80 字，退化用 URL。"""
    for line in text.splitlines():
        line = line.strip().lstrip("#").strip()
        if line:
            return line[:80]
    return url


def _make_sandbox_resolver(ctx: RunContext) -> Callable[[str], Path]:
    """模型传入的 fs 路径按"相对 wiki-data"约定归一：
    相对路径拼到沙箱根下，绝对路径原样交给 safe_* 做越界检查。
    """

    def resolve(path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else ctx.wiki_root / p

    return resolve


def _register_research_tools(
    registry: ToolRegistry, ctx: RunContext, *, with_dispatch: bool
) -> None:
    """注册基础研究工具；with_dispatch 控制是否给主循环开放子 agent 派发（防递归）。"""
    _resolve_in_sandbox = _make_sandbox_resolver(ctx)

    def search_handler(args: dict[str, Any]) -> str:
        query = str(args.get("query") or "").strip()
        if not query:
            raise ValueError("query 不能为空")
        max_results = int(args.get("max_results") or 5)
        ctx.search_calls += 1
        hits = ctx.search_provider.search(query, max_results=max_results)
        for h in hits:
            ctx.add_source(h.url, h.title)
        return json.dumps(
            {
                "query": query,
                "results": [
                    {"title": h.title, "url": h.url, "snippet": h.snippet} for h in hits
                ],
            },
            ensure_ascii=False,
        )

    def fetch_handler(args: dict[str, Any]) -> str:
        url = str(args.get("url") or "").strip()
        if not url:
            raise ValueError("url 不能为空")
        ctx.fetch_calls += 1
        result: FetchResult = fetch_url(
            url,
            max_chars=ctx.max_fetch_chars,
            sources_dir=ctx.sources_dir,
            transport=ctx.fetch_transport,
            timeout=ctx.fetch_timeout,
        )
        ctx.add_source(
            result.final_url or url,
            _title_from_text(result.text, url),
            result.content_hash,
        )
        return json.dumps(
            {
                "url": result.url,
                "final_url": result.final_url,
                "http_status": result.http_status,
                "truncated": result.truncated,
                "content_hash": result.content_hash,
                "text": result.text,
            },
            ensure_ascii=False,
        )

    def fs_read_handler(args: dict[str, Any]) -> str:
        path = str(args.get("path") or "")
        content = safe_read(_resolve_in_sandbox(path), root=ctx.wiki_root)
        return json.dumps({"path": path, "content": content}, ensure_ascii=False)

    def fs_write_handler(args: dict[str, Any]) -> str:
        path = str(args.get("path") or "")
        content = str(args.get("content") or "")
        if not path:
            raise ValueError("path 不能为空")
        safe_write(_resolve_in_sandbox(path), content, root=ctx.wiki_root)
        return json.dumps(
            {"path": path, "bytes": len(content.encode("utf-8"))}, ensure_ascii=False
        )

    def fs_list_handler(args: dict[str, Any]) -> str:
        path = str(args.get("path") or "")
        entries = [
            {"name": p.name, "type": "dir" if p.is_dir() else "file"}
            for p in safe_list(_resolve_in_sandbox(path), root=ctx.wiki_root)
        ]
        return json.dumps({"path": path, "entries": entries}, ensure_ascii=False)

    def dispatch_handler(args: dict[str, Any]) -> str:
        topic = str(args.get("topic") or "").strip()
        if not topic:
            raise ValueError("topic 不能为空")
        brief = str(args.get("brief") or "")
        if ctx.subagent_factory is None:
            raise RuntimeError("子 agent 未配置")
        task_id = ctx.next_task_id()
        ctx.push_task(task_id, f"子 agent 研究：{topic}", "running", "独立上下文与预算运行中")
        result = ctx.subagent_factory(topic, brief)
        ctx.push_task(
            task_id,
            f"子 agent 研究：{topic}",
            "done",
            f"{result.steps} 步 · {len(result.sources)} 个来源"
            f" · findings {len(result.findings)} 字",
        )
        return json.dumps(
            {
                "topic": topic,
                "findings": result.findings,
                "notes": result.notes,
                "sources": result.sources,
                "steps": result.steps,
            },
            ensure_ascii=False,
        )

    registry.register(
        Tool(
            name="web_search",
            description="网页检索：返回与 query 相关的标题、URL 与摘要列表。",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索词"},
                    "max_results": {
                        "type": "integer",
                        "description": "结果条数上限，默认 5",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
            handler=search_handler,
        )
    )
    registry.register(
        Tool(
            name="fetch_url",
            description="抓取网页并提取正文（超长截断），原文快照自动落盘 sources/。",
            parameters={
                "type": "object",
                "properties": {"url": {"type": "string", "description": "完整 URL"}},
                "required": ["url"],
            },
            handler=fetch_handler,
        )
    )
    registry.register(
        Tool(
            name="fs_read",
            description="读取 wiki-data 沙箱内的文件内容。",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "相对 wiki-data/ 的路径"}
                },
                "required": ["path"],
            },
            handler=fs_read_handler,
        )
    )
    registry.register(
        Tool(
            name="fs_write",
            description="原子写入 wiki-data 沙箱内的文件（自动建父目录）。",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "相对 wiki-data/ 的路径",
                    },
                    "content": {"type": "string", "description": "文件内容"},
                },
                "required": ["path", "content"],
            },
            handler=fs_write_handler,
        )
    )
    registry.register(
        Tool(
            name="fs_list",
            description="列出 wiki-data 沙箱内目录的条目。",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "相对 wiki-data/ 的路径，根目录传 .",
                    }
                },
                "required": ["path"],
            },
            handler=fs_list_handler,
        )
    )
    if with_dispatch:
        registry.register(
            Tool(
                name="dispatch_research",
                description="派发独立上下文的研究子 agent 深入研究一个子问题，"
                "返回 JSON（findings / notes / sources）。",
                parameters={
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string", "description": "子问题主题"},
                        "brief": {"type": "string", "description": "补充要求（可选）"},
                    },
                    "required": ["topic"],
                },
                handler=dispatch_handler,
            )
        )


def build_registry(ctx: RunContext) -> ToolRegistry:
    """主循环工具集：基础工具 + dispatch_research（子 agent 入口）。"""
    registry = ToolRegistry()
    _register_research_tools(registry, ctx, with_dispatch=True)
    return registry


def build_sub_registry(ctx: RunContext) -> ToolRegistry:
    """子 agent 工具集：只有检索与阅读，不含 dispatch_research（防止递归派发）。"""
    registry = ToolRegistry()
    _register_research_tools(registry, ctx, with_dispatch=False)
    return registry


# ---- 主循环 ----------------------------------------------------------------


class AgentLoop:
    """真实研究 run：强模型驱动 plan→act→observe，产出与 mock 同构的事件流。

    公开入口只有 events()（Iterator[dict]，SSE 层逐条序列化）；
    run 结束后 report_text / plan_text / input_tokens 等属性可供评测层读取。
    """

    def __init__(
        self,
        question: str,
        *,
        router: TierRouter,
        llm_config: Mapping[str, Any] | None = None,
        accountant: TokenAccountant | None = None,
        search_provider: SearchProvider | None = None,
        wiki_root: str | Path = DEFAULT_WIKI_DATA_ROOT,
        run_dir: str | Path | None = None,
        max_steps: int = 12,
        token_budget: int = 200_000,
        subagent_max_steps: int = 6,
        subagent_token_budget: int = 50_000,
        fetch_transport: Any = None,
        max_fetch_chars: int = 8000,
        clock: Callable[[], float] = time.perf_counter,
        embedding: EmbeddingProvider | None = None,
        wiki_config: Mapping[str, Any] | None = None,
        prior_config: Mapping[str, Any] | None = None,
        formation_config: Mapping[str, Any] | None = None,
        build_pages: bool = False,
    ) -> None:
        self.question = question
        self.router = router
        self.llm_config = llm_config
        self.accountant = accountant
        self.trace_id = accountant.new_trace_id() if accountant else "real00000000"
        self.wiki_root = Path(wiki_root)
        default_run_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}-{self.trace_id}"
        self.run_dir = Path(run_dir) if run_dir else self.wiki_root / "runs" / default_run_id
        self.max_steps = max_steps
        self.token_budget = token_budget
        self.clock = clock

        # 档位选择：主循环恒用 strong；蒸馏与子 agent 优先 cheap，未配置复用 strong
        self.provider = router.get("strong")
        self._aux_tier = pick_tier(llm_config, "cheap")
        self.distill_provider = router.get(self._aux_tier)
        self.subagent_provider = router.get(self._aux_tier)

        self.ctx = RunContext(
            search_provider=search_provider or get_search_provider(),
            sources_dir=self.wiki_root / "sources",
            wiki_root=self.wiki_root,
            fetch_transport=fetch_transport,
            max_fetch_chars=max_fetch_chars,
        )
        self.registry = build_registry(self.ctx)
        self.sub_registry = build_sub_registry(self.ctx)
        self.tools_schema = self.registry.schemas()
        self.ctx.subagent_factory = self._run_subagent
        self.subagent_budget = subagent_token_budget
        self.subagent_steps = subagent_max_steps

        # wiki 三层存储（notes/ pages/ conflicts/，与 loop 层既有目录布局互读兼容）：
        # run 内蒸馏走 Distiller（报告素材 → 原子笔记）→ Ingestor（查重合并 + 规范 ID）。
        self._build_wiki_layer(
            embedding=embedding,
            wiki_config=wiki_config,
            prior_config=prior_config,
            formation_config=formation_config,
        )
        self.build_pages = build_pages
        # 滚动状态（state.md 的数据源）
        self.input_tokens = 0
        self.output_tokens = 0
        self.step_count = 0
        self.tool_calls_count = 0
        self.notes_written = 0
        self.plan_text = ""
        self.report_text = ""
        self.research_summary = ""
        self.forced_reason = ""
        # 子 agent 带回的候选笔记，蒸馏阶段与主循环抽取结果一并入库
        self.pending_candidates: list[CandidateNote] = []
        # Prior 检索结果（events() 开头填充；None = 未启用或尚未检索）与入库动作计数
        self.prior_context: PriorContext | None = None
        self.notes_created = 0
        self.notes_merged = 0
        self.notes_superseded = 0
        # formation 判定计数（RQ1）：候选 = 拒绝 + 入库（逃生阀关闭时拒绝恒 0）
        self.formation_stats = {"candidates": 0, "persisted": 0, "rejected": 0}

    # ---- 基础设施 ----------------------------------------------------------

    def _build_wiki_layer(
        self,
        *,
        embedding: EmbeddingProvider | None,
        wiki_config: Mapping[str, Any] | None,
        prior_config: Mapping[str, Any] | None,
        formation_config: Mapping[str, Any] | None,
    ) -> None:
        """装配 wiki 层：WikiStore / EntityRegistry / Ingestor / Distiller / Prior 索引。

        方法内延迟导入的原因见文件头 TYPE_CHECKING 注释：wiki.store 反向依赖
        loop.notes，模块级导入会在「先 import wiki 包」的顺序下形成循环导入。
        """
        from researchwiki.wiki.distiller import Distiller
        from researchwiki.wiki.entities import EntityRegistry
        from researchwiki.wiki.formation import FormationSettings, from_config
        from researchwiki.wiki.index import SearchIndex, wiki_settings
        from researchwiki.wiki.ingest import Ingestor, dedup_settings
        from researchwiki.wiki.prior import prior_settings
        from researchwiki.wiki.store import WikiStore

        # 去重阈值：config [wiki] 段（dedup_similarity / dedup_entity_overlap）可覆盖
        self.dedup = dedup_settings(wiki_config)
        self.wiki_store = WikiStore(self.wiki_root)
        self.entity_registry = EntityRegistry(self.wiki_root)
        self.ingestor = Ingestor(
            self.wiki_store,
            embedding=embedding,
            similarity_threshold=self.dedup.similarity,
            entity_overlap_min=self.dedup.entity_overlap_min,
            trace_id=self.trace_id,
            clock=self.clock,
            on_usage=self._absorb_usage,
        )
        self.distiller = Distiller(
            self.wiki_store,
            provider=self.distill_provider,
            entity_registry=self.entity_registry,
            accountant=self.accountant,
            trace_id=self.trace_id,
            clock=self.clock,
            step="distill",
            on_usage=self._absorb_usage,
        )
        # Prior 检索层（PLAN §4.2）：[prior] 段缺省 = enabled + 默认值，不强制用户配置。
        # 索引复用 AgentLoop 的 embedding 与 [wiki] 的 tokenizer/半衰期配置，
        # 构造方式同 mcp_server/service.py 的 _open_index；禁用时不建索引（零副作用）。
        self.prior_settings = prior_settings(prior_config)
        self.prior_index: SearchIndex | None = None
        if self.prior_settings.enabled:
            # wiki_config 的既定口径与 dedup_settings 一致：[wiki] 段或完整 config
            # 皆可（server / smoke 脚本传的是段）；wiki_settings 只认完整 config，
            # 这里归一后再解析。
            if isinstance(wiki_config, Mapping) and "wiki" in wiki_config:
                full_cfg: Mapping[str, Any] = wiki_config
            else:
                full_cfg = {"wiki": wiki_config or {}}
            settings = wiki_settings(full_cfg)
            self.prior_index = SearchIndex(
                self.wiki_root,
                embedding=embedding,
                tokenizer=settings.fts_tokenizer,
                half_life_days=settings.half_life_days,
            )
        # formation（RQ1：什么时候应该记住）：显式传入 [formation] 段才启用入库
        # 判定；None = 未配置（脚本 / 既有测试等老调用方），跳过判定照旧入库、
        # 零行为变化。server 始终传 config.get("formation")，段内 enabled=false
        # 是逃生阀（同样跳过判定，但计数照记）。
        if formation_config is None:
            self.formation_settings = FormationSettings(enabled=False)
        else:
            self.formation_settings = from_config(formation_config)

    def _retrieve_priors(self) -> None:
        """run 开始前的 Prior 检索（PLAN §4.2）：索引新鲜检查（落后即 rebuild）→ 检索。

        结果存 ``self.prior_context``（None = 未启用）；Prior 的 URL **不得**进
        SourcePool（不调用 ctx.add_source），只随 ``format()`` 注入 plan 步骤的
        user 消息——报告 [n] 编号因此只指向本轮 fresh 来源。
        """
        if self.prior_index is None:
            self.prior_context = None
            return
        # wiki 层延迟导入，见文件头 TYPE_CHECKING 说明
        from researchwiki.wiki.prior import ensure_index_fresh, retrieve_priors

        # 正确性优先：索引落后于 store 就整体 rebuild（性能后优化）
        ensure_index_fresh(self.wiki_store, self.prior_index)
        self.prior_context = retrieve_priors(
            self.question,
            self.wiki_store,
            self.prior_index,
            k=self.prior_settings.k,
            max_chars=self.prior_settings.max_chars,
        )

    # ---- formation 判定（RQ1：什么时候应该记住）-----------------------------

    def _formation_decision(self, candidate: CandidateNote) -> FormationDecision | None:
        """对蒸馏候选做 formation 判定；未启用（逃生阀 / 未传配置）返回 None。

        similarity_max 用入库层同一套嵌入与余弦实现现算（确定性、零网络）：
        嵌入失败或 Wiki 为空时为 None，交给 formation 跳过近乎重复规则。
        """
        if not self.formation_settings.enabled:
            return None
        from researchwiki.wiki.formation import evaluate_candidate  # 延迟导入，见文件头说明

        return evaluate_candidate(
            candidate.text,
            entities=candidate.entities,
            sources=candidate.source_refs,
            similarity_max=self._similarity_max(candidate.text),
            settings=self.formation_settings,
        )

    def _similarity_max(self, text: str) -> float | None:
        """候选正文与既有 active 记忆的最大余弦相似度；无笔记或嵌入失败返回 None。"""
        existing = self.wiki_store.list_notes()  # 只与 active 笔记比对（与查重同口径）
        if not existing:
            return None
        try:
            vectors = self.ingestor.embedding.embed(
                [text, *(note.body.strip() for note in existing)]
            )
        except Exception as exc:  # noqa: BLE001 -- 嵌入失败降级为"不做近乎重复判定"
            print(
                f"[agent-loop] formation 相似度计算失败，跳过近乎重复规则："
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return None
        from researchwiki.wiki.ingest import cosine  # 延迟导入，见文件头说明

        return max(cosine(vectors[0], vector) for vector in vectors[1:])

    def _annotate_formation(self, note: Note, decision: FormationDecision) -> None:
        """把 formation 判定写回已入库笔记：importance/kind 进 frontmatter，
        判定理由与置信度进 extra（extra["formation_reason"] / extra["formation_confidence"]）。

        原地覆写（note_id 传回，created 保留），其余元数据字段透传不变；
        属辅助产物：失败只记 stderr，不推翻已完成的入库。
        """
        try:
            meta = note.meta
            extra: dict[str, Any] = dict(meta.extra)
            extra["formation_reason"] = decision.reason
            extra["formation_confidence"] = decision.confidence
            self.wiki_store.save_note(
                note.body,
                note_id=note.id,
                title=note.title,
                entities=list(note.entities),
                confidence=meta.confidence,
                status=meta.status,
                redirect_to=meta.redirect_to,
                superseded_by=meta.superseded_by,
                volatility=meta.volatility,
                kind=decision.kind,
                importance=decision.importance,
                observed_at=meta.observed_at,
                reviewed_at=meta.reviewed_at,
                trace_id=meta.trace_id,
                sources=list(meta.sources),
                extra=extra,
                created=meta.created,
            )
        except Exception as exc:  # noqa: BLE001 -- 标注失败不能推翻入库
            print(
                f"[agent-loop] formation 标注入库笔记失败：{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    def _run_subagent(self, topic: str, brief: str) -> SubagentResult:
        sub = ResearchSubagent(
            topic=topic,
            brief=brief,
            provider=self.subagent_provider,
            registry=self.sub_registry,
            accountant=self.accountant,
            trace_id=self.trace_id,
            max_steps=self.subagent_steps,
            token_budget=self.subagent_budget,
            clock=self.clock,
            # 子 agent 用量回流主循环滚动计数（独立预算只管它自己的熔断），
            # 保证 run-metrics 与 tokens.jsonl 按 trace_id 可对账
            on_usage=self._absorb_usage,
        )
        result = sub.run()
        # 子 agent 产物汇入本次 run：笔记进蒸馏队列（同样走入库去重），来源进引用池
        for s in result.sources:
            self.ctx.add_source(str(s.get("url") or ""), str(s.get("title") or ""))
        # 子 agent 未按条给出来源，这里把它检索到的来源集合作为其笔记的候选来源
        # （粗粒度溯源：至少能定位到证据集合，比空 sources 更接近证据链）
        from researchwiki.wiki.distiller import CandidateNote  # 延迟导入，见文件头说明

        refs = self.ctx.source_refs()
        urls = [str(s.get("url") or "") for s in result.sources if str(s.get("url") or "")]
        self.pending_candidates.extend(
            CandidateNote.from_dict({**note, "source_urls": urls}, sources=refs)
            for note in result.notes
        )
        return result

    def _absorb_usage(self, usage: TokenUsage | None) -> None:
        if usage is None:
            return
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens

    def _record(self, step: str, provider: Provider, usage: TokenUsage | None, t0: float) -> None:
        """每步一行记账；usage 缺失时记 0（行数仍是每步一行，可对账调用次数）。"""
        if self.accountant is None:
            return
        self.accountant.record(
            trace_id=self.trace_id,
            step=step,
            model=provider.model,
            usage=usage or TokenUsage(),
            latency_ms=(self.clock() - t0) * 1000.0,
        )

    def _drain_tasks(self) -> Iterator[dict[str, Any]]:
        while self.ctx.task_sink:
            yield self.ctx.task_sink.pop(0)

    def _write_plan_file(self) -> None:
        content = (
            "# 研究计划\n\n"
            f"- 问题：{self.question}\n"
            f"- trace_id：{self.trace_id}\n"
            f"- 生成时间：{datetime.now(UTC).isoformat(timespec='seconds')}\n\n"
            f"{self.plan_text}\n"
        )
        atomic_write_text(self.run_dir / "research-plan.md", content)

    def _write_state(self, phase: str) -> None:
        """滚动状态原子重写：阶段推进与每个研究步骤后各写一次。"""
        pct = round(self.input_tokens / self.token_budget * 100, 1) if self.token_budget else 0.0
        digest = self.research_summary or self.plan_text
        lines = [
            "# Run 状态",
            "",
            f"- trace_id：{self.trace_id}",
            f"- 问题：{self.question}",
            f"- 阶段：{phase}",
            f"- 步骤：{self.step_count}/{self.max_steps}（工具调用 {self.tool_calls_count} 次）",
            f"- Token：input {self.input_tokens} / output {self.output_tokens}"
            f"（预算 {self.token_budget}，已用 {pct}%）",
            f"- 来源：{len(self.ctx.source_pool.entries)} 个",
            f"- 笔记：本次已写 {self.notes_written} 条",
            f"- 记忆形成：候选 {self.formation_stats['candidates']}，"
            f"入库 {self.formation_stats['persisted']}，拒绝 {self.formation_stats['rejected']}",
        ]
        if self.forced_reason:
            lines.append(f"- 熔断：{self.forced_reason}")
        lines += ["", "## 关键发现", "", (digest[:400] or "（暂无）"), ""]
        atomic_write_text(self.run_dir / "state.md", "\n".join(lines))

    # ---- 事件主流程 ----------------------------------------------------------

    def events(self) -> Iterator[dict[str, Any]]:
        """SSE 主流程：start … finish（与 mock 同构的事件序列）。

        run-metrics.json 恰好落盘一次：正常结束、中途异常、生成器被 close 三种
        关闭路径都经 try/finally 写一次（数据取当时实况）；落盘自身失败不抛出，
        既不压过流中的原异常，也不让已写出 report.md 的 run 在收尾时崩掉。
        """
        t_start = self.clock()
        yield {"type": "start"}
        try:
            yield from self._events()
        finally:
            self._write_run_metrics_guarded(t_start)

    def _events(self) -> Iterator[dict[str, Any]]:
        # ---- 阶段 0：Prior 检索（PLAN §4.2：plan 步骤之前，不污染来源池）----
        self._retrieve_priors()
        plan_content = self.question
        if self.prior_context is not None:
            prior_block = self.prior_context.format()
            if prior_block:
                # Prior 块放在问题之前、带分隔线；自带"仅供核验"标签行。
                # 系统提示词一字不动——稳定 prompt 前缀是阶段 3 的前提，
                # Prior 属于动态后缀区。
                plan_content = f"{prior_block}\n---\n\n研究问题：{self.question}"
        plan_messages = [Message(role="user", content=plan_content)]

        # ---- 阶段 1：规划（reasoning 流式转发，计划文本并入同一思考块）----
        t0 = self.clock()
        yield {"type": "reasoning-start", "id": "plan"}
        parts: list[str] = []
        usage: TokenUsage | None = None
        for ev in self.provider.stream(plan_messages, system=PLAN_SYSTEM, tools=None):
            if ev.type == "reasoning_delta":
                yield {"type": "reasoning-delta", "id": "plan", "delta": ev.delta}
            elif ev.type == "text_delta":
                parts.append(ev.delta)
                yield {"type": "reasoning-delta", "id": "plan", "delta": ev.delta}
            elif ev.type == "usage" and ev.usage is not None:
                usage = ev.usage
        yield {"type": "reasoning-end", "id": "plan"}
        self.plan_text = "".join(parts)
        self._absorb_usage(usage)
        self._record("plan", self.provider, usage, t0)
        self._write_plan_file()
        task_plan = self.ctx.next_task_id()
        self.ctx.push_task(task_plan, "制定研究计划", "done", "计划已写入 research-plan.md")
        yield from self._drain_tasks()
        self._write_state("researching")

        # ---- 阶段 2：执行（带 tools 的循环，tool_calls → 执行 → role="tool" 回传）----
        history: list[Message] = [
            plan_messages[0],
            Message(role="assistant", content=self.plan_text),
        ]
        steps_used = 0
        forced = ""
        if self.input_tokens >= self.token_budget:
            forced = "token_budget"
        elif self.max_steps <= 0:
            forced = "max_steps"

        task_act: str | None = None
        if not forced:
            task_act = self.ctx.next_task_id()
            self.ctx.push_task(task_act, "执行研究（工具调用）", "running", "第 0 步 · 计划已就绪")
            yield from self._drain_tasks()

        while not forced:
            if steps_used >= self.max_steps:
                forced = "max_steps"
                break
            if self.input_tokens >= self.token_budget:
                forced = "token_budget"
                break
            steps_used += 1
            self.step_count = steps_used
            t0 = self.clock()
            parts = []
            tool_calls: list[dict[str, Any]] | None = None
            usage = None
            for ev in self.provider.stream(history, system=ACT_SYSTEM, tools=self.tools_schema):
                if ev.type == "text_delta":
                    parts.append(ev.delta)
                elif ev.type == "tool_calls" and ev.tool_calls:
                    tool_calls = ev.tool_calls
                elif ev.type == "usage" and ev.usage is not None:
                    usage = ev.usage
            step_text = "".join(parts)
            self._absorb_usage(usage)
            self._record(f"act:{steps_used}", self.provider, usage, t0)

            if not tool_calls:
                # 纯文本结尾：研究阶段自然结束
                self.research_summary = step_text
                history.append(Message(role="assistant", content=step_text))
                break

            self.tool_calls_count += len(tool_calls)
            history.append(
                Message(role="assistant", content=step_text, tool_calls=tool_calls)
            )
            names: list[str] = []
            for tc in tool_calls:
                fn = tc.get("function") or {}
                name = str(fn.get("name") or "unknown")
                names.append(name)
                result = self.registry.dispatch(name, fn.get("arguments") or "{}")
                history.append(
                    Message(
                        role="tool",
                        content=result,
                        tool_call_id=str(tc.get("id") or ""),
                        name=name,
                    )
                )
                yield from self._drain_tasks()
            if task_act:
                self.ctx.push_task(
                    task_act,
                    "执行研究（工具调用）",
                    "running",
                    f"第 {steps_used} 步 · 调用 {'、'.join(names)}"
                    f" · 来源 {len(self.ctx.source_pool.entries)} 个",
                )
            yield from self._drain_tasks()
            self._write_state("researching")

        if forced:
            self.forced_reason = _FORCED_REASONS[forced]

        if task_act is not None:
            if forced:
                detail = f"{steps_used} 步后熔断 · 来源 {len(self.ctx.source_pool.entries)} 个"
            else:
                detail = (
                    f"{steps_used} 步 · {self.tool_calls_count} 次工具调用"
                    f" · 来源 {len(self.ctx.source_pool.entries)} 个"
                )
            self.ctx.push_task(task_act, "执行研究（工具调用）", "done", detail)
            yield from self._drain_tasks()

        # 研究总结以第二个思考块呈现（与最终报告的正文流区分）
        if self.research_summary.strip():
            yield {"type": "reasoning-start", "id": "observe"}
            for piece in chunk_text(self.research_summary, 48):
                yield {"type": "reasoning-delta", "id": "observe", "delta": piece}
            yield {"type": "reasoning-end", "id": "observe"}
        self._write_state("distilling")

        # ---- 阶段 3：蒸馏（cheap 档；报告素材 → Distiller 抽取 → Ingestor 去重入库）----
        #
        # 顺序说明（与既有事件协议/调用序列的取舍）：data-note 出现在报告正文之前
        # （与 mock 演示 ResearchRun 同构，前端零改动），且 LLM 调用序列里 distill
        # 必须先于 report（tests/test_loop.py 断言了调用顺序与记账步骤），因此这里
        # 蒸馏的是**报告素材**（研究总结 + 工具结果摘编，与随后的报告同源），而不是
        # 报告成品。Distiller.extract_notes 的输入契约仍是报告正文：阶段 4 的
        # consolidation / 单跑蒸馏可以直接把 run_dir/report.md 喂给它。
        task_distill = self.ctx.next_task_id()
        self.ctx.push_task(task_distill, "蒸馏原子笔记 → wiki", "running", "抽取事实中")
        yield from self._drain_tasks()

        sources = self.ctx.source_refs()
        material = self._report_material(history)
        extracted = self.distiller.extract_notes(
            material, question=self.question, sources=sources
        )
        candidates = [*self.pending_candidates, *extracted]

        note_seq = 0
        created = 0
        merged = 0
        ingested: list[Note] = []
        for candidate in candidates:
            # formation 判定（RQ1：什么时候应该记住）：未启用返回 None（逃生阀，
            # 全部照旧入库）；拒绝的候选不入库、只计数，防无差别堆积。
            self.formation_stats["candidates"] += 1
            decision = self._formation_decision(candidate)
            if decision is not None and not decision.persist:
                self.formation_stats["rejected"] += 1
                continue
            result = self.ingestor.add(candidate, trace_id=self.trace_id)
            note_seq += 1
            self.notes_written += 1
            self.formation_stats["persisted"] += 1
            if decision is not None:
                # importance/kind 写进 frontmatter，判定理由写进 extra（辅助产物，
                # 失败不推翻入库）
                self._annotate_formation(result.note, decision)
            created += 1 if result.action == "created" else 0
            merged += 1 if result.action == "merged" else 0
            ingested.append(result.note)
            # data-note 结构不变；data.id 恒为规范 ID（合并命中时引用既有笔记）
            yield {
                "type": "data-note",
                "id": f"note-{note_seq}",
                "data": {
                    "id": result.note.id,
                    "text": result.note.body.strip(),
                    "entities": list(result.note.entities),
                    "confidence": result.note.confidence,
                },
            }
        # 入库动作计数 → run-metrics.json（IngestResult.action 只有 created/merged，
        # superseded 在阶段 1 恒为 0，字段保位以稳住 §4.4 契约）
        self.notes_created = created
        self.notes_merged = merged
        self.notes_superseded = 0
        conflicts = self.distiller.last_conflicts
        for i, conflict in enumerate(conflicts, start=1):
            yield {
                "type": "data-conflict",
                "id": f"conflict-{i}",
                "data": {
                    "id": conflict.id,
                    "summary": conflict.summary,
                    "action": conflict.action,
                },
            }
        if self.build_pages and ingested:
            # 页面聚合（LLM）默认关闭：run 内多一次模型调用就会拉长时延，
            # 由调用方显式开启（阶段 4 的 consolidation 或 MCP 侧写路径）
            for draft in self.distiller.build_pages(ingested):
                self.wiki_store.save_page(draft.slug, draft.title, draft.body)
        if task_distill:
            self.ctx.push_task(
                task_distill,
                "蒸馏原子笔记 → wiki",
                "done",
                f"新增 {created} 条笔记 · 合并 {merged} 条 · {len(conflicts)} 条冲突",
            )
            yield from self._drain_tasks()
        self._write_state("reporting")

        # ---- 阶段 4：报告（[n] 引用 + source-url 事件出自实际来源池）----
        sources = self.ctx.source_pool.numbered()
        t0 = self.clock()
        yield {"type": "text-start", "id": "report"}
        parts = []
        usage = None
        report_messages = [*history, Message(role="user", content=REPORT_REQUEST)]
        for ev in self.provider.stream(
            report_messages,
            system=report_system(sources, forced_reason=self.forced_reason),
            tools=None,
        ):
            if ev.type == "text_delta":
                parts.append(ev.delta)
                yield {"type": "text-delta", "id": "report", "delta": ev.delta}
            elif ev.type == "usage" and ev.usage is not None:
                usage = ev.usage
        yield {"type": "text-end", "id": "report"}
        self.report_text = "".join(parts)
        self._absorb_usage(usage)
        self._record("report", self.provider, usage, t0)

        for s in sources:
            yield {
                "type": "source-url",
                "sourceId": f"s{s['n']}",
                "url": s["url"],
                "title": s["title"],
            }

        atomic_write_text(self.run_dir / "report.md", self.report_text + "\n")
        self._write_state("done")
        yield {"type": "finish"}

    # ---- 辅助 ----------------------------------------------------------

    def _write_run_metrics_guarded(self, t_start: float) -> None:
        """run-metrics.json 恰好落盘一次的守卫入口（events() 的 try/finally 配套）。

        落盘自身失败只记 stderr、不抛出：不得压过流中的原异常，也不得让
        已写出 report.md 的 run 在收尾阶段崩掉。
        """
        try:
            self._write_run_metrics(t_start)
        except Exception as exc:  # noqa: BLE001 -- 指标是辅助产物，失败不能推翻本次 run
            print(
                f"[agent-loop] run-metrics.json 落盘失败：{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    def _write_run_metrics(self, t_start: float) -> None:
        """落盘 run-metrics.json（PLAN §4.4/§4.5；恰好写一次由 events() 的
        try/finally 守卫保证，含异常/被 close 的关闭路径）。

        token 口径：self.input_tokens / self.output_tokens 含主循环、蒸馏与
        子 agent 三路用量（子 agent 经 ResearchSubagent.on_usage → _absorb_usage
        回流；蒸馏/入库走 on_usage 同机制），与 tokens.jsonl 按 trace_id 经
        sum_tokens_from_jsonl 可复算对账。latency 为 events() 全程的
        self.clock 差取整（异常关闭路径取到抛错时刻）。
        """
        prior = self.prior_context
        source_count = len(self.ctx.source_pool.entries)
        # 引用覆盖率精度口径（Task 2 遗留决定）：round(x, 4)
        coverage = compute_citation_coverage(self.report_text, source_count)
        if coverage is not None:
            coverage = round(coverage, 4)
        metrics = RunMetrics.from_loop(
            trace_id=self.trace_id,
            prior_hit_count=len(prior.hits) if prior else 0,
            prior_note_ids=[h.note_id for h in prior.hits] if prior else [],
            prior_context_chars=prior.context_chars if prior else 0,
            fresh_search_count=self.ctx.search_calls,
            fresh_fetch_count=self.ctx.fetch_calls,
            source_count=source_count,
            notes_created=self.notes_created,
            notes_merged=self.notes_merged,
            notes_superseded=self.notes_superseded,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            latency_ms=int((self.clock() - t_start) * 1000),
            citation_coverage=coverage,
        )
        write_run_metrics(self.run_dir, metrics)

    def _report_material(self, history: list[Message]) -> str:
        """蒸馏输入 = 研究总结 + 工具结果摘编（研究问题由 extract_notes 的 question 参数带入）。

        见 events() 蒸馏阶段的顺序说明：既有事件协议与 LLM 调用序列都要求蒸馏发生在
        报告生成之前，所以这里喂给 Distiller 的是"报告素材"而非报告成品；
        Distiller.extract_notes 的输入契约仍是报告正文（阶段 4 可消费 report.md）。
        """
        parts: list[str] = []
        if self.research_summary.strip():
            parts.append(f"研究总结：\n{self.research_summary.strip()}")
        parts.append(f"研究过程记录（工具结果摘编）：\n{self._tool_digest(history)}")
        return "\n\n".join(parts)

    def _tool_digest(self, history: list[Message], *, per_result_chars: int = 800) -> str:
        """工具结果摘编：每个 role="tool" 消息截取前 N 字，作为蒸馏输入。"""
        blocks = []
        for m in history:
            if m.role != "tool":
                continue
            content = m.content or ""
            if len(content) > per_result_chars:
                content = content[:per_result_chars] + "…[截断]"
            blocks.append(f"[{m.name or 'tool'}] {content}")
        return "\n\n".join(blocks) if blocks else "（本次研究没有工具调用记录）"
