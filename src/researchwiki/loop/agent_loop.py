"""真实 agent loop：plan → act(tools) → observe → distill → report。

强模型驱动的主循环，产出与 ResearchRun（mock）同构的 UI Message Stream 事件：
start / reasoning-* / data-task / data-note / data-conflict / text-* / source-url / finish。
前端零改动；蒸馏与子 agent 优先用 cheap 档（未配置则复用 strong）。

熔断：max_steps（默认 12，计 act 循环的模型调用次数）与 token_budget
（默认 200_000，主循环 LLM 调用的 input tokens 累计）任一超限即强制收尾，
给报告模型一条"预算耗尽，基于已有信息直接写报告"的指示。

state 约定：每次 run 落盘 wiki-data/runs/{时间戳}-{trace_id}/：
- research-plan.md：规划阶段产出的研究计划；
- state.md：滚动状态（阶段、步骤、token 消耗、来源数、关键发现），每阶段原子重写；
- report.md：最终报告（供阶段 3 Distiller 消费）。
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from researchwiki.llm.accounting import TokenAccountant
from researchwiki.llm.provider import Message, Provider, TokenUsage, chunk_text
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
    from researchwiki.wiki.frontmatter import SourceRef
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
        self._build_wiki_layer(embedding=embedding, wiki_config=wiki_config)
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

    # ---- 基础设施 ----------------------------------------------------------

    def _build_wiki_layer(
        self, *, embedding: EmbeddingProvider | None, wiki_config: Mapping[str, Any] | None
    ) -> None:
        """装配 wiki 层：WikiStore / EntityRegistry / Ingestor（去重）/ Distiller（蒸馏）。

        方法内延迟导入的原因见文件头 TYPE_CHECKING 注释：wiki.store 反向依赖
        loop.notes，模块级导入会在「先 import wiki 包」的顺序下形成循环导入。
        """
        from researchwiki.wiki.distiller import Distiller
        from researchwiki.wiki.entities import EntityRegistry
        from researchwiki.wiki.ingest import Ingestor, dedup_settings
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
        ]
        if self.forced_reason:
            lines.append(f"- 熔断：{self.forced_reason}")
        lines += ["", "## 关键发现", "", (digest[:400] or "（暂无）"), ""]
        atomic_write_text(self.run_dir / "state.md", "\n".join(lines))

    # ---- 事件主流程 ----------------------------------------------------------

    def events(self) -> Iterator[dict[str, Any]]:
        yield {"type": "start"}

        # ---- 阶段 1：规划（reasoning 流式转发，计划文本并入同一思考块）----
        plan_messages = [Message(role="user", content=self.question)]
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
            result = self.ingestor.add(candidate, trace_id=self.trace_id)
            note_seq += 1
            self.notes_written += 1
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
