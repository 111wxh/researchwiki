#!/usr/bin/env python
"""评测四条件 runner：C1 无 Memory / C2 Vector RAG / C3 Hybrid RAG / C4 ExternalMemory warm。

── 用途（评测支线 PLAN §7.2 的产数件）────────────────────────────────────
对同一题集逐题跑四个条件，产出 ``<out>/<provider>_<ts>/results.jsonl``（每题
每条件一行）与同目录 ``manifest.json``（provenance 留痕，供 Task 7 报告引用）：

  C1 无 Memory        每题一次性 wiki 根目录（c1_<qid>/）：prior 关闭、
                      memory_update / formation 不接——纯冷启动研究 loop；
  C2 Vector RAG       run_rag(mode="vector")，不建 wiki；
  C3 Hybrid RAG       run_rag(mode="hybrid")，不建 wiki；
  C4 ExternalMemory   先播种（--seed-runs N，默认 1）：在 <out>/seed/ 用种子
                      问题（缺省取题集第一道 multi_hop 题）跑 forced deep 的
                      完整研究，蒸馏沉淀记忆；随后每题把 seed 根目录 copytree
                      成独立副本（c4_<qid>/，排除 index.db*——Windows sqlite
                      锁 + WAL 尾巴，adaptive_smoke.copy_seed_into 同款先例）
                      再跑 warm run（prior / memory_update / formation 开），
                      记忆演化按题序隔离、互不污染。

── 同口径（四条件可比的前提）────────────────────────────────────────────
同一生成 provider（real=ModelRouter strong 档 / mock=同形 ScriptedProvider）、
同一 embedding 实例、同一题集、同一受控语料；检索侧全部走 FixtureSearch
（C1/C4 hybrid 检索经 loop 的 web_search，C2/C3 由 run_rag 直连索引）。

回答长度的口径差异（诚实边界，manifest.honest_boundary 与行级 provider_note
均有声明）：C2/C3 的 300 字约束来自 RAG_SYSTEM（硬约束）；C1/C4 的 loop 报告
经 retrieval_config={"enabled": True, "forced_mode": "deep",
"deep": {"report_style": "brief"}} 近似对齐——深检索限额不变（max_fresh_
searches/max_steps 用 deep 的），仅报告样式受 brief 约束，**非硬约束**。

── fixture-corpus 的 URL 口径（诚实映射）────────────────────────────────
受控语料的 manifest URL 形如 ``fixture://ai-frameworks/<doc_id>``，不是 http
方案——loop 条件的 LLM 拿到检索命中后要用 fetch_url（httpx）抓取，直接传
fixture:// 会失败。本 runner 给 loop 的搜索 provider 包一层 URL 改写
（fixture://<domain>/<doc_id> → http://fixture.local/<domain>/<doc_id>.md），
再用 httpx.MockTransport 按 path 回放语料 markdown（content-type text/markdown，
同时兜住 Jina Reader 降级路径）——fetch_url 走真实链路（trafilatura 提取、
快照落盘 sources/）。因此 results 行里 loop 条件的来源/引用池记录的是**改写后
的 http URL**；C2/C3 不经 fetch，记录原始 fixture:// URL。映射关系在
manifest.fixture_url_mapping 与行级 provider_note 声明，各条件内 [n] 编号自洽。

── 两种 provider 模式（--provider）──────────────────────────────────────
  mock（默认）  ScriptedProvider 全链路（生成与 judge 均 mock，judge 用
                build_judge_provider(None) 的占位评分——分数无意义，仅验证
                管线）；embedding 用确定性 MockEmbeddingProvider，tokenizer
                固定 trigram。零网络、零 key、不读 .env，可进 CI。
                剧本（每 run 全新实例）：strong 5 轮 plan → web_search →
                fetch_url → 研究总结 → 报告（adaptive_smoke 的 deep 剧本同形）；
                cheap 1 轮蒸馏——笔记正文嵌入对应问题原文（Prior 检索 FTS+
                向量双通道必然命中，cold_warm_smoke 同款保证），来源指向本轮
                检索 top 命中（formation 的 require_source_for_knowledge 放行）。
                C4 的种子 run 蒸馏脚本覆盖**题集全部题目**（每题一条笔记），
                保证每个 warm 副本的 Prior 检索都能命中对应笔记；
                warm run 的蒸馏候选与 seed 笔记同文，formation 按近重复拒绝
                入库（cold_warm 同款预期语义，memory_update 无动作可比）。
  real          config.toml 的真实 strong/cheap/judge 档位 + 真实 embedding；
                检索/fetch 仍走 fixture-corpus（受控语料评测与本模式无关）。
                会产生真实 API 费用，开始前打印成本警告。启动时硬校验 judge：
                [llm.judge] 缺段或 base_url 空（judge 落 MockProvider）即干净
                报错退出、零落盘——mock judge 的占位分数无意义，不得冒充真实
                评测；确需管线自证请显式走 --provider mock。真模型蒸馏若没
                产出可命中的笔记，C4 的 prior_hit_count 可能为 0——这是真实
                测量结果，不做任何静默补偿。

── 记账与对账（不变量 ⑥）───────────────────────────────────────────────
loop 条件：TokenAccountant 指向 <wiki_root>/tokens.jsonl，run 的 input/output
tokens 与 sum_tokens_from_jsonl(<wiki_root>/tokens.jsonl, trace_id) 完全一致；
judge 调用记在同一文件但用独立 trace（<trace_id>-judge 后缀）——不污染主
trace 的对账。RAG 条件：记账集中在 <out>/tokens.jsonl（每行 trace_id 独立）。
judge sources：C1/C4 传 SourcePool.numbered()（与报告 [n] 同源），C2/C3 传
检索命中的 {url, title} 列表。

── 输出契约 ────────────────────────────────────────────────────────────
results.jsonl 每行一个 JSON 对象（UTF-8、ensure_ascii=False）：
  {"qid", "qtype", "condition", "answer", "gold_points", "em", "point_hits",
   "refusal", "citation_coverage", "judge": {...}|None,
   "metrics": {…loop=run-metrics.json 全量 / RAG={"mode","input_tokens",
     "output_tokens","latency_ms","hit_count"}…},
   "trace_id", "wiki_root", "input_tokens", "output_tokens", "latency_ms",
   "provider_note"[, "fresh_search_count"]}
fresh_search_count 仅 loop 条件携带（RAG 行不写该键，也不写 None——对齐
metrics.aggregate_rows 的缺失语义）。em/point_hits/refusal 来自
evals.metrics 纯函数；citation_coverage 用 loop/metrics.compute_citation_
coverage（RAG source_count=len(hits)，loop 取 run-metrics 落盘值）。
主循环异常时已累积行先原子写入同目录 ``results.partial.jsonl``（同 schema）
再退出非 0，错误信息注明部分结果路径与已保存行数。
--out 缺省 evals/results；结果写 <out>/<provider>_<UTC时间戳>/。

── 怎么跑 ─────────────────────────────────────────────────────────────
  # 零 key 零网络：mock 全链路（2 题 × 4 条件）
  uv run --no-sync python scripts/run_eval.py --provider mock --limit 2
  # 全题集（30 题）
  uv run --no-sync python scripts/run_eval.py --provider mock
  # 真实测量：config.toml 的 strong/cheap/judge（真实费用）
  uv run --no-sync python scripts/run_eval.py --provider real --conditions c1,c4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.parse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ---- 同目录复用 cold_warm_smoke（scripts/ 不是包，先把目录挂上 sys.path）----
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import cold_warm_smoke as cws  # noqa: E402  -- 挂 path 后才能 import 兄弟脚本
import httpx  # noqa: E402

from researchwiki.env import load_env_file  # noqa: E402
from researchwiki.evals.corpus import CorpusIndex, FixtureSearch, load_corpus  # noqa: E402
from researchwiki.evals.judge import build_judge_provider, judge_answer  # noqa: E402
from researchwiki.evals.metrics import (  # noqa: E402
    aggregate_rows,
    normalized_em,
    point_hit,
    refusal_detected,
)
from researchwiki.evals.qa import QaItem, load_qa  # noqa: E402
from researchwiki.evals.rag_harness import run_rag  # noqa: E402
from researchwiki.llm.accounting import TokenAccountant  # noqa: E402
from researchwiki.llm.provider import MockProvider, Provider, TokenUsage  # noqa: E402
from researchwiki.llm.router import ModelRouter  # noqa: E402
from researchwiki.loop.agent_loop import AgentLoop  # noqa: E402
from researchwiki.loop.metrics import compute_citation_coverage  # noqa: E402
from researchwiki.tools import atomic_write_text  # noqa: E402
from researchwiki.tools.search import SearchHit  # noqa: E402
from researchwiki.wiki.embeddings import (  # noqa: E402
    MockEmbeddingProvider,
    get_embedding_provider,
)

PROJECT_ROOT = cws.PROJECT_ROOT
DEFAULT_QA = PROJECT_ROOT / "evals" / "qa" / "ai-frameworks.jsonl"
DEFAULT_FIXTURES = PROJECT_ROOT / "evals" / "fixtures" / "ai-frameworks"
DEFAULT_OUT = PROJECT_ROOT / "evals" / "results"
DEFAULT_CONFIG = cws.DEFAULT_CONFIG
DEFAULT_ENV_FILE = cws.DEFAULT_ENV_FILE

EXIT_PASS = 0
EXIT_RUN_ERROR = 2

# ---- 条件矩阵 ---------------------------------------------------------------

CONDITION_C1 = "c1"
CONDITION_C2 = "c2"
CONDITION_C3 = "c3"
CONDITION_C4 = "c4"
CONDITIONS: tuple[str, ...] = (CONDITION_C1, CONDITION_C2, CONDITION_C3, CONDITION_C4)
#: RAG 条件 → run_rag 的检索 mode
RAG_MODES: dict[str, str] = {CONDITION_C2: "vector", CONDITION_C3: "hybrid"}

CONDITION_NOTES: dict[str, str] = {
    CONDITION_C1: "C1 无 Memory（每题一次性 root，prior/记忆更新/formation 关闭）",
    CONDITION_C2: "C2 Vector RAG（run_rag mode=vector，300 字硬约束）",
    CONDITION_C3: "C3 Hybrid RAG（run_rag mode=hybrid，300 字硬约束）",
    CONDITION_C4: "C4 ExternalMemory warm（seed 播种 + 每题副本，prior/记忆更新/formation 开）",
}

# ---- 口径常量（裁定留档）------------------------------------------------------

#: C1/C4 的 retrieval_config：强制 deep（检索深度限额与 seed 一致）+ 报告 brief
#: 样式（回答长度近似对齐，非硬约束——manifest.honest_boundary 声明）。
LOOP_RETRIEVAL_CONFIG: dict[str, Any] = {
    "enabled": True,
    "forced_mode": "deep",
    "deep": {"report_style": "brief"},
}
#: C4 播种 run 的 retrieval_config：仅强制 deep（seed 不计分，报告样式不约束）。
SEED_RETRIEVAL_CONFIG: dict[str, Any] = {"enabled": True, "forced_mode": "deep"}
#: C1 的 prior_config：显式关闭（每题一次性 root，Prior 无从命中）。
C1_PRIOR_CONFIG: dict[str, Any] = {"enabled": False}

#: fixture URL 映射前缀：fixture://<domain>/<doc_id> → 前缀 + <domain>/<doc_id>.md
FIXTURE_HTTP_PREFIX = "http://fixture.local/"
#: 行级诚实边界声明（C1/C4 报告长度口径）
BRIEF_BOUNDARY_NOTE = "C1/C4 报告长度经 brief 样式近似对齐，非硬约束"

# ---- mock 剧本补充件（plan/summary/report 沿用 cold_warm 常量）----------------

#: mock RAG 生成脚本的占位回答（带 [1] 引用，让 RAG 行的 citation_coverage 可算）
MOCK_RAG_ANSWER = "（mock 模式）占位回答：依据资料[1]的要点作答。"


# ---- fixture URL 改写与本地回放 ---------------------------------------------


def rewrite_fixture_url(url: str) -> str:
    """fixture://<domain>/<doc_id> → http://<doc_id>.md 形态（fetch_url 可抓取）。"""
    if url.startswith("fixture://"):
        return FIXTURE_HTTP_PREFIX + url[len("fixture://") :] + ".md"
    return url


class FixtureHttpSearch:
    """搜索 provider 包装：FixtureSearch 命中的 fixture:// URL 改写为可 fetch 的 http URL。

    AgentLoop 的 web_search 把命中 URL 交给 LLM，LLM 再用 fetch_url（httpx）抓取
    ——fixture:// 不是 http 方案，httpx 无法处理。改写后 :func:`make_fixture_fetch_
    transport` 的 MockTransport 按 path 回放语料 markdown，fetch_url 走真实链路
    （trafilatura 提取 + sources/ 快照落盘）。``calls`` 透传内层 FixtureSearch 的
    调用序列（口径与 MockSearch 惯例一致，供测试与统计）。
    """

    def __init__(self, inner: FixtureSearch) -> None:
        self._inner = inner

    @property
    def calls(self) -> list[tuple[str, int]]:
        return self._inner.calls

    def search(self, query: str, max_results: int = 5) -> list[SearchHit]:
        return [
            SearchHit(title=h.title, url=rewrite_fixture_url(h.url), snippet=h.snippet)
            for h in self._inner.search(query, max_results=max_results)
        ]


def make_fixture_fetch_transport(fixture_dir: str | Path) -> httpx.MockTransport:
    """httpx.MockTransport：按 URL path 回放语料 markdown（fetch_url 真实链路零网络）。

    同时兜住 Jina Reader 降级路径（https://r.jina.ai/http://fixture.local/...）；
    未登记的 URL 返回 404（fetch_url 会抛 FetchError，工具层如实回传错误）。
    """
    root = Path(fixture_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    docs: dict[str, str] = {}
    for doc in manifest["docs"]:
        doc_id = str(doc["doc_id"])
        docs[doc_id] = (root / "docs" / f"{doc_id}.md").read_text(encoding="utf-8")

    def doc_id_from_url(url: str) -> str | None:
        pos = url.find(FIXTURE_HTTP_PREFIX)
        if pos < 0:
            return None
        rest = url[pos + len(FIXTURE_HTTP_PREFIX) :].split("?", 1)[0].split("#", 1)[0]
        filename = rest.rsplit("/", 1)[-1]
        if not filename.endswith(".md"):
            return None
        return urllib.parse.unquote(filename[: -len(".md")])

    def handler(request: httpx.Request) -> httpx.Response:
        doc_id = doc_id_from_url(str(request.url))
        body = docs.get(doc_id) if doc_id else None
        if body is None:
            return httpx.Response(404, text="fixture doc not found")
        return httpx.Response(
            200, text=body, headers={"content-type": "text/markdown; charset=utf-8"}
        )

    return httpx.MockTransport(handler)


def top_fixture_http_url(index: CorpusIndex, query: str) -> str | None:
    """预计算本轮检索 top 命中的改写 URL（剧本 fetch 轮与蒸馏笔记来源共用）。

    与注入 loop 的 FixtureSearch(index, "hybrid") 同参检索（max_results=5），
    命中完全确定；直接查 index 不经 FixtureSearch 包装，不污染 calls 计数。
    空语料返回 None（剧本退化为无 fetch 轮、笔记不带来源）。
    """
    hits = index.search(query, max_results=5, mode="hybrid")
    if not hits:
        return None
    return rewrite_fixture_url(hits[0].url)


# ---- mock 剧本构造（cold_warm 的事件助手同款）--------------------------------


def mock_note_text(question: str, qid: str) -> str:
    """mock 蒸馏笔记正文：嵌入问题原文（Prior 检索 FTS+向量双通道必然命中，
    cold_warm_smoke 同款保证）与 qid 标记（多条笔记互不相同，避免入库去重合并）。"""
    return (
        f"关于「{question}」的研究结论（{qid}）：Letta 用后台 subagent 在会话空闲期"
        "整理长期记忆，跨会话保留长期事实。"
    )


def mock_distill_json(
    notes: Sequence[tuple[str, str, Sequence[str]]], source_url: str | None
) -> str:
    """mock 蒸馏脚本输出：notes = [(text, qid, entities)]。

    source_url 必须是本轮 web_search 结果里的 URL（改写后的 fixture http URL）：
    CandidateNote.from_dict 会把 source_urls 过滤到本轮来源池内，formation 的
    require_source_for_knowledge 才会放行知识类候选。source_url 为 None（空语料
    兜底）时笔记不带来源——C4 的种子笔记会被 formation 拒绝，属降级路径。
    """
    urls = [source_url] if source_url else []
    return json.dumps(
        {
            "notes": [
                {
                    "text": text,
                    "entities": list(entities) or ["Letta"],
                    "confidence": "high",
                    "source_urls": urls,
                }
                for text, _qid, entities in notes
            ],
            "conflicts": [],
        },
        ensure_ascii=False,
    )


def make_mock_router(
    *,
    search_query: str,
    fetch_url_value: str | None,
    distill_json: str,
) -> cws.ScriptedTierRouter:
    """一次 run 用的全新脚本化 router（cold_warm.make_mock_router 的评测版）。

    strong 4–5 轮：plan → web_search → fetch_url（语料可抓取时）→ 研究总结 →
    报告（与 adaptive_smoke 的 deep 剧本同形）；cheap 1 轮：蒸馏。每份剧本最后
    一轮都是纯文本——ScriptedProvider 耗尽后重复最后轮，任何限额下都不会死循环。
    """
    turns: list[list[Any]] = [
        cws._text_turn(cws.MOCK_PLAN_TEXT),
        cws._tool_turn(
            [cws._call("web_search", {"query": search_query}, call_id="eval-search")]
        ),
    ]
    if fetch_url_value:
        turns.append(
            cws._tool_turn(
                [cws._call("fetch_url", {"url": fetch_url_value}, call_id="eval-fetch")]
            )
        )
    turns.append(cws._text_turn(cws.MOCK_SUMMARY_TEXT))
    turns.append(cws._text_turn(cws.MOCK_REPORT_TEXT))
    strong = cws.ScriptedProvider(
        turns,
        tier="strong",
        model="mock-strong",
        usage=TokenUsage(input_tokens=1200, output_tokens=180),
    )
    cheap = cws.ScriptedProvider(
        [cws._text_turn(distill_json)],
        tier="cheap",
        model="mock-cheap",
        usage=TokenUsage(input_tokens=1200, output_tokens=180),
    )
    return cws.ScriptedTierRouter(strong, cheap)


def make_mock_rag_provider() -> cws.ScriptedProvider:
    """mock 模式 RAG 行的生成 provider（与 loop 条件同形：strong 档、同 usage 口径）。"""
    return cws.ScriptedProvider(
        [cws._text_turn(MOCK_RAG_ANSWER)],
        tier="strong",
        model="mock-strong",
        usage=TokenUsage(input_tokens=1200, output_tokens=180),
    )


# ---- 共享依赖与 run 执行 -----------------------------------------------------


@dataclass
class EvalDeps:
    """一次评测的共享依赖（四条件同口径：同一语料索引 / embedding / fetch 服务）。

    - ``router``：real 模式的 ModelRouter（无状态，可跨 run 复用）；mock 为 None；
    - ``formation_base`` / ``memory_update_base`` / ``verification_base``：C4 用
      的三段配置底座（mock=内置最小等价配置；real=config.toml 对应段）——C4 语义
      要求记忆演化开启，主流程用 :func:`enabled_section` 强制 enabled=True；
    - ``_top_urls``：top_fixture_http_url 的按查询缓存（同一问题在 C1/C4/种子
      之间复用同一预计算结果）。
    """

    provider_mode: str
    index: CorpusIndex
    embedding: Any
    wiki_config: Mapping[str, Any] | None
    llm_config: Mapping[str, Any] | None
    router: Any | None
    judge_provider: Provider
    fetch_transport: Any
    formation_base: Mapping[str, Any] | None
    memory_update_base: Mapping[str, Any] | None
    verification_base: Mapping[str, Any] | None
    corpus_note: str
    _top_urls: dict[str, str | None] = field(default_factory=dict)

    def top_url(self, query: str) -> str | None:
        if query not in self._top_urls:
            self._top_urls[query] = top_fixture_http_url(self.index, query)
        return self._top_urls[query]


def enabled_section(section: Mapping[str, Any] | None) -> dict[str, Any]:
    """C4 的三段配置：以用户段为底、强制 enabled=True（C4 语义要求记忆演化开启）。"""
    cfg = dict(section or {})
    cfg["enabled"] = True
    return cfg


def execute_loop(
    deps: EvalDeps,
    *,
    label: str,
    question: str,
    wiki_root: Path,
    prior_config: Mapping[str, Any] | None,
    retrieval_config: Mapping[str, Any] | None,
    formation_config: Mapping[str, Any] | None,
    memory_update_config: Mapping[str, Any] | None,
    verification_config: Mapping[str, Any] | None,
    distill_notes: Sequence[tuple[str, str, Sequence[str]]],
) -> tuple[AgentLoop, dict[str, Any], TokenAccountant]:
    """跑一次完整 AgentLoop（构造照抄 cold_warm_smoke.run_once 的参数集）。

    检索 provider = FixtureHttpSearch(FixtureSearch(index, "hybrid"))（fixture URL
    改写）；fetch 经 MockTransport 回放语料 markdown（fetch_url 真实链路写快照）。
    返回 (loop, run-metrics.json 落盘原文, accountant)——供行构造与 judge 记账。
    """
    wiki_root.mkdir(parents=True, exist_ok=True)
    accountant = TokenAccountant(path=wiki_root / "tokens.jsonl")
    if deps.provider_mode == "mock":
        router: Any = make_mock_router(
            search_query=question,
            fetch_url_value=deps.top_url(question),
            distill_json=mock_distill_json(distill_notes, deps.top_url(question)),
        )
    else:
        router = deps.router
    loop = AgentLoop(
        question,
        router=router,
        llm_config=deps.llm_config,
        accountant=accountant,
        search_provider=FixtureHttpSearch(FixtureSearch(deps.index, "hybrid")),
        wiki_root=wiki_root,
        embedding=deps.embedding,
        wiki_config=deps.wiki_config,
        prior_config=prior_config,
        retrieval_config=retrieval_config,
        formation_config=formation_config,
        memory_update_config=memory_update_config,
        verification_config=verification_config,
        fetch_transport=deps.fetch_transport,
    )
    for event in loop.events():
        if event.get("type") == "data-note":
            data = event.get("data") or {}
            print(
                f"  [{label}] note {data.get('id')} conf={data.get('confidence')}",
                flush=True,
            )
    metrics = json.loads((loop.run_dir / "run-metrics.json").read_text(encoding="utf-8"))
    return loop, metrics, accountant


# ---- 结果行构造 --------------------------------------------------------------


def _judge_payload(verdict: Any) -> dict[str, Any] | None:
    """JudgeVerdict → 行内 JSON（全字段含 raw 留档）；无评审分（解析失败）→ None。"""
    if verdict is None:
        return None
    return asdict(verdict)


def build_row(
    *,
    item: QaItem,
    condition: str,
    answer: str,
    judge_verdict: Any,
    metrics: Mapping[str, Any],
    trace_id: str,
    wiki_root: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: float | int,
    citation_coverage: float | None,
    corpus_note: str,
) -> dict[str, Any]:
    """一行 results.jsonl（字段契约见模块 docstring 的输出契约）。

    em/point_hits/refusal 用 evals.metrics 纯函数现算；fresh_search_count 仅
    loop 条件携带（RAG 行不写该键——对齐 aggregate_rows 的缺失语义，不写 None）。
    """
    row: dict[str, Any] = {
        "qid": item.qid,
        "qtype": item.qtype,
        "condition": condition,
        "answer": answer,
        "gold_points": list(item.gold_points),
        "em": normalized_em(answer, item.gold_points),
        "point_hits": sum(1 for gold in item.gold_points if point_hit(answer, gold)),
        "refusal": refusal_detected(answer),
        "citation_coverage": citation_coverage,
        "judge": _judge_payload(judge_verdict),
        "metrics": dict(metrics),
        "trace_id": trace_id,
        "wiki_root": wiki_root,
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "latency_ms": latency_ms,
        "provider_note": corpus_note,
    }
    if condition in (CONDITION_C1, CONDITION_C4):
        note = CONDITION_NOTES[condition]
        row["provider_note"] = f"{corpus_note}；{note}；{BRIEF_BOUNDARY_NOTE}"
        row["fresh_search_count"] = int(metrics.get("fresh_search_count") or 0)
    elif condition in RAG_MODES:
        row["provider_note"] = f"{corpus_note}；{CONDITION_NOTES[condition]}"
    return row


def run_loop_row(
    deps: EvalDeps,
    *,
    condition: str,
    item: QaItem,
    wiki_root: Path,
    prior_config: Mapping[str, Any] | None,
    retrieval_config: Mapping[str, Any] | None,
    formation_config: Mapping[str, Any] | None,
    memory_update_config: Mapping[str, Any] | None,
    verification_config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """跑一个 loop 条件（C1/C4）的一次 run + 立即 judge，构造结果行。

    judge sources 传 SourcePool.numbered()（与报告 [n] 编号同源）；judge 记账用
    独立 trace（<trace_id>-judge）——不污染主 trace 的 sum_tokens_from_jsonl 对账。
    """
    loop, metrics, accountant = execute_loop(
        deps,
        label=f"{condition}/{item.qid}",
        question=item.question,
        wiki_root=wiki_root,
        prior_config=prior_config,
        retrieval_config=retrieval_config,
        formation_config=formation_config,
        memory_update_config=memory_update_config,
        verification_config=verification_config,
        distill_notes=[(mock_note_text(item.question, item.qid), item.qid, item.entities)],
    )
    verdict = judge_answer(
        deps.judge_provider,
        question=item.question,
        gold_points=item.gold_points,
        answer=loop.report_text,
        sources=loop.ctx.source_pool.numbered(),
        accountant=accountant,
        trace_id=f"{loop.trace_id}-judge",
    )
    return build_row(
        item=item,
        condition=condition,
        answer=loop.report_text,
        judge_verdict=verdict,
        metrics=metrics,
        trace_id=loop.trace_id,
        wiki_root=str(wiki_root),
        input_tokens=int(metrics.get("input_tokens") or 0),
        output_tokens=int(metrics.get("output_tokens") or 0),
        latency_ms=int(metrics.get("latency_ms") or 0),
        citation_coverage=metrics.get("citation_coverage"),
        corpus_note=deps.corpus_note,
    )


def run_rag_row(
    deps: EvalDeps,
    *,
    condition: str,
    item: QaItem,
    provider: Provider,
    accountant: TokenAccountant,
) -> dict[str, Any]:
    """跑一个 RAG 条件（C2/C3）的一次 run_rag + 立即 judge，构造结果行。

    sources 传检索命中的 {url, title} 列表（原始 fixture:// URL——RAG 不经
    fetch，无改写问题）；记账集中在 <out>/tokens.jsonl（judge 同文件独立 trace）。
    """
    mode = RAG_MODES[condition]
    trace_id = accountant.new_trace_id()
    result = run_rag(
        provider,
        deps.index,
        question=item.question,
        mode=mode,
        accountant=accountant,
        trace_id=trace_id,
    )
    verdict = judge_answer(
        deps.judge_provider,
        question=item.question,
        gold_points=item.gold_points,
        answer=result.answer,
        sources=[{"url": h.url, "title": h.title} for h in result.hits],
        accountant=accountant,
        trace_id=f"{trace_id}-judge",
    )
    coverage = compute_citation_coverage(result.answer, len(result.hits))
    if coverage is not None:
        coverage = round(coverage, 4)
    metrics = {
        "mode": mode,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "latency_ms": round(result.latency_ms, 1),
        "hit_count": len(result.hits),
    }
    return build_row(
        item=item,
        condition=condition,
        answer=result.answer,
        judge_verdict=verdict,
        metrics=metrics,
        trace_id=trace_id,
        wiki_root="",
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        latency_ms=metrics["latency_ms"],
        citation_coverage=coverage,
        corpus_note=deps.corpus_note,
    )


# ---- C4 播种 ----------------------------------------------------------------


def default_seed_question(items: Sequence[QaItem]) -> str:
    """缺省种子问题：题集第一道 multi_hop 题；无 multi_hop（如 --limit 截断）回退首题。"""
    for item in items:
        if item.qtype == "multi_hop":
            return item.question
    return items[0].question


def copy_seed_into(root: Path, seed_root: Path) -> None:
    """播种根 → 每题独立副本（warm 起跑、互不污染）。

    排除 index.db*：播种 run 的 SearchIndex 连接可能仍存活（Windows 文件锁 /
    WAL 尾巴）；warm run 开头的 ensure_index_fresh 会探测到索引落后并按 store
    整体重建——检索口径不变（adaptive_smoke.copy_seed_into 同款先例）。
    """
    if root.exists():
        shutil.rmtree(root)
    shutil.copytree(
        seed_root,
        root,
        ignore=shutil.ignore_patterns("index.db", "index.db-wal", "index.db-shm"),
    )


def run_seed_phase(
    deps: EvalDeps,
    *,
    items: Sequence[QaItem],
    seed_root: Path,
    seed_question: str,
    seed_runs: int,
) -> None:
    """C4 播种：在 seed_root 连续跑 N 次 forced deep 的完整研究。

    mock 模式的种子蒸馏脚本覆盖**题集全部题目**（每题一条笔记、正文嵌入问题
    原文、来源指向种子 run 自己的检索 top 命中）——保证每个 warm 副本的 Prior
    检索都能命中对应笔记。播种 run 不产 results 行（它不是评测题）。
    """
    seed_root.mkdir(parents=True, exist_ok=True)
    for i in range(max(1, seed_runs)):
        label = f"seed-{i + 1}/{seed_runs}"
        print(f"[{label}] run 开始（forced deep，question={seed_question}）", flush=True)
        _loop, metrics, _accountant = execute_loop(
            deps,
            label=label,
            question=seed_question,
            wiki_root=seed_root,
            prior_config=None,  # 缺省 = enabled + 默认（seed 根为空，冷启动播种）
            retrieval_config=SEED_RETRIEVAL_CONFIG,
            formation_config=enabled_section(deps.formation_base),
            memory_update_config=enabled_section(deps.memory_update_base),
            verification_config=deps.verification_base,
            distill_notes=[
                (mock_note_text(item.question, item.qid), item.qid, item.entities)
                for item in items
            ],
        )
        print(
            f"[{label}] trace={metrics.get('trace_id', '')}  "
            f"notes(created/merged)={metrics.get('notes_created')}/{metrics.get('notes_merged')}  "
            f"in/out={metrics.get('input_tokens')}/{metrics.get('output_tokens')}",
            flush=True,
        )


# ---- 入口 -------------------------------------------------------------------


def conditions_arg(raw: str) -> list[str]:
    """--conditions 解析：逗号分隔、去重保序、非法值在 argparse 层拒绝（exit 2）。"""
    conds = [c.strip().lower() for c in str(raw).split(",") if c.strip()]
    invalid = [c for c in conds if c not in CONDITIONS]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"--conditions 只支持 {'/'.join(CONDITIONS)}，得到 {invalid!r}"
        )
    if not conds:
        raise argparse.ArgumentTypeError("--conditions 不能为空")
    return list(dict.fromkeys(conds))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="评测四条件 runner（C1/C2/C3/C4，详见文件头 docstring）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--qa", default=str(DEFAULT_QA), help="题集 JSONL 路径")
    parser.add_argument("--fixtures", default=str(DEFAULT_FIXTURES), help="受控语料 fixture 目录")
    parser.add_argument(
        "--conditions", type=conditions_arg, default=",".join(CONDITIONS),
        help="条件矩阵（逗号分隔，缺省 c1,c2,c3,c4）",
    )
    parser.add_argument(
        "--provider", choices=("mock", "real"), default="mock",
        help="mock=零网络零 key 的剧本全链路（默认）；real=走 config.toml 真实档位",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="config.toml 路径")
    parser.add_argument(
        "--env-file", default=str(DEFAULT_ENV_FILE),
        help=".env 路径（仅 real 模式读取；传空串跳过）",
    )
    parser.add_argument(
        "--seed-question", default="",
        help="C4 播种问题（缺省取题集第一道 multi_hop 题，无 multi_hop 回退首题）",
    )
    parser.add_argument("--seed-runs", type=int, default=1, help="C4 播种 run 次数（默认 1）")
    parser.add_argument(
        "--limit", type=int, default=0,
        help="只取题集前 N 题（0 = 全部；冒烟/测试用）",
    )
    parser.add_argument(
        "--out", default=str(DEFAULT_OUT),
        help="输出根目录（默认 evals/results；结果写 <out>/<provider>_<ts>/）",
    )
    return parser


def _model_info(provider: Any) -> dict[str, str]:
    return {
        "model": str(getattr(provider, "model", "") or ""),
        "base_url": str(getattr(provider, "base_url", "") or ""),
    }


def mock_judge_error(judge_provider: Provider) -> str | None:
    """real 模式的 judge 硬校验：judge provider 为 MockProvider 时返回错误文案。

    [llm.judge] 缺段或 base_url 空时 build_judge_provider 回落 MockProvider——
    其占位分数无意义，在 real 模式冒充真实评测会污染全部 judge 字段。返回
    中文错误文案（main 打印后以非 0 退出、零落盘）；真实端点 provider 返回
    None（放行）。
    """
    if isinstance(judge_provider, MockProvider):
        return (
            "real 模式要求 [llm.judge].base_url 非空（评审必须强模型）；"
            "如确需 mock judge 跑管线请用 --provider mock"
        )
    return None


def _fmt_metric(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def format_condition_summary(rows: Sequence[Mapping[str, Any]]) -> str:
    """按条件聚合的人类可读汇总（aggregate_rows 的直读视图 + 防误读声明）。"""
    summary = aggregate_rows(list(rows))
    lines = ["── 条件聚合（aggregate_rows）──"]
    for condition in CONDITIONS:
        s = summary.get(condition)
        if s is None:
            continue
        cells = (
            f"n={s['n']}  em_mean={s['em_mean']:.4f}  refusal_rate={s['refusal_rate']:.4f}"
            f"  latency_p50={s['latency_p50']:.0f}ms"
            f"  in_tok={s['input_tokens_mean']:.0f}  out_tok={s['output_tokens_mean']:.0f}"
        )
        if "fresh_search_mean" in s:
            cells += f"  fresh_search_mean={s['fresh_search_mean']:.2f}"
        lines.append(f"  {condition:<4}{cells}")
    lines.append(
        "注：小样本评测，不构成收益结论（PLAN §11.1）；条件口径差异见 manifest.json 的"
        " honest_boundary 与 fixture_url_mapping。"
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001 -- 老环境/被替换的 stdout 上忽略
        pass
    args = build_parser().parse_args(argv)
    conditions: list[str] = list(args.conditions)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out) / f"{args.provider}_{stamp}"

    config: dict[str, Any] = {}
    if args.provider == "real":
        if args.env_file:
            load_env_file(args.env_file)
        config = cws.load_config(Path(args.config))
        print(
            "⚠ 成本警告：real 模式将用真实模型完整跑（题数 × 条件数 + C4 播种）次"
            "研究与评审，会产生真实 API 费用；请确认 key 与配额后再继续。"
        )

    items = load_qa(args.qa)
    if args.limit > 0:
        items = items[: args.limit]
    if not items:
        print("✗ 题集为空，无法评测")
        return EXIT_RUN_ERROR

    # ---- 依赖组装（四条件同口径：同一生成 provider / embedding / 语料 / fetch 服务）----
    if args.provider == "mock":
        embedding: Any = MockEmbeddingProvider(dim=512)
        wiki_config: Mapping[str, Any] | None = {"fts_tokenizer": "trigram"}
        llm_config: Mapping[str, Any] | None = {
            "strong": {"base_url": "https://mock"},
            "cheap": {"base_url": "https://mock-cheap"},
        }
        router: ModelRouter | None = None
        judge_provider = build_judge_provider(None)
        formation_base: Mapping[str, Any] | None = cws.MOCK_FORMATION_CONFIG
        memory_update_base: Mapping[str, Any] | None = cws.MOCK_MEMORY_UPDATE_CONFIG
        verification_base: Mapping[str, Any] | None = cws.MOCK_VERIFICATION_CONFIG
        model_summary = {
            "strong": {"model": "mock-strong", "base_url": ""},
            "cheap": {"model": "mock-cheap", "base_url": ""},
            "judge": {"model": "mock-judge", "base_url": ""},
        }
    else:
        # 同一 embedding 配置贯穿语料索引与全部 loop（缓存关闭：语料索引只算一次，
        # 各 wiki 根单次 run 无复算收益；vector 口径由模型决定，与缓存无关）
        embedding = get_embedding_provider(config, cache_path=None)
        wiki_config = config.get("wiki") or {}
        llm_config = config.get("llm") or {}
        router = ModelRouter(dict(llm_config))
        judge_provider = build_judge_provider(config)
        formation_base = config.get("formation")
        memory_update_base = config.get("memory_update")
        verification_base = config.get("verification")
        model_summary = {
            "strong": _model_info(router.get("strong")),
            "cheap": _model_info(router.get("cheap")),
            "judge": _model_info(judge_provider),
        }

    # real 模式 judge 硬校验（在创建任何输出目录之前：校验失败零落盘）
    if args.provider == "real":
        judge_error = mock_judge_error(judge_provider)
        if judge_error:
            print(f"✗ {judge_error}")
            return EXIT_RUN_ERROR

    out_dir.mkdir(parents=True, exist_ok=True)

    fixture_dir = Path(args.fixtures)
    index = load_corpus(fixture_dir, embedding=embedding)
    corpus_manifest = json.loads((fixture_dir / "manifest.json").read_text(encoding="utf-8"))
    deps = EvalDeps(
        provider_mode=args.provider,
        index=index,
        embedding=embedding,
        wiki_config=wiki_config,
        llm_config=llm_config,
        router=router,
        judge_provider=judge_provider,
        fetch_transport=make_fixture_fetch_transport(fixture_dir),
        formation_base=formation_base,
        memory_update_base=memory_update_base,
        verification_base=verification_base,
        corpus_note=str(corpus_manifest.get("provider_note") or ""),
    )

    seed_question = str(args.seed_question).strip() or default_seed_question(items)
    seed_root = out_dir / "seed"
    manifest: dict[str, Any] = {
        "gate": "run_eval",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "provider_mode": args.provider,
        "model": model_summary,
        "qa_path": str(Path(args.qa).resolve()),
        "qa_sha256": hashlib.sha256(Path(args.qa).read_bytes()).hexdigest(),
        "qa_count": len(items),
        "limit": args.limit or None,
        "fixtures_dir": str(fixture_dir.resolve()),
        "corpus_provider_note": deps.corpus_note,
        "fixture_url_mapping": (
            "loop 条件（C1/C4）的检索命中 URL 经 fixture://<domain>/<doc_id> → "
            f"{FIXTURE_HTTP_PREFIX}<domain>/<doc_id>.md 改写后由 httpx.MockTransport "
            "本地回放语料 markdown（fetch_url 真实链路写 sources/ 快照），来源池记录"
            "改写后 URL；RAG 条件（C2/C3）不经 fetch，记录原始 fixture:// URL。"
            "各条件内报告 [n] 引用编号与来源池一一对应、自洽。"
        ),
        "conditions": conditions,
        "condition_notes": {c: CONDITION_NOTES[c] for c in conditions},
        "retrieval_config_loop": LOOP_RETRIEVAL_CONFIG,
        "honest_boundary": (
            "C2/C3 的 300 字约束来自 RAG_SYSTEM（硬约束）；C1/C4 的报告长度经 brief "
            "样式近似对齐（retrieval_config 强制 deep + deep.report_style=brief，"
            "深检索限额不变），非硬约束——agent 条件的回答结构与 RAG 不同，长度对齐"
            "是近似口径。"
        ),
        "seed": {
            "question": seed_question if CONDITION_C4 in conditions else None,
            "runs": args.seed_runs if CONDITION_C4 in conditions else 0,
            "root": str(seed_root) if CONDITION_C4 in conditions else None,
        },
        "rag_tokens_path": str(out_dir / "tokens.jsonl"),
        "judge_trace_suffix": "-judge",
        "embedding_model": str(getattr(embedding, "model", "") or ""),
    }

    strong = model_summary["strong"]
    judge = model_summary["judge"]
    print("=== 评测四条件 runner（C1 无 Memory / C2 Vector / C3 Hybrid / C4 warm）===")
    print(f"provider    : {args.provider}")
    print(
        f"model       : strong={strong['model']}（{strong['base_url'] or '—'}）  "
        f"judge={judge['model']}（{judge['base_url'] or '—'}）"
    )
    print(f"qa          : {args.qa}（{len(items)} 题" + (f"，limit={args.limit}" if args.limit else "") + "）")
    print(f"conditions  : {', '.join(conditions)}")
    print(f"seed        : {seed_question if CONDITION_C4 in conditions else '—（未选 C4）'}")
    print(f"out         : {out_dir}")
    print("-" * 76, flush=True)

    rows: list[dict[str, Any]] = []
    try:
        if CONDITION_C4 in conditions:
            run_seed_phase(
                deps,
                items=items,
                seed_root=seed_root,
                seed_question=seed_question,
                seed_runs=args.seed_runs,
            )

        rag_accountant = TokenAccountant(path=out_dir / "tokens.jsonl")
        for item in items:
            for condition in conditions:
                if condition in RAG_MODES:
                    provider: Provider = (
                        make_mock_rag_provider()
                        if deps.provider_mode == "mock"
                        else deps.router.get("strong")  # type: ignore[union-attr]
                    )
                    row = run_rag_row(
                        deps, condition=condition, item=item, provider=provider,
                        accountant=rag_accountant,
                    )
                elif condition == CONDITION_C1:
                    row = run_loop_row(
                        deps,
                        condition=condition,
                        item=item,
                        wiki_root=out_dir / f"c1_{item.qid}",
                        prior_config=C1_PRIOR_CONFIG,
                        retrieval_config=LOOP_RETRIEVAL_CONFIG,
                        formation_config=None,
                        memory_update_config=None,
                        verification_config=None,
                    )
                else:  # c4
                    root = out_dir / f"c4_{item.qid}"
                    copy_seed_into(root, seed_root)
                    row = run_loop_row(
                        deps,
                        condition=condition,
                        item=item,
                        wiki_root=root,
                        prior_config=None,  # 缺省 = enabled + 默认
                        retrieval_config=LOOP_RETRIEVAL_CONFIG,
                        formation_config=enabled_section(deps.formation_base),
                        memory_update_config=enabled_section(deps.memory_update_base),
                        verification_config=deps.verification_base,
                    )
                rows.append(row)
                print(
                    f"[{item.qid}/{condition}] em={row['em']:.2f} "
                    f"point_hits={row['point_hits']} refusal={row['refusal']} "
                    f"cov={_fmt_metric(row['citation_coverage'])} "
                    f"in/out={row['input_tokens']}/{row['output_tokens']}",
                    flush=True,
                )
    except Exception as exc:  # noqa: BLE001 -- 环境/网络问题不作数，明确退出而非裸栈
        # 部分行落盘（终审修复波）：已累积的结果不随异常蒸发——先原子写入
        # results.partial.jsonl（同 schema）再退出非 0，错误信息注明路径与行数。
        partial_path = out_dir / "results.partial.jsonl"
        atomic_write_text(
            partial_path,
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        )
        print(f"✗ 运行异常：{type(exc).__name__}: {exc}")
        print(
            f"（已完成 {len(rows)} 行已部分落盘 {partial_path}；"
            "real 模式请检查 key / base_url / 网络；mock 模式不应出现本行，视为 bug）"
        )
        return EXIT_RUN_ERROR

    results_path = out_dir / "results.jsonl"
    results_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("-" * 76)
    print(format_condition_summary(rows))
    print(f"results.jsonl 已写入 {results_path}")
    print(f"manifest.json 已写入 {manifest_path}")
    return EXIT_PASS


if __name__ == "__main__":
    raise SystemExit(main())
