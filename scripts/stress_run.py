#!/usr/bin/env python
"""连续问题流压测：单持久 wiki root + RunWatchdog 看门狗（P4b T8 阶段验收件）。

── 用途（PLAN §3.5 退出条件的证据链落点）──────────────────────────────────
单持久 wiki root 上连续问题流驱动 AgentLoop（每问题一个 run、共享同一 root），
同时锻炼 compaction（[context]）、checkpoint（[checkpoint]）与记忆演化全链路
（formation / memory_update / verification / unanswered）：

  闸5  连续运行后 lint 指标不恶化 → 每 LINT_EVERY 问拍一次 lint --json 快照并
       逐对比较（口径见下"不恶化口径"）；
  闸6  10+ 连续问题 / 2h 零未解释崩溃 → --questions / --duration 驱动的问题流，
       questions.jsonl 逐问留痕（含错误），看门狗暂停 = 唯一合法的非零收尾；
  闸7  token 曲线在阈值内、compaction 次数/前后 token/保留摘要可审计 → 汇总打印
       per-question tokens（计量）+ compactions-summary.json（跨 run 汇总）；
  闸9  cache 命中率只在 provider 返回 cache 字段时报告 → 当前 air 不返回，
       本脚本输出恒写"未报告：provider 无 cache 字段"，不伪造 cache 数字。

维护执行（consolidate --run，T5）不进本脚本：Global Constraints 规定 job queue
是"进程内可重复执行 + CLI 驱动"。压测结束后对同一 root 执行
  uv run --no-sync python -m researchwiki.cli consolidate --run --root <root>
即得维护动作证据（汇总里会打印这条命令指针）。

── 不恶化口径（写死在此，供评审对照）─────────────────────────────────────
比较相邻两次 lint 快照（基线 → 每 5 问 → final）的三个计数，后 ≤ 前 即"不恶化"：

  - broken_links       len(report.broken_links)     断链
  - orphan_notes       len(report.orphan_notes)     孤立笔记（无页面引用的 active 笔记）
  - dangling_evidence  report.dangling_evidence     无来源/证据链断裂（笔记声明的
                                                    来源快照文件不存在或无 content_hash）

任一计数增大即闸失败（退出码 1）。lint --json 的完整 dict 原样落盘可复核。

孤立指标与 build_pages 的取舍：AgentLoop 默认不建页（build_pages=False）——没有
页面引用时所有 active 笔记都是"孤立"，孤立计数会随问题数线性增长、闸必挂。
故本脚本显式 build_pages=True（每问蒸馏入库后按实体聚合一页；mock 模式页面输出
走骨架页兜底，确定性带笔记 ID 标注）。这是脚本级选择，不改 loop 默认行为。

── 诚实计量 ──────────────────────────────────────────────────────────────
questions.jsonl 的 tokens 来自 **run 的 token JSONL 汇总**（tokens.jsonl 按
trace_id 复算，与 run-metrics.json 可对账）——计量非估算；duration_s 是
wall-clock 计量。为避免 2h 压测下逐问全量扫描 tokens.jsonl 的 O(n²) 成本，
按文件字节偏移增量读取（每问 run 的记账行都是追加在文件尾部，偏移读取与全量
复算同值；tests/test_stress.py 用全量复算交叉验证这一点）。

── 两种 provider 模式（--provider）────────────────────────────────────────
mock（默认）  ScriptedProvider 两档（每问全新实例，剧本按问题定制：plan →
              web_search → fetch_url → 总结 → 报告；cheap 一轮蒸馏 + 一轮页面）
              + 显式锁定的 MockSearch + MockTransport 抓取（零网络、快照照常
              落盘）+ 确定性 MockEmbeddingProvider。全链路零 key 零网络零 API
              成本（结构保证）。问题清单内置 12 问（AI 框架域，与
              evals/qa/ai-frameworks.jsonl 同域），超出后循环使用——同题二轮的
              蒸馏候选与一轮笔记近乎重复，formation 按 near_duplicate_similarity
              拒绝入库（预期语义：防无差别堆积，压测照常锻炼判定链路）。
real          config.toml 全量接线（llm / search / embedding / wiki / prior /
              formation / memory_update / verification / unanswered / context /
              retrieval / checkpoint / freshness），与 server real 通路同一传法。
              会产生真实 API 费用——跑前先做预算估算（九闸第 6 条）。

── 产物（--out 目录，默认 wiki-data/stress/{UTC时间戳}/）───────────────────
  questions.jsonl           每问一行（append + flush，中断保留已完成部分）
  lint-0000.json            基线快照（第 0 问前）
  lint-0005.json …          每 LINT_EVERY 问一次
  lint-final.json           结束时必拍
  compactions-summary.json  跨 run 的 compactions.jsonl 汇总（闸 7）
  summary.json              机器可读汇总（含 lint 闸结论与逐问结果）

── 怎么跑 ────────────────────────────────────────────────────────────────
  # 快速路径（测试/自检，秒级）
  uv run --no-sync python scripts/stress_run.py --questions 3 --provider mock \
      --root wiki-data/stress/quick --out wiki-data/stress/quick-out
  # mock 2 小时压测（零成本，建议后台跑）
  uv run --no-sync python scripts/stress_run.py --provider mock --duration 2h \
      --root wiki-data/stress/mock2h
  # real 压测（先做预算估算，见任务简报九闸第 6 条）
  uv run --no-sync python scripts/stress_run.py --provider real --duration 2h \
      --root wiki-data/stress/real2h --per-question-timeout 600
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
import shutil
import sys
import time
import tomllib
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from researchwiki.env import load_env_file
from researchwiki.llm.accounting import TokenAccountant
from researchwiki.llm.provider import ScriptedProvider, StreamEvent, TokenUsage
from researchwiki.llm.router import ModelRouter
from researchwiki.loop.agent_loop import AgentLoop
from researchwiki.loop.metrics import sum_tokens_from_jsonl
from researchwiki.loop.watchdog import RunWatchdog, WatchdogSettings
from researchwiki.tools import fetch_url, get_search_provider
from researchwiki.wiki.embeddings import MockEmbeddingProvider, get_embedding_provider
from researchwiki.wiki.freshness import from_config as freshness_from_config
from researchwiki.wiki.lint import lint_wiki
from researchwiki.wiki.store import WikiStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config.toml"

# 每 LINT_EVERY 问拍一次 lint 快照（简报口径：每 5 问）
LINT_EVERY = 5

EXIT_PASS = 0
EXIT_ASSERT_FAILED = 1  # lint 不恶化闸失败
EXIT_RUN_ERROR = 2  # 参数错误 / 看门狗暂停 / 零问题执行

# ---- 内置问题清单（AI 框架域，与 evals/qa/ai-frameworks.jsonl 同域；10+ 问）----
# (问题, 实体)：实体进蒸馏笔记（每问一实体，页面按实体聚合并引用其笔记——
# 孤立闸的前提），mock 剧本与 real 模式共用本清单。
STRESS_QUESTIONS: tuple[tuple[str, str], ...] = (
    ("LangChain 的 RunnableRetry 组件默认重试几次？", "LangChain"),
    ("LangGraph 的 checkpoint 持久化组件如何配置？", "LangGraph"),
    ("ConversationSummaryMemory 如何压缩对话历史？", "ConversationSummaryMemory"),
    ("FAISS 的 IndexFlatL2 属于什么类型的检索？", "FAISS"),
    ("Chroma 通过哪个参数配置本地持久化目录？", "Chroma"),
    ("BM25 在 RAG 混合检索中承担什么角色？", "BM25"),
    ("bge-m3 模型的最大输入长度是多少？", "bge-m3"),
    ("ParentDocumentRetriever 的召回与返回机制是什么？", "ParentDocumentRetriever"),
    ("text-embedding-3-large 的 dimensions 参数最低可降到多少？", "text-embedding-3-large"),
    ("AgentExecutor 默认最多迭代多少轮？", "AgentExecutor"),
    ("Milvus 支持哪几档一致性级别？", "Milvus"),
    ("Assistants API 的 tool_choice 设为 auto 是什么含义？", "AssistantsAPI"),
)

# ---- mock 剧本（每问一份，同题循环；与 tests/test_loop.py 的事件造法同构）----
MOCK_FETCH_URL = "https://github.com/letta-ai/letta"  # MockSearch 夹具第一条
MOCK_FETCH_HTML = (
    "<html><body><article>"
    "Letta 通过后台 subagent 在会话间整理记忆（sleep-time compute），"
    "把记忆维护成本移出交互窗口。本文给出工程实践要点与取舍。"
    "</article></body></html>"
)
MOCK_USAGE = TokenUsage(input_tokens=1200, output_tokens=180)

# mock 模式内置的接线配置（mock 不读 config.toml；语义与 config.toml 同形，
# 值取"开着 + 默认阈值"——与 kill_resume_smoke 的 MOCK_*_CONFIG 同口径）：
# [context]/[checkpoint] 开着正是本压测要锻炼的 compaction/checkpoint 链路。
MOCK_FORMATION_CONFIG: dict[str, Any] = {"enabled": True}
MOCK_MEMORY_UPDATE_CONFIG: dict[str, Any] = {"enabled": True, "dry_run": False}
MOCK_VERIFICATION_CONFIG: dict[str, Any] = {}
MOCK_UNANSWERED_CONFIG: dict[str, Any] = {"enabled": True}
MOCK_CONTEXT_CONFIG: dict[str, Any] = {
    "enabled": True,
    "max_context_tokens": 131072,
    "compaction_threshold": 0.7,
    "keep_recent_turns": 4,
}
MOCK_CHECKPOINT_CONFIG: dict[str, Any] = {"enabled": True, "every_step": True}


def _text_turn(text: str) -> list[StreamEvent]:
    return [StreamEvent(type="text_delta", delta=text)]


def _tool_turn(calls: list[dict[str, Any]]) -> list[StreamEvent]:
    return [StreamEvent(type="tool_calls", tool_calls=calls)]


def _call(name: str, arguments: dict[str, Any], *, call_id: str) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
    }


def mock_distill_json(question: str, entity: str) -> str:
    """蒸馏阶段的脚本输出：一条嵌入问题原文的原子笔记，只引用已抓取的来源。

    - 正文嵌问题原文：同题循环时与一轮笔记近乎重复 → formation 拒绝（预期语义）；
    - source_urls 只给 MOCK_FETCH_URL：fetch 已把该 URL 带 content_hash 入来源池，
      笔记 frontmatter 拿到哈希、快照真实落盘 → 悬空证据（无来源闸）保持 0。
    """
    note = (
        f"关于「{question}」的研究结论：{entity} 的关键机制已核实；"
        "Letta 用后台 subagent 在会话空闲期整理长期记忆，跨会话保留长期事实。"
    )
    return json.dumps(
        {
            "notes": [
                {
                    "text": note,
                    "entities": [entity],
                    "confidence": "high",
                    "source_urls": [MOCK_FETCH_URL],
                }
            ],
            "conflicts": [],
        },
        ensure_ascii=False,
    )


class ScriptedTierRouter:
    """mock 模式的极简 TierRouter（kill_resume_smoke 同款）。"""

    def __init__(self, strong: ScriptedProvider, cheap: ScriptedProvider) -> None:
        self._providers = {"strong": strong, "cheap": cheap}

    def get(self, tier: str) -> ScriptedProvider:
        return self._providers[tier]


# ---- provider 装配（--provider mock / real）---------------------------------


@dataclass
class StressStack:
    """按 --provider 组装好的依赖（AgentLoop 逐问构造参数的统一视图）。

    配置传法与 server real 通路一致（formation/memory_update/verification/
    unanswered/context/retrieval/checkpoint 逐段显式传入，段缺失 = None = 不接线）；
    build_pages 恒 True（孤立闸的前提，见模块 docstring 的取舍说明）。
    """

    mode: str
    model_info: dict[str, Any]
    llm_config: Mapping[str, Any]
    wiki_config: Mapping[str, Any] | None
    prior_config: Mapping[str, Any] | None
    formation_config: Mapping[str, Any] | None
    memory_update_config: Mapping[str, Any] | None
    verification_config: Mapping[str, Any] | None
    unanswered_config: Mapping[str, Any] | None
    context_config: Mapping[str, Any] | None
    checkpoint_config: Mapping[str, Any] | None
    retrieval_config: Mapping[str, Any] | None
    search_provider: Any
    embedding: Any
    fetch_transport: Any
    freshness_settings: Any
    make_router: Callable[[str], Any]


def build_mock_stack() -> StressStack:
    """mock 模式：每问全新 ScriptedProvider 剧本 + 锁定 MockSearch + MockTransport。

    零网络是结构性的：搜索显式传 {"search": {"provider": "mock"}}（不看环境变量），
    抓取走 httpx.MockTransport（快照照常落盘，悬空证据闸才有意义）。
    """
    questions_entity = dict(STRESS_QUESTIONS)

    def make_router(question: str) -> ScriptedTierRouter:
        entity = questions_entity.get(question, "AI框架")
        strong = ScriptedProvider(
            [
                _text_turn(f"- 任务一：围绕「{question}」检索资料\n- 任务二：蒸馏原子笔记\n"),
                _tool_turn([
                    _call("web_search", {"query": question}, call_id="stress-s1"),
                    # 单次 fetch 即可：正常路径下快照已由 _warm_mock_snapshot 校验
                    # 落地，本抓取走"目录已存在即跳过"的纯计算分支，零磁盘写。
                    # 写失败场景下重试抓取也无济于事——半成品目录（有目录无
                    # content.md）会让重试拿到假成功，真正的防线在预热校验。
                    _call("fetch_url", {"url": MOCK_FETCH_URL}, call_id="stress-f1"),
                ]),
                _text_turn("已检索并阅读来源，信息足够，研究结束。"),
                _text_turn(f"## 研究报告\n\n关于「{question}」的核心结论见正文与笔记[1]。\n"),
            ],
            tier="strong",
            model="mock-strong",
            usage=MOCK_USAGE,
        )
        cheap = ScriptedProvider(
            [
                _text_turn(mock_distill_json(question, entity)),
                # 页面聚合的一轮输出：<20 字 → 骨架页兜底（确定性、带笔记 ID 标注）
                _text_turn("页"),
            ],
            tier="cheap",
            model="mock-cheap",
            usage=MOCK_USAGE,
        )
        return ScriptedTierRouter(strong, cheap)

    def mock_transport(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, html=MOCK_FETCH_HTML)

    return StressStack(
        mode="mock",
        model_info={"strong": {"model": "mock-strong", "base_url": ""},
                    "cheap": {"model": "mock-cheap", "base_url": ""}},
        llm_config={
            "strong": {"base_url": "https://mock"},
            "cheap": {"base_url": "https://mock-cheap"},
        },
        wiki_config={"fts_tokenizer": "trigram"},
        prior_config=None,  # 缺省 = enabled + 默认预算（与生产缺省一致）
        formation_config=MOCK_FORMATION_CONFIG,
        memory_update_config=MOCK_MEMORY_UPDATE_CONFIG,
        verification_config=MOCK_VERIFICATION_CONFIG,
        unanswered_config=MOCK_UNANSWERED_CONFIG,
        context_config=MOCK_CONTEXT_CONFIG,
        checkpoint_config=MOCK_CHECKPOINT_CONFIG,
        retrieval_config=None,
        search_provider=get_search_provider({"search": {"provider": "mock"}}),
        embedding=MockEmbeddingProvider(dim=512),
        fetch_transport=httpx.MockTransport(mock_transport),
        freshness_settings=None,  # 模块默认参数（与 lint 缺配置时逐值一致）
        make_router=make_router,
    )


def _mock_url_key(sources_root: Path, url: str) -> Path:
    """mock 抓取 URL 的快照一级目录：sources/{sha1(url)}/（与 tools.fetch 布局对齐）。"""
    return sources_root / hashlib.sha1(url.encode("utf-8")).hexdigest()


def _purge_half_made_snapshot_dirs(sources_root: Path, url: str) -> int:
    """删除 {sha1(url)}/ 下缺 content.md 的半成品快照目录，返回删除数。

    背景（评审 C1 复现的毒化链）：atomic_write_text 先 mkdir 后写文件——写失败
    （如 Defender 瞬时锁打断了 os.replace）会留下"有目录、无 content.md"的半成品；
    _save_snapshot 的跳过判据只看**目录**存在，于是此后每次抓取都直接 return
    （带 content_hash 假成功），笔记引用哈希但正文永远缺失 → 悬空证据 → 闸 5
    误报。删除半成品不破坏 append-only 语义：合法快照目录必含 content.md
    （_save_snapshot 先写 content.md、后写 meta.json），缺 content.md 的目录
    只可能是写失败残留、且按现有跳过判据永远不会被补写——删掉让重试完整重建
    （目录内至多为空；meta.json 晚于 content.md 写出，顺序上不会单独残留，
    防御性容忍其存在）。
    """
    url_key = _mock_url_key(sources_root, url)
    if not url_key.is_dir():
        return 0
    purged = 0
    for subdir in sorted(url_key.iterdir()):
        if subdir.is_dir() and not (subdir / "content.md").is_file():
            shutil.rmtree(subdir, ignore_errors=True)
            purged += 1
    return purged


def _warm_mock_snapshot(wiki_root: Path, stack: StressStack) -> None:
    """预热 mock 抓取的来源快照（脚本启动时写一次，之后每问的抓取零磁盘写）。

    为什么需要：mock 剧本里唯一的**工具内**磁盘写是 fetch_url 的快照落盘——
    registry.dispatch 把工具异常折算成 error JSON（run 照常 done），来源池里该
    URL 已有搜索命中（无 content_hash），蒸馏笔记引用它即成悬空证据；而写失败
    留下的半成品目录更会经"目录存在即跳过"判据变成永久毒化（见
    _purge_half_made_snapshot_dirs 的背景说明）。预热把
    sources/{sha1(url)}/{content_hash}/ **连同 content.md** 先建好并校验落地，
    之后 _save_snapshot 见目录已存在即跳过写盘（append-only 语义），每问抓取
    只剩纯内存计算——悬空证据从机制上不可能被写入抖动制造。

    成功判据是**文件已落地**（sources/{sha1}/{hash}/content.md is_file），不是
    "fetch_url 不抛异常"——后者在半成品目录存在时会假成功（跳过判据不看文件）。
    每次尝试前先清半成品目录再抓取；重试 3 次（0.2s 间隔，针对瞬时文件锁）。
    预热最终失败只记 stderr 不中断——逐问抓取退回原路径（抓取失败如实降级为
    error JSON），预热是稳健性优化，不是前置条件。
    """
    import time as _time

    sources_root = wiki_root / "sources"
    last_error: Exception | None = None
    for _attempt in range(1, 4):
        _purge_half_made_snapshot_dirs(sources_root, MOCK_FETCH_URL)
        try:
            result = fetch_url(
                MOCK_FETCH_URL,
                sources_dir=sources_root,
                transport=stack.fetch_transport,
            )
        except Exception as exc:  # noqa: BLE001 -- 瞬时文件锁重试，见上
            last_error = exc
            _time.sleep(0.2)
            continue
        if _mock_url_key_hash_file(sources_root, result.content_hash).is_file():
            return  # 成功判据：快照正文真正落地
        # fetch_url"成功"但正文缺失 = 半成品目录骗过了跳过判据（评审 C1 链），
        # 记为失败、下一轮清理后重写
        last_error = RuntimeError(f"预热快照未落地：{_mock_url_key(sources_root, MOCK_FETCH_URL)}")
        _time.sleep(0.2)
    print(
        f"[warm ] mock 快照预热 3 次均未确认落地（逐问抓取将走原路径）："
        f"{type(last_error).__name__}: {last_error}",
        file=sys.stderr,
    )


def _mock_url_key_hash_file(sources_root: Path, content_hash: str) -> Path:
    """预热成功判据的落点：sources/{sha1(mock url)}/{content_hash}/content.md。"""
    return _mock_url_key(sources_root, MOCK_FETCH_URL) / content_hash / "content.md"


def load_config() -> dict[str, Any]:
    """读运行配置（RESEARCHWIKI_CONFIG 优先，缺省项目根 config.toml）；读不到回退空。"""
    path = Path(os.environ.get("RESEARCHWIKI_CONFIG") or DEFAULT_CONFIG)
    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def build_real_stack(config: Mapping[str, Any], *, wiki_root: Path) -> StressStack:
    """real 模式：config.toml 全量接线，与 server real 通路同一传法（会产生真实费用）。"""
    llm_cfg = dict(config.get("llm") or {})
    router = ModelRouter(llm_cfg)

    def provider_info(provider: Any) -> dict[str, str]:
        return {
            "model": str(getattr(provider, "model", "") or ""),
            "base_url": str(getattr(provider, "base_url", "") or ""),
        }

    return StressStack(
        mode="real",
        model_info={"strong": provider_info(router.get("strong")),
                    "cheap": provider_info(router.get("cheap"))},
        llm_config=llm_cfg,
        wiki_config=config.get("wiki"),
        prior_config=config.get("prior"),
        formation_config=config.get("formation"),
        memory_update_config=config.get("memory_update"),
        verification_config=config.get("verification"),
        unanswered_config=config.get("unanswered"),
        context_config=config.get("context"),
        checkpoint_config=config.get("checkpoint"),
        retrieval_config=config.get("retrieval"),
        search_provider=get_search_provider(config),
        embedding=get_embedding_provider(config, cache_path=wiki_root / "index.db"),
        fetch_transport=None,  # 真实抓取
        freshness_settings=freshness_from_config(config),
        make_router=lambda question: router,
    )


def make_loop_factory(
    stack: StressStack, *, wiki_root: Path, tokens_path: Path
) -> Callable[[str], AgentLoop]:
    """逐问 loop 工厂：每问一个全新 AgentLoop、同一持久 wiki root、同一 tokens.jsonl。"""

    def factory(question: str) -> AgentLoop:
        return AgentLoop(
            question,
            router=stack.make_router(question),
            llm_config=stack.llm_config,
            accountant=TokenAccountant(path=tokens_path),
            search_provider=stack.search_provider,
            wiki_root=wiki_root,
            fetch_transport=stack.fetch_transport,
            embedding=stack.embedding,
            wiki_config=stack.wiki_config,
            prior_config=stack.prior_config,
            formation_config=stack.formation_config,
            memory_update_config=stack.memory_update_config,
            verification_config=stack.verification_config,
            unanswered_config=stack.unanswered_config,
            context_config=stack.context_config,
            checkpoint_config=stack.checkpoint_config,
            retrieval_config=stack.retrieval_config,
            freshness_settings=stack.freshness_settings,
            build_pages=True,  # 孤立闸的前提（模块 docstring 的取舍说明）
        )

    return factory


# ---- 计量与闸（纯函数，便于测试）-------------------------------------------


def sum_tokens_since(path: Path, trace_id: str, start_offset: int) -> tuple[int, int]:
    """从字节偏移起读 tokens.jsonl，按 trace_id 汇总 (input, output)。

    与 loop/metrics.sum_tokens_from_jsonl 同一套行解析容错（损坏行跳过、error_type
    非空计入——错误调用同样消耗 token）；区别只在起点：压测里每问 run 的记账行
    都追加在文件尾部，从上一问结束时的文件大小读起即可，避免 2h 压测逐问全量
    扫描的 O(n²) 成本。全量复算同值（tests/test_stress.py 交叉验证）。
    """
    if not start_offset or not path.is_file():
        return sum_tokens_from_jsonl(path, trace_id)
    total_in = 0
    total_out = 0
    with path.open("r", encoding="utf-8") as f:
        f.seek(start_offset)
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict) or row.get("trace_id") != trace_id:
                continue
            input_tokens = row.get("input_tokens")
            output_tokens = row.get("output_tokens")
            if isinstance(input_tokens, int) and isinstance(output_tokens, int):
                total_in += input_tokens
                total_out += output_tokens
    return (total_in, total_out)


def gate_counts(snapshot: Mapping[str, Any]) -> dict[str, int]:
    """不恶化闸的三个计数（口径见模块 docstring"不恶化口径"）。"""
    return {
        "broken_links": len(snapshot.get("broken_links") or []),
        "orphan_notes": len(snapshot.get("orphan_notes") or []),
        "dangling_evidence": int(snapshot.get("dangling_evidence") or 0),
    }


def lint_gate_violations(prev: Mapping[str, Any], curr: Mapping[str, Any]) -> list[str]:
    """比较相邻两次快照，返回违规描述（空 = 不恶化）。"""
    prev_counts = gate_counts(prev)
    curr_counts = gate_counts(curr)
    return [
        f"{key} {prev_counts[key]} → {curr_counts[key]}（增大）"
        for key in prev_counts
        if curr_counts[key] > prev_counts[key]
    ]


def lint_snapshot(wiki_root: Path, freshness_settings: Any) -> dict[str, Any]:
    """拍一次 lint 快照（与 `researchwiki lint --json` 同接线：lint_wiki + to_dict）。"""
    report = lint_wiki(WikiStore(wiki_root), freshness_settings=freshness_settings)
    return report.to_dict()


def summarize_compactions(wiki_root: Path) -> dict[str, Any]:
    """跨 run 汇总 runs/*/compactions.jsonl（闸 7：压缩次数/前后 token 可审计）。"""
    runs_dir = wiki_root / "runs"
    events: list[dict[str, Any]] = []
    run_dirs = 0
    if runs_dir.is_dir():
        for path in sorted(runs_dir.glob("*/compactions.jsonl")):
            run_dirs += 1
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    events.append({"run_dir": path.parent.name, **event})
    pre_total = sum(int(event.get("pre_tokens") or 0) for event in events)
    post_total = sum(int(event.get("post_tokens") or 0) for event in events)
    return {
        "run_dirs_with_file": run_dirs,
        "events": len(events),
        "pre_tokens_total": pre_total,
        "post_tokens_total": post_total,
        "details": events,
    }


# ---- CLI -------------------------------------------------------------------


def parse_duration(text: str) -> float:
    """'2h' / '30m' / '90s' / '90' → 秒；非法值抛 ValueError（argparse 报错）。"""
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([hms]?)", text.strip(), re.IGNORECASE)
    if match is None:
        raise ValueError(f"无法解析时长：{text!r}（示例：2h / 30m / 90s / 90）")
    unit = match.group(2).lower()
    return float(match.group(1)) * {"h": 3600.0, "m": 60.0, "s": 1.0, "": 1.0}[unit]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "连续问题流压测：单持久 wiki root + RunWatchdog（详见文件头 docstring）"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--questions", type=int, default=None,
                        help="问题数（与 --duration 至少给一个；超清单后循环使用）")
    parser.add_argument("--root", required=True, help="wiki 数据根目录（单持久，整个问题流共用）")
    parser.add_argument("--provider", choices=["mock", "real"], default="mock",
                        help="mock = 零成本剧本压测（默认）；real = 全量接线（真实费用）")
    parser.add_argument("--per-question-timeout", type=float, default=600.0,
                        help="单问超时秒数（默认 600，与看门狗默认一致）")
    parser.add_argument("--duration", default="",
                        help="压测时长上限（如 2h / 30m / 90s）；到点在问题边界优雅收尾")
    parser.add_argument("--out", default="",
                        help="产物目录（默认 wiki-data/stress/{UTC时间戳}/）")
    return parser


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    """追加一行 JSON 并立即落盘（中断保留已完成部分——崩了就停，账不丢）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _on_event_printer() -> Callable[[Mapping[str, Any]], None]:
    """事件回调：笔记落库打一行（2h 长跑的 liveness 证据）。"""

    def on_event(event: Mapping[str, Any]) -> None:
        if event.get("type") == "data-note":
            data = event.get("data") or {}
            print(f"    [note] {data.get('id')} conf={data.get('confidence')}", flush=True)

    return on_event


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001 -- 老环境/被替换的 stdout 上忽略
        pass
    # real 模式才读 .env 的 key（adaptive_smoke/cold_warm_smoke 同款约定：mock 模式
    # 不碰进程环境，测试进程里不留全局副作用）
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.provider == "real":
        load_env_file()

    # 参数校验：问题数与时长至少给一个；都给了以先到者收尾
    duration_s: float | None = None
    if args.duration:
        try:
            duration_s = parse_duration(args.duration)
        except ValueError as exc:
            print(f"参数错误：{exc}", file=sys.stderr)
            return EXIT_RUN_ERROR
    if args.questions is None and duration_s is None:
        print("参数错误：--questions 与 --duration 至少给一个（否则问题流没有边界）。",
              file=sys.stderr)
        return EXIT_RUN_ERROR
    if args.questions is not None and args.questions < 1:
        print("参数错误：--questions 必须 >= 1。", file=sys.stderr)
        return EXIT_RUN_ERROR

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    wiki_root = Path(args.root)
    tokens_path = wiki_root / "tokens.jsonl"
    out_dir = Path(args.out) if args.out else PROJECT_ROOT / "wiki-data" / "stress" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    questions_path = out_dir / "questions.jsonl"

    stack = build_mock_stack() if args.provider == "mock" else build_real_stack(
        load_config(), wiki_root=wiki_root
    )
    if args.provider == "mock":
        _warm_mock_snapshot(wiki_root, stack)
    factory = make_loop_factory(stack, wiki_root=wiki_root, tokens_path=tokens_path)
    watchdog = RunWatchdog(
        WatchdogSettings(per_question_timeout=args.per_question_timeout),
        on_event=_on_event_printer(),
    )

    print("=== 连续问题流压测（P4b T8 阶段验收件）===")
    print(f"provider    : {stack.mode}（{stack.model_info['strong']['model']} /"
          f" {stack.model_info['cheap']['model']}）")
    print(f"wiki root   : {wiki_root}")
    print(f"产物目录    : {out_dir}")
    print(f"边界        : questions={args.questions} duration={args.duration or '不限'}"
          f" 单问超时={args.per_question_timeout:g}s 连续失败上限="
          f"{watchdog.settings.max_consecutive_failures}")
    print(f"问题清单    : {len(STRESS_QUESTIONS)} 问（AI 框架域，超出后循环）")
    print("-" * 76, flush=True)

    # 基线快照（第 0 问前）：不恶化闸的起点
    snapshots: list[tuple[int, dict[str, Any]]] = []
    violations: list[str] = []

    def take_snapshot(index: int, *, final: bool = False) -> dict[str, Any]:
        snap = lint_snapshot(wiki_root, stack.freshness_settings)
        name = "lint-final.json" if final else f"lint-{index:04d}.json"
        (out_dir / name).write_text(
            json.dumps(snap, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if snapshots:
            prev_index, prev = snapshots[-1]
            row_violations = lint_gate_violations(prev, snap)
            violations.extend(f"lint 闸（第 {prev_index} → {index} 问）：{v}"
                              for v in row_violations)
        snapshots.append((index, snap))
        counts = "、".join(f"{k}={v}" for k, v in gate_counts(snap).items())
        print(f"[lint ] 第 {index} 问快照 → {name}（{counts}）", flush=True)
        return snap

    take_snapshot(0)

    started = time.monotonic()
    question_source: Iterator[tuple[str, str]] = itertools.cycle(STRESS_QUESTIONS)
    counter: Iterator[int] = (
        iter(range(1, args.questions + 1)) if args.questions is not None else itertools.count(1)
    )
    try:
        for _ in counter:
            if watchdog.paused:
                break
            if duration_s is not None and time.monotonic() - started >= duration_s:
                print(f"[stop ] 已达 --duration {args.duration}，在问题边界收尾。", flush=True)
                break
            question, _entity = next(question_source)
            started_at = datetime.now(UTC).isoformat(timespec="seconds")
            offset = tokens_path.stat().st_size if tokens_path.is_file() else 0
            outcome = watchdog.run_question(question, factory)
            tokens_in, tokens_out = sum_tokens_since(tokens_path, outcome.trace_id, offset)
            append_jsonl(
                questions_path,
                {
                    **outcome.to_dict(),
                    "started_at": started_at,
                    "input_tokens": tokens_in,
                    "output_tokens": tokens_out,
                    "tokens_source": "tokens.jsonl 按 trace_id 汇总（计量非估算）",
                },
            )
            print(
                f"[q{outcome.index:03d}] {outcome.status}"
                f" {outcome.elapsed_s:6.1f}s in/out={tokens_in}/{tokens_out}"
                f"{' ' + outcome.error_type if outcome.error_type else ''}"
                f" | {watchdog.status_line()}",
                flush=True,
            )
            if outcome.index % LINT_EVERY == 0:
                take_snapshot(outcome.index)
    except KeyboardInterrupt:
        print("[stop ] 收到 KeyboardInterrupt，在问题边界收尾（已完成部分已逐问落盘）。",
              flush=True)

    ran = len(watchdog.results)
    if ran == 0:
        print("✗ 零个问题被执行（--duration 过短或首问前中断），压测没有产生证据。",
              flush=True)
        return EXIT_RUN_ERROR

    # final 快照：无论最后一问是否恰逢 5 的倍数，收尾必拍并参与闸比较
    take_snapshot(ran, final=True)

    compactions = summarize_compactions(wiki_root)
    (out_dir / "compactions-summary.json").write_text(
        json.dumps(compactions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # ---- 汇总 ---------------------------------------------------------------
    done = sum(1 for outcome in watchdog.results if outcome.status == "done")
    timeouts = sum(1 for outcome in watchdog.results if outcome.status == "timeout")
    errors = sum(1 for outcome in watchdog.results if outcome.status == "error")
    elapsed = time.monotonic() - started
    # 逐问 tokens 从已落盘的 questions.jsonl 读回（测的就是落盘契约，不另算副本）
    rows = [
        json.loads(line)
        for line in questions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    inputs = [int(row.get("input_tokens") or 0) for row in rows]
    total_in = sum(inputs)
    total_out = sum(int(row.get("output_tokens") or 0) for row in rows)
    first_snap_index, first_snap = snapshots[0]
    last_snap_index, last_snap = snapshots[-1]
    first_counts = gate_counts(first_snap)
    last_counts = gate_counts(last_snap)

    print("-" * 76)
    print("=== 压测汇总 ===")
    print(f"问题流      : {ran} 问（完成 {done} · 超时 {timeouts} · 错误 {errors}）"
          f" · 墙钟 {elapsed:.1f}s")
    print(f"看门狗      : {watchdog.status_line()}")
    if watchdog.paused:
        print(f"暂停原因    : {watchdog.pause_reason}")
    per_question = (
        f"（每问 input min/avg/max = {min(inputs)}/{total_in // ran}/{max(inputs)}）"
        if inputs
        else ""
    )
    print(
        f"tokens 计量 : input {total_in} / output {total_out}"
        + per_question
        + "——来自 tokens.jsonl 按 trace 汇总（计量非估算）"
    )
    print(
        f"compaction  : {compactions['run_dirs_with_file']} 个 run 目录有记录、"
        f"{compactions['events']} 次压缩、pre→post tokens "
        f"{compactions['pre_tokens_total']}→{compactions['post_tokens_total']}"
        "（详见 compactions-summary.json）"
    )
    print(
        f"lint 快照   : 第 {first_snap_index} 问 → 第 {last_snap_index} 问："
        + "、".join(
            f"{key} {first_counts[key]}→{last_counts[key]}" for key in first_counts
        )
    )
    if violations:
        print(f"lint 闸（不恶化）：✗ 失败，{len(violations)} 处违规：")
        for violation in violations:
            print(f"  - {violation}")
    else:
        print(f"lint 闸（不恶化）：✓ 通过（{len(snapshots)} 份快照逐对比较，口径见脚本 docstring）")
    print("cache 命中率：未报告：provider 无 cache 字段")
    print(
        "维护执行指针：uv run --no-sync python -m researchwiki.cli consolidate --run "
        f"--root {wiki_root}"
    )

    summary = {
        "gate": "stress_run",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "provider_mode": stack.mode,
        "model": stack.model_info,
        "root": str(wiki_root),
        "out_dir": str(out_dir),
        "questions_requested": args.questions,
        "duration_limit_s": duration_s,
        "per_question_timeout_s": args.per_question_timeout,
        "elapsed_s": round(elapsed, 3),
        "watchdog": {
            "paused": watchdog.paused,
            "pause_reason": watchdog.pause_reason,
            "consecutive_failures": watchdog.consecutive_failures,
            "status_line": watchdog.status_line(),
            "results": [outcome.to_dict() for outcome in watchdog.results],
        },
        "tokens": {
            "input_total": total_in,
            "output_total": total_out,
            "source": "tokens.jsonl 按 trace_id 汇总（计量非估算）",
        },
        "lint": {
            "snapshots": [
                {"index": index, "counts": gate_counts(snap)} for index, snap in snapshots
            ],
            "violations": violations,
            "passed": not violations,
        },
        "compactions": compactions,
        "cache_hit_rate": "未报告：provider 无 cache 字段",
        "maintenance_pointer": (
            f"uv run --no-sync python -m researchwiki.cli consolidate --run --root {wiki_root}"
        ),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"机器可读汇总已写入 {out_dir / 'summary.json'}")

    if watchdog.paused:
        print("✗ 看门狗暂停（连续失败达到上限）——问题流按暂停语义收尾，退出码 2。")
        return EXIT_RUN_ERROR
    if violations:
        return EXIT_ASSERT_FAILED
    print("✓ 压测完成：问题流跑完、lint 指标不恶化、无未解释崩溃。")
    return EXIT_PASS


if __name__ == "__main__":
    raise SystemExit(main())
