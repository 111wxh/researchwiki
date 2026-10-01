#!/usr/bin/env python
"""adaptive 冒烟：同题跨模式成本/质量对照（P3 Dynamic Retrieval 阶段验收证据）。

── 用途（PLAN §3.4 的阶段验收证据）──────────────────────────────────────
同一个 --question 在模式矩阵下各跑一次 AgentLoop（直接驱动，不走 server）：

  播种      强制 deep 跑一次研究（retrieval_config 强制 forced_mode=deep），
            在基础 wiki 根目录沉淀记忆；随后**手工**写入一条 stale 记忆
            （volatility=volatile + observed_at=2026-01-01，正文嵌入问题原文，
            探测检索必命中），让"播种含 stale 记忆"成为确定性前提；
  每模式    把播种后的 wiki 根目录 shutil.copytree 成独立副本（simple/update/
            deep/auto 各一份，warm 起跑、互不污染），在副本里各跑一次：
            simple/update/deep 用 {"enabled": True, "forced_mode": <mode>} 钉死；
            auto 不带 forced_mode（自然判定，守卫红线复验的载体）。

每次 run 收集 <run_dir>/run-metrics.json（PLAN §4.4，14 字段契约不动）与
<run_dir>/policy.json（P3 模式判定留痕），verify_rows 做实五道硬断言：

  1. 每行 policy 含 features 与非空 reasons（PLAN §3.4 验收：不允许裸模式字符串）；
  2. mock 下 simple.input_tokens < deep.input_tokens 且 simple.fresh_search_count == 0
     （simple 不做 fresh 搜索：帽=0 + 剧本也不给搜索轮）；
  3. simple ≤ update ≤ deep 的 fresh_search_count 单调（矩阵的"检索深度阶梯"）；
  4. 播种含 stale 记忆时 auto.policy.mode != simple（守卫红线复验：stale 在判，
     simple 不做 fresh 搜索，不得把守卫信号路由成"无需搜索"）；
  5. input_tokens_checked == metrics.input_tokens（tokens.jsonl 按 trace_id 经
     sum_tokens_from_jsonl 复算对账——成本数字可复算约束，不变量 ⑥）。

任一断言不满足即打印原因并以非零退出码结束——绝不静默通过。

── 与 cold_warm_smoke 的关系（复用而非复制）────────────────────────────────
同目录 import cold_warm_smoke 复用其 ProviderStack / build_config_stack /
ScriptedTierRouter / mock_distill_json / 剧本事件助手（_text_turn/_tool_turn/
_call）/ 三段 MOCK_*_CONFIG / load_config / write_json / 退出码常量。
**run 函数是本脚本自己的**（run_mode_once）：cold_warm 的 run_once 不收
retrieval_config / fetch_transport，为保持 P2 证据脚本不被触碰，这里按同一
构造顺序自建 AgentLoop（与 server real 通路一致的传法）并多传两个参数。

── mock 剧本按模式分层（每模式独立 ScriptedProvider，可重复）──────────────
  simple  plan 轮 + act 纯文本轮 + report 轮（3 轮 strong + 1 轮 cheap 蒸馏）；
          act 首轮即纯文本 → 研究阶段第 1 步自然结束（max_steps=3 绰绰有余）。
  update  plan 轮 + act 带 **一次** web_search 工具调用 + act 纯文本收尾 +
          report 轮（4 轮 strong）；fresh 搜索 1 次（帽 2）。
  deep    plan 轮 + web_search + fetch_url（离线 MockTransport，见下）+
          act 纯文本收尾 + report 轮（5 轮 strong）；搜索 1 次 + 抓取 1 次
          （帽 4），input_tokens 因此严格大于 update/simple。
  auto    与 update 同一份剧本（判定结果由 stale 记忆决定为 update，确定性）。
  剧本安全性：每个剧本的**最后一轮都是纯文本**——ScriptedProvider 耗尽后重复
  最后一轮（防御意外多调用），纯文本轮只会让 act 循环自然终止，任何模式限额下
  都不可能死循环；effective max_steps = simple 3 / update 6 / deep 12。
  注意：forced simple 时 dispatch_research 根本不注册（subagents=False）且
  fresh 搜索帽=0（达帽即拒、provider 不被调用），剧本里也不安排搜索轮。

── 离线 fetch（mock 模式）──────────────────────────────────────────────
deep/播种剧本含一次 fetch_url 调用；mock 模式给 AgentLoop 注入
httpx.MockTransport（回放固定 HTML，tests/test_loop.py 同款、trafilatura 可
提取正文）——抓取链路被真实执行但零网络。config 模式不注入（真实抓取）。

── 副本目录与 index.db ─────────────────────────────────────────────────
模式副本 = copytree(播种根, 播种根同级的 <name>-<mode>)，但**排除 index.db*
（播种 run 的 SearchIndex sqlite 连接可能仍存活：Windows 文件锁 + WAL 尾巴
都不可靠）；且 ensure_index_fresh 会在每个模式 run 开头探测到索引落后并按
store 整体重建（含手工 stale 笔记），检索口径不变、确定性更佳。
模式副本目录是脚本派生物：已存在时整体重建（rmtree 后 copytree）。

── 两种 provider 模式（--provider）──────────────────────────────────────
  mock（默认）  与 cold_warm 同口径：ScriptedProvider 两档 + 显式锁定
                MockSearch + 确定性 MockEmbeddingProvider + trigram tokenizer，
                全程零网络、零 API key、不读 .env；三段记忆演化配置用内置
                最小等价配置（MOCK_*_CONFIG）。
  config        从 config.toml 读真实档位与 provider（复用 cold_warm 的
                build_config_stack，embedding 缓存指向各模式副本根目录）；
                retrieval 配置以 config.toml 的 [retrieval] 段为底、按模式
                覆盖 forced_mode（auto 剔除该键）。会产生真实 API 费用，
                开始前打印成本警告。真实对照（含真模型判定是否与 mock 同构）
                由调用方决定何时花费 token，本脚本不做任何静默降级。

── 输出契约 ────────────────────────────────────────────────────────────
--out 每行一个 JSON 对象（UTF-8、ensure_ascii=False），每模式一行：
  {"mode", "question", "trace_id", "wiki_root", "run_dir", "provider_mode",
   "model", "metrics": {…run-metrics.json 14 字段原文…},
   "policy": {…policy.json 原文（mode/forced/features/reasons/limits/outcome）…},
   "report_chars", "citation_coverage", "input_tokens_checked"}
--out 缺省 PROJECT_ROOT/smoke_out/adaptive_{provider}_{ts}.jsonl；传目录
（如 --out smoke_out）则把默认文件名写进该目录；传 *.jsonl 则按字面路径。
--json <path> 另存机器可读汇总（含断言结论）。stdout 末段是模式对照表并注明
"小样本冒烟，不构成收益结论"（PLAN §11.1）。

── 怎么跑 ─────────────────────────────────────────────────────────────
  # 零 key 零网络：mock 剧本自证全矩阵与五道硬断言
  uv run --no-sync python scripts/adaptive_smoke.py --provider mock --out smoke_out
  # 真实对照：走 config.toml 的 strong/cheap + 真实搜索/embedding（真实费用）
  uv run --no-sync python scripts/adaptive_smoke.py --provider config --out smoke_out
  # 指定问题 / 播种根目录 / 模式子集
  uv run --no-sync python scripts/adaptive_smoke.py --provider mock \
      --question "..." --wiki-root wiki-data/smoke/my-run --modes simple,deep
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ---- 同目录复用 cold_warm_smoke（scripts/ 不是包，先把目录挂上 sys.path）----
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import cold_warm_smoke as cws  # noqa: E402  -- 挂 path 后才能 import 兄弟脚本
import httpx  # noqa: E402

from researchwiki.llm.accounting import TokenAccountant  # noqa: E402
from researchwiki.llm.provider import TokenUsage  # noqa: E402
from researchwiki.loop.agent_loop import AgentLoop  # noqa: E402
from researchwiki.loop.metrics import sum_tokens_from_jsonl  # noqa: E402
from researchwiki.tools import get_search_provider  # noqa: E402
from researchwiki.wiki.embeddings import MockEmbeddingProvider  # noqa: E402
from researchwiki.wiki.store import WikiStore  # noqa: E402

PROJECT_ROOT = cws.PROJECT_ROOT
SMOKE_OUT_DIR = PROJECT_ROOT / "smoke_out"

DEFAULT_QUESTION = cws.DEFAULT_QUESTION
DEFAULT_CONFIG = cws.DEFAULT_CONFIG
DEFAULT_ENV_FILE = cws.DEFAULT_ENV_FILE

EXIT_PASS = cws.EXIT_PASS
EXIT_ASSERT_FAILED = cws.EXIT_ASSERT_FAILED
EXIT_RUN_ERROR = cws.EXIT_RUN_ERROR

# ---- 模式矩阵 ---------------------------------------------------------------
# simple/update/deep 是 research_policy 的三个策略模式（可被 forced_mode 钉死）；
# auto 是矩阵语义层的关键字 = 不带 forced_mode 的自然判定（非策略模式名）。
MODE_SIMPLE = "simple"
MODE_UPDATE = "update"
MODE_DEEP = "deep"
MODE_AUTO = "auto"
FORCED_MODES: tuple[str, ...] = (MODE_SIMPLE, MODE_UPDATE, MODE_DEEP)
ALL_MODES: tuple[str, ...] = (*FORCED_MODES, MODE_AUTO)

# ---- mock 剧本补充件（plan/report/搜索词/来源 URL 沿用 cold_warm 的常量）------
# simple 的 act 首轮就是纯文本收尾：既有记忆足够，无需新检索。
MOCK_ACT_SIMPLE_TEXT = "既有记忆已覆盖问题，无需新检索，直接作答。"

# 离线 fetch 的固定 HTML（tests/test_loop.py 同款：trafilatura 确定可提取正文，
# primary 通道即成功、不会降级到 Jina Reader）。
MOCK_FETCH_HTML = (
    "<html><body><h1>Agent 记忆综述</h1>"
    "<p>智能体的长期记忆是核心议题。Letta 提出 sleep-time compute 思路，"
    "由后台子代理在会话空闲期整理记忆，把维护成本移出交互窗口，上下文保持精简，"
    "同时保留跨会话可复用的长期事实，这些内容足以通过正文提取阈值。</p></body></html>"
)

# ---- stale 播种（硬断言 ④ 的确定性前提）---------------------------------------
# observed_at 固定在 2026-01-01：相对"现在"已过 volatile（半衰期 30 天）的多个
# 半衰期 → decay 远低于 stale_ratio(0.25) → 判 stale（与 tests/test_loop_policy.py
# 的 stale 夹具同一约定，对任意"今天"都成立）。
STALE_OBSERVED_AT = "2026-01-01T00:00:00+00:00"
STALE_NOTE_TITLE = "过期的旧结论（待 fresh 核验）"


def stale_note_text(question: str) -> str:
    """stale 记忆正文：**嵌入问题原文**——探测检索（FTS trigram + 向量双通道）
    对"含问题原文的笔记"必然命中，保证 auto run 的 collect_features 一定看到它。"""
    return (
        f"关于「{question}」的早期观察（{STALE_OBSERVED_AT[:10]} 记录）："
        "当时的结论可能已过时，未经 fresh 核验前不得免检引用。"
    )


def write_stale_seed(wiki_root: Path, question: str) -> str:
    """播种后直接往 wiki 根目录写一条 stale 记忆（绕过蒸馏，确定性优先）。

    confidence=high：让"stale"成为该笔记**唯一**的守卫信号——auto 判定为
    update 的理由链干净可读（stale_hits ≥ 1 → 需 fresh verification）。
    """
    store = WikiStore(wiki_root)
    note = store.save_note(
        stale_note_text(question),
        title=STALE_NOTE_TITLE,
        entities=["Letta"],
        confidence="high",
        volatility="volatile",
        observed_at=STALE_OBSERVED_AT,
    )
    return note.id


# ---- mock 剧本构造（复用 cold_warm 的事件助手与蒸馏脚本）----------------------


def strong_turns_for_mode(mode: str) -> list[list[Any]]:
    """按模式给出 strong 档的剧本轮次（每轮 = 一次 LLM 调用的事件列表）。

    轮次经济学（act 循环按"消费到纯文本轮即终止"运转，max_steps 限额见括号）：
      simple (3)   plan → act 纯文本（第 1 步即收尾）→ report；
      update (6)   plan → act web_search×1 → act 纯文本收尾 → report；
      deep   (12)  plan → act web_search → act fetch_url → act 纯文本收尾 → report；
      auto         与 update 同一份（stale 记忆下判定确定性地落在 update）。
    每份剧本最后一轮都是纯文本：ScriptedProvider 耗尽后重复最后一轮，
    任何限额下都只会让 act 循环自然终止，不可能死循环。
    """
    plan = cws._text_turn(cws.MOCK_PLAN_TEXT)
    summary = cws._text_turn(cws.MOCK_SUMMARY_TEXT)
    report = cws._text_turn(cws.MOCK_REPORT_TEXT)
    search = cws._tool_turn([_search_call()])
    if mode == MODE_SIMPLE:
        return [plan, cws._text_turn(MOCK_ACT_SIMPLE_TEXT), report]
    if mode in (MODE_UPDATE, MODE_AUTO):
        return [plan, search, summary, report]
    if mode == MODE_DEEP:
        fetch = cws._tool_turn([_fetch_call()])
        return [plan, search, fetch, summary, report]
    raise ValueError(f"未知模式：{mode!r}（支持 {'/'.join(ALL_MODES)}）")


def _search_call() -> dict[str, Any]:
    return cws._call(
        "web_search", {"query": cws.MOCK_SEARCH_QUERY}, call_id="adaptive-search"
    )


def _fetch_call() -> dict[str, Any]:
    # 抓取 MockSearch 夹具第一条 URL（必定在来源池内）；离线由 MockTransport 兜住
    return cws._call("fetch_url", {"url": cws.MOCK_SOURCE_URL}, call_id="adaptive-fetch")


def make_mode_router(mode: str, question: str) -> cws.ScriptedTierRouter:
    """一次 run 用的全新脚本化 router（cold_warm 的 make_mock_router 按模式分层版）。

    strong 2–5 次调用（plan/act×n/report），cheap 1 次（蒸馏，脚本与 cold_warm
    同一份：笔记正文嵌入问题原文、带来源 URL——播种 run 的候选能过 formation）。
    """
    strong = cws.ScriptedProvider(
        strong_turns_for_mode(mode),
        tier="strong",
        model="mock-strong",
        usage=TokenUsage(input_tokens=1200, output_tokens=180),
    )
    cheap = cws.ScriptedProvider(
        [cws._text_turn(cws.mock_distill_json(question))],
        tier="cheap",
        model="mock-cheap",
        usage=TokenUsage(input_tokens=1200, output_tokens=180),
    )
    return cws.ScriptedTierRouter(strong, cheap)


def mock_fetch_transport() -> httpx.MockTransport:
    """离线 fetch 传输层：任何 URL 都回放固定 HTML（零网络、确定性）。"""
    return httpx.MockTransport(lambda request: httpx.Response(200, html=MOCK_FETCH_HTML))


def build_mock_stack_for_mode(mode: str, question: str) -> cws.ProviderStack:
    """mock 模式 stack：与 cold_warm.build_mock_stack 同口径，仅 router 换成按模式剧本。

    搜索显式锁定 MockSearch（不受 SEARCH_PROVIDER / 各家 key 环境变量影响）、
    trigram tokenizer、确定性 Mock embedding；三段记忆演化配置沿用内置最小
    等价配置——不接就等于把 P1-B/P2-E 关掉，冒烟就验证不到它们。
    """
    llm_config = {
        "strong": {"base_url": "https://mock"},
        "cheap": {"base_url": "https://mock-cheap"},
    }
    return cws.ProviderStack(
        mode="mock",
        model_info={
            "strong": {"model": "mock-strong", "base_url": ""},
            "cheap": {"model": "mock-cheap", "base_url": ""},
        },
        llm_config=llm_config,
        wiki_config={"fts_tokenizer": "trigram"},
        prior_config=None,
        search_provider=get_search_provider({"search": {"provider": "mock"}}),
        embedding=MockEmbeddingProvider(dim=512),
        make_router=lambda: make_mode_router(mode, question),
        formation_config=cws.MOCK_FORMATION_CONFIG,
        memory_update_config=cws.MOCK_MEMORY_UPDATE_CONFIG,
        verification_config=cws.MOCK_VERIFICATION_CONFIG,
    )


def build_stack(
    provider_mode: str,
    config: Mapping[str, Any],
    question: str,
    mode: str,
    wiki_root: Path,
) -> cws.ProviderStack:
    """按 provider 组装一次 run 的依赖；config 模式 embedding 缓存指向本次根目录。"""
    if provider_mode == "config":
        return cws.build_config_stack(config, wiki_root=wiki_root)
    return build_mock_stack_for_mode(mode, question)


def retrieval_config_for_mode(
    mode: str, base: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """构造本次 run 的 retrieval_config（Task 5 接线键）。

    - base：config 模式取 config.toml 的 [retrieval] 段（保留用户阈值，如
      probe_k / budget_floor），mock 模式为 None；
    - enabled 恒置 True（矩阵的存在意义就是让判定真的跑）；
    - simple/update/deep 置 forced_mode=<mode>；auto **剔除** forced_mode 键
      （自然判定——policy.json 的 forced 应为 false）。
    """
    cfg = dict(base or {})
    cfg["enabled"] = True
    if mode == MODE_AUTO:
        cfg.pop("forced_mode", None)
    else:
        cfg["forced_mode"] = mode
    return cfg


# ---- 单次 run（本脚本自有的 run_once：多传 retrieval_config / fetch_transport）--


def run_mode_once(
    label: str,
    question: str,
    wiki_root: Path,
    stack: cws.ProviderStack,
    *,
    retrieval_config: Mapping[str, Any],
    fetch_transport: Any = None,
) -> dict[str, Any]:
    """跑一次完整 AgentLoop，读回 run-metrics.json / policy.json / report.md。

    构造顺序与 cold_warm.run_once（进而与 server real 通路）一致，仅多传：
    - retrieval_config：P3 模式判定（Task 5 键位），启用时 run 落 policy.json；
    - fetch_transport：mock 模式的离线抓取层（config 模式传 None = 真实抓取）。
    """
    accountant = TokenAccountant(path=wiki_root / "tokens.jsonl")
    loop = AgentLoop(
        question,
        router=stack.make_router(),
        llm_config=stack.llm_config,
        accountant=accountant,
        search_provider=stack.search_provider,
        wiki_root=wiki_root,
        embedding=stack.embedding,
        wiki_config=stack.wiki_config,
        prior_config=stack.prior_config,
        retrieval_config=retrieval_config,
        formation_config=stack.formation_config,
        memory_update_config=stack.memory_update_config,
        verification_config=stack.verification_config,
        fetch_transport=fetch_transport,
    )
    for event in loop.events():
        if event.get("type") == "data-note":
            data = event.get("data") or {}
            print(f"  [note] {data.get('id')} conf={data.get('confidence')}", flush=True)
    metrics = json.loads((loop.run_dir / "run-metrics.json").read_text(encoding="utf-8"))
    # retrieval_config 恒非 None（enabled=True）→ policy.json 必在；缺失即 bug，炸出来
    policy = json.loads((loop.run_dir / "policy.json").read_text(encoding="utf-8"))
    report_chars = len((loop.run_dir / "report.md").read_text(encoding="utf-8"))
    # 成本对账值：tokens.jsonl 按 trace_id 复算（硬断言 ⑤ 的左操作数，复算口径唯一）
    tok_in, _tok_out = sum_tokens_from_jsonl(wiki_root / "tokens.jsonl", loop.trace_id)
    return {
        "mode": label,
        "question": question,
        "trace_id": loop.trace_id,
        "wiki_root": str(wiki_root),
        "run_dir": str(loop.run_dir),
        "provider_mode": stack.mode,
        "model": stack.model_info,
        "metrics": metrics,
        "policy": policy,
        "report_chars": report_chars,
        "citation_coverage": metrics.get("citation_coverage"),
        "input_tokens_checked": tok_in,
    }


# ---- 硬断言（verify_rows：返回失败原因列表，空 = 全部通过）---------------------


def verify_rows(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """五道硬断言（详见模块 docstring；任一不满足即非空失败列表）。"""
    failures: list[str] = []
    if not rows:
        return ["没有任何结果行，无法验证（矩阵至少要跑出一个模式）"]

    # ① 每行 policy 含 features 与非空 reasons（不允许裸模式字符串）
    for row in rows:
        label = str(row.get("mode"))
        policy = row.get("policy")
        if not isinstance(policy, Mapping):
            failures.append(f"[{label}] policy 缺失（retrieval 未启用或 policy.json 未落盘）")
            continue
        if not policy.get("features"):
            failures.append(f"[{label}] policy.features 缺失或为空——判定输入必须留痕可复算")
        if not [str(r) for r in (policy.get("reasons") or []) if str(r).strip()]:
            failures.append(
                f"[{label}] policy.reasons 为空——PLAN §3.4 验收不允许只有模式字符串"
            )

    by_mode = {str(row.get("mode")): row for row in rows}
    provider_mode = str(rows[0].get("provider_mode"))

    # ② mock 成本对照：simple < deep 且 simple 零 fresh 搜索（config 模式不设此断言：
    #    真模型各模式成本受真实剧本/搜索影响，只做 ③⑤ 的口径校验）
    if provider_mode == "mock":
        simple = by_mode.get(MODE_SIMPLE)
        deep = by_mode.get(MODE_DEEP)
        if simple is not None and deep is not None:
            s_in = int((simple.get("metrics") or {}).get("input_tokens") or 0)
            d_in = int((deep.get("metrics") or {}).get("input_tokens") or 0)
            if s_in >= d_in:
                failures.append(
                    f"mock 成本对照失败：simple.input_tokens={s_in} 必须 < "
                    f"deep.input_tokens={d_in}（轻量模式的成本优势必须可测得）"
                )
        if simple is not None:
            s_searches = int(
                (simple.get("metrics") or {}).get("fresh_search_count") or 0
            )
            if s_searches != 0:
                failures.append(
                    f"mock simple fresh_search_count={s_searches}，必须 == 0"
                    "（simple 的搜索帽=0：fresh 检索一律不做）"
                )

    # ③ fresh 搜索沿 simple ≤ update ≤ deep 单调（缺席的模式跳过该档比较）
    chain = [m for m in FORCED_MODES if m in by_mode]
    for lower, upper in zip(chain, chain[1:], strict=False):
        lo = int((by_mode[lower].get("metrics") or {}).get("fresh_search_count") or 0)
        hi = int((by_mode[upper].get("metrics") or {}).get("fresh_search_count") or 0)
        if lo > hi:
            failures.append(
                f"检索深度阶梯断裂：{lower}.fresh_search_count={lo} > "
                f"{upper}.fresh_search_count={hi}（必须单调不减）"
            )

    # ④ 守卫红线复验：播种含 stale 记忆时 auto 不落 simple
    auto = by_mode.get(MODE_AUTO)
    if auto is not None:
        auto_policy = auto.get("policy") or {}
        if str(auto_policy.get("mode") or "") == MODE_SIMPLE:
            failures.append(
                "守卫红线复验失败：auto 判定为 simple——播种含 stale 记忆时，"
                "simple（零 fresh 搜索）不得成为自然判定结果（PLAN §3.4 红线）"
            )

    # ⑤ 成本可复算：input_tokens 与 tokens.jsonl 按 trace_id 复算一致
    for row in rows:
        label = str(row.get("mode"))
        trace_id = str(row.get("trace_id") or "")
        metrics = dict(row.get("metrics") or {})
        tokens_path = Path(str(row.get("wiki_root") or "")) / "tokens.jsonl"
        tok_in, _tok_out = sum_tokens_from_jsonl(tokens_path, trace_id)
        metrics_in = int(metrics.get("input_tokens") or 0)
        if metrics_in != tok_in:
            failures.append(
                f"[{label}] token 对账失败：run-metrics input_tokens={metrics_in}，"
                f"sum_tokens_from_jsonl({tokens_path}, trace_id={trace_id})={tok_in}"
            )
        checked = row.get("input_tokens_checked")
        if not isinstance(checked, int) or checked != metrics_in:
            failures.append(
                f"[{label}] 行内 input_tokens_checked={checked!r} 与 "
                f"metrics.input_tokens={metrics_in} 不一致（复算值必须随行落盘）"
            )
    return failures


# ---- 汇总表（纯函数，供单测）-------------------------------------------------


def format_summary(rows: Sequence[Mapping[str, Any]]) -> str:
    """同题跨模式对照表：每模式一行的成本/检索/质量摘要（含防误读声明）。"""
    # (metrics 字段, 列宽)；coverage 取 metrics.citation_coverage 的显示别名
    columns: tuple[tuple[str, int], ...] = (
        ("fresh_search_count", 19),
        ("fresh_fetch_count", 18),
        ("source_count", 14),
        ("notes_created", 15),
        ("input_tokens", 14),
        ("output_tokens", 15),
        ("latency_ms", 12),
        ("citation_coverage", 18),
    )
    lines = [
        "── 同题跨模式对照（播种 = forced deep + 手工 stale 记忆；每模式独立副本 warm 起跑）──",
        f"{'mode':<8}{'policy':<8}{'forced':<8}"
        + "".join(name.rjust(width) for name, width in columns),
    ]
    for row in rows:
        policy = row.get("policy") or {}
        metrics = dict(row.get("metrics") or {})
        cells = "".join(
            cws._fmt_metric(metrics.get(name)).rjust(width) for name, width in columns
        )
        lines.append(
            f"{str(row.get('mode')):<8}{str(policy.get('mode')):<8}"
            f"{str(policy.get('forced')).lower():<8}{cells}"
        )
    lines.append(
        "注：小样本冒烟，不构成收益结论（PLAN §11.1）；"
        "各模式限额见各 run_dir/policy.json 的 limits。"
    )
    return "\n".join(lines)


# ---- 入口 -------------------------------------------------------------------


def modes_arg(raw: str) -> list[str]:
    """--modes 解析：逗号分隔、去重保序、非法值在 argparse 层拒绝（exit 2）。"""
    modes = [m.strip() for m in str(raw).split(",") if m.strip()]
    invalid = [m for m in modes if m not in ALL_MODES]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"--modes 只支持 {'/'.join(ALL_MODES)}，得到 {invalid!r}"
        )
    if not modes:
        raise argparse.ArgumentTypeError("--modes 不能为空")
    return list(dict.fromkeys(modes))


def resolve_out_path(out_arg: str, provider_mode: str, stamp: str) -> Path:
    """--out 三态：缺省 = smoke_out/adaptive_{provider}_{ts}.jsonl；
    传 *.jsonl = 字面文件路径；传目录（如 smoke_out）= 默认文件名写进该目录。"""
    if not out_arg:
        return SMOKE_OUT_DIR / f"adaptive_{provider_mode}_{stamp}.jsonl"
    path = Path(out_arg)
    if path.suffix == ".jsonl":
        return path
    return path / f"adaptive_{provider_mode}_{stamp}.jsonl"


def copy_seed_into(mode_root: Path, seed_root: Path) -> None:
    """播种根目录 → 模式独立副本（warm 起跑、互不污染）。

    排除 index.db*：播种 run 的 SearchIndex 连接可能仍存活（Windows 文件锁 /
    WAL 尾巴），且各模式 run 开头的 ensure_index_fresh 会探测到索引落后并按
    store 整体重建（含手工 stale 笔记）——检索口径不变，确定性更佳。
    模式副本是脚本派生物：已存在则整体重建。
    """
    if mode_root.exists():
        shutil.rmtree(mode_root)
    shutil.copytree(
        seed_root,
        mode_root,
        ignore=shutil.ignore_patterns("index.db", "index.db-wal", "index.db-shm"),
    )


def mode_root_for(seed_root: Path, mode: str) -> Path:
    """模式副本目录：播种根的同级 <name>-<mode>（与播种根、彼此两两不相交）。"""
    return seed_root.parent / f"{seed_root.name}-{mode}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="adaptive 冒烟：同题跨模式成本/质量对照（详见文件头 docstring）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--provider", choices=("mock", "config"), default="mock",
                        help="mock=零网络零 key 的剧本自证（默认）；config=走 config.toml 真实档位")
    parser.add_argument("--question", default=DEFAULT_QUESTION,
                        help="研究问题（矩阵内全部 run 共用）")
    parser.add_argument("--wiki-root", default="",
                        help="播种用 wiki 根目录（默认 wiki-data/smoke/adaptive-<UTC时间戳>/；"
                             "各模式副本为同级 <name>-<mode>）")
    parser.add_argument("--out", default="",
                        help="JSONL 输出路径（默认 smoke_out/adaptive_{provider}_{ts}.jsonl；"
                             "传目录则写入该目录下的默认文件名）")
    parser.add_argument("--modes", type=modes_arg, default=",".join(ALL_MODES),
                        help="模式矩阵（逗号分隔，缺省 simple,update,deep,auto；auto=自然判定）")
    parser.add_argument("--json", default="", help="机器可读汇总另存为 JSON（可选）")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE),
                        help=".env 路径（仅 config 模式读取；传空串跳过）")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="config.toml 路径")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001 -- 老环境/被替换的 stdout 上忽略
        pass
    args = build_parser().parse_args(argv)
    modes: list[str] = list(args.modes)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    seed_root = (
        Path(args.wiki_root) if args.wiki_root else cws.SMOKE_DIR / f"adaptive-{stamp}"
    )
    out_path = resolve_out_path(args.out, args.provider, stamp)

    config: dict[str, Any] = {}
    retrieval_base: Mapping[str, Any] | None = None
    if args.provider == "config":
        if args.env_file:
            from researchwiki.env import load_env_file

            load_env_file(args.env_file)
        config = cws.load_config(Path(args.config))
        raw = config.get("retrieval")
        retrieval_base = raw if isinstance(raw, Mapping) else None
        print(
            "⚠ 成本警告：config 模式将用真实模型与真实搜索完整跑 "
            f"{1 + len(modes)} 次研究（播种 + {len(modes)} 个模式），"
            "会产生真实 API 费用；请确认 key 与配额后再继续。"
        )

    question = args.question
    seed_stack = build_stack(args.provider, config, question, MODE_DEEP, seed_root)
    fetch_transport = mock_fetch_transport() if args.provider == "mock" else None

    strong_info = seed_stack.model_info["strong"]
    cheap_info = seed_stack.model_info["cheap"]
    print("=== adaptive 冒烟：同题跨模式成本/质量对照（P3 Dynamic Retrieval）===")
    print(f"question    : {question}")
    print(f"provider    : {args.provider}")
    print(f"modes       : {', '.join(modes)}（auto = 不带 forced_mode 的自然判定）")
    print(
        f"model       : strong={strong_info['model']}"
        f"（{strong_info['base_url'] or '—'}）  "
        f"cheap={cheap_info['model']}（{cheap_info['base_url'] or '—'}）"
    )
    print(f"features    : {cws.describe_features(seed_stack)}")
    print(f"seed root   : {seed_root}")
    print(f"out         : {out_path}")
    print("-" * 76, flush=True)

    rows: list[dict[str, Any]] = []
    try:
        # ---- 播种：forced deep 跑一次研究，在基础根目录沉淀记忆 ----
        print("[seed] run 开始（forced deep，空 Wiki 冷启动播种）", flush=True)
        seed_row = run_mode_once(
            "seed",
            question,
            seed_root,
            seed_stack,
            retrieval_config=retrieval_config_for_mode(MODE_DEEP, retrieval_base),
            fetch_transport=fetch_transport,
        )
        stale_id = write_stale_seed(seed_root, question)
        seed_metrics = seed_row["metrics"]
        print(
            f"[seed] trace_id={seed_row['trace_id']}  mode={seed_row['policy']['mode']}"
            f"  notes={seed_metrics['notes_created']}  in/out="
            f"{seed_metrics['input_tokens']}/{seed_metrics['output_tokens']}"
            f"  + stale 记忆 note_id={stale_id}（observed_at={STALE_OBSERVED_AT}）",
            flush=True,
        )

        # ---- 每模式：独立副本上各跑一次（simple/update/deep 钉模式，auto 自然判定）----
        for mode in modes:
            mode_root = mode_root_for(seed_root, mode)
            copy_seed_into(mode_root, seed_root)
            stack = build_stack(args.provider, config, question, mode, mode_root)
            print(f"[{mode}] run 开始（wiki_root={mode_root}）", flush=True)
            row = run_mode_once(
                mode,
                question,
                mode_root,
                stack,
                retrieval_config=retrieval_config_for_mode(mode, retrieval_base),
                fetch_transport=fetch_transport,
            )
            rows.append(row)
            metrics = row["metrics"]
            policy = row["policy"]
            print(
                f"[{mode}] trace_id={row['trace_id']}  policy={policy['mode']}"
                f"(forced={str(policy['forced']).lower()})"
                f"  prior_hit={metrics['prior_hit_count']}"
                f"  search/fetch={metrics['fresh_search_count']}/{metrics['fresh_fetch_count']}"
                f"  sources={metrics['source_count']}"
                f"  in/out={metrics['input_tokens']}/{metrics['output_tokens']}"
                f"  run_dir={row['run_dir']}",
                flush=True,
            )
    except Exception as exc:  # noqa: BLE001 -- 环境/网络问题不作数，明确退出而非裸栈
        print(f"✗ 运行异常：{type(exc).__name__}: {exc}")
        print("（config 模式请检查 key / base_url / 网络；mock 模式不应出现本行，视为 bug）")
        return EXIT_RUN_ERROR

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    failures = verify_rows(rows)
    print("-" * 76)
    print(format_summary(rows))
    if args.json:
        cws.write_json(
            Path(args.json),
            {
                "gate": "adaptive_smoke",
                "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "provider_mode": args.provider,
                "question": question,
                "modes": modes,
                "seed_root": str(seed_root),
                "out": str(out_path),
                "passed": not failures,
                "failures": failures,
                "runs": rows,
            },
        )
        print(f"机器可读汇总已写入 {args.json}")

    if failures:
        print("-" * 76)
        for failure in failures:
            print(f"✗ 断言失败：{failure}")
        print(f"JSONL 已写入 {out_path}（断言未通过，退出码 {EXIT_ASSERT_FAILED}）")
        return EXIT_ASSERT_FAILED

    auto_row = next((row for row in rows if row["mode"] == MODE_AUTO), None)
    print("-" * 76)
    print(
        "✓ 断言通过：① 每行 policy 全量带 features+reasons；"
        "② mock 下 simple 成本 < deep 且 simple 零 fresh 搜索；"
        "③ fresh 搜索沿模式阶梯单调；"
        + (
            f"④ auto 守卫红线复验（stale_hits={auto_row['policy']['features']['stale_hits']} "
            f"→ mode={auto_row['policy']['mode']} ≠ simple）；"
            if auto_row is not None
            else ""
        )
        + "⑤ input_tokens 与 tokens.jsonl 按 trace_id 复算一致（可复算）。"
    )
    print(f"JSONL 已写入 {out_path}")
    return EXIT_PASS


if __name__ == "__main__":
    raise SystemExit(main())
