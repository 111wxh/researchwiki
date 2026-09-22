#!/usr/bin/env python
"""cold/warm 对照冒烟：同一问题先空 Wiki 冷启动、再同根目录二次研究（阶段 1 Prior MVP 收官件）。

── 用途（PLAN §4.5 的阶段验收证据）──────────────────────────────────────
同一 --question 在同一 wiki 根目录连续跑两次 AgentLoop（直接驱动，不走 server）：

  cold run  空 Wiki 冷启动：Prior 无从命中（prior_hit_count == 0），蒸馏沉淀
            ≥1 条与问题相关的 active note 进 wiki；
  warm run  同一 wiki 根目录、同一问题、全新 AgentLoop 实例：Prior 检索必须
            命中冷启动沉淀的笔记（prior_hit_count > 0）。

每次 run 收集 <run_dir>/run-metrics.json（PLAN §4.4，14 字段），并把硬断言做实：

  1. cold run prior_hit_count == 0（必须是空 Wiki 冷启动；--wiki-root 指向
     已含相关笔记的目录时在此拦住"假绿"对照）；
  2. cold run notes_created >= 1（cold 必须沉淀 ≥1 条 active note，否则 warm
     的命中不可能来自本次 cold 产物）；
  3. warm run prior_hit_count > 0；
  4. 两次 run 的 input/output tokens 与
     sum_tokens_from_jsonl(<wiki_root>/tokens.jsonl, trace_id) 完全一致
     （TokenAccountant 由本脚本构造、指向同一 tokens.jsonl）。

任一断言不满足即打印原因并以非零退出码结束——这是"复用收益可复算"的
脚本化闸，绝不静默通过。

── 两种 provider 模式（--provider）──────────────────────────────────────
  mock（默认）  ScriptedProvider 驱动 strong/cheap 两档（极简 TierRouter 注入，
                gate 脚本 / tests/test_loop_prior.py 同款造法）+ 显式锁定的
                MockSearch（get_search_provider 收 {"search": {"provider":
                "mock"}}，不受 SEARCH_PROVIDER / TAVILY_API_KEY /
                BOCHA_API_KEY 环境变量影响）；embedding 用确定性
                MockEmbeddingProvider，tokenizer 固定 trigram。全程零网络、
                零 API key、不读 .env；两次 run 用同一份剧本、各自全新
                provider 实例，模型响应完全脚本化、可重复。
  config        从 config.toml 读真实 strong/cheap 档位（ModelRouter）、真实搜索
                provider 与真实 embedding（get_embedding_provider，缓存指向
                <wiki_root>/index.db）；输出行记录 model 名与 base_url
                （PLAN §10.3：评测记录 provider）。真模型蒸馏若没产出可命中的
                笔记，warm prior_hit_count == 0 → 明确报失败，不静默通过。

── 输出契约 ────────────────────────────────────────────────────────────
--out 每行一个 JSON 对象（UTF-8、ensure_ascii=False）：
  {"run": "cold"|"warm", "question", "trace_id", "wiki_root", "run_dir",
   "provider_mode",
   "model": {"strong": {"model", "base_url"}, "cheap": {"model", "base_url"}},
   "metrics": {…run-metrics.json 14 字段原文…}}
--json <path> 另存机器可读汇总（含断言结论）。stdout 末段是 cold vs warm
对照表（含差值），并注明"小样本冒烟，不构成收益结论"（PLAN §11.1）。

── 成本警告（config 模式）──────────────────────────────────────────────
config 模式会调用真实 LLM 与真实搜索、完整跑两次研究（cold + warm），
产生真实 API 费用；key 从 .env / 环境变量读取（脚本内零硬编码）。
开始前会打印一行成本警告，请确认配额后再运行。

── 怎么跑 ─────────────────────────────────────────────────────────────
  # 零 key 零网络：mock 剧本自证 cold/warm 全链路与硬断言
  uv run --no-sync python scripts/cold_warm_smoke.py --provider mock
  # 真实测量：走 config.toml 的 strong/cheap + 真实搜索与 embedding
  uv run --no-sync python scripts/cold_warm_smoke.py --provider config
  # 指定问题 / 落盘位置
  uv run --no-sync python scripts/cold_warm_smoke.py --provider mock \
      --question "..." --wiki-root wiki-data/smoke/my-run --out wiki-data/smoke/my-run.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from researchwiki.env import load_env_file
from researchwiki.llm.accounting import TokenAccountant
from researchwiki.llm.provider import ScriptedProvider, StreamEvent, TokenUsage
from researchwiki.llm.router import ModelRouter
from researchwiki.loop.agent_loop import AgentLoop
from researchwiki.loop.metrics import sum_tokens_from_jsonl
from researchwiki.tools import get_search_provider
from researchwiki.wiki.embeddings import MockEmbeddingProvider, get_embedding_provider

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config.toml"
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"
SMOKE_DIR = PROJECT_ROOT / "wiki-data" / "smoke"

DEFAULT_QUESTION = "Agent 长期记忆的主流方案（Letta、Mem0 等）如何做记忆固化与检索？"

EXIT_PASS = 0
EXIT_ASSERT_FAILED = 1
EXIT_RUN_ERROR = 2

# ---- mock 剧本（两次 run 同一份；每次 run 全新 provider 实例，保证可重复）----
MOCK_PLAN_TEXT = "- 任务一：检索主流 agent 长期记忆方案\n- 任务二：把关键结论蒸馏成原子笔记\n"
MOCK_SEARCH_QUERY = "agent 长期记忆 方案"
MOCK_SUMMARY_TEXT = "已检索到主流方案，信息足够，研究完成。"
MOCK_REPORT_TEXT = "## 研究报告\n\nLetta 的后台 subagent 方案[1] 值得优先试点。\n"

# 对照表字段（PLAN §4.5 阶段验收口径 + prior 命中证据行；差值 = warm − cold）
COMPARE_FIELDS: tuple[str, ...] = (
    "prior_hit_count",
    "fresh_search_count",
    "fresh_fetch_count",
    "source_count",
    "notes_created",
    "input_tokens",
    "output_tokens",
    "latency_ms",
    "citation_coverage",
)


# ---- mock 剧本构造（与 tests/test_loop_prior.py 同构的事件造法）--------------


def _text_turn(text: str) -> list[StreamEvent]:
    return [StreamEvent(type="text_delta", delta=text)]


def _tool_turn(calls: list[dict[str, Any]]) -> list[StreamEvent]:
    return [
        StreamEvent(type="tool_calls", tool_calls=calls),
        StreamEvent(type="usage", usage=TokenUsage(input_tokens=900, output_tokens=40)),
    ]


def _call(name: str, arguments: dict[str, Any], *, call_id: str) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
    }


def mock_distill_json(question: str) -> str:
    """蒸馏阶段的脚本输出：一条与问题强相关（含问题原文）的原子笔记。

    笔记正文嵌入问题原文，保证 warm run 的 Prior 检索（FTS trigram + 向量双通道）
    必然命中——这是"cold 蒸馏产出 ≥1 条与问题相关的 active note"的脚本化保证。
    """
    note = (
        f"关于「{question}」的研究结论：Letta 用后台 subagent 在会话空闲期整理长期记忆，"
        "跨会话保留长期事实。"
    )
    return json.dumps(
        {
            "notes": [{"text": note, "entities": ["Letta"], "confidence": "high"}],
            "conflicts": [],
        },
        ensure_ascii=False,
    )


class ScriptedTierRouter:
    """mock 模式的极简 TierRouter：按档位返回脚本化 Provider（测试 FakeRouter 同款）。"""

    def __init__(self, strong: ScriptedProvider, cheap: ScriptedProvider) -> None:
        self._providers = {"strong": strong, "cheap": cheap}

    def get(self, tier: str) -> ScriptedProvider:
        return self._providers[tier]


def make_mock_router(question: str) -> ScriptedTierRouter:
    """一次 run 用的全新脚本化 router：strong 4 次调用（plan/act/收尾/report），cheap 1 次（蒸馏）。

    每次调用都重新构造 provider 实例——cold 与 warm 各自从头消费同一份剧本，
    模型响应完全脚本化、两次 run 可重复。
    """
    strong_turns = [
        _text_turn(MOCK_PLAN_TEXT),
        _tool_turn([_call("web_search", {"query": MOCK_SEARCH_QUERY}, call_id="smoke-1")]),
        _text_turn(MOCK_SUMMARY_TEXT),
        _text_turn(MOCK_REPORT_TEXT),
    ]
    strong = ScriptedProvider(
        strong_turns,
        tier="strong",
        model="mock-strong",
        usage=TokenUsage(input_tokens=1200, output_tokens=180),
    )
    cheap = ScriptedProvider(
        [_text_turn(mock_distill_json(question))],
        tier="cheap",
        model="mock-cheap",
        usage=TokenUsage(input_tokens=1200, output_tokens=180),
    )
    return ScriptedTierRouter(strong, cheap)


# ---- provider 组装（--provider mock / config，参照 gate 脚本的组装模式）-------


@dataclass
class ProviderStack:
    """按 --provider 组装好的依赖（两种装配的统一视图）。

    make_router 每次调用返回一个可用于一次 run 的 router（mock 是全新脚本化
    实例；config 的 ModelRouter 无状态，直接复用）。
    """

    mode: str
    model_info: dict[str, dict[str, str]]
    llm_config: Mapping[str, Any]
    wiki_config: Mapping[str, Any]
    prior_config: Mapping[str, Any] | None
    search_provider: Any
    embedding: Any
    make_router: Callable[[], Any]


def build_mock_stack(question: str) -> ProviderStack:
    """mock 模式：ScriptedProvider 两档 + 显式锁定的 MockSearch + 确定性 Mock embedding。

    搜索组装显式传 ``{"search": {"provider": "mock"}}``：get_search_provider 的
    provider 名命中不了 tavily/bocha 分支，也不再看 SEARCH_PROVIDER /
    TAVILY_API_KEY / BOCHA_API_KEY 环境变量兜底——mock 模式的零网络不依赖
    环境巧合。llm_config 按 tests/test_loop_prior.py 的口径给 cheap 配置
    base_url，使蒸馏/子 agent 明确走 cheap 档；wiki_config 固定 trigram
    （不探测 vendor DLL，保证确定性）；prior_config 缺省 = enabled + 默认预算。
    """
    llm_config = {
        "strong": {"base_url": "https://mock"},
        "cheap": {"base_url": "https://mock-cheap"},
    }
    return ProviderStack(
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
        make_router=lambda: make_mock_router(question),
    )


def build_config_stack(config: Mapping[str, Any], *, wiki_root: Path) -> ProviderStack:
    """config 模式：真实 ModelRouter / 搜索 provider / embedding（缓存指向本次 wiki 根目录）。"""
    llm_cfg = config.get("llm") or {}
    router = ModelRouter(dict(llm_cfg))
    strong = router.get("strong")
    cheap = router.get("cheap")

    def provider_info(provider: Any) -> dict[str, str]:
        return {
            "model": str(getattr(provider, "model", "") or ""),
            "base_url": str(getattr(provider, "base_url", "") or ""),
        }

    return ProviderStack(
        mode="config",
        model_info={"strong": provider_info(strong), "cheap": provider_info(cheap)},
        llm_config=llm_cfg,
        wiki_config=config.get("wiki") or {},
        prior_config=config.get("prior"),
        search_provider=get_search_provider(config),
        embedding=get_embedding_provider(config, cache_path=wiki_root / "index.db"),
        make_router=lambda: router,
    )


# ---- 单次 run 与硬断言 ------------------------------------------------------


def run_once(
    label: str,
    question: str,
    wiki_root: Path,
    stack: ProviderStack,
    tokens_path: Path,
) -> dict[str, Any]:
    """跑一次完整 AgentLoop，从 <run_dir>/run-metrics.json 读回指标（测的就是落盘契约）。"""
    accountant = TokenAccountant(path=tokens_path)
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
    )
    for event in loop.events():
        if event.get("type") == "data-note":
            data = event.get("data") or {}
            print(f"  [note] {data.get('id')} conf={data.get('confidence')}", flush=True)
    metrics_path = loop.run_dir / "run-metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "run": label,
        "question": question,
        "trace_id": loop.trace_id,
        "wiki_root": str(wiki_root),
        "run_dir": str(loop.run_dir),
        "provider_mode": stack.mode,
        "model": stack.model_info,
        "metrics": metrics,
    }


def verify_rows(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """硬断言（PLAN §4.5 阶段验收口径），返回失败原因列表（空 = 全部通过）。

    1. cold run 的 prior_hit_count == 0（空 Wiki 冷启动的口径本身也要被验证；
       --wiki-root 指向已含相关笔记的目录时，cold 直接命中旧笔记、对照假绿，
       在此拦住）；
    2. cold run 的 notes_created >= 1（cold 必须沉淀 ≥1 条 active note，否则
       warm 的命中不可能来自本次 cold 产物）；
    3. warm run 的 prior_hit_count > 0（mock / config 都断言；config 真模型
       蒸馏没产出可命中笔记时在此明确报失败）；
    4. 两次 run 的 input/output tokens 与
       sum_tokens_from_jsonl(<wiki_root>/tokens.jsonl, trace_id) 完全一致。
    """
    failures: list[str] = []
    cold = next((row for row in rows if str(row.get("run")) == "cold"), None)
    if cold is None:
        failures.append("缺少 cold run 的结果行，无法验证冷启动口径")
    else:
        metrics = dict(cold.get("metrics") or {})
        hits = int(metrics.get("prior_hit_count") or 0)
        if hits != 0:
            failures.append(
                f"cold run prior_hit_count={hits}，必须 == 0：cold 必须从空 Wiki 冷启动，"
                "命中说明 --wiki-root 已含相关笔记、本次对照是假绿"
                "（换一个空目录或清空后重跑）"
            )
        created = int(metrics.get("notes_created") or 0)
        if created < 1:
            failures.append(
                f"cold run notes_created={created}，必须 >= 1：cold 蒸馏必须沉淀"
                " ≥1 条 active note，否则 warm 的 prior 命中不可能来自本次 cold 产物"
            )
    warm = next((row for row in rows if str(row.get("run")) == "warm"), None)
    if warm is None:
        failures.append("缺少 warm run 的结果行，无法验证 Prior 复用")
    else:
        hits = int((warm.get("metrics") or {}).get("prior_hit_count") or 0)
        if hits <= 0:
            failures.append(
                f"warm run prior_hit_count={hits}，必须 > 0：同一 wiki 根目录二次研究"
                "必须命中 cold 沉淀的 active note"
                "（config 模式下若真模型蒸馏没产出可命中的笔记，这就是明确失败）"
            )
    for row in rows:
        label = str(row.get("run"))
        trace_id = str(row.get("trace_id") or "")
        metrics = dict(row.get("metrics") or {})
        tokens_path = Path(str(row.get("wiki_root") or "")) / "tokens.jsonl"
        tok_in, tok_out = sum_tokens_from_jsonl(tokens_path, trace_id)
        if (
            int(metrics.get("input_tokens") or 0) != tok_in
            or int(metrics.get("output_tokens") or 0) != tok_out
        ):
            failures.append(
                f"[{label}] token 对账失败：run-metrics in/out="
                f"{metrics.get('input_tokens')}/{metrics.get('output_tokens')}，"
                f"sum_tokens_from_jsonl({tokens_path}, trace_id={trace_id})={tok_in}/{tok_out}"
            )
    return failures


# ---- 汇总表（纯函数，供单测）-------------------------------------------------


def _fmt_metric(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _fmt_delta(cold: Any, warm: Any) -> str:
    if cold is None or warm is None:
        return "—"
    delta = warm - cold
    if isinstance(delta, float):
        return f"{delta:+.4f}"
    return f"{delta:+d}"


def format_summary(rows: Sequence[Mapping[str, Any]]) -> str:
    """cold vs warm 人类可读对照表（含差值与 PLAN §11.1 的防误读声明）。"""
    by_run = {str(row.get("run")): dict(row.get("metrics") or {}) for row in rows}
    cold = by_run.get("cold", {})
    warm = by_run.get("warm", {})
    lines = [
        "── cold vs warm 对照（差值 = warm − cold）──",
        f"{'指标':<18}{'cold':>14}{'warm':>14}{'差值':>14}",
    ]
    for field in COMPARE_FIELDS:
        cold_value = cold.get(field)
        warm_value = warm.get(field)
        lines.append(
            f"{field:<18}{_fmt_metric(cold_value):>14}"
            f"{_fmt_metric(warm_value):>14}{_fmt_delta(cold_value, warm_value):>14}"
        )
    lines.append("注：小样本冒烟，不构成收益结论（PLAN §11.1）。")
    return "\n".join(lines)


# ---- 入口 -------------------------------------------------------------------


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", "utf-8")


def load_config(path: Path) -> dict[str, Any]:
    """读 config.toml；缺失/损坏按空配置处理（等价 mock，不炸）。"""
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except (FileNotFoundError, tomllib.TOMLDecodeError):
        return {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="cold/warm 对照冒烟：Prior 复用收益的阶段验收证据（详见文件头 docstring）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--provider", choices=("mock", "config"), default="mock",
                        help="mock=零网络零 key 的剧本自证（默认）；config=走 config.toml 真实档位")
    parser.add_argument("--question", default=DEFAULT_QUESTION,
                        help="研究问题（两次 run 共用）")
    parser.add_argument("--wiki-root", default="",
                        help="wiki 根目录（默认 wiki-data/smoke/cold_warm-<UTC时间戳>/，"
                             "cold 从空目录开始，warm 复用同一根目录）")
    parser.add_argument("--out", default="",
                        help="JSONL 输出路径（默认 wiki-data/smoke/cold_warm-<UTC时间戳>.jsonl）")
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

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    wiki_root = Path(args.wiki_root) if args.wiki_root else SMOKE_DIR / f"cold_warm-{stamp}"
    out_path = Path(args.out) if args.out else SMOKE_DIR / f"cold_warm-{stamp}.jsonl"
    wiki_root.mkdir(parents=True, exist_ok=True)

    config: dict[str, Any] = {}
    if args.provider == "config":
        if args.env_file:
            load_env_file(args.env_file)
        config = load_config(Path(args.config))
        print(
            "⚠ 成本警告：config 模式将用真实模型与真实搜索完整跑两次研究（cold + warm），"
            "会产生真实 API 费用；请确认 key 与配额后再继续。"
        )

    question = args.question
    stack = (
        build_config_stack(config, wiki_root=wiki_root)
        if args.provider == "config"
        else build_mock_stack(question)
    )

    tokens_path = wiki_root / "tokens.jsonl"
    strong_info = stack.model_info["strong"]
    cheap_info = stack.model_info["cheap"]
    print("=== cold/warm 对照冒烟（Prior MVP 阶段验收证据）===")
    print(f"question    : {question}")
    print(f"provider    : {stack.mode}")
    print(
        f"model       : strong={strong_info['model']}"
        f"（{strong_info['base_url'] or '—'}）  "
        f"cheap={cheap_info['model']}（{cheap_info['base_url'] or '—'}）"
    )
    print(f"wiki_root   : {wiki_root}")
    print(f"out         : {out_path}")
    print("-" * 76, flush=True)

    rows: list[dict[str, Any]] = []
    try:
        for label in ("cold", "warm"):
            note = "空 Wiki 冷启动" if label == "cold" else "同一 wiki 根目录二次研究"
            print(f"[{label}] run 开始（{note}）", flush=True)
            row = run_once(label, question, wiki_root, stack, tokens_path)
            rows.append(row)
            metrics = row["metrics"]
            print(
                f"[{label}] trace_id={row['trace_id']}  prior_hit={metrics['prior_hit_count']}"
                f"  notes(created/merged)={metrics['notes_created']}/{metrics['notes_merged']}"
                f"  sources={metrics['source_count']}"
                f"  in/out={metrics['input_tokens']}/{metrics['output_tokens']}"
                f"  latency={metrics['latency_ms']}ms",
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
        write_json(
            Path(args.json),
            {
                "gate": "cold_warm_smoke",
                "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "provider_mode": stack.mode,
                "question": question,
                "wiki_root": str(wiki_root),
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

    cold = next(row for row in rows if row["run"] == "cold")
    warm = next(row for row in rows if row["run"] == "warm")
    print("-" * 76)
    print(
        f"✓ 断言通过：cold 空 Wiki 冷启动（prior_hit_count=0，沉淀 "
        f"{cold['metrics']['notes_created']} 条 active note）；"
        f"warm run prior_hit_count={warm['metrics']['prior_hit_count']} > 0；"
        "两次 run 的 input/output tokens 与 "
        "sum_tokens_from_jsonl(<wiki_root>/tokens.jsonl, trace_id) 完全一致（可复算）。"
    )
    print(f"JSONL 已写入 {out_path}")
    return EXIT_PASS


if __name__ == "__main__":
    raise SystemExit(main())
