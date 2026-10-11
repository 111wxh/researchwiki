#!/usr/bin/env python
"""kill/resume 冒烟：子进程起 run → 硬 kill → 新进程 resume_from 续跑（T7 阶段验收件）。

── 用途（P4b 支撑设施 T7 的阶段验收证据）──────────────────────────────────
mock provider 全程驱动（ScriptedProvider + MockSearch + MockEmbeddingProvider，
零 key 零网络零 API 成本），三个子进程对照：

  baseline  一次性跑完整场研究（对照基线）；
  victim    同剧本起 run，在 kill-after-step（默认 act:1）的 checkpoint 落盘后、
            下一次模型调用前暂停（PauseBeforeCallProvider），父进程等 ready 标记
            出现即 ``proc.kill()``（Windows = TerminateProcess，等同 kill -9，
            Python 清理与 finally 一概不执行）；
  resume    新进程、同 run_dir、``resume_from`` 指向 checkpoint 续跑至完成。

四道硬闸（任一不满足即打印原因、非零退出码，绝不静默通过）：

  闸1  未完成任务继续执行完成：resume 子进程退出码 0、report.md 落盘、
       state.md 阶段 done；且 victim 的 run-metrics.json 缺席（硬 kill 证据：
       events() 的 finally 未执行）。
  闸2  已完成 step 不重复执行：victim + resume 的 provider 调用次数
       ≤ baseline 一次性跑完的调用次数（重复执行必然超限）。
  闸3  笔记无重复：killed 根目录 store 的 active 笔记数与正文与一次性基线一致。
  闸4  全程零 API 成本（结构保证：仅 mock provider）+ 计量连续可复算：
       victim 的 tokens.jsonl 按 trace 汇总 == checkpoint.budget_used（暂停点在
       下一次调用前，无浪费调用）；checkpoint 基线 + resume 新增 == baseline
       全量（预算衔接）；baseline/resume 的 metrics 与 tokens.jsonl 按 trace
       对账一致。

── kill 时机机制（写明）──────────────────────────────────────────────────
用 **PauseBeforeCallProvider**（暂停型 provider 包装）而非父进程轮询：包装在每次
strong 调用前检查 checkpoint 的 completed_step_ids，目标 step 已记 → 写 summary、
落 ready 标记、睡眠等待父进程 kill。父进程轮询标记文件后 kill——不靠时间窗竞速，
victim 的已耗计量恰好等于 checkpoint.budget_used（闸4 的精确对账依赖这一点）。
父进程未按期 kill 的兜底：睡眠到 deadline 后子进程自行退出（退出码 3），绝不
静默继续。

── 怎么跑 ─────────────────────────────────────────────────────────────
  # 零 key 零网络（默认 mock，无 config 模式——闸4 的零成本是结构性的）
  uv run --no-sync python scripts/kill_resume_smoke.py
  # 指定落盘位置 / kill 步
  uv run --no-sync python scripts/kill_resume_smoke.py \
      --root wiki-data/smoke/kr --kill-after-step act:2
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from researchwiki.llm.accounting import TokenAccountant
from researchwiki.llm.provider import ScriptedProvider, StreamEvent, TokenUsage
from researchwiki.loop import checkpoint
from researchwiki.loop.agent_loop import AgentLoop
from researchwiki.loop.metrics import sum_tokens_from_jsonl
from researchwiki.tools import MockSearch
from researchwiki.wiki.embeddings import MockEmbeddingProvider
from researchwiki.wiki.store import WikiStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "wiki-data" / "smoke"

EXIT_PASS = 0
EXIT_ASSERT_FAILED = 1
EXIT_RUN_ERROR = 2
EXIT_WAIT_TIMEOUT = 3  # 子进程专用：父进程未按期 kill，到 deadline 自行退出

DEFAULT_QUESTION = "Agent 长期记忆的主流方案如何做记忆固化与检索？"
DEFAULT_KILL_STEP = "act:1"
READY_TIMEOUT_SECONDS = 60.0
PAUSE_DEADLINE_SECONDS = 120.0

# ---- mock 剧本（与 tests/test_resume.py 同构：plan → act:1(search) → act:2(fetch)
#      → act:3(文本收尾) → report；cheap 一轮蒸馏）------------------------------
MOCK_PLAN_TEXT = "- 任务一：检索主流方案\n- 任务二：把结论蒸馏成笔记\n"
MOCK_SEARCH_QUERY = "agent 长期记忆 方案"
MOCK_FETCH_URL = "https://example.com/article"
MOCK_SUMMARY_TEXT = "已检索到主流方案，信息足够，研究结束。"
MOCK_REPORT_TEXT = "## 研究报告\n\nLetta 的后台 subagent 方案[1] 值得优先试点。\n"
MOCK_SOURCE_URL = "https://github.com/letta-ai/letta"  # MockSearch 夹具第一条
MOCK_DISTILL_JSON = json.dumps(
    {
        "notes": [
            {
                "text": "关于「Agent 长期记忆」的研究结论：Letta 用后台 subagent 在会话空闲期"
                "整理长期记忆，跨会话保留长期事实。",
                "entities": ["Letta"],
                "confidence": "high",
                "source_urls": [MOCK_SOURCE_URL],
            }
        ],
        "conflicts": [],
    },
    ensure_ascii=False,
)
MOCK_USAGE = TokenUsage(input_tokens=1200, output_tokens=180)

# 记忆演化三段（与 cold_warm_smoke 的 MOCK_*_CONFIG 同口径：开着 + 默认阈值）
MOCK_FORMATION_CONFIG: dict[str, Any] = {"enabled": True}
MOCK_MEMORY_UPDATE_CONFIG: dict[str, Any] = {"enabled": True, "dry_run": False}
MOCK_VERIFICATION_CONFIG: dict[str, Any] = {}
CHECKPOINT_CONFIG: dict[str, Any] = {"enabled": True, "every_step": True}


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


def strong_turns() -> list[list[StreamEvent]]:
    return [
        _text_turn(MOCK_PLAN_TEXT),
        _tool_turn([_call("web_search", {"query": MOCK_SEARCH_QUERY}, call_id="s1")]),
        _tool_turn([_call("fetch_url", {"url": MOCK_FETCH_URL}, call_id="f1")]),
        _text_turn(MOCK_SUMMARY_TEXT),
        _text_turn(MOCK_REPORT_TEXT),
    ]


def distill_turn() -> list[StreamEvent]:
    return [_text_turn(MOCK_DISTILL_JSON)]


class ScriptedTierRouter:
    """mock 模式的极简 TierRouter（cold_warm_smoke 同款）。"""

    def __init__(self, strong: Any, cheap: Any) -> None:
        self._providers = {"strong": strong, "cheap": cheap}

    def get(self, tier: str) -> Any:
        return self._providers[tier]


class PauseBeforeCallProvider:
    """在目标 step 的 checkpoint 落盘后、下一次模型调用前暂停的 Provider 包装。

    kill 时机机制（见模块 docstring）：每次 stream() 前检查 checkpoint，目标 step
    已记 → 回调 on_pause（写 summary、落 ready 标记）→ 睡眠等待父进程 kill；
    到 deadline 仍未被 kill（父进程异常）→ 明确退出，绝不静默继续。
    """

    def __init__(
        self,
        inner: Any,
        *,
        run_dir: Path,
        kill_after_step: str,
        on_pause: Any,
        deadline_seconds: float = PAUSE_DEADLINE_SECONDS,
    ) -> None:
        self.inner = inner
        self.model = inner.model
        self.tier = inner.tier
        self.run_dir = run_dir
        self.kill_after_step = kill_after_step
        self.on_pause = on_pause
        self.deadline_seconds = deadline_seconds
        self.paused = False

    def stream(
        self,
        messages: list[Any],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        if not self.paused:
            ck = checkpoint.load(self.run_dir)
            if ck and self.kill_after_step in (ck.get("completed_step_ids") or []):
                self.paused = True
                self.on_pause()
                deadline = time.monotonic() + self.deadline_seconds
                while time.monotonic() < deadline:
                    time.sleep(0.05)
                raise SystemExit(
                    f"kill_resume_smoke：等待父进程 kill 超时（{self.deadline_seconds}s），"
                    "子进程自行退出"
                )
        yield from self.inner.stream(messages, system=system, tools=tools)


# ---- 子进程模式 ------------------------------------------------------------


def build_child_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--wiki-root", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--kill-after-step", default="")
    parser.add_argument("--ready-marker", default="")
    return parser


def write_summary(
    path: Path,
    *,
    role: str,
    strong: Any,
    cheap: Any,
    loop: AgentLoop | None,
    finished: bool,
    run_dir: Path,
) -> None:
    """子进程 run 摘要落盘：victim 的版本在暂停时刻写（kill 后 stdout 不可依赖）。

    wiki 根按本脚本固定布局从 run_dir 反推（``wiki_root/run``）。
    """
    store = WikiStore(run_dir.parent)
    payload = {
        "role": role,
        "strong_calls": len(
            strong.inner.calls if isinstance(strong, PauseBeforeCallProvider) else strong.calls
        ),
        "cheap_calls": len(cheap.calls),
        "trace_id": loop.trace_id if loop else "",
        "finished": finished,
        "input_tokens": loop.input_tokens if loop else 0,
        "output_tokens": loop.output_tokens if loop else 0,
        "notes": len(store.list_notes()),
        "run_dir": str(run_dir),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_child(args: argparse.Namespace) -> int:
    """子进程主体：按 resume_from 裁剪剧本 → 跑 AgentLoop → 写 run 摘要。"""
    run_dir = Path(args.run_dir)
    wiki_root = Path(args.wiki_root)
    resume_from = Path(args.resume_from) if args.resume_from else None

    # 剧本裁剪：从第一个未完成 step 起（恢复进程不重放已完成的模型调用）
    ck = checkpoint.load(resume_from) if resume_from else None
    completed = set(ck.get("completed_step_ids") or []) if ck else set()
    plan_done = "plan" in completed
    act_done = max(
        (
            int(m.group(1))
            for sid in completed
            if (m := re.fullmatch(r"act:(\d+)", sid)) is not None
        ),
        default=0,
    )
    distill_done = "distill" in completed
    strong_script = strong_turns()[(1 if plan_done else 0) + act_done :] or [
        _text_turn("（无剩余 strong 调用）")
    ]
    cheap_script = [] if distill_done else distill_turn()  # type: ignore[assignment]

    strong = ScriptedProvider(strong_script, tier="strong", model="mock-strong", usage=MOCK_USAGE)
    cheap = ScriptedProvider(
        cheap_script or distill_turn(), tier="cheap", model="mock-cheap", usage=MOCK_USAGE
    )

    def write_run_summary(finished: bool) -> None:
        write_summary(
            Path(args.summary_out),
            role="victim" if args.kill_after_step else ("resume" if resume_from else "baseline"),
            strong=strong,
            cheap=cheap,
            loop=loop_holder.get("loop"),
            finished=finished,
            run_dir=run_dir,
        )

    def pause_hook() -> None:
        """victim 暂停钩子：先写 run 摘要（kill 后 stdout 不可依赖），再落 ready 标记。"""
        write_run_summary(False)
        if args.ready_marker:
            Path(args.ready_marker).write_text("ready", encoding="utf-8")

    accountant = TokenAccountant(path=wiki_root / "tokens.jsonl")
    if args.kill_after_step:
        strong = PauseBeforeCallProvider(  # type: ignore[assignment]
            strong,
            run_dir=run_dir,
            kill_after_step=args.kill_after_step,
            on_pause=pause_hook,
        )
    loop_holder: dict[str, AgentLoop | None] = {"loop": None}
    loop = AgentLoop(
        args.question,
        router=ScriptedTierRouter(strong, cheap),
        llm_config={
            "strong": {"base_url": "https://mock"},
            "cheap": {"base_url": "https://mock-cheap"},
        },
        accountant=accountant,
        search_provider=MockSearch(),
        wiki_root=wiki_root,
        run_dir=run_dir,
        embedding=MockEmbeddingProvider(dim=512),
        wiki_config={"fts_tokenizer": "trigram"},
        formation_config=MOCK_FORMATION_CONFIG,
        memory_update_config=MOCK_MEMORY_UPDATE_CONFIG,
        verification_config=MOCK_VERIFICATION_CONFIG,
        checkpoint_config=CHECKPOINT_CONFIG,
        resume_from=resume_from,
    )
    loop_holder["loop"] = loop
    finished = False
    try:
        for event in loop.events():
            if event.get("type") == "data-note":
                data = event.get("data") or {}
                print(f"  [note] {data.get('id')} conf={data.get('confidence')}", flush=True)
        finished = True
    finally:
        if not args.kill_after_step:
            # 正常完成的子进程在收尾写终版摘要（victim 的摘要由暂停回调写）
            write_run_summary(finished)
    return EXIT_PASS if finished else EXIT_ASSERT_FAILED


# ---- 父进程模式 ------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "kill/resume 冒烟：硬 kill 后从 checkpoint 续跑的四闸验收"
            "（详见文件头 docstring）"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--root", default="",
                        help="冒烟落盘根目录（默认 wiki-data/smoke/kill_resume-<UTC时间戳>/）")
    parser.add_argument("--question", default=DEFAULT_QUESTION, help="研究问题（三个子进程共用）")
    parser.add_argument("--kill-after-step", default=DEFAULT_KILL_STEP,
                        help="victim 在该 step 的 checkpoint 落盘后被 kill（默认 act:1）")
    parser.add_argument("--ready-timeout", type=float, default=READY_TIMEOUT_SECONDS,
                        help="等待 victim 就绪标记的超时秒数（默认 60）")
    parser.add_argument("--json", default="", help="机器可读汇总另存为 JSON（可选）")
    return parser


def child_command(
    *,
    run_dir: Path,
    wiki_root: Path,
    question: str,
    summary_out: Path,
    resume_from: Path | None = None,
    kill_after_step: str = "",
    ready_marker: Path | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "kill_resume_smoke.py"),
        "--child",
        "--run-dir", str(run_dir),
        "--wiki-root", str(wiki_root),
        "--question", question,
        "--summary-out", str(summary_out),
    ]
    if resume_from is not None:
        cmd += ["--resume-from", str(resume_from)]
    if kill_after_step:
        cmd += ["--kill-after-step", kill_after_step]
        if ready_marker is not None:
            cmd += ["--ready-marker", str(ready_marker)]
    return cmd


def child_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def read_summary(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def verify_gates(
    *,
    baseline: dict[str, Any],
    victim: dict[str, Any],
    resume: dict[str, Any],
    killed_run_dir: Path,
    ck_at_kill: dict[str, Any],
    baseline_wiki_root: Path,
    killed_wiki_root: Path,
    kill_step: str,
    baseline_rc: int,
    resume_rc: int,
) -> tuple[dict[str, bool], list[str]]:
    """四道硬闸（纯函数，便于单测），返回（闸结论, 失败原因列表）。

    baseline_wiki_root / killed_wiki_root 是 wiki-data 根（notes/、tokens.jsonl 所在），
    对应 run 目录按本脚本固定布局为 ``wiki_root/run``；``ck_at_kill`` 是父进程在
    kill 之后、resume 之前快照的 checkpoint（resume 会覆写同目录 checkpoint.json，
    kill 时刻的证据必须提前留档）。
    """
    failures: list[str] = []
    ck_payload = ck_at_kill
    completed = list(ck_payload.get("completed_step_ids") or [])
    budget_used = dict(ck_payload.get("budget_used") or {})
    metrics_baseline = json.loads(
        (baseline_wiki_root / "run" / "run-metrics.json").read_text(encoding="utf-8")
    )
    metrics_resume = json.loads(
        (killed_run_dir / "run-metrics.json").read_text(encoding="utf-8")
    )

    # 闸1 未完成任务继续执行完成（victim_metrics_missing 取 kill 后、resume 前的快照
    # ——resume 会补写 run-metrics.json，事后检查就不是 kill 证据了）
    state_text = ""
    state_path = killed_run_dir / "state.md"
    if state_path.is_file():
        state_text = state_path.read_text(encoding="utf-8")
    victim_metrics_missing = bool(victim.get("run_metrics_missing"))
    gate1 = (
        baseline_rc == EXIT_PASS
        and resume_rc == EXIT_PASS
        and bool(resume.get("finished"))
        and (killed_run_dir / "report.md").is_file()
        and "阶段：done" in state_text
        and victim_metrics_missing
        and kill_step in completed
    )
    if not gate1:
        failures.append(
            f"闸1 未完成任务继续执行完成：baseline_rc={baseline_rc} resume_rc={resume_rc} "
            f"finished={resume.get('finished')} "
            f"report.md={(killed_run_dir / 'report.md').is_file()} "
            f"state_done={'阶段：done' in state_text} "
            f"victim_metrics_missing={victim_metrics_missing} "
            f"kill_step({kill_step})_in_completed={kill_step in completed}"
        )

    # 闸2 已完成 step 不重复执行：victim + resume ≤ baseline（重复执行必然超限）
    gate2 = all(
        int(victim.get(tier) or 0) + int(resume.get(tier) or 0) <= int(baseline.get(tier) or 0)
        for tier in ("strong_calls", "cheap_calls")
    )
    if not gate2:
        failures.append(
            f"闸2 已完成 step 不重复执行：strong {victim.get('strong_calls')}"
            f"+{resume.get('strong_calls')} vs baseline {baseline.get('strong_calls')}；"
            f"cheap {victim.get('cheap_calls')}+{resume.get('cheap_calls')} "
            f"vs baseline {baseline.get('cheap_calls')}"
        )

    # 闸3 笔记无重复：与一次性基线一致（数量与正文）
    baseline_notes = WikiStore(baseline_wiki_root).list_notes()
    killed_notes = WikiStore(killed_wiki_root).list_notes()
    gate3 = (
        len(baseline_notes) == len(killed_notes)
        and sorted(n.body.strip() for n in baseline_notes)
        == sorted(n.body.strip() for n in killed_notes)
    )
    if not gate3:
        failures.append(
            f"闸3 笔记无重复：baseline {len(baseline_notes)} 条 vs killed {len(killed_notes)} 条"
            "（数量或正文不一致，存在重复/缺失创建）"
        )

    # 闸4 零 API 成本（结构性：仅 mock provider）+ 计量连续可复算
    victim_in, victim_out = sum_tokens_from_jsonl(
        killed_wiki_root / "tokens.jsonl", str(victim.get("trace_id") or "")
    )
    baseline_in, baseline_out = sum_tokens_from_jsonl(
        baseline_wiki_root / "tokens.jsonl", str(baseline.get("trace_id") or "")
    )
    resume_in, resume_out = sum_tokens_from_jsonl(
        killed_wiki_root / "tokens.jsonl", str(resume.get("trace_id") or "")
    )
    reconciliations = [
        # victim 暂停在目标 step 保存后、下一次调用前：已耗 == checkpoint 基线
        (
            victim_in == int(budget_used.get("tokens_in") or 0)
            and victim_out == int(budget_used.get("tokens_out") or 0),
            f"victim tokens.jsonl({victim_in}/{victim_out}) == checkpoint.budget_used"
            f"({budget_used.get('tokens_in')}/{budget_used.get('tokens_out')})",
        ),
        # 预算衔接：checkpoint 基线 + resume 新增 == baseline 全量
        (
            int(budget_used.get("tokens_in") or 0) + int(metrics_resume.get("input_tokens") or 0)
            == int(metrics_baseline.get("input_tokens") or 0)
            and int(budget_used.get("tokens_out") or 0)
            + int(metrics_resume.get("output_tokens") or 0)
            == int(metrics_baseline.get("output_tokens") or 0),
            "budget_used + resume metrics == baseline metrics（预算衔接）",
        ),
        # 正常完成的 run：metrics 与 tokens.jsonl 按 trace 对账一致
        (
            baseline_in == int(metrics_baseline.get("input_tokens") or 0)
            and baseline_out == int(metrics_baseline.get("output_tokens") or 0),
            "baseline metrics == sum_tokens_from_jsonl(trace)",
        ),
        (
            resume_in == int(metrics_resume.get("input_tokens") or 0)
            and resume_out == int(metrics_resume.get("output_tokens") or 0),
            "resume metrics == sum_tokens_from_jsonl(trace)",
        ),
    ]
    gate4 = all(ok for ok, _ in reconciliations)
    if not gate4:
        failures.extend(
            f"闸4 计量对账失败：{why}" for ok, why in reconciliations if not ok
        )

    gates = {
        "task_completed": gate1,
        "no_step_redone": gate2,
        "notes_not_duplicated": gate3,
        "zero_api_cost": gate4,
    }
    return gates, failures


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001 -- 老环境/被替换的 stdout 上忽略
        pass
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--child" in argv:
        # 子进程模式：独立的参数面（--run-dir/--wiki-root/...），不进父进程四闸流程
        return run_child(build_child_parser().parse_args(argv))
    args = build_parser().parse_args(argv)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    root = Path(args.root) if args.root else DEFAULT_ROOT / f"kill_resume-{stamp}"
    root.mkdir(parents=True, exist_ok=True)
    baseline_root = root / "baseline"
    killed_root = root / "killed"
    baseline_run_dir = baseline_root / "run"
    killed_run_dir = killed_root / "run"
    marker = root / "ready.marker"

    print("=== kill/resume 冒烟（P4b T7 阶段验收证据）===")
    print(f"question    : {args.question}")
    print("provider    : mock（ScriptedProvider + MockSearch，全程零 API 成本——结构保证）")
    print(f"kill_step   : {args.kill_after_step}（checkpoint 落盘后、下一次模型调用前）")
    print(f"root        : {root}")
    print("-" * 76, flush=True)

    try:
        # 1) baseline：一次性跑完（对照基线）
        baseline_summary = root / "summary-baseline.json"
        proc = subprocess.run(
            child_command(
                run_dir=baseline_run_dir,
                wiki_root=baseline_root,
                question=args.question,
                summary_out=baseline_summary,
            ),
            cwd=PROJECT_ROOT,
            env=child_env(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
        )
        baseline_rc = proc.returncode
        if baseline_rc != EXIT_PASS:
            print(proc.stdout[-2000:])
            print(proc.stderr[-2000:])
            print(f"✗ baseline 子进程异常退出（{baseline_rc}）")
            return EXIT_RUN_ERROR
        baseline = read_summary(baseline_summary)
        print(
            f"[baseline] 一次跑完：strong {baseline['strong_calls']} 次 /"
            f" cheap {baseline['cheap_calls']} 次 / notes {baseline['notes']}"
            f" / in/out={baseline['input_tokens']}/{baseline['output_tokens']}",
            flush=True,
        )

        # 2) victim：跑到目标 step checkpoint 落盘 → 硬 kill
        victim_summary = root / "summary-victim.json"
        victim_proc = subprocess.Popen(
            child_command(
                run_dir=killed_run_dir,
                wiki_root=killed_root,
                question=args.question,
                summary_out=victim_summary,
                kill_after_step=args.kill_after_step,
                ready_marker=marker,
            ),
            cwd=PROJECT_ROOT,
            env=child_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        try:
            deadline = time.monotonic() + args.ready_timeout
            while time.monotonic() < deadline and not marker.is_file():
                if victim_proc.poll() is not None:
                    break  # 子进程提前退出（剧本/断点异常）：下方 summary 校验会拦住
                time.sleep(0.02)
        finally:
            killed = victim_proc.poll() is None
            if killed:
                victim_proc.kill()  # Windows = TerminateProcess：finally 一概不执行
            stdout_text = victim_proc.communicate()[0] or ""
        marker.unlink(missing_ok=True)
        if not victim_summary.is_file():
            print((stdout_text or "")[-2000:])
            print(
                f"✗ victim 未在 {args.ready_timeout}s 内到达 "
                f"{args.kill_after_step} 的 checkpoint（未 kill 或剧本异常）"
            )
            return EXIT_RUN_ERROR
        victim = read_summary(victim_summary)
        victim["killed"] = killed
        victim["killed_at_step"] = args.kill_after_step
        # kill 证据快照（必须在 resume 之前取：resume 会补写 run-metrics 与
        # checkpoint.json，事后检查/读取就不是 kill 时刻的证据了）
        victim["run_metrics_missing"] = not (killed_run_dir / "run-metrics.json").is_file()
        ck_at_kill_path = root / "checkpoint-at-kill.json"
        if (killed_run_dir / "checkpoint.json").is_file():
            ck_at_kill_path.write_text(
                (killed_run_dir / "checkpoint.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            ck_at_kill = dict(json.loads(ck_at_kill_path.read_text(encoding="utf-8")))
        else:
            ck_at_kill = {}
        victim["completed_step_ids"] = list(ck_at_kill.get("completed_step_ids") or [])
        print(
            f"[victim]   checkpoint({args.kill_after_step}) 落盘后硬 kill："
            f"strong {victim['strong_calls']} 次 / cheap {victim['cheap_calls']} 次"
            f" / run-metrics 缺席={victim['run_metrics_missing']}（finally 未执行 = 硬 kill 证据）",
            flush=True,
        )

        # 3) resume：新进程从断点续跑至完成
        resume_summary = root / "summary-resume.json"
        proc = subprocess.run(
            child_command(
                run_dir=killed_run_dir,
                wiki_root=killed_root,
                question=args.question,
                summary_out=resume_summary,
                resume_from=killed_run_dir,
            ),
            cwd=PROJECT_ROOT,
            env=child_env(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
        )
        resume_rc = proc.returncode
        if not resume_summary.is_file():
            print(proc.stdout[-2000:])
            print(proc.stderr[-2000:])
            print(f"✗ resume 子进程异常退出（{resume_rc}），未写 run 摘要")
            return EXIT_RUN_ERROR
        resume = read_summary(resume_summary)
        print(
            f"[resume]   续跑至完成（rc={resume_rc}）：strong {resume['strong_calls']} 次 /"
            f" cheap {resume['cheap_calls']} 次 / notes {resume['notes']}"
            f" / in/out={resume['input_tokens']}/{resume['output_tokens']}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001 -- 环境/子进程问题不作数，明确退出而非裸栈
        print(f"✗ 运行异常：{type(exc).__name__}: {exc}")
        return EXIT_RUN_ERROR

    gates, failures = verify_gates(
        baseline=baseline,
        victim=victim,
        resume=resume,
        killed_run_dir=killed_run_dir,
        ck_at_kill=ck_at_kill,
        baseline_wiki_root=baseline_root,
        killed_wiki_root=killed_root,
        kill_step=args.kill_after_step,
        baseline_rc=baseline_rc,
        resume_rc=resume_rc,
    )
    print("-" * 76)
    for name, ok in gates.items():
        print(f"{'✓' if ok else '✗'} 闸：{name}")
    if args.json:
        payload = {
            "gate": "kill_resume_smoke",
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "question": args.question,
            "kill_step": args.kill_after_step,
            "root": str(root),
            "passed": not failures,
            "failures": failures,
            "gates": gates,
            "baseline": baseline,
            "victim": victim,
            "resume": resume,
        }
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"机器可读汇总已写入 {args.json}")
    if failures:
        for failure in failures:
            print(f"✗ 断言失败：{failure}")
        return EXIT_ASSERT_FAILED
    print(
        f"✓ 四闸通过：victim 在 {args.kill_after_step} checkpoint 后被硬 kill（run-metrics 缺席），"
        "resume 续跑完成；已完成 step 未重复执行；笔记与一次性基线一致；"
        "计量连续（checkpoint 基线 + resume 新增 == baseline 全量），全程零 API 成本。"
    )
    return EXIT_PASS


if __name__ == "__main__":
    raise SystemExit(main())
