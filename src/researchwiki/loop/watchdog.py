"""RunWatchdog：连续问题流的看门狗包裹层（P4b 支撑设施 T8）。

PLAN §3.5 支撑设施 6 的口径：连续问题流、超时、工具失败和单个 provider 失败都
必须有可读状态。本层逐问题驱动 AgentLoop（每问题一个 run、共享同一个持久
wiki root），提供四件事：

1. **per-question timeout**：单问超过 ``per_question_timeout`` 秒即收束该 run
   （生成器 ``close()``——``events()`` 的 finally 照常写 run-metrics、关闭索引
   连接，不留半开的 run，已耗计量不丢），失败账本记一条 timeout；
2. **连续失败上限**：连续 ``max_consecutive_failures`` 个问题失败（超时或异常）
   → 暂停（``paused=True``）而不是崩溃——之后 ``run_question`` 拒绝执行并抛
   :class:`WatchdogPaused`，由上层脚本决定收尾方式（ Global Constraints：所有
   自动动作可暂停）；
3. **失败账本**：``failures`` 里每条带 trace（traceback 末段 / 超时判定口径）
   + 错误类型 + 所在 step + trace_id/run_dir，可审计；
4. **status_line()**：一行可读状态（进度、成败计数、连续失败、暂停态、当前
   问题与 step）。

**不侵入 AgentLoop 内部**：T19/T20/T21 刚在 ``events()``/try-finally 上落了
索引关闭 / 压缩触发 / checkpoint 三层结构，本层只在 ``events()`` 迭代外层包裹，
问题怎么研究由调用方给的 ``loop_factory(question)`` 决定（mock 剧本或真实
provider 都行）；对 loop 的全部读取只有 ``trace_id`` / ``run_dir`` 两个属性
（getattr 容错，非 AgentLoop 的替身对象也能包）。

超时判定口径（诚实边界）：**协作式检查**——在事件流每次消费间隙比对时钟；
provider 单次模型调用内部长时间不产事件时，由 provider 自身的 HTTP 超时
（OpenAICompatibleProvider 默认 120s × 3 次重试）兜底。watchdog 不开线程、
不强杀进程。测试注入点：``clock``（默认 ``time.monotonic``）+ 慢 provider 在
stream() 前快进假时钟，见 tests/test_stress.py（零真实 sleep）。

step 推断口径：从事件流尽力推断（start / plan / act:N / observe / distill /
report / finish），解析不了的保持上一值——这是粗粒度的近似，异常的精确位置以
账本里的 traceback 末段为准。
"""

from __future__ import annotations

import re
import sys
import time
import traceback
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass
from typing import Any

# 简报口径的默认值：单问 600s、连续失败 3 次 → 暂停
DEFAULT_PER_QUESTION_TIMEOUT = 600.0
DEFAULT_MAX_CONSECUTIVE_FAILURES = 3

# 失败账本里 traceback 末段的最大长度（整段堆栈可能很长，账本保尾段即可定位）
TRACEBACK_TAIL_CHARS = 2000
# status_line() 里当前问题的截断长度
_QUESTION_SHOWN_CHARS = 24

# data-task detail 里的 act 步号（"第 2 步 · 调用 web_search"）
_ACT_STEP_RE = re.compile(r"第 (\d+) 步")


class WatchdogPaused(RuntimeError):
    """看门狗已暂停（连续失败达到上限）时请求新问题。"""


@dataclass
class WatchdogSettings:
    """看门狗参数（简报口径的默认值见模块常量）。"""

    per_question_timeout: float = DEFAULT_PER_QUESTION_TIMEOUT
    max_consecutive_failures: int = DEFAULT_MAX_CONSECUTIVE_FAILURES


@dataclass
class QuestionOutcome:
    """单问结果：成功 / 超时 / 异常三态，账本与脚本 JSONL 共用同一形态。"""

    index: int  # 问题序号（1 起）
    question: str
    status: str  # "done" | "timeout" | "error"
    step: str  # 事件流推断的所在 step（best-effort，异常精确位置以 trace 为准）
    trace_id: str
    run_dir: str
    elapsed_s: float
    error_type: str = ""  # timeout / 异常类名；成功为空串
    error: str = ""  # 错误消息（截断）
    trace: str = ""  # traceback 末段 / 超时判定口径

    @property
    def failed(self) -> bool:
        return self.status != "done"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def step_from_event(prev: str, event: Mapping[str, Any]) -> str:
    """从事件流推断当前所在 step（尽力而为：解析不了保持上一值）。

    映射依据 agent_loop 的事件协议：start → plan（reasoning-start id=plan）→
    act:N（data-task「执行研究」的 detail 步号）→ observe（研究总结思考块）→
    distill（data-note/data-conflict/蒸馏任务）→ report（text-* id=report）→
    finish。
    """
    etype = str(event.get("type") or "")
    if etype == "start":
        return "start"
    if etype.startswith("reasoning-"):
        return str(event.get("id") or prev) or prev
    if etype == "data-task":
        data = event.get("data") or {}
        title = str(data.get("title") or "")
        detail = str(data.get("detail") or "")
        if "研究计划" in title:
            return "plan"
        if "执行研究" in title:
            match = _ACT_STEP_RE.search(detail)
            return f"act:{match.group(1)}" if match else "act"
        if "蒸馏" in title:
            return "distill"
        return prev
    if etype in ("data-note", "data-conflict"):
        return "distill"
    if etype.startswith("text-"):
        return str(event.get("id") or "report") or prev
    if etype == "finish":
        return "finish"
    return prev


class RunWatchdog:
    """逐问题驱动 AgentLoop 的看门狗（包裹层，见模块 docstring 的职责与边界）。"""

    def __init__(
        self,
        settings: WatchdogSettings | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        on_event: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        self.settings = settings if settings is not None else WatchdogSettings()
        self.clock = clock
        self.on_event = on_event
        self.results: list[QuestionOutcome] = []
        self.consecutive_failures = 0
        self.paused = False
        self.pause_reason = ""
        # 当前问题上下文（status_line 用）
        self.current_index = 0
        self.current_question = ""
        self.current_step = ""

    # ---- 只读视图 ---------------------------------------------------------

    @property
    def failures(self) -> list[QuestionOutcome]:
        """失败账本：所有非 done 的问题结果（每条带 trace + 错误类型 + step）。"""
        return [outcome for outcome in self.results if outcome.failed]

    def status_line(self) -> str:
        """一行可读状态（PLAN §3.5 支撑设施 6：连续问题流必须有可读状态）。"""
        done = sum(1 for outcome in self.results if outcome.status == "done")
        timeouts = sum(1 for outcome in self.results if outcome.status == "timeout")
        errors = sum(1 for outcome in self.results if outcome.status == "error")
        pause = f"暂停=是（{self.pause_reason}）" if self.paused else "暂停=否"
        line = (
            f"watchdog 状态：已驱动 {len(self.results)} 问"
            f"（完成 {done} · 超时 {timeouts} · 错误 {errors}）"
            f" · 连续失败 {self.consecutive_failures}/{self.settings.max_consecutive_failures}"
            f" · {pause}"
        )
        if self.current_question:
            question = self.current_question
            if len(question) > _QUESTION_SHOWN_CHARS:
                question = question[:_QUESTION_SHOWN_CHARS] + "…"
            line += f" · 当前 #{self.current_index}「{question}」@ {self.current_step}"
        return line

    # ---- 驱动 -------------------------------------------------------------

    def run_question(self, question: str, loop_factory: Callable[[str], Any]) -> QuestionOutcome:
        """跑一个问题；返回 :class:`QuestionOutcome`（失败同时进失败账本）。

        loop_factory(question) 返回一个带 ``events()`` 的 run 对象（AgentLoop 或
        行为兼容的替身）。超时 / 异常都收束为该问的失败结果——watchdog 自身绝不
        因单问失败崩溃；连续失败达到上限即暂停（paused=True），之后再调用本方法
        一律抛 :class:`WatchdogPaused`。
        """
        if self.paused:
            raise WatchdogPaused(
                f"看门狗已暂停（{self.pause_reason}），run_question 拒绝新问题"
            )
        self.current_index = len(self.results) + 1
        self.current_question = question
        self.current_step = "start"

        loop: Any = None
        outcome = QuestionOutcome(
            index=self.current_index,
            question=question,
            status="error",
            step="start",
            trace_id="",
            run_dir="",
            elapsed_s=0.0,
        )
        t0 = self.clock()
        deadline = t0 + self.settings.per_question_timeout
        try:
            loop = loop_factory(question)
            outcome.trace_id = str(getattr(loop, "trace_id", "") or "")
            outcome.run_dir = str(getattr(loop, "run_dir", "") or "")
            outcome.step = "start"
            gen: Iterator[Mapping[str, Any]] = loop.events()
            timed_out = False
            for event in gen:
                if self.on_event is not None:
                    self.on_event(event)
                outcome.step = step_from_event(outcome.step, event)
                self.current_step = outcome.step
                # finish 是最后一件事件：自然收尾不算超时（避免"压线完成"误判）
                if event.get("type") != "finish" and self.clock() > deadline:
                    timed_out = True
                    break
            if timed_out:
                self._close_guarded(gen)
                outcome.status = "timeout"
                outcome.error_type = "timeout"
                outcome.error = f"单问超时（{self.settings.per_question_timeout:g}s）"
                outcome.trace = (
                    f"超时判定：事件间隙协作式检查，clock={self.clock() - t0:.1f}s > "
                    f"per_question_timeout={self.settings.per_question_timeout:g}s；"
                    "provider 单次调用内部阻塞由其 HTTP 超时兜底，watchdog 不开线程不强杀。"
                    "事件流已 close()，run-metrics 经 finally 照常落盘。"
                )
            else:
                outcome.status = "done"
        except Exception as exc:  # noqa: BLE001 -- 单问失败必须收束为账本记录，不能掀翻问题流
            outcome.status = "error"
            outcome.error_type = type(exc).__name__
            outcome.error = str(exc)[:500]
            outcome.trace = "".join(traceback.format_exception(exc))[-TRACEBACK_TAIL_CHARS:]
        outcome.elapsed_s = self.clock() - t0
        self._record(outcome)
        return outcome

    # ---- 内部 -------------------------------------------------------------

    def _record(self, outcome: QuestionOutcome) -> None:
        """结果入账本并维护连续失败计数 / 暂停态（成功归零，"连续"语义）。"""
        self.results.append(outcome)
        if not outcome.failed:
            self.consecutive_failures = 0
            return
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.settings.max_consecutive_failures:
            self.paused = True
            self.pause_reason = (
                f"连续失败 {self.consecutive_failures} 个问题达到上限"
                f"（最近：{outcome.error_type}@{outcome.step}）"
            )

    @staticmethod
    def _close_guarded(gen: Iterator[Mapping[str, Any]]) -> None:
        """收束事件流（close() 触发 events() 的 finally）；收束失败不改变超时结论。"""
        try:
            gen.close()
        except Exception as exc:  # noqa: BLE001 -- 清理自身失败只记 stderr，不推翻账本
            print(
                f"[watchdog] 事件流收束异常（超时结论不变）："
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
