"""P4b 支撑设施 T8（task-22）单元测试：RunWatchdog 三行为 + stress_run 快速路径。

对应 task-22 简报 Step 1：

① 看门狗超时：per-question timeout 到点 → 该问标记 timeout、失败账本带 trace
   （判定口径写明）+ 错误类型 + 所在 step；事件流被 close()（events() 的 finally
   照常写 run-metrics——已耗计量不丢）；不崩（paused 不触发）。
② 连续失败暂停：连续失败达到上限 → paused=True（暂停不是崩溃）；失败账本每条带
   traceback 末段 + 异常类型；暂停后 run_question 拒绝执行（WatchdogPaused）。
③ 状态行可读：status_line() 一行给出进度/成败计数/连续失败/暂停态。
④ 成功后连续失败计数归零（"连续"语义）。
⑤ stress_run `--questions 3 --provider mock` 快速路径：真跑（ScriptedProvider +
   MockSearch + MockTransport，零网络零 key）、questions.jsonl 真实落盘且 tokens
   与 tokens.jsonl 按 trace 复算一致（诚实计量）、lint 快照真拍且不恶化闸通过、
   退出码 0；汇总写明 cache 命中率未报告（九闸第 9 条）。

超时注入用假时钟 + 慢 provider（现有测试惯例），全程零真实 sleep、零网络。
"""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from researchwiki.llm.accounting import TokenAccountant
from researchwiki.llm.provider import ScriptedProvider, StreamEvent
from researchwiki.loop.agent_loop import AgentLoop
from researchwiki.loop.metrics import sum_tokens_from_jsonl
from researchwiki.loop.watchdog import RunWatchdog, WatchdogPaused, WatchdogSettings
from researchwiki.tools import MockSearch
from researchwiki.wiki.embeddings import MockEmbeddingProvider
from researchwiki.wiki.store import WikiStore

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str) -> ModuleType:
    """按路径加载脚本；先注册进 sys.modules，否则 dataclass 解析注解会失败。"""
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


stress = _load("stress_run", "scripts/stress_run.py")

# ---- 看门狗测试脚手架（ScriptedProvider + MockSearch + tmp_path 沙箱）---------


class FakeClock:
    """可快进的假时钟：watchdog 与慢 provider 共享，零真实等待。"""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class SlowBeforeCallProvider:
    """从第 free_calls+1 次模型调用起，每次 stream() 前把共享假时钟快进 advance 秒。

    与 kill_resume_smoke 的 PauseBeforeCallProvider 同一包装思路：在 stream()
    边界做动作。首次调用放行（plan 正常完成、usage 照常记账），后续调用快进
    ——watchdog 在事件消费间隙的协作式检查即可命中超时，不需要真实 sleep。
    """

    def __init__(
        self, inner: Any, fake_clock: FakeClock, advance: float, *, free_calls: int = 1
    ) -> None:
        self.inner = inner
        self.fake_clock = fake_clock
        self.advance = advance
        self.free_calls = free_calls
        self.calls = 0
        # loop 的记账（_record）会读 model/tier：包装层照 PauseBeforeCallProvider
        # 同款透传，行为与 inner 一致
        self.model = getattr(inner, "model", "slow-mock")
        self.tier = getattr(inner, "tier", "strong")

    def stream(
        self,
        messages: list[Any],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        self.calls += 1
        if self.calls > self.free_calls:
            self.fake_clock.advance(self.advance)
        yield from self.inner.stream(messages, system=system, tools=tools)


class BoomProvider:
    """stream() 即抛 RuntimeError 的 provider（连续失败路径的测试注入）。"""

    model = "boom-strong"
    tier = "strong"

    def stream(
        self,
        messages: list[Any],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        raise RuntimeError("provider 爆炸（测试注入）")
        yield  # pragma: no cover -- 让本函数成为生成器，异常在首次 next() 时抛出


class FakeRouter:
    """按档位返回预置 Provider（test_loop.py 同款）。"""

    def __init__(self, strong: Any, cheap: Any) -> None:
        self._providers = {"strong": strong, "cheap": cheap}

    def get(self, tier: str) -> Any:
        return self._providers[tier]


def _text_turn(text: str) -> list[StreamEvent]:
    return [StreamEvent(type="text_delta", delta=text)]


MOCK_PLAN_TEXT = "- 任务一：检索主流方案\n- 任务二：蒸馏笔记\n"
MOCK_SUMMARY_TEXT = "已检索到主流方案，信息足够，研究结束。"
MOCK_DISTILL_JSON = json.dumps(
    {
        "notes": [
            {
                "text": "Letta 用后台 subagent 在会话空闲期整理长期记忆，跨会话保留长期事实。",
                "entities": ["Letta"],
                "confidence": "high",
            }
        ],
        "conflicts": [],
    },
    ensure_ascii=False,
)


def _good_strong_turns() -> list[list[StreamEvent]]:
    """最小可完成剧本：plan → act（纯文本收尾）→ report。"""
    return [_text_turn(MOCK_PLAN_TEXT), _text_turn(MOCK_SUMMARY_TEXT)]


def make_factory(
    tmp_path: Path,
    *,
    strong: Any = None,
    cheap: Any = None,
    clock: Any = None,
    **kwargs: Any,
) -> Any:
    """构造逐问题的 loop 工厂：每问全新 ScriptedProvider 实例、同一 tmp 沙箱。"""

    def factory(question: str) -> AgentLoop:
        strong_p = strong if strong is not None else ScriptedProvider(
            _good_strong_turns(), model="mock-strong"
        )
        cheap_p = cheap if cheap is not None else ScriptedProvider(
            [_text_turn(MOCK_DISTILL_JSON)], tier="cheap", model="mock-cheap"
        )
        if clock is not None:
            kwargs.setdefault("clock", clock)
        return AgentLoop(
            question,
            router=FakeRouter(strong_p, cheap_p),
            llm_config={
                "strong": {"base_url": "https://mock"},
                "cheap": {"base_url": "https://mock-cheap"},
            },
            accountant=TokenAccountant(tmp_path / "tokens.jsonl"),
            search_provider=MockSearch(),
            wiki_root=tmp_path / "wiki-data",
            embedding=MockEmbeddingProvider(dim=512),
            wiki_config={"fts_tokenizer": "trigram"},
            **kwargs,
        )

    return factory


# ---- ① 默认参数（简报口径：600s / 连续 3 次）----------------------------------


def test_watchdog_settings_defaults() -> None:
    """默认设置与简报一致：单问超时 600s、连续失败上限 3。"""
    settings = WatchdogSettings()
    assert settings.per_question_timeout == 600.0
    assert settings.max_consecutive_failures == 3


# ---- ② 超时：跳过该问、账本带 trace/类型/step、计量不丢、不暂停 ----------------


def test_watchdog_timeout_skips_with_trace(tmp_path: Path) -> None:
    """单问超时 → timeout 结果 + 失败账本（trace 含判定口径）+ close 落盘 run-metrics。"""
    fake = FakeClock()
    slow = SlowBeforeCallProvider(ScriptedProvider(_good_strong_turns()), fake, advance=700.0)
    factory = make_factory(tmp_path, strong=slow, clock=fake)
    watchdog = RunWatchdog(WatchdogSettings(per_question_timeout=600.0), clock=fake)

    outcome = watchdog.run_question("超时问题", factory)

    assert outcome.status == "timeout"
    assert outcome.step.startswith("act")  # plan 完成后、执行阶段超时（事件流推断）
    assert outcome.error_type == "timeout"
    assert "600" in outcome.trace and "协作式" in outcome.trace  # 判定口径写进 trace
    assert outcome.trace_id  # loop 已构造，trace_id 可追溯
    assert len(watchdog.failures) == 1
    record = watchdog.failures[0]
    assert record.error_type == "timeout" and record.step == outcome.step
    assert watchdog.consecutive_failures == 1
    assert not watchdog.paused  # 一次超时远未到上限：暂停语义不触发

    # close() 触发 events() 的 finally：run-metrics 照常落盘（已耗计量不丢）
    run_dir = Path(outcome.run_dir)
    metrics = json.loads((run_dir / "run-metrics.json").read_text(encoding="utf-8"))
    assert metrics["input_tokens"] > 0


# ---- ③ 连续失败：达到上限暂停（不是崩溃）、账本带 traceback、暂停后拒绝执行 ----


def test_watchdog_pauses_after_consecutive_failures(tmp_path: Path) -> None:
    """连续 2 次失败 → paused；账本两条带异常类型与 traceback 末段；再问抛 WatchdogPaused。"""
    boom = BoomProvider()
    factory = make_factory(tmp_path, strong=boom)
    watchdog = RunWatchdog(
        WatchdogSettings(per_question_timeout=600.0, max_consecutive_failures=2),
        clock=FakeClock(),
    )

    first = watchdog.run_question("问题一", factory)
    assert first.status == "error"
    assert first.error_type == "RuntimeError"
    assert "provider 爆炸" in first.trace  # traceback 末段进账本
    assert first.step  # 所在 step 尽力推断（start 之后异常，精确位置以 trace 为准）
    assert not watchdog.paused  # 第 1 次失败未到上限

    watchdog.run_question("问题二", factory)
    assert watchdog.paused
    assert "连续失败" in watchdog.pause_reason
    assert len(watchdog.failures) == 2
    assert watchdog.consecutive_failures == 2

    # 暂停后拒绝新问题（暂停不是崩溃：上层脚本据此收尾）
    with pytest.raises(WatchdogPaused):
        watchdog.run_question("问题三", factory)


# ---- ④ 状态行可读 + 成功后连续失败归零 ---------------------------------------


def test_status_line_readable(tmp_path: Path) -> None:
    """status_line() 一行给出进度/成败/连续失败/暂停态；初始态也可读。"""
    fake = FakeClock()
    watchdog = RunWatchdog(clock=fake)

    line0 = watchdog.status_line()
    assert "完成 0" in line0 and "暂停=否" in line0

    done = watchdog.run_question("问题甲", make_factory(tmp_path, clock=fake))
    assert done.status == "done"
    assert done.step == "finish"  # 正常跑完的 step 推断到 finish
    line1 = watchdog.status_line()
    assert "完成 1" in line1
    assert "连续失败 0/3" in line1  # 成功归零（"连续"语义）
    assert "暂停=否" in line1

    watchdog.run_question("问题乙", make_factory(tmp_path, strong=BoomProvider(), clock=fake))
    line2 = watchdog.status_line()
    assert "完成 1" in line2 and "错误 1" in line2
    assert "连续失败 1/3" in line2
    assert "暂停=否" in line2


def test_consecutive_failures_reset_on_success(tmp_path: Path) -> None:
    """超时失败后再成功：连续失败计数归零，不累积到暂停。"""
    fake = FakeClock()
    slow = SlowBeforeCallProvider(ScriptedProvider(_good_strong_turns()), fake, advance=700.0)
    watchdog = RunWatchdog(WatchdogSettings(per_question_timeout=600.0), clock=fake)

    watchdog.run_question("慢问题", make_factory(tmp_path, strong=slow, clock=fake))
    assert watchdog.consecutive_failures == 1

    watchdog.run_question("好问题", make_factory(tmp_path, clock=fake))
    assert watchdog.consecutive_failures == 0
    assert not watchdog.paused


# ---- ⑤ stress_run 快速路径：真跑、真 JSONL、真 lint 快照、退出码 0 -------------


@pytest.fixture(autouse=True)
def _no_search_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """零网络保证：mock 模式已显式锁定 MockSearch，这里再清掉搜索环境变量兜底。"""
    for var in ("SEARCH_PROVIDER", "TAVILY_API_KEY", "BOCHA_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def test_stress_run_mock_fast_path(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """--questions 3 --provider mock：真跑 3 问、JSONL 落盘、lint 快照、退出码 0。"""
    root = tmp_path / "stress-root"
    out_dir = tmp_path / "stress-out"

    rc = stress.main(
        [
            "--questions",
            "3",
            "--provider",
            "mock",
            "--root",
            str(root),
            "--out",
            str(out_dir),
        ]
    )

    assert rc == 0

    # questions.jsonl 真实落盘：3 行，字段齐全
    rows = [
        json.loads(line)
        for line in (out_dir / "questions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 3
    for row in rows:
        assert row["status"] == "done"
        assert row["question"]
        assert row["trace_id"]
        assert row["elapsed_s"] >= 0.0
        assert row["error_type"] == ""
        # 诚实计量：tokens 与 tokens.jsonl 按 trace 复算一致（计量非估算）
        tokens_in, tokens_out = sum_tokens_from_jsonl(root / "tokens.jsonl", row["trace_id"])
        assert row["input_tokens"] == tokens_in > 0
        assert row["output_tokens"] == tokens_out > 0

    # lint 快照真拍：基线（第 0 问前）+ final（3 < 5，无中间快照）
    assert (out_dir / "lint-0000.json").is_file()
    final = json.loads((out_dir / "lint-final.json").read_text(encoding="utf-8"))
    assert final["notes_total"] >= 3
    # 不恶化闸通过：断链/孤立/无来源三类计数全为 0（骨架页兜底让笔记都被页面引用）
    assert stress.gate_counts(final) == {
        "broken_links": 0,
        "orphan_notes": 0,
        "dangling_evidence": 0,
    }

    # 记忆演化链路真的被锻炼：笔记沉淀 + 页面聚合
    store = WikiStore(root)
    assert len(store.list_notes()) >= 3
    assert len(store.list_pages()) >= 3

    # 机器可读汇总 + 九闸第 9 条（cache 命中率未报告）
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["lint"]["passed"] is True
    assert summary["watchdog"]["paused"] is False
    assert summary["cache_hit_rate"] == "未报告：provider 无 cache 字段"
    stdout_text = capsys.readouterr().out
    assert "未报告" in stdout_text and "cache" in stdout_text


def test_stress_run_requires_questions_or_duration(tmp_path: Path) -> None:
    """--questions 与 --duration 都缺省 → 参数校验失败（非零退出码，不裸跑）。"""
    rc = stress.main(["--provider", "mock", "--root", str(tmp_path / "r"), "--out",
                      str(tmp_path / "o")])
    assert rc == 2
