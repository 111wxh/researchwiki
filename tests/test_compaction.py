"""P4b 支撑设施 T6（task-20）测试：上下文预算估算 + 稳定前缀 + 滚动压缩。

六组用例对应 task-20 简报 Step 1：

① 估算函数（accounting）对中英文样例的确定性与单调性（更长 → 更大）；
② 触发数学：估算跨过阈值 → 压缩（摘要消息在保留区之前、最近 K 轮原文保留、
   计划里的未解决任务清单原文入摘要）；未跨阈值 → compacted=False 原样返回；
③ ``[context].enabled = false`` 逃生阀永不压缩；段缺失 = 不接线 = 零行为变化；
④ compactions.jsonl 审计事件字段完整（追加式 + 断电容错，手法同 T17 jobs.jsonl）；
⑤ 稳定前缀不变量：同配置两次构造 AgentLoop → system prompt 与工具 schema 逐字节
   相同（工具注册顺序相同）；
⑥ summarizer 抛异常 → 保守降级不压缩 + 事件记录 status="failed"。

全部零网络、零真实模型（ScriptedProvider / fake summarizer / MockSearch），
估算与计量分离：估算数字只进 compactions.jsonl，绝不进 tokens.jsonl。
"""

import json
from pathlib import Path

import httpx
import pytest

from researchwiki.llm.accounting import (
    TokenAccountant,
    estimate_context_tokens,
    estimate_message_tokens,
    estimate_tools_tokens,
)
from researchwiki.llm.provider import Message, ScriptedProvider, StreamEvent
from researchwiki.loop.agent_loop import AgentLoop
from researchwiki.loop.compaction import (
    CompactionResult,
    ContextSettings,
    context_settings_from_config,
    maybe_compact,
    read_compaction_events,
    record_compaction_event,
)
from researchwiki.tools import MockSearch

QUESTION = "agent 记忆方案对比"
PLAN_TEXT = "- 任务一：检索主流方案\n- 任务二：阅读核心来源\n- [ ] 校验引用编号\n"
REPORT_TEXT = "## 研究报告\n\n基于已有信息给出结论。\n"
SUMMARY_TEXT = "已确认：Letta 与 MemGPT 是主流方案；尚未完成：引用编号校验。"
EMPTY_DISTILL_JSON = json.dumps({"notes": [], "conflicts": []}, ensure_ascii=False)

_BIG_SENTENCE = "智能体的长期记忆需要滚动压缩来保住上下文预算。"
BIG_PAGE = "<html><body>" + _BIG_SENTENCE * 100 + "</body></html>"


# ---- 脚手架 ------------------------------------------------------------------


def text_turn(text: str) -> list[StreamEvent]:
    return [StreamEvent(type="text_delta", delta=text)]


def tool_turn(calls: list[dict]) -> list[StreamEvent]:
    return [StreamEvent(type="tool_calls", tool_calls=calls)]


def call(name: str, arguments: dict, *, id: str = "call-1") -> dict:
    return {
        "id": id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
    }


class FakeRouter:
    def __init__(self, strong: ScriptedProvider, cheap: ScriptedProvider | None = None) -> None:
        self._providers = {"strong": strong}
        if cheap is not None:
            self._providers["cheap"] = cheap

    def get(self, tier: str) -> ScriptedProvider:
        if tier not in self._providers:
            raise AssertionError(f"测试未配置 {tier} 档 Provider，却被请求了")
        return self._providers[tier]


class RecordingProvider(ScriptedProvider):
    """记录每次调用收到的 system 与 tools 实参（前缀稳定性断言用）。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.systems: list[str | None] = []
        self.tools_seen: list[list[dict] | None] = []

    def stream(self, messages, *, system=None, tools=None):
        self.systems.append(system)
        self.tools_seen.append(tools)
        yield from super().stream(messages, system=system, tools=tools)


def big_history(*, turns: int = 4, chars: int = 400) -> list[Message]:
    """合成一段研究 history：问题 + 计划 + N 个「assistant(tool_calls) + tool」轮。

    tool 结果为纯中文长文（chars 字 → 约 chars token 估算），用于稳定跨过阈值。
    """
    messages = [
        Message(role="user", content=f"研究问题：{QUESTION}"),
        Message(role="assistant", content=PLAN_TEXT),
    ]
    for i in range(turns):
        messages.append(
            Message(
                role="assistant",
                content=f"第 {i} 步：检索",
                tool_calls=[
                    {
                        "id": f"c{i}",
                        "type": "function",
                        "function": {"name": "web_search", "arguments": "{}"},
                    }
                ],
            )
        )
        messages.append(
            Message(role="tool", tool_call_id=f"c{i}", name="web_search", content="发现" * chars)
        )
    return messages


def compact_settings(**overrides) -> ContextSettings:
    defaults = {
        "max_context_tokens": 1000,
        "compaction_threshold": 0.5,
        "keep_recent_turns": 2,
        "enabled": True,
    }
    defaults.update(overrides)
    return ContextSettings(**defaults)


def fake_summarizer(messages: list[Message]) -> str:
    """fake summarizer：拼接旧轮次首条内容，证明它收到的是待压缩的旧消息。"""
    return SUMMARY_TEXT


# ---- ① 估算函数：确定性与单调性 ------------------------------------------------


def test_estimate_text_tokens_cjk_and_ascii_samples() -> None:
    """中英文样例：CJK 1 token/字、ASCII 1 token/4 字符，同输入恒同值。"""
    from researchwiki.llm.accounting import _estimate_text_tokens

    assert _estimate_text_tokens("") == 0
    assert _estimate_text_tokens("研究记忆") == 4  # 4 个汉字 → 4 token
    assert _estimate_text_tokens("研究记忆。") == 5  # 中文句号也是东亚字符 → 1 token
    assert _estimate_text_tokens("abcd") == 1  # 4 个 ASCII → 1 token
    assert _estimate_text_tokens("abc") == 1  # 不足 4 字符向上取整
    assert _estimate_text_tokens("研究ab") == 3  # 2 汉字 + ceil(2/4)
    # 确定性：同一输入重复计算结果一致
    sample = "上下文预算 Context Budget，估算而非计量 12345"
    assert _estimate_text_tokens(sample) == _estimate_text_tokens(sample)


def test_estimate_message_tokens_monotonic_and_covers_tool_payload() -> None:
    """消息估算单调（更长 → 更大），且覆盖 tool_calls 载荷与 dict 形态。"""
    base = Message(role="tool", tool_call_id="c1", name="web_search", content="结果" * 50)
    longer = Message(role="tool", tool_call_id="c1", name="web_search", content="结果" * 500)
    assert estimate_message_tokens(base) < estimate_message_tokens(longer)

    with_payload = Message(
        role="assistant", content="", tool_calls=[call("web_search", {"query": "q"})]
    )
    without_payload = Message(role="assistant", content="")
    assert estimate_message_tokens(without_payload) < estimate_message_tokens(with_payload)

    # Message dataclass 与等价 dict 形态估算一致（接口接受两种输入）
    as_dict = {
        "role": "tool",
        "content": "结果" * 50,
        "tool_call_id": "c1",
        "name": "web_search",
        "tool_calls": None,
    }
    assert estimate_message_tokens(as_dict) == estimate_message_tokens(base)

    # 单调性（中英文混合，user 角色样例）：按族递增 + 逐字追加不减
    ascii_family = ["a", "ab", "abcd", "abcd" * 10]
    cjk_family = ["研", "研究", "研究" * 100, "研究" * 100 + "混合文本"]
    for family in (ascii_family, cjk_family):
        ests = [estimate_message_tokens(Message(role="user", content=t)) for t in family]
        assert ests == sorted(ests)
    # 逐字符追加：估算不回退；追加一个汉字恰好 +1 token
    growing = ""
    estimates = []
    for ch in "研究abcd记忆":
        growing += ch
        estimates.append(estimate_message_tokens(Message(role="user", content=growing)))
    assert estimates == sorted(estimates)
    assert estimate_message_tokens(Message(role="user", content="研究")) == (
        estimate_message_tokens(Message(role="user", content="研")) + 1
    )


def test_estimate_tools_and_context_tokens_deterministic() -> None:
    """工具 schema 估算：空列表为 0、单调、确定性（与顺序无关）；上下文 = 消息 + 工具。"""
    tools = [
        {
            "type": "function",
            "function": {"name": "web_search", "description": "网页检索", "parameters": {}},
        },
        {
            "type": "function",
            "function": {"name": "fetch_url", "description": "抓取网页", "parameters": {}},
        },
    ]
    assert estimate_tools_tokens([]) == 0
    assert estimate_tools_tokens(None) == 0
    assert estimate_tools_tokens(tools[:1]) < estimate_tools_tokens(tools)
    # 逐个 schema 序列化后求和：与列表顺序无关（确定性）
    assert estimate_tools_tokens(tools) == estimate_tools_tokens([tools[1], tools[0]])

    messages = big_history(turns=2)
    assert estimate_context_tokens(messages, tools) == (
        sum(estimate_message_tokens(m) for m in messages) + estimate_tools_tokens(tools)
    )
    assert estimate_context_tokens(messages) == estimate_context_tokens(messages, None)


def test_estimate_is_not_metering_pure_functions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """估算非计量：估算函数是纯函数——不实例化 TokenAccountant、不落任何计量文件。

    TokenAccountant 构造即创建 tokens.jsonl；chdir 到空目录后调用估算函数，
    目录里不得出现任何文件（估算不写盘的直接证明）。
    """
    monkeypatch.chdir(tmp_path)
    messages = [Message(role="user", content="问题"), Message(role="assistant", content="回答")]
    assert estimate_message_tokens(messages[0]) > 0
    assert estimate_context_tokens(messages, [{"type": "function", "function": {}}]) > 0
    assert estimate_tools_tokens([{"type": "function", "function": {}}]) > 0
    assert list(tmp_path.iterdir()) == []  # 零写盘


# ---- ② 触发数学 ----------------------------------------------------------------


def test_maybe_compact_triggers_and_keeps_recent_turns_verbatim() -> None:
    """估算跨过阈值 → 压缩：摘要消息在保留区之前，最近 K 轮原文（同一批对象）保留。"""
    messages = big_history(turns=4, chars=400)
    before = list(messages)
    pre_estimate = sum(estimate_message_tokens(m) for m in messages)

    result = maybe_compact(messages, compact_settings(), summarizer=fake_summarizer)

    assert result.compacted is True
    assert result.pre_tokens == pre_estimate
    assert result.post_tokens < result.pre_tokens  # 压缩后估算确实变小
    assert result.kept_turns == 2
    assert result.summary == SUMMARY_TEXT
    # 历史被原地改写：[摘要 user 消息, 最近 2 轮原文]
    # （big_history 共 5 轮：问题+计划合为一轮 + 4 个工具轮；keep=2 → 压 3 轮）
    assert len(messages) == 1 + 4  # 摘要 + 2 轮 × (assistant + tool)
    head = messages[0]
    assert head.role == "user"
    assert head.content.startswith("【研究进展摘要·已压缩 3 轮】")
    assert SUMMARY_TEXT in head.content
    # 摘要消息在保留区之前，保留区是原对象（原文未动）
    assert messages[1:] == before[-4:]
    assert messages[1] is before[-4]
    # 未解决任务清单（计划原文）保留进摘要消息
    assert "- [ ] 校验引用编号" in head.content


def test_maybe_compact_below_threshold_returns_unchanged() -> None:
    """未跨阈值 → compacted=False，消息列表原样返回（同对象、同长度）。"""
    messages = [Message(role="user", content="小问题"), Message(role="assistant", content="小计划")]
    snapshot = list(messages)

    result = maybe_compact(messages, compact_settings(), summarizer=fake_summarizer)

    assert result.compacted is False
    assert result.pre_tokens == result.post_tokens
    assert messages == snapshot and len(messages) == 2
    assert messages[0] is snapshot[0]


def test_maybe_compact_never_compacts_down_to_nothing() -> None:
    """旧轮次不足（轮数 ≤ keep_recent_turns）时不压缩：不能把历史压到只剩摘要。"""
    messages = big_history(turns=2, chars=5000)
    snapshot = list(messages)

    result = maybe_compact(
        messages, compact_settings(keep_recent_turns=4), summarizer=fake_summarizer
    )

    assert result.compacted is False
    assert messages == snapshot


def test_maybe_compact_keeps_whole_turns_for_protocol() -> None:
    """整轮保留：assistant(tool_calls) 与其 tool 结果永不分离（OpenAI 协议配对）。"""
    messages = big_history(turns=4, chars=400)
    maybe_compact(messages, compact_settings(keep_recent_turns=1), summarizer=fake_summarizer)
    roles = [m.role for m in messages]
    assert roles[0] == "user"  # 摘要
    # 每条 assistant(tool_calls) 之后必须紧跟其 tool 结果
    for i, m in enumerate(messages[1:], start=1):
        if m.role == "assistant" and m.tool_calls:
            assert i + 1 < len(messages) and messages[i + 1].role == "tool"


def test_context_settings_fallback_chain() -> None:
    """[context] 段缺失 = 不接线（None）；threshold 缺键回退 [runtime]，再缺用默认。"""
    assert context_settings_from_config(None) is None
    # 段缺失（显式空 full-config 形态等价于没传 [context]）→ 不接线
    assert context_settings_from_config(None, {"compaction_threshold": 0.7}) is None

    # [context] 显式值优先
    settings = context_settings_from_config({"compaction_threshold": 0.5})
    assert settings is not None and settings.compaction_threshold == 0.5
    # [context] 缺键 → 回退 [runtime].compaction_threshold（迁移路径）
    settings = context_settings_from_config({}, {"compaction_threshold": 0.8})
    assert settings is not None and settings.compaction_threshold == 0.8
    # 两处都没有 → 默认 0.7
    settings = context_settings_from_config({})
    assert settings is not None and settings.compaction_threshold == 0.7
    # 默认值：131072 / 0.7 / 4 / enabled
    assert settings.max_context_tokens == 131072
    assert settings.keep_recent_turns == 4
    assert settings.enabled is True
    # 逃生阀开关
    settings = context_settings_from_config({"enabled": False})
    assert settings is not None and settings.enabled is False


# ---- ③ 逃生阀与不接线：零行为变化 ----------------------------------------------


def test_maybe_compact_never_runs_when_disabled() -> None:
    """enabled=false 逃生阀：即使估算远超阈值也永不压缩、历史原样。"""
    messages = big_history(turns=4, chars=400)
    snapshot = list(messages)

    result = maybe_compact(
        messages, compact_settings(enabled=False), summarizer=fake_summarizer
    )

    assert result.compacted is False
    assert messages == snapshot


def _compaction_loop_turns() -> list[list[StreamEvent]]:
    """strong 脚本：计划 → 两次抓取大页面（tool 结果巨大）→ 文本收尾 → 报告。"""
    return [
        text_turn(PLAN_TEXT),
        tool_turn([call("fetch_url", {"url": "https://example.com/a"}, id="ca")]),
        tool_turn([call("fetch_url", {"url": "https://example.com/b"}, id="cb")]),
        text_turn("两次抓取完成，研究结束。"),
        text_turn(REPORT_TEXT),
    ]


def _big_fetch_transport() -> httpx.MockTransport:
    return httpx.MockTransport(lambda request: httpx.Response(200, html=BIG_PAGE))


def _make_compaction_loop(tmp_path: Path, run_name: str, *, context_config, cheap_turns=None):
    cheap = ScriptedProvider(
        cheap_turns or [text_turn(EMPTY_DISTILL_JSON)], tier="cheap", model="mock-cheap"
    )
    strong = ScriptedProvider(_compaction_loop_turns(), model="mock-strong")
    loop = AgentLoop(
        QUESTION,
        router=FakeRouter(strong, cheap),
        llm_config={
            "strong": {"base_url": "https://mock"},
            "cheap": {"base_url": "https://mock-cheap"},
        },
        accountant=TokenAccountant(tmp_path / run_name / "tokens.jsonl"),
        search_provider=MockSearch(),
        wiki_root=tmp_path / run_name / "wiki-data",
        run_dir=tmp_path / run_name / "run",
        fetch_transport=_big_fetch_transport(),
        context_config=context_config,
    )
    return loop, strong, cheap


def accountant_steps(tmp_path: Path, run_name: str) -> list[dict]:
    lines = (tmp_path / run_name / "tokens.jsonl").read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines]


def test_agent_loop_compacts_history_when_threshold_crossed(tmp_path: Path) -> None:
    """loop 通路：act 循环内触发压缩——act:3 请求以摘要消息开头，事件落盘、记账含 compact。"""
    context = {
        "enabled": True,
        "max_context_tokens": 500,
        "compaction_threshold": 0.1,
        "keep_recent_turns": 2,
    }
    loop, strong, cheap = _make_compaction_loop(
        tmp_path,
        "trigger",
        context_config=context,
        cheap_turns=[text_turn(SUMMARY_TEXT), text_turn(EMPTY_DISTILL_JSON)],
    )
    events = list(loop.events())
    assert events[0]["type"] == "start" and events[-1]["type"] == "finish"

    # 调用序列：plan / act:1 / act:2 / act:3 / report；act:3 与 report 以摘要开头
    assert len(strong.calls) == 5
    act3 = strong.calls[3]
    assert act3[0].role == "user"
    assert act3[0].content.startswith("【研究进展摘要·已压缩 1 轮】")
    assert SUMMARY_TEXT in act3[0].content
    # 摘要之后是最近 2 轮的原文（两次 fetch 的 tool 结果）
    assert [m.role for m in act3[1:]] == ["assistant", "tool", "assistant", "tool"]
    # report 请求同样以摘要开头（压缩后的 history + 报告指令）
    assert strong.calls[4][0].content.startswith("【研究进展摘要·已压缩 1 轮】")

    # cheap 档先当 summarizer（第 1 次调用，收到的是待压缩旧轮次：问题+计划）
    # 再当蒸馏器（第 2 次调用）
    assert len(cheap.calls) == 2
    assert [m.role for m in cheap.calls[0]] == ["user", "assistant"]
    assert cheap.calls[0][0].content == QUESTION

    # 审计事件：字段完整（④ 的一部分在这里顺带断言 loop 侧接线）
    events_lines = read_compaction_events(tmp_path / "trigger" / "run")
    assert len(events_lines) == 1
    event = events_lines[0]
    assert {"index", "at", "pre_tokens", "post_tokens", "kept_turns", "summary_chars"} <= set(
        event
    )
    assert event["index"] == 1
    # 触发时估算确实跨过阈值（0.1 × 500）。注意 post 可能大于 pre：本场景被压缩
    # 的旧轮次是「问题+计划」小轮，保留区是两个大工具轮——压缩判定只看触发时的
    # 估算（post<pre 的缩减断言在单元测试 ② 用大体量旧轮次验证）
    assert event["pre_tokens"] > 500 * 0.1
    assert event["post_tokens"] > 0
    assert event["kept_turns"] == 2
    assert event["summary_chars"] == len(SUMMARY_TEXT)

    # state.md / run-metrics 不出现压缩字段（run-metrics 14 字段契约不动）
    metrics = json.loads(
        (tmp_path / "trigger" / "run" / "run-metrics.json").read_text(encoding="utf-8")
    )
    assert "compactions" not in metrics and "compaction" not in " ".join(metrics)

    # 估算与计量分账的端到端证据：真实 summarizer 调用正常计量（step="compact"
    # 一行进 tokens.jsonl，模型 mock-cheap），估算数字只进 compactions.jsonl
    steps = [r["step"] for r in accountant_steps(tmp_path, "trigger")]
    assert steps == ["plan", "act:1", "act:2", "compact", "act:3", "distill", "report"]
    compact_row = next(r for r in accountant_steps(tmp_path, "trigger") if r["step"] == "compact")
    assert compact_row["model"] == "mock-cheap"


def test_agent_loop_disabled_and_missing_section_change_nothing(tmp_path: Path) -> None:
    """enabled=false 逃生阀与段缺失（不接线）：历史不压缩、零留痕，行为与接入前一致。"""
    for run_name, context_config in (("disabled", {"enabled": False}), ("missing", None)):
        loop, strong, _ = _make_compaction_loop(tmp_path, run_name, context_config=context_config)
        list(loop.events())
        # act:3 请求仍是完整历史（无摘要消息）
        act3 = strong.calls[3]
        assert act3[0].content == QUESTION  # 原始问题原样在首位
        assert [m.role for m in act3] == [
            "user",
            "assistant",
            "assistant",
            "tool",
            "assistant",
            "tool",
        ]
        # 零留痕：compactions.jsonl 不存在；context_settings 未接线
        assert not (tmp_path / run_name / "run" / "compactions.jsonl").exists()
        assert loop.compaction_count == 0
        # 接线状态：段缺失 = settings 为 None；enabled=false = settings 保留但逃生阀关
        if context_config is None:
            assert loop.context_settings is None
        else:
            assert loop.context_settings is not None
            assert loop.context_settings.enabled is False


# ---- ④ compactions.jsonl 审计：追加式 + 断电容错 -------------------------------


def test_compaction_event_fields_and_incremental_append(tmp_path: Path) -> None:
    """事件字段完整（index/at/pre_tokens/post_tokens/kept_turns/summary_chars）。"""
    run_dir = tmp_path / "run"
    result = CompactionResult(
        compacted=True, pre_tokens=1200, post_tokens=480, kept_turns=2, summary="摘要内容四个字"
    )
    event1 = record_compaction_event(run_dir, index=1, result=result)
    assert event1["index"] == 1
    assert event1["pre_tokens"] == 1200 and event1["post_tokens"] == 480
    assert event1["kept_turns"] == 2
    assert event1["summary_chars"] == len("摘要内容四个字")
    assert event1["at"]  # 时间戳存在

    event2 = record_compaction_event(run_dir, index=2, result=result)
    assert event2["index"] == 2

    events = read_compaction_events(run_dir)
    assert [e["index"] for e in events] == [1, 2]


def test_compaction_ledger_tolerates_truncated_tail(tmp_path: Path) -> None:
    """断电容错（照抄 T17 jobs.jsonl 手法）：尾行残缺/UTF-8 截断不拖垮读取与追加。"""
    run_dir = tmp_path / "run"
    result = CompactionResult(
        compacted=True, pre_tokens=100, post_tokens=40, kept_turns=2, summary="摘要"
    )
    record_compaction_event(run_dir, index=1, result=result)

    # 模拟断电：在 UTF-8 序列中间截断的半行（无换行结尾）
    with (run_dir / "compactions.jsonl").open("ab") as f:
        f.write(b'{"index": 2, "at": "20' + "残".encode()[:2])

    # 读取：坏行跳过，既有记录不受影响
    events = read_compaction_events(run_dir)
    assert [e["index"] for e in events] == [1]

    # 追加：先补换行再写，新记录不与截断行粘连
    record_compaction_event(run_dir, index=3, result=result)
    events = read_compaction_events(run_dir)
    assert [e["index"] for e in events] == [1, 3]


# ---- ⑤ 稳定前缀不变量 ----------------------------------------------------------


def _stability_loop(tmp_path: Path, run_name: str) -> tuple[AgentLoop, RecordingProvider]:
    """同配置构造一个带记录能力的 loop（每次调用独立 Provider 实例）。"""
    strong = RecordingProvider(_compaction_loop_turns(), model="mock-strong")
    cheap = ScriptedProvider([text_turn(EMPTY_DISTILL_JSON)], tier="cheap", model="mock-cheap")
    loop = AgentLoop(
        QUESTION,
        router=FakeRouter(strong, cheap),
        llm_config={"strong": {"base_url": "https://mock"}, "cheap": {"base_url": ""}},
        search_provider=MockSearch(),
        wiki_root=tmp_path / run_name / "wiki-data",
        run_dir=tmp_path / run_name / "run",
        embedding=None,
        wiki_config={"fts_tokenizer": "trigram"},
    )
    return loop, strong


def test_stable_prefix_two_constructions_byte_identical(tmp_path: Path) -> None:
    """同配置两次构造 + 完整运行：system prompt 逐字节相同、工具 schema 列表逐项相同。"""
    loop1, strong1 = _stability_loop(tmp_path, "run-a")
    loop2, strong2 = _stability_loop(tmp_path, "run-b")
    list(loop1.events())
    list(loop2.events())

    # system prompt：每次调用的 system 实参逐字节相同（plan/act/report 三个阶段）
    assert strong1.systems == strong2.systems
    assert all(s is not None for s in strong1.systems)
    # 工具 schema：每次 act 调用收到的列表逐项相等（注册顺序即 schema 顺序）
    assert strong1.tools_seen == strong2.tools_seen
    schemas = strong1.tools_seen[1]
    assert [t["function"]["name"] for t in schemas] == [
        "web_search", "fetch_url", "fs_read", "fs_write", "fs_list", "dispatch_research",
    ]
    # 两个实例的注册表顺序与 schema 序列化逐字节相同
    assert loop1.registry.names() == loop2.registry.names()
    assert json.dumps(loop1.tools_schema, ensure_ascii=False, sort_keys=True) == json.dumps(
        loop2.tools_schema, ensure_ascii=False, sort_keys=True
    )


def test_stable_prefix_plan_message_keeps_dynamic_blocks_in_suffix(tmp_path: Path) -> None:
    """前缀/后缀分区：T15 护栏块、T18 冲突块、Prior 块都只出现在 user 消息（后缀区），
    system prompt 不含任何动态注入内容。"""
    loop, strong = _stability_loop(tmp_path, "run-c")
    list(loop.events())
    plan_system = strong.systems[0]
    plan_message = strong.calls[0][0].content
    # system（前缀区）是纯静态文案
    assert plan_system.startswith("你是「自进化研究 Wiki」的研究 agent")
    assert QUESTION not in plan_system
    # 问题在 user 消息（动态后缀区）
    assert QUESTION in plan_message


# ---- ⑥ summarizer 异常：保守降级 ------------------------------------------------


def test_maybe_compact_summarizer_failure_degrades_conservatively(tmp_path: Path) -> None:
    """summarizer 抛异常 → 不压缩（原文零改动）+ 结果 status="failed"。"""
    messages = big_history(turns=4, chars=400)
    snapshot = list(messages)

    def boom(old: list[Message]) -> str:
        raise RuntimeError("摘要模型爆炸")

    result = maybe_compact(messages, compact_settings(), summarizer=boom)

    assert result.compacted is False
    assert result.status == "failed"
    assert "RuntimeError" in result.error and "摘要模型爆炸" in result.error
    assert messages == snapshot  # 保守降级：历史一个字都不丢
    assert messages[0] is snapshot[0]


def test_compaction_event_records_failed_status(tmp_path: Path) -> None:
    """failed 结果照记事件：status="failed"、summary_chars=0、post=pre（未压缩）。"""
    run_dir = tmp_path / "run"
    failed = CompactionResult(
        compacted=False,
        pre_tokens=1200,
        post_tokens=1200,
        kept_turns=0,
        summary="",
        status="failed",
        error="RuntimeError: 摘要模型爆炸",
    )
    event = record_compaction_event(run_dir, index=1, result=failed)
    assert event["status"] == "failed"
    assert event["summary_chars"] == 0
    assert event["pre_tokens"] == event["post_tokens"] == 1200
    (loaded,) = read_compaction_events(run_dir)
    assert loaded["status"] == "failed" and loaded["error"] == "RuntimeError: 摘要模型爆炸"


def test_agent_loop_summarizer_failure_keeps_run_alive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """loop 通路：summarizer 中途爆炸 → 研究循环照常走完（保守降级），事件记 failed。"""
    context = {
        "enabled": True,
        "max_context_tokens": 500,
        "compaction_threshold": 0.1,
        "keep_recent_turns": 2,
    }
    loop, strong, _ = _make_compaction_loop(tmp_path, "boom", context_config=context)

    def boom(self, messages):  # noqa: ARG001 -- 替换绑定方法
        raise RuntimeError("摘要模型爆炸")

    monkeypatch.setattr(AgentLoop, "_compact_summarizer", boom)
    events = list(loop.events())
    assert events[-1]["type"] == "finish"  # run 不被压缩故障打断
    # act:3 仍是完整历史（未压缩），报告照常产出
    assert strong.calls[3][0].content == QUESTION
    assert "".join(e.get("delta", "") for e in events if e["type"] == "text-delta") == REPORT_TEXT
    (event,) = read_compaction_events(tmp_path / "boom" / "run")
    assert event["status"] == "failed"
