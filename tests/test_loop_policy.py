"""P3：AgentLoop 模式判定接线测试（零网络，ScriptedProvider + MockSearch）。

脚手架与 tests/test_loop.py / tests/test_loop_prior.py 同构（text_turn /
tool_turn / call 助手 + FakeRouter + tmp_path 沙箱）；wiki 预置笔记用
WikiStore.save_note + MockEmbeddingProvider(dim=512) + trigram tokenizer
（确定性、零网络、不依赖 vendor DLL）。模式探测复用 Prior 索引，events()
开头的 ensure_index_fresh 会真实 rebuild，预置笔记因此能被探测检索命中。

时效确定性：fresh 夹具不写 observed_at（回退 created=真实现在 → fresh）；
stale 夹具 observed_at 固定在 2026-01-01（相对现在已过多个半衰期 → stale），
与 tests/test_research_policy.py 的约定一致。
"""

import json
from pathlib import Path

from researchwiki.llm.accounting import TokenAccountant
from researchwiki.llm.provider import ScriptedProvider, StreamEvent, TokenUsage
from researchwiki.loop.agent_loop import AgentLoop
from researchwiki.tools import MockSearch
from researchwiki.wiki.embeddings import MockEmbeddingProvider
from researchwiki.wiki.store import WikiStore

QUESTION = "Zephyr 框架的内存占用是多少"
PLAN_TEXT = "- 任务一：核对既有记忆\n- 任务二：撰写结论\n"
REPORT_TEXT = "## 研究报告\n\nZephyr 内存占用约 2KB。\n"
DISTILL_JSON = json.dumps({"notes": [], "conflicts": []}, ensure_ascii=False)

# 简报样式约束（接线常量 _REPORT_STYLE_SUFFIX 的可辨识片段）
BRIEF_SUFFIX_FRAGMENT = "限 200 字以内"


# ---- 脚手架（照抄 tests/test_loop.py / tests/test_loop_prior.py 现有约定）----


def text_turn(text: str) -> list[StreamEvent]:
    return [StreamEvent(type="text_delta", delta=text)]


def tool_turn(calls: list[dict]) -> list[StreamEvent]:
    return [
        StreamEvent(type="tool_calls", tool_calls=calls),
        StreamEvent(type="usage", usage=TokenUsage(input_tokens=900, output_tokens=40)),
    ]


def call(name: str, arguments: dict, *, id: str = "c1") -> dict:
    return {
        "id": id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
    }


class FakeRouter:
    """按档位返回预置 Provider（未配置的档位被请求即失败，防止测试里静默串线）。"""

    def __init__(self, strong, cheap=None) -> None:
        self._providers = {"strong": strong}
        if cheap is not None:
            self._providers["cheap"] = cheap

    def get(self, tier):
        if tier not in self._providers:
            raise AssertionError(f"测试未配置 {tier} 档 Provider，却被请求了")
        return self._providers[tier]


class SystemCaptureProvider:
    """包装 ScriptedProvider，记录每次调用的 system 提示词（断言报告样式接线用）。

    与 tests/test_loop.py 的 ToolUntilForcedProvider 同一先例：按 Provider 协议
    的最小测试替身，只转发不篡改。
    """

    model = "mock-strong"
    tier = "strong"

    def __init__(self, inner: ScriptedProvider) -> None:
        self._inner = inner
        self.systems: list[str | None] = []

    @property
    def calls(self) -> list:
        return self._inner.calls

    def stream(self, messages, *, system=None, tools=None):
        self.systems.append(system)
        yield from self._inner.stream(messages, system=system, tools=tools)


def seed_fresh_stable(store: WikiStore) -> None:
    """两条 fresh/stable/high 笔记，正文与问题强重叠（trigram 探测必命中）。"""
    store.save_note(
        "Zephyr 框架的内存占用约为 2KB，官方文档发布于 2026-09。",
        title="内存占用（历史）",
        entities=["Zephyr"],
        confidence="high",
        volatility="stable",
    )
    store.save_note(
        "Zephyr 框架的内存占用很低，社区评测确认了这一读数。",
        title="内存占用（评测）",
        entities=["Zephyr"],
        confidence="high",
        volatility="stable",
    )


def make_loop(
    tmp_path: Path,
    *,
    strong_turns: list[list[StreamEvent]],
    question: str = QUESTION,
    retrieval_config: dict | None = None,
    **kwargs,
) -> AgentLoop:
    strong = ScriptedProvider(strong_turns, model="mock-strong")
    # cheap 未配置 base_url → 蒸馏/子 agent 复用 strong（与 tests/test_loop.py 同口径）
    llm_config = {"strong": {"base_url": "https://mock"}, "cheap": {"base_url": ""}}
    return AgentLoop(
        question,
        router=FakeRouter(strong),
        llm_config=llm_config,
        accountant=TokenAccountant(tmp_path / "tokens.jsonl"),
        search_provider=MockSearch(),
        wiki_root=tmp_path / "wiki-data",
        run_dir=tmp_path / "run",
        # 检索确定性：dim=512 夹具间可区分；trigram 不探测 vendor DLL（同 test_loop_prior）
        embedding=MockEmbeddingProvider(dim=512),
        wiki_config={"fts_tokenizer": "trigram"},
        retrieval_config=retrieval_config,
        **kwargs,
    )


def default_turns() -> list[list[StreamEvent]]:
    """plan 文本 → act 文本收尾 → distill → report（无工具调用、无 fresh 搜索）。"""
    return [
        text_turn(PLAN_TEXT),
        text_turn("记忆已核对，研究完成。"),
        text_turn(DISTILL_JSON),
        text_turn(REPORT_TEXT),
    ]


def load_policy(tmp_path: Path) -> dict:
    path = tmp_path / "run" / "policy.json"
    assert path.is_file(), "模式判定启用时 policy.json 必须存在"
    return json.loads(path.read_text(encoding="utf-8"))


def load_metrics(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "run" / "run-metrics.json").read_text(encoding="utf-8"))


def load_state(tmp_path: Path) -> str:
    return (tmp_path / "run" / "state.md").read_text(encoding="utf-8")


# ---- 1. retrieval_config=None：模式概念不存在，行为与 P2 一致 ------------------


def test_retrieval_disabled_keeps_legacy_behavior(tmp_path: Path) -> None:
    loop = make_loop(tmp_path, strong_turns=default_turns(), retrieval_config=None)
    events = list(loop.events())

    assert events[0]["type"] == "start" and events[-1]["type"] == "finish"
    # 无 policy.json、state.md 无 retrieval 行、判定结论为 None
    assert not (tmp_path / "run" / "policy.json").exists()
    assert "retrieval:" not in load_state(tmp_path)
    assert loop.policy_decision is None
    # 工具面完整（dispatch_research 仍注册）、搜索帽不生效
    assert "dispatch_research" in loop.registry.names()
    assert "dispatch_research" in [s["function"]["name"] for s in loop.tools_schema]
    # 记账序列与 P2 逐位一致：plan → act:1 → distill → report
    lines = (tmp_path / "tokens.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert [json.loads(line)["step"] for line in lines] == [
        "plan",
        "act:1",
        "distill",
        "report",
    ]
    # report system 提示词不含简报约束（standard 现状不变）
    report = (tmp_path / "run" / "report.md").read_text(encoding="utf-8")
    assert REPORT_TEXT.strip() in report
    assert BRIEF_SUFFIX_FRAGMENT not in report


def test_retrieval_enabled_false_treated_as_off(tmp_path: Path) -> None:
    """段内 enabled=false 是逃生阀（eval baseline）：与 None 同口径零痕迹，
    且搜索帽 / 子代理门都不生效（P2 行为完整保留）。"""
    loop = make_loop(
        tmp_path,
        strong_turns=[
            text_turn(PLAN_TEXT),
            tool_turn([call("web_search", {"query": "Zephyr 内存"}, id="c1")]),
            text_turn("研究完成。"),
            text_turn(DISTILL_JSON),
            text_turn(REPORT_TEXT),
        ],
        retrieval_config={"enabled": False},
    )
    events = list(loop.events())

    assert events[-1]["type"] == "finish"
    assert not (tmp_path / "run" / "policy.json").exists()
    assert "retrieval:" not in load_state(tmp_path)
    assert loop.policy_decision is None
    # 搜索真实发生（帽不生效）、dispatch_research 仍注册
    assert loop.ctx.search_calls == 1
    assert "dispatch_research" in loop.registry.names()


# ---- 2. 判定留痕：features + reasons 落盘 policy.json，state.md 记一行 --------


def test_policy_decision_logged_with_features_and_reasons(tmp_path: Path) -> None:
    store = WikiStore(tmp_path / "wiki-data")
    seed_fresh_stable(store)
    strong = ScriptedProvider(default_turns(), model="mock-strong")
    capture = SystemCaptureProvider(strong)
    llm_config = {"strong": {"base_url": "https://mock"}, "cheap": {"base_url": ""}}
    loop = AgentLoop(
        QUESTION,
        router=FakeRouter(capture),
        llm_config=llm_config,
        accountant=TokenAccountant(tmp_path / "tokens.jsonl"),
        search_provider=MockSearch(),
        wiki_root=tmp_path / "wiki-data",
        run_dir=tmp_path / "run",
        embedding=MockEmbeddingProvider(dim=512),
        wiki_config={"fts_tokenizer": "trigram"},
        retrieval_config={"enabled": True},
    )
    events = list(loop.events())

    assert events[-1]["type"] == "finish"
    # 自动判定：fresh/stable 覆盖充分 → simple（非 forced）
    assert loop.policy_decision is not None
    assert loop.policy_decision.mode == "simple"
    assert loop.policy_decision.forced is False
    # policy.json：trace_id / decided_at / features / reasons / limits 全量留痕
    policy = load_policy(tmp_path)
    assert policy["trace_id"] == loop.trace_id
    assert policy["decided_at"]
    assert policy["mode"] == "simple" and policy["forced"] is False
    assert policy["features"]["hit_count"] >= 1
    assert policy["features"]["stale_hits"] == 0
    assert policy["features"]["low_confidence_hits"] == 0
    assert policy["reasons"]
    assert policy["limits"]["max_fresh_searches"] == 0  # simple 不做 fresh 搜索
    assert policy["limits"]["subagents"] is False
    # state.md 记 retrieval 行
    assert "retrieval: mode=simple forced=false" in load_state(tmp_path)
    # subagents=False → 模型工具面不再提供 dispatch_research
    assert "dispatch_research" not in loop.registry.names()
    assert "dispatch_research" not in [s["function"]["name"] for s in loop.tools_schema]
    # 简报样式：仅报告阶段的 system 提示追加 ≤200 字约束（plan/act/distill 不变）
    assert len(capture.systems) == 4
    assert BRIEF_SUFFIX_FRAGMENT in (capture.systems[3] or "")
    assert all(BRIEF_SUFFIX_FRAGMENT not in (s or "") for s in capture.systems[:3])


# ---- 3. simple 模式搜索帽：达帽即拒，provider 不被调用、计数不增长 --------------


def test_simple_mode_blocks_fresh_search(tmp_path: Path) -> None:
    loop = make_loop(
        tmp_path,
        strong_turns=[
            text_turn(PLAN_TEXT),
            tool_turn([call("web_search", {"query": "Zephyr 内存"}, id="c1")]),
            text_turn("搜索被模式帽子拦截，基于既有记忆作答。"),
            text_turn(DISTILL_JSON),
            text_turn(REPORT_TEXT),
        ],
        retrieval_config={"enabled": True, "forced_mode": "simple"},
    )
    events = list(loop.events())

    assert events[-1]["type"] == "finish"
    assert loop.policy_decision.mode == "simple" and loop.policy_decision.forced is True
    # forced 仍照常留痕特征与理由（空 wiki 的规则判定 deep 被覆盖，理由可审计）
    policy = load_policy(tmp_path)
    assert policy["features"]["hit_count"] == 0
    assert any("forced_mode=simple" in r for r in policy["reasons"])
    # act:2 请求快照里才有 assistant(tool_calls) + role="tool" 结果（与 test_loop.py 同序）
    tool_msg = loop.provider.calls[2][3]
    assert tool_msg.role == "tool" and tool_msg.name == "web_search"
    payload = json.loads(tool_msg.content)
    assert payload["error"] == "search_budget_exhausted"
    assert payload["mode"] == "simple"
    assert payload["limit"] == 0
    # provider（MockSearch）从未被调用；fresh_search_count 计数不增长
    assert loop.ctx.search_provider.calls == []
    assert loop.ctx.search_calls == 0
    assert load_metrics(tmp_path)["fresh_search_count"] == 0


# ---- 4. 守卫红线：stale 记忆在判，自动判定不得为 simple ------------------------


def test_stale_memory_never_routed_simple(tmp_path: Path) -> None:
    store = WikiStore(tmp_path / "wiki-data")
    # 一条 stale（volatile + 2026-01-01，远超 30 天半衰期）+ 一条 fresh stable：
    # 覆盖度达标，但守卫信号在判 → 必须 update
    store.save_note(
        "Zephyr 框架的内存占用约为 2KB。",
        title="旧读数",
        entities=["Zephyr"],
        confidence="high",
        volatility="volatile",
        observed_at="2026-01-01T00:00:00+00:00",
    )
    store.save_note(
        "Zephyr 框架的内存占用很低，社区评测确认了这一读数。",
        title="新读数",
        entities=["Zephyr"],
        confidence="high",
        volatility="stable",
    )
    loop = make_loop(
        tmp_path,
        strong_turns=default_turns(),
        retrieval_config={"enabled": True},
    )
    events = list(loop.events())

    assert events[-1]["type"] == "finish"
    assert loop.policy_decision is not None
    assert loop.policy_decision.mode == "update"
    assert loop.policy_decision.forced is False
    policy = load_policy(tmp_path)
    assert policy["mode"] == "update"
    assert policy["features"]["stale_hits"] >= 1
    # update 的限额允许少量 fresh 核验搜索（与 simple 的 0 帽形成对照）
    assert policy["limits"]["max_fresh_searches"] == 2
    assert "retrieval: mode=update forced=false" in load_state(tmp_path)


# ---- 5. outcome 回填：报告写盘后核验 fresh 来源数，不足追加诚实边界块 ----------


def test_policy_json_outcome_backfilled_after_report(tmp_path: Path) -> None:
    # forced_mode=update、min_fresh_sources=1；MockSearch 有结果但脚本不调用搜索
    # → 来源池为空，fresh_source_count=0 < 1 → 诚实边界提示写入 report.md 末尾
    loop = make_loop(
        tmp_path,
        strong_turns=default_turns(),
        retrieval_config={
            "enabled": True,
            "forced_mode": "update",
            "update": {"min_fresh_sources": 1},
        },
    )
    events = list(loop.events())

    assert events[-1]["type"] == "finish"
    assert loop.policy_decision.mode == "update" and loop.policy_decision.forced is True
    policy = load_policy(tmp_path)
    assert policy["outcome"] == {
        "fresh_source_count": 0,
        "min_fresh_sources": 1,
        "min_fresh_sources_satisfied": False,
    }
    # 诚实边界块追加在报告正文之后（不污染正文事件流，report_text 不变）
    report = (tmp_path / "run" / "report.md").read_text(encoding="utf-8")
    assert report.startswith(REPORT_TEXT)
    assert "新鲜来源不足" in report
    assert "诚实边界" in report and "非引用缺失错误" in report
    # 正文 text-delta 事件仍是纯报告（提示块只落盘、不进流）
    assert "".join(e["delta"] for e in events if e["type"] == "text-delta") == REPORT_TEXT
