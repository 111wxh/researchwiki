"""闸脚本（scripts/gate_*.py）内纯函数的单测：判定逻辑与指标计算。

只测不依赖网络与 key 的纯逻辑——闸脚本本身用 --provider mock / --embedding mock
做端到端自证，这里只钉住"判得对不对、算得对不对"。脚本不是包，用 importlib 按路径加载。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str) -> ModuleType:
    """按路径加载闸脚本；先注册进 sys.modules，否则 dataclass 解析注解会失败。"""
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


gate1 = _load("gate_cheap_toolcalls", "scripts/gate_cheap_toolcalls.py")
gate2 = _load("gate_retrieval", "scripts/gate_retrieval.py")


# ---- 闸 1：工具调用解析与参数校验 -------------------------------------------


def _tool(name: str = "web_search", required: tuple[str, ...] = ("query",)) -> object:
    return gate1.Tool(
        name=name,
        description="测试用",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}, "max_results": {"type": "integer"}},
            "required": list(required),
        },
        handler=lambda args: "{}",
    )


def _round(*, expected: tuple[str, ...] = ("web_search",), follow_up: bool = False) -> object:
    return gate1.Round(
        idx=1, category="测试", prompt="问题", expected=expected, follow_up=follow_up
    )


def _turn(
    *,
    calls: list[tuple[str, str]] | None = None,
    error: str = "",
) -> object:
    """造一次 stream 的观测；calls 走真实的 parse_tool_calls（坏 JSON 自然落成 args=None）。"""
    payload = [
        {"id": f"call-{i}", "type": "function", "function": {"name": name, "arguments": raw}}
        for i, (name, raw) in enumerate(calls or [])
    ]
    return gate1.TurnOutcome(calls=gate1.parse_tool_calls(payload), error=error)


def test_parse_tool_calls_keeps_broken_json_as_unparsed():
    parsed = gate1.parse_tool_calls(
        [
            {"function": {"name": "web_search", "arguments": '{"query": "上下文压缩"}'}},
            {"function": {"name": "fs_read", "arguments": "{坏 JSON"}},
            {"function": {"name": "fs_list", "arguments": '["not", "an", "object"]'}},
        ]
    )
    assert [call.name for call in parsed] == ["web_search", "fs_read", "fs_list"]
    assert parsed[0].args == {"query": "上下文压缩"}
    assert parsed[1].args is None and "JSON 破损" in parsed[1].parse_error
    assert parsed[2].args is None and "不是 JSON object" in parsed[2].parse_error


def test_validate_arguments_requires_fields_and_checks_types():
    tool = _tool()
    assert gate1.validate_arguments({"query": "x"}, tool)[0] is True
    assert gate1.validate_arguments({}, tool)[0] is False
    assert gate1.validate_arguments({"query": ""}, tool)[0] is False
    assert gate1.validate_arguments({"query": "x", "max_results": "5"}, tool) == (
        False,
        "字段 max_results 类型应为 integer，实得 str",
    )
    # 未声明的额外字段不判错
    assert gate1.validate_arguments({"query": "x", "extra": [1]}, tool)[0] is True


@pytest.mark.parametrize(
    ("turns", "expected_verdict", "expected_args_ok"),
    [
        ([_turn(calls=[("web_search", '{"query": "x"}')])], "ok", True),
        ([_turn()], "no_call", False),
        ([_turn(calls=[("nope", "{}")])], "name_hallucinated", False),
        ([_turn(calls=[("fs_read", '{"query": "x"}')])], "wrong_tool", False),
        ([_turn(calls=[("web_search", "{坏}")])], "bad_arguments", False),
        ([_turn(error="ProviderError: HTTP 401")], "provider_error", False),
    ],
)
def test_classify_round_expected_call_cases(turns, expected_verdict, expected_args_ok):
    known = {"web_search": _tool(), "fs_read": _tool("fs_read", required=("path",))}
    verdict, args_ok, _detail = gate1.classify_round(_round(), turns, known=known)
    assert verdict == expected_verdict
    assert args_ok is expected_args_ok


def test_classify_round_flags_unwanted_call_and_repeat():
    known = {"web_search": _tool()}
    no_call_round = _round(expected=())
    verdict, args_ok, _ = gate1.classify_round(
        no_call_round, [_turn(calls=[("web_search", '{"query": "x"}')])], known=known
    )
    assert verdict == "unwanted_call" and args_ok is True
    assert gate1.classify_round(no_call_round, [_turn()], known=known)[0] == "ok"

    follow_up = _round(follow_up=True)
    first = _turn(calls=[("web_search", '{"query": "x"}')])
    repeat = _turn(calls=[("web_search", '{"query": "x"}')])
    other = _turn(calls=[("web_search", '{"query": "y"}')])
    assert gate1.classify_round(follow_up, [first, repeat], known=known)[0] == "repeated_call"
    assert gate1.classify_round(follow_up, [first, other], known=known)[0] == "ok"
    assert gate1.classify_round(follow_up, [first, _turn()], known=known)[0] == "ok"


def test_percentile_linear_interpolation():
    assert gate1.percentile([], 50) == 0.0
    assert gate1.percentile([10.0], 95) == 10.0
    assert gate1.percentile([10.0, 20.0], 50) == 15.0
    assert gate1.percentile([10.0, 20.0, 30.0, 40.0], 95) == pytest.approx(38.5)


def test_compute_summary_rates_and_pass_line():
    ok_round = _round(expected=("web_search",))
    silent_round = _round(expected=("web_search",))
    no_tool_round = _round(expected=())
    results = [
        (ok_round, "ok", True, 100.0, 10, 5, 0),
        (silent_round, "no_call", False, 300.0, 20, 8, 0),
        (no_tool_round, "ok", True, 200.0, 30, 9, 0),
    ]
    summary = gate1.compute_summary(results, price_in=1.0, price_out=2.0)
    assert summary.rounds == 3 and summary.expect_calls == 2
    assert summary.name_rate == 0.5 and summary.args_rate == 0.5
    assert summary.ok_rate == pytest.approx(2 / 3)
    assert summary.latency_p50 == 200.0 and summary.latency_p95 == pytest.approx(290.0)
    assert summary.input_tokens == 60 and summary.output_tokens == 22
    assert summary.cost == pytest.approx(60 / 1_000_000 * 1.0 + 22 / 1_000_000 * 2.0)
    assert summary.passed(90) is False
    assert summary.passed(50) is True


def test_select_rounds_keeps_every_category():
    rounds = gate1.build_rounds()
    assert len(rounds) == 20
    picked = gate1.select_rounds(rounds, 5)
    assert len(picked) == 5
    assert {rd.category for rd in picked} == {rd.category for rd in rounds}
    assert [rd.idx for rd in picked] == sorted(rd.idx for rd in picked)
    assert gate1.select_rounds(rounds, 99) == rounds


def test_build_rounds_expected_tools_are_registered_names():
    names = {"web_search", "fetch_url", "fs_read", "fs_write", "fs_list", "dispatch_research"}
    for rd in gate1.build_rounds():
        assert set(rd.expected) <= names
        if rd.expected:
            assert rd.sample or rd.expected[0] == "web_search"


# ---- 闸 2：检索指标与夹具一致性 ---------------------------------------------


def test_recall_and_reciprocal_rank():
    truth = ("a", "z")
    assert gate2.recall_at_k(["a", "b"], truth, 5) == 0.5
    assert gate2.recall_at_k(["a", "z"], truth, 2) == 1.0
    assert gate2.recall_at_k([], truth, 5) == 0.0
    assert gate2.recall_at_k(["x"], (), 5) == 1.0
    assert gate2.reciprocal_rank(["x", "y", "a"], ("a",), 5) == pytest.approx(1 / 3)
    assert gate2.reciprocal_rank(["x", "y", "a"], ("a",), 2) == 0.0
    assert gate2.mean([]) == 0.0 and gate2.mean([1.0, 0.0]) == 0.5


def test_channel_metrics_counts_misses_and_forbidden():
    def row(qid: str, ranked: dict[str, list[str]]) -> object:
        return gate2.EvalRow(
            qid=qid,
            query="q",
            kind="术语",
            truth=("N-0001",),
            forbid=("N-0002",),
            ranked=ranked,
        )

    hit = row("Q1", {"fts": ["N-0001", "N-0002"], "vector": ["N-0001"], "hybrid": ["N-0001"]})
    miss = row("Q2", {"fts": ["N-0009"], "vector": ["N-0001"], "hybrid": ["N-0001"]})
    fts = gate2.channel_metrics("fts", [hit, miss])
    assert fts.scored_queries == 2
    assert fts.recall5 == 0.5 and fts.mrr == 0.5
    assert fts.forbidden_hits == 1 and fts.forbidden_checks == 2
    assert len(fts.failures) == 2  # 一条误召 + 一条漏召
    vector = gate2.channel_metrics("vector", [hit, miss])
    assert vector.recall5 == 1.0 and vector.forbidden_hits == 0 and vector.failures == []
    hybrid = gate2.channel_metrics("hybrid", [hit, miss])
    assert gate2.build_conclusion([fts, vector, hybrid], 0.8).startswith("项目默认 hybrid")


def test_fixture_ground_truth_ids_all_exist():
    note_ids = {note.note_id for note in gate2.FIXTURE_NOTES}
    assert len(note_ids) == 24
    assert len({query.qid for query in gate2.QUERIES}) == 12
    for query in gate2.QUERIES:
        assert query.truth, f"{query.qid} 缺少 ground truth"
        assert set(query.truth) <= note_ids, f"{query.qid} 的期望笔记不在夹具里"
        assert set(query.forbid) <= note_ids, f"{query.qid} 的禁用笔记不在夹具里"
        assert not set(query.truth) & set(query.forbid), f"{query.qid} 的期望与禁用重叠"
