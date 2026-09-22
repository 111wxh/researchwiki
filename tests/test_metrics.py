"""run-metrics 指标模块单测：契约键集合、落盘 round-trip、tokens 对账、引用覆盖率。

全部零网络、零 Wiki 依赖；tokens.jsonl 用临时文件手工构造。
"""

import json
from pathlib import Path

from researchwiki.loop.metrics import (
    RunMetrics,
    compute_citation_coverage,
    sum_tokens_from_jsonl,
    write_run_metrics,
)

# PLAN §4.4 最小指标集：逐字契约，防漂移
EXPECTED_KEYS = {
    "trace_id",
    "prior_hit_count",
    "prior_note_ids",
    "prior_context_chars",
    "fresh_search_count",
    "fresh_fetch_count",
    "source_count",
    "notes_created",
    "notes_merged",
    "notes_superseded",
    "input_tokens",
    "output_tokens",
    "latency_ms",
    "citation_coverage",
}


# ---- RunMetrics 结构契约 -------------------------------------------------------


class TestRunMetricsContract:
    def test_to_dict_keys_match_plan_verbatim(self) -> None:
        # 集合相等断言：多键、少键、改名都会失败
        assert set(RunMetrics().to_dict()) == EXPECTED_KEYS

    def test_defaults_match_plan_json(self) -> None:
        d = RunMetrics().to_dict()
        # 计数全 0
        for key in (
            "prior_hit_count",
            "prior_context_chars",
            "fresh_search_count",
            "fresh_fetch_count",
            "source_count",
            "notes_created",
            "notes_merged",
            "notes_superseded",
            "input_tokens",
            "output_tokens",
            "latency_ms",
        ):
            assert d[key] == 0, f"{key} 默认值应为 0"
        # 列表为空、None 字段
        assert d["prior_note_ids"] == []
        assert d["citation_coverage"] is None
        assert d["trace_id"] == ""

    def test_from_loop_populates_all_fields(self) -> None:
        m = RunMetrics.from_loop(
            trace_id="abc123def456",
            prior_hit_count=3,
            prior_note_ids=["N-0001", "N-0002"],
            prior_context_chars=1500,
            fresh_search_count=10,
            fresh_fetch_count=4,
            source_count=12,
            notes_created=5,
            notes_merged=1,
            notes_superseded=0,
            input_tokens=9000,
            output_tokens=2200,
            latency_ms=48000,
            citation_coverage=0.75,
        )
        d = m.to_dict()
        assert d["trace_id"] == "abc123def456"
        assert d["prior_hit_count"] == 3
        assert d["prior_note_ids"] == ["N-0001", "N-0002"]
        assert d["prior_context_chars"] == 1500
        assert d["fresh_search_count"] == 10
        assert d["fresh_fetch_count"] == 4
        assert d["source_count"] == 12
        assert d["notes_created"] == 5
        assert d["notes_merged"] == 1
        assert d["notes_superseded"] == 0
        assert d["input_tokens"] == 9000
        assert d["output_tokens"] == 2200
        assert d["latency_ms"] == 48000
        assert d["citation_coverage"] == 0.75

    def test_from_loop_none_list_becomes_empty_list(self) -> None:
        m = RunMetrics.from_loop(trace_id="t")
        assert m.prior_note_ids == []


# ---- 落盘 round-trip -----------------------------------------------------------


class TestWriteRunMetrics:
    def test_round_trip_equals_to_dict(self, tmp_path: Path) -> None:
        metrics = RunMetrics.from_loop(
            trace_id="trace000001",
            prior_hit_count=2,
            prior_note_ids=["N-0001"],
            fresh_search_count=6,
            source_count=8,
            notes_created=3,
            input_tokens=1000,
            output_tokens=200,
            latency_ms=12000,
            citation_coverage=0.5,
        )
        out = write_run_metrics(tmp_path / "runs" / "2026-09-22", metrics)
        assert out.name == "run-metrics.json"
        # 目录不存在则创建
        assert out.parent.is_dir()
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded == metrics.to_dict()

    def test_file_is_utf8_unescaped_indented(self, tmp_path: Path) -> None:
        metrics = RunMetrics(trace_id="中文trace")
        out = write_run_metrics(tmp_path, metrics)
        raw = out.read_text(encoding="utf-8")
        assert "中文trace" in raw  # ensure_ascii=False：中文不转义
        assert '\n  "trace_id"' in raw  # 缩进 2

    def test_overwrite_existing_file(self, tmp_path: Path) -> None:
        write_run_metrics(tmp_path, RunMetrics(trace_id="first"))
        out = write_run_metrics(tmp_path, RunMetrics(trace_id="second"))
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded["trace_id"] == "second"


# ---- tokens.jsonl 对账 -----------------------------------------------------------


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    return path


class TestSumTokensFromJsonl:
    def test_sum_filters_by_trace_id(self, tmp_path: Path) -> None:
        path = _write_jsonl(
            tmp_path / "tokens.jsonl",
            [
                {"trace_id": "aaa", "input_tokens": 100, "output_tokens": 10},
                {"trace_id": "aaa", "input_tokens": 200, "output_tokens": 20},
                {"trace_id": "bbb", "input_tokens": 999, "output_tokens": 999},
                {"trace_id": "aaa", "input_tokens": 50, "output_tokens": 5},
            ],
        )
        assert sum_tokens_from_jsonl(path, "aaa") == (350, 35)

    def test_missing_file_returns_zero(self, tmp_path: Path) -> None:
        assert sum_tokens_from_jsonl(tmp_path / "nope.jsonl", "aaa") == (0, 0)

    def test_corrupted_and_missing_field_lines_skipped(self, tmp_path: Path) -> None:
        content = (
            json.dumps({"trace_id": "aaa", "input_tokens": 100, "output_tokens": 10}) + "\n"
            "{not valid json\n"
            + json.dumps({"input_tokens": 100, "output_tokens": 10}) + "\n"  # 缺 trace_id
            + json.dumps({"trace_id": "aaa", "input_tokens": 100}) + "\n"  # 缺 output_tokens
            + json.dumps({"trace_id": "aaa", "input_tokens": "x", "output_tokens": 1}) + "\n"
            + json.dumps({"trace_id": "aaa", "input_tokens": 30, "output_tokens": 3}) + "\n"
            "\n"  # 空行
        )
        path = tmp_path / "tokens.jsonl"
        path.write_text(content, encoding="utf-8")
        # 只有第 1 行和最后一条完整记录计入
        assert sum_tokens_from_jsonl(path, "aaa") == (130, 13)

    def test_error_rows_are_counted(self, tmp_path: Path) -> None:
        # error_type 非空（调用失败）也消耗 token，必须计入
        path = _write_jsonl(
            tmp_path / "tokens.jsonl",
            [
                {
                    "trace_id": "aaa",
                    "input_tokens": 500,
                    "output_tokens": 0,
                    "error_type": "rate_limit",
                },
                {"trace_id": "aaa", "input_tokens": 100, "output_tokens": 20},
            ],
        )
        assert sum_tokens_from_jsonl(path, "aaa") == (600, 20)

    def test_empty_file_returns_zero(self, tmp_path: Path) -> None:
        path = tmp_path / "tokens.jsonl"
        path.write_text("", encoding="utf-8")
        assert sum_tokens_from_jsonl(path, "aaa") == (0, 0)

    def test_real_accountant_row_shape_is_reconcilable(self, tmp_path: Path) -> None:
        # 与 TokenAccountant.record 写出的行结构对账（字段名来自 asdict(TokenUsage)）
        from researchwiki.llm.accounting import TokenAccountant
        from researchwiki.llm.provider import TokenUsage

        accountant = TokenAccountant(tmp_path / "tokens.jsonl")
        accountant.record(
            trace_id="abc123def456",
            step="plan",
            model="mock-strong",
            usage=TokenUsage(input_tokens=900, output_tokens=120),
            latency_ms=1.5,
        )
        accountant.record(
            trace_id="abc123def456",
            step="report",
            model="mock-strong",
            usage=TokenUsage(input_tokens=300, output_tokens=80, cache_read_tokens=400),
            latency_ms=2.0,
            error_type=None,
        )
        accountant.record(
            trace_id="other-trace",
            step="plan",
            model="mock-strong",
            usage=TokenUsage(input_tokens=777, output_tokens=77),
            latency_ms=1.0,
        )
        assert sum_tokens_from_jsonl(accountant.path, "abc123def456") == (1200, 200)


# ---- 引用覆盖率 -----------------------------------------------------------------


class TestComputeCitationCoverage:
    def test_partial_coverage(self) -> None:
        text = "结论 A[1]；结论 B[3]；结论 C 重复引用[1]。"
        assert compute_citation_coverage(text, source_count=4) == 2 / 4

    def test_full_coverage_with_duplicates(self) -> None:
        text = "[1] 与 [2] 与 [3]，再引用 [1] [2]。"
        assert compute_citation_coverage(text, source_count=3) == 1.0

    def test_zero_sources_returns_none(self) -> None:
        assert compute_citation_coverage("正文 [1] [2]", source_count=0) is None

    def test_no_markers_with_sources_returns_zero(self) -> None:
        # 口径：source_count>0 且正文无引用标记 → 0.0（有来源但零覆盖）
        assert compute_citation_coverage("正文没有任何引用标记", source_count=5) == 0.0

    def test_out_of_range_markers_not_counted(self) -> None:
        # 越界编号指向不存在的来源，不计入覆盖
        text = "引用 [1] [9]"
        assert compute_citation_coverage(text, source_count=4) == 1 / 4
