"""scripts/cold_warm_smoke.py 的测试：mock 模式端到端 + 硬断言/汇总表纯函数。

脚本不是包，用 importlib 按路径加载（与 tests/test_gate_scripts.py 同款）。
端到端跑真实 AgentLoop（ScriptedProvider + MockSearch + MockEmbeddingProvider），
零网络、零 key；config 模式只测"真模型不可用时明确报失败"的路径
（config 读不到 → MockProvider 占位回复 → 蒸馏无产出 → warm prior 命中 0
→ 退出码 1，绝不静默通过）。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from researchwiki.loop.metrics import RunMetrics, sum_tokens_from_jsonl
from researchwiki.tools import MockSearch

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str) -> ModuleType:
    """按路径加载脚本；先注册进 sys.modules，否则 dataclass 解析注解会失败。"""
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


smoke = _load("cold_warm_smoke", "scripts/cold_warm_smoke.py")


@pytest.fixture(autouse=True)
def _no_search_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """零网络保证：mock 模式已显式锁定 MockSearch；这里再清掉搜索相关环境变量，
    兜住 config 模式测试（空 config 时 get_search_provider 会回退读环境变量）。"""
    for var in ("SEARCH_PROVIDER", "TAVILY_API_KEY", "BOCHA_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def _run(tmp_path: Path, argv: list[str]) -> tuple[int, list[dict], Path]:
    """跑一次脚本 main，读回（退出码, JSONL 行, 输出路径）。"""
    out = tmp_path / "cold_warm.jsonl"
    rc = smoke.main([*argv, "--out", str(out)])
    lines = out.read_text(encoding="utf-8").splitlines()
    return rc, [json.loads(line) for line in lines], out


# ---- mock 模式端到端（真实 AgentLoop 两次 run）--------------------------------


def test_mock_end_to_end_prior_hit_and_token_reconciliation(tmp_path, capsys) -> None:
    wiki_root = tmp_path / "wiki"
    rc, rows, out = _run(tmp_path, ["--provider", "mock", "--wiki-root", str(wiki_root)])

    assert rc == smoke.EXIT_PASS

    # JSONL 恰好两行：cold + warm，字段与 run-metrics 14 字段契约齐全
    assert [row["run"] for row in rows] == ["cold", "warm"]
    expected_keys = {
        "run",
        "question",
        "trace_id",
        "wiki_root",
        "run_dir",
        "provider_mode",
        "model",
        "metrics",
    }
    assert all(set(row) == expected_keys for row in rows)
    assert all(set(row["metrics"]) == set(RunMetrics().to_dict()) for row in rows)
    assert all(row["provider_mode"] == "mock" for row in rows)
    assert all(row["question"] == smoke.DEFAULT_QUESTION for row in rows)
    assert all(Path(row["wiki_root"]) == wiki_root for row in rows)
    assert rows[0]["model"] == {
        "strong": {"model": "mock-strong", "base_url": ""},
        "cheap": {"model": "mock-cheap", "base_url": ""},
    }

    # 两次 run 是独立 AgentLoop 实例：独立 trace_id，各自 run-metrics.json 落盘
    assert rows[0]["trace_id"] != rows[1]["trace_id"]
    assert rows[0]["run_dir"] != rows[1]["run_dir"]
    for row in rows:
        assert (Path(row["run_dir"]) / "run-metrics.json").is_file()

    # 阶段验收口径：cold 空 Wiki 不命中；warm 必须命中 cold 沉淀的 active note
    assert rows[0]["metrics"]["prior_hit_count"] == 0
    assert rows[0]["metrics"]["prior_note_ids"] == []
    assert rows[1]["metrics"]["prior_hit_count"] > 0
    assert rows[1]["metrics"]["prior_note_ids"], "warm 命中必须带 note id"
    # cold 沉淀 ≥1 条 active note（warm 能命中的前提），且运行时硬断言全绿
    assert rows[0]["metrics"]["notes_created"] >= 1
    assert smoke.verify_rows(rows) == []

    # 同一份剧本的两次 run 行为一致（完全脚本化、可重复）
    assert rows[0]["metrics"]["fresh_search_count"] == rows[1]["metrics"]["fresh_search_count"]
    assert rows[0]["metrics"]["fresh_search_count"] == 1

    # token 可复算：两次 run 的 metrics 与 tokens.jsonl 按 trace_id 复算完全一致
    for row in rows:
        tok_in, tok_out = sum_tokens_from_jsonl(wiki_root / "tokens.jsonl", row["trace_id"])
        assert tok_in > 0, "对账不应空转"
        assert row["metrics"]["input_tokens"] == tok_in
        assert row["metrics"]["output_tokens"] == tok_out

    # 汇总表可打印：对照字段、差值列、防误读声明
    printed = capsys.readouterr().out
    for field in (
        "fresh_search_count",
        "fresh_fetch_count",
        "source_count",
        "notes_created",
        "input_tokens",
        "output_tokens",
        "latency_ms",
        "citation_coverage",
    ):
        assert field in printed
    assert "差值" in printed
    assert "小样本冒烟" in printed
    assert f"JSONL 已写入 {out}" in printed


def test_mock_mode_supports_custom_question(tmp_path) -> None:
    """自定义问题下剧本同样自洽：笔记正文嵌入问题原文，warm 依然命中。"""
    rc, rows, _out = _run(
        tmp_path,
        ["--provider", "mock", "--question", "上下文压缩的工程实践有哪些"],
    )
    assert rc == smoke.EXIT_PASS
    assert rows[0]["metrics"]["prior_hit_count"] == 0
    assert rows[1]["metrics"]["prior_hit_count"] > 0


def test_mock_mode_locks_mock_search_even_with_env_keys(monkeypatch, tmp_path) -> None:
    """环境里恰好有搜索 key 时，mock 模式仍显式锁定 MockSearch——零网络不靠环境巧合。"""
    monkeypatch.setenv("TAVILY_API_KEY", "smoke-test-not-a-real-key")
    stack = smoke.build_mock_stack(smoke.DEFAULT_QUESTION)
    assert isinstance(stack.search_provider, MockSearch)
    # 端到端仍然通过（不会向真实搜索发起请求）
    rc, _rows, _out = _run(
        tmp_path, ["--provider", "mock", "--wiki-root", str(tmp_path / "wiki")]
    )
    assert rc == smoke.EXIT_PASS


# ---- config 模式：真模型不可用时必须明确报失败 ---------------------------------


def test_config_mode_fails_loudly_without_usable_models(tmp_path, capsys) -> None:
    """config 读不到 → MockProvider 占位回复 → 蒸馏无产出 → warm prior 命中 0 → 退出码 1。

    同时验证：config 模式开始前打印成本警告、运行本身零网络（MockSearch 兜底）。
    """
    rc, rows, _out = _run(
        tmp_path,
        [
            "--provider",
            "config",
            "--config",
            str(tmp_path / "missing.toml"),
            "--env-file",
            "",  # 不读项目 .env（测试零 key 依赖）
            "--wiki-root",
            str(tmp_path / "wiki"),
        ],
    )
    assert rc == smoke.EXIT_ASSERT_FAILED
    assert [row["run"] for row in rows] == ["cold", "warm"]
    assert rows[1]["metrics"]["prior_hit_count"] == 0
    printed = capsys.readouterr().out
    assert "成本警告" in printed
    assert "断言失败" in printed and "prior_hit_count" in printed


# ---- 硬断言与汇总表：纯函数 ---------------------------------------------------


def test_verify_rows_flags_zero_prior_and_token_mismatch(tmp_path) -> None:
    (tmp_path / "tokens.jsonl").write_text(
        json.dumps({"trace_id": "t1", "input_tokens": 100, "output_tokens": 10}) + "\n",
        encoding="utf-8",
    )
    rows = [
        {"run": "cold", "trace_id": "t1", "wiki_root": str(tmp_path),
         "metrics": {"input_tokens": 100, "output_tokens": 10, "prior_hit_count": 0,
                     "notes_created": 1}},
        {"run": "warm", "trace_id": "t2", "wiki_root": str(tmp_path),
         "metrics": {"input_tokens": 5, "output_tokens": 5, "prior_hit_count": 0,
                     "notes_created": 0}},
    ]
    failures = smoke.verify_rows(rows)
    assert any("prior_hit_count" in failure for failure in failures)
    # 只有对不上的 warm 行被点名；cold 行与 tokens.jsonl 一致，不进失败列表
    mismatches = [failure for failure in failures if "对账失败" in failure]
    assert len(mismatches) == 1 and "warm" in mismatches[0]

    # warm 命中后仅剩的问题是对账失败
    rows[1]["metrics"]["prior_hit_count"] = 2
    assert all("prior_hit_count" not in failure for failure in smoke.verify_rows(rows))


def test_verify_rows_flags_dirty_cold_start(tmp_path) -> None:
    """cold 闸：--wiki-root 已含相关笔记（cold 直接命中）或蒸馏零产出时必须拦住假绿对照。"""
    rows = [
        {"run": "cold", "trace_id": "t1", "wiki_root": str(tmp_path),
         "metrics": {"input_tokens": 0, "output_tokens": 0, "prior_hit_count": 1,
                     "notes_created": 0}},
        {"run": "warm", "trace_id": "t2", "wiki_root": str(tmp_path),
         "metrics": {"input_tokens": 0, "output_tokens": 0, "prior_hit_count": 1,
                     "notes_created": 0}},
    ]
    failures = smoke.verify_rows(rows)
    assert any(
        "cold run prior_hit_count=1" in failure and "假绿" in failure
        for failure in failures
    )
    assert any("cold run notes_created=0" in failure for failure in failures)

    # 修正 cold 口径后这两条失败消失（warm 命中 1 保持合法）
    rows[0]["metrics"]["prior_hit_count"] = 0
    rows[0]["metrics"]["notes_created"] = 1
    remaining = smoke.verify_rows(rows)
    assert not any("cold run" in failure for failure in remaining)


def test_verify_rows_flags_missing_warm_row() -> None:
    failures = smoke.verify_rows([{"run": "cold", "trace_id": "t", "wiki_root": "x",
                                   "metrics": {}}])
    assert any("缺少 warm run" in failure for failure in failures)


def test_format_summary_shows_compare_fields_delta_and_disclaimer() -> None:
    rows = [
        {"run": "cold", "metrics": {"prior_hit_count": 0, "fresh_search_count": 1,
                                    "fresh_fetch_count": 0, "source_count": 3,
                                    "notes_created": 1, "input_tokens": 100,
                                    "output_tokens": 20, "latency_ms": 5,
                                    "citation_coverage": 0.3333}},
        {"run": "warm", "metrics": {"prior_hit_count": 1, "fresh_search_count": 1,
                                    "fresh_fetch_count": 0, "source_count": 3,
                                    "notes_created": 0, "input_tokens": 90,
                                    "output_tokens": 20, "latency_ms": 4,
                                    "citation_coverage": None}},
    ]
    text = smoke.format_summary(rows)
    assert "cold" in text and "warm" in text and "差值" in text
    assert "小样本冒烟，不构成收益结论" in text
    # citation_coverage 一侧为 None 时差值显示占位符，不硬算
    assert "—" in text
    for field in smoke.COMPARE_FIELDS:
        assert field in text
