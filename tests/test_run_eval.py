"""scripts/run_eval.py 的测试：mock 模式两题×四条件端到端 + provenance 留痕。

脚本不是包，用 importlib 按路径加载（与 tests/test_adaptive_smoke.py 同款）。
端到端跑真实 AgentLoop 与 run_rag（ScriptedProvider + FixtureSearch 受控语料 +
httpx.MockTransport 本地回放语料 markdown），零网络、零 key；断言落在
results.jsonl 行契约、四条件语义（C1 无 Memory / C2 Vector / C3 Hybrid /
C4 ExternalMemory warm）与 token 对账上。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from researchwiki.loop.metrics import sum_tokens_from_jsonl
from researchwiki.wiki.store import WikiStore

ROOT = Path(__file__).resolve().parents[1]
QA_PATH = ROOT / "evals" / "qa" / "ai-frameworks.jsonl"
FIXTURES_DIR = ROOT / "evals" / "fixtures" / "ai-frameworks"


def _load(name: str, relative: str) -> ModuleType:
    """按路径加载脚本；先注册进 sys.modules，否则 dataclass 解析注解会失败。"""
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


run_eval = _load("run_eval", "scripts/run_eval.py")

# mock 模式、题集取前 2 题（Q001/Q002，均 single_hop）的公共参数
BASE_ARGS = [
    "--provider",
    "mock",
    "--qa",
    str(QA_PATH),
    "--fixtures",
    str(FIXTURES_DIR),
    "--limit",
    "2",
    "--out",
]


@pytest.fixture(autouse=True)
def _no_search_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """零网络保证：mock 模式已显式走 FixtureSearch + MockTransport；
    这里再清掉搜索相关环境变量，兜住一切环境巧合（与 adaptive_smoke 测试同约定）。"""
    for var in ("SEARCH_PROVIDER", "TAVILY_API_KEY", "BOCHA_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def _read_rows(out_dir: Path) -> list[dict[str, Any]]:
    lines = (out_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


BASE_ROW_KEYS = {
    "qid",
    "qtype",
    "condition",
    "answer",
    "gold_points",
    "em",
    "point_hits",
    "refusal",
    "citation_coverage",
    "judge",
    "metrics",
    "trace_id",
    "wiki_root",
    "input_tokens",
    "output_tokens",
    "latency_ms",
    "provider_note",
}


def test_mock_four_conditions_end_to_end(tmp_path, capsys) -> None:
    out_root = tmp_path / "results"
    rc = run_eval.main([*BASE_ARGS, str(out_root), "--env-file", ""])

    assert rc == 0
    run_dirs = list(out_root.glob("mock_*"))
    assert len(run_dirs) == 1
    out_dir = run_dirs[0]
    rows = _read_rows(out_dir)

    # 恰好 8 行 = 2 题 × 4 条件；每行字段契约齐全（fresh_search_count 仅 loop 条件携带）
    assert len(rows) == 8
    for row in rows:
        expected = set(BASE_ROW_KEYS)
        if row["condition"] in ("c1", "c4"):
            expected.add("fresh_search_count")
        assert set(row) == expected, f"{row['condition']} 行字段不齐：{sorted(set(row) ^ expected)}"
        assert 0.0 <= row["em"] <= 1.0
        assert isinstance(row["point_hits"], int) and row["point_hits"] >= 0
        assert isinstance(row["refusal"], bool)
        assert row["citation_coverage"] is None or 0.0 <= row["citation_coverage"] <= 1.0
        assert row["provider_note"]
        judge = row["judge"]
        assert isinstance(judge, dict), "mock judge 输出是可解析占位 JSON，必须产出 verdict"
        for field in ("coverage", "citation", "temporal"):
            assert 1 <= judge[field] <= 5

    by_cond: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_cond.setdefault(row["condition"], []).append(row)
    assert sorted(by_cond) == ["c1", "c2", "c3", "c4"]
    for cond, cond_rows in by_cond.items():
        assert sorted(r["qid"] for r in cond_rows) == ["Q001", "Q002"]
        assert all(r["qtype"] == "single_hop" for r in cond_rows)
        assert all(r["gold_points"] for r in cond_rows)

    # C1 无 Memory：fresh 检索真的发生（scripted web_search），prior 从未命中
    for row in by_cond["c1"]:
        assert row["fresh_search_count"] > 0
        assert row["metrics"]["prior_hit_count"] == 0

    # C4 warm：seed 播种后每题独立副本的 prior 必须命中，且副本里确有 seed 笔记
    for row in by_cond["c4"]:
        assert row["metrics"]["prior_hit_count"] > 0
        assert Path(row["wiki_root"]).is_dir()
        assert WikiStore(row["wiki_root"]).list_notes(), "C4 副本必须含 seed 沉淀的笔记"

    # C2/C3：RAG 条件——mode 留痕、有命中、不建 wiki（wiki_root 为空串）
    assert all(r["metrics"]["mode"] == "vector" for r in by_cond["c2"])
    assert all(r["metrics"]["mode"] == "hybrid" for r in by_cond["c3"])
    assert all(r["metrics"]["hit_count"] > 0 for r in by_cond["c2"] + by_cond["c3"])
    assert all(r["wiki_root"] == "" for r in by_cond["c2"] + by_cond["c3"])

    # token 对账：loop 条件按 per-root tokens.jsonl + trace_id 复算一致
    for row in by_cond["c1"] + by_cond["c4"]:
        tok_in, tok_out = sum_tokens_from_jsonl(
            Path(row["wiki_root"]) / "tokens.jsonl", row["trace_id"]
        )
        assert tok_in > 0, "对账不应空转"
        assert (row["input_tokens"], row["output_tokens"]) == (tok_in, tok_out)
    # RAG 条件按 <out>/tokens.jsonl 复算一致
    for row in by_cond["c2"] + by_cond["c3"]:
        tok_in, tok_out = sum_tokens_from_jsonl(out_dir / "tokens.jsonl", row["trace_id"])
        assert tok_in > 0
        assert (row["input_tokens"], row["output_tokens"]) == (tok_in, tok_out)

    # 8 次 run 全部独立 trace；loop 条件的 per-question root 互不相同
    assert len({row["trace_id"] for row in rows}) == 8
    loop_roots = [row["wiki_root"] for row in rows if row["condition"] in ("c1", "c4")]
    assert len(set(loop_roots)) == 4

    # manifest provenance：题集 sha256、语料 provider_note、URL 映射声明、seed 记录
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["qa_sha256"] == hashlib.sha256(QA_PATH.read_bytes()).hexdigest()
    corpus_manifest = json.loads(
        (FIXTURES_DIR / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["corpus_provider_note"] == corpus_manifest["provider_note"]
    assert "fixture://" in manifest["fixture_url_mapping"]
    assert manifest["provider_mode"] == "mock"
    assert manifest["seed"]["question"]
    assert manifest["conditions"] == ["c1", "c2", "c3", "c4"]

    # stdout 汇总带防误读声明与落盘位置
    printed = capsys.readouterr().out
    assert "小样本" in printed
    assert "results.jsonl" in printed


def test_conditions_subset_rag_only(tmp_path) -> None:
    """--conditions 子集：只跑请求的条件；RAG-only 不播种、不建 wiki 根目录。"""
    out_root = tmp_path / "results"
    rc = run_eval.main(
        [*BASE_ARGS, str(out_root), "--env-file", "", "--conditions", "c2,c3"]
    )

    assert rc == 0
    out_dir = next(out_root.glob("mock_*"))
    rows = _read_rows(out_dir)
    assert len(rows) == 4
    assert {row["condition"] for row in rows} == {"c2", "c3"}
    assert all("fresh_search_count" not in row for row in rows)
    assert not (out_dir / "seed").exists()
    assert not list(out_root.glob("mock_*/c1_*"))
    assert not list(out_root.glob("mock_*/c4_*"))


def test_default_seed_question_prefers_multi_hop() -> None:
    """缺省种子问题：题集第一道 multi_hop 题；截断后无 multi_hop 回退首题。"""
    items = run_eval.load_qa(QA_PATH)
    first_multi = next(item for item in items if item.qtype == "multi_hop")
    assert run_eval.default_seed_question(items) == first_multi.question
    assert run_eval.default_seed_question(items[:2]) == items[0].question


def test_invalid_condition_rejected_by_argparse(tmp_path) -> None:
    """非法条件名在 argparse 层拒绝（exit code 2），不产生任何输出目录。"""
    out_root = tmp_path / "results"
    with pytest.raises(SystemExit) as excinfo:
        run_eval.main([*BASE_ARGS, str(out_root), "--conditions", "c1,c9"])
    assert excinfo.value.code != 0
    assert not out_root.exists()
