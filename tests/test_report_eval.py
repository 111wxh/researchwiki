"""scripts/report_eval.py 的测试：mock results（手工构造 8 行）→ 报告聚合契约。

脚本不是包，用 importlib 按路径加载（与 tests/test_run_eval.py 同款）。手工构造
results.jsonl 行比真实跑 runner 快且稳，行覆盖三类边界：judge None 行、em None
行、fresh_search 缺失的 RAG 行。断言落在条件×指标矩阵（四条件行齐、每格 n= 与
judge n 列）、"—"脚注（RAG 条件无此指标）、逐题明细（judge None → "—"、理由截
断 60 字）与文末凭证声明（manifest 路径 + sha256）上。零网络零模型。
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

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str) -> ModuleType:
    """按路径加载脚本；先注册进 sys.modules，否则模块级注解解析会失败。"""
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


report_eval = _load("report_eval", "scripts/report_eval.py")

LONG_REASON = "字" * 80  # 超过 60 字的 judge 理由（测截断）


def _row(qid: str, condition: str, **overrides: Any) -> dict[str, Any]:
    """一行 results.jsonl（run_eval.build_row 的字段契约；fresh_search_count 按需携带）。"""
    row: dict[str, Any] = {
        "qid": qid,
        "qtype": "single_hop",
        "condition": condition,
        "answer": f"{qid}/{condition} 的 mock 回答[1]。",
        "gold_points": ["要点一"],
        "em": 0.5,
        "point_hits": 1,
        "refusal": False,
        "citation_coverage": 0.5,
        "judge": None,
        "metrics": {},
        "trace_id": f"t-{qid}-{condition}",
        "wiki_root": "",
        "input_tokens": 100,
        "output_tokens": 10,
        "latency_ms": 100.0,
        "provider_note": "note",
    }
    row.update(overrides)
    return row


def _mock_rows() -> list[dict[str, Any]]:
    """2 题 × 4 条件 = 8 行，聚合结果全部手算可验：

    - c1：em {1.0, 0.5}、refusal 1/2、judge n=1（Q001 有 Q002 None）、fresh {2,1}；
    - c2：Q001 em None + judge 有分、Q002 cov None + judge None（RAG 无 fresh 键）；
    - c3：judge n=2（均值 4.5/4.0/3.5）、Q001 理由超长（测截断）；
    - c4：Q001 em None、fresh {3,0}（测得 0 计入均值，非"未测"）、Q001 理由含
      换行与竖线（测明细单元格净化，真实 judge reasons 常见形态）。
    """
    return [
        _row(
            "Q001", "c1", em=1.0, citation_coverage=0.5,
            judge={"coverage": 3, "citation": 4, "temporal": 5, "reasons": "理由甲"},
            fresh_search_count=2, input_tokens=1000, output_tokens=100, latency_ms=100.0,
        ),
        _row(
            "Q002", "c1", em=0.5, refusal=True, citation_coverage=1.0,
            fresh_search_count=1, input_tokens=1200, output_tokens=150, latency_ms=200.0,
        ),
        _row(
            "Q001", "c2", em=None, citation_coverage=0.25,
            judge={"coverage": 2, "citation": 2, "temporal": 2, "reasons": "理由乙"},
            input_tokens=100, output_tokens=50, latency_ms=50.0,
        ),
        _row(
            "Q002", "c2", em=0.0, citation_coverage=None,
            input_tokens=110, output_tokens=55, latency_ms=70.0,
        ),
        _row(
            "Q001", "c3", em=1.0, citation_coverage=1.0,
            judge={"coverage": 5, "citation": 5, "temporal": 4, "reasons": LONG_REASON},
            input_tokens=200, output_tokens=60, latency_ms=90.0,
        ),
        _row(
            "Q002", "c3", em=0.5, citation_coverage=0.5,
            judge={"coverage": 4, "citation": 3, "temporal": 3, "reasons": "理由丙"},
            input_tokens=210, output_tokens=65, latency_ms=110.0,
        ),
        _row(
            "Q001", "c4", em=None, citation_coverage=0.75,
            judge={
                "coverage": 3,
                "citation": 3,
                "temporal": 3,
                "reasons": "理由丁：第一段\n第二段|含|竖线",
            },
            fresh_search_count=3, input_tokens=900, output_tokens=200, latency_ms=300.0,
        ),
        _row(
            "Q002", "c4", em=1.0, citation_coverage=0.5,
            fresh_search_count=0, input_tokens=800, output_tokens=180, latency_ms=400.0,
        ),
    ]


def _mock_manifest(results_dir: Path) -> dict[str, Any]:
    """手写 manifest（run_eval.main 落盘字段的同形子集，够报告头部与凭证引用）。"""
    return {
        "gate": "run_eval",
        "generated_at": "2026-10-03T00:00:00+00:00",
        "provider_mode": "mock",
        "model": {
            "strong": {"model": "mock-strong", "base_url": ""},
            "cheap": {"model": "mock-cheap", "base_url": ""},
            "judge": {"model": "mock-judge", "base_url": ""},
        },
        "qa_path": str(results_dir / "qa.jsonl"),
        "qa_sha256": "ab" * 32,
        "qa_count": 2,
        "fixtures_dir": str(results_dir / "fixtures"),
        "corpus_provider_note": "受控语料 provider_note 原文（测试引用）",
        "fixture_url_mapping": "fixture:// → http://fixture.local/（测试）",
        "conditions": ["c1", "c2", "c3", "c4"],
        "condition_notes": {
            "c1": "C1 无 Memory",
            "c2": "C2 Vector RAG",
            "c3": "C3 Hybrid RAG",
            "c4": "C4 warm",
        },
        "honest_boundary": "C2/C3 硬约束；C1/C4 近似对齐（测试原文）。",
        "rag_tokens_path": str(results_dir / "tokens.jsonl"),
        "embedding_model": "",
    }


@pytest.fixture(scope="module")
def generated(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """生成一次报告，供矩阵 / 明细 / 凭证三组断言共享（手工 mock 行，零网络）。"""
    root = tmp_path_factory.mktemp("eval")
    results_dir = root / "mock_20261003T000000Z"
    results_dir.mkdir()
    (results_dir / "results.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in _mock_rows()),
        encoding="utf-8",
    )
    manifest = _mock_manifest(results_dir)
    (results_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    out_dir = root / "reports"
    rc = report_eval.main(["--results", str(results_dir), "--out", str(out_dir)])
    assert rc == 0
    report_path = out_dir / "mock_20261003T000000Z.md"
    assert report_path.is_file(), "报告文件必须以 <results 目录名>.md 生成"
    return {
        "results_dir": results_dir,
        "report_path": report_path,
        "text": report_path.read_text(encoding="utf-8"),
        "manifest": manifest,
        "rows": _mock_rows(),
    }


def _matrix_row(text: str, condition: str) -> list[str]:
    """取矩阵中某条件行的单元格列表（列序见 report_eval.MATRIX_COLUMNS）。"""
    for line in text.splitlines():
        if line.startswith(f"| {condition} |"):
            return [cell.strip() for cell in line.strip().strip("|").split("|")]
    raise AssertionError(f"矩阵缺 {condition} 行：\n{text}")


def test_matrix_four_conditions_with_n_annotations(generated: dict[str, Any]) -> None:
    """矩阵四条件行齐；EM/citation_coverage 格标 n=、judge n 独立列；"—"有脚注。"""
    text = generated["text"]
    assert "| 条件 | n | EM 均值 | 拒答率 |" in text
    assert "judge n" in text
    for cond in ("c1", "c2", "c3", "c4"):
        _matrix_row(text, cond)  # 四条件行齐全

    # c1：n=2、EM 均值 0.75（分母 2）、judge n=1 → 三维均值 3/4/5、fresh 均值 1.50
    cells = _matrix_row(text, "c1")
    assert cells[1] == "2"
    assert cells[2] == "0.7500（n=2）"
    assert cells[3] == "0.5000"
    assert cells[4] == "0.7500（n=2）"
    assert cells[5] == "1"
    assert (cells[6], cells[7], cells[8]) == ("3.0", "4.0", "5.0")
    assert (cells[9], cells[10]) == ("1100.0", "125.0")
    assert (cells[11], cells[12]) == ("150", "195")
    assert cells[13] == "1.50"

    # c2：em None 行不计入 EM 分母、cov None 行不计入 cov 分母、无 fresh 键 → "—"¹
    cells = _matrix_row(text, "c2")
    assert cells[1] == "2"
    assert cells[2] == "0.0000（n=1）"
    assert cells[4] == "0.2500（n=1）"
    assert cells[5] == "1"
    assert (cells[6], cells[7], cells[8]) == ("2.0", "2.0", "2.0")
    assert cells[13] == "—¹"

    # c3：judge n=2 → 三维均值 4.5/4.0/3.5；RAG 行无 fresh 键 → "—"¹
    cells = _matrix_row(text, "c3")
    assert cells[5] == "2"
    assert (cells[6], cells[7], cells[8]) == ("4.5", "4.0", "3.5")
    assert cells[13] == "—¹"

    # c4：em None 行 → EM 分母 1；fresh {3,0} 测得 0 计入均值（≠"未测"）
    cells = _matrix_row(text, "c4")
    assert cells[2] == "1.0000（n=1）"
    assert cells[4] == "0.6250（n=2）"
    assert cells[13] == "1.50"

    # "—"脚注（RAG 条件无此指标）与分母口径、小样本防误读声明
    assert "RAG 条件无此指标" in text
    assert "小样本" in text


def test_detail_table_and_judge_none_rows(generated: dict[str, Any]) -> None:
    """逐题明细 8 行齐：judge None → "—"、拒答显示、60 字截断加省略号。"""
    text = generated["text"]
    assert "| qid | qtype | 条件 |" in text
    for row in generated["rows"]:
        assert f"| {row['qid']} | {row['qtype']} | {row['condition']} |" in text

    # 有分行：三分 + 一句话理由
    q1c1 = next(ln for ln in text.splitlines() if ln.startswith("| Q001 | single_hop | c1 |"))
    assert "3/4/5" in q1c1 and "理由甲" in q1c1
    # judge None 行：judge 与理由两格都是 "—"；拒答行显示"是"
    q2c1 = next(ln for ln in text.splitlines() if ln.startswith("| Q002 | single_hop | c1 |"))
    assert q2c1.endswith("| — | — |")
    assert "| 是 |" in q2c1
    # em None 行：EM 格 "—"
    q1c2 = next(ln for ln in text.splitlines() if ln.startswith("| Q001 | single_hop | c2 |"))
    assert "| — | 否 | 2/2/2 |" in q1c2
    # 超长理由截断到 60 字 + "…"，第 61 字绝不出现
    q1c3 = next(ln for ln in text.splitlines() if ln.startswith("| Q001 | single_hop | c3 |"))
    assert ("字" * 60 + "…") in q1c3
    assert ("字" * 61) not in text


def test_detail_reason_cell_sanitizes_pipes_and_newlines(generated: dict[str, Any]) -> None:
    """judge 理由含 | 与换行 → 单元格净化后明细表仍是单一表格行。

    真实 judge reasons 常含竖线/换行；裸 | 会把 markdown 表格撑破——
    竖线须换全角斜杠、换行折成空格（与报告声明"折行折叠、竖线换全角斜杠"一致）。
    """
    text = generated["text"]
    q1c4 = next(ln for ln in text.splitlines() if ln.startswith("| Q001 | single_hop | c4 |"))
    cells = [c.strip() for c in q1c4.strip().strip("|").split("|")]
    assert len(cells) == 7, f"裸 | 撑破表格行：{q1c4}"
    # 换行折成空格、竖线换全角斜杠，原句完整落在最后一个单元格内
    assert cells[6] == "理由丁：第一段 第二段／含／竖线"
    # 原始片段（含裸 |）绝不出现在该行
    assert "第二段|含|竖线" not in q1c4


def test_credentials_and_manifest_provenance(generated: dict[str, Any]) -> None:
    """凭证声明：results/tokens/manifest 路径与 sha256；头部引用 manifest 原文。"""
    text = generated["text"]
    results_dir = generated["results_dir"]
    manifest = generated["manifest"]

    assert "所有结论可由原始凭证重算" in text
    assert f"`{results_dir / 'results.jsonl'}`" in text
    results_sha = hashlib.sha256((results_dir / "results.jsonl").read_bytes()).hexdigest()
    assert results_sha in text
    assert str(manifest["rag_tokens_path"]) in text
    assert f"`{results_dir / 'manifest.json'}`" in text
    assert manifest["qa_sha256"] in text

    # 头部：生成时间、results 目录、config 摘要、题集路径 + sha256、原文引用
    assert "生成时间" in text
    assert str(results_dir.resolve()) in text
    assert "mock-strong" in text and "mock-judge" in text
    assert str(manifest["qa_path"]) in text
    assert manifest["corpus_provider_note"] in text
    assert manifest["honest_boundary"] in text


def test_empty_results_exit_code_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """results.jsonl 为空（0 行）→ 友好报错、退出码 2、不产出报告文件。"""
    results_dir = tmp_path / "mock_empty"
    results_dir.mkdir()
    (results_dir / "results.jsonl").write_text("", encoding="utf-8")
    (results_dir / "manifest.json").write_text("{}", encoding="utf-8")
    out_dir = tmp_path / "reports"

    rc = report_eval.main(["--results", str(results_dir), "--out", str(out_dir)])

    assert rc == 2
    assert "results.jsonl" in capsys.readouterr().out
    assert not list(out_dir.glob("*.md"))


def test_missing_manifest_exit_code_2(tmp_path: Path) -> None:
    """缺 manifest.json（报告头与凭证声明的必要输入）→ 退出码 2，不产出报告。"""
    results_dir = tmp_path / "mock_broken"
    results_dir.mkdir()
    (results_dir / "results.jsonl").write_text(
        '{"qid": "Q1", "condition": "c1"}\n', encoding="utf-8"
    )

    rc = report_eval.main(["--results", str(results_dir), "--out", str(tmp_path / "reports")])

    assert rc == 2
    assert not list((tmp_path / "reports").glob("*.md"))
