"""scripts/report_sequence.py 的测试：手造 events 行 → 四段报告与预注册裁决契约。

脚本不是包，用 importlib 按路径加载（与 tests/test_report_eval.py 同款）。手工构造
events.jsonl 行比真实跑 harness 快且稳，行覆盖简报要求的边界：judge None 行、
gold_source=override 行、簇内重复对与 straddle 对同在、无 simple 路由（变体 B）、
RAG 行无 fresh/wiki_root 键。judge 成本聚合用 tmp_path 造两类 tokens.jsonl
（RAG 记账 <out>/tokens.jsonl + loop 行 wiki_root/tokens.jsonl）验证。

主 fixture 时间线（9 事件 × 4 条件，全部数字手算可验）：
  e0 ingest（c4 study 1000） e1 SEQ001(簇内 1st) e2 SEQ001(簇内 2nd)
  e3 SEQ009(straddle 1st)   e4 SEQ016(无答案)    e5 update（c4 study 500）
  e6 SEQ013(时效)           e7 SEQ009(straddle 2nd) e8 SEQ004(override 单列)

成本账本（主 in_tok；judge in_tok 每问 300）：
  c1 主 1400 / judge 2100 / 全口径 3500；c2 主 350 / 2450；
  c3 主 420 / 2520；c4 主 1500(study)+470(query)=1970 / 4070。
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


report_sequence = _load("report_sequence", "scripts/report_sequence.py")

DIR_NAME = "sequence_mock_20261010T000000Z"


# ---- manifest.pre_registered 原文（run_sequence.PRE_REGISTERED 逐字拷贝）--------


def _pre_registered() -> list[dict[str, str]]:
    return [
        {
            "dimension": "经济性",
            "成立需要": "c4-auto 累计成本在 ≤22 问内 ≤ c2 同期，或 judge cov 优势 ≥+0.5",
            "证伪条件": "全程 c4 累计 > c2 且无任何质量维度优势",
        },
        {
            "dimension": "复用质量",
            "成立需要": "重复问第二次 judge cov ≥ 第一次且 simple 路由发生（率 >0）",
            "证伪条件": "simple 路由率为 0 或复用后质量下降",
        },
        {
            "dimension": "演化价值",
            "成立需要": "更新后时效题 c4 优于 c2/c3",
            "证伪条件": "c4 时效题无优势（supersede 未转化为答案正确性）",
        },
        {
            "dimension": "反幻觉",
            "成立需要": "（测量，无成立条件）",
            "证伪条件": "c4 重演 RQ028 编造",
        },
    ]


# ---- 手造 events 行（run_sequence 行 schema 的手工同形子集）--------------------


def _row(
    condition: str,
    event_id: int,
    kind: str,
    qid: str = "",
    *,
    in_tok: int = 0,
    out_tok: int = 0,
    trace_id: str = "",
    judge: dict[str, Any] | None = None,
    em: float | None = None,
    point_hits: int | None = None,
    refusal: bool = False,
    citation_coverage: float | None = None,
    gold_source: str = "base",
    policy: str | None = None,
    wiki_root: str | None = None,
    fresh: int = 0,
    latency: float = 100.0,
    hit_urls: list[str] | None = None,
    embedding: int | None = None,
) -> dict[str, Any]:
    """一行 events.jsonl（loop 行带 wiki_root/run_dir/fresh_search_count；RAG 行带
    hit_urls 且无 fresh 键；索引行带 embedding_calls；c4 行带 policy_mode）。"""
    row: dict[str, Any] = {
        "condition": condition,
        "event_id": event_id,
        "event_kind": kind,
        "qid": qid,
        "in_tok": in_tok,
        "out_tok": out_tok,
        "latency_ms": latency,
        "em": em,
        "point_hits": point_hits,
        "refusal": refusal,
        "citation_coverage": citation_coverage,
        "judge": judge,
        "gold_source": gold_source,
        "trace_id": trace_id,
    }
    if wiki_root is not None:
        row["wiki_root"] = wiki_root
        row["run_dir"] = wiki_root
        row["fresh_search_count"] = fresh
    if policy is not None:
        row["policy_mode"] = policy
    if hit_urls is not None:
        row["hit_urls"] = hit_urls
    if embedding is not None:
        row["embedding_calls"] = embedding
    return row


def _judge(coverage: int) -> dict[str, Any]:
    return {"coverage": coverage, "citation": 3, "temporal": 3, "reasons": "理由"}


def _base_rows(results_dir: Path) -> list[dict[str, Any]]:
    """主 fixture 34 行（7 query 事件 × 4 条件 + c4 study×2 + c2/c3 索引×2×2）。

    c4 query 路由：e1–e7 simple ×6、e8 update ×1 → simple 率 6/7。
    latency 手算锚点：c1 全行 [100..700] → p50=400 / p95=670；
    c4 全行 [110..170, 800, 1000] → p50=150 / p95=920。
    """
    root = str(results_dir)
    return [
        # e0 ingest：c4 study + c2/c3 索引行（c1 不产行）
        _row("c4", 0, "ingest", in_tok=1000, out_tok=100, trace_id="t-c4-s0",
             wiki_root=f"{root}/c4", fresh=4, policy="deep", latency=1000.0),
        _row("c2", 0, "ingest", trace_id="", embedding=12, latency=5.0),
        _row("c3", 0, "ingest", trace_id="", embedding=12, latency=5.0),
        # e1 query SEQ001（簇内重复对第 1 次）
        _row("c1", 1, "query", "SEQ001", in_tok=200, out_tok=20, trace_id="t-c1-e1",
             wiki_root=f"{root}/c1_e1_SEQ001", fresh=3, em=1.0, point_hits=2,
             judge=_judge(3), latency=100.0),
        _row("c2", 1, "query", "SEQ001", in_tok=50, out_tok=10, trace_id="t-c2-e1",
             em=0.0, point_hits=1, judge=_judge(2), citation_coverage=0.5,
             hit_urls=["http://fixture.local/x/doc-a.md"], latency=50.0),
        _row("c3", 1, "query", "SEQ001", in_tok=60, out_tok=12, trace_id="t-c3-e1",
             em=0.0, point_hits=1, judge=_judge(2), latency=60.0),
        _row("c4", 1, "query", "SEQ001", in_tok=80, out_tok=15, trace_id="t-c4-e1",
             wiki_root=f"{root}/c4", fresh=1, em=1.0, point_hits=2,
             judge=_judge(3), policy="simple", latency=110.0),
        # e2 query SEQ001（簇内重复对第 2 次：c4 cov 3→4 复用不掉分）
        _row("c1", 2, "query", "SEQ001", in_tok=200, out_tok=20, trace_id="t-c1-e2",
             wiki_root=f"{root}/c1_e2_SEQ001", fresh=3, em=1.0, judge=_judge(3),
             latency=200.0),
        _row("c2", 2, "query", "SEQ001", in_tok=50, out_tok=10, trace_id="t-c2-e2",
             em=0.0, judge=_judge(2), latency=50.0),
        _row("c3", 2, "query", "SEQ001", in_tok=60, out_tok=12, trace_id="t-c3-e2",
             em=0.0, judge=_judge(2), latency=60.0),
        _row("c4", 2, "query", "SEQ001", in_tok=40, out_tok=15, trace_id="t-c4-e2",
             wiki_root=f"{root}/c4", fresh=0, em=1.0, judge=_judge(4),
             policy="simple", latency=120.0),
        # e3 query SEQ009（straddle 第 1 次，update 前）
        _row("c1", 3, "query", "SEQ009", in_tok=200, out_tok=20, trace_id="t-c1-e3",
             wiki_root=f"{root}/c1_e3_SEQ009", fresh=3, em=1.0, judge=_judge(3),
             latency=300.0),
        _row("c2", 3, "query", "SEQ009", in_tok=50, out_tok=10, trace_id="t-c2-e3",
             em=1.0, judge=_judge(2), latency=50.0),
        _row("c3", 3, "query", "SEQ009", in_tok=60, out_tok=12, trace_id="t-c3-e3",
             em=1.0, judge=_judge(2), latency=60.0),
        _row("c4", 3, "query", "SEQ009", in_tok=70, out_tok=15, trace_id="t-c4-e3",
             wiki_root=f"{root}/c4", fresh=1, em=1.0, judge=_judge(4),
             policy="simple", latency=130.0),
        # e4 query SEQ016（无答案题：四条件 judge cov 5 = 干净拒答，非编造）
        _row("c1", 4, "query", "SEQ016", in_tok=200, out_tok=20, trace_id="t-c1-e4",
             wiki_root=f"{root}/c1_e4_SEQ016", fresh=3, em=0.0, judge=_judge(5),
             latency=400.0),
        _row("c2", 4, "query", "SEQ016", in_tok=50, out_tok=10, trace_id="t-c2-e4",
             em=0.0, judge=_judge(5), latency=50.0),
        _row("c3", 4, "query", "SEQ016", in_tok=60, out_tok=12, trace_id="t-c3-e4",
             em=0.0, judge=_judge(5), latency=60.0),
        _row("c4", 4, "query", "SEQ016", in_tok=70, out_tok=15, trace_id="t-c4-e4",
             wiki_root=f"{root}/c4", fresh=1, em=0.0, judge=_judge(5),
             policy="simple", latency=140.0),
        # e5 update：c4 study + c2/c3 索引行
        _row("c4", 5, "update", in_tok=500, out_tok=60, trace_id="t-c4-s5",
             wiki_root=f"{root}/c4", fresh=2, policy="deep", latency=800.0),
        _row("c2", 5, "update", trace_id="", embedding=15, latency=6.0),
        _row("c3", 5, "update", trace_id="", embedding=15, latency=6.0),
        # e6 query SEQ013（时效题：c4 优于 c2/c3）
        _row("c1", 6, "query", "SEQ013", in_tok=200, out_tok=20, trace_id="t-c1-e6",
             wiki_root=f"{root}/c1_e6_SEQ013", fresh=3, em=0.0, judge=_judge(2),
             latency=500.0),
        _row("c2", 6, "query", "SEQ013", in_tok=50, out_tok=10, trace_id="t-c2-e6",
             em=0.0, judge=_judge(1), latency=50.0),
        _row("c3", 6, "query", "SEQ013", in_tok=60, out_tok=12, trace_id="t-c3-e6",
             em=0.0, judge=_judge(1), latency=60.0),
        _row("c4", 6, "query", "SEQ013", in_tok=90, out_tok=15, trace_id="t-c4-e6",
             wiki_root=f"{root}/c4", fresh=1, em=1.0, judge=_judge(5),
             policy="simple", latency=150.0),
        # e7 query SEQ009（straddle 第 2 次，update 后——不计入复用摊销判定）
        _row("c1", 7, "query", "SEQ009", in_tok=200, out_tok=20, trace_id="t-c1-e7",
             wiki_root=f"{root}/c1_e7_SEQ009", fresh=3, em=1.0, judge=_judge(3),
             latency=600.0),
        _row("c2", 7, "query", "SEQ009", in_tok=50, out_tok=10, trace_id="t-c2-e7",
             em=1.0, judge=_judge(2), latency=50.0),
        _row("c3", 7, "query", "SEQ009", in_tok=60, out_tok=12, trace_id="t-c3-e7",
             em=1.0, judge=_judge(2), latency=60.0),
        _row("c4", 7, "query", "SEQ009", in_tok=60, out_tok=15, trace_id="t-c4-e7",
             wiki_root=f"{root}/c4", fresh=1, em=1.0, judge=_judge(4),
             policy="simple", latency=160.0),
        # e8 query SEQ004（override 单列：演化正确性 v2 口径；c4 路由 update）
        _row("c1", 8, "query", "SEQ004", in_tok=200, out_tok=20, trace_id="t-c1-e8",
             wiki_root=f"{root}/c1_e8_SEQ004", fresh=3, em=1.0, judge=_judge(4),
             gold_source="override", latency=700.0),
        _row("c2", 8, "query", "SEQ004", in_tok=50, out_tok=10, trace_id="t-c2-e8",
             em=0.0, judge=_judge(2), gold_source="override", latency=50.0),
        _row("c3", 8, "query", "SEQ004", in_tok=60, out_tok=12, trace_id="t-c3-e8",
             em=0.0, judge=_judge(2), gold_source="override", latency=60.0),
        _row("c4", 8, "query", "SEQ004", in_tok=60, out_tok=15, trace_id="t-c4-e8",
             wiki_root=f"{root}/c4", fresh=1, em=1.0, judge=_judge(5),
             gold_source="override", policy="update", latency=170.0),
    ]


def _patch_rows(
    rows: list[dict[str, Any]], overrides: list[tuple[str, int, dict[str, Any]]]
) -> list[dict[str, Any]]:
    """按 (condition, event_id) 覆盖字段（就地修改，供变体构造）。"""
    for condition, event_id, patch in overrides:
        target = next(
            r for r in rows
            if r["condition"] == condition and r["event_id"] == event_id
        )
        target.update(patch)
    return rows


def _write_qa(results_dir: Path) -> None:
    """序列题集子集（qtype 是时效/无答案判定的必要输入）。"""
    items = [
        {"qid": "SEQ001", "question": "Q1", "qtype": "multi_hop",
         "gold_points": ["要点一", "要点二"], "entities": ["甲", "乙"], "notes": ""},
        {"qid": "SEQ009", "question": "Q9", "qtype": "multi_hop",
         "gold_points": ["要点一", "要点二"], "entities": ["甲", "乙"], "notes": ""},
        {"qid": "SEQ016", "question": "Q16", "qtype": "unanswerable",
         "gold_points": [], "entities": ["甲"], "notes": "语料不含此信息"},
        {"qid": "SEQ013", "question": "Q13", "qtype": "temporal",
         "gold_points": ["要点一", "要点二"], "entities": ["甲", "乙"], "notes": "v1/v2 对"},
        {"qid": "SEQ004", "question": "Q4", "qtype": "single_hop",
         "gold_points": ["要点一", "要点二"], "entities": ["甲", "乙"], "notes": ""},
    ]
    (results_dir / "qa.jsonl").write_text(
        "".join(json.dumps(o, ensure_ascii=False) + "\n" for o in items),
        encoding="utf-8",
    )


def _write_tokens(results_dir: Path, rows: list[dict[str, Any]]) -> None:
    """按行造两类 tokens.jsonl：loop 行 → wiki_root/tokens.jsonl；RAG 行 →
    <out>/tokens.jsonl。主 trace 行内 in/out 如实；query 行追加 -judge trace
    （in=300 out=50——judge 成本聚合断言的数据源）。"""
    files: dict[Path, list[str]] = {}
    for row in rows:
        trace = row.get("trace_id") or ""
        if not trace:
            continue  # 索引行无记账
        target = (
            Path(row["wiki_root"]) / "tokens.jsonl"
            if row.get("wiki_root")
            else results_dir / "tokens.jsonl"
        )
        lines = files.setdefault(target, [])
        lines.append(json.dumps(
            {"trace_id": trace, "input_tokens": row["in_tok"],
             "output_tokens": row["out_tok"]}, ensure_ascii=False))
        if row["event_kind"] == "query":
            lines.append(json.dumps(
                {"trace_id": f"{trace}-judge", "input_tokens": 300,
                 "output_tokens": 50}, ensure_ascii=False))
    for path, lines in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")


def _write_manifest(results_dir: Path) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "gate": "run_sequence",
        "generated_at": "2026-10-10T00:00:00+00:00",
        "provider_mode": "mock",
        "model": {
            "strong": {"model": "mock-strong", "base_url": ""},
            "cheap": {"model": "mock-cheap", "base_url": ""},
            "judge": {"model": "mock-judge", "base_url": ""},
        },
        "pre_registered": _pre_registered(),
        "conditions": ["c1", "c2", "c3", "c4"],
        "condition_notes": {
            "c1": "C1 无 Memory", "c2": "C2 Vector RAG",
            "c3": "C3 Hybrid RAG", "c4": "C4 ExternalMemory",
        },
        "event_count": 9,
        "qa_path": str(results_dir / "qa.jsonl"),
        "qa_sha256": "cd" * 32,
        "rag_tokens_path": str(results_dir / "tokens.jsonl"),
        "judge_trace_suffix": "-judge",
        "corpus_provider_note": "受控语料 provider_note 原文（测试引用）",
    }
    (results_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _write_fixture(results_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """写全 events/qa/tokens/manifest 四件套（目录须已存在）。"""
    (results_dir / "events.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    _write_qa(results_dir)
    _write_tokens(results_dir, rows)
    return _write_manifest(results_dir)


@pytest.fixture(scope="module")
def generated(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """主 fixture 生成一次报告，供曲线 / 拐点 / 裁决断言共享（零网络零模型）。"""
    root = tmp_path_factory.mktemp("seq")
    results_dir = root / DIR_NAME
    results_dir.mkdir()
    manifest = _write_fixture(results_dir, _base_rows(results_dir))
    out_dir = root / "reports"
    rc = report_sequence.main(["--results", str(results_dir), "--out", str(out_dir)])
    assert rc == 0
    report_path = out_dir / f"{DIR_NAME}.md"
    assert report_path.is_file(), "报告必须以 <results 目录名>.md 生成"
    return {
        "results_dir": results_dir,
        "manifest": manifest,
        "rows": _base_rows(results_dir),
        "text": report_path.read_text(encoding="utf-8"),
    }


def _verdict_section(text: str, dimension: str) -> str:
    """取 "### 预注册判定·<dimension>" 小节（到下一个标题为止）。"""
    start = text.index(f"### 预注册判定·{dimension}")
    rest = text[start + len(f"### 预注册判定·{dimension}"):]
    cuts = [p for p in (rest.find("\n### "), rest.find("\n## ")) if p >= 0]
    return rest if not cuts else rest[: min(cuts)]


# ---- 四段齐全 + 头部（预注册原文逐字在头部）------------------------------------


def test_report_four_sections_and_preregistered_verbatim(
    generated: dict[str, Any],
) -> None:
    """四段标题齐全；预注册四条原文（成立需要/证伪条件共 8 句）逐字出现在头部。"""
    text = generated["text"]
    assert text.startswith(f"# 复用序列实验报告：{DIR_NAME}")
    assert "## 预注册判定（manifest 原文" in text
    assert "## 累计成本曲线" in text
    assert "## 摊销拐点" in text
    assert "## 预注册逐条裁决" in text

    # 头部：生成时间 / results 目录 / provider / 模型摘要
    assert "生成时间" in text
    assert str(generated["results_dir"].resolve()) in text
    assert "provider 模式：mock" in text
    assert "mock-strong" in text and "mock-judge" in text

    # 预注册判定原文逐字引用（manifest.pre_registered，跑前写死的证据）
    for item in _pre_registered():
        assert item["成立需要"] in text
        assert item["证伪条件"] in text
        assert item["dimension"] in text


# ---- 累计成本曲线（累计数字 + c4 策略标注 + judge 两类文件聚合）----------------


def test_curve_cumulative_policy_and_judge_aggregation(
    generated: dict[str, Any],
) -> None:
    """曲线表逐事件累计 in_tok 正确（含 study/索引行）、c4 query 行标注 policy、
    judge 列按 -judge trace 从两类 tokens.jsonl 聚合（c1/c4 ← wiki_root，
    c2/c3 ← <out>/tokens.jsonl）；embedding 列累计索引行 embedding_calls
    （spec §1 计量入账——c2/c3 e0=12、e5 update 后 27，c1/c4 恒 0）。"""
    text = generated["text"]
    assert (
        "| 事件 | c4 策略 | c1 累计 in | c2 累计 in | c3 累计 in | c4 累计 in "
        "| c1 累计 emb | c2 累计 emb | c3 累计 emb | c4 累计 emb "
        "| c1 judge | c2 judge | c3 judge | c4 judge |" in text
    )
    expected_rows = [
        "| e0/ingest | — | 0 | 0 | 0 | 1000 | 0 | 12 | 12 | 0 | 0 | 0 | 0 | 0 |",
        "| e1/query/SEQ001 | simple | 200 | 50 | 60 | 1080 | 0 | 12 | 12 | 0 | 300 | 300 | 300 | 300 |",
        "| e2/query/SEQ001 | simple | 400 | 100 | 120 | 1120 | 0 | 12 | 12 | 0 | 600 | 600 | 600 | 600 |",
        "| e3/query/SEQ009 | simple | 600 | 150 | 180 | 1190 | 0 | 12 | 12 | 0 | 900 | 900 | 900 | 900 |",
        "| e4/query/SEQ016 | simple | 800 | 200 | 240 | 1260 | 0 | 12 | 12 | 0 | 1200 | 1200 | 1200 | 1200 |",
        "| e5/update | — | 800 | 200 | 240 | 1760 | 0 | 27 | 27 | 0 | 1200 | 1200 | 1200 | 1200 |",
        "| e6/query/SEQ013 | simple | 1000 | 250 | 300 | 1850 | 0 | 27 | 27 | 0 | 1500 | 1500 | 1500 | 1500 |",
        "| e7/query/SEQ009 | simple | 1200 | 300 | 360 | 1910 | 0 | 27 | 27 | 0 | 1800 | 1800 | 1800 | 1800 |",
        "| e8/query/SEQ004 | update | 1400 | 350 | 420 | 1970 | 0 | 27 | 27 | 0 | 2100 | 2100 | 2100 | 2100 |",
    ]
    for row in expected_rows:
        assert row in text, f"缺曲线行：{row}"
    # 合计行：主 in_tok / embedding / judge 累计（judge 与 embedding 均不在主 in_tok 内）
    assert (
        "| 合计（全程） | — | 1400 | 350 | 420 | 1970 | 0 | 27 | 27 | 0 "
        "| 2100 | 2100 | 2100 | 2100 |" in text
    )
    # 条件总账：行数 / 主 / judge / 全口径 / out / 累计 embedding_calls / p50 p95
    assert "| c1 | 7 | 1400 | 2100 | 3500 | 140 | 0 | 400 | 670 |" in text
    assert "| c2 | 9 | 350 | 2100 | 2450 | 70 | 27 | 50 | 50 |" in text
    assert "| c4 | 9 | 1970 | 2100 | 4070 | 265 | 0 | 150 | 920 |" in text

    # judge 聚合口径直查：loop 行走 wiki_root/tokens.jsonl，RAG 行走出 tokens.jsonl，
    # 非 query 行（study/索引）不聚合 judge
    jt = report_sequence.judge_tokens(
        generated["rows"], generated["manifest"], generated["results_dir"]
    )
    by_key = {
        (r["condition"], r["event_id"]): tok for r, tok in zip(generated["rows"], jt)
    }
    assert by_key[("c1", 1)] == (300, 50)
    assert by_key[("c4", 2)] == (300, 50)
    assert by_key[("c2", 1)] == (300, 50)
    assert by_key[("c3", 8)] == (300, 50)
    assert by_key[("c4", 0)] == (0, 0)  # study 行不聚合 judge
    assert by_key[("c2", 5)] == (0, 0)  # 索引行不聚合 judge
    # 凭证声明披露两类 tokens.jsonl 来源
    assert str(generated["results_dir"] / "tokens.jsonl") in text
    assert str(generated["results_dir"] / "c4" / "tokens.jsonl") in text


# ---- 摊销拐点（build_cost、交叉或未交叉、摊销表）------------------------------


def test_amortization_build_cost_and_no_crossing(generated: dict[str, Any]) -> None:
    """c4 累计（含 study 构建成本 1500）全程高于 c2/c1 → 两处"全程未交叉"；
    摊销表第 7 次查询后 c4=4070/7=581.4、c2=350.0、c3=360.0、c1=500.0（全口径）。"""
    text = generated["text"]
    assert "1500" in text  # build_cost = c4 study 行 in_tok 合计
    assert text.count("全程未交叉") == 2  # vs c2 与 vs c1
    assert "4070" in text and "2450" in text and "3500" in text
    # 预注册摊销口径（build_cost/reuse_count + query_cost）在第 N 次查询后的值
    assert "| 7 | SEQ004 | e8/query/SEQ004 | 581.4 | 350.0 | 360.0 | 500.0 |" in text
    # embedding 口径声明（c2/c3 真金 embedding 成本不因不折算而被隐没）
    assert "embedding 调用次数" in text
    assert "c2=27、c3=27" in text


# ---- 预注册逐条裁决（主 fixture：三态之"成立"+"未证伪"）------------------------


def test_verdicts_hold_with_data_citations(generated: dict[str, Any]) -> None:
    """主 fixture：经济性（质量优势路径）/复用质量/演化价值 成立、反幻觉未证伪；
    每条裁决都有数据引用；straddle 对排除、simple 率、override 单列齐备。"""
    text = generated["text"]

    eco = _verdict_section(text, "经济性")
    assert "裁决：成立" in eco
    assert "4070" in eco and "2450" in eco  # c4/c2 全口径累计
    assert "+2.33" in eco and "4.17" in eco and "1.83" in eco  # cov 均值差路径
    # embedding 口径声明（spec §1 计量入账——c2/c3 的 embedding_calls 来自行数据）
    assert "embedding 调用次数" in eco
    assert "c2=27、c3=27" in eco
    assert "不折算货币" in eco and "不含 embedding tokens" in eco

    reuse = _verdict_section(text, "复用质量")
    assert "裁决：成立" in reuse
    assert "| SEQ001 | e1 | 3.0 | e2 | 4.0 | 是 |" in reuse  # 簇内重复对
    assert "SEQ009" in reuse and "straddle" in reuse  # straddle 对排除披露
    assert "6/7" in reuse and "0.857" in reuse  # simple 路由率 >0
    assert "simple×6" in reuse and "update×1" in reuse  # policy_mode 统计

    evolution = _verdict_section(text, "演化价值")
    assert "裁决：成立" in evolution
    assert "SEQ013" in evolution and "5.0" in evolution and "1.0" in evolution
    # SEQ004 ev8 override 单列（演化正确性，v2 口径命中）
    assert "SEQ004" in evolution and "override" in evolution
    assert "| SEQ004 | e8 | c4 |" in evolution and "5.0" in evolution

    anti = _verdict_section(text, "反幻觉")
    assert "裁决：未证伪（测量）" in anti
    assert "SEQ016" in anti and "RQ028" in anti
    assert "| SEQ016 | c4 | 5.0 | 否 |" in anti

    # 主 fixture 未触发双双证伪条款
    assert "未双双证伪" in text


# ---- 裁决三态之"证伪"（复用掉分 + 时效无优势 + 无答案编造）----------------------


def test_verdicts_falsified_scenario(tmp_path: Path) -> None:
    """变体 A：c4 重复对掉分→复用证伪；时效题劣于 c2/c3→演化证伪；无答案 judge
    cov=1→反幻觉证伪（RQ028 编造信号）；经济性随双双证伪落证伪，结论条款触发。"""
    results_dir = tmp_path / DIR_NAME
    results_dir.mkdir()
    rows = _patch_rows(_base_rows(results_dir), [
        # 复用质量：簇内重复对第二次掉分（4→2）
        ("c4", 1, {"judge": _judge(4)}),
        ("c4", 2, {"judge": _judge(2)}),
        # 其余可答题 cov 压低 → cov 均值差转负
        ("c4", 3, {"judge": _judge(1)}),
        ("c4", 7, {"judge": _judge(1)}),
        ("c4", 8, {"judge": _judge(1)}),
        # 演化价值：时效题 c4 劣于 c2/c3
        ("c4", 6, {"judge": _judge(0), "em": 0.0}),
        ("c2", 6, {"judge": _judge(3), "em": 1.0}),
        ("c3", 6, {"judge": _judge(3), "em": 1.0}),
        # 反幻觉：无答案题 judge cov=1 → 编造信号
        ("c4", 4, {"judge": _judge(1)}),
    ])
    _write_fixture(results_dir, rows)
    out_dir = tmp_path / "reports_a"
    rc = report_sequence.main(["--results", str(results_dir), "--out", str(out_dir)])
    assert rc == 0
    text = (out_dir / f"{DIR_NAME}.md").read_text(encoding="utf-8")

    assert "裁决：证伪" in _verdict_section(text, "复用质量")
    assert "裁决：证伪" in _verdict_section(text, "演化价值")
    assert "裁决：证伪" in _verdict_section(text, "反幻觉")
    assert "裁决：证伪" in _verdict_section(text, "经济性")
    # spec §4 结论条款：经济性与演化价值双双证伪
    assert "双双证伪" in text
    assert "未证明存在 RAG 之外的价值" in text


# ---- 复用质量部分对 judge 缺失 → 不可判定（修复 1 + 修复 3）--------------------


def test_reuse_undetermined_when_pairs_partially_evaluable(tmp_path: Path) -> None:
    """变体 C：两个簇内重复对中一个的 judge 缺失 → 复用质量不可判定（注明 N/M
    可评估），绝不把缺失对合并进有利结论出"成立"。

    顺带覆盖修复 3：把 e5 的 update 行改写为 query 行（时间线无 update 事件），
    repeat_pairs 应按"全部 query 对=簇内对"识别出 SEQ001 与 SEQ009 两对——
    无 update 事件不再整表丢弃重复对。
    """
    results_dir = tmp_path / DIR_NAME
    results_dir.mkdir()
    rows = _base_rows(results_dir)
    for row in rows:
        if row["event_id"] == 5:
            row["event_kind"] = "query"  # 抹掉 update 事件（qid 为空不影响行识别）
    next(
        r for r in rows if r["condition"] == "c4" and r["event_id"] == 7
    )["judge"] = None  # SEQ009 对的第二侧 judge 缺失
    _write_fixture(results_dir, rows)
    out_dir = tmp_path / "reports_c"
    rc = report_sequence.main(["--results", str(results_dir), "--out", str(out_dir)])
    assert rc == 0
    text = (out_dir / f"{DIR_NAME}.md").read_text(encoding="utf-8")

    reuse = _verdict_section(text, "复用质量")
    assert "裁决：不可判定" in reuse
    assert "1/2" in reuse  # N/M 可评估注明
    # 缺失侧 judge cov 显示 "—"（不是 0，也不显示"否"冒充下降）
    assert "| SEQ009 | e3 | 4.0 | e7 | — | — |" in reuse
    assert "| SEQ001 | e1 | 3.0 | e2 | 4.0 | 是 |" in reuse


# ---- 裁决三态之"不可判定"（c4 judge 全缺 + 无 simple 路由）---------------------


def test_verdicts_undetermined_when_judge_missing(tmp_path: Path) -> None:
    """变体 B：c4 query 行 judge 全 None 且无 policy_mode（simple 路由率 0）→
    复用质量按预注册证伪（路由率为 0），经济性/演化价值/反幻觉不可判定——
    绝不在证据缺失时编造结论。"""
    results_dir = tmp_path / DIR_NAME
    results_dir.mkdir()
    rows = _base_rows(results_dir)
    for row in rows:
        if row["condition"] == "c4" and row["event_kind"] == "query":
            row.pop("policy_mode", None)
            row["judge"] = None
            if row["event_id"] == 6:
                row["em"] = 0.0  # 时效题 EM 与 c2/c3 持平 → 无 EM 优势路径可判
    _write_fixture(results_dir, rows)
    out_dir = tmp_path / "reports_b"
    rc = report_sequence.main(["--results", str(results_dir), "--out", str(out_dir)])
    assert rc == 0
    text = (out_dir / f"{DIR_NAME}.md").read_text(encoding="utf-8")

    assert "裁决：证伪" in _verdict_section(text, "复用质量")  # simple 路由率 0
    assert "无 policy_mode×7" in _verdict_section(text, "复用质量")
    for dimension in ("经济性", "演化价值", "反幻觉"):
        assert "裁决：不可判定" in _verdict_section(text, dimension)


# ---- 凭证声明与坏输入退出码 -----------------------------------------------------


def test_credentials_and_bad_inputs_exit_2(
    generated: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """凭证声明（events/manifest sha256 + 可重算声明）；events/manifest/qa 缺失
    或 events 为空 → 退出码 2 且不产出报告。"""
    text = generated["text"]
    results_dir = generated["results_dir"]
    assert "所有结论可由原始凭证重算" in text
    events_sha = hashlib.sha256((results_dir / "events.jsonl").read_bytes()).hexdigest()
    assert events_sha in text
    assert generated["manifest"]["qa_sha256"] in text

    out = tmp_path / "out"
    # 缺 events.jsonl
    missing = tmp_path / "sequence_mock_missing"
    missing.mkdir()
    assert report_sequence.main(["--results", str(missing), "--out", str(out)]) == 2
    # events.jsonl 为空
    empty = tmp_path / "sequence_mock_empty"
    empty.mkdir()
    (empty / "events.jsonl").write_text("", encoding="utf-8")
    _write_qa(empty)
    _write_manifest(empty)
    assert report_sequence.main(["--results", str(empty), "--out", str(out)]) == 2
    # 缺 manifest.json
    nomanifest = tmp_path / "sequence_mock_nomanifest"
    nomanifest.mkdir()
    (nomanifest / "events.jsonl").write_text('{"condition": "c1"}\n', encoding="utf-8")
    assert report_sequence.main(["--results", str(nomanifest), "--out", str(out)]) == 2
    # manifest.qa_path 悬空（时效/无答案判定缺输入 → 拒绝出报告）
    noqa = tmp_path / "sequence_mock_noqa"
    noqa.mkdir()
    (noqa / "events.jsonl").write_text(
        json.dumps(_row("c4", 0, "ingest"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest = _write_manifest(noqa)
    manifest["qa_path"] = str(noqa / "dangling.jsonl")
    (noqa / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    assert report_sequence.main(["--results", str(noqa), "--out", str(out)]) == 2
    # 预注册对比锚点缺失（events 无 c4 行）→ 干净退出码 2，非裸 KeyError
    noc4 = tmp_path / "sequence_mock_noc4"
    noc4.mkdir()
    rows = [r for r in _base_rows(noc4) if r["condition"] != "c4"]
    (noc4 / "events.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    _write_qa(noc4)
    _write_tokens(noc4, rows)
    _write_manifest(noc4)
    assert report_sequence.main(["--results", str(noc4), "--out", str(out)]) == 2

    assert not list(out.glob("*.md"))
    assert capsys.readouterr().out  # 坏输入有友好报错输出
