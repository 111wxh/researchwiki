#!/usr/bin/env python
"""评测报告聚合器：从评测产物目录生成条件×指标矩阵 markdown 报告。

── 用途（评测支线 PLAN §7.3 的消费件）────────────────────────────────────
读取 ``<provider>_<ts>/`` 目录（Task 6 run_eval 的产物：results.jsonl 每题每
条件一行 + manifest.json provenance），聚合出人读报告 ``<目录名>.md``：

  ① 头部：生成时间、results 目录、manifest 的 config 摘要（provider/模型/条件）
     与 corpus_provider_note / honest_boundary 原文引用、题集路径 + sha256；
  ② 条件×指标矩阵（行=条件，列=n、EM 均值、拒答率、citation_coverage 均值、
     judge 三维均值、input/output tokens 均值、latency p50/p95、fresh_search 均
     值）——EM / judge / citation_coverage 的分母可能与行数不同（None 行不计入），
     每格旁标 n；fresh_search 键整组缺失显示 "—" 并脚注（RAG 条件无此指标）；
  ③ 逐题明细表（qid / qtype / condition / em / refusal / judge 三分 / 理由截
     断 60 字）；
  ④ 文末凭证声明：results.jsonl / tokens.jsonl / manifest 路径与 sha256，
     声明"所有结论可由原始凭证重算"。

── 聚合口径 ────────────────────────────────────────────────────────────
条件级 n / EM 均值 / 拒答率 / latency p50/p95 / token 均值 / fresh_search 均值
直接复用 ``evals.metrics.aggregate_rows``（与 run_eval stdout 汇总同口径，行字
段兼容其缺失语义）；judge 三维均值与 citation_coverage 均值在本脚本补算
（aggregate_rows 不输出这两项）：只统计对应字段非 None 的行，分母以 n= 标注写
进报告——缺失不计入分母，绝不把缺失当 0。

── 怎么跑 ─────────────────────────────────────────────────────────────
  uv run --no-sync python scripts/report_eval.py --results evals/results/mock_20261002T101112Z
  uv run --no-sync python scripts/report_eval.py --results <dir> --out evals/reports
退出码：0 成功；2 输入不可用（目录 / results.jsonl / manifest.json 缺失、凭证
损坏或 0 行）——报告缺输入宁可报错，不产半份误导性报告。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from researchwiki.evals.metrics import aggregate_rows

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "evals" / "reports"

#: 条件展示顺序的缺省口径（manifest.conditions 缺失/非法时回退）
DEFAULT_CONDITION_ORDER: tuple[str, ...] = ("c1", "c2", "c3", "c4")

#: judge 三个分数维度（与 evals.judge.JudgeVerdict 字段同序）
JUDGE_FIELDS: tuple[str, str, str] = ("coverage", "citation", "temporal")

#: 逐题明细 judge 理由的截断长度（字）
REASON_LIMIT = 60

#: 矩阵列（顺序即渲染顺序；单元格断言按本表下标取值）
MATRIX_COLUMNS: tuple[str, ...] = (
    "条件",
    "n",
    "EM 均值",
    "拒答率",
    "citation_coverage 均值",
    "judge n",
    "judge coverage 均值",
    "judge citation 均值",
    "judge temporal 均值",
    "input_tokens 均值",
    "output_tokens 均值",
    "latency p50 (ms)",
    "latency p95 (ms)",
    "fresh_search 均值",
)

EXIT_PASS = 0
EXIT_BAD_INPUT = 2


# ---- 输入读取 ----------------------------------------------------------------


def load_rows(results_path: Path) -> list[dict[str, Any]]:
    """results.jsonl → 行 dict 列表（跳过空行；坏行抛 ValueError，由上层拒绝聚合）。"""
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(results_path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"第 {lineno} 行不是合法 JSON：{exc}") from exc
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _numeric(value: Any) -> float | None:
    """bool 不算数值（JSON true/false 不是评分/指标）；int/float → float。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def condition_order(rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]) -> list[str]:
    """矩阵行序：manifest.conditions 声明优先（只留结果里出现的），其余条件按名追加。"""
    present = {str(r["condition"]) for r in rows if r.get("condition") is not None}
    declared = manifest.get("conditions")
    if not isinstance(declared, (list, tuple)):
        declared = DEFAULT_CONDITION_ORDER
    order = [c for c in declared if c in present]
    order += sorted(present - set(order))
    return order


# ---- 聚合（aggregate_rows 之外的补算项）---------------------------------------


def judge_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """按条件聚合 judge 三维均值（仅 judge 非 None 的行；aggregate_rows 不含此项）。

    返回 ``{condition: {"n": 行数, "coverage": 均值, "citation": 均值, "temporal":
    均值}}``；某维分数缺失/非数值的行不计入该维均值（不编造分数）。
    """
    counts: dict[str, int] = {}
    sums: dict[str, list[float]] = {}
    dims: dict[str, list[int]] = {}
    for row in rows:
        cond = row.get("condition")
        judge = row.get("judge")
        if cond is None or not isinstance(judge, Mapping):
            continue
        key = str(cond)
        counts[key] = counts.get(key, 0) + 1
        acc = sums.setdefault(key, [0.0, 0.0, 0.0])
        dim_n = dims.setdefault(key, [0, 0, 0])
        for i, field in enumerate(JUDGE_FIELDS):
            value = _numeric(judge.get(field))
            if value is not None:
                acc[i] += value
                dim_n[i] += 1
    return {
        key: {
            "n": counts[key],
            **{
                name: (sums[key][i] / dims[key][i] if dims[key][i] else 0.0)
                for i, name in enumerate(JUDGE_FIELDS)
            },
        }
        for key in counts
    }


def coverage_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float | int]]:
    """按条件聚合 citation_coverage 均值（仅非 None 行；n= 真实分母）。"""
    counts: dict[str, int] = {}
    sums: dict[str, float] = {}
    for row in rows:
        cond = row.get("condition")
        value = _numeric(row.get("citation_coverage"))
        if cond is None or value is None:
            continue
        key = str(cond)
        counts[key] = counts.get(key, 0) + 1
        sums[key] = sums.get(key, 0.0) + value
    return {
        key: {"n": counts[key], "mean": (sums[key] / counts[key] if counts[key] else 0.0)}
        for key in counts
    }


def em_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """每条件 em 非 None 行数（EM 均值的真实分母，aggregate_rows 不输出）。"""
    counts: dict[str, int] = {}
    for row in rows:
        cond = row.get("condition")
        if cond is None or row.get("em") is None:
            continue
        key = str(cond)
        counts[key] = counts.get(key, 0) + 1
    return counts


# ---- 渲染 --------------------------------------------------------------------


def _cell(value: Any) -> str:
    """表格单元格净化：竖线换全角斜杠、空白折成单行（markdown 表格不能断行）。"""
    return " ".join(str(value).split()).replace("|", "／")


def _truncate(text: str, limit: int = REASON_LIMIT) -> str:
    """折叠空白后按字数截断，超长加省略号（"截断 60 字"）。"""
    folded = " ".join(str(text).split())
    if len(folded) <= limit:
        return folded
    return folded[:limit] + "…"


def _model_line(manifest: Mapping[str, Any]) -> str:
    """manifest.model 摘要 → ``strong=x / cheap=y / judge=z``（缺失档位显示 —）。"""
    model = manifest.get("model")
    if not isinstance(model, Mapping) or not model:
        return "—"
    parts = []
    for tier in ("strong", "cheap", "judge"):
        info = model.get(tier)
        name = info.get("model") if isinstance(info, Mapping) else None
        parts.append(f"{tier}={_cell(name) or '—'}")
    return " / ".join(parts)


def _render_header(
    results_dir: Path, manifest: Mapping[str, Any], generated_at: str
) -> list[str]:
    """头部：生成时间 / results 目录 / config 摘要 / 题集路径+sha256 / 原文引用。"""
    lines = [
        f"# 评测报告：{results_dir.name}",
        "",
        f"- 生成时间：{generated_at}",
        f"- results 目录：`{results_dir.resolve()}`",
        f"- provider 模式：{_cell(manifest.get('provider_mode')) or '—'}",
        f"- 模型（strong / cheap / judge）：{_model_line(manifest)}",
        f"- embedding：{_cell(manifest.get('embedding_model')) or '—'}",
        f"- 题集：`{_cell(manifest.get('qa_path')) or '—'}`"
        f"（sha256：`{_cell(manifest.get('qa_sha256')) or '—'}`，"
        f"共 {_cell(manifest.get('qa_count', '—'))} 题）",
    ]
    conditions = manifest.get("conditions")
    if isinstance(conditions, (list, tuple)) and conditions:
        lines.append(f"- 条件：{'、'.join(_cell(c) for c in conditions)}")
    notes = manifest.get("condition_notes")
    if isinstance(notes, Mapping) and notes:
        lines.extend(["", "条件口径："])
        lines.extend(f"- {cond}：{_cell(note)}" for cond, note in notes.items())
    lines.extend(["", "## 口径原文引用（manifest）", ""])
    for key in ("corpus_provider_note", "honest_boundary"):
        value = manifest.get(key)
        if value:
            lines.append(f"**{key}**：")
            lines.extend(f"> {line}" for line in str(value).splitlines() or [""])
            lines.append("")
    return lines


def _render_matrix(rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]) -> list[str]:
    """条件×指标矩阵：aggregate_rows 直读 + judge / citation_coverage 均值补算。"""
    summary = aggregate_rows(list(rows))
    jstats = judge_stats(rows)
    cstats = coverage_stats(rows)
    em_n = em_counts(rows)

    lines = [
        "## 条件 × 指标矩阵",
        "",
        "| " + " | ".join(MATRIX_COLUMNS) + " |",
        "|" + "---|" * len(MATRIX_COLUMNS),
    ]
    for cond in condition_order(rows, manifest):
        s = summary[cond]
        j = jstats.get(cond)
        cov = cstats.get(cond)
        fresh_cell = f"{s['fresh_search_mean']:.2f}" if "fresh_search_mean" in s else "—¹"
        cells = [
            cond,
            str(s["n"]),
            f"{s['em_mean']:.4f}（n={em_n.get(cond, 0)}）",
            f"{s['refusal_rate']:.4f}",
            f"{cov['mean']:.4f}（n={cov['n']}）" if cov else "—（n=0）",
            str(j["n"]) if j else "0",
            f"{j['coverage']:.1f}" if j else "—",
            f"{j['citation']:.1f}" if j else "—",
            f"{j['temporal']:.1f}" if j else "—",
            f"{s['input_tokens_mean']:.1f}",
            f"{s['output_tokens_mean']:.1f}",
            f"{s['latency_p50']:.0f}",
            f"{s['latency_p95']:.0f}",
            fresh_cell,
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "注：",
            "- EM 均值分母 = 该条件 em 非 None 行数；citation_coverage 均值分母 = 非 None 行数"
            "（格内 n= 标注）；judge 三维均值仅统计 judge 非 None 的行（分母见 judge n 列）"
            "——缺失行不计入分母，绝不把缺失当 0。",
            "- ¹ “—”＝该条件组内无任何行携带 fresh_search_count（RAG 条件无此指标）——"
            "区分“未测”与“测得 0”。",
            "- 小样本评测，不构成收益结论（PLAN §11.1）；条件口径差异见头部 honest_boundary 原文。",
            "",
        ]
    )
    return lines


def _render_detail(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """逐题明细：qid/qtype/condition/em/refusal/judge 三分/一句话理由（≤60 字）。"""
    lines = [
        "## 逐题明细",
        "",
        "| qid | qtype | 条件 | EM | 拒答 | judge（覆盖/引用/时效） | judge 理由（≤60 字） |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        em = _numeric(row.get("em"))
        judge = row.get("judge")
        if isinstance(judge, Mapping):
            scores = [_numeric(judge.get(field)) for field in JUDGE_FIELDS]
            judge_cell = (
                "/".join(f"{value:.0f}" for value in scores)
                if all(value is not None for value in scores)
                else "—"
            )
            reason_cell = _truncate(str(judge.get("reasons") or "—"))
        else:
            judge_cell = "—"
            reason_cell = "—"
        lines.append(
            "| {qid} | {qtype} | {cond} | {em} | {refusal} | {judge} | {reason} |".format(
                qid=_cell(row.get("qid") or "—"),
                qtype=_cell(row.get("qtype") or "—"),
                cond=_cell(row.get("condition") or "—"),
                em=f"{em:.2f}" if em is not None else "—",
                refusal="是" if row.get("refusal") else "否",
                judge=judge_cell,
                reason=reason_cell,
            )
        )
    lines.append("")
    return lines


def _render_credentials(
    results_dir: Path, rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> list[str]:
    """文末凭证声明：原始凭证路径 + sha256 + "所有结论可由原始凭证重算"。"""
    base = results_dir.resolve()
    results_sha = hashlib.sha256((base / "results.jsonl").read_bytes()).hexdigest()
    rag_tokens = manifest.get("rag_tokens_path") or str(base / "tokens.jsonl")
    loop_conditions = sorted(
        {
            str(r["condition"])
            for r in rows
            if r.get("condition") is not None and "fresh_search_count" in r
        }
    )
    lines = [
        "## 凭证声明",
        "",
        "本报告是聚合视图，**所有结论可由原始凭证重算**：",
        "",
        f"- results.jsonl（每题每条件一行）：`{base / 'results.jsonl'}`"
        f"（报告生成时 sha256：`{results_sha}`）",
        f"- manifest.json（run provenance）：`{base / 'manifest.json'}`"
        f"（题集 qa_sha256：`{_cell(manifest.get('qa_sha256')) or '—'}`）",
        f"- tokens.jsonl（RAG 条件记账）：`{_cell(rag_tokens)}`",
    ]
    if loop_conditions:
        lines.append(
            f"- loop 条件（{'、'.join(loop_conditions)}）逐题记账：各行 `wiki_root` 下的"
            " tokens.jsonl（judge 记账在 `<trace_id>-judge` 独立 trace；trace_id 见"
            " results.jsonl 对应行）"
        )
    lines.extend(
        [
            "",
            "聚合口径：条件级指标复用 `evals.metrics.aggregate_rows`（与 run_eval stdout 同口径）；"
            "judge 三维与 citation_coverage 均值为本脚本补算，分母见矩阵标注。",
            "",
        ]
    )
    return lines


def build_report(
    results_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    *,
    generated_at: str | None = None,
) -> str:
    """整份报告 markdown（头部 → 矩阵 → 逐题明细 → 凭证声明）。"""
    generated_at = generated_at or datetime.now(UTC).isoformat(timespec="seconds")
    lines = _render_header(results_dir, manifest, generated_at)
    lines += _render_matrix(rows, manifest)
    lines += _render_detail(rows)
    lines += _render_credentials(results_dir, rows, manifest)
    return "\n".join(lines) + "\n"


# ---- 入口 --------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="评测报告聚合器（条件×指标矩阵 markdown，详见文件头 docstring）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results", required=True,
        help="评测产物目录（run_eval 输出：含 results.jsonl 与 manifest.json）",
    )
    parser.add_argument(
        "--out", default=str(DEFAULT_OUT),
        help="报告输出目录（默认 evals/reports；报告名为 <results 目录名>.md）",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001 -- 老环境/被替换的 stdout 上忽略
        pass
    args = build_parser().parse_args(argv)
    results_dir = Path(args.results)
    results_path = results_dir / "results.jsonl"
    manifest_path = results_dir / "manifest.json"

    if not results_path.is_file():
        print(
            f"✗ 找不到 {results_path}——--results 应指向 run_eval 产物目录"
            "（含 results.jsonl 与 manifest.json）"
        )
        return EXIT_BAD_INPUT
    if not manifest_path.is_file():
        print(f"✗ 找不到 {manifest_path}——manifest.json 是报告头部与凭证声明的必要输入")
        return EXIT_BAD_INPUT
    try:
        rows = load_rows(results_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except ValueError as exc:  # json.JSONDecodeError 是其子类
        print(f"✗ 凭证文件损坏，拒绝聚合：{exc}")
        return EXIT_BAD_INPUT
    if not rows:
        print(f"✗ {results_path} 为空（0 行），无法聚合报告")
        return EXIT_BAD_INPUT
    if not isinstance(manifest, dict):
        print(f"✗ {manifest_path} 顶层不是 JSON 对象，无法引用其 provenance")
        return EXIT_BAD_INPUT

    report = build_report(results_dir, rows, manifest)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{results_dir.name}.md"
    out_path.write_text(report, encoding="utf-8")

    print(f"✓ 报告已写入 {out_path}")
    summary = aggregate_rows(rows)
    for cond in condition_order(rows, manifest):
        s = summary[cond]
        extra = (
            f"  fresh_search_mean={s['fresh_search_mean']:.2f}"
            if "fresh_search_mean" in s
            else ""
        )
        print(
            f"  {cond:<4} n={s['n']}  em_mean={s['em_mean']:.4f}"
            f"  refusal_rate={s['refusal_rate']:.4f}{extra}"
        )
    print("注：小样本评测，不构成收益结论（PLAN §11.1）。")
    return EXIT_PASS


if __name__ == "__main__":
    raise SystemExit(main())
