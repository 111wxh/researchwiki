#!/usr/bin/env python
"""复用序列实验报告：累计成本曲线 + 摊销拐点 + 预注册判定逐条裁决。

── 用途（复用序列实验 PLAN Task 3）──────────────────────────────────────
消费 Task 2 的产物目录 ``evals/results/sequence_<provider>_<ts>/``（events.jsonl
每行一个条件×事件 + manifest.json 含 pre_registered 预注册判定原文），产出人读
报告 ``evals/reports/<results 目录名>.md``——**报告的核心是对 spec §4 的预注册
判定四条逐条裁决**（每条给 成立|证伪|不可判定 + 数据引用，不许只有结论词）：

  ① 头部：生成时间、results 目录、manifest 的 provider/config 摘要、**预注册
     判定四条原文**（逐字引用 manifest.pre_registered——跑前写死的证据）；
  ② 累计成本曲线表：每条件逐事件累计 in_tok（含 ingest/update 事件行），行=
     事件序（event_id+kind+qid），列=四条件累计；c4 行标注 policy_mode（query
     事件）；**judge 成本单列**（按 ``-judge`` trace 从两类 tokens.jsonl 聚合、
     累计：c1/c4 的 judge 在各 loop 行 wiki_root/tokens.jsonl，c2/c3 在
     manifest.rag_tokens_path 的 <out>/tokens.jsonl）；
  ③ 摊销拐点：c4 累计（含 study 构建成本）vs c2/c1 累计——交叉事件序或
     "全程未交叉"；摊销口径 ``build_cost/reuse_count + query_cost`` 在第 N 次
     查询后的值（全口径含 judge）；
  ④ 预注册逐条裁决：
     - 经济性：c4 累计在 22 问内是否 ≤ c2 同期，或可答题 judge cov 差 ≥+0.5；
     - 复用质量：**只按簇内重复对**（同 qid 两次查询都在 update 前；straddle
       更新后重复对排除披露）第二次 judge cov ≥ 第一次，且 simple 路由率 >0
       （c4 query 行 policy_mode 统计一并给出）；
     - 演化价值：update 后时效题（qtype=temporal、gold_source=base）c4 vs
       c2/c3 的 EM 与 judge cov；gold_source=override 行单列演化正确性（v2
       口径 diff-gold 命中）；
     - 反幻觉：无答案题（qtype=unanswerable）c4 的 judge cov 与 refusal——
       对照 RQ028 编造基线（judge cov=1 即编造信号），测量报告。

── 裁决规则（对预注册原文逐条机械化，不在证据缺失时编造结论）──────────────
  经济性：c4 全口径累计 ≤ c2 → 成立；否则可答题 judge cov 均值差（c4−c2）≥+0.5
    → 成立；否则仅当复用质量与演化价值**双双证伪**（=无任何质量维度优势）→
    证伪；其余（含质量维度不可判定）→ 不可判定。
  复用质量：有可评重复对（两侧 judge cov 齐）且任一对下降 → 证伪；simple 路由
    率 =0 → 证伪（预注册原文）；全部可评对第二次 ≥ 第一次且率 >0 → 成立；
    无簇内重复对或对侧 judge 缺失且率 >0 → 不可判定。
  演化价值：c4 时效题 judge cov 或 EM 均值同时优于 c2 与 c3 → 成立；四项均值
    齐且 c4 对 c2/c3 全不占优 → 证伪；数据不齐（judge 缺失/条件缺行）→ 不可
    判定。
  反幻觉（测量项，无成立条件）：c4 无答案题任一 judge cov ≤1 → 证伪（编造
    信号）；无行或 judge 全缺 → 不可判定；否则 → 未证伪（测量）。

── 怎么跑 ─────────────────────────────────────────────────────────────
  uv run --no-sync python scripts/report_sequence.py --results evals/results/sequence_mock_20261010T090000Z
  uv run --no-sync python scripts/report_sequence.py --results <dir> --out evals/reports
退出码：0 成功；2 输入不可用（目录 / events.jsonl / manifest.json / 题集缺失、
凭证损坏或 0 行）——报告缺输入宁可报错，不产半份误导性报告（report_eval 同款）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from researchwiki.evals.metrics import mean, percentile
from researchwiki.evals.qa import load_qa
from researchwiki.loop.metrics import sum_tokens_from_jsonl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "evals" / "reports"

#: spec §4（预注册判定的权威来源，头部引用）
SPEC_PATH = "docs/superpowers/specs/2026-10-10-reuse-sequence-eval-design.md"

#: 条件展示顺序的缺省口径（manifest.conditions 缺失/非法时回退）
DEFAULT_CONDITION_ORDER: tuple[str, ...] = ("c1", "c2", "c3", "c4")

#: judge 独立 trace 后缀（run_sequence manifest.judge_trace_suffix 同款）
JUDGE_SUFFIX = "-judge"

#: 裁决三态（反幻觉是测量项，无成立条件，用"未证伪（测量）"表达干净测量）
VERDICT_HOLD = "成立"
VERDICT_FALSIFIED = "证伪"
VERDICT_UNDETERMINED = "不可判定"
VERDICT_NOT_FALSIFIED = "未证伪（测量）"

EXIT_PASS = 0
EXIT_BAD_INPUT = 2


# ---- 基础工具（_cell 与 report_eval 同款两行逻辑：防裸 | 撑破 markdown 表格）--


def _cell(value: Any) -> str:
    """表格单元格净化：竖线换全角斜杠、空白折成单行（markdown 表格不能断行）。"""
    return " ".join(str(value).split()).replace("|", "／")


def _numeric(value: Any) -> float | None:
    """bool 不算数值（JSON true/false 不是评分/指标）；int/float → float。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _fmt(value: float | None, nd: int = 2) -> str:
    """数值单元格：None → "—"（区分"未测"与"测得 0"）。"""
    return "—" if value is None else f"{value:.{nd}f}"


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


def _eid(row: Mapping[str, Any]) -> int:
    """行的事件序号（event_id 缺失/非数值按 0——排序与聚合的兜底口径）。"""
    value = _numeric(row.get("event_id"))
    return int(value) if value is not None else 0


# ---- 输入读取 ----------------------------------------------------------------


def load_rows(events_path: Path) -> list[dict[str, Any]]:
    """events.jsonl → 行 dict 列表（跳过空行；坏行抛 ValueError，由上层拒绝聚合）。"""
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(events_path.read_text(encoding="utf-8").splitlines(), 1):
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


def condition_order(rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]) -> list[str]:
    """条件列序：manifest.conditions 声明优先（只留结果里出现的），其余按名追加。"""
    present = {str(r["condition"]) for r in rows if r.get("condition") is not None}
    declared = manifest.get("conditions")
    if not isinstance(declared, (list, tuple)):
        declared = DEFAULT_CONDITION_ORDER
    order = [c for c in declared if c in present]
    order += sorted(present - set(order))
    return order


def load_qa_types(manifest: Mapping[str, Any]) -> dict[str, str]:
    """manifest.qa_path → {qid: qtype}（时效/无答案判定与可答题口径的必要输入）。

    题集缺失/损坏抛 ValueError（演化价值与反幻觉两条裁决没有 qtype 就无法对
    预注册口径——缺输入宁可报错，不产半份误导性报告）。
    """
    qa_path = manifest.get("qa_path")
    if not isinstance(qa_path, str) or not qa_path or not Path(qa_path).is_file():
        raise ValueError(f"manifest.qa_path 不可用（时效/无答案判定需要 qtype）：{qa_path}")
    return {item.qid: item.qtype for item in load_qa(qa_path)}


# ---- judge 成本聚合（两类 tokens.jsonl：loop 行 wiki_root + RAG 记账 out）------


def judge_tokens(
    rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any], results_dir: Path
) -> list[tuple[int, int]]:
    """逐行聚合 judge 成本 ``(judge_in_tok, judge_out_tok)``。

    T2 移交口径：c1/c4 的 judge token 在各 loop 行 ``wiki_root/tokens.jsonl``；
    c2/c3 在 manifest.rag_tokens_path（<out>/tokens.jsonl）。每行按
    ``trace_id + "-judge"`` 从对应文件汇总（sum_tokens_from_jsonl，文件缺失
    返回 (0,0)）；仅 query 行聚合（study/索引行无 judge trace）。
    """
    rag_path = Path(str(manifest.get("rag_tokens_path") or (Path(results_dir) / "tokens.jsonl")))
    result: list[tuple[int, int]] = []
    for row in rows:
        if row.get("event_kind") != "query" or not row.get("trace_id"):
            result.append((0, 0))
            continue
        if row.get("wiki_root"):
            path = Path(str(row["wiki_root"])) / "tokens.jsonl"
        else:
            path = rag_path
        result.append(sum_tokens_from_jsonl(path, f"{row['trace_id']}{JUDGE_SUFFIX}"))
    return result


# ---- 曲线与聚合 ---------------------------------------------------------------


def build_curve(
    rows: Sequence[Mapping[str, Any]],
    conditions: Sequence[str],
    judge_toks: Sequence[tuple[int, int]],
) -> list[dict[str, Any]]:
    """逐事件累计曲线：行=事件序（event_id 升序），每条件累计主 in_tok 与累计
    judge in_tok（judge 单列口径）；c4 query 事件附 policy_mode 标注。

    累计=截至该事件（含该事件行）的合计；c1 在 ingest/update 事件无行、累计
    保持不变（如实呈现为数字，不编造"—"）。
    """
    ranked = sorted(range(len(rows)), key=lambda i: (_eid(rows[i]), i))
    events: list[dict[str, Any]] = []
    index_of: dict[int, int] = {}
    by_event: dict[int, list[int]] = {}
    for i in ranked:
        row = rows[i]
        eid = _eid(row)
        by_event.setdefault(eid, []).append(i)
        if eid not in index_of:
            index_of[eid] = len(events)
            events.append(
                {
                    "event_id": eid,
                    "kind": str(row.get("event_kind") or ""),
                    "qid": str(row.get("qid") or ""),
                }
            )
        elif row.get("qid") and not events[index_of[eid]]["qid"]:
            events[index_of[eid]]["qid"] = str(row["qid"])

    cum = {c: {"main": 0, "judge": 0} for c in conditions}
    curve: list[dict[str, Any]] = []
    for event in events:
        for i in by_event.get(event["event_id"], []):
            cond = str(rows[i].get("condition") or "")
            if cond in cum:
                cum[cond]["main"] += int(rows[i].get("in_tok") or 0)
                cum[cond]["judge"] += judge_toks[i][0]
        policy = None
        if event["kind"] == "query":
            for i in by_event.get(event["event_id"], []):
                row = rows[i]
                if str(row.get("condition")) == "c4" and row.get("policy_mode") is not None:
                    policy = str(row["policy_mode"])
                    break
        curve.append(
            {
                **event,
                "policy": policy,
                "cum": {c: dict(cum[c]) for c in conditions},
            }
        )
    return curve


def condition_totals(
    rows: Sequence[Mapping[str, Any]],
    conditions: Sequence[str],
    judge_toks: Sequence[tuple[int, int]],
) -> dict[str, dict[str, Any]]:
    """条件总账：行数 / 主 in_tok / judge in_tok / 全口径 / 主 out_tok / latency 列表。"""
    totals: dict[str, dict[str, Any]] = {}
    for i, row in enumerate(rows):
        cond = str(row.get("condition") or "")
        acc = totals.setdefault(
            cond,
            {"rows": 0, "main_in": 0, "judge_in": 0, "main_out": 0, "latencies": []},
        )
        acc["rows"] += 1
        acc["main_in"] += int(row.get("in_tok") or 0)
        acc["judge_in"] += judge_toks[i][0]
        acc["main_out"] += int(row.get("out_tok") or 0)
        latency = _numeric(row.get("latency_ms"))
        if latency is not None:
            acc["latencies"].append(latency)
    for cond in conditions:  # 声明了但零行的条件也给空账（—）
        totals.setdefault(
            cond, {"rows": 0, "main_in": 0, "judge_in": 0, "main_out": 0, "latencies": []}
        )
    return totals


def repeat_pairs(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, int]], list[dict[str, int]], int | None]:
    """从 events 行识别重复问结构（同 qid 多次 query，按事件序去重）：

    - 簇内重复对：两次都在 update 事件之前（预注册复用质量的唯一判定口径）；
    - straddle 更新后重复对：一前一后跨 update（演化语义，排除披露）。
    返回 (intra, straddle, update_event_id)；无 update 事件时两表皆空。
    """
    update_ids = [_eid(r) for r in rows if r.get("event_kind") == "update"]
    update_id = min(update_ids) if update_ids else None
    seen: set[tuple[int, str]] = set()
    by_qid: dict[str, list[int]] = {}
    for row in rows:
        if row.get("event_kind") != "query" or not row.get("qid"):
            continue
        eid, qid = _eid(row), str(row["qid"])
        if (eid, qid) in seen:
            continue
        seen.add((eid, qid))
        by_qid.setdefault(qid, []).append(eid)
    intra: list[dict[str, int]] = []
    straddle: list[dict[str, int]] = []
    if update_id is not None:
        for qid, eids in by_qid.items():
            if len(eids) < 2:
                continue
            first, second = eids[0], eids[1]
            if second < update_id:
                intra.append({"qid": qid, "first": first, "second": second})
            elif first < update_id < second:
                straddle.append({"qid": qid, "first": first, "second": second})
    return intra, straddle, update_id


def _row_at(
    rows: Sequence[Mapping[str, Any]], condition: str, event_id: int
) -> Mapping[str, Any] | None:
    """取 (condition, event_id) 的第一行（每条件每事件至多一行）。"""
    for row in rows:
        if str(row.get("condition") or "") == condition and _eid(row) == event_id:
            return row
    return None


def _judge_cov(row: Mapping[str, Any] | None) -> float | None:
    """行 judge cov（judge None / 缺 coverage → None——缺失不计入分母）。"""
    if row is None:
        return None
    judge = row.get("judge")
    if not isinstance(judge, Mapping):
        return None
    return _numeric(judge.get("coverage"))


def _query_rows(rows: Sequence[Mapping[str, Any]], condition: str) -> list[Mapping[str, Any]]:
    """某条件的全部 query 行（保持 events 文件序）。"""
    return [
        r
        for r in rows
        if str(r.get("condition") or "") == condition and r.get("event_kind") == "query"
    ]


def _score_stats(
    cond_rows: Sequence[Mapping[str, Any]], exclude_qids: frozenset[str] = frozenset()
) -> dict[str, Any]:
    """一组行的 EM / judge cov 均值（缺失不计入分母；全缺 → None——不编造 0）。"""
    covs = [
        cov
        for r in cond_rows
        if str(r.get("qid") or "") not in exclude_qids and (cov := _judge_cov(r)) is not None
    ]
    ems = [
        em
        for r in cond_rows
        if str(r.get("qid") or "") not in exclude_qids and (em := _numeric(r.get("em"))) is not None
    ]
    return {
        "n": len(cond_rows),
        "cov": mean(covs) if covs else None,
        "cov_n": len(covs),
        "em": mean(ems) if ems else None,
        "em_n": len(ems),
    }


def build_stats(
    rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    results_dir: Path,
    qa_types: Mapping[str, str],
) -> dict[str, Any]:
    """裁决所需的全部聚合值（一次算好，渲染与裁决只读不重算）。"""
    conditions = condition_order(rows, manifest)
    judge_toks = judge_tokens(rows, manifest, results_dir)
    totals = condition_totals(rows, conditions, judge_toks)
    curve = build_curve(rows, conditions, judge_toks)
    intra, straddle, update_id = repeat_pairs(rows)
    unanswerable_qids = frozenset(q for q, t in qa_types.items() if t == "unanswerable")
    temporal_qids = frozenset(q for q, t in qa_types.items() if t == "temporal")

    def post_update(row: Mapping[str, Any]) -> bool:
        return update_id is not None and _eid(row) > update_id

    # 可答题（排除无答案题）的 judge cov 均值——经济性质量路径口径
    cov_answerable = {
        c: _score_stats(_query_rows(rows, c), exclude_qids=unanswerable_qids)
        for c in conditions
    }
    # 时效题：update 后、gold_source=base、qtype=temporal
    temporal = {
        c: _score_stats(
            [
                r
                for r in _query_rows(rows, c)
                if post_update(r)
                and r.get("gold_source") == "base"
                and r.get("qid") in temporal_qids
            ]
        )
        for c in conditions
    }
    # 无答案题逐行（反幻觉测量）
    unanswerable = {
        c: [
            {
                "qid": str(r.get("qid") or ""),
                "event_id": _eid(r),
                "cov": _judge_cov(r),
                "refusal": bool(r.get("refusal")),
            }
            for r in _query_rows(rows, c)
            if r.get("qid") in unanswerable_qids
        ]
        for c in conditions
    }
    # override 行（演化正确性 diff-gold，单列）
    override_rows = [
        r for r in rows if r.get("event_kind") == "query" and r.get("gold_source") == "override"
    ]
    # c4 study 构建成本（ingest/update 行主 trace in_tok）与 query 路由统计
    build_cost = sum(
        int(r.get("in_tok") or 0)
        for r in rows
        if str(r.get("condition") or "") == "c4" and r.get("event_kind") in ("ingest", "update")
    )
    c4_query = _query_rows(rows, "c4")
    policy_counts = Counter(
        str(r["policy_mode"]) if r.get("policy_mode") is not None else "无 policy_mode"
        for r in c4_query
    )
    simple_n = policy_counts.get("simple", 0)
    simple_rate = (simple_n / len(c4_query)) if c4_query else None
    return {
        "rows": list(rows),
        "conditions": conditions,
        "totals": totals,
        "curve": curve,
        "intra": intra,
        "straddle": straddle,
        "update_id": update_id,
        "temporal_qids": temporal_qids,
        "unanswerable_qids": unanswerable_qids,
        "cov_answerable": cov_answerable,
        "temporal": temporal,
        "unanswerable": unanswerable,
        "override_rows": override_rows,
        "build_cost": build_cost,
        "policy_counts": policy_counts,
        "simple_rate": simple_rate,
        "c4_query_n": len(c4_query),
    }


# ---- 摊销拐点 -----------------------------------------------------------------


def _all_in(cum: Mapping[str, Any], cond: str) -> int | None:
    """某条件截至某事件的全口径累计（主 + judge）；条件不在曲线内 → None。"""
    entry = cum.get(cond)
    return None if entry is None else entry["main"] + entry["judge"]


def amortization(stats: Mapping[str, Any]) -> dict[str, Any]:
    """摊销拐点：交叉点（c4 vs c2 / c4 vs c1）+ 第 N 次查询后的摊销值表。

    摊销口径（预注册 build_cost/reuse_count + query_cost）：第 N 次查询后
    c4 摊销/问 =（build_cost + c4 query 主累计 + c4 judge 累计）/ N——数值上
    等于 c4 全口径累计 / N（c4 全口径 = study 构建成本 + query 成本 + judge）；
    c2/c3/c1 无构建成本，均值/问 = 全口径累计 / N。
    """
    curve: list[dict[str, Any]] = stats["curve"]
    crossings: dict[str, dict[str, Any] | None] = {}
    for target in ("c2", "c1"):
        crossings[target] = None
        for n, entry in enumerate(
            (e for e in curve if e["kind"] == "query"), start=1
        ):
            c4 = _all_in(entry["cum"], "c4")
            base = _all_in(entry["cum"], target)
            if c4 is None or base is None:
                break  # c4 或对照条件不在结果里——无法对比
            if c4 <= base:
                crossings[target] = {
                    "n": n,
                    "label": _event_label(entry),
                    "c4": c4,
                    "target": base,
                }
                break
    amort_rows: list[dict[str, Any]] = []
    for n, entry in enumerate((e for e in curve if e["kind"] == "query"), start=1):
        values = {c: _all_in(entry["cum"], c) for c in ("c2", "c3", "c1")}
        c4 = _all_in(entry["cum"], "c4")
        amort_rows.append(
            {
                "n": n,
                "qid": entry["qid"],
                "label": _event_label(entry),
                "c4": None if c4 is None else c4 / n,
                **{c: (None if v is None else v / n) for c, v in values.items()},
            }
        )
    return {"crossings": crossings, "rows": amort_rows}


def _event_label(entry: Mapping[str, Any]) -> str:
    """事件行标签：``e<id>/<kind>`` + query 事件追加 ``/<qid>``。"""
    label = f"e{entry['event_id']}/{entry['kind']}"
    if entry.get("qid"):
        label += f"/{entry['qid']}"
    return label


# ---- 预注册逐条裁决 ------------------------------------------------------------


def compute_verdicts(stats: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """预注册四条的机械化裁决：每条返回 {state, citations, prereg}。

    规则见模块 docstring「裁决规则」——对预注册原文逐条机械化，证据缺失时给
    不可判定而非编造结论；每条 citations 都带具体聚合值（数据引用硬要求）。
    """
    totals: Mapping[str, Mapping[str, Any]] = stats["totals"]

    def total_all(cond: str) -> int | None:
        acc = totals.get(cond)
        return None if acc is None else acc["main_in"] + acc["judge_in"]

    # ---- 复用质量（先算：经济性的"无任何质量维度优势"要引用它）------------------
    pair_rows: list[dict[str, Any]] = []
    for pair in stats["intra"]:
        first = _judge_cov(_row_at(stats["rows"], "c4", pair["first"]))
        second = _judge_cov(_row_at(stats["rows"], "c4", pair["second"]))
        pair_rows.append({**pair, "first_cov": first, "second_cov": second})
    evaluable = [p for p in pair_rows if p["first_cov"] is not None and p["second_cov"] is not None]
    declined = [p for p in evaluable if (p["second_cov"] or 0.0) < (p["first_cov"] or 0.0)]
    simple_rate: float | None = stats["simple_rate"]
    if not pair_rows:
        reuse_state = VERDICT_UNDETERMINED
        reuse_why = "events 里没有簇内重复对（同 qid 两次查询都在 update 前）"
    elif declined:
        reuse_state = VERDICT_FALSIFIED
        bad = declined[0]
        reuse_why = (
            f"重复对 {bad['qid']} 第二次 judge cov={_fmt(bad['second_cov'], 1)} < "
            f"第一次 {_fmt(bad['first_cov'], 1)}——复用后质量下降（预注册证伪条件）"
        )
    elif simple_rate is not None and stats["c4_query_n"] > 0 and simple_rate == 0.0:
        reuse_state = VERDICT_FALSIFIED
        reuse_why = f"simple 路由率 = 0（c4 query 共 {stats['c4_query_n']} 行无一 simple）"
    elif evaluable and all(
        (p["second_cov"] or 0.0) >= (p["first_cov"] or 0.0) for p in evaluable
    ) and simple_rate is not None and simple_rate > 0:
        reuse_state = VERDICT_HOLD
        reuse_why = (
            f"{len(evaluable)}/{len(pair_rows)} 个簇内重复对可评且第二次 judge cov 全部 ≥ 第一次，"
            f"simple 路由率 = {simple_rate:.3f} > 0"
        )
    else:
        reuse_state = VERDICT_UNDETERMINED
        reuse_why = "重复对 judge cov 缺失（无可评对）且无法确认 simple 路由 >0"

    # ---- 演化价值 ---------------------------------------------------------------
    temporal: Mapping[str, Mapping[str, Any]] = stats["temporal"]
    t4, t2, t3 = temporal.get("c4"), temporal.get("c2"), temporal.get("c3")
    if not stats["temporal_qids"]:
        evo_state, evo_why = VERDICT_UNDETERMINED, "题集无 temporal 题（qtype 缺失）"
    elif t4 is None or t4.get("n", 0) == 0:
        evo_state, evo_why = VERDICT_UNDETERMINED, "c4 无 update 后时效题行"
    elif t2 is None or t3 is None or (t2["cov"] is None and t2["em"] is None):
        evo_state, evo_why = VERDICT_UNDETERMINED, "c2/c3 无 update 后时效题行——无法对比"
    else:
        adv_cov = (
            t4["cov"] is not None and t2["cov"] is not None and t3["cov"] is not None
            and t4["cov"] > t2["cov"] and t4["cov"] > t3["cov"]
        )
        adv_em = (
            t4["em"] is not None and t2["em"] is not None and t3["em"] is not None
            and t4["em"] > t2["em"] and t4["em"] > t3["em"]
        )
        dominated = (
            t4["cov"] is not None and t2["cov"] is not None and t3["cov"] is not None
            and t4["em"] is not None and t2["em"] is not None and t3["em"] is not None
            and t4["cov"] <= t2["cov"] and t4["cov"] <= t3["cov"]
            and t4["em"] <= t2["em"] and t4["em"] <= t3["em"]
        )
        if adv_cov or adv_em:
            metric = "judge cov" if adv_cov else "EM"
            evo_state = VERDICT_HOLD
            evo_why = (
                f"时效题 {metric} 均值 c4={_fmt(t4['cov'] if adv_cov else t4['em'])} "
                f"> c2={_fmt(t2['cov'] if adv_cov else t2['em'])} / "
                f"c3={_fmt(t3['cov'] if adv_cov else t3['em'])}（对两个 RAG 条件均占优）"
            )
        elif dominated:
            evo_state = VERDICT_FALSIFIED
            evo_why = (
                f"时效题 c4 对 c2/c3 全不占优（cov {_fmt(t4['cov'])} vs "
                f"{_fmt(t2['cov'])}/{_fmt(t3['cov'])}；EM {_fmt(t4['em'])} vs "
                f"{_fmt(t2['em'])}/{_fmt(t3['em'])}）——supersede 未转化为答案正确性"
            )
        else:
            evo_state = VERDICT_UNDETERMINED
            evo_why = "时效题均值缺失或对比结果混合（优于一个条件、不优于另一个）"

    # ---- 经济性（引用复用/演化的裁决：证伪条件=无任何质量维度优势）----------------
    c4_total, c2_total = total_all("c4"), total_all("c2")
    cov4 = stats["cov_answerable"].get("c4", {})
    cov2 = stats["cov_answerable"].get("c2", {})
    cov_diff = (
        cov4["cov"] - cov2["cov"]
        if cov4.get("cov") is not None and cov2.get("cov") is not None
        else None
    )
    if c4_total is None or c2_total is None:
        eco_state, eco_why = VERDICT_UNDETERMINED, "c4 或 c2 无行——无法对比累计成本"
    elif c4_total <= c2_total:
        eco_state = VERDICT_HOLD
        eco_why = f"c4 全口径累计 {c4_total} tok ≤ c2 同期 {c2_total} tok（≤22 问内成立）"
    elif cov_diff is not None and cov_diff >= 0.5:
        eco_state = VERDICT_HOLD
        eco_why = (
            f"成本路径不成立（c4 {c4_total} > c2 {c2_total} tok），但可答题 judge cov "
            f"均值差（c4−c2）= {cov_diff:+.2f} ≥ +0.5 → 按「judge cov 优势 ≥+0.5」成立"
        )
    elif reuse_state == VERDICT_FALSIFIED and evo_state == VERDICT_FALSIFIED:
        eco_state = VERDICT_FALSIFIED
        eco_why = (
            f"全程 c4 累计 {c4_total} > c2 {c2_total} tok 且 cov 均值差 "
            f"{_fmt(cov_diff, 2) if cov_diff is not None else '—'} < +0.5，复用质量与演化价值"
            "双双证伪（无任何质量维度优势）"
        )
    else:
        eco_state = VERDICT_UNDETERMINED
        eco_why = (
            f"c4 累计 {c4_total} > c2 {c2_total} tok 且 cov 均值差未达 +0.5，"
            "但质量维度（复用/演化）存在未证伪项——「无任何质量维度优势」无法确认"
        )

    # ---- 反幻觉（测量项：judge cov=1 即 RQ028 编造信号）--------------------------
    c4_unans: list[Mapping[str, Any]] = stats["unanswerable"].get("c4", [])
    covs = [u["cov"] for u in c4_unans if u["cov"] is not None]
    if not c4_unans or not covs:
        anti_state = VERDICT_UNDETERMINED
        anti_why = "c4 无答案题无行或 judge 全缺——编造信号不可测"
    elif any(cov <= 1.0 for cov in covs):
        anti_state = VERDICT_FALSIFIED
        anti_why = (
            f"c4 无答案题 judge cov={_fmt(min(covs), 1)} ≤ 1——复现 RQ028 编造信号"
        )
    else:
        anti_state = VERDICT_NOT_FALSIFIED
        anti_why = (
            f"c4 无答案题 judge cov={_fmt(min(covs), 1)}（全部 >1，无编造信号）、"
            f"refusal 拒答 {sum(1 for u in c4_unans if u['refusal'])}/{len(c4_unans)} 行"
        )

    return {
        "经济性": {"state": eco_state, "why": eco_why},
        "复用质量": {"state": reuse_state, "why": reuse_why},
        "演化价值": {"state": evo_state, "why": evo_why},
        "反幻觉": {"state": anti_state, "why": anti_why},
    }


# ---- 渲染 ---------------------------------------------------------------------


def _render_header(
    results_dir: Path,
    manifest: Mapping[str, Any],
    stats: Mapping[str, Any],
    generated_at: str,
) -> list[str]:
    """头部：生成时间 / results 目录 / config 摘要 / 预注册判定四条原文（逐字）。"""
    event_count = manifest.get("event_count")
    query_n = sum(1 for e in stats["curve"] if e["kind"] == "query")
    update_cell = f"e{stats['update_id']}" if stats["update_id"] is not None else "—"
    lines = [
        f"# 复用序列实验报告：{results_dir.name}",
        "",
        f"- 生成时间：{generated_at}",
        f"- results 目录：`{results_dir.resolve()}`",
        f"- provider 模式：{_cell(manifest.get('provider_mode')) or '—'}",
        f"- 模型（strong / cheap / judge）：{_model_line(manifest)}",
        f"- 事件数：{_cell(event_count if event_count is not None else len(stats['curve']))}"
        f"（query {query_n} 个；update 事件 {update_cell}）",
        f"- 题集：`{_cell(manifest.get('qa_path')) or '—'}`"
        f"（sha256：`{_cell(manifest.get('qa_sha256')) or '—'}`）",
        f"- spec：`{SPEC_PATH}` §4（预注册判定与口径以 spec 为权威）",
    ]
    notes = manifest.get("condition_notes")
    if isinstance(notes, Mapping) and notes:
        lines.append("")
        lines.append("条件口径：")
        lines.extend(f"- {cond}：{_cell(note)}" for cond, note in notes.items())
    lines.extend(
        [
            "",
            "## 预注册判定（manifest 原文，跑前写死）",
            "",
            "以下四条逐字引用 manifest.pre_registered（run_sequence 落盘，跑前写死——"
            "本报告「预注册逐条裁决」一节对表裁决，实现不得弱化或改动）：",
            "",
            "| 维度 | 成立需要 | 证伪条件 |",
            "|---|---|---|",
        ]
    )
    for item in manifest.get("pre_registered") or []:
        if isinstance(item, Mapping):
            lines.append(
                f"| {_cell(item.get('dimension', '—'))} | {_cell(item.get('成立需要', '—'))} "
                f"| {_cell(item.get('证伪条件', '—'))} |"
            )
    lines.append("")
    return lines


def _render_curve(results_dir: Path, stats: Mapping[str, Any]) -> list[str]:
    """累计成本曲线：条件总账 + 逐事件累计表（judge 成本单列）。"""
    conditions: list[str] = stats["conditions"]
    totals: Mapping[str, Mapping[str, Any]] = stats["totals"]

    lines = [
        "## 累计成本曲线",
        "",
        f"条件总账（主 in_tok 含 ingest/update 事件行；judge 成本按 `{JUDGE_SUFFIX}` "
        "trace 从两类 tokens.jsonl 聚合——loop 行的 wiki_root 记账与 RAG 记账 "
        "manifest.rag_tokens_path，**不在主 in_tok 内**）：",
        "",
        "| 条件 | 行数 | 主 in_tok | judge in_tok | 全口径（主+judge） | 主 out_tok "
        "| latency p50 (ms) | latency p95 (ms) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for cond in conditions:
        acc = totals[cond]
        lat = acc["latencies"]
        lines.append(
            f"| {_cell(cond)} | {acc['rows']} | {acc['main_in']} | {acc['judge_in']} "
            f"| {acc['main_in'] + acc['judge_in']} | {acc['main_out']} "
            f"| {percentile(lat, 50):.0f} | {percentile(lat, 95):.0f} |"
        )
    lines.extend(
        [
            "",
            "逐事件累计（行=事件序；c4 策略=query 事件的 P3 policy_mode 标注；"
            "c1 在 ingest/update 事件无行、累计不变）：",
            "",
        ]
    )
    header = ["事件"] + (["c4 策略"] if "c4" in conditions else [])
    header += [f"{c} 累计 in" for c in conditions] + [f"{c} judge" for c in conditions]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for entry in stats["curve"]:
        cells = [_cell(_event_label(entry))]
        if "c4" in conditions:
            cells.append(_cell(entry["policy"]) if entry["policy"] else "—")
        cells.extend(str(entry["cum"][c]["main"]) for c in conditions)
        cells.extend(str(entry["cum"][c]["judge"]) for c in conditions)
        lines.append("| " + " | ".join(cells) + " |")
    final = stats["curve"][-1]["cum"] if stats["curve"] else {c: {"main": 0, "judge": 0} for c in conditions}
    cells = ["合计（全程）"] + (["—"] if "c4" in conditions else [])
    cells += [str(final[c]["main"]) for c in conditions]
    cells += [str(final[c]["judge"]) for c in conditions]
    lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _render_amortization(stats: Mapping[str, Any]) -> list[str]:
    """摊销拐点：build_cost、交叉点或"全程未交叉"、第 N 次查询后的摊销值表。"""
    amort = amortization(stats)
    build_cost: int = stats["build_cost"]
    query_n: int = stats["c4_query_n"]
    final_c4 = next(
        (_all_in(e["cum"], "c4") for e in reversed(stats["curve"]) if e["kind"] == "query"),
        None,
    )
    lines = [
        "## 摊销拐点",
        "",
        f"- c4 study 构建成本（ingest/update 行主 trace in_tok 合计）：**{build_cost} tok**；"
        f"reuse_count = query 事件数 {query_n}",
        "- 预注册摊销口径 `build_cost/reuse_count + query_cost`：第 N 次查询后 "
        f"c4 摊销/问 =（build_cost {build_cost} + c4 query 主累计 + c4 judge 累计）/ N"
        "（全口径含 judge，数值上等于 c4 全口径累计 / N；c2/c3/c1 无构建成本，均值=累计/N）",
        "- 交叉点（全口径累计，judge 计入）：",
    ]
    for target, name in (("c2", "c2（Vector RAG）"), ("c1", "c1（无 Memory）")):
        cross = amort["crossings"][target]
        if cross is not None:
            lines.append(
                f"  - c4 vs {name}：第 {cross['n']} 次查询（{cross['label']}）c4 全口径累计"
                f"首次 ≤ {target}（{cross['c4']} ≤ {cross['target']} tok）"
            )
        elif _all_in(stats["curve"][-1]["cum"], target) is not None and final_c4 is not None:
            final_target = _all_in(stats["curve"][-1]["cum"], target)
            lines.append(
                f"  - c4 vs {name}：**全程未交叉**（第 {query_n} 次查询后 c4 全口径累计 "
                f"{final_c4} tok > {target} {final_target} tok）"
            )
        else:
            lines.append(f"  - c4 vs {name}：结果中无 {target} 行——无法对比")
    lines.extend(
        [
            "",
            "| 第 N 次查询 | qid | 事件 | c4 摊销/问 | c2 累计/问 | c3 累计/问 | c1 累计/问 |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for row in amort["rows"]:
        lines.append(
            f"| {row['n']} | {_cell(row['qid'])} | {_cell(row['label'])} "
            f"| {_fmt(row['c4'], 1)} | {_fmt(row['c2'], 1)} | {_fmt(row['c3'], 1)} "
            f"| {_fmt(row['c1'], 1)} |"
        )
    lines.append("")
    return lines


def _prereg_lines(manifest: Mapping[str, Any], dimension: str) -> list[str]:
    """预注册原文引用行（裁决小节开头，逐字）。"""
    for item in manifest.get("pre_registered") or []:
        if isinstance(item, Mapping) and str(item.get("dimension")) == dimension:
            return [
                f"- 预注册：成立需要「{_cell(item.get('成立需要', '—'))}」；"
                f"证伪条件「{_cell(item.get('证伪条件', '—'))}」。"
            ]
    return [f"- 预注册：manifest.pre_registered 缺「{dimension}」条目。"]


def _render_verdicts(
    manifest: Mapping[str, Any], stats: Mapping[str, Any]
) -> list[str]:
    """预注册逐条裁决：四条小节（每条 裁决 + 预注册原文 + 数据引用）+ 总结论。"""
    verdicts = compute_verdicts(stats)
    update_cell = f"e{stats['update_id']}" if stats["update_id"] is not None else "—"
    lines = [
        "## 预注册逐条裁决",
        "",
        "对 spec §4 四条预注册判定逐条机械化裁决（规则：证据缺失给「不可判定」，"
        "绝不编造结论；每条均带数据引用）。",
        "",
    ]

    # ---- 经济性 ------------------------------------------------------------------
    eco = verdicts["经济性"]
    cov4 = stats["cov_answerable"].get("c4", {})
    cov2 = stats["cov_answerable"].get("c2", {})
    totals = stats["totals"]
    lines.extend(
        [
            "### 预注册判定·经济性",
            "",
            f"**裁决：{eco['state']}**",
            "",
            *_prereg_lines(manifest, "经济性"),
            f"- 数据引用：c4 全程累计（主 {totals['c4']['main_in']} + judge "
            f"{totals['c4']['judge_in']}）= {totals['c4']['main_in'] + totals['c4']['judge_in']} tok，"
            f"c2 同期 = {totals['c2']['main_in'] + totals['c2']['judge_in']} tok"
            f"（主 {totals['c2']['main_in']} + judge {totals['c2']['judge_in']}）——曲线见上节。",
            f"- 数据引用：可答题（排除无答案题）judge cov 均值 c4={_fmt(cov4.get('cov'))}"
            f"（n={cov4.get('cov_n', 0)}）vs c2={_fmt(cov2.get('cov'))}"
            f"（n={cov2.get('cov_n', 0)}）——均值差见裁决理由。",
            f"- 裁决理由：{eco['why']}",
            "",
        ]
    )

    # ---- 复用质量 ----------------------------------------------------------------
    reuse = verdicts["复用质量"]
    lines.extend(
        [
            "### 预注册判定·复用质量",
            "",
            f"**裁决：{reuse['state']}**",
            "",
            *_prereg_lines(manifest, "复用质量"),
            f"簇内重复对（同 qid 两次查询都在 update 事件 {update_cell} 之前，"
            f"n={len(stats['intra'])}）——预注册只按这些对判定：",
            "",
            "| qid | 第 1 次 | judge cov | 第 2 次 | judge cov | 第二次 ≥ 第一次 |",
            "|---|---|---|---|---|---|",
        ]
    )
    for pair in stats["intra"]:
        first = _judge_cov(_row_at(stats["rows"], "c4", pair["first"]))
        second = _judge_cov(_row_at(stats["rows"], "c4", pair["second"]))
        holds = "是" if (first is not None and second is not None and second >= first) else "否"
        lines.append(
            f"| {_cell(pair['qid'])} | e{pair['first']} | {_fmt(first, 1)} "
            f"| e{pair['second']} | {_fmt(second, 1)} | {holds} |"
        )
    counts: Counter = stats["policy_counts"]
    ordered = ["simple", "update", "deep", "无 policy_mode"]
    stat_cell = "、".join(f"{key}×{counts.get(key, 0)}" for key in ordered)
    rate = stats["simple_rate"]
    rate_cell = (
        f"{counts.get('simple', 0)}/{stats['c4_query_n']} = {rate:.3f}" if rate is not None else "—"
    )
    lines.extend(
        [
            "",
            f"c4 query 路由统计（policy_mode）：{stat_cell} → **simple 路由率 = {rate_cell}**。",
            (
                "排除披露（straddle 更新后重复对，一前一后跨 update，属演化语义、"
                "不计入复用摊销判定）："
                + "、".join(
                    f"{p['qid']}（e{p['first']} → e{p['second']}）" for p in stats["straddle"]
                )
                if stats["straddle"]
                else "排除披露（straddle）：无。"
            ),
            f"- 裁决理由：{reuse['why']}",
            "",
        ]
    )

    # ---- 演化价值 ----------------------------------------------------------------
    evo = verdicts["演化价值"]
    temporal: Mapping[str, Mapping[str, Any]] = stats["temporal"]
    lines.extend(
        [
            "### 预注册判定·演化价值",
            "",
            f"**裁决：{evo['state']}**",
            "",
            *_prereg_lines(manifest, "演化价值"),
            f"时效题（qtype=temporal、update（{update_cell}）后、gold_source=base，"
            f"n={len(stats['temporal_qids'])} 题："
            f"{'、'.join(sorted(stats['temporal_qids'])) or '—'}）：",
            "",
            "| 条件 | 行数 | EM 均值 | judge cov 均值 |",
            "|---|---|---|---|",
        ]
    )
    for cond in ("c4", "c2", "c3"):
        acc = temporal.get(cond)
        if acc is None:
            lines.append(f"| {cond} | 0 | — | — |")
            continue
        lines.append(
            f"| {cond} | {max(acc['em_n'], acc['cov_n'])} | {_fmt(acc['em'])} "
            f"| {_fmt(acc['cov'], 1)} |"
        )
    lines.extend(["", "演化正确性单列（gold_source=override 行，v2 口径 diff-gold 命中）：", ""])
    if stats["override_rows"]:
        lines.append("| qid | 事件 | 条件 | gold_source | EM | judge cov |")
        lines.append("|---|---|---|---|---|---|")
        for row in stats["override_rows"]:
            lines.append(
                f"| {_cell(row.get('qid') or '—')} | e{_eid(row)} "
                f"| {_cell(row.get('condition') or '—')} | override "
                f"| {_fmt(_numeric(row.get('em')))} | {_fmt(_judge_cov(row), 1)} |"
            )
    else:
        lines.append("无 override 行（本题集无演化正确性 diff-gold 测量）。")
    lines.extend(["", f"- 裁决理由：{evo['why']}", ""])

    # ---- 反幻觉（测量项）----------------------------------------------------------
    anti = verdicts["反幻觉"]
    lines.extend(
        [
            "### 预注册判定·反幻觉",
            "",
            f"**裁决：{anti['state']}**",
            "",
            *_prereg_lines(manifest, "反幻觉"),
            "无答案题逐条件测量（RQ028 编造基线：judge cov=1 即编造信号；"
            "拒答判定以 judge 为准口径）：",
            "",
            "| qid | 条件 | judge cov | refusal |",
            "|---|---|---|---|",
        ]
    )
    for cond in ("c4", "c2", "c3", "c1"):
        for entry in stats["unanswerable"].get(cond, []):
            lines.append(
                f"| {_cell(entry['qid'])} | {cond} | {_fmt(entry['cov'], 1)} "
                f"| {'是' if entry['refusal'] else '否'} |"
            )
    lines.extend(["", f"- 裁决理由：{anti['why']}", ""])

    # ---- 总结论（spec §4：经济性与演化价值双双证伪 → P4 决策依据）------------------
    if eco["state"] == VERDICT_FALSIFIED and evo["state"] == VERDICT_FALSIFIED:
        lines.extend(
            [
                "**总结论**：经济性与演化价值**双双证伪** → 记忆架构在当前形态下"
                "未证明存在 RAG 之外的价值（P4 决策依据，spec §4）。",
                "",
            ]
        )
    else:
        lines.extend(
            [
                f"**总结论**：经济性={eco['state']}、演化价值={evo['state']}——"
                "未双双证伪，不触发 spec §4「未证明价值」结论条款。",
                "",
            ]
        )
    return lines


def _render_credentials(
    results_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> list[str]:
    """文末凭证声明：原始凭证路径 + sha256 + judge 聚合来源 + 可重算声明。"""
    events_sha = hashlib.sha256((results_dir / "events.jsonl").read_bytes()).hexdigest()
    wiki_roots = sorted({str(r["wiki_root"]) for r in rows if r.get("wiki_root")})
    lines = [
        "## 凭证声明",
        "",
        "本报告是聚合视图，**所有结论可由原始凭证重算**：",
        "",
        f"- events.jsonl（每条件×事件一行）：`{results_dir / 'events.jsonl'}`"
        f"（报告生成时 sha256：`{events_sha}`）",
        f"- manifest.json（run provenance + pre_registered 原文）：`{results_dir / 'manifest.json'}`"
        f"（题集 qa_sha256：`{_cell(manifest.get('qa_sha256')) or '—'}`）",
        f"- RAG 条件记账（主 trace + `-judge`）：`{_cell(manifest.get('rag_tokens_path')) or '—'}`",
    ]
    if wiki_roots:
        lines.append(
            "- loop 条件（c1/c4）逐 run 记账（主 trace + `-judge`）："
            + "、".join(f"`{Path(root) / 'tokens.jsonl'}`" for root in wiki_roots)
        )
    lines.extend(
        [
            "",
            "聚合口径：mean/percentile 复用 `evals.metrics`；judge 成本复用 "
            "`sum_tokens_from_jsonl`（trace 后缀 `-judge`）；单元格净化与 report_eval "
            "同款（竖线换全角斜杠、空白折行）。",
            "",
        ]
    )
    return lines


def build_report(
    results_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    *,
    qa_types: Mapping[str, str],
    generated_at: str | None = None,
) -> str:
    """整份报告 markdown：头部 → 累计成本曲线 → 摊销拐点 → 预注册逐条裁决 → 凭证。"""
    generated_at = generated_at or datetime.now(UTC).isoformat(timespec="seconds")
    stats = build_stats(rows, manifest, results_dir, qa_types)
    lines = _render_header(results_dir, manifest, stats, generated_at)
    lines += _render_curve(results_dir, stats)
    lines += _render_amortization(stats)
    lines += _render_verdicts(manifest, stats)
    lines += _render_credentials(results_dir, rows, manifest)
    return "\n".join(lines) + "\n"


# ---- 入口 ---------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="复用序列实验报告（累计成本曲线 + 摊销拐点 + 预注册逐条裁决）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results", required=True,
        help="run_sequence 产物目录（含 events.jsonl 与 manifest.json）",
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
    events_path = results_dir / "events.jsonl"
    manifest_path = results_dir / "manifest.json"

    if not events_path.is_file():
        print(
            f"✗ 找不到 {events_path}——--results 应指向 run_sequence 产物目录"
            "（含 events.jsonl 与 manifest.json）"
        )
        return EXIT_BAD_INPUT
    if not manifest_path.is_file():
        print(f"✗ 找不到 {manifest_path}——manifest.json（含预注册判定原文）是必要输入")
        return EXIT_BAD_INPUT
    try:
        rows = load_rows(events_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("manifest.json 顶层不是 JSON 对象")
        qa_types = load_qa_types(manifest)
    except ValueError as exc:  # json.JSONDecodeError 是其子类
        print(f"✗ 凭证文件损坏或缺失，拒绝聚合：{exc}")
        return EXIT_BAD_INPUT
    if not rows:
        print(f"✗ {events_path} 为空（0 行），无法聚合报告")
        return EXIT_BAD_INPUT

    report = build_report(results_dir, rows, manifest, qa_types=qa_types)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{results_dir.name}.md"
    out_path.write_text(report, encoding="utf-8")

    print(f"✓ 报告已写入 {out_path}")
    verdicts = compute_verdicts(build_stats(rows, manifest, results_dir, qa_types))
    for dimension, verdict in verdicts.items():
        print(f"  {dimension}：{verdict['state']}")
    print("注：预注册判定逐条裁决见报告；小样本评测，不构成收益结论。")
    return EXIT_PASS


if __name__ == "__main__":
    raise SystemExit(main())
