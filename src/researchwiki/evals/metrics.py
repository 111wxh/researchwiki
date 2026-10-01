"""确定性指标：答案归一化、要点级 EM、拒答启发式与分组聚合。

评测支线（PLAN §7.3）的打分口径——全部是纯函数，零网络、零模型调用，
供 Task 6 的 runner 产数、Task 7 的报告消费。所有规则写死在本模块，
同一输入永远得到同一输出，保证评测可复现、可单测。

归一化规则（:func:`normalize_answer`，judge/报告的口径依据）：

1. NFKC 兼容归一：全角字母/数字/标点折算为半角（``ＡＢＣ``→``ABC``、
   ``１．２``→``1.2``）；
2. 英文统一小写（汉字不受影响）；
3. 去中英文标点：只保留 Unicode 字母（含汉字）、数字、下划线与 ``%``，
   其余标点/符号一律删除（替换为空串，不引入空格）——两侧按同一规则归一，
   等值内符号（百分号等）得以保留：``93.7%``→``937%``、``15,600``→``15600``；
4. 空白全部删除：中文排版空格无语义（"重试 3 次"与"重试3次"同义），
   删除后要点命中对空格排版不敏感——"合并空白"在本模块取最强形式
   （合并为 0 个空白），使 gold 要点与模型答句的空格差异不致漏判。

要点命中（:func:`point_hit`）：归一后 gold_point 是归一后 answer 的子串即命中；
gold 归一后为空（空串/纯标点/纯空白）一律不命中，避免空要点虚增 EM。

已知局限（备案）：子串包含语义下，过短的要点可能被更长的数字串意外包含
（如要点 ``78%`` 命中回答里的 ``178%``）——要点应写成带上下文的短语。

拒答启发式（:func:`refusal_detected`）：词典 :data:`REFUSAL_PHRASES` 命中
**且** 归一后长度小于 :data:`REFUSAL_MAX_LEN`（80 字）两条件同时成立才算拒答，
任一不满足即 False——长答案即使带免责声明也按正常回答计。

聚合（:func:`aggregate_rows`）的各字段缺失语义见其 docstring。
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

#: 拒答词典（确定性启发式短语表；模块常量，可直接单测与扩充）
REFUSAL_PHRASES: tuple[str, ...] = (
    "无法确定",
    "无法回答",
    "我不知道",
    "资料未提及",
    "没有找到",
    "不确定",
)

#: 拒答判定的长度上限（归一后字符数，严格小于才算短答案）
REFUSAL_MAX_LEN: int = 80

#: 归一化保留字符：Unicode 字母（含汉字）、数字、下划线、``%``；其余（含空白）全删
_KEEP_PATTERN = re.compile(r"[^\w%]+")


def normalize_answer(text: str) -> str:
    """按模块 docstring 的四步规则归一化答案文本。

    步骤：NFKC（全角→半角）→ 小写 → 删除保留集（字母/数字/下划线/``%``）
    以外的全部标点与符号 → 删除全部空白。确定性纯函数。
    """
    folded = unicodedata.normalize("NFKC", text).lower()
    return _KEEP_PATTERN.sub("", folded)


def point_hit(answer: str, gold_point: str) -> bool:
    """要点级命中：归一后 gold_point 是归一后 answer 的子串。

    gold 归一后为空（空串/纯标点/纯空白）返回 False，防止空要点虚增 EM。
    """
    norm_gold = normalize_answer(gold_point)
    if not norm_gold:
        return False
    return norm_gold in normalize_answer(answer)


def normalized_em(answer: str, gold_points: Sequence[str]) -> float:
    """要点级 EM：命中要点数 / 要点总数；空要点列表返回 0.0。

    每个要点独立按 :func:`point_hit` 判定（包含即中，不要求逐字），
    要点级判分比整句 EM 更稳。
    """
    if not gold_points:
        return 0.0
    hits = sum(1 for gold in gold_points if point_hit(answer, gold))
    return hits / len(gold_points)


def refusal_detected(answer: str) -> bool:
    """确定性拒答启发式：命中 :data:`REFUSAL_PHRASES` 且归一后长度 < 80 字。

    在归一化后的答案上匹配词典（短语本身不含标点，归一无损）；
    两条件同时成立才判拒答，任一不满足即 False。
    """
    normalized = normalize_answer(answer)
    if len(normalized) >= REFUSAL_MAX_LEN:
        return False
    return any(phrase in normalized for phrase in REFUSAL_PHRASES)


def mean(values: Sequence[float]) -> float:
    """算术平均；空序列返回 0.0（与 scripts/gate_retrieval.py 同口径）。"""
    return sum(values) / len(values) if values else 0.0


def percentile(values: Sequence[float], p: float) -> float:
    """线性插值分位数（p ∈ [0,100]）；空输入返回 0.0。

    与 ``scripts/gate_retrieval.py`` 的 percentile 逐行同口径（实现一致，
    单测 ``test_percentile_matches_gate_script`` 钉住两边结果相等），
    保证评测报告的 p50/p95 与检索闸门可直接对比。
    """
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (p / 100.0) * (len(ordered) - 1)
    low, high = math.floor(rank), math.ceil(rank)
    if low == high:
        return ordered[int(rank)]
    frac = rank - low
    return ordered[low] * (1 - frac) + ordered[high] * frac


def aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float | int]]:
    """按 ``row["condition"]`` 分组聚合观测行（Task 6 产数、Task 7 报告的契约）。

    行字段约定与缺失语义（用 ``row.get()`` 容错，缺失键不抛错）：

    - ``condition``：分组键（非字符串值转 str）；缺失/None 的行无法归组，整体跳过；
    - ``em``：0–1 浮点，或缺失（键不存在 / None）——缺失行**不计入 em_mean 的
      分母**，但 ``n`` 仍计；整组无 em 值时 em_mean 为 0.0；
    - ``refusal``：bool——缺失按 False 计入 refusal_rate 的分母；
    - ``latency_ms`` / ``input_tokens`` / ``output_tokens``：数值——缺失行跳过
      该字段的统计，整组无值时输出 0.0（分位为 percentile 的空输入口径）；
    - ``fresh_search_count``：仅 loop 条件的行携带——组内**至少一行**携带时输出
      ``fresh_search_mean``（缺失行按 0 计入均值），整组都未携带则**跳过该字段**
      （输出 dict 中不含该键，便于报告区分"未测"与"测得 0"）。

    每组输出：``{"n": int, "em_mean": float, "refusal_rate": float,
    "latency_p50": float, "latency_p95": float, "input_tokens_mean": float,
    "output_tokens_mean": float[, "fresh_search_mean": float]}``；
    空输入返回空 dict。
    """
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        condition = row.get("condition")
        if condition is None:
            continue
        groups.setdefault(str(condition), []).append(row)

    result: dict[str, dict[str, float | int]] = {}
    for condition, group in groups.items():
        em_values = [float(r["em"]) for r in group if r.get("em") is not None]
        refusal_rate = mean([1.0 if r.get("refusal") else 0.0 for r in group])
        latencies = [float(r["latency_ms"]) for r in group if r.get("latency_ms") is not None]
        input_tokens = [
            float(r["input_tokens"]) for r in group if r.get("input_tokens") is not None
        ]
        output_tokens = [
            float(r["output_tokens"]) for r in group if r.get("output_tokens") is not None
        ]
        has_fresh = any("fresh_search_count" in r for r in group)

        summary: dict[str, float | int] = {
            "n": len(group),
            "em_mean": mean(em_values),
            "refusal_rate": refusal_rate,
            "latency_p50": percentile(latencies, 50.0),
            "latency_p95": percentile(latencies, 95.0),
            "input_tokens_mean": mean(input_tokens),
            "output_tokens_mean": mean(output_tokens),
        }
        if has_fresh:
            fresh = [float(r.get("fresh_search_count") or 0) for r in group]
            summary["fresh_search_mean"] = mean(fresh)
        result[condition] = summary
    return result
