"""评测题集（QA JSONL）的加载与 schema 校验。

评测支线（PLAN §7.1）的数据契约：

- 每行一个 JSON 对象，字段：qid / question / qtype / gold_points / entities / notes；
- qtype 白名单四种：single_hop（单跳）、multi_hop（多跳）、temporal（时效）、
  unanswerable（无答案）；
- gold_points 是可判定的参考要点列表——答句包含要点即可判对，不要求逐字；
  无答案题的 gold_points 必须为空数组；
- entities 记录题目涉及的实体，多跳题在此填跨文档串联实体；
- notes 备注：无答案题注明"语料不含此信息"，时效题注明依赖的 v1/v2 时效对。

校验失败的行抛 ``ValueError``，错误信息带物理行号（含空行计数），便于定位坏行；
多余的 JSON 键一律忽略，保证题集向后兼容地加字段。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

#: 题型白名单（MVP 口径：单跳 / 多跳 / 时效 / 无答案）
QA_TYPES: frozenset[str] = frozenset(
    {"single_hop", "multi_hop", "temporal", "unanswerable"}
)

#: 必填字段（notes 可省略，默认空串）
_REQUIRED_FIELDS: tuple[str, ...] = ("qid", "question", "qtype", "gold_points", "entities")


@dataclass(frozen=True)
class QaItem:
    """一道评测题。

    - qid：全局唯一的问题编号（如 Q001）；
    - question：问题原文；
    - qtype：题型，取值见 :data:`QA_TYPES`；
    - gold_points：参考要点（答句包含要点即可判对，不必逐字）；
    - entities：题目涉及的实体（多跳题填串联实体，至少 2 个）；
    - notes：备注（无答案题注明无答案原因，时效题注明依赖的 v1/v2 对）。
    """

    qid: str
    question: str
    qtype: str
    gold_points: list[str]
    entities: list[str]
    notes: str = ""


def _fail(lineno: int, reason: str) -> ValueError:
    """构造带物理行号的校验错误。"""
    return ValueError(f"第 {lineno} 行：{reason}")


def _check_str_field(lineno: int, obj: dict, field: str) -> str:
    """校验非空字符串字段，返回去空白后的值。"""
    value = obj[field]
    if not isinstance(value, str) or not value.strip():
        raise _fail(lineno, f"{field} 必须是非空字符串")
    return value


def _check_str_list(
    lineno: int, obj: dict, field: str, *, allow_empty: bool
) -> list[str]:
    """校验字符串列表字段：类型必须为 list，元素为非空字符串。"""
    value = obj[field]
    if not isinstance(value, list):
        raise _fail(lineno, f"{field} 必须是字符串列表")
    if not allow_empty and not value:
        raise _fail(lineno, f"{field} 不能为空列表")
    for element in value:
        if not isinstance(element, str) or not element.strip():
            raise _fail(lineno, f"{field} 的每个元素都必须是非空字符串")
    return value


def _parse_line(lineno: int, raw: str, seen: dict[str, int]) -> QaItem:
    """解析并校验单行 JSONL，返回 :class:`QaItem`。"""
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise _fail(lineno, f"不是合法 JSON（{exc.msg}）") from exc
    if not isinstance(obj, dict):
        raise _fail(lineno, "每行必须是一个 JSON 对象")

    missing = [name for name in _REQUIRED_FIELDS if name not in obj]
    if missing:
        raise _fail(lineno, f"缺少字段：{'、'.join(missing)}")
    # 多余 JSON 键按契约忽略，不做检查。

    qid = _check_str_field(lineno, obj, "qid")
    if qid in seen:
        raise _fail(lineno, f"qid 重复：{qid}（首次出现在第 {seen[qid]} 行）")
    seen[qid] = lineno

    question = _check_str_field(lineno, obj, "question")

    qtype = _check_str_field(lineno, obj, "qtype")
    if qtype not in QA_TYPES:
        allowed = "、".join(sorted(QA_TYPES))
        raise _fail(lineno, f"未知 qtype：{qtype}（允许值：{allowed}）")

    gold_points = _check_str_list(lineno, obj, "gold_points", allow_empty=True)
    if qtype == "unanswerable":
        # 无答案题不设参考要点：语料明确不含答案，必须留空数组。
        if gold_points:
            raise _fail(lineno, "unanswerable 题的 gold_points 必须为空数组")
    elif not 2 <= len(gold_points) <= 4:
        # 本题集的内容硬口径：可判定要点 2–4 条。
        raise _fail(lineno, f"gold_points 需要 2–4 条要点，实际 {len(gold_points)} 条")

    entities = _check_str_list(lineno, obj, "entities", allow_empty=False)

    notes = obj.get("notes", "")
    if not isinstance(notes, str):
        raise _fail(lineno, "notes 必须是字符串")

    return QaItem(
        qid=qid,
        question=question,
        qtype=qtype,
        gold_points=gold_points,
        entities=entities,
        notes=notes,
    )


def load_qa(path: str | Path) -> list[QaItem]:
    """逐行加载并校验题集 JSONL，返回 :class:`QaItem` 列表。

    - 空行（纯空白行）跳过，但计入物理行号；
    - 任一行校验失败即抛 ``ValueError``，错误信息带物理行号。
    """
    text = Path(path).read_text(encoding="utf-8")
    items: list[QaItem] = []
    seen: dict[str, int] = {}
    for lineno, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        items.append(_parse_line(lineno, raw, seen))
    return items


def qa_type_counts(items: list[QaItem]) -> dict[str, int]:
    """按题型统计题数；四种题型键齐全（未出现的题型计 0）。"""
    counts = {qtype: 0 for qtype in sorted(QA_TYPES)}
    for item in items:
        counts[item.qtype] += 1
    return counts
