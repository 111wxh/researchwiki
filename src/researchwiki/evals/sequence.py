"""序列实验事件流（scenario）的 schema 与校验器（复用序列实验 task-1）。

复用序列实验（PLAN Task 1）的数据契约：四条件重放**同一条事件流**，逐事件
计量——事件流文件 ``evals/sequence/scenario.jsonl`` 每行一个 JSON 对象：

    {"event_id": 0, "kind": "ingest|query|update",
     "doc_ids": ["doc-a", ...],   # 仅 ingest / update 携带
     "qid": "SEQ001",             # 仅 query 携带
     "note": "..."}

时间线语义（spec §3）：

- ``t0`` 恰 1 个 ingest 事件：全量 t0 语料（v1 快照与无版本概念文档）；
- ``t1–t4`` 四个主题簇的 query 块（每块 4 问，其中同一 qid 出现两次 = 簇内
  重复对，两次都在 update 前，测记忆摊销）；
- ``t5`` 恰 1 个 update 事件：3 对双 SHA 时效文档的 v2 版到达（v2 替换 v1）；
- ``t6`` 时效问（gold=v2 口径）+ 无答案问 + 更新后重复问（重复对的后一次，
  qid 复用，测记忆演化）。

校验规则（:func:`load_scenario`，失败抛 ``ValueError``，错误信息带物理行号）：

1. 行级 schema：合法 JSON 对象；kind 白名单；event_id 从 0 连续递增（文件行序
   即时间线顺序）；query 必须带 qid 且不带 doc_ids；ingest/update 必须带
   doc_ids 且不带 qid；
2. query 的 qid ∈ 题集（题集经 :func:`researchwiki.evals.qa.load_qa` 加载复用）；
3. qid 恰出现 1 次或 2 次；出现 2 次必须"分居 update 两侧"（更新后重复对）
   或"同在 update 前"（簇内重复对）——同在 update 后非法；重复对的两次事件
   note 必须写明属于哪种（报告审计用）；
4. update 恰 1 个，且全部时效题（题集 qtype==temporal）的 query 事件都在
   update 之后（时效 gold 按 v2 口径，v2 未到达前提问无意义）；
5. ingest/update 的 doc_ids ⊆ 语料 manifest 的 doc_id 集；
6. 文档依赖自查：每道 query 题目的来源文档（题集 notes 的「来源：」段解析，
   无答案题豁免）必须在首个引用它的 query 事件之前由某个 ingest 或 update
   事件提供——保证 gold 要点在该时刻可被检索看到。

:func:`clusters_from_manifest` 给出语料的主题簇分组（由实现者按语料主题划定，
分组依据在 scenario.jsonl 各簇首问的 note 字段与 task 报告中披露）。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from researchwiki.evals.qa import QaItem, load_qa

#: 事件类型白名单
EVENT_KINDS: frozenset[str] = frozenset({"ingest", "query", "update"})

#: 来源段解析：notes 中的「来源：」段（顿号分隔 doc_id，句号或行尾终止）
_SOURCE_PATTERN = re.compile(r"来源：(.+)")


@dataclass(frozen=True)
class SeqEvent:
    """事件流中的一个事件。

    - event_id：时间线序号（从 0 连续递增，文件行序即时间线顺序）；
    - kind：事件类型，取值见 :data:`EVENT_KINDS`；
    - doc_ids：ingest/update 携带的文档集（query 事件为空 tuple）；
    - qid：query 携带的题目编号（其余事件为空串）；
    - note：备注（重复对的两次事件必须写明重复对类型；各簇首问写明簇分组依据）。
    """

    event_id: int
    kind: str
    doc_ids: tuple[str, ...] = ()
    qid: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        # doc_ids 归一化为 tuple：frozen dataclass 里保持真正不可变且可哈希
        object.__setattr__(self, "doc_ids", tuple(self.doc_ids))


def _fail(lineno: int, reason: str) -> ValueError:
    """构造带物理行号的校验错误。"""
    return ValueError(f"第 {lineno} 行：{reason}")


def parse_source_docs(notes: str) -> list[str]:
    """从题集 notes 的「来源：」段解析 doc_id 列表。

    - 「来源：」后到第一个句号（或行尾）为止，按顿号切分；
    - 无「来源：」段返回空列表（无答案题豁免依赖自查）。
    """
    match = _SOURCE_PATTERN.search(notes)
    if not match:
        return []
    segment = match.group(1).split("。")[0]
    return [part.strip() for part in segment.split("、") if part.strip()]


def _parse_event_line(lineno: int, raw: str) -> SeqEvent:
    """解析单行事件 JSONL（行级 schema，规则 1）。"""
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise _fail(lineno, f"不是合法 JSON（{exc.msg}）") from exc
    if not isinstance(obj, dict):
        raise _fail(lineno, "每行必须是一个 JSON 对象")

    kind = obj.get("kind")
    if kind not in EVENT_KINDS:
        allowed = "、".join(sorted(EVENT_KINDS))
        raise _fail(lineno, f"未知事件类型：{kind}（允许值：{allowed}）")

    event_id = obj.get("event_id")
    if not isinstance(event_id, int) or isinstance(event_id, bool):
        raise _fail(lineno, "event_id 必须是整数")

    doc_ids = obj.get("doc_ids") or ()
    if not isinstance(doc_ids, (list, tuple)) or any(
        not isinstance(d, str) or not d.strip() for d in doc_ids
    ):
        raise _fail(lineno, "doc_ids 必须是非空字符串列表")

    qid = obj.get("qid") or ""
    if not isinstance(qid, str):
        raise _fail(lineno, "qid 必须是字符串")

    note = obj.get("note") or ""
    if not isinstance(note, str):
        raise _fail(lineno, "note 必须是字符串")

    # kind 与携带字段的对应关系：文档集只属于 ingest/update，qid 只属于 query
    if kind == "query":
        if doc_ids:
            raise _fail(lineno, "query 事件不应携带 doc_ids（文档集只属于 ingest/update）")
        if not qid.strip():
            raise _fail(lineno, "query 事件必须携带非空 qid")
    else:
        if qid.strip():
            raise _fail(lineno, f"{kind} 事件不应携带 qid（qid 只属于 query）")
        if not doc_ids:
            raise _fail(lineno, f"{kind} 事件必须携带非空 doc_ids")

    return SeqEvent(
        event_id=event_id, kind=kind, doc_ids=tuple(doc_ids), qid=qid, note=note
    )


def load_scenario(
    path: str | Path, qa_path: str | Path, corpus_manifest_path: str | Path
) -> list[SeqEvent]:
    """加载并校验事件流，返回 :class:`SeqEvent` 列表（文件行序）。

    - ``path``：事件流 JSONL；空行跳过但计入物理行号；
    - ``qa_path``：序列题集 JSONL（复用 :func:`researchwiki.evals.qa.load_qa`
      加载与 schema 校验）；
    - ``corpus_manifest_path``：语料 manifest.json（doc_id 事实来源）；
    - 任一校验规则失败即抛 ``ValueError``，错误信息带物理行号。
    """
    qa_items: dict[str, QaItem] = {item.qid: item for item in load_qa(qa_path)}
    manifest = json.loads(Path(corpus_manifest_path).read_text(encoding="utf-8"))
    manifest_doc_ids = {str(doc["doc_id"]) for doc in manifest["docs"]}

    text = Path(path).read_text(encoding="utf-8")
    events: list[SeqEvent] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        event = _parse_event_line(lineno, raw)
        # event_id 从 0 连续递增：文件行序即时间线顺序（顺序即事件流的语义）
        if event.event_id != len(events):
            raise _fail(
                lineno,
                f"event_id 必须从 0 连续递增，期望 {len(events)}，实际 {event.event_id}",
            )
        events.append(event)

    _validate_timeline(events, qa_items, manifest_doc_ids)
    return events


def _validate_timeline(
    events: list[SeqEvent], qa_items: dict[str, QaItem], manifest_doc_ids: set[str]
) -> None:
    """跨行校验（规则 2–6）：qid 合法性与重复对结构、update 唯一性与时效题
    顺序、doc_ids ⊆ manifest、文档依赖自查。"""
    # update 恰 1 个（规则 4 前半）
    updates = [e for e in events if e.kind == "update"]
    if len(updates) != 1:
        raise ValueError(f"事件流必须恰好 1 个 update 事件，实际 {len(updates)} 个")
    update_id = updates[0].event_id

    # 规则 5：ingest/update 的 doc_ids ⊆ manifest
    for event in events:
        if event.kind == "query":
            continue
        unknown = [d for d in event.doc_ids if d not in manifest_doc_ids]
        if unknown:
            lineno = event.event_id + 1  # event_id 连续递增 ⇒ 行号 = event_id + 1
            raise _fail(
                lineno,
                f"{event.kind} 的 doc_ids 不在语料 manifest 中：{'、'.join(unknown)}",
            )

    # 规则 2 + 3：qid ∈ 题集；出现次数 ≤2；重复对结构
    occurrences: dict[str, list[SeqEvent]] = {}
    for event in events:
        if event.kind != "query":
            continue
        if event.qid not in qa_items:
            lineno = event.event_id + 1
            raise _fail(lineno, f"query 的 qid 不在题集中：{event.qid}")
        occurrences.setdefault(event.qid, []).append(event)

    for qid, evts in occurrences.items():
        if len(evts) > 2:
            first_lineno = evts[0].event_id + 1
            raise ValueError(
                f"qid {qid} 出现 {len(evts)} 次（首次在第 {first_lineno} 行）："
                "每题至多出现 2 次（簇内重复对或更新后重复对）"
            )
        if len(evts) == 2:
            before, after = evts
            straddles = before.event_id < update_id < after.event_id
            both_before = after.event_id < update_id
            if not straddles and not both_before:
                raise ValueError(
                    f"qid {qid} 出现两次但都在 update 之后（第 "
                    f"{before.event_id + 1}、{after.event_id + 1} 行）：合法形态只有"
                    "簇内重复对（同在 update 前）与更新后重复对（分居两侧）"
                )
            # 重复对类型须写进 note（报告审计用）
            for event in evts:
                if not event.note.strip():
                    lineno = event.event_id + 1
                    raise _fail(
                        lineno,
                        f"重复对 qid {qid} 的事件 note 为空：须写明"
                        "「簇内重复对（两次均在 update 前）」或「更新后重复对」",
                    )

    # 规则 4 后半：全部时效题的 query 事件都在 update 之后
    for qid, evts in occurrences.items():
        if qa_items[qid].qtype != "temporal":
            continue
        for event in evts:
            if event.event_id < update_id:
                lineno = event.event_id + 1
                raise _fail(
                    lineno,
                    f"时效题 {qid} 的 query 事件在 update（event_id={update_id}）之前："
                    "时效 gold 按 v2 口径，v2 文档未到达前提问无意义",
                )

    # 规则 6：文档依赖自查——query 的来源文档必须已被更早的 ingest/update 提供
    available: set[str] = set()
    cursor = 0
    for event in events:
        # 先推进可用文档集（ingest/update），再检查本事件（严格"之前"语义）
        if event.kind in ("ingest", "update"):
            available.update(event.doc_ids)
            cursor = event.event_id
            continue
        item = qa_items[event.qid]
        if item.qtype == "unanswerable":
            continue  # 无答案题不依赖任何文档
        sources = parse_source_docs(item.notes)
        if not sources:
            lineno = event.event_id + 1
            raise _fail(
                lineno,
                f"qid {event.qid} 的题集行缺少「来源：」段，无法做文档依赖自查"
                "（无答案题除外均须标注来源文档）",
            )
        missing = [d for d in sources if d not in available]
        if missing:
            lineno = event.event_id + 1
            raise _fail(
                lineno,
                f"query {event.qid} 的来源文档在首个引用它之前未被 ingest/update："
                f"{'、'.join(missing)}（当前可用：{sorted(available)}）",
            )


# ---------------------------------------------------------------------------
# 语料主题簇分组（实现者按语料主题划定；依据在各簇首问 note 与 task 报告披露）
# ---------------------------------------------------------------------------

#: 4 簇主题分组：簇名 → 该簇 doc_id（含 v1/v2 对，两者强制同簇）。
#: 分组依据（报告同步披露）：
#: - 发行定位与agent演进：README 层三篇——0.3.30 根/主包定位与 1.0 新 agent 架构
#:   （root-readme、libs-langchain-readme、langchain-v1-readme）；
#: - 包架构与Core基抽象：包层次概念文档 + Core README（抽象/版本策略）；
#: - RAG数据链路：切分→嵌入→向量存储→RAG 检索的数据通路四篇；
#: - 模型接口与对话上下文：工具调用/结构化输出两类模型输出接口 + 对话历史管理。
_CLUSTER_DEFS: dict[str, tuple[str, ...]] = {
    "发行定位与agent演进": (
        "root-readme",
        "root-readme@v2",
        "libs-langchain-readme",
        "libs-langchain-readme@v2",
        "langchain-v1-readme",
    ),
    "包架构与Core基抽象": (
        "concepts-architecture",
        "libs-core-readme",
        "libs-core-readme@v2",
    ),
    "RAG数据链路": (
        "concepts-text-splitters",
        "concepts-embedding-models",
        "concepts-vectorstores",
        "concepts-rag",
    ),
    "模型接口与对话上下文": (
        "concepts-tool-calling",
        "concepts-structured-outputs",
        "concepts-chat-history",
    ),
}


def clusters_from_manifest(manifest: Mapping[str, Any] | str | Path) -> dict[str, list[str]]:
    """按语料主题返回 4 簇分组：簇名 → doc_id 列表（定义序，见 :data:`_CLUSTER_DEFS`）。

    ``manifest`` 可为已加载的 dict 或 manifest.json 路径。校验：manifest 的每篇
    doc_id 恰归入一簇（未知 doc_id 抛 ``ValueError``——簇定义随语料同步维护）。
    """
    if isinstance(manifest, (str, Path)):
        manifest = json.loads(Path(manifest).read_text(encoding="utf-8"))
    manifest_doc_ids = [str(doc["doc_id"]) for doc in manifest["docs"]]

    clustered: dict[str, list[str]] = {}
    seen: dict[str, str] = {}
    for cluster_name, doc_ids in _CLUSTER_DEFS.items():
        clustered[cluster_name] = list(doc_ids)
        for doc_id in doc_ids:
            if doc_id in seen:
                raise ValueError(f"doc_id {doc_id} 同时属于多簇（{seen[doc_id]}、{cluster_name}）")
            seen[doc_id] = cluster_name

    unknown = [d for d in manifest_doc_ids if d not in seen]
    if unknown:
        raise ValueError(f"manifest 含未归簇的 doc_id：{'、'.join(unknown)}")
    return clustered


__all__ = [
    "EVENT_KINDS",
    "SeqEvent",
    "clusters_from_manifest",
    "load_scenario",
    "parse_source_docs",
]
