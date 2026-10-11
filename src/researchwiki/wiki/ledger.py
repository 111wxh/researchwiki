"""未答问题台账（P4a Task 15）：kind=experience 的反幻觉记忆。

背景（复用序列实验的反幻觉维度**证伪**）：无答案问题上 loop 仍会编造完整报告
（SEQ016 复现 RQ028，cov=1.0；RAG 基线反而干净拒绝）。本模块把"这个问题此前已
判定证据不足"变成**可检索的记忆**：run 收尾用确定性规则（零模型调用）判定本轮
是否"没答上"，命中即写一条 kind=experience 的高 importance 台账笔记；此后同题
或相似题再到来时，Prior 阶段（wiki/prior.py）据此注入"不可编造"护栏。

判定与写入的边界（Global Constraints 的诚实口径）：

- **判定是确定性代理**：``notes_created == 0`` 且 fresh 来源数为 0 →
  ``no_notes``；fresh 来源数低于当前模式的 ``min_fresh_sources`` →
  ``below_min_fresh``。"有部分证据但仍不足"的问题不在捕获范围（SEQ016 形态
  恰好落在 no_notes，但不是所有形态都如此）——这是声明的局限，不是疏漏。
- **显式写入通道**：走 ``WikiStore.save_note(kind="experience")``——与 MCP
  memory.store 同通道，**不经 formation 判定**（formation 只约束 run 内蒸馏
  自动入库；台账是收尾期的显式审计记忆，被 formation 拒掉就失去护栏意义）。
- **幂等**：同一 trace_id（同一 run）同题只记一条——重复 record 返回既有笔记，
  不新增、不覆写。
- **处置契约**：台账是 active 记忆；再次研究后若已可答，以 supersede 了结本条
  （保留历史）——superseded / tombstone 自动退出护栏匹配（见
  ``iter_active_unanswered``）。

相似度口径（与 [verification] 注释"不是同一尺度"的告诫一致）：护栏匹配复用
``wiki/verification.py`` 的确定性 bigram 特征哈希余弦（零网络、可复算）——本
模块**只提供复用入口** ``question_similarity``，不另写一套度量；guard_similarity
等阈值都在这套（verification）尺度上标定。
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from researchwiki.wiki.store import Note, WikiStore
from researchwiki.wiki.verification import (
    DEFAULT_SIMILARITY_DIM,
    VerificationSettings,
    token_similarity,
)

# ---- 契约常量 ----------------------------------------------------------------

# extra.ledger 的标记值（护栏匹配按它识别台账笔记）
LEDGER_MARKER = "unanswered_question"
# 触发原因枚举（extra.reason；写入前由调用方经 decide_unanswered_reason 产生）
REASON_NO_NOTES = "no_notes"
REASON_BELOW_MIN_FRESH = "below_min_fresh"
REASONS: tuple[str, ...] = (REASON_NO_NOTES, REASON_BELOW_MIN_FRESH)
# 反幻觉护栏是高价值记忆，importance 固定 0.8
LEDGER_IMPORTANCE = 0.8
LEDGER_TITLE = "未答问题台账"

# [unanswered] 段缺省值（与 config.toml 的注释逐字对应）
DEFAULT_GUARD_SIMILARITY = 0.4
DEFAULT_GUARD_TOP_K = 2

# 台账正文首段（写入与解析共用同一形态；解析失败 = 非台账正文，护栏跳过）。
# 问题原文夹在「」里、非贪婪匹配到"」于"——问题文本本身含「」紧跟"于"的形态
# 极罕见，遇到时该条护栏静默失效（宁可少注入，不可注入错误条目）。
_LEDGER_LINE_RE = re.compile(
    r"问题「(.+?)」于 (\d{4}-\d{2}-\d{2}) 被判定证据不足（原因 ([a-z_]+)）。"
)


# ---- 数据结构 ----------------------------------------------------------------


@dataclass(frozen=True)
class UnansweredEntry:
    """从台账笔记解析出的结构化条目（护栏匹配与注入用）。

    - ``asked_date``：判定日期（ISO 日期部分，护栏文案引用）；完整时间戳在
      笔记的 observed_at / created，正文契约只承载日期。
    """

    note_id: str
    question: str
    asked_date: str
    reason: str


@dataclass
class UnansweredSettings:
    """``[unanswered]`` 段的解析结果。

    ``enabled = false`` 是逃生阀（完全禁用：收尾零动作、Prior 零护栏）；
    ``dry_run = true`` 是影子模式（判定与留痕照跑，不写台账笔记、不注入护栏）；
    ``guard_similarity`` 是护栏触发的问题相似度（verification 尺度，校准项）；
    ``guard_top_k`` 是护栏注入条数上限（按相似度降序）。
    """

    enabled: bool = True
    dry_run: bool = False
    guard_similarity: float = DEFAULT_GUARD_SIMILARITY
    guard_top_k: int = DEFAULT_GUARD_TOP_K


def unanswered_settings(config: Mapping[str, Any] | None) -> UnansweredSettings | None:
    """解析 ``[unanswered]`` 段；**段缺失（None）返回 None = 不接线**。

    不接线时 AgentLoop 收尾零动作、Prior 零护栏（与 [formation]/[memory_update]
    的 guarded 模式逐字一致）。段存在时非法值宽容回退默认、越界值夹取（比率到
    [0.0, 1.0]、条数下限 1），不抛异常——配置错误不打断研究链路。
    """
    if not isinstance(config, Mapping):
        return None
    settings = UnansweredSettings()
    enabled = config.get("enabled")
    if enabled is not None:
        settings.enabled = bool(enabled)
    dry_run = config.get("dry_run")
    if dry_run is not None:
        settings.dry_run = bool(dry_run)
    similarity = config.get("guard_similarity")
    if similarity is not None:
        try:
            number = float(similarity)
        except (TypeError, ValueError):
            number = DEFAULT_GUARD_SIMILARITY
        if number == number:  # NaN 回退默认
            settings.guard_similarity = min(1.0, max(0.0, number))
    top_k = config.get("guard_top_k")
    if top_k is not None:
        try:
            settings.guard_top_k = max(1, int(top_k))
        except (TypeError, ValueError):
            settings.guard_top_k = DEFAULT_GUARD_TOP_K
    return settings


# ---- 触发判定（run 收尾，确定性，无模型调用）---------------------------------


def decide_unanswered_reason(
    *, notes_created: int, fresh_sources: int, min_fresh_sources: int
) -> str | None:
    """run 收尾的确定性触发判定；返回 REASONS 之一，不触发返回 None。

    - ``no_notes``：本轮 ``notes_created == 0`` 且 fresh 来源数为 0（零产出 run）；
    - ``below_min_fresh``：fresh 来源数 < 当前模式的 ``min_fresh_sources``
      （mode 不可得时按 [retrieval.deep] 段，见 AgentLoop._min_fresh_sources）；
    - 两者同时命中时 ``no_notes`` 优先（零产出是更强的"没答上"信号）；
    - 其余（有 fresh 来源且达到模式要求，或无来源但有笔记且不低于下限）不记台账。
    """
    if notes_created == 0 and fresh_sources == 0:
        return REASON_NO_NOTES
    if fresh_sources < min_fresh_sources:
        return REASON_BELOW_MIN_FRESH
    return None


# ---- 写入与遍历 --------------------------------------------------------------


def record_unanswered(
    store: WikiStore,
    *,
    question: str,
    reason: str,
    trace_id: str,
    sources_used: int,
    notes_created: int,
    asked_at: str | None = None,
) -> Note:
    """写一条 kind=experience 的未答问题台账笔记（显式写入，不经 formation）。

    - **幂等**：同 ``trace_id``（同一 run）已有 active 台账时直接返回既有笔记，
      不新增、不覆写——同一 run 内同题只记一条；
    - ``reason`` 必须是 REASONS 之一（no_notes | below_min_fresh），非法值按
      ValueError 抛出（判定函数的输出本就只有这两类，写错是调用方 bug）；
    - ``asked_at`` 缺省取当前 UTC 时刻；正文的判定日期取其前 10 位（ISO 日期）。
    """
    if reason not in REASONS:
        raise ValueError(f"reason 必须是 {REASONS} 之一，得到 {reason!r}")
    for note in iter_active_unanswered(store):
        # trace_id 是 NoteMeta 的类型化字段（来源追溯位）：round-trip 后从 meta 读；
        # extra 里的同名键只在新写的内存对象上可达，一并兜底
        recorded = str(note.meta.trace_id or note.meta.extra.get("trace_id") or "")
        if recorded == trace_id:
            return note  # 同 run 幂等：不新增、不覆写
    now = asked_at or datetime.now(UTC).isoformat(timespec="seconds")
    # 问题原文进正文一行（护栏解析按同一形态读回）；换行会破坏"一行一判定"的
    # 形态，压平成空白归一的单行
    normalized_question = " ".join(str(question).split())
    asked_date = str(now)[:10]
    body = (
        f"# {LEDGER_TITLE}\n"
        "\n"
        f"问题「{normalized_question}」于 {asked_date} 被判定证据不足（原因 {reason}）。\n"
        f"本次尝试：{sources_used} 个来源、{notes_created} 条笔记、trace {trace_id}。\n"
        "\n"
        "处置：后续相似问题注入反编造护栏；再次研究后若已可答，"
        "以 supersede 了结本条（保留历史）。\n"
    )
    return store.save_note(
        body,
        title=LEDGER_TITLE,
        kind="experience",
        importance=LEDGER_IMPORTANCE,
        # observed_at = 判定时刻（"这个问题答不上"是一条带时间的观察）
        observed_at=now,
        trace_id=trace_id,
        extra={
            "ledger": LEDGER_MARKER,
            "reason": reason,
            "trace_id": trace_id,
            "sources_used": int(sources_used),
            "notes_created": int(notes_created),
        },
    )


def iter_active_unanswered(store: WikiStore) -> list[Note]:
    """列出活跃台账：status=active 且 extra.ledger == unanswered_question。

    superseded（已被"再次研究可答"了结）与 tombstone（失效裁定审计记录）自动
    排除——它们不再是"当前仍然成立的不可编造护栏"，不该继续注入 plan 上下文。
    """
    return [
        note
        for note in store.list_notes(status="active")
        if note.meta.extra.get("ledger") == LEDGER_MARKER and not note.tombstone
    ]


def parse_unanswered_entry(note: Note) -> UnansweredEntry | None:
    """从台账笔记正文解析结构化条目；形态不符（非台账正文）返回 None。

    写入（``record_unanswered``）与解析共用同一正文契约；解析失败 = 该笔记
    不是按契约写的台账（手工改动 / 旧数据），护栏宁可跳过也不注入错误条目。
    """
    match = _LEDGER_LINE_RE.search(note.body)
    if match is None:
        return None
    return UnansweredEntry(
        note_id=note.id,
        question=match.group(1),
        asked_date=match.group(2),
        reason=match.group(3),
    )


# ---- 相似度（复用 verification 的度量，同一把尺）-----------------------------


def question_similarity(a: str, b: str, dim: int = DEFAULT_SIMILARITY_DIM) -> float:
    """问题文本相似度：verification 的确定性 bigram 特征哈希余弦（零网络、可复算）。

    刻意**委托** ``verification.token_similarity`` 而不另写度量——护栏匹配与
    记忆更新判定必须同一把尺，verification 侧口径演进（如换维度归一）时护栏
    自动跟随；``dim`` 透传其 ``similarity_dim``（缺省与 [verification] 同值 128）。
    任一侧为空串（零向量）→ 0.0，结果恒在 [0.0, 1.0]。
    """
    return token_similarity(a, b, settings=VerificationSettings(similarity_dim=dim))


__all__ = [
    "DEFAULT_GUARD_SIMILARITY",
    "DEFAULT_GUARD_TOP_K",
    "LEDGER_IMPORTANCE",
    "LEDGER_MARKER",
    "LEDGER_TITLE",
    "REASON_BELOW_MIN_FRESH",
    "REASON_NO_NOTES",
    "REASONS",
    "UnansweredEntry",
    "UnansweredSettings",
    "decide_unanswered_reason",
    "iter_active_unanswered",
    "parse_unanswered_entry",
    "question_similarity",
    "record_unanswered",
    "unanswered_settings",
]
