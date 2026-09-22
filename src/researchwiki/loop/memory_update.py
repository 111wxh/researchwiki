"""新证据 vs 旧记忆的**动作执行**（RQ2 闭环收尾，P2-E）。

``wiki/verification.py``（P2-C）只判定与建议——不改笔记、不写台账、不写盘，
"不静默覆盖"是设计约束。本模块负责把判定变成**动作**：遍历"每条命中 Prior ×
本轮证据"的比较结论，按 ``suggested_action`` 分派并留下可审计记录。判定与执行
分离是刻意的：判定零副作用、可复算；写入只在这一个模块发生。

动作分发表（逐条可测）
----------------------

- ``consistent`` → ``refresh_reviewed_at``：只刷新 ``reviewed_at=now``（只改这一个
  字段；sources 原样不动；其余 frontmatter 字段逐字段透传）。
- ``newer`` → ``supersede``：旧笔记 ``superseded_by`` 指向替代版本；**必须带来源**
  （URL **与 content_hash 都非空**），否则降级 ``open_conflict``（宁可开台账，也不造
  无证据/悬空来源的替代记忆）。替代版本优先**复用本轮入库笔记**（``evidence_notes``
  映射），无映射时才新建。
- ``more_specific`` → ``merge``：update 语义（正文追加要点 + 来源追加 + 刷新
  reviewed_at），保留规范 ID；正文由本模块拼接（见 ``_merged_body``）。无实际变更
  （正文已含该证据、来源无新增）时如实记 skipped。
- ``conflicting`` → ``open_conflict``：``store.save_conflict``，**必须带两侧证据**
  （旧 note ID + 正文摘录 + 来源；新证据正文 + 来源 + 冲突槽位取值）。
- ``uncertain`` → ``none``：不写盘，只计数（计入 skipped 桶，理由写明 uncertain）。

三条硬约束
----------

1. **不静默覆盖**：任何 supersede / merge 之前都先构造含 ``verdict`` + 关键值
   （冲突槽位取值 / 相似度）+ **证据来源 URL** 的 reason 字符串，并把它写进笔记
   extra（``supersede_reason`` / ``update_reasons``，字段名与 P2-D 的 MCP 通道
   一致）或冲突台账；旧笔记在 supersede 后按状态机原样保留（``status=superseded``
   + ``superseded_by`` 链）。
2. **幂等**：双通道键 —— ``update_key(prior, body)`` = ``{prior_id}:{sha1(正文)[:16]}``
   与 ``body_key(body)`` = ``sha1(正文)[:16]``（不带 prior 前缀，见其 docstring：
   supersede 之后 Prior 检索只回 active 的替代版本，带前缀的键在那里永远对不上，
   跨 run 的重放只能靠正文哈希键拦住）。两个键都记进被处置笔记 extra
   （``memory_update_keys`` / ``memory_update_body_keys``，各自保留最近
   ``MAX_RECORDED_KEYS`` 条）与冲突台账两侧 claim（``memory_update_key`` /
   ``memory_update_body_key``）；命中任一 → 不写盘、计入 skipped。同一批内重复的
   证据也会被去重。
3. **dry_run**：``dry_run=True`` 只返回报告，零写盘（不落笔记、不写台账、不改
   extra），但幂等与护栏判定照常执行——所以"先 dry 后真"看到的动作集合一致。
   ``[memory_update] dry_run = true`` 是生产侧的同款影子开关。

为什么不用 MCP 的 memory_* 通道写
--------------------------------

P2-D 的 ``memory_supersede`` / ``memory_update`` 带 ``source_urls`` 证据入口
（来源=提供值、按 ``(url, content_hash)`` 去重、证据空则返回 ``evidence: none``）。
本模块在 ``WikiStore`` 层复刻同一套**语义**，而不是调用 ``mcp_server``：

- 分层：``loop/*`` 至今不依赖 ``mcp_server``（依赖方向是 cli → mcp_server，
  loop ↔ wiki 平行），从 loop 反向调用会把 MCP 服务层拖进研究链路；
- 签名与依赖：本模块按简报接收 ``store``（纯 wiki 层），
  ``WikiService`` 还要 root/config/索引/线程锁与 JSON 错误协议，都是执行
  动作不需要的东西；
- 可审计性等价：来源按 (url, hash) 去重后写入替代版本、reason 落 extra、
  旧笔记保留（MCP 那层的备份/索引同步留给交互式写路径；本研究 run 的写路径
  本来就统一走 ``WikiStore``，例如 Ingestor 合并与 formation 标注）。

风险控制与 P2-D 一致：**不允许造出无证据的替代记忆**（新证据无 URL 或无
content_hash 时降级为开冲突台账）；confidence 也不为"新证据"继承旧值——按
**formation 的确定性口径**现算（``_supersede_confidence``：有来源 + 具体事实要素
→ high），这样自动通道也能给出 high，而不是被硬编码成 medium 塞进 Prior 注入块。
替代版本带上 ``observed_at = 证据观察时间``（MCP 通道不知道证据时间，只能回退
created），与 P2-A freshness 的时间基准口径一致。

判官越权拦截（防御纵深）
------------------------

``wiki/verification.py`` 已把"护栏触发时判官只允许保守判定"做成模块内不变量
（``EvidenceComparison.guard`` / ``judge_verdict``）。本模块**再按
``suggested_action`` 与 ``guard`` 双重复核**：``guard`` 非 None 且建议动作是
``supersede`` / ``merge`` 时一律拒绝执行（理论上不可达，作为断言与纵深防御；
理由行写明护栏名）。方向一致：宁可少一次自动更新，不可多一次静默覆盖。
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from researchwiki.wiki.distiller import CandidateNote
from researchwiki.wiki.formation import evaluate_candidate
from researchwiki.wiki.frontmatter import SourceRef
from researchwiki.wiki.store import Note, WikiStore
from researchwiki.wiki.verification import (
    ACTION_MERGE,
    ACTION_NONE,
    ACTION_OPEN_CONFLICT,
    ACTION_REFRESH_REVIEWED_AT,
    ACTION_SUPERSEDE,
    EvidenceComparison,
    EvidenceItem,
)

# ---- applied 取值（MemoryUpdateAction.applied）------------------------------

APPLIED_REVIEWED_AT = "reviewed_at"
APPLIED_SUPERSEDED = "superseded"
APPLIED_MERGED = "merged"
APPLIED_CONFLICT_OPENED = "conflict_opened"
APPLIED_SKIPPED = "skipped"

# applied → counts 桶名（counts 的键集合固定为这五个，恒出现，便于消费方逐键取值）
APPLIED_COUNTS: Mapping[str, str] = {
    APPLIED_REVIEWED_AT: "reviewed",
    APPLIED_SUPERSEDED: "superseded",
    APPLIED_MERGED: "merged",
    APPLIED_CONFLICT_OPENED: "conflicts",
    APPLIED_SKIPPED: "skipped",
}
COUNT_KEYS: tuple[str, ...] = ("reviewed", "superseded", "merged", "conflicts", "skipped")

# ---- 落盘字段键（extra / 冲突台账）-----------------------------------------
#
# 键名与 P2-D 的 MCP 通道保持一致（update_reasons / supersede_reason），
# 让"研究 run 自动更新"与"交互式写入"两条路径在同一套字段上可互相审计。

MEMORY_UPDATE_KEYS = "memory_update_keys"  # list[str]，幂等键（按处置顺序去重累积）
MEMORY_UPDATE_BODY_KEYS = "memory_update_body_keys"  # list[str]，正文哈希键（跨 run 幂等）
MEMORY_UPDATE_REASONS = "memory_update_reasons"  # list[dict]，本模块的动作留痕
UPDATE_REASONS_KEY = "update_reasons"  # P2-D 同名字段：修订原因列表（merge 用）
SUPERSEDE_REASON_KEY = "supersede_reason"  # P2-D 同名字段（supersede 用）
CONFLICT_KEY_FIELD = "memory_update_key"  # 冲突台账两侧 claim 里的幂等键字段
CONFLICT_BODY_KEY_FIELD = "memory_update_body_key"  # 台账里的正文哈希键（跨 run 幂等）
EXCERPT_CHARS = 300  # 台账/理由里正文摘录的截断长度
# 键/留痕列表的保留上限（修复轮 1/5 M-4）：只留最近的 N 条，防止 frontmatter 随
# run 次数无限膨胀（口径与 mark_source_changed 的"上界 = URL 数"一致）
MAX_RECORDED_KEYS = 50

MAX_ACTIONS_IN_REASON = 3  # 理由里最多列几个冲突槽位

# 护栏触发时**接线层**允许放行的动作（修复轮 1/5 Minor M-1）：与 verification 的
# CONSERVATIVE_VERDICTS（conflicting / uncertain）一一镜像——只有开台账与不做动作
# 两种，refresh_reviewed_at（宣称"仍然成立"）也一并拒绝，避免"假一致"绕过护栏。
GUARD_ALLOWED_ACTIONS: tuple[str, ...] = (ACTION_NONE, ACTION_OPEN_CONFLICT)


def _now_iso() -> str:
    """当前 UTC 秒级 ISO 时间（与 store / mcp_server 的落盘口径一致）。"""
    return datetime.now(UTC).isoformat(timespec="seconds")


# ---- 配置 -------------------------------------------------------------------


@dataclass
class MemoryUpdateSettings:
    """``[memory_update]`` 段的解析结果：开关 + 影子模式。

    - ``enabled``：false 是逃生阀（不执行任何动作，只计数与留痕）；
    - ``dry_run``：true = 影子模式（判定与分派照常跑，零写盘），用于在生产上
      先观察"如果开启会改动什么"再放行。

    与 formation 的保守默认一致：AgentLoop 侧 ``memory_update_config=None``
    （未配置，脚本 / 老调用方）时**整个阶段不执行**（连 state.md 行与
    memory-update.json 都不产出，逐字段零行为变化）；server 始终传
    ``config.get("memory_update")``。
    """

    enabled: bool = True
    dry_run: bool = False


def _coerce_bool(value: Any, default: bool) -> bool:
    """宽容布尔解析（与 formation._coerce_bool 同风格：字符串按词表认，非法回默认）。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("false", "0", "no", "off"):
            return False
        if text in ("true", "1", "yes", "on"):
            return True
        return default
    return bool(value)


def memory_update_settings(config: Mapping[str, Any] | None = None) -> MemoryUpdateSettings:
    """解析 ``[memory_update]`` 段（接受段本身或完整 config，两者皆可）。

    缺省 / None = 段缺失（调用方据此决定是否跳过整个阶段；本函数只回默认值）；
    非法值宽容回退默认，不让配置错误打断研究链路（与 formation / prior 同风格）。
    """
    section: Mapping[str, Any] = {}
    if isinstance(config, Mapping):
        raw = config.get("memory_update")
        section = raw if isinstance(raw, Mapping) else config
    settings = MemoryUpdateSettings()
    value = section.get("enabled")
    if value is not None:
        settings.enabled = _coerce_bool(value, settings.enabled)
    value = section.get("dry_run")
    if value is not None:
        settings.dry_run = _coerce_bool(value, settings.dry_run)
    return settings


# ---- 报告结构 ---------------------------------------------------------------


@dataclass
class MemoryUpdateAction:
    """一次处置（或一次跳过）的留痕：判定 → 动作 → 落盘结果。

    - ``applied``：实际执行的动作，取值 ``reviewed_at`` / ``superseded`` /
      ``merged`` / ``conflict_opened`` / ``skipped``（``merged`` 是简报
      ``applied`` 清单的**增项**：``more_specific → merge`` 真的并入时要有个
      可读的名字，不能记成 reviewed 或 skipped 而看不出发生过修订）。
    - ``note_id``：新建/目标笔记 ID（reviewed/merged 为原笔记 ID；supersede 为新的
      或**复用的**替代版本 ID；冲突为台账 ID；skipped 为 None）。
    - ``key``：幂等键（审计用；跳过时也写明是"哪个证据"被跳过）。
    """

    prior_note_id: str
    verdict: str
    suggested_action: str
    applied: str
    note_id: str | None
    reason: str
    key: str = ""
    evidence_index: int = -1
    guard: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "prior_note_id": self.prior_note_id,
            "verdict": self.verdict,
            "suggested_action": self.suggested_action,
            "applied": self.applied,
            "note_id": self.note_id,
            "key": self.key,
            "evidence_index": self.evidence_index,
            "guard": self.guard,
            "reason": self.reason,
        }


@dataclass
class MemoryUpdateReport:
    """一次记忆更新阶段的结果：动作清单 + 五桶计数（``to_dict`` 供落盘/断言）。"""

    actions: list[MemoryUpdateAction] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    dry_run: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "dry_run": self.dry_run,
            "counts": {key: int(self.counts.get(key, 0)) for key in COUNT_KEYS},
            "actions": [action.to_dict() for action in self.actions],
        }

    def state_line(self) -> str:
        """state.md 那一行（``- 记忆更新：复核 M，替代 S，合并 G，冲突 C，跳过 K``）。

        简报要求的前四项（复核/替代/冲突/跳过）按原顺序、原措辞给出；
        ``合并 G`` 是增项——``more_specific → merge`` 真的会发生正文修订，
        不列出来会让"复核"与"跳过"之间的差额无从解释。
        """
        counts = self.counts
        return (
            f"- 记忆更新：复核 {counts.get('reviewed', 0)}，"
            f"替代 {counts.get('superseded', 0)}，"
            f"合并 {counts.get('merged', 0)}，"
            f"冲突 {counts.get('conflicts', 0)}，"
            f"跳过 {counts.get('skipped', 0)}"
        )


# ---- 幂等键 -----------------------------------------------------------------


def body_key(evidence_text: str) -> str:
    """证据**正文**的哈希键（不带 prior 前缀）：跨 run 幂等的"内容指纹"。

    为什么需要它（修复轮 1/5 Important I-1）：带 prior 前缀的 ``update_key`` 只
    能回答"这条 prior 处置过这段证据吗"，回答不了"这段证据在本 wiki 里处置过吗"。
    典型失效场景是 supersede 之后：下一轮 Prior 检索只回 **active**（链尾）——
    也就是替代笔记，它身上记的键是 ``{旧 prior id}:{hash}``，而按新笔记 id 算出的
    键是 ``{新 note id}:{hash}``，两者永不相等，于是同一段证据每次 run 都重新
    supersede 一遍（旧笔记则因为 ``status != active`` 根本不会被读到）。
    "旧笔记仍可读"不等于"会被去读"——所以键必须有一条**不依赖 prior 身份**的
    形态，落在被处置笔记的 extra 里。
    """
    return hashlib.sha1(str(evidence_text or "").strip().encode("utf-8")).hexdigest()[:16]


def update_key(prior_note_id: str, evidence_text: str) -> str:
    """幂等键：``{prior_note_id}:{sha1(证据正文)[:16]}``（纯函数、确定性）。

    选型理由（简报明确二选一：evidence content_hash **或** 正文 hash）：

    - 用**正文 hash** 而不是 ``EvidenceItem.content_hash``：后者是**来源快照**
      的哈希，只在"蒸馏候选引用了某个来源 URL"时才有值（候选没有来源时为空串
      ——同批所有无来源证据会共享同一个空串，键失去区分度）；正文 hash 恒可用，
      且"同一 prior 的同一段证据"这一语义正是我们要的幂等粒度。
    - 前缀 ``prior_note_id``：同一个证据正文可能同时命中多条 Prior（各自独立
      判定、独立处置），键必须带上被处置的旧记忆，否则第二条会被误判成重复。
    - 16 位（64 bit）截断：碰撞概率对"一次 run 内每条 prior 的十几条证据"可忽略，
      同时让 frontmatter 里的键列表保持可读。
    """
    digest = body_key(evidence_text)
    return f"{prior_note_id}:{digest}"


def recorded_keys(note: Note) -> list[str]:
    """旧笔记 extra 里已记录的**带 prior 前缀**的幂等键（形态容错，非列表回空）。"""
    return _string_list(note.meta.extra, MEMORY_UPDATE_KEYS)


def recorded_body_keys(note: Note) -> list[str]:
    """旧笔记 extra 里已记录的**不带前缀**的正文哈希键（跨 run 幂等的查重依据）。"""
    return _string_list(note.meta.extra, MEMORY_UPDATE_BODY_KEYS)


def _string_list(extra: Mapping[str, Any], field_name: str) -> list[str]:
    """extra 里某个键的字符串列表视图（非列表 / 非字符串项一律忽略，容错）。"""
    raw = dict(extra).get(field_name)
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw]


def _bounded(values: Sequence[str], *, limit: int = MAX_RECORDED_KEYS) -> list[str]:
    """键列表保留**最近** ``limit`` 条（修复轮 1/5 Minor M-4：无上限累积）。

    取舍与 ``store.mark_source_changed`` 的"上界 = 引用过的 URL 数"一致：留痕要
    可审计，但不能随 run 次数无限膨胀（每次 run 的每条证据都会添一个键）。
    退化代价明确且方向安全：被裁掉的老键若再次出现同一段证据，会**多做一次**动作
    （而不是漏做）；同一段证据通常在自己的 run 里就已处置过，跨 run 重放概率极低。
    """
    if limit <= 0 or len(values) <= limit:
        return list(values)
    return list(values[-limit:])


# ---- 证据构造 ---------------------------------------------------------------


def _pool_refs(source_urls: Sequence[str], source_refs: Sequence[SourceRef]) -> list[SourceRef]:
    """候选的 URL 列表 → 来源池里对得上的 SourceRef（保序、去重、原 URL 写法）。"""
    by_url: dict[str, SourceRef] = {}
    for ref in source_refs:
        if ref.url and ref.url not in by_url:
            by_url[ref.url] = ref
    out: list[SourceRef] = []
    seen: set[str] = set()
    for url in source_urls:
        url = str(url or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(by_url.get(url) or SourceRef(url=url, content_hash=""))
    return out


def candidate_source_refs(
    candidate: CandidateNote, *, source_refs: Sequence[SourceRef]
) -> list[SourceRef]:
    """候选实际引用的来源（已匹配的 source_refs 优先，其次按 URL 回落到来源池写法）。

    与 ``CandidateNote.from_dict`` 的既有口径一致：``source_refs`` 是"与来源快照
    对上"的引用（带 content_hash），``source_urls`` 是保底列表；两者都为空即
    "这条候选没有可用来源"（supersede 路径据此降级为开冲突台账）。
    """
    refs = [ref for ref in candidate.source_refs if ref.url]
    if refs:
        return refs
    return _pool_refs(candidate.source_urls, source_refs)


def _primary_source(refs: Sequence[SourceRef]) -> SourceRef | None:
    """从候选引用的来源里挑"最能触发来源变化判定"的一条作为证据来源。

    ``EvidenceItem`` 只有一个 ``source_url`` 槽位，取"首个带 content_hash 的来源"
    （有 hash 才可能命中 P2-C 规则 2 的来源变化分支），退化为首个非空 URL；
    无来源返回 None（supersede 路径会据此降级为 open_conflict）。

    多 URL 候选的其余引用**不丢**：比较器每条证据只认一条来源，所以这里只挑主来源；
    候选入库时（Distiller → Ingestor）写进笔记 frontmatter 的是完整 ``source_refs``，
    来源池 ``SourcePool`` 也全量保留。记忆更新动作引用主来源即可定位证据链，
    完整链条在同一条候选笔记上可查。
    """
    usable = [ref for ref in refs if ref.url]
    for ref in usable:
        if ref.content_hash:
            return ref
    return usable[0] if usable else None


def evidence_from_run(
    candidates: Sequence[CandidateNote],
    *,
    source_refs: Sequence[SourceRef],
    now: str | None = None,
) -> list[EvidenceItem]:
    """把本轮蒸馏候选转成 EvidenceItem（保序，与比较/回填下标一一对应）。

    - ``text``：候选正文（strip 后；空正文保留在列表里——P2-C 的
      ``evidence_index`` 必须能回填到输入下标，跳过会让下游错位）；
    - ``observed_at``：本轮观察时间（缺省当前 UTC，可由调用方注入以便复算）；
    - ``source_url`` / ``content_hash``：候选实际引用来源里的"主来源"（无则空串）；
    - ``entities``：候选实体（P2-C 的实体护栏依据）。
    """
    stamp = now or _now_iso()
    items: list[EvidenceItem] = []
    for candidate in candidates:
        refs = candidate_source_refs(candidate, source_refs=source_refs)
        primary = _primary_source(refs)
        items.append(
            EvidenceItem(
                text=str(candidate.text or "").strip(),
                observed_at=stamp,
                source_url=primary.url if primary is not None else "",
                content_hash=primary.content_hash if primary is not None else "",
                entities=list(candidate.entities),
            )
        )
    return items


def _evidence_source_refs(item: EvidenceItem) -> list[SourceRef]:
    """证据条目 → 可写入 frontmatter 的来源列表（**URL 与 content_hash 都非空**才算可用）。

    ``EvidenceItem`` 只有单来源槽位，所以这里最多一条；返回空列表是 supersede
    路径的降级信号（见 ``apply_comparisons``）。

    `content_hash` **必须非空**（修复轮 1/5 Minor M-6）：候选的 ``source_urls``
    里可能有"模型自造 / 只被检索到、从未抓取"的 URL（来源池里的 hash 是空串），
    放行它会让 supersede 造出一条**带无哈希来源**的替代记忆——frontmatter 上有 URL
    但定位不到任何快照，与 ``store.missing_snapshots`` 把空哈希视作"悬空证据"
    的口径直接冲突。空哈希一律按"证据链不完整"处理 → 降级开台账（宁可少一次
    自动更新，不可多一条无法追溯的替代记忆）。
    """
    url = str(item.source_url or "").strip()
    content_hash = str(item.content_hash or "").strip()
    if not url or not content_hash:
        return []
    return [SourceRef(url=url, content_hash=content_hash)]


def _source_gap(item: EvidenceItem) -> str:
    """supersede 降级时的一句话原因（区分"无 URL"与"无 content_hash"两种退化）。"""
    if not str(item.source_url or "").strip():
        return (
            "新证据没有可用来源（无 URL）→ 降级为 open_conflict："
            "宁可开冲突台账，也不造一条无证据的替代记忆"
        )
    return (
        f"新证据的来源 {str(item.source_url).strip()} 没有 content_hash"
        "（无法定位快照 → 证据链不完整）→ 降级为 open_conflict："
        "宁可开冲突台账，也不造一条带悬空来源的替代记忆"
    )


def _merge_sources(
    existing: Sequence[SourceRef], provided: Sequence[SourceRef]
) -> tuple[list[SourceRef], int]:
    """追加式合并来源（merge 语义）：按 ``(url, content_hash)`` 去重、旧来源在前。

    与 P2-D ``service._merge_sources`` 同一口径（去重键、顺序、返回新增条数），
    保证两条写入路径在 frontmatter 上产出同形态的 sources。
    """
    merged = list(existing)
    seen = {(ref.url, ref.content_hash) for ref in merged}
    added = 0
    for ref in provided:
        key = (ref.url, ref.content_hash)
        if key in seen:
            continue
        seen.add(key)
        merged.append(ref)
        added += 1
    return merged, added


# ---- 动作执行 ---------------------------------------------------------------


def apply_comparisons(
    store: WikiStore,
    comparisons: Sequence[EvidenceComparison],
    *,
    evidence: Sequence[EvidenceItem],
    now: str,
    dry_run: bool = False,
    trace_id: str = "",
    evidence_notes: Mapping[int, str] | None = None,
) -> MemoryUpdateReport:
    """按判定执行动作（分发表见模块 docstring）；``dry_run=True`` 零写盘。

    **输入契约**：``comparisons`` 必须由 ``wiki/verification.compare_batch`` 产出
    ——它保证 ``evidence_index`` 与 ``evidence`` 的下标一一对齐；手搓
    ``EvidenceComparison`` 时下标错了会把**别的**证据当作本次证据处置（幂等键、
    正文摘录、来源都会错位）。判定-执行分离的前提就是这个下标契约。

    ``evidence_notes``（修复轮 1/5 Important I-4，接口演进）：``{证据下标: 笔记 ID}``
    ——本轮蒸馏候选经 Ingestor 入库后得到的**规范笔记 ID**（新建 = 新 ID，命中合并
    = 既有规范 ID）。给了映射且目标笔记存在时，``newer`` 路径**复用它当替代版本**
    （只退役旧记忆、不再另建笔记），避免"同一段证据既入库又 supersede"留下两条正文
    相同的 active 记忆（RQ3 检索重复命中、Prior 预算双份消耗）。缺省 None = 老行为
    （新建替代笔记）。

    逐条独立的防御：旧记忆不存在 / ``evidence_index`` 越界 / 批内重复 /
    幂等命中（带前缀键或正文哈希键）/ 护栏拦截 / 非 active 状态 / 本轮已被替代 /
    替代目标就是旧记忆本身 —— 都只记一条 skipped 动作，不抛异常、不中断后续比较
    （写入失败同样降级为 skipped：收尾阶段的辅助产物不得推翻已完成的 run）。
    """
    actions: list[MemoryUpdateAction] = []
    seen_keys: set[str] = set()
    ledger = _conflict_keys(store)  # 台账侧已用过的键（冲突不进笔记 extra）
    retired: set[str] = set()  # 本轮已被替代的旧记忆（后续比较不再落到它身上）
    note_map: Mapping[int, str] = evidence_notes if evidence_notes is not None else {}
    for comparison in comparisons:
        prior = store.get_note(comparison.prior_note_id)
        if prior is None:
            actions.append(
                _skip(comparison, f"旧记忆 {comparison.prior_note_id} 不存在，跳过")
            )
            continue
        item = _evidence_at(evidence, comparison.evidence_index)
        if item is None:
            actions.append(
                _skip(
                    comparison,
                    f"evidence_index={comparison.evidence_index} 越界"
                    f"（本轮证据 {len(evidence)} 条），跳过",
                    prior=prior,
                )
            )
            continue
        key = update_key(prior.id, item.text)
        content_key = body_key(item.text)
        if key in seen_keys:
            actions.append(
                _skip(comparison, "同一批证据内重复：同一 prior 的同一正文已处置", prior, key)
            )
            continue
        seen_keys.add(key)
        # 幂等双通道：带 prior 前缀的键（本轮内/同 prior 重放）+ 不带前缀的正文哈希键
        # （跨 run 重放：supersede 后 prior 检索只回 active 的替代笔记，那时按新 ID
        # 算出的前缀键永远对不上，只能靠正文哈希键命中——修复轮 1/5 I-1）。
        if (
            key in recorded_keys(prior)
            or content_key in recorded_body_keys(prior)
            or key in ledger["keys"]
            or content_key in ledger["body_keys"]
        ):
            actions.append(
                _skip(
                    comparison,
                    "幂等：该证据此前已处置过"
                    f"（键见 {MEMORY_UPDATE_KEYS} / {MEMORY_UPDATE_BODY_KEYS} / 冲突台账），跳过",
                    prior,
                    key,
                )
            )
            continue
        if prior.id in retired:
            actions.append(
                _skip(
                    comparison,
                    f"旧记忆 {prior.id} 本轮已被替代（superseded 版本不再接受动作），跳过",
                    prior,
                    key,
                )
            )
            continue
        if prior.meta.status != "active":
            actions.append(
                _skip(comparison, f"旧记忆 {prior.id} 状态为 {prior.meta.status}，跳过", prior, key)
            )
            continue
        action = comparison.suggested_action
        # 防御纵深（判官越权拦截的接线侧复核）：护栏非 None 时只放行 open_conflict /
        # none——与 verification 的 CONSERVATIVE_VERDICTS 一一镜像（修复轮 1/5 M-1）。
        # 原来只拦 supersede/merge，会照旧放行 refresh_reviewed_at（"一致 → 仍然成立"），
        # 而"一致"恰恰是 verification 在实体不交时特意拒绝的另一类结论：主体同一性
        # 都没确认就宣称旧断言仍然成立，等于用弱信号给旧记忆背书。护栏触发时一律拒绝。
        if comparison.guard is not None and action not in GUARD_ALLOWED_ACTIONS:
            actions.append(
                _skip(
                    comparison,
                    f"护栏「{comparison.guard}」触发：拒绝执行 {action}"
                    f"（护栏触发时只放行 {'/'.join(GUARD_ALLOWED_ACTIONS)}；"
                    "主体同一性未确认时不得覆盖旧记忆、也不得宣称旧断言仍然成立）",
                    prior,
                    key,
                )
            )
            continue
        try:
            if action == ACTION_REFRESH_REVIEWED_AT:
                actions.append(
                    _apply_reviewed(store, prior, item, comparison, key, now, dry_run)
                )
            elif action == ACTION_SUPERSEDE:
                sources = _evidence_source_refs(item)
                if not sources:
                    actions.append(
                        _apply_conflict(
                            store,
                            prior,
                            item,
                            comparison,
                            key,
                            now,
                            dry_run,
                            trace_id,
                            degraded=_source_gap(item),
                        )
                    )
                else:
                    actions.append(
                        _apply_supersede(
                            store,
                            prior,
                            item,
                            comparison,
                            key,
                            now,
                            dry_run,
                            trace_id,
                            sources,
                            reuse_note_id=note_map.get(comparison.evidence_index, ""),
                        )
                    )
                    retired.add(prior.id)
            elif action == ACTION_MERGE:
                actions.append(
                    _apply_merge(store, prior, item, comparison, key, now, dry_run, trace_id)
                )
            elif action == ACTION_OPEN_CONFLICT:
                actions.append(
                    _apply_conflict(
                        store, prior, item, comparison, key, now, dry_run, trace_id
                    )
                )
            elif action == ACTION_NONE:
                actions.append(
                    _skip(
                        comparison,
                        "uncertain → 不做动作（只计数：确定性判据不足，交人工/判官）",
                        prior,
                        key,
                    )
                )
            else:
                actions.append(
                    _skip(comparison, f"未知建议动作 {action}，不执行", prior, key)
                )
        except Exception as exc:  # noqa: BLE001 -- 辅助产物失败不得中断 run 收尾
            actions.append(
                _skip(
                    comparison,
                    f"{action} 执行失败（{type(exc).__name__}: {exc}），"
                    "本次不落盘（旧记忆保持原样）",
                    prior,
                    key,
                )
            )
    return MemoryUpdateReport(actions=actions, counts=_counts(actions), dry_run=dry_run)


def _evidence_at(evidence: Sequence[EvidenceItem], index: int) -> EvidenceItem | None:
    """按下标取证据（越界 / 负下标返回 None）。"""
    if 0 <= index < len(evidence):
        return evidence[index]
    return None


def _counts(actions: Sequence[MemoryUpdateAction]) -> dict[str, int]:
    """五桶计数（键恒齐全，零值也出现——消费方不必做缺键兜底）。"""
    counts = {key: 0 for key in COUNT_KEYS}
    for action in actions:
        bucket = APPLIED_COUNTS.get(action.applied)
        if bucket is not None:
            counts[bucket] += 1
    return counts


def _action(
    comparison: EvidenceComparison,
    applied: str,
    *,
    note_id: str | None,
    reason: str,
    key: str = "",
    prior_note_id: str = "",
) -> MemoryUpdateAction:
    return MemoryUpdateAction(
        prior_note_id=prior_note_id or comparison.prior_note_id,
        verdict=comparison.verdict,
        suggested_action=comparison.suggested_action,
        applied=applied,
        note_id=note_id,
        reason=reason,
        key=key,
        evidence_index=comparison.evidence_index,
        guard=comparison.guard,
    )


def _skip(
    comparison: EvidenceComparison,
    reason: str,
    prior: Note | None = None,
    key: str = "",
) -> MemoryUpdateAction:
    """一条跳过动作（``prior`` 只用于把 prior_note_id 归一为实际读到的 ID）。"""
    return _action(
        comparison,
        APPLIED_SKIPPED,
        note_id=None,
        reason=reason,
        key=key,
        prior_note_id=prior.id if prior is not None else "",
    )


def _reason(
    comparison: EvidenceComparison,
    item: EvidenceItem,
    tail: str,
    *,
    extra: str = "",
    sources: Sequence[SourceRef] = (),
) -> str:
    """审计理由：动作 + verdict + 关键值（冲突槽位 / 相似度）+ 证据来源 URL。

    "不静默覆盖"的落点就在这里——任何写盘动作的 reason 都必须能独立回答
    "凭什么改、改了什么、证据在哪"，不必回读判定模块。
    """
    bits = [f"verdict={comparison.verdict}", f"建议动作={comparison.suggested_action}"]
    if comparison.similarity:
        bits.append(f"相似度={comparison.similarity:.3f}")
    if comparison.conflicts:
        described = "；".join(
            conflict.describe() for conflict in comparison.conflicts[:MAX_ACTIONS_IN_REASON]
        )
        if len(comparison.conflicts) > MAX_ACTIONS_IN_REASON:
            described += f"（另有 {len(comparison.conflicts) - MAX_ACTIONS_IN_REASON} 个槽位）"
        bits.append(f"冲突槽位：{described}")
    if item.source_url:
        bits.append(f"证据来源={item.source_url}")
        if item.content_hash:
            bits.append(f"content_hash={item.content_hash[:8]}")
    else:
        bits.append("证据来源=(无 URL)")
    if sources:
        bits.append("写入来源=" + "、".join(ref.url for ref in sources))
    if comparison.guard is not None:
        bits.append(f"护栏={comparison.guard}")
    if extra:
        bits.append(extra)
    return f"{tail}｜" + "｜".join(bits)


def _audit_extra(
    extra: Mapping[str, Any],
    key: str,
    content_key: str,
    reason: str,
    comparison: EvidenceComparison,
    now: str,
) -> dict[str, Any]:
    """在 extra 上追加本次处置的留痕（前缀键 + 正文哈希键 + 理由记录），其余键保留。

    两个键列表都按 ``MAX_RECORDED_KEYS`` 保留最近若干条（修复轮 1/5 M-4）：键要能
    被无限次重跑查到，但不能让 frontmatter 随 run 次数无限膨胀。
    """
    out = dict(extra)
    out[MEMORY_UPDATE_KEYS] = _bounded(
        _append_unique(_string_list(out, MEMORY_UPDATE_KEYS), key)
    )
    out[MEMORY_UPDATE_BODY_KEYS] = _bounded(
        _append_unique(_string_list(out, MEMORY_UPDATE_BODY_KEYS), content_key)
    )
    history = out.get(MEMORY_UPDATE_REASONS)
    records = list(history) if isinstance(history, list) else []
    records.append(
        {
            "at": now,
            "reason": reason,
            "verdict": comparison.verdict,
            "suggested_action": comparison.suggested_action,
        }
    )
    out[MEMORY_UPDATE_REASONS] = records[-MAX_RECORDED_KEYS:]
    return out


def _append_unique(values: Sequence[str], value: str) -> list[str]:
    """把 value 追加进列表（空串 / 已存在则原样返回，保序）。"""
    out = list(values)
    if value and value not in out:
        out.append(value)
    return out


def _audit_only_extra(
    key: str,
    content_key: str,
    reason: str,
    comparison: EvidenceComparison,
    now: str,
) -> dict[str, Any]:
    """给**新建笔记**用的留痕 extra（不从任何既有 extra 继承，防旧来源标记漂移）。"""
    return _audit_extra({}, key, content_key, reason, comparison, now)


def _save_preserving(
    store: WikiStore,
    note: Note,
    *,
    body: str | None = None,
    status: str | None = None,
    superseded_by: str | None = None,
    reviewed_at: str | None = None,
    sources: Sequence[SourceRef] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Note:
    """覆写一条笔记：走 ``meta.replace(...) + store.save_meta(...)``（P2-F 的推荐入口）。

    **为什么不逐参数透传 ``save_note``**（终审 F1）：那条路要手写每一个字段，漏一个
    就静默丢一个——本方法原来就漏了 ``tombstone``（``save_note`` 的缺省 False），
    于是一条墓碑（失效裁定记录）只要被 consistent / merge 动作碰到就会被"复活"成
    可召回的正常笔记：这是**静默改变记忆语义**，而 ``apply_comparisons`` 是公开
    函数（P4 refresh 是天然的下一个调用方）。改成 ``replace`` 之后，"没点名的字段
    必然保留"是语法保证，而不是靠记性——tombstone 与 kind / importance / valid_* /
    redirect_to / created 全都在内。与 ``store.mark_source_changed`` / ``ingest._merge``
    的重建点同一手法。
    """
    changes: dict[str, Any] = {}
    if status is not None:
        changes["status"] = status
    if superseded_by is not None:
        changes["superseded_by"] = superseded_by
    if reviewed_at is not None:
        changes["reviewed_at"] = reviewed_at
    if sources is not None:
        changes["sources"] = list(sources)
    if extra is not None:
        changes["extra"] = dict(extra)
    return store.save_meta(
        note.meta.replace(**changes),
        body if body is not None else note.body,
    )


def _derive_title(text: str, fallback: str) -> str:
    """从正文猜标题：首个非空行去掉 # 后截 80 字，退化用 fallback（P2-D 同口径）。"""
    for line in str(text or "").splitlines():
        line = line.strip().lstrip("#").strip()
        if line:
            return line[:80]
    return fallback


# ---- 各动作的实现 -----------------------------------------------------------


def _apply_reviewed(
    store: WikiStore,
    prior: Note,
    item: EvidenceItem,
    comparison: EvidenceComparison,
    key: str,
    now: str,
    dry_run: bool,
) -> MemoryUpdateAction:
    """consistent → 只刷新 ``reviewed_at``（sources 与其他字段逐字段不动）。"""
    reason = _reason(
        comparison,
        item,
        f"一致确认：{prior.id} 的断言在本轮证据下仍成立 → 刷新 reviewed_at={now}",
    )
    if not dry_run:
        _save_preserving(
            store,
            prior,
            reviewed_at=now,
            extra=_audit_extra(
                prior.meta.extra, key, body_key(item.text), reason, comparison, now
            ),
        )
    return _action(comparison, APPLIED_REVIEWED_AT, note_id=prior.id, reason=reason, key=key)


def _supersede_confidence(text: str, entities: Sequence[str], sources: Sequence[SourceRef]) -> str:
    """按 **formation 的确定性口径**给替代记忆定 confidence（修复轮 1/5 I-3）。

    口径只有一份：直接复用 ``formation.evaluate_candidate`` 的 ``confidence``
    ——``has_source and has_specifics → high``、``has_source → medium``、否则 low。
    不这么做就会有可见后果：同一 run 里同一段证据留下两条 active 记忆，入库的那条
    拿 distiller 的 confidence、替代的那条硬编码 medium（``prior.py`` 会把
    "置信度: medium" 直接注入模型的核验上下文）——而 RQ2 的自动通道**没有交互式
    调用方**可以补一个 high（MCP 的 ``memory_supersede(confidence=...)`` 那条路径
    是给显式调用者的），不在这里落地就永远落不了地。

    ``similarity_max=None``：近乎重复是**入库**的拒绝规则（防堆积），不是给替代
    记忆定置信度的依据；此处不重复判它。
    """
    decision = evaluate_candidate(
        text, entities=list(entities), sources=list(sources), similarity_max=None
    )
    return decision.confidence


def _apply_supersede(
    store: WikiStore,
    prior: Note,
    item: EvidenceItem,
    comparison: EvidenceComparison,
    key: str,
    now: str,
    dry_run: bool,
    trace_id: str,
    sources: Sequence[SourceRef],
    *,
    reuse_note_id: str = "",
) -> MemoryUpdateAction:
    """newer → 用新证据内容替代旧记忆（**必须带来源**，调用方已保证非空）。

    两种落盘形态：

    - **复用本轮入库笔记**（``reuse_note_id`` 指向本轮 Ingestor 产出的规范笔记，且
      该笔记存在、active、不等于旧记忆本身；修复轮 1/5 I-4）：把旧记忆退役到该笔记
      上（``superseded_by=入库笔记 ID``），不再另建——"同一段证据既入库又 supersede"
      会留下两条正文相同的 active 记忆，RQ3 检索重复命中、Prior 预算双份消耗。
      同时把幂等键/理由写进该笔记 extra、必要时补上证据来源（入库时若被合并进别的
      规范笔记，来源可能不含本次证据）。
    - **新建替代笔记**（无映射，或映射目标不合格）：正文 = 证据正文，sources =
      证据来源（不继承旧来源——旧证据不该被当作新结论的证据），entities /
      volatility / kind / importance 继承旧记忆（身份连续性），``observed_at`` =
      证据观察时间（时间基准追到证据本身）。

    两侧共同的硬约束：confidence 按 formation 口径算（``_supersede_confidence``；
    不继承旧证据的置信度）；旧笔记 ``status=superseded`` + ``superseded_by`` 链 +
    reason/键留痕（旧笔记原样保留，沿链可达替代版本）。
    """
    content_key = body_key(item.text)
    target = store.get_note(reuse_note_id) if reuse_note_id else None
    if target is not None and target.id == prior.id:
        # 候选被 Ingestor 合并进了这条旧记忆本身：事实已经在旧记忆里了，
        # 再"退役到它自己"只会把一条 active 记忆标成 superseded_by 指向自己。
        return _skip(
            comparison,
            f"本轮候选已被 Ingestor 合并进该记忆本身（{prior.id}）：事实已并入，"
            "无需再 supersede/merge（不写盘）",
            prior,
            key,
        )
    if target is not None and target.meta.status != "active":
        target = None  # 目标不是 active（异常状态）：退回新建路径，不把链挂到退役版本上
    confidence = _supersede_confidence(item.text, prior.meta.entities, sources)
    target_note = f"复用本轮入库笔记 {reuse_note_id}" if target is not None else "新建替代笔记"
    reason = _reason(
        comparison,
        item,
        f"更新替代：{prior.id} → {target_note}"
        f"（证据更新，旧 ID 沿 superseded_by 可达；confidence={confidence}）",
        sources=sources,
    )
    new_id: str | None = None
    if not dry_run:
        if target is not None:
            # 复用路径：留痕必须**追加在目标既有 extra 上**（终审 F5）——原来从 {}
            # 起算并与目标 extra 做浅合并，会把目标自己的 memory_update_keys /
            # body_keys / reasons 覆写成单元素，跨 run 的幂等记录随之丢失（同一段
            # 证据会被重复处置，甚至把目标自己再退役一次）。
            merged_sources, _ = _merge_sources(target.meta.sources, sources)
            target_extra = _audit_extra(
                target.meta.extra, key, content_key, reason, comparison, now
            )
            target_extra[SUPERSEDE_REASON_KEY] = reason
            target_extra["superseded_from"] = prior.id
            _save_preserving(store, target, sources=merged_sources, extra=target_extra)
            new_id = target.id
        else:
            # 新建路径：新笔记没有既有 extra，用干净的留痕（不继承旧来源标记）
            extra = _audit_only_extra(key, content_key, reason, comparison, now)
            extra[SUPERSEDE_REASON_KEY] = reason
            extra["superseded_from"] = prior.id
            written = store.save_note(
                item.text,
                title=_derive_title(item.text, fallback=prior.title or "替代记忆"),
                entities=list(prior.meta.entities),
                confidence=confidence,
                volatility=prior.meta.volatility,
                kind=prior.meta.kind,
                importance=prior.meta.importance,
                observed_at=item.observed_at or now,
                reviewed_at=now,
                trace_id=trace_id or prior.meta.trace_id,
                sources=list(sources),
                extra=extra,
            )
            new_id = written.id
        old_extra = _audit_extra(prior.meta.extra, key, content_key, reason, comparison, now)
        old_extra[SUPERSEDE_REASON_KEY] = reason
        _save_preserving(
            store,
            prior,
            status="superseded",
            superseded_by=new_id,
            extra=old_extra,
        )
    elif target is not None:
        new_id = target.id  # dry_run：目标已存在，如实报出（不是伪造的占位值）
    return _action(
        comparison,
        APPLIED_SUPERSEDED,
        note_id=new_id,
        reason=reason,
        key=key,
    )


def _merged_body(prior_body: str, evidence_text: str, now: str) -> str:
    """合并正文：旧正文 + 一条"补充"要点（保留原规范 ID 的修订语义）。

    幂等第二道闸（第一道是幂等键）：证据正文已经出现在旧正文里就不再追加，
    防"键被外部清掉后重复并入同一段事实"。
    """
    addition_text = str(evidence_text or "").strip()
    base = str(prior_body or "").rstrip()
    if addition_text and addition_text in base:
        return prior_body
    addition = f"- 补充（{now}）：{addition_text}"
    return f"{base}\n\n{addition}" if base else addition


def _apply_merge(
    store: WikiStore,
    prior: Note,
    item: EvidenceItem,
    comparison: EvidenceComparison,
    key: str,
    now: str,
    dry_run: bool,
    trace_id: str,
) -> MemoryUpdateAction:
    """more_specific → 并入新事实、保留规范 ID（update 语义，不是 supersede）。

    简报允许在"update 通道不支持正文合并语义"时降级为 open_conflict。读码结论：
    P2-D 的 ``memory_update`` 是**修订**语义（正文由调用方给出、来源按
    (url, hash) 去重追加、刷新 reviewed_at、保留 ID、reason 落 extra），
    "合并"正是调用方的职责——所以本模块选定 **merge**（简报的首选分支），
    由 ``_merged_body`` 拼接正文、``_merge_sources`` 追加来源。

    **无实际变更时如实记 skipped**（修复轮 1/5 M-2）：正文里已有该证据、
    来源也没有新增时，没有任何东西可写——此时若仍记 ``applied=merged``，报告会说
    "改了"而盘上没改（幂等键也没持久化）。变更 = 正文变化 **或** 来源有新增；
    来源有新增就照常落盘（那也是真实的修订）。
    """
    sources = _evidence_source_refs(item)
    merged_body = _merged_body(prior.body, item.text, now)
    merged_sources, added = _merge_sources(prior.meta.sources, sources)
    changed = merged_body != prior.body or added > 0
    if not changed:
        return _skip(
            comparison,
            "并入无实际变更：正文已包含该证据、证据来源也已记录（幂等第二道闸）→ "
            "不写盘（不把未发生的修订记成 merged）",
            prior,
            key,
        )
    # 理由在写盘前一次成形（新增来源条数也进去）：笔记 extra 里的 update_reasons
    # 与报告/state 里的 reason 必须是**同一串**，否则审计时两处对不上。
    reason = _reason(
        comparison, item, f"并入新事实：{prior.id} 正文追加证据要点、来源追加（保留规范 ID）"
    )
    if sources:
        reason += f"｜新增来源 {added} 条"
    else:
        reason += "｜证据无可用来源（无 URL 或 content_hash）：只并入正文要点"
    extra = _audit_extra(prior.meta.extra, key, body_key(item.text), reason, comparison, now)
    records = extra.get(UPDATE_REASONS_KEY)
    history = list(records) if isinstance(records, list) else []
    history.append({"reason": reason, "at": now})
    # 也按 MAX_RECORDED_KEYS 封顶（终审 F6）：P2-D 的 update_reasons 是全量追加契约，
    # 本模块把上限统一到同一常量，四个列表（keys / body_keys / reasons / update_reasons）
    # 的口径才一致——报告 §M-4 声称的"四个列表都保留最近 50 条"由此成立。
    extra[UPDATE_REASONS_KEY] = history[-MAX_RECORDED_KEYS:]
    if not dry_run:
        _save_preserving(
            store,
            prior,
            body=merged_body,
            sources=merged_sources,
            reviewed_at=now,
            extra=extra,
        )
    return _action(
        comparison,
        APPLIED_MERGED,
        note_id=prior.id,
        reason=reason,
        key=key,
    )


def _apply_conflict(
    store: WikiStore,
    prior: Note,
    item: EvidenceItem,
    comparison: EvidenceComparison,
    key: str,
    now: str,
    dry_run: bool,
    trace_id: str,
    *,
    degraded: str = "",
) -> MemoryUpdateAction:
    """conflicting → 写冲突台账（**必须带两侧证据**，不能被 merge 吞掉）。

    - claim_a（旧记忆）：note_id / 标题 / 正文摘录 / 来源（url+hash）/ 判定字段；
    - claim_b（新证据）：正文摘录 / 来源 URL + hash / 观察时间 / 命中槽位取值；
    - 两侧都带 ``memory_update_key``（带 prior 前缀）与 ``memory_update_body_key``
      （正文哈希，跨 run 幂等）：台账没有独立 extra 槽，幂等查重就按这两个字段
      （``_conflict_keys``），重跑不会重复开台账、换了一条 prior 命中同一段证据也
      不会重复开。

    ``degraded`` 非空 = "newer 但来源不可用"的降级路径：verdict 记 newer、
    applied 记 conflict_opened，理由写明降级原因（动作分发表里的硬约束）。
    """
    tail = degraded or (
        f"冲突登记：{prior.id} 与本轮新证据在同一槽位取值矛盾 → 开台账（不静默覆盖）"
    )
    reason = _reason(comparison, item, tail)
    conflict_id: str | None = None
    if not dry_run:
        claim_a: dict[str, Any] = {
            "note_id": prior.id,
            "title": prior.title,
            "excerpt": prior.body.strip()[:EXCERPT_CHARS],
            "observed_at": prior.meta.observed_at,
            "created": prior.meta.created,
            "reviewed_at": prior.meta.reviewed_at,
            "sources": [
                {"url": ref.url, "content_hash": ref.content_hash}
                for ref in prior.meta.sources
            ],
            "verdict": comparison.verdict,
            "suggested_action": comparison.suggested_action,
            "guard": comparison.guard,
            "reason": reason,
            CONFLICT_KEY_FIELD: key,
            CONFLICT_BODY_KEY_FIELD: body_key(item.text),
        }
        claim_b: dict[str, Any] = {
            "evidence_index": comparison.evidence_index,
            "text": item.text.strip()[:EXCERPT_CHARS],
            "source_url": item.source_url,
            "content_hash": item.content_hash,
            "observed_at": item.observed_at,
            "entities": list(item.entities),
            "conflicts": [conflict.to_dict() for conflict in comparison.conflicts],
            "similarity": comparison.similarity,
            "reason": reason,
            CONFLICT_KEY_FIELD: key,
            CONFLICT_BODY_KEY_FIELD: body_key(item.text),
        }
        conflict_id = store.save_conflict(
            _conflict_question(prior, comparison, degraded=degraded),
            claim_a,
            claim_b,
            trace_id=trace_id,
        ).id
    return _action(
        comparison,
        APPLIED_CONFLICT_OPENED,
        note_id=conflict_id,  # dry_run 下恒 None（台账 ID 写入时才分配）
        reason=reason,
        key=key,
    )


def _conflict_question(
    prior: Note, comparison: EvidenceComparison, *, degraded: str = ""
) -> str:
    """冲突台账的问题行：引用具体槽位与取值（人类可读，第一眼看出矛盾在哪）。"""
    label = prior.title or prior.id
    if degraded:
        return f"{label}（{prior.id}）本轮证据更新缺少可用来源，无法自动替代：{degraded}"
    described = "；".join(
        conflict.describe() for conflict in comparison.conflicts[:MAX_ACTIONS_IN_REASON]
    )
    if not described:
        return f"{label}（{prior.id}）与本轮新证据矛盾（verdict={comparison.verdict}）"
    return f"{label}（{prior.id}）与本轮新证据在同一槽位取值矛盾：{described}"


def conflict_has_key(conflict: Any, key: str) -> bool:
    """冲突台账条目是否由该幂等键开出（查两侧 claim 的 ``memory_update_key``）。

    公开给测试与后续 refresh 包复用；容错：claim 非映射 / 无该字段都当 False。
    """
    return _conflict_claim_has(conflict, CONFLICT_KEY_FIELD, key)


def conflict_has_body_key(conflict: Any, content_key: str) -> bool:
    """冲突台账条目是否由该**正文哈希键**开出（跨 run 幂等查重用）。"""
    return _conflict_claim_has(conflict, CONFLICT_BODY_KEY_FIELD, content_key)


def _conflict_claim_has(conflict: Any, field_name: str, value: str) -> bool:
    """台账两侧 claim 的某个字段是否等于给定值（容错：非映射/缺字段当 False）。"""
    for claim in (getattr(conflict, "claim_a", None), getattr(conflict, "claim_b", None)):
        if isinstance(claim, Mapping) and str(claim.get(field_name) or "") == value:
            return True
    return False


def _conflict_keys(store: WikiStore) -> dict[str, set[str]]:
    """台账里出现过的幂等键：``{"keys": {带前缀键}, "body_keys": {正文哈希键}}``。

    冲突路径不写笔记，所以"这条证据处置过"只记在台账里；重跑（或换了一条 prior
    命中同一段证据）时要按它去重，否则同一批证据每跑一次就多开一条冲突。
    ``status=None``：已裁决的冲突同样算处置过。
    """
    keys: set[str] = set()
    body_keys: set[str] = set()
    for conflict in store.list_conflicts(status=None):
        for claim in (conflict.claim_a, conflict.claim_b):
            if not isinstance(claim, Mapping):
                continue
            found = str(claim.get(CONFLICT_KEY_FIELD) or "")
            if found:
                keys.add(found)
            found_body = str(claim.get(CONFLICT_BODY_KEY_FIELD) or "")
            if found_body:
                body_keys.add(found_body)
    return {"keys": keys, "body_keys": body_keys}


__all__ = [
    "APPLIED_CONFLICT_OPENED",
    "APPLIED_MERGED",
    "APPLIED_REVIEWED_AT",
    "APPLIED_SKIPPED",
    "APPLIED_SUPERSEDED",
    "CONFLICT_BODY_KEY_FIELD",
    "CONFLICT_KEY_FIELD",
    "COUNT_KEYS",
    "GUARD_ALLOWED_ACTIONS",
    "MAX_RECORDED_KEYS",
    "MEMORY_UPDATE_BODY_KEYS",
    "MEMORY_UPDATE_KEYS",
    "MEMORY_UPDATE_REASONS",
    "MemoryUpdateAction",
    "MemoryUpdateReport",
    "MemoryUpdateSettings",
    "SUPERSEDE_REASON_KEY",
    "UPDATE_REASONS_KEY",
    "apply_comparisons",
    "body_key",
    "candidate_source_refs",
    "conflict_has_body_key",
    "conflict_has_key",
    "evidence_from_run",
    "memory_update_settings",
    "recorded_body_keys",
    "recorded_keys",
    "update_key",
]
