"""新证据 vs 旧记忆的**动作执行**（RQ2 闭环收尾，P2-E）。

``wiki/verification.py``（P2-C）只判定与建议——不改笔记、不写台账、不写盘，
"不静默覆盖"是设计约束。本模块负责把判定变成**动作**：遍历"每条命中 Prior ×
本轮证据"的比较结论，按 ``suggested_action`` 分派并留下可审计记录。判定与执行
分离是刻意的：判定零副作用、可复算；写入只在这一个模块发生。

动作分发表（逐条可测）
----------------------

- ``consistent`` → ``refresh_reviewed_at``：只刷新 ``reviewed_at=now``（只改这一个
  字段；sources 原样不动；其余 frontmatter 字段逐字段透传）。
- ``newer`` → ``supersede``：新笔记 + 旧笔记 ``superseded_by``；**必须带来源**，
  无可用 URL → 降级 ``open_conflict``（宁可开台账，也不造无证据的替代记忆）。
- ``more_specific`` → ``merge``：update 语义（正文追加要点 + 来源追加 + 刷新
  reviewed_at），保留规范 ID；正文由本模块拼接（见 ``_merged_body``）。
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
2. **幂等**：幂等键 ``update_key(prior_note_id, evidence_text)`` =
   ``{prior_id}:{sha1(正文)[:16]}``（见该函数 docstring 的选型理由）。键记在
   **旧笔记** extra 的 ``memory_update_keys`` 列表里（supersede 后旧笔记仍可读，
   所以重跑能查到），同一批内重复的证据也会被去重。命中已记录的键 → 不写盘、
   计入 skipped。冲突台账没有独立 extra 槽，键写进两侧 claim 的
   ``memory_update_key`` 字段并按它查重。
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
- 可审计性等价：来源按 (url, hash) 去重后写入新笔记、reason 落 extra、
  旧笔记保留（MCP 那层的备份/索引同步留给交互式写路径；本研究 run 的写路径
  本来就统一走 ``WikiStore``，例如 Ingestor 合并与 formation 标注）。

风险控制与 P2-D 一致：**不允许造出无证据的替代记忆**（新证据无 URL 时降级为
开冲突台账），也不为"新证据"继承旧证据的 confidence（一律 ``medium``，见
``_apply_supersede``）。差别只有一处、且更严格：新笔记带上
``observed_at = 证据观察时间``（MCP 通道不知道证据时间，只能回退 created），
这与 P2-A freshness 的时间基准口径一致。

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
MEMORY_UPDATE_REASONS = "memory_update_reasons"  # list[dict]，本模块的动作留痕
UPDATE_REASONS_KEY = "update_reasons"  # P2-D 同名字段：修订原因列表（merge 用）
SUPERSEDE_REASON_KEY = "supersede_reason"  # P2-D 同名字段（supersede 用）
CONFLICT_KEY_FIELD = "memory_update_key"  # 冲突台账两侧 claim 里的幂等键字段
EXCERPT_CHARS = 300  # 台账/理由里正文摘录的截断长度

MAX_ACTIONS_IN_REASON = 3  # 理由里最多列几个冲突槽位


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
    - ``note_id``：新建/目标笔记 ID（reviewed/merged 为原笔记 ID；supersede 为
      新笔记 ID；冲突为台账 ID；skipped 为 None）。
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
    digest = hashlib.sha1(str(evidence_text or "").strip().encode("utf-8")).hexdigest()
    return f"{prior_note_id}:{digest[:16]}"


def recorded_keys(note: Note) -> list[str]:
    """旧笔记 extra 里已记录的幂等键（形态容错：非列表 / 非字符串一律忽略）。"""
    raw = dict(note.meta.extra).get(MEMORY_UPDATE_KEYS)
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw]


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
    """证据条目 → 可写入 frontmatter 的来源列表（无 URL 即空列表 = 无可用来源）。

    ``EvidenceItem`` 只有单来源槽位，所以这里最多一条；返回空列表是 supersede
    路径的降级信号（见 ``apply_comparisons``）。
    """
    url = str(item.source_url or "").strip()
    if not url:
        return []
    return [SourceRef(url=url, content_hash=str(item.content_hash or ""))]


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
) -> MemoryUpdateReport:
    """按判定执行动作（分发表见模块 docstring）；``dry_run=True`` 零写盘。

    **输入契约**：``comparisons`` 必须由 ``wiki/verification.compare_batch`` 产出
    ——它保证 ``evidence_index`` 与 ``evidence`` 的下标一一对齐；手搓
    ``EvidenceComparison`` 时下标错了会把**别的**证据当作本次证据处置（幂等键、
    正文摘录、来源都会错位）。判定-执行分离的前提就是这个下标契约。

    逐条独立的防御：旧记忆不存在 / ``evidence_index`` 越界 / 批内重复 /
    幂等命中 / 护栏拦截 / 非 active 状态 / 本轮已被替代 —— 都只记一条 skipped
    动作，不抛异常、不中断后续比较（写入失败同样降级为 skipped：收尾阶段的
    辅助产物不得推翻已完成的 run）。
    """
    actions: list[MemoryUpdateAction] = []
    seen_keys: set[str] = set()
    ledger_keys = _conflict_keys(store)  # 台账侧已用过的幂等键（冲突不进笔记 extra）
    retired: set[str] = set()  # 本轮已被替代的旧记忆（后续比较不再落到它身上）
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
        if key in seen_keys:
            actions.append(
                _skip(comparison, "同一批证据内重复：同一 prior 的同一正文已处置", prior, key)
            )
            continue
        seen_keys.add(key)
        if key in recorded_keys(prior) or key in ledger_keys:
            actions.append(
                _skip(
                    comparison,
                    "幂等：该证据此前已处置过"
                    f"（键见 {MEMORY_UPDATE_KEYS} / 冲突台账），跳过",
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
        # 防御纵深（判官越权拦截的接线侧复核）：护栏非 None 时拒绝覆盖类动作。
        # verification 层已保证不可达（护栏触发 → uncertain/none 或 conflicting），
        # 这里作为断言保留——漏一次就是一次静默覆盖。
        if comparison.guard is not None and action in (ACTION_SUPERSEDE, ACTION_MERGE):
            actions.append(
                _skip(
                    comparison,
                    f"护栏「{comparison.guard}」触发：拒绝执行 {action}"
                    "（主体同一性未确认时不得覆盖旧记忆）",
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
                            degraded=(
                                "新证据没有可用来源（无 URL）→ 降级为 open_conflict："
                                "宁可开冲突台账，也不造一条无证据的替代记忆"
                            ),
                        )
                    )
                else:
                    actions.append(
                        _apply_supersede(
                            store, prior, item, comparison, key, now, dry_run, trace_id, sources
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
    reason: str,
    comparison: EvidenceComparison,
    now: str,
) -> dict[str, Any]:
    """在 extra 上追加本次处置的留痕（键列表 + 理由记录），其余键原样保留。"""
    out = dict(extra)
    keys = out.get(MEMORY_UPDATE_KEYS)
    recorded = [str(item) for item in keys] if isinstance(keys, list) else []
    if key and key not in recorded:
        recorded.append(key)
    out[MEMORY_UPDATE_KEYS] = recorded
    records = out.get(MEMORY_UPDATE_REASONS)
    history = list(records) if isinstance(records, list) else []
    history.append(
        {
            "at": now,
            "reason": reason,
            "verdict": comparison.verdict,
            "suggested_action": comparison.suggested_action,
        }
    )
    out[MEMORY_UPDATE_REASONS] = history
    return out


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
    """覆写一条笔记，**逐字段透传**未显式覆盖的 frontmatter 字段。

    与 ``store.mark_source_changed`` / ``AgentLoop._annotate_formation`` 同一手法：
    ``save_note`` 的参数默认值会静默丢字段，所以这里把 title / entities /
    confidence / volatility / kind / importance / observed_at / valid_* /
    source_changed_at / created 逐字段搬过去（只改调用方点名的那几项）。
    """
    meta = note.meta
    return store.save_note(
        body if body is not None else note.body,
        note_id=note.id,
        title=meta.title,
        entities=list(meta.entities),
        confidence=meta.confidence,
        status=status if status is not None else meta.status,
        redirect_to=meta.redirect_to,
        superseded_by=superseded_by if superseded_by is not None else meta.superseded_by,
        volatility=meta.volatility,
        kind=meta.kind,
        importance=meta.importance,
        observed_at=meta.observed_at,
        reviewed_at=reviewed_at if reviewed_at is not None else meta.reviewed_at,
        valid_from=meta.valid_from,
        valid_until=meta.valid_until,
        source_changed_at=meta.source_changed_at,
        trace_id=meta.trace_id,
        sources=list(sources) if sources is not None else list(meta.sources),
        extra=dict(extra) if extra is not None else dict(meta.extra),
        created=meta.created,
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
            extra=_audit_extra(prior.meta.extra, key, reason, comparison, now),
        )
    return _action(comparison, APPLIED_REVIEWED_AT, note_id=prior.id, reason=reason, key=key)


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
) -> MemoryUpdateAction:
    """newer → 用新证据内容替代旧记忆（**必须带来源**，调用方已保证非空）。

    新笔记：正文 = 证据正文，sources = 证据来源（不继承旧来源——旧证据不该被
    当作新结论的证据），entities/volatility/kind/importance 继承旧记忆（身份
    连续性），confidence 一律 ``medium``（不继承旧证据的置信度，与 P2-D 的
    「有来源 → medium」口径一致），``observed_at`` = 证据观察时间（时间基准追到
    证据本身）。
    旧笔记：``status=superseded`` + ``superseded_by=新 ID``，reason 与幂等键写进
    extra（旧笔记原样保留，沿链可达新记忆）。
    """
    reason = _reason(
        comparison,
        item,
        f"更新替代：{prior.id} → 新笔记（证据更新，旧 ID 沿 superseded_by 可达）",
        sources=sources,
    )
    new_id: str | None = None
    if not dry_run:
        written = store.save_note(
            item.text,
            title=_derive_title(item.text, fallback=prior.title or "替代记忆"),
            entities=list(prior.meta.entities),
            confidence="medium",
            volatility=prior.meta.volatility,
            kind=prior.meta.kind,
            importance=prior.meta.importance,
            observed_at=item.observed_at or now,
            reviewed_at=now,
            trace_id=trace_id or prior.meta.trace_id,
            sources=list(sources),
            extra={
                MEMORY_UPDATE_KEYS: [key],
                MEMORY_UPDATE_REASONS: [
                    {
                        "at": now,
                        "reason": reason,
                        "verdict": comparison.verdict,
                        "suggested_action": comparison.suggested_action,
                    }
                ],
                SUPERSEDE_REASON_KEY: reason,
                "superseded_from": prior.id,
            },
        )
        new_id = written.id
        old_extra = _audit_extra(prior.meta.extra, key, reason, comparison, now)
        old_extra[SUPERSEDE_REASON_KEY] = reason
        _save_preserving(
            store,
            prior,
            status="superseded",
            superseded_by=new_id,
            extra=old_extra,
        )
    return _action(
        comparison,
        APPLIED_SUPERSEDED,
        note_id=new_id,  # dry_run 下恒 None：新 ID 是写入时才分配的，不伪造占位值
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
    """
    sources = _evidence_source_refs(item)
    merged_body = _merged_body(prior.body, item.text, now)
    merged_sources, added = _merge_sources(prior.meta.sources, sources)
    # 理由在写盘前一次成形（新增来源条数也进去）：笔记 extra 里的 update_reasons
    # 与报告/state 里的 reason 必须是**同一串**，否则审计时两处对不上。
    reason = _reason(
        comparison, item, f"并入新事实：{prior.id} 正文追加证据要点、来源追加（保留规范 ID）"
    )
    if sources:
        reason += f"｜新增来源 {added} 条"
    else:
        reason += "｜证据无可用来源：只并入正文要点"
    extra = _audit_extra(prior.meta.extra, key, reason, comparison, now)
    records = extra.get(UPDATE_REASONS_KEY)
    history = list(records) if isinstance(records, list) else []
    history.append({"reason": reason, "at": now})
    extra[UPDATE_REASONS_KEY] = history
    if not dry_run and merged_body != prior.body:
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
    - 两侧都带 ``memory_update_key``：台账没有独立 extra 槽，幂等键按它查重
      （``_conflict_exists``），重跑不会重复开台账。

    ``degraded`` 非空 = "newer 但没有可用来源"的降级路径：verdict 记 newer、
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
        return f"{label}（{prior.id}）本轮证据更新缺少来源，无法自动替代：{degraded}"
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
    for claim in (getattr(conflict, "claim_a", None), getattr(conflict, "claim_b", None)):
        if isinstance(claim, Mapping) and str(claim.get(CONFLICT_KEY_FIELD) or "") == key:
            return True
    return False


def _conflict_keys(store: WikiStore) -> set[str]:
    """台账里出现过的全部幂等键（``status=None``：已裁决的冲突同样算处置过）。

    冲突路径不写笔记，所以"这条证据处置过"只记在台账里；重跑时要按它去重，
    否则同一批证据每跑一次就多开一条冲突（幂等要求）。
    """
    keys: set[str] = set()
    for conflict in store.list_conflicts(status=None):
        for claim in (conflict.claim_a, conflict.claim_b):
            if isinstance(claim, Mapping):
                found = str(claim.get(CONFLICT_KEY_FIELD) or "")
                if found:
                    keys.add(found)
    return keys


__all__ = [
    "APPLIED_CONFLICT_OPENED",
    "APPLIED_MERGED",
    "APPLIED_REVIEWED_AT",
    "APPLIED_SKIPPED",
    "APPLIED_SUPERSEDED",
    "CONFLICT_KEY_FIELD",
    "COUNT_KEYS",
    "MEMORY_UPDATE_KEYS",
    "MEMORY_UPDATE_REASONS",
    "MemoryUpdateAction",
    "MemoryUpdateReport",
    "MemoryUpdateSettings",
    "SUPERSEDE_REASON_KEY",
    "UPDATE_REASONS_KEY",
    "apply_comparisons",
    "candidate_source_refs",
    "conflict_has_key",
    "evidence_from_run",
    "memory_update_settings",
    "recorded_keys",
    "update_key",
]
