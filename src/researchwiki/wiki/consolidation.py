"""后台维护选择器（P4a Task 16 / T2）：扫描 active 记忆，产出四类维护候选。

``plan_consolidation`` 对 ``store.list_notes(status="active")`` 做**一次扫描**
（列表只取一次、所有候选计算复用），按四条确定性规则（顺序固定、零模型调用、
零网络）选出维护候选：

1. **merge**：active knowledge 笔记两两之间，实体交集 ≥1（slug 归一后）且
   verification 尺度相似度 ∈ ``[merge_min_similarity, MERGE_MAX_SIMILARITY)``
   → 并入候选。canonical 取 created 更早者（并列取字典序更小 ID），payload 记
   canonical/absorbed、相似度与相交实体。
   **merge_floor 明文化（P2 移交裁定）**：merge 下限 0.5，独立于替换门 0.6——
   merge 是保守动作（双内容都保留在规范 ID 下），supersede 是覆盖动作；0.6–0.8
   灰区的新证据不 supersede（落 uncertain 人工看），但允许 merge（内容不丢），
   与 [verification] 现有注释逐字对齐。校准锚点：真实 run 在 sim 0.31 曾并入
   正文——0.31 < 0.5 新地板，该形态不再自动 merge。
   上界 ``MERGE_MAX_SIMILARITY = 0.95``：≥0.95 属近乎重复带（[wiki].dedup 与
   formation.near_duplicate 的辖区），consolidation 只兜"相关但表述不同"的漏网对。
2. **refresh**：freshness 三态 ∈ {review_due, stale} 且 ≥1 个来源 URL 的笔记
   → 重抓候选（payload 记 url/urls）；**无来源的 review_due/stale 笔记改派
   rejudge**（无从重抓，只能重判）。
3. **rejudge**：confidence=="low" 的 active 笔记 ∪（无来源且 review_due/stale，
   来自规则 2 改派）；payload 记执行器（T3）要用的模型档位。
4. **conflict**：同实体 knowledge 笔记对过 verification 比较（respect
   similarity_floor；实体护栏是全局前置——同实体预筛之后护栏不会触发，保留它
   是双保险），verdict=="conflicting" → 开台账候选，payload 记双方断言槽位。
   每计划最多扫 ``max_conflict_pairs`` 对（确定性排序后取前 N，防组合爆炸）。

排序与截断：每类候选先按 (note_ids, idempotency_key) 字典序排好、再取前
``max_*`` 个；最终动作列表按 (action, note_ids) 字典序输出——**同一 store 状态 +
同一 settings + 同一 now 必须产出逐字节相同的 plan**（可复算测试锁定）。

幂等键：``idempotency_key = sha256(action + sorted(note_ids) + 各笔记 body_hash +
action 参数)``——T3 幂等重试的钥匙。body_hash 是正文 sha256（刻意不用
index.note_index_hash：那是检索视图指纹，含 title/confidence 等 meta；幂等键只应
跟着"断言内容"变——reviewed_at 刷新等 meta 动作不换键，重试才安全；正文一变键
即变，重复执行自然被视为新工作）。材料编码：``"\n"`` 连接、payload 用 sort_keys
JSON（tests/test_consolidation.py 按公式逐字节锁定）。

边界与逃生阀（Global Constraints 逐条落地）：

- **纯选择零写盘**：本模块不调用 store 的任何写接口（``store.save_conflict``
  由 T3 执行器才真正调用）；dry-run 前后目录逐字节不变（测试锁定）。
- **段缺失 = 不接线 = 整段无操作**：``consolidation_settings`` 对缺
  ``[consolidation]`` 段的 config 返回 None（CLI 报"未配置"退出 1，与旧桩语义
  衔接）；``enabled = false`` 即完全禁用（零判定，plan 直接为空、scanned=0）。
- **墓碑排除**：tombstone 笔记是失效裁定审计记录（P2-F 裁定一），不是现役知识，
  四类候选都不收（与 ledger.iter_active_unanswered 的排除方向一致）；它们计入
  scanned（被扫到），只是不产候选。
- **判定尺度接线**：verification / freshness 生效参数由 ``consolidation_settings``
  从完整 config 的 ``[verification]`` / ``[freshness]``（半衰期回退 ``[wiki]``）
  段读入（与 lint 的 ``freshness_from_config(load_config())`` 同手法），并随
  settings_snapshot 进 plan——计划自带"用什么尺判的"完整快照，可离线复算。
  ``rejudge_stale_ratio`` 是 [freshness].stale_ratio 的**声明值**（config 注释
  明文镜像），stale/review_due 的判定本身委托 freshness 模块（口径只有一份）。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from itertools import combinations
from typing import Any

from researchwiki.wiki.freshness import (
    FRESHNESS_REVIEW_DUE,
    FRESHNESS_STALE,
    FreshnessSettings,
    evaluate_freshness,
    parse_ts,
)
from researchwiki.wiki.freshness import from_config as freshness_from_config
from researchwiki.wiki.ingest import entity_keys
from researchwiki.wiki.store import Note, WikiStore
from researchwiki.wiki.verification import (
    VERDICT_CONFLICTING,
    EvidenceComparison,
    EvidenceItem,
    VerificationSettings,
    compare_prior_and_evidence,
    token_similarity,
)

# ---- 动作常量（T3 执行器按此分派）--------------------------------------------

ACTION_MERGE = "merge"  # 并入规范笔记（保守：双内容保留）
ACTION_REFRESH = "refresh"  # 重抓来源并比对快照
ACTION_REJUDGE = "rejudge"  # 重判 confidence
ACTION_CONFLICT = "conflict"  # 开冲突台账

# ---- 缺省参数（[consolidation] 段可覆盖）--------------------------------------

# merge 候选的相似度下限（verification 尺度）：独立于替换门 0.6 的明文下限
# （P2 移交裁定，理由见模块 docstring 与 config.toml [consolidation] 注释）。
DEFAULT_MERGE_MIN_SIMILARITY = 0.5
# merge 候选的相似度上界（**排他**，简报给定、不可配置）：≥0.95 属近乎重复带，
# 是 [wiki].dedup（0.9）与 formation.near_duplicate（0.95）的辖区。
MERGE_MAX_SIMILARITY = 0.95
# rejudge_stale_ratio 缺省值：等于 [freshness].stale_ratio 的声明镜像（见
# ConsolidationSettings 注释——判定本身委托 freshness，本值只作锚点随快照留痕）。
DEFAULT_REJUDGE_STALE_RATIO = 0.25
# conflict 规则每计划最多扫的笔记对数（防组合爆炸；0 = 不扫）
DEFAULT_MAX_CONFLICT_PAIRS = 50
# 每类批量上限（0 = 该类禁用）
DEFAULT_MAX_PER_CLASS = 10
# rejudge 动作默认的模型档位（payload 记录，T3 执行器按此调用）：与笔记蒸馏同档
# 的成本口径——重判的是单条笔记的 confidence，不是整轮综述。
REJUDGE_TIER = "cheap"

# created 缺失/不可解析时的排序哨兵：视为最晚，canonical 让给有时间基准的笔记
_LATEST = datetime.max.replace(tzinfo=UTC)


# ---- 数据结构 ----------------------------------------------------------------


@dataclass
class PlannedAction:
    """一条维护候选（T3 执行器的输入；本模块只选择、不执行）。

    - ``action``：merge / refresh / rejudge / conflict 之一；
    - ``note_ids``：涉及的笔记 ID（升序；merge/conflict 是笔记对，其余是单条）；
    - ``reason``：人类可读的触发依据，引用具体数值（相似度、decay、槽位取值）；
    - ``idempotency_key``：幂等键（公式见模块 docstring；T3 重试按它去重）；
    - ``payload``：动作参数——merge 的 canonical_id/absorbed_id、refresh 的
      url/urls、rejudge 的模型档位、conflict 的双方断言槽位。
    """

    action: str
    note_ids: list[str]
    reason: str
    idempotency_key: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """序列化（consolidate --json / T3 审计留痕复用；payload 原样保留）。"""
        return {
            "action": self.action,
            "note_ids": list(self.note_ids),
            "reason": self.reason,
            "idempotency_key": self.idempotency_key,
            "payload": dict(self.payload),
        }


@dataclass
class ConsolidationPlan:
    """一次选择的完整产出（可直接 to_dict 进 --json；format_text 进人读表）。

    ``settings_snapshot`` 是生效 settings 的 dict 快照（含 verification/freshness
    尺度参数）——计划可离线复算的依据；刻意**不含任何墙钟时间**，同一状态两次
    plan 的 JSON 逐字节相同。
    """

    actions: list[PlannedAction]
    scanned: int
    settings_snapshot: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """序列化（consolidate --json 的输出形态）。"""
        return {
            "actions": [action.to_dict() for action in self.actions],
            "scanned": self.scanned,
            "settings_snapshot": self.settings_snapshot,
        }

    def format_text(self, *, root: str = "") -> str:
        """人读表（consolidate 不带 --json 时的输出）：标题行 + 每动作一行。"""
        if not self.settings_snapshot.get("enabled", True):
            return f"consolidation dry-run（root={root}）：已禁用（enabled=false）→ 零判定零留痕。"
        lines = [
            f"consolidation dry-run（root={root}）：扫描 {self.scanned} 条 active 笔记，"
            f"维护候选 {len(self.actions)} 个"
        ]
        if not self.actions:
            lines.append("  （无维护候选）")
        for action in self.actions:
            lines.append(f"  [{action.action:>8}] {'、'.join(action.note_ids)}  {action.reason}")
        return "\n".join(lines)


@dataclass
class ConsolidationSettings:
    """``[consolidation]`` 段的解析结果 + 判定尺度（verification/freshness）接线。

    - ``enabled = false`` 逃生阀：完全禁用（零判定，plan 直接为空）；
    - ``merge_min_similarity``：merge 候选的相似度下限（verification 尺度）——
      独立于替换门 0.6 的明文下限（P2 移交裁定，理由见模块 docstring）；
    - ``rejudge_stale_ratio``：等于 [freshness].stale_ratio 的**声明值**（config
      注释明文镜像）。stale/review_due 的判定本身委托 freshness 模块（口径只有
      一份，``freshness`` 字段是它的生效参数），本值只作校准锚点随快照留痕；
    - ``max_conflict_pairs``：conflict 规则每计划最多扫的笔记对数（防组合爆炸）；
    - ``max_merges / max_refresh / max_rejudge / max_conflicts``：每类批量上限
      （0 = 该类禁用）；
    - ``verification`` / ``freshness``：判定尺度的生效参数——
      ``consolidation_settings`` 从完整 config 接线（[verification] / [freshness]
      段，半衰期回退 [wiki]），直接构造时用模块默认值（与 config.toml 现值一致）。

    非法值宽容回退默认、数值越界夹取，不抛异常（与 formation / freshness 的
    宽容风格一致：配置错误不打断维护链路）。
    """

    enabled: bool = True
    merge_min_similarity: float = DEFAULT_MERGE_MIN_SIMILARITY
    rejudge_stale_ratio: float = DEFAULT_REJUDGE_STALE_RATIO
    max_conflict_pairs: int = DEFAULT_MAX_CONFLICT_PAIRS
    max_merges: int = DEFAULT_MAX_PER_CLASS
    max_refresh: int = DEFAULT_MAX_PER_CLASS
    max_rejudge: int = DEFAULT_MAX_PER_CLASS
    max_conflicts: int = DEFAULT_MAX_PER_CLASS
    verification: VerificationSettings = field(default_factory=VerificationSettings)
    freshness: FreshnessSettings = field(default_factory=FreshnessSettings)

    def __post_init__(self) -> None:
        self.merge_min_similarity = _ratio(
            self.merge_min_similarity, DEFAULT_MERGE_MIN_SIMILARITY
        )
        self.rejudge_stale_ratio = _ratio(self.rejudge_stale_ratio, DEFAULT_REJUDGE_STALE_RATIO)
        self.max_conflict_pairs = _count(self.max_conflict_pairs, DEFAULT_MAX_CONFLICT_PAIRS)
        self.max_merges = _count(self.max_merges, DEFAULT_MAX_PER_CLASS)
        self.max_refresh = _count(self.max_refresh, DEFAULT_MAX_PER_CLASS)
        self.max_rejudge = _count(self.max_rejudge, DEFAULT_MAX_PER_CLASS)
        self.max_conflicts = _count(self.max_conflicts, DEFAULT_MAX_PER_CLASS)


def consolidation_settings(config: Mapping[str, Any] | None) -> ConsolidationSettings | None:
    """解析 ``[consolidation]`` 段；**段缺失返回 None = 不接线**。

    与 [formation]/[memory_update]/[unanswered] 的 guarded 语义逐字一致：缺段时
    CLI 报"未配置"退出 1（与旧桩语义衔接），不做任何扫描。段存在时非法值宽容
    回退默认、越界值夹取。判定尺度随完整 config 一起接线：``[verification]`` 段
    进 ``settings.verification``（conflict 判定 respect 配置的 similarity_floor），
    ``[freshness]``（半衰期回退 ``[wiki]``）段进 ``settings.freshness``。
    """
    if not isinstance(config, Mapping):
        return None
    section = config.get("consolidation")
    if not isinstance(section, Mapping):
        return None
    settings = ConsolidationSettings(
        enabled=_flag(section.get("enabled"), True),
        merge_min_similarity=_ratio(
            section.get("merge_min_similarity"), DEFAULT_MERGE_MIN_SIMILARITY
        ),
        rejudge_stale_ratio=_ratio(
            section.get("rejudge_stale_ratio"), DEFAULT_REJUDGE_STALE_RATIO
        ),
        max_conflict_pairs=_count(section.get("max_conflict_pairs"), DEFAULT_MAX_CONFLICT_PAIRS),
        max_merges=_count(section.get("max_merges"), DEFAULT_MAX_PER_CLASS),
        max_refresh=_count(section.get("max_refresh"), DEFAULT_MAX_PER_CLASS),
        max_rejudge=_count(section.get("max_rejudge"), DEFAULT_MAX_PER_CLASS),
        max_conflicts=_count(section.get("max_conflicts"), DEFAULT_MAX_PER_CLASS),
    )
    settings.verification = VerificationSettings.from_config(config)
    settings.freshness = freshness_from_config(config)
    return settings


# ---- 宽容归一（与 verification / freshness 的私有助手同语义）------------------


def _ratio(value: Any, default: float) -> float:
    """比率宽容归一：None / 非法 / bool / NaN 回退 default，其余夹到 [0.0, 1.0]。"""
    if value is None or isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:  # NaN
        return default
    return min(1.0, max(0.0, number))


def _count(value: Any, default: int) -> int:
    """条数宽容归一：None / 非法 / bool 回退 default，负值夹到 0（0 = 该类禁用）。"""
    if value is None or isinstance(value, bool):
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(0, number)


_TRUE_WORDS = ("1", "true", "yes", "on", "y", "t")
_FALSE_WORDS = ("0", "false", "no", "off", "n", "f")


def _flag(value: Any, default: bool) -> bool:
    """布尔开关宽容归一：真 bool 直接认，字符串按真/假词表认，其余回 default。"""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, int | float):
        return bool(value)
    text = str(value).strip().lower()
    if text in _TRUE_WORDS:
        return True
    if text in _FALSE_WORDS:
        return False
    return default


# ---- 幂等键 ------------------------------------------------------------------


def _body_hash(note: Note) -> str:
    """正文内容哈希（幂等键材料）：sha256(body) 的 hex。

    刻意不用 index.note_index_hash（那是检索视图指纹，含 title/confidence 等
    meta）：幂等键只应跟着"断言内容"变——reviewed_at 刷新等 meta 动作不换键，
    T3 重试才安全；正文一变键即变，重复执行自然被视为新工作。
    """
    return hashlib.sha256(note.body.encode("utf-8")).hexdigest()


def _idempotency_key(
    action: str,
    note_ids: Sequence[str],
    body_hashes: Mapping[str, str],
    payload: Mapping[str, Any],
) -> str:
    """幂等键：``sha256(action + sorted(note_ids) + 各笔记 body_hash + action 参数)``。

    材料编码：``"\\n"`` 连接（action / 排序后的 note_ids / 对应 body_hash /
    sort_keys JSON 的 payload），UTF-8 编码后取 sha256 hex。同一 (action, notes,
    body, 参数) 恒得同一键（T3 幂等重试按它去重）；笔记正文一变键即变。
    公式由 tests/test_consolidation.py 逐字节锁定。
    """
    ordered = sorted(note_ids)
    material = "\n".join(
        [
            action,
            *ordered,
            *(body_hashes[note_id] for note_id in ordered),
            json.dumps(dict(payload), ensure_ascii=False, sort_keys=True),
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


# ---- 主入口 ------------------------------------------------------------------


def plan_consolidation(
    store: WikiStore,
    settings: ConsolidationSettings,
    *,
    embedding: Any | None = None,
    now: datetime | None = None,
) -> ConsolidationPlan:
    """扫描 active 记忆产出维护候选（**纯选择，零写盘**；同一状态恒得同一计划）。

    - ``now``：freshness 判定的时钟（缺省当前 UTC；测试注入固定时钟可完全复现）；
    - ``embedding``：预留参数（签名由简报给定）——当前判定尺度固定为 verification
      的确定性度量（零网络、可复算），传入非 None 抛 NotImplementedError：换尺度
      是一次显式取舍（见 wiki/verification.py docstring），不是补一个参数；
    - ``settings.enabled = False`` → 完全禁用：零判定，plan 为空（scanned=0）。

    流程：``list_notes(status="active")`` **只取一次**（T15 评审已知线性扫描代价，
    整计划复用这一份列表），merge/conflict 与 refresh/rejudge 两族候选在同一份
    数据上算完，最后按 (action, note_ids) 字典序合并输出。
    """
    if embedding is not None:
        raise NotImplementedError(
            "plan_consolidation 的 embedding 参数是预留位：判定尺度固定为 verification"
            " 的确定性度量（零网络、可复算），注入其他尺度是一次显式取舍"
            "（见 wiki/verification.py docstring），当前不支持。"
        )
    snapshot = asdict(settings)
    if not settings.enabled:
        # 逃生阀（Global Constraints）：enabled=false 即完全禁用——零判定零留痕
        return ConsolidationPlan(actions=[], scanned=0, settings_snapshot=snapshot)
    moment = now if now is not None else datetime.now(UTC)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)

    notes = store.list_notes(status="active")  # 一次扫描：整计划复用这一份列表
    # 墓碑是失效裁定审计记录（P2-F 裁定一），不是现役知识 → 四类候选都不收
    pool = sorted((note for note in notes if not note.tombstone), key=lambda note: note.id)
    body_hashes = {note.id: _body_hash(note) for note in pool}
    actions: list[PlannedAction] = []
    actions.extend(_merge_and_conflict_candidates(pool, settings, body_hashes))
    actions.extend(_refresh_and_rejudge_candidates(pool, settings, body_hashes, moment))
    actions.sort(key=lambda item: (item.action, tuple(item.note_ids), item.idempotency_key))
    return ConsolidationPlan(actions=actions, scanned=len(notes), settings_snapshot=snapshot)


# ---- 规则 1 / 4：merge 与 conflict（同实体 knowledge 笔记对）------------------


def _merge_and_conflict_candidates(
    pool: Sequence[Note],
    settings: ConsolidationSettings,
    body_hashes: Mapping[str, str],
) -> list[PlannedAction]:
    """同实体 knowledge 笔记对上的两类候选（一次配对枚举，两族规则共用）。

    - merge：verification 尺度相似度 ∈ [merge_floor, 0.95)（便宜，全量算）；
    - conflict：完整 verification 比较（贵，只扫前 max_conflict_pairs 对）。
    配对顺序 = 笔记按 ID 升序的两两组合（确定性，防组合爆炸的截断因此可复算）。
    """
    knowledge = [note for note in pool if note.kind == "knowledge"]
    entity_map = {note.id: entity_keys(note.entities) for note in knowledge}
    pairs: list[tuple[Note, Note, set[str]]] = []
    for left, right in combinations(knowledge, 2):
        shared = entity_map[left.id] & entity_map[right.id]
        if shared:
            pairs.append((left, right, shared))

    merges: list[PlannedAction] = []
    for left, right, shared in pairs:
        similarity = token_similarity(left.body, right.body, settings=settings.verification)
        if settings.merge_min_similarity <= similarity < MERGE_MAX_SIMILARITY:
            merges.append(_merge_action(left, right, shared, similarity, settings, body_hashes))
    merges.sort(key=lambda item: (tuple(item.note_ids), item.idempotency_key))

    conflicts: list[PlannedAction] = []
    # 实体护栏（verification 全局前置）在此不会触发：预筛保证两侧实体相交；
    # 保留比较器内部的护栏是双保险（不绕过、不重写判定）。
    for left, right, _shared in pairs[: settings.max_conflict_pairs]:
        comparison = compare_prior_and_evidence(
            left,
            EvidenceItem(text=right.body, entities=list(right.entities)),
            settings=settings.verification,
        )
        if comparison.verdict != VERDICT_CONFLICTING:
            continue
        conflicts.append(_conflict_action(left, right, comparison, body_hashes))
    conflicts.sort(key=lambda item: (tuple(item.note_ids), item.idempotency_key))
    return [*merges[: settings.max_merges], *conflicts[: settings.max_conflicts]]


def _created_order(note: Note) -> tuple[datetime, str]:
    """canonical 排序键：created 更早者优先（缺失/不可解析视为最晚）；并列取更小 ID。"""
    parsed = parse_ts(note.meta.created)
    if parsed is None:
        return (_LATEST, note.id)
    return (parsed, note.id)


def _merge_action(
    left: Note,
    right: Note,
    shared: set[str],
    similarity: float,
    settings: ConsolidationSettings,
    body_hashes: Mapping[str, str],
) -> PlannedAction:
    """组装一条 merge 候选：canonical = created 更早者，payload 记 canonical/absorbed。"""
    canonical, absorbed = sorted((left, right), key=_created_order)
    payload = {
        "canonical_id": canonical.id,
        "absorbed_id": absorbed.id,
        "similarity": round(similarity, 4),
        "entities": sorted(shared),
    }
    note_ids = sorted((left.id, right.id))
    reason = (
        f"实体相交（{'、'.join(sorted(shared))}）且 verification 相似度 {similarity:.4f} "
        f"∈ [{settings.merge_min_similarity:.2f}, {MERGE_MAX_SIMILARITY:.2f})："
        f"canonical={canonical.id}（created={canonical.meta.created or '缺失'} 更早），"
        f"absorbed={absorbed.id}（T3 并入规范 ID 并留痕 merged）"
    )
    return PlannedAction(
        action=ACTION_MERGE,
        note_ids=note_ids,
        reason=reason,
        idempotency_key=_idempotency_key(ACTION_MERGE, note_ids, body_hashes, payload),
        payload=payload,
    )


def _conflict_action(
    left: Note,
    right: Note,
    comparison: EvidenceComparison,
    body_hashes: Mapping[str, str],
) -> PlannedAction:
    """组装一条 conflict 候选：payload 记双方断言槽位（SlotConflict 的完整 dict）。"""
    slots = [conflict.to_dict() for conflict in comparison.conflicts]
    described = "；".join(
        f"{slot['slot']}（旧={slot['prior_value']} 新={slot['evidence_value']}）" for slot in slots
    )
    payload = {"slots": slots, "similarity": round(comparison.similarity, 4)}
    note_ids = sorted((left.id, right.id))
    reason = (
        f"同实体笔记过 verification 判定 verdict=conflicting：{len(slots)} 个槽位取值冲突"
        f"（{described}）→ 开冲突台账候选（写 conflicts/ 由 T3 执行）"
    )
    return PlannedAction(
        action=ACTION_CONFLICT,
        note_ids=note_ids,
        reason=reason,
        idempotency_key=_idempotency_key(ACTION_CONFLICT, note_ids, body_hashes, payload),
        payload=payload,
    )


# ---- 规则 2 / 3：refresh 与 rejudge（单条笔记）--------------------------------


def _refresh_and_rejudge_candidates(
    pool: Sequence[Note],
    settings: ConsolidationSettings,
    body_hashes: Mapping[str, str],
    moment: datetime,
) -> list[PlannedAction]:
    """单条笔记上的两类候选：refresh（review_due/stale + 有来源）与 rejudge。

    - review_due/stale 且 ≥1 个来源 URL → refresh（payload 记 url/urls）；
    - review_due/stale 但无来源 → 改派 rejudge（规则 2 的显式改派）；
    - confidence=="low" → rejudge（规则 3 第一析取支，与 freshness 无关）。
    同一笔记同时命中 rejudge 的两个来源时只出**一条**动作（∪ 语义），触发原因
    依触发顺序拼接。refresh 按 note_ids 排序后取前 max_refresh 条；rejudge 按
    note_ids 排序后取前 max_rejudge 条。
    """
    refreshes: list[PlannedAction] = []
    rejudge_triggers: dict[str, list[str]] = {}
    for note in pool:
        state = evaluate_freshness(note, now=moment, settings=settings.freshness)
        due = state.state in (FRESHNESS_REVIEW_DUE, FRESHNESS_STALE)
        # 来源 URL 去重保序（frontmatter 顺序即确定性顺序；空 URL 不算"有来源"）
        urls = list(dict.fromkeys(s.url for s in note.meta.sources if s.url))
        if due and urls:
            payload = {"url": urls[0], "urls": urls}
            note_ids = [note.id]
            reason = (
                f"freshness={state.state}（decay {state.decay:.3f}）且带 {len(urls)} 条来源"
                f"（{urls[0]}）→ 重抓候选（T3 重新抓取并比对快照）"
            )
            refreshes.append(
                PlannedAction(
                    action=ACTION_REFRESH,
                    note_ids=note_ids,
                    reason=reason,
                    idempotency_key=_idempotency_key(
                        ACTION_REFRESH, note_ids, body_hashes, payload
                    ),
                    payload=payload,
                )
            )
        elif due:
            rejudge_triggers.setdefault(note.id, []).append(
                f"freshness={state.state}（decay {state.decay:.3f}）"
                "且无来源可重抓 → 由 refresh 规则改派重判"
            )
        if note.confidence == "low":
            rejudge_triggers.setdefault(note.id, []).append(
                "confidence=low（低置信记忆需重新判定）"
            )

    rejudges: list[PlannedAction] = []
    for note_id in sorted(rejudge_triggers)[: settings.max_rejudge]:
        payload = {"tier": REJUDGE_TIER}
        note_ids = [note_id]
        reason = (
            "；".join(rejudge_triggers[note_id]) + f"（T3 按 {REJUDGE_TIER} 档重判 confidence）"
        )
        rejudges.append(
            PlannedAction(
                action=ACTION_REJUDGE,
                note_ids=note_ids,
                reason=reason,
                idempotency_key=_idempotency_key(ACTION_REJUDGE, note_ids, body_hashes, payload),
                payload=payload,
            )
        )
    refreshes.sort(key=lambda item: (tuple(item.note_ids), item.idempotency_key))
    return [*refreshes[: settings.max_refresh], *rejudges]


__all__ = [
    "ACTION_CONFLICT",
    "ACTION_MERGE",
    "ACTION_REJUDGE",
    "ACTION_REFRESH",
    "DEFAULT_MERGE_MIN_SIMILARITY",
    "DEFAULT_MAX_CONFLICT_PAIRS",
    "DEFAULT_MAX_PER_CLASS",
    "DEFAULT_REJUDGE_STALE_RATIO",
    "MERGE_MAX_SIMILARITY",
    "REJUDGE_TIER",
    "ConsolidationPlan",
    "ConsolidationSettings",
    "PlannedAction",
    "consolidation_settings",
    "plan_consolidation",
]
