"""时间感知的记忆有效性计算（RQ2：什么时候一条记忆不再应该被当成当前事实）。

本模块把"这条记忆现在还算不算当前事实"变成**可计算、可复算、可解释**的确定
状态：输入一条 ``Note``（frontmatter 里的 volatility / observed_at / created /
valid_from / valid_until / kind）+ 一个"现在"的时刻，输出一个 ``FreshnessState``
（state + age_days + half_life_days + decay + 引用具体数值的 reasons）。

三层语义（逐条可测；判定按 1 → 2 → 3 → 4 的顺序短路，命中即定状态）：

1. **时间基准**：``observed_at`` → ``created``（前者缺失或不可解析才回退，与
   ``index.freshness_factor`` 同口径）。两者都拿不到 → "未知年龄"：``age_days``
   按 0.0 计、state 一律 ``review_due``（既不判 fresh 也不判 stale）、理由写明
   "缺少时间基准，建议人工确认"。
2. **显式有效期优先**：``valid_until`` 存在且 ``now > valid_until`` → 直接
   ``stale``（**不看 volatility 衰减**）；``valid_from`` 存在且
   ``now < valid_from`` → "尚未生效"，state ``review_due``（不判 fresh）。两个
   字段写了但不是合法 ISO 时不报错，按"未声明"处理并在理由里说明。
3. **volatility 衰减**：``stable`` 不衰减（``decay`` 恒 1.0、``half_life_days``
   为 None，state 恒 ``fresh``，除非命中规则 2 或规则 1 的"缺基准"）；
   ``drifting`` / ``volatile`` 按半衰期指数衰减 ``decay = 0.5 ** (age / half)``，
   再按阈值分类：``decay > review_due_ratio`` → fresh；
   ``stale_ratio < decay <= review_due_ratio`` → review_due；
   ``decay <= stale_ratio`` → stale（**阈值取闭区间**："降到该比例"即命中，故
   恰好一个半衰期 = decay 0.5 → review_due，恰好两个半衰期 = decay 0.25 → stale）。
4. **kind 差异留出接口**：``FreshnessSettings.per_kind``（如
   ``{"user": {"review_due_ratio": 0.7}}``）按 kind 覆盖默认参数（可覆盖
   ``review_due_ratio`` / ``stale_ratio`` / ``half_life_days``）；**缺省为空 dict，
   默认行为等价于不区分 kind**（PLAN §3.3：user 记忆与 knowledge 记忆的时效参数
   "可能需要区分"，具体取值留给后续实现与评测校准，现在不写死）。

与 ``index.py`` 的关系（口径约束）：``decay`` 与检索层的
``index.freshness_factor(...)`` **数值口径完全一致**——同一套
"observed_at 回退 created、age <= 0 不衰减、半衰期 <= 0 不衰减、0.5 ** (age/half)"
规则。但本模块是**独立实现、不 import index.py**（计算层不反向依赖检索层，
避免把检索层耦合进时序计算）；``DEFAULT_HALF_LIFE_DAYS`` 与 index 的同名常量
语义相同，tests/test_freshness.py 用"逐值一致"断言锁住两份常量与两份公式，
防未来单侧漂移。

其它约定：

- **零网络、零模型调用、全确定性**：同一 (note, now, settings) 恒得同一结果。
- **时钟可注入**：``now`` 缺省 ``datetime.now(UTC)``；传入的 naive 时间按 UTC 解释。
- 本模块只回答"该不该复核/该怎么看待"，**不**检索过滤、**不**写盘、**不**改
  笔记状态（supersede / conflict 留痕属后续 verification 包）。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from researchwiki.wiki.store import Note

# ---- 状态常量 ---------------------------------------------------------------

FRESHNESS_FRESH = "fresh"
FRESHNESS_REVIEW_DUE = "review_due"
FRESHNESS_STALE = "stale"
FRESHNESS_STATES: tuple[str, ...] = (FRESHNESS_FRESH, FRESHNESS_REVIEW_DUE, FRESHNESS_STALE)

# ---- 缺省参数（[freshness] 段可覆盖）-----------------------------------------

# 半衰期（天）：与 index.DEFAULT_HALF_LIFE_DAYS 同值同语义（tests 逐值断言锁死）。
# stable 不在表内 = 不衰减；配置里给 <= 0 也按"不衰减"处理（与 freshness_factor 一致）。
DEFAULT_HALF_LIFE_DAYS: Mapping[str, float] = {"volatile": 30.0, "drifting": 90.0}
DEFAULT_REVIEW_DUE_RATIO = 0.5
DEFAULT_STALE_RATIO = 0.25

# 构造默认 freshness 统计桶（lint 用；顺序即输出顺序）
_ZERO_COUNTS: Mapping[str, int] = dict.fromkeys(FRESHNESS_STATES, 0)

# 队列排序权重：stale 比 review_due 更该先复核
_QUEUE_RANK: Mapping[str, int] = {FRESHNESS_STALE: 0, FRESHNESS_REVIEW_DUE: 1}


# ---- 数据结构 ---------------------------------------------------------------


@dataclass
class FreshnessState:
    """一条记忆在某个时刻的时效判定结论（全部字段可读、可复算、可审计）。

    - state: fresh | review_due | stale（判定规则见模块 docstring）
    - age_days: 相对"现在"的年龄（由基准时间字段算出，负数按 0.0 计）
    - half_life_days: 该 volatility 生效的半衰期；None = 不衰减（stable 或配置 <= 0）
    - decay: 0.0–1.0 的有效性衰减，1.0 = 完全有效（口径同 index.freshness_factor）
    - reasons: 人类可读的判定依据，每条都引用具体数值（含基准时间/年龄/decay/阈值）
    """

    note_id: str
    state: str
    age_days: float
    half_life_days: float | None
    decay: float
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """序列化（lint --json / MCP 复用；reasons 原样保留为字符串列表）。"""
        return {
            "note_id": self.note_id,
            "state": self.state,
            "age_days": self.age_days,
            "half_life_days": self.half_life_days,
            "decay": self.decay,
            "reasons": list(self.reasons),
        }


@dataclass
class FreshnessSettings:
    """时效判定的全部参数（集中可配；默认值与 PLAN/§3.3 的"先不区分 kind"一致）。

    - half_life_days: volatility → 半衰期（天）；缺省沿用 DEFAULT_HALF_LIFE_DAYS
      （与 index 同表），表外的 volatility（如 stable）不衰减。
    - review_due_ratio / stale_ratio: decay 降到该比例即判 review_due / stale
      （**闭区间**：``decay <= ratio`` 命中；stale_ratio 不得大于 review_due_ratio，
      越界会被夹到 review_due_ratio，避免判定退化）。
    - per_kind: ``{kind: {参数名: 值}}`` 按 kind 覆盖上面三项（可只覆盖其中一项）。
      缺省空 dict = 不区分 kind（默认行为与"无 per_kind"逐字节一致）。

    本类在 ``__post_init__`` 里做宽容归一：比率夹到 [0.0, 1.0]、非法值回退默认、
    半衰期表剔除非数值项——配置错误不抛异常，也不打断研究链路（与
    formation.from_config / prior_settings 的宽容风格一致）。
    """

    half_life_days: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_HALF_LIFE_DAYS))
    review_due_ratio: float = DEFAULT_REVIEW_DUE_RATIO
    stale_ratio: float = DEFAULT_STALE_RATIO
    per_kind: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        base = dict(DEFAULT_HALF_LIFE_DAYS)
        base.update(_half_life_overrides(self.half_life_days))
        self.half_life_days = base
        self.review_due_ratio = _ratio(self.review_due_ratio, DEFAULT_REVIEW_DUE_RATIO)
        self.stale_ratio = min(
            _ratio(self.stale_ratio, DEFAULT_STALE_RATIO), self.review_due_ratio
        )
        self.per_kind = {
            str(kind): dict(override)
            for kind, override in (self.per_kind or {}).items()
            if isinstance(override, Mapping)
        }

    def for_kind(self, kind: str | None) -> FreshnessSettings:
        """叠加该 kind 的覆盖项，返回生效参数（无覆盖时返回 self，零开销）。"""
        override = self.per_kind.get(str(kind))
        if not override:
            return self
        half_life = dict(self.half_life_days)
        raw_half_life = override.get("half_life_days")
        if isinstance(raw_half_life, Mapping):
            half_life.update(_half_life_overrides(raw_half_life))
        review_due = _ratio(override.get("review_due_ratio"), self.review_due_ratio)
        stale = min(_ratio(override.get("stale_ratio"), self.stale_ratio), review_due)
        return FreshnessSettings(
            half_life_days=half_life,
            review_due_ratio=review_due,
            stale_ratio=stale,
            per_kind={},  # 覆盖只应用一层，防 per_kind 里再嵌 per_kind
        )


def _ratio(value: Any, default: float) -> float:
    """比率宽容归一：None / 非法 / bool 回退 default，其余夹到 [0.0, 1.0]。"""
    if value is None or isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:  # NaN
        return default
    return min(1.0, max(0.0, number))


def _half_life_overrides(raw: Any) -> dict[str, float]:
    """半衰期覆盖项宽容解析：只收可转 float 的值，None / 非法值丢弃。"""
    out: dict[str, float] = {}
    if not isinstance(raw, Mapping):
        return out
    for key, value in raw.items():
        if value is None or isinstance(value, bool):
            continue
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def from_config(config: Mapping[str, Any] | None = None) -> FreshnessSettings:
    """解析时效配置：接受完整 config（取 ``[freshness]`` 段）或直接给段。

    支持形态（缺省 / None = 全默认；非法值宽容回退，不抛异常）::

        [freshness]
        review_due_ratio = 0.5
        stale_ratio = 0.25
        half_life_days = { volatile = 30, drifting = 90 }   # 覆盖项按 volatility 合并

        [freshness.user]          # 按 kind 覆盖（键名不是已知参数时即视为 kind 段）
        review_due_ratio = 0.7

    half_life_days 与 index.wiki_settings 同策略：在默认表上 merge（只写 volatile
    时 drifting 的默认值保留）。per_kind 段缺省为空 = 默认不区分 kind。
    """
    section: Mapping[str, Any] = {}
    if isinstance(config, Mapping):
        raw = config.get("freshness")
        section = raw if isinstance(raw, Mapping) else config
    known = {"half_life_days", "review_due_ratio", "stale_ratio"}
    half_life = dict(DEFAULT_HALF_LIFE_DAYS)
    half_life.update(_half_life_overrides(section.get("half_life_days")))
    return FreshnessSettings(
        half_life_days=half_life,
        review_due_ratio=_ratio(section.get("review_due_ratio"), DEFAULT_REVIEW_DUE_RATIO),
        stale_ratio=_ratio(section.get("stale_ratio"), DEFAULT_STALE_RATIO),
        # 映射型非已知键 = 该 kind 的覆盖段（如 [freshness.user]）
        per_kind={
            str(key): dict(value)
            for key, value in section.items()
            if key not in known and isinstance(value, Mapping)
        },
    )


# ---- 时间工具（与 index.freshness_factor 同口径的独立实现）-------------------


def _parse_ts(text: Any) -> datetime | None:
    """宽松解析 ISO 时间戳；无时区的按 UTC 解释，不可解析返回 None。

    与 index._parse_ts / mcp_server.service._parse_ts 同约定（本模块独立实现，
    不 import index）。
    """
    if text is None or isinstance(text, bool):
        return None
    raw = str(text).strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _resolve_base_time(meta: Any) -> tuple[str | None, datetime | None]:
    """基准时间：``(字段名, 时间)``；observed_at 优先，缺失/不可解析回退 created。

    两者都拿不到返回 ``(None, None)`` —— 即"未知年龄"（规则 1）。
    """
    observed = _parse_ts(meta.observed_at)
    if observed is not None:
        return "observed_at", observed
    created = _parse_ts(meta.created)
    if created is not None:
        return "created", created
    return None, None


def _effective_half_life(half_life_days: Mapping[str, float], volatility: str) -> float | None:
    """生效半衰期：表内为正数才衰减，否则 None（stable / 未配置 / <= 0 不衰减）。

    与 index.freshness_factor 的 ``half_life is None or half_life <= 0 → 1.0`` 同口径。
    """
    value = half_life_days.get(volatility)
    if value is None or value <= 0:
        return None
    return float(value)


def decay_factor(age_days: float, half_life_days: float | None) -> float:
    """半衰期指数衰减：``0.5 ** (age / half)``；不衰减（None/<=0）或 age <= 0 → 1.0。

    与 index.freshness_factor 的数值口径一致（见模块 docstring 的"口径约束"）。
    """
    if half_life_days is None or half_life_days <= 0 or age_days <= 0:
        return 1.0
    return 0.5 ** (age_days / half_life_days)


# ---- 主入口 -----------------------------------------------------------------


def evaluate_freshness(
    note: Note,
    *,
    now: datetime | None = None,
    settings: FreshnessSettings | None = None,
) -> FreshnessState:
    """对一条笔记做确定性时效判定（规则 1–4 见模块 docstring，按序短路）。

    ``now`` 缺省 ``datetime.now(UTC)``（测试传固定时钟即可完全复现）；
    ``settings`` 缺省全默认（= 不区分 kind，与 PLAN §3.3"先不写死"一致）。
    本函数不改笔记、不写盘、不访问网络。
    """
    cfg = (settings if settings is not None else FreshnessSettings()).for_kind(note.meta.kind)
    moment = now if now is not None else datetime.now(UTC)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)

    meta = note.meta
    reasons: list[str] = []

    # 基准时间与年龄（规则 1）
    base_field, base_ts = _resolve_base_time(meta)
    if base_ts is None:
        age_days = 0.0
        reasons.append(
            "缺少时间基准（observed_at 与 created 均缺失或不可解析）"
            "，年龄按 0.0 天计，建议人工确认"
        )
    else:
        age_days = max(0.0, (moment - base_ts).total_seconds() / 86400.0)
        reasons.append(f"时间基准 {base_field}（{base_ts.isoformat()}），年龄 {age_days:.1f} 天")

    # 显式有效期（规则 2）：先记录不可解析的写法，再判两个方向
    valid_from = _parse_ts(meta.valid_from)
    valid_until = _parse_ts(meta.valid_until)
    for field_name, raw, parsed in (
        ("valid_from", meta.valid_from, valid_from),
        ("valid_until", meta.valid_until, valid_until),
    ):
        if raw and parsed is None:
            reasons.append(f"{field_name}={raw} 不是合法 ISO 时间，按未声明处理（不参与判定）")

    half_life = _effective_half_life(cfg.half_life_days, meta.volatility)
    decay = decay_factor(age_days, half_life)
    if half_life is None:
        reasons.append(f"volatility={meta.volatility} 不衰减（无生效半衰期），decay 1.000")
    else:
        reasons.append(
            f"volatility={meta.volatility} 半衰期 {half_life:.1f} 天，"
            f"年龄 {age_days:.1f} 天 → decay {decay:.3f}"
        )

    # 规则 2：显式失效 / 显式未生效，优先于 volatility 衰减
    if valid_until is not None and moment > valid_until:
        reasons.append(
            f"valid_until={valid_until.isoformat()} 已过期"
            f"（now {moment.isoformat()} > valid_until）→ stale"
            "（显式失效，不看 volatility 衰减）"
        )
        return FreshnessState(note.id, FRESHNESS_STALE, age_days, half_life, decay, reasons)
    if valid_from is not None and moment < valid_from:
        reasons.append(
            f"valid_from={valid_from.isoformat()} 未到生效时间"
            f"（now {moment.isoformat()} < valid_from）→ review_due（未生效，不判 fresh）"
        )
        return FreshnessState(note.id, FRESHNESS_REVIEW_DUE, age_days, half_life, decay, reasons)

    # 规则 1 的"未知年龄"：既不判 fresh 也不判 stale，交人工确认
    if base_ts is None:
        reasons.append(
            "缺少时间基准 → review_due（decay 按 1.000 计但不足以判 fresh，也不够判 stale）"
        )
        return FreshnessState(note.id, FRESHNESS_REVIEW_DUE, age_days, half_life, decay, reasons)

    # 规则 3：volatility 衰减分类（阈值闭区间）
    if half_life is None:
        reasons.append(f"volatility={meta.volatility} 不衰减 → fresh（decay 1.000）")
        state = FRESHNESS_FRESH
    elif decay <= cfg.stale_ratio:
        reasons.append(
            f"decay {decay:.3f} ≤ stale_ratio {cfg.stale_ratio:.2f} → stale"
        )
        state = FRESHNESS_STALE
    elif decay <= cfg.review_due_ratio:
        reasons.append(
            f"decay {decay:.3f} ≤ review_due_ratio {cfg.review_due_ratio:.2f}"
            f"（> stale_ratio {cfg.stale_ratio:.2f}）→ review_due"
        )
        state = FRESHNESS_REVIEW_DUE
    else:
        reasons.append(
            f"decay {decay:.3f} > review_due_ratio {cfg.review_due_ratio:.2f} → fresh"
        )
        state = FRESHNESS_FRESH
    return FreshnessState(note.id, state, age_days, half_life, decay, reasons)


def freshness_queue(
    notes: Sequence[Note],
    *,
    now: datetime | None = None,
    settings: FreshnessSettings | None = None,
) -> list[FreshnessState]:
    """待复核队列：只留 review_due + stale，按"该优先复核"排序。

    排序：state（stale 先于 review_due）→ age_days 降序（越老越先）→ note_id 升序
    （同级同龄时保证确定性输出）。范围由调用方决定（lint 只传 active 笔记；
    本函数不按 status 过滤，也不改笔记）。
    """
    cfg = settings if settings is not None else FreshnessSettings()
    moment = now if now is not None else datetime.now(UTC)
    states = [evaluate_freshness(note, now=moment, settings=cfg) for note in notes]
    pending = [s for s in states if s.state in _QUEUE_RANK]
    pending.sort(key=lambda s: (_QUEUE_RANK[s.state], -s.age_days, s.note_id))
    return pending


def freshness_counts(states: Sequence[FreshnessState]) -> dict[str, int]:
    """把一批判定结果折成 ``{fresh: n, review_due: n, stale: n}``（lint 统计用）。

    未在 FRESHNESS_STATES 内的值（理论上不会出现）不计入，避免统计口径被污染。
    """
    counts = dict(_ZERO_COUNTS)
    for state in states:
        if state.state in counts:
            counts[state.state] += 1
    return counts


__all__ = [
    "DEFAULT_HALF_LIFE_DAYS",
    "DEFAULT_REVIEW_DUE_RATIO",
    "DEFAULT_STALE_RATIO",
    "FRESHNESS_FRESH",
    "FRESHNESS_REVIEW_DUE",
    "FRESHNESS_STALE",
    "FRESHNESS_STATES",
    "FreshnessSettings",
    "FreshnessState",
    "decay_factor",
    "evaluate_freshness",
    "freshness_counts",
    "freshness_queue",
    "from_config",
]
