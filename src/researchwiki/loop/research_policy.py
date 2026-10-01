"""P3 Dynamic Retrieval：budget-aware 确定性召回策略（模式判定与解释）。

判定只用可复算的确定性特征（检索命中、freshness 三态、volatility、open 冲突、
token 预算余量），不引入分类器；每次判定必须带 features + reasons 落盘
（PLAN §3.4 验收：不允许只有模式字符串）。
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from researchwiki.wiki.freshness import (
    FRESHNESS_REVIEW_DUE,
    FRESHNESS_STALE,
    FreshnessSettings,
    evaluate_freshness,
)
from researchwiki.wiki.index import SearchIndex
from researchwiki.wiki.store import WikiStore
from researchwiki.wiki.verification import token_similarity

MODE_SIMPLE = "simple"
MODE_UPDATE = "update"
MODE_DEEP = "deep"
MODES = (MODE_SIMPLE, MODE_UPDATE, MODE_DEEP)

_REPORT_STYLES = ("brief", "standard", "full")

# 报告样式后缀（P3 Task 5 接线点）：report_style="brief" 时追加到报告阶段
# system 提示词末尾的简报约束；standard/full 不追加（现状不变）。
_REPORT_STYLE_SUFFIX = "限 200 字以内，直接回答，仅列关键结论"


@dataclass(frozen=True)
class ModeLimits:
    """一个模式的检索/预算限额（决定"召回多少、研究多深"）。"""
    prior_k: int
    prior_max_chars: int
    max_fresh_searches: int
    max_steps: int
    subagents: bool
    report_style: str          # brief | standard | full
    min_fresh_sources: int     # >0 时报告须核验新鲜来源数，不足写明缺口（诚实边界）


DEFAULT_LIMITS: dict[str, ModeLimits] = {
    # simple：直接回答或轻量检索——不动 fresh 搜索，只读少量 Prior
    MODE_SIMPLE: ModeLimits(prior_k=3, prior_max_chars=1200, max_fresh_searches=0,
                            max_steps=3, subagents=False, report_style="brief",
                            min_fresh_sources=0),
    # update：读取旧记忆 + 少量 fresh verification
    MODE_UPDATE: ModeLimits(prior_k=5, prior_max_chars=4000, max_fresh_searches=2,
                            max_steps=6, subagents=False, report_style="standard",
                            min_fresh_sources=1),
    # deep：预算完整的研究 loop（与 P2 现状等价的默认上限）
    MODE_DEEP: ModeLimits(prior_k=5, prior_max_chars=4000, max_fresh_searches=4,
                          max_steps=12, subagents=True, report_style="full",
                          min_fresh_sources=0),
}

_LIMIT_INT_BOUNDS = {"prior_k": (1, 20), "prior_max_chars": (200, 20000),
                     "max_fresh_searches": (0, 20), "max_steps": (1, 50),
                     "min_fresh_sources": (0, 20)}


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _mode_limits(mode: str, override: Mapping[str, Any] | None) -> ModeLimits:
    base = DEFAULT_LIMITS[mode]
    if not override:
        return base
    kwargs: dict[str, Any] = {}
    for f in ("prior_k", "prior_max_chars", "max_fresh_searches",
              "max_steps", "min_fresh_sources"):
        if f in override:
            low, high = _LIMIT_INT_BOUNDS[f]
            kwargs[f] = _clamp(int(override[f]), low, high)
    if "subagents" in override:
        kwargs["subagents"] = bool(override["subagents"])
    if "report_style" in override:
        style = str(override["report_style"])
        if style not in _REPORT_STYLES:
            raise ValueError(f"report_style 必须是 {_REPORT_STYLES} 之一，得到 {style!r}")
        kwargs["report_style"] = style
    return replace(base, **kwargs)


@dataclass(frozen=True)
class PolicySettings:
    """确定性策略的阈值与各模式限额。"""
    enabled: bool = True
    probe_k: int = 8                    # 覆盖度探测检索条数
    limits: Mapping[str, ModeLimits] = field(default_factory=lambda: dict(DEFAULT_LIMITS))
    # open 冲突与当前问题的 bigram 相似度门槛（同 [verification].similarity_floor 量级）
    conflict_similarity: float = 0.3
    coverage_min_hits: int = 2          # simple 要求的最小命中数
    # simple 要求的 top 分数下限。量级标定：RRF k=60 单通道 top1≈1/61≈0.0164，
    # 双通道 top1≈0.0328，再乘 confidence×freshness×importance 因子（0.5–1.0）；
    # 0.010 ≈ "单通道前二且因子不塌"，评测期再校准。
    coverage_min_score: float = 0.010
    budget_floor: float = 0.15          # 剩余 token 预算比例低于它 → 模式降一级
    time_budget_seconds: float | None = None
    forced_mode: str | None = None      # 冒烟/评测钉死模式；仍照常记录 features+reasons


def policy_settings_from_config(config: Mapping[str, Any] | None) -> PolicySettings:
    cfg = dict(config or {})
    forced = cfg.get("forced_mode")
    if forced is not None and forced not in MODES:
        raise ValueError(f"forced_mode 必须是 {MODES} 之一，得到 {forced!r}")
    limits = {m: _mode_limits(m, cfg.get(m)) for m in MODES}
    return PolicySettings(
        enabled=bool(cfg.get("enabled", True)),
        probe_k=_clamp(int(cfg.get("probe_k", 8)), 1, 32),
        limits=limits,
        conflict_similarity=float(cfg.get("conflict_similarity", 0.3)),
        coverage_min_hits=_clamp(int(cfg.get("coverage_min_hits", 2)), 0, 32),
        coverage_min_score=float(cfg.get("coverage_min_score", 0.010)),
        budget_floor=min(1.0, max(0.0, float(cfg.get("budget_floor", 0.15)))),
        time_budget_seconds=(float(cfg["time_budget_seconds"])
                             if cfg.get("time_budget_seconds") is not None else None),
        forced_mode=forced,
    )


# ---- 特征采集 collect_features（五因子的具体化，PLAN §3.4 因子表） -----------
# 五因子特征语义（对应 PLAN §3.4 因子表；判定输入全部可复算、可落盘）：
# - coverage   = hit_count + top_score（探测检索命中覆盖度）
# - freshness  = fresh/review_due/stale_hits（命中记忆时间有效性三态计数）
# - volatility = volatile_hits + max_volatility（命中实体的变化风险）
# - uncertainty = low_confidence_hits + conflict_hits（低置信 / open 冲突；
#   PLAN 的 "unanswered signals" 在本 MVP 映射为：零覆盖→deep、open 冲突相关→deep）
# - budget     = budget_remaining_ratio + elapsed_seconds（由调用方传入：
#   loop 用 `1 - input_tokens/token_budget`，MCP recall 传 1.0）
# 依赖的 wiki 模块导入统一放在文件顶部导入区。

_VOLATILITY_RANK = {"stable": 0, "drifting": 1, "volatile": 2}


@dataclass
class PolicyFeatures:
    """策略判定的输入特征（全部可复算、可落盘）。"""
    hit_count: int = 0
    top_score: float = 0.0
    fresh_hits: int = 0
    review_due_hits: int = 0
    stale_hits: int = 0
    volatile_hits: int = 0
    max_volatility: str = "stable"
    low_confidence_hits: int = 0
    conflict_hits: int = 0
    budget_remaining_ratio: float = 1.0
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "hit_count": self.hit_count, "top_score": round(self.top_score, 6),
            "fresh_hits": self.fresh_hits, "review_due_hits": self.review_due_hits,
            "stale_hits": self.stale_hits, "volatile_hits": self.volatile_hits,
            "max_volatility": self.max_volatility,
            "low_confidence_hits": self.low_confidence_hits,
            "conflict_hits": self.conflict_hits,
            "budget_remaining_ratio": round(self.budget_remaining_ratio, 6),
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }


def _open_conflicts_related(store: WikiStore, question: str,
                            settings: PolicySettings) -> int:
    """与当前问题相关的 open 冲突数：确定性 bigram 相似度 ≥ conflict_similarity。"""
    related = 0
    for conflict in store.list_conflicts(status="open"):
        if token_similarity(conflict.question, question) >= settings.conflict_similarity:
            related += 1
    return related


def collect_features(question: str, store: WikiStore, index: SearchIndex, *,
                     settings: PolicySettings,
                     freshness_settings: FreshnessSettings | None = None,
                     now: Any = None,
                     budget_remaining_ratio: float = 1.0,
                     elapsed_seconds: float = 0.0) -> PolicyFeatures:
    """探测检索 + 五因子特征采集（零模型调用，全部可复算）。

    守卫信号（stale / review_due / 低置信）在探测窗口之外再做一次**全库扫描 +
    相关性门控**：对探测窗口未覆盖的 active 笔记，state ∈ {review_due, stale}
    或 confidence=low 时，须 token_similarity(title + body[:400], question) ≥
    conflict_similarity 才计入对应计数。理由（真实冒烟教训，PLAN §3.4 红线）：
    seed deep run 产出的同题 fresh 笔记会把同题 stale 笔记挤出 probe_k 窗口，
    只看窗口就会把 stale 路由成"无需搜索"的 simple——守卫信号必须全库扫描，
    不得被新鲜内容挤出探测窗口（open 冲突本就是全库扫描 + 同门槛门控，此处
    拉齐口径）。hit_count / top_score / volatile_hits / max_volatility 仍为
    探测窗口口径：覆盖度本来就是窗口量，volatility 是风险信号而非免检红线信号。
    """
    fresh_cfg = freshness_settings or FreshnessSettings()
    matches = index.search(question, k=settings.probe_k)
    feats = PolicyFeatures(budget_remaining_ratio=budget_remaining_ratio,
                           elapsed_seconds=elapsed_seconds)
    top = 0.0
    max_vol = "stable"
    for m in matches:
        note = store.get_note(m.note_id)
        if note is None:
            continue
        top = max(top, m.score)
        state = evaluate_freshness(note, now=now, settings=fresh_cfg)
        if state.state == FRESHNESS_STALE:
            feats.stale_hits += 1
        elif state.state == FRESHNESS_REVIEW_DUE:
            feats.review_due_hits += 1
        else:
            feats.fresh_hits += 1
        if _VOLATILITY_RANK[note.volatility] > _VOLATILITY_RANK[max_vol]:
            max_vol = note.volatility
        if note.volatility == "volatile":
            feats.volatile_hits += 1
        if note.confidence == "low":
            feats.low_confidence_hits += 1
    feats.hit_count = len(matches)
    feats.top_score = top
    feats.max_volatility = max_vol
    # 守卫信号全库扫描（真实冒烟教训，PLAN §3.4 红线）：stale/review_due/低置信
    # 不得被同题 fresh 内容挤出 probe_k 窗口——遍历窗口未覆盖的 active 笔记，
    # 相关性门控通过才计数（与 open 冲突的全库扫描同口径；volatile 不扫，它
    # 是覆盖/风险信号而非免检红线信号）。
    seen = {m.note_id for m in matches}
    for note in store.list_notes(status="active"):
        if note.id in seen:
            continue
        state = evaluate_freshness(note, now=now, settings=fresh_cfg)
        if (state.state not in (FRESHNESS_STALE, FRESHNESS_REVIEW_DUE)
                and note.confidence != "low"):
            continue
        if token_similarity(f"{note.title} {note.body[:400]}",
                            question) < settings.conflict_similarity:
            continue
        if state.state == FRESHNESS_STALE:
            feats.stale_hits += 1
        elif state.state == FRESHNESS_REVIEW_DUE:
            feats.review_due_hits += 1
        if note.confidence == "low":
            feats.low_confidence_hits += 1
    feats.conflict_hits = _open_conflicts_related(store, question, settings)
    return feats


# ---- 确定性判定 decide_mode（守卫红线 + 预算降级 + 理由） ----------------------

@dataclass
class PolicyDecision:
    """decide_mode 的输出：模式 + 判定输入特征 + 理由（落盘留痕）+ 该模式限额。"""
    mode: str
    features: PolicyFeatures
    reasons: list[str]
    limits: ModeLimits
    forced: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode, "forced": self.forced,
            "features": self.features.to_dict(), "reasons": list(self.reasons),
            "limits": {
                "prior_k": self.limits.prior_k,
                "prior_max_chars": self.limits.prior_max_chars,
                "max_fresh_searches": self.limits.max_fresh_searches,
                "max_steps": self.limits.max_steps,
                "subagents": self.limits.subagents,
                "report_style": self.limits.report_style,
                "min_fresh_sources": self.limits.min_fresh_sources,
            },
        }


def _downgrade(mode: str) -> str:
    return {MODE_DEEP: MODE_UPDATE, MODE_UPDATE: MODE_SIMPLE, MODE_SIMPLE: MODE_SIMPLE}[mode]


def decide_mode(features: PolicyFeatures, *,
                settings: PolicySettings) -> PolicyDecision:
    """确定性模式判定。规则顺序即优先级：守卫红线 > 覆盖度 > 预算降级 > forced。

    预算/时间降级受"守卫地板"约束：判定链上任一守卫信号（conflict、零覆盖、stale、
    低置信、review_due、volatile，含覆盖不足路径）在判时，降级下限为 update——
    simple 不做 fresh 搜索，不得把守卫信号路由成"无需搜索"（PLAN §3.4 验收红线）。
    """
    reasons: list[str] = []
    if features.conflict_hits > 0:
        mode = MODE_DEEP
        reasons.append(f"open_conflict_hits={features.conflict_hits} → 冲突需完整研究回路重新调查")
    elif features.hit_count == 0:
        mode = MODE_DEEP
        reasons.append("memory_hit_count=0 → 无记忆覆盖，走完整研究")
    elif features.stale_hits > 0 or features.low_confidence_hits > 0:
        mode = MODE_UPDATE
        reasons.append(f"stale_hits={features.stale_hits}, low_confidence_hits="
                       f"{features.low_confidence_hits} → 旧记忆不可免检，需 fresh verification")
    elif features.review_due_hits > 0 or features.volatile_hits > 0:
        mode = MODE_UPDATE
        reasons.append(f"review_due_hits={features.review_due_hits}, "
                       f"volatile_hits={features.volatile_hits} → 变化风险需少量 fresh 核验")
    elif (features.hit_count >= settings.coverage_min_hits
          and features.top_score >= settings.coverage_min_score):
        mode = MODE_SIMPLE
        reasons.append(f"hit_count={features.hit_count} ≥ {settings.coverage_min_hits} 且 "
                       f"top_score={features.top_score:.4f} ≥ {settings.coverage_min_score} → "
                       "记忆覆盖充分且全部 fresh/stable，直接轻量作答")
    else:
        mode = MODE_UPDATE
        reasons.append(f"覆盖度不足（hit_count={features.hit_count} < "
                       f"{settings.coverage_min_hits} 或 top_score={features.top_score:.4f} < "
                       f"{settings.coverage_min_score}）→ 轻量研究")
    # 守卫地板：守卫信号或覆盖不足在判（mode ≠ simple）→ 降级下限 update（PLAN §3.4 红线）。
    guard_floored = mode != MODE_SIMPLE
    if features.budget_remaining_ratio < settings.budget_floor and mode != MODE_SIMPLE:
        candidate = _downgrade(mode)
        if guard_floored and candidate == MODE_SIMPLE:
            reasons.append(f"budget_remaining_ratio={features.budget_remaining_ratio:.3f} < "
                           f"budget_floor={settings.budget_floor} → 守卫地板：降级止于 update")
            candidate = MODE_UPDATE
        else:
            reasons.append(f"budget_remaining_ratio={features.budget_remaining_ratio:.3f} < "
                           f"budget_floor={settings.budget_floor} → 模式降一级")
        mode = candidate
    if (settings.time_budget_seconds is not None
            and features.elapsed_seconds > settings.time_budget_seconds
            and mode != MODE_SIMPLE):
        candidate = _downgrade(mode)
        if guard_floored and candidate == MODE_SIMPLE:
            reasons.append(f"elapsed_seconds={features.elapsed_seconds:.1f} > "
                           f"time_budget_seconds={settings.time_budget_seconds} → "
                           "守卫地板：降级止于 update")
            candidate = MODE_UPDATE
        else:
            reasons.append(f"elapsed_seconds={features.elapsed_seconds:.1f} > "
                           f"time_budget_seconds={settings.time_budget_seconds} → 模式降一级")
        mode = candidate
    forced = settings.forced_mode is not None
    if forced and settings.forced_mode != mode:
        reasons.append(f"forced_mode={settings.forced_mode} 覆盖规则判定 {mode}")
        mode = settings.forced_mode
    return PolicyDecision(mode=mode, features=features, reasons=reasons,
                          limits=settings.limits[mode], forced=forced)
