"""P3 Dynamic Retrieval：budget-aware 确定性召回策略（模式判定与解释）。

判定只用可复算的确定性特征（检索命中、freshness 三态、volatility、open 冲突、
token 预算余量），不引入分类器；每次判定必须带 features + reasons 落盘
（PLAN §3.4 验收：不允许只有模式字符串）。
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

MODE_SIMPLE = "simple"
MODE_UPDATE = "update"
MODE_DEEP = "deep"
MODES = (MODE_SIMPLE, MODE_UPDATE, MODE_DEEP)

_REPORT_STYLES = ("brief", "standard", "full")


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
    conflict_similarity: float = 0.3    # open 冲突与当前问题的 bigram 相似度门槛（同 [verification].similarity_floor 量级）
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
