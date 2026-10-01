# P3 · Dynamic Retrieval（budget-aware 确定性检索策略）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让系统对不同问题以不同深度召回记忆——确定性策略按 coverage / freshness / volatility / uncertainty / budget 五因子判定 `simple|update|deep` 模式，决定召回什么、召回多少，且每次判定留痕可解释。

**Architecture:** 新增纯函数策略模块 `loop/research_policy.py`（特征采集 + 确定性判定 + 模式限额表），AgentLoop 在 Prior 注入前探测检索并判定模式，模式限额约束 prior 深度 / fresh search 次数 / 步数 / 子代理 / 报告样式 / 最小新鲜来源；SearchIndex 排名补 importance 因子（PLAN §2.4 "P3 进因子"）；MCP `memory_recall` 从 passthrough 升级为策略入口。不引入分类器，不改 run-metrics 契约。

**Tech Stack:** Python 3.11+ / pytest / SQLite(sqlite-vec, FTS5) / FastMCP / uv。测试零真实网络（ScriptedProvider + MockEmbeddingProvider + MockSearch + 注入时钟）。

**Spec:** `PLAN.md` §3.4 "P3 · Dynamic Retrieval"（line 321 起）+ §2.4 因子表（line 189 起）+ 验收口径（"同题不同模式下成本下降、质量不劣化"）。

## Global Constraints

- **确定性策略先行**，不引入分类器；只有规则策略在评测中明显误判才考虑 cheap classifier（PLAN §3.4）。
- **总闸验收**：同题不同模式下成本下降、质量不劣化；stale / conflict / 低置信旧记忆**不得**被路由为"无需搜索"；每次模式决策必须记录输入特征与理由，不允许只有模式字符串。
- **run-metrics.json 14 字段契约不动**（P2 先例：新增旁路文件 `policy.json`，见 Task 5）。
- **AgentLoop 配置约定**：每个特性一个 `*_config` 构造参数，`None` = 功能关闭（旧行为不变）；由 `server/main.py` 用 `config.get("<section>")` 接线（同 prior/formation/memory_update/verification 模式，`server/main.py:130-149`）。
- **旧数据兼容 / 历史版本永不删除**：index schema 加列必须带迁移路径，`note_index_hash` 扩展字段后靠 drift→rebuild 重建（`index.py:219` docstring 即此约定）。
- **可复算**：成本数字必须能由 `tokens.jsonl`、`run-metrics.json` 重算（不变量 ⑥）；冒烟断言用 `sum_tokens_from_jsonl` 对账。
- **测试零真实网络**；`pyproject.toml` 无 pytest marker，全部走 tmp_path。
- **注释中文**，与 config.toml / 各模块现文风一致；commit 用 conventional commits；**git add 只加显式路径**；每个 Task 交付后 commit+push（用户既定协议）。
- **范围外**（P2 worklog §7 移交项，不在本计划）：①替换门与相似度带关系明文化 ②merge 独立下限 ③F7 loop 写路径 id 形态护栏（P4 触发）。

---

### Task 1: 策略配置与模式限额表（`research_policy.py` 骨架）

**Files:**
- Create: `src/researchwiki/loop/research_policy.py`
- Test: `tests/test_research_policy.py`

**Interfaces:**
- Produces: `MODE_SIMPLE/MODE_UPDATE/MODE_DEEP/MODES` 常量；`ModeLimits`（frozen dataclass：`prior_k, prior_max_chars, max_fresh_searches, max_steps, subagents, report_style, min_fresh_sources`）；`PolicySettings`（frozen dataclass：`enabled, probe_k, limits, conflict_similarity, coverage_min_hits, coverage_min_score, budget_floor, time_budget_seconds, forced_mode`）；`policy_settings_from_config(config: Mapping | None) -> PolicySettings`。后续所有 Task 只经这两个类型读配置。

- [ ] **Step 1: 写失败测试**

```python
"""P3 Dynamic Retrieval：确定性 recall 策略测试。"""
from researchwiki.loop.research_policy import (
    MODE_DEEP, MODE_SIMPLE, MODE_UPDATE, ModeLimits,
    policy_settings_from_config,
)

def test_default_settings_match_plan_ladder():
    s = policy_settings_from_config(None)
    assert s.enabled is True
    assert s.probe_k == 8
    assert s.limits[MODE_SIMPLE].max_fresh_searches == 0
    assert s.limits[MODE_SIMPLE].subagents is False
    assert s.limits[MODE_SIMPLE].report_style == "brief"
    assert s.limits[MODE_UPDATE].max_fresh_searches == 2
    assert s.limits[MODE_UPDATE].min_fresh_sources == 1
    assert s.limits[MODE_DEEP].subagents is True
    assert s.limits[MODE_DEEP].report_style == "full"
    assert s.forced_mode is None

def test_config_overrides_and_clamp():
    s = policy_settings_from_config({
        "probe_k": 3,
        "budget_floor": 0.5,
        "forced_mode": "update",
        "simple": {"prior_k": 2, "max_fresh_searches": 1, "max_steps": 99},
    })
    assert s.probe_k == 3
    assert s.budget_floor == 0.5
    assert s.forced_mode == "update"
    assert s.limits[MODE_SIMPLE].prior_k == 2
    assert s.limits[MODE_SIMPLE].max_fresh_searches == 1
    # 越界值夹取：步数不得超上限、比例夹到 [0,1]
    s2 = policy_settings_from_config({"probe_k": 64, "budget_floor": 7})
    assert s2.probe_k == 32
    assert s2.budget_floor == 1.0

def test_invalid_forced_mode_rejected():
    import pytest
    with pytest.raises(ValueError):
        policy_settings_from_config({"forced_mode": "turbo"})
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_research_policy.py -q`
Expected: FAIL（`ModuleNotFoundError: researchwiki.loop.research_policy`）

- [ ] **Step 3: 最小实现**

```python
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
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_research_policy.py -q`
Expected: PASS（3 个测试）

- [ ] **Step 5: Commit + push**

```bash
git add src/researchwiki/loop/research_policy.py tests/test_research_policy.py
git commit -m "feat(policy): P3 策略配置与模式限额表（simple/update/deep）"
git push
```

---

### Task 2: 特征采集 `collect_features`（五因子的具体化）

**Files:**
- Modify: `src/researchwiki/loop/research_policy.py`（追加）
- Test: `tests/test_research_policy.py`（追加）

**Interfaces:**
- Consumes: Task 1 的 `PolicySettings`；现有 `WikiStore.get_note/list_conflicts`（`store.py:252/534`）、`SearchIndex.search(query, k)`（`index.py:521`）、`evaluate_freshness(note, *, now, settings) -> FreshnessState`（`freshness.py:331`）、`token_similarity(a, b)`（`verification.py:846`，确定性 bigram 余弦）。
- Produces: `PolicyFeatures`（dataclass，含 `to_dict()`）：`hit_count, top_score, fresh_hits, review_due_hits, stale_hits, volatile_hits, max_volatility, low_confidence_hits, conflict_hits, budget_remaining_ratio, elapsed_seconds`；`collect_features(question, store, index, *, settings, freshness_settings=None, now=None, budget_remaining_ratio=1.0, elapsed_seconds=0.0) -> PolicyFeatures`。Task 3/5/6/8 消费。

**特征语义（写进模块 docstring，对应 PLAN §3.4 因子表）：**
- coverage = `hit_count` + `top_score`（探测检索命中覆盖度）
- freshness = `fresh/review_due/stale_hits`（命中记忆时间有效性三态计数）
- volatility = `volatile_hits` + `max_volatility`（命中实体的变化风险）
- uncertainty = `low_confidence_hits` + `conflict_hits`（低置信 / open 冲突；PLAN 的 "unanswered signals" 在本 MVP 映射为：零覆盖→deep、open 冲突相关→deep）
- budget = `budget_remaining_ratio` + `elapsed_seconds`（由调用方传入：loop 用 `1 - input_tokens/token_budget`，MCP recall 传 1.0）

- [ ] **Step 1: 写失败测试**（seed 出 PLAN 要求的五类场景）

```python
import datetime as dt
import pytest
from researchwiki.wiki.store import WikiStore
from researchwiki.wiki.index import SearchIndex
from researchwiki.wiki.freshness import FreshnessSettings
from researchwiki.loop.research_policy import (
    collect_features, policy_settings_from_config,
)

NOW = dt.datetime(2026, 10, 1, 12, 0, 0)

def _seed(store: WikiStore, *, body: str, volatility: str = "stable",
          confidence: str = "high", observed_at: str | None = None,
          entities: list[str] | None = None, title: str = "笔记") -> None:
    store.save_note(body, title=title, entities=entities or ["实体"],
                    volatility=volatility, confidence=confidence,
                    observed_at=observed_at)

def _index(store: WikiStore) -> SearchIndex:
    idx = SearchIndex(store.root, clock=lambda: NOW)
    idx.rebuild(store)
    return idx

def test_features_on_fresh_stable_hits(tmp_path):
    store = WikiStore(tmp_path / "wiki-data")
    _seed(store, body="Zephyr 框架的内存占用约为 2KB，发布于 2026-09。")
    _seed(store, body="Zephyr 的许可证是 Apache 2.0。")
    feats = collect_features("Zephyr 框架的内存占用是多少", store, _index(store),
                             settings=policy_settings_from_config(None),
                             freshness_settings=FreshnessSettings(), now=NOW)
    assert feats.hit_count >= 1 and feats.top_score > 0
    assert feats.stale_hits == 0 and feats.low_confidence_hits == 0
    assert feats.max_volatility == "stable"
    assert feats.budget_remaining_ratio == 1.0

def test_features_count_stale_and_low_confidence(tmp_path):
    store = WikiStore(tmp_path / "wiki-data")
    _seed(store, body="Zephyr 内存占用约 2KB。", volatility="volatile",
          observed_at="2026-01-01T00:00:00+00:00")   # 半衰期 30 天 → stale
    _seed(store, body="Zephyr 许可证是 Apache 2.0。", confidence="low")
    feats = collect_features("Zephyr 内存占用", store, _index(store),
                             settings=policy_settings_from_config(None),
                             freshness_settings=FreshnessSettings(), now=NOW)
    assert feats.stale_hits >= 1
    assert feats.low_confidence_hits >= 1

def test_features_conflict_overlap(tmp_path):
    store = WikiStore(tmp_path / "wiki-data")
    _seed(store, body="Zephyr 内存占用约 2KB。")
    store.save_conflict("Zephyr 的内存占用到底是多少",
                        {"text": "2KB", "observed_at": "2026-01-01"},
                        {"text": "4KB", "observed_at": "2026-09-01"})
    feats = collect_features("Zephyr 的内存占用是多少", store, _index(store),
                             settings=policy_settings_from_config(None),
                             freshness_settings=FreshnessSettings(), now=NOW)
    assert feats.conflict_hits == 1

def test_features_empty_wiki(tmp_path):
    store = WikiStore(tmp_path / "wiki-data")
    feats = collect_features("量子纠错的表面码阈值", store, _index(store),
                             settings=policy_settings_from_config(None),
                             freshness_settings=FreshnessSettings(), now=NOW)
    assert feats.hit_count == 0 and feats.top_score == 0.0
    assert feats.max_volatility == "stable" and feats.conflict_hits == 0

def test_features_to_dict_roundtrip(tmp_path):
    store = WikiStore(tmp_path / "wiki-data")
    _seed(store, body="Zephyr 内存占用约 2KB。")
    feats = collect_features("Zephyr 内存占用", store, _index(store),
                             settings=policy_settings_from_config(None),
                             freshness_settings=FreshnessSettings(), now=NOW,
                             budget_remaining_ratio=0.4, elapsed_seconds=1.5)
    d = feats.to_dict()
    assert d["budget_remaining_ratio"] == 0.4 and d["elapsed_seconds"] == 1.5
    assert set(d) >= {"hit_count", "top_score", "fresh_hits", "review_due_hits",
                      "stale_hits", "volatile_hits", "max_volatility",
                      "low_confidence_hits", "conflict_hits"}
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_research_policy.py -q`
Expected: FAIL（`ImportError: cannot import name 'collect_features'`）

- [ ] **Step 3: 最小实现**（追加到 `research_policy.py`）

```python
from dataclasses import dataclass  # 文件顶部已有；以下为新追加段
from researchwiki.wiki.freshness import (
    FRESHNESS_REVIEW_DUE, FRESHNESS_STALE, FreshnessSettings,
    evaluate_freshness,
)
from researchwiki.wiki.index import SearchIndex
from researchwiki.wiki.store import WikiStore
from researchwiki.wiki.verification import token_similarity

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
    """探测检索 + 五因子特征采集（零模型调用，全部可复算）。"""
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
    feats.conflict_hits = _open_conflicts_related(store, question, settings)
    return feats
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_research_policy.py -q`
Expected: PASS（8 个测试）

- [ ] **Step 5: Commit + push**

```bash
git add src/researchwiki/loop/research_policy.py tests/test_research_policy.py
git commit -m "feat(policy): 五因子特征采集 collect_features（覆盖/时效/波动/不确定/预算）"
git push
```

---

### Task 3: 确定性判定 `decide_mode`（守卫红线 + 预算降级 + 理由）

**Files:**
- Modify: `src/researchwiki/loop/research_policy.py`（追加）
- Test: `tests/test_research_policy.py`（追加）

**Interfaces:**
- Consumes: Task 1 `PolicySettings`/`ModeLimits`、Task 2 `PolicyFeatures`。
- Produces: `PolicyDecision`（dataclass：`mode, features, reasons: list[str], limits: ModeLimits, forced: bool` + `to_dict()`）；`decide_mode(features, *, settings) -> PolicyDecision`。Task 5/6/8 消费。

**判定规则（确定性、有序、写进 docstring）：**
1. **守卫（验收红线，优先级最高）**：`conflict_hits > 0` → deep；`hit_count == 0` → deep；`stale_hits > 0 或 low_confidence_hits > 0` → 至少 update；`review_due_hits > 0 或 volatile_hits > 0` → 至少 update。
2. **覆盖度**：`hit_count ≥ coverage_min_hits 且 top_score ≥ coverage_min_score` 且无上述降级信号 → simple；否则 update。
3. **预算降级**：`budget_remaining_ratio < budget_floor`（或 `elapsed_seconds > time_budget_seconds`，当配置了时间预算）→ 模式下移一级（deep→update→simple），理由记录。
4. **forced_mode**：钉死模式但**照常计算并记录 features + reasons**（冒烟/评测用）。

- [ ] **Step 1: 写失败测试**

```python
from researchwiki.loop.research_policy import (
    PolicyFeatures, decide_mode, policy_settings_from_config,
)

def _feats(**kw) -> PolicyFeatures:
    base = dict(hit_count=3, top_score=0.03, fresh_hits=3, review_due_hits=0,
                stale_hits=0, volatile_hits=0, max_volatility="stable",
                low_confidence_hits=0, conflict_hits=0,
                budget_remaining_ratio=1.0, elapsed_seconds=0.0)
    base.update(kw)
    return PolicyFeatures(**base)

def test_fresh_stable_covered_routes_simple():
    d = decide_mode(_feats(), settings=policy_settings_from_config(None))
    assert d.mode == MODE_SIMPLE
    assert d.reasons and d.limits is not None

def test_stale_never_routes_simple_guard():
    d = decide_mode(_feats(stale_hits=1, fresh_hits=2),
                    settings=policy_settings_from_config(None))
    assert d.mode == MODE_UPDATE
    assert any("stale" in r for r in d.reasons)

def test_low_confidence_never_routes_simple_guard():
    d = decide_mode(_feats(low_confidence_hits=1),
                    settings=policy_settings_from_config(None))
    assert d.mode == MODE_UPDATE

def test_conflict_routes_deep():
    d = decide_mode(_feats(conflict_hits=1),
                    settings=policy_settings_from_config(None))
    assert d.mode == MODE_DEEP

def test_empty_wiki_routes_deep():
    d = decide_mode(_feats(hit_count=0, top_score=0.0, fresh_hits=0),
                    settings=policy_settings_from_config(None))
    assert d.mode == MODE_DEEP

def test_volatile_hits_route_update():
    d = decide_mode(_feats(volatile_hits=1, max_volatility="volatile"),
                    settings=policy_settings_from_config(None))
    assert d.mode == MODE_UPDATE

def test_budget_downgrade_one_level():
    s = policy_settings_from_config({"budget_floor": 0.5})
    d = decide_mode(_feats(conflict_hits=1, budget_remaining_ratio=0.3), settings=s)
    assert d.mode == MODE_UPDATE      # deep → update
    assert any("budget" in r for r in d.reasons)

def test_forced_mode_pins_but_logs_features():
    s = policy_settings_from_config({"forced_mode": "simple"})
    d = decide_mode(_feats(conflict_hits=1), settings=s)
    assert d.mode == MODE_SIMPLE and d.forced is True
    assert d.features.conflict_hits == 1 and d.reasons  # 照常留痕

def test_decision_to_dict_shape():
    d = decide_mode(_feats(), settings=policy_settings_from_config(None))
    payload = d.to_dict()
    assert payload["mode"] == MODE_SIMPLE and payload["forced"] is False
    assert isinstance(payload["features"], dict) and payload["reasons"]
    assert payload["limits"]["max_fresh_searches"] == 0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_research_policy.py -q`
Expected: FAIL（`ImportError: cannot import name 'decide_mode'`）

- [ ] **Step 3: 最小实现**

```python
@dataclass
class PolicyDecision:
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
    """确定性模式判定。规则顺序即优先级：守卫红线 > 覆盖度 > 预算降级 > forced。"""
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
        reasons.append("覆盖度不足（命中数或分数低于 simple 门槛）→ 轻量研究")
    if features.budget_remaining_ratio < settings.budget_floor and mode != MODE_SIMPLE:
        reasons.append(f"budget_remaining_ratio={features.budget_remaining_ratio:.3f} < "
                       f"budget_floor={settings.budget_floor} → 模式降一级")
        mode = _downgrade(mode)
    if (settings.time_budget_seconds is not None
            and features.elapsed_seconds > settings.time_budget_seconds
            and mode != MODE_SIMPLE):
        reasons.append(f"elapsed_seconds={features.elapsed_seconds:.1f} > "
                       f"time_budget_seconds={settings.time_budget_seconds} → 模式降一级")
        mode = _downgrade(mode)
    forced = settings.forced_mode is not None
    if forced and settings.forced_mode != mode:
        reasons.append(f"forced_mode={settings.forced_mode} 覆盖规则判定 {mode}")
        mode = settings.forced_mode
    return PolicyDecision(mode=mode, features=features, reasons=reasons,
                          limits=settings.limits[mode], forced=forced)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_research_policy.py -q`
Expected: PASS（17 个测试）

- [ ] **Step 5: Commit + push**

```bash
git add src/researchwiki/loop/research_policy.py tests/test_research_policy.py
git commit -m "feat(policy): decide_mode 确定性判定（守卫红线/覆盖度/预算降级/理由留痕）"
git push
```

---

### Task 4: importance 进检索排名因子（PLAN §2.4 "P3 进因子"）

**Files:**
- Modify: `src/researchwiki/wiki/index.py`（schema、`note_index_hash:219`、upsert/select、`search:584` 打分）
- Test: `tests/test_research_policy.py`（追加排名测试）

**Interfaces:**
- Consumes: `NoteMeta.importance: float | None`（`frontmatter.py`，P1 已落库）。
- Produces: `SearchIndex.search` 最终分 = RRF 融合分 × confidence × freshness × **importance**；模块级 `importance_factor(importance: float | None) -> float` 与常量 `IMPORTANCE_FLOOR=0.5`、`IMPORTANCE_DEFAULT=0.6`；`note_meta` 表新增 `importance REAL` 列（带旧库 ALTER 迁移）。

**因子公式**（温和乘子，避免 importance 线性压制相关性）：
`factor = IMPORTANCE_FLOOR + (1 - IMPORTANCE_FLOOR) * value`，`value = importance if importance is not None else IMPORTANCE_DEFAULT`。
量级：importance 1.0→1.0，0.3→0.65，None→0.8，0.0→0.5。

- [ ] **Step 1: 写失败测试**

```python
from researchwiki.wiki.index import importance_factor

def test_importance_factor_scale():
    assert importance_factor(1.0) == 1.0
    assert abs(importance_factor(0.3) - 0.65) < 1e-9
    assert abs(importance_factor(None) - 0.8) < 1e-9
    assert abs(importance_factor(0.0) - 0.5) < 1e-9
    assert importance_factor(1.5) == 1.0      # 越界夹取
    assert importance_factor(-1.0) == 0.5

def test_importance_reorders_ranking(tmp_path):
    store = WikiStore(tmp_path / "wiki-data")
    # 同题两条：高重要度 vs 低重要度，其余特征一致
    store.save_note("Zephyr 内存占用约 2KB，来自官方文档。", title="高价值",
                    entities=["Zephyr"], importance=1.0)
    store.save_note("Zephyr 内存占用大约 2KB。", title="低价值",
                    entities=["Zephyr"], importance=0.0)
    idx = SearchIndex(store.root, clock=lambda: NOW)
    idx.rebuild(store)
    hits = idx.search("Zephyr 内存占用", k=2)
    assert hits[0].note_id != hits[1].note_id
    high = next(n for n in store.list_notes() if n.title == "高价值")
    assert hits[0].note_id == high.id   # 高重要度排前

def test_old_index_without_importance_column_rebuilds(tmp_path):
    # 旧库迁移路径：先建库（新代码会带列），模拟旧 schema 的方式是：
    # 建库后手工 DROP COLUMN 不可行（SQLite 限制），改为直接断言
    # "schema 缺列时 search 不崩、rebuild 修复" —— 实现：在 SearchIndex 初始化
    # 里用 ALTER TABLE ... ADD COLUMN 守护迁移；测试通过手工建旧表验证。
    import sqlite3
    root = tmp_path / "wiki-data"
    store = WikiStore(root)
    note = store.save_note("Zephyr 内存占用约 2KB。", entities=["Zephyr"])
    root.mkdir(exist_ok=True)
    conn = sqlite3.connect(root / "index.db")
    # 造一个没有 importance 列的旧版 note_meta
    conn.execute("DROP TABLE note_meta")
    conn.execute("CREATE TABLE note_meta (id TEXT PRIMARY KEY, status TEXT, "
                 "title TEXT, body_hash TEXT, confidence TEXT, volatility TEXT, "
                 "kind TEXT, redirect_to TEXT, superseded_by TEXT, "
                 "observed_at TEXT, created TEXT, tombstone INTEGER)")
    conn.execute("INSERT INTO note_meta VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                 (note.id, "active", "t", "h", "high", "stable", "knowledge",
                  None, None, None, "2026-10-01", 0))
    conn.commit(); conn.close()
    idx = SearchIndex(root, clock=lambda: NOW)
    idx.rebuild(store)          # 迁移 + 重建
    hits = idx.search("Zephyr 内存占用", k=3)
    assert hits and hits[0].note_id == note.id
```

（注意：旧表列名以 `index.py` 现有 `CREATE TABLE note_meta` 实际 DDL 为准——实现者先读该 DDL 再改写测试里的旧表构造，保证列清单与真实旧 schema 一致、仅缺 `importance`。）

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_research_policy.py -q`
Expected: FAIL（`ImportError: cannot import name 'importance_factor'`）

- [ ] **Step 3: 实现**（`index.py` 四处改动）

1. 模块常量与函数（放在 `CONFIDENCE_FACTOR` 附近，`index.py:47-51` 区域）：

```python
IMPORTANCE_FLOOR = 0.5    # importance 因子下限：factor = floor + (1-floor)*value
IMPORTANCE_DEFAULT = 0.6  # 旧数据/显式写入未标 importance 时的中性值


def importance_factor(importance: float | None) -> float:
    """P3 进因子（PLAN §2.4）：温和乘子，避免 importance 线性压制相关性。"""
    if importance is None:
        value = IMPORTANCE_DEFAULT
    else:
        value = max(0.0, min(1.0, importance))
    return IMPORTANCE_FLOOR + (1.0 - IMPORTANCE_FLOOR) * value
```

2. `note_meta` 建表 DDL 加 `importance REAL` 列；初始化后执行守护迁移：

```python
try:
    conn.execute("ALTER TABLE note_meta ADD COLUMN importance REAL")
except sqlite3.OperationalError:
    pass  # 列已存在（新库或已迁移）
```

3. `note_index_hash`（`index.py:219`）把 `importance` 加进哈希清单——旧行存储哈希随之失配，`index_drift` → rebuild 自动重建（docstring 里预留的正是这一步）。
4. upsert 写入 `importance`（取 `note.meta.importance`）；`search()` 候选 SELECT 带出该列；最终打分（`index.py:584`）改为：

```python
score = fused * CONFIDENCE_FACTOR[confidence] * freshness * importance_factor(importance)
```

- [ ] **Step 4: 跑测试确认通过 + 全量回归**

Run: `uv run pytest tests/test_research_policy.py tests/test_wiki.py tests/test_index_freshness.py tests/test_prior.py -q`
Expected: PASS（rebuild 语义、drift、prior 打分不受破坏）

- [ ] **Step 5: Commit + push**

```bash
git add src/researchwiki/wiki/index.py tests/test_research_policy.py
git commit -m "feat(index): importance 进检索排名因子（P3 §2.4）+ 旧库 ALTER 迁移与 hash 扩展重建"
git push
```

---

### Task 5: AgentLoop 接线（探测→判定→留痕→模式限额）

**Files:**
- Modify: `src/researchwiki/loop/agent_loop.py`（`__init__:502` 加参数；`_events:878` Prior 段前置判定；`_register_research_tools:274` 加搜索帽；报告段加样式与来源核验）
- Modify: `src/researchwiki/server/main.py:130-149`（接线 `retrieval_config`、`freshness_settings`）
- Test: `tests/test_loop_policy.py`（新建）

**Interfaces:**
- Consumes: Task 1–3 全部产物；现有 `retrieve_priors`（`prior.py:133`）、`RunContext.search_calls`、`_write_state(phase)`（`agent_loop.py:851`）、`SourcePool.numbered()`。
- Produces:
  - `AgentLoop(..., retrieval_config: Mapping | None = None, freshness_settings=None)`；`retrieval_config=None` → 行为与现状逐字节等价（模式概念不存在）。
  - run_dir 新增 `policy.json`：`{"trace_id", "decided_at", "mode", "forced", "features", "reasons", "limits", "outcome": {"fresh_source_count", "min_fresh_sources", "min_fresh_sources_satisfied"}}`（outcome 在报告结束后回填；fresh_source_count = SourcePool 计数，Prior URL 不计入）。
  - state.md 新增一行 `retrieval: mode=<mode> forced=<bool> reasons=<首条理由>`。
  - 模式限额生效点：prior `k`/`max_chars` 用 `limits.prior_k/prior_max_chars`（覆盖 `[prior]` 段值）；`web_search` 调用达 `limits.max_fresh_searches` 后返回 `{"error": "search_budget_exhausted", "mode", "limit"}`（不再调用 provider，不计 fresh_search_count）；`limits.subagents=False` 时不注册 `dispatch_research`；`effective_max_steps = limits.max_steps`；报告提示词按 `report_style` 追加后缀（brief="限 200 字以内，直接回答，仅列关键结论"；standard/full = 现状不变）；`min_fresh_sources > 0` 且报告后 `fresh_source_count < min` → report.md 末尾追加 `> ⚠️ 新鲜来源不足：本轮 fresh source N 篇，低于该模式要求下限 M（诚实边界，非引用缺失错误）。`

- [ ] **Step 1: 写失败测试**（新建 `tests/test_loop_policy.py`，ScriptedProvider 模式沿用 `tests/test_loop.py:30-46` 的 `text_turn/tool_turn/call` 助手）

```python
"""P3：AgentLoop 模式判定接线测试（零网络，ScriptedProvider + MockSearch）。"""
import json
import pytest
from researchwiki.loop.agent_loop import AgentLoop
from researchwiki.llm.provider import ScriptedProvider, StreamEvent, TokenUsage
from researchwiki.llm.accounting import TokenAccountant
from researchwiki.tools import MockSearch

# turn 助手与断言所需的QUESTION/turns 构造，照抄 test_loop.py 现有 helper：
# text_turn(text), tool_turn(call(name, args)) —— 以 test_loop.py 实际签名为准。


def _loop(tmp_path, question, retrieval_config, turns, **kw):
    return AgentLoop(
        question,
        router=None, llm_config=None,
        accountant=TokenAccountant(tmp_path / "tokens.jsonl"),
        search_provider=MockSearch(),
        wiki_root=tmp_path / "wiki-data",
        run_dir=tmp_path / "run",
        retrieval_config=retrieval_config,
        **kw,
    ), turns


def test_retrieval_disabled_keeps_legacy_behavior(tmp_path):
    # retrieval_config=None：run_dir 不出现 policy.json，行为与 P2 一致
    ...

def test_policy_decision_logged_with_features_and_reasons(tmp_path):
    # 预置 store 两条 fresh stable 笔记 → 期待 simple；断言 policy.json 存在且
    # mode/forced/features.hit_count/reasons 非空、limits.max_fresh_searches == 0
    ...

def test_simple_mode_blocks_fresh_search(tmp_path):
    # forced_mode=simple；脚本 plan 轮含 web_search 调用 → 工具返回
    # search_budget_exhausted 且 fresh_search_count == 0（读 run-metrics.json）
    ...

def test_stale_memory_never_routed_simple(tmp_path):
    # seed 一条 volatile + observed_at 很旧的笔记（stale）→ 自动判定不得为 simple
    ...

def test_policy_json_outcome_backfilled_after_report(tmp_path):
    # forced_mode=update、min_fresh_sources=1、MockSearch 有结果但脚本不调用搜索
    # → outcome.min_fresh_sources_satisfied is False 且 report.md 末尾有"新鲜来源不足"
    ...
```

（测试体在实现时补全：turn 脚本按 `test_loop.py` 现有 helper 写；simple 模式脚本 = plan 纯文本轮 + act 纯文本轮 + report 轮；`_loop` 里 `router` 参数按 `test_loop.py` 现有构造方式传 ScriptedProvider 包装，此处以现有测试为准，不新造机制。）

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_loop_policy.py -q`
Expected: FAIL（`TypeError: AgentLoop got an unexpected keyword argument 'retrieval_config'`）

- [ ] **Step 3: 实现 AgentLoop 接线**（五处，全部在 `agent_loop.py`）

1. `__init__` 签名追加 `retrieval_config=None, freshness_settings=None`；`retrieval_config` 非 None 时 `self.policy_settings = policy_settings_from_config(retrieval_config)`，否则 `None`。
2. `_events()` Prior 段（`agent_loop.py:893` 附近）之前插入判定（复用同一段已 `ensure_index_fresh` 的 store/index）：

```python
policy_decision = None
if self.policy_settings is not None:
    from researchwiki.loop.research_policy import collect_features, decide_mode
    feats = collect_features(
        self.question, store, prior_index,
        settings=self.policy_settings, freshness_settings=self.freshness_settings,
        budget_remaining_ratio=max(0.0, 1.0 - self.input_tokens / max(1, self.token_budget)),
        elapsed_seconds=self.clock() - self._t0,
    )
    policy_decision = decide_mode(feats, settings=self.policy_settings)
    limits = policy_decision.limits
    (run_dir / "policy.json").write_text(
        json.dumps({"trace_id": self.trace_id, "decided_at": <iso now>,
                    **policy_decision.to_dict(),
                    "outcome": None}, ensure_ascii=False, indent=2), encoding="utf-8")
else:
    limits = None
```

（`self._t0`：`__init__` 里 `self.clock()` 初值；iso 时间用现有 run 的时间源约定。判定后照 `_write_state` 的现有写法在 state.md 追加 `retrieval:` 一行。）
3. Prior 调用改带模式限额：`retrieve_priors(question, store, prior_index, k=limits.prior_k if limits else prior_settings.k, max_chars=limits.prior_max_chars if limits else prior_settings.max_chars)`。
4. 生效步数：act 循环的停止判据（`agent_loop.py:946`）把 `self.max_steps` 换成 `effective_max_steps = limits.max_steps if limits else self.max_steps`。
5. `_register_research_tools`（`agent_loop.py:274`）：接受可选 `limits`；`limits.subagents is False` → 不注册 `dispatch_research`；`web_search` 包装层在调用 provider 前检查 `ctx.search_calls >= limits.max_fresh_searches` → 返回 `{"error": "search_budget_exhausted", "mode": policy_decision.mode, "limit": limits.max_fresh_searches}`（不调用 provider、`search_calls` 不增长 fresh 计数口径按 run-metrics 现有统计点为准）。
6. 报告段（`agent_loop.py:1120`）：`report_style == "brief"` 时在 report system 提示后追加一句简报约束（`_REPORT_STYLE_SUFFIX` 常量放 `research_policy.py`）；报告写盘后回填 `policy.json` 的 `outcome`（读 SourcePool 计数，`min_fresh_sources > 0` 且不足时向 report.md 追加缺口提示块）。

`server/main.py`（`130-149` 区域）按现有约定接线：

```python
retrieval_config=config.get("retrieval"),
freshness_settings=freshness_from_config(config),
```

- [ ] **Step 4: 跑测试确认通过 + 旧路径回归**

Run: `uv run pytest tests/test_loop_policy.py tests/test_loop.py tests/test_loop_prior.py tests/test_server.py tests/test_cold_warm_smoke.py -q`
Expected: PASS（`retrieval_config=None` 时旧行为不变，包括 cold_warm 冒烟断言）

- [ ] **Step 5: Commit + push**

```bash
git add src/researchwiki/loop/agent_loop.py src/researchwiki/server/main.py tests/test_loop_policy.py
git commit -m "feat(loop): P3 模式判定接线——探测/留痕 policy.json/模式限额（搜索帽、子代理门、步数、报告样式）"
git push
```

---

### Task 6: MCP `memory_recall` 升级为 budget-aware 入口

**Files:**
- Modify: `src/researchwiki/mcp_server/service.py`（`recall:763`，替换 passthrough）
- Test: `tests/test_mcp_recall.py`（新建）

**Interfaces:**
- Consumes: Task 1–3 产物；`WikiService`（`service.py:362`，持有完整 config）；`search` 内部的索引同步路径（`_index_is_stale:467` / `_sync_index:487`）。
- Produces: `recall(query, k=5, kind=None, include_tombstones=False)` 返回 payload 在现有字段基础上新增：`"mode"`（判定模式，替换 `"passthrough"`）、`"policy": {"features": {...}, "reasons": [...], "forced": bool}`；结果条数 = `min(k, limits.prior_k)`（simple 收窄召回宽度，update/deep 尊重调用方 k）；每条结果维持现有字段（note_id/title/snippet/score/match_type/redirected_from/redirect_note/status/observed_at/kind/importance）。MCP 工具签名不变（`server.py:461`）。

- [ ] **Step 1: 写失败测试**（`tests/test_mcp_recall.py`，fixture 沿用 `tests/test_mcp.py:50-66` 三件套风格）

```python
"""P3：memory_recall budget-aware 升级测试。"""
import datetime as dt
from researchwiki.wiki.store import WikiStore

NOW = dt.datetime(2026, 10, 1, 12, 0, 0)

def test_recall_reports_mode_features_reasons(service, store):
    store.save_note("Zephyr 内存占用约 2KB，来源官方文档。", entities=["Zephyr"])
    payload = service.recall("Zephyr 内存占用", k=5)
    assert payload["ok"] is True
    assert payload["mode"] in {"simple", "update", "deep"}
    assert payload["policy"]["reasons"]
    assert "hit_count" in payload["policy"]["features"]

def test_recall_simple_mode_narrows_results(service, store):
    for i in range(5):
        store.save_note(f"Zephyr 知识条目 {i}：官方规格 {i}。", entities=["Zephyr"])
    payload = service.recall("Zephyr 知识条目", k=5)   # 全 fresh stable → simple
    if payload["mode"] == "simple":
        assert payload["count"] <= 3                    # simple limits.prior_k=3
    else:
        assert payload["count"] <= 5

def test_recall_with_conflict_routes_deep_and_still_returns_results(service, store):
    store.save_note("Zephyr 内存占用约 2KB。", entities=["Zephyr"])
    store.save_conflict("Zephyr 内存占用是多少",
                        {"text": "2KB"}, {"text": "4KB"})
    payload = service.recall("Zephyr 内存占用是多少", k=5)
    assert payload["mode"] == "deep"
    assert payload["count"] >= 1

def test_mcp_tool_surface_unchanged(server):
    # 走 in-process MCP client 调 memory_recall，确认工具签名与 payload 透传
    ...  # 按 test_mcp.py 现有 call_tool 助手调用并断言 payload["mode"] 存在
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_mcp_recall.py -q`
Expected: FAIL（`payload` 无 `"policy"` 键）

- [ ] **Step 3: 实现**（`service.py` `recall` 替换 passthrough 段）

```python
def recall(self, query, k=5, kind=None, include_tombstones=False) -> dict[str, Any]:
    """budget-aware 动态召回（P3）：先判定模式，再按模式收窄召回宽度。

    MCP 侧没有 loop 的 token 账本，budget_remaining_ratio 恒 1.0
    （预算因子只在 loop 内生效）；判定照常带 features + reasons 透传。
    """
    payload = self.search(query, k=k, kind=kind, include_tombstones=include_tombstones)
    retrieval_cfg = (self._config or {}).get("retrieval")
    settings = policy_settings_from_config(retrieval_cfg) if retrieval_cfg else None
    if settings is None or settings.enabled is False:
        payload["mode"] = "passthrough"
        return payload
    store, index = self._store, <复用 search 内部已同步的 index 实例>
    features = collect_features(query, store, index, settings=settings,
                                freshness_settings=freshness_from_config(self._config))
    decision = decide_mode(features, settings=settings)
    width = min(int(payload.get("count", k)) if payload.get("results") else k,
                decision.limits.prior_k)
    payload["results"] = payload["results"][:width]
    payload["count"] = len(payload["results"])
    payload["k"] = width
    payload["mode"] = decision.mode
    payload["policy"] = {"features": decision.features.to_dict(),
                         "reasons": decision.reasons, "forced": decision.forced}
    return payload
```

（`self._config`/index 复用方式以 `service.py` 现有字段名为准——实现者先读 `search` 的索引获取路径再落笔；`freshness_from_config` 接收完整 config dict。）

- [ ] **Step 4: 跑测试确认通过 + MCP 回归**

Run: `uv run pytest tests/test_mcp_recall.py tests/test_mcp.py tests/test_server.py -q`
Expected: PASS

- [ ] **Step 5: Commit + push**

```bash
git add src/researchwiki/mcp_server/service.py tests/test_mcp_recall.py
git commit -m "feat(mcp): memory_recall 升级 budget-aware——模式判定+召回宽度收窄+policy 透传"
git push
```

---

### Task 7: config.toml `[retrieval]` 段（模式预算/阈值/最小新鲜来源数）

**Files:**
- Modify: `config.toml`（`[memory_update]` 段后追加）
- Test: `tests/test_research_policy.py`（追加一条"默认 config.toml 可解析"测试）

**Interfaces:**
- Consumes: Task 1 `policy_settings_from_config` 的键位。
- Produces: config.toml 的 `[retrieval]` 段（缺省段 = enabled + 默认值，同 `[prior]` 风格）。

- [ ] **Step 1: 写失败测试**

```python
def test_default_config_toml_parses_into_settings():
    import tomllib
    from pathlib import Path
    raw = tomllib.loads((Path(__file__).resolve().parents[1] / "config.toml").read_text("utf-8"))
    s = policy_settings_from_config(raw.get("retrieval"))
    assert s.enabled is True
    assert s.limits[MODE_UPDATE].min_fresh_sources == 1
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_research_policy.py -q`
Expected: FAIL（当前 config.toml 无 `[retrieval]` 段也能过——此测试锁"注释里声明的默认值与代码一致"，先跑确认当前通过、加入段后再跑确认不回归；若 Step 1 直接通过则该 Step 记录 PASS 并继续）

- [ ] **Step 3: 追加配置段**（含中文注释，风格对齐现有段）

```toml
[retrieval]
# 动态检索策略（P3 / RQ3）：不是每个问题都召回同样多、同样深的记忆。
# run 开始时先做一次探测检索（probe_k 条），从命中里算确定性特征——
# coverage（命中数/分数）、freshness（fresh/review_due/stale 计数）、
# volatility（volatile 命中）、uncertainty（低置信命中 + open 冲突相关度）、
# budget（剩余 token 比例）——按守卫红线 > 覆盖度 > 预算降级的顺序判定模式：
#   simple → 直接回答或轻量检索（不 fresh 搜索，只读少量 Prior）
#   update → 读取旧记忆 + 少量 fresh verification
#   deep   → 预算完整的研究 loop（与 P2 现状等价）
# 红线：stale / conflict / 低置信旧记忆不得被路由为 simple（无需搜索）。
# 每次判定写入 run_dir/policy.json（features + reasons + limits），可复算可审计。
# enabled = false 为逃生阀：行为回到 P2 单一 deep 路径，不判定不留痕。
enabled = true
# 探测检索条数（特征来源，不直接等于注入条数）
probe_k = 8
# 判定阈值（确定性规则，评测期校准）
conflict_similarity = 0.3
coverage_min_hits = 2
coverage_min_score = 0.010
budget_floor = 0.15
# time_budget_seconds = 600   # 可选：判定耗时超它降一级；缺省不启用
# forced_mode = "simple"      # 可选：冒烟/评测钉死模式（仍记录 features+reasons）

[retrieval.simple]
prior_k = 3
prior_max_chars = 1200
max_fresh_searches = 0
max_steps = 3
subagents = false
report_style = "brief"
min_fresh_sources = 0

[retrieval.update]
prior_k = 5
prior_max_chars = 4000
max_fresh_searches = 2
max_steps = 6
subagents = false
report_style = "standard"
min_fresh_sources = 1

[retrieval.deep]
prior_k = 5
prior_max_chars = 4000
max_fresh_searches = 4
max_steps = 12
subagents = true
report_style = "full"
min_fresh_sources = 0
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_research_policy.py -q`
Expected: PASS

- [ ] **Step 5: Commit + push**

```bash
git add config.toml tests/test_research_policy.py
git commit -m "feat(config): [retrieval] 段——模式预算/判定阈值/最小新鲜来源数"
git push
```

---

### Task 8: `scripts/adaptive_smoke.py` 同题跨模式成本/质量对照

**Files:**
- Create: `scripts/adaptive_smoke.py`
- Test: `tests/test_adaptive_smoke.py`（新建，沿用 `tests/test_cold_warm_smoke.py` 对脚本的测试方式）

**Interfaces:**
- Consumes: `cold_warm_smoke.py` 的 `ProviderStack`/`build_mock_stack`/`run_once`（`scripts/cold_warm_smoke.py:258/282/395`，同目录 import）；Task 5 的 `retrieval_config`（`forced_mode` 钉模式）。
- Produces: CLI `--provider mock|config --question --wiki-root --out --json --env-file --config`；输出 JSONL 每行：`{"mode", "question", "trace_id", "wiki_root", "run_dir", "provider_mode", "model", "metrics"(run-metrics 全量), "policy"(policy.json 的 mode/forced/features/reason), "report_chars", "citation_coverage", "input_tokens_checked"(sum_tokens_from_jsonl 对账值)}`。`--modes` 缺省 `simple,update,deep,auto`。

**矩阵语义**：同一问题先跑一次"播种" run（forced_mode=deep，产生记忆），把播种后的 wiki-data 目录 `shutil.copytree` 成每模式独立副本再跑各模式（warm 起跑、互不污染）；`auto` = 不带 forced_mode 的自然判定。验证函数 `verify_rows(rows)` 硬断言：
1. 每行 policy 含 features 与非空 reasons（验收：不允许裸模式字符串）；
2. mock 下 `simple.input_tokens < deep.input_tokens` 且 `simple.fresh_search_count == 0`；
3. `simple.fresh_search_count <= update.fresh_search_count <= deep.fresh_search_count`；
4. 播种含 stale 记忆时 `auto.policy.mode != simple`（守卫红线复验）；
5. `input_tokens_checked == metrics.input_tokens`（tokens.jsonl 对账，不变量 ⑥）。

- [ ] **Step 1: 写失败测试**（`tests/test_adaptive_smoke.py`：importlib 按路径加载脚本，跑 mock 矩阵于 tmp_path，断言 JSONL 行数=4、verify_rows 通过、seeded stale 场景 auto 不落 simple）

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_adaptive_smoke.py -q`
Expected: FAIL（脚本不存在）

- [ ] **Step 3: 实现脚本**（结构照 `cold_warm_smoke.py`：argparse → build stacks → 播种 → copytree → 逐模式 `run_once`（注入 `retrieval_config={"enabled": True, "forced_mode": m}` 或 auto 不注入）→ `verify_rows` → 写 `smoke_out/adaptive_{provider}_{ts}.jsonl`；mock 脚本 turn 按 simple/update/deep 分别构造：simple 两轮纯文本（plan/act）+ report 轮；update 在 plan 轮带一次 web_search 工具调用；deep 沿用 cold_warm 的完整脚本）

- [ ] **Step 4: 跑测试确认通过 + 真实模型冒烟（可选，有 key 时）**

Run: `uv run pytest tests/test_adaptive_smoke.py -q`（必过）
Run: `uv run python scripts/adaptive_smoke.py --provider mock --out smoke_out`（必过，证据落 smoke_out/）
Run: `uv run python scripts/adaptive_smoke.py --provider config --out smoke_out`（真实对照，成本记录进 worklog；无 key 或超预算时明确跳过并在 worklog 写明）

- [ ] **Step 5: Commit + push**

```bash
git add scripts/adaptive_smoke.py tests/test_adaptive_smoke.py
git commit -m "feat(smoke): adaptive_smoke 同题跨模式成本/质量对照（simple/update/deep/auto）"
git push
```

---

### Task 9: 终验 + worklog 交付记录

**Files:**
- Modify: `docs/agent-worklog.md`（追加 P3 交付段）

- [ ] **Step 1: 全量回归**

Run: `uv run pytest -q`
Expected: 全绿（P2 基线 703 + P3 新增）

- [ ] **Step 2: mock 冒烟证据复核**

Run: `uv run python scripts/adaptive_smoke.py --provider mock --out smoke_out`
Expected: verify_rows 全过，`smoke_out/adaptive_mock_*.jsonl` 生成；抽查 simple/deep 两行的 `input_tokens` 与 `fresh_search_count` 方向正确。

- [ ] **Step 3: worklog 追加**：P3 交付段记录——交付物清单（对照 PLAN §3.4 五项）、验收逐条勾（决策留痕 / simple p50 低于 deep / 守卫红线 / 质量不劣化基线说明：mock 矩阵 + 真实 run 数据或"未跑真实"的诚实边界）、实测数字（含样本数标注）、范围外三项重申。

- [ ] **Step 4: Commit + push**

```bash
git add docs/agent-worklog.md
git commit -m "docs(worklog): P3 Dynamic Retrieval 交付记录与验收证据"
git push
```

---

## Self-Review 记录

- **Spec 覆盖**：PLAN §3.4 五项交付物 → Task 1–3（research_policy.py）、Task 5（agent_loop 模式限额/报告/验证要求）、Task 7（config.toml 三类配置）、Task 8（adaptive_smoke）；tests/test_research_policy.py 五类用例 → Task 2/3（stable/fresh、volatile/stale、conflict、空库、无答案=零覆盖）。§2.4 importance 进因子 → Task 4；budget 进因子 → 模式限额（Task 1/5）+ 预算降级（Task 3）。MCP recall 钩子位 → Task 6。验收"记录理由"→ policy.json（Task 5）；"简单 p50 低于 deep"→ Task 8 断言 2；"stale/conflict 不得免检路由"→ Task 3 守卫 + Task 8 断言 4；"质量不低于无路由 baseline"→ retrieval.enabled=false 即 baseline（Task 5 旧行为等价性测试）+ Task 8 对照。
- **占位符扫描**：Task 5/6 部分测试体标注"按现有 helper 实现"与"以现有字段名为准"——这是对现有代码事实的引用约束（helper 名/字段名以仓库为准），非待定设计；实现者第一步先读指定行号。其余步骤均含实际代码/内容。
- **类型一致性**：`ModeLimits` 字段名在 Task 1 定义、Task 3 `to_dict`、Task 5/6/8 消费处一致；`PolicyFeatures` 字段在 Task 2 定义、Task 3 消费一致；`policy_settings_from_config` 签名各处一致。
