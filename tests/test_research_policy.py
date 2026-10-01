"""P3 Dynamic Retrieval：确定性 recall 策略测试。"""
import datetime as dt

from researchwiki.loop.research_policy import (
    MODE_DEEP,
    MODE_SIMPLE,
    MODE_UPDATE,
    ModeLimits,
    PolicyFeatures,
    collect_features,
    decide_mode,
    policy_settings_from_config,
)
from researchwiki.wiki.freshness import FreshnessSettings
from researchwiki.wiki.index import SearchIndex
from researchwiki.wiki.store import WikiStore


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


# ---- 特征采集 collect_features（五因子的具体化） ------------------------------

# 带时区的固定时钟：SearchIndex 的 freshness_factor 会拿 clock() 与落盘的
# aware 时间戳直接相减，naive datetime 会抛 TypeError（与 test_wiki 的
# FIXED_NOW 同一约定）。
NOW = dt.datetime(2026, 10, 1, 12, 0, 0, tzinfo=dt.UTC)

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


# ---- 确定性判定 decide_mode（守卫红线 + 预算降级 + 理由） ----------------------

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
