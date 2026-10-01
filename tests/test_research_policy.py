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
from researchwiki.wiki.index import SearchIndex, importance_factor
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

def test_default_config_toml_parses_into_settings():
    import tomllib
    from pathlib import Path
    raw = tomllib.loads((Path(__file__).resolve().parents[1] / "config.toml").read_text("utf-8"))
    s = policy_settings_from_config(raw.get("retrieval"))
    assert s.enabled is True
    assert s.limits[MODE_UPDATE].min_fresh_sources == 1


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

def test_budget_downgrade_respects_guard_floor():
    # 守卫信号（stale）在判时，预算降级止于 update：simple 不做 fresh 搜索，
    # 不得把守卫信号路由成"无需搜索"（PLAN §3.4 验收红线）。
    s = policy_settings_from_config({"budget_floor": 0.5})
    d = decide_mode(_feats(stale_hits=1, fresh_hits=2, budget_remaining_ratio=0.0), settings=s)
    assert d.mode == MODE_UPDATE
    assert any("守卫地板" in r for r in d.reasons)

def test_budget_downgrade_pure_coverage_to_simple():
    # 守卫全零且覆盖达标的"覆盖充分"纯路径：simple 不受预算降级影响。
    s = policy_settings_from_config({"budget_floor": 0.5})
    d = decide_mode(_feats(budget_remaining_ratio=0.0), settings=s)
    assert d.mode == MODE_SIMPLE

def test_insufficient_coverage_reason_records_values():
    # 覆盖度不足分支的理由必须嵌入具体数值与阈值（全局红线：决策理由可审计）。
    d = decide_mode(_feats(hit_count=1, top_score=0.005),
                    settings=policy_settings_from_config(None))
    assert d.mode == MODE_UPDATE
    assert "hit_count=1" in d.reasons[0] and "0.0050" in d.reasons[0]


# ---- P3 §2.4：importance 进检索排名因子 ---------------------------------------

def test_importance_factor_scale():
    # 温和乘子：factor = 0.5 + 0.5*value（None 视为中性默认 0.6 → 0.8），越界夹取
    assert importance_factor(1.0) == 1.0
    assert abs(importance_factor(0.3) - 0.65) < 1e-9
    assert abs(importance_factor(None) - 0.8) < 1e-9
    assert abs(importance_factor(0.0) - 0.5) < 1e-9
    assert importance_factor(1.5) == 1.0      # 越界夹取
    assert importance_factor(-1.0) == 0.5

def test_importance_reorders_ranking(tmp_path):
    store = WikiStore(tmp_path / "wiki-data")
    # 同题两条：高重要度 vs 低重要度，其余特征一致（confidence/volatility/created 同默认）
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
    # 旧库迁移路径：先用新代码建库，再把 note_meta 降级成"真实旧 schema"
    # （= index.py 现有 CREATE TABLE note_meta 的列清单，仅缺 importance），
    # 重新打开时守护迁移补列，rebuild 用新代码重写全部行，search 照常命中。
    import sqlite3
    root = tmp_path / "wiki-data"
    store = WikiStore(root)
    note = store.save_note("Zephyr 内存占用约 2KB。", entities=["Zephyr"])
    SearchIndex(root, clock=lambda: NOW).rebuild(store)   # 先按新代码建库（带列）
    conn = sqlite3.connect(root / "index.db")
    # 造一个没有 importance 列的旧版 note_meta（列名/列序与 P2 末版 DDL 一致）
    conn.execute("DROP TABLE note_meta")
    conn.execute(
        "CREATE TABLE note_meta ("
        "note_id TEXT PRIMARY KEY, title TEXT, body TEXT, confidence TEXT, "
        "volatility TEXT, kind TEXT, status TEXT, redirect_to TEXT, superseded_by TEXT, "
        "observed_at TEXT, created TEXT, body_hash TEXT, tombstone INTEGER)"
    )
    conn.execute(
        "INSERT INTO note_meta (note_id, title, body, confidence, volatility, kind, "
        "status, redirect_to, superseded_by, observed_at, created, body_hash, tombstone) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (note.id, "t", "Zephyr 内存占用约 2KB。", "medium", "stable", "knowledge",
         "active", None, None, None, "2026-10-01", "h", 0),
    )
    conn.commit()
    conn.close()
    idx = SearchIndex(root, clock=lambda: NOW)   # 重新打开：初始化即触发守护迁移
    columns = {row[1] for row in idx._conn.execute("PRAGMA table_info(note_meta)")}
    assert "importance" in columns               # 缺列 → ALTER 已补上
    idx.rebuild(store)          # 迁移 + 重建
    hits = idx.search("Zephyr 内存占用", k=3)
    assert hits and hits[0].note_id == note.id
