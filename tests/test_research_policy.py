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
