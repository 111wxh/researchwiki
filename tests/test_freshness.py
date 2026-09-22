"""时间感知的记忆有效性计算测试（P2-A / RQ2 freshness 侧）。

覆盖 PLAN §3.3 验收第 1 条（"stable、drifting、volatile 三类记忆在固定时钟下
得到可预测的 freshness 状态"）与 task-8 简报的六条语义规则：

- 规则 1 基准时间选择（observed_at → created → 未知年龄）
- 规则 2 显式 expiration 优先（valid_until 过期 / valid_from 未生效 / 非 ISO 宽容）
- 规则 3 volatility 衰减（时点-状态对照表：0 / 半衰期 / 2 个半衰期）
- 规则 4 kind 差异接口（默认不区分；per_kind 覆盖；[freshness.user] 段）
- 规则 5 decay 与 index.freshness_factor 数值口径一致（独立实现 + 防漂移断言）
- 规则 6 时钟可注入、全确定性

全部零网络、零模型调用；固定时钟 NOW，断言精确到数值。
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from researchwiki.wiki import freshness as fr
from researchwiki.wiki.frontmatter import NoteMeta, dump, parse
from researchwiki.wiki.index import DEFAULT_HALF_LIFE_DAYS as INDEX_HALF_LIFE_DAYS
from researchwiki.wiki.index import freshness_factor, wiki_settings
from researchwiki.wiki.store import Note, WikiStore

NOW = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
HL = {"volatile": 30.0, "drifting": 90.0}
# 哨兵：区分"没给 observed_at"（按 age_days 反推）与"显式给 None"（不带观察时间）
_UNSET: object = object()


def make_note(
    *,
    note_id: str = "N-0001",
    volatility: str = "drifting",
    kind: str = "knowledge",
    age_days: float | None = 0.0,
    observed_at: object = _UNSET,
    created: str | None = "",
    status: str = "active",
    **extra: object,
) -> Note:
    """构造一条测试笔记（时间基准相对固定时钟 NOW，全部字段可显式覆盖）。

    - observed_at 不给时按 ``age_days`` 反推（``age_days=None`` = 不带观察时间）；
    - 显式传 ``observed_at=None`` 表示这条笔记没有观察时间（回退 created 或未知年龄）。
    """
    if observed_at is _UNSET:
        observed_at = None if age_days is None else (NOW - timedelta(days=age_days)).isoformat()
    meta = NoteMeta(
        id=note_id,
        title="时效测试",
        volatility=volatility,
        kind=kind,
        status=status,
        observed_at=observed_at,
        created=("" if created is None else created),
        **extra,  # type: ignore[arg-type]
    )
    return Note(id=note_id, title="时效测试", body="正文。", meta=meta)


# ---- 规则 3：三类 volatility 的时点-状态对照表 --------------------------------


@pytest.mark.parametrize(
    ("volatility", "age_days", "state", "decay", "half_life"),
    [
        # stable：不衰减，任何年龄都 fresh，half_life 为 None
        ("stable", 0.0, fr.FRESHNESS_FRESH, 1.0, None),
        ("stable", 90.0, fr.FRESHNESS_FRESH, 1.0, None),
        ("stable", 3650.0, fr.FRESHNESS_FRESH, 1.0, None),
        # drifting（半衰期 90 天）：1 个半衰期 = 0.5 → review_due（阈值闭区间），2 个 = 0.25 → stale
        ("drifting", 0.0, fr.FRESHNESS_FRESH, 1.0, 90.0),
        ("drifting", 45.0, fr.FRESHNESS_FRESH, 0.5**0.5, 90.0),
        ("drifting", 90.0, fr.FRESHNESS_REVIEW_DUE, 0.5, 90.0),
        ("drifting", 90.1, fr.FRESHNESS_REVIEW_DUE, 0.5 ** (90.1 / 90.0), 90.0),
        ("drifting", 180.0, fr.FRESHNESS_STALE, 0.25, 90.0),
        ("drifting", 900.0, fr.FRESHNESS_STALE, 0.5**10, 90.0),
        # volatile（半衰期 30 天）
        ("volatile", 0.0, fr.FRESHNESS_FRESH, 1.0, 30.0),
        ("volatile", 15.0, fr.FRESHNESS_FRESH, 0.5**0.5, 30.0),
        ("volatile", 30.0, fr.FRESHNESS_REVIEW_DUE, 0.5, 30.0),
        ("volatile", 45.0, fr.FRESHNESS_REVIEW_DUE, 0.5**1.5, 30.0),
        ("volatile", 60.0, fr.FRESHNESS_STALE, 0.25, 30.0),
    ],
)
def test_state_table_by_volatility_and_age(
    volatility: str, age_days: float, state: str, decay: float, half_life: float | None
) -> None:
    """固定时钟下三类记忆在 0 / 半衰期 / 2 个半衰期等时点得到可预测状态与 decay。"""
    result = fr.evaluate_freshness(make_note(volatility=volatility, age_days=age_days), now=NOW)
    assert result.state == state
    assert result.decay == pytest.approx(decay)
    assert result.half_life_days == half_life
    assert result.age_days == pytest.approx(age_days)
    # 每条理由都引用具体数值（年龄 / decay / 阈值 / 半衰期）
    assert any(f"年龄 {age_days:.1f} 天" in reason for reason in result.reasons)
    assert any(f"decay {decay:.3f}" in reason for reason in result.reasons)


def test_category_reason_lines_are_explicit() -> None:
    """判定行必须写出命中的规则与阈值（可审计）。"""
    fresh = fr.evaluate_freshness(make_note(volatility="volatile", age_days=1.0), now=NOW)
    due = fr.evaluate_freshness(make_note(volatility="volatile", age_days=30.0), now=NOW)
    stale = fr.evaluate_freshness(make_note(volatility="volatile", age_days=60.0), now=NOW)
    stable = fr.evaluate_freshness(make_note(volatility="stable", age_days=9999.0), now=NOW)
    assert any("> review_due_ratio 0.50 → fresh" in r for r in fresh.reasons)
    assert any("≤ review_due_ratio 0.50" in r and "→ review_due" in r for r in due.reasons)
    assert any("≤ stale_ratio 0.25 → stale" in r for r in stale.reasons)
    assert any("不衰减 → fresh" in r for r in stable.reasons)


def test_non_positive_half_life_means_no_decay() -> None:
    """配置给出 <= 0 的半衰期 = 不衰减（与 index.freshness_factor 同语义）。"""
    settings = fr.FreshnessSettings(half_life_days={"volatile": 0.0, "drifting": -1.0})
    note = make_note(volatility="volatile", age_days=3650.0)
    result = fr.evaluate_freshness(note, now=NOW, settings=settings)
    assert result.state == fr.FRESHNESS_FRESH
    assert result.decay == 1.0 and result.half_life_days is None


# ---- 规则 1：基准时间选择与"未知年龄" ----------------------------------------


def test_base_time_prefers_observed_at_then_created() -> None:
    observed = fr.evaluate_freshness(
        make_note(volatility="volatile", observed_at=(NOW - timedelta(days=60)).isoformat(),
                  created=(NOW - timedelta(days=1)).isoformat()),
        now=NOW,
    )
    assert observed.age_days == pytest.approx(60.0)
    assert any("时间基准 observed_at" in r for r in observed.reasons)

    fallback = fr.evaluate_freshness(
        make_note(volatility="volatile", observed_at=None,
                  created=(NOW - timedelta(days=60)).isoformat()),
        now=NOW,
    )
    assert fallback.age_days == pytest.approx(60.0)
    assert fallback.state == fr.FRESHNESS_STALE
    assert any("时间基准 created" in r for r in fallback.reasons)


def test_unparseable_observed_at_falls_back_to_created() -> None:
    """observed_at 写了但不是 ISO → 回退 created（与 index 同口径）。"""
    result = fr.evaluate_freshness(
        make_note(volatility="volatile", observed_at="上周三",
                  created=(NOW - timedelta(days=60)).isoformat()),
        now=NOW,
    )
    assert result.age_days == pytest.approx(60.0)
    assert any("时间基准 created" in r for r in result.reasons)


@pytest.mark.parametrize("volatility", ["stable", "drifting", "volatile"])
def test_missing_base_time_is_review_due(volatility: str) -> None:
    """无时间基准 → age 0.0、state review_due（不判 fresh 也不判 stale）。"""
    result = fr.evaluate_freshness(
        make_note(volatility=volatility, age_days=None, observed_at=None, created=""), now=NOW
    )
    assert result.state == fr.FRESHNESS_REVIEW_DUE
    assert result.age_days == 0.0
    assert result.decay == 1.0
    assert any("缺少时间基准" in r and "建议人工确认" in r for r in result.reasons)


def test_missing_base_time_with_unparseable_created() -> None:
    result = fr.evaluate_freshness(
        make_note(volatility="volatile", age_days=None, observed_at=None, created="某天"), now=NOW
    )
    assert result.state == fr.FRESHNESS_REVIEW_DUE
    assert any("缺少时间基准" in r for r in result.reasons)


def test_future_observed_at_clamps_age_to_zero() -> None:
    """时间基准在未来（时钟回拨 / 预置断言）→ 年龄按 0.0 计、不衰减。"""
    result = fr.evaluate_freshness(
        make_note(volatility="volatile", age_days=-30.0), now=NOW
    )
    assert result.age_days == 0.0
    assert result.decay == 1.0
    assert result.state == fr.FRESHNESS_FRESH


# ---- 规则 2：显式 expiration / valid_from 优先 -------------------------------


def test_valid_until_expired_is_stale_regardless_of_volatility() -> None:
    """valid_until 过期 → 直接 stale，stable 也不例外，且不看 decay。"""
    for volatility in ("stable", "drifting", "volatile"):
        result = fr.evaluate_freshness(
            make_note(
                volatility=volatility,
                age_days=0.0,
                valid_until=(NOW - timedelta(days=1)).isoformat(),
            ),
            now=NOW,
        )
        assert result.state == fr.FRESHNESS_STALE
        assert result.decay == 1.0  # 年龄 0：decay 仍是纯函数值，状态由 valid_until 决定
        assert any("valid_until" in r and "已过期" in r and "→ stale" in r for r in result.reasons)


def test_valid_until_boundary_is_exclusive() -> None:
    """now == valid_until 视为仍在有效期内（只有 now > valid_until 才过期）。"""
    exact = fr.evaluate_freshness(
        make_note(volatility="volatile", age_days=0.0, valid_until=NOW.isoformat()), now=NOW
    )
    assert exact.state == fr.FRESHNESS_FRESH
    one_second_past = fr.evaluate_freshness(
        make_note(
            volatility="volatile",
            age_days=0.0,
            valid_until=(NOW - timedelta(seconds=1)).isoformat(),
        ),
        now=NOW,
    )
    assert one_second_past.state == fr.FRESHNESS_STALE


def test_valid_until_in_future_keeps_normal_decay_classification() -> None:
    result = fr.evaluate_freshness(
        make_note(
            volatility="volatile",
            age_days=60.0,
            valid_until=(NOW + timedelta(days=30)).isoformat(),
        ),
        now=NOW,
    )
    assert result.state == fr.FRESHNESS_STALE  # 未过期，按 decay 0.25 判 stale
    assert any("≤ stale_ratio 0.25 → stale" in r for r in result.reasons)


def test_valid_from_in_future_is_review_due_not_fresh() -> None:
    """valid_from 未到 → review_due（未生效，不判 fresh），即便 decay 很高。"""
    result = fr.evaluate_freshness(
        make_note(
            volatility="volatile",
            age_days=0.0,
            valid_from=(NOW + timedelta(days=7)).isoformat(),
        ),
        now=NOW,
    )
    assert result.state == fr.FRESHNESS_REVIEW_DUE
    assert any("valid_from" in r and "未到生效时间" in r and "→ review_due" in r
               for r in result.reasons)


def test_valid_from_in_past_does_not_affect_state() -> None:
    ok = fr.evaluate_freshness(
        make_note(volatility="volatile", age_days=0.0,
                  valid_from=(NOW - timedelta(days=7)).isoformat()),
        now=NOW,
    )
    assert ok.state == fr.FRESHNESS_FRESH
    assert not any("未到生效时间" in r for r in ok.reasons)


def test_valid_until_beats_valid_from_when_both_decide() -> None:
    """两个显式字段同时命中时，失效优先（规则 2 内部顺序：expiration 先行）。"""
    result = fr.evaluate_freshness(
        make_note(
            volatility="volatile",
            age_days=0.0,
            valid_from=(NOW + timedelta(days=7)).isoformat(),
            valid_until=(NOW - timedelta(days=7)).isoformat(),
        ),
        now=NOW,
    )
    assert result.state == fr.FRESHNESS_STALE


@pytest.mark.parametrize(
    ("window", "state"),
    [
        ({"valid_until": "expired"}, fr.FRESHNESS_STALE),
        ({"valid_from": "future"}, fr.FRESHNESS_REVIEW_DUE),
        ({}, fr.FRESHNESS_REVIEW_DUE),
    ],
)
def test_expiry_rules_beat_missing_base(window: dict[str, str], state: str) -> None:
    """规则 2 命中时优先于"缺基准"（R1 → R2 的优先级由此锁定）。

    - 过期的 valid_until + 无基准 → stale（有效期不需要年龄基准）；
    - 未到的 valid_from + 无基准 → review_due（同上）；
    - 两者都没有 + 无基准 → review_due，且带"建议人工确认"提示。
    前两种情况下理由里不得再出现"建议人工确认"（否则与判定行自相牵制）。
    """
    window_values = {
        "expired": (NOW - timedelta(days=1)).isoformat(),
        "future": (NOW + timedelta(days=7)).isoformat(),
    }
    kwargs = {key: window_values[value] for key, value in window.items()}
    result = fr.evaluate_freshness(
        make_note(volatility="volatile", age_days=None, observed_at=None, created="", **kwargs),
        now=NOW,
    )
    assert result.state == state
    assert result.age_days == 0.0 and result.decay == 1.0
    assert result.half_life_days == 30.0
    if window:
        assert any("无时间基准" in r and "不影响有效期判定" in r for r in result.reasons)
        assert not any("建议人工确认" in r for r in result.reasons)
        assert not any("缺少时间基准 → review_due" in r for r in result.reasons)
    else:
        assert any("建议人工确认" in r for r in result.reasons)
    if "valid_until" in kwargs:
        assert any("已过期" in r and "→ stale" in r for r in result.reasons)
    if "valid_from" in kwargs:
        assert any("未到生效时间" in r and "→ review_due" in r for r in result.reasons)


def test_expired_with_base_does_not_advise_manual_review_base() -> None:
    """有基准时过期判定同样不夹带缺基准提示（只有有效期判定行）。"""
    result = fr.evaluate_freshness(
        make_note(
            volatility="volatile", age_days=10.0,
            valid_until=(NOW - timedelta(days=1)).isoformat(),
        ),
        now=NOW,
    )
    assert result.state == fr.FRESHNESS_STALE
    assert any("时间基准 observed_at" in r for r in result.reasons)
    assert not any("建议人工确认" in r or "无时间基准" in r for r in result.reasons)


def test_valid_window_unparseable_is_treated_as_declared_nothing() -> None:
    """valid_from/valid_until 非 ISO → 不抛异常、按未声明处理，理由里说明。"""
    result = fr.evaluate_freshness(
        make_note(volatility="volatile", age_days=0.0, valid_from="下周", valid_until="待定"),
        now=NOW,
    )
    assert result.state == fr.FRESHNESS_FRESH
    assert any("valid_from=下周 不是合法 ISO 时间" in r for r in result.reasons)
    assert any("valid_until=待定 不是合法 ISO 时间" in r for r in result.reasons)


def test_expired_note_keeps_decay_and_age_observable() -> None:
    """过期不影响 age/decay 的纯数值（判定依据是有效期，但数值仍可审计）。"""
    result = fr.evaluate_freshness(
        make_note(
            volatility="volatile",
            age_days=10.0,
            valid_until=(NOW - timedelta(days=1)).isoformat(),
        ),
        now=NOW,
    )
    assert result.age_days == pytest.approx(10.0)
    assert result.decay == pytest.approx(0.5 ** (10.0 / 30.0))
    assert result.state == fr.FRESHNESS_STALE


# ---- 规则 4：kind 差异接口（默认不区分，可覆盖） ------------------------------


def test_default_settings_do_not_distinguish_kind() -> None:
    """默认参数下 user 与 knowledge 走同一套规则（测试锁死"默认等价于不区分"）。"""
    user = fr.evaluate_freshness(
        make_note(volatility="volatile", kind="user", age_days=20.0), now=NOW
    )
    knowledge = fr.evaluate_freshness(
        make_note(volatility="volatile", kind="knowledge", age_days=20.0), now=NOW
    )
    assert (user.state, user.decay, user.half_life_days) == (
        knowledge.state,
        knowledge.decay,
        knowledge.half_life_days,
    )
    # 显式传默认 settings 与不传 settings 逐字段一致
    explicit = fr.evaluate_freshness(
        make_note(volatility="volatile", kind="user", age_days=20.0),
        now=NOW,
        settings=fr.FreshnessSettings(),
    )
    assert explicit == user
    # per_kind 缺省为空 dict
    assert fr.FreshnessSettings().per_kind == {}


def test_per_kind_overrides_only_that_kind() -> None:
    """per_kind 覆盖 review_due_ratio：同一 decay 下 user 判 review_due、knowledge 仍 fresh。"""
    settings = fr.FreshnessSettings(per_kind={"user": {"review_due_ratio": 0.7}})
    decay_ok_age = 20.0  # decay = 0.5 ** (20/30) ≈ 0.630 > 0.5 且 <= 0.7
    user = fr.evaluate_freshness(
        make_note(volatility="volatile", kind="user", age_days=decay_ok_age),
        now=NOW,
        settings=settings,
    )
    knowledge = fr.evaluate_freshness(
        make_note(volatility="volatile", kind="knowledge", age_days=decay_ok_age),
        now=NOW,
        settings=settings,
    )
    assert user.decay == pytest.approx(knowledge.decay)
    assert user.state == fr.FRESHNESS_REVIEW_DUE
    assert knowledge.state == fr.FRESHNESS_FRESH
    # 未被覆盖的参数（stale_ratio）保持默认
    assert settings.for_kind("user").stale_ratio == 0.25
    assert settings.for_kind("experience") is settings  # 无覆盖的 kind 零开销复用 self


def test_per_kind_half_life_override() -> None:
    """per_kind 可覆盖半衰期：user 记忆用更短的半衰期，decay 随之更快衰减。"""
    settings = fr.FreshnessSettings(per_kind={"user": {"half_life_days": {"volatile": 10.0}}})
    user = fr.evaluate_freshness(
        make_note(volatility="volatile", kind="user", age_days=10.0), now=NOW, settings=settings
    )
    knowledge = fr.evaluate_freshness(
        make_note(volatility="volatile", kind="knowledge", age_days=10.0),
        now=NOW,
        settings=settings,
    )
    assert user.half_life_days == 10.0 and user.decay == pytest.approx(0.5)
    assert knowledge.half_life_days == 30.0 and knowledge.decay == pytest.approx(0.5 ** (1 / 3))
    # 覆盖项只应用一层：生效参数里不再带 per_kind（防嵌套递归）
    assert settings.for_kind("user").per_kind == {}


def test_from_config_reads_kind_sections() -> None:
    """[freshness] / [freshness.user] 两段都能读；缺省段 = 全默认。"""
    config = {
        "freshness": {
            "review_due_ratio": 0.6,
            "stale_ratio": 0.3,
            "half_life_days": {"volatile": 15},
            "user": {"review_due_ratio": 0.9},
        }
    }
    settings = fr.from_config(config)
    assert settings.review_due_ratio == 0.6 and settings.stale_ratio == 0.3
    # half_life_days 在默认表上 merge（未写的 drifting 保留默认 90）
    assert settings.half_life_days == {"volatile": 15.0, "drifting": 90.0}
    assert settings.per_kind == {"user": {"review_due_ratio": 0.9}}
    user = settings.for_kind("user")
    assert (user.review_due_ratio, user.stale_ratio) == (0.9, 0.3)


# ---- 规则 5：与 index.freshness_factor 的口径一致性 ---------------------------


def _factor(note: Note) -> float:
    return freshness_factor(
        note.meta.volatility,
        note.meta.observed_at,
        note.meta.created,
        half_life_days=HL,
        now=NOW,
    )


@pytest.mark.parametrize(
    ("volatility", "age_days", "created"),
    [
        ("stable", 0.0, ""),
        ("stable", 3650.0, ""),
        ("drifting", 0.0, ""),
        ("drifting", 90.0, ""),
        ("drifting", 180.0, ""),
        ("volatile", 15.0, ""),
        ("volatile", 30.0, ""),
        ("volatile", 60.0, ""),
        ("volatile", -5.0, ""),  # 基准在未来
        ("volatile", None, (NOW - timedelta(days=60)).isoformat()),  # 回退 created
        ("volatile", None, ""),  # 无基准
        ("volatile", None, "不是时间"),  # 不可解析
    ],
)
def test_decay_matches_index_freshness_factor(
    volatility: str, age_days: float | None, created: str
) -> None:
    """同一组 note，FreshnessState.decay 与 index.freshness_factor 结果一致（防漂移）。"""
    note = make_note(volatility=volatility, age_days=age_days, created=created)
    state = fr.evaluate_freshness(note, now=NOW, settings=fr.FreshnessSettings(half_life_days=HL))
    expected = _factor(note)
    assert state.decay == pytest.approx(expected)
    if state.half_life_days is None:
        assert expected == 1.0
    else:
        # 统一公式口径：decay = 0.5 ** (age / half)，age <= 0 → 1.0
        assert state.decay == pytest.approx(
            1.0 if state.age_days <= 0 else 0.5 ** (state.age_days / state.half_life_days)
        )


def test_default_half_life_table_matches_index() -> None:
    """默认半衰期表与 index.DEFAULT_HALF_LIFE_DAYS 逐值一致（两份常量不许漂移）。"""
    assert dict(fr.DEFAULT_HALF_LIFE_DAYS) == dict(INDEX_HALF_LIFE_DAYS)


def test_freshness_module_does_not_import_index() -> None:
    """计算层不依赖检索层：freshness 模块源码的任何 import 行都不出现 index。"""
    assert fr.__file__ is not None
    lines = [line.strip() for line in Path(fr.__file__).read_text(encoding="utf-8").splitlines()]
    imports = [line for line in lines if line.startswith(("import ", "from "))]
    assert imports  # 至少要有 import 行，避免空列表假阳性
    assert not any("index" in line for line in imports)


# ---- 规则 6：时钟注入与确定性 -------------------------------------------------


def test_clock_injection_is_required_for_determinism() -> None:
    """同一 (note, now, settings) 恒得同一结果；不传 now 时用真实时钟且不报错。"""
    note = make_note(volatility="volatile", age_days=30.0)
    first = fr.evaluate_freshness(note, now=NOW)
    second = fr.evaluate_freshness(note, now=NOW)
    assert first == second

    live_now = datetime.now(UTC)
    live = fr.evaluate_freshness(
        make_note(volatility="volatile", observed_at=live_now.isoformat())
    )
    assert live.state == fr.FRESHNESS_FRESH  # 刚发生：真实时钟下也必然是 fresh
    assert live.age_days < 1.0


def test_naive_now_is_interpreted_as_utc() -> None:
    naive = NOW.replace(tzinfo=None)
    aware = fr.evaluate_freshness(make_note(volatility="volatile", age_days=30.0), now=NOW)
    result = fr.evaluate_freshness(make_note(volatility="volatile", age_days=30.0), now=naive)
    assert result == aware


def test_naive_timestamp_in_frontmatter_is_utc() -> None:
    """无时区的 frontmatter 时间戳按 UTC 解释（与 index / mcp 同约定）。"""
    note = make_note(
        volatility="volatile", age_days=None, observed_at=None, created="2026-05-02T12:00:00"
    )
    result = fr.evaluate_freshness(note, now=NOW)
    assert result.age_days == pytest.approx(30.0)
    assert result.state == fr.FRESHNESS_REVIEW_DUE


def test_evaluate_does_not_mutate_note() -> None:
    note = make_note(volatility="volatile", age_days=60.0, valid_until="下周")
    before = dump(note.meta.to_dict(), note.body)
    fr.evaluate_freshness(note, now=NOW)
    assert dump(note.meta.to_dict(), note.body) == before


# ---- 队列与统计 --------------------------------------------------------------


def test_queue_orders_stale_first_then_age_desc() -> None:
    notes = [
        make_note(note_id="N-0001", volatility="volatile", age_days=15.0),  # fresh
        make_note(note_id="N-0002", volatility="volatile", age_days=30.0),  # review_due
        make_note(note_id="N-0003", volatility="volatile", age_days=60.0),  # stale
        make_note(note_id="N-0004", volatility="drifting", age_days=900.0),  # stale 更老
        make_note(note_id="N-0005", volatility="stable", age_days=1.0),  # fresh
    ]
    queue = fr.freshness_queue(notes, now=NOW)
    assert [s.note_id for s in queue] == ["N-0004", "N-0003", "N-0002"]
    assert [s.state for s in queue] == [
        fr.FRESHNESS_STALE,
        fr.FRESHNESS_STALE,
        fr.FRESHNESS_REVIEW_DUE,
    ]


def test_queue_tiebreak_is_note_id() -> None:
    """同级同龄时按 note_id 升序，保证输出确定。"""
    notes = [
        make_note(note_id="N-0009", volatility="volatile", age_days=60.0),
        make_note(note_id="N-0002", volatility="volatile", age_days=60.0),
    ]
    assert [s.note_id for s in fr.freshness_queue(notes, now=NOW)] == ["N-0002", "N-0009"]


def test_queue_excludes_fresh_and_unknown_age_notes_are_included() -> None:
    notes = [
        make_note(note_id="N-0001", volatility="volatile", age_days=1.0),  # fresh
        make_note(note_id="N-0002", volatility="stable", age_days=None, created=""),  # 未知年龄
    ]
    queue = fr.freshness_queue(notes, now=NOW)
    assert [s.note_id for s in queue] == ["N-0002"]
    assert queue[0].state == fr.FRESHNESS_REVIEW_DUE


def test_queue_respects_injected_settings() -> None:
    settings = fr.FreshnessSettings(per_kind={"user": {"review_due_ratio": 0.9}})
    notes = [
        make_note(note_id="N-0001", volatility="volatile", kind="user", age_days=5.0),
        make_note(note_id="N-0002", volatility="volatile", kind="knowledge", age_days=5.0),
    ]
    assert [s.note_id for s in fr.freshness_queue(notes, now=NOW, settings=settings)] == ["N-0001"]


def test_freshness_counts() -> None:
    states = [
        fr.evaluate_freshness(make_note(volatility="volatile", age_days=1.0), now=NOW),
        fr.evaluate_freshness(make_note(volatility="volatile", age_days=30.0), now=NOW),
        fr.evaluate_freshness(make_note(volatility="volatile", age_days=60.0), now=NOW),
        fr.evaluate_freshness(make_note(note_id="N-0004", volatility="stable", age_days=0.0),
                              now=NOW),
    ]
    counts = fr.freshness_counts(states)
    assert counts == {"fresh": 2, "review_due": 1, "stale": 1}
    assert fr.freshness_counts([]) == {"fresh": 0, "review_due": 0, "stale": 0}


def test_state_to_dict_is_json_shaped() -> None:
    state = fr.evaluate_freshness(make_note(volatility="volatile", age_days=60.0), now=NOW)
    payload = state.to_dict()
    assert payload["note_id"] == "N-0001" and payload["state"] == "stale"
    assert payload["half_life_days"] == 30.0
    assert isinstance(payload["reasons"], list) and payload["reasons"]


# ---- from_config 宽容回退 ----------------------------------------------------


def test_from_config_defaults_and_tolerance() -> None:
    assert fr.from_config(None) == fr.FreshnessSettings()
    assert fr.from_config({}) == fr.FreshnessSettings()
    # 直接给段（无 "freshness" 键）= 把整个 mapping 当段读
    assert fr.from_config({"review_due_ratio": 0.5, "stale_ratio": 0.25}) == fr.FreshnessSettings()
    # 非法值一律宽容回退，不抛异常
    settings = fr.from_config(
        {
            "freshness": {
                "review_due_ratio": "0.8",  # 字符串数字：可转则收
                "stale_ratio": "不是数",
                "half_life_days": {"volatile": "bad", "drifting": 10, "stable": None},
                "user": ["不是映射"],  # 非映射的 kind 段忽略
            }
        }
    )
    assert settings.review_due_ratio == 0.8
    assert settings.stale_ratio == fr.DEFAULT_STALE_RATIO
    assert settings.half_life_days == {"volatile": 30.0, "drifting": 10.0}
    assert settings.per_kind == {}
    # 比率越界夹取；stale_ratio 不得大于 review_due_ratio（否则判定退化）
    clamped = fr.FreshnessSettings(review_due_ratio=1.5, stale_ratio=0.9)
    assert clamped.review_due_ratio == 1.0 and clamped.stale_ratio == 0.9
    inverted = fr.FreshnessSettings(review_due_ratio=0.2, stale_ratio=0.9)
    assert inverted.stale_ratio == 0.2
    # 布尔不算数值（bool 是 int 子类，必须排除）
    booleans = fr.FreshnessSettings(review_due_ratio=True)
    assert booleans.review_due_ratio == fr.DEFAULT_REVIEW_DUE_RATIO


def test_from_config_reads_wiki_half_life_fallback() -> None:
    """只给项目既有的 [wiki].half_life_days 时，freshness 必须用它（不能退回默认 30/90）。"""
    config = {
        "wiki": {"fts_tokenizer": "trigram", "half_life_days": {"volatile": 15, "drifting": 45}}
    }
    settings = fr.from_config(config)
    assert settings.half_life_days == {"volatile": 15.0, "drifting": 45.0}
    # 判定确实按 [wiki] 的半衰期算：volatile 15 天 = 恰好一个半衰期
    state = fr.evaluate_freshness(
        make_note(volatility="volatile", age_days=15.0), now=NOW, settings=settings
    )
    assert state.half_life_days == 15.0
    assert state.decay == pytest.approx(0.5)
    assert state.state == fr.FRESHNESS_REVIEW_DUE


def test_from_config_freshness_section_wins_over_wiki() -> None:
    """两段都给时 [freshness] 优先；[wiki] 独有的键继续生效（逐键合并）。"""
    config = {
        "wiki": {"half_life_days": {"volatile": 15, "drifting": 45}},
        "freshness": {"half_life_days": {"volatile": 20}, "review_due_ratio": 0.6},
    }
    settings = fr.from_config(config)
    assert settings.half_life_days == {"volatile": 20.0, "drifting": 45.0}
    assert settings.review_due_ratio == 0.6
    # 两段都没写的 volatility 仍用默认表
    assert fr.from_config({"wiki": {"half_life_days": {"volatile": 1}}}).half_life_days == {
        "volatile": 1.0,
        "drifting": 90.0,
    }


def test_from_config_half_life_matches_index_wiki_settings() -> None:
    """配置态口径一致：同一份 config，freshness 的 decay 与按 [wiki] 半衰期算的
    index.freshness_factor 逐点一致（防"配置态静默漂移"）。"""
    config = {"wiki": {"half_life_days": {"volatile": 7, "drifting": 14}}}
    settings = fr.from_config(config)
    index_settings = wiki_settings(config)
    assert settings.half_life_days == index_settings.half_life_days
    for volatility in ("stable", "drifting", "volatile"):
        for age in (0.0, 7.0, 14.0, 60.0):
            note = make_note(volatility=volatility, age_days=age)
            state = fr.evaluate_freshness(note, now=NOW, settings=settings)
            expected = freshness_factor(
                volatility,
                note.meta.observed_at,
                note.meta.created,
                half_life_days=index_settings.half_life_days,
                now=NOW,
            )
            assert state.decay == pytest.approx(expected), (volatility, age)


# ---- 字段读写（frontmatter / store round-trip） -------------------------------


def test_validity_fields_round_trip() -> None:
    meta = NoteMeta(
        id="N-0001",
        valid_from="2026-01-01T00:00:00+00:00",
        valid_until="2026-12-31T00:00:00+00:00",
    )
    meta_dict, body = parse(dump(meta.to_dict(), "正文"))
    assert meta_dict["valid_from"] == "2026-01-01T00:00:00+00:00"
    assert meta_dict["valid_until"] == "2026-12-31T00:00:00+00:00"
    restored = NoteMeta.from_dict(meta_dict)
    assert restored.valid_from == meta.valid_from and restored.valid_until == meta.valid_until
    assert restored == meta


def test_validity_fields_omitted_when_none() -> None:
    plain = NoteMeta(id="N-0001")
    assert "valid_from" not in plain.to_dict()
    assert "valid_until" not in plain.to_dict()
    assert NoteMeta.from_dict({"id": "N-0001"}).valid_from is None


def test_validity_fields_tolerate_non_iso_and_non_string() -> None:
    """非 ISO 字符串 / 数字 / YAML 时间戳都不抛异常（数字按标量转字符串）。"""
    meta = NoteMeta.from_dict({"id": "N-0001", "valid_from": "下周", "valid_until": 20261231})
    assert meta.valid_from == "下周"
    assert meta.valid_until == "20261231"
    # 计算侧把不可解析写法按"未声明"处理（不抛异常、不改判）
    state = fr.evaluate_freshness(
        make_note(volatility="volatile", age_days=0.0, valid_from="下周", valid_until="待定"),
        now=NOW,
    )
    assert state.state == fr.FRESHNESS_FRESH
    assert any("valid_until=待定 不是合法 ISO 时间" in r for r in state.reasons)
    # YAML 把未加引号的时间戳解析成 datetime → 统一转回 ISO 字符串
    meta_dict, _ = parse("---\nid: N-0001\nvalid_from: 2026-01-01 08:30:00\n---\n正文\n")
    assert NoteMeta.from_dict(meta_dict).valid_from.startswith("2026-01-01T08:30:00")


def test_store_save_and_get_validity_fields(tmp_path) -> None:
    store = WikiStore(tmp_path / "wiki-data")
    note = store.save_note(
        "有效期内的断言。",
        valid_from="2026-01-01T00:00:00+00:00",
        valid_until="2026-12-31T00:00:00+00:00",
        volatility="volatile",
    )
    loaded = store.get_note(note.id)
    assert loaded is not None
    assert loaded.valid_from == "2026-01-01T00:00:00+00:00"
    assert loaded.valid_until == "2026-12-31T00:00:00+00:00"
    assert loaded.meta.valid_from == note.meta.valid_from
    # 未声明有效期时字段为 None（frontmatter 里省略），端到端判定回落到 volatility 衰减
    plain = store.save_note(
        "普通断言。", volatility="volatile", created="2026-05-02T12:00:00"
    )
    reloaded = store.get_note(plain.id)
    assert reloaded is not None and reloaded.valid_until is None
    state = fr.evaluate_freshness(reloaded, now=NOW)
    assert state.age_days == pytest.approx(30.0)
    assert state.state == fr.FRESHNESS_REVIEW_DUE
