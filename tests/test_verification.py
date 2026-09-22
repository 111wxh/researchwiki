"""旧记忆与新证据的结构化比较测试（P2-C / RQ2 判定侧）。

覆盖 task-10 简报的验收清单：

- 五类 verdict 各至少一例（judge=None 下全部可达）+ verdict → suggested_action 映射；
- 来源变化两分支（无冲突 → newer；有冲突 → conflicting）与"来源未变 → consistent"；
- 时间相等/更早不判 newer；基准时间 observed_at → created 且与 P2-A freshness 同口径；
- 相似度地板生效（低于地板 → uncertain/none，且不调用 judge）；
- judge 只在 uncertain 被调用（计数 stub 断言）、返回 None/非法值/抛异常都保持
  uncertain，judge=None 全绿；
- 规则优先级（冲突门 > 来源变化 > 时间；一致 > 时间；时间 > 具体度）；
- 冲突槽位识别纯函数（同量词上下文、单侧多值跳过、裸数字不比较、归一化）；
- from_config 宽容回退与开关；
- 理由行引用具体值（hash 前 8 位、数值、时间戳），to_dict 可序列化；
- 不写盘、不写台账、不改笔记（"不静默覆盖"）。

全部零网络、零模型调用；固定时间戳，断言精确到数值。
"""

import json

import pytest

from researchwiki.wiki import verification as vf
from researchwiki.wiki.embeddings import MockEmbeddingProvider
from researchwiki.wiki.freshness import evaluate_freshness
from researchwiki.wiki.frontmatter import NoteMeta, SourceRef
from researchwiki.wiki.store import Note, WikiStore

NOW = "2026-06-01T00:00:00+00:00"
LATER = "2026-09-01T00:00:00+00:00"
EARLIER = "2026-01-01T00:00:00+00:00"
MID = "2026-03-01T00:00:00+00:00"
URL = "https://example.test/doc"
MODEL_BODY = "模型 A 的上下文窗口是 128k。"
MODEL_EVIDENCE = "模型 A 的上下文窗口是 128k，另有 256k 可选。"
HASH_OLD = "1a2b3c4d5e6f7788"
HASH_NEW = "9f8e7d6c5b4a3928"


# ---- 构造工具 ---------------------------------------------------------------


def make_note(
    *,
    body: str = "项目使用 uv 管理依赖。",
    note_id: str = "N-0001",
    observed_at: str | None = NOW,
    created: str = NOW,
    sources: list[SourceRef] | None = None,
    source_changed_at: str | None = None,
    reviewed_at: str | None = None,
    entities: list[str] | None = None,
) -> Note:
    """构造一条测试旧记忆（时间基准与来源均可显式覆盖）。"""
    meta = NoteMeta(
        id=note_id,
        title="判定测试",
        observed_at=observed_at,
        created=created,
        sources=list(sources or []),
        source_changed_at=source_changed_at,
        reviewed_at=reviewed_at,
        entities=list(entities or []),
    )
    return Note(id=note_id, title=meta.title, body=body, meta=meta, path=None)


def evi(
    text: str,
    *,
    observed_at: str | None = None,
    source_url: str = "",
    content_hash: str = "",
    entities: list[str] | None = None,
) -> vf.EvidenceItem:
    """构造一条新证据。"""
    return vf.EvidenceItem(
        text=text,
        observed_at=observed_at,
        source_url=source_url,
        content_hash=content_hash,
        entities=list(entities or []),
    )


def counting_judge(verdict: str | None = None, *, error: Exception | None = None):
    """计数判官桩：记录调用参数，按需返回 verdict / None / 抛异常。"""
    calls: list[tuple[str, str]] = []

    def judge(prior: Note, evidence: vf.EvidenceItem) -> str | None:
        calls.append((prior.id, evidence.text))
        if error is not None:
            raise error
        return verdict

    return judge, calls


def source_url_note(**kwargs) -> Note:
    """带一条来源快照（hash = HASH_OLD）的旧记忆。"""
    kwargs.setdefault("body", "模型 A 的上下文窗口是 128k。")
    kwargs.setdefault("sources", [SourceRef(url=URL, content_hash=HASH_OLD)])
    return make_note(**kwargs)


# ---- 五类 verdict -----------------------------------------------------------


def test_consistent_when_evidence_restates_prior() -> None:
    note = make_note(body="项目使用 uv 管理依赖，Python 3.12。")
    result = vf.compare_prior_and_evidence(note, evi("项目使用 uv 管理依赖，Python 3.12。"))
    assert result.verdict == vf.VERDICT_CONSISTENT
    assert result.suggested_action == vf.ACTION_REFRESH_REVIEWED_AT
    assert result.prior_note_id == "N-0001"
    assert result.evidence_index == 0
    assert any("一致：相似度" in reason for reason in result.reasons)


def test_newer_when_evidence_is_later_and_reworded() -> None:
    note = make_note(body="GLM-5.3 支持工具调用。")
    result = vf.compare_prior_and_evidence(
        note, evi("GLM-5.3 现在支持工具调用与函数并行。", observed_at=LATER)
    )
    assert result.verdict == vf.VERDICT_NEWER
    assert result.suggested_action == vf.ACTION_SUPERSEDE
    assert NOW in result.reasons[2] and LATER in result.reasons[2]


def test_newer_when_source_content_changed() -> None:
    note = source_url_note(source_changed_at="2026-08-01T00:00:00+00:00")
    result = vf.compare_prior_and_evidence(
        note,
        evi(
            "模型 A 的上下文窗口是 128k，另有 256k 可选。",
            source_url=URL,
            content_hash=HASH_NEW,
        ),
    )
    assert result.verdict == vf.VERDICT_NEWER
    assert result.suggested_action == vf.ACTION_SUPERSEDE
    assert "来源变化" in result.reasons[2]
    assert "1a2b3c4d" in result.reasons[2] and "9f8e7d6c" in result.reasons[2]


def test_more_specific_when_evidence_carries_more_facts() -> None:
    note = make_note(body="该系列有 3 个版本。")
    result = vf.compare_prior_and_evidence(
        note, evi("该系列有 3 个版本，其中 Pro 版本参数 70B，发布于 2026-09。")
    )
    assert result.verdict == vf.VERDICT_MORE_SPECIFIC
    assert result.suggested_action == vf.ACTION_MERGE
    assert any("更具体" in reason for reason in result.reasons)
    assert any("number:70" in reason and "date:2026-09" in reason for reason in result.reasons)


def test_conflicting_when_slot_values_differ() -> None:
    note = make_note(body="GLM-5.3 的上下文窗口是 128k。")
    result = vf.compare_prior_and_evidence(
        note, evi("GLM-4.5 的上下文窗口是 64k。", observed_at=LATER)
    )
    assert result.verdict == vf.VERDICT_CONFLICTING
    assert result.suggested_action == vf.ACTION_OPEN_CONFLICT
    joined = "\n".join(result.reasons)
    assert "旧=128" in joined and "新=64" in joined
    assert "旧=5.3" in joined and "新=4.5" in joined
    assert len(result.conflicts) == 2
    assert [c.to_dict()["slot"] for c in result.conflicts] == [
        "number:k:上下文窗口是",
        "version:GLM",
    ]


def test_uncertain_when_no_deterministic_signal() -> None:
    note = make_note(body="用户偏好中文回答。")
    result = vf.compare_prior_and_evidence(note, evi("用户喜欢用中文交流。"))
    assert result.verdict == vf.VERDICT_UNCERTAIN
    assert result.suggested_action == vf.ACTION_NONE
    assert result.judge_used is False
    assert any("无确定性判据" in reason for reason in result.reasons)


def test_all_five_verdicts_reachable_without_judge() -> None:
    """judge=None 下五类判定全部可达（本包零模型调用也能工作）。"""
    cases = [
        (
            vf.VERDICT_CONSISTENT,
            make_note(body="项目使用 uv 管理依赖。"),
            evi("项目使用 uv 管理依赖。"),
        ),
        (
            vf.VERDICT_NEWER,
            make_note(body="GLM-5.3 支持工具调用。"),
            evi("GLM-5.3 现在支持工具调用与函数并行。", observed_at=LATER),
        ),
        (
            vf.VERDICT_MORE_SPECIFIC,
            make_note(body="该系列有 3 个版本。"),
            evi("该系列有 3 个版本，Pro 版本参数 70B。"),
        ),
        (
            vf.VERDICT_CONFLICTING,
            make_note(body="GLM-5.3 的上下文窗口是 128k。"),
            evi("GLM-4.5 的上下文窗口是 64k。"),
        ),
        (
            vf.VERDICT_UNCERTAIN,
            make_note(body="用户偏好中文回答。"),
            evi("用户喜欢用中文交流。"),
        ),
    ]
    for expected, note, evidence in cases:
        assert vf.compare_prior_and_evidence(note, evidence).verdict == expected


def test_verdict_action_table_is_total() -> None:
    assert set(vf.VERDICT_ACTIONS) == set(vf.VERDICTS)
    assert vf.VERDICT_ACTIONS == {
        "consistent": "refresh_reviewed_at",
        "newer": "supersede",
        "more_specific": "merge",
        "conflicting": "open_conflict",
        "uncertain": "none",
    }


# ---- 来源变化两分支与"来源未变" --------------------------------------------


def test_source_change_without_conflict_is_newer() -> None:
    note = source_url_note(source_changed_at="2026-08-01T00:00:00+00:00")
    result = vf.compare_prior_and_evidence(
        note, evi(MODEL_EVIDENCE, source_url=URL, content_hash=HASH_NEW)
    )
    joined = "\n".join(result.reasons)
    assert result.verdict == vf.VERDICT_NEWER
    assert "source_changed_at=2026-08-01T00:00:00+00:00" in joined
    assert "另：证据新增事实令牌 number:256" in joined  # 供 P2-D supersede 时保留


def test_source_change_with_conflict_is_conflicting() -> None:
    note = source_url_note(source_changed_at="2026-08-01T00:00:00+00:00")
    result = vf.compare_prior_and_evidence(
        note, evi("模型 A 的上下文窗口是 256k。", source_url=URL, content_hash=HASH_NEW)
    )
    assert result.verdict == vf.VERDICT_CONFLICTING
    assert result.suggested_action == vf.ACTION_OPEN_CONFLICT


def test_unchanged_source_hash_is_not_newer() -> None:
    note = source_url_note(body="模型 A 上下文 128k。")
    result = vf.compare_prior_and_evidence(
        note, evi("模型 A 上下文 128k。", source_url=URL, content_hash=HASH_OLD)
    )
    assert result.verdict == vf.VERDICT_CONSISTENT
    joined = "\n".join(result.reasons)
    assert "来源未变" in joined and "1a2b3c4d" in joined


def test_source_url_hit_without_hash_is_not_newer() -> None:
    note = source_url_note(source_changed_at="2026-08-01T00:00:00+00:00")
    result = vf.compare_prior_and_evidence(
        note, evi(MODEL_EVIDENCE, source_url=URL)
    )
    assert result.verdict != vf.VERDICT_NEWER
    assert result.verdict == vf.VERDICT_MORE_SPECIFIC
    assert any("未给 content_hash" in reason for reason in result.reasons)


def test_hash_differs_without_source_changed_at_is_not_newer() -> None:
    note = source_url_note()  # 没有 source_changed_at 标记
    result = vf.compare_prior_and_evidence(
        note, evi(MODEL_EVIDENCE, source_url=URL, content_hash=HASH_NEW)
    )
    assert result.verdict == vf.VERDICT_MORE_SPECIFIC
    joined = "\n".join(result.reasons)
    assert "未声明 source_changed_at" in joined
    assert "旧=1a2b3c4d" in joined and "新=9f8e7d6c" in joined


def test_unparseable_source_changed_at_is_treated_as_undeclared() -> None:
    note = source_url_note(source_changed_at="昨天")
    result = vf.compare_prior_and_evidence(
        note, evi(MODEL_EVIDENCE, source_url=URL, content_hash=HASH_NEW)
    )
    assert result.verdict != vf.VERDICT_NEWER
    joined = "\n".join(result.reasons)
    assert "source_changed_at=昨天 不是合法 ISO 时间" in joined
    assert "未声明 source_changed_at" in joined


def test_source_url_not_matching_prior_refs_is_ignored() -> None:
    note = source_url_note(source_changed_at="2026-08-01T00:00:00+00:00")
    result = vf.compare_prior_and_evidence(
        note,
        evi(MODEL_EVIDENCE, source_url="https://other.test/doc", content_hash=HASH_NEW),
    )
    joined = "\n".join(result.reasons)
    assert result.verdict != vf.VERDICT_NEWER
    assert "来源变化" not in joined


# ---- 时间规则 ---------------------------------------------------------------


def test_later_evidence_time_makes_newer() -> None:
    note = make_note(body="GLM-5.3 支持工具调用。")
    result = vf.compare_prior_and_evidence(
        note, evi("GLM-5.3 现在支持工具调用与函数并行。", observed_at=LATER)
    )
    assert result.verdict == vf.VERDICT_NEWER
    assert f"晚于旧记忆 observed_at={NOW}" in result.reasons[2]


def test_equal_timestamp_is_not_newer() -> None:
    note = make_note(body="GLM-5.3 支持工具调用。", observed_at=NOW)
    result = vf.compare_prior_and_evidence(
        note, evi("GLM-5.3 现在支持工具调用与函数并行。", observed_at=NOW)
    )
    assert result.verdict != vf.VERDICT_NEWER
    assert any("（相等或更早）→ 不因时间判更新" in reason for reason in result.reasons)


def test_earlier_timestamp_is_not_newer() -> None:
    note = make_note(body="GLM-5.3 支持工具调用。", observed_at=NOW)
    result = vf.compare_prior_and_evidence(
        note, evi("GLM-5.3 现在支持工具调用与函数并行。", observed_at=EARLIER)
    )
    assert result.verdict != vf.VERDICT_NEWER


def test_unparseable_evidence_time_is_ignored() -> None:
    note = make_note(body="GLM-5.3 支持工具调用。")
    result = vf.compare_prior_and_evidence(
        note, evi("GLM-5.3 现在支持工具调用与函数并行。", observed_at="不是时间")
    )
    assert result.verdict != vf.VERDICT_NEWER
    joined = "\n".join(result.reasons)
    assert "observed_at=不是时间 不是合法 ISO 时间" in joined
    assert "证据未提供可解析的 observed_at" in joined


def test_prior_without_base_time_is_not_newer() -> None:
    note = make_note(body="GLM-5.3 支持工具调用。", observed_at=None, created="")
    result = vf.compare_prior_and_evidence(
        note, evi("GLM-5.3 现在支持工具调用与函数并行。", observed_at=LATER)
    )
    assert result.verdict != vf.VERDICT_NEWER
    assert any("旧记忆无时间基准" in reason for reason in result.reasons)


def test_base_time_falls_back_to_created() -> None:
    note = make_note(body="GLM-5.3 支持工具调用。", observed_at=None, created=NOW)
    result = vf.compare_prior_and_evidence(
        note, evi("GLM-5.3 现在支持工具调用与函数并行。", observed_at=LATER)
    )
    assert result.verdict == vf.VERDICT_NEWER
    assert f"旧记忆 created={NOW}" in result.reasons[2]


def test_base_time_prefers_observed_at_over_created() -> None:
    """证据时间落在 created 与 observed_at 之间：只有用 observed_at 做基准才不判 newer。"""
    note = make_note(body="用户偏好中文回答。", observed_at=NOW, created=EARLIER)
    result = vf.compare_prior_and_evidence(note, evi("用户喜欢用中文交流。", observed_at=MID))
    assert result.verdict != vf.VERDICT_NEWER
    assert any(f"旧记忆 observed_at={NOW}" in reason for reason in result.reasons)


def test_base_time_agrees_with_freshness() -> None:
    """基准时间口径与 P2-A freshness 一致（同一 note、同一字段、同一时间戳）。"""
    note = make_note(body="GLM-5.3 支持工具调用。", observed_at=None, created=NOW)
    result = vf.compare_prior_and_evidence(
        note, evi("GLM-5.3 现在支持工具调用与函数并行。", observed_at=LATER)
    )
    freshness = evaluate_freshness(note)
    assert f"时间基准 created（{NOW}）" in freshness.reasons[0]
    assert f"旧记忆 created={NOW}" in result.reasons[2]

    with_observed = make_note(body="用户偏好中文回答。", observed_at=NOW, created=EARLIER)
    observed_result = vf.compare_prior_and_evidence(
        with_observed, evi("用户喜欢用中文交流。", observed_at=MID)
    )
    assert any(f"旧记忆 observed_at={NOW}" in reason for reason in observed_result.reasons)
    assert f"时间基准 observed_at（{NOW}）" in evaluate_freshness(with_observed).reasons[0]


def test_missing_evidence_time_is_not_newer() -> None:
    note = make_note(body="GLM-5.3 支持工具调用。")
    result = vf.compare_prior_and_evidence(note, evi("GLM-5.3 现在支持工具调用与函数并行。"))
    assert result.verdict != vf.VERDICT_NEWER
    assert any("证据未提供可解析的 observed_at" in reason for reason in result.reasons)


# ---- 相似度地板 -------------------------------------------------------------


def test_similarity_floor_short_circuits_to_uncertain() -> None:
    note = make_note(body="项目使用 uv 管理依赖。")
    result = vf.compare_prior_and_evidence(note, evi("今天天气不错，适合出门散步。"))
    assert result.verdict == vf.VERDICT_UNCERTAIN
    assert result.suggested_action == vf.ACTION_NONE
    joined = "\n".join(result.reasons)
    assert f"相似度 {result.similarity:.3f} 低于地板 0.30" in joined
    assert "不进入确定性比较" in joined


def test_similarity_floor_does_not_call_judge() -> None:
    """地板短路路径不调用 judge：地板的意义就是不让无关证据消耗比较器。"""
    judge, calls = counting_judge(vf.VERDICT_NEWER)
    note = make_note(body="项目使用 uv 管理依赖。")
    result = vf.compare_prior_and_evidence(
        note, evi("今天天气不错，适合出门散步。"), judge=judge
    )
    assert result.verdict == vf.VERDICT_UNCERTAIN
    assert result.judge_used is False
    assert calls == []


def test_similarity_dimension_is_configurable_and_quoted() -> None:
    settings = vf.VerificationSettings(similarity_dim=64)
    result = vf.compare_prior_and_evidence(
        make_note(body="项目使用 uv 管理依赖。"),
        evi("项目使用 uv 管理依赖。"),
        settings=settings,
    )
    assert "dim=64" in result.reasons[0]
    assert result.verdict == vf.VERDICT_CONSISTENT


def test_empty_evidence_text_is_uncertain() -> None:
    result = vf.compare_prior_and_evidence(make_note(), evi(""))
    assert result.verdict == vf.VERDICT_UNCERTAIN
    assert result.suggested_action == vf.ACTION_NONE
    assert result.similarity == 0.0
    assert "相似度 0.000 低于地板 0.30" in "\n".join(result.reasons)


def test_similarity_is_symmetric_and_bounded() -> None:
    assert vf.token_similarity("项目使用 uv 管理依赖。", "项目使用 uv 管理依赖。") == 1.0
    forward = vf.token_similarity("项目使用 uv 管理依赖。", "Python 版本为 3.12。")
    backward = vf.token_similarity("Python 版本为 3.12。", "项目使用 uv 管理依赖。")
    assert forward == backward
    assert 0.0 <= forward <= 1.0
    assert vf.token_similarity("", "项目使用 uv 管理依赖。") == 0.0


# ---- judge 注入 -------------------------------------------------------------


def test_judge_only_called_for_uncertain() -> None:
    """已确定的判定绝不调用 judge（五类里的四个确定态各一例，来源变化含内）。"""
    judge, calls = counting_judge(vf.VERDICT_CONFLICTING)
    determined = [
        (make_note(body="项目使用 uv 管理依赖。"), evi("项目使用 uv 管理依赖。")),
        (
            make_note(body="GLM-5.3 支持工具调用。"),
            evi("GLM-5.3 现在支持工具调用与函数并行。", observed_at=LATER),
        ),
        (make_note(body="该系列有 3 个版本。"), evi("该系列有 3 个版本，Pro 版本参数 70B。")),
        (make_note(body="GLM-5.3 的上下文窗口是 128k。"), evi("GLM-4.5 的上下文窗口是 64k。")),
        (
            source_url_note(source_changed_at=NOW),
            evi("模型 A 上下文 128k。", source_url=URL, content_hash=HASH_NEW),
        ),
    ]
    verdicts = [
        vf.compare_prior_and_evidence(note, evidence, judge=judge).verdict
        for note, evidence in determined
    ]
    assert calls == []
    assert verdicts == [
        vf.VERDICT_CONSISTENT,
        vf.VERDICT_NEWER,
        vf.VERDICT_MORE_SPECIFIC,
        vf.VERDICT_CONFLICTING,
        vf.VERDICT_NEWER,
    ]


def test_judge_verdict_is_applied_when_uncertain() -> None:
    judge, calls = counting_judge(vf.VERDICT_CONFLICTING)
    result = vf.compare_prior_and_evidence(
        make_note(body="用户偏好中文回答。"), evi("用户喜欢用中文交流。"), judge=judge
    )
    assert len(calls) == 1 and calls[0][0] == "N-0001"
    assert result.verdict == vf.VERDICT_CONFLICTING
    assert result.suggested_action == vf.ACTION_OPEN_CONFLICT
    assert result.judge_used is True
    assert any("语义判官判定 conflicting" in reason for reason in result.reasons)


def test_judge_returning_none_keeps_uncertain() -> None:
    judge, calls = counting_judge(None)
    result = vf.compare_prior_and_evidence(
        make_note(body="用户偏好中文回答。"), evi("用户喜欢用中文交流。"), judge=judge
    )
    assert len(calls) == 1
    assert result.verdict == vf.VERDICT_UNCERTAIN
    assert result.judge_used is True
    assert any("未给出判定（返回 None）" in reason for reason in result.reasons)


def test_judge_returning_illegal_value_keeps_uncertain() -> None:
    judge, _ = counting_judge("probably-newer")
    result = vf.compare_prior_and_evidence(
        make_note(body="用户偏好中文回答。"), evi("用户喜欢用中文交流。"), judge=judge
    )
    assert result.verdict == vf.VERDICT_UNCERTAIN
    assert any("不是合法判定" in reason for reason in result.reasons)


def test_judge_verdict_is_normalized() -> None:
    judge, _ = counting_judge("  NEWER ")
    result = vf.compare_prior_and_evidence(
        make_note(body="用户偏好中文回答。"), evi("用户喜欢用中文交流。"), judge=judge
    )
    assert result.verdict == vf.VERDICT_NEWER
    assert result.suggested_action == vf.ACTION_SUPERSEDE


def test_judge_exception_keeps_uncertain_without_raising() -> None:
    judge, _ = counting_judge(error=ValueError("boom"))
    result = vf.compare_prior_and_evidence(
        make_note(body="用户偏好中文回答。"), evi("用户喜欢用中文交流。"), judge=judge
    )
    assert result.verdict == vf.VERDICT_UNCERTAIN
    assert any("语义判官调用失败（ValueError: boom）" in reason for reason in result.reasons)


def test_judge_none_still_covers_every_uncertain_case() -> None:
    """judge=None 时"信息不足"与"地板"两条 uncertain 路径都完整可用。"""
    fallback = vf.compare_prior_and_evidence(
        make_note(body="用户偏好中文回答。"), evi("用户喜欢用中文交流。")
    )
    floored = vf.compare_prior_and_evidence(
        make_note(body="用户偏好中文回答。"), evi("今天天气不错。")
    )
    assert fallback.verdict == floored.verdict == vf.VERDICT_UNCERTAIN
    assert fallback.suggested_action == floored.suggested_action == vf.ACTION_NONE
    assert fallback.judge_used is False and floored.judge_used is False


# ---- 规则优先级 -------------------------------------------------------------


def test_source_change_beats_time() -> None:
    """证据时间更晚 + 来源换版（无冲突）→ 走来源分支（理由里出现 hash 前 8 位）。"""
    note = source_url_note(source_changed_at="2026-08-01T00:00:00+00:00")
    result = vf.compare_prior_and_evidence(
        note,
        evi(MODEL_EVIDENCE, observed_at=LATER, source_url=URL, content_hash=HASH_NEW),
    )
    assert result.verdict == vf.VERDICT_NEWER
    assert any("来源变化" in reason for reason in result.reasons)
    assert not any("晚于旧记忆" in reason for reason in result.reasons)


def test_conflict_beats_time_and_specificity() -> None:
    """更晚且事实更多的证据，只要同槽位取值不同 → 冲突台账，不 supersede 掩盖。"""
    note = make_note(body="GLM-5.3 的上下文窗口是 128k。")
    result = vf.compare_prior_and_evidence(
        note, evi("GLM-4.5 的上下文窗口是 64k，参数量 70B。", observed_at=LATER)
    )
    assert result.verdict == vf.VERDICT_CONFLICTING
    joined = "\n".join(result.reasons)
    assert "优先级高于来源变化与时间规则" in joined
    assert "时间：" not in joined


def test_consistency_beats_time() -> None:
    """时间更晚但断言未变（高相似、无新增令牌）→ consistent，只刷新 reviewed_at。"""
    note = make_note(body="项目使用 uv 管理依赖，Python 3.12。", observed_at=NOW)
    result = vf.compare_prior_and_evidence(
        note, evi("项目使用 uv 管理依赖，Python 3.12。", observed_at=LATER)
    )
    assert result.verdict == vf.VERDICT_CONSISTENT
    assert not any("时间：" in reason for reason in result.reasons)


def test_time_beats_specificity() -> None:
    """证据更晚且事实更多 → newer（时间规则优先于具体度，具体度只在时间不更新时生效）。"""
    note = make_note(body="该系列有 3 个版本。")
    result = vf.compare_prior_and_evidence(
        note, evi("该系列有 3 个版本，Pro 版本参数 70B。", observed_at=LATER)
    )
    assert result.verdict == vf.VERDICT_NEWER


def test_specificity_applies_when_time_not_later() -> None:
    note = make_note(body="该系列有 3 个版本。", observed_at=NOW)
    result = vf.compare_prior_and_evidence(
        note, evi("该系列有 3 个版本，Pro 版本参数 70B。", observed_at=NOW)
    )
    assert result.verdict == vf.VERDICT_MORE_SPECIFIC
    assert result.suggested_action == vf.ACTION_MERGE


def test_entity_disjoint_blocks_specificity() -> None:
    """实体不交的长证据不得判 more_specific（修复轮 2/5 后连具体度也被全局护栏短路）。"""
    note = make_note(body="该系列有 3 个版本。", entities=["GLM-5.3"])
    result = vf.compare_prior_and_evidence(
        note, evi("该系列有 3 个版本，Pro 版本参数 70B。", entities=["Qwen-3"])
    )
    assert result.verdict != vf.VERDICT_MORE_SPECIFIC
    assert result.verdict == vf.VERDICT_UNCERTAIN
    joined = "\n".join(result.reasons)
    assert "两侧实体不交" in joined
    assert "无法确认主体同一性 → 不做更新判定" in joined

    with_time = vf.compare_prior_and_evidence(
        note,
        evi("该系列有 3 个版本，Pro 版本参数 70B。", observed_at=LATER, entities=["Qwen-3"]),
    )
    assert with_time.verdict == vf.VERDICT_UNCERTAIN
    assert "具体度：" not in "\n".join(with_time.reasons)  # 规则 5 整条被短路


def test_entity_overlap_allows_specificity() -> None:
    note = make_note(body="该系列有 3 个版本。", entities=["GLM-5.3"])
    result = vf.compare_prior_and_evidence(
        note, evi("该系列有 3 个版本，Pro 版本参数 70B。", entities=["glm-5-3"])
    )
    assert result.verdict == vf.VERDICT_MORE_SPECIFIC


# ---- 冲突槽位识别（纯函数）-------------------------------------------------


@pytest.mark.parametrize(
    ("prior_text", "evidence_text", "prior_value", "evidence_value"),
    [
        ("共 3 个模型", "共 5 个模型", "3", "5"),
        ("上下文窗口 128k", "上下文窗口 256k", "128", "256"),
        ("支持 30% 的折扣", "支持 20% 的折扣", "30%", "20%"),
        ("第 3 章讲了并发模型", "第 4 章讲了并发模型", "3章", "4章"),
        ("Chapter 3 covers retries", "Chapter 4 covers retries", "3章", "4章"),
        ("v1.2.3 修复了问题", "v1.2.4 修复了问题", "1.2.3", "1.2.4"),
        ("模型发布于 2026-08", "模型发布于 2026-09", "2026-08", "2026-09"),
        ("模型发布于 2026年8月", "模型发布于 2026年9月", "2026-08", "2026-09"),
    ],
)
def test_slot_conflicts_by_kind(prior_text, evidence_text, prior_value, evidence_value) -> None:
    conflicts = vf.detect_slot_conflicts(prior_text, evidence_text)
    assert len(conflicts) == 1
    assert conflicts[0].prior_value == prior_value
    assert conflicts[0].evidence_value == evidence_value
    assert f"旧={prior_value}" in conflicts[0].describe()
    assert f"新={evidence_value}" in conflicts[0].describe()


def test_slot_conflict_reports_kind_and_anchor() -> None:
    conflicts = vf.detect_slot_conflicts("上下文窗口 128k", "上下文窗口 256k")
    conflict = conflicts[0]
    assert conflict.kind == "number"
    assert conflict.context == "上下文窗口"
    assert conflict.slot == "number:k:上下文窗口"
    assert conflict.to_dict() == {
        "slot": "number:k:上下文窗口",
        "kind": "number",
        "context": "上下文窗口",
        "prior_value": "128",
        "evidence_value": "256",
    }


def test_different_anchors_are_different_slots() -> None:
    """锚点不同 = 不同槽位（参数量与上下文窗口不是同一个事实）→ 不判冲突。"""
    assert vf.detect_slot_conflicts("参数量 128k", "上下文窗口 256k") == []


def test_bare_number_without_unit_or_context_is_not_comparable() -> None:
    assert vf.detect_slot_conflicts("3", "4") == []
    assert vf.detect_slot_conflicts("共 3 个", "共 4 个") != []


def test_same_value_different_units_is_not_a_conflict() -> None:
    assert vf.detect_slot_conflicts("共 3 个 共 3个", "共 3 个") == []


def test_single_side_multi_value_is_skipped() -> None:
    scan = vf.scan_slots("版本 3.5 版本 4.0 都支持", "版本 3.5 都支持")
    assert scan.conflicts == []
    assert len(scan.skipped) == 1
    assert "在旧记忆有多个取值（旧=3.5、4.0；新=3.5）" in scan.skipped[0]


def test_skipped_slot_reason_lands_in_comparison_reasons() -> None:
    """被跳过的槽位会让"一致"退回 uncertain（可比性没走完，不宣称仍然成立）。"""
    note = make_note(body="版本 3.5 版本 4.0 都支持。", observed_at=None, created="")
    result = vf.compare_prior_and_evidence(note, evi("版本 3.5 都支持。"))
    assert result.verdict == vf.VERDICT_UNCERTAIN
    assert result.conflicts == []
    assert any("跳过该槽位" in reason for reason in result.reasons)
    assert any("一致判定被保留" in reason for reason in result.reasons)


def test_tokens_are_ordered_and_deduplicated() -> None:
    tokens = vf.extract_fact_tokens("共 3 个，共 3 个，另加 5 个。")
    assert [t.label() for t in tokens] == ["number:3", "number:5"]
    assert [t.raw for t in tokens] == ["3", "5"]


def test_extract_fact_tokens_priority_and_values() -> None:
    text = "GLM-5.3 的上下文窗口是 128k，发布于 2026-09，第 3 章讲了并发，提升 30%。"
    tokens = vf.extract_fact_tokens(text)
    assert [(t.kind, t.value) for t in tokens] == [
        ("version", "5.3"),
        ("number", "128"),
        ("date", "2026-09"),
        ("chapter", "3章"),
        ("percent", "30%"),
    ]
    assert tokens[1].unit == "k"
    assert tokens[0].slot == "version:GLM"
    assert tokens[1].slot == "number:k:上下文窗口是"


def test_no_tokens_means_no_conflict() -> None:
    assert vf.detect_slot_conflicts("用户偏好中文回答。", "用户喜欢用中文交流。") == []
    assert vf.scan_slots("用户偏好中文回答。", "用户喜欢用中文交流。").new_tokens == []


def test_scan_slots_reports_new_tokens() -> None:
    scan = vf.scan_slots("该系列有 3 个版本。", "该系列有 3 个版本，Pro 版本参数 70B。")
    assert [t.label() for t in scan.new_tokens] == ["number:70"]
    assert len(scan.prior_tokens) == 1 and len(scan.evidence_tokens) == 2


# ---- 配置（from_config 宽容回退）-------------------------------------------


def test_default_settings_values_and_dim_lock() -> None:
    settings = vf.VerificationSettings()
    assert settings.similarity_floor == 0.3
    assert settings.consistent_similarity == 0.8
    assert settings.context_chars == 6
    assert settings.similarity_dim == vf.DEFAULT_SIMILARITY_DIM
    # 与复用嵌入器的默认维度同值（防单侧漂移）
    assert settings.similarity_dim == MockEmbeddingProvider().dim


def test_from_config_accepts_full_config_and_section() -> None:
    section = {
        "similarity_floor": 0.4,
        "consistent_similarity": 0.9,
        "similarity_dim": 256,
        "context_chars": 8,
    }
    from_full = vf.VerificationSettings.from_config({"wiki": {}, "verification": section})
    from_section = vf.VerificationSettings.from_config(section)
    assert from_full == from_section
    assert from_full.similarity_floor == 0.4
    assert from_full.consistent_similarity == 0.9
    assert from_full.similarity_dim == 256
    assert from_full.context_chars == 8


def test_from_config_defaults_on_missing_or_junk() -> None:
    default = vf.VerificationSettings()
    assert vf.VerificationSettings.from_config() == default
    assert vf.VerificationSettings.from_config(None) == default
    assert vf.VerificationSettings.from_config({"verification": "junk"}) == default
    assert vf.VerificationSettings.from_config([]) == default
    assert vf.VerificationSettings.from_config(
        {"verification": {"similarity_floor": "abc", "similarity_dim": "big", "nope": 1}}
    ) == default


def test_from_config_clamps_out_of_range_values() -> None:
    settings = vf.VerificationSettings.from_config(
        {"verification": {"similarity_floor": 2.0, "similarity_dim": -5, "context_chars": 999}}
    )
    assert settings.similarity_floor == 1.0
    assert settings.similarity_dim == 8
    assert settings.context_chars == 24


def test_consistent_similarity_never_below_floor() -> None:
    settings = vf.VerificationSettings(similarity_floor=0.6, consistent_similarity=0.4)
    assert settings.consistent_similarity == 0.6


def test_settings_tolerate_bad_bool_values() -> None:
    settings = vf.VerificationSettings(
        extract_dates="yes",  # type: ignore[arg-type]
        extract_versions="off",  # type: ignore[arg-type]
    )
    assert settings.extract_dates is True
    assert settings.extract_versions is False


def test_from_config_switches_off_extraction() -> None:
    settings = vf.VerificationSettings.from_config(
        {"verification": {"extract_dates": False, "extract_chapters": "no"}}
    )
    tokens = vf.extract_fact_tokens("发布于 2026-09，第 3 章，共 3 个。", settings=settings)
    kinds = {t.kind for t in tokens}
    assert "date" not in kinds and "chapter" not in kinds
    assert {"number", "percent"} & kinds  # 数值开关未关，仍抽取
    assert vf.detect_slot_conflicts("发布于 2026-08", "发布于 2026-09", settings=settings) == []


def test_from_config_switches_off_versions() -> None:
    settings = vf.VerificationSettings.from_config({"verification": {"extract_versions": False}})
    kinds = {t.kind for t in vf.extract_fact_tokens("v1.2.3 修复了问题", settings=settings)}
    assert "version" not in kinds
    assert vf.extract_fact_tokens("v1.2.3 修复了问题")[0].kind == "version"


def test_settings_flow_through_batch() -> None:
    settings = vf.VerificationSettings(similarity_floor=0.99)
    results = vf.compare_batch(
        make_note(body="项目使用 uv 管理依赖。"),
        [evi("项目使用 uv 管理依赖，Python 3.12。")],
        settings=settings,
    )
    assert results[0].verdict == vf.VERDICT_UNCERTAIN
    assert "地板 0.99" in results[0].reasons[0]
    assert "低于地板 0.99" in results[0].reasons[1]


# ---- 留痕、可审计、无副作用 -------------------------------------------------


def test_reasons_quote_concrete_values() -> None:
    conflict = vf.compare_prior_and_evidence(
        source_url_note(source_changed_at="2026-08-01T00:00:00+00:00"),
        evi("模型 A 的上下文窗口是 256k。", source_url=URL, content_hash=HASH_NEW),
    )
    joined = "\n".join(conflict.reasons)
    assert "旧=128" in joined and "新=256" in joined  # 冲突槽位的两侧取值
    assert f"{conflict.similarity:.3f}" in joined  # 相似度数值
    assert "地板 0.30" in joined and "一致阈值 0.80" in joined  # 阈值具体值

    changed = vf.compare_prior_and_evidence(
        source_url_note(source_changed_at="2026-08-01T00:00:00+00:00"),
        evi(MODEL_EVIDENCE, source_url=URL, content_hash=HASH_NEW),
    )
    changed_joined = "\n".join(changed.reasons)
    assert "旧=1a2b3c4d" in changed_joined and "新=9f8e7d6c" in changed_joined  # hash 前 8 位
    assert "source_changed_at=2026-08-01T00:00:00+00:00" in changed_joined


def test_comparison_to_dict_is_json_serializable() -> None:
    note = make_note(body="GLM-5.3 的上下文窗口是 128k。")
    result = vf.compare_prior_and_evidence(note, evi("GLM-4.5 的上下文窗口是 64k。"))
    payload = json.loads(json.dumps(result.to_dict(), ensure_ascii=False))
    assert payload["verdict"] == vf.VERDICT_CONFLICTING
    assert payload["suggested_action"] == vf.ACTION_OPEN_CONFLICT
    assert payload["prior_note_id"] == "N-0001"
    assert payload["evidence_index"] == 0
    assert payload["judge_used"] is False
    assert [c["prior_value"] for c in payload["conflicts"]] == ["128", "5.3"]
    assert [c["evidence_value"] for c in payload["conflicts"]] == ["64", "4.5"]
    assert payload["reasons"] == result.reasons


def test_comparison_is_deterministic() -> None:
    note = source_url_note(source_changed_at="2026-08-01T00:00:00+00:00")
    evidence = evi(MODEL_EVIDENCE, source_url=URL, content_hash=HASH_NEW)
    first = vf.compare_prior_and_evidence(note, evidence)
    second = vf.compare_prior_and_evidence(note, evidence)
    assert first == second


def test_comparison_does_not_write_notes_or_conflicts(tmp_path) -> None:
    """不静默覆盖：判定只产出建议与留痕数据，笔记与冲突台账都不被写。"""
    store = WikiStore(tmp_path)
    note = store.save_note(
        "GLM-5.3 的上下文窗口是 128k。",
        note_id="N-0001",
        observed_at=NOW,
        created=NOW,
        sources=[SourceRef(url=URL, content_hash=HASH_OLD)],
        source_changed_at="2026-08-01T00:00:00+00:00",
    )
    path = tmp_path / "notes" / "N-0001.md"
    before = path.read_text(encoding="utf-8")
    result = vf.compare_prior_and_evidence(
        note,
        evi("GLM-4.5 的上下文窗口是 64k。", observed_at=LATER, source_url=URL,
            content_hash=HASH_NEW),
    )
    assert result.verdict == vf.VERDICT_CONFLICTING
    assert path.read_text(encoding="utf-8") == before
    assert store.list_conflicts() == []
    reloaded = store.get_note("N-0001")
    assert reloaded is not None
    assert reloaded.meta.status == "active"
    assert reloaded.meta.reviewed_at is None


def test_compare_batch_keeps_index_alignment() -> None:
    note = make_note(body="用户偏好中文回答。")
    evidence = [
        evi("用户偏好中文回答。"),
        evi("用户喜欢用中文交流。"),
        evi(""),  # 空正文不跳过：下标必须能回填到输入列表
    ]
    results = vf.compare_batch(note, evidence)
    assert [r.evidence_index for r in results] == [0, 1, 2]
    assert [r.verdict for r in results] == [
        vf.VERDICT_CONSISTENT,
        vf.VERDICT_UNCERTAIN,
        vf.VERDICT_UNCERTAIN,
    ]
    assert all(r.prior_note_id == "N-0001" for r in results)
    assert vf.compare_batch(note, []) == []


def test_compare_batch_judge_call_count_matches_uncertain_items() -> None:
    judge, calls = counting_judge(vf.VERDICT_NEWER)
    note = make_note(body="用户偏好中文回答。")
    results = vf.compare_batch(
        note,
        [evi("用户偏好中文回答。"), evi("用户喜欢用中文交流。"), evi("今天天气不错。")],
        judge=judge,
    )
    assert len(calls) == 1  # 第 2 条（信息不足）才需要判官；第 3 条被相似度地板挡下
    assert calls[0][1] == "用户喜欢用中文交流。"
    assert [r.verdict for r in results] == [
        vf.VERDICT_CONSISTENT,
        vf.VERDICT_NEWER,
        vf.VERDICT_UNCERTAIN,
    ]
    assert [r.judge_used for r in results] == [False, True, False]


# ---- 修复轮 1/5：Critical（一致判据改槽位级）--------------------------------


SWAPPED_PRIOR = "上下文窗口 128k，参数量 70B。"
SWAPPED_EVIDENCE = "上下文窗口 70B，参数量 128k。"


def test_critical_swapped_values_is_not_consistent() -> None:
    """Critical 复现用例：取值错位互换不得判 consistent。

    旧 `上下文窗口 128k，参数量 70B` 对新 `上下文窗口 70B，参数量 128k`：两侧用到的
    (kind, value) 集合完全相同、相似度 0.946、无槽位冲突，集合级检查会判"仍然成立"。
    实际 verdict = uncertain（action=none）——这才是正确的：两个槽位在两份文本间
    根本建立不起对应关系（旧 `number:k:上下文窗口` 在证据里变成 `number:b:上下文窗口`、
    旧 `number:b:参数量` 变成 `number:k:参数量`），既不能说"取值没变"（一致），也不能
    说"同一个槽位上无法同时成立"（冲突只对得上槽位时才成立）——这正是"来源换版时
    表格列序/标签错位"的形态，交人工/判官复核是正确的方向。
    """
    result = vf.compare_prior_and_evidence(make_note(body=SWAPPED_PRIOR), evi(SWAPPED_EVIDENCE))
    assert result.verdict == vf.VERDICT_UNCERTAIN
    assert result.suggested_action == vf.ACTION_NONE
    assert result.verdict != vf.VERDICT_CONSISTENT
    assert result.similarity > 0.9  # 字面极像，所以"看起来没变"最危险
    assert result.conflicts == []  # 槽位键都对不上，冲突门本就命中不了
    joined = "\n".join(result.reasons)
    assert "一致判定被保留" in joined
    assert "槽位级取值对不上" in joined
    assert "number:k:上下文窗口=128" in joined and "number:b:上下文窗口=70" in joined


def test_slot_level_agreement_pure_function() -> None:
    """槽位级一致判据（`scan_slots(...).slots_agree`）的四个边界。"""
    # 逐槽位同值 → 一致
    assert vf.scan_slots("共 3 个模型。", "共 3 个模型。").slots_agree is True
    # 错位互换 → 不一致（集合级看不见）
    assert vf.scan_slots(SWAPPED_PRIOR, SWAPPED_EVIDENCE).slots_agree is False
    # 证据多出一个可比槽位（该取值在旧文里也出现过）→ 不一致
    assert vf.scan_slots("参数量 128k。", "参数量 128k，上下文窗口 128k。").slots_agree is False
    # 两侧都抽不到可比令牌 → 恒等（无可比事实即无可反驳）
    assert vf.scan_slots("用户偏好中文回答。", "用户喜欢用中文交流。").slots_agree is True


def test_same_slot_same_value_still_consistent() -> None:
    """Critical 修复不得把正常的"同槽位同值"重述赶出 consistent。"""
    body = "共 3 个模型，上下文窗口 128k。"
    result = vf.compare_prior_and_evidence(make_note(body=body), evi(body))
    assert result.verdict == vf.VERDICT_CONSISTENT
    assert result.suggested_action == vf.ACTION_REFRESH_REVIEWED_AT
    assert vf.scan_slots(body, body).slots_agree is True


def test_swapped_values_with_later_time_is_not_consistent() -> None:
    """错位互换 + 时间更晚：不判 consistent；时间规则给 newer（supersede 保留旧版本）。

    为什么这里 newer 可以接受：时间确实更晚，而"取值错位"只能说明两边对不上号，
    不能说明新证据错了；按 PLAN"历史版本永不删除"的口径，supersede 会把旧记忆整条
    留档（旧 ID 仍可读），审计链完整。真正不能接受的是 consistent（＝断言仍然成立）
    ——那才会让错位的事实静默留在"当前事实"里。
    """
    result = vf.compare_prior_and_evidence(
        make_note(body=SWAPPED_PRIOR), evi(SWAPPED_EVIDENCE, observed_at=LATER)
    )
    assert result.verdict == vf.VERDICT_NEWER
    assert result.verdict != vf.VERDICT_CONSISTENT
    assert "一致判定被保留" in "\n".join(result.reasons)


# ---- 修复轮 1/5：Important（实体护栏纳入冲突门）+ 修复轮 2/5：提升为全局前置 ---


def test_entity_disjoint_skips_conflict_gate() -> None:
    """修复轮 1/5 复现用例：两侧实体不交时不得判 conflicting（修复轮 2/5 后为全局短路）。

    旧「模型 A 参数 70B。」(entities=["GLM-5.3"]) 对新「模型 B 参数 128B。」
    (entities=["Qwen-3"])：槽位键里不含主语，`number:b:参数` 同槽位不同值看似冲突，
    但两个句子说的是两个模型。判 conflicting 会直接产出 open_conflict、污染冲突台账
    与评测冲突计数 → 实际 verdict = uncertain（交人工/判官）。
    """
    result = vf.compare_prior_and_evidence(
        make_note(body="模型 A 参数 70B。", entities=["GLM-5.3"]),
        evi("模型 B 参数 128B。", entities=["Qwen-3"]),
    )
    assert result.verdict != vf.VERDICT_CONFLICTING
    assert result.verdict == vf.VERDICT_UNCERTAIN
    assert result.suggested_action == vf.ACTION_NONE
    assert result.conflicts == []
    joined = "\n".join(result.reasons)
    assert "两侧实体不交（旧=glm-5-3；新=qwen-3）：槽位键里不含主语 → 跳过冲突判定" in joined
    assert "冲突检测命中" not in joined


def test_entity_overlap_keeps_conflict_gate() -> None:
    """同一对文本 + 实体相交 → 冲突门照常命中（证明跳过是实体信号触发的）。"""
    result = vf.compare_prior_and_evidence(
        make_note(body="模型 A 参数 70B。", entities=["GLM-5.3"]),
        evi("模型 B 参数 128B。", entities=["glm-5-3", "qwen-3"]),
    )
    assert result.verdict == vf.VERDICT_CONFLICTING
    assert any("冲突检测命中" in reason for reason in result.reasons)


def test_missing_entities_keep_conflict_gate() -> None:
    """任一侧实体为空 = 无从判断主语 → 护栏不生效，冲突门照常。"""
    result = vf.compare_prior_and_evidence(
        make_note(body="模型 A 参数 70B。"), evi("模型 B 参数 128B。")
    )
    assert result.verdict == vf.VERDICT_CONFLICTING
    assert any("冲突检测命中" in reason for reason in result.reasons)


# ---- 修复轮 2/5：重要（实体护栏提升为全局前置）-----------------------------


def test_entity_disjoint_with_later_time_is_not_newer() -> None:
    """①实体不交 + 时间更晚 → uncertain（不得升级为 newer/supersede）。

    未修复前：冲突门被跳过 → 时间规则接手 → newer/supersede，判定从"开台账"变成
    "建议覆盖"，而冲突门根本没跑、没有任何取值被反证。实体信号只说明"无法确认同一
    主语"，不足以支撑任何更新判定。
    """
    result = vf.compare_prior_and_evidence(
        make_note(body="模型 A 参数 70B。", entities=["GLM-5.3"]),
        evi("模型 B 参数 128B。", observed_at=LATER, entities=["Qwen-3"]),
    )
    assert result.verdict == vf.VERDICT_UNCERTAIN
    assert result.suggested_action == vf.ACTION_NONE
    joined = "\n".join(result.reasons)
    assert "无法确认主体同一性 → 不做更新判定" in joined
    assert "时间：" not in joined  # 时间规则整条被短路，不再产出更新理由


def test_entity_disjoint_with_source_change_is_not_newer() -> None:
    """②实体不交 + 来源命中且 hash 不同 → uncertain（不得判 newer）。"""
    result = vf.compare_prior_and_evidence(
        source_url_note(entities=["GLM-5.3"], source_changed_at="2026-08-01T00:00:00+00:00"),
        evi(
            "模型 B 的上下文窗口是 128k，另有 256k 可选。",
            source_url=URL,
            content_hash=HASH_NEW,
            entities=["Qwen-3"],
        ),
    )
    assert result.verdict == vf.VERDICT_UNCERTAIN
    assert result.suggested_action == vf.ACTION_NONE
    joined = "\n".join(result.reasons)
    assert "无法确认主体同一性 → 不做更新判定" in joined
    assert "来源变化" not in joined  # 规则 2 被短路
    # 留痕仍带来源全量 hash（结构化字段不受短路影响）
    assert result.source is not None and result.source.evidence_content_hash == HASH_NEW


def test_entity_disjoint_family_versions_do_not_get_pressed_into_newer() -> None:
    """同一家族不同版本（GLM-5.3 vs GLM-4.5）也判不交：真冲突不得被压成 newer。"""
    result = vf.compare_prior_and_evidence(
        make_note(body="GLM-5.3 的上下文窗口是 128k。", entities=["GLM-5.3"]),
        evi("GLM-4.5 的上下文窗口是 64k。", observed_at=LATER, entities=["GLM-4.5"]),
    )
    assert result.verdict == vf.VERDICT_UNCERTAIN
    assert result.verdict not in {vf.VERDICT_NEWER, vf.VERDICT_CONFLICTING}
    assert "无法确认主体同一性 → 不做更新判定" in "\n".join(result.reasons)


def test_entity_disjoint_blocks_consistent() -> None:
    """规则 3 的关联残留：实体不交但槽位取值相同也不得判 consistent（假一致）。"""
    result = vf.compare_prior_and_evidence(
        make_note(body="模型 A 参数 70B。", entities=["GLM-5.3"]),
        evi("模型 B 参数 70B。", entities=["Qwen-3"]),
    )
    assert result.verdict != vf.VERDICT_CONSISTENT
    assert result.verdict == vf.VERDICT_UNCERTAIN
    assert result.suggested_action == vf.ACTION_NONE
    joined = "\n".join(result.reasons)
    assert "无法确认主体同一性 → 不做更新判定" in joined
    assert "一致：" not in joined


@pytest.mark.parametrize(
    ("prior_entities", "evidence_entities"),
    [
        (["GLM-5.3"], ["glm-5-3"]),  # 同实体不同写法（slugify 归一后相交）
        (["GLM-5.3"], ["GLM-5.3", "Qwen-3"]),  # 部分相交
        ([], ["Qwen-3"]),  # 旧记忆无实体
        (["GLM-5.3"], []),  # 证据无实体
        ([], []),  # 两侧都无实体
    ],
)
def test_entity_signal_absent_keeps_original_verdicts(
    prior_entities: list[str], evidence_entities: list[str]
) -> None:
    """③实体相交或任一侧为空 → 原有判定不受影响（回归：时间更晚仍判 newer）。"""
    result = vf.compare_prior_and_evidence(
        make_note(body="GLM-5.3 支持工具调用。", entities=prior_entities),
        evi(
            "GLM-5.3 现在支持工具调用与函数并行。",
            observed_at=LATER,
            entities=evidence_entities,
        ),
    )
    assert result.verdict == vf.VERDICT_NEWER
    assert any("晚于旧记忆" in reason for reason in result.reasons)


def test_entity_disjoint_judge_can_still_override() -> None:
    """实体短路走同一兜底出口：judge 仍可改判（并写明最终判定）。"""
    judge, calls = counting_judge(vf.VERDICT_CONFLICTING)
    result = vf.compare_prior_and_evidence(
        make_note(body="模型 A 参数 70B。", entities=["GLM-5.3"]),
        evi("模型 B 参数 128B。", observed_at=LATER, entities=["Qwen-3"]),
        judge=judge,
    )
    assert len(calls) == 1
    assert result.verdict == vf.VERDICT_CONFLICTING
    assert result.suggested_action == vf.ACTION_OPEN_CONFLICT
    assert result.judge_used is True
    assert "→ 最终判定 conflicting（由语义判官给出，建议动作 open_conflict）" in "\n".join(
        result.reasons
    )


# ---- 修复轮 1/5：随修（Minor a–e）------------------------------------------


def test_similarity_calibration_samples_are_locked() -> None:
    """标定样例实测值锁定（可复算产物；报告 §9.2 的区间以本断言为准）。

    `consistent_similarity=0.8` 是经验阈值，必须能用固定样例复算：同一句话 = 1.0；
    "只换一个数值"的改写落在 0.68–0.95（几乎都在 0.8 之上，所以"换数值"要靠冲突门
    而不是相似度阈值兜住，见下一个用例）；语义相近但换说法 ≈ 0.35；不相关 ≈ 0.08。
    数值改动会让本断言失败——那是提醒重新标定，不是允许放宽。
    """
    samples = {
        ("项目使用 uv 管理依赖，Python 3.12。", "项目使用 uv 管理依赖，Python 3.12。"): 1.0,
        ("该系列有 3 个版本。", "该系列有 4 个版本。"): 0.9259,
        ("支持 30% 的折扣。", "支持 20% 的折扣。"): 0.8828,
        ("发布于 2026-08。", "发布于 2026-09。"): 0.8889,
        ("共 3 个模型。", "共 5 个模型。"): 0.8293,
        ("上下文窗口 128k。", "上下文窗口 256k。"): 0.7009,
        ("GLM-5.3 的上下文窗口是 128k。", "GLM-4.5 的上下文窗口是 64k。"): 0.6844,
        (SWAPPED_PRIOR, SWAPPED_EVIDENCE): 0.9459,
        ("用户偏好中文回答。", "用户喜欢用中文交流。"): 0.3504,
        ("项目使用 uv 管理依赖。", "今天天气不错，适合出门散步。"): 0.0770,
    }
    for (left, right), expected in samples.items():
        assert vf.token_similarity(left, right) == pytest.approx(expected, abs=1e-3), left


def test_value_change_above_consistency_threshold_hits_conflict_gate() -> None:
    """相似度高于一致阈值也不能把"换了数值"判成一致——冲突门必须先兜住。"""
    result = vf.compare_prior_and_evidence(
        make_note(body="该系列有 3 个版本。"), evi("该系列有 4 个版本。")
    )
    assert result.similarity > 0.8
    assert result.verdict == vf.VERDICT_CONFLICTING
    joined = "\n".join(result.reasons)
    assert "旧=3" in joined and "新=4" in joined


def test_source_change_ignores_reviewed_at_and_says_so() -> None:
    """Minor a：来源变化规则不看 reviewed_at（与 freshness 规则 5 取舍不同），留痕写明。"""
    result = vf.compare_prior_and_evidence(
        source_url_note(
            source_changed_at="2026-05-01T00:00:00+00:00",
            reviewed_at="2026-06-05T00:00:00+00:00",
        ),
        evi(MODEL_EVIDENCE, source_url=URL, content_hash=HASH_NEW),
    )
    assert result.verdict == vf.VERDICT_NEWER
    joined = "\n".join(result.reasons)
    assert "注：source_changed_at 不晚于 reviewed_at=2026-06-05T00:00:00+00:00" in joined
    assert "本模块仍按来源变化处理" in joined


def test_comparison_to_dict_carries_full_hashes() -> None:
    """Minor d：结构化留痕带完整新旧 hash（理由里只有 8 位截断，碰撞时字面相同）。"""
    result = vf.compare_prior_and_evidence(
        source_url_note(source_changed_at="2026-08-01T00:00:00+00:00"),
        evi(MODEL_EVIDENCE, source_url=URL, content_hash=HASH_NEW),
    )
    assert result.source is not None
    assert result.source.to_dict() == {
        "url": URL,
        "prior_content_hash": HASH_OLD,
        "evidence_content_hash": HASH_NEW,
        "source_changed_at": "2026-08-01T00:00:00+00:00",
    }
    payload = json.loads(json.dumps(result.to_dict(), ensure_ascii=False))
    assert payload["source"]["prior_content_hash"] == HASH_OLD
    assert payload["source"]["evidence_content_hash"] == HASH_NEW
    assert HASH_OLD not in "\n".join(result.reasons)  # 理由仍只有前 8 位
    assert "1a2b3c4d" in "\n".join(result.reasons)


def test_source_trace_is_none_without_url_hit() -> None:
    result = vf.compare_prior_and_evidence(make_note(), evi("项目使用 uv 管理依赖。"))
    assert result.source is None
    assert result.to_dict()["source"] is None


def test_judge_override_rewrites_uncertain_tail() -> None:
    """Minor c：judge 改判后不得再留"→ uncertain"结句（留痕与 verdict 自相牵制）。"""
    judge, _ = counting_judge(vf.VERDICT_NEWER)
    result = vf.compare_prior_and_evidence(
        make_note(body="用户偏好中文回答。"), evi("用户喜欢用中文交流。"), judge=judge
    )
    assert result.verdict == vf.VERDICT_NEWER
    joined = "\n".join(result.reasons)
    assert "→ 最终判定 newer（由语义判官给出，建议动作 supersede）" in joined
    assert "→ uncertain" not in joined
    assert result.reasons[-1].endswith("（由语义判官给出，建议动作 supersede）")


def test_uncertain_tail_kept_when_judge_gives_nothing() -> None:
    """judge 没给出结论时结句仍是 uncertain（结句随最终判定走）。"""
    judge, _ = counting_judge(None)
    result = vf.compare_prior_and_evidence(
        make_note(body="用户偏好中文回答。"), evi("用户喜欢用中文交流。"), judge=judge
    )
    assert result.verdict == vf.VERDICT_UNCERTAIN
    assert result.reasons[-1].endswith("→ uncertain（建议人工或语义判官复核，本轮不做动作）")


def test_time_helpers_are_public_in_freshness_and_shared() -> None:
    """Minor e：基准时间/时间解析提升为 freshness 公开名，旧私有名保持可用且同一对象。"""
    from researchwiki.wiki import freshness as fr

    assert fr.parse_ts is fr._parse_ts
    assert fr.resolve_base_time is fr._resolve_base_time
    assert "parse_ts" in fr.__all__ and "resolve_base_time" in fr.__all__
    # verification 复用的是同一个实现（不是另抄一份）
    assert vf.parse_ts is fr.parse_ts
    assert vf.resolve_base_time is fr.resolve_base_time
    meta = NoteMeta(id="N-1", observed_at=None, created=NOW)
    assert fr.resolve_base_time(meta) == ("created", fr.parse_ts(NOW))


def test_public_time_helper_keeps_lenient_parsing() -> None:
    """公开名行为与旧口径一致：无时区按 UTC、非法值回 None、bool 不当时间。"""
    from researchwiki.wiki import freshness as fr

    assert fr.parse_ts("2026-06-01T00:00:00").isoformat() == NOW
    assert fr.parse_ts("昨天") is None
    assert fr.parse_ts("") is None
    assert fr.parse_ts(True) is None
    assert fr.parse_ts(None) is None


# ---- 修复轮 3/5：判官越权拦截（结构性不变量）--------------------------------


def test_guard_is_machine_readable_on_entity_shortfall() -> None:
    """①护栏触发时 guard / judge_verdict 机器可读（接线层不必匹配中文理由）。"""
    judge, calls = counting_judge(None)
    result = vf.compare_prior_and_evidence(
        make_note(body="模型 A 参数 70B。", entities=["GLM-5.3"]),
        evi("模型 B 参数 128B。", observed_at=LATER, entities=["Qwen-3"]),
        judge=judge,
    )
    assert len(calls) == 1
    assert result.guard == vf.GUARD_ENTITIES_DISJOINT
    assert result.judge_verdict is None  # 判官没给合法 verdict
    payload = json.loads(json.dumps(result.to_dict(), ensure_ascii=False))
    assert payload["guard"] == "entities_disjoint"
    assert payload["judge_verdict"] is None


def test_guard_rejects_judge_override_to_newer() -> None:
    """②护栏触发 + judge 返回 newer → 最终 uncertain / none，judge_verdict 仍可读。

    判官给出的覆盖动作（新证据更晚 → supersede）在主体同一性未确认时被结构性拒绝：
    最终判定保持 uncertain、建议动作 none（memory_update 层据此不会写盘），
    但"判官说了什么"完整留痕（判官越权拦截，修复轮 3/5）。
    """
    judge, calls = counting_judge(vf.VERDICT_NEWER)
    result = vf.compare_prior_and_evidence(
        make_note(body="模型 A 参数 70B。", entities=["GLM-5.3"]),
        evi("模型 B 参数 128B。", observed_at=LATER, entities=["Qwen-3"]),
        judge=judge,
    )
    assert len(calls) == 1
    assert result.guard == vf.GUARD_ENTITIES_DISJOINT
    assert result.judge_verdict == vf.VERDICT_NEWER  # 原始 verdict 可读
    assert result.verdict == vf.VERDICT_UNCERTAIN
    assert result.suggested_action == vf.ACTION_NONE
    assert result.judge_used is True
    joined = "\n".join(result.reasons)
    assert "判官建议被护栏拒绝" in joined
    assert "保持 uncertain" in joined
    # 不得再出现"最终判定 newer"这类结句（判官建议没被采纳）
    assert "→ 最终判定 newer" not in joined
    assert result.reasons[-1].endswith("→ uncertain（建议人工或语义判官复核，本轮不做动作）")


@pytest.mark.parametrize(
    "rejected",
    [vf.VERDICT_NEWER, vf.VERDICT_MORE_SPECIFIC, vf.VERDICT_CONSISTENT],
)
def test_guard_rejects_every_non_conservative_judge_verdict(rejected: str) -> None:
    """护栏只放行 conflicting / uncertain：三个非保守判定一律被拒（覆盖动作 + 假一致）。"""
    judge, _ = counting_judge(rejected)
    result = vf.compare_prior_and_evidence(
        make_note(body="模型 A 参数 70B。", entities=["GLM-5.3"]),
        evi("模型 B 参数 128B。", entities=["Qwen-3"]),
        judge=judge,
    )
    assert result.verdict == vf.VERDICT_UNCERTAIN
    assert result.judge_verdict == rejected


def test_guard_allows_judge_conflicting() -> None:
    """③护栏触发 + judge 返回 conflicting → 允许（保守动作 open_conflict）。"""
    judge, calls = counting_judge(vf.VERDICT_CONFLICTING)
    result = vf.compare_prior_and_evidence(
        make_note(body="模型 A 参数 70B。", entities=["GLM-5.3"]),
        evi("模型 B 参数 128B。", entities=["Qwen-3"]),
        judge=judge,
    )
    assert len(calls) == 1
    assert result.verdict == vf.VERDICT_CONFLICTING
    assert result.suggested_action == vf.ACTION_OPEN_CONFLICT
    assert result.judge_verdict == vf.VERDICT_CONFLICTING
    assert result.guard == vf.GUARD_ENTITIES_DISJOINT
    assert "判官建议被护栏拒绝" not in "\n".join(result.reasons)


def test_no_guard_keeps_judge_behaviour_unchanged() -> None:
    """④无护栏时判官行为逐字段不变（回归）：newer 照常采纳，guard 恒 None。"""
    judge, calls = counting_judge(vf.VERDICT_NEWER)
    result = vf.compare_prior_and_evidence(
        make_note(body="用户偏好中文回答。"), evi("用户喜欢用中文交流。"), judge=judge
    )
    assert len(calls) == 1
    assert result.guard is None
    assert result.judge_verdict == vf.VERDICT_NEWER
    assert result.verdict == vf.VERDICT_NEWER
    assert result.suggested_action == vf.ACTION_SUPERSEDE
    assert vf.CONSERVATIVE_VERDICTS == (vf.VERDICT_CONFLICTING, vf.VERDICT_UNCERTAIN)


def test_deterministic_paths_leave_guard_and_judge_verdict_none() -> None:
    """judge=None 的确定性路径：guard / judge_verdict 恒 None（不误报护栏）。"""
    conflict = vf.compare_prior_and_evidence(
        make_note(body="共 3 个模型"), evi("共 5 个模型")
    )
    assert conflict.verdict == vf.VERDICT_CONFLICTING
    assert conflict.guard is None and conflict.judge_verdict is None
    consistent = vf.compare_prior_and_evidence(
        make_note(body="项目使用 uv 管理依赖。"), evi("项目使用 uv 管理依赖。")
    )
    assert consistent.verdict == vf.VERDICT_CONSISTENT
    assert consistent.guard is None and consistent.judge_verdict is None


# ---- 终审 F2：替换门（supersede_min_similarity，监察者裁定）------------------


# 主题相邻、facet 不同的一对（终审实测 sim≈0.52）：旧记忆讲 Letta 的后台子代理，
# 证据讲 MemGPT 的分页机制，两侧都没有事实令牌（具体度规则不会命中），时间更晚。
ADJACENT_PRIOR = "Letta 的后台子代理方案在长会话里更省 token。"
ADJACENT_EVIDENCE = "MemGPT 的分页机制在长文档里更省 token。"
# 低相似度**冲突**对（sim≈0.52，命中 number:k:上下文窗口 槽位）：开台账是保守动作，
# 替换门不该拦它。
LOW_SIM_CONFLICT_PRIOR = "上下文窗口 128k，模型 A 定位多轮长会话。"
LOW_SIM_CONFLICT_EVIDENCE = "上下文窗口 256k，模型 B 面向单轮短任务，成本结构完全不同。"


def test_supersede_gate_blocks_facet_adjacent_later_evidence() -> None:
    """F2①：异 facet 的更晚证据（sim≈0.52 < 替换门 0.60）不得判 newer/supersede。

    没有这道门时规则 4（时间）会直接判 newer——地板 0.3 只保证"不是完全无关"，
    实体为空时实体护栏也不设防，于是"主题相邻、facet 不同"的更晚证据能把仍然有效
    的旧记忆覆盖掉（静默覆盖最现实的形态）。
    """
    note = make_note(body=ADJACENT_PRIOR)
    evidence = evi(ADJACENT_EVIDENCE, observed_at=LATER)
    similarity = vf.token_similarity(ADJACENT_PRIOR, ADJACENT_EVIDENCE)
    assert 0.3 <= similarity < vf.DEFAULT_SUPERSEDE_MIN_SIMILARITY  # 前提：落在地板与门之间

    result = vf.compare_prior_and_evidence(note, evidence)
    assert result.verdict == vf.VERDICT_UNCERTAIN
    assert result.suggested_action == vf.ACTION_NONE
    assert result.similarity == similarity
    joined = "\n".join(result.reasons)
    assert "低于替换门 0.60" in joined
    assert f"相似度 {similarity:.3f}" in joined  # 理由写出实际相似度
    assert "不足以放行覆盖动作（supersede）" in joined
    assert "→ 新证据更新（建议 supersede）" not in joined
    assert result.to_dict()["verdict"] == vf.VERDICT_UNCERTAIN


def test_supersede_gate_keeps_legitimate_update_and_conflict() -> None:
    """F2②：同主题同 facet 的高相似更新仍判 newer；conflicting 不受门约束。"""
    # 高相似（sim≈0.83 > 0.60）的更晚 / 更具体证据 → 照旧 newer
    legit = vf.compare_prior_and_evidence(
        make_note(body="模型 A 的上下文窗口是 128k。"),
        evi("模型 A 的上下文窗口是 128k，另有 256k 可选。", observed_at=LATER),
    )
    assert legit.verdict == vf.VERDICT_NEWER
    assert legit.suggested_action == vf.ACTION_SUPERSEDE
    assert "→ 新证据更新（建议 supersede）" in "\n".join(legit.reasons)

    # 低相似（sim≈0.52）但同槽位取值矛盾 → 仍判 conflicting（开台账是保守动作）
    low_sim = vf.token_similarity(LOW_SIM_CONFLICT_PRIOR, LOW_SIM_CONFLICT_EVIDENCE)
    assert low_sim < vf.DEFAULT_SUPERSEDE_MIN_SIMILARITY
    conflict = vf.compare_prior_and_evidence(
        make_note(body=LOW_SIM_CONFLICT_PRIOR),
        evi(LOW_SIM_CONFLICT_EVIDENCE, observed_at=LATER),
    )
    assert conflict.verdict == vf.VERDICT_CONFLICTING
    assert conflict.suggested_action == vf.ACTION_OPEN_CONFLICT
    assert "低于替换门" not in "\n".join(conflict.reasons)


def test_supersede_gate_default_is_above_floor_and_configurable() -> None:
    """F2③：缺省高于地板、可经 [verification] 段覆盖，且不低于地板。"""
    default = vf.VerificationSettings()
    assert default.supersede_min_similarity == vf.DEFAULT_SUPERSEDE_MIN_SIMILARITY == 0.6
    assert default.similarity_floor < default.supersede_min_similarity < (
        default.consistent_similarity
    )
    assert "DEFAULT_SUPERSEDE_MIN_SIMILARITY" in vf.__all__

    # 段内覆盖（两种口径：段本身 / 完整 config）都生效（0.7 是收窄，也在区间内）
    section = {"supersede_min_similarity": 0.7}
    assert vf.VerificationSettings.from_config(section).supersede_min_similarity == 0.7
    assert (
        vf.VerificationSettings.from_config({"verification": section}).supersede_min_similarity
        == 0.7
    )
    # 非法值宽容回退默认；低于地板时被夹到地板（地板是"连比较都不做"的绝对下界）
    assert (
        vf.VerificationSettings.from_config({"supersede_min_similarity": "乱写"})
        .supersede_min_similarity
        == vf.DEFAULT_SUPERSEDE_MIN_SIMILARITY
    )
    assert vf.VerificationSettings(supersede_min_similarity=0.1).supersede_min_similarity == (
        vf.DEFAULT_SIMILARITY_FLOOR
    )

    # 把门放到最低（0.0 被夹到地板 0.3）：同一对异 facet 证据恢复判 newer
    # ——证明是这道门在拦，而不是别的原因
    lowered = vf.VerificationSettings(supersede_min_similarity=0.0)
    relaxed = vf.compare_prior_and_evidence(
        make_note(body=ADJACENT_PRIOR),
        evi(ADJACENT_EVIDENCE, observed_at=LATER),
        settings=lowered,
    )
    assert relaxed.verdict == vf.VERDICT_NEWER
    assert "低于替换门" not in "\n".join(relaxed.reasons)
