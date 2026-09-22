"""记忆更新（RQ2 闭环：新证据 vs 旧记忆）测试。

覆盖 task-12 简报的验收清单：

- 五类动作分发各一例（含 newer 无来源 → 降级 open_conflict、more_specific → merge）；
- dry_run 零写盘（真实 store 断言笔记 md 逐字节未变、台账未增）；
- 幂等（同批 evidence 跑两次：第二次全 skipped、无新笔记/新冲突）；
- 冲突台账真写入且带两侧证据（读 list_conflicts 断言）；supersede 后旧 ID 沿链
  可达新记忆、新记忆带来源；
- refresh_reviewed_at 只改 reviewed_at（NoteMeta 逐字段断言）；
- 护栏防御纵深（guard 非 None 时拒绝 supersede/merge）；
- AgentLoop 集成（脚本化 provider）：state.md 出现"记忆更新"行、
  run_dir/memory-update.json 生成、run-metrics 14 字段契约不变；
  memory_update_config=None / enabled=false / dry_run=true 三种形态各自的行为。

全部零网络、零模型调用（比较判定用确定性 Mock 嵌入，AgentLoop 用脚本 provider）。
"""

import json
from dataclasses import fields
from pathlib import Path

import pytest

from researchwiki.llm.accounting import TokenAccountant
from researchwiki.llm.provider import ScriptedProvider, StreamEvent, TokenUsage
from researchwiki.loop.agent_loop import AgentLoop
from researchwiki.loop.memory_update import (
    APPLIED_CONFLICT_OPENED,
    APPLIED_MERGED,
    APPLIED_REVIEWED_AT,
    APPLIED_SKIPPED,
    APPLIED_SUPERSEDED,
    CONFLICT_KEY_FIELD,
    COUNT_KEYS,
    MAX_RECORDED_KEYS,
    MEMORY_UPDATE_KEYS,
    MEMORY_UPDATE_REASONS,
    SUPERSEDE_REASON_KEY,
    UPDATE_REASONS_KEY,
    MemoryUpdateReport,
    apply_comparisons,
    body_key,
    candidate_source_refs,
    conflict_has_body_key,
    conflict_has_key,
    evidence_from_run,
    memory_update_settings,
    recorded_body_keys,
    recorded_keys,
    update_key,
)
from researchwiki.tools import MockSearch
from researchwiki.wiki.distiller import CandidateNote
from researchwiki.wiki.embeddings import MockEmbeddingProvider
from researchwiki.wiki.frontmatter import NoteMeta, SourceRef
from researchwiki.wiki.store import Note, WikiStore
from researchwiki.wiki.verification import (
    ACTION_MERGE,
    ACTION_SUPERSEDE,
    GUARD_ENTITIES_DISJOINT,
    VERDICT_CONFLICTING,
    VERDICT_CONSISTENT,
    VERDICT_MORE_SPECIFIC,
    VERDICT_NEWER,
    VERDICT_UNCERTAIN,
    EvidenceComparison,
    EvidenceItem,
    compare_prior_and_evidence,
)

NOW = "2026-09-01T00:00:00+00:00"
LATER = "2026-10-01T00:00:00+00:00"
EARLIER = "2026-06-01T00:00:00+00:00"
URL = "https://example.test/doc"
URL_B = "https://example.test/other"
HASH_OLD = "1a2b3c4d5e6f7788"
HASH_NEW = "9f8e7d6c5b4a3928"

# 五类判定的固定样例（实测：同一对文本恒得同一 verdict，见 tests/test_verification.py）
BODY_CONSISTENT = "项目使用 uv 管理依赖，Python 3.12。"
BODY_NEWER = "模型 A 的上下文窗口是 128k。"
EVIDENCE_NEWER = "模型 A 的上下文窗口是 128k，另有 256k 可选。"
BODY_SPECIFIC = "该系列有 3 个版本。"
EVIDENCE_SPECIFIC = "该系列有 3 个版本，Pro 版本参数 70B。"
BODY_CONFLICT = "共 3 个方案支持长期记忆。"
EVIDENCE_CONFLICT = "共 5 个方案支持长期记忆。"


# ---- 脚手架 -----------------------------------------------------------------


def make_store(tmp_path: Path) -> WikiStore:
    return WikiStore(tmp_path / "wiki-data")


def pairs_for(
    store: WikiStore, specs: list[tuple[Note, EvidenceItem]]
) -> list[tuple[EvidenceComparison, EvidenceItem]]:
    """(prior, item) 列表 → (comparison, item) 列表，走真实判定模块（不是手搓结论）。

    ``evidence_index`` 必须与证据列表下标对齐——这是 `compare_batch` 在生产路径
    里给出的契约（执行层按下标取证据），所以测试也按同一约定构造（下标错位会
    把别的证据当成本次证据处置）。
    """
    del store  # 判定零副作用，store 只是调用方的自然入口
    return [
        (compare_prior_and_evidence(prior, item, evidence_index=index), item)
        for index, (prior, item) in enumerate(specs)
    ]


def apply(
    store: WikiStore,
    pairs: list[tuple[EvidenceComparison, EvidenceItem]],
    *,
    now: str = NOW,
    dry_run: bool = False,
) -> MemoryUpdateReport:
    return apply_comparisons(
        store,
        [comparison for comparison, _ in pairs],
        evidence=[item for _, item in pairs],
        now=now,
        dry_run=dry_run,
        trace_id="trace-test",
    )


def one(store: WikiStore, prior: Note, item: EvidenceItem, **kwargs):
    """比较 + 执行一条（返回 (comparison, report, action)）。"""
    pairs = pairs_for(store, [(prior, item)])
    comparison = pairs[0][0]
    report = apply(store, pairs, **kwargs)
    assert len(report.actions) == 1
    return comparison, report, report.actions[0]


def note_files(root: Path) -> dict[str, str]:
    """notes/ 与 conflicts/ 下所有 md 的字节内容快照（dry_run 的"零写盘"断言用）。"""
    out: dict[str, str] = {}
    for sub in ("notes", "conflicts"):
        directory = root / sub
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.md")):
            out[f"{sub}/{path.name}"] = path.read_text(encoding="utf-8")
    return out


def assert_meta_intact(before: Note, after: Note, *, skip: tuple[str, ...] = ("reviewed_at",)):
    """NoteMeta 逐字段断言（extra 单看：原键必须原样保留，新键允许追加）。"""
    for f in fields(NoteMeta):
        if f.name == "extra" or f.name in skip:
            continue
        assert getattr(before.meta, f.name) == getattr(after.meta, f.name), f.name
    assert before.body == after.body
    for key, value in before.meta.extra.items():
        assert after.meta.extra.get(key) == value, f"extra[{key}] 被改动"


def make_prior(store: WikiStore, body: str, **kwargs) -> Note:
    kwargs.setdefault("note_id", "N-0001")
    kwargs.setdefault("title", "判定测试")
    kwargs.setdefault("observed_at", EARLIER)
    kwargs.setdefault("created", EARLIER)
    return store.save_note(body, **kwargs)


# ---- 证据构造与配置 ---------------------------------------------------------


def test_evidence_from_run_maps_candidates_to_items() -> None:
    """候选 → EvidenceItem：正文/时间/来源/hash/实体逐字段落位（保序一一对应）。"""
    refs = [SourceRef(url=URL, content_hash=HASH_NEW), SourceRef(url=URL_B, content_hash="")]
    candidates = [
        CandidateNote(text="  第一条事实。  ", entities=["Letta"], source_refs=[refs[0]]),
        CandidateNote(text="第二条事实。", source_urls=[URL_B], source_refs=[]),
        CandidateNote(text="无来源的第三条。"),
    ]
    items = evidence_from_run(candidates, source_refs=refs, now=NOW)
    assert [item.text for item in items] == ["第一条事实。", "第二条事实。", "无来源的第三条。"]
    assert [item.observed_at for item in items] == [NOW, NOW, NOW]
    assert items[0].source_url == URL and items[0].content_hash == HASH_NEW
    # source_refs 为空 → 按 URL 回落到来源池的写法（hash 仍从池里取）
    assert items[1].source_url == URL_B
    assert items[2].source_url == "" and items[2].content_hash == ""
    assert items[0].entities == ["Letta"]
    # 观察时间缺省 = 当前 UTC（不注入 now 时仍合法 ISO）
    assert evidence_from_run(candidates[:1], source_refs=refs)[0].observed_at


def test_candidate_source_refs_prefers_matched_refs_then_urls() -> None:
    pool = [SourceRef(url=URL, content_hash=HASH_NEW)]
    matched = CandidateNote(text="x", source_refs=[SourceRef(url=URL, content_hash=HASH_OLD)])
    assert candidate_source_refs(matched, source_refs=pool) == [
        SourceRef(url=URL, content_hash=HASH_OLD)
    ]
    url_only = CandidateNote(text="x", source_urls=[URL, URL_B], source_refs=[])
    assert candidate_source_refs(url_only, source_refs=pool) == [
        SourceRef(url=URL, content_hash=HASH_NEW),
        SourceRef(url=URL_B, content_hash=""),
    ]


def test_memory_update_settings_parsing() -> None:
    """[memory_update] 段：缺省全默认；段/完整 config 两种口径都能解析；非法值回默认。"""
    default = memory_update_settings(None)
    assert default.enabled is True and default.dry_run is False
    assert memory_update_settings({"enabled": False, "dry_run": True}) == memory_update_settings(
        {"memory_update": {"enabled": False, "dry_run": True}}
    )
    assert memory_update_settings({"enabled": "false"}) .enabled is False  # 字符串词表
    assert memory_update_settings({"enabled": "乱写"}).enabled is True  # 非法值回默认
    assert memory_update_settings({"dry_run": "on"}).dry_run is True


def test_update_key_is_stable_and_scoped_to_prior() -> None:
    first = update_key("N-0001", "同一条证据")
    assert first == update_key("N-0001", "  同一条证据  ")  # strip 后同键
    assert first != update_key("N-0002", "同一条证据")  # 同一证据命中多条 Prior 不互相去重
    assert first != update_key("N-0001", "另一条证据")
    assert first.startswith("N-0001:")


# ---- 1. consistent → 只刷新 reviewed_at ------------------------------------


def test_consistent_refreshes_only_reviewed_at(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    prior = make_prior(
        store,
        BODY_CONSISTENT,
        title="依赖管理",
        entities=["Letta"],
        confidence="high",
        volatility="drifting",
        kind="user",
        importance=0.7,
        reviewed_at=None,
        valid_from=EARLIER,
        valid_until="2027-01-01T00:00:00+00:00",
        source_changed_at="2026-08-01T00:00:00+00:00",
        trace_id="trace-old",
        sources=[SourceRef(url=URL, content_hash=HASH_OLD)],
        extra={
            "formation_reason": "既有理由",
            UPDATE_REASONS_KEY: [{"reason": "旧", "at": EARLIER}],
        },
    )
    item = EvidenceItem(text=BODY_CONSISTENT, observed_at=None)
    comparison, report, action = one(store, prior, item)

    assert comparison.verdict == VERDICT_CONSISTENT  # 前提：样例确实判一致
    assert action.applied == APPLIED_REVIEWED_AT and action.note_id == "N-0001"
    assert report.counts["reviewed"] == 1

    after = store.get_note("N-0001")
    assert after is not None
    assert after.meta.reviewed_at == NOW
    assert_meta_intact(prior, after)  # 除 reviewed_at 外逐字段不变（含 sources）
    assert after.meta.sources == [SourceRef(url=URL, content_hash=HASH_OLD)]
    # 留痕：幂等键 + 理由记录，原 extra 键原样保留
    assert MEMORY_UPDATE_KEYS in after.meta.extra  # 幂等键落盘位置（重跑据此跳过）
    assert recorded_keys(after) == [update_key("N-0001", BODY_CONSISTENT)]
    assert after.meta.extra[UPDATE_REASONS_KEY] == [{"reason": "旧", "at": EARLIER}]
    history = after.meta.extra[MEMORY_UPDATE_REASONS]
    assert history[-1]["verdict"] == VERDICT_CONSISTENT
    assert f"verdict={VERDICT_CONSISTENT}" in action.reason


# ---- 2. newer → supersede（带来源、旧 ID 沿链可达）--------------------------


def test_newer_with_source_supersedes_and_keeps_chain(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    prior = make_prior(
        store,
        BODY_NEWER,
        title="模型 A 规格",
        entities=["模型A"],
        volatility="drifting",
        kind="experience",
        importance=0.6,
        sources=[SourceRef(url=URL, content_hash=HASH_OLD)],
        source_changed_at="2026-08-01T00:00:00+00:00",
    )
    item = EvidenceItem(
        text=EVIDENCE_NEWER, observed_at=LATER, source_url=URL, content_hash=HASH_NEW
    )
    comparison, report, action = one(store, prior, item)

    assert comparison.verdict == VERDICT_NEWER  # 前提：来源换版 → newer
    assert action.applied == APPLIED_SUPERSEDED
    assert report.counts["superseded"] == 1
    new_id = action.note_id
    assert new_id == "N-0002"

    # 旧 ID 沿链可达新记忆，新记忆带来源（不是无证据的替代记忆）
    final = store.follow_redirect("N-0001")
    assert final is not None and final.id == new_id
    assert final.body == EVIDENCE_NEWER
    assert final.meta.sources == [SourceRef(url=URL, content_hash=HASH_NEW)]
    assert final.meta.observed_at == LATER  # 时间基准追到证据本身
    # 不继承旧证据的置信度，按 **formation 确定性口径**现算：正文含具体事实要素
    # （128k/256k）+ 有来源 → high（修复轮 1/5 I-3：原来硬编码 medium，自动通道
    # 再也给不出 high，而 prior.py 会把 "置信度: medium" 直接注入模型的核验上下文）
    assert final.meta.confidence == "high"
    assert final.meta.entities == ["模型A"]  # 身份连续性
    assert final.meta.volatility == "drifting" and final.meta.kind == "experience"
    assert final.meta.importance == 0.6
    assert final.meta.extra[SUPERSEDE_REASON_KEY] == action.reason

    # 旧笔记原样保留（status=superseded + 链 + 留痕），正文不被覆盖
    old = store.get_note("N-0001")
    assert old is not None and old.status == "superseded"
    assert old.meta.superseded_by == new_id and old.body == BODY_NEWER
    assert old.meta.extra[SUPERSEDE_REASON_KEY] == action.reason
    assert update_key("N-0001", EVIDENCE_NEWER) in recorded_keys(old)

    # 理由含 verdict + 关键值 + 证据来源 URL（"不静默覆盖"的审计要求）
    assert f"verdict={VERDICT_NEWER}" in action.reason
    assert URL in action.reason and HASH_NEW[:8] in action.reason


# ---- 3. newer 但证据无来源 → 降级 open_conflict -------------------------------


def test_newer_without_source_degrades_to_conflict(tmp_path: Path) -> None:
    """宁可开台账也不造无证据的替代记忆（动作分发表的硬约束）。"""
    store = make_store(tmp_path)
    prior = make_prior(store, BODY_NEWER)
    item = EvidenceItem(text=EVIDENCE_NEWER, observed_at=LATER)  # 无 source_url
    comparison, report, action = one(store, prior, item)

    assert comparison.verdict == VERDICT_NEWER and comparison.suggested_action == ACTION_SUPERSEDE
    assert action.applied == APPLIED_CONFLICT_OPENED
    assert action.verdict == VERDICT_NEWER and action.suggested_action == ACTION_SUPERSEDE
    assert "降级为 open_conflict" in action.reason
    assert report.counts["conflicts"] == 1 and report.counts["superseded"] == 0
    # 旧记忆没被替代：仍是 active、没建新笔记
    assert [n.id for n in store.list_notes()] == ["N-0001"]
    conflicts = store.list_conflicts()
    assert len(conflicts) == 1
    assert "本轮证据更新缺少可用来源" in conflicts[0].question


# ---- 4. more_specific → merge（并入正文与来源、保留规范 ID）------------------


def test_more_specific_merges_body_and_sources(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    prior = make_prior(
        store,
        BODY_SPECIFIC,
        title="版本清单",
        entities=["GLM-5.3"],
        sources=[SourceRef(url=URL, content_hash=HASH_OLD)],
        extra={"formation_reason": "保留我"},
    )
    item = EvidenceItem(
        text=EVIDENCE_SPECIFIC, observed_at=None, source_url=URL_B, content_hash=HASH_NEW
    )
    comparison, report, action = one(store, prior, item)

    assert comparison.verdict == VERDICT_MORE_SPECIFIC
    assert action.applied == APPLIED_MERGED and action.note_id == "N-0001"
    assert report.counts["merged"] == 1

    after = store.get_note("N-0001")
    assert after is not None and after.status == "active"  # 保留规范 ID（不 supersede）
    assert after.body.startswith(BODY_SPECIFIC)
    assert EVIDENCE_SPECIFIC in after.body  # 新事实并入正文
    assert after.meta.sources == [
        SourceRef(url=URL, content_hash=HASH_OLD),
        SourceRef(url=URL_B, content_hash=HASH_NEW),
    ]  # 追加来源，不替换旧证据
    assert after.meta.reviewed_at == NOW
    assert after.meta.title == "版本清单" and after.meta.entities == ["GLM-5.3"]
    assert after.meta.extra["formation_reason"] == "保留我"
    assert after.meta.extra[UPDATE_REASONS_KEY][-1]["reason"] == action.reason
    assert "新增来源 1 条" in action.reason
    assert sum(1 for _ in store.notes_dir.glob("N-*.md")) == 1  # 没有新 ID


# ---- 5. conflicting → 冲突台账（带两侧证据）--------------------------------


def test_conflicting_writes_ledger_with_both_sides(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    prior = make_prior(
        store,
        BODY_CONFLICT,
        sources=[SourceRef(url=URL, content_hash=HASH_OLD)],
    )
    item = EvidenceItem(
        text=EVIDENCE_CONFLICT, observed_at=LATER, source_url=URL_B, content_hash=HASH_NEW
    )
    comparison, report, action = one(store, prior, item)

    assert comparison.verdict == VERDICT_CONFLICTING
    assert action.applied == APPLIED_CONFLICT_OPENED and report.counts["conflicts"] == 1
    conflicts = store.list_conflicts()
    assert len(conflicts) == 1
    conflict = conflicts[0]
    assert conflict.status == "open"
    assert conflict.id == action.note_id
    # 两侧证据都在（旧 note ID + 正文摘录 + 来源；新证据正文 + 来源 + 槽位取值）
    assert conflict.claim_a["note_id"] == "N-0001"
    assert conflict.claim_a["excerpt"] == BODY_CONFLICT
    assert conflict.claim_a["sources"] == [{"url": URL, "content_hash": HASH_OLD}]
    assert conflict.claim_a["verdict"] == VERDICT_CONFLICTING
    assert conflict.claim_b["text"] == EVIDENCE_CONFLICT
    assert conflict.claim_b["source_url"] == URL_B
    assert conflict.claim_b["content_hash"] == HASH_NEW
    assert conflict.claim_b["conflicts"][0]["prior_value"] == "3"
    assert conflict.claim_b["conflicts"][0]["evidence_value"] == "5"
    # 幂等键两侧都写（台账没有独立 extra 槽）
    assert conflict.claim_a[CONFLICT_KEY_FIELD] == update_key("N-0001", EVIDENCE_CONFLICT)
    assert conflict_has_key(conflict, update_key("N-0001", EVIDENCE_CONFLICT))
    assert not conflict_has_key(conflict, "别的键")
    assert "旧=" in conflict.question and "新=" in conflict.question
    # 旧记忆不被覆盖、也不被 merge 吞掉
    after = store.get_note("N-0001")
    assert after is not None and after.body == BODY_CONFLICT and after.status == "active"


# ---- 6. uncertain → none（不写盘，只计数）-----------------------------------


def test_uncertain_does_nothing(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    prior = make_prior(store, "项目使用 uv 管理依赖。")
    item = EvidenceItem(text="今天天气很好，适合出门散步。", observed_at=None)
    before = note_files(tmp_path / "wiki-data")
    comparison, report, action = one(store, prior, item)

    assert comparison.verdict == VERDICT_UNCERTAIN
    assert action.applied == APPLIED_SKIPPED and action.note_id is None
    assert "uncertain" in action.reason
    assert report.counts == {
        "reviewed": 0,
        "superseded": 0,
        "merged": 0,
        "conflicts": 0,
        "skipped": 1,
    }
    assert note_files(tmp_path / "wiki-data") == before  # 零写盘
    after = store.get_note("N-0001")
    assert after is not None and after.meta.reviewed_at is None and after.meta.extra == {}


# ---- 7. dry_run：只报告不写盘 -----------------------------------------------


def test_dry_run_reports_without_writing(tmp_path: Path) -> None:
    """同一批（一致 / 更新 / 冲突各一）dry_run：动作与计数照常，磁盘逐字节未变。"""
    store = make_store(tmp_path)
    consistent = make_prior(store, BODY_CONSISTENT, note_id="N-0001")
    newer = make_prior(
        store,
        BODY_NEWER,
        note_id="N-0002",
        sources=[SourceRef(url=URL, content_hash=HASH_OLD)],
        source_changed_at="2026-08-01T00:00:00+00:00",
    )
    conflicting = make_prior(store, BODY_CONFLICT, note_id="N-0003")
    pairs = pairs_for(
        store,
        [
            (consistent, EvidenceItem(text=BODY_CONSISTENT, observed_at=None)),
            (
                newer,
                EvidenceItem(
                    text=EVIDENCE_NEWER,
                    observed_at=LATER,
                    source_url=URL,
                    content_hash=HASH_NEW,
                ),
            ),
            (conflicting, EvidenceItem(text=EVIDENCE_CONFLICT, observed_at=None)),
        ],
    )
    before = note_files(tmp_path / "wiki-data")
    report = apply(store, pairs, dry_run=True)

    assert report.dry_run is True
    assert [a.applied for a in report.actions] == [
        APPLIED_REVIEWED_AT,
        APPLIED_SUPERSEDED,
        APPLIED_CONFLICT_OPENED,
    ]
    assert report.counts == {
        "reviewed": 1,
        "superseded": 1,
        "merged": 0,
        "conflicts": 1,
        "skipped": 0,
    }
    assert note_files(tmp_path / "wiki-data") == before
    assert [n.id for n in store.list_notes()] == ["N-0001", "N-0002", "N-0003"]
    assert store.list_conflicts() == []
    assert store.get_note("N-0001").meta.reviewed_at is None
    # dry_run 的报告形态与真跑一致（可先 dry 后真）
    assert report.to_dict()["dry_run"] is True and report.state_line().endswith("跳过 0")


# ---- 8. 幂等：同批证据跑两次，第二次全 skipped --------------------------------


def test_second_run_with_same_evidence_is_all_skipped(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    consistent = make_prior(store, BODY_CONSISTENT, note_id="N-0001")
    specific = make_prior(store, BODY_SPECIFIC, note_id="N-0002")
    conflicting = make_prior(store, BODY_CONFLICT, note_id="N-0003")

    def build():
        """同一批证据重建一遍比较（重跑时判定输入完全一致）。"""
        return pairs_for(
            store,
            [
                (consistent, EvidenceItem(text=BODY_CONSISTENT, observed_at=None)),
                (
                    specific,
                    EvidenceItem(
                        text=EVIDENCE_SPECIFIC,
                        observed_at=None,
                        source_url=URL_B,
                        content_hash=HASH_NEW,
                    ),
                ),
                (conflicting, EvidenceItem(text=EVIDENCE_CONFLICT, observed_at=None)),
            ],
        )

    first = apply(store, build(), now=NOW)
    assert [a.applied for a in first.actions] == [
        APPLIED_REVIEWED_AT,
        APPLIED_MERGED,
        APPLIED_CONFLICT_OPENED,
    ]
    notes_after_first = sorted(n.id for n in store.list_notes(status=None))
    conflicts_after_first = [c.id for c in store.list_conflicts()]
    bodies_after_first = {n.id: n.body for n in store.list_notes(status=None)}

    second = apply(store, build(), now=LATER)
    assert [a.applied for a in second.actions] == [APPLIED_SKIPPED] * 3
    assert second.counts == {
        "reviewed": 0,
        "superseded": 0,
        "merged": 0,
        "conflicts": 0,
        "skipped": 3,
    }
    assert all("幂等" in a.reason for a in second.actions)
    assert sorted(n.id for n in store.list_notes(status=None)) == notes_after_first
    assert [c.id for c in store.list_conflicts()] == conflicts_after_first
    # 第二次没有改写任何正文、也没刷新 reviewed_at（幂等命中就直接跳过）
    assert {n.id: n.body for n in store.list_notes(status=None)} == bodies_after_first
    assert store.get_note("N-0001").meta.reviewed_at == NOW


def test_supersede_rerun_is_skipped_even_though_prior_is_retired(tmp_path: Path) -> None:
    """supersede 后旧笔记不再 active：重跑必须靠幂等键跳过（而不是重复建笔记）。"""
    store = make_store(tmp_path)
    prior = make_prior(
        store,
        BODY_NEWER,
        sources=[SourceRef(url=URL, content_hash=HASH_OLD)],
        source_changed_at="2026-08-01T00:00:00+00:00",
    )
    item = EvidenceItem(
        text=EVIDENCE_NEWER, observed_at=LATER, source_url=URL, content_hash=HASH_NEW
    )
    first = apply(store, pairs_for(store, [(prior, item)]))
    assert first.actions[0].applied == APPLIED_SUPERSEDED
    second = apply(store, pairs_for(store, [(prior, item)]))
    assert second.actions[0].applied == APPLIED_SKIPPED
    assert "幂等" in second.actions[0].reason
    assert [n.id for n in store.list_notes(status=None)] == ["N-0001", "N-0002"]


# ---- 9. 边界与防御纵深 -------------------------------------------------------


def test_guard_blocks_override_actions_even_if_verdict_says_newer(tmp_path: Path) -> None:
    """判官越权拦截的接线侧复核：guard 非 None 时拒绝 supersede/merge（理论上不可达）。"""
    store = make_store(tmp_path)
    make_prior(store, BODY_NEWER)
    item = EvidenceItem(
        text=EVIDENCE_NEWER, observed_at=LATER, source_url=URL, content_hash=HASH_NEW
    )
    comparison = EvidenceComparison(
        verdict=VERDICT_NEWER,
        reasons=["人为构造：护栏标记 + 覆盖动作"],
        prior_note_id="N-0001",
        evidence_index=0,
        suggested_action=ACTION_SUPERSEDE,
        guard=GUARD_ENTITIES_DISJOINT,
    )
    report = apply(store, [(comparison, item)])
    action = report.actions[0]
    assert action.applied == APPLIED_SKIPPED and action.guard == GUARD_ENTITIES_DISJOINT
    assert "拒绝执行 supersede" in action.reason
    assert [n.id for n in store.list_notes(status=None)] == ["N-0001"]

    merged = EvidenceComparison(
        verdict=VERDICT_MORE_SPECIFIC,
        reasons=["人为构造"],
        prior_note_id="N-0001",
        evidence_index=0,
        suggested_action=ACTION_MERGE,
        guard=GUARD_ENTITIES_DISJOINT,
    )
    assert apply(store, [(merged, item)]).actions[0].applied == APPLIED_SKIPPED


def test_duplicate_evidence_in_one_batch_is_deduped(tmp_path: Path) -> None:
    """同一 prior 的同一段证据在一批里出现两次：第二条按批内重复跳过。"""
    store = make_store(tmp_path)
    prior = make_prior(store, BODY_CONSISTENT)
    item = EvidenceItem(text=BODY_CONSISTENT, observed_at=None)
    comparison = pairs_for(store, [(prior, item)])[0][0]
    report = apply(store, [(comparison, item), (comparison, item)])
    assert [a.applied for a in report.actions] == [APPLIED_REVIEWED_AT, APPLIED_SKIPPED]
    assert "批内" in report.actions[1].reason or "同一批证据内重复" in report.actions[1].reason


def test_missing_prior_and_bad_evidence_index_are_skipped(tmp_path: Path) -> None:
    """旧记忆不存在 / evidence_index 越界：只记 skipped，不抛异常。"""
    store = make_store(tmp_path)
    item = EvidenceItem(text=BODY_CONSISTENT, observed_at=None)
    missing = EvidenceComparison(
        verdict=VERDICT_CONSISTENT,
        reasons=["旧记忆不存在"],
        prior_note_id="N-9999",
        evidence_index=0,
        suggested_action="refresh_reviewed_at",
    )
    out_of_range = EvidenceComparison(
        verdict=VERDICT_CONSISTENT,
        reasons=["下标越界"],
        prior_note_id="N-0001",
        evidence_index=7,
        suggested_action="refresh_reviewed_at",
    )
    make_prior(store, BODY_CONSISTENT)
    report = apply(store, [(missing, item), (out_of_range, item)])
    assert [a.applied for a in report.actions] == [APPLIED_SKIPPED, APPLIED_SKIPPED]
    assert "不存在" in report.actions[0].reason
    assert "越界" in report.actions[1].reason


def test_empty_batch_reports_zero_counts(tmp_path: Path) -> None:
    """无命中 Prior（或本轮无候选）时报告零动作、计数键齐全。"""
    store = make_store(tmp_path)
    report = apply(store, [])
    assert report.actions == []
    assert report.counts == {key: 0 for key in COUNT_KEYS}
    assert report.to_dict()["counts"] == {key: 0 for key in COUNT_KEYS}


def test_report_to_dict_and_state_line_shape() -> None:
    """报告序列化形态：counts 五键齐全、state_line 含简报要求的四个词位。"""
    report = MemoryUpdateReport(
        counts={"reviewed": 2, "superseded": 1, "merged": 1, "conflicts": 3, "skipped": 4},
        dry_run=False,
    )
    assert report.state_line() == "- 记忆更新：复核 2，替代 1，合并 1，冲突 3，跳过 4"
    payload = json.loads(json.dumps(report.to_dict(), ensure_ascii=False))
    assert payload == {
        "dry_run": False,
        "counts": {"reviewed": 2, "superseded": 1, "merged": 1, "conflicts": 3, "skipped": 4},
        "actions": [],
    }


# ---- 10. AgentLoop 集成（脚本化 provider）------------------------------------

QUESTION = "agent 记忆方案对比"
PLAN_TEXT = "- 任务一：检索主流方案\n- 任务二：蒸馏笔记\n"
REPORT_TEXT = "## 研究报告\n\n共 5 个方案支持长期记忆[1]。\n"
CANDIDATE_TEXT = "agent 记忆方案对比：共 5 个方案支持长期记忆。"
PRIOR_BODY = "agent 记忆方案对比：共 3 个方案支持长期记忆。"
DISTILL_JSON = json.dumps(
    {"notes": [{"text": CANDIDATE_TEXT, "entities": [], "confidence": "high"}], "conflicts": []},
    ensure_ascii=False,
)
METRICS_FIELDS = {
    "trace_id",
    "prior_hit_count",
    "prior_note_ids",
    "prior_context_chars",
    "fresh_search_count",
    "fresh_fetch_count",
    "source_count",
    "notes_created",
    "notes_merged",
    "notes_superseded",
    "input_tokens",
    "output_tokens",
    "latency_ms",
    "citation_coverage",
}


def text_turn(text: str) -> list[StreamEvent]:
    return [StreamEvent(type="text_delta", delta=text)]


def tool_turn(calls: list[dict]) -> list[StreamEvent]:
    return [
        StreamEvent(type="tool_calls", tool_calls=calls),
        StreamEvent(type="usage", usage=TokenUsage(input_tokens=900, output_tokens=40)),
    ]


def call(name: str, arguments: dict, *, id: str = "c1") -> dict:
    return {
        "id": id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
    }


class FakeRouter:
    def __init__(self, strong: ScriptedProvider) -> None:
        self._providers = {"strong": strong}

    def get(self, tier: str) -> ScriptedProvider:
        if tier not in self._providers:
            raise AssertionError(f"测试未配置 {tier} 档 Provider，却被请求了")
        return self._providers[tier]


def default_turns() -> list[list[StreamEvent]]:
    return [
        text_turn(PLAN_TEXT),
        tool_turn([call("web_search", {"query": "agent 记忆"}, id="c1")]),
        text_turn("已检索到主流方案，研究完成。"),
        text_turn(DISTILL_JSON),
        text_turn(REPORT_TEXT),
    ]


def preload_prior(tmp_path: Path) -> None:
    store = WikiStore(tmp_path / "wiki-data")
    store.save_note(
        PRIOR_BODY,
        note_id="N-0001",
        title="记忆方案（历史）",
        observed_at=EARLIER,
        created=EARLIER,
    )


def make_loop(
    tmp_path: Path,
    *,
    turns: list[list[StreamEvent]] | None = None,
    run_dir: Path | None = None,
    **kwargs,
) -> AgentLoop:
    return AgentLoop(
        QUESTION,
        router=FakeRouter(ScriptedProvider(turns or default_turns(), model="mock-strong")),
        llm_config={"strong": {"base_url": "https://mock"}, "cheap": {"base_url": ""}},
        accountant=TokenAccountant(tmp_path / "tokens.jsonl"),
        search_provider=MockSearch(),
        wiki_root=tmp_path / "wiki-data",
        run_dir=run_dir or tmp_path / "run",
        embedding=MockEmbeddingProvider(dim=512),
        wiki_config={"fts_tokenizer": "trigram"},
        **kwargs,
    )


def read_state(tmp_path: Path) -> str:
    return (tmp_path / "run" / "state.md").read_text(encoding="utf-8")


def memory_update_actions(tmp_path: Path) -> list[dict]:
    payload = json.loads((tmp_path / "run" / "memory-update.json").read_text(encoding="utf-8"))
    assert set(payload["counts"]) == set(COUNT_KEYS)
    return payload["actions"]


def test_loop_runs_memory_update_phase_and_writes_state_line(tmp_path: Path) -> None:
    """启用 [memory_update]：新证据（本轮入库候选）与命中 Prior 冲突 → 开台账 + 留痕。"""
    preload_prior(tmp_path)
    loop = make_loop(tmp_path, memory_update_config={"enabled": True})
    events = list(loop.events())
    assert events[-1]["type"] == "finish"

    state = read_state(tmp_path)
    assert "- 记忆更新：复核 0，替代 0，合并 0，冲突 1，跳过 0" in state
    assert "记忆形成" in state  # 既有行不受影响

    actions = memory_update_actions(tmp_path)
    assert len(actions) == 1
    assert actions[0]["applied"] == APPLIED_CONFLICT_OPENED
    assert actions[0]["verdict"] == VERDICT_CONFLICTING
    assert actions[0]["prior_note_id"] == "N-0001"

    store = WikiStore(tmp_path / "wiki-data")
    conflicts = store.list_conflicts()
    assert len(conflicts) == 1
    assert conflicts[0].claim_a["note_id"] == "N-0001"
    assert conflicts[0].claim_b["text"] == CANDIDATE_TEXT
    # 缺失 Prior 记忆没有被覆盖，仍是 active
    prior = store.get_note("N-0001")
    assert prior is not None and prior.status == "active" and prior.body == PRIOR_BODY
    # run-metrics 的 14 字段契约不被触碰
    metrics = json.loads((tmp_path / "run" / "run-metrics.json").read_text(encoding="utf-8"))
    assert set(metrics) == METRICS_FIELDS
    assert metrics["prior_hit_count"] == 1 and metrics["prior_note_ids"] == ["N-0001"]


def test_loop_without_memory_update_config_changes_nothing(tmp_path: Path) -> None:
    """memory_update_config=None：不比较、不写盘，state.md 与 run 目录零变化（回归）。"""
    preload_prior(tmp_path)
    loop = make_loop(tmp_path)  # 不传 memory_update_config
    events = list(loop.events())
    assert events[-1]["type"] == "finish"

    assert loop.memory_update_settings is None
    assert loop.memory_update_report is None
    assert "记忆更新" not in read_state(tmp_path)
    assert not (tmp_path / "run" / "memory-update.json").exists()
    store = WikiStore(tmp_path / "wiki-data")
    assert store.list_conflicts() == []
    # 蒸馏照旧入库：新增一条候选笔记（N-0002），Prior 未被任何记忆更新动作触碰
    assert sorted(n.id for n in store.list_notes()) == ["N-0001", "N-0002"]
    metrics = json.loads((tmp_path / "run" / "run-metrics.json").read_text(encoding="utf-8"))
    assert set(metrics) == METRICS_FIELDS and metrics["notes_created"] == 1


def test_loop_memory_update_disabled_is_recorded_but_does_no_action(tmp_path: Path) -> None:
    """[memory_update].enabled=false 逃生阀：留痕写明已禁用、零写盘动作。"""
    preload_prior(tmp_path)
    loop = make_loop(tmp_path, memory_update_config={"enabled": False})
    list(loop.events())

    assert "- 记忆更新：已禁用（[memory_update].enabled=false）" in read_state(tmp_path)
    payload = json.loads((tmp_path / "run" / "memory-update.json").read_text(encoding="utf-8"))
    assert payload["enabled"] is False and payload["actions"] == []
    assert payload["counts"] == {key: 0 for key in COUNT_KEYS}
    assert WikiStore(tmp_path / "wiki-data").list_conflicts() == []


def test_loop_memory_update_dry_run_shadow_mode(tmp_path: Path) -> None:
    """dry_run=true：判定与分派照跑（报告有冲突动作），但零写盘。"""
    preload_prior(tmp_path)
    loop = make_loop(tmp_path, memory_update_config={"enabled": True, "dry_run": True})
    list(loop.events())

    assert "- 记忆更新：复核 0，替代 0，合并 0，冲突 1，跳过 0（dry-run，未落盘）" in read_state(
        tmp_path
    )
    payload = json.loads((tmp_path / "run" / "memory-update.json").read_text(encoding="utf-8"))
    assert payload["dry_run"] is True and payload["counts"]["conflicts"] == 1
    assert payload["actions"][0]["applied"] == APPLIED_CONFLICT_OPENED
    assert WikiStore(tmp_path / "wiki-data").list_conflicts() == []


def test_loop_memory_update_needs_prior_hits_and_evidence(tmp_path: Path) -> None:
    """空 Wiki（无命中 Prior）：阶段照跑、报告零动作、state 行计数为 0。"""
    loop = make_loop(tmp_path, memory_update_config={"enabled": True})
    list(loop.events())

    assert "- 记忆更新：复核 0，替代 0，合并 0，冲突 0，跳过 0" in read_state(tmp_path)
    payload = json.loads((tmp_path / "run" / "memory-update.json").read_text(encoding="utf-8"))
    assert payload["prior_hit_count"] == 0 and payload["evidence_count"] == 1
    assert payload["actions"] == []


def test_loop_memory_update_refreshes_reviewed_at_on_consistency(tmp_path: Path) -> None:
    """一致场景全链路：本轮候选与命中 Prior 断言**逐字一致** → 只刷新 reviewed_at。

    修复轮 1/5 I-2：原版只是"蒸馏候选与 Prior 正文不同"的泛化断言，且末尾那条
    ``counts 之和 == len(actions)`` 恒真（``_counts`` 对每条动作恰好计一次）——
    等于没有覆盖。这里让蒸馏候选与 Prior 正文完全相同（相似度 1.0 → 确定性判
    ``consistent``），逐字段断言真实动作与真实落盘。
    """
    consistent_text = f"{QUESTION}：共 3 个方案支持长期记忆。"
    store = WikiStore(tmp_path / "wiki-data")
    store.save_note(
        consistent_text,
        note_id="N-0001",
        title="记忆方案（历史）",
        observed_at=EARLIER,
        created=EARLIER,
        reviewed_at=None,
    )
    distill = json.dumps(
        {
            "notes": [{"text": consistent_text, "entities": [], "confidence": "high"}],
            "conflicts": [],
        },
        ensure_ascii=False,
    )
    loop = make_loop(
        tmp_path,
        memory_update_config={"enabled": True},
        turns=[
            text_turn(PLAN_TEXT),
            tool_turn([call("web_search", {"query": "agent 记忆"}, id="c1")]),
            text_turn("已检索到主流方案，研究完成。"),
            text_turn(distill),
            text_turn(REPORT_TEXT),
        ],
    )
    list(loop.events())

    payload = json.loads((tmp_path / "run" / "memory-update.json").read_text(encoding="utf-8"))
    assert payload["prior_hit_count"] == 1 and payload["prior_note_ids"] == ["N-0001"]
    assert payload["evidence_count"] == 1
    actions = payload["actions"]
    assert len(actions) == 1
    assert actions[0]["verdict"] == VERDICT_CONSISTENT
    assert actions[0]["applied"] == APPLIED_REVIEWED_AT
    assert actions[0]["note_id"] == "N-0001"
    assert actions[0]["prior_note_id"] == "N-0001"
    assert payload["counts"]["reviewed"] == 1
    assert "- 记忆更新：复核 1，替代 0，合并 0，冲突 0，跳过 0" in read_state(tmp_path)

    after = store.get_note("N-0001")
    assert after is not None
    assert after.meta.reviewed_at == payload["now"]  # 真的刷到了 run 的 now
    assert after.body == consistent_text and after.status == "active"
    assert after.meta.sources == []  # 不动 sources
    assert after.meta.created == EARLIER and after.meta.title == "记忆方案（历史）"
    assert recorded_keys(after) == [update_key("N-0001", consistent_text)]
    assert store.list_conflicts() == []  # 一致不开台账


@pytest.mark.parametrize("verdict", [VERDICT_CONSISTENT, VERDICT_UNCERTAIN])
def test_unknown_actions_never_reach_disk(verdict: str, tmp_path: Path) -> None:
    """动作分派只认五种建议动作；未知动作只记 skipped（不做任何猜测性写入）。"""
    store = make_store(tmp_path)
    make_prior(store, BODY_CONSISTENT)
    item = EvidenceItem(text=BODY_CONSISTENT, observed_at=None)
    comparison = EvidenceComparison(
        verdict=verdict,
        reasons=["人为构造的未知动作"],
        prior_note_id="N-0001",
        evidence_index=0,
        suggested_action="unknown_action",
    )
    report = apply(store, [(comparison, item)])
    assert report.actions[0].applied == APPLIED_SKIPPED
    assert "未知建议动作" in report.actions[0].reason
    assert store.get_note("N-0001").meta.reviewed_at is None


# ---- 11. AgentLoop 集成：来源换版 → supersede（写入路径 + 证据链）-------------

SAME_URL = "https://example.com/spec"
SUPERSEDE_PRIOR = "agent 记忆方案对比：模型 A 的上下文窗口是 128k。"
SUPERSEDE_CANDIDATE = "agent 记忆方案对比：模型 A 的上下文窗口是 128k，另有 256k 可选。"
SUPERSEDE_DISTILL = json.dumps(
    {
        "notes": [
            {
                "text": SUPERSEDE_CANDIDATE,
                "entities": [],
                "confidence": "high",
                "source_urls": [SAME_URL],
            }
        ],
        "conflicts": [],
    },
    ensure_ascii=False,
)


def test_loop_supersede_on_source_change_keeps_chain(tmp_path: Path) -> None:
    """来源换版（来源池里的快照 hash ≠ 旧记忆记录）→ supersede：旧 ID 沿链可达。

    来源池用 ``ctx.add_source(url, title, content_hash=...)`` 直接预置（等价于本轮
    fetch_url 抓到了这一页）：不把测试绑死在 Windows 的临时目录深度上（快照落盘
    路径 = tmp + sources/{sha1(url)}/{hash}/content.md，深路径会撞 MAX_PATH），
    fetch 自身的哈希/落盘由 tests/test_loop_prior.py 与 tests/test_tools.py 覆盖。
    """
    store = WikiStore(tmp_path / "wiki-data")
    store.save_note(
        SUPERSEDE_PRIOR,
        note_id="N-0001",
        title="模型 A 规格（历史）",
        observed_at=EARLIER,
        created=EARLIER,
        sources=[SourceRef(url=SAME_URL, content_hash="a" * 64)],
        source_changed_at="2026-08-01T00:00:00+00:00",
    )
    loop = make_loop(
        tmp_path,
        memory_update_config={"enabled": True},
        turns=[
            text_turn(PLAN_TEXT),
            text_turn("已抓取规格页，研究完成。"),
            text_turn(SUPERSEDE_DISTILL),
            text_turn(REPORT_TEXT),
        ],
    )
    loop.ctx.add_source(SAME_URL, "模型 A 规格", content_hash=HASH_NEW)  # = 本轮 fresh 快照
    list(loop.events())

    assert "- 记忆更新：复核 0，替代 1，合并 0，冲突 0，跳过 0" in read_state(tmp_path)
    actions = memory_update_actions(tmp_path)
    assert len(actions) == 1
    assert actions[0]["applied"] == APPLIED_SUPERSEDED
    assert actions[0]["verdict"] == VERDICT_NEWER
    assert actions[0]["prior_note_id"] == "N-0001"
    new_id = actions[0]["note_id"]
    assert new_id and new_id != "N-0001"
    assert SAME_URL in actions[0]["reason"]  # 审计留痕带证据来源 URL

    final = store.follow_redirect("N-0001")
    assert final is not None and final.id == new_id
    assert final.body == SUPERSEDE_CANDIDATE
    # 新记忆带的是**本轮快照**的哈希（不是旧记录里那个），证据链指向真实来源
    assert final.meta.sources == [SourceRef(url=SAME_URL, content_hash=HASH_NEW)]
    assert final.meta.sources[0].content_hash != "a" * 64
    old = store.get_note("N-0001")
    assert old is not None and old.status == "superseded" and old.meta.superseded_by == new_id
    assert old.meta.extra[SUPERSEDE_REASON_KEY] == actions[0]["reason"]
    assert old.meta.extra[MEMORY_UPDATE_REASONS][-1]["verdict"] == VERDICT_NEWER



# ---- 修复轮 1/5：I-1 跨 run 幂等（正文哈希键）--------------------------------


def test_cross_run_supersede_is_idempotent_by_body_key(tmp_path: Path) -> None:
    """I-1：supersede 后重跑（下一轮 Prior 只回 active 的替代版本）必须仍被判幂等。

    真实失效链：第一轮把 N-0001 退役、替代版本 N-0002 上记的键是 ``N-0001:{hash}``；
    第二轮 Prior 检索只回 **active** 的 N-0002（旧笔记根本不会被读到），按 N-0002
    算出的键是 ``N-0002:{hash}`` —— 与记录值不等，于是同一段证据每轮都重新
    supersede 一遍。正文哈希键（不带 prior 前缀）就是为这条链准备的。
    """
    store = make_store(tmp_path)
    prior = make_prior(
        store,
        BODY_NEWER,
        sources=[SourceRef(url=URL, content_hash=HASH_OLD)],
        source_changed_at="2026-08-01T00:00:00+00:00",
    )
    item = EvidenceItem(
        text=EVIDENCE_NEWER, observed_at=LATER, source_url=URL, content_hash=HASH_NEW
    )
    first = apply(store, pairs_for(store, [(prior, item)]))
    assert first.actions[0].applied == APPLIED_SUPERSEDED
    tail_id = first.actions[0].note_id
    assert tail_id is not None
    # 替代版本上同时记着带前缀键与正文哈希键（后者跨 run 有效）
    tail = store.get_note(tail_id)
    assert tail is not None
    assert recorded_keys(tail) == [update_key("N-0001", EVIDENCE_NEWER)]
    assert recorded_body_keys(tail) == [body_key(EVIDENCE_NEWER)]

    # 第二轮：命中的 Prior 就是链尾（active），证据一字不变
    before_notes = sorted(n.id for n in store.list_notes(status=None))
    second = apply(store, pairs_for(store, [(tail, item)]), now=LATER)
    action = second.actions[0]
    assert action.applied == APPLIED_SKIPPED
    assert "幂等" in action.reason
    assert sorted(n.id for n in store.list_notes(status=None)) == before_notes  # 不新建笔记
    assert store.get_note(tail_id).status == "active"  # 替代版本没被自己退役


def test_conflict_idempotency_also_survives_prior_identity_change(tmp_path: Path) -> None:
    """I-1 的台账侧：同一段证据在**后续一次处置**里换一条 prior 命中也不重复开台账。

    语义边界（刻意保留）：**同一批**里两条 prior 各自命中同一段证据时，仍各开一条
    台账——判定与处置是按 prior 独立的（``update_key`` 的前缀就是为此存在），两条
    冲突记录的是两条不同旧记忆与同一段证据的矛盾，合并成一条会丢掉一侧的证据。
    正文哈希键拦的是**跨处置**的重放：换了一条 prior、换了一个 run，台账里已有
    "这段证据处置过"的记录，就不该再多开一条。
    """
    store = make_store(tmp_path)
    prior_a = make_prior(store, BODY_CONFLICT, note_id="N-0001")
    prior_b = make_prior(store, BODY_CONFLICT, note_id="N-0002")
    item = EvidenceItem(text=EVIDENCE_CONFLICT, observed_at=None)
    first = apply(store, pairs_for(store, [(prior_a, item), (prior_b, item)]))
    assert [a.applied for a in first.actions] == [
        APPLIED_CONFLICT_OPENED,
        APPLIED_CONFLICT_OPENED,
    ]
    assert len(store.list_conflicts()) == 2
    assert conflict_has_body_key(store.list_conflicts()[0], body_key(EVIDENCE_CONFLICT))

    # 后续一次处置：换一条此前没参与过的 prior（N-0003）命中同一段证据
    prior_c = make_prior(store, BODY_CONFLICT, note_id="N-0003")
    second = apply(store, pairs_for(store, [(prior_c, item)]), now=LATER)
    action = second.actions[0]
    assert action.applied == APPLIED_SKIPPED
    assert "幂等" in action.reason
    assert len(store.list_conflicts()) == 2  # 没有新台账


def test_loop_cross_run_supersede_is_idempotent(tmp_path: Path) -> None:
    """I-1 的 run 级复现：两次 run 走同一条来源换版路径，第二次不再产生替代动作。"""
    store = WikiStore(tmp_path / "wiki-data")
    store.save_note(
        SUPERSEDE_PRIOR,
        note_id="N-0001",
        title="模型 A 规格（历史）",
        observed_at=EARLIER,
        created=EARLIER,
        sources=[SourceRef(url=SAME_URL, content_hash="a" * 64)],
        source_changed_at="2026-08-01T00:00:00+00:00",
    )
    turns = [
        text_turn(PLAN_TEXT),
        text_turn("已抓取规格页，研究完成。"),
        text_turn(SUPERSEDE_DISTILL),
        text_turn(REPORT_TEXT),
    ]
    first_loop = make_loop(tmp_path, memory_update_config={"enabled": True}, turns=turns)
    first_loop.ctx.add_source(SAME_URL, "模型 A 规格", content_hash=HASH_NEW)
    list(first_loop.events())
    assert first_loop.memory_update_report is not None
    assert first_loop.memory_update_report.counts["superseded"] == 1
    active_after_first = sorted(n.id for n in store.list_notes())

    second_loop = make_loop(
        tmp_path,
        memory_update_config={"enabled": True},
        turns=turns,
        run_dir=tmp_path / "run2",
    )
    second_loop.ctx.add_source(SAME_URL, "模型 A 规格", content_hash=HASH_NEW)
    list(second_loop.events())
    assert second_loop.memory_update_report is not None
    # 第二轮：证据（正文 + 来源哈希）都没变 → 不再 supersede、不新建记忆
    assert second_loop.memory_update_report.counts["superseded"] == 0
    payload = json.loads((tmp_path / "run2" / "memory-update.json").read_text(encoding="utf-8"))
    assert payload["counts"]["superseded"] == 0
    chain_tail = store.follow_redirect("N-0001")
    assert chain_tail is not None and chain_tail.status == "active"
    assert chain_tail.id in active_after_first


# ---- 修复轮 1/5：I-3 confidence 走 formation 口径 ----------------------------


def test_supersede_confidence_follows_formation_rule(tmp_path: Path) -> None:
    """I-3：有来源 + 具体事实要素 → high；有来源但无具体要素 → medium。"""
    store = make_store(tmp_path)
    prior = make_prior(
        store,
        "记忆方案对比：Letta 的方案最成熟。",
        sources=[SourceRef(url=URL, content_hash=HASH_OLD)],
        source_changed_at="2026-08-01T00:00:00+00:00",
    )
    specific = EvidenceItem(
        text="记忆方案对比：Letta 的方案最成熟，实测延迟 120ms。",
        observed_at=LATER,
        source_url=URL,
        content_hash=HASH_NEW,
    )
    _, _, action = one(store, prior, specific)
    assert action.applied == APPLIED_SUPERSEDED
    replacement = store.get_note(action.note_id)
    assert replacement is not None and replacement.meta.confidence == "high"
    assert "confidence=high" in action.reason

    # 另一条 prior + 无数字/拉丁字母/引号的证据 → 有来源但无具体要素 → medium
    plain_prior = make_prior(
        store,
        "记忆方案对比：后台整理是当前主流做法。",
        note_id="N-0009",
        sources=[SourceRef(url=URL_B, content_hash=HASH_OLD)],
        source_changed_at="2026-08-01T00:00:00+00:00",
    )
    plain = EvidenceItem(
        text="记忆方案对比：后台整理是当前主流做法，另有离线整理。",
        observed_at=LATER,
        source_url=URL_B,
        content_hash=HASH_NEW,
    )
    _, _, plain_action = one(store, plain_prior, plain)
    assert plain_action.applied == APPLIED_SUPERSEDED
    plain_replacement = store.get_note(plain_action.note_id)
    assert plain_replacement is not None and plain_replacement.meta.confidence == "medium"


# ---- 修复轮 1/5：I-4 supersede 复用本轮入库笔记 ------------------------------


def test_supersede_reuses_ingested_note_instead_of_creating_duplicate(tmp_path: Path) -> None:
    """I-4：给了 evidence_notes 映射时复用本轮入库笔记，不再另建替代笔记。"""
    store = make_store(tmp_path)
    prior = make_prior(
        store,
        BODY_NEWER,
        sources=[SourceRef(url=URL, content_hash=HASH_OLD)],
        source_changed_at="2026-08-01T00:00:00+00:00",
    )
    ingested = store.save_note(EVIDENCE_NEWER, note_id="N-0002", title="本轮入库")
    item = EvidenceItem(
        text=EVIDENCE_NEWER, observed_at=LATER, source_url=URL, content_hash=HASH_NEW
    )
    comparison = pairs_for(store, [(prior, item)])[0][0]
    report = apply_comparisons(
        store,
        [comparison],
        evidence=[item],
        now=NOW,
        evidence_notes={0: ingested.id},
    )
    action = report.actions[0]
    assert action.applied == APPLIED_SUPERSEDED
    assert action.note_id == "N-0002"
    assert "复用本轮入库笔记 N-0002" in action.reason
    # 没有新增笔记（同源记忆只有一条 active）
    assert sorted(n.id for n in store.list_notes()) == ["N-0002"]
    assert store.follow_redirect("N-0001").id == "N-0002"
    old = store.get_note("N-0001")
    assert old is not None and old.status == "superseded" and old.meta.superseded_by == "N-0002"
    # 替代版本（入库笔记）拿到留痕 + 证据来源（幂等键与理由都落在它身上）
    target = store.get_note("N-0002")
    assert target is not None
    assert recorded_body_keys(target) == [body_key(EVIDENCE_NEWER)]
    assert target.meta.extra[SUPERSEDE_REASON_KEY] == action.reason
    assert target.meta.extra["superseded_from"] == "N-0001"
    assert SourceRef(url=URL, content_hash=HASH_NEW) in target.meta.sources
    assert target.meta.title == "本轮入库"  # 入库笔记的身份不被改写


def test_supersede_skips_when_ingested_note_is_the_prior_itself(tmp_path: Path) -> None:
    """I-4 边界：候选被 Ingestor 合并进旧记忆本身 → 事实已并入，不做 supersede。"""
    store = make_store(tmp_path)
    prior = make_prior(
        store,
        BODY_NEWER,
        sources=[SourceRef(url=URL, content_hash=HASH_OLD)],
        source_changed_at="2026-08-01T00:00:00+00:00",
    )
    item = EvidenceItem(
        text=EVIDENCE_NEWER, observed_at=LATER, source_url=URL, content_hash=HASH_NEW
    )
    comparison = pairs_for(store, [(prior, item)])[0][0]
    report = apply_comparisons(
        store, [comparison], evidence=[item], now=NOW, evidence_notes={0: "N-0001"}
    )
    action = report.actions[0]
    assert action.applied == APPLIED_SKIPPED
    assert "合并进该记忆本身" in action.reason
    after = store.get_note("N-0001")
    assert after is not None and after.status == "active"  # 没被自己退役
    assert after.meta.superseded_by is None


def test_evidence_notes_target_must_be_active_or_falls_back_to_new_note(tmp_path: Path) -> None:
    """I-4 边界：映射目标不是 active（异常状态）→ 退回新建替代笔记，不挂错链。"""
    store = make_store(tmp_path)
    prior = make_prior(
        store,
        BODY_NEWER,
        sources=[SourceRef(url=URL, content_hash=HASH_OLD)],
        source_changed_at="2026-08-01T00:00:00+00:00",
    )
    store.save_note(EVIDENCE_NEWER, note_id="N-0002", title="已退役", status="superseded")
    item = EvidenceItem(
        text=EVIDENCE_NEWER, observed_at=LATER, source_url=URL, content_hash=HASH_NEW
    )
    comparison = pairs_for(store, [(prior, item)])[0][0]
    report = apply_comparisons(
        store, [comparison], evidence=[item], now=NOW, evidence_notes={0: "N-0002"}
    )
    action = report.actions[0]
    assert action.applied == APPLIED_SUPERSEDED
    assert action.note_id != "N-0002"  # 新建，而不是挂到退役版本上
    assert "新建替代笔记" in action.reason
    assert store.follow_redirect("N-0001").id == action.note_id


# ---- 修复轮 1/5：M-1 护栏 allowlist 镜像 / M-2 无变更记 skipped --------------


def test_guard_also_blocks_refresh_reviewed_at(tmp_path: Path) -> None:
    """M-1：护栏触发时 consistent 也不放行（假一致正是 verification 特意拒的一类）。"""
    store = make_store(tmp_path)
    make_prior(store, BODY_CONSISTENT)
    item = EvidenceItem(text=BODY_CONSISTENT, observed_at=None)
    comparison = EvidenceComparison(
        verdict=VERDICT_CONSISTENT,
        reasons=["人为构造：护栏标记 + 一致判定"],
        prior_note_id="N-0001",
        evidence_index=0,
        suggested_action="refresh_reviewed_at",
        guard=GUARD_ENTITIES_DISJOINT,
    )
    report = apply(store, [(comparison, item)])
    action = report.actions[0]
    assert action.applied == APPLIED_SKIPPED
    assert "拒绝执行 refresh_reviewed_at" in action.reason
    assert "主体同一性未确认时不得覆盖旧记忆、也不得宣称旧断言仍然成立" in action.reason
    assert store.get_note("N-0001").meta.reviewed_at is None  # 没写盘
    # 护栏下 open_conflict 仍放行（与 verification 的 CONSERVATIVE_VERDICTS 镜像）
    conflict = EvidenceComparison(
        verdict=VERDICT_CONFLICTING,
        reasons=["人为构造"],
        prior_note_id="N-0001",
        evidence_index=0,
        suggested_action="open_conflict",
        guard=GUARD_ENTITIES_DISJOINT,
    )
    assert apply(store, [(conflict, item)]).actions[0].applied == APPLIED_CONFLICT_OPENED


def merge_comparison() -> EvidenceComparison:
    """构造一条 merge 判定（执行层的"无实际变更"分支用它驱动）。

    为什么手搓：正文已逐字包含证据时，"证据事实令牌数 > 旧记忆"这条具体度判据
    （P2-C 规则 5）不再成立，所以确定性链路走不到 ``more_specific``——该分支在
    生产上由**判官注入**的 more_specific 触发（``compare_batch(judge=...)``），
    而 run 收尾刻意不注入 judge。这里构造判定只为覆盖执行层的审计不变量
    （M-2：报告与落盘必须一致），判定侧的真实路径由
    ``test_more_specific_merges_body_and_sources`` 用真实 compare 覆盖。
    """
    return EvidenceComparison(
        verdict=VERDICT_MORE_SPECIFIC,
        reasons=["人为构造：证据事实已被旧正文逐字包含（merge 无实际变更）"],
        prior_note_id="N-0001",
        evidence_index=0,
        suggested_action=ACTION_MERGE,
    )


def test_merge_without_actual_change_is_skipped(tmp_path: Path) -> None:
    """M-2：正文已含该证据、来源无新增 → 如实记 skipped（不谎报 merged）。"""
    store = make_store(tmp_path)
    body = f"{BODY_SPECIFIC}\n\n- 补充（{EARLIER}）：{EVIDENCE_SPECIFIC}"
    make_prior(store, body, reviewed_at=None)
    item = EvidenceItem(text=EVIDENCE_SPECIFIC, observed_at=None)
    note_path = tmp_path / "wiki-data" / "notes" / "N-0001.md"
    store_before = note_path.read_text(encoding="utf-8")
    report = apply(store, [(merge_comparison(), item)])
    action = report.actions[0]

    assert action.applied == APPLIED_SKIPPED
    assert "并入无实际变更" in action.reason
    assert report.counts["merged"] == 0 and report.counts["skipped"] == 1
    after = store.get_note("N-0001")
    assert after is not None and after.meta.reviewed_at is None  # 没写盘
    assert note_path.read_text(encoding="utf-8") == store_before
    assert UPDATE_REASONS_KEY not in after.meta.extra


def test_merge_records_change_when_only_sources_are_new(tmp_path: Path) -> None:
    """M-2 的另一半：正文无变化但来源有新增 → 照常落盘并记 merged（真实修订）。"""
    store = make_store(tmp_path)
    body = f"{BODY_SPECIFIC}\n\n- 补充（{EARLIER}）：{EVIDENCE_SPECIFIC}"
    make_prior(store, body, sources=[SourceRef(url=URL, content_hash=HASH_OLD)])
    item = EvidenceItem(
        text=EVIDENCE_SPECIFIC, observed_at=None, source_url=URL_B, content_hash=HASH_NEW
    )
    report = apply(store, [(merge_comparison(), item)])
    action = report.actions[0]
    assert action.applied == APPLIED_MERGED
    assert report.counts["merged"] == 1
    after = store.get_note("N-0001")
    assert after is not None
    assert after.body == body  # 正文没被重复追加
    assert after.meta.sources == [
        SourceRef(url=URL, content_hash=HASH_OLD),
        SourceRef(url=URL_B, content_hash=HASH_NEW),
    ]
    assert after.meta.extra[UPDATE_REASONS_KEY][-1]["reason"] == action.reason


# ---- 修复轮 1/5：M-4 键/留痕保留上限 ----------------------------------------


def test_recorded_keys_are_bounded(tmp_path: Path) -> None:
    """M-4：键与留痕列表只保留最近 MAX_RECORDED_KEYS 条（不随 run 次数无限膨胀）。"""
    store = make_store(tmp_path)
    make_prior(store, BODY_CONSISTENT)
    for index in range(MAX_RECORDED_KEYS + 5):
        item = EvidenceItem(text=f"{BODY_CONSISTENT}（第 {index} 次重述）", observed_at=None)
        comparison = EvidenceComparison(
            verdict=VERDICT_CONSISTENT,
            reasons=["人为构造：连续多次一致"],
            prior_note_id="N-0001",
            evidence_index=0,
            suggested_action="refresh_reviewed_at",
        )
        report = apply(store, [(comparison, item)], now=NOW)
        assert report.actions[0].applied == APPLIED_REVIEWED_AT
    after = store.get_note("N-0001")
    assert after is not None
    assert len(recorded_keys(after)) == MAX_RECORDED_KEYS
    assert len(recorded_body_keys(after)) == MAX_RECORDED_KEYS
    assert len(after.meta.extra[MEMORY_UPDATE_REASONS]) == MAX_RECORDED_KEYS
    # 保留的是**最近**的键：最后一次的键仍在，最老的那条已被裁掉
    newest = f"{BODY_CONSISTENT}（第 {MAX_RECORDED_KEYS + 4} 次重述）"
    assert update_key("N-0001", newest) in recorded_keys(after)
    assert update_key("N-0001", f"{BODY_CONSISTENT}（第 0 次重述）") not in recorded_keys(after)


# ---- 修复轮 1/5：M-6 supersede 要求来源 hash 非空 ----------------------------


def test_supersede_requires_content_hash_not_just_url(tmp_path: Path) -> None:
    """M-6：只有 URL、没有 content_hash 的来源视为证据链不完整 → 降级开台账。"""
    store = make_store(tmp_path)
    prior = make_prior(store, BODY_NEWER)
    item = EvidenceItem(text=EVIDENCE_NEWER, observed_at=LATER, source_url=URL, content_hash="")
    comparison, report, action = one(store, prior, item)

    assert comparison.verdict == VERDICT_NEWER
    assert action.applied == APPLIED_CONFLICT_OPENED
    assert "没有 content_hash" in action.reason
    assert "无法定位快照" in action.reason
    assert report.counts["conflicts"] == 1 and report.counts["superseded"] == 0
    assert [n.id for n in store.list_notes()] == ["N-0001"]  # 没造替代记忆
    conflict = store.list_conflicts()[0]
    assert conflict.claim_b["source_url"] == URL and conflict.claim_b["content_hash"] == ""
