"""后台维护选择器测试（P4a Task 16 / T2：consolidation 选择器 + consolidate --dry-run）。

覆盖 task-16 简报的验收清单（十一组）+ 逃生阀 / 墓碑 / 接线补充：

① merge 候选：相似度 ≥0.5 + 实体重叠的对，canonical = created 更早者（并列取更小 ID）；
② ~0.31 校准锚点带（[0.3 地板, 0.5 merge_floor)）→ 不自动 merge（真实 run 在 sim
   0.31 曾并入正文，0.5 新地板后该形态不再出现）；
③ review_due + 有来源 → refresh 候选（payload 带 url）；review_due + 无来源 → rejudge 改派；
④ confidence=low → rejudge（payload 记模型档位）；
⑤ 同实体 slot 级矛盾对 → conflict 候选（构造法参照 tests/test_verification.py；
   实体不相交的对不产 conflict——实体护栏方向被预筛尊重）；
⑥ max_* 截断生效（max_rejudge / max_merges / max_conflict_pairs / max_conflicts）；
⑦ dry-run 零变异：plan 与 CLI 前后目录逐字节对比（含 mtime）；
⑧ 同一 store 状态两次 plan 输出逐字节相同；
⑨ idempotency_key 同输入稳定、对 body 变化敏感、材料公式逐字节锁定；
⑩ [consolidation] 段缺失 → CLI 退出 1 提示未配置；
⑪ --json 形态可解析且含 settings_snapshot。

全部零网络、零模型调用；freshness 判定用固定时钟（now 注入）；相似度复用
wiki/verification.py 的确定性 bigram 特征哈希度量（同一把尺）。
"""

import hashlib
import json
import tomllib
from datetime import UTC, datetime
from pathlib import Path

from researchwiki import cli
from researchwiki.wiki.consolidation import (
    ACTION_CONFLICT,
    ACTION_MERGE,
    ACTION_REFRESH,
    ACTION_REJUDGE,
    REJUDGE_TIER,
    ConsolidationSettings,
    consolidation_settings,
    plan_consolidation,
)
from researchwiki.wiki.frontmatter import SourceRef
from researchwiki.wiki.store import WikiStore
from researchwiki.wiki.verification import token_similarity

NOW = datetime(2026, 6, 1, tzinfo=UTC)

# 测试文本对（verification 尺度实测）：merge 对 0.8447 ∈ [0.5, 0.95)；锚点对
# 0.3720 ∈ [0.3 地板, 0.5 merge_floor)——真实 run sim 0.31 曾并入正文，同属此带；
# 冲突对 0.8473，同槽位（number:k:上下文窗口是）取值不同。
MERGE_A = "GLM-5.3 支持工具调用。"
MERGE_B = "GLM-5.3 支持工具调用与函数并行。"
ANCHOR_A = "GLM-5.3 的上下文窗口是 128k，面向长文档任务。"
ANCHOR_B = "GLM-5.3 的语音克隆需要三秒音频样本。"
CONFLICT_A = "GLM-5.3 的上下文窗口是 128k。"
CONFLICT_B = "GLM-5.3 的上下文窗口是 64k。"

# [consolidation] 段（与 config.toml 逐字一致，测试用它验证 CLI 接线）
CONSOLIDATION_TOML = """\
[consolidation]
enabled = true
merge_min_similarity = 0.5
rejudge_stale_ratio = 0.25
max_conflict_pairs = 50
max_merges = 10
max_refresh = 10
max_rejudge = 10
max_conflicts = 10
"""


# ---- 脚手架 -----------------------------------------------------------------


def make_store(tmp_path: Path) -> WikiStore:
    return WikiStore(tmp_path / "wiki-data")


def save(
    store: WikiStore,
    note_id: str,
    body: str,
    *,
    entities: list[str] | None = None,
    kind: str = "knowledge",
    confidence: str = "medium",
    volatility: str = "stable",
    created: str = "2026-01-01T00:00:00+00:00",
    observed_at: str | None = None,
    sources: list[SourceRef] | None = None,
    valid_until: str | None = None,
    tombstone: bool = False,
):
    """按测试默认值写一条笔记（字段显式给出，不吃 save_note 的静默默认）。"""
    return store.save_note(
        body,
        note_id=note_id,
        title=f"t-{note_id}",
        entities=list(entities or []),
        kind=kind,
        confidence=confidence,
        volatility=volatility,
        created=created,
        observed_at=observed_at,
        sources=list(sources or []),
        valid_until=valid_until,
        tombstone=tombstone,
    )


def actions_of(plan, action: str) -> list:
    return [a for a in plan.actions if a.action == action]


def snapshot_tree(root: Path) -> dict[str, dict[str, object]]:
    """目录逐字节快照：相对路径 → mtime_ns（目录）/ mtime_ns + size + sha256（文件）。

    dry-run 零变异的判据：plan 前后两次快照完全相等（读文件不改变 mtime）。
    """
    out: dict[str, dict[str, object]] = {}
    for path in sorted(Path(root).rglob("*")):
        stat = path.stat()
        entry: dict[str, object] = {"mtime_ns": stat.st_mtime_ns}
        if path.is_file():
            entry["size"] = stat.st_size
            entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        out[path.relative_to(root).as_posix()] = entry
    return out


def rich_store(tmp_path: Path) -> WikiStore:
    """覆盖四类候选的混合库（族间实体不相交，互不串扰）：

    - merge 族（GLM-5.3）：N-0001/N-0002 相似度 0.8447 → merge；
    - conflict 族（DeepSeek-V3）：N-0003/N-0004 同槽位取值冲突 → conflict（且
      相似度 ≥0.5 同时产 merge——两类判定独立，都是保守动作）；
    - N-0005 volatile + 有来源 → refresh；N-0006 volatile 无来源 → rejudge 改派；
      N-0007 confidence=low → rejudge。
    """
    store = make_store(tmp_path)
    save(store, "N-0001", MERGE_A, entities=["GLM-5.3"], created="2026-03-01T00:00:00+00:00")
    save(store, "N-0002", MERGE_B, entities=["GLM-5.3"], created="2026-01-01T00:00:00+00:00")
    save(
        store,
        "N-0003",
        "DeepSeek-V3 的上下文窗口是 128k。",
        entities=["DeepSeek-V3"],
    )
    save(store, "N-0004", "DeepSeek-V3 的上下文窗口是 64k。", entities=["DeepSeek-V3"])
    save(
        store,
        "N-0005",
        "张三负责项目部署与发布。",
        entities=["张三"],
        volatility="volatile",
        observed_at="2026-04-17T00:00:00+00:00",  # 45 天 → decay 0.354 → review_due
        sources=[SourceRef(url="https://example.test/a", content_hash="hash-a")],
    )
    save(
        store,
        "N-0006",
        "李四负责测试排期。",
        entities=["李四"],
        volatility="volatile",
        observed_at="2026-04-17T00:00:00+00:00",
    )
    save(
        store,
        "N-0007",
        "王五管理发布流程。",
        entities=["王五"],
        confidence="low",
        observed_at="2026-05-30T00:00:00+00:00",
    )
    return store


# ---- ① merge 候选 -----------------------------------------------------------


def test_merge_candidate_canonical_is_earlier_created(tmp_path: Path) -> None:
    """①相似度 ≥0.5 + 实体重叠 → merge 候选；canonical = created 更早者（非 ID 序）。"""
    store = make_store(tmp_path)
    save(store, "N-0001", MERGE_A, entities=["GLM-5.3"], created="2026-03-01T00:00:00+00:00")
    save(store, "N-0002", MERGE_B, entities=["GLM-5.3"], created="2026-01-01T00:00:00+00:00")
    plan = plan_consolidation(store, ConsolidationSettings(), now=NOW)

    merges = actions_of(plan, ACTION_MERGE)
    assert len(merges) == 1
    action = merges[0]
    assert action.note_ids == ["N-0001", "N-0002"]
    # created 更早的 N-0002 是规范 ID——尽管它的字典序更大（锁定"按 created 不按 ID"）
    assert action.payload["canonical_id"] == "N-0002"
    assert action.payload["absorbed_id"] == "N-0001"
    assert action.payload["entities"] == ["glm-5-3"]
    assert 0.5 <= action.payload["similarity"] < 0.95
    assert "N-0002" in action.reason and "相似度" in action.reason
    # 两条 stable/fresh/medium 笔记不触发其他类候选
    assert len(plan.actions) == 1


def test_merge_canonical_ties_break_to_smaller_id(tmp_path: Path) -> None:
    """①补充：created 并列时取字典序更小 ID 为 canonical。"""
    store = make_store(tmp_path)
    save(store, "N-0001", MERGE_A, entities=["GLM-5.3"], created="2026-03-01T00:00:00+00:00")
    save(store, "N-0002", MERGE_B, entities=["GLM-5.3"], created="2026-03-01T00:00:00+00:00")
    plan = plan_consolidation(store, ConsolidationSettings(), now=NOW)
    assert actions_of(plan, ACTION_MERGE)[0].payload["canonical_id"] == "N-0001"


def test_merge_skips_below_floor_and_near_duplicate_band(tmp_path: Path) -> None:
    """②校准锚点带不 merge；≥0.95 近乎重复带也不 merge（归入库去重/formation 辖区）。"""
    store = make_store(tmp_path)
    save(store, "N-0001", ANCHOR_A, entities=["GLM-5.3"])
    save(store, "N-0002", ANCHOR_B, entities=["GLM-5.3"])
    save(store, "N-0003", "项目使用 uv 管理依赖。", entities=["uv"])
    save(store, "N-0004", "项目使用 uv 管理依赖。", entities=["uv"])
    # 锚点对实测相似度落在 [0.3 地板, 0.5 merge_floor)——校准锁：度量漂移时在此失败
    anchor_sim = token_similarity(ANCHOR_A, ANCHOR_B)
    assert 0.3 <= anchor_sim < 0.5
    plan = plan_consolidation(store, ConsolidationSettings(), now=NOW)
    assert actions_of(plan, ACTION_MERGE) == []
    # 锚点对无可比令牌差异、近乎重复对无冲突，两组都不开台账
    assert actions_of(plan, ACTION_CONFLICT) == []


def test_merge_only_applies_to_knowledge_pairs(tmp_path: Path) -> None:
    """①补充：merge/conflict 只看 knowledge 笔记——user/experience 相似对不产候选。"""
    store = make_store(tmp_path)
    save(store, "N-0001", MERGE_A, entities=["GLM-5.3"], kind="user")
    save(store, "N-0002", MERGE_B, entities=["GLM-5.3"], kind="user")
    plan = plan_consolidation(store, ConsolidationSettings(), now=NOW)
    assert plan.actions == []


# ---- ③ refresh / ④ rejudge --------------------------------------------------


def test_refresh_with_source_and_rejudge_reassignment(tmp_path: Path) -> None:
    """③review_due + 有来源 → refresh（payload 带 url）；无来源 → rejudge 改派。"""
    store = make_store(tmp_path)
    save(
        store,
        "N-0001",
        "张三负责项目部署与发布。",
        entities=["张三"],
        volatility="volatile",
        observed_at="2026-04-17T00:00:00+00:00",  # 45 天 → decay 0.354 → review_due
        sources=[SourceRef(url="https://example.test/a", content_hash="hash-a")],
    )
    save(
        store,
        "N-0002",
        "李四负责测试排期。",
        entities=["李四"],
        volatility="volatile",
        observed_at="2026-04-17T00:00:00+00:00",  # review_due 但无来源 → 改派 rejudge
    )
    save(
        store,
        "N-0003",
        "王五管理发布流程。",
        entities=["王五"],
        valid_until="2026-05-01T00:00:00+00:00",  # 显式过期 → stale（不看衰减）
        sources=[SourceRef(url="https://example.test/b", content_hash="hash-b")],
    )
    save(
        store,
        "N-0004",
        "赵六负责运维值班。",
        entities=["赵六"],
        observed_at="2026-05-30T00:00:00+00:00",  # fresh + 有来源 → 无候选
        sources=[SourceRef(url="https://example.test/c", content_hash="hash-c")],
    )
    plan = plan_consolidation(store, ConsolidationSettings(), now=NOW)

    refreshes = actions_of(plan, ACTION_REFRESH)
    assert [a.note_ids for a in refreshes] == [["N-0001"], ["N-0003"]]
    by_id = {a.note_ids[0]: a for a in refreshes}
    assert by_id["N-0001"].payload["url"] == "https://example.test/a"
    assert by_id["N-0001"].payload["urls"] == ["https://example.test/a"]
    assert by_id["N-0003"].payload["url"] == "https://example.test/b"
    assert "review_due" in by_id["N-0001"].reason and "decay" in by_id["N-0001"].reason
    assert "stale" in by_id["N-0003"].reason

    rejudges = actions_of(plan, ACTION_REJUDGE)
    assert [a.note_ids for a in rejudges] == [["N-0002"]]
    assert "无来源" in rejudges[0].reason
    assert len(plan.actions) == 3  # fresh 的 N-0004 不产任何候选


def test_low_confidence_goes_to_rejudge(tmp_path: Path) -> None:
    """④confidence=low → rejudge 候选（与 freshness 无关，fresh 也收）。"""
    store = make_store(tmp_path)
    fresh_observed = "2026-05-30T00:00:00+00:00"
    save(store, "N-0001", "低置信断言待复核。", confidence="low", observed_at=fresh_observed)
    save(store, "N-0002", "中等置信断言。", confidence="medium", observed_at=fresh_observed)
    plan = plan_consolidation(store, ConsolidationSettings(), now=NOW)

    rejudges = actions_of(plan, ACTION_REJUDGE)
    assert [a.note_ids for a in rejudges] == [["N-0001"]]
    assert "confidence=low" in rejudges[0].reason
    assert rejudges[0].payload == {"tier": REJUDGE_TIER}
    assert len(plan.actions) == 1


# ---- ⑤ conflict 候选 --------------------------------------------------------


def test_conflict_candidate_for_same_entity_slot_clash(tmp_path: Path) -> None:
    """⑤同实体 slot 级取值冲突（构造法同 test_verification）→ conflict 候选。

    该对相似度 0.8473 ∈ [0.5, 0.95)，同时产 merge 候选——两类判定独立，
    merge（内容保留）与 conflict（开台账）都是保守动作，互不抑制。
    """
    store = make_store(tmp_path)
    save(store, "N-0001", CONFLICT_A, entities=["GLM-5.3"])
    save(store, "N-0002", CONFLICT_B, entities=["GLM-5.3"])
    plan = plan_consolidation(store, ConsolidationSettings(), now=NOW)

    conflicts = actions_of(plan, ACTION_CONFLICT)
    assert len(conflicts) == 1
    action = conflicts[0]
    assert action.note_ids == ["N-0001", "N-0002"]
    assert action.payload["slots"] == [
        {
            "slot": "number:k:上下文窗口是",
            "kind": "number",
            "context": "上下文窗口是",
            "prior_value": "128",
            "evidence_value": "64",
        }
    ]
    assert "conflicting" in action.reason
    assert len(actions_of(plan, ACTION_MERGE)) == 1  # 同一对的 merge 候选并存


def test_conflict_respects_entity_guard(tmp_path: Path) -> None:
    """⑤补充：实体不相交的矛盾对不产 conflict（同实体预筛 = 尊重实体护栏全局前置）。"""
    store = make_store(tmp_path)
    save(store, "N-0001", "GLM-5.3 的上下文窗口是 128k。", entities=["GLM-5.3"])
    save(store, "N-0002", "GLM-4.5 的上下文窗口是 64k。", entities=["GLM-4.5"])
    plan = plan_consolidation(store, ConsolidationSettings(), now=NOW)
    assert actions_of(plan, ACTION_CONFLICT) == []
    assert actions_of(plan, ACTION_MERGE) == []  # 实体不交同样不 merge


# ---- ⑥ max_* 截断 ----------------------------------------------------------


def test_max_rejudge_caps_batch(tmp_path: Path) -> None:
    """⑥12 条 low 笔记在默认 max_rejudge=10 下只出 10 条（按 note_ids 字典序取前 N）。"""
    store = make_store(tmp_path)
    for i in range(1, 13):
        save(store, f"N-{i:04d}", f"低置信断言 {i}。", confidence="low", entities=[f"主体{i}"])
    plan = plan_consolidation(store, ConsolidationSettings(), now=NOW)
    rejudges = actions_of(plan, ACTION_REJUDGE)
    assert len(rejudges) == 10
    assert [a.note_ids[0] for a in rejudges] == [f"N-{i:04d}" for i in range(1, 11)]


def test_max_merges_and_conflicts_caps(tmp_path: Path) -> None:
    """⑥max_merges / max_conflicts / max_conflict_pairs 截断（确定性排序后取前 N）。"""
    store = make_store(tmp_path)
    for i, value in enumerate(["32k", "64k", "128k", "200k"], start=1):
        save(
            store,
            f"N-{i:04d}",
            f"GLM-5.3 的上下文窗口是 {value}。",
            entities=["GLM-5.3"],
        )
    # 4 条同实体窗口笔记：6 个两两对全部相似（≥0.5）且全部槽位冲突
    plan = plan_consolidation(store, ConsolidationSettings(max_conflict_pairs=3), now=NOW)
    conflicts = actions_of(plan, ACTION_CONFLICT)
    assert [a.note_ids for a in conflicts] == [
        ["N-0001", "N-0002"],
        ["N-0001", "N-0003"],
        ["N-0001", "N-0004"],
    ]  # 只扫前 3 对（max_conflict_pairs=3），后续对根本不进比较器
    assert len(actions_of(plan, ACTION_MERGE)) == 6  # merge 不受 pairs 截断（上限 10 未触顶）

    plan = plan_consolidation(store, ConsolidationSettings(max_conflicts=2), now=NOW)
    assert [a.note_ids for a in actions_of(plan, ACTION_CONFLICT)] == [
        ["N-0001", "N-0002"],
        ["N-0001", "N-0003"],
    ]

    plan = plan_consolidation(store, ConsolidationSettings(max_merges=2), now=NOW)
    assert [a.note_ids for a in actions_of(plan, ACTION_MERGE)] == [
        ["N-0001", "N-0002"],
        ["N-0001", "N-0003"],
    ]


# ---- ⑦ dry-run 零变异 / ⑧ 可复算 --------------------------------------------


def test_dry_run_never_mutates_store(tmp_path: Path, monkeypatch) -> None:
    """⑦对临时 root 跑 plan 与 CLI dry-run，前后目录逐字节对比（含 mtime）零差异。"""
    store = rich_store(tmp_path)
    root = store.root
    before = snapshot_tree(root)

    plan_consolidation(store, ConsolidationSettings(), now=NOW)
    assert snapshot_tree(root) == before

    config_path = tmp_path / "config.toml"
    config_path.write_text(CONSOLIDATION_TOML, encoding="utf-8")
    monkeypatch.setattr(cli, "CONFIG_PATH", config_path)
    assert cli.main(["consolidate", "--root", str(root), "--dry-run"]) == 0
    assert snapshot_tree(root) == before
    assert cli.main(["consolidate", "--root", str(root), "--json"]) == 0
    assert snapshot_tree(root) == before


def test_same_state_produces_byte_identical_plan(tmp_path: Path) -> None:
    """⑧同一 store 状态 + 同一 settings + 同一 now → 两次 plan 逐字节相同。"""
    store = rich_store(tmp_path)
    settings = ConsolidationSettings()
    first = plan_consolidation(store, settings, now=NOW)
    second = plan_consolidation(store, settings, now=NOW)
    text_first = json.dumps(first.to_dict(), ensure_ascii=False, sort_keys=True)
    text_second = json.dumps(second.to_dict(), ensure_ascii=False, sort_keys=True)
    assert text_first == text_second
    assert len(first.actions) == 6  # 2 merge + 1 conflict + 1 refresh + 2 rejudge
    # 动作排序 (action, note_ids) 字典序
    keys = [(a.action, tuple(a.note_ids)) for a in first.actions]
    assert keys == sorted(keys)


# ---- ⑨ 幂等键 ---------------------------------------------------------------


def test_idempotency_key_stable_sensitive_and_formula_locked(tmp_path: Path) -> None:
    """⑨幂等键：同输入稳定；body 变化敏感；材料公式逐字节锁定（T3 的重试契约）。"""
    store = rich_store(tmp_path)
    settings = ConsolidationSettings()
    first = plan_consolidation(store, settings, now=NOW)
    second = plan_consolidation(store, settings, now=NOW)
    assert [a.idempotency_key for a in first.actions] == [
        a.idempotency_key for a in second.actions
    ]

    # 公式锁定：sha256(action + sorted(note_ids) + 各笔记 body_hash + action 参数)，
    # 材料以 "\n" 连接、payload 用 sort_keys JSON（实现 docstring 与本测试互为契约）。
    notes = {n.id: n for n in store.list_notes(status="active")}
    refresh = next(a for a in first.actions if a.action == ACTION_REFRESH)
    material = "\n".join(
        [
            refresh.action,
            *sorted(refresh.note_ids),
            *(
                hashlib.sha256(notes[nid].body.encode("utf-8")).hexdigest()
                for nid in sorted(refresh.note_ids)
            ),
            json.dumps(refresh.payload, ensure_ascii=False, sort_keys=True),
        ]
    )
    assert refresh.idempotency_key == hashlib.sha256(material.encode("utf-8")).hexdigest()

    # body 变化 → 该笔记动作的键变化，其余动作的键不变
    stale_keys = {
        a.note_ids[0]: a.idempotency_key for a in first.actions if a.action == ACTION_REJUDGE
    }
    note = store.get_note("N-0006")
    store.save_meta(note.meta, note.body + "（正文补充说明）")
    after = plan_consolidation(store, settings, now=NOW)
    new_keys = {
        a.note_ids[0]: a.idempotency_key for a in after.actions if a.action == ACTION_REJUDGE
    }
    assert set(new_keys) == set(stale_keys)
    assert new_keys["N-0006"] != stale_keys["N-0006"]
    assert new_keys["N-0007"] == stale_keys["N-0007"]


# ---- ⑩⑪ CLI 接线 -----------------------------------------------------------


def test_cli_exits_1_when_section_missing(tmp_path: Path, capsys, monkeypatch) -> None:
    """⑩[consolidation] 段缺失 → CLI 退出 1，提示未配置（与旧桩语义衔接）。"""
    store = make_store(tmp_path)
    save(store, "N-0001", MERGE_A, entities=["GLM-5.3"])
    config_path = tmp_path / "config.toml"
    config_path.write_text("[unanswered]\nenabled = true\n", encoding="utf-8")
    monkeypatch.setattr(cli, "CONFIG_PATH", config_path)
    exit_code = cli.main(["consolidate", "--root", str(store.root)])
    assert exit_code == 1
    assert "未配置" in capsys.readouterr().err


def test_cli_json_plan_shape(tmp_path: Path, capsys, monkeypatch) -> None:
    """⑪--json 输出可解析、字段齐全、含 settings_snapshot（merge 下限 0.5 在快照里）。"""
    store = make_store(tmp_path)
    save(store, "N-0001", "低置信断言待复核。", confidence="low")
    config_path = tmp_path / "config.toml"
    config_path.write_text(CONSOLIDATION_TOML, encoding="utf-8")
    monkeypatch.setattr(cli, "CONFIG_PATH", config_path)
    assert cli.main(["consolidate", "--root", str(store.root), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {"actions", "scanned", "settings_snapshot"}
    assert payload["scanned"] == 1
    snapshot = payload["settings_snapshot"]
    assert snapshot["enabled"] is True
    assert snapshot["merge_min_similarity"] == 0.5
    assert snapshot["rejudge_stale_ratio"] == 0.25
    assert snapshot["max_conflict_pairs"] == 50
    assert snapshot["max_merges"] == 10
    assert payload["actions"][0]["action"] == ACTION_REJUDGE
    assert set(payload["actions"][0]) == {
        "action",
        "note_ids",
        "reason",
        "idempotency_key",
        "payload",
    }


def test_cli_run_stub_returns_1(tmp_path: Path, capsys) -> None:
    """--run 在本任务保留桩语义：打印"执行器在 T3 接线"返回 1（T17 接走）。"""
    exit_code = cli.main(["consolidate", "--run", "--root", str(tmp_path / "wiki-data")])
    assert exit_code == 1
    assert "执行器在 T3 接线" in capsys.readouterr().err


def test_cli_text_output_is_human_readable(tmp_path: Path, capsys, monkeypatch) -> None:
    """非 --json 的人读表：标题行带扫描数与候选数，每个动作一行。"""
    store = make_store(tmp_path)
    save(store, "N-0001", MERGE_A, entities=["GLM-5.3"], created="2026-03-01T00:00:00+00:00")
    save(store, "N-0002", MERGE_B, entities=["GLM-5.3"], created="2026-01-01T00:00:00+00:00")
    config_path = tmp_path / "config.toml"
    config_path.write_text(CONSOLIDATION_TOML, encoding="utf-8")
    monkeypatch.setattr(cli, "CONFIG_PATH", config_path)
    assert cli.main(["consolidate", "--root", str(store.root)]) == 0
    out = capsys.readouterr().out
    assert "consolidation dry-run" in out
    assert "扫描 2 条" in out and "候选 1 个" in out
    assert "merge" in out and "N-0001" in out


# ---- 配置解析与逃生阀 -------------------------------------------------------


def test_settings_missing_section_returns_none() -> None:
    """段缺失 = 不接线：consolidation_settings 返回 None（CLI 报"未配置"）。"""
    assert consolidation_settings(None) is None
    assert consolidation_settings({}) is None
    assert consolidation_settings({"unanswered": {"enabled": True}}) is None


def test_settings_parses_section_with_tolerant_fallback() -> None:
    """段内非法值宽容回退默认、越界夹取、enabled 认字符串；不抛异常。"""
    config = tomllib.loads(
        "[consolidation]\n"
        'enabled = "false"\n'
        'merge_min_similarity = "bad"\n'
        "max_merges = 3\n"
        "max_rejudge = -2\n"
    )
    settings = consolidation_settings(config)
    assert settings is not None
    assert settings.enabled is False
    assert settings.merge_min_similarity == 0.5  # 非法回退默认（P2 移交裁定的独立下限）
    assert settings.rejudge_stale_ratio == 0.25
    assert settings.max_merges == 3
    assert settings.max_rejudge == 0  # 负值夹到 0（0 = 该类禁用）
    assert settings.max_conflict_pairs == 50
    assert settings.verification.similarity_floor == 0.3  # 未配置 [verification] → 模块默认
    assert settings.freshness.stale_ratio == 0.25


def test_settings_wires_verification_and_freshness_from_config() -> None:
    """判定尺度接线：[verification]/[freshness] 段随完整 config 注入（与 lint 同手法）。"""
    config = tomllib.loads(
        "[verification]\nsimilarity_floor = 0.4\n"
        "[freshness]\nreview_due_ratio = 0.99\nstale_ratio = 0.98\n"
        "[wiki]\nhalf_life_days = { volatile = 0.01 }\n"
        "[consolidation]\nenabled = true\n"
    )
    settings = consolidation_settings(config)
    assert settings is not None
    assert settings.verification.similarity_floor == 0.4  # conflict 判定 respect 配置地板
    assert settings.freshness.stale_ratio == 0.98
    assert settings.freshness.half_life_days["volatile"] == 0.01  # 半衰期回退 [wiki] 段


def test_enabled_false_produces_empty_plan(tmp_path: Path) -> None:
    """逃生阀：enabled=false → 零判定（plan 为空、scanned=0），不扫库。"""
    store = make_store(tmp_path)
    save(store, "N-0001", "低置信断言。", confidence="low")
    settings = ConsolidationSettings(enabled=False)
    plan = plan_consolidation(store, settings, now=NOW)
    assert plan.actions == []
    assert plan.scanned == 0
    assert plan.settings_snapshot["enabled"] is False


def test_tombstone_notes_are_excluded(tmp_path: Path) -> None:
    """墓碑是失效裁定审计记录（P2-F 裁定一）：四类候选都不收。"""
    store = make_store(tmp_path)
    save(store, "N-0001", MERGE_A, entities=["GLM-5.3"], tombstone=True)
    save(store, "N-0002", MERGE_B, entities=["GLM-5.3"], tombstone=True)
    save(
        store,
        "N-0003",
        "墓碑的低置信断言。",
        confidence="low",
        tombstone=True,
        volatility="volatile",
        observed_at="2026-01-01T00:00:00+00:00",
    )
    plan = plan_consolidation(store, ConsolidationSettings(), now=NOW)
    assert plan.actions == []
    assert plan.scanned == 3  # 墓碑计入扫描数（被扫到但不收候选）


def test_plan_format_text(tmp_path: Path) -> None:
    """人读表：标题行 + 每动作一行；禁用时输出禁用说明。"""
    store = rich_store(tmp_path)
    plan = plan_consolidation(store, ConsolidationSettings(), now=NOW)
    text = plan.format_text(root=str(store.root))
    assert "consolidation dry-run" in text
    assert "扫描 7 条" in text and "候选 6 个" in text
    for action in plan.actions:
        assert action.reason in text

    disabled = plan_consolidation(store, ConsolidationSettings(enabled=False), now=NOW)
    assert "已禁用" in disabled.format_text(root=str(store.root))
