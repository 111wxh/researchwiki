"""维护执行器测试（P4a Task 17 / T3：maintenance job queue + consolidate --run）。

覆盖 task-17 简报的验收清单（七组）：

① merge 执行：canonical body 含双方内容、absorbed status=merged+redirect_to 留痕、
   canonical sources 取并集；
② refresh：Mock fetcher 返回相同内容 → 仅 reviewed_at 变、无新快照；返回不同内容 →
   新证据笔记（kind=knowledge、带来源）+ 旧快照文件仍在 + mark_source_changed 留痕；
   fetcher 抛异常 → job failed、原笔记 body/frontmatter 逐字节不变；
③ rejudge：合法 confirm / supersede JSON 两条路径；畸形 JSON / 超字段 → failed +
   原笔记逐字节不变；usage 记进 job（无 usage 记 0 并在 result 注明来源——诚实计量）；
④ 幂等：同 plan 跑两遍，第二遍全 skipped、笔记数/快照数/台账数不增（第二遍只追加
   skipped 行），模型零重复调用（零重复 token 记账）；
⑤ retry（仅 failed 可重试，重试成功 attempt+1）、skip（含 reason 留痕，仅
   pending/failed）、list 按状态过滤；
⑥ conflict 动作开台账且两笔记均保持 active（绝不自动 merge 冲突对）；台账侧幂等：
   同 idempotency_key 不重复开台账；
⑦ JSONL 断电模拟：行中途被杀留下截断尾行 → 既有记录逐行可解析、list_jobs 容忍坏尾行、
   断电后继续追加不与截断行粘连。

全部零网络、零真实模型：fetcher 注入 fake，Provider 用 MockProvider / 不回 usage 的
极简 fake（诚实计量路径）。
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from researchwiki import cli
from researchwiki.llm.provider import MockProvider, StreamEvent, TokenUsage
from researchwiki.tools.fetch import FetchError, FetchResult
from researchwiki.wiki.consolidation import (
    ACTION_CONFLICT,
    ACTION_MERGE,
    ACTION_REFRESH,
    ACTION_REJUDGE,
    REJUDGE_TIER,
    ConsolidationPlan,
    ConsolidationSettings,
    plan_consolidation,
)
from researchwiki.wiki.frontmatter import SourceRef
from researchwiki.wiki.maintenance import (
    MaintenanceRunner,
    MaintenanceSettings,
    maintenance_settings,
)
from researchwiki.wiki.store import WikiStore, source_snapshot_path

NOW = datetime(2026, 6, 1, tzinfo=UTC)
URL = "https://example.test/a"

# 与 tests/test_consolidation.py 同一组实测文本（verification 尺度：merge 对
# 0.8447 ∈ [0.5, 0.95)；冲突对 0.8473 同槽位取值不同，且同样落在 merge 带）
MERGE_A = "GLM-5.3 支持工具调用。"
MERGE_B = "GLM-5.3 支持工具调用与函数并行。"
CONFLICT_A = "GLM-5.3 的上下文窗口是 128k。"
CONFLICT_B = "GLM-5.3 的上下文窗口是 64k。"

CONSOLIDATION_TOML = """\
[consolidation]
enabled = true
"""

MAINTENANCE_TOML = """\
[maintenance]
enabled = true
"""


# ---- 脚手架 ------------------------------------------------------------------


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
    )


def actions_of(plan, action: str) -> list:
    return [a for a in plan.actions if a.action == action]


def single_action_plan(store: WikiStore, action: str) -> ConsolidationPlan:
    """只保留指定动作的 plan（conflict/merge 并存的 对 用它隔离出单动作执行）。"""
    full = plan_consolidation(store, ConsolidationSettings(), now=NOW)
    return ConsolidationPlan(
        actions=actions_of(full, action),
        scanned=full.scanned,
        settings_snapshot=full.settings_snapshot,
    )


def snapshot_count(store: WikiStore) -> int:
    """sources/ 下快照正文数（append-only 的"旧快照仍在"判据）。"""
    if not store.sources_dir.is_dir():
        return 0
    return sum(1 for _ in store.sources_dir.rglob("content.md"))


def fetch_result(content_hash: str, text: str = "") -> FetchResult:
    return FetchResult(
        url=URL, final_url=URL, text=text, content_hash=content_hash, http_status=200,
        truncated=False,
    )


class FakeFetcher:
    """注入的假抓取器：按脚本返回结果或抛异常，记录调用。"""

    def __init__(self, result: FetchResult | None = None, error: Exception | None = None):
        self.result = result
        self.error = error
        self.calls: list[str] = []

    def __call__(self, url: str) -> FetchResult:
        self.calls.append(url)
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


class FakeRouter:
    """注入的假路由：恒回同一 provider，记录请求的档位。"""

    def __init__(self, provider):
        self.provider = provider
        self.tiers: list[str] = []

    def get(self, tier: str):
        self.tiers.append(tier)
        return self.provider


class NoUsageProvider:
    """不回 usage 事件的极简 Provider：验证诚实计量（没有 usage 记 0 并注明来源）。"""

    model = "fake-no-usage"
    tier = "cheap"

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def stream(self, messages, *, system=None, tools=None):
        self.calls += 1
        yield StreamEvent(type="text_delta", delta=self.text)


def make_runner(store: WikiStore, *, fetcher=None, provider=None) -> MaintenanceRunner:
    return MaintenanceRunner(
        store,
        FakeRouter(provider or MockProvider("", tier="cheap", model="mock-cheap")),
        MaintenanceSettings(),
        fetcher=fetcher,
    )


def refresh_store(tmp_path: Path) -> WikiStore:
    """review_due + 带来源的 volatile 笔记（refresh 候选），并预落旧快照文件。"""
    store = make_store(tmp_path)
    save(
        store,
        "N-0001",
        "张三负责项目部署与发布。",
        entities=["张三"],
        volatility="volatile",
        observed_at="2026-04-17T00:00:00+00:00",  # 45 天 → decay 0.354 → review_due
        sources=[SourceRef(url=URL, content_hash="hash-a")],
    )
    snapshot = source_snapshot_path(store.sources_dir, URL, "hash-a")
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_text("旧快照内容", encoding="utf-8")
    return store


# ---- ① merge 执行 ------------------------------------------------------------


def test_merge_merges_body_and_leaves_trail(tmp_path: Path) -> None:
    """①canonical body 含双方内容、sources 并集；absorbed 置 merged + redirect_to 留痕。"""
    store = make_store(tmp_path)
    save(
        store, "N-0001", MERGE_A, entities=["GLM-5.3"],
        created="2026-03-01T00:00:00+00:00",
        sources=[SourceRef(url="https://example.test/a", content_hash="hash-a")],
    )
    save(
        store, "N-0002", MERGE_B, entities=["GLM-5.3"],
        created="2026-01-01T00:00:00+00:00",
        sources=[SourceRef(url="https://example.test/b", content_hash="hash-b")],
    )
    runner = make_runner(store)
    records = runner.run(single_action_plan(store, ACTION_MERGE))

    assert len(records) == 1
    record = records[0]
    assert record.status == "done" and record.action == ACTION_MERGE
    assert record.error is None and record.model is None
    assert record.tokens_in == 0 and record.tokens_out == 0  # 非模型动作零 token

    canonical = store.get_note("N-0002")  # created 更早者是规范
    assert canonical is not None and canonical.meta.status == "active"
    assert MERGE_A in canonical.body and MERGE_B in canonical.body  # 双方内容都在
    merged_urls = {(s.url, s.content_hash) for s in canonical.meta.sources}
    assert merged_urls == {
        ("https://example.test/a", "hash-a"),
        ("https://example.test/b", "hash-b"),
    }

    absorbed = store.get_note("N-0001")
    assert absorbed is not None
    assert absorbed.meta.status == "merged" and absorbed.meta.redirect_to == "N-0002"
    assert absorbed.body == MERGE_A  # 留痕记录保留原始表述（旧 ID 永不消失）
    assert store.follow_redirect("N-0001").id == "N-0002"  # 链路可达规范笔记

    # 账本落盘：JSONL 一行 + 单 job 详情 JSON
    ledger = store.root / "maintenance" / "jobs.jsonl"
    assert ledger.is_file()
    assert (store.root / "maintenance" / "jobs" / f"{record.job_id}.json").is_file()


def test_merge_refuses_when_state_changed(tmp_path: Path) -> None:
    """①补充：目标已非 active（状态漂移）→ job failed、双方原笔记零改动。"""
    store = make_store(tmp_path)
    save(store, "N-0001", MERGE_A, entities=["GLM-5.3"], created="2026-03-01T00:00:00+00:00")
    save(store, "N-0002", MERGE_B, entities=["GLM-5.3"], created="2026-01-01T00:00:00+00:00")
    plan = single_action_plan(store, ACTION_MERGE)
    # 计划之后、执行之前 absorbed 已被外部处置（merged）——执行器拒绝
    absorbed = store.get_note("N-0001")
    store.save_meta(absorbed.meta.replace(status="merged", redirect_to="N-0002"), absorbed.body)

    before = (store.root / "notes" / "N-0002.md").read_bytes()
    record = make_runner(store).run(plan)[0]
    assert record.status == "failed" and record.error is not None
    assert (store.root / "notes" / "N-0002.md").read_bytes() == before  # canonical 不动


# ---- ② refresh 执行 ----------------------------------------------------------


def test_refresh_unchanged_only_refreshes_reviewed_at(tmp_path: Path) -> None:
    """②内容未变 → 仅 reviewed_at 刷新、无新笔记、无新快照，result 标 source_unchanged。"""
    store = refresh_store(tmp_path)
    note = store.get_note("N-0001")
    assert note.meta.reviewed_at is None
    snapshots_before = snapshot_count(store)
    dirs_before = {p for p in store.sources_dir.rglob("*") if p.is_dir()}

    runner = make_runner(store, fetcher=FakeFetcher(fetch_result("hash-a", "旧快照内容")))
    record = runner.run(single_action_plan(store, ACTION_REFRESH))[0]

    assert record.status == "done"
    assert record.result["source_unchanged"] is True
    after = store.get_note("N-0001")
    assert after.meta.reviewed_at is not None  # 仅刷新 reviewed_at
    assert after.body == note.body and after.meta.status == "active"  # 正文零改动
    assert len(store.list_notes(status=None)) == 1  # 无新笔记
    assert snapshot_count(store) == snapshots_before  # 无新快照
    assert {p for p in store.sources_dir.rglob("*") if p.is_dir()} == dirs_before


def test_refresh_changed_creates_evidence_note_and_marks(tmp_path: Path) -> None:
    """②内容变化 → mark_source_changed 留痕 + 新证据笔记（kind=knowledge 带来源）；
    旧快照文件逐字节仍在，旧笔记保持 active 且正文不动。"""
    store = refresh_store(tmp_path)
    old_snapshot = source_snapshot_path(store.sources_dir, URL, "hash-a")
    old_bytes = old_snapshot.read_bytes()
    note = store.get_note("N-0001")
    new_text = "新内容：张三现在负责架构评审。"

    runner = make_runner(store, fetcher=FakeFetcher(fetch_result("hash-b", new_text)))
    record = runner.run(single_action_plan(store, ACTION_REFRESH))[0]

    assert record.status == "done"
    assert record.result["source_unchanged"] is False
    assert record.result["new_content_hash"] == "hash-b"

    after = store.get_note("N-0001")
    assert after.meta.status == "active" and after.body == note.body  # 旧笔记零改动
    assert after.meta.source_changed_at is not None  # mark_source_changed 留痕
    assert after.meta.extra["source_changed_hash"][URL] == "hash-b"

    notes = {n.id: n for n in store.list_notes(status="active")}
    new_id = record.result["new_note_id"]
    assert new_id in notes and new_id != "N-0001"
    new_note = notes[new_id]
    assert new_note.kind == "knowledge"
    assert new_note.body == new_text  # 新证据可追溯：正文即抓取内容
    assert [(s.url, s.content_hash) for s in new_note.meta.sources] == [(URL, "hash-b")]

    assert old_snapshot.read_bytes() == old_bytes  # 旧快照一律保留
    assert snapshot_count(store) == 1  # 快照版本化是 fetch_url 的内部事务，执行器不自写


def test_refresh_fetch_failure_keeps_note_byte_identical(tmp_path: Path) -> None:
    """②抓取异常 → job failed、原笔记文件逐字节不变、无新笔记、无留痕写入。"""
    store = refresh_store(tmp_path)
    note_path = store.root / "notes" / "N-0001.md"
    before = note_path.read_bytes()
    notes_before = len(store.list_notes(status=None))

    runner = make_runner(
        store, fetcher=FakeFetcher(error=FetchError(URL, "connection reset"))
    )
    record = runner.run(single_action_plan(store, ACTION_REFRESH))[0]

    assert record.status == "failed"
    assert "FetchError" in record.error
    assert note_path.read_bytes() == before  # 逐字节不变（失败保原文）
    assert len(store.list_notes(status=None)) == notes_before
    assert store.get_note("N-0001").meta.source_changed_at is None


# ---- ③ rejudge 执行 ----------------------------------------------------------


def rejudge_store(tmp_path: Path) -> WikiStore:
    store = make_store(tmp_path)
    save(
        store,
        "N-0001",
        "低置信断言待复核。",
        confidence="low",
        observed_at="2026-05-30T00:00:00+00:00",
    )
    return store


def test_rejudge_confirm_refreshes_reviewed_at(tmp_path: Path) -> None:
    """③合法 confirm JSON → reviewed_at 刷新、笔记保持 active；usage 记进 job。"""
    store = rejudge_store(tmp_path)
    provider = MockProvider(
        '{"verdict": "confirm", "reason": "复核后仍然成立"}',
        tier="cheap",
        model="mock-cheap",
        usage=TokenUsage(input_tokens=111, output_tokens=22),
    )
    record = make_runner(store, provider=provider).run(
        single_action_plan(store, ACTION_REJUDGE)
    )[0]

    assert record.status == "done"
    assert record.model == "mock-cheap"  # job 记 model
    assert record.tokens_in == 111 and record.tokens_out == 22  # usage 如实记账
    assert record.result["tokens_source"] == "provider_usage"
    assert record.result["verdict"] == "confirm"
    note = store.get_note("N-0001")
    assert note.meta.status == "active" and note.meta.reviewed_at is not None
    assert note.body == "低置信断言待复核。"  # 正文不动


def test_rejudge_supersede_marks_without_replacement(tmp_path: Path) -> None:
    """③合法 supersede JSON → 原笔记置 superseded（带理由、不删历史、不建替代笔记）。"""
    store = rejudge_store(tmp_path)
    provider = MockProvider(
        '{"verdict": "supersede", "reason": "来源已撤回该说法"}',
        tier="cheap",
        model="mock-cheap",
    )
    record = make_runner(store, provider=provider).run(
        single_action_plan(store, ACTION_REJUDGE)
    )[0]

    assert record.status == "done" and record.result["verdict"] == "supersede"
    note = store.get_note("N-0001")
    assert note.meta.status == "superseded"
    assert note.body == "低置信断言待复核。"  # 不删除历史
    assert note.meta.extra["supersede_reason"] == "来源已撤回该说法"
    assert note.meta.extra["supersede_reason_at"] is not None
    assert len(store.list_notes(status=None)) == 1  # 不创建替代笔记


@pytest.mark.parametrize(
    "raw",
    [
        "这不是 JSON",
        '{"verdict": "confirm", "reason": "ok", "extra": 1}',  # 超字段
        '{"verdict": "maybe", "reason": "ok"}',  # verdict 不在词表
        '{"verdict": "confirm"}',  # 缺 reason
        '{"verdict": "confirm", "reason": 3}',  # reason 非字符串
    ],
)
def test_rejudge_malformed_output_fails_and_keeps_note(tmp_path: Path, raw: str) -> None:
    """③畸形/超字段输出 → job failed，原笔记文件逐字节不变（模型输出畸形绝不落新事实）。"""
    store = rejudge_store(tmp_path)
    note_path = store.root / "notes" / "N-0001.md"
    before = note_path.read_bytes()
    provider = MockProvider(raw, tier="cheap", model="mock-cheap")

    record = make_runner(store, provider=provider).run(
        single_action_plan(store, ACTION_REJUDGE)
    )[0]

    assert record.status == "failed"
    assert record.error is not None
    assert note_path.read_bytes() == before
    assert store.get_note("N-0001").meta.status == "active"


def test_rejudge_without_usage_records_zero_honestly(tmp_path: Path) -> None:
    """③诚实计量：provider 不回 usage → tokens 记 0，result 注明来源。"""
    store = rejudge_store(tmp_path)
    provider = NoUsageProvider('{"verdict": "confirm", "reason": "仍然成立"}')
    record = make_runner(store, provider=provider).run(
        single_action_plan(store, ACTION_REJUDGE)
    )[0]

    assert record.status == "done"
    assert record.model == "fake-no-usage"
    assert record.tokens_in == 0 and record.tokens_out == 0
    assert record.result["tokens_source"] == "none_reported"


# ---- ④ 幂等 ------------------------------------------------------------------


def idempotent_store(tmp_path: Path) -> WikiStore:
    """覆盖三类动作的库：merge 对 + refresh 候选（带来源）+ low 笔记（rejudge）。"""
    store = make_store(tmp_path)
    save(
        store, "N-0001", MERGE_A, entities=["GLM-5.3"], created="2026-03-01T00:00:00+00:00",
        sources=[SourceRef(url="https://example.test/a", content_hash="hash-a")],
    )
    save(
        store, "N-0002", MERGE_B, entities=["GLM-5.3"], created="2026-01-01T00:00:00+00:00",
        sources=[SourceRef(url="https://example.test/b", content_hash="hash-b")],
    )
    save(
        store, "N-0003", "张三负责项目部署与发布。", entities=["张三"],
        volatility="volatile", observed_at="2026-04-17T00:00:00+00:00",
        sources=[SourceRef(url=URL, content_hash="hash-a")],
    )
    save(store, "N-0004", "低置信断言。", confidence="low", observed_at="2026-05-30T00:00:00+00:00")
    snapshot = source_snapshot_path(store.sources_dir, URL, "hash-a")
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_text("旧快照内容", encoding="utf-8")
    return store


def test_rerunning_same_plan_is_all_skipped(tmp_path: Path) -> None:
    """④同 plan 跑两遍：第二遍全 skipped、笔记/快照/台账数不增、只追加 skipped 行、
    模型零重复调用（零重复 token 记账）。"""
    store = idempotent_store(tmp_path)
    full = plan_consolidation(store, ConsolidationSettings(), now=NOW)
    provider = MockProvider(
        '{"verdict": "confirm", "reason": "仍然成立"}', tier="cheap", model="mock-cheap"
    )
    runner = make_runner(
        store, fetcher=FakeFetcher(fetch_result("hash-a", "旧快照内容")), provider=provider
    )

    first = runner.run(full)
    assert all(r.status == "done" for r in first)
    assert len(first) == len(full.actions)
    notes_after_first = len(store.list_notes(status=None))
    snapshots_after_first = snapshot_count(store)
    conflicts_after_first = len(store.list_conflicts(status=None))
    ledger = store.root / "maintenance" / "jobs.jsonl"
    lines_after_first = len(ledger.read_text(encoding="utf-8").splitlines())
    provider_calls_after_first = len(provider.calls)

    second = runner.run(full)  # 同一 plan 重放
    assert len(second) == len(full.actions)
    assert all(r.status == "skipped" for r in second)
    assert all(r.result["reason"] == "idempotent" for r in second)

    assert len(store.list_notes(status=None)) == notes_after_first  # 零重复记忆
    assert snapshot_count(store) == snapshots_after_first  # 零重复快照
    assert len(store.list_conflicts(status=None)) == conflicts_after_first
    lines = ledger.read_text(encoding="utf-8").splitlines()
    assert len(lines) == lines_after_first + len(full.actions)  # 只追加 skipped 行
    new_lines = lines[lines_after_first:]
    assert all(json.loads(line)["status"] == "skipped" for line in new_lines)
    assert len(provider.calls) == provider_calls_after_first  # 模型零重复调用


# ---- ⑤ retry / skip / list ---------------------------------------------------


def test_retry_failed_refresh_job(tmp_path: Path) -> None:
    """⑤failed job 可重试：新 job 记录 attempt+1，重试成功 done；旧 failed 记录仍在账本。"""
    store = refresh_store(tmp_path)
    fetcher = FakeFetcher(error=FetchError(URL, "boom"))
    runner = make_runner(store, fetcher=fetcher)
    failed = runner.run(single_action_plan(store, ACTION_REFRESH))[0]
    assert failed.status == "failed" and failed.attempt == 1

    fetcher.error = None  # 故障恢复后重试
    fetcher.result = fetch_result("hash-a", "旧快照内容")
    retried = runner.retry_job(failed.job_id)
    assert retried.status == "done"
    assert retried.job_id != failed.job_id  # 新 job 记录（账本追加式）
    assert retried.attempt == 2
    assert retried.idempotency_key == failed.idempotency_key

    statuses = {r.job_id: r.status for r in runner.list_jobs()}
    assert statuses[failed.job_id] == "failed"  # 失败历史保留
    assert statuses[retried.job_id] == "done"
    assert len(runner.list_jobs(status="failed")) == 1

    with pytest.raises(ValueError):
        runner.retry_job(retried.job_id)  # done 不可重试
    with pytest.raises(ValueError):
        runner.retry_job("J-9999")  # 不存在


def test_skip_failed_job_records_reason(tmp_path: Path) -> None:
    """⑤skip（仅 pending/failed）：reason 留痕进 result；skip 后不可再重试。"""
    store = refresh_store(tmp_path)
    runner = make_runner(store, fetcher=FakeFetcher(error=FetchError(URL, "boom")))
    failed = runner.run(single_action_plan(store, ACTION_REFRESH))[0]

    skipped = runner.skip_job(failed.job_id, "来源永久失效，人工裁决保留原记忆")
    assert skipped.job_id == failed.job_id  # 同一 job 的状态流转（账本追加新行）
    assert skipped.status == "skipped"
    assert skipped.result["reason"] == "来源永久失效，人工裁决保留原记忆"
    assert runner.list_jobs(status="skipped")[0].job_id == failed.job_id

    with pytest.raises(ValueError):
        runner.retry_job(failed.job_id)  # skipped 不可重试
    with pytest.raises(ValueError):
        runner.skip_job(failed.job_id, "再跳一次")  # 已 skipped 不可再跳


def test_skip_rejects_done_job(tmp_path: Path) -> None:
    """⑤补充：done 状态不可跳过。"""
    store = rejudge_store(tmp_path)
    runner = make_runner(
        store,
        provider=MockProvider('{"verdict": "confirm", "reason": "ok"}', tier="cheap"),
    )
    done = runner.run(single_action_plan(store, ACTION_REJUDGE))[0]
    assert done.status == "done"
    with pytest.raises(ValueError):
        runner.skip_job(done.job_id, "不该成功")


def test_list_jobs_filters_by_status(tmp_path: Path) -> None:
    """⑤list 过滤：status=None 全量（按 job_id 升序），指定状态只回该状态。"""
    store = idempotent_store(tmp_path)
    runner = make_runner(
        store,
        fetcher=FakeFetcher(error=FetchError(URL, "boom")),
        provider=MockProvider("坏输出", tier="cheap"),
    )
    runner.run(plan_consolidation(store, ConsolidationSettings(), now=NOW))
    all_records = runner.list_jobs()
    assert [r.job_id for r in all_records] == sorted(r.job_id for r in all_records)
    assert {r.status for r in all_records} >= {"failed"}
    failed_ids = {r.job_id for r in runner.list_jobs(status="failed")}
    assert failed_ids and failed_ids < {r.job_id for r in all_records}
    assert runner.list_jobs(status="done") == [] or all(
        r.status == "done" for r in runner.list_jobs(status="done")
    )


# ---- ⑥ conflict 执行 ---------------------------------------------------------


def conflict_store(tmp_path: Path) -> WikiStore:
    store = make_store(tmp_path)
    save(store, "N-0001", CONFLICT_A, entities=["GLM-5.3"], created="2026-03-01T00:00:00+00:00")
    save(store, "N-0002", CONFLICT_B, entities=["GLM-5.3"], created="2026-01-01T00:00:00+00:00")
    return store


def test_conflict_opens_ledger_and_keeps_both_notes(tmp_path: Path) -> None:
    """⑥conflict 动作开台账（claim 带双方断言槽位），两笔记均保持 active——绝不自动 merge。"""
    store = conflict_store(tmp_path)
    plan = single_action_plan(store, ACTION_CONFLICT)
    action = plan.actions[0]

    record = make_runner(store).run(plan)[0]
    assert record.status == "done" and record.result["reused"] is False

    conflicts = store.list_conflicts(status="open")
    assert len(conflicts) == 1
    conflict = conflicts[0]
    assert conflict.claim_a["note_id"] == "N-0001"  # ID 小者为 prior（T16 方向）
    assert conflict.claim_b["note_id"] == "N-0002"
    assert conflict.claim_a["role"] == "prior" and conflict.claim_b["role"] == "evidence"
    # 台账内容取自 payload 的断言槽位（不重新比较）
    assert conflict.claim_a["slots"] == action.payload["slots"]
    assert conflict.claim_a["idempotency_key"] == action.idempotency_key

    for note_id in ("N-0001", "N-0002"):  # 双方保持 active、原文零改动
        note = store.get_note(note_id)
        assert note.meta.status == "active"
    assert store.get_note("N-0001").body == CONFLICT_A
    assert store.get_note("N-0002").body == CONFLICT_B


def test_conflict_does_not_duplicate_ledger_entry(tmp_path: Path) -> None:
    """⑥台账侧幂等：job 账本丢失后重放同动作，同 idempotency_key 不重复开台账。"""
    store = conflict_store(tmp_path)
    plan = single_action_plan(store, ACTION_CONFLICT)
    runner = make_runner(store)
    assert runner.run(plan)[0].status == "done"
    assert len(store.list_conflicts(status=None)) == 1

    (store.root / "maintenance" / "jobs.jsonl").unlink()  # 模拟 job 账本丢失
    record = make_runner(store).run(plan)[0]
    assert record.status == "done"
    assert record.result["reused"] is True
    assert len(store.list_conflicts(status=None)) == 1  # 不重复开台账


# ---- ⑦ JSONL 断电模拟 --------------------------------------------------------


def test_jsonl_survives_power_cut_mid_append(tmp_path: Path) -> None:
    """⑦行中途被杀留下截断尾行：既有记录逐行可解析、list_jobs 容忍坏尾行、
    断电后继续追加不与截断行粘连。"""
    store = rejudge_store(tmp_path)
    runner = make_runner(
        store,
        provider=MockProvider('{"verdict": "confirm", "reason": "ok"}', tier="cheap"),
    )
    (runner.ledger_path.parent).mkdir(parents=True, exist_ok=True)
    first = runner.run(single_action_plan(store, ACTION_REJUDGE))
    assert len(first) == 1

    # 模拟断电：追加动作在行中途被杀（尾行残缺、无换行符）
    with runner.ledger_path.open("ab") as f:
        f.write(b'{"job_id": "J-0099", "act')

    # 既有记录完好：完整行逐行可解析；坏尾行被容忍、不影响 list
    lines = runner.ledger_path.read_text(encoding="utf-8").splitlines()
    parsed = []
    for line in lines[:-1]:
        parsed.append(json.loads(line))  # 断电前的记录全部可解析
    assert [item["job_id"] for item in parsed] == [first[0].job_id]
    assert [r.job_id for r in runner.list_jobs()] == [first[0].job_id]

    # 断电恢复后继续执行：新记录补换行后追加，不与截断行粘连
    second = runner.run(single_action_plan(store, ACTION_REJUDGE))
    assert second[0].job_id == "J-0002"  # 编号只认可解析记录，不被坏尾行带偏
    records = runner.list_jobs()
    assert [r.job_id for r in records] == [first[0].job_id, "J-0002"]
    # 截断行被补换行隔离在自己的行上（不粘连新记录）：读取侧跳过它，
    # 其余行仍逐行可解析
    lines_after = runner.ledger_path.read_text(encoding="utf-8").splitlines()
    assert len(lines_after) == 3
    with pytest.raises(json.JSONDecodeError):
        json.loads(lines_after[1])  # 人为截断行原样保留
    json.loads(lines_after[2])  # 断电后追加的新记录完整可解析


# ---- 配置解析 ----------------------------------------------------------------


def test_maintenance_settings_missing_section_returns_none() -> None:
    """段缺失 = 不接线：maintenance_settings 返回 None（CLI 报"未配置"）。"""
    assert maintenance_settings(None) is None
    assert maintenance_settings({}) is None
    assert maintenance_settings({"consolidation": {"enabled": True}}) is None


def test_maintenance_settings_parses_with_tolerant_fallback() -> None:
    """enabled 认字符串；非法模型档回退 cheap；不抛异常。"""
    config = {"maintenance": {"enabled": "false", "rejudge_model_tier": "ultra"}}
    settings = maintenance_settings(config)
    assert settings is not None
    assert settings.enabled is False
    assert settings.rejudge_model_tier == REJUDGE_TIER

    ok = maintenance_settings({"maintenance": {"rejudge_model_tier": "strong"}})
    assert ok.enabled is True and ok.rejudge_model_tier == "strong"


def test_runner_refuses_when_disabled(tmp_path: Path) -> None:
    """逃生阀：settings.enabled=false → run 零动作零留痕。"""
    store = rejudge_store(tmp_path)
    runner = MaintenanceRunner(
        store,
        FakeRouter(MockProvider("", tier="cheap")),
        MaintenanceSettings(enabled=False),
        fetcher=FakeFetcher(fetch_result("hash-a")),
    )
    assert runner.run(single_action_plan(store, ACTION_REJUDGE)) == []
    assert not (store.root / "maintenance").exists()


# ---- CLI 接线 ----------------------------------------------------------------


def test_cli_run_executes_plan(tmp_path: Path, capsys, monkeypatch) -> None:
    """--run 端到端：计划执行、笔记合并、job 账本落盘、退出 0。"""
    store = make_store(tmp_path)
    save(store, "N-0001", MERGE_A, entities=["GLM-5.3"], created="2026-03-01T00:00:00+00:00")
    save(store, "N-0002", MERGE_B, entities=["GLM-5.3"], created="2026-01-01T00:00:00+00:00")
    config_path = tmp_path / "config.toml"
    config_path.write_text(CONSOLIDATION_TOML + MAINTENANCE_TOML, encoding="utf-8")
    monkeypatch.setattr(cli, "CONFIG_PATH", config_path)

    assert cli.main(["consolidate", "--run", "--root", str(store.root)]) == 0
    out = capsys.readouterr().out
    assert "done 1" in out and "J-0001" in out
    assert store.get_note("N-0001").meta.status == "merged"
    assert (store.root / "maintenance" / "jobs.jsonl").is_file()

    assert cli.main(["consolidate", "--run", "--root", str(store.root), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    # --run 每次重新计划：merge 已完成，同样的候选不再出现（状态层面的幂等）；
    # 账本层面同 plan 重放 → skipped 的幂等由 test_rerunning_same_plan_is_all_skipped 锁定
    assert payload["jobs"] == [] and payload["scanned"] == 1


def test_cli_run_without_maintenance_section_exits_1(tmp_path: Path, capsys, monkeypatch) -> None:
    """[maintenance] 段缺失 = 不接线：--run 不做任何事，退出 1。"""
    store = make_store(tmp_path)
    save(store, "N-0001", MERGE_A, entities=["GLM-5.3"], created="2026-03-01T00:00:00+00:00")
    save(store, "N-0002", MERGE_B, entities=["GLM-5.3"], created="2026-01-01T00:00:00+00:00")
    config_path = tmp_path / "config.toml"
    config_path.write_text(CONSOLIDATION_TOML, encoding="utf-8")
    monkeypatch.setattr(cli, "CONFIG_PATH", config_path)

    exit_code = cli.main(["consolidate", "--run", "--root", str(store.root)])
    assert exit_code == 1
    assert "未配置" in capsys.readouterr().err
    assert store.get_note("N-0001").meta.status == "active"  # 零动作
    assert not (store.root / "maintenance").exists()


def test_cli_run_disabled_is_noop(tmp_path: Path, capsys, monkeypatch) -> None:
    """maintenance enabled=false：--run 零动作零留痕，退出 0。"""
    store = make_store(tmp_path)
    save(store, "N-0001", MERGE_A, entities=["GLM-5.3"], created="2026-03-01T00:00:00+00:00")
    save(store, "N-0002", MERGE_B, entities=["GLM-5.3"], created="2026-01-01T00:00:00+00:00")
    config_path = tmp_path / "config.toml"
    config_path.write_text(CONSOLIDATION_TOML + "[maintenance]\nenabled = false\n")
    monkeypatch.setattr(cli, "CONFIG_PATH", config_path)

    assert cli.main(["consolidate", "--run", "--root", str(store.root)]) == 0
    assert "已禁用" in capsys.readouterr().out
    assert store.get_note("N-0001").meta.status == "active"
    assert not (store.root / "maintenance").exists()


def seed_failed_rejudge_job(store: WikiStore) -> str:
    """直接用执行器造一个 failed 的 rejudge job（CLI 侧 --jobs/--skip 测试的垫材）。"""
    runner = make_runner(store, provider=MockProvider("坏输出", tier="cheap", model="mock-cheap"))
    return runner.run(single_action_plan(store, ACTION_REJUDGE))[0].job_id


def test_cli_jobs_and_skip(tmp_path: Path, capsys, monkeypatch) -> None:
    """--jobs（含 --status 过滤）与 --skip --reason：账本可查、failed 可跳过（退出 0）。"""
    store = rejudge_store(tmp_path)
    job_id = seed_failed_rejudge_job(store)
    config_path = tmp_path / "config.toml"
    config_path.write_text(CONSOLIDATION_TOML + MAINTENANCE_TOML, encoding="utf-8")
    monkeypatch.setattr(cli, "CONFIG_PATH", config_path)

    assert cli.main(["consolidate", "--jobs", "--root", str(store.root)]) == 0
    out = capsys.readouterr().out
    assert job_id in out and "failed" in out

    assert cli.main(["consolidate", "--jobs", "--status", "done", "--root", str(store.root)]) == 0
    assert "共 0 条" in capsys.readouterr().out

    assert (
        cli.main(
            ["consolidate", "--skip", job_id, "--reason", "人工保留", "--root", str(store.root)]
        )
        == 0
    )
    assert "已跳过" in capsys.readouterr().out
    record = MaintenanceRunner(
        store, FakeRouter(MockProvider("", tier="cheap")), MaintenanceSettings()
    ).list_jobs(status="skipped")[0]
    assert record.result["reason"] == "人工保留"


def test_cli_retry_failed_rejudge_with_mock(tmp_path: Path, capsys, monkeypatch) -> None:
    """--retry：mock 档重试 failed rejudge（输出仍畸形 → 再次 failed，退出 1；attempt+1）。"""
    store = rejudge_store(tmp_path)
    job_id = seed_failed_rejudge_job(store)
    config_path = tmp_path / "config.toml"
    config_path.write_text(CONSOLIDATION_TOML + MAINTENANCE_TOML, encoding="utf-8")
    monkeypatch.setattr(cli, "CONFIG_PATH", config_path)

    assert cli.main(["consolidate", "--retry", job_id, "--root", str(store.root)]) == 1
    runner = MaintenanceRunner(
        store, FakeRouter(MockProvider("", tier="cheap")), MaintenanceSettings()
    )
    records = {r.job_id: r for r in runner.list_jobs()}
    retried = [r for r in records.values() if r.attempt == 2]
    assert len(retried) == 1 and retried[0].status == "failed"
    assert records[job_id].status == "failed"  # 原失败记录保留


def test_cli_skip_requires_reason(tmp_path: Path, capsys, monkeypatch) -> None:
    """--skip 不带 --reason：用法错误，退出 1，零写盘。"""
    store = rejudge_store(tmp_path)
    job_id = seed_failed_rejudge_job(store)
    config_path = tmp_path / "config.toml"
    config_path.write_text(CONSOLIDATION_TOML + MAINTENANCE_TOML, encoding="utf-8")
    monkeypatch.setattr(cli, "CONFIG_PATH", config_path)

    before = (store.root / "maintenance" / "jobs.jsonl").read_bytes()
    assert cli.main(["consolidate", "--skip", job_id, "--root", str(store.root)]) == 1
    assert "reason" in capsys.readouterr().err
    assert (store.root / "maintenance" / "jobs.jsonl").read_bytes() == before
