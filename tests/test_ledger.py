"""未答问题台账（P4a Task 15：kind=experience 反幻觉记忆）测试。

覆盖 task-15 简报的验收清单（六组）：

① record_unanswered 写出 kind=experience / extra.ledger="unanswered_question" /
   importance 0.8 的 active 台账笔记，正文含问题原文、原因与判定日期；
② 同 trace_id 重复 record 幂等（不新增台账，首条不被覆写）；
③ iter_active_unanswered 排除 superseded 与 tombstone；
④ AgentLoop 收尾触发路径：no_notes（零产出 run）与 below_min_fresh（fresh 来源
   不足 run）；段缺失 / enabled=false 零动作零留痕；dry_run=true 不写台账笔记但
   run_dir/unanswered.json 留痕；
⑤ 护栏注入：相似问题（≥guard_similarity）注入护栏块、不相似问题不注入、
   dry_run 台账不注入；
⑥ 护栏块出现在 Prior 块之前，且不占用 [prior].top_k 配额。

全部零网络、零模型调用（AgentLoop 用 ScriptedProvider，相似度复用
wiki/verification.py 的确定性 bigram 特征哈希度量——同一把尺）。
"""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from researchwiki.llm.accounting import TokenAccountant
from researchwiki.llm.provider import ScriptedProvider, StreamEvent, TokenUsage
from researchwiki.loop.agent_loop import AgentLoop
from researchwiki.tools import MockSearch
from researchwiki.wiki.embeddings import MockEmbeddingProvider
from researchwiki.wiki.ledger import (
    LEDGER_IMPORTANCE,
    LEDGER_MARKER,
    REASON_BELOW_MIN_FRESH,
    REASON_NO_NOTES,
    decide_unanswered_reason,
    iter_active_unanswered,
    parse_unanswered_entry,
    question_similarity,
    record_unanswered,
    unanswered_settings,
)
from researchwiki.wiki.prior import (
    PRIOR_CONTEXT_LABEL,
    UNANSWERED_GUARD_HEADER,
    format_unanswered_guards,
    retrieve_unanswered_guards,
)
from researchwiki.wiki.store import WikiStore
from researchwiki.wiki.verification import VerificationSettings, token_similarity

QUESTION = "agent 记忆方案对比"
PLAN_TEXT = "- 任务一：检索主流方案\n- 任务二：蒸馏笔记\n"
REPORT_TEXT = "## 研究报告\n\n本轮证据不足，无法给出结论。\n"
# 相似问题对（verification 尺度实测 0.8227 ≥ guard_similarity 0.4）与不相似对（0.1549）
SIMILAR_QUESTION = "主流 agent 记忆方案对比有哪些"
DISSIMILAR_QUESTION = "量子纠缠的贝尔不等式实验结论"
EMPTY_DISTILL_JSON = json.dumps({"notes": [], "conflicts": []}, ensure_ascii=False)
ONE_NOTE_DISTILL_JSON = json.dumps(
    {
        "notes": [
            {
                "text": "agent 记忆方案对比：共 5 个方案支持长期记忆。",
                "entities": [],
                "confidence": "high",
            }
        ],
        "conflicts": [],
    },
    ensure_ascii=False,
)


# ---- 脚手架（与 tests/test_memory_update.py 同构）---------------------------


def make_store(tmp_path: Path) -> WikiStore:
    return WikiStore(tmp_path / "wiki-data")


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


def zero_output_turns() -> list[list[StreamEvent]]:
    """零产出 run：无工具调用（fresh 来源 0）、蒸馏 0 条笔记 → no_notes 路径。"""
    return [
        text_turn(PLAN_TEXT),
        text_turn("没有检索到可用来源，研究结束。"),
        text_turn(EMPTY_DISTILL_JSON),
        text_turn(REPORT_TEXT),
    ]


def one_note_fresh_turns() -> list[list[StreamEvent]]:
    """fresh 来源不足 run：1 次 web_search（max_results=2 → fresh 2 篇）、1 条笔记，
    配 [retrieval.update].min_fresh_sources=3 → below_min_fresh 路径。"""
    return [
        text_turn(PLAN_TEXT),
        tool_turn([call("web_search", {"query": "agent 记忆", "max_results": 2})]),
        text_turn("已检索到部分方案，研究结束。"),
        text_turn(ONE_NOTE_DISTILL_JSON),
        text_turn(REPORT_TEXT),
    ]


def make_loop(tmp_path: Path, *, turns: list[list[StreamEvent]], **kwargs) -> AgentLoop:
    return AgentLoop(
        QUESTION,
        router=FakeRouter(ScriptedProvider(turns, model="mock-strong")),
        llm_config={"strong": {"base_url": "https://mock"}, "cheap": {"base_url": ""}},
        accountant=TokenAccountant(tmp_path / "tokens.jsonl"),
        search_provider=MockSearch(),
        wiki_root=tmp_path / "wiki-data",
        run_dir=tmp_path / "run",
        embedding=MockEmbeddingProvider(dim=512),
        wiki_config={"fts_tokenizer": "trigram"},
        **kwargs,
    )


def read_state(tmp_path: Path) -> str:
    return (tmp_path / "run" / "state.md").read_text(encoding="utf-8")


def ledger_notes(tmp_path: Path) -> list:
    return iter_active_unanswered(WikiStore(tmp_path / "wiki-data"))


def load_unanswered_json(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "run" / "unanswered.json").read_text(encoding="utf-8"))


def seed_ledger(store: WikiStore, *, question: str = QUESTION, trace_id: str = "t-old") -> None:
    record_unanswered(
        store,
        question=question,
        reason=REASON_NO_NOTES,
        trace_id=trace_id,
        sources_used=0,
        notes_created=0,
        asked_at="2026-10-11T08:00:00+00:00",
    )


# ---- ① record_unanswered：台账笔记形态 --------------------------------------


def test_record_unanswered_writes_experience_note(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    note = record_unanswered(
        store,
        question="RQ028 的原问题文本",
        reason=REASON_NO_NOTES,
        trace_id="20261011T-abc",
        sources_used=0,
        notes_created=0,
        asked_at="2026-10-11T08:00:00+00:00",
    )
    loaded = store.get_note(note.id)
    assert loaded is not None
    # kind=experience、active、importance 0.8（反幻觉护栏是高价值记忆）
    assert loaded.kind == "experience"
    assert loaded.status == "active"
    assert loaded.importance == LEDGER_IMPORTANCE == 0.8
    # frontmatter extra 承载结构化字段（契约字段名逐字；trace_id 是 NoteMeta 的
    # 类型化来源追溯字段，round-trip 后从 meta 读——见 NoteMeta.from_dict）
    assert loaded.meta.extra["ledger"] == LEDGER_MARKER == "unanswered_question"
    assert loaded.meta.extra["reason"] == "no_notes"
    assert loaded.meta.trace_id == "20261011T-abc"
    assert loaded.meta.extra["sources_used"] == 0
    assert loaded.meta.extra["notes_created"] == 0
    # 正文含问题原文、原因与判定日期（契约措辞）
    assert "RQ028 的原问题文本" in loaded.body
    assert "no_notes" in loaded.body
    assert "2026-10-11" in loaded.body
    assert "supersede" in loaded.body  # 处置说明保留历史


def test_record_unanswered_below_min_fresh_body_counts(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    note = record_unanswered(
        store,
        question=QUESTION,
        reason=REASON_BELOW_MIN_FRESH,
        trace_id="t2",
        sources_used=2,
        notes_created=1,
    )
    loaded = store.get_note(note.id)
    assert loaded is not None
    assert "2 个来源、1 条笔记" in loaded.body
    assert "below_min_fresh" in loaded.body
    assert loaded.meta.extra["reason"] == REASON_BELOW_MIN_FRESH


# ---- ② 同 trace_id 幂等 ------------------------------------------------------


def test_record_unanswered_same_trace_id_is_idempotent(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    first = record_unanswered(
        store, question=QUESTION, reason=REASON_NO_NOTES, trace_id="t1",
        sources_used=0, notes_created=0,
    )
    second = record_unanswered(
        store, question=SIMILAR_QUESTION, reason=REASON_BELOW_MIN_FRESH, trace_id="t1",
        sources_used=1, notes_created=0,
    )
    assert second.id == first.id  # 不新增
    notes = iter_active_unanswered(store)
    assert len(notes) == 1
    # 首条不被覆写
    assert notes[0].meta.extra["reason"] == REASON_NO_NOTES
    assert notes[0].meta.extra["sources_used"] == 0


# ---- ③ iter_active_unanswered 排除 superseded / tombstone -------------------


def test_iter_active_unanswered_excludes_superseded_and_tombstone(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    active = record_unanswered(
        store, question="问题一", reason=REASON_NO_NOTES, trace_id="t1",
        sources_used=0, notes_created=0,
    )
    superseded = record_unanswered(
        store, question="问题二", reason=REASON_BELOW_MIN_FRESH, trace_id="t2",
        sources_used=0, notes_created=2,
    )
    tombstoned = record_unanswered(
        store, question="问题三", reason=REASON_NO_NOTES, trace_id="t3",
        sources_used=0, notes_created=0,
    )
    # supersede / tombstone 都走 meta.replace + save_meta（save_note 会静默丢字段）
    meta = replace(superseded.meta, status="superseded", superseded_by=active.id)
    store.save_meta(meta, superseded.body)
    meta = replace(tombstoned.meta, tombstone=True)
    store.save_meta(meta, tombstoned.body)
    remaining = iter_active_unanswered(store)
    assert [n.id for n in remaining] == [active.id]


# ---- 判定与配置（确定性单元）-------------------------------------------------


def test_decide_unanswered_reason_truth_table() -> None:
    # 零产出 → no_notes（零产出优先于 below_min_fresh）
    assert decide_unanswered_reason(
        notes_created=0, fresh_sources=0, min_fresh_sources=0
    ) == REASON_NO_NOTES
    assert decide_unanswered_reason(
        notes_created=0, fresh_sources=0, min_fresh_sources=3
    ) == REASON_NO_NOTES
    # fresh 来源数低于模式下限 → below_min_fresh（有笔记、无/少来源）
    assert decide_unanswered_reason(
        notes_created=1, fresh_sources=0, min_fresh_sources=3
    ) == REASON_BELOW_MIN_FRESH
    assert decide_unanswered_reason(
        notes_created=2, fresh_sources=2, min_fresh_sources=3
    ) == REASON_BELOW_MIN_FRESH
    # 有来源或达模式要求 → 不记台账
    assert decide_unanswered_reason(notes_created=0, fresh_sources=2, min_fresh_sources=0) is None
    assert decide_unanswered_reason(notes_created=1, fresh_sources=1, min_fresh_sources=1) is None


def test_unanswered_settings_parsing() -> None:
    # 段缺失 = 不接线
    assert unanswered_settings(None) is None
    # 全默认（接线但未覆盖任何键）
    settings = unanswered_settings({})
    assert (settings.enabled, settings.dry_run) == (True, False)
    assert (settings.guard_similarity, settings.guard_top_k) == (0.4, 2)
    # 显式值
    settings = unanswered_settings(
        {"enabled": True, "dry_run": False, "guard_similarity": 0.4, "guard_top_k": 2}
    )
    assert (settings.enabled, settings.dry_run) == (True, False)
    assert (settings.guard_similarity, settings.guard_top_k) == (0.4, 2)
    # enabled=false 逃生阀照常解析（接线层据此零动作）
    assert unanswered_settings({"enabled": False}).enabled is False
    # 非法值宽容回退默认
    settings = unanswered_settings({"guard_similarity": "abc", "guard_top_k": "x"})
    assert (settings.guard_similarity, settings.guard_top_k) == (0.4, 2)
    # 越界夹取：比率到 [0,1]，条数下限 1
    settings = unanswered_settings({"guard_similarity": 1.5, "guard_top_k": 0})
    assert (settings.guard_similarity, settings.guard_top_k) == (1.0, 1)


def test_question_similarity_reuses_verification_scale() -> None:
    assert question_similarity(QUESTION, QUESTION) == pytest.approx(1.0)
    # 同一把尺：与 verification.token_similarity（默认 128 维）逐值一致
    assert question_similarity(QUESTION, SIMILAR_QUESTION) == pytest.approx(
        token_similarity(QUESTION, SIMILAR_QUESTION, settings=VerificationSettings())
    )
    assert question_similarity(QUESTION, SIMILAR_QUESTION) >= 0.4
    assert question_similarity(QUESTION, DISSIMILAR_QUESTION) < 0.4
    # 空串（零向量）→ 0.0
    assert question_similarity("", QUESTION) == 0.0


def test_parse_unanswered_entry_reads_contract_body(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    note = record_unanswered(
        store, question="RQ028 的原问题文本", reason=REASON_NO_NOTES, trace_id="t1",
        sources_used=0, notes_created=0, asked_at="2026-10-11T08:00:00+00:00",
    )
    loaded = store.get_note(note.id)
    assert loaded is not None
    entry = parse_unanswered_entry(loaded)
    assert entry is not None
    assert entry.question == "RQ028 的原问题文本"
    assert entry.asked_date == "2026-10-11"
    assert entry.reason == REASON_NO_NOTES
    assert entry.note_id == note.id


def test_parse_unanswered_entry_returns_none_for_malformed_body(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    # 非台账正文（即使被误标 ledger 标记）解析不出条目 → 护栏侧跳过
    note = store.save_note(
        "随手写的笔记，不是台账。",
        note_id="N-0001",
        title="普通",
        kind="experience",
        extra={"ledger": LEDGER_MARKER},
    )
    assert parse_unanswered_entry(note) is None


# ---- ④ AgentLoop 收尾触发路径 ------------------------------------------------


def test_loop_zero_output_run_records_no_notes_ledger(tmp_path: Path) -> None:
    loop = make_loop(tmp_path, turns=zero_output_turns(), unanswered_config={"enabled": True})
    events = list(loop.events())
    assert events[-1]["type"] == "finish"
    notes = ledger_notes(tmp_path)
    assert len(notes) == 1
    note = notes[0]
    assert note.kind == "experience" and note.status == "active"
    assert note.importance == 0.8
    assert note.meta.extra["ledger"] == LEDGER_MARKER
    assert note.meta.extra["reason"] == REASON_NO_NOTES
    assert note.meta.trace_id == loop.trace_id
    assert note.meta.extra["sources_used"] == 0
    assert note.meta.extra["notes_created"] == 0
    assert QUESTION in note.body
    payload = load_unanswered_json(tmp_path)
    assert payload["reason"] == REASON_NO_NOTES
    assert payload["recorded"] is True
    assert payload["note_id"] == note.id
    assert payload["fresh_source_count"] == 0
    assert payload["notes_created"] == 0
    assert payload["enabled"] is True and payload["dry_run"] is False
    # state.md 一行（run_dir/*.json + state.md 一行的产物契约）
    assert f"- 未答问题台账：{REASON_NO_NOTES}（{note.id}）" in read_state(tmp_path)


def test_loop_below_min_fresh_run_records_ledger(tmp_path: Path) -> None:
    loop = make_loop(
        tmp_path,
        turns=one_note_fresh_turns(),
        unanswered_config={"enabled": True},
        # update 模式要求 3 个 fresh 来源，本轮只搜到 2 个 → below_min_fresh
        retrieval_config={
            "enabled": True,
            "forced_mode": "update",
            "update": {"min_fresh_sources": 3},
        },
    )
    events = list(loop.events())
    assert events[-1]["type"] == "finish"
    payload = load_unanswered_json(tmp_path)
    assert payload["reason"] == REASON_BELOW_MIN_FRESH
    assert payload["mode"] == "update"
    assert payload["min_fresh_sources"] == 3
    assert 0 < payload["fresh_source_count"] < 3
    assert payload["notes_created"] == 1
    assert payload["recorded"] is True
    notes = ledger_notes(tmp_path)
    assert len(notes) == 1
    assert notes[0].meta.extra["reason"] == REASON_BELOW_MIN_FRESH
    assert "below_min_fresh" in notes[0].body


def test_loop_below_min_fresh_falls_back_to_deep_section(tmp_path: Path) -> None:
    """mode 不可得（retrieval 判定关闭）时 below_min_fresh 按 [retrieval.deep] 段。"""
    loop = make_loop(
        tmp_path,
        turns=one_note_fresh_turns(),
        unanswered_config={"enabled": True},
        retrieval_config={"enabled": False, "deep": {"min_fresh_sources": 3}},
    )
    events = list(loop.events())
    assert events[-1]["type"] == "finish"
    payload = load_unanswered_json(tmp_path)
    assert payload["mode"] is None  # 模式不可得
    assert payload["min_fresh_sources"] == 3  # deep 段兜底
    assert payload["reason"] == REASON_BELOW_MIN_FRESH
    assert payload["fresh_source_count"] == 2
    assert payload["notes_created"] == 1


def test_loop_wired_but_not_triggered(tmp_path: Path) -> None:
    """接线但判定不触发（有来源、达模式要求）：零台账，json 记 reason=null。"""
    loop = make_loop(tmp_path, turns=one_note_fresh_turns(), unanswered_config={"enabled": True})
    events = list(loop.events())
    assert events[-1]["type"] == "finish"
    payload = load_unanswered_json(tmp_path)
    assert payload["reason"] is None
    assert payload["recorded"] is False
    assert payload["note_id"] is None
    assert payload["notes_created"] == 1
    assert payload["fresh_source_count"] >= 1
    assert ledger_notes(tmp_path) == []
    # 未触发也留一行判定输入（state.md 一行的产物契约），判定可复算
    assert "未答问题台账：不触发" in read_state(tmp_path)


def test_loop_without_unanswered_config_does_nothing(tmp_path: Path) -> None:
    """段缺失 = 不接线：零动作零留痕（无台账、无 json、state.md 无行）。"""
    loop = make_loop(tmp_path, turns=zero_output_turns())  # 不传 unanswered_config
    events = list(loop.events())
    assert events[-1]["type"] == "finish"
    assert ledger_notes(tmp_path) == []
    assert not (tmp_path / "run" / "unanswered.json").exists()
    assert "未答问题台账" not in read_state(tmp_path)


def test_loop_disabled_unanswered_does_nothing(tmp_path: Path) -> None:
    """enabled=false 逃生阀：完全禁用（零动作零留痕）。"""
    loop = make_loop(
        tmp_path, turns=zero_output_turns(), unanswered_config={"enabled": False}
    )
    events = list(loop.events())
    assert events[-1]["type"] == "finish"
    assert ledger_notes(tmp_path) == []
    assert not (tmp_path / "run" / "unanswered.json").exists()
    assert "未答问题台账" not in read_state(tmp_path)


def test_loop_dry_run_records_trace_without_note(tmp_path: Path) -> None:
    """dry_run 影子模式：判定与留痕照跑，不写台账笔记。"""
    loop = make_loop(
        tmp_path,
        turns=zero_output_turns(),
        unanswered_config={"enabled": True, "dry_run": True},
    )
    events = list(loop.events())
    assert events[-1]["type"] == "finish"
    assert ledger_notes(tmp_path) == []  # 不写台账笔记
    payload = load_unanswered_json(tmp_path)
    assert payload["dry_run"] is True
    assert payload["reason"] == REASON_NO_NOTES
    assert payload["recorded"] is False
    assert payload["note_id"] is None
    state = read_state(tmp_path)
    assert "未答问题台账" in state and "dry-run" in state


# ---- ⑤ 护栏注入：相似注入 / 不相似不注入 / dry_run 不注入 --------------------


def test_retrieve_unanswered_guards_matches_similar_question_only(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    seed_ledger(store)
    # 同义改写（实测 sim 0.8227 ≥ 0.4）→ 注入
    guards = retrieve_unanswered_guards(SIMILAR_QUESTION, store)
    assert len(guards) == 1
    guard = guards[0]
    assert guard.question == QUESTION
    assert guard.asked_date == "2026-10-11"
    assert guard.reason == REASON_NO_NOTES
    assert guard.similarity >= 0.4
    # 不相似问题（实测 sim 0.1549 < 0.4）→ 不注入
    assert retrieve_unanswered_guards(DISSIMILAR_QUESTION, store) == []


def test_retrieve_unanswered_guards_top_k_and_ordering(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    record_unanswered(
        store, question=QUESTION, reason=REASON_NO_NOTES, trace_id="t1",
        sources_used=0, notes_created=0, asked_at="2026-10-11T08:00:00+00:00",
    )
    record_unanswered(
        store, question=SIMILAR_QUESTION, reason=REASON_BELOW_MIN_FRESH, trace_id="t2",
        sources_used=0, notes_created=2, asked_at="2026-10-10T08:00:00+00:00",
    )
    # guard_top_k=1：只留相似度最高的一条（原题 sim 1.0 > 改写 0.8227）
    top1 = retrieve_unanswered_guards(QUESTION, store, top_k=1)
    assert len(top1) == 1
    assert top1[0].question == QUESTION
    # guard_top_k=2：两条都注入，按相似度降序
    top2 = retrieve_unanswered_guards(QUESTION, store, top_k=2)
    assert [g.question for g in top2] == [QUESTION, SIMILAR_QUESTION]
    assert top2[0].similarity >= top2[1].similarity


def test_format_unanswered_guards_block_format(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    seed_ledger(store)
    guards = retrieve_unanswered_guards(QUESTION, store)
    block = format_unanswered_guards(guards)
    assert block.startswith(UNANSWERED_GUARD_HEADER + "\n")
    assert UNANSWERED_GUARD_HEADER == "### 未答问题护栏（不可编造）"
    assert "问题「agent 记忆方案对比」于 2026-10-11 被判定证据不足（原因 no_notes）。" in block
    assert '必须明说"证据不足"，禁止编造。' in block
    assert format_unanswered_guards([]) == ""


def test_loop_injects_guard_for_similar_ledger_question(tmp_path: Path) -> None:
    store = WikiStore(tmp_path / "wiki-data")
    seed_ledger(store)
    loop = make_loop(tmp_path, turns=zero_output_turns(), unanswered_config={"enabled": True})
    events = list(loop.events())
    assert events[-1]["type"] == "finish"
    plan_message = loop.provider.calls[0][0].content  # plan 步骤的 user 消息
    assert UNANSWERED_GUARD_HEADER in plan_message
    assert f"问题「{QUESTION}」" in plan_message
    assert '必须明说"证据不足"，禁止编造。' in plan_message


def test_loop_dry_run_ledger_not_injected(tmp_path: Path) -> None:
    """dry_run 影子模式：台账照常在库里，但本轮不注入护栏。"""
    store = WikiStore(tmp_path / "wiki-data")
    seed_ledger(store)
    loop = make_loop(
        tmp_path,
        turns=zero_output_turns(),
        unanswered_config={"enabled": True, "dry_run": True},
    )
    events = list(loop.events())
    assert events[-1]["type"] == "finish"
    plan_message = loop.provider.calls[0][0].content
    assert "未答问题护栏" not in plan_message


def test_loop_without_unanswered_config_no_guard(tmp_path: Path) -> None:
    """段缺失 = 不接线：Prior 零护栏（库里有台账也不注入）。"""
    store = WikiStore(tmp_path / "wiki-data")
    seed_ledger(store)
    loop = make_loop(tmp_path, turns=zero_output_turns())  # 不传 unanswered_config
    events = list(loop.events())
    assert events[-1]["type"] == "finish"
    plan_message = loop.provider.calls[0][0].content
    assert "未答问题护栏" not in plan_message


# ---- ⑥ 护栏块在 Prior 块之前且不占 top_k 配额 --------------------------------


def test_guard_block_precedes_prior_and_does_not_consume_top_k(tmp_path: Path) -> None:
    store = WikiStore(tmp_path / "wiki-data")
    # 两条与问题匹配的普通笔记 + 一条台账：Prior 配额 top_k=1
    store.save_note(
        "agent 记忆方案对比：Letta 用后台 subagent 整理记忆。",
        note_id="N-0001",
        title="Letta（历史）",
    )
    store.save_note(
        "agent 记忆方案对比：MemGPT 用分页管理上下文。",
        note_id="N-0002",
        title="MemGPT（历史）",
    )
    seed_ledger(store)
    loop = make_loop(
        tmp_path,
        turns=zero_output_turns(),
        unanswered_config={"enabled": True},
        prior_config={"top_k": 1},
    )
    events = list(loop.events())
    assert events[-1]["type"] == "finish"
    plan_message = loop.provider.calls[0][0].content
    # 顺序：护栏块在前，Prior 标签行在后
    guard_pos = plan_message.index(UNANSWERED_GUARD_HEADER)
    prior_pos = plan_message.index(PRIOR_CONTEXT_LABEL)
    assert guard_pos < prior_pos
    # 配额：Prior 块恰好 1 条（护栏不占 [prior].top_k、也不把注入条数撑大）
    assert len(loop.prior_context.hits) == 1
    assert plan_message.count("### [") == 1  # Prior 条目格式 "### [n] ..."
    assert plan_message.count(UNANSWERED_GUARD_HEADER) == 1
