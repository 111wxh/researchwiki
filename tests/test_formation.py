"""Memory Formation MVP（RQ1：什么时候应该记住）的策略与接入测试。

- 策略判定：好候选持久化、四类拒绝（短文本 / 无来源 / 低分 / 近乎重复）各自
  reason 引用信号值；confidence 三档与 kind 恒 knowledge；
- 权重表：构造仅差单一信号的候选对，importance 差值与 formation.py docstring
  的权重表一致（常量即表）；
- from_config：缺省回退、段/完整 config 双口径、非法值宽容、enabled 逃生阀；
- loop 接入：参考 tests/test_loop_prior.py 的脚本化 provider 模式——门控计数、
  state.md「记忆形成」行、importance/extra 注写、近乎重复端到端拒绝、以及
  enabled=false / 未传配置两条照旧入库路径（行为与既有基线逐字段一致）。
"""

import json
from pathlib import Path
from typing import Any

import pytest

from researchwiki.llm.accounting import TokenAccountant
from researchwiki.llm.provider import ScriptedProvider, StreamEvent, TokenUsage
from researchwiki.loop.agent_loop import AgentLoop
from researchwiki.tools import MockSearch
from researchwiki.wiki.embeddings import MockEmbeddingProvider
from researchwiki.wiki.formation import (
    BASE_IMPORTANCE,
    BODY_FULL_CHARS,
    WEIGHT_BODY_FULL,
    WEIGHT_BODY_MIN,
    WEIGHT_PER_ENTITY,
    WEIGHT_SOURCE,
    WEIGHT_SPECIFICS,
    FormationSettings,
    evaluate_candidate,
    from_config,
)
from researchwiki.wiki.frontmatter import SourceRef
from researchwiki.wiki.store import WikiStore

QUESTION = "agent 记忆方案对比"
PLAN_TEXT = "- 任务一：检索主流方案\n- 任务二：蒸馏笔记\n"
REPORT_TEXT = "## 研究报告\n\nLetta 方案[1] 值得优先试点。\n"

# 好候选：有来源 + 具体事实要素（数字/拉丁词元）+ 实体 + 正文过完整档（>= 80 字）
GOOD_BODY = (
    "Letta 于 2025 年推出 sleep-time compute：由后台 subagent 在会话空闲期整理记忆，"
    "把维护成本移出交互窗口，相关配置与成本见官方文档。"
)
SOURCE = SourceRef(url="https://example.com/letta", content_hash="")
# 模糊候选：纯中文、无数字/拉丁/引号、无来源、无实体（20 <= 长度 < 80）
VAGUE_BODY = "这个方案整体感觉还可以接受，运行起来也算稳定可靠，没有更多值得展开的细节内容。"

assert len(GOOD_BODY) >= BODY_FULL_CHARS, "夹具必须落在正文完整档（权重表前提）"
assert 20 <= len(VAGUE_BODY) < BODY_FULL_CHARS, "夹具必须落在正文达标档"


# ---- 策略判定 ---------------------------------------------------------------


def test_good_candidate_persists_with_expected_importance() -> None:
    decision = evaluate_candidate(
        GOOD_BODY, entities=("Letta", "MemGPT"), sources=(SOURCE,)
    )
    # 权重表：0.15 基础 + 0.25 来源 + 0.25 具体要素 + 2 实体 0.20 + 正文完整档 0.10
    assert decision.persist is True
    assert decision.importance == pytest.approx(0.95)
    assert decision.confidence == "high"
    assert decision.kind == "knowledge"  # MVP 恒 knowledge（user/experience 走显式 store）
    assert decision.signals.body_chars == len(GOOD_BODY)
    assert decision.signals.entity_count == 2
    assert decision.signals.has_source is True
    assert decision.signals.has_specifics is True
    # reason 引用具体信号值
    assert f"正文 {len(GOOD_BODY)} 字" in decision.reason
    assert "实体 2 个" in decision.reason
    assert "来源有" in decision.reason and "具体要素有" in decision.reason
    assert "0.95" in decision.reason


def test_similarity_none_skips_near_duplicate_rule() -> None:
    decision = evaluate_candidate(GOOD_BODY, entities=("Letta",), sources=(SOURCE,))
    assert decision.persist is True  # 无嵌入上下文（None）不做近乎重复判定
    assert decision.signals.similarity_max is None


def test_reject_short_body_reason_cites_signal() -> None:
    decision = evaluate_candidate("太短了")
    assert decision.persist is False
    assert decision.signals.body_chars == 3
    assert "正文 3 字" in decision.reason
    assert "阈值 20" in decision.reason


def test_reject_knowledge_without_source_reason_cites_signal() -> None:
    decision = evaluate_candidate(GOOD_BODY, entities=("Letta",))
    assert decision.persist is False
    assert decision.signals.has_source is False
    assert "无来源" in decision.reason
    assert "has_source=False" in decision.reason
    assert "require_source_for_knowledge=True" in decision.reason


def test_reject_low_importance_reason_cites_signal() -> None:
    settings = FormationSettings(require_source_for_knowledge=False)
    decision = evaluate_candidate(VAGUE_BODY, settings=settings)
    # 0.15 基础 + 0.05 正文达标档 = 0.20 < 0.3
    assert decision.persist is False
    assert decision.importance == pytest.approx(0.20)
    assert "importance 0.20" in decision.reason
    assert "0.30" in decision.reason


def test_reject_near_duplicate_reason_cites_signal() -> None:
    decision = evaluate_candidate(
        GOOD_BODY,
        entities=("Letta",),
        sources=(SOURCE,),
        similarity_max=0.9623,
    )
    assert decision.persist is False
    assert "相似度 0.962" in decision.reason
    assert "0.95" in decision.reason
    assert "近乎重复" in decision.reason


def test_confidence_levels() -> None:
    high = evaluate_candidate(GOOD_BODY, entities=("Letta",), sources=(SOURCE,))
    assert high.confidence == "high"  # 有来源 + 具体要素
    medium = evaluate_candidate(VAGUE_BODY, sources=(SOURCE,))
    assert medium.persist is True  # 0.15 + 0.25 + 0.05 = 0.45 >= 0.3
    assert medium.confidence == "medium"  # 有来源、无具体要素
    low = evaluate_candidate(
        VAGUE_BODY, settings=FormationSettings(require_source_for_knowledge=False)
    )
    assert low.confidence == "low"  # 无来源


# ---- 权重表：仅差单一信号的候选对，差值 == 表中权重 -----------------------------


def test_weight_table_source_contribution() -> None:
    settings = FormationSettings(require_source_for_knowledge=False)
    with_source = evaluate_candidate(
        GOOD_BODY, entities=("Letta",), sources=(SOURCE,), settings=settings
    )
    without = evaluate_candidate(GOOD_BODY, entities=("Letta",), settings=settings)
    assert with_source.importance - without.importance == pytest.approx(WEIGHT_SOURCE)


def test_weight_table_specifics_contribution() -> None:
    filler = (
        "这个方案的整体思路是先把任务拆分再逐个完成，遇到问题就回退到上一个稳定状态"
        "重新尝试，直到所有步骤都能顺利通过为止，并且全程保持记录便于事后回顾与总结，"
        "也方便团队其他成员随时查阅。"
    )
    assert len(filler) >= BODY_FULL_CHARS  # 两条同档，唯一差异是具体事实要素
    with_specifics = evaluate_candidate(GOOD_BODY, entities=("Letta",), sources=(SOURCE,))
    without = evaluate_candidate(filler, entities=("Letta",), sources=(SOURCE,))
    assert without.signals.has_specifics is False
    assert with_specifics.importance - without.importance == pytest.approx(
        WEIGHT_SPECIFICS
    )


def test_weight_table_entity_contribution_and_cap() -> None:
    one = evaluate_candidate(GOOD_BODY, entities=("A",), sources=(SOURCE,))
    two = evaluate_candidate(GOOD_BODY, entities=("A", "B"), sources=(SOURCE,))
    assert two.importance - one.importance == pytest.approx(WEIGHT_PER_ENTITY)
    # 封顶：3 个已到 +0.30 上限，第 4 个不再加分
    three = evaluate_candidate(
        GOOD_BODY, entities=("A", "B", "C"), sources=(SOURCE,)
    )
    four = evaluate_candidate(
        GOOD_BODY, entities=("A", "B", "C", "D"), sources=(SOURCE,)
    )
    assert three.importance == four.importance == pytest.approx(1.0)  # 1.05 夹到 1.0


def test_weight_table_body_tier_contribution() -> None:
    settings = FormationSettings(require_source_for_knowledge=False)
    short_ok = evaluate_candidate(VAGUE_BODY, settings=settings)
    long_text = VAGUE_BODY + (
        "补充的说明文字依然只有中文汉字，用来把正文长度推过完整档位的门槛线，"
        "仍旧不包含任何其他类型的要素，也不引入数字与字母。"
    )
    assert len(long_text) >= BODY_FULL_CHARS
    full = evaluate_candidate(long_text, settings=settings)
    assert full.signals.has_specifics is False  # 两条都无具体要素，唯一差异是正文档位
    assert full.importance - short_ok.importance == pytest.approx(
        WEIGHT_BODY_FULL - WEIGHT_BODY_MIN
    )


def test_weight_table_base_and_clamp() -> None:
    bare = evaluate_candidate(
        VAGUE_BODY, settings=FormationSettings(require_source_for_knowledge=False)
    )
    assert bare.importance == pytest.approx(BASE_IMPORTANCE + WEIGHT_BODY_MIN)
    everything = evaluate_candidate(
        GOOD_BODY,
        entities=("A", "B", "C", "D", "E"),
        sources=(SOURCE,),
    )
    assert everything.importance == pytest.approx(1.0)  # 0.15+0.25+0.25+0.30+0.10 夹顶


# ---- from_config -------------------------------------------------------------


def test_from_config_defaults_on_none_and_empty() -> None:
    for config in (None, {}, {"formation": {}}):
        settings = from_config(config)
        assert settings == FormationSettings()  # enabled=True + 全部默认阈值


def test_from_config_reads_section_and_full_config() -> None:
    section = {
        "enabled": False,
        "min_body_chars": 40,
        "require_source_for_knowledge": False,
        "min_importance_to_persist": 0.5,
        "near_duplicate_similarity": 0.9,
    }
    assert from_config(section) == FormationSettings(**section)
    assert from_config({"formation": section}) == FormationSettings(**section)


def test_from_config_tolerates_illegal_values() -> None:
    settings = from_config(
        {
            "min_body_chars": "abc",  # 非法 → 回默认 20
            "min_importance_to_persist": "not-a-number",  # → 0.3
            "near_duplicate_similarity": 7,  # 越界 → 夹到 1.0
            "require_source_for_knowledge": 0,  # bool 化
        }
    )
    assert settings.min_body_chars == 20
    assert settings.min_importance_to_persist == pytest.approx(0.3)
    assert settings.near_duplicate_similarity == pytest.approx(1.0)
    assert settings.require_source_for_knowledge is False
    # 负数夹到 0
    assert from_config({"min_body_chars": -5}).min_body_chars == 0


# ---- loop 接入（脚本化 provider 模式，与 tests/test_loop_prior.py 同构）--------


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


# 两条蒸馏候选：好候选（来源 URL 在 MockSearch 夹具池内）+ 注定被拒的短文本无来源候选
GOOD_NOTE_TEXT = GOOD_BODY
DOOMED_NOTE_TEXT = "记忆需要整理。"
DISTILL_JSON = json.dumps(
    {
        "notes": [
            {
                "text": GOOD_NOTE_TEXT,
                "entities": ["Letta"],
                "confidence": "high",
                "source_urls": ["https://github.com/letta-ai/letta"],
            },
            {"text": DOOMED_NOTE_TEXT, "entities": ["Letta"], "confidence": "low"},
        ],
        "conflicts": [],
    },
    ensure_ascii=False,
)


def run_turns() -> list[list[StreamEvent]]:
    return [
        text_turn(PLAN_TEXT),
        tool_turn([call("web_search", {"query": "agent 记忆"}, id="c1")]),
        text_turn("已检索到主流方案，研究完成。"),
        text_turn(DISTILL_JSON),
        text_turn(REPORT_TEXT),
    ]


def make_loop(
    tmp_path: Path,
    strong_turns: list[list[StreamEvent]],
    formation_config: dict[str, Any] | None = None,
) -> AgentLoop:
    return AgentLoop(
        QUESTION,
        router=FakeRouter(ScriptedProvider(strong_turns, model="mock-strong")),
        llm_config={"strong": {"base_url": "https://mock"}, "cheap": {"base_url": ""}},
        accountant=TokenAccountant(tmp_path / "tokens.jsonl"),
        search_provider=MockSearch(),
        wiki_root=tmp_path / "wiki-data",
        run_dir=tmp_path / "run",
        embedding=MockEmbeddingProvider(dim=512),
        wiki_config={"fts_tokenizer": "trigram"},
        formation_config=formation_config,
    )


def load_metrics(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "run" / "run-metrics.json").read_text(encoding="utf-8"))


def read_state(tmp_path: Path) -> str:
    return (tmp_path / "run" / "state.md").read_text(encoding="utf-8")


def test_loop_gates_candidates_counts_and_annotates_notes(tmp_path: Path) -> None:
    loop = make_loop(tmp_path, run_turns(), formation_config={"enabled": True})
    events = list(loop.events())

    notes = [e for e in events if e["type"] == "data-note"]
    assert len(notes) == 1  # 短文本无来源候选被拒，不入库不发事件
    assert notes[0]["data"]["text"].strip() == GOOD_NOTE_TEXT
    assert loop.formation_stats == {"candidates": 2, "persisted": 1, "rejected": 1}
    assert load_metrics(tmp_path)["notes_created"] == 1
    assert "记忆形成：候选 2，入库 1，拒绝 1" in read_state(tmp_path)

    # 判定结果写进笔记：importance/kind 进 frontmatter，理由进 extra
    saved = WikiStore(tmp_path / "wiki-data").list_notes()
    assert len(saved) == 1
    assert saved[0].importance == pytest.approx(0.85)  # 权重表：0.15+0.25+0.25+0.10+0.10
    assert saved[0].kind == "knowledge"
    assert saved[0].meta.extra["formation_confidence"] == "high"
    assert "正文" in saved[0].meta.extra["formation_reason"]


def test_loop_rejects_near_duplicate_of_existing_note(tmp_path: Path) -> None:
    store = WikiStore(tmp_path / "wiki-data")
    store.save_note(GOOD_NOTE_TEXT, entities=["Letta"], note_id="N-0001")
    loop = make_loop(tmp_path, run_turns(), formation_config={"enabled": True})
    events = list(loop.events())

    # 好候选与既有 N-0001 正文相同 → 相似度 1.0 ≥ 0.95 拒绝；短文本候选拒于规则 1
    assert loop.formation_stats == {"candidates": 2, "persisted": 0, "rejected": 2}
    assert [e for e in events if e["type"] == "data-note"] == []
    assert [n.id for n in store.list_notes()] == ["N-0001"]  # 无差别堆积被防住
    assert "记忆形成：候选 2，入库 0，拒绝 2" in read_state(tmp_path)


def test_loop_escape_valve_disabled_ingests_everything(tmp_path: Path) -> None:
    loop = make_loop(tmp_path, run_turns(), formation_config={"enabled": False})
    events = list(loop.events())

    notes = [e for e in events if e["type"] == "data-note"]
    assert len(notes) == 2  # 全部照旧入库
    assert loop.formation_stats == {"candidates": 2, "persisted": 2, "rejected": 0}
    assert load_metrics(tmp_path)["notes_created"] == 2
    saved = WikiStore(tmp_path / "wiki-data").list_notes()
    assert len(saved) == 2
    assert all(note.importance is None for note in saved)  # 不做任何 formation 注写
    assert all("formation_reason" not in note.meta.extra for note in saved)
    assert "记忆形成：候选 2，入库 2，拒绝 0" in read_state(tmp_path)


def test_loop_without_formation_config_keeps_legacy_behavior(tmp_path: Path) -> None:
    """未传 [formation] 段（脚本 / 既有调用方）：跳过判定，行为与基线逐字段一致。"""
    loop = make_loop(tmp_path, run_turns())
    list(loop.events())

    assert loop.formation_settings.enabled is False
    assert loop.formation_stats == {"candidates": 2, "persisted": 2, "rejected": 0}
    saved = WikiStore(tmp_path / "wiki-data").list_notes()
    assert len(saved) == 2
    assert all(note.importance is None for note in saved)
