"""Distiller 单测：ScriptedProvider 脚本 + tmp_path 隔离、零真实网络。

覆盖：抽取 JSON 解析（缺字段/类型漂移容错）、非法 JSON 与超量降级、来源 URL 越界丢弃、
实体注册、冲突落台账；页面生成的稳定 ID 双链与断言行 ID 标注（含模型偷懒时的确定性兜底）。
"""

import json
from pathlib import Path

import pytest

from researchwiki.llm.provider import ScriptedProvider, StreamEvent
from researchwiki.wiki.distiller import (
    Distiller,
    annotate_citations,
    ensure_entity_links,
    skeleton_page,
)
from researchwiki.wiki.entities import Entity, EntityRegistry
from researchwiki.wiki.frontmatter import SourceRef
from researchwiki.wiki.lint import NOTE_ID_RE, iter_assertion_lines
from researchwiki.wiki.store import WikiStore

SOURCE_A = SourceRef(url="https://example.com/a", content_hash="a" * 64)
SOURCE_B = SourceRef(url="https://example.com/b", content_hash="b" * 64)


def text_turn(text: str) -> list[StreamEvent]:
    return [StreamEvent(type="text_delta", delta=text)]


def payload_turn(payload: object) -> list[StreamEvent]:
    return text_turn(json.dumps(payload, ensure_ascii=False))


@pytest.fixture()
def store(tmp_path: Path) -> WikiStore:
    return WikiStore(tmp_path / "wiki-data")


def make_distiller(store: WikiStore, turns, **kwargs) -> tuple[Distiller, ScriptedProvider]:
    provider = ScriptedProvider(turns, model="mock-cheap")
    distiller = Distiller(
        store,
        provider=provider,
        entity_registry=EntityRegistry(store.root),
        trace_id="run-1",
        **kwargs,
    )
    return distiller, provider


# ---- extract_notes：解析与降级 ----------------------------------------------


def test_extract_notes_parses_items_and_tolerates_drift(store: WikiStore):
    payload = {
        "notes": [
            {
                "text": "GLM-5.3 支持 1M 上下文窗口",
                "entities": ["GLM-5.3"],
                "confidence": "high",
                "volatility": "volatile",
                "source_urls": [SOURCE_A.url, "https://example.com/outside"],
            },
            {"text": "KV-cache 到 32 万 token 仍有效", "entities": "Prompt Caching"},
            {"text": "缺少 confidence 与 entities 的笔记"},
        ],
        "conflicts": [{"summary": "上下文窗口 128K vs 1M", "action": "核实官方文档"}],
    }
    distiller, provider = make_distiller(store, [payload_turn(payload)])
    notes = distiller.extract_notes(
        "# 报告\n\nGLM-5.3 支持 1M 上下文窗口。\n",
        question="GLM-5.3 窗口多大",
        sources=[SOURCE_A, SOURCE_B],
        max_notes=8,
    )

    assert [note.text for note in notes] == [
        "GLM-5.3 支持 1M 上下文窗口",
        "KV-cache 到 32 万 token 仍有效",
        "缺少 confidence 与 entities 的笔记",
    ]
    first = notes[0]
    assert first.entities == ["GLM-5.3"] and first.confidence == "high"
    assert first.volatility == "volatile"
    # 来源列表之外的 URL 被丢弃，保留的 URL 带上 content_hash
    assert first.source_urls == [SOURCE_A.url]
    assert first.source_refs == [SOURCE_A]
    # 字段漂移：字符串 entities → 单元素列表；缺字段走默认值
    assert notes[1].entities == ["Prompt Caching"]
    assert notes[1].confidence == "medium" and notes[1].volatility == "stable"
    assert notes[2].confidence == "medium" and notes[2].entities == []
    # 越界 URL 记录了降级原因
    assert any("来源列表之外" in item for item in distiller.degradations)
    # 抽取阶段注册实体（别名归一先行），但不写笔记
    assert EntityRegistry(store.root).resolve("GLM-5.3") is not None
    assert store.list_notes(status=None) == []
    # 冲突落台账 + 供事件层使用
    assert [c.id for c in distiller.last_conflicts] == ["C-0001"]
    assert distiller.last_conflicts[0].summary == "上下文窗口 128K vs 1M"
    assert store.get_conflict("C-0001").claim_b == {"action": "核实官方文档"}
    # 记账：一次抽取一次调用
    assert [len(call) for call in provider.calls] == [1]


def test_extract_notes_degrades_on_invalid_json(store: WikiStore):
    distiller, _ = make_distiller(store, [text_turn("抱歉，我无法输出 JSON")])
    assert distiller.extract_notes("报告正文", question="q") == []
    assert distiller.last_conflicts == []
    assert any("不是合法 JSON" in item for item in distiller.degradations)


def test_extract_notes_degrades_on_missing_notes_array(store: WikiStore):
    distiller, _ = make_distiller(store, [payload_turn({"conflicts": []})])
    assert distiller.extract_notes("报告正文") == []
    assert any("缺少 notes 数组" in item for item in distiller.degradations)


def test_extract_notes_truncates_over_max_notes(store: WikiStore):
    payload = {
        "notes": [{"text": f"事实 {i}", "entities": ["E"]} for i in range(4)],
        "conflicts": [],
    }
    distiller, _ = make_distiller(store, [payload_turn(payload)])
    notes = distiller.extract_notes("报告", max_notes=2)
    assert [note.text for note in notes] == ["事实 0", "事实 1"]
    assert any("超过上限 2" in item for item in distiller.degradations)


def test_extract_notes_drops_malformed_items_and_truncates_long_text(store: WikiStore):
    long_text = "长" * 300
    payload = {
        "notes": ["不是对象", {"entities": ["E"]}, {"text": long_text, "entities": ["E"]}],
        "conflicts": [{"summary": ""}, "垃圾", {"summary": "有效矛盾"}],
    }
    distiller, _ = make_distiller(store, [payload_turn(payload)])
    notes = distiller.extract_notes("报告")
    assert len(notes) == 1
    assert len(notes[0].text) == 240
    assert any("截断到 240" in item for item in distiller.degradations)
    assert any("结构非法" in item for item in distiller.degradations)
    # 空 summary / 非对象的冲突被丢弃，只登记有效的那条
    assert [c.summary for c in distiller.last_conflicts] == ["有效矛盾"]


def test_extract_notes_degrades_on_llm_failure(store: WikiStore):
    """LLM 调用本身抛异常也必须降级为空列表（不打断整轮研究），原因写进 degradations。"""

    class BoomProvider:
        model = "boom"

        def stream(self, messages, *, system=None, tools=None):
            raise RuntimeError("上游 502")

    distiller = Distiller(store, provider=BoomProvider())
    assert distiller.extract_notes("报告正文", question="q") == []
    assert distiller.last_conflicts == []
    assert any("RuntimeError: 上游 502" in item for item in distiller.degradations)


def test_extract_notes_skips_llm_when_report_empty(store: WikiStore):
    distiller, provider = make_distiller(store, [text_turn("不该被调用")])
    assert distiller.extract_notes("   ", question="q") == []
    assert provider.calls == []
    assert distiller.degradations == ["蒸馏输入为空，跳过抽取"]


def test_extract_notes_truncates_oversized_report(store: WikiStore):
    payload = {"notes": [{"text": "事实", "entities": ["E"]}], "conflicts": []}
    distiller, provider = make_distiller(store, [payload_turn(payload)], max_report_chars=100)
    distiller.extract_notes("报" * 300)
    assert any("截断到 100" in item for item in distiller.degradations)
    assert len(provider.calls[0][0].content) < 400  # 提示词里只带截断后的报告


def test_extract_notes_conflict_cap(store: WikiStore):
    payload = {
        "notes": [],
        "conflicts": [{"summary": f"矛盾 {i}", "action": "核实"} for i in range(7)],
    }
    distiller, _ = make_distiller(store, [payload_turn(payload)])
    distiller.extract_notes("报告")
    assert len(distiller.last_conflicts) == 5
    assert any("矛盾点超过 5 条" in item for item in distiller.degradations)


# ---- build_pages：双链与行内引用 --------------------------------------------


def test_build_pages_adds_stable_links_and_inline_citations(store: WikiStore):
    note_a = store.save_note("GLM-5.3 支持 1M 上下文窗口", entities=["GLM-5.3"])
    note_b = store.save_note("KV-cache 到 32 万 token 仍有效", entities=["GLM-5.3"])
    # 模型偷懒：正文没有行内标注、也没写双链
    page_body = "# GLM-5.3\n\n## 能力\n\n支持 1M 上下文窗口。\n- 缓存到 32 万 token 仍有效\n"
    distiller, _ = make_distiller(store, [text_turn(page_body)])
    drafts = distiller.build_pages([note_a, note_b])

    assert len(drafts) == 1
    draft = drafts[0]
    assert draft.slug == "glm-5-3" and draft.title == "GLM-5.3"
    assert draft.note_ids == [note_a.id, note_b.id]
    # 稳定 ID 双链（链接用 ID、显示名可读）
    assert "[[entity:glm-5-3|GLM-5.3]]" in draft.body
    # 每个断言行都有笔记 ID 标注（模型没标时由确定性兜底补上）
    assertions = list(iter_assertion_lines(draft.body))
    assert assertions
    assert all(NOTE_ID_RE.search(line) for _, line in assertions)
    assert "（N-0001）" in draft.body and "（N-0002）" in draft.body


def test_build_pages_groups_by_entity_and_caps(store: WikiStore):
    for text in ["A 事实一", "A 事实二", "B 事实一"]:
        entities = ["实体A"] if text.startswith("A") else ["实体B"]
        store.save_note(text, entities=entities)
    notes = store.list_notes()
    page_body = "# 页面\n\n## 小节\n\n一条断言。\n"
    distiller, provider = make_distiller(store, [text_turn(page_body)])
    drafts = distiller.build_pages(notes, max_pages=1)
    # 按笔记数降序取前 max_pages；实体A 有 2 条笔记
    assert [draft.slug for draft in drafts] == ["实体a"]
    assert len(provider.calls) == 1


def test_build_pages_uses_skeleton_when_model_output_unusable(store: WikiStore):
    note = store.save_note("一条断言", entities=["实体A"])
    distiller, _ = make_distiller(store, [text_turn("嗯")])
    draft = distiller.build_pages([note])[0]
    assert draft.body.startswith("# 实体A")
    assert f"一条断言（{note.id}）" in draft.body
    assert any("骨架页" in item for item in distiller.degradations)


def test_build_pages_dedupes_same_note_and_skips_empty(store: WikiStore):
    note = store.save_note("断言", entities=["实体A"])
    page = "# 实体A\n\n## 小节\n\n断言（N-0001）。\n"
    distiller, provider = make_distiller(store, [text_turn(page)])
    drafts = distiller.build_pages([note, note])
    assert len(drafts) == 1
    assert drafts[0].note_ids == [note.id]
    assert len(provider.calls) == 1
    # 无实体/无正文的空候选不产生页面，也不炸
    assert distiller.build_pages([]) == []


def test_page_keeps_registered_links_and_degrades_broken_ones(store: WikiStore):
    EntityRegistry(store.root).get_or_create("实体B")  # 已注册 → 双链保留
    store.save_note("断言", entities=["实体A"])
    page = (
        "```markdown\n# 实体A\n\n## 小节\n\n"
        "断言（N-0001），参见 [[entity:实体b|实体B]] 与 [[entity:幽灵|幽灵]]。\n```\n"
    )
    distiller, _ = make_distiller(store, [text_turn(page)])
    draft = distiller.build_pages(store.list_notes())[0]
    assert "```" not in draft.body
    assert draft.body.startswith("# 实体A")
    assert "[[entity:实体b|实体B]]" in draft.body  # 已注册：保留模型写的双链
    assert "[[entity:幽灵" not in draft.body  # 未注册：降级为纯文本，页面不产生断链
    assert "幽灵" in draft.body
    assert any("未注册实体双链已降级" in item for item in distiller.degradations)
    assert draft.entity_ids == ["实体a"]


# ---- 纯函数 ------------------------------------------------------------------


def test_annotate_citations_skips_headings_links_and_cited_lines():
    body = (
        "# 标题\n\n"
        "- [[entity:letta|Letta]]\n\n"
        "Letta 用后台 sub agent 整理记忆。\n"
        "已经标过的断言（N-0002）。\n"
    )
    out = annotate_citations(body, {"N-0001": "Letta 用后台 sub agent 整理记忆"})
    lines = out.splitlines()
    assert lines[0] == "# 标题"
    assert lines[2] == "- [[entity:letta|Letta]]"  # 纯双链导航行不加标注
    assert lines[4] == "Letta 用后台 sub agent 整理记忆。（N-0001）"
    assert lines[5] == "已经标过的断言（N-0002）。"
    # 空 note_texts 原样返回
    assert annotate_citations(body, {}) == body


def test_annotate_citations_falls_back_to_first_note():
    out = annotate_citations("# T\n\n完全不相干的一句话。\n", {"N-0007": "另外一条断言"})
    assert "（N-0007）" in out


def test_ensure_entity_links_appends_or_fills_section():
    entity = Entity(id="glm-5-3", name="GLM-5.3")
    other = Entity(id="letta", name="Letta")
    assert ensure_entity_links("# T\n\n[[entity:glm-5-3|GLM-5.3]] 说明（N-1）。\n", [entity]) == (
        "# T\n\n[[entity:glm-5-3|GLM-5.3]] 说明（N-1）。\n"
    )
    filled = ensure_entity_links("# T\n\n断言（N-1）。\n\n## 相关实体\n", [entity, other])
    assert filled.index("## 相关实体") < filled.index("[[entity:glm-5-3|GLM-5.3]]")
    assert "[[entity:letta|Letta]]" in filled
    appended = ensure_entity_links("# T\n\n断言（N-1）。\n", [other])
    assert appended.endswith("- [[entity:letta|Letta]]\n")


def test_skeleton_page_lists_notes_with_ids():
    body = skeleton_page(Entity(id="e", name="实体"), {"N-0001": "断言一", "N-0002": "断言二"})
    assert body.splitlines()[0] == "# 实体"
    assert "- 断言一（N-0001）" in body and "- 断言二（N-0002）" in body
