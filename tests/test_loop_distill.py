"""loop × 蒸馏入库集成：事件协议同构 + data-note 的 id 是规范 ID + 去重合并真的落盘。

脚手架与 tests/test_loop.py 同构（ScriptedProvider + MockSearch + tmp_path），
但独立成文件：既有测试的断言一字不改，这里只加"蒸馏 → 入库去重"这一层的集成断言。
"""

import json
from collections.abc import Iterator
from pathlib import Path

from researchwiki.llm.accounting import TokenAccountant
from researchwiki.llm.provider import Message, ScriptedProvider, StreamEvent, TokenUsage
from researchwiki.loop.agent_loop import AgentLoop
from researchwiki.tools import MockSearch
from researchwiki.wiki.entities import EntityRegistry
from researchwiki.wiki.store import WikiStore

PLAN_TEXT = "- 任务一：检索主流方案\n- 任务二：蒸馏笔记\n"
REPORT_TEXT = "## 研究报告\n\nMem0 在写入前做相似度查重[1]。\n"
DUP_DISTILL = json.dumps(
    {
        "notes": [
            {"text": "Mem0 在写入记忆前做相似度查重", "entities": ["Mem0"], "confidence": "high"},
            {"text": "Mem0 在写入记忆前做相似度查重", "entities": ["Mem0"], "confidence": "high"},
        ],
        "conflicts": [],
    },
    ensure_ascii=False,
)


def text_turn(text: str) -> list[StreamEvent]:
    return [StreamEvent(type="text_delta", delta=text)]


def tool_turn(calls: list[dict]) -> list[StreamEvent]:
    return [StreamEvent(type="tool_calls", tool_calls=calls, usage=TokenUsage(10, 5))]


def call(name: str, arguments: dict, *, id: str = "c1") -> dict:
    return {
        "id": id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
    }


class FakeRouter:
    """按档位返回预置 Provider（未配置的档位被请求即失败，防止测试里静默串线）。"""

    def __init__(self, strong: ScriptedProvider, cheap: ScriptedProvider | None = None) -> None:
        self._providers = {"strong": strong}
        if cheap is not None:
            self._providers["cheap"] = cheap

    def get(self, tier: str) -> ScriptedProvider:
        return self._providers[tier]


def make_loop(
    tmp_path: Path,
    *,
    strong_turns: list[list[StreamEvent]],
    cheap_turns: list[list[StreamEvent]] | None = None,
    **kwargs,
) -> AgentLoop:
    strong = ScriptedProvider(strong_turns, model="mock-strong")
    cheap = None
    llm_config = {"strong": {"base_url": "https://mock"}, "cheap": {"base_url": ""}}
    if cheap_turns is not None:
        cheap = ScriptedProvider(cheap_turns, tier="cheap", model="mock-cheap")
        llm_config = {
            "strong": {"base_url": "https://mock"},
            "cheap": {"base_url": "https://mock-cheap"},
        }
    return AgentLoop(
        "agent 记忆方案对比",
        router=FakeRouter(strong, cheap),
        llm_config=llm_config,
        accountant=TokenAccountant(tmp_path / "tokens.jsonl"),
        search_provider=MockSearch(),
        wiki_root=tmp_path / "wiki-data",
        run_dir=tmp_path / "run",
        **kwargs,
    )


def accountant_steps(tmp_path: Path) -> list[str]:
    lines = (tmp_path / "tokens.jsonl").read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line)["step"] for line in lines]


def note_events(events: list[dict]) -> Iterator[dict]:
    return (e for e in events if e["type"] == "data-note")


def test_loop_notes_are_ingested_with_dedup_and_canonical_ids(tmp_path: Path):
    loop = make_loop(
        tmp_path,
        strong_turns=[
            text_turn(PLAN_TEXT),
            text_turn("研究完成。"),
            text_turn(DUP_DISTILL),
            text_turn(REPORT_TEXT),
        ],
    )
    events = list(loop.events())
    types = [e["type"] for e in events]

    # 事件协议不变：start/finish 包裹，笔记事件仍在报告正文之前（与 mock 演示同构）
    assert types[0] == "start" and types[-1] == "finish"
    assert types.index("data-note") < types.index("text-start") < types.index("finish")
    assert "".join(e["delta"] for e in events if e["type"] == "text-delta") == REPORT_TEXT

    notes = list(note_events(events))
    assert [e["id"] for e in notes] == ["note-1", "note-2"]
    # 第二条与第一条是同一事实 → 命中合并，两条事件都用规范 ID 指向同一条笔记
    assert [e["data"]["id"] for e in notes] == ["N-0001", "N-0001"]
    assert notes[0]["data"]["entities"] == ["Mem0"]
    assert notes[0]["data"]["confidence"] == "high"
    assert notes[0]["data"]["text"] == "Mem0 在写入记忆前做相似度查重"

    store = WikiStore(tmp_path / "wiki-data")
    canonical = store.get_note("N-0001")
    trail = store.get_note("N-0002")
    assert canonical.status == "active"
    assert canonical.meta.extra["merged_from"] == ["N-0002"]
    assert trail.status == "merged" and trail.meta.redirect_to == "N-0001"
    assert trail.body == "Mem0 在写入记忆前做相似度查重"
    assert store.follow_redirect("N-0002").id == "N-0001"
    # 笔记 frontmatter 是 wiki 层的完整格式（loop 与 wiki 互读兼容）
    raw = (tmp_path / "wiki-data" / "notes" / "N-0001.md").read_text(encoding="utf-8")
    assert "id: N-0001" in raw and "status: active" in raw and "volatility: stable" in raw
    # 实体注册表已建条目（后续页面双链可用）
    assert EntityRegistry(tmp_path / "wiki-data").resolve("Mem0") is not None
    # 蒸馏仍是一次 LLM 调用、一个记账步骤（序列与既有测试一致）
    assert accountant_steps(tmp_path) == ["plan", "act:1", "distill", "report"]
    # 任务卡统计：新增 1 条 · 合并 1 条
    distill_done = [
        e["data"]["detail"]
        for e in events
        if e["type"] == "data-task"
        and e["data"]["title"] == "蒸馏原子笔记 → wiki"
        and e["data"]["status"] == "done"
    ]
    assert distill_done == ["新增 1 条笔记 · 合并 1 条 · 0 条冲突"]


def test_loop_conflicts_come_from_the_distill_ledger(tmp_path: Path):
    distill = json.dumps(
        {
            "notes": [],
            "conflicts": [{"summary": "窗口 128K vs 1M", "action": "核实官方文档"}],
        },
        ensure_ascii=False,
    )
    loop = make_loop(
        tmp_path,
        strong_turns=[
            text_turn(PLAN_TEXT),
            text_turn("研究完成。"),
            text_turn(distill),
            text_turn(REPORT_TEXT),
        ],
    )
    events = list(loop.events())
    conflicts = [e for e in events if e["type"] == "data-conflict"]
    assert conflicts[0]["id"] == "conflict-1"
    assert conflicts[0]["data"] == {
        "id": "C-0001",
        "summary": "窗口 128K vs 1M",
        "action": "核实官方文档",
    }
    # 冲突同时落台账（下轮研究注入用）
    ledger = WikiStore(tmp_path / "wiki-data").get_conflict("C-0001")
    assert ledger is not None and ledger.status == "open"
    assert ledger.claim_a["question"] == "agent 记忆方案对比"


def test_loop_subagent_notes_are_ingested_with_source_refs(tmp_path: Path):
    sub_json = json.dumps(
        {
            "findings": "子问题结论：70% 阈值触发压缩。",
            "notes": [
                {"text": "压缩阈值 70% 触发", "entities": ["Compaction"], "confidence": "medium"}
            ],
            "sources": [{"url": "https://example.com/sub", "title": "子来源"}],
        },
        ensure_ascii=False,
    )
    main_distill = json.dumps(
        {"notes": [{"text": "Mem0 写入前查重", "entities": ["Mem0"]}], "conflicts": []},
        ensure_ascii=False,
    )
    loop = make_loop(
        tmp_path,
        strong_turns=[
            text_turn(PLAN_TEXT),
            tool_turn([call("dispatch_research", {"topic": "上下文压缩"}, id="c1")]),
            text_turn("汇总完毕。"),
            text_turn(REPORT_TEXT),
        ],
        cheap_turns=[
            tool_turn([call("web_search", {"query": "上下文压缩"}, id="s1")]),
            text_turn(sub_json),
            text_turn(main_distill),
        ],
    )
    events = list(loop.events())
    notes = list(note_events(events))

    # 子 agent 笔记先入库、主循环笔记接续（编号连续），两条都是新建
    assert [e["data"]["text"] for e in notes] == ["压缩阈值 70% 触发", "Mem0 写入前查重"]
    assert [e["data"]["id"] for e in notes] == ["N-0001", "N-0002"]
    store = WikiStore(tmp_path / "wiki-data")
    sub_note = store.get_note("N-0001")
    assert sub_note.entities == ["Compaction"] and sub_note.confidence == "medium"
    # 子 agent 的来源集合写进它笔记的 frontmatter.sources（粗粒度溯源）
    assert [s.url for s in sub_note.meta.sources] == ["https://example.com/sub"]
    assert store.get_note("N-0002").meta.confidence == "medium"


def test_loop_build_pages_optin_writes_linked_pages(tmp_path: Path):
    page_body = "# Mem0\n\n## 写入流程\n\n写入前先做相似度查重。\n"
    loop = make_loop(
        tmp_path,
        strong_turns=[
            text_turn(PLAN_TEXT),
            text_turn("研究完成。"),
            text_turn(DUP_DISTILL),
            text_turn(page_body),
            text_turn(REPORT_TEXT),
        ],
        build_pages=True,
    )
    events = list(loop.events())

    store = WikiStore(tmp_path / "wiki-data")
    page = store.get_page("mem0")
    assert page is not None and page.title == "Mem0"
    assert "[[entity:mem0|Mem0]]" in page.body  # 稳定 ID 双链
    assert "N-0001" in page.body  # 断言行内标注规范笔记 ID
    # 页面聚合是显式开启的额外一步（默认关闭，保持 run 内调用序列不变）
    assert accountant_steps(tmp_path) == ["plan", "act:1", "distill", "distill:pages", "report"]
    assert [e["type"] for e in events][-1] == "finish"


def test_loop_does_not_merge_notes_without_entity_overlap(tmp_path: Path):
    """默认门槛 dedup_entity_overlap=1：实体不同就不合并（宁可多写一条）。"""
    distill = json.dumps(
        {
            "notes": [
                {"text": "同一条事实的重复表述", "entities": ["实体A"]},
                {"text": "同一条事实的重复表述", "entities": ["实体B"]},
            ],
            "conflicts": [],
        },
        ensure_ascii=False,
    )
    loop = make_loop(
        tmp_path,
        strong_turns=[
            text_turn(PLAN_TEXT),
            text_turn("研究完成。"),
            text_turn(distill),
            text_turn(REPORT_TEXT),
        ],
    )
    events = list(loop.events())
    assert [e["data"]["id"] for e in note_events(events)] == ["N-0001", "N-0002"]
    store = WikiStore(tmp_path / "wiki-data")
    assert [n.id for n in store.list_notes()] == ["N-0001", "N-0002"]


def test_loop_distill_keeps_single_message_call_and_usage_accounting(tmp_path: Path):
    """蒸馏调用形态不变：单条 user 消息（含来源列表 + 研究记录）；token 计入主循环状态。"""
    loop = make_loop(
        tmp_path,
        strong_turns=[
            text_turn(PLAN_TEXT),
            tool_turn([call("web_search", {"query": "agent 记忆"}, id="c1")]),
            text_turn("研究完成。"),
            text_turn(DUP_DISTILL),
            text_turn(REPORT_TEXT),
        ],
    )
    events = list(loop.events())
    types = [e["type"] for e in events]
    assert types[0] == "start" and types[-1] == "finish"

    provider: ScriptedProvider = loop.distill_provider  # type: ignore[assignment]
    # 调用序列 plan → act:1 → act:2 → distill → report：第 4 次调用是蒸馏
    distill_call = provider.calls[3]
    assert [m.role for m in distill_call] == ["user"]
    assert len(distill_call) == 1
    assert isinstance(distill_call[0], Message)
    content = distill_call[0].content
    assert "来源列表：" in content and "https://github.com/letta-ai/letta" in content
    assert "研究总结：\n研究完成。" in content  # 研究总结进了蒸馏输入
    assert "研究过程记录（工具结果摘编）" in content and "输出严格 JSON" in content

    # 记账行数 == 主循环调用次数，input_tokens 覆盖全部调用（预算判断不漏账）
    rows = [
        json.loads(line)
        for line in (tmp_path / "tokens.jsonl").read_text(encoding="utf-8").strip().splitlines()
    ]
    assert [row["step"] for row in rows] == ["plan", "act:1", "act:2", "distill", "report"]
    assert loop.input_tokens == sum(row["input_tokens"] for row in rows)
    assert loop.notes_written == 2
