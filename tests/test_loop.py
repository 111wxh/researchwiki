"""loop 层单测：ScriptedProvider 多轮脚本 + MockSearch + tmp_path 沙箱。

零真实网络、零真实等待：LLM 全部走脚本化 Provider，搜索走 MockSearch 夹具，
fetch 走 httpx.MockTransport，落盘全部重定向到 tmp_path。
"""

import json
from pathlib import Path

import httpx

from researchwiki.llm.accounting import TokenAccountant
from researchwiki.llm.openai_provider import OpenAICompatibleProvider
from researchwiki.llm.provider import Message, ScriptedProvider, StreamEvent, TokenUsage
from researchwiki.loop.agent_loop import (
    AgentLoop,
    RunContext,
    SourcePool,
    build_registry,
    build_sub_registry,
    pick_tier,
)
from researchwiki.loop.notes import NoteStore, scan_max_note_id
from researchwiki.loop.registry import Tool, ToolRegistry
from researchwiki.loop.subagent import ResearchSubagent, extract_json
from researchwiki.tools import MockSearch

# ---- 脚手架 ----------------------------------------------------------------


def text_turn(text: str) -> list[StreamEvent]:
    """一轮纯文本响应（usage 由 ScriptedProvider 自动补齐）。"""
    return [StreamEvent(type="text_delta", delta=text)]


def tool_turn(calls: list[dict], usage: TokenUsage | None = None) -> list[StreamEvent]:
    events = [StreamEvent(type="tool_calls", tool_calls=calls)]
    if usage is not None:
        events.append(StreamEvent(type="usage", usage=usage))
    return events


def call(name: str, arguments: dict, *, id: str = "call-1") -> dict:
    """OpenAI 格式的单个工具调用（arguments 为 JSON 字符串）。"""
    return {
        "id": id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
    }


class FakeRouter:
    """按档位返回预置 Provider；未配置的档位被请求即失败（防止测试里静默串线）。"""

    def __init__(self, strong, cheap=None) -> None:
        self._providers = {"strong": strong}
        if cheap is not None:
            self._providers["cheap"] = cheap

    def get(self, tier):
        if tier not in self._providers:
            raise AssertionError(f"测试未配置 {tier} 档 Provider，却被请求了")
        return self._providers[tier]


PLAN_TEXT = "- 任务一：检索主流方案\n- 任务二：阅读核心来源\n- 任务三：蒸馏笔记\n"
REPORT_TEXT = "## 研究报告\n\nLetta 方案[1] 值得优先试点。\n"
DISTILL_JSON = json.dumps(
    {
        "notes": [
            {"text": "Letta 用后台 subagent 整理记忆", "entities": ["Letta"], "confidence": "high"}
        ],
        "conflicts": [{"summary": "上下文窗口 128K vs 200K", "action": "下轮优先核实"}],
    },
    ensure_ascii=False,
)


def make_loop(strong_turns, tmp_path, *, cheap_turns=None, llm_config=None, **kwargs):
    """构造带记账与 tmp 沙箱的 AgentLoop；返回 (loop, strong, cheap)。"""
    strong = ScriptedProvider(strong_turns, model="mock-strong")
    cheap = None
    if cheap_turns is not None:
        cheap = ScriptedProvider(cheap_turns, tier="cheap", model="mock-cheap")
    if llm_config is None:
        llm_config = (
            {"strong": {"base_url": "https://mock"}, "cheap": {"base_url": "https://mock-cheap"}}
            if cheap_turns is not None
            else {"strong": {"base_url": "https://mock"}, "cheap": {"base_url": ""}}
        )
    loop = AgentLoop(
        "agent 记忆方案对比",
        router=FakeRouter(strong, cheap),
        llm_config=llm_config,
        accountant=TokenAccountant(tmp_path / "tokens.jsonl"),
        search_provider=MockSearch(),
        wiki_root=tmp_path / "wiki-data",
        run_dir=tmp_path / "run",
        **kwargs,
    )
    return loop, strong, cheap


def accountant_steps(tmp_path) -> list[dict]:
    lines = (tmp_path / "tokens.jsonl").read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines]


# ---- pick_tier / 来源池 / 笔记编号 -------------------------------------------


def test_pick_tier_prefers_configured_cheap_else_strong():
    both = {"strong": {"base_url": "https://s"}, "cheap": {"base_url": "https://c"}}
    assert pick_tier(both, "cheap") == "cheap"
    # cheap 未配置但 strong 已配置 → 复用 strong
    assert pick_tier({"strong": {"base_url": "https://s"}, "cheap": {}}, "cheap") == "strong"
    # 两档皆空（纯 mock）或无配置 → 沿用 preferred
    assert pick_tier({"strong": {}, "cheap": {}}, "cheap") == "cheap"
    assert pick_tier(None, "cheap") == "cheap"


def test_source_pool_dedupes_and_numbers():
    pool = SourcePool()
    assert pool.add("https://a", "A") is True
    assert pool.add("https://a", "A2") is False  # 同 URL 去重
    assert pool.add("", "空") is False
    pool.add("https://b")
    assert pool.numbered() == [
        {"n": 1, "url": "https://a", "title": "A"},
        {"n": 2, "url": "https://b", "title": "https://b"},
    ]


def test_note_store_persists_markdown_and_continues_ids(tmp_path):
    notes_dir = tmp_path / "notes"
    store = NoteStore(notes_dir, trace_id="trace01")
    saved = store.save({"text": "一条事实", "entities": ["实体A"], "confidence": "high"})
    assert saved == {
        "id": "N-0001",
        "text": "一条事实",
        "entities": ["实体A"],
        "confidence": "high",
    }
    raw = (notes_dir / "N-0001.md").read_text(encoding="utf-8")
    assert "一条事实" in raw and "id: N-0001" in raw and "trace_id: trace01" in raw
    assert scan_max_note_id(notes_dir) == 1
    # 新实例接续现有最大编号
    assert NoteStore(notes_dir).next_id() == "N-0002"
    assert scan_max_note_id(tmp_path / "missing") == 0


# ---- 工具注册表 --------------------------------------------------------------


def make_ctx(tmp_path: Path, *, transport=None) -> RunContext:
    return RunContext(
        search_provider=MockSearch(),
        sources_dir=tmp_path / "sources",
        wiki_root=tmp_path / "wiki-data",
        fetch_transport=transport,
    )


def test_registry_registers_all_tools_with_openai_schemas(tmp_path):
    registry = build_registry(make_ctx(tmp_path))
    assert set(registry.names()) == {
        "web_search",
        "fetch_url",
        "fs_read",
        "fs_write",
        "fs_list",
        "dispatch_research",
    }
    schemas = registry.schemas()
    ws = next(s for s in schemas if s["function"]["name"] == "web_search")
    assert ws["type"] == "function"
    assert ws["function"]["parameters"]["required"] == ["query"]
    # 子 agent 工具集不含 dispatch_research（防递归）
    sub = build_sub_registry(make_ctx(tmp_path))
    assert "dispatch_research" not in sub.names()
    assert {"web_search", "fetch_url", "fs_read", "fs_write", "fs_list"} <= set(sub.names())


def test_registry_dispatches_web_search_and_fills_source_pool(tmp_path):
    search = MockSearch()
    ctx = RunContext(search_provider=search, sources_dir=tmp_path / "s", wiki_root=tmp_path / "w")
    registry = build_registry(ctx)
    payload = json.loads(
        registry.dispatch("web_search", json.dumps({"query": "agent 记忆", "max_results": 2}))
    )
    assert len(payload["results"]) == 2
    assert payload["results"][0]["url"].startswith("http")
    assert search.calls == [("agent 记忆", 2)]
    assert [e["url"] for e in ctx.source_pool.entries] == [h["url"] for h in payload["results"]]


def test_registry_fs_tools_roundtrip_within_sandbox(tmp_path):
    registry = build_registry(make_ctx(tmp_path))
    written = json.loads(registry.dispatch("fs_write", {"path": "notes/a.md", "content": "hello"}))
    assert written == {"path": "notes/a.md", "bytes": 5}
    read = json.loads(registry.dispatch("fs_read", {"path": "notes/a.md"}))
    assert read["content"] == "hello"
    listed = json.loads(registry.dispatch("fs_list", {"path": "notes"}))
    assert listed["entries"] == [{"name": "a.md", "type": "file"}]


def test_registry_dispatch_errors_become_error_json(tmp_path):
    registry = build_registry(make_ctx(tmp_path))
    # 未知工具
    assert "error" in registry.dispatch("nope", "{}")
    # arguments 非法 JSON
    assert "error" in registry.dispatch("web_search", "{bad json")
    # 沙箱越界（SandboxError 被降级）
    escaped = json.loads(registry.dispatch("fs_read", json.dumps({"path": "../outside.txt"})))
    assert "error" in escaped and "sandbox" in escaped["error"]
    # 缺必要参数
    assert "error" in registry.dispatch("web_search", json.dumps({"max_results": 3}))


def test_registry_fetch_tool_pools_source_and_snapshots(tmp_path):
    html = (
        "<html><body><h1>Agent 记忆综述</h1>"
        "<p>智能体的长期记忆是核心议题。Letta 提出 sleep-time compute 思路，"
        "由后台子代理在会话空闲期整理记忆，把维护成本移出交互窗口，上下文保持精简，"
        "同时保留跨会话可复用的长期事实，这些内容足以通过正文提取阈值。</p></body></html>"
    )
    transport = httpx.MockTransport(lambda request: httpx.Response(200, html=html))
    ctx = make_ctx(tmp_path, transport=transport)
    registry = build_registry(ctx)
    payload = json.loads(registry.dispatch("fetch_url", {"url": "https://example.com/memo"}))
    assert payload["http_status"] == 200
    assert payload["final_url"] == "https://example.com/memo"
    assert "sleep-time compute" in payload["text"]
    assert ctx.source_pool.entries[0]["url"] == "https://example.com/memo"
    # 标题约定 = 正文第一个非空行截 80 字（不绑定 trafilatura 是否保留 h1）
    first_line = payload["text"].splitlines()[0].strip()
    assert ctx.source_pool.entries[0]["title"] == first_line[:80]
    # 快照已落盘 sources/{sha1(url)}/{content_hash}/
    url_dirs = list((tmp_path / "sources").iterdir())
    assert len(url_dirs) == 1 and (url_dirs[0] / payload["content_hash"] / "content.md").exists()


def test_registry_fetch_failure_returns_error_json(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "r.jina.ai":
            return httpx.Response(503)
        return httpx.Response(404)

    registry = build_registry(make_ctx(tmp_path, transport=httpx.MockTransport(handler)))
    payload = json.loads(registry.dispatch("fetch_url", {"url": "https://example.com/gone"}))
    assert "error" in payload and "FetchError" in payload["error"]


def test_registry_truncates_oversized_results():
    registry = ToolRegistry(max_result_chars=100)
    registry.register(
        Tool(
            name="big",
            description="",
            parameters={"type": "object", "properties": {}},
            handler=lambda args: "x" * 500,
        )
    )
    out = registry.dispatch("big", {})
    assert out.startswith("x" * 100)
    assert "已截断" in out and len(out) < 200


# ---- LLM 层：Message 新字段与 ScriptedProvider --------------------------------


def test_openai_body_serializes_optional_tool_fields():
    provider = OpenAICompatibleProvider(base_url="https://example.com", model="m", api_key="k")
    body = provider._build_body(
        [
            Message(role="assistant", content="", tool_calls=[{"id": "c1"}]),
            Message(role="tool", content="结果", tool_call_id="c1", name="web_search"),
            Message(role="user", content="普通消息"),
        ]
    )
    msgs = body["messages"]
    assert msgs[0]["tool_calls"] == [{"id": "c1"}]
    assert msgs[1]["role"] == "tool"
    assert msgs[1]["tool_call_id"] == "c1" and msgs[1]["name"] == "web_search"
    assert msgs[2] == {"role": "user", "content": "普通消息"}  # 旧格式不受影响


def test_scripted_provider_consumes_turns_in_order_and_appends_usage():
    provider = ScriptedProvider(
        [
            [StreamEvent(type="tool_calls", tool_calls=[{"id": "c1"}])],
            [StreamEvent(type="text_delta", delta="报告")],
        ],
        model="mock-strong",
    )
    first = list(provider.stream([Message(role="user", content="q")]))
    assert first[0].type == "tool_calls"
    assert first[-1].type == "usage"  # 轮次缺 usage 时自动补齐
    second = list(provider.stream([Message(role="user", content="q")]))
    assert "".join(e.delta for e in second if e.type == "text_delta") == "报告"
    assert len(provider.calls) == 2


def test_extract_json_tolerates_fences_and_prose():
    assert extract_json('说明\n```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('前缀 {"findings": "x", "n": {"deep": [1, 2]}} 后缀')["findings"] == "x"
    assert extract_json("完全没有 JSON") is None
    assert extract_json('{"bad": truncated') is None


# ---- AgentLoop：消息序列与工具回传 ---------------------------------------------


def test_loop_tool_calls_become_role_tool_messages(tmp_path):
    strong_turns = [
        text_turn(PLAN_TEXT),
        tool_turn([call("web_search", {"query": "agent 记忆"}, id="c1")]),
        text_turn("已检索到主流方案，研究完成。"),
        text_turn(DISTILL_JSON),
        text_turn(REPORT_TEXT),
    ]
    loop, strong, _ = make_loop(strong_turns, tmp_path)
    list(loop.events())  # 驱动完整 run，断言只关注 provider 侧的消息序列与记账

    # 调用序列：plan → act:1(tool) → act:2(text 收尾) → distill → report
    assert [len(c) for c in strong.calls] == [1, 2, 4, 1, 6]
    # act:1 请求：用户问题 + 助手计划（计划进入上下文）
    assert [m.role for m in strong.calls[1]] == ["user", "assistant"]
    assert strong.calls[1][1].content == PLAN_TEXT
    # act:2 请求：追加 assistant(tool_calls) + role="tool" 结果
    act2 = strong.calls[2]
    assert [m.role for m in act2] == ["user", "assistant", "assistant", "tool"]
    assert act2[2].tool_calls[0]["id"] == "c1"
    tool_msg = act2[3]
    assert tool_msg.role == "tool"
    assert tool_msg.tool_call_id == "c1" and tool_msg.name == "web_search"
    tool_payload = json.loads(tool_msg.content)
    assert tool_payload["query"] == "agent 记忆" and "results" in tool_payload
    # report 请求：完整历史 + 追加的撰写指令
    report_req = strong.calls[4]
    assert report_req[-1].role == "user" and "报告" in report_req[-1].content

    # 记账：每步一行
    rows = accountant_steps(tmp_path)
    assert [r["step"] for r in rows] == ["plan", "act:1", "act:2", "distill", "report"]
    assert all(r["model"] == "mock-strong" for r in rows)


def test_loop_event_protocol_is_frontend_homogeneous(tmp_path):
    strong_turns = [
        text_turn(PLAN_TEXT),
        tool_turn([call("web_search", {"query": "agent 记忆"}, id="c1")]),
        text_turn("已检索到主流方案，研究完成。"),
        text_turn(DISTILL_JSON),
        text_turn(REPORT_TEXT),
    ]
    loop, _, _ = make_loop(strong_turns, tmp_path)
    events = list(loop.events())
    types = [e["type"] for e in events]

    assert types[0] == "start" and types[-1] == "finish"
    # 思考块：规划 + 研究总结
    reasoning_ids = [e["id"] for e in events if e["type"] == "reasoning-start"]
    assert reasoning_ids == ["plan", "observe"]
    # 任务流：计划 done → 执行 running…done → 蒸馏 running/done，taskId 与事件 id 一致
    tasks = [e for e in events if e["type"] == "data-task"]
    assert [(t["id"], t["data"]["status"]) for t in tasks] == [
        ("task-1", "done"),
        ("task-2", "running"),
        ("task-2", "running"),
        ("task-2", "done"),
        ("task-3", "running"),
        ("task-3", "done"),
    ]
    assert all(t["data"]["taskId"] == t["id"] for t in tasks)
    # 笔记：事件 id 递增，data.id 从现有 wiki 最大编号 +1
    notes = [e for e in events if e["type"] == "data-note"]
    assert [e["id"] for e in notes] == ["note-1"]
    assert notes[0]["data"]["id"] == "N-0001"
    assert notes[0]["data"]["entities"] == ["Letta"]
    # 冲突
    conflicts = [e for e in events if e["type"] == "data-conflict"]
    assert conflicts[0]["data"] == {
        "id": "C-0001",
        "summary": "上下文窗口 128K vs 200K",
        "action": "下轮优先核实",
    }
    # 报告：单一 text 块，delta 拼出全文；来源在文本后、finish 前
    assert types.index("text-start") < types.index("source-url") < types.index("finish")
    assert "".join(e["delta"] for e in events if e["type"] == "text-delta") == REPORT_TEXT
    sources = [e for e in events if e["type"] == "source-url"]
    assert sources[0]["sourceId"] == "s1"
    assert sources[0]["url"] == "https://github.com/letta-ai/letta"  # MockSearch 首条夹具

    # state / plan / report 落盘
    run_dir = tmp_path / "run"
    assert PLAN_TEXT in (run_dir / "research-plan.md").read_text(encoding="utf-8")
    state = (run_dir / "state.md").read_text(encoding="utf-8")
    assert "done" in state and "Token" in state
    assert (run_dir / "report.md").read_text(encoding="utf-8").strip() == REPORT_TEXT.strip()
    # 笔记文件已写入 wiki
    assert (tmp_path / "wiki-data" / "notes" / "N-0001.md").exists()


def test_loop_note_ids_continue_from_existing_wiki(tmp_path):
    notes_dir = tmp_path / "wiki-data" / "notes"
    notes_dir.mkdir(parents=True)
    (notes_dir / "N-0007.md").write_text("旧笔记\n", encoding="utf-8")
    distill = json.dumps(
        {
            "notes": [
                {"text": "笔记一", "entities": [], "confidence": "high"},
                {"text": "笔记二", "entities": [], "confidence": "low"},
            ],
            "conflicts": [],
        },
        ensure_ascii=False,
    )
    strong_turns = [
        text_turn(PLAN_TEXT),
        text_turn("研究完成。"),
        text_turn(distill),
        text_turn(REPORT_TEXT),
    ]
    loop, _, _ = make_loop(strong_turns, tmp_path)
    events = list(loop.events())
    notes = [e for e in events if e["type"] == "data-note"]
    assert [e["data"]["id"] for e in notes] == ["N-0008", "N-0009"]
    assert [e["id"] for e in notes] == ["note-1", "note-2"]


# ---- 熔断 --------------------------------------------------------------------


def test_loop_max_steps_circuit_breaker_still_reports(tmp_path):
    strong_turns = [
        text_turn(PLAN_TEXT),
        tool_turn([call("web_search", {"query": "q"}, id="c1")]),
        text_turn(DISTILL_JSON),
        text_turn(REPORT_TEXT),
    ]
    loop, strong, _ = make_loop(strong_turns, tmp_path, max_steps=1)
    events = list(loop.events())
    types = [e["type"] for e in events]

    # 恰好 1 次 act 调用（plan / act:1 / distill / report）
    assert [r["step"] for r in accountant_steps(tmp_path)] == ["plan", "act:1", "distill", "report"]
    # 熔断后仍产出完整报告文本事件与 start/finish 包裹
    assert types[0] == "start" and types[-1] == "finish"
    assert "".join(e["delta"] for e in events if e["type"] == "text-delta") == REPORT_TEXT
    # 熔断可见：任务 done 详情与 state.md 都带标记
    act_done = [
        e
        for e in events
        if e["type"] == "data-task" and e["data"]["title"] == "执行研究（工具调用）"
        and e["data"]["status"] == "done"
    ]
    assert act_done and "熔断" in act_done[0]["data"]["detail"]
    assert "max_steps" in (tmp_path / "run" / "state.md").read_text(encoding="utf-8")


def test_loop_token_budget_circuit_breaker_skips_research(tmp_path):
    plan_turn = [
        StreamEvent(type="text_delta", delta=PLAN_TEXT),
        StreamEvent(type="usage", usage=TokenUsage(input_tokens=5_000, output_tokens=10)),
    ]
    strong_turns = [
        plan_turn,
        text_turn(DISTILL_JSON),
        text_turn(REPORT_TEXT),
    ]
    loop, strong, _ = make_loop(strong_turns, tmp_path, token_budget=5_000)
    events = list(loop.events())
    types = [e["type"] for e in events]

    # plan 烧完预算 → act 一步都没跑，直接收尾（distill + report）
    assert [r["step"] for r in accountant_steps(tmp_path)] == ["plan", "distill", "report"]
    assert len(strong.calls) == 3
    assert types[0] == "start" and types[-1] == "finish"
    assert "".join(e["delta"] for e in events if e["type"] == "text-delta") == REPORT_TEXT
    # "执行研究"任务从未出现
    assert not any(
        e["type"] == "data-task" and e["data"]["title"] == "执行研究（工具调用）" for e in events
    )
    assert "token_budget" in (tmp_path / "run" / "state.md").read_text(encoding="utf-8")


# ---- Research 子 agent ---------------------------------------------------------


def test_loop_dispatches_subagent_with_independent_context(tmp_path):
    sub_json = json.dumps(
        {
            "findings": "子问题结论：滚动 compaction 在 70% 阈值触发。",
            "notes": [
                {"text": "子 agent 笔记", "entities": ["Compaction"], "confidence": "medium"}
            ],
            "sources": [{"url": "https://example.com/sub", "title": "子来源"}],
        },
        ensure_ascii=False,
    )
    strong_turns = [
        text_turn(PLAN_TEXT),
        tool_turn([call("dispatch_research", {"topic": "上下文压缩"}, id="c1")]),
        text_turn("子任务完成，汇总完毕。"),
        text_turn(REPORT_TEXT),
    ]
    cheap_turns = [
        tool_turn([call("web_search", {"query": "上下文压缩"}, id="s1")]),
        text_turn(sub_json),
        text_turn(DISTILL_JSON),
    ]
    loop, strong, cheap = make_loop(strong_turns, tmp_path, cheap_turns=cheap_turns)
    events = list(loop.events())

    # 子 agent 独立上下文：首轮消息只有研究指令，主循环历史不掺入
    assert cheap.calls[0] == [Message(role="user", content="研究主题：上下文压缩")]
    # 主循环第二次 act 请求：工具结果只有 dispatch_research 一条，无子 agent 中间步骤
    act2 = strong.calls[2]
    assert [m.role for m in act2] == ["user", "assistant", "assistant", "tool"]
    assert act2[3].name == "dispatch_research" and act2[3].tool_call_id == "c1"
    payload = json.loads(act2[3].content)
    assert payload["steps"] == 2
    assert "滚动 compaction" in payload["findings"]
    # 调用归属：主循环 4 次 strong（plan/act/act/report），子 agent 2 步 + 蒸馏走 cheap
    assert [r["step"] for r in accountant_steps(tmp_path)] == [
        "plan",
        "act:1",
        "subagent:step:1",
        "subagent:step:2",
        "act:2",
        "distill",
        "report",
    ]
    # 子 agent 笔记先发出，蒸馏笔记接续，编号连续
    notes = [e for e in events if e["type"] == "data-note"]
    assert [e["data"]["text"] for e in notes] == ["子 agent 笔记", "Letta 用后台 subagent 整理记忆"]
    assert [e["data"]["id"] for e in notes] == ["N-0001", "N-0002"]
    # 子 agent 来源进入引用池 → source-url 事件
    urls = [e["url"] for e in events if e["type"] == "source-url"]
    assert "https://example.com/sub" in urls
    assert any("letta-ai" in u for u in urls)
    # 子 agent 任务卡：running → done
    sub_tasks = [e for e in events if e["type"] == "data-task" and "子 agent" in e["data"]["title"]]
    assert [t["data"]["status"] for t in sub_tasks] == ["running", "done"]


def test_subagent_stops_on_its_own_token_budget_only(tmp_path):
    heavy = TokenUsage(input_tokens=60_000, output_tokens=10)
    strong_turns = [
        text_turn(PLAN_TEXT),
        tool_turn([call("dispatch_research", {"topic": "子问题"}, id="c1")]),
        text_turn("汇总。"),
        text_turn(REPORT_TEXT),
    ]
    cheap_turns = [
        tool_turn([call("web_search", {"query": "q1"}, id="s1")], usage=heavy),
        tool_turn([call("web_search", {"query": "q2"}, id="s2")], usage=heavy),
        text_turn(DISTILL_JSON),  # 熔断后主循环的蒸馏调用
        text_turn("多余轮次不应被消费"),
    ]
    loop, strong, cheap = make_loop(
        strong_turns, tmp_path, cheap_turns=cheap_turns, subagent_token_budget=70_000
    )
    events = list(loop.events())

    # 子 agent 第 3 步被自身预算拦截（60k+60k ≥ 70k），主循环与蒸馏照常完成
    sub_calls = len(cheap.calls) - 1  # 去掉蒸馏那次
    assert sub_calls == 2
    payload = json.loads(strong.calls[2][3].content)
    assert payload["steps"] == 2
    assert "".join(e["delta"] for e in events if e["type"] == "text-delta") == REPORT_TEXT
    # 主循环 input tokens 不含子 agent 消耗（独立预算），state 只记主循环
    assert loop.input_tokens == sum(
        r["input_tokens"]
        for r in accountant_steps(tmp_path)
        if not r["step"].startswith("subagent")
    )


def test_subagent_standalone_run_returns_fixed_schema(tmp_path):
    provider = ScriptedProvider(
        [
            tool_turn([call("web_search", {"query": "q"}, id="s1")]),
            text_turn('```json\n{"findings": "结论", "notes": [], "sources": []}\n```'),
        ],
        model="mock-cheap",
    )
    registry = build_sub_registry(
        RunContext(
            search_provider=MockSearch(), sources_dir=tmp_path / "s", wiki_root=tmp_path / "w"
        )
    )
    sub = ResearchSubagent(
        topic="测试主题",
        provider=provider,
        registry=registry,
        max_steps=3,
        token_budget=100_000,
    )
    result = sub.run()
    assert result.findings == "结论"
    assert result.steps == 2
    assert result.notes == [] and result.sources == []
    # 非法 JSON 回退：findings 用原文
    provider2 = ScriptedProvider([[StreamEvent(type="text_delta", delta="这不是 JSON")]])
    sub2 = ResearchSubagent(
        topic="t", provider=provider2, registry=registry, max_steps=2, token_budget=100_000
    )
    assert sub2.run().findings == "这不是 JSON"
