"""AgentLoop × Prior 集成：注入、不污染来源池、计数与 run-metrics 落盘（PLAN §4.2/§4.5）。

脚手架与 tests/test_loop.py 同构（ScriptedProvider + MockSearch + tmp_path）；
wiki 预置笔记用 WikiStore.save_note，检索口径与 tests/test_prior.py 一致：
MockEmbeddingProvider(dim=512) + tokenizer=trigram（确定性、零网络、不依赖
vendor DLL）。索引在 AgentLoop 构造时为空快照，events() 开头的
ensure_index_fresh 会真实走一遍 rebuild，预置笔记因此能被检索到。
"""

import json
from pathlib import Path

import httpx
import pytest

from researchwiki.llm.accounting import TokenAccountant
from researchwiki.llm.provider import ScriptedProvider, StreamEvent, TokenUsage
from researchwiki.loop.agent_loop import AgentLoop
from researchwiki.loop.metrics import sum_tokens_from_jsonl
from researchwiki.tools import MockSearch
from researchwiki.wiki.embeddings import MockEmbeddingProvider
from researchwiki.wiki.frontmatter import SourceRef
from researchwiki.wiki.prior import PRIOR_CONTEXT_LABEL
from researchwiki.wiki.store import WikiStore

QUESTION = "agent 记忆方案对比"
PLAN_TEXT = "- 任务一：检索主流方案\n- 任务二：蒸馏笔记\n"
REPORT_TEXT = "## 研究报告\n\nLetta 方案[1] 值得优先试点。\n"
DISTILL_JSON = json.dumps(
    {
        "notes": [
            {"text": "Letta 用后台 subagent 整理记忆", "entities": ["Letta"], "confidence": "high"}
        ],
        "conflicts": [],
    },
    ensure_ascii=False,
)
PRIOR_SOURCE_URL = "https://example.com/prior-source"


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
    """按档位返回预置 Provider（未配置的档位被请求即失败，防止测试里静默串线）。"""

    def __init__(self, strong: ScriptedProvider, cheap: ScriptedProvider | None = None) -> None:
        self._providers = {"strong": strong}
        if cheap is not None:
            self._providers["cheap"] = cheap

    def get(self, tier: str) -> ScriptedProvider:
        if tier not in self._providers:
            raise AssertionError(f"测试未配置 {tier} 档 Provider，却被请求了")
        return self._providers[tier]


def make_loop(
    tmp_path: Path,
    *,
    strong_turns: list[list[StreamEvent]],
    cheap_turns: list[list[StreamEvent]] | None = None,
    prior_config: dict | None = None,
    fetch_transport: httpx.BaseTransport | None = None,
    **kwargs,
) -> AgentLoop:
    strong = ScriptedProvider(strong_turns, model="mock-strong")
    cheap = None
    # cheap 未配置 base_url → 蒸馏/子 agent 复用 strong（与 tests/test_loop.py 同口径）
    llm_config = {"strong": {"base_url": "https://mock"}, "cheap": {"base_url": ""}}
    if cheap_turns is not None:
        cheap = ScriptedProvider(cheap_turns, tier="cheap", model="mock-cheap")
        llm_config = {
            "strong": {"base_url": "https://mock"},
            "cheap": {"base_url": "https://mock-cheap"},
        }
    return AgentLoop(
        QUESTION,
        router=FakeRouter(strong, cheap),
        llm_config=llm_config,
        accountant=TokenAccountant(tmp_path / "tokens.jsonl"),
        search_provider=MockSearch(),
        wiki_root=tmp_path / "wiki-data",
        run_dir=tmp_path / "run",
        # 检索确定性：dim=512 下夹具间正交/可区分；trigram 不探测 vendor DLL。
        # wiki_config 传 [wiki] 段（与 server/main.py、real_run_smoke.py 同口径）
        embedding=MockEmbeddingProvider(dim=512),
        wiki_config={"fts_tokenizer": "trigram"},
        prior_config=prior_config,
        fetch_transport=fetch_transport,
        **kwargs,
    )


def preload_note(store: WikiStore, body: str, **kwargs) -> None:
    store.save_note(body, **kwargs)


def load_metrics(tmp_path: Path) -> dict:
    path = tmp_path / "run" / "run-metrics.json"
    assert path.is_file(), "run 结束后 run-metrics.json 必须存在"
    return json.loads(path.read_text(encoding="utf-8"))


def default_turns() -> list[list[StreamEvent]]:
    return [
        text_turn(PLAN_TEXT),
        tool_turn([call("web_search", {"query": "agent 记忆"}, id="c1")]),
        text_turn("已检索到主流方案，研究完成。"),
        text_turn(DISTILL_JSON),
        text_turn(REPORT_TEXT),
    ]


# ---- 1. 命中注入 + 不污染来源池 -----------------------------------------------


def test_prior_hit_injected_into_plan_and_kept_out_of_source_pool(tmp_path: Path) -> None:
    store = WikiStore(tmp_path / "wiki-data")
    preload_note(
        store,
        f"历史结论：关于{QUESTION}，Letta 的后台 subagent 方案最成熟。",
        note_id="N-0001",
        title="记忆方案（历史）",
        confidence="high",
        sources=[SourceRef(url=PRIOR_SOURCE_URL, content_hash="a" * 64)],
    )
    loop = make_loop(tmp_path, strong_turns=default_turns())
    events = list(loop.events())

    # [wiki] 段的 tokenizer 配置真实传到了 Prior 索引（段/完整 config 两种口径都收）
    assert loop.prior_index is not None and loop.prior_index.tokenizer == "trigram"

    # plan 步骤的 user 消息含标签行原文，且问题仍在（Prior 在问题之前）
    plan_msg = loop.provider.calls[0][0]  # type: ignore[attr-defined]
    assert plan_msg.role == "user"
    assert PRIOR_CONTEXT_LABEL in plan_msg.content
    assert "研究问题：" + QUESTION in plan_msg.content
    assert plan_msg.content.index(PRIOR_CONTEXT_LABEL) < plan_msg.content.index(QUESTION)

    # metrics：命中 1 条，来源池只有 fresh 来源
    metrics = load_metrics(tmp_path)
    assert metrics["prior_hit_count"] == 1
    assert metrics["prior_note_ids"] == ["N-0001"]
    assert metrics["prior_context_chars"] > 0

    pool_urls = [e["url"] for e in loop.ctx.source_pool.entries]
    assert pool_urls, "本轮应有 fresh 来源"
    assert PRIOR_SOURCE_URL not in pool_urls
    assert all("prior-source" not in u for u in pool_urls)
    # 报告 [n] 来源列表（source-url 事件）同样不含 prior URL
    source_events = [e["url"] for e in events if e["type"] == "source-url"]
    assert source_events == pool_urls
    assert PRIOR_SOURCE_URL not in source_events


# ---- 2. 空 Wiki：run 正常完成，metrics 仍落盘 ----------------------------------


def test_empty_wiki_run_completes_and_metrics_written(tmp_path: Path) -> None:
    loop = make_loop(tmp_path, strong_turns=default_turns())
    events = list(loop.events())

    assert events[0]["type"] == "start" and events[-1]["type"] == "finish"
    # 空 Wiki：不注入标签行，plan 消息就是问题原文
    plan_msg = loop.provider.calls[0][0]  # type: ignore[attr-defined]
    assert plan_msg.content == QUESTION
    metrics = load_metrics(tmp_path)
    assert metrics["prior_hit_count"] == 0
    assert metrics["prior_note_ids"] == []
    assert metrics["prior_context_chars"] == 0
    assert metrics["source_count"] > 0


# ---- 3. merged→active 链：只注入最终 active note --------------------------------


def test_merged_chain_injects_only_final_active_note(tmp_path: Path) -> None:
    store = WikiStore(tmp_path / "wiki-data")
    # 查询短语只出现在 merged 旧笔记正文里 → 命中必须走重定向路径
    preload_note(
        store,
        f"关于{QUESTION}的早期草稿介绍，内容已过时。",
        note_id="N-0001",
        title="记忆方案（旧）",
        status="merged",
        redirect_to="N-0002",
    )
    preload_note(
        store,
        "最终结论：后台 subagent 方案最成熟，配置与成本见官方文档。",
        note_id="N-0002",
        title="记忆方案",
        confidence="high",
    )
    loop = make_loop(
        tmp_path,
        strong_turns=[
            text_turn(PLAN_TEXT),
            text_turn("研究完成。"),
            text_turn(DISTILL_JSON),
            text_turn(REPORT_TEXT),
        ],
    )
    events = list(loop.events())
    assert events[-1]["type"] == "finish"

    plan_msg = loop.provider.calls[0][0]  # type: ignore[attr-defined]
    assert PRIOR_CONTEXT_LABEL in plan_msg.content
    assert "N-0002" in plan_msg.content  # 注入的是最终 active note
    assert "早期草稿" not in plan_msg.content  # merged 旧笔记正文不注入
    metrics = load_metrics(tmp_path)
    assert metrics["prior_hit_count"] == 1
    assert metrics["prior_note_ids"] == ["N-0002"]


# ---- 4. run-metrics 与 tokens.jsonl 对账 ----------------------------------------


def test_run_metrics_reconciles_with_tokens_jsonl(tmp_path: Path) -> None:
    store = WikiStore(tmp_path / "wiki-data")
    preload_note(
        store,
        f"历史结论：关于{QUESTION}，Letta 的后台 subagent 方案最成熟。",
        note_id="N-0001",
        title="记忆方案（历史）",
    )
    loop = make_loop(tmp_path, strong_turns=default_turns())
    list(loop.events())

    metrics = load_metrics(tmp_path)
    token_in, token_out = sum_tokens_from_jsonl(tmp_path / "tokens.jsonl", loop.trace_id)
    assert metrics["trace_id"] == loop.trace_id
    assert metrics["input_tokens"] == token_in == loop.input_tokens
    assert metrics["output_tokens"] == token_out == loop.output_tokens
    assert token_in > 0  # ScriptedProvider 自动补默认 usage，对账不是空转
    assert isinstance(metrics["latency_ms"], int) and metrics["latency_ms"] >= 0
    # 入库动作计数：蒸馏出的 1 条笔记为 created
    assert metrics["notes_created"] == 1
    assert metrics["notes_merged"] == 0
    assert metrics["notes_superseded"] == 0


# ---- 5. search/fetch 计数 ------------------------------------------------------


def test_fresh_search_and_fetch_counts(tmp_path: Path) -> None:
    html = (
        "<html><body><h1>Agent 记忆综述</h1>"
        "<p>智能体的长期记忆是核心议题。Letta 提出 sleep-time compute 思路，"
        "由后台子代理在会话空闲期整理记忆，把维护成本移出交互窗口，上下文保持精简，"
        "同时保留跨会话可复用的长期事实，这些内容足以通过正文提取阈值。</p></body></html>"
    )
    transport = httpx.MockTransport(lambda request: httpx.Response(200, html=html))
    loop = make_loop(
        tmp_path,
        strong_turns=[
            text_turn(PLAN_TEXT),
            tool_turn(
                [
                    call("web_search", {"query": "agent 记忆"}, id="c1"),
                    call("fetch_url", {"url": "https://example.com/memo"}, id="c2"),
                ]
            ),
            text_turn("研究完成。"),
            text_turn(DISTILL_JSON),
            text_turn(REPORT_TEXT),
        ],
        fetch_transport=transport,
    )
    events = list(loop.events())
    assert events[-1]["type"] == "finish"

    assert loop.ctx.search_calls == 1
    assert loop.ctx.fetch_calls == 1
    metrics = load_metrics(tmp_path)
    assert metrics["fresh_search_count"] == 1
    assert metrics["fresh_fetch_count"] == 1
    assert metrics["source_count"] == len(loop.ctx.source_pool.entries)
    # 引用覆盖率精度口径 round(x, 4)：REPORT_TEXT 只引了 [1]
    assert metrics["citation_coverage"] == round(1 / len(loop.ctx.source_pool.entries), 4)


# ---- 6. enabled=false：完全不检索不注入 -----------------------------------------


def test_prior_disabled_skips_retrieval_and_injection(tmp_path: Path) -> None:
    store = WikiStore(tmp_path / "wiki-data")
    preload_note(
        store,
        f"历史结论：关于{QUESTION}，Letta 的后台 subagent 方案最成熟。",
        note_id="N-0001",
        title="记忆方案（历史）",
    )
    loop = make_loop(
        tmp_path,
        strong_turns=default_turns(),
        prior_config={"enabled": False},
    )
    events = list(loop.events())

    assert events[-1]["type"] == "finish"
    assert loop.prior_index is None  # 禁用时连索引都不建（零副作用）
    assert loop.prior_context is None
    plan_msg = loop.provider.calls[0][0]  # type: ignore[attr-defined]
    assert plan_msg.content == QUESTION  # 无任何注入
    assert PRIOR_CONTEXT_LABEL not in plan_msg.content
    metrics = load_metrics(tmp_path)
    assert metrics["prior_hit_count"] == 0
    assert metrics["prior_note_ids"] == []


# ---- 7. dispatch 路径：子 agent 用量回流主计数，metrics 可对账 -------------------


def test_dispatch_run_tokens_reconcile_with_tokens_jsonl(tmp_path: Path) -> None:
    """含 dispatch_research 的 run：子 agent 的 token 经 on_usage 回流主循环滚动
    计数，run-metrics.json 与 tokens.jsonl 按 trace_id 仍可精确对账（评审修复项）。"""
    sub_json = json.dumps(
        {
            "findings": "子问题结论：70% 阈值触发压缩。",
            "notes": [],
            "sources": [{"url": "https://example.com/sub", "title": "子来源"}],
        },
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
            text_turn(DISTILL_JSON),  # cheap 配置后蒸馏也走 cheap 档
        ],
    )
    events = list(loop.events())
    assert events[-1]["type"] == "finish"

    # 子 agent 真实跑过：2 步收尾、其来源进了引用池
    payload = json.loads(loop.provider.calls[2][3].content)  # type: ignore[attr-defined]
    assert payload["steps"] == 2
    assert "https://example.com/sub" in [e["url"] for e in events if e["type"] == "source-url"]

    # 对账：run-metrics == tokens.jsonl 按 trace_id 全量合计 == 主循环滚动计数
    metrics = load_metrics(tmp_path)
    token_in, token_out = sum_tokens_from_jsonl(tmp_path / "tokens.jsonl", loop.trace_id)
    assert token_in > 0  # 子 agent 与蒸馏的用量真实入账
    assert metrics["input_tokens"] == token_in == loop.input_tokens
    assert metrics["output_tokens"] == token_out == loop.output_tokens


# ---- 8. 流中途异常：run-metrics 仍恰好写一次 -------------------------------------


class _BoomProvider:
    """第一次调用即抛错的 Provider：模拟流在 plan 步骤中途死亡。"""

    model = "mock-strong"
    tier = "strong"

    def stream(self, messages, *, system=None, tools=None):
        raise RuntimeError("模型连接中断")


def test_run_metrics_still_written_when_stream_dies_midway(tmp_path: Path) -> None:
    """生成器异常关闭：metrics 经 try/finally 写一次（取当时实况），原异常照常抛出。"""
    store = WikiStore(tmp_path / "wiki-data")
    preload_note(
        store,
        f"历史结论：关于{QUESTION}，Letta 的后台 subagent 方案最成熟。",
        note_id="N-0001",
        title="记忆方案（历史）",
    )
    loop = AgentLoop(
        QUESTION,
        router=FakeRouter(_BoomProvider()),  # type: ignore[arg-type]
        llm_config={"strong": {"base_url": "https://mock"}, "cheap": {"base_url": ""}},
        accountant=TokenAccountant(tmp_path / "tokens.jsonl"),
        search_provider=MockSearch(),
        wiki_root=tmp_path / "wiki-data",
        run_dir=tmp_path / "run",
        embedding=MockEmbeddingProvider(dim=512),
        wiki_config={"fts_tokenizer": "trigram"},
    )
    with pytest.raises(RuntimeError, match="模型连接中断"):
        list(loop.events())

    # Prior 已检索（异常发生在 plan 调用）、metrics 已落盘且与记账一致
    assert loop.prior_context is not None and len(loop.prior_context.hits) == 1
    metrics = load_metrics(tmp_path)
    token_in, token_out = sum_tokens_from_jsonl(tmp_path / "tokens.jsonl", loop.trace_id)
    assert metrics["input_tokens"] == token_in == loop.input_tokens == 0
    assert metrics["prior_hit_count"] == 1
    assert metrics["prior_note_ids"] == ["N-0001"]
    assert metrics["citation_coverage"] is None  # 无报告无来源，取当时实况


def test_run_metrics_written_exactly_once_on_happy_path(monkeypatch, tmp_path: Path) -> None:
    """正常结束：落盘恰好一次（try/finally 单一写点，不因守卫重复写）。"""
    loop = make_loop(tmp_path, strong_turns=default_turns())
    calls: list[float] = []
    original = AgentLoop._write_run_metrics

    def spy(self: AgentLoop, t_start: float) -> None:
        calls.append(t_start)
        original(self, t_start)

    monkeypatch.setattr(AgentLoop, "_write_run_metrics", spy)
    list(loop.events())

    assert len(calls) == 1
    assert (tmp_path / "run" / "run-metrics.json").is_file()
