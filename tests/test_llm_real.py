"""真实 Provider 层单测：全部用 httpx.MockTransport 构造假响应，零真实网络。"""

import json
import tomllib
from pathlib import Path

import httpx
import pytest

from researchwiki.llm.openai_provider import OpenAICompatibleProvider, ProviderError
from researchwiki.llm.provider import Message, MockProvider, StreamEvent, TokenUsage
from researchwiki.llm.replay import ReplayProvider
from researchwiki.llm.router import ModelRouter

CONFIG_TOML = Path(__file__).parents[1] / "config.toml"


# ---- 假响应构造 -----------------------------------------------------------


def sse_body(chunks: list[dict]) -> bytes:
    """把 chunk 列表拼成 OpenAI 兼容 SSE 报文，以 data: [DONE] 结尾。"""
    lines = [f"data: {json.dumps(c, ensure_ascii=False)}" for c in chunks]
    lines.append("data: [DONE]")
    return ("\n\n".join(lines) + "\n\n").encode("utf-8")


def text_chunk(delta: str, *, usage: dict | None = None) -> dict:
    chunk: dict = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
    }
    if usage is not None:
        chunk["usage"] = usage
    return chunk


def tool_chunk(tool_calls: list[dict], *, finish: str | None = None) -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"tool_calls": tool_calls}, "finish_reason": finish}],
    }


def make_provider(handler, *, recorded: list | None = None, **kwargs):
    """构造注入 MockTransport 的 provider；sleep 被替换为记录器以免测试真实等待。"""
    sleep_log: list[float] = []

    def wrapping(request):
        if recorded is not None:
            recorded.append(request)
        return handler(request)

    provider = OpenAICompatibleProvider(
        base_url="http://test.local/v1",
        model="glm-test",
        tier="strong",
        api_key="test-key",
        transport=httpx.MockTransport(wrapping),
        sleep=sleep_log.append,
        **kwargs,
    )
    return provider, sleep_log


# ---- OpenAICompatibleProvider ---------------------------------------------


def test_openai_stream_text_and_usage():
    recorded: list = []
    chunks = [
        text_chunk("你好，"),
        text_chunk("世界！"),
        text_chunk("", usage={"prompt_tokens": 100, "completion_tokens": 20}),
    ]
    provider, sleeps = make_provider(
        lambda request: httpx.Response(200, content=sse_body(chunks)), recorded=recorded
    )
    events = list(provider.stream([Message(role="user", content="打个招呼")]))

    text = "".join(e.delta for e in events if e.type == "text_delta")
    assert text == "你好，世界！"
    usage_events = [e for e in events if e.type == "usage"]
    assert len(usage_events) == 1
    usage = usage_events[0].usage
    assert usage is not None
    assert (usage.input_tokens, usage.output_tokens, usage.cache_read_tokens) == (100, 20, 0)
    assert usage.total_tokens == 120
    assert sleeps == []  # 成功路径不重试

    request = recorded[0]
    body = json.loads(request.content)
    assert body["model"] == "glm-test" and body["stream"] is True
    assert request.headers["Authorization"] == "Bearer test-key"
    assert request.url.path == "/v1/chat/completions"


def test_openai_aggregates_tool_call_increments():
    chunks = [
        text_chunk("让我查一下。"),
        tool_chunk(
            [
                {
                    "index": 0,
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search_wiki", "arguments": ""},
                }
            ]
        ),
        tool_chunk([{"index": 0, "function": {"arguments": "{\"query\": "}}]),
        tool_chunk([{"index": 0, "function": {"arguments": "\"缓存\"}"}}]),
        tool_chunk([], finish="tool_calls"),
    ]
    provider, _ = make_provider(lambda request: httpx.Response(200, content=sse_body(chunks)))
    events = list(provider.stream([Message(role="user", content="查缓存")]))

    tc_events = [e for e in events if e.type == "tool_calls"]
    assert len(tc_events) == 1  # 聚合完成后只产出一次
    assert tc_events[0].tool_calls == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "search_wiki", "arguments": "{\"query\": \"缓存\"}"},
        }
    ]


def test_openai_tool_calls_multiple_indices_sorted():
    chunks = [
        tool_chunk(
            [
                {"index": 1, "id": "call_b", "function": {"name": "read_url", "arguments": ""}},
                {"index": 0, "id": "call_a", "function": {"name": "search_wiki", "arguments": ""}},
            ]
        ),
        tool_chunk(
            [
                {"index": 1, "function": {"arguments": "{\"url\": \"https://x\"}"}},
                {"index": 0, "function": {"arguments": "{\"query\": \"q\"}"}},
            ]
        ),
    ]
    provider, _ = make_provider(lambda request: httpx.Response(200, content=sse_body(chunks)))
    events = list(provider.stream([Message(role="user", content="两个工具")]))
    tc_events = [e for e in events if e.type == "tool_calls"]
    assert [tc["id"] for tc in tc_events[0].tool_calls] == ["call_a", "call_b"]  # 按 index 排序
    assert tc_events[0].tool_calls[1]["function"]["arguments"] == "{\"url\": \"https://x\"}"


def test_openai_maps_cached_tokens():
    chunks = [
        text_chunk(
            "带缓存命中的回复",
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "prompt_tokens_details": {"cached_tokens": 64},
            },
        )
    ]
    provider, _ = make_provider(lambda request: httpx.Response(200, content=sse_body(chunks)))
    events = list(provider.stream([Message(role="user", content="hi")]))
    usage = [e for e in events if e.type == "usage"][0].usage
    assert usage is not None and usage.cache_read_tokens == 64


def test_openai_system_and_tools_passthrough():
    recorded: list = []
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_wiki",
                "description": "检索个人 wiki",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
        }
    ]
    provider, _ = make_provider(
        lambda request: httpx.Response(200, content=sse_body([text_chunk("ok")])),
        recorded=recorded,
        temperature=0.3,
    )
    list(provider.stream([Message(role="user", content="q")], system="你是研究助手", tools=tools))

    body = json.loads(recorded[0].content)
    assert body["messages"][0] == {"role": "system", "content": "你是研究助手"}
    assert body["messages"][1] == {"role": "user", "content": "q"}
    assert body["tools"] == tools
    assert body["temperature"] == 0.3


def test_openai_retries_on_429_then_succeeds():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] <= 2:
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(
            200,
            content=sse_body(
                [text_chunk("重试成功", usage={"prompt_tokens": 5, "completion_tokens": 2})]
            ),
        )

    provider, sleeps = make_provider(handler)
    events = list(provider.stream([Message(role="user", content="hi")]))

    assert "".join(e.delta for e in events if e.type == "text_delta") == "重试成功"
    assert calls["n"] == 3  # 初始 + 2 次重试
    assert len(sleeps) == 2
    assert sleeps[0] >= 0.5 and sleeps[1] >= 1.0  # 指数退避


def test_openai_retry_exhausted_on_500():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(500, json={"error": "boom"})

    provider, sleeps = make_provider(handler)
    with pytest.raises(ProviderError):
        list(provider.stream([Message(role="user", content="hi")]))
    assert calls["n"] == 4  # 最多 3 次重试
    assert len(sleeps) == 3


def test_openai_no_retry_on_client_error():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(400, json={"error": "bad request"})

    provider, sleeps = make_provider(handler)
    with pytest.raises(ProviderError):
        list(provider.stream([Message(role="user", content="hi")]))
    assert calls["n"] == 1 and sleeps == []  # 其余 4xx 立即失败


def test_openai_retries_on_connect_error():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        raise httpx.ConnectError("connection refused", request=request)

    provider, sleeps = make_provider(handler)
    with pytest.raises(ProviderError):
        list(provider.stream([Message(role="user", content="hi")]))
    assert calls["n"] == 4 and len(sleeps) == 3


def test_openai_stream_options_include_usage():
    recorded: list = []
    provider, _ = make_provider(
        lambda request: httpx.Response(200, content=sse_body([text_chunk("ok")])),
        recorded=recorded,
    )
    list(provider.stream([Message(role="user", content="hi")]))
    assert json.loads(recorded[0].content)["stream_options"] == {"include_usage": True}


# ---- ModelRouter -----------------------------------------------------------


def test_router_falls_back_to_mock_when_base_url_empty():
    router = ModelRouter(
        {
            "strong": {"model": "", "base_url": "", "api_key_env": ""},
            "cheap": {"model": "glm-flash", "base_url": "", "api_key_env": ""},
        }
    )
    strong = router.get("strong")
    assert isinstance(strong, MockProvider)
    assert strong.model == "mock-strong" and strong.tier == "strong"
    cheap = router.get("cheap")
    assert isinstance(cheap, MockProvider)
    assert cheap.model == "glm-flash" and cheap.tier == "cheap"


def test_router_returns_openai_provider_for_real_config():
    router = ModelRouter(
        {
            "cheap": {
                "model": "glm-4.7-flash",
                "base_url": "http://test.local/v1",
                "api_key_env": "RESEARCHWIKI_CHEAP_API_KEY",
            }
        }
    )
    provider = router.get("cheap")
    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.model == "glm-4.7-flash" and provider.tier == "cheap"
    assert provider.base_url == "http://test.local/v1"


def test_router_matches_project_config_toml():
    """config.toml 的 [llm] 段能被 router 直接消费：结构完整、两档可解析。

    仓库里的 config.toml 会写真实模型与 base_url（便于 clone 后改 mode 即用），
    所以断言的**不是**"默认 mock"，而是：
    - 两档都存在且结构合法（model / base_url / api_key_env 键齐全）；
    - base_url 为空才回退 mock——真正保证演示与测试不外呼的是 [server].mode="mock"，
      该约束由 tests/test_server.py 的 mode 开关测试覆盖。
    """
    with CONFIG_TOML.open("rb") as f:
        data = tomllib.load(f)
    llm_cfg = data["llm"]
    assert set(llm_cfg) >= {"strong", "cheap"}
    for tier in ("strong", "cheap"):
        section = llm_cfg[tier]
        assert {"model", "base_url", "api_key_env"} <= set(section)

    router = ModelRouter(llm_cfg)
    for tier in ("strong", "cheap"):
        provider = router.get(tier)
        if str(llm_cfg[tier].get("base_url") or "").strip():
            assert isinstance(provider, OpenAICompatibleProvider)
        else:  # 配置留空 = mock，无需 key 即可跑
            assert isinstance(provider, MockProvider)

    # 仓库默认必须仍是演示模式：不动配置直接起服务不会带 key 外出
    assert str((data.get("server") or {}).get("mode") or "mock").lower() == "mock"


# ---- ReplayProvider --------------------------------------------------------


class ExplodeProvider:
    """一旦被触达就失败——用于证明回放根本没打到内部 Provider。"""

    model = "mock-strong"
    tier = "strong"

    def stream(self, messages, *, system=None, tools=None):
        raise AssertionError("命中缓存不应触达内部 Provider")
        yield  # pragma: no cover —— 使本函数成为生成器


def test_replay_records_then_replays_from_cache(tmp_path):
    cache = tmp_path / "replay.jsonl"
    inner = MockProvider(["第一段", "第二段"], usage=TokenUsage(input_tokens=10, output_tokens=5))
    replay = ReplayProvider(inner, cache_path=cache)
    messages = [Message(role="user", content="问题")]

    first = list(replay.stream(messages))
    assert "".join(e.delta for e in first if e.type == "text_delta") == "第一段第二段"

    lines = cache.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert len(record["hash"]) == 64  # sha256 hex
    assert record["model"] == "mock-strong"
    assert record["text"] == "第一段第二段"
    assert record["tool_calls"] is None
    assert record["usage"] == {
        "input_tokens": 10,
        "output_tokens": 5,
        "cache_read_tokens": 0,
        "cache_creation_tokens": 0,
    }

    replayed = ReplayProvider(ExplodeProvider(), cache_path=cache)
    second = list(replayed.stream(messages))
    assert "".join(e.delta for e in second if e.type == "text_delta") == "第一段第二段"
    usage = [e for e in second if e.type == "usage"][0].usage
    assert (usage.input_tokens, usage.output_tokens) == (10, 5)


class FakeToolProvider:
    """产出 tool_calls 事件的假 Provider，验证回放保留工具调用。"""

    model = "fake-tool-model"
    tier = "cheap"

    def __init__(self) -> None:
        self.calls = 0

    def stream(self, messages, *, system=None, tools=None):
        self.calls += 1
        yield StreamEvent(type="text_delta", delta="查一下：")
        yield StreamEvent(
            type="tool_calls",
            tool_calls=[
                {
                    "id": "call_9",
                    "type": "function",
                    "function": {"name": "search_wiki", "arguments": "{\"query\": \"缓存\"}"},
                }
            ],
        )
        yield StreamEvent(type="usage", usage=TokenUsage(input_tokens=7, output_tokens=3))


class ExplodeToolProvider:
    model = "fake-tool-model"
    tier = "cheap"

    def stream(self, messages, *, system=None, tools=None):
        raise AssertionError("命中缓存不应触达内部 Provider")
        yield  # pragma: no cover


def test_replay_preserves_tool_calls_on_replay(tmp_path):
    cache = tmp_path / "replay.jsonl"
    fake = FakeToolProvider()
    replay = ReplayProvider(fake, cache_path=cache)
    messages = [Message(role="user", content="查缓存")]

    first = list(replay.stream(messages))
    assert [e.type for e in first] == ["text_delta", "tool_calls", "usage"]
    assert fake.calls == 1

    second = list(ReplayProvider(ExplodeToolProvider(), cache_path=cache).stream(messages))
    tc_events = [e for e in second if e.type == "tool_calls"]
    assert len(tc_events) == 1
    assert tc_events[0].tool_calls[0]["function"]["name"] == "search_wiki"
    usage = [e for e in second if e.type == "usage"][0].usage
    assert (usage.input_tokens, usage.output_tokens) == (7, 3)


def test_replay_cache_key_covers_system_and_tools(tmp_path):
    """system / tools 参与请求哈希：变了就是缓存未命中。"""
    cache = tmp_path / "replay.jsonl"
    inner = MockProvider(["a"])
    replay = ReplayProvider(inner, cache_path=cache)
    messages = [Message(role="user", content="q")]

    list(replay.stream(messages))
    list(replay.stream(messages))  # 同请求 → 命中
    list(replay.stream(messages, tools=[{"type": "function", "function": {"name": "t"}}]))
    list(replay.stream(messages, system="不同系统提示"))

    assert len(inner.calls) == 3  # 仅首次命中缓存
    lines = cache.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
