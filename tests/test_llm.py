import json

from researchwiki.llm.accounting import TokenAccountant
from researchwiki.llm.provider import Message, MockProvider, TokenUsage


def test_mock_provider_streams_script_chunks():
    provider = MockProvider(
        ["第一段", "第二段"], usage=TokenUsage(input_tokens=10, output_tokens=5)
    )
    events = list(provider.stream([Message(role="user", content="问题")]))
    text = "".join(e.delta for e in events if e.type == "text_delta")
    assert text == "第一段第二段"
    usage = [e for e in events if e.type == "usage"]
    assert len(usage) == 1 and usage[0].usage.total_tokens == 15
    assert provider.calls == [[Message(role="user", content="问题")]]


def test_accountant_appends_jsonl(tmp_path):
    acc = TokenAccountant(tmp_path / "tokens.jsonl")
    row = acc.record(
        trace_id="abc123",
        step="reasoning:plan",
        model="mock-strong",
        usage=TokenUsage(input_tokens=100, output_tokens=20, cache_read_tokens=50),
        latency_ms=12.3,
    )
    assert row["cache_read_tokens"] == 50
    lines = (tmp_path / "tokens.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["trace_id"] == "abc123" and parsed["model"] == "mock-strong"
