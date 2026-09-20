import json

from fastapi.testclient import TestClient

from researchwiki.server.main import app

client = TestClient(app)


def _parse_sse(body: str) -> list[dict]:
    events = []
    for line in body.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            events.append(json.loads(line[len("data: "):]))
    return events


def test_health():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_chat_stream_protocol():
    payload = {"messages": [{"role": "user", "content": "agent 记忆方案对比"}]}
    with client.stream("POST", "/api/chat", json=payload) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = "".join(chunk for chunk in resp.iter_text())

    assert body.endswith("data: [DONE]\n\n")
    events = _parse_sse(body)
    types = [e["type"] for e in events]

    # 协议骨架：start → reasoning → tasks/notes/conflict → text → sources → finish
    assert types[0] == "start"
    assert "reasoning-start" in types and "reasoning-delta" in types and "reasoning-end" in types
    assert "data-task" in types and "data-note" in types and "data-conflict" in types
    assert "text-start" in types and "text-delta" in types and "text-end" in types
    assert "source-url" in types
    assert types[-1] == "finish"

    # 事件顺序约束：报告文本在 finish 前，来源在文本后
    assert types.index("text-start") < types.index("source-url") < types.index("finish")


def test_chat_stream_records_tokens(tmp_path, monkeypatch):
    import researchwiki.server.main as sm

    real_accountant = sm.TokenAccountant
    monkeypatch.setattr(
        sm, "TokenAccountant", lambda: real_accountant(tmp_path / "tokens.jsonl")
    )
    payload = {"messages": [{"role": "user", "content": "测试"}]}
    with client.stream("POST", "/api/chat", json=payload) as resp:
        "".join(chunk for chunk in resp.iter_text())
    lines = (tmp_path / "tokens.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) > 0


# ---- mode 开关：mode="real" 走真实 loop，其余一律 mock（默认行为不变）--------


class _StubRouter:
    """测试替身：按档位返回预置 Provider，避免 mode=real 时触发真实网络。"""

    def __init__(self, providers):
        self._providers = providers

    def get(self, tier):
        return self._providers[tier]


def test_real_loop_enabled_requires_real_mode_and_base_url():
    import researchwiki.server.main as sm

    assert sm.real_loop_enabled({}) is False
    assert sm.real_loop_enabled({"server": {"mode": "real"}}) is False
    assert (
        sm.real_loop_enabled({"server": {"mode": "real"}, "llm": {"strong": {"base_url": ""}}})
        is False
    )
    assert (
        sm.real_loop_enabled(
            {"server": {"mode": "real"}, "llm": {"strong": {"base_url": "https://x"}}}
        )
        is True
    )
    # 显式 mock 模式即便配了 base_url 也不走真实 loop
    assert (
        sm.real_loop_enabled(
            {"server": {"mode": "mock"}, "llm": {"strong": {"base_url": "https://x"}}}
        )
        is False
    )


def test_chat_mode_real_runs_agent_loop(tmp_path, monkeypatch):
    import researchwiki.server.main as sm
    from researchwiki.llm.provider import ScriptedProvider, StreamEvent

    report = "## 真实 loop 报告\n\n来自 ScriptedProvider 的报告，mock 演示不会输出这段话。"
    strong = ScriptedProvider(
        [
            [StreamEvent(type="text_delta", delta="- 计划任务一\n- 计划任务二")],
            [StreamEvent(type="text_delta", delta="研究总结：信息已足够。")],
            [StreamEvent(type="text_delta", delta='{"notes": [], "conflicts": []}')],
            [StreamEvent(type="text_delta", delta=report)],
        ]
    )
    monkeypatch.setattr(
        sm,
        "load_config",
        lambda: {
            "server": {"mode": "real", "wiki_data_dir": str(tmp_path / "wiki-data")},
            "llm": {"strong": {"model": "mock-strong", "base_url": "https://mock"}},
        },
    )
    monkeypatch.setattr(
        sm, "ModelRouter", lambda llm_cfg: _StubRouter({"strong": strong, "cheap": strong})
    )
    real_accountant = sm.TokenAccountant
    monkeypatch.setattr(sm, "TokenAccountant", lambda: real_accountant(tmp_path / "tokens.jsonl"))

    payload = {"messages": [{"role": "user", "content": "真实模式冒烟"}]}
    with client.stream("POST", "/api/chat", json=payload) as resp:
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = "".join(chunk for chunk in resp.iter_text())

    events = _parse_sse(body)
    assert events[0]["type"] == "start" and events[-1]["type"] == "finish"
    # 报告正文来自真实 loop 的 ScriptedProvider（ResearchRun 的 mock 脚本没有这段话）
    text = "".join(
        e.get("delta", "") for e in events if e["type"] == "text-delta"
    )
    assert "mock 演示不会输出这段话" in text
    types = [e["type"] for e in events]
    assert "reasoning-start" in types and "data-task" in types and "text-delta" in types
    # 落盘与记账都发生在重定向后的 wiki-data 内
    runs = list((tmp_path / "wiki-data" / "runs").iterdir())
    assert len(runs) == 1
    assert (runs[0] / "research-plan.md").exists()
    assert (runs[0] / "state.md").exists()
    assert (tmp_path / "tokens.jsonl").exists()


def test_chat_mode_real_without_base_url_falls_back_to_mock(monkeypatch):
    import researchwiki.server.main as sm

    monkeypatch.setattr(
        sm,
        "load_config",
        lambda: {"server": {"mode": "real"}, "llm": {"strong": {"model": "", "base_url": ""}}},
    )
    payload = {"messages": [{"role": "user", "content": "回退 mock"}]}
    with client.stream("POST", "/api/chat", json=payload) as resp:
        body = "".join(chunk for chunk in resp.iter_text())
    # mock ResearchRun 的标志性任务标题仍然出现
    assert "搜索主流 agent 记忆方案" in body
