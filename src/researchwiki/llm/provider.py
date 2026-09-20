"""LLM Provider 抽象与 Mock 实现。

骨架先行的关键：接口与测试不依赖任何 API key；真模型接入只是换 Provider 实现 + 改 config。
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class Message:
    """一条对话消息。

    基础字段覆盖普通对话；可选字段服务 agent loop 的工具调用回传
    （OpenAI 协议的 role="tool" / assistant.tool_calls），不填即保持原行为。
    """

    role: Role
    content: str
    # role="tool" 时必带：本条结果对应的工具调用 id 与工具名
    tool_call_id: str | None = None
    name: str | None = None
    # role="assistant" 发起工具调用时携带（OpenAI function 格式）
    tool_calls: list[dict[str, Any]] | None = None


@dataclass
class TokenUsage:
    """provider 返回什么记什么：不支持的 cache 字段保持 0，不硬造。"""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class StreamEvent:
    type: Literal["text_delta", "reasoning_delta", "tool_calls", "usage"]
    delta: str = ""
    usage: TokenUsage | None = None
    tool_calls: list[dict[str, Any]] | None = None


class Provider(Protocol):
    model: str
    tier: Literal["strong", "cheap"]

    def stream(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> Iterator[StreamEvent]: ...


class MockProvider:
    """按脚本回放流式事件。

    用途：骨架单测、SSE 联调、前端演示——零成本、可重复、无网络依赖。
    """

    def __init__(
        self,
        script: str | list[str],
        *,
        tier: Literal["strong", "cheap"] = "strong",
        model: str = "mock-strong",
        usage: TokenUsage | None = None,
        delay: float = 0.0,
    ) -> None:
        self.script = [script] if isinstance(script, str) else list(script)
        self.tier: Literal["strong", "cheap"] = tier
        self.model = model
        self.usage = usage or TokenUsage(input_tokens=1200, output_tokens=180)
        self.delay = delay
        self.calls: list[list[Message]] = []

    def stream(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> Iterator[StreamEvent]:
        self.calls.append(messages)
        for chunk in self.script:
            if self.delay:
                time.sleep(self.delay)
            yield StreamEvent(type="text_delta", delta=chunk)
        yield StreamEvent(type="usage", usage=self.usage)


class ScriptedProvider:
    """多轮脚本化 Provider：每次 stream() 消费一轮事件脚本。

    agent loop 一次 run 会发起多次 LLM 调用（规划 / 带工具的执行 / 蒸馏 / 报告），
    MockProvider 只能回放同一份响应，无法表达"先回 tool_calls 再回纯文本"的
    多步剧本——本类补齐这个空缺，同样零网络、可记录调用：

        turns = [
            [StreamEvent(type="text_delta", delta="计划……")],          # 第 1 次调用
            [StreamEvent(type="tool_calls", tool_calls=[{...}])],      # 第 2 次调用
            [StreamEvent(type="text_delta", delta="报告……")],          # 第 3 次调用
        ]

    轮次里没有 usage 事件时自动补一条默认 usage（对齐 MockProvider 行为，
    保证记账链路可测）；脚本用尽后重复最后一轮（防御意外的多余调用）。
    """

    def __init__(
        self,
        turns: list[list[StreamEvent]],
        *,
        tier: Literal["strong", "cheap"] = "strong",
        model: str = "mock-strong",
        usage: TokenUsage | None = None,
    ) -> None:
        self.turns = [list(turn) for turn in turns]
        self.tier: Literal["strong", "cheap"] = tier
        self.model = model
        self.usage = usage or TokenUsage(input_tokens=1200, output_tokens=180)
        self.calls: list[list[Message]] = []
        self._cursor = 0

    def stream(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> Iterator[StreamEvent]:
        # 快照而非引用：调用方（agent loop）会原地复用/扩写同一个历史列表
        self.calls.append(list(messages))
        index = min(self._cursor, len(self.turns) - 1)
        self._cursor += 1
        events = list(self.turns[index])
        if not any(e.type == "usage" for e in events):
            events.append(StreamEvent(type="usage", usage=self.usage))
        yield from events


@dataclass
class MockEvent:
    """供 loop 层编排完整研究过程的复合事件（reasoning/text/usage 混合）。"""

    events: list[StreamEvent] = field(default_factory=list)


def chunk_text(text: str, size: int = 24) -> list[str]:
    """把一段文本切成流式小块，模拟 token 级增量。"""
    return [text[i : i + size] for i in range(0, len(text), size)]


def script_events(segments: Iterable[str]) -> list[StreamEvent]:
    events: list[StreamEvent] = []
    for seg in segments:
        events.extend(StreamEvent(type="text_delta", delta=c) for c in chunk_text(seg))
    return events
