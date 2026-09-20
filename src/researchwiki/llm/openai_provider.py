"""OpenAI 兼容协议的真实 Provider（/chat/completions SSE 流式）。

适配任意 OpenAI 兼容端点（GLM、DeepSeek、vLLM、OpenAI 等）：
流式文本 delta、工具调用增量聚合、usage 记账、连接级指数退避重试。
"""

from __future__ import annotations

import json
import os
import random
import time
from collections.abc import Callable, Iterator
from typing import Any, Literal

import httpx

from researchwiki.llm.provider import Message, StreamEvent, TokenUsage

# 视为可重试的 HTTP 状态码：限流 + 服务端错误
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class ProviderError(RuntimeError):
    """上游 OpenAI 兼容端点错误：不可重试的 4xx，或重试次数耗尽后的终态。"""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class OpenAICompatibleProvider:
    """通过 OpenAI 兼容 chat/completions 接口产出 StreamEvent 流。

    配置项：
    - base_url：API 根地址（不含 /chat/completions），如 "https://open.bigmodel.cn/api/paas/v4"。
    - api_key / api_key_env：显式 key 优先，否则读 api_key_env 指定的环境变量；都没有则不带鉴权头。
    - model / tier / temperature：模型名、档位与可选采样温度。
    - transport / sleep：测试注入点（httpx.MockTransport、免真实退避等待），生产传默认值。
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        tier: Literal["strong", "cheap"] = "strong",
        api_key: str | None = None,
        api_key_env: str | None = None,
        temperature: float | None = None,
        max_retries: int = 3,
        retry_base_delay: float = 0.5,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.tier: Literal["strong", "cheap"] = tier
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.api_key = api_key if api_key is not None else (
            os.environ.get(api_key_env, "") if api_key_env else ""
        )
        self._sleep = sleep
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout, connect=10.0),
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    # ---- 请求构造 --------------------------------------------------------

    def _build_body(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        msgs: list[dict[str, Any]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        for m in messages:
            entry: dict[str, Any] = {"role": m.role, "content": m.content}
            # 工具调用回传 / 发起字段：仅在填写时携带，普通消息请求体不变
            if m.tool_call_id is not None:
                entry["tool_call_id"] = m.tool_call_id
            if m.name is not None:
                entry["name"] = m.name
            if m.tool_calls is not None:
                entry["tool_calls"] = m.tool_calls
            msgs.append(entry)
        body: dict[str, Any] = {
            "model": self.model,
            "messages": msgs,
            "stream": True,
            # OpenAI 协议下 usage 需显式开启；不兼容端点忽略该字段
            "stream_options": {"include_usage": True},
        }
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if tools:
            body["tools"] = tools
        return body

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "text/event-stream"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    # ---- 连接与重试 ------------------------------------------------------

    def _backoff(self, attempt: int) -> float:
        """指数退避：base * 2^attempt + 抖动，attempt 从 0 计。"""
        return self.retry_base_delay * (2**attempt) + random.uniform(0, 0.25)

    def _open_stream(self, body: dict[str, Any]) -> httpx.Response:
        """发起请求并拿到可迭代的 SSE 响应；重试只发生在这里（首个事件产出之前）。"""
        request = self._client.build_request(
            "POST", "/chat/completions", json=body, headers=self._headers()
        )
        last_error: str | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.send(request, stream=True)
            except httpx.TransportError as exc:  # 连接错误 / 超时等传输层异常
                last_error = f"transport error: {exc!r}"
                if attempt < self.max_retries:
                    self._sleep(self._backoff(attempt))
                    continue
                break
            if response.status_code in _RETRYABLE_STATUS:
                response.close()
                last_error = f"HTTP {response.status_code}"
                if attempt < self.max_retries:
                    self._sleep(self._backoff(attempt))
                    continue
                break
            if response.is_error:  # 其余 4xx：不可重试，立即终态
                response.read()
                raise ProviderError(
                    f"上游返回不可重试状态 HTTP {response.status_code}: "
                    f"{response.text[:200]}",
                    status_code=response.status_code,
                )
            return response
        raise ProviderError(
            f"请求 {self.base_url}/chat/completions 失败（重试 {self.max_retries} 次后）："
            f"{last_error}"
        )

    # ---- SSE 解析 --------------------------------------------------------

    def stream(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> Iterator[StreamEvent]:
        body = self._build_body(messages, system=system, tools=tools)
        response = self._open_stream(body)  # 重试全部发生在首个 yield 之前
        try:
            yield from self._iter_events(response)  # 流中途出错不重试，直接抛出
        finally:
            response.close()

    def _iter_events(self, response: httpx.Response) -> Iterator[StreamEvent]:
        tool_calls_acc: dict[int, dict[str, Any]] = {}
        usage: TokenUsage | None = None
        for line in response.iter_lines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:") :].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue  # 跳过不完整/非 JSON 的 SSE 行
            chunk_usage = chunk.get("usage")
            if chunk_usage:
                usage = self._parse_usage(chunk_usage)
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    yield StreamEvent(type="text_delta", delta=delta["content"])
                if delta.get("reasoning_content"):
                    yield StreamEvent(type="reasoning_delta", delta=delta["reasoning_content"])
                for tc in delta.get("tool_calls") or []:
                    self._accumulate_tool_call(tool_calls_acc, tc)
        if tool_calls_acc:
            yield StreamEvent(
                type="tool_calls",
                tool_calls=[tool_calls_acc[i] for i in sorted(tool_calls_acc)],
            )
        if usage is not None:
            yield StreamEvent(type="usage", usage=usage)

    @staticmethod
    def _accumulate_tool_call(acc: dict[int, dict[str, Any]], tc: dict[str, Any]) -> None:
        """按 index 聚合工具调用增量：id / name 首片给出，arguments 分片拼接。"""
        index = tc.get("index", 0)
        entry = acc.setdefault(
            index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
        )
        if tc.get("id"):
            entry["id"] = tc["id"]
        if tc.get("type"):
            entry["type"] = tc["type"]
        function = tc.get("function") or {}
        if function.get("name"):
            entry["function"]["name"] = function["name"]
        if function.get("arguments"):
            entry["function"]["arguments"] += function["arguments"]

    @staticmethod
    def _parse_usage(raw: dict[str, Any]) -> TokenUsage:
        """provider 返回什么记什么：cached_tokens 存在才映射，否则保持 0。"""
        details = raw.get("prompt_tokens_details") or {}
        return TokenUsage(
            input_tokens=int(raw.get("prompt_tokens") or 0),
            output_tokens=int(raw.get("completion_tokens") or 0),
            cache_read_tokens=int(details.get("cached_tokens") or 0),
        )
