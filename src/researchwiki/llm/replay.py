"""ReplayProvider：请求哈希级 JSONL 缓存，开发/评测成本可控可重放。

首次真实调用：透传 wrapped Provider 的事件流，结束后把
「请求哈希 → 合并完整文本 + tool_calls + usage」追加写入 JSONL。
命中缓存：直接回放（逐块 text_delta + tool_calls + usage），不发网络请求。
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from researchwiki.llm.provider import Message, StreamEvent, TokenUsage, chunk_text

# 回放时文本切块大小（字符），模拟流式增量
_REPLAY_CHUNK_SIZE = 48


def request_hash(
    model: str,
    messages: list[Message],
    *,
    system: str | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> str:
    """请求哈希 = sha256(model + messages(规范化 json) + tools)。

    system 提示折叠进消息列表参与规范化，保证同请求同哈希、异请求异哈希。
    """
    msgs: list[dict[str, Any]] = []
    if system:
        msgs.append({"role": "system", "content": system})
    for m in messages:
        entry: dict[str, Any] = {"role": m.role, "content": m.content}
        # 工具消息字段参与规范化：不填时与旧哈希完全一致（不影响既有缓存）
        if m.tool_call_id is not None:
            entry["tool_call_id"] = m.tool_call_id
        if m.name is not None:
            entry["name"] = m.name
        if m.tool_calls is not None:
            entry["tool_calls"] = m.tool_calls
        msgs.append(entry)
    payload = {"model": model, "messages": msgs, "tools": tools}
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class ReplayProvider:
    """包装任意 Provider，按请求哈希做 JSONL 透写缓存与回放。"""

    def __init__(self, inner: Any, cache_path: str | Path = "wiki-data/replay.jsonl") -> None:
        self.inner = inner
        self.model = inner.model
        self.tier = inner.tier
        self.path = Path(cache_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, dict[str, Any]] | None = None

    # ---- 缓存读写 --------------------------------------------------------

    def _load(self) -> dict[str, dict[str, Any]]:
        if self._cache is None:
            cache: dict[str, dict[str, Any]] = {}
            if self.path.exists():
                with self.path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            record = json.loads(line)
                            cache[record["hash"]] = record
            self._cache = cache
        return self._cache

    def _append(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._load()[record["hash"]] = record

    # ---- 主流程 ----------------------------------------------------------

    def stream(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> Iterator[StreamEvent]:
        digest = request_hash(self.model, messages, system=system, tools=tools)
        cached = self._load().get(digest)
        if cached is not None:
            yield from self._replay(cached)
            return

        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] | None = None
        usage: TokenUsage | None = None
        for event in self.inner.stream(messages, system=system, tools=tools):
            if event.type == "text_delta":
                text_parts.append(event.delta)
            elif event.type == "tool_calls":
                tool_calls = event.tool_calls
            elif event.type == "usage" and event.usage is not None:
                usage = event.usage
            yield event

        self._append(
            {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "hash": digest,
                "model": self.model,
                "text": "".join(text_parts),
                "tool_calls": tool_calls,
                "usage": (
                    {
                        "input_tokens": usage.input_tokens,
                        "output_tokens": usage.output_tokens,
                        "cache_read_tokens": usage.cache_read_tokens,
                        "cache_creation_tokens": usage.cache_creation_tokens,
                    }
                    if usage is not None
                    else None
                ),
            }
        )

    def _replay(self, record: dict[str, Any]) -> Iterator[StreamEvent]:
        for piece in chunk_text(record.get("text") or "", _REPLAY_CHUNK_SIZE):
            yield StreamEvent(type="text_delta", delta=piece)
        if record.get("tool_calls"):
            yield StreamEvent(type="tool_calls", tool_calls=record["tool_calls"])
        usage = record.get("usage")
        if usage:
            yield StreamEvent(type="usage", usage=TokenUsage(**usage))
