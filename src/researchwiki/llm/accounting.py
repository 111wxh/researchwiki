"""Token 记账：每次 LLM 调用追加一行 JSONL。

复用收益"token 降低 X%"的原始凭证；cache 字段 provider 返回什么记什么。
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from researchwiki.llm.provider import TokenUsage


class TokenAccountant:
    def __init__(self, path: str | Path = "wiki-data/tokens.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def new_trace_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def record(
        self,
        *,
        trace_id: str,
        step: str,
        model: str,
        usage: TokenUsage,
        latency_ms: float,
        error_type: str | None = None,
    ) -> dict:
        row = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "trace_id": trace_id,
            "step": step,
            "model": model,
            **asdict(usage),
            "latency_ms": round(latency_ms, 1),
            "error_type": error_type,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row
