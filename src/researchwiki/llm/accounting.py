"""Token 记账：每次 LLM 调用追加一行 JSONL。

复用收益"token 降低 X%"的原始凭证；cache 字段 provider 返回什么记什么。

估算与计量的边界（P4b 支撑设施 T6）：本文件后半段的 ``estimate_*`` 函数是
**估算非计量**——给压缩触发决策用的确定性字符启发式，不经过 TokenAccountant、
不写 tokens.jsonl、不覆盖 run-metrics 任何字段；计量数字只来自 provider 的
usage 回传（上面的 ``record()``），两者严格分账。
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from researchwiki.llm.provider import Message, TokenUsage


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


# ---- 上下文 token 估算（P4b 支撑设施 T6）-------------------------------------
#
# 再强调一次边界：以下是**估算非计量**。估算服务于滚动压缩的触发决策
# （loop/compaction.py），是零网络、零模型、可复算的确定性字符启发式；
# 估算值不得进 tokens.jsonl（计量 JSONL），也不得覆盖 run-metrics 任何字段。

# 东亚文字区下界：U+2E80 起（CJK 部首/假名/谚文/汉字/中文标点/全角等）按
# 「1 token/字」估；其余字符（拉丁/数字/ASCII 符号等）按「1 token/4 字符」估。
_CJK_BOUNDARY = 0x2E80


def _estimate_text_tokens(text: str) -> int:
    """确定性字符启发式：东亚文字 1 token/字，其余 1 token/4 字符（向上取整）。

    这是粗估（真实 tokenizer 按子词切分，数字与代码会偏），只用于压缩触发的
    相对比较，不用于任何成本口径。
    """
    if not text:
        return 0
    cjk = 0
    for ch in text:
        if ord(ch) >= _CJK_BOUNDARY:
            cjk += 1
    other = len(text) - cjk
    return cjk + (other + 3) // 4


def _message_payload(msg: Message | dict[str, Any]) -> dict[str, Any]:
    """接受 provider.Message 或其 dict 形态（估算接口对两种输入都开放）。"""
    if isinstance(msg, dict):
        return msg
    return asdict(msg)


def estimate_message_tokens(msg: Message | dict[str, Any]) -> int:
    """估算单条消息的 token 数（**估算非计量**，见模块 docstring 的边界说明）。

    覆盖 content 与 tool_calls 载荷（OpenAI function 格式的 arguments JSON）
    以及 role / name / tool_call_id 定长字段；确定性、可复算。
    """
    data = _message_payload(msg)
    total = _estimate_text_tokens(str(data.get("content") or ""))
    for key in ("role", "name", "tool_call_id"):
        total += _estimate_text_tokens(str(data.get(key) or ""))
    tool_calls = data.get("tool_calls")
    if tool_calls:
        # sort_keys 保证同一载荷（无论 dict 构造顺序如何）估算恒一致
        total += _estimate_text_tokens(json.dumps(tool_calls, ensure_ascii=False, sort_keys=True))
    return total


def estimate_tools_tokens(tools: list[dict[str, Any]] | None) -> int:
    """估算工具 schema 列表的 token 数（**估算非计量**）。"""
    if not tools:
        return 0
    return sum(
        _estimate_text_tokens(json.dumps(tool, ensure_ascii=False, sort_keys=True))
        for tool in tools
    )


def estimate_context_tokens(
    messages: list[Message | dict[str, Any]] | None,
    tools: list[dict[str, Any]] | None = None,
) -> int:
    """估算一次模型调用的上下文 token（消息 + 工具 schema；**估算非计量**）。"""
    total = sum(estimate_message_tokens(m) for m in messages or [])
    return total + estimate_tools_tokens(tools)
