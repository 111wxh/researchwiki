"""Research 子 agent：独立 Message 上下文 + 独立 token 预算 + 固定 JSON 返回 schema。

主循环通过 dispatch_research 工具派发子 agent；子 agent 内部自跑"带 tools 的多步
小循环"（MVP 不单设规划步，研究指令即计划）。关键约束：
- 中间步骤（工具调用、工具结果）只存在于子 agent 自己的消息列表，绝不回流主循环；
- 超过自身 max_steps / token_budget 即收尾，把已有信息整理成 JSON 交回；
- 固定返回 schema：{"findings": str, "notes": [...], "sources": [...]}。
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from researchwiki.llm.accounting import TokenAccountant
from researchwiki.llm.provider import Message, Provider, TokenUsage
from researchwiki.loop.registry import ToolRegistry

SUBAGENT_SYSTEM = (
    "你是「自进化研究 Wiki」的研究子 agent，独立完成一个子问题的检索与阅读。\n"
    "可用工具：web_search(query, max_results?) 检索；fetch_url(url) 精读网页正文。\n"
    "规则：围绕主题做 1-3 轮检索与精读，不要偏离主题；引用的信息标注来源 URL。\n"
    "研究完成后输出严格 JSON（不要输出任何其他文本，不要用代码围栏以外的说明文字）：\n"
    '{"findings": "要点综述（300 字内，关键结论附 [来源URL]）", '
    '"notes": [{"text": "单一事实，不超过 80 字", "entities": ["实体"], '
    '"confidence": "high|medium|low"}], '
    '"sources": [{"url": "https://...", "title": "标题"}]}'
)


def extract_json(text: str) -> dict[str, Any] | None:
    """从模型输出提取第一个完整 JSON 对象（容忍 ``` 围栏与前后缀说明文字）。"""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    # 括号配对扫描：找到第一个能整体解析的 {...} 块
    start = cleaned.find("{")
    while start != -1:
        depth, in_str, escaped = 0, False, False
        for i in range(start, len(cleaned)):
            ch = cleaned[i]
            if in_str:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(cleaned[start : i + 1])
                    except json.JSONDecodeError:
                        break
                    return obj if isinstance(obj, dict) else None
        start = cleaned.find("{", start + 1)
    return None


@dataclass
class SubagentResult:
    """子 agent 的固定返回：主循环只看这个结构，不接触其内部消息。"""

    topic: str
    findings: str
    notes: list[dict[str, Any]] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    steps: int = 0
    raw: str = ""
    stopped_by_budget: bool = False


class ResearchSubagent:
    """一次子研究任务：独立上下文 + 独立预算的多步工具循环，产出结构化结果。"""

    def __init__(
        self,
        *,
        topic: str,
        brief: str = "",
        provider: Provider,
        registry: ToolRegistry,
        accountant: TokenAccountant | None = None,
        trace_id: str = "",
        max_steps: int = 6,
        token_budget: int = 50_000,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.topic = topic
        self.brief = brief
        self.provider = provider
        self.registry = registry
        self.accountant = accountant
        self.trace_id = trace_id
        self.max_steps = max_steps
        self.token_budget = token_budget
        self.clock = clock
        self.input_tokens = 0

    # ---- 内部 ----------------------------------------------------------

    def _record(self, step: str, usage: TokenUsage | None, t0: float) -> None:
        if self.accountant is None:
            return
        self.accountant.record(
            trace_id=self.trace_id,
            step=step,
            model=self.provider.model,
            usage=usage or TokenUsage(),
            latency_ms=(self.clock() - t0) * 1000.0,
        )

    # ---- 主流程 ----------------------------------------------------------

    def run(self) -> SubagentResult:
        """跑完子研究并返回结构化结果；任何一步都不触碰主循环的消息历史。"""
        brief = f"研究主题：{self.topic}"
        if self.brief.strip():
            brief += f"\n补充要求：{self.brief.strip()}"
        messages: list[Message] = [Message(role="user", content=brief)]
        schemas = self.registry.schemas()

        steps_used = 0
        stopped_by_budget = False
        last_text = ""
        for _ in range(self.max_steps):
            if self.input_tokens >= self.token_budget:
                stopped_by_budget = True
                break
            t0 = self.clock()
            parts: list[str] = []
            tool_calls: list[dict[str, Any]] | None = None
            usage: TokenUsage | None = None
            for ev in self.provider.stream(messages, system=SUBAGENT_SYSTEM, tools=schemas):
                if ev.type == "text_delta":
                    parts.append(ev.delta)
                elif ev.type == "tool_calls" and ev.tool_calls:
                    tool_calls = ev.tool_calls
                elif ev.type == "usage" and ev.usage is not None:
                    usage = ev.usage
            last_text = "".join(parts)
            steps_used += 1
            if usage is not None:
                self.input_tokens += usage.input_tokens
            self._record(f"subagent:step:{steps_used}", usage, t0)

            if not tool_calls:
                break  # 纯文本结尾：子研究完成
            messages.append(
                Message(role="assistant", content=last_text, tool_calls=tool_calls)
            )
            for tc in tool_calls:
                fn = tc.get("function") or {}
                name = str(fn.get("name") or "unknown")
                result = self.registry.dispatch(name, fn.get("arguments") or "{}")
                messages.append(
                    Message(
                        role="tool",
                        content=result,
                        tool_call_id=str(tc.get("id") or ""),
                        name=name,
                    )
                )

        parsed = extract_json(last_text) or {}
        findings = str(parsed.get("findings") or "").strip() or last_text.strip()
        notes = [
            n
            for n in (parsed.get("notes") or [])
            if isinstance(n, dict) and str(n.get("text") or "").strip()
        ]
        sources = [
            {"url": str(s.get("url") or ""), "title": str(s.get("title") or "")}
            for s in (parsed.get("sources") or [])
            if isinstance(s, dict) and str(s.get("url") or "").strip()
        ]
        return SubagentResult(
            topic=self.topic,
            findings=findings,
            notes=notes,
            sources=sources,
            steps=steps_used,
            raw=last_text,
            stopped_by_budget=stopped_by_budget,
        )
