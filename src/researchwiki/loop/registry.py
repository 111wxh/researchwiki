"""工具注册表：agent loop 的"手脚"统一登记、出 schema、按名分发。

设计要点：
- 工具 = 名字 + OpenAI function schema + 处理函数（args dict -> 结果字符串）；
- dispatch 统一捕获异常并折算成 {"error": ...} JSON 回传模型，单工具失败不炸整个 loop；
- 所有工具结果统一截断（PLAN.md 阶段 2 硬约定：不做这个复杂问题就会撑爆上下文）。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

Handler = Callable[[dict[str, Any]], str]


@dataclass
class Tool:
    """一个可供模型调用的工具。handler 入参是已解析的 arguments dict，返回值是结果文本。"""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Handler


@dataclass
class ToolRegistry:
    """具名工具集合：给 provider 出 OpenAI schema，给 loop 按名分发执行。"""

    max_result_chars: int = 12_000
    _tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        """OpenAI function calling 格式：{"type":"function","function":{...}}。"""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self._tools.values()
        ]

    def dispatch(self, name: str, arguments: str | dict[str, Any]) -> str:
        """按名执行工具，返回回传给模型的结果文本（统一 JSON + 截断）。

        任何失败（未知工具 / arguments 非法 JSON / 工具内部异常）都折算成
        {"error": ...}，让模型有机会换路重试而不是让整个 run 崩掉。
        """
        tool = self._tools.get(name)
        if tool is None:
            return self._error_result(f"unknown tool: {name!r}")
        if isinstance(arguments, str):
            try:
                args = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError as exc:
                return self._error_result(f"invalid arguments JSON: {exc}")
        else:
            args = arguments
        if not isinstance(args, dict):
            got = type(args).__name__
            return self._error_result(f"arguments must be a JSON object, got: {got}")
        try:
            result = tool.handler(args)
        except Exception as exc:  # noqa: BLE001 -- 工具异常统一降级为错误 JSON
            return self._error_result(f"{type(exc).__name__}: {exc}")
        return self._truncate(result)

    def _error_result(self, message: str) -> str:
        return self._truncate(json.dumps({"error": message}, ensure_ascii=False))

    def _truncate(self, text: str) -> str:
        """工具结果统一截断：超限部分丢弃并附提示，模型能感知被裁剪。"""
        if len(text) <= self.max_result_chars:
            return text
        cut = text[: self.max_result_chars]
        return cut + f"\n…[工具结果已截断：仅保留前 {self.max_result_chars} 字符]"
