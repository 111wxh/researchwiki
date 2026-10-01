"""memory_recall 的 budget-aware 升级测试（P3 Dynamic Retrieval）。

层级与 tests/test_mcp.py 同一约定：
1. **逻辑层**：直接调 ``WikiService.recall``，断言模式判定、policy 透传与
   召回宽度收窄；fixture 三件套（wiki_root/store/service/server）沿用同款风格。
2. **协议层**：用 FastMCP 的 in-process Client 走一次完整 MCP 往返，
   确认 memory_recall 工具签名与 payload 透传不变（升级只动 service 层）。

与 test_mcp.py 的差别只有一处：CONFIG 显式带 ``retrieval = {enabled = true}``，
让 recall 走判定路径；retrieval 段缺席时的 passthrough 基线由
test_mcp.py::test_recall_annotates_memory_state 锁定，这里不重复。
零真实网络（embedding 走 mock）、tmp_path 隔离；笔记直接用 ``store.save_note``
落盘，recall 内部的 search 会按共享判据（index_drift）发现索引落后并重建。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client, FastMCP

from researchwiki.mcp_server import build_server
from researchwiki.mcp_server.service import WikiService
from researchwiki.wiki.store import WikiStore

# trigram：确定性、不探测 vendor DLL（同 test_mcp.py）；retrieval 显式启用判定路径
CONFIG = {"wiki": {"fts_tokenizer": "trigram"}, "retrieval": {"enabled": True}}


def _tool_call(server: FastMCP, name: str, arguments: dict[str, Any] | None = None) -> dict:
    """在独立事件循环里做一次 in-process MCP 工具调用，返回解析后的 JSON payload。"""

    async def run() -> Any:
        async with Client(server) as client:
            return await client.call_tool(name, arguments or {})

    result = asyncio.run(run())
    assert result.is_error is False, f"结构化错误不应变成 tool error：{result.content}"
    return json.loads(result.content[0].text)


@pytest.fixture()
def wiki_root(tmp_path: Path) -> Path:
    return tmp_path / "wiki-data"


@pytest.fixture()
def store(wiki_root: Path) -> WikiStore:
    return WikiStore(wiki_root)


@pytest.fixture()
def service(wiki_root: Path) -> WikiService:
    return WikiService(wiki_root, config=CONFIG)


@pytest.fixture()
def server(wiki_root: Path) -> FastMCP:
    return build_server(root=wiki_root, config=CONFIG)


def test_recall_reports_mode_features_reasons(service: WikiService, store: WikiStore) -> None:
    """启用 retrieval 后 recall 必须给出模式 + 判定理由 + 特征（不允许只有模式字符串）。"""
    store.save_note("Zephyr 内存占用约 2KB，来源官方文档。", entities=["Zephyr"])
    payload = service.recall("Zephyr 内存占用", k=5)
    assert payload["ok"] is True
    assert payload["mode"] in {"simple", "update", "deep"}
    assert payload["policy"]["reasons"]
    assert "hit_count" in payload["policy"]["features"]


def test_recall_simple_mode_narrows_results(service: WikiService, store: WikiStore) -> None:
    """simple 模式（prior_k=3）收窄召回宽度；update/deep 尊重调用方 k。"""
    for i in range(5):
        store.save_note(f"Zephyr 知识条目 {i}：官方规格 {i}。", entities=["Zephyr"])
    payload = service.recall("Zephyr 知识条目", k=5)  # 全 fresh stable → simple
    assert payload["ok"] is True
    if payload["mode"] == "simple":
        assert payload["count"] <= 3  # simple 模式限额 prior_k=3
        assert payload["k"] <= 3
    else:
        assert payload["count"] <= 5


def test_recall_with_conflict_routes_deep_and_still_returns_results(
    service: WikiService, store: WikiStore
) -> None:
    """open 冲突相关的问题路由 deep，且召回结果照常返回（判定不裁掉可用记忆）。"""
    store.save_note("Zephyr 内存占用约 2KB。", entities=["Zephyr"])
    store.save_conflict("Zephyr 内存占用是多少", {"text": "2KB"}, {"text": "4KB"})
    payload = service.recall("Zephyr 内存占用是多少", k=5)
    assert payload["mode"] == "deep"
    assert payload["count"] >= 1


def test_mcp_tool_surface_unchanged(server: FastMCP, store: WikiStore) -> None:
    """协议层：memory_recall 工具签名不变，payload 经 MCP 往返后 mode/policy 完整透传。"""

    async def list_tools() -> list[Any]:
        async with Client(server) as client:
            return await client.list_tools()

    tools = {t.name: t for t in asyncio.run(list_tools())}
    schema = tools["memory_recall"].input_schema
    assert schema["required"] == ["query"]
    assert set(schema["properties"]) == {"query", "k", "kind", "include_tombstones"}

    store.save_note("Zephyr 内存占用约 2KB。", entities=["Zephyr"])
    payload = _tool_call(server, "memory_recall", {"query": "Zephyr 内存占用", "k": 5})
    assert payload["ok"] is True
    assert payload["mode"] in {"simple", "update", "deep"}
    assert "policy" in payload
    hit = payload["results"][0]
    # 每条结果维持既有标注字段（P1-A 契约逐字段不变）
    assert {"note_id", "title", "snippet", "score", "match_type", "status", "kind"} <= set(hit)
