"""统一 web_search：SearchProvider 协议 + Tavily / 博查实现 + 中文 Mock 夹具。

骨架先行的同一套思路：loop 层只依赖 SearchProvider 协议与 SearchHit；
没有配置任何 API key 时，工厂返回 MockSearch，保证无 key 全链路可跑、可测。
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

TAVILY_ENDPOINT = "https://api.tavily.com/search"
BOCHA_ENDPOINT = "https://api.bochaai.com/v1/web-search"
REQUEST_TIMEOUT = 15.0

# 统一限速：所有 provider 共享最近一次请求时间戳，两次请求间隔不小于 min_interval
_rate_lock = threading.Lock()
_last_request_at = 0.0


class SearchError(RuntimeError):
    """搜索失败：重试耗尽仍网络错误、HTTP 状态异常或响应无法解析。"""

    def __init__(self, provider: str, reason: str) -> None:
        self.provider = provider
        self.reason = reason
        super().__init__(f"[{provider}] search failed: {reason}")


@dataclass
class SearchHit:
    title: str
    url: str
    snippet: str


class SearchProvider(Protocol):
    def search(self, query: str, max_results: int = 5) -> list[SearchHit]: ...


def _throttle(min_interval: float) -> None:
    """模块级统一限速：距上次请求不足 min_interval 秒则等待。"""
    global _last_request_at
    with _rate_lock:
        wait = min_interval - (time.monotonic() - _last_request_at)
        if wait > 0:
            time.sleep(wait)
        _last_request_at = time.monotonic()


def _post_with_retry(
    client: httpx.Client,
    provider: str,
    url: str,
    *,
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
    retries: int = 2,
    backoff: float = 0.5,
) -> httpx.Response:
    """POST 并对网络错误做指数退避重试（共 retries+1 次尝试）。"""
    last_error = ""
    for attempt in range(retries + 1):
        try:
            return client.post(url, json=payload, headers=headers)
        except httpx.TransportError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries and backoff > 0:
                time.sleep(backoff * (2**attempt))
    raise SearchError(provider, f"network error after {retries + 1} attempts: {last_error}")


def _parse_hits(
    raw: list[dict[str, Any]], *, title: str, url: str, snippet: str
) -> list[SearchHit]:
    return [
        SearchHit(
            title=str(r.get(title, "")),
            url=str(r.get(url, "")),
            snippet=str(r.get(snippet, "")),
        )
        for r in raw
    ]


class TavilySearch:
    """Tavily 搜索：POST {api_key, query, max_results}，解析 results[].{title,url,content}。"""

    def __init__(
        self,
        api_key: str,
        *,
        min_interval: float = 1.0,
        retries: int = 2,
        backoff: float = 0.5,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.api_key = api_key
        self.min_interval = min_interval
        self.retries = retries
        self.backoff = backoff
        self.transport = transport

    def search(self, query: str, max_results: int = 5) -> list[SearchHit]:
        _throttle(self.min_interval)
        payload: dict[str, Any] = {
            "api_key": self.api_key,
            "query": query,
            "max_results": max_results,
        }
        with httpx.Client(transport=self.transport, timeout=REQUEST_TIMEOUT) as client:
            resp = _post_with_retry(
                client,
                "tavily",
                TAVILY_ENDPOINT,
                payload=payload,
                retries=self.retries,
                backoff=self.backoff,
            )
        if resp.status_code != 200:
            raise SearchError("tavily", f"HTTP {resp.status_code}: {resp.text[:200]}")
        try:
            data: dict[str, Any] = resp.json()
        except ValueError as exc:
            raise SearchError("tavily", f"invalid JSON response: {exc}") from exc
        hits = _parse_hits(data.get("results", []), title="title", url="url", snippet="content")
        return hits[:max_results]


class BochaSearch:
    """博查搜索：Bearer token 鉴权，解析 data.webPages.value[].{name,url,snippet}。"""

    def __init__(
        self,
        api_key: str,
        *,
        min_interval: float = 1.0,
        retries: int = 2,
        backoff: float = 0.5,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.api_key = api_key
        self.min_interval = min_interval
        self.retries = retries
        self.backoff = backoff
        self.transport = transport

    def search(self, query: str, max_results: int = 5) -> list[SearchHit]:
        _throttle(self.min_interval)
        payload: dict[str, Any] = {"query": query, "count": max_results, "summary": True}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        with httpx.Client(transport=self.transport, timeout=REQUEST_TIMEOUT) as client:
            resp = _post_with_retry(
                client,
                "bocha",
                BOCHA_ENDPOINT,
                payload=payload,
                headers=headers,
                retries=self.retries,
                backoff=self.backoff,
            )
        if resp.status_code != 200:
            raise SearchError("bocha", f"HTTP {resp.status_code}: {resp.text[:200]}")
        try:
            data: dict[str, Any] = resp.json()
        except ValueError as exc:
            raise SearchError("bocha", f"invalid JSON response: {exc}") from exc
        pages = ((data.get("data") or {}).get("webPages") or {}).get("value", [])
        hits = _parse_hits(pages, title="name", url="url", snippet="snippet")
        return hits[:max_results]


# 内置中文夹具（agent 记忆相关主题），与 loop 层 mock 演示的场景一致
_MOCK_FIXTURES: tuple[tuple[str, str, str], ...] = (
    (
        "Letta (MemGPT)：操作系统式的 agent 记忆",
        "https://github.com/letta-ai/letta",
        "Letta 通过后台 subagent 在会话间整理记忆（sleep-time compute），"
        "把记忆维护成本移出交互窗口。",
    ),
    (
        "Mem0：可扩展的 agent 记忆层",
        "https://github.com/mem0ai/mem0",
        "Mem0 的记忆流水线包含 add / update / merge 三类操作，写入前做相似度查重。",
    ),
    (
        "滚动 compaction：上下文压缩的工程实践",
        "https://www.anthropic.com/engineering/claude-code-best-practices",
        "上下文占用超过窗口 70% 时触发滚动压缩，早期轮次折叠进状态文件，保持 KV-cache 前缀稳定。",
    ),
    (
        "Anthropic Prompt Caching 官方文档",
        "https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching",
        "prompt caching 需要显式设置 cache_control 断点；OpenAI 兼容端点依赖前缀稳定。",
    ),
    (
        "sqlite-vec：嵌入式向量检索",
        "https://github.com/asg017/sqlite-vec",
        "单文件零运维的向量检索方案，适合个人 wiki 规模的关键词+向量混合检索。",
    ),
)


class MockSearch:
    """返回内置中文夹具（3-5 条），零 key 零网络；单测与无 key 演示用。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, max_results: int = 5) -> list[SearchHit]:
        self.calls.append((query, max_results))
        return [SearchHit(title=t, url=u, snippet=s) for t, u, s in _MOCK_FIXTURES[:max_results]]


def get_search_provider(config: Mapping[str, Any] | None = None) -> SearchProvider:
    """按配置选择搜索 provider；找不到任何可用 key 时返回 MockSearch。

    读取 config["search"]（与 config.toml 的节对应）：provider / tavily_api_key /
    bocha_api_key / api_key；环境变量 SEARCH_PROVIDER、TAVILY_API_KEY、BOCHA_API_KEY
    作为兜底。指定了 provider 但 key 缺失时同样回退 MockSearch（不抛错、不断链路）。
    """
    search_cfg: Mapping[str, Any] = {}
    if config:
        candidate = config.get("search")
        if isinstance(candidate, Mapping):
            search_cfg = candidate

    def _cfg(key: str) -> str:
        value = search_cfg.get(key)
        return str(value) if value is not None else ""

    provider_name = (_cfg("provider") or os.getenv("SEARCH_PROVIDER") or "").strip().lower()
    tavily_key = _cfg("tavily_api_key") or os.getenv("TAVILY_API_KEY") or ""
    bocha_key = _cfg("bocha_api_key") or os.getenv("BOCHA_API_KEY") or ""

    if provider_name == "tavily":
        key = _cfg("api_key") or tavily_key
        return TavilySearch(key) if key else MockSearch()
    if provider_name == "bocha":
        key = _cfg("api_key") or bocha_key
        return BochaSearch(key) if key else MockSearch()
    if not provider_name:
        if tavily_key:
            return TavilySearch(tavily_key)
        if bocha_key:
            return BochaSearch(bocha_key)
    return MockSearch()
