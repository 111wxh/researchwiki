"""工具层：web_search / fetch_url / 沙箱文件读写——真实 agent loop 的手脚。

loop 层统一从 `researchwiki.tools` 导入，不感知各 provider 细节。
"""

from __future__ import annotations

from researchwiki.tools.fetch import DEFAULT_SOURCES_DIR, FetchError, FetchResult, fetch_url
from researchwiki.tools.fs import (
    DEFAULT_WIKI_DATA_ROOT,
    SandboxError,
    atomic_write_text,
    safe_list,
    safe_read,
    safe_write,
)
from researchwiki.tools.search import (
    BochaSearch,
    MockSearch,
    SearchError,
    SearchHit,
    SearchProvider,
    TavilySearch,
    get_search_provider,
)

__all__ = [
    "DEFAULT_SOURCES_DIR",
    "DEFAULT_WIKI_DATA_ROOT",
    "BochaSearch",
    "FetchError",
    "FetchResult",
    "MockSearch",
    "SandboxError",
    "SearchError",
    "SearchHit",
    "SearchProvider",
    "TavilySearch",
    "atomic_write_text",
    "fetch_url",
    "get_search_provider",
    "safe_list",
    "safe_read",
    "safe_write",
]
