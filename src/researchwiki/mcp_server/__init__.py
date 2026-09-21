"""wiki MCP server：把研究 wiki 装进 Claude Code 等 MCP 客户端。

对外入口：
- ``build_server(root=..., config=...)`` 装配 FastMCP server（测试可用 tmp_path 隔离）；
- ``WikiService`` 服务层：检索/读取/写入/增量列表/健康检查，返回 JSON 友好的 dict；
- ``main(argv)`` / ``python -m researchwiki.mcp_server``：stdio（默认）或 http 启动。

工具：wiki_search / wiki_read / wiki_write / wiki_list_changes / wiki_health。
"""

from researchwiki.mcp_server.server import (
    SERVER_INSTRUCTIONS,
    SERVER_NAME,
    build_server,
    main,
    resolve_wiki_root,
)
from researchwiki.mcp_server.service import WikiService, error_payload

__all__ = [
    "SERVER_INSTRUCTIONS",
    "SERVER_NAME",
    "WikiService",
    "build_server",
    "error_payload",
    "main",
    "resolve_wiki_root",
]
