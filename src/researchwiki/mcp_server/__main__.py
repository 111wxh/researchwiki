"""``python -m researchwiki.mcp_server`` 入口：默认 stdio 传输。

stdout 只走 MCP 协议报文，日志/提示一律 stderr（MCP 客户端按行解析 stdout）。
"""

from researchwiki.mcp_server.server import main

if __name__ == "__main__":
    raise SystemExit(main())
