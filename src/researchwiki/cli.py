"""开发用单命令入口（非交互式 CLI，交互走 Web UI）。"""

from __future__ import annotations

import argparse
import os
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="researchwiki", description="自进化研究 Wiki 智能体")
    sub = parser.add_subparsers(dest="command", required=True)
    serve_parser = sub.add_parser("serve", help="启动 FastAPI server（SSE）")
    # 默认只监听本机；容器/服务器部署时用参数或环境变量放开到 0.0.0.0
    serve_parser.add_argument(
        "--host", default=os.environ.get("RESEARCHWIKI_HOST", "127.0.0.1")
    )
    serve_parser.add_argument(
        "--port", type=int, default=int(os.environ.get("RESEARCHWIKI_PORT", "8000"))
    )
    sub.add_parser("consolidate", help="后台 consolidation：merge / refresh / conflict（阶段 3）")
    sub.add_parser("lint", help="wiki 健康度检查（阶段 2）")
    sub.add_parser("arbitrate", help="冲突人工仲裁入口（阶段 3）")
    sub.add_parser("serve-mcp", help="启动 wiki MCP server（阶段 2）")
    args = parser.parse_args(argv)

    if args.command == "serve":
        import uvicorn

        uvicorn.run(
            "researchwiki.server.main:app", host=args.host, port=args.port, reload=False
        )
        return 0

    print(f"「{args.command}」在后续阶段实现，见 PLAN.md 对应任务。", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
