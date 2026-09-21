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
    lint_parser = sub.add_parser("lint", help="wiki 健康度检查（阶段 2）")
    lint_parser.add_argument(
        "--root",
        default=os.environ.get("RESEARCHWIKI_WIKI_DATA", "wiki-data"),
        help="wiki 数据目录（默认 wiki-data，可用 RESEARCHWIKI_WIKI_DATA 覆盖）",
    )
    lint_parser.add_argument("--json", action="store_true", help="输出 JSON（便于 CI 消费）")
    sub.add_parser("arbitrate", help="冲突人工仲裁入口（阶段 3）")
    mcp_parser = sub.add_parser(
        "serve-mcp", help="启动 wiki MCP server（标准 MCP 客户端如 Claude Code 用 stdio）"
    )
    mcp_parser.add_argument(
        "--root", default=None, help="wiki 目录（默认 wiki-data，可用 RESEARCHWIKI_WIKI_DATA 覆盖）"
    )
    mcp_parser.add_argument(
        "--transport",
        default="stdio",
        choices=["stdio", "http", "sse", "streamable-http"],
        help="MCP 传输方式，默认 stdio",
    )
    mcp_parser.add_argument("--host", default="127.0.0.1", help="http/sse 传输的监听地址")
    mcp_parser.add_argument("--port", type=int, default=8765, help="http/sse 传输的监听端口")
    args = parser.parse_args(argv)

    if args.command == "serve":
        import uvicorn

        uvicorn.run(
            "researchwiki.server.main:app", host=args.host, port=args.port, reload=False
        )
        return 0

    if args.command == "serve-mcp":
        # 参数语义与 `python -m researchwiki.mcp_server` 保持单一实现，这里只做转发
        from researchwiki.mcp_server.server import main as mcp_main

        forwarded = ["--transport", args.transport, "--host", args.host, "--port", str(args.port)]
        if args.root:
            forwarded += ["--root", args.root]
        return mcp_main(forwarded)

    if args.command == "lint":
        # 延迟导入：lint 之外的命令不需要拉起 wiki 子系统
        import json

        from researchwiki.wiki.lint import lint_wiki
        from researchwiki.wiki.store import WikiStore

        store = WikiStore(args.root)
        report = lint_wiki(store)
        if args.json:
            print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(report.format_text(root=str(args.root)))
        return report.exit_code()

    print(f"「{args.command}」在后续阶段实现，见 PLAN.md 对应任务。", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
