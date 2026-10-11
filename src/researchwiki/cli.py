"""开发用单命令入口（非交互式 CLI，交互走 Web UI）。"""

from __future__ import annotations

import argparse
import os
import sys
import tomllib
from pathlib import Path

# 运行配置路径：与 server.main.CONFIG_PATH 同一约定（env 覆盖，缺省项目根 config.toml）
CONFIG_PATH = Path(os.environ.get("RESEARCHWIKI_CONFIG", "config.toml"))


def load_config() -> dict:
    """读取运行配置；读不到（缺失/损坏/路径是目录/无读权限）时按空配置处理。

    与 ``server.main.load_config`` 同语义（lint 命令只需 tomllib，不值得为它
    拉起 FastAPI 那一整串导入，故此处独立实现，不 import server 模块）。
    宽容面比 server 版更大：``RESEARCHWIKI_CONFIG`` 指向目录、或无读权限时，
    "读配置"不该让 lint 整体崩掉——一律回退空配置（调用方随即按默认参数接线，
    时效统计与不接线时逐值一致）。
    """
    try:
        with CONFIG_PATH.open("rb") as f:
            return tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        # OSError 覆盖 FileNotFoundError / IsADirectoryError / PermissionError 等
        return {}


def main(argv: list[str] | None = None) -> int:
    # 先加载 .env：所有子命令（serve / lint / serve-mcp）都要能读到 key
    from researchwiki.env import load_env_file

    load_env_file()

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
    consolidate_parser = sub.add_parser(
        "consolidate",
        help="后台 consolidation：merge / refresh / rejudge / conflict 候选选择（P4a）",
    )
    consolidate_parser.add_argument(
        "--root",
        default=os.environ.get("RESEARCHWIKI_WIKI_DATA", "wiki-data"),
        help="wiki 数据目录（默认 wiki-data，可用 RESEARCHWIKI_WIKI_DATA 覆盖）",
    )
    consolidate_parser.add_argument(
        "--json", action="store_true", help="输出 JSON（plan 的 dict 形态，便于 CI 消费）"
    )
    consolidate_run_group = consolidate_parser.add_mutually_exclusive_group()
    consolidate_run_group.add_argument(
        "--dry-run",
        action="store_true",
        help="只选择零写盘（默认行为；此参数仅为显式声明）",
    )
    consolidate_run_group.add_argument(
        "--run",
        action="store_true",
        help="执行维护动作（执行器在 T3 接线；当前为桩）",
    )
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

    if args.command == "consolidate":
        # 延迟导入：consolidate 之外的命令不需要拉起 wiki 子系统
        import json

        from researchwiki.wiki.consolidation import consolidation_settings, plan_consolidation
        from researchwiki.wiki.store import WikiStore

        if args.run:
            # T17 会接走执行器；当前保留桩语义（返回 1，零写盘）
            print("执行器在 T3 接线：consolidate --run 尚未实现，本次未做任何事。", file=sys.stderr)
            return 1
        # 配置接线与 lint 同手法：load_config() 宽容读（缺失/损坏回退空配置），
        # consolidation_settings 对缺 [consolidation] 段返回 None = 不接线。
        settings = consolidation_settings(load_config())
        if settings is None:
            print(
                "[consolidation] 段未配置（config.toml 缺段 = 不接线）："
                "consolidate 不做任何事。",
                file=sys.stderr,
            )
            return 1
        # dry-run 是默认路径（--dry-run 仅为显式声明）：纯选择零写盘
        plan = plan_consolidation(WikiStore(args.root), settings)
        if args.json:
            print(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(plan.format_text(root=str(args.root)))
        return 0

    if args.command == "lint":
        # 延迟导入：lint 之外的命令不需要拉起 wiki 子系统
        import json

        from researchwiki.wiki.freshness import from_config as freshness_from_config
        from researchwiki.wiki.lint import lint_wiki
        from researchwiki.wiki.store import WikiStore

        store = WikiStore(args.root)
        # 时效参数接线：freshness.from_config 读 [freshness] 段，半衰期回退 [wiki]
        # 段（与检索层 index.wiki_settings 同一张表）。config.toml 缺失/无该段
        # 时得到全默认参数，时效统计行为与接线前逐值一致。
        report = lint_wiki(store, freshness_settings=freshness_from_config(load_config()))
        if args.json:
            print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(report.format_text(root=str(args.root)))
        return report.exit_code()

    print(f"「{args.command}」在后续阶段实现，见 PLAN.md 对应任务。", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
