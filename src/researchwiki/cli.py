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


def _run_maintenance(store, config, consolidation_settings, *, as_json: bool) -> int:
    """consolidate --run：按 [consolidation] 的计划逐动作执行（[maintenance] 接线）。

    逃生阀与不接线语义（Global Constraints 逐字）：[maintenance] 段缺失 = 不接线 =
    整段无操作（退出 1，与 [consolidation] 缺段同口径）；enabled = false 即完全
    禁用（零动作零留痕，退出 0）。有 failed 时退出 1，供脚本感知部分失败。
    """
    import json

    from researchwiki.llm.router import ModelRouter
    from researchwiki.wiki.consolidation import plan_consolidation
    from researchwiki.wiki.maintenance import MaintenanceRunner, maintenance_settings

    maintenance = maintenance_settings(config)
    if maintenance is None:
        print(
            "[maintenance] 段未配置（config.toml 缺段 = 不接线）："
            "consolidate --run 不做任何事。",
            file=sys.stderr,
        )
        return 1
    if not maintenance.enabled:
        print("maintenance 已禁用（enabled=false）：维护执行零动作零留痕。")
        return 0
    plan = plan_consolidation(store, consolidation_settings)
    runner = MaintenanceRunner(store, ModelRouter(config), maintenance)
    records = runner.run(plan)
    counts: dict[str, int] = {}
    for record in records:
        counts[record.status] = counts.get(record.status, 0) + 1
    summary = "、".join(f"{name} {counts.get(name, 0)}" for name in ("done", "failed", "skipped"))
    if as_json:
        print(
            json.dumps(
                {"scanned": plan.scanned, "jobs": [r.to_dict() for r in records]},
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(
            f"维护执行（root={store.root}）：候选 {len(records)} 个 → {summary}"
            if records
            else f"维护执行（root={store.root}）：无维护候选，零动作。"
        )
        for record in records:
            line = (
                f"  {record.job_id} [{record.status:>7}] {record.action:<8} "
                f"{'、'.join(record.note_ids)}"
            )
            if record.error:
                line += f"  错误：{record.error}"
            print(line)
    return 1 if counts.get("failed") else 0


def _print_jobs(runner, *, status: str | None, as_json: bool) -> int:
    """--jobs：列出维护 job 账本（可按状态过滤）。"""
    import json

    records = runner.list_jobs(status=status)
    if as_json:
        print(json.dumps([record.to_dict() for record in records], ensure_ascii=False, indent=2))
        return 0
    label = f"（status={status}）" if status else ""
    print(f"维护 job 账本（root={runner.store.root}）{label}：共 {len(records)} 条")
    for record in records:
        line = (
            f"  {record.job_id} [{record.status:>7}] {record.action:<8} "
            f"{'、'.join(record.note_ids)} attempt={record.attempt} "
            f"tokens={record.tokens_in}/{record.tokens_out}"
        )
        if record.error:
            line += f"  错误：{record.error}"
        print(line)
    return 0


def _retry_job(runner, job_id: str, *, as_json: bool) -> int:
    """--retry：重试 failed 的 job（退出码跟随重试结果：done 0 / failed 1）。"""
    import json

    try:
        record = runner.retry_job(job_id)
    except ValueError as exc:
        print(f"重试失败：{exc}", file=sys.stderr)
        return 1
    if as_json:
        print(json.dumps(record.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(
            f"重试完成：{record.job_id} [{record.status}] {record.action} "
            f"{'、'.join(record.note_ids)} attempt={record.attempt}"
        )
        if record.error:
            print(f"  错误：{record.error}")
    return 0 if record.status == "done" else 1


def _skip_job(runner, job_id: str, reason: str, *, as_json: bool) -> int:
    """--skip --reason：跳过 pending/failed 的 job，理由留痕进 job 记录。"""
    import json

    try:
        record = runner.skip_job(job_id, reason)
    except ValueError as exc:
        print(f"跳过失败：{exc}", file=sys.stderr)
        return 1
    if as_json:
        print(json.dumps(record.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(f"已跳过：{record.job_id}（reason={reason}）")
    return 0


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
        help="执行维护动作：按 [consolidation] 计划逐动作执行（[maintenance] 接线）",
    )
    consolidate_run_group.add_argument(
        "--jobs",
        action="store_true",
        help="查看维护 job 账本（可配 --status 过滤）",
    )
    consolidate_run_group.add_argument(
        "--retry",
        metavar="JOB_ID",
        default=None,
        help="重试指定 job（仅 failed 可重试；按原计划参数重放，attempt+1）",
    )
    consolidate_run_group.add_argument(
        "--skip",
        metavar="JOB_ID",
        default=None,
        help="跳过指定 job（仅 pending/failed 可跳过；需配 --reason 留痕）",
    )
    consolidate_parser.add_argument(
        "--status",
        default=None,
        help="--jobs 的状态过滤：pending / running / done / failed / skipped",
    )
    consolidate_parser.add_argument(
        "--reason",
        default=None,
        help="--skip 的跳过理由（必填，写入 job 记录留痕）",
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

        from researchwiki.llm.router import ModelRouter
        from researchwiki.wiki.consolidation import consolidation_settings, plan_consolidation
        from researchwiki.wiki.maintenance import (
            MaintenanceRunner,
            MaintenanceSettings,
            maintenance_settings,
        )
        from researchwiki.wiki.store import WikiStore

        # 跨参数依赖校验（argparse 的互斥组表达不了）
        if args.skip and not args.reason:
            print("--skip 需要配合 --reason 提供跳过理由（审计留痕）。", file=sys.stderr)
            return 1
        if args.status and not args.jobs:
            print("--status 需要配合 --jobs 使用。", file=sys.stderr)
            return 1

        store = WikiStore(args.root)
        config = load_config()

        # --jobs / --retry / --skip：对既有账本的运维动作。不要求 [maintenance] 段
        # 存在——账本可能产生于段被移除之前，运维不该被配置缺失挡在门外；
        # 缺段时用默认 settings（retry 的 rejudge 档位回退 cheap）。
        if args.jobs or args.retry or args.skip:
            runner = MaintenanceRunner(
                store,
                ModelRouter(config),
                maintenance_settings(config) or MaintenanceSettings(),
            )
            if args.jobs:
                return _print_jobs(runner, status=args.status, as_json=args.json)
            if args.retry:
                return _retry_job(runner, args.retry, as_json=args.json)
            return _skip_job(runner, args.skip, args.reason, as_json=args.json)

        # 配置接线与 lint 同手法：load_config() 宽容读（缺失/损坏回退空配置），
        # consolidation_settings 对缺 [consolidation] 段返回 None = 不接线。
        settings = consolidation_settings(config)
        if settings is None:
            print(
                "[consolidation] 段未配置（config.toml 缺段 = 不接线）："
                "consolidate 不做任何事。",
                file=sys.stderr,
            )
            return 1

        if args.run:
            return _run_maintenance(store, config, settings, as_json=args.json)

        # dry-run 是默认路径（--dry-run 仅为显式声明）：纯选择零写盘
        plan = plan_consolidation(store, settings)
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
