"""真模型端到端冒烟：真实 LLM 驱动完整 agent loop，产出可复算的证据。

用途（PLAN 阶段 2 的验收项「真模型冒烟」）：
    跑一个陌生主题的完整研究，落盘报告，并给出按步骤的 token 记账，
    用于核对「一次研究的成本可从 JSONL 精确复算」。

用法：
    uv run researchwiki real-run --question "MCP 的传输机制与安全边界"
    python scripts/real_run_smoke.py --question "..." --max-steps 8 --budget 60000

前置：项目根 .env 里配好 key（见 .env.example）；搜索未配 key 时回退 MockSearch，
此时来源 URL 由模型自行给出（模型仍会真实抓取，但检索这一步不是全网搜索）。

产物：默认写到临时目录（不污染仓库 wiki-data），用 --out 指定输出目录可保留报告。
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

from researchwiki.env import load_env_file

load_env_file()

from researchwiki.llm.accounting import TokenAccountant  # noqa: E402
from researchwiki.llm.router import ModelRouter  # noqa: E402
from researchwiki.loop.agent_loop import AgentLoop  # noqa: E402
from researchwiki.tools import get_search_provider  # noqa: E402
from researchwiki.wiki.embeddings import get_embedding_provider  # noqa: E402
from researchwiki.wiki.store import WikiStore  # noqa: E402

DEFAULT_QUESTION = "MCP（Model Context Protocol）的传输机制与安全边界"


def load_config(path: Path) -> dict:
    """读 config.toml；缺失时按空配置处理（等价 mock，不炸）。"""
    if not path.is_file():
        return {}
    import tomllib

    with path.open("rb") as f:
        return tomllib.load(f)


def run(question: str, *, config_path: Path, wiki_root: Path, max_steps: int, budget: int) -> dict:
    cfg = load_config(config_path)
    accountant = TokenAccountant(path=wiki_root / "tokens.jsonl")
    loop = AgentLoop(
        question,
        router=ModelRouter(cfg.get("llm") or {}),
        llm_config=cfg.get("llm") or {},
        accountant=accountant,
        search_provider=get_search_provider(cfg),
        wiki_root=wiki_root,
        wiki_config=cfg.get("wiki") or {},
        embedding=get_embedding_provider(cfg, cache_path=wiki_root / "index.db"),
        max_steps=max_steps,
        token_budget=budget,
    )

    counts: dict[str, int] = {}
    sources: list[dict] = []
    started = time.perf_counter()
    for ev in loop.events():
        kind = str(ev.get("type") or "?")
        counts[kind] = counts.get(kind, 0) + 1
        if kind == "data-task":
            data = ev.get("data") or {}
            print(f"[task] {str(data.get('status', '?')):8s} {data.get('title', '')}", flush=True)
        elif kind == "data-note":
            data = ev.get("data") or {}
            print(f"[note] {data.get('id')} conf={data.get('confidence')}", flush=True)
        elif kind == "data-conflict":
            data = ev.get("data") or {}
            print(f"[conflict] {str(data.get('summary'))[:60]}", flush=True)
        elif kind == "source-url":
            sources.append({"url": ev.get("url"), "title": ev.get("title")})

    elapsed = time.perf_counter() - started
    report_text = getattr(loop, "report_text", "") or ""
    rows = [
        json.loads(line)
        for line in (wiki_root / "tokens.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    notes = WikiStore(wiki_root).list_notes()
    return {
        "elapsed_s": round(elapsed, 1),
        "report": report_text,
        "report_chars": len(report_text),
        "sources": sources,
        "notes": [n.id for n in notes],
        "counts": counts,
        "usage": {
            "calls": len(rows),
            "input_tokens": sum(r["input_tokens"] for r in rows),
            "output_tokens": sum(r["output_tokens"] for r in rows),
            "cache_read_tokens": sum(r.get("cache_read_tokens") or 0 for r in rows),
            "by_step": [
                {
                    "step": r["step"],
                    "model": r["model"],
                    "in": r["input_tokens"],
                    "out": r["output_tokens"],
                    "ms": r["latency_ms"],
                }
                for r in rows
            ],
        },
        "wiki_root": str(wiki_root),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="真模型端到端冒烟")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--wiki-root", default=None, help="默认用临时目录（不污染仓库）")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--budget", type=int, default=60_000, help="主循环 input token 预算")
    parser.add_argument("--out", default=None, help="输出目录（写 report.md 与 result.json）")
    parser.add_argument("--json", action="store_true", help="只输出 JSON 结果")
    args = parser.parse_args(argv)

    wiki_root = (
        Path(args.wiki_root) if args.wiki_root else Path(tempfile.mkdtemp(prefix="smoke-wiki-"))
    )
    if not args.json:
        print(f"问题: {args.question}")
        print(f"wiki_root: {wiki_root}")
        print(f"预算: {args.max_steps} 步 / {args.budget} input tokens")
        print("-" * 60, flush=True)

    result = run(
        args.question,
        config_path=Path(args.config),
        wiki_root=wiki_root,
        max_steps=args.max_steps,
        budget=args.budget,
    )

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "report.md").write_text(result["report"], encoding="utf-8")
        summary = {k: v for k, v in result.items() if k != "report"}
        (out_dir / "result.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    if args.json:
        print(json.dumps({k: v for k, v in result.items() if k != "report"}, ensure_ascii=False))
        return 0

    usage = result["usage"]
    print("-" * 60)
    print(
        f"耗时 {result['elapsed_s']}s | 报告 {result['report_chars']} 字 | "
        f"来源 {len(result['sources'])}"
    )
    print(f"入库笔记 {len(result['notes'])} 条: {result['notes']}")
    print(
        f"记账 {usage['calls']} 行 | input={usage['input_tokens']} "
        f"output={usage['output_tokens']} cache_read={usage['cache_read_tokens']}"
    )
    print("\n报告开头:\n" + result["report"][:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
