#!/usr/bin/env python
"""盲答验证（真实层题集协议的最小可复现实现，复用序列实验 task-1）。

── 用途 ─────────────────────────────────────────────────────────────────
复用序列实验的新综合问在定稿前须过盲答验证协议（evals/qa/ai-frameworks-real.
validation.md §2.4 同款）：把**语料全文 + 问题**（不给 gold、不给 qtype——
无泄漏）交 cheap 档模型作答，用 ``researchwiki.evals.metrics.point_hit`` 判
gold 要点命中；未命中修复一次（改写 gold 第 1 条为文档逐字短证据）再验，
再不过丢弃该题（不迭代）。逐次作答落 JSONL 留痕（committed，审计链）。

真实层会话（commits df94774/22002ac）的盲答验证为会话内临时产物（scratch/，
不随仓库提交）——本脚本是该协议的最小固化，供序列题集与后续题集复用。

── 用法 ─────────────────────────────────────────────────────────────────
  uv run python scripts/blind_validate.py \
      --qa evals/qa/ai-frameworks-seq.jsonl \
      --corpus evals/fixtures/ai-frameworks-real \
      --out evals/qa/ai-frameworks-seq.blind.jsonl \
      [--qids SEQ003,SEQ006,SEQ012]   # 缺省自动识别 notes 含「新题」的行
      [--allow-mock]                  # 仅管线自证（占位答案无判定意义）

真实调用走 config.toml 的 [llm.cheap] 档（ModelRouter）；cheap 的 base_url 为
空即 mock 模式——盲答验证的判定对象是真实模型回答，mock 占位回答无意义，
默认拒绝（--allow-mock 显式自证除外）。key 从 .env / 环境变量读取。

── 输出契约 ─────────────────────────────────────────────────────────────
``out`` JSONL 每行：{"qid", "question", "attempt", "model", "tier", "answer",
"point_hits", "hits", "em", "pass", "ts", "provider_note"}。pass 判据与
real 题集漏斗一致：hits ≥ 1（至少命中一条 gold 要点）。退出码：全部 PASS=0，
存在 FAIL=1（供自动化检查）。
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from researchwiki.env import load_env_file  # noqa: E402
from researchwiki.evals.metrics import normalized_em, point_hit  # noqa: E402
from researchwiki.evals.qa import load_qa  # noqa: E402
from researchwiki.llm.provider import Message  # noqa: E402
from researchwiki.llm.router import ModelRouter  # noqa: E402

#: 盲答协议 system 提示（validation.md §2.4 同款硬规则；与 judge 的 temporal
#: 扣分口径一致——新旧版本以 SHA-B 为准，此为协议约定而非 gold 泄漏）
BLIND_SYSTEM = (
    "你只能依据用户提供的语料原文回答问题。硬规则：\n"
    "1. 只准使用语料中明确写出的事实，禁止编造与使用语料之外的知识；\n"
    "2. 英文术语、命令、包名、API 名必须逐字引用原文；\n"
    "3. 语料不含所需信息时，只回答：语料不含此信息；\n"
    "4. 涉及新旧版本的事实，以最新版本（SHA-B / 1.0.0 快照）口径为准；\n"
    "5. 用中文回答，简洁直接，不要复述本题之外的语料内容。"
)

#: pass 判据：至少命中一条 gold 要点（与 real 题集漏斗口径一致）
PASS_MIN_HITS = 1


def load_config(path: Path) -> dict[str, Any]:
    """读 config.toml；缺失/损坏按空配置处理（cold_warm_smoke.load_config 同款）。"""
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except (FileNotFoundError, tomllib.TOMLDecodeError):
        return {}


def corpus_full_text(fixture_dir: Path) -> str:
    """全部语料按 manifest 顺序拼接（doc_id 标题分隔）——盲答的唯一信息源。"""
    manifest = json.loads((fixture_dir / "manifest.json").read_text(encoding="utf-8"))
    parts: list[str] = []
    for doc in manifest["docs"]:
        doc_id = str(doc["doc_id"])
        raw = (fixture_dir / "docs" / f"{doc_id}.md").read_text(encoding="utf-8")
        parts.append(f"===== 语料文档 {doc_id} =====\n{raw.strip()}")
    return "\n\n".join(parts)


def select_new_questions(qa_path: Path, qids_arg: str) -> list[Any]:
    """选出待验证的题：显式 --qids 优先；缺省取 notes 含「新题」的行。"""
    items = load_qa(qa_path)
    if qids_arg:
        wanted = [q.strip() for q in qids_arg.split(",") if q.strip()]
        by_id = {item.qid: item for item in items}
        missing = [q for q in wanted if q not in by_id]
        if missing:
            raise SystemExit(f"--qids 含题集中不存在的 qid：{'、'.join(missing)}")
        return [by_id[q] for q in wanted]
    return [item for item in items if "新题" in item.notes]


def blind_answer(provider: Any, corpus_text: str, question: str) -> str:
    """一次盲答调用：语料全文 + 问题（无 gold、无题型提示）→ 回答全文。"""
    messages = [Message(role="user", content=f"{corpus_text}\n\n===== 问题 =====\n{question}")]
    chunks: list[str] = []
    for event in provider.stream(messages, system=BLIND_SYSTEM):
        if event.type == "text_delta":
            chunks.append(event.delta)
    return "".join(chunks).strip()


def validate_question(provider: Any, corpus_text: str, item: Any, model_name: str,
                      attempt: int) -> dict:
    """对一道题执行盲答判定，返回一行留痕记录（不在此做修复迭代——修复是
    人工改写 gold 后重跑本脚本，attempt 序号区分轮次）。"""
    answer = blind_answer(provider, corpus_text, item.question)
    hits_flags = [point_hit(answer, gp) for gp in item.gold_points]
    hits = sum(1 for flag in hits_flags if flag)
    em = normalized_em(answer, item.gold_points)
    return {
        "qid": item.qid,
        "question": item.question,
        "attempt": attempt,
        "model": model_name,
        "tier": "cheap",
        "answer": answer,
        "point_hits": hits_flags,
        "hits": hits,
        "em": round(em, 3),
        "pass": hits >= PASS_MIN_HITS,
        "ts": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "provider_note": "盲答协议：语料全文+问题（无 gold 无题型提示），point_hit 判 gold",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="盲答验证：语料全文+问题 → cheap 档 → point_hit 判 gold")
    parser.add_argument("--qa", required=True, help="序列题集 JSONL（待验证题所在文件）")
    parser.add_argument("--corpus", required=True, help="语料 fixture 目录（含 manifest.json 与 docs/）")
    parser.add_argument("--out", required=True, help="盲答留痕 JSONL 输出路径（追加写）")
    parser.add_argument("--qids", default="", help="逗号分隔的待验证 qid；缺省自动识别 notes 含「新题」的行")
    parser.add_argument("--attempt", type=int, default=1, help="盲答轮次（修复重跑记 2，留痕区分）")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.toml"), help="config.toml 路径")
    parser.add_argument("--env-file", default=str(PROJECT_ROOT / ".env"), help=".env 路径（传空串跳过）")
    parser.add_argument("--allow-mock", action="store_true", help="允许 mock 档（占位回答，仅管线自证）")
    args = parser.parse_args(argv)

    if args.env_file:
        load_env_file(args.env_file)

    config = load_config(Path(args.config))
    router = ModelRouter(config.get("llm") or {})
    provider = router.get("cheap")
    base_url = str((config.get("llm") or {}).get("cheap", {}).get("base_url") or "").strip()
    if not base_url and not args.allow_mock:
        print("[blind] [llm.cheap].base_url 为空（mock 档）：盲答判定对象是真实模型回答，"
              "mock 占位回答无意义。确需管线自证请加 --allow-mock。", file=sys.stderr)
        return 2

    qa_path = Path(args.qa)
    corpus_text = corpus_full_text(Path(args.corpus))
    questions = select_new_questions(qa_path, args.qids)
    if not questions:
        print("[blind] 无待验证题（--qids 为空且题集无「新题」行）。", file=sys.stderr)
        return 2

    records = [
        validate_question(provider, corpus_text, item, provider.model, args.attempt)
        for item in questions
    ]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    for record in records:
        status = "PASS" if record["pass"] else "FAIL"
        print(f"[blind] {record['qid']} hits={record['hits']}/{len(record['point_hits'])} "
              f"em={record['em']} -> {status}")
    all_pass = all(record["pass"] for record in records)
    print(f"[blind] 留痕：{out_path}（{'全部 PASS' if all_pass else '存在 FAIL——修复 gold 后重跑，再不过丢弃'}）")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
