#!/usr/bin/env python
"""复用序列实验重放 harness：四条件重放同一条事件时间线，逐事件计量。

── 用途（复用序列实验 PLAN Task 2）──────────────────────────────────────
消费 Task 1 的事件流（``evals/sequence/scenario.jsonl``，:func:`researchwiki.
evals.sequence.load_scenario` 校验）与序列题集，把**同一条事件流**重放到四个
条件上，产出 ``<out>/sequence_<provider>_<ts>/events.jsonl``（每行一个条件×
事件）与 ``manifest.json``（provenance + 预注册判定原文 + 本 harness 裁定）：

  C1 无 Memory        每问一次性 wiki 根目录（c1_e<event_id>_<qid>/）：prior/
                      formation/memory_update 全关——ingest/update 事件对 c1
                      无操作（不产行）；信息权限对等靠检索侧：c1 的 fresh 搜索
                      走同一份"当前语料索引"，update 事件后同样检索到 v2 语料。
  C2 Vector RAG       run_rag(mode="vector")，持久索引对象；update 事件换 chunk。
  C3 Hybrid RAG       run_rag(mode="hybrid")，同上。
  C4 ExternalMemory   **一个持久 wiki 根目录（<out>/c4/）贯穿全时间线**（记忆
                      累积=被测对象）：ingest/update 事件=一次 study run（完整
                      loop，formation + memory_update 开，强制 deep，把该事件
                      的文档集研究入库——LLM 提取是架构真实构建成本，如实记入
                      该事件行）；query 事件=warm loop（prior 开），检索策略
                      **auto**（mock={"enabled": true} 全默认；real=接 config
                      [retrieval] 段，不设 forced_mode），formation/memory_
                      update 关（见裁定①：查询不写记忆）。

── 三条实现裁定（manifest.rulings 留档，报告披露）────────────────────────
① 查询=读操作：query 事件的 loop 不开 formation/memory_update（不蒸馏、不
  触发 supersede）——c4 的记忆只在 ingest/update 事件边界演化，干净可归因；
  代价是查询期内的新信息不回流记忆（演化语义由 update 事件承担）。
② 信息权限对等：四条件的检索语料集**同随事件切换**（同一索引实例：ingest 扩
  集、update 后 @v2 替换 base v1）——c1 也能在 query 时搜到当前语料，c2/c3
  update 后检索命中 v2 chunk（取对 RAG 最有利解释，避免沙包化）。
③ 索引整体重建：update 事件按新文档集整体重建 CorpusIndex（选简单正确），
  构建成本（embedding 调用次数，零 LLM 成本）如实记入 update 事件行；
  c2/c3 运行时共享同一索引实例，构建成本按条件镜像入账（per-condition 曲线
  可比性优先，manifest 声明）。

study run 的研究问题为**合成问题**（:func:`study_question` 由文档标题拼装，
原文在 manifest.rulings.study_synthetic_question 披露）——它不属于评测题，
仅驱动 study run 的检索与蒸馏。mock 模式下 study run 的蒸馏剧本覆盖该事件
时段会出现的题集题目（每题一条笔记，run_eval 播种同款保证 warm prior 命中）；
real 模式蒸馏由真实模型完成，剧本清单不消费。

── 与 real run（evals/results/real_*）的口径差异（报告头部声明）──────────
①c4 由 forced deep 改为 auto（修正"重复查询承担完整构建成本"）；②同一信息流
分事件进入（修正信息权限不对称）；③c1/c4 的查询行 judge 与 run_eval 同链路
（judge glm-4.7 不变；生成模型 real=glm-4.5-air 不变）；c1 检索口径不变
（forced deep + brief，成本曲线可与 real run 对齐）。

── 复用边界（run_eval 机械，importlib 同目录 import）────────────────────
条件实现/URL 改写/fetch 回放/mock 剧本/judge 接线全部复用 ``scripts/run_eval.
py``（execute_loop / EvalDeps / FixtureHttpSearch / make_mock_router /
make_mock_rag_provider / enabled_section / conditions_arg / mock_judge_error /
SEED_RETRIEVAL_CONFIG 等）；本脚本只新增事件时间线驱动、语料状态机、
events.jsonl 行 schema 与 manifest。``src/researchwiki/evals/corpus.py`` 的
load_corpus 加了可选 doc_ids 子集过滤（缺省行为不变，导入性薄改）。
cold_warm_smoke.py 冻结不动。

── fetch 回放（real 语料的零网络保证）──────────────────────────────────
``make_corpus_fetch_transport``：manifest 登记的 URL（real 语料是 github 形态、
fixture 语料经 FixtureHttpSearch 改写成 http://fixture.local/...）一律回放本地
``docs/<doc_id>.md``（content-type text/markdown），并兜住 Jina Reader 降级
前缀；未登记 URL 返回 404（fetch_url 如实抛 FetchError）。搜索命中只含
manifest URL，故 loop 条件的 fetch_url 全链路（trafilatura 提取 + sources/
快照落盘）零网络可跑。

── 输出契约（events.jsonl 每行一个 JSON 对象，UTF-8、ensure_ascii=False）──
  公共键：{"condition", "event_id", "event_kind", "qid", "in_tok", "out_tok",
  "latency_ms", "em", "point_hits", "refusal", "citation_coverage",
  "judge": {...}|None, "gold_source": "base"|"override", "trace_id"}
  loop 行（c1/c4 的 query 与 c4 的 ingest/update study run）追加：
  {"fresh_search_count", "wiki_root", "run_dir"}，c4 再追加 {"policy_mode"}
  （来自 run_dir/policy.json 的 P3 判定 mode）；
  RAG query 行（c2/c3）追加：{"hit_urls"}（检索命中 URL——update 后时效问的
  v2 命中可审计）；
  索引行（c2/c3 的 ingest/update）追加：{"embedding_calls"}（本次重建嵌入的
  chunk 数；in_tok/out_tok 恒 0——embedding 零 LLM 成本，如实记录）。
  非 query 行（ingest/update）：em/point_hits/refusal/citation_coverage/judge
  均为 None（不评分）；c1 在 ingest/update 事件不产行。
  judge：每问每条件恰一次（复用 -judge 独立 trace，不污染主 trace 对账）；
  gold_override 存在的 query 事件（演化正确性 diff-gold）用 override 要点判
  em/point_hits 与 judge，行级 gold_source="override" 区分。
  主循环异常时已累积行先原子写入 events.partial.jsonl 再退出非 0。

── 记账与对账（不变量 ⑥）───────────────────────────────────────────────
loop 行：TokenAccountant 指向 <wiki_root>/tokens.jsonl（c4 全时间线同一文件、
逐 run 独立 trace），行内 in_tok/out_tok 与 sum_tokens_from_jsonl(wiki_root/
tokens.jsonl, trace_id) 完全一致；RAG 行：记账集中在 <out>/tokens.jsonl。

── 怎么跑 ─────────────────────────────────────────────────────────────
  # 零 key 零网络：mock 全链路（真实场景 24 事件 × 4 条件）
  uv run --no-sync python scripts/run_sequence.py --provider mock
  # 真实测量：config.toml 的 strong/cheap/judge + 真实 embedding（真实费用）
  uv run --no-sync python scripts/run_sequence.py --provider real
"""

from __future__ import annotations

import argparse
import hashlib
import httpx
import json
import sys
import time
import urllib.parse
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ---- 同目录复用 run_eval（scripts/ 不是包，先把目录挂上 sys.path；run_eval
#      同款先例：它对 cold_warm_smoke 也这么做）--------------------------------
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import cold_warm_smoke as cws  # noqa: E402  -- MOCK_*_CONFIG / load_config 复用
import run_eval  # noqa: E402  -- 条件实现与 fixture 机械化复用（见模块 docstring）

from researchwiki.evals.corpus import load_corpus  # noqa: E402
from researchwiki.evals.judge import build_judge_provider, judge_answer  # noqa: E402
from researchwiki.evals.metrics import (  # noqa: E402
    normalized_em,
    point_hit,
    refusal_detected,
)
from researchwiki.evals.qa import QaItem, load_qa  # noqa: E402
from researchwiki.evals.rag_harness import run_rag  # noqa: E402
from researchwiki.evals.sequence import SeqEvent, load_scenario  # noqa: E402
from researchwiki.llm.accounting import TokenAccountant  # noqa: E402
from researchwiki.loop.metrics import (  # noqa: E402
    compute_citation_coverage,
    sum_tokens_from_jsonl,
)
from researchwiki.tools import atomic_write_text  # noqa: E402
from researchwiki.wiki.embeddings import (  # noqa: E402
    MockEmbeddingProvider,
    get_embedding_provider,
)

PROJECT_ROOT = run_eval.PROJECT_ROOT
DEFAULT_SCENARIO = PROJECT_ROOT / "evals" / "sequence" / "scenario.jsonl"
DEFAULT_QA = PROJECT_ROOT / "evals" / "qa" / "ai-frameworks-seq.jsonl"
DEFAULT_FIXTURES = PROJECT_ROOT / "evals" / "fixtures" / "ai-frameworks-real"
DEFAULT_OUT = run_eval.DEFAULT_OUT
DEFAULT_CONFIG = run_eval.DEFAULT_CONFIG
DEFAULT_ENV_FILE = run_eval.DEFAULT_ENV_FILE

SPEC_PATH = "docs/superpowers/specs/2026-10-10-reuse-sequence-eval-design.md"
PLAN_PATH = "docs/superpowers/plans/2026-10-10-reuse-sequence-eval.md"

#: spec §4 预注册判定（原文引用，跑完对表——实现不得弱化或改动，manifest 留档）
PRE_REGISTERED: tuple[dict[str, str], ...] = (
    {
        "dimension": "经济性",
        "成立需要": "c4-auto 累计成本在 ≤22 问内 ≤ c2 同期，或 judge cov 优势 ≥+0.5",
        "证伪条件": "全程 c4 累计 > c2 且无任何质量维度优势",
    },
    {
        "dimension": "复用质量",
        "成立需要": "重复问第二次 judge cov ≥ 第一次且 simple 路由发生（率 >0）",
        "证伪条件": "simple 路由率为 0 或复用后质量下降",
    },
    {
        "dimension": "演化价值",
        "成立需要": "更新后时效题 c4 优于 c2/c3",
        "证伪条件": "c4 时效题无优势（supersede 未转化为答案正确性）",
    },
    {
        "dimension": "反幻觉",
        "成立需要": "（测量，无成立条件）",
        "证伪条件": "c4 重演 RQ028 编造",
    },
)

#: c4 query 事件的 auto 检索策略（mock：全默认自适应判定，不设 forced_mode；
#: real：接 config [retrieval] 段——见 build_parser/main 的 deps 组装）
C4_AUTO_RETRIEVAL_MOCK: dict[str, Any] = {"enabled": True}
#: c4 study run（ingest/update 事件）的检索策略：强制 deep 完整研究
#: （复用 run_eval 播种口径——构建成本要真实，但 study run 不计分、不约束报告样式）
STUDY_RETRIEVAL_CONFIG: dict[str, Any] = run_eval.SEED_RETRIEVAL_CONFIG
#: c1 query 事件的检索策略：与 run_eval C1 同口径（forced deep + brief），
#: 每问一次性 root 的成本曲线可与 real run 的 c1 对齐
C1_RETRIEVAL_CONFIG: dict[str, Any] = run_eval.LOOP_RETRIEVAL_CONFIG

EXIT_PASS = 0
EXIT_RUN_ERROR = 2


# ---- 语料状态机（事件时间线 → 当前文档集 → 子集索引）-------------------------


def apply_corpus_change(
    current: Sequence[str], change_doc_ids: Sequence[str]
) -> list[str]:
    """把 ingest/update 事件的文档集应用到当前语料状态，返回新文档集（保序去重）。

    - ingest 语义：追加新文档；重复提供已在语料中的文档 → ``ValueError``
      （事件流校验器不查此项，harness 层 fail fast）；
    - update 语义：``<base>@v2`` 替换 base 的 v1（v2 到达即淘汰旧版——对 RAG
      取最有利解释，避免沙包化，spec §1）；不带 @v2 后缀的按追加处理。
    """
    docs = list(dict.fromkeys(current))
    for doc_id in change_doc_ids:
        base = doc_id[: -len("@v2")] if doc_id.endswith("@v2") else doc_id
        if doc_id.endswith("@v2"):
            docs = [d for d in docs if d != base]
        if doc_id in docs:
            raise ValueError(f"文档 {doc_id} 已在语料中，不能重复 ingest/update")
        docs.append(doc_id)
    return docs


def build_corpus_index(
    fixture_dir: str | Path, doc_ids: Sequence[str], embedding: Any
) -> Any:
    """按当前文档集建语料索引（load_corpus 同款切段/URL/字段，doc_ids 子集过滤）。

    切段与检索口径与全量 load_corpus 逐字同源（同一函数实现，仅文档集不同），
    保证 c2/c3/c1/c4 的检索口径在事件间只随文档集演化、不随实现漂移。
    """
    return load_corpus(fixture_dir, embedding=embedding, doc_ids=list(doc_ids))


def gold_for(item: QaItem, event: SeqEvent) -> tuple[list[str], str]:
    """query 事件的判分要点与口径来源：带 gold_override 用 override（演化正确性
    diff-gold，EM 单列演化口径），否则用题集 gold（base）。"""
    if event.gold_override:
        return list(event.gold_override), "override"
    return list(item.gold_points), "base"


# ---- fetch 回放（real 语料 URL + fixture 改写 URL 的零网络本地回放）-----------


def make_corpus_fetch_transport(fixture_dir: str | Path) -> httpx.MockTransport:
    """httpx.MockTransport：manifest 登记 URL → 本地 docs/<doc_id>.md（零网络）。

    覆盖三类 URL 形态（搜索命中只回放这些）：

    - real 语料 manifest 的原始 http(s) URL（如 github blob——按 manifest 精确
      匹配，忽略 query/fragment 与尾斜杠）；
    - fixture:// 语料经 run_eval.FixtureHttpSearch 改写出的
      ``http://fixture.local/<domain>/<doc_id>.md``；
    - 以上两者的 Jina Reader 降级前缀（https://r.jina.ai/<url>）变体。

    未登记 URL 返回 404（fetch_url 如实抛 FetchError，工具层回传错误——不静默
    兜底）；命中时回放语料 markdown 原文（content-type text/markdown），fetch_url
    走真实链路（trafilatura 提取 + sources/ 快照落盘）。
    """
    root = Path(fixture_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    docs: dict[str, str] = {}
    url_to_doc: dict[str, str] = {}
    for doc in manifest["docs"]:
        doc_id = str(doc["doc_id"])
        docs[doc_id] = (root / "docs" / f"{doc_id}.md").read_text(encoding="utf-8")
        url_to_doc[str(doc["url"]).rstrip("/")] = doc_id

    jina_prefix = "https://r.jina.ai/"

    def doc_id_from_url(url: str) -> str | None:
        candidate = url.strip()
        if candidate.startswith(jina_prefix):
            candidate = candidate[len(jina_prefix) :]
        candidate = candidate.split("?", 1)[0].split("#", 1)[0].rstrip("/")
        if candidate in url_to_doc:
            return url_to_doc[candidate]
        # fixture:// 改写形态：http://fixture.local/<domain>/<doc_id>.md
        pos = candidate.find(run_eval.FIXTURE_HTTP_PREFIX)
        if pos >= 0:
            rest = candidate[pos + len(run_eval.FIXTURE_HTTP_PREFIX) :]
            filename = rest.rsplit("/", 1)[-1]
            if filename.endswith(".md"):
                return urllib.parse.unquote(filename[: -len(".md")])
        return None

    def handler(request: httpx.Request) -> httpx.Response:
        doc_id = doc_id_from_url(str(request.url))
        body = docs.get(doc_id) if doc_id else None
        if body is None:
            return httpx.Response(404, text="fixture doc not found")
        return httpx.Response(
            200, text=body, headers={"content-type": "text/markdown; charset=utf-8"}
        )

    return httpx.MockTransport(handler)


# ---- study run（ingest/update 事件的 c4 记忆构建）-----------------------------


def study_question(event: SeqEvent, titles: Mapping[str, str]) -> str:
    """ingest/update 事件 study run 的综合研究问题（合成，非评测题）。

    study run 用完整 loop 消化文档（LLM 提取入库=架构真实构建成本）；问题由
    文档标题确定性拼装、只驱动检索与蒸馏——原文在 manifest 披露，不参与判分。
    """
    titles_txt = "、".join(f"《{titles.get(doc_id, doc_id)}》" for doc_id in event.doc_ids)
    if event.kind == "ingest":
        return (
            f"【ingest 综合研究问题】请检索并研读以下 {len(event.doc_ids)} 篇文档："
            f"{titles_txt}；提取各文档可长期复用的关键事实、机制与结论，形成结构化研究笔记。"
        )
    return (
        f"【update 研究问题】以下文档发布了新版本：{titles_txt}；"
        "请检索研读新版本内容，识别相对既有记忆中旧口径的新增、变更与移除事实，"
        "形成更新研究笔记。"
    )


def study_distill_notes(
    events: Sequence[SeqEvent],
    qa_items: Mapping[str, QaItem],
    *,
    phase: str,
    update_event_id: int,
) -> list[tuple[str, str, Sequence[str]]]:
    """mock study run 的蒸馏剧本：该事件时段会出现的题集题目，每题一条笔记。

    phase="pre"：update 前出现过的 qid（ingest study run 用）；phase="post"：
    update 后出现的 qid（update study run 用）。笔记正文嵌入问题原文（Prior
    检索 FTS+向量双通道必然命中，run_eval 播种同款保证）。**real 模式不消费
    本清单**（execute_loop 仅 mock 分支用 distill_notes，蒸馏由真实模型完成）。
    """
    qids: list[str] = []
    for event in events:
        if event.kind != "query" or event.qid in qids:
            continue
        if (event.event_id < update_event_id) == (phase == "pre"):
            qids.append(event.qid)
    return [
        (
            run_eval.mock_note_text(qa_items[qid].question, qid),
            qid,
            qa_items[qid].entities,
        )
        for qid in qids
    ]


# ---- events.jsonl 行构造（schema 见模块 docstring 的输出契约）-----------------


def _judge_payload(verdict: Any) -> dict[str, Any] | None:
    """JudgeVerdict → 行内 JSON（全字段含 raw 留档）；解析失败 → None。"""
    return None if verdict is None else dict(vars(verdict))


def _base_row(*, event: SeqEvent, condition: str, qid: str) -> dict[str, Any]:
    """events.jsonl 行的公共键（非 query 行的 em/point_hits/… 均为 None——不评分）。"""
    return {
        "condition": condition,
        "event_id": event.event_id,
        "event_kind": event.kind,
        "qid": qid,
        "in_tok": 0,
        "out_tok": 0,
        "latency_ms": 0,
        "em": None,
        "point_hits": None,
        "refusal": None,
        "citation_coverage": None,
        "judge": None,
        "gold_source": "base",
        "trace_id": "",
    }


def build_study_row(
    *,
    event: SeqEvent,
    loop: Any,
    metrics: Mapping[str, Any],
    question: str,
) -> dict[str, Any]:
    """c4 study run 的 ingest/update 行：记记忆构建成本（qid=""，不评分）。"""
    row = _base_row(event=event, condition=run_eval.CONDITION_C4, qid="")
    row.update(
        {
            "in_tok": int(metrics.get("input_tokens") or 0),
            "out_tok": int(metrics.get("output_tokens") or 0),
            "latency_ms": int(metrics.get("latency_ms") or 0),
            "trace_id": loop.trace_id,
            "fresh_search_count": int(metrics.get("fresh_search_count") or 0),
            "wiki_root": str(loop.wiki_root),
            "run_dir": str(loop.run_dir),
        }
    )
    policy_mode = read_policy_mode(loop.run_dir)
    if policy_mode is not None:
        row["policy_mode"] = policy_mode
    # study 问题不进行 schema（manifest.rulings 披露拼装规则），打印留痕供审计
    row["_study_question"] = question  # 落盘前剥离（见 write_events）
    return row


def build_index_row(
    *, event: SeqEvent, condition: str, chunk_count: int, latency_ms: float
) -> dict[str, Any]:
    """c2/c3 的 ingest/update 行：记索引重建成本（embedding 调用，零 LLM 成本）。"""
    row = _base_row(event=event, condition=condition, qid="")
    row.update(
        {
            "latency_ms": round(latency_ms, 1),
            "embedding_calls": int(chunk_count),
        }
    )
    return row


def build_loop_query_row(
    *,
    event: SeqEvent,
    item: QaItem,
    condition: str,
    gold_points: Sequence[str],
    gold_source: str,
    loop: Any,
    metrics: Mapping[str, Any],
    verdict: Any,
) -> dict[str, Any]:
    """c1/c4 的 query 行：warm/cold loop + judge，fresh_search_count 仅 loop 行携带。"""
    answer = loop.report_text
    row = _base_row(event=event, condition=condition, qid=item.qid)
    row.update(
        {
            "in_tok": int(metrics.get("input_tokens") or 0),
            "out_tok": int(metrics.get("output_tokens") or 0),
            "latency_ms": int(metrics.get("latency_ms") or 0),
            "em": normalized_em(answer, gold_points),
            "point_hits": sum(1 for gold in gold_points if point_hit(answer, gold)),
            "refusal": refusal_detected(answer),
            "citation_coverage": metrics.get("citation_coverage"),
            "judge": _judge_payload(verdict),
            "gold_source": gold_source,
            "trace_id": loop.trace_id,
            "fresh_search_count": int(metrics.get("fresh_search_count") or 0),
            "wiki_root": str(loop.wiki_root),
            "run_dir": str(loop.run_dir),
        }
    )
    if condition == run_eval.CONDITION_C4:
        policy_mode = read_policy_mode(loop.run_dir)
        if policy_mode is not None:
            row["policy_mode"] = policy_mode
    return row


def build_rag_query_row(
    *,
    event: SeqEvent,
    item: QaItem,
    condition: str,
    gold_points: Sequence[str],
    gold_source: str,
    result: Any,
    verdict: Any,
    trace_id: str,
    latency_ms: float,
) -> dict[str, Any]:
    """c2/c3 的 query 行：run_rag + judge；hit_urls 留痕检索命中（v2 换 chunk 可审计）。"""
    coverage = compute_citation_coverage(result.answer, len(result.hits))
    row = _base_row(event=event, condition=condition, qid=item.qid)
    row.update(
        {
            "in_tok": int(result.input_tokens),
            "out_tok": int(result.output_tokens),
            "latency_ms": round(latency_ms, 1),
            "em": normalized_em(result.answer, gold_points),
            "point_hits": sum(1 for gold in gold_points if point_hit(result.answer, gold)),
            "refusal": refusal_detected(result.answer),
            "citation_coverage": None if coverage is None else round(coverage, 4),
            "judge": _judge_payload(verdict),
            "gold_source": gold_source,
            "trace_id": trace_id,
            "hit_urls": [hit.url for hit in result.hits],
        }
    )
    return row


def read_policy_mode(run_dir: str | Path) -> str | None:
    """读 run_dir/policy.json 的 P3 判定 mode；文件缺失（未启用判定）→ None。"""
    path = Path(run_dir) / "policy.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    mode = payload.get("mode") if isinstance(payload, dict) else None
    return mode if mode in ("simple", "update", "deep") else None


# ---- 依赖组装与入口 -----------------------------------------------------------


def _model_info(provider: Any) -> dict[str, str]:
    return {
        "model": str(getattr(provider, "model", "") or ""),
        "base_url": str(getattr(provider, "base_url", "") or ""),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="复用序列实验重放 harness（四条件同事件流，详见文件头 docstring）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--scenario", default=str(DEFAULT_SCENARIO), help="事件流 JSONL 路径")
    parser.add_argument("--qa", default=str(DEFAULT_QA), help="序列题集 JSONL 路径")
    parser.add_argument(
        "--fixtures", default=str(DEFAULT_FIXTURES), help="受控语料 fixture 目录"
    )
    parser.add_argument(
        "--conditions", type=run_eval.conditions_arg, default=",".join(run_eval.CONDITIONS),
        help="条件矩阵（逗号分隔，缺省 c1,c2,c3,c4）",
    )
    parser.add_argument(
        "--provider", choices=("mock", "real"), default="mock",
        help="mock=零网络零 key 的剧本全链路（默认）；real=走 config.toml 真实档位",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="config.toml 路径")
    parser.add_argument(
        "--env-file", default=str(DEFAULT_ENV_FILE),
        help=".env 路径（仅 real 模式读取；传空串跳过）",
    )
    parser.add_argument(
        "--out", default=str(DEFAULT_OUT),
        help="输出根目录（默认 evals/results；结果写 <out>/sequence_<provider>_<ts>/）",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001 -- 老环境/被替换的 stdout 上忽略
        pass
    args = build_parser().parse_args(argv)
    conditions: list[str] = list(args.conditions)
    fixture_dir = Path(args.fixtures)

    # ---- 事件流与题集先加载校验（校验失败零落盘退出）--------------------------
    try:
        events = load_scenario(args.scenario, args.qa, fixture_dir / "manifest.json")
    except ValueError as exc:
        print(f"✗ 事件流校验失败：{exc}")
        return EXIT_RUN_ERROR
    qa_items: dict[str, QaItem] = {item.qid: item for item in load_qa(args.qa)}
    update_event = next(e for e in events if e.kind == "update")  # 校验器保证恰 1 个

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out) / f"sequence_{args.provider}_{stamp}"

    # ---- 依赖组装（四条件同口径：同一生成 provider / embedding / 语料 / fetch 服务）----
    if args.provider == "mock":
        embedding: Any = MockEmbeddingProvider(dim=512)
        wiki_config: Mapping[str, Any] | None = {"fts_tokenizer": "trigram"}
        llm_config: Mapping[str, Any] | None = {
            "strong": {"base_url": "https://mock"},
            "cheap": {"base_url": "https://mock-cheap"},
        }
        router: Any | None = None
        judge_provider = build_judge_provider(None)
        formation_base: Mapping[str, Any] | None = cws.MOCK_FORMATION_CONFIG
        memory_update_base: Mapping[str, Any] | None = cws.MOCK_MEMORY_UPDATE_CONFIG
        verification_base: Mapping[str, Any] | None = cws.MOCK_VERIFICATION_CONFIG
        c4_retrieval_config: Mapping[str, Any] | None = C4_AUTO_RETRIEVAL_MOCK
        model_summary = {
            "strong": {"model": "mock-strong", "base_url": ""},
            "cheap": {"model": "mock-cheap", "base_url": ""},
            "judge": {"model": "mock-judge", "base_url": ""},
        }
    else:
        if args.env_file:
            from researchwiki.env import load_env_file

            load_env_file(args.env_file)
        config = cws.load_config(Path(args.config))
        # 同一 embedding 配置贯穿语料索引与全部 loop（缓存关闭：索引逐事件重建，
        # 各 wiki 根单次 run 无复算收益；vector 口径由模型决定，与缓存无关）
        embedding = get_embedding_provider(config, cache_path=None)
        wiki_config = config.get("wiki") or {}
        llm_config = config.get("llm") or {}
        router = run_eval.ModelRouter(dict(llm_config))
        judge_provider = build_judge_provider(config)
        formation_base = config.get("formation")
        memory_update_base = config.get("memory_update")
        verification_base = config.get("verification")
        c4_retrieval_config = config.get("retrieval")
        model_summary = {
            "strong": _model_info(router.get("strong")),
            "cheap": _model_info(router.get("cheap")),
            "judge": _model_info(judge_provider),
        }

    # real 模式 judge 硬校验（run_eval 同款：在创建任何输出目录之前，失败零落盘）
    if args.provider == "real":
        judge_error = run_eval.mock_judge_error(judge_provider)
        if judge_error:
            print(f"✗ {judge_error}")
            return EXIT_RUN_ERROR
        retrieval_cfg = c4_retrieval_config if isinstance(c4_retrieval_config, Mapping) else {}
        if retrieval_cfg.get("forced_mode"):
            print(
                "⚠ 警告：config [retrieval].forced_mode 已设置"
                f"（{retrieval_cfg.get('forced_mode')}）——预注册口径为 auto（不设"
                "forced_mode），判定结果将与预注册口径不一致（manifest 已留档）。"
            )

    out_dir.mkdir(parents=True, exist_ok=True)

    corpus_manifest = json.loads((fixture_dir / "manifest.json").read_text(encoding="utf-8"))
    corpus_note = str(corpus_manifest.get("provider_note") or "")
    doc_titles = {
        str(doc["doc_id"]): str(doc.get("title") or doc["doc_id"])
        for doc in corpus_manifest["docs"]
    }

    def make_deps(index: Any) -> run_eval.EvalDeps:
        """按当前语料索引组装共享依赖（索引逐事件重建 → 新 EvalDeps 实例，
        连带 _top_urls 预计算缓存一并失效，保证 top_url 与当前语料同源）。"""
        return run_eval.EvalDeps(
            provider_mode=args.provider,
            index=index,
            embedding=embedding,
            wiki_config=wiki_config,
            llm_config=llm_config,
            router=router,
            judge_provider=judge_provider,
            fetch_transport=make_corpus_fetch_transport(fixture_dir),
            formation_base=formation_base,
            memory_update_base=memory_update_base,
            verification_base=verification_base,
            corpus_note=corpus_note,
        )

    strong = model_summary["strong"]
    judge = model_summary["judge"]
    print("=== 复用序列实验重放 harness（四条件同事件流）===")
    print(f"provider    : {args.provider}")
    print(
        f"model       : strong={strong['model']}（{strong['base_url'] or '—'}）  "
        f"judge={judge['model']}（{judge['base_url'] or '—'}）"
    )
    print(f"scenario    : {args.scenario}（{len(events)} 事件）")
    print(f"qa          : {args.qa}（{len(qa_items)} 题）")
    print(f"conditions  : {', '.join(conditions)}")
    print(f"out         : {out_dir}")
    print("-" * 76, flush=True)

    rows: list[dict[str, Any]] = []
    rag_accountant = TokenAccountant(path=out_dir / "tokens.jsonl")
    c4_root = out_dir / "c4"
    doc_state: list[str] = []
    deps: run_eval.EvalDeps | None = None
    corpus_timeline: list[dict[str, Any]] = []

    try:
        for event in events:
            if event.kind in ("ingest", "update"):
                # ---- 语料状态机：应用事件 → 整体重建索引（裁定②③）------------
                doc_state = apply_corpus_change(doc_state, event.doc_ids)
                t0 = time.perf_counter()
                index = build_corpus_index(fixture_dir, doc_state, embedding)
                build_ms = (time.perf_counter() - t0) * 1000.0
                deps = make_deps(index)
                corpus_timeline.append(
                    {
                        "event_id": event.event_id,
                        "kind": event.kind,
                        "doc_ids": list(event.doc_ids),
                        "doc_set_after": list(doc_state),
                    }
                )
                print(
                    f"[e{event.event_id}/{event.kind}] 语料 → {len(doc_state)} 篇"
                    f"（索引重建 {len(index.chunks)} chunks，{build_ms:.0f}ms）",
                    flush=True,
                )
                # c4：一次 study run 吃下该事件文档集（formation + memory_update 开）
                if run_eval.CONDITION_C4 in conditions:
                    question = study_question(event, doc_titles)
                    phase = "pre" if event.event_id < update_event.event_id else "post"
                    distill = study_distill_notes(
                        events, qa_items, phase=phase, update_event_id=update_event.event_id
                    )
                    print(
                        f"[e{event.event_id}/c4-study] run 开始（forced deep，"
                        f"{len(event.doc_ids)} 篇文档，phase={phase}）",
                        flush=True,
                    )
                    loop, metrics, _accountant = run_eval.execute_loop(
                        deps,
                        label=f"e{event.event_id}/c4-study",
                        question=question,
                        wiki_root=c4_root,
                        prior_config=None,  # 缺省 = enabled + 默认（update run 可读旧记忆）
                        retrieval_config=STUDY_RETRIEVAL_CONFIG,
                        formation_config=run_eval.enabled_section(formation_base),
                        memory_update_config=run_eval.enabled_section(memory_update_base),
                        verification_config=verification_base,
                        distill_notes=distill,
                    )
                    rows.append(
                        build_study_row(
                            event=event, loop=loop, metrics=metrics, question=question
                        )
                    )
                # c2/c3：索引构建成本行（共享同一索引实例，按条件镜像入账——裁定③）
                for cond in (run_eval.CONDITION_C2, run_eval.CONDITION_C3):
                    if cond in conditions:
                        rows.append(
                            build_index_row(
                                event=event,
                                condition=cond,
                                chunk_count=len(index.chunks),
                                latency_ms=build_ms,
                            )
                        )
                # c1：ingest/update 无操作（检索语料切换已随共享索引生效）
            else:
                # ---- query 事件：四条件在各自当时状态下作答 --------------------
                assert deps is not None, "query 事件之前必有 ingest 事件（校验器保证）"
                item = qa_items[event.qid]
                gold_points, gold_source = gold_for(item, event)
                for cond in conditions:
                    if cond in run_eval.RAG_MODES:
                        provider: Any = (
                            run_eval.make_mock_rag_provider()
                            if args.provider == "mock"
                            else router.get("strong")  # type: ignore[union-attr]
                        )
                        trace_id = rag_accountant.new_trace_id()
                        result = run_rag(
                            provider,
                            deps.index,
                            question=item.question,
                            mode=run_eval.RAG_MODES[cond],
                            accountant=rag_accountant,
                            trace_id=trace_id,
                        )
                        verdict = judge_answer(
                            judge_provider,
                            question=item.question,
                            gold_points=gold_points,
                            answer=result.answer,
                            sources=[{"url": h.url, "title": h.title} for h in result.hits],
                            accountant=rag_accountant,
                            trace_id=f"{trace_id}-judge",
                        )
                        rows.append(
                            build_rag_query_row(
                                event=event,
                                item=item,
                                condition=cond,
                                gold_points=gold_points,
                                gold_source=gold_source,
                                result=result,
                                verdict=verdict,
                                trace_id=trace_id,
                                latency_ms=result.latency_ms,
                            )
                        )
                    elif cond == run_eval.CONDITION_C1:
                        loop, metrics, accountant = run_eval.execute_loop(
                            deps,
                            label=f"e{event.event_id}/c1/{item.qid}",
                            question=item.question,
                            wiki_root=out_dir / f"c1_e{event.event_id}_{item.qid}",
                            prior_config=run_eval.C1_PRIOR_CONFIG,
                            retrieval_config=C1_RETRIEVAL_CONFIG,
                            formation_config=None,
                            memory_update_config=None,
                            verification_config=None,
                            distill_notes=[
                                (
                                    run_eval.mock_note_text(item.question, item.qid),
                                    item.qid,
                                    item.entities,
                                )
                            ],
                        )
                        verdict = judge_answer(
                            judge_provider,
                            question=item.question,
                            gold_points=gold_points,
                            answer=loop.report_text,
                            sources=loop.ctx.source_pool.numbered(),
                            accountant=accountant,
                            trace_id=f"{loop.trace_id}-judge",
                        )
                        rows.append(
                            build_loop_query_row(
                                event=event,
                                item=item,
                                condition=cond,
                                gold_points=gold_points,
                                gold_source=gold_source,
                                loop=loop,
                                metrics=metrics,
                                verdict=verdict,
                            )
                        )
                    else:  # c4：warm loop，auto 策略，查询不写记忆（裁定①）
                        loop, metrics, accountant = run_eval.execute_loop(
                            deps,
                            label=f"e{event.event_id}/c4/{item.qid}",
                            question=item.question,
                            wiki_root=c4_root,
                            prior_config=None,  # 缺省 = enabled + 默认（warm 读记忆）
                            retrieval_config=c4_retrieval_config,
                            formation_config=None,
                            memory_update_config=None,
                            verification_config=None,
                            distill_notes=[
                                (
                                    run_eval.mock_note_text(item.question, item.qid),
                                    item.qid,
                                    item.entities,
                                )
                            ],
                        )
                        verdict = judge_answer(
                            judge_provider,
                            question=item.question,
                            gold_points=gold_points,
                            answer=loop.report_text,
                            sources=loop.ctx.source_pool.numbered(),
                            accountant=accountant,
                            trace_id=f"{loop.trace_id}-judge",
                        )
                        rows.append(
                            build_loop_query_row(
                                event=event,
                                item=item,
                                condition=cond,
                                gold_points=gold_points,
                                gold_source=gold_source,
                                loop=loop,
                                metrics=metrics,
                                verdict=verdict,
                            )
                        )
                last_row = rows[-1]
                print(
                    f"[e{event.event_id}/query/{item.qid}] 四条件完成（"
                    f"gold_source={gold_source}，末行 {last_row['condition']} "
                    f"in/out={last_row['in_tok']}/{last_row['out_tok']}）",
                    flush=True,
                )
    except Exception as exc:  # noqa: BLE001 -- 环境/网络问题不作数，明确退出而非裸栈
        # 部分行落盘（real run 保护，run_eval 同款）：已累积行先原子写 events.partial.jsonl
        partial_path = out_dir / "events.partial.jsonl"
        atomic_write_text(
            partial_path,
            "".join(
                json.dumps(_strip_row(row), ensure_ascii=False) + "\n" for row in rows
            ),
        )
        print(f"✗ 运行异常：{type(exc).__name__}: {exc}")
        print(
            f"（已完成 {len(rows)} 行已部分落盘 {partial_path}；"
            "real 模式请检查 key / base_url / 网络；mock 模式不应出现本行，视为 bug）"
        )
        return EXIT_RUN_ERROR

    events_path = out_dir / "events.jsonl"
    events_path.write_text(
        "".join(json.dumps(_strip_row(row), ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    manifest: dict[str, Any] = {
        "gate": "run_sequence",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "spec_path": SPEC_PATH,
        "plan_path": PLAN_PATH,
        "pre_registered": [dict(item) for item in PRE_REGISTERED],
        "provider_mode": args.provider,
        "model": model_summary,
        "scenario_path": str(Path(args.scenario).resolve()),
        "scenario_sha256": hashlib.sha256(Path(args.scenario).read_bytes()).hexdigest(),
        "event_count": len(events),
        "qa_path": str(Path(args.qa).resolve()),
        "qa_sha256": hashlib.sha256(Path(args.qa).read_bytes()).hexdigest(),
        "fixtures_dir": str(fixture_dir.resolve()),
        "corpus_provider_note": corpus_note,
        "fixture_url_mapping": (
            "loop 条件（C1/C4）的检索命中 URL 交 fetch_url 抓取；real 语料的原始"
            " http(s) URL 与 fixture:// 语料改写后的 fixture.local URL 一律由 "
            "httpx.MockTransport（make_corpus_fetch_transport）按 manifest 回放本地"
            " docs/<doc_id>.md（零网络，fetch_url 真实链路写 sources/ 快照）；RAG "
            "条件（C2/C3）不经 fetch，行级 hit_urls 记录检索命中的原始 URL。"
        ),
        "conditions": conditions,
        "condition_notes": {c: run_eval.CONDITION_NOTES[c] for c in conditions},
        "retrieval_config": {
            "c4_query": dict(c4_retrieval_config) if c4_retrieval_config else None,
            "c4_query_note": "auto 自适应（不设 forced_mode）；real 接 config [retrieval] 段",
            "c1_query": dict(C1_RETRIEVAL_CONFIG),
            "c1_query_note": "与 run_eval C1 同口径（forced deep + brief）",
            "c4_study": dict(STUDY_RETRIEVAL_CONFIG),
            "c4_study_note": "ingest/update 的 study run：forced deep 完整研究（构建成本）",
        },
        "rulings": {
            # 裁定①：查询=读操作——c4 记忆只在事件边界演化
            "query_readonly": True,
            "query_readonly_note": (
                "query 事件的 loop formation/memory_update 均为 None（不蒸馏、不触发"
                " supersede）；只有 ingest/update 的 study run 开 formation+memory_update。"
            ),
            # 裁定②：信息权限对等——检索语料集同随事件切换
            "search_corpus_switches_with_update": list(conditions),
            "search_corpus_switches_note": (
                "四条件共用同一语料状态机（ingest 扩集、update 后 @v2 替换 base v1），"
                "c1 查询时同样能搜到当前语料。"
            ),
            # 裁定③：索引整体重建 + 成本记账
            "index_full_rebuild": True,
            "index_full_rebuild_note": (
                "ingest/update 事件按当前文档集整体重建 CorpusIndex；c2/c3 行记 "
                "embedding_calls（重建嵌入的 chunk 数，零 LLM 成本，in/out tok 恒 0）；"
                "c2/c3 运行时共享同一索引实例，构建成本按条件镜像入账（per-condition "
                "曲线可比性优先）。"
            ),
            # study run 的合成研究问题披露
            "study_synthetic_question": True,
            "study_synthetic_question_note": (
                "ingest/update 的 study run 用 study_question() 由文档标题拼装的合成"
                "研究问题驱动（非评测题、不判分）；mock 模式蒸馏剧本覆盖该事件时段"
                "会出现的题集题目（run_eval 播种同款），real 模式由真实模型蒸馏。"
            ),
            "c1_retrieval_unchanged": True,
            "c1_retrieval_unchanged_note": (
                "c1 保持 run_eval 口径（forced deep + brief），成本曲线可与 real run 对齐；"
                "c4 query 改 auto 属预注册口径差异（spec §2）。"
            ),
        },
        "corpus_timeline": corpus_timeline,
        "row_schema": {
            "公共键": "condition/event_id/event_kind/qid/in_tok/out_tok/latency_ms/em/"
            "point_hits/refusal/citation_coverage/judge/gold_source/trace_id",
            "loop 行（c1/c4 query 与 c4 study）追加": "fresh_search_count/wiki_root/run_dir",
            "c4 loop 行追加": "policy_mode（run_dir/policy.json 的 P3 判定 mode）",
            "RAG query 行追加": "hit_urls（检索命中 URL——update 后 v2 命中可审计）",
            "索引行（c2/c3 ingest/update）追加": "embedding_calls（重建嵌入 chunk 数，零 LLM 成本）",
            "非 query 行": "em/point_hits/refusal/citation_coverage/judge 均为 None（不评分）；c1 不产行",
            "gold_override": "演化正确性 diff-gold——override 事件行 gold_source=override，"
            "em/point_hits/judge 按 override 要点判",
        },
        "rag_tokens_path": str(out_dir / "tokens.jsonl"),
        "judge_trace_suffix": "-judge",
        "embedding_model": str(getattr(embedding, "model", "") or ""),
        "c4_root": str(c4_root) if run_eval.CONDITION_C4 in conditions else None,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("-" * 76)
    print("── 条件×事件累计（in_tok；小样本评测，不构成收益结论——预注册判定见 manifest）──")
    for cond in conditions:
        cond_rows = [r for r in rows if r["condition"] == cond]
        total_in = sum(int(r["in_tok"]) for r in cond_rows)
        total_out = sum(int(r["out_tok"]) for r in cond_rows)
        print(f"  {cond:<4}n={len(cond_rows):<3}in_tok={total_in:<8}out_tok={total_out}")
    print(f"events.jsonl 已写入 {events_path}")
    print(f"manifest.json 已写入 {manifest_path}")
    return EXIT_PASS


def _strip_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """落盘前剥离行内临时键（build_study_row 的 _study_question 留痕打印用）。"""
    return {k: v for k, v in row.items() if not k.startswith("_")}


if __name__ == "__main__":
    raise SystemExit(main())
