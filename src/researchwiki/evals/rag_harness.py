"""RAG harness（检索-生成器）：C2 Vector RAG / C3 Hybrid RAG 条件共用。

评测支线（PLAN §7.2）生成侧的数据口径：

- 四个评测条件（C2/C3 等）共用同一个 harness 与同一份 prompt 口径：检索
  top-3 chunk 按 ``[n] url`` 编号拼进 user prompt，**统一长度约束
  "回答不超过 300 字"与引用要求写死在 :data:`RAG_SYSTEM`**（docstring 即
  声明处），条件之间只有检索 mode 不同（C2="vector"、C3="hybrid"），
  生成链路逐字同款——复用 :func:`researchwiki.wiki.distiller.call_text`
  （step="evals:rag"），token 记账与蒸馏/评审同一条链路；
- snippet 用 :class:`researchwiki.evals.corpus.CorpusIndex` 返回的原样文本
  （其内部已截 200 字），本模块不重截、不加工；
- 空命中口径：检索结果为空（空语料或无命中）时 prompt 写明
  "（无检索资料）"，仍正常走一次生成调用、正常返回 :class:`RagResult`
  （模型应按 system 的"资料未提及"口径拒答，而不是报错跳过）；
- RagResult 的 input/output tokens 通过 call_text 的 ``on_usage`` 回调捕获
  （call_text 只返回 str；usage 为 None 时计 (0, 0)），记账本身由
  accountant 落 tokens.jsonl。
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from researchwiki.evals.corpus import CorpusIndex
from researchwiki.llm.accounting import TokenAccountant
from researchwiki.llm.provider import Provider, TokenUsage
from researchwiki.tools.search import SearchHit
from researchwiki.wiki.distiller import call_text

#: RAG 生成统一 system prompt：引用要求（[n] 标注）、拒答口径（资料未提及、
#: 不得编造）与统一长度约束（不超过 300 字）写死在这里——四个评测条件
#: 共用同一口径，保证条件间只差检索 mode、生成侧零变量。
RAG_SYSTEM = (
    "你是受控语料问答助手，只能基于给出的检索资料回答问题。"
    "回答中必须用 [n] 标注所依据资料的编号；"
    "资料不足以回答时，明确说明资料未提及，不得编造；"
    "回答不超过 300 字。"
)

#: 检索命中条数（top-3 拼 prompt）
TOP_K = 3
#: 空命中时 prompt 中的占位说明
_NO_MATERIAL = "（无检索资料）"


def build_rag_user_prompt(question: str, hits: Sequence[SearchHit]) -> str:
    """拼 RAG user prompt：要求行 → 检索资料块 → 问题（最后一行）。

    - 每块格式 ``[n] {url}\\n{title}\\n{snippet 全文}``，n 从 1 按传入顺序编号
      （snippet 用 CorpusIndex.search 返回的原样文本，不重截）；
    - hits 为空时资料区写明"（无检索资料）"，prompt 仍可构造（空命中口径）。
    """
    lines = [
        "回答要求：仅依据下列编号资料作答，用 [n] 标注所依据资料的编号，"
        "资料未提及的如实说明，回答不超过 300 字。",
        "",
        "检索资料：",
        "",
    ]
    if hits:
        for i, hit in enumerate(hits, start=1):
            lines.append(f"[{i}] {hit.url}")
            lines.append(hit.title)
            lines.append(hit.snippet)
            lines.append("")
    else:
        lines.append(_NO_MATERIAL)
        lines.append("")
    lines.append(f"问题：{question}")
    return "\n".join(lines)


@dataclass(frozen=True)
class RagResult:
    """一次 RAG 问答的完整产物（检索命中 + 生成回答 + token 记账）。

    - hits：检索命中（run_rag 固定 top-3，SearchHit：title/url/snippet）；
    - input/output_tokens：来自生成调用的 usage（on_usage 回调捕获），
      provider 未回 usage 时计 (0, 0)；
    - latency_ms：检索 + 生成的总耗时（clock 注入，可测）。
    """

    answer: str
    hits: list[SearchHit]
    input_tokens: int
    output_tokens: int
    latency_ms: float


def run_rag(
    provider: Provider,
    index: CorpusIndex,
    *,
    question: str,
    mode: str = "hybrid",
    accountant: TokenAccountant | None = None,
    trace_id: str = "",
    clock: Callable[[], float] = time.perf_counter,
) -> RagResult:
    """跑一次"检索 top-3 → 拼 prompt → 生成"的 RAG 问答，返回 :class:`RagResult`。

    C2/C3 共用本函数：mode="vector"（纯余弦）/ "hybrid"（融合打分）只影响
    ``index.search`` 的检索排序，prompt 与生成链路完全相同。

    生成调用逐字同款复用 :func:`researchwiki.wiki.distiller.call_text`
    （system=:data:`RAG_SYSTEM`、step="evals:rag"）；accountant 非 None 时
    tokens.jsonl 落一行；RagResult 的 tokens 经 ``on_usage`` 回调捕获，
    usage 为 None 时计 (0, 0)。
    """
    t0 = clock()
    hits = index.search(question, max_results=TOP_K, mode=mode)
    user = build_rag_user_prompt(question, hits)
    captured: list[TokenUsage] = []
    answer = call_text(
        provider,
        system=RAG_SYSTEM,
        user=user,
        step="evals:rag",
        accountant=accountant,
        trace_id=trace_id,
        clock=clock,
        on_usage=captured.append,
    )
    latency_ms = (clock() - t0) * 1000.0
    usage = captured[-1] if captured else None
    return RagResult(
        answer=answer,
        hits=list(hits),
        input_tokens=getattr(usage, "input_tokens", 0) or 0,
        output_tokens=getattr(usage, "output_tokens", 0) or 0,
        latency_ms=latency_ms,
    )


__all__ = ["RAG_SYSTEM", "RagResult", "TOP_K", "build_rag_user_prompt", "run_rag"]
