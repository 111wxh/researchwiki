"""RAG harness（检索-生成器）测试（评测支线 task-5）。

覆盖 task-5 简报的契约：

- build_rag_user_prompt：检索命中按顺序 `[n] url` 编号，块内含 title 与
  snippet 全文（原样、不重截），长度约束"不超过 300 字"出现在 prompt 中，
  问题放最后一行；
- run_rag：ScriptedProvider 回固定回答；tokens 来自 call_text 的 on_usage
  回调（ScriptedProvider 默认 usage=TokenUsage(1200, 180)）；RagResult frozen；
- 记账：accountant 落 tokens.jsonl 一行，step="evals:rag"、model 随 provider；
- mode 透传：vector / hybrid 两次调用经计数 wrapper 证明 mode 真传到
  index.search，且 max_results=3（hits ≤ 3）；
- 空语料口径：检索为空时 prompt 写明"（无检索资料）"，仍正常走生成、
  RagResult 正常返回（见 rag_harness 模块 docstring）。

全部零网络、零真实模型调用；语料用真实 fixtures（只读）。
"""

import json
from pathlib import Path

import pytest

from researchwiki.evals.corpus import CorpusIndex, load_corpus
from researchwiki.evals.rag_harness import (
    RAG_SYSTEM,
    RagResult,
    build_rag_user_prompt,
    run_rag,
)
from researchwiki.llm.accounting import TokenAccountant
from researchwiki.llm.provider import ScriptedProvider, StreamEvent

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "evals" / "fixtures" / "ai-frameworks"

QUESTION = "LangChain 的核心架构是什么？"
ANSWER = "LangChain 采用分层架构 [1]。（ScriptedProvider 固定回答）"


def _scripted(answer: str = ANSWER) -> ScriptedProvider:
    """单回合脚本 Provider：回一段固定文本（usage 走默认 1200/180）。"""
    return ScriptedProvider([[StreamEvent(type="text_delta", delta=answer)]], model="mock-strong")


class _ModeSpy:
    """包一层 CorpusIndex 的计数 wrapper：记录 search 收到的 mode/max_results。"""

    def __init__(self, inner: CorpusIndex) -> None:
        self.inner = inner
        self.modes: list[str] = []
        self.max_results: list[int] = []

    def search(self, query: str, max_results: int = 5, mode: str = "hybrid"):
        self.modes.append(mode)
        self.max_results.append(max_results)
        return self.inner.search(query, max_results=max_results, mode=mode)


# ---------------------------------------------------------------------------
# build_rag_user_prompt：编号 / url / 长度约束 / 问题末行
# ---------------------------------------------------------------------------


def test_build_rag_user_prompt_numbers_hits_and_puts_question_last() -> None:
    """top-3 命中按 [n] url 编号，title/snippet 原样入 prompt，问题在末行。"""
    index = load_corpus(FIXTURE_DIR)
    hits = index.search(QUESTION, max_results=3, mode="hybrid")
    assert len(hits) == 3
    prompt = build_rag_user_prompt(QUESTION, hits)
    for i, hit in enumerate(hits, start=1):
        assert f"[{i}] {hit.url}" in prompt
        assert hit.title in prompt
        assert hit.snippet in prompt  # snippet 全文原样（CorpusIndex 已截 200 字）
    assert "不超过 300 字" in prompt
    assert prompt.splitlines()[-1] == f"问题：{QUESTION}"


def test_build_rag_user_prompt_empty_hits_notes_no_material() -> None:
    """空命中口径：prompt 写明"（无检索资料）"、无 [n] 块，问题仍在末行。"""
    prompt = build_rag_user_prompt(QUESTION, [])
    assert "（无检索资料）" in prompt
    assert "[1]" not in prompt
    assert prompt.splitlines()[-1] == f"问题：{QUESTION}"


# ---------------------------------------------------------------------------
# run_rag：RagResult / tokens 来自 on_usage / frozen
# ---------------------------------------------------------------------------


def test_run_rag_returns_answer_hits_tokens_and_latency() -> None:
    """固定回答落位；tokens 取自 on_usage（默认 1200/180）；hits ≤ 3；frozen。"""
    provider = _scripted()
    index = load_corpus(FIXTURE_DIR)
    result = run_rag(provider, index, question=QUESTION, mode="hybrid")
    assert isinstance(result, RagResult)
    assert result.answer == ANSWER
    assert 1 <= len(result.hits) <= 3
    assert (result.input_tokens, result.output_tokens) == (1200, 180)
    assert isinstance(result.latency_ms, float)
    assert result.latency_ms >= 0.0
    # 发出的 user prompt：带 [1] 编号、问题在末行
    [user_msg] = provider.calls[0]
    assert "[1]" in user_msg.content
    assert user_msg.content.splitlines()[-1] == f"问题：{QUESTION}"
    # 统一长度/引用口径写死在 RAG_SYSTEM（四个评测条件共用同一口径）
    assert "不超过 300 字" in RAG_SYSTEM
    assert "[n]" in RAG_SYSTEM
    assert "不得编造" in RAG_SYSTEM
    with pytest.raises(AttributeError):  # frozen：禁止改字段（FrozenInstanceError）
        result.answer = "改不动"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 记账：step="evals:rag" 落 tokens.jsonl
# ---------------------------------------------------------------------------


def test_run_rag_records_step_in_tokens_jsonl(tmp_path: Path) -> None:
    """accountant 恰记一行：step/model/trace_id/usage 与本次调用对齐。"""
    path = tmp_path / "tokens.jsonl"
    accountant = TokenAccountant(path)
    provider = _scripted()
    index = load_corpus(FIXTURE_DIR)
    run_rag(provider, index, question=QUESTION, accountant=accountant, trace_id="trace-rag-1")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["step"] == "evals:rag"
    assert row["model"] == "mock-strong"
    assert row["trace_id"] == "trace-rag-1"
    assert row["input_tokens"] == 1200
    assert row["output_tokens"] == 180


# ---------------------------------------------------------------------------
# mode 透传与空语料
# ---------------------------------------------------------------------------


def test_run_rag_mode_passthrough_vector_and_hybrid() -> None:
    """计数 wrapper 证明 mode 真传到 index.search；max_results=3（hits ≤ 3）。"""
    index = load_corpus(FIXTURE_DIR)
    spy = _ModeSpy(index)
    provider = _scripted()
    r_vector = run_rag(provider, spy, question=QUESTION, mode="vector")
    r_hybrid = run_rag(provider, spy, question=QUESTION, mode="hybrid")
    assert spy.modes == ["vector", "hybrid"]
    assert spy.max_results == [3, 3]
    assert 1 <= len(r_vector.hits) <= 3
    assert 1 <= len(r_hybrid.hits) <= 3


def test_run_rag_empty_corpus_still_answers() -> None:
    """空语料（CorpusIndex([])）：prompt 注明（无检索资料），RagResult 正常返回。"""
    provider = _scripted()
    result = run_rag(provider, CorpusIndex([]), question=QUESTION, mode="hybrid")
    assert result.answer == ANSWER
    assert result.hits == []
    assert (result.input_tokens, result.output_tokens) == (1200, 180)
    [user_msg] = provider.calls[0]
    assert "（无检索资料）" in user_msg.content
    assert user_msg.content.splitlines()[-1] == f"问题：{QUESTION}"
