"""受控语料检索（fixture-corpus）：CorpusChunk / CorpusIndex / FixtureSearch。

评测支线（PLAN §7）检索侧的数据口径：

- 语料来自 ``evals/fixtures/<domain>/``，manifest.json 是事实来源（doc_id/
  title/url/version），provider 边界由 manifest.provider_note 声明——本模块
  是 fixture-corpus（受控合成语料，事实自洽）口径的检索器，零网络、不触
  真实 web 搜索；
- 两版本时效文档（doc_id 形如 ``<base>`` 与 ``<base>@v2``）同时入库：版本
  事实写在各自正文与文末修订说明里，检索靠查询中的版本词（版本号、人名等）
  区分命中；
- 切段：去掉 ``# 标题`` 行后按 ≤500 字贪心合并段落（段落在空白处断）；单段
  超长时再按句末标点切句合并——只在段落/句子边界断，不切断句子；
- 打分：vector=纯余弦；hybrid=0.5×余弦 + 0.5×词项重叠（字符 bigram Jaccard，
  query 对 chunk 正文）；余弦与 Jaccard 均为手写纯函数，不引 numpy；
- 嵌入向量化在索引构造时一次算齐（语料约几十 chunk），查询向量每次 search
  现算；同分按 chunk_id 升序 tie-break，结果完全可复现。
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from researchwiki.tools.search import SearchHit
from researchwiki.wiki.embeddings import EmbeddingProvider, MockEmbeddingProvider

#: 单 chunk 正文上限（字）
MAX_CHUNK_CHARS = 500
#: SearchHit.snippet 截断长度（字）
SNIPPET_CHARS = 200
#: hybrid 融合权重：score = VECTOR_WEIGHT×余弦 + TERM_WEIGHT×Jaccard
VECTOR_WEIGHT = 0.5
TERM_WEIGHT = 0.5

#: 句子边界：中文句末标点（含全角分号）或"英文句点+空白"（避免切碎 0.4.2 这类版本号）
_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？；!?;])|(?<=\. )")
#: 段落边界：空行（允许夹空白）
_PARAGRAPH_BOUNDARY = re.compile(r"\n\s*\n")


# ---------------------------------------------------------------------------
# 打分纯函数（手写，不引 numpy）
# ---------------------------------------------------------------------------


def _cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度；零向量或维度不齐按 0 分处理（不抛错）。"""
    if not a or len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / ((norm_a**0.5) * (norm_b**0.5))


def _bigrams(text: str) -> frozenset[str]:
    """字符 bigram 集合：先去空白，避免 bigram 跨词界拼接出假词项。"""
    compact = "".join(text.split())
    return frozenset(compact[i : i + 2] for i in range(len(compact) - 1))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Jaccard 重叠 |A∩B| / |A∪B|；任一集合为空记 0。"""
    if not a or not b:
        return 0.0
    union = len(a | b)
    return (len(a & b) / union) if union else 0.0


# ---------------------------------------------------------------------------
# 切段：段落优先整段并入，超长段落切句无缝并入
# ---------------------------------------------------------------------------


def _split_sentences(paragraph: str) -> list[str]:
    """按句末标点切段内句子（标点随前句保留），空段剔除。"""
    return [
        part for part in (s.strip() for s in _SENTENCE_BOUNDARY.split(paragraph)) if part
    ]


def _pack(units: Sequence[tuple[str, str]]) -> list[str]:
    """贪心装箱：units 为 (文本, 与前一文本的连接符)，放不下先落盘再开新块。

    单个文本自身超长时兜底硬截（正常语料——段落、句子都 ≤ 上限——到不了该分支）。
    """
    chunks: list[str] = []
    buf = ""
    for unit, joiner in units:
        candidate = f"{buf}{joiner}{unit}" if buf else unit
        if len(candidate) <= MAX_CHUNK_CHARS:
            buf = candidate
            continue
        if buf:
            chunks.append(buf)
        while len(unit) > MAX_CHUNK_CHARS:
            chunks.append(unit[:MAX_CHUNK_CHARS])
            unit = unit[MAX_CHUNK_CHARS:]
        buf = unit
    if buf:
        chunks.append(buf)
    return chunks


def _chunk_body(body: str) -> list[str]:
    """正文 → chunk 文本列表：段落在空白处（空行）断，句子按句末标点断。"""
    units: list[tuple[str, str]] = []
    for paragraph in (p.strip() for p in _PARAGRAPH_BOUNDARY.split(body)):
        if not paragraph:
            continue
        if len(paragraph) <= MAX_CHUNK_CHARS:
            units.append((paragraph, "\n\n"))
        else:
            for i, sentence in enumerate(_split_sentences(paragraph)):
                # 同段句子无缝拼接（保住原段落文本）；段首句与前一 chunk 用空行隔开
                units.append((sentence, "\n\n" if i == 0 else ""))
    return _pack(units)


# ---------------------------------------------------------------------------
# 数据结构与索引
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusChunk:
    """一个语料块。

    - chunk_id：全局唯一（``<doc_id>#c<序号>``），同分 tie-break 用；
    - doc_id：来源文档（manifest.docs[].doc_id，@v2 时效版含在 id 里）；
    - url / title：直接取 manifest（url 形如 fixture://ai-frameworks/<doc_id>）；
    - text：chunk 正文（不含 `# 标题` 行）。
    """

    chunk_id: str
    doc_id: str
    url: str
    title: str
    text: str


class CorpusIndex:
    """受控语料索引：构造时对全部 chunk 一次算齐向量，search 融合打分排序。"""

    def __init__(
        self,
        chunks: Sequence[CorpusChunk],
        embedding: EmbeddingProvider | None = None,
    ) -> None:
        self.chunks = list(chunks)
        self.embedding: EmbeddingProvider = (
            embedding if embedding is not None else MockEmbeddingProvider()
        )
        # 嵌入输入用 title + 正文：标题里的实体词（langchain-core 等）参与向量区分
        inputs = [f"{c.title}\n{c.text}" for c in self.chunks]
        vectors = self.embedding.embed(inputs) if inputs else []
        if len(vectors) != len(inputs):
            raise ValueError("嵌入 provider 返回的向量数量与输入不一致")
        self._vectors = vectors

    def search(
        self, query: str, max_results: int = 5, mode: str = "hybrid"
    ) -> list[SearchHit]:
        """检索语料块：vector=纯余弦；hybrid=0.5×余弦 + 0.5×bigram Jaccard。

        同分按 chunk_id 升序 tie-break；返回 SearchHit（url 取 manifest 的
        fixture:// 形态，snippet=chunk 正文前 200 字）。
        """
        if mode not in ("vector", "hybrid"):
            raise ValueError(f"未知检索模式：{mode}（允许：vector、hybrid）")
        (query_vec,) = self.embedding.embed([query])
        query_grams = _bigrams(query)
        scored: list[tuple[float, CorpusChunk]] = []
        for chunk, chunk_vec in zip(self.chunks, self._vectors, strict=True):
            cosine = _cosine(query_vec, chunk_vec)
            if mode == "vector":
                score = cosine
            else:
                overlap = _jaccard(query_grams, _bigrams(chunk.text))
                score = VECTOR_WEIGHT * cosine + TERM_WEIGHT * overlap
            scored.append((score, chunk))
        scored.sort(key=lambda pair: (-pair[0], pair[1].chunk_id))
        return [
            SearchHit(title=c.title, url=c.url, snippet=c.text[:SNIPPET_CHARS])
            for _, c in scored[:max_results]
        ]


def load_corpus(
    fixture_dir: str | Path,
    *,
    embedding: EmbeddingProvider | None = None,
    doc_ids: Sequence[str] | None = None,
) -> CorpusIndex:
    """读 manifest → 逐篇 markdown → 切段 → 建索引。

    manifest.docs 是事实来源：doc_id 对应 ``docs/<doc_id>.md``，url/title 直接
    取 manifest，两版本时效文档都入库（不做去重）；embedding 缺省
    MockEmbeddingProvider（确定性 bigram 哈希，零网络）。

    ``doc_ids``：可选文档集过滤（缺省 None = 全部 manifest 文档，行为不变）——
    序列重放 harness 按事件时间线建子集索引用；传入时仅索引列出的 doc_id
    （顺序仍按 manifest 定义序），不在 manifest 中的名字抛 ``ValueError``。
    """
    root = Path(fixture_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    docs = manifest["docs"]
    if doc_ids is not None:
        wanted = list(doc_ids)
        known = {str(doc["doc_id"]) for doc in docs}
        unknown = [d for d in wanted if d not in known]
        if unknown:
            raise ValueError(f"doc_ids 不在语料 manifest 中：{'、'.join(unknown)}")
        docs = [doc for doc in docs if str(doc["doc_id"]) in set(wanted)]
    chunks: list[CorpusChunk] = []
    for doc in docs:
        doc_id = str(doc["doc_id"])
        raw = (root / "docs" / f"{doc_id}.md").read_text(encoding="utf-8")
        # 去 `# 标题` 行（标题经 manifest.title 单独保存，不入正文重复计分）
        body = "\n".join(line for line in raw.splitlines() if not line.startswith("#"))
        for i, text in enumerate(_chunk_body(body)):
            chunks.append(
                CorpusChunk(
                    chunk_id=f"{doc_id}#c{i:02d}",
                    doc_id=doc_id,
                    url=str(doc["url"]),
                    title=str(doc["title"]),
                    text=text,
                )
            )
    return CorpusIndex(chunks, embedding=embedding)


class FixtureSearch:
    """把 CorpusIndex 包成 SearchProvider 协议实现（评测 runner 直接可注入）。

    calls 记录 (query, max_results) 调用序列，对齐 MockSearch 惯例，
    供测试与 runner 统计 fresh search 次数。
    """

    def __init__(self, index: CorpusIndex, mode: str = "hybrid") -> None:
        self.index = index
        self.mode = mode
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, max_results: int = 5) -> list[SearchHit]:
        """委托 index.search 并透传构造时的 mode，同时计数。"""
        self.calls.append((query, max_results))
        return self.index.search(query, max_results=max_results, mode=self.mode)


__all__ = [
    "CorpusChunk",
    "CorpusIndex",
    "FixtureSearch",
    "MAX_CHUNK_CHARS",
    "SNIPPET_CHARS",
    "TERM_WEIGHT",
    "VECTOR_WEIGHT",
    "load_corpus",
]
