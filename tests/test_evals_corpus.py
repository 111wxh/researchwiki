"""受控语料检索测试（评测支线 task-2）。

覆盖 task-2 简报的契约：

- CorpusChunk frozen dataclass：chunk_id/doc_id/url/title/text；
- load_corpus：manifest 为事实来源 → 逐篇 markdown → 去 `# 标题` 行 →
  ≤500 字贪心切段（段落在空白处断、不切断句子），两版本时效文档同时入库；
  embedding 缺省 MockEmbeddingProvider（零网络）；
- CorpusIndex.search：vector=纯余弦（手写，不引 numpy）；hybrid=0.5×余弦 +
  0.5×字符 bigram Jaccard；同分 tie-break 按 chunk_id 升序；
  SearchHit.url 以 fixture:// 开头、snippet=正文前 200 字、max_results 截断；
- 时效对版本区分：v2 特有事实词命中 @v2 chunk，v1 特有词命中 v1 chunk；
- FixtureSearch：SearchProvider 协议 + calls 计数器（对齐 MockSearch 惯例）。

全部零网络：真实 fixtures 目录只读使用；融合语义用 stub 嵌入精确构造
（余弦 0.6 vs 0.0 + Jaccard 1.0），避免 Mock 嵌入下混合排序断言不可写死的问题。
"""

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from researchwiki.evals.corpus import (
    CorpusChunk,
    CorpusIndex,
    FixtureSearch,
    load_corpus,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "evals" / "fixtures" / "ai-frameworks"


class StubEmbedding:
    """按精确文本映射预制向量的嵌入 stub：融合排序断言用，完全可复现。"""

    model = "stub"

    def __init__(self, table: dict[str, list[float]]) -> None:
        self._table = table

    def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            return [list(self._table[t]) for t in texts]
        except KeyError as exc:  # 防呆：键写错立刻炸，而不是给出错向量
            raise AssertionError(f"StubEmbedding 缺少键：{exc}") from exc


def _make_fixture(root: Path, docs: dict[str, tuple[str, str]]) -> None:
    """写一个最小 fixture 目录：docs 为 doc_id → (标题, 正文)。"""
    (root / "docs").mkdir(parents=True, exist_ok=True)
    manifest = {
        "domain": "demo",
        "provider_note": "fixture-corpus（测试合成语料）",
        "docs": [
            {
                "doc_id": doc_id,
                "title": title,
                "url": f"fixture://demo/{doc_id}",
                "version": "v1",
            }
            for doc_id, (title, _) in docs.items()
        ],
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    for doc_id, (title, body) in docs.items():
        (root / "docs" / f"{doc_id}.md").write_text(
            f"# {title}\n\n{body}\n", encoding="utf-8"
        )


@pytest.fixture(scope="module")
def index() -> CorpusIndex:
    """真实 fixtures 目录（仓库数据，只读）上建一次索引，模块内复用。"""
    return load_corpus(FIXTURE_DIR)


# ---------------------------------------------------------------------------
# CorpusChunk 与 load_corpus
# ---------------------------------------------------------------------------


def test_corpus_chunk_is_frozen() -> None:
    """frozen dataclass：五个字段齐全，禁止就地修改。"""
    chunk = CorpusChunk(
        chunk_id="d#c00", doc_id="d", url="fixture://d", title="t", text="x"
    )
    assert (chunk.chunk_id, chunk.doc_id, chunk.url, chunk.title, chunk.text) == (
        "d#c00",
        "d",
        "fixture://d",
        "t",
        "x",
    )
    with pytest.raises(FrozenInstanceError):
        chunk.text = "y"  # type: ignore[misc]


def test_load_corpus_indexes_all_manifest_docs(index: CorpusIndex) -> None:
    """manifest 21 篇（18 基础 + 3 个 @v2）全部入库；url/title 取 manifest；
    每块正文 ≤500 字；chunk_id 唯一；缺省嵌入为 Mock。"""
    manifest = json.loads((FIXTURE_DIR / "manifest.json").read_text(encoding="utf-8"))
    by_doc: dict[str, list[CorpusChunk]] = {}
    for chunk in index.chunks:
        by_doc.setdefault(chunk.doc_id, []).append(chunk)
    assert set(by_doc) == {d["doc_id"] for d in manifest["docs"]}
    for doc in manifest["docs"]:
        chunks = by_doc[doc["doc_id"]]
        assert chunks, doc["doc_id"]
        for chunk in chunks:
            assert chunk.url == doc["url"]
            assert chunk.title == doc["title"]
            assert 0 < len(chunk.text) <= 500
    chunk_ids = [c.chunk_id for c in index.chunks]
    assert len(chunk_ids) == len(set(chunk_ids))
    assert index.embedding.model == "mock-embedding"


def test_load_corpus_strips_heading_keeps_body_and_revision_note(
    index: CorpusIndex,
) -> None:
    """`# 标题` 行不入正文；正文与 v2 文末修订说明保留（版本事实供检索区分）。"""
    texts = [c.text for c in index.chunks]
    assert not any(t.lstrip().startswith("#") for t in texts)
    assert any("LCEL" in t for t in texts)
    v2_texts = [c.text for c in index.chunks if c.doc_id == "langchain-core@v2"]
    assert any("修订说明：" in t for t in v2_texts)
    v1_texts = [c.text for c in index.chunks if c.doc_id == "langchain-core"]
    assert any("重试风暴" in t for t in v1_texts)


def test_chunking_respects_limit_and_never_cuts_units(tmp_path: Path) -> None:
    """切段口径：段落整段并入（不切断）；单段超长时按句切分（不切断句子）；
    所有 chunk ≤500 字。"""
    paras = [f"第{i}段：" + "内容句子。" * 19 + "收尾句。" for i in range(7)]
    sentences = [f"第{i:02d}号句子记录一条用于切块测试的事实内容并给出结论。" for i in range(20)]
    root = tmp_path / "fx"
    _make_fixture(
        root,
        {
            "paras-doc": ("分段演示", "\n\n".join(paras)),
            "long-doc": ("长段演示", "".join(sentences)),
        },
    )
    chunks = load_corpus(root).chunks
    assert all(len(c.text) <= 500 for c in chunks)
    paras_chunks = [c for c in chunks if c.doc_id == "paras-doc"]
    # 每个 ≤500 的段落完整落在唯一 chunk 中（出现两次/丢失都算切断段落）
    placed = [p for p in paras for c in paras_chunks if p in c.text]
    assert len(placed) == len(paras)
    long_chunks = [c for c in chunks if c.doc_id == "long-doc"]
    assert len(long_chunks) >= 2  # 600+ 字单段确被切分
    for sentence in sentences:  # 每个句子完整出现在某个 chunk 中
        assert any(sentence in c.text for c in long_chunks), sentence


# ---------------------------------------------------------------------------
# 检索排序（真实语料 + stub 精确构造）
# ---------------------------------------------------------------------------


def test_relevant_query_ranks_target_doc_first(index: CorpusIndex) -> None:
    """相关 query（含 0.4.2 版本事实词）→ 对应文档 chunk 排最前（top1 断言）。"""
    hits = index.search("LangChain 版本 0.4.2", mode="hybrid")
    assert hits[0].url == "fixture://ai-frameworks/langchain-core@v2"


def test_temporal_pair_versions_discriminated(index: CorpusIndex) -> None:
    """时效对区分：v2 特有事实词（赵岚——仅修订版"共同负责"）命中 @v2 chunk，
    v1 特有事实词（重试风暴——修订时删除的表述）命中 v1 chunk。

    查询不带 "langchain-core" 前缀词做 v2 检索：两版本标题相同，标题词会把
    v1 chunk 一并拉高，掩盖版本词的区分信号（探针验证过的口径）。
    """
    v2_hits = index.search("赵岚 0.4.2", mode="hybrid")
    assert v2_hits[0].url == "fixture://ai-frameworks/langchain-core@v2"
    v1_hits = index.search("langchain-core 重试风暴", mode="hybrid")
    assert v1_hits[0].url == "fixture://ai-frameworks/langchain-core"


def test_hybrid_fusion_flips_ranking_by_term_overlap() -> None:
    """hybrid 受词项重叠影响：余弦占优的 chunk 在 vector 下居前；
    词项完全重叠（Jaccard=1.0）的 chunk 在 hybrid 下反超。
    stub 嵌入精确控制余弦 0.6 vs 0.0，断言完全可复现。"""
    query = "数据库连接池配置"
    text_a = "向量检索索引构建"  # 与 query 零 bigram 重叠
    text_b = "数据库连接池配置"  # 与 query bigram 集合一致 → Jaccard 1.0
    chunk_a = CorpusChunk("demo#a", "demo-a", "fixture://demo/a", "文档甲", text_a)
    chunk_b = CorpusChunk("demo#b", "demo-b", "fixture://demo/b", "文档乙", text_b)
    stub = StubEmbedding(
        {
            f"文档甲\n{text_a}": [0.6, 0.8],  # 与 query [1,0] 余弦 0.6
            f"文档乙\n{text_b}": [0.0, 1.0],  # 与 query 余弦 0.0
            query: [1.0, 0.0],
        }
    )
    idx = CorpusIndex([chunk_a, chunk_b], embedding=stub)
    vector_order = [h.url for h in idx.search(query, mode="vector")]
    hybrid_order = [h.url for h in idx.search(query, mode="hybrid")]
    assert vector_order == ["fixture://demo/a", "fixture://demo/b"]  # 0.6 > 0.0
    assert hybrid_order == ["fixture://demo/b", "fixture://demo/a"]  # 0.5 > 0.3


def test_same_score_ties_break_by_chunk_id() -> None:
    """同分 tie-break：两 chunk 向量分完全相同（纯 vector 模式）时按 chunk_id 升序。"""
    chunk_low = CorpusChunk("demo#01", "demo", "fixture://demo", "同题", "正文乙")
    chunk_high = CorpusChunk("demo#02", "demo", "fixture://demo", "同题", "正文甲")
    stub = StubEmbedding(
        {"同题\n正文乙": [1.0, 0.0], "同题\n正文甲": [1.0, 0.0], "正文": [1.0, 0.0]}
    )
    hits = CorpusIndex([chunk_high, chunk_low], embedding=stub).search(
        "正文", max_results=2, mode="vector"
    )
    assert [h.snippet for h in hits] == ["正文乙", "正文甲"]


def test_search_hit_shape_truncation_and_mode_validation(index: CorpusIndex) -> None:
    """SearchHit 形状：url 以 fixture:// 开头、snippet=对应 chunk 正文前 200 字；
    max_results 截断生效（含默认 5 与超出总量两种）；未知 mode 报 ValueError。"""
    hits = index.search("Milvus 一致性级别 Strong", max_results=3, mode="hybrid")
    assert len(hits) == 3
    for hit in hits:
        assert hit.url.startswith("fixture://ai-frameworks/")
        candidates = [c for c in index.chunks if c.url == hit.url]
        assert candidates
        assert any(
            c.title == hit.title and c.text[:200] == hit.snippet for c in candidates
        )
    assert len(index.search("Milvus", max_results=len(index.chunks) + 10)) == len(
        index.chunks
    )
    assert len(index.search("Milvus")) == 5  # 默认 max_results=5
    assert len(index.search("Milvus", max_results=5, mode="vector")) == 5
    with pytest.raises(ValueError, match="未知检索模式"):
        index.search("任意", mode="bm25")


# ---------------------------------------------------------------------------
# FixtureSearch：协议实现 + calls 计数 + mode 透传
# ---------------------------------------------------------------------------


def test_fixture_search_delegates_counts_and_mode_passthrough(
    index: CorpusIndex,
) -> None:
    """FixtureSearch 委托 index.search 并透传 mode；calls 记录 (query, max_results)。"""
    fs = FixtureSearch(index)
    hits = fs.search("FAISS 向量检索", max_results=2)
    assert len(hits) == 2
    assert fs.calls == [("FAISS 向量检索", 2)]
    fs.search("Chroma 轻量嵌入式")  # 默认 max_results=5
    assert fs.calls[-1] == ("Chroma 轻量嵌入式", 5)
    assert len(fs.calls) == 2
    # mode 透传：vector 模式的 FixtureSearch 与 index.search(mode="vector") 逐条一致
    fs_vec = FixtureSearch(index, mode="vector")
    assert fs_vec.search("FAISS 向量检索") == index.search(
        "FAISS 向量检索", mode="vector"
    )
    assert len(fs_vec.calls) == 1
