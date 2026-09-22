"""Prior Reader 单测：active/merged/superseded 命中、空 Wiki、索引落后、标签行、max_chars 截断。

全部零网络：MockEmbeddingProvider(dim=512) + tokenizer=trigram（不依赖 vendor DLL）。
dim=512 是实测选择：blake2b 特征哈希在低维度（64/128）下会让无关中文串产生
非零余弦（向量通道误命中），512 维下本文件全部夹具对均正交或按预期区分，
且哈希确定性（跨进程可复现）保证该结论稳定。merged/superseded 用例让查询
短语只出现在旧笔记正文里，确保重定向命中路径（而非直接命中最终 active
note）被真实走到。
"""

from pathlib import Path

import pytest

from researchwiki.wiki.embeddings import MockEmbeddingProvider
from researchwiki.wiki.frontmatter import SourceRef
from researchwiki.wiki.index import SearchIndex
from researchwiki.wiki.prior import (
    PRIOR_CONTEXT_LABEL,
    PriorContext,
    ensure_index_fresh,
    format_prior_context,
    retrieve_priors,
)
from researchwiki.wiki.store import WikiStore


@pytest.fixture()
def store(tmp_path: Path) -> WikiStore:
    return WikiStore(tmp_path / "wiki-data")


def _make_index(store: WikiStore) -> SearchIndex:
    return SearchIndex(
        store.root,
        embedding=MockEmbeddingProvider(dim=512),
        tokenizer="trigram",
    )


# ---- active 命中 -------------------------------------------------------------


class TestActiveHit:
    def test_active_note_hit_content_correct(self, store: WikiStore) -> None:
        store.save_note(
            "上下文压缩技术可以把长对话压缩成摘要，保留关键事实与出处。",
            note_id="N-0001",
            title="上下文压缩",
            entities=["GLM-5.3"],
            confidence="high",
            volatility="stable",
            observed_at="2026-08-01T00:00:00+00:00",
            sources=[SourceRef(url="https://example.com/a", content_hash="a" * 64)],
        )
        with _make_index(store) as index:
            index.rebuild(store)
            ctx = retrieve_priors("上下文压缩技术", store, index, k=5)

            assert isinstance(ctx, PriorContext)
            assert [h.note_id for h in ctx.hits] == ["N-0001"]
            hit = ctx.hits[0]
            assert hit.title == "上下文压缩"
            assert hit.confidence == "high"
            assert hit.volatility == "stable"
            assert hit.observed_at == "2026-08-01T00:00:00+00:00"
            assert hit.source_urls == ["https://example.com/a"]
            assert hit.redirected_from == []
            assert "压缩成摘要" in hit.body
            assert hit.score > 0
            assert ctx.context_chars == len(ctx.format())
            assert ctx.context_chars > 0

    def test_observed_at_falls_back_to_created(self, store: WikiStore) -> None:
        store.save_note(
            "RAG 检索增强生成把外部检索与生成模型结合。",
            note_id="N-0001",
            title="RAG",
            observed_at=None,
            created="2026-07-15T00:00:00+00:00",
        )
        with _make_index(store) as index:
            index.rebuild(store)
            ctx = retrieve_priors("检索增强生成", store, index, k=5)
            assert ctx.hits[0].observed_at == "2026-07-15T00:00:00+00:00"


# ---- merged / superseded 命中 -------------------------------------------------


class TestRedirectedHit:
    def test_merged_hit_returns_final_active_note(self, store: WikiStore) -> None:
        # 查询短语"早期草稿介绍"只出现在 merged 旧笔记正文里 → 必走重定向路径
        store.save_note(
            "上下文压缩技术的早期草稿介绍，内容已过时。",
            note_id="N-0001",
            title="上下文压缩（旧）",
            status="merged",
            redirect_to="N-0002",
        )
        store.save_note(
            "上下文压缩技术可以把长对话压缩成摘要，保留关键事实。",
            note_id="N-0002",
            title="上下文压缩",
            confidence="high",
        )
        with _make_index(store) as index:
            index.rebuild(store)
            ctx = retrieve_priors("早期草稿介绍", store, index, k=5)

            assert [h.note_id for h in ctx.hits] == ["N-0002"]
            hit = ctx.hits[0]
            # 内容取最终 active note，不取 merged 旧笔记
            assert hit.title == "上下文压缩"
            assert "压缩成摘要" in hit.body
            assert "早期草稿" not in hit.body
            assert hit.confidence == "high"
            # 重定向链信息保留旧 ID
            assert hit.redirected_from == ["N-0001"]
            assert "redirected_from: N-0001" in ctx.format()

    def test_superseded_hit_walks_to_final_active_note(self, store: WikiStore) -> None:
        store.save_note(
            "向量数据库选型的旧结论：推荐 Milvus 1.x，此说法已被取代。",
            note_id="N-0001",
            title="向量库选型（旧）",
            status="superseded",
            superseded_by="N-0002",
        )
        store.save_note(
            "向量数据库选型的新结论：优先考虑托管服务与生态成熟度。",
            note_id="N-0002",
            title="向量库选型",
            confidence="medium",
        )
        with _make_index(store) as index:
            index.rebuild(store)
            ctx = retrieve_priors("此说法已被取代", store, index, k=5)

            assert [h.note_id for h in ctx.hits] == ["N-0002"]
            hit = ctx.hits[0]
            assert hit.title == "向量库选型"
            assert "托管服务" in hit.body
            assert hit.redirected_from == ["N-0001"]

    def test_chained_redirect_lists_all_old_ids(self, store: WikiStore) -> None:
        # N-0001 merged → N-0002 merged → N-0003 active：链上所有旧 ID 都要列出
        store.save_note(
            "提示词缓存的第一版说明，已合并进后续条目。",
            note_id="N-0001",
            title="提示词缓存 v1",
            status="merged",
            redirect_to="N-0002",
        )
        store.save_note(
            "提示词缓存的第二版说明，又被合并。",
            note_id="N-0002",
            title="提示词缓存 v2",
            status="merged",
            redirect_to="N-0003",
        )
        store.save_note(
            "提示词缓存：稳定前缀可显著降低重复请求的 token 成本。",
            note_id="N-0003",
            title="提示词缓存",
        )
        with _make_index(store) as index:
            index.rebuild(store)
            ctx = retrieve_priors("第一版说明", store, index, k=5)

            assert [h.note_id for h in ctx.hits] == ["N-0003"]
            assert ctx.hits[0].redirected_from == ["N-0001", "N-0002"]


# ---- 空 Wiki -----------------------------------------------------------------


class TestEmptyWiki:
    def test_empty_store_returns_empty_context(self, store: WikiStore) -> None:
        with _make_index(store) as index:
            index.rebuild(store)
            ctx = retrieve_priors("任何问题", store, index, k=5)
            assert isinstance(ctx, PriorContext)
            assert ctx.hits == []
            assert ctx.context_chars == 0
            assert ctx.format() == ""  # 空 Wiki：不抛异常，context 为空字符串

    def test_no_match_returns_empty_context(self, store: WikiStore) -> None:
        store.save_note("完全无关的笔记内容。", note_id="N-0001", title="无关")
        with _make_index(store) as index:
            index.rebuild(store)
            ctx = retrieve_priors("拓扑量子计算纠错", store, index, k=5)
            assert ctx.hits == []
            assert ctx.format() == ""


# ---- 索引落后 -----------------------------------------------------------------


class TestEnsureIndexFresh:
    def test_fresh_index_needs_no_rebuild(self, store: WikiStore) -> None:
        store.save_note("上下文压缩技术把长对话压缩成摘要。", note_id="N-0001", title="上下文压缩")
        with _make_index(store) as index:
            index.rebuild(store)
            rebuilt, reason = ensure_index_fresh(store, index)
            assert rebuilt is False
            assert "无需 rebuild" in reason

    def test_stale_index_detected_and_rebuilt(self, store: WikiStore) -> None:
        store.save_note("上下文压缩技术把长对话压缩成摘要。", note_id="N-0001", title="上下文压缩")
        with _make_index(store) as index:
            index.rebuild(store)
            # store 新增笔记但索引未更新 → 检测出落后并 rebuild
            store.save_note(
                "RAG 检索增强生成把外部检索与生成模型结合。", note_id="N-0002", title="RAG"
            )
            rebuilt, reason = ensure_index_fresh(store, index)
            assert rebuilt is True
            assert "N-0002" in reason
            # rebuild 之后检索能命中新笔记
            ctx = retrieve_priors("检索增强生成", store, index, k=5)
            assert [h.note_id for h in ctx.hits] == ["N-0002"]
            # 再次检查：已新鲜，不再 rebuild
            rebuilt2, reason2 = ensure_index_fresh(store, index)
            assert rebuilt2 is False
            assert "无需 rebuild" in reason2

    def test_status_change_counts_as_stale(self, store: WikiStore) -> None:
        store.save_note("上下文压缩技术把长对话压缩成摘要。", note_id="N-0001", title="上下文压缩")
        store.save_note("RAG 检索增强生成把外部检索与生成模型结合。", note_id="N-0002", title="RAG")
        with _make_index(store) as index:
            index.rebuild(store)
            # N-0002 被 merged（status 变化）→ 索引落后
            store.save_note(
                "RAG 检索增强生成把外部检索与生成模型结合。",
                note_id="N-0002",
                title="RAG",
                status="merged",
                redirect_to="N-0001",
            )
            rebuilt, reason = ensure_index_fresh(store, index)
            assert rebuilt is True
            assert "status 变化" in reason
            assert "N-0002" in reason


# ---- 格式化与预算 ---------------------------------------------------------------


class TestFormatAndBudget:
    def test_label_line_present_verbatim(self, store: WikiStore) -> None:
        store.save_note("上下文压缩技术把长对话压缩成摘要。", note_id="N-0001", title="上下文压缩")
        with _make_index(store) as index:
            index.rebuild(store)
            ctx = retrieve_priors("上下文压缩技术", store, index, k=5)
            text = ctx.format()
            assert PRIOR_CONTEXT_LABEL in text
            assert "历史 Prior，仅供核验，不是本轮 fresh evidence" in text
            # 标签独占一行
            assert f"## {PRIOR_CONTEXT_LABEL}\n" in text
            # format_prior_context 与 ctx.format() 对同一批 hits 输出一致
            assert format_prior_context(ctx.hits) == text

    def test_empty_hits_format_has_no_label(self) -> None:
        assert format_prior_context([]) == ""

    def test_max_chars_truncation_keeps_top_scores(self, store: WikiStore) -> None:
        for i in range(1, 4):
            store.save_note(
                f"主题{i}的长正文，包含大量细节描述。" * 25,
                note_id=f"N-{i:04d}",
                title=f"主题{i}",
            )
        with _make_index(store) as index:
            index.rebuild(store)
            ctx = retrieve_priors("主题", store, index, k=3, max_chars=1200)
            assert len(ctx.format()) <= 1200
            assert ctx.context_chars == len(ctx.format())
            assert 0 < len(ctx.hits) < 3  # 预算内放不下全部 → 截断
            # 按分数从高到低保留
            scores = [h.score for h in ctx.hits]
            assert scores == sorted(scores, reverse=True)

    def test_generous_budget_keeps_all_hits(self, store: WikiStore) -> None:
        for i in range(1, 4):
            store.save_note(
                f"主题{i}的长正文，包含大量细节描述。" * 25,
                note_id=f"N-{i:04d}",
                title=f"主题{i}",
            )
        with _make_index(store) as index:
            index.rebuild(store)
            ctx = retrieve_priors("主题", store, index, k=3, max_chars=8000)
            assert len(ctx.hits) == 3
            assert ctx.context_chars <= 8000

    def test_extreme_small_budget_returns_empty(self, store: WikiStore) -> None:
        store.save_note("上下文压缩技术把长对话压缩成摘要。", note_id="N-0001", title="上下文压缩")
        with _make_index(store) as index:
            index.rebuild(store)
            ctx = retrieve_priors("上下文压缩技术", store, index, k=5, max_chars=10)
            assert ctx.hits == []
            assert ctx.format() == ""
            assert ctx.context_chars == 0

    def test_body_truncated_to_max_body_chars(self, store: WikiStore) -> None:
        store.save_note("很长的正文。" * 200, note_id="N-0001", title="长文")
        with _make_index(store) as index:
            index.rebuild(store)
            ctx = retrieve_priors("很长的正文", store, index, k=5)
            assert ctx.hits[0].body.endswith("…")
            assert len(ctx.hits[0].body) <= 601  # 截断上限 + 省略号
