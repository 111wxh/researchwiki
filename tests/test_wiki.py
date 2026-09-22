"""wiki 子系统单测：tmp_path 隔离、MockEmbedding/MockTransport、零真实网络、中文夹具。

覆盖：frontmatter round-trip（未知字段保留、loop 层轻量格式兼容）、实体别名归一、
redirect 链式跟随与防环、store 只回 active、FTS5 中文 recall（simple/trigram 两种
tokenizer 都要过）、向量检索 mock 相似度断言、RRF 合并、置信/新鲜度因子（注入
clock）、嵌入缓存命中、工厂无 key 回退 mock。
"""

import hashlib
import json
import sqlite3
import tomllib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from researchwiki.loop.notes import NoteStore
from researchwiki.wiki.embeddings import (
    CachedEmbeddingProvider,
    EmbeddingError,
    MockEmbeddingProvider,
    OpenAICompatibleEmbedding,
    get_embedding_provider,
)
from researchwiki.wiki.entities import EntityRegistry, slugify
from researchwiki.wiki.frontmatter import NoteMeta, SourceRef, dump, parse
from researchwiki.wiki.index import (
    SearchIndex,
    SearchMatch,
    freshness_factor,
    rrf_fuse,
    wiki_search,
    wiki_settings,
)
from researchwiki.wiki.store import WikiStore, source_snapshot_path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXED_NOW = datetime(2026, 9, 20, 12, 0, 0, tzinfo=UTC)


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b, strict=True))
    den = sum(x * x for x in a) ** 0.5 * sum(y * y for y in b) ** 0.5
    return num / den if den else 0.0


@pytest.fixture()
def store(tmp_path: Path) -> WikiStore:
    return WikiStore(tmp_path / "wiki-data")


# ---- frontmatter ------------------------------------------------------


class TestFrontmatter:
    def test_round_trip_full(self):
        meta = NoteMeta(
            id="N-0001",
            title="上下文压缩",
            entities=["glm-5-3"],
            confidence="high",
            status="merged",
            redirect_to="N-0002",
            volatility="drifting",
            observed_at="2026-05-01T00:00:00+00:00",
            created="2026-05-01T08:00:00+00:00",
            trace_id="run-42",
            sources=[SourceRef(url="https://example.com/a", content_hash="a" * 64)],
        )
        meta.extra = {"custom_note": {"nested": [1, 2]}}
        body = "正文第一行。\n第二行。\n"
        text = dump(meta.to_dict(), body)

        meta_dict, parsed_body = parse(text)
        assert parsed_body == body
        restored = NoteMeta.from_dict(meta_dict)
        assert restored == meta
        # dump 是稳定的：再 dump 一次内容一致
        assert parse(dump(meta_dict, parsed_body)) == (meta_dict, parsed_body)

    def test_unknown_fields_preserved(self):
        text = "---\nid: N-0001\nweird_field: 你好\nnested:\n  a: 1\n---\n正文\n"
        meta_dict, body = parse(text)
        meta = NoteMeta.from_dict(meta_dict)
        assert meta.extra == {"weird_field": "你好", "nested": {"a": 1}}
        # round-trip 后未知字段原样保留
        meta2_dict, _ = parse(dump(meta.to_dict(), body))
        assert NoteMeta.from_dict(meta2_dict).extra == meta.extra

    def test_light_loop_format_compatible(self):
        """loop 层 NoteStore 写的轻量 frontmatter（无 title/status 等）必须能读。"""
        text = (
            "---\n"
            "id: N-0001\n"
            'entities: ["GLM-5.3"]\n'
            "confidence: medium\n"
            "created: 2026-05-01T00:00:00+00:00\n"
            "trace_id: run-42\n"
            "---\n"
            "\n"
            "笔记正文。\n"
        )
        meta_dict, body = parse(text)
        meta = NoteMeta.from_dict(meta_dict)
        assert body == "\n笔记正文。\n"
        assert meta.id == "N-0001"
        assert meta.entities == ["GLM-5.3"]
        assert meta.confidence == "medium"
        # YAML 时间戳对象被归一回 ISO 字符串
        assert meta.created == "2026-05-01T00:00:00+00:00"
        assert meta.status == "active" and meta.volatility == "stable"
        assert meta.sources == [] and meta.title == ""

    def test_parse_without_frontmatter(self):
        meta, body = parse("纯正文，没有 frontmatter。")
        assert meta == {} and body == "纯正文，没有 frontmatter。"

    def test_enum_normalization(self):
        meta = NoteMeta.from_dict({"id": "N-1", "confidence": "HIGH", "status": "Merged"})
        assert meta.confidence == "high" and meta.status == "merged"
        meta2 = NoteMeta.from_dict({"id": "N-1", "confidence": "whatever"})
        assert meta2.confidence == "medium"

    def test_kind_round_trip(self):
        """kind 全值域 round-trip；to_dict 始终输出 kind。"""
        for kind in ("knowledge", "user", "experience"):
            meta = NoteMeta.from_dict({"id": "N-1", "kind": kind})
            assert meta.kind == kind
            meta_dict, _ = parse(dump(meta.to_dict(), "正文"))
            assert meta_dict["kind"] == kind
            assert NoteMeta.from_dict(meta_dict).kind == kind

    def test_kind_tolerant_parsing(self):
        """kind 宽容解析：大小写归一、非法值回退 knowledge。"""
        assert NoteMeta.from_dict({"id": "N-1", "kind": "USER"}).kind == "user"
        assert NoteMeta.from_dict({"id": "N-1", "kind": " Experience "}).kind == "experience"
        assert NoteMeta.from_dict({"id": "N-1", "kind": "diary"}).kind == "knowledge"
        assert NoteMeta.from_dict({"id": "N-1", "kind": 123}).kind == "knowledge"
        assert NoteMeta.from_dict({"id": "N-1"}).kind == "knowledge"

    def test_importance_round_trip(self):
        """importance 合法值 round-trip；None 时 to_dict 省略该字段。"""
        meta = NoteMeta.from_dict({"id": "N-1", "importance": 0.8})
        assert meta.importance == 0.8
        meta_dict, _ = parse(dump(meta.to_dict(), "正文"))
        assert meta_dict["importance"] == 0.8
        assert NoteMeta.from_dict(meta_dict).importance == 0.8
        # None → 序列化时省略（旧数据兼容：frontmatter 里不出现该键）
        plain = NoteMeta(id="N-1")
        assert "importance" not in plain.to_dict()

    def test_importance_tolerant_parsing(self):
        """importance 宽容解析：仅认 0.0–1.0 的数值，其余一律 None。"""
        assert NoteMeta.from_dict({"id": "N-1", "importance": 0}).importance == 0.0
        assert NoteMeta.from_dict({"id": "N-1", "importance": 1}).importance == 1.0
        assert NoteMeta.from_dict({"id": "N-1", "importance": 0.5}).importance == 0.5
        assert NoteMeta.from_dict({"id": "N-1", "importance": "0.8"}).importance is None
        assert NoteMeta.from_dict({"id": "N-1", "importance": True}).importance is None
        assert NoteMeta.from_dict({"id": "N-1", "importance": 1.5}).importance is None
        assert NoteMeta.from_dict({"id": "N-1", "importance": -0.1}).importance is None
        assert NoteMeta.from_dict({"id": "N-1"}).importance is None

    def test_old_note_without_kind_importance_compat(self):
        """旧笔记（无 kind/importance）round-trip 后 kind=knowledge、importance=None。"""
        text = (
            "---\n"
            "id: N-0001\n"
            "title: 旧笔记\n"
            "confidence: high\n"
            "created: 2026-05-01T00:00:00+00:00\n"
            "---\n正文\n"
        )
        meta_dict, body = parse(text)
        meta = NoteMeta.from_dict(meta_dict)
        assert meta.kind == "knowledge" and meta.importance is None
        restored = NoteMeta.from_dict(parse(dump(meta.to_dict(), body))[0])
        assert restored.kind == "knowledge" and restored.importance is None
        assert restored.confidence == "high" and restored.created == meta.created


# ---- entities ---------------------------------------------------------


class TestEntities:
    def test_slugify(self):
        assert slugify("GLM-5.3") == "glm-5-3"
        assert slugify("  上下文压缩 ") == "上下文压缩"
        assert slugify("A/B*C") == "a-b-c"
        assert slugify("***").startswith("e-")

    def test_get_or_create_and_resolve(self, store: WikiStore):
        reg = EntityRegistry(store.root)
        entity = reg.get_or_create("GLM-5.3", ["智谱旗舰"])
        assert entity.id == "glm-5-3"
        # 文件落盘 + 持久化
        assert (store.root / "entities" / "glm-5-3.md").is_file()
        assert reg.resolve(" glm-5.3 ") is entity  # 大小写/空白归一
        assert reg.resolve("智谱旗舰") is entity
        assert reg.resolve("不存在的实体") is None

    def test_alias_merge_no_duplicates(self, store: WikiStore):
        reg = EntityRegistry(store.root)
        reg.get_or_create("GLM-5.3", ["智谱旗舰"])
        again = reg.get_or_create("GLM-5.3", ["智谱旗舰", "GLM 5.3 旗舰版"])
        assert again.id == "glm-5-3"
        assert again.aliases.count("智谱旗舰") == 1
        assert "GLM 5.3 旗舰版" in again.aliases

    def test_slug_collision_merges(self, store: WikiStore):
        """不同写法 slug 相同（glm-5.3 / glm-5-3）→ 归并到同一实体。"""
        reg = EntityRegistry(store.root)
        first = reg.get_or_create("GLM-5.3")
        second = reg.get_or_create("glm-5-3")
        assert second.id == first.id
        assert second.name == "GLM-5.3"
        assert "glm-5-3" in second.aliases

    def test_persistence_and_reload(self, store: WikiStore):
        EntityRegistry(store.root).get_or_create("上下文压缩")
        # 新实例从磁盘重建索引
        fresh = EntityRegistry(store.root)
        assert fresh.resolve("上下文压缩") is not None
        assert [e.id for e in fresh.list_entities()] == ["上下文压缩"]
        # reload 拾取进程外新增
        fresh.get_or_create("向量检索")
        EntityRegistry(store.root).reload()
        assert EntityRegistry(store.root).resolve("向量检索") is not None


# ---- store ------------------------------------------------------------


class TestStore:
    def test_save_and_get_round_trip(self, store: WikiStore):
        note = store.save_note(
            "上下文压缩技术显著降低长会话成本。",
            title="上下文压缩",
            entities=["glm-5-3"],
            confidence="high",
            volatility="volatile",
            observed_at="2026-05-01T00:00:00+00:00",
            trace_id="run-1",
            sources=[SourceRef(url="https://example.com/a", content_hash="a" * 64)],
        )
        loaded = store.get_note(note.id)
        assert loaded is not None
        assert loaded.id == note.id and loaded.title == "上下文压缩"
        assert loaded.body == "上下文压缩技术显著降低长会话成本。"
        assert loaded.meta.confidence == "high"
        assert loaded.meta.sources[0].url == "https://example.com/a"
        assert loaded.meta.entities == ["glm-5-3"]

    def test_save_note_kind_importance_round_trip(self, store: WikiStore):
        """save_note 透传 kind/importance：落盘后读回一致，默认值向后兼容。"""
        note = store.save_note(
            "用户偏好深色主题。",
            title="用户偏好",
            kind="user",
            importance=0.9,
        )
        assert note.kind == "user" and note.importance == 0.9
        loaded = store.get_note(note.id)
        assert loaded is not None
        assert loaded.kind == "user" and loaded.importance == 0.9
        default = store.save_note("普通知识笔记", title="知识")
        assert default.kind == "knowledge" and default.importance is None

    def test_numbering_continues_from_loop_notes(self, store: WikiStore):
        """loop 层已写过 N-0001 时，WikiStore 必须续接编号而非从 N-0001 重来。"""
        NoteStore(store.notes_dir).save({"text": "loop 层笔记", "entities": ["x"]})
        assert store.list_notes(status=None)[0].id == "N-0001"
        note = store.save_note("wiki 层笔记")
        assert note.id == "N-0002"

    def test_list_notes_active_only(self, store: WikiStore):
        store.save_note("现役", title="a")
        store.save_note("已合并", title="b", status="merged", redirect_to="N-0001")
        store.save_note("已废止", title="c", status="superseded", superseded_by="N-0001")
        active = store.list_notes()
        assert [n.id for n in active] == ["N-0001"]
        everything = store.list_notes(status=None)
        assert [n.id for n in everything] == ["N-0001", "N-0002", "N-0003"]

    def test_follow_redirect_chain(self, store: WikiStore):
        final = store.save_note("最终版本")
        merged = store.save_note("", status="merged", redirect_to=final.id)
        superseded = store.save_note("", status="superseded", superseded_by=merged.id)
        assert store.follow_redirect(superseded.id).id == final.id
        assert store.follow_redirect(merged.id).id == final.id
        assert store.follow_redirect(final.id).id == final.id
        assert store.follow_redirect("N-9999") is None
        # 链条断裂（merged 指向不存在的笔记）→ None
        dangling = store.save_note("", status="merged", redirect_to="N-9999")
        assert store.follow_redirect(dangling.id) is None

    def test_follow_redirect_cycle(self, store: WikiStore):
        a = store.save_note("", status="merged", redirect_to="N-0002")
        store.save_note("", status="merged", redirect_to=a.id)
        with pytest.raises(ValueError, match="cycle"):
            store.follow_redirect(a.id)

    def test_snapshot_paths(self, store: WikiStore):
        note = store.save_note(
            "正文",
            sources=[SourceRef(url="https://example.com/x", content_hash="b" * 64)],
        )
        paths = store.note_snapshot_paths(store.get_note(note.id))
        assert len(paths) == 1
        # 与 tools.fetch 的落盘布局对齐：sources/{sha1(url)}/{content_hash}/content.md
        snapshot = source_snapshot_path(store.sources_dir, "https://example.com/x", "b" * 64)
        assert paths[0] == snapshot
        expected = store.sources_dir / _sha1("https://example.com/x") / ("b" * 64) / "content.md"
        assert paths[0] == expected

    def test_pages(self, store: WikiStore):
        page = store.save_page("context-compression", "上下文压缩", "第一版")
        assert page.created == page.updated
        assert store.get_page("context-compression").body == "第一版"
        updated = store.save_page("context-compression", "上下文压缩", "第二版")
        assert updated.created == page.created  # 更新保留 created
        assert updated.updated >= page.updated
        assert store.get_page("context-compression").body == "第二版"
        assert [p.id for p in store.list_pages()] == ["context-compression"]
        assert store.get_page("missing") is None

    def test_conflicts(self, store: WikiStore):
        c1 = store.save_conflict(
            "GLM-5.3 的上下文窗口是多大？",
            {"note_id": "N-0001", "excerpt": "128k"},
            {"note_id": "N-0002", "excerpt": "1M"},
        )
        c2 = store.save_conflict("第二个问题", {"note_id": "N-0003"}, {"note_id": "N-0004"})
        assert (c1.id, c2.id) == ("C-0001", "C-0002")
        assert [c.id for c in store.list_conflicts()] == ["C-0001", "C-0002"]
        resolved = store.resolve_conflict(c1.id, verdict="以官方文档为准", resolved_with="N-0002")
        assert resolved.status == "resolved"
        assert [c.id for c in store.list_conflicts()] == ["C-0002"]  # 默认只回 open
        assert [c.id for c in store.list_conflicts(status=None)] == ["C-0001", "C-0002"]
        loaded = store.get_conflict(c1.id)
        assert loaded.resolution == {"verdict": "以官方文档为准", "resolved_with": "N-0002"}


# ---- embeddings -------------------------------------------------------


class TestEmbeddings:
    def test_mock_deterministic_and_normalized(self):
        emb = MockEmbeddingProvider(dim=64)
        v1 = emb.embed(["上下文压缩", "向量检索"])
        v2 = emb.embed(["上下文压缩", "向量检索"])
        assert v1 == v2  # 跨调用可复现（hashlib，无进程加盐）
        for vec in v1:
            assert len(vec) == 64
            assert abs(sum(x * x for x in vec) - 1.0) < 1e-9  # L2 归一化

    def test_mock_similarity(self):
        emb = MockEmbeddingProvider(dim=128)
        a, a2, b = emb.embed(["上下文压缩技术", "上下文压缩显著降低成本", "数据库索引原理"])
        sim_close = cosine(a, a2)
        sim_far = cosine(a, b)
        assert sim_close > 0
        assert sim_close > sim_far  # 相似串相似度 > 不相似串

    def test_openai_compatible_request(self):
        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["auth"] = request.headers.get("Authorization")
            seen["json"] = json.loads(request.content)
            data = [
                {"index": 1, "embedding": [0.4, 0.5, 0.6]},
                {"index": 0, "embedding": [0.1, 0.2, 0.3]},
            ]
            return httpx.Response(200, json={"data": data})

        provider = OpenAICompatibleEmbedding(
            base_url="https://api.example.com/v1",
            model="embedding-3",
            api_key="sk-test",
            transport=httpx.MockTransport(handler),
        )
        vectors = provider.embed(["第一条", "第二条"])
        assert seen["url"] == "https://api.example.com/v1/embeddings"
        assert seen["auth"] == "Bearer sk-test"
        assert seen["json"] == {"model": "embedding-3", "input": ["第一条", "第二条"]}
        assert vectors == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]  # 按 index 复原顺序

    def test_openai_compatible_reads_env_key(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("MY_KEY", "sk-env")

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})

        provider = OpenAICompatibleEmbedding(
            base_url="https://api.example.com",
            model="m",
            api_key_env="MY_KEY",
            transport=httpx.MockTransport(handler),
        )
        assert provider.embed(["x"]) == [[1.0]]

    def test_openai_compatible_error(self):
        provider = OpenAICompatibleEmbedding(
            base_url="https://api.example.com",
            model="m",
            transport=httpx.MockTransport(lambda request: httpx.Response(401, text="unauthorized")),
        )
        with pytest.raises(EmbeddingError, match="401"):
            provider.embed(["x"])

    def test_cache_hit_skips_transport(self, tmp_path: Path):
        calls: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            inputs = json.loads(request.content)["input"]
            data = [{"index": i, "embedding": [0.25] * 4} for i in range(len(inputs))]
            return httpx.Response(200, json={"data": data})

        inner = OpenAICompatibleEmbedding(
            base_url="https://api.example.com",
            model="embedding-3",
            api_key="k",
            transport=httpx.MockTransport(handler),
        )
        cached = CachedEmbeddingProvider(inner, tmp_path / "cache.db")
        first = cached.embed(["你好世界", "第二条"])
        second = cached.embed(["你好世界", "第二条"])
        assert len(calls) == 1  # 第二次全部命中缓存，不再触发 transport
        assert first == second
        fresh = cached.embed(["你好世界", "新文本"])
        assert len(calls) == 2 and fresh[:1] == first[:1]
        cached.close()

    def test_store_commits_second_connection_can_write(self, tmp_path: Path):
        """缓存未命中→写缓存后，另一连接写同库不锁死（模拟 SearchIndex rebuild）。

        缓存与检索索引共享同一 index.db（设计如此）：修复前 _store 只 INSERT
        不 commit，未提交写事务持有 RESERVED 锁直到连接关闭（且关闭时回滚），
        第二连接的写操作忙等超时即抛 "database is locked"。本测试以 200ms
        短忙等钉住该失败模式：_store 后必须立即可被另一连接写入。
        """
        db = tmp_path / "shared.db"
        cached = CachedEmbeddingProvider(MockEmbeddingProvider(dim=8), db)
        vec = cached.embed(["某条需要缓存的文本"])  # 未命中 → _store 写入
        assert len(vec[0]) == 8
        other = sqlite3.connect(db, timeout=0.2)
        try:
            other.execute("CREATE TABLE IF NOT EXISTS probe (id INTEGER PRIMARY KEY)")
            other.execute("INSERT INTO probe (id) VALUES (1)")
            other.commit()
        finally:
            other.close()
        cached.close()

    def test_cache_persists_across_instances(self, tmp_path: Path):
        """缓存写入须真正持久化：第二次实例化同一 cache_path 全命中，底层不再被调。"""
        calls: list[str] = []

        class CountingMock(MockEmbeddingProvider):
            def embed(self, texts: list[str]) -> list[list[float]]:
                calls.extend(texts)
                return super().embed(texts)

        db = tmp_path / "cache.db"
        first = CachedEmbeddingProvider(CountingMock(dim=8), db)
        v1 = first.embed(["持久化检查文本"])
        first.close()
        second = CachedEmbeddingProvider(CountingMock(dim=8), db)
        v2 = second.embed(["持久化检查文本"])
        second.close()
        assert calls == ["持久化检查文本"]  # 第二实例全命中，未再调底层 embedding
        # 缓存以 float32（struct 'f'）序列化，回读有 ~1e-7 级精度损失，用 approx 比
        assert v2[0] == pytest.approx(v1[0])

    def test_store_visible_to_fresh_connection(self, tmp_path: Path):
        """_store 后数据必须已提交：关闭 provider 后用全新连接能读到缓存行。"""
        db = tmp_path / "cache.db"
        cached = CachedEmbeddingProvider(MockEmbeddingProvider(dim=4), db)
        cached.embed(["提交语义钉"])
        cached.close()
        conn = sqlite3.connect(db)
        try:
            row = conn.execute("SELECT COUNT(*) FROM embedding_cache").fetchone()
        finally:
            conn.close()
        assert row is not None and row[0] == 1

    def test_index_rebuild_with_shared_cache_db_no_deadlock(self, store: WikiStore):
        """SearchIndex 与嵌入缓存连接共享同一 index.db：rebuild（缓存全未命中）不锁死。

        复现真模型 warm run 的死锁：若 index_note 在本连接写事务中途调 embed，
        缓存连接的 INSERT 要等 SearchIndex 的 RESERVED 锁、SearchIndex 在等
        embed 返回，busy_timeout 只能以 locked 收场。embed 已移到事务前；
        以 200ms 短忙等（任一侧竞争即快速失败）钉住 rebuild 必须正常完成。
        """
        store.save_note("上下文压缩技术显著降低长会话成本", title="上下文压缩")
        cached = CachedEmbeddingProvider(MockEmbeddingProvider(dim=8), store.root / "index.db")
        index = SearchIndex(store.root, embedding=cached, tokenizer="trigram")
        try:
            assert index.rebuild(store) == 1
            assert index.search("上下文压缩")
        finally:
            index.close()
            cached.close()

    def test_factory_falls_back_to_mock(self):
        assert isinstance(get_embedding_provider({}), MockEmbeddingProvider)
        empty = {"embedding": {"model": "", "base_url": ""}}
        assert isinstance(get_embedding_provider(empty), MockEmbeddingProvider)
        # 有 model 没 base_url 同样回退
        no_base = {"embedding": {"model": "embedding-3", "base_url": ""}}
        assert isinstance(get_embedding_provider(no_base, cache_path=None), MockEmbeddingProvider)

    def test_factory_with_config(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        config = {
            "embedding": {
                "model": "embedding-3",
                "base_url": "https://open.bigmodel.cn/api/paas/v4",
                "api_key_env": "RESEARCHWIKI_TEST_KEY",
            }
        }
        # 未设 key：工厂不炸，调用时才由 OpenAICompatibleEmbedding 读环境变量
        provider = get_embedding_provider(config, cache_path=tmp_path / "c.db")
        assert isinstance(provider, CachedEmbeddingProvider)
        assert provider.inner.model == "embedding-3"
        uncached_cfg = {"embedding": dict(config["embedding"])}
        uncached = get_embedding_provider(uncached_cfg, cache_path=None)
        assert isinstance(uncached, OpenAICompatibleEmbedding)


# ---- index ------------------------------------------------------------


class FixedEmbedding:
    """测试替身：查表返回预置向量，让向量通道的排名完全可控。"""

    model = "fixed"
    dim = 4

    def __init__(self, table: dict[str, list[float]]) -> None:
        self.table = table

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(self.table[text]) for text in texts]


class TestIndex:
    @pytest.mark.parametrize("tok", ["trigram", "simple"])
    def test_fts_chinese_recall(self, store: WikiStore, tok: str):
        """两种 tokenizer 都必须过同一组中文 recall 断言。"""
        n1 = store.save_note("上下文压缩技术显著降低长会话的 token 成本", title="上下文压缩")
        store.save_note("向量数据库方案支持多种检索模式", title="向量库")
        with SearchIndex(store.root, tokenizer=tok) as idx:
            assert idx.rebuild(store) == 2
            hits = idx.search("上下文压缩", k=3)
            assert hits, f"tokenizer={tok} 无命中"
            assert hits[0].note_id == n1.id

    def test_fts_tokenizer_auto(self, store: WikiStore):
        n1 = store.save_note("上下文压缩技术显著降低长会话的 token 成本", title="上下文压缩")
        with SearchIndex(store.root) as idx:  # tokenizer 缺省 auto
            assert idx.tokenizer in ("simple", "trigram")
            idx.rebuild(store)
            assert idx.search("上下文压缩")[0].note_id == n1.id

    def test_trigram_short_query_like_fallback(self, store: WikiStore):
        """trigram 无法索引 <3 字符查询，LIKE 兜底必须补上召回。"""
        n1 = store.save_note("向量数据库方案支持多种检索模式", title="向量库")
        with SearchIndex(store.root, tokenizer="trigram") as idx:
            idx.rebuild(store)
            hits = idx.search("向量", k=5)
            assert [h.note_id for h in hits] == [n1.id]

    def test_vector_channel_recall(self, store: WikiStore):
        """查询"向量检索"命中"向量数据库方案"笔记：FTS 不命中（无 3 字公共子串），
        靠向量通道（bigram 特征哈希共享"向量"）经 RRF 融合召回。"""
        store.save_note("上下文压缩技术显著降低长会话成本", title="上下文压缩")
        target = store.save_note("向量数据库方案支持多种检索模式", title="向量库")
        with SearchIndex(store.root, tokenizer="trigram") as idx:
            idx.rebuild(store)
            hits = idx.search("向量检索", k=5)
            match = next(m for m in hits if m.note_id == target.id)
            assert match.match_type == "vector"

    def test_rrf_fuse(self):
        scores = rrf_fuse([["a", "b"], ["b", "c"]])
        assert sorted(scores, key=scores.get, reverse=True) == ["b", "a", "c"]
        assert scores["b"] == pytest.approx(1 / 61 + 1 / 62)
        assert scores["a"] == pytest.approx(1 / 61)

    def test_confidence_factor(self, store: WikiStore):
        """其余条件相同（同正文），high 必须排在 low/medium 之前。"""
        store.save_note("上下文压缩技术", title="高置信", confidence="high")
        store.save_note("上下文压缩技术", title="低置信", confidence="low")
        store.save_note("上下文压缩技术", title="中置信", confidence="medium")
        with SearchIndex(store.root, tokenizer="trigram") as idx:
            idx.rebuild(store)
            hits = idx.search("上下文压缩", k=5)
            by_title = {h.title: h.score for h in hits}
            assert by_title["高置信"] > by_title["中置信"] > by_title["低置信"]

    def test_freshness_factor_pure(self):
        hl = {"volatile": 30.0, "drifting": 90.0}
        f = freshness_factor
        stale_created = "2020-01-01T00:00:00+00:00"
        assert f("stable", None, stale_created, half_life_days=hl, now=FIXED_NOW) == 1.0
        assert f(
            "volatile", "2026-08-21T12:00:00+00:00", "", half_life_days=hl, now=FIXED_NOW
        ) == pytest.approx(0.5)
        assert f(
            "drifting", "2026-08-21T12:00:00+00:00", "", half_life_days=hl, now=FIXED_NOW
        ) == pytest.approx(0.5 ** (30 / 90))
        # observed_at 缺省回退 created；时间非法/缺失 → 不衰减
        assert f(
            "volatile", None, "2026-08-21T12:00:00+00:00", half_life_days=hl, now=FIXED_NOW
        ) == pytest.approx(0.5)
        assert f("volatile", None, "", half_life_days=hl, now=FIXED_NOW) == 1.0

    def test_freshness_in_search_with_injected_clock(self, store: WikiStore):
        """volatile 笔记 60 天后（30 天半衰期）因子 0.25，同正文必排到 fresh 之后。"""
        old = FIXED_NOW - timedelta(days=60)
        store.save_note("上下文压缩技术", title="新鲜", created=FIXED_NOW.isoformat())
        store.save_note(
            "上下文压缩技术",
            title="陈旧",
            created=old.isoformat(),
            observed_at=old.isoformat(),
            volatility="volatile",
        )
        with SearchIndex(store.root, tokenizer="trigram", clock=lambda: FIXED_NOW) as idx:
            idx.rebuild(store)
            hits = idx.search("上下文压缩", k=5)
            assert hits[0].title == "新鲜"

    def test_redirect_annotation_in_search(self, store: WikiStore):
        """命中 merged 笔记（目标自身不匹配查询）→ 结果为重定向目标并标注 redirected_from。"""
        target = store.save_note("实现细节调整", title="最终")
        store.save_note(
            "上下文压缩技术降低成本", title="别名页", status="merged", redirect_to=target.id
        )
        with SearchIndex(store.root, tokenizer="trigram") as idx:
            idx.rebuild(store)
            hits = idx.search("上下文压缩", k=5)
            assert [h.note_id for h in hits] == [target.id]
            assert hits[0].redirected_from == "N-0002"

    def test_merged_note_not_returned_directly(self, store: WikiStore):
        """目标被直接命中时不标注重定向；别名页自身从不出现在结果里。"""
        store.save_note("上下文压缩技术", title="最终")
        store.save_note(
            "上下文压缩技术", title="合并", status="merged", redirect_to="N-0001"
        )
        with SearchIndex(store.root, tokenizer="trigram") as idx:
            idx.rebuild(store)
            hits = idx.search("上下文压缩", k=5)
            assert [h.note_id for h in hits] == ["N-0001"]
            assert hits[0].redirected_from is None

    def test_incremental_index_and_rebuild(self, store: WikiStore):
        with SearchIndex(store.root, tokenizer="trigram") as idx:
            assert idx.search("上下文压缩") == []  # 空索引
            n1 = store.save_note("上下文压缩技术")
            idx.index_note(n1)
            assert idx.search("上下文压缩")[0].note_id == n1.id
            # 原地更新：改正文后旧关键词不再命中、新关键词命中
            updated = store.save_note("向量数据库方案", note_id=n1.id)
            idx.index_note(updated)
            assert idx.search("上下文压缩") == []
            assert idx.search("向量数据库方案")[0].note_id == n1.id
            # 全量重建：清空后重新索引 store 里的全部笔记
            assert idx.rebuild(store) == 1
            assert idx.search("向量数据库方案")[0].note_id == n1.id

    def test_brute_force_fallback_same_semantics(self, store: WikiStore):
        """禁用 vec0 后走纯 Python 余弦暴力扫描，结果语义与 KNN 路径一致。"""
        store.save_note("上下文压缩技术", title="a")
        store.save_note("向量数据库方案", title="b")
        with SearchIndex(store.root, tokenizer="trigram") as idx:
            idx.rebuild(store)
            knn_hits = idx.search("向量检索", k=5)
            idx._vec_ready = False  # 模拟 sqlite-vec 不可用
            brute_hits = idx.search("向量检索", k=5)
            assert [h.note_id for h in brute_hits] == [h.note_id for h in knn_hits]
            assert brute_hits[0].match_type == knn_hits[0].match_type

    def test_dim_change_recreates_vec_table(self, store: WikiStore):
        store.save_note("上下文压缩技术", title="a")
        with SearchIndex(store.root, tokenizer="trigram") as idx:
            idx.rebuild(store)
        # 同库换 embedding 维度：vec0 表自动重建，可继续检索
        dim16 = MockEmbeddingProvider(dim=16)
        with SearchIndex(store.root, tokenizer="trigram", embedding=dim16) as idx2:
            idx2.rebuild(store)
            assert idx2.search("上下文压缩")[0].title == "a"

    def test_wiki_search_convenience(self, store: WikiStore):
        n1 = store.save_note("上下文压缩技术", title="a")
        with SearchIndex(store.root, tokenizer="trigram") as idx:
            idx.rebuild(store)
        hits = wiki_search("上下文压缩", store=store, tokenizer="trigram")
        assert isinstance(hits[0], SearchMatch)
        assert hits[0].note_id == n1.id
        assert hits[0].snippet  # 摘要非空
        assert wiki_search("  ", store=store) == []  # 空查询

    def test_old_index_db_without_kind_column_migrates(self, store: WikiStore):
        """旧 schema（无 kind 列）的 index.db 打开时自动升级，检索与写入不受影响。

        手工搭建 v1 schema（P1-A 之前：note_meta 无 kind 列）+ 一条已索引笔记，
        SearchIndex 打开该库后必须：kind 列出现、存量行回填 knowledge、
        FTS 检索照常可用——绝不能让老用户索引报废或要求手工重建。
        """
        db_path = store.root / "index.db"
        store.root.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path)
        try:
            conn.execute(
                "CREATE TABLE note_meta ("
                "note_id TEXT PRIMARY KEY, title TEXT, body TEXT, confidence TEXT, "
                "volatility TEXT, status TEXT, redirect_to TEXT, superseded_by TEXT, "
                "observed_at TEXT, created TEXT)"
            )
            conn.execute(
                "CREATE VIRTUAL TABLE note_fts USING fts5("
                "note_id UNINDEXED, title, body, tokenize='trigram')"
            )
            conn.execute("CREATE TABLE index_meta (key TEXT PRIMARY KEY, value TEXT)")
            conn.execute(
                "CREATE TABLE note_vec_cache "
                "(note_id TEXT PRIMARY KEY, dim INTEGER, vec BLOB)"
            )
            conn.execute("INSERT INTO index_meta VALUES ('tokenizer', 'trigram')")
            conn.execute(
                "INSERT INTO note_meta (note_id, title, body, confidence, volatility, "
                "status, created) VALUES ('N-0001', '上下文压缩', '旧库里的上下文压缩笔记', "
                "'high', 'stable', 'active', '2026-05-01T00:00:00+00:00')"
            )
            conn.execute(
                "INSERT INTO note_fts (note_id, title, body) "
                "VALUES ('N-0001', '上下文压缩', '旧库里的上下文压缩笔记')"
            )
            conn.commit()
        finally:
            conn.close()

        with SearchIndex(store.root, tokenizer="trigram") as idx:
            # 列已补上，存量行回填默认类型
            columns = {row[1] for row in idx._conn.execute("PRAGMA table_info(note_meta)")}
            assert "kind" in columns
            row = idx._conn.execute("SELECT kind FROM note_meta WHERE note_id='N-0001'").fetchone()
            assert row is not None and row[0] == "knowledge"
            # 旧索引内容原样可用
            hits = idx.search("上下文压缩", k=5)
            assert [h.note_id for h in hits] == ["N-0001"]
            assert idx.search("上下文压缩", kind="knowledge")[0].note_id == "N-0001"
            assert idx.search("上下文压缩", kind="user") == []
            # 迁移后的库还能继续增量写入（带 kind）
            note = store.save_note(
                "向量数据库方案支持多种检索模式", title="向量库", kind="experience"
            )
            idx.index_note(note)
            assert idx.search("向量数据库", kind="experience")[0].note_id == note.id

    def test_kind_filter_hit_and_exclude(self, store: WikiStore):
        """kind 过滤：结果里只出现目标类型的笔记；None 时行为与不过滤完全一致。"""
        user_note = store.save_note("用户偏好深色主题的界面", title="用户偏好", kind="user")
        store.save_note("上下文压缩技术降低成本", title="知识", kind="knowledge")
        with SearchIndex(store.root, tokenizer="trigram") as idx:
            idx.rebuild(store)
            hits = idx.search("用户偏好", kind="user")
            assert [h.note_id for h in hits] == [user_note.id]
            # 过滤后绝不允许混入其他类型（mock 向量通道的弱相关命中也要被挡掉）
            for other_kind in ("knowledge", "experience"):
                assert user_note.id not in {
                    h.note_id for h in idx.search("用户偏好", kind=other_kind)
                }
            # 不过滤时两种都能召回（查询放宽到共享词）
            both = idx.search("偏好", k=5, kind=None)
            assert user_note.id in {h.note_id for h in both}

    def test_kind_filter_follows_redirect_target(self, store: WikiStore):
        """kind 过滤看的是重定向落点（最终 active 笔记）的类型，不是别名笔记。"""
        store.save_note("量子退火旧记录里提到用户偏好", title="用户偏好旧版",
                        kind="user", status="superseded", superseded_by="N-0002")
        store.save_note("用户偏好深色主题的界面", title="用户偏好新版", kind="knowledge")
        with SearchIndex(store.root, tokenizer="trigram") as idx:
            idx.rebuild(store)
            # 别名（user）被唯一词命中 → 重定向落点 N-0002（knowledge）
            hits = idx.search("量子退火", kind="knowledge")
            assert [h.note_id for h in hits] == ["N-0002"]
            assert hits[0].redirected_from == "N-0001"
            # 落点不是 user → kind=user 的结果里不允许出现
            assert "N-0002" not in {h.note_id for h in idx.search("量子退火", kind="user")}

    def test_wiki_settings_and_invalid_tokenizer(self, store: WikiStore):
        config = {
            "wiki": {"fts_tokenizer": "trigram", "half_life_days": {"volatile": 10, "drifting": 20}}
        }
        settings = wiki_settings(config)
        assert settings.fts_tokenizer == "trigram"
        assert settings.half_life_days == {"volatile": 10.0, "drifting": 20.0}
        with pytest.raises(ValueError, match="fts_tokenizer"):
            SearchIndex(store.root, tokenizer="bogus")


# ---- config.toml ------------------------------------------------------


class TestConfig:
    def test_config_sections(self):
        with open(PROJECT_ROOT / "config.toml", "rb") as f:
            config = tomllib.load(f)
        embedding = config["embedding"]
        # 仓库配置会填真实模型与 base_url（clone 后填入自己的 key 即可用），
        # 所以断言结构完整而非"值为空"；真正的安全属性是 [server].mode 默认 mock——
        # 不动配置直接起服务，不会有任何带 key 的外呼。
        assert {"model", "base_url", "api_key_env"} <= set(embedding)
        assert embedding["api_key_env"] == "RESEARCHWIKI_EMBEDDING_API_KEY"
        assert config["wiki"]["half_life_days"] == {"volatile": 30, "drifting": 90}
        assert config["wiki"]["fts_tokenizer"] == "auto"
        # 既有段语义未被破坏
        assert config["server"]["mode"] == "mock"
        assert config["runtime"]["compaction_threshold"] == 0.7
        settings = wiki_settings(config)
        assert settings.half_life_days == {"volatile": 30.0, "drifting": 90.0}

    def test_embedding_blank_means_mock(self):
        """base_url 或 model 留空 = MockEmbeddingProvider（无 key 也能跑）。"""
        provider = get_embedding_provider(
            {"embedding": {"model": "", "base_url": "", "api_key_env": "X"}}
        )
        assert isinstance(provider, MockEmbeddingProvider)
