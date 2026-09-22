"""索引指纹与 ensure_index_fresh 三维比对（P2-B：carried 一致性修复）。

覆盖 task-9 简报需求 1 与需求 2：

1. ``note_meta.body_hash`` 列 + 旧库迁移（探测缺列 → ALTER，存量行留空）+ 只读
   方法 ``indexed_fingerprints()``（``indexed_status()`` 契约不变）；
2. ``ensure_index_fresh`` 的三维落后判定：id 集合 / status / **索引指纹**
   （覆盖写入索引且参与检索/过滤的全部字段），以及回归用例：
   a. ``store.save_note`` 同 id 原地改写正文 → 判落后并 rebuild，之后检索命中
      新正文、不再命中旧正文；
   b. ``memory_update`` 走完整 MCP 服务面改正文 → 索引一致性可检出（service 层
      直接调用，不起 server、不走 stdio）；
   c. status 变化与新增笔记两个旧维度不回归（另见 test_prior.py 的同名用例）；
   d. **只改元数据**（kind / volatility / observed_at / title / redirect_to …，
      正文与 status 都不变）同样检出——覆盖 ``_annotate_formation`` 回写 kind
      这条已被代码预判的路径。

P2-F 追加两组：``tombstone`` 列（裁定一：墓碑标记进指纹、旧库 ALTER 回填 0），
以及 ``index_drift`` 共享判据（裁定二：MCP 与 loop 两条路径对同一状态给出一致
结论——含"md 重写但指纹没变"时**两边都不重建**的"白做功"回归）。

零网络：MockEmbeddingProvider（dim=128，与 service 层缺省一致）+ tokenizer=trigram。

检索断言的写法说明：Mock 向量通道对任何查询都可能返回命中（余弦只要 > 1e-6 即算
命中，见 index.VECTOR_MIN_COSINE），所以"不命中旧正文"的断言落在 **FTS 通道**
（match_type 为 fts/both）——那才是真正存文本的通道；helper ``_fts_hits`` 就是干
这个的，避免断言被向量噪声污染。同理，"命中新正文"断言到 snippet 文本上。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from researchwiki.mcp_server.service import WikiService
from researchwiki.wiki.embeddings import MockEmbeddingProvider
from researchwiki.wiki.index import SearchIndex, SearchMatch, index_drift, note_index_hash
from researchwiki.wiki.prior import ensure_index_fresh
from researchwiki.wiki.store import Note, WikiStore

CONFIG = {"wiki": {"fts_tokenizer": "trigram"}}
# 旧/新两版正文：首行是中性标题（memory_update 沿用旧标题，若标题里带旧短语会
# 让 FTS 断言被标题命中污染），查询短语只出现在正文里
TITLE = "观测记录"
OLD_BODY = "观测记录\n旧版本：量子退火炉的初代读数甲。"
NEW_BODY = "观测记录\n新版本：闪电风暴仪的次代读数乙。"
OLD_PHRASE = "量子退火炉"
NEW_PHRASE = "闪电风暴仪"


@pytest.fixture()
def store(tmp_path: Path) -> WikiStore:
    return WikiStore(tmp_path / "wiki-data")


def make_index(store: WikiStore) -> SearchIndex:
    """与 service 层缺省同参数的索引（Mock 嵌入 dim=128 + trigram）。"""
    return SearchIndex(
        store.root,
        embedding=MockEmbeddingProvider(dim=128),
        tokenizer="trigram",
    )


def _fts_hits(index: SearchIndex, query: str, kind: str | None = None) -> list[SearchMatch]:
    """只保留真正来自索引正文的命中（fts / both），滤掉向量通道噪声。

    ``kind`` 透传给 ``search`` 的过滤参数（kind 过滤发生在融合打分之后，
    与通道无关；这里同时带上通道过滤，断言才不会被向量命中污染）。
    """
    return [
        m for m in index.search(query, k=5, kind=kind) if m.match_type in ("fts", "both")
    ]


# ---- 需求 1：指纹列 / 只读方法 / 旧库迁移 -------------------------------------


class TestBodyHashColumn:
    def test_indexed_fingerprints_matches_indexed_content(self, store: WikiStore) -> None:
        note = store.save_note(OLD_BODY, note_id="N-0001", title=TITLE)
        with make_index(store) as index:
            index.rebuild(store)
            fingerprints = index.indexed_fingerprints()
            assert fingerprints == {
                "N-0001": ("active", note_index_hash(note))
            }
            # 契约不动：indexed_status 仍是 {note_id: status}（P1-A 调用方依赖）
            assert index.indexed_status() == {"N-0001": "active"}

    def test_fingerprint_covers_title_body_and_metadata(self, store: WikiStore) -> None:
        """指纹覆盖"写入索引且参与检索/过滤的全部字段"，逐字段敏感。"""
        base = store.save_note(OLD_BODY, note_id="N-0001", title=TITLE)
        assert note_index_hash(base) == note_index_hash(store.get_note("N-0001"))
        # title / body
        for kwargs in (
            {"title": "另一个标题"},
            {"body": "换一段正文"},
            {"confidence": "high"},
            {"volatility": "volatile"},
            {"kind": "user"},
            {"observed_at": "2026-05-01T00:00:00+00:00"},
            {"created": "2026-05-01T00:00:00+00:00"},
        ):
            changed = store.save_note(
                kwargs.get("body", OLD_BODY),
                note_id="N-0001",
                title=str(kwargs.get("title", TITLE)),
                confidence=str(kwargs.get("confidence", "medium")),
                volatility=str(kwargs.get("volatility", "stable")),
                kind=str(kwargs.get("kind", "knowledge")),
                observed_at=kwargs.get("observed_at"),
                created=str(kwargs.get("created", base.meta.created)),
                sources=list(base.meta.sources),
            )
            assert note_index_hash(changed) != note_index_hash(base), kwargs
        # redirect_to / superseded_by（merged 笔记的跳转目标变了 = 命中落点变了）
        merged_a = store.save_note(
            "留痕", note_id="N-0002", status="merged", redirect_to="N-0001", title="留痕甲"
        )
        merged_b = store.save_note(
            "留痕", note_id="N-0002", status="merged", redirect_to="N-0003", title="留痕甲"
        )
        assert note_index_hash(merged_a) != note_index_hash(merged_b)

    def test_title_only_change_is_detected(self, store: WikiStore) -> None:
        store.save_note(OLD_BODY, note_id="N-0001", title="旧标题")
        with make_index(store) as index:
            index.rebuild(store)
            # 只改标题（正文一字不动）：检索可见的 title 变了，索引同样失真
            store.save_note(OLD_BODY, note_id="N-0001", title="新标题")
            rebuilt, reason = ensure_index_fresh(store, index)
            assert rebuilt is True
            assert "索引字段变化" in reason and "N-0001" in reason

    def test_kind_only_change_is_detected(self, store: WikiStore) -> None:
        """只改 kind（body/title/status 不变）→ 判落后并 rebuild。

        这条路径已被代码预判：``loop/agent_loop._annotate_formation`` 原地回写
        ``kind=decision.kind``，其 docstring 写明"若 formation 未来赋非 knowledge
        kind，此处必须触发索引同步，否则检索的 kind 过滤会失真"。指纹不含元数据
        时这条路径检不出，索引会静默按旧 kind 过滤。
        """
        store.save_note(OLD_BODY, note_id="N-0001", title=TITLE, kind="knowledge")
        with make_index(store) as index:
            index.rebuild(store)
            assert index.indexed_fingerprints()["N-0001"][0] == "active"
            # 模拟 _annotate_formation 的原地回写：正文/标题/status 全不变，只换 kind
            store.save_note(OLD_BODY, note_id="N-0001", title=TITLE, kind="user")
            # 陈旧态证据：索引里仍是 kind=knowledge，按 user 过滤一条也召不回
            assert _fts_hits(index, OLD_PHRASE, kind="user") == []
            assert [m.note_id for m in _fts_hits(index, OLD_PHRASE, kind="knowledge")] == ["N-0001"]
            rebuilt, reason = ensure_index_fresh(store, index)
            assert rebuilt is True
            assert "索引字段变化" in reason and "N-0001" in reason
            # rebuild 之后 kind 过滤才与 store 一致
            assert [m.note_id for m in _fts_hits(index, OLD_PHRASE, kind="user")] == ["N-0001"]
            assert _fts_hits(index, OLD_PHRASE, kind="knowledge") == []

    def test_volatility_only_change_is_detected(self, store: WikiStore) -> None:
        """只改 volatility/observed_at（参与检索打分的字段）→ 同样判落后。"""
        store.save_note(
            OLD_BODY, note_id="N-0001", title=TITLE, volatility="stable",
            observed_at="2026-01-01T00:00:00+00:00",
        )
        with make_index(store) as index:
            index.rebuild(store)
            store.save_note(
                OLD_BODY, note_id="N-0001", title=TITLE, volatility="volatile",
                observed_at="2026-01-01T00:00:00+00:00",
            )
            rebuilt, reason = ensure_index_fresh(store, index)
            assert rebuilt is True
            assert "索引字段变化" in reason and "N-0001" in reason

    def test_legacy_index_without_body_hash_is_migrated_and_rebuilt(
        self, store: WikiStore
    ) -> None:
        """旧库（无 body_hash 列）打开即自动补列；首次检查判落后并 rebuild。"""
        note = store.save_note(OLD_BODY, note_id="N-0001", title=TITLE)
        with make_index(store) as index:
            index.rebuild(store)
        # 模拟 P2 之前的旧库：删掉列（旧 schema 里没有它）
        conn = sqlite3.connect(store.root / "index.db")
        conn.execute("ALTER TABLE note_meta DROP COLUMN body_hash")
        conn.commit()
        conn.close()
        with make_index(store) as index:
            columns = {row[1] for row in index._conn.execute("PRAGMA table_info(note_meta)")}
            assert "body_hash" in columns  # 探测缺列 → ALTER 已补上
            # 存量行不回填：归一成空串（"不知道是哪版检索视图"）
            assert index.indexed_fingerprints()["N-0001"] == ("active", "")
            assert index.indexed_status() == {"N-0001": "active"}
            rebuilt, reason = ensure_index_fresh(store, index)
            assert rebuilt is True
            assert "旧索引缺指纹" in reason
            # rebuild 之后指纹回填，再检查即新鲜
            assert index.indexed_fingerprints()["N-0001"][1] == note_index_hash(note)
            again, reason2 = ensure_index_fresh(store, index)
            assert again is False and "无需 rebuild" in reason2

    def test_indexed_fingerprints_is_read_only(self, store: WikiStore) -> None:
        """只读方法不改库：两次读取结果一致，也不引入旧版本的命中。"""
        store.save_note(OLD_BODY, note_id="N-0001", title=TITLE)
        with make_index(store) as index:
            index.rebuild(store)
            before = index.indexed_fingerprints()
            assert index.indexed_fingerprints() == before
            assert _fts_hits(index, NEW_PHRASE) == []


# ---- 需求 2：同 id 原地改写（正文 / 元数据）的检出与 rebuild ------------------


class TestBodyRewriteDetection:
    def test_same_id_body_rewrite_detected_and_rebuilt(self, store: WikiStore) -> None:
        """a. 同 id 换正文：判落后 → rebuild → 检索命中新正文、不再命中旧正文。"""
        store.save_note(OLD_BODY, note_id="N-0001", title=TITLE)
        with make_index(store) as index:
            index.rebuild(store)
            assert [m.note_id for m in _fts_hits(index, OLD_PHRASE)] == ["N-0001"]
            # 同 id 原地改写正文（status/集合都没变）：旧实现只比 status → 判"新鲜"，
            # 索引静默保留陈旧正文
            store.save_note(NEW_BODY, note_id="N-0001", title=TITLE)
            assert [m.note_id for m in _fts_hits(index, NEW_PHRASE)] == []  # 陈旧态证据
            rebuilt, reason = ensure_index_fresh(store, index)
            assert rebuilt is True
            assert "索引字段变化" in reason and "N-0001" in reason
            # rebuild 之后：新正文可召回（snippet 就是新文本），旧正文不再可召回
            new_hits = _fts_hits(index, NEW_PHRASE)
            assert [m.note_id for m in new_hits] == ["N-0001"]
            assert NEW_PHRASE in new_hits[0].snippet
            assert _fts_hits(index, OLD_PHRASE) == []
            # 再检查一次：已新鲜，不再 rebuild
            again, reason2 = ensure_index_fresh(store, index)
            assert again is False and "无需 rebuild" in reason2

    def test_memory_update_through_service_is_detectable(self, tmp_path: Path) -> None:
        """b. memory_update 走完整 MCP 服务面改正文：索引一致性可检出（不起 server）。"""
        root = tmp_path / "wiki-data"
        service = WikiService(root, config=CONFIG)
        written = service.store_memory(OLD_BODY, kind="knowledge")
        assert written["ok"] is True
        note_id = str(written["note_id"])
        store = WikiStore(root)
        before = store.get_note(note_id)
        assert before is not None
        with make_index(store) as index:
            index.rebuild(store)
            updated = service.update_memory(note_id, NEW_BODY, "来源内容更新，修订正文")
            assert updated["ok"] is True and updated["indexed"] is True
            # ① 服务面自身做了增量索引同步 → 正常路径下检查判定为新鲜
            rebuilt, reason = ensure_index_fresh(store, index)
            assert rebuilt is False and "无需 rebuild" in reason
            # ② 复现 carried 缺陷的陈旧态：索引里仍是改写前的正文（等价于 service 的
            #    索引同步失败，或外部进程改了 md 而索引没人同步）——只比 status 完全
            #    看不出来，指纹维度必须检出
            index.index_note(before)
            assert [m.note_id for m in _fts_hits(index, OLD_PHRASE)] == [note_id]
            rebuilt, reason = ensure_index_fresh(store, index)
            assert rebuilt is True
            assert "索引字段变化" in reason and note_id in reason
            assert [m.note_id for m in _fts_hits(index, NEW_PHRASE)] == [note_id]
            assert _fts_hits(index, OLD_PHRASE) == []

    def test_new_note_and_status_change_still_detected(self, store: WikiStore) -> None:
        """c. 两个旧维度不回归：新增笔记、status 变化（test_prior.py 有同源用例）。"""
        store.save_note("上下文压缩技术把长对话压缩成摘要。", note_id="N-0001", title="上下文压缩")
        with make_index(store) as index:
            index.rebuild(store)
            store.save_note(
                "RAG 检索增强生成把外部检索与生成模型结合。", note_id="N-0002", title="RAG"
            )
            rebuilt, reason = ensure_index_fresh(store, index)
            assert rebuilt is True
            assert "新增: N-0002" in reason
            assert ensure_index_fresh(store, index)[0] is False
            # 同 id 只改 status（正文不变）→ status 维度检出
            store.save_note(
                "RAG 检索增强生成把外部检索与生成模型结合。",
                note_id="N-0002",
                title="RAG",
                status="merged",
                redirect_to="N-0001",
            )
            rebuilt2, reason2 = ensure_index_fresh(store, index)
            assert rebuilt2 is True
            assert "status 变化" in reason2 and "N-0002" in reason2

    def test_extra_index_row_counts_as_stale(self, store: WikiStore) -> None:
        """索引里多出 store 没有的笔记（md 被外部删除）→ 仍判落后并 rebuild。"""
        note = store.save_note(OLD_BODY, note_id="N-0001", title=TITLE)
        with make_index(store) as index:
            index.rebuild(store)
            ghost = Note(id="N-9999", title="幽灵", body="md 已被删除的笔记", meta=note.meta)
            index.index_note(ghost)
            rebuilt, reason = ensure_index_fresh(store, index)
            assert rebuilt is True
            assert "索引多余: N-9999" in reason
            assert "N-9999" not in index.indexed_fingerprints()


# ---- 裁定一：墓碑标记进指纹 / 列迁移 -------------------------------------------


class TestTombstoneFlag:
    """墓碑标记（``tombstone``）参与索引指纹与检索过滤（P2-F 裁定一）。"""

    def test_fingerprint_covers_tombstone_flag(self, store: WikiStore) -> None:
        """只改 tombstone（正文/标题/元数据全不动）→ 指纹必须变。"""
        plain = store.save_note(OLD_BODY, note_id="N-0001", title=TITLE)
        marked = store.save_note(
            OLD_BODY, note_id="N-0001", title=TITLE, tombstone=True,
            created=plain.meta.created,
        )
        assert note_index_hash(plain) != note_index_hash(marked)
        # 反向：标回 False 又回到原指纹（标记是二值、可复原的）
        unmarked = store.save_note(
            OLD_BODY, note_id="N-0001", title=TITLE, tombstone=False,
            created=plain.meta.created,
        )
        assert note_index_hash(unmarked) == note_index_hash(plain)

    def test_tombstone_flag_change_is_detected_and_filters_retrieval(
        self, store: WikiStore
    ) -> None:
        """裁定一⑤：标记变化被 ensure_index_fresh 检出，rebuild 后检索排除墓碑。"""
        store.save_note(OLD_BODY, note_id="N-0001", title=TITLE)
        with make_index(store) as index:
            index.rebuild(store)
            # 陈旧态证据：索引里 tombstone=0，检索照常把它当当前知识返回
            assert [m.note_id for m in _fts_hits(index, OLD_PHRASE)] == ["N-0001"]
            # 原地标成墓碑（正文/标题/status 全不变）
            store.save_note(OLD_BODY, note_id="N-0001", title=TITLE, tombstone=True)
            rebuilt, reason = ensure_index_fresh(store, index)
            assert rebuilt is True
            assert "索引字段变化" in reason and "N-0001" in reason
            # rebuild 后：默认检索里没有它，显式 include_tombstones 才有
            assert _fts_hits(index, OLD_PHRASE) == []
            assert index.search(OLD_PHRASE) == []
            assert [m.note_id for m in index.search(OLD_PHRASE, include_tombstones=True)] == [
                "N-0001"
            ]

    def test_legacy_index_without_tombstone_column_is_migrated(self, store: WikiStore) -> None:
        """旧库（有 body_hash、无 tombstone 列）探测缺列即 ALTER，存量行回填 0。"""
        store.save_note(OLD_BODY, note_id="N-0001", title=TITLE)
        with make_index(store) as index:
            index.rebuild(store)
        # 模拟 P2-B~P2-F 之间建的库：列不存在（旧 schema）
        conn = sqlite3.connect(store.root / "index.db")
        conn.execute("ALTER TABLE note_meta DROP COLUMN tombstone")
        conn.commit()
        conn.close()
        with make_index(store) as index:
            columns = {row[1] for row in index._conn.execute("PRAGMA table_info(note_meta)")}
            assert "tombstone" in columns
            rows = index._conn.execute("SELECT note_id, tombstone FROM note_meta").fetchall()
            assert [(str(r[0]), int(r[1])) for r in rows] == [("N-0001", 0)]
            # 回填 0 是对的（旧数据里不存在墓碑）→ 指纹没变 → 判新鲜，不白做 rebuild
            fresh, reason = ensure_index_fresh(store, index)
            assert fresh is False and "无需 rebuild" in reason
            assert index.search(OLD_PHRASE, include_tombstones=True)


# ---- 裁定二：共享判据（MCP 与 loop 两条路径只允许一个答案）---------------------


class TestSharedDriftJudgement:
    """``index.index_drift`` 是两条路径共用的判据（P2-F 裁定二）。

    ``WikiService._index_is_stale`` 在 v1 是 mtime + 条数口径（返回 bool），现在
    委托给共享判据并返回落后原因（``str | None``）——本类直接对它断言，因为这正是
    契约变更的那一处；同时用公开行为（``service.search`` 是否 rebuild）兜一层，
    避免断言只锁在私有方法上。
    """

    @staticmethod
    def _service(root: Path) -> WikiService:
        return WikiService(root, config=CONFIG)

    def test_both_paths_agree_stale_when_md_changed_without_index_update(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "wiki-data"
        service = self._service(root)
        written = service.store_memory(OLD_BODY, kind="knowledge")
        note_id = str(written["note_id"])
        store = WikiStore(root)
        # 外部进程改 md（绕过 MCP，索引没跟上）
        store.save_note(NEW_BODY, note_id=note_id, title=TITLE)
        with make_index(store) as index:
            # 同一状态、同一索引实例：共享判据 / MCP 判据 / loop 判据三方同答案
            stale, reason = index_drift(store, index)
            assert stale is True
            assert "索引字段变化" in reason and note_id in reason
            assert service._index_is_stale(index) == reason  # 逐字同一原因（同一份判据）
            loop_stale, loop_reason = ensure_index_fresh(store, index)
            assert loop_stale is True
            assert loop_reason == f"{reason}；已执行 rebuild（索引 1 条笔记）"
        # 落点一致：loop 路径那次 rebuild 对 MCP 路径同样生效 → 检索命中新正文
        # 且不再需要重建（两条路径不会互相判对方的成果为"落后"）
        payload = service.search(NEW_PHRASE)
        assert "index_rebuilt" not in payload
        assert [r["note_id"] for r in payload["results"]] == [note_id]
        assert payload["results"][0]["snippet"] and NEW_PHRASE in payload["results"][0]["snippet"]

    def test_both_paths_agree_fresh_when_index_matches(self, tmp_path: Path) -> None:
        root = tmp_path / "wiki-data"
        service = self._service(root)
        service.store_memory(OLD_BODY, kind="knowledge")
        store = WikiStore(root)
        with make_index(store) as index:
            assert index_drift(store, index) == (
                False,
                "索引与 store 一致（1 条笔记，集合、status 与索引指纹均相同）",
            )
            assert service._index_is_stale(index) is None
            fresh, reason = ensure_index_fresh(store, index)
            assert fresh is False and "无需 rebuild" in reason
        payload = service.search(OLD_PHRASE)
        assert "index_rebuilt" not in payload  # 判新鲜 → 零重建
        assert payload["count"] == 1

    def test_frontmatter_only_rewrite_is_fresh_for_both_paths(self, tmp_path: Path) -> None:
        """回归"白做功"：``mark_source_changed`` 只写 md 不写索引 → 两边都判新鲜。

        该路径推后 ``source_changed_at``（不在指纹输入内），旧 mtime 判据必然判
        stale 并白做一次全量重建；共享判据只看集合/status/指纹，正确判新鲜。
        """
        root = tmp_path / "wiki-data"
        service = self._service(root)
        url = "https://example.com/quantum"
        written = service.store_memory(
            OLD_BODY, kind="knowledge", source_urls=[{"url": url, "content_hash": "sha1:old"}]
        )
        note_id = str(written["note_id"])
        store = WikiStore(root)
        assert store.mark_source_changed(url, "sha1:new") == [note_id]  # md 真被重写
        with make_index(store) as index:
            assert index_drift(store, index)[0] is False
            assert service._index_is_stale(index) is None
            assert ensure_index_fresh(store, index)[0] is False
        payload = service.search(OLD_PHRASE)
        assert "index_rebuilt" not in payload
        assert [r["note_id"] for r in payload["results"]] == [note_id]
