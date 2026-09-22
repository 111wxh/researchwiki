"""索引正文指纹与 ensure_index_fresh 三维比对（P2-B：carried 一致性修复）。

覆盖 task-9 简报需求 1 与需求 2：

1. ``note_meta.body_hash`` 列 + 旧库迁移（探测缺列 → ALTER，存量行留空）+ 只读
   方法 ``indexed_fingerprints()``（``indexed_status()`` 契约不变）；
2. ``ensure_index_fresh`` 的三维落后判定：id 集合 / status / **正文指纹**，以及
   回归用例：
   a. ``store.save_note`` 同 id 原地改写正文 → 判落后并 rebuild，之后检索命中
      新正文、不再命中旧正文；
   b. ``memory_update`` 走完整 MCP 服务面改正文 → 索引一致性可检出（service 层
      直接调用，不起 server、不走 stdio）；
   c. status 变化与新增笔记两个旧维度不回归（另见 test_prior.py 的同名用例）。

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
from researchwiki.wiki.index import SearchIndex, SearchMatch, note_body_hash
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


def _fts_hits(index: SearchIndex, query: str) -> list[SearchMatch]:
    """只保留真正来自索引正文的命中（fts / both），滤掉向量通道噪声。"""
    return [m for m in index.search(query, k=5) if m.match_type in ("fts", "both")]


# ---- 需求 1：指纹列 / 只读方法 / 旧库迁移 -------------------------------------


class TestBodyHashColumn:
    def test_indexed_fingerprints_matches_indexed_content(self, store: WikiStore) -> None:
        note = store.save_note(OLD_BODY, note_id="N-0001", title=TITLE)
        with make_index(store) as index:
            index.rebuild(store)
            fingerprints = index.indexed_fingerprints()
            assert fingerprints == {
                "N-0001": ("active", note_body_hash(note.title, note.body))
            }
            # 契约不动：indexed_status 仍是 {note_id: status}（P1-A 调用方依赖）
            assert index.indexed_status() == {"N-0001": "active"}

    def test_fingerprint_covers_title_and_body(self) -> None:
        """指纹覆盖 title + body（两者都进 FTS 与 note_meta，改任一个都算变化）。"""
        assert note_body_hash("甲", "同样的正文") != note_body_hash("乙", "同样的正文")
        assert note_body_hash("甲", "正文一") != note_body_hash("甲", "正文二")
        assert note_body_hash("甲", "正文一") == note_body_hash("甲", "正文一")

    def test_title_only_change_is_detected(self, store: WikiStore) -> None:
        store.save_note(OLD_BODY, note_id="N-0001", title="旧标题")
        with make_index(store) as index:
            index.rebuild(store)
            # 只改标题（正文一字不动）：检索可见的 title 变了，索引同样失真
            store.save_note(OLD_BODY, note_id="N-0001", title="新标题")
            rebuilt, reason = ensure_index_fresh(store, index)
            assert rebuilt is True
            assert "正文变化" in reason and "N-0001" in reason

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
            # 存量行不回填：归一成空串（"不知道是哪版正文"）
            assert index.indexed_fingerprints()["N-0001"] == ("active", "")
            assert index.indexed_status() == {"N-0001": "active"}
            rebuilt, reason = ensure_index_fresh(store, index)
            assert rebuilt is True
            assert "旧索引缺正文指纹" in reason
            # rebuild 之后指纹回填，再检查即新鲜
            assert index.indexed_fingerprints()["N-0001"][1] == note_body_hash(
                note.title, note.body
            )
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


# ---- 需求 2：同 id 原地改写正文的检出与 rebuild --------------------------------


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
            assert "正文变化" in reason and "N-0001" in reason
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
            assert "正文变化" in reason and note_id in reason
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
