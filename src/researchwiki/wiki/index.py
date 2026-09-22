"""检索索引：SQLite（wiki-data/index.db）= FTS5 关键词 + sqlite-vec 向量 + RRF 融合。

中文 tokenizer 决策（config [wiki].fts_tokenizer = auto|simple|trigram）：
- 优先尝试 wangfenjin/simple（vendor/libsimple-windows-x64/simple.dll）。
- 实测 blocker：本项目路径含中文，cppjieba 的 std::ifstream 打不开非 ASCII
  路径下的 jieba 词库（C++ 层 FATAL abort，Python 无法捕获）；且该 Windows
  构建没有 simple2 tokenizer。词库不可用时 simple 退化为"单字+拼音"分词，
  与有词库行为不一致，故 auto 策略要求"DLL 可加载 且 词库目录纯 ASCII 且
  词库文件齐全"才启用 simple，否则退回内置 trigram（SQLite >= 3.34 自带，
  中文按 3-gram 子串召回，"上下文压缩"→"上下文压缩技术"这类断言成立）。
- trigram 的短查询（<3 字符）天然无法命中，search() 会退回 note_meta 表的
  LIKE 扫描补召回（个人 wiki 语料量级下全扫可接受）。

向量索引：sqlite-vec（vec0, cosine 距离）。Windows 上实测可加载；加载失败时
自动退回纯 Python 余弦暴力扫描。向量始终同时落一份到普通表 note_vec_cache，
两条路径读同源数据、同 API 同结果语义。

一致性策略（MVP）：查询前【不】自动同步 md 文件与索引——由调用方决定
rebuild(store)（全量重建）或 index_note(note)（增量 upsert）。索引存了
body 快照（供向量命中出摘要与短查询 LIKE），md 改动后需重新 index_note。
"""

from __future__ import annotations

import sqlite3
import struct
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from researchwiki.wiki.embeddings import EmbeddingProvider, MockEmbeddingProvider
from researchwiki.wiki.store import Note, WikiStore

RRF_K = 60
CONFIDENCE_FACTOR: Mapping[str, float] = {"high": 1.0, "medium": 0.9, "low": 0.75}
DEFAULT_HALF_LIFE_DAYS: Mapping[str, float] = {"volatile": 30.0, "drifting": 90.0}  # stable 不衰减
# 向量通道的最低余弦相似度：正交（0 相似）不算命中，避免"唯一向量也排第一"
VECTOR_MIN_COSINE = 1e-6

# vendor/simple 的默认探测目录（仓库根下；wheel 安装场景自然探测不到 → trigram）
_VENDOR_SUBDIR = Path("vendor/libsimple-windows-x64")
_REQUIRED_DICT_FILES = ("jieba.dict.utf8", "hmm_model.utf8", "user.dict.utf8")


# ---- tokenizer 探测 ---------------------------------------------------------


def _default_vendor_dir() -> Path | None:
    """仓库根/vendor（src/researchwiki/wiki/index.py 上溯三级）；不存在返回 None。"""
    try:
        candidate = Path(__file__).resolve().parents[3] / _VENDOR_SUBDIR
    except IndexError:  # pragma: no cover - 打包进 zipapp 等极端布局
        return None
    return candidate if candidate.is_dir() else None


def _try_load_simple_dll(dll_path: Path) -> bool:
    """尝试加载 simple.dll 并建一张探针 FTS5 表；任何一步失败都返回 False。"""
    try:
        conn = sqlite3.connect(":memory:")
        try:
            conn.enable_load_extension(True)
            conn.load_extension(str(dll_path))
            conn.enable_load_extension(False)
            conn.execute("CREATE VIRTUAL TABLE probe USING fts5(x, tokenize='simple')")
            conn.execute("INSERT INTO probe VALUES ('上下文压缩测试')")
            conn.execute("SELECT rowid FROM probe WHERE probe MATCH '上下文'")
        finally:
            conn.close()
    except Exception:  # noqa: BLE001 -- 扩展加载失败一律退回 trigram
        return False
    return True


def _jieba_dict_usable(dict_dir: Path) -> bool:
    """jieba 词库可用性检查。

    cppjieba 用 std::ifstream 打词库，Windows 下非 ASCII 路径会触发
    不可捕获的 C++ FATAL abort，因此路径必须纯 ASCII 且关键文件齐全。
    """
    if not all((dict_dir / name).is_file() for name in _REQUIRED_DICT_FILES):
        return False
    try:
        str(dict_dir).encode("ascii")
    except UnicodeEncodeError:
        return False
    return True


def resolve_tokenizer(setting: str, *, vendor_dir: Path | None = None) -> str:
    """把 config 的 fts_tokenizer 取值解析成实际 tokenizer：simple | trigram。

    - "trigram" → 直接 trigram；
    - "simple"  → DLL 可加载即用（无词库时退化为单字分词，仍可 AND 召回），
      否则退 trigram；
    - "auto"（默认）→ 要求 DLL 与 jieba 词库都可用才用 simple，否则 trigram。
    """
    if setting == "trigram":
        return "trigram"
    base = vendor_dir if vendor_dir is not None else _default_vendor_dir()
    dll = base / "simple.dll" if base is not None else None
    dll_ok = dll is not None and dll.is_file() and _try_load_simple_dll(dll)
    if setting == "simple":
        return "simple" if dll_ok else "trigram"
    if setting == "auto":
        dict_ok = base is not None and _jieba_dict_usable(base / "dict")
        return "simple" if dll_ok and dict_ok else "trigram"
    raise ValueError(f"fts_tokenizer 取值非法: {setting!r}（可选 auto/simple/trigram）")


# ---- 数据结构与纯函数 -------------------------------------------------------


@dataclass
class SearchMatch:
    """一条检索命中。redirected_from 非 None 表示该结果是跟随重定向得来的。"""

    note_id: str
    title: str
    snippet: str
    score: float
    match_type: str  # 'fts' | 'vector' | 'both'
    redirected_from: str | None = None


@dataclass
class WikiSettings:
    """[wiki] 段的解析结果：tokenizer 策略与新鲜度半衰期（天）。"""

    fts_tokenizer: str = "auto"
    half_life_days: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_HALF_LIFE_DAYS))


def wiki_settings(config: Mapping[str, Any] | None) -> WikiSettings:
    """从完整 config（tomllib 解析结果）读 [wiki] 段；缺省值见 WikiSettings。"""
    section: object = (config or {}).get("wiki") or {}
    half_life = dict(DEFAULT_HALF_LIFE_DAYS)
    tokenizer = "auto"
    if isinstance(section, Mapping):
        tokenizer = str(section.get("fts_tokenizer") or "auto")
        raw = section.get("half_life_days")
        if isinstance(raw, Mapping):
            # 0 合法（freshness_factor 语义：<=0 不衰减），只过滤 None
            half_life.update({str(k): float(v) for k, v in raw.items() if v is not None})
    return WikiSettings(fts_tokenizer=tokenizer, half_life_days=half_life)


def rrf_fuse(ranked_lists: list[Iterable[str]], *, k: int = RRF_K) -> dict[str, float]:
    """Reciprocal Rank Fusion：score = Σ 1/(k + rank)，rank 从 1 计。

    入参为各通道的有序 note_id 序列（dict 迭代顺序即排名顺序）；同一文档
    多通道命中分数叠加。
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for position, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + position)
    return scores


def freshness_factor(
    volatility: str,
    observed_at: str | None,
    created: str,
    *,
    half_life_days: Mapping[str, float],
    now: datetime,
) -> float:
    """新鲜度因子：volatile/drifting 按 half_life 指数衰减（0.5 ** age/half），stable 恒 1。

    观察时间优先 observed_at，缺省回退 created；都缺失或不合法视为不衰减。
    """
    half_life = half_life_days.get(volatility)
    if half_life is None or half_life <= 0:
        return 1.0
    timestamp = _parse_ts(observed_at) or _parse_ts(created)
    if timestamp is None:
        return 1.0
    age_days = (now - timestamp).total_seconds() / 86400.0
    if age_days <= 0:
        return 1.0
    return 0.5 ** (age_days / half_life)


def _parse_ts(text: str | None) -> datetime | None:
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _serialize_vector(vec: list[float]) -> bytes:
    """float32 小端打包（sqlite-vec raw bytes 格式，暴力扫描路径复用）。"""
    return struct.pack(f"{len(vec)}f", *vec)


def _deserialize_vector(blob: bytes) -> list[float]:
    return list(struct.unpack(f"{len(blob) // 4}f", blob))


# ---- 索引主体 ---------------------------------------------------------------


class SearchIndex:
    """FTS5 + 向量双通道检索索引（SQLite 单库 wiki-data/index.db）。

    embedding 缺省为 MockEmbeddingProvider（skeleton-first）；接入真模型时
    传入 get_embedding_provider(config) 的结果即可，索引/检索代码不变。
    vec0 表维度在建表时固定：换 embedding（维度变化）会自动 DROP 重建，
    需要随后 rebuild()。
    """

    def __init__(
        self,
        root: str | Path = "wiki-data",
        *,
        embedding: EmbeddingProvider | None = None,
        tokenizer: str = "auto",
        half_life_days: Mapping[str, float] | None = None,
        clock: Callable[[], datetime] | None = None,
        vendor_dir: Path | None = None,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "index.db"
        self.embedding = embedding or MockEmbeddingProvider()
        self.half_life_days = dict(DEFAULT_HALF_LIFE_DAYS)
        if half_life_days:
            self.half_life_days.update(half_life_days)
        self.clock: Callable[[], datetime] = clock or (lambda: datetime.now(UTC))
        self._conn = sqlite3.connect(self.db_path)
        self._vec_ready = self._load_vec_extension()
        self._tokenizer = resolve_tokenizer(tokenizer, vendor_dir=vendor_dir)
        if self._tokenizer == "simple":
            self._activate_simple_on_conn(vendor_dir)  # 扩展按连接生效，主连接必须单独加载
        self._jieba_usable = self._setup_jieba(vendor_dir)
        self._vec_dim: int | None = getattr(self.embedding, "dim", None)
        self._direct_hits: set[str] = set()
        self._redirect_source: dict[str, str] = {}
        self._ensure_tables()

    @property
    def tokenizer(self) -> str:
        """实际生效的 tokenizer（auto 解析后的结果）。"""
        return self._tokenizer

    # ---- 初始化 ----

    def _load_vec_extension(self) -> bool:
        """加载 sqlite-vec；失败（Windows 扩展问题等）退回纯 Python 暴力扫描。"""
        try:
            import sqlite_vec

            self._conn.enable_load_extension(True)
            self._conn.load_extension(sqlite_vec.loadable_path())
            self._conn.enable_load_extension(False)
            return True
        except Exception:  # noqa: BLE001 -- 扩展不可用一律走暴力扫描
            return False

    def _activate_simple_on_conn(self, vendor_dir: Path | None) -> None:
        """把 simple.dll 装载到当前连接（SQLite 扩展按连接生效）；失败退回 trigram。"""
        base = vendor_dir if vendor_dir is not None else _default_vendor_dir()
        dll = base / "simple.dll" if base is not None else None
        if dll is None or not dll.is_file():
            self._tokenizer = "trigram"
            return
        try:
            self._conn.enable_load_extension(True)
            self._conn.load_extension(str(dll))
            self._conn.enable_load_extension(False)
        except sqlite3.Error:
            self._tokenizer = "trigram"

    def _setup_jieba(self, vendor_dir: Path | None) -> bool:
        """simple tokenizer 下尝试装载 jieba 词库（查询期词级短语分词用）。"""
        if self._tokenizer != "simple":
            return False
        base = vendor_dir if vendor_dir is not None else _default_vendor_dir()
        if base is None or not _jieba_dict_usable(base / "dict"):
            return False
        try:
            self._conn.execute("SELECT jieba_dict(?)", (str(base / "dict"),))
        except sqlite3.Error:
            return False
        return True

    def _ensure_tables(self) -> None:
        conn = self._conn
        conn.execute(
            "CREATE TABLE IF NOT EXISTS note_meta ("
            "note_id TEXT PRIMARY KEY, title TEXT, body TEXT, confidence TEXT, "
            "volatility TEXT, status TEXT, redirect_to TEXT, superseded_by TEXT, "
            "observed_at TEXT, created TEXT)"
        )
        conn.execute("CREATE TABLE IF NOT EXISTS index_meta (key TEXT PRIMARY KEY, value TEXT)")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS note_vec_cache ("
            "note_id TEXT PRIMARY KEY, dim INTEGER NOT NULL, vec BLOB NOT NULL)"
        )
        # tokenizer 与已建 fts 表不一致（换机器/改配置）时重建 fts 表，需随后 rebuild
        row = conn.execute("SELECT value FROM index_meta WHERE key = 'tokenizer'").fetchone()
        if row is not None and row[0] != self._tokenizer:
            conn.execute("DROP TABLE IF EXISTS note_fts")
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS note_fts USING fts5("
            f"note_id UNINDEXED, title, body, tokenize='{self._tokenizer}')"
        )
        conn.execute(
            "INSERT OR REPLACE INTO index_meta (key, value) VALUES ('tokenizer', ?)",
            (self._tokenizer,),
        )
        stored_dim = conn.execute("SELECT value FROM index_meta WHERE key = 'vec_dim'").fetchone()
        stored_int = int(stored_dim[0]) if stored_dim is not None else None
        if self._vec_dim is not None and stored_int is not None and stored_int != self._vec_dim:
            conn.execute("DROP TABLE IF EXISTS note_vec")
            conn.execute("DELETE FROM index_meta WHERE key = 'vec_dim'")
        conn.commit()

    def _ensure_vec_table(self, dim: int) -> bool:
        """惰性建 vec0 表；sqlite-vec 不可用或建表失败返回 False（走暴力扫描）。"""
        if not self._vec_ready:
            return False
        row = self._conn.execute("SELECT value FROM index_meta WHERE key = 'vec_dim'").fetchone()
        if row is None or int(row[0]) != dim:
            self._conn.execute("DROP TABLE IF EXISTS note_vec")
            self._conn.execute(
                f"CREATE VIRTUAL TABLE note_vec USING vec0("
                f"note_id TEXT PRIMARY KEY, emb float[{dim}] distance_metric=cosine)"
            )
            self._conn.execute(
                "INSERT OR REPLACE INTO index_meta (key, value) VALUES ('vec_dim', ?)", (str(dim),)
            )
            self._conn.commit()
        return True

    # ---- 写入 ----

    def index_note(self, note: Note) -> None:
        """增量 upsert 一条笔记（fts / note_meta / 向量三处同步）。"""
        conn = self._conn
        conn.execute("DELETE FROM note_fts WHERE note_id = ?", (note.id,))
        conn.execute(
            "INSERT INTO note_fts (note_id, title, body) VALUES (?, ?, ?)",
            (note.id, note.title, note.body),
        )
        conn.execute(
            "INSERT OR REPLACE INTO note_meta (note_id, title, body, confidence, volatility, "
            "status, redirect_to, superseded_by, observed_at, created) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                note.id,
                note.title,
                note.body,
                note.meta.confidence,
                note.meta.volatility,
                note.meta.status,
                note.meta.redirect_to,
                note.meta.superseded_by,
                note.meta.observed_at,
                note.meta.created,
            ),
        )
        embedding_text = f"{note.title}\n{note.body}"
        vec = self.embedding.embed([embedding_text])[0]
        conn.execute(
            "INSERT OR REPLACE INTO note_vec_cache (note_id, dim, vec) VALUES (?, ?, ?)",
            (note.id, len(vec), _serialize_vector(vec)),
        )
        if self._ensure_vec_table(len(vec)):
            conn.execute("DELETE FROM note_vec WHERE note_id = ?", (note.id,))
            conn.execute(
                "INSERT INTO note_vec (note_id, emb) VALUES (?, ?)",
                (note.id, _serialize_vector(vec)),
            )
        conn.commit()

    def rebuild(self, store: WikiStore) -> int:
        """全量重建：清空索引表后逐条 index_note；返回索引条数。

        会一并索引进 merged/superseded 笔记（检索时跟随重定向并标注来源）。
        """
        conn = self._conn
        conn.execute("DELETE FROM note_fts")
        conn.execute("DELETE FROM note_meta")
        conn.execute("DELETE FROM note_vec_cache")
        if self._vec_ready:
            conn.execute("DROP TABLE IF EXISTS note_vec")
            conn.execute("DELETE FROM index_meta WHERE key = 'vec_dim'")
        conn.commit()
        count = 0
        for note in store.list_notes(status=None):
            self.index_note(note)
            count += 1
        return count

    def indexed_status(self) -> dict[str, str]:
        """索引内 ``{note_id: status}`` 的只读快照。

        仅供索引新鲜度检查（如 ``prior.ensure_index_fresh``）比对 store 全量
        状态用；不代表检索可用性，调用方不得据此写入或改判笔记状态。
        """
        rows = self._conn.execute("SELECT note_id, status FROM note_meta").fetchall()
        return {str(r[0]): str(r[1] or "") for r in rows}

    # ---- 检索 ----

    def search(self, query: str, k: int = 5) -> list[SearchMatch]:
        """双通道检索 + RRF 融合 + 置信/新鲜度调节；默认只回 active 笔记。

        命中 merged/superseded 时跟随重定向到最终 active 笔记并在结果上
        标注 redirected_from（值是被命中的那条别名笔记 id）。
        """
        query = query.strip()
        if not query:
            return []
        meta_map = self._load_meta_map()
        pool = max(k * 4, 16)

        fts_hits = self._fts_candidates(query, pool)
        vector_hits = self._vector_candidates(query, pool)

        # 重定向解析：别名笔记 → 最终 active 笔记（防环、断裂丢弃）
        self._direct_hits = set()
        self._redirect_source = {}
        fts_ranked = self._resolve_redirected(fts_hits, meta_map)
        vector_ranked = self._resolve_redirected(vector_hits, meta_map)

        fused = rrf_fuse([fts_ranked, vector_ranked])
        now = self.clock()
        matches: list[SearchMatch] = []
        for note_id, fused_score in fused.items():
            info = meta_map[note_id]
            confidence = CONFIDENCE_FACTOR.get(info["confidence"], CONFIDENCE_FACTOR["medium"])
            fresh = freshness_factor(
                info["volatility"],
                info["observed_at"],
                info["created"],
                half_life_days=self.half_life_days,
                now=now,
            )
            in_fts, in_vec = note_id in fts_ranked, note_id in vector_ranked
            match_type = "both" if in_fts and in_vec else ("fts" if in_fts else "vector")
            snippet = fts_ranked.get(note_id) or self._body_snippet(info)
            direct = note_id in self._direct_hits
            matches.append(
                SearchMatch(
                    note_id=note_id,
                    title=str(info["title"]),
                    snippet=snippet,
                    score=fused_score * confidence * fresh,
                    match_type=match_type,
                    redirected_from=None if direct else self._redirect_source.get(note_id),
                )
            )
        matches.sort(key=lambda m: (-m.score, m.note_id))
        return matches[:k]

    def _load_meta_map(self) -> dict[str, dict[str, str | None]]:
        rows = self._conn.execute(
            "SELECT note_id, title, confidence, volatility, status, redirect_to, "
            "superseded_by, observed_at, created, body FROM note_meta"
        ).fetchall()
        keys = (
            "note_id",
            "title",
            "confidence",
            "volatility",
            "status",
            "redirect_to",
            "superseded_by",
            "observed_at",
            "created",
            "body",
        )
        return {str(row[0]): dict(zip(keys, row, strict=True)) for row in rows}

    def _fts_candidates(self, query: str, pool: int) -> list[tuple[str, str]]:
        """关键词通道：返回 (note_id, snippet) 有序列表。

        simple tokenizer 且 jieba 词库可用时用 jieba_query 做词级短语查询；
        trigram 用短语包裹（子串语义）；trigram 下超短查询（<3 字符）FTS
        无法命中，退回 note_meta 的 LIKE 全扫补召回（snippet 取 body 前缀）。
        """
        expr = self._fts_match_expr(query)
        if expr is not None:
            try:
                rows = self._conn.execute(
                    "SELECT note_id, snippet(note_fts, 2, '[', ']', '…', 12) "
                    "FROM note_fts WHERE note_fts MATCH ? ORDER BY rank LIMIT ?",
                    (expr, pool),
                ).fetchall()
                return [(str(r[0]), str(r[1])) for r in rows]
            except sqlite3.OperationalError:
                pass  # 语法不兼容等异常 → 落到 LIKE 兜底
        like = f"%{query}%"
        rows = self._conn.execute(
            "SELECT note_id, substr(body, 1, 60) FROM note_meta "
            "WHERE title LIKE ? OR body LIKE ? ORDER BY note_id LIMIT ?",
            (like, like, pool),
        ).fetchall()
        return [(str(r[0]), str(r[1])) for r in rows]

    def _fts_match_expr(self, query: str) -> str | None:
        """构造 MATCH 表达式；返回 None 表示走 LIKE 兜底。"""
        if self._tokenizer == "simple" and self._jieba_usable:
            try:
                row = self._conn.execute("SELECT jieba_query(?)", (query,)).fetchone()
                if row and row[0]:
                    return str(row[0])
            except sqlite3.Error:
                pass
        if self._tokenizer == "trigram" and len(query) < 3:
            return None
        escaped = query.replace('"', '""')
        return f'"{escaped}"'

    def _vector_candidates(self, query: str, pool: int) -> list[tuple[str, str]]:
        """向量通道：返回 (note_id, '') 有序列表（vec0 KNN 或纯 Python 暴力扫描）。"""
        if not self._conn.execute("SELECT 1 FROM note_vec_cache LIMIT 1").fetchone():
            return []
        try:
            query_vec = self.embedding.embed([query])[0]
        except Exception:  # noqa: BLE001 -- 嵌入失败时向量通道静默降级
            return []
        if self._ensure_vec_table(len(query_vec)):
            rows = self._conn.execute(
                "SELECT note_id, distance FROM note_vec WHERE emb MATCH ? AND k = ?",
                (_serialize_vector(query_vec), pool),
            ).fetchall()
            # cosine 距离 = 1 - 相似度；正交（distance >= 1）不算命中
            return [
                (str(r[0]), "")
                for r in rows
                if float(r[1]) < 1.0 - VECTOR_MIN_COSINE
            ]
        return self._brute_force_candidates(query_vec, pool)

    def _brute_force_candidates(self, query_vec: list[float], pool: int) -> list[tuple[str, str]]:
        """纯 Python 余弦暴力扫描（sqlite-vec 不可用时的同语义退路）。"""
        rows = self._conn.execute("SELECT note_id, vec FROM note_vec_cache").fetchall()
        norm_q = sum(x * x for x in query_vec) ** 0.5
        scored: list[tuple[float, str]] = []
        for note_id, blob in rows:
            stored = _deserialize_vector(blob)
            if len(stored) != len(query_vec):
                continue
            norm_v = sum(x * x for x in stored) ** 0.5
            if not norm_q or not norm_v:
                continue
            cosine = sum(a * b for a, b in zip(query_vec, stored, strict=True)) / (norm_q * norm_v)
            if cosine <= VECTOR_MIN_COSINE:
                continue  # 与 KNN 路径同语义：正交不算命中
            scored.append((cosine, str(note_id)))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [(note_id, "") for _, note_id in scored[:pool]]

    def _resolve_redirected(
        self, candidates: list[tuple[str, str]], meta_map: Mapping[str, Mapping[str, str | None]]
    ) -> dict[str, str]:
        """候选列表 → {最终 active note_id: payload(snippet)}，去重保首现（最优）排名。

        直接命中（自身即 active）记入 self._direct_hits；经重定向到达的把
        target → 被命中的别名 id 记入 self._redirect_source 供结果标注。
        """
        out: dict[str, str] = {}
        for note_id, payload in candidates:
            target, source = self._follow(note_id, meta_map)
            if target is None:
                continue
            if source is None:
                self._direct_hits.add(target)
            else:
                self._redirect_source.setdefault(target, source)
            if target not in out:
                out[target] = payload
        return out

    @staticmethod
    def _follow(
        note_id: str, meta_map: Mapping[str, Mapping[str, str | None]]
    ) -> tuple[str | None, str | None]:
        """沿 redirect 链走到最终 active 笔记；环路/断裂/落点非 active 返回 None。"""
        visited = {note_id}
        current = note_id
        first_alias: str | None = None
        while True:
            info = meta_map.get(current)
            if info is None:
                return None, None
            status = info["status"]
            if status == "active":
                return current, first_alias
            if status == "merged":
                nxt = info["redirect_to"]
            elif status == "superseded":
                nxt = info["superseded_by"]
            else:
                return None, None
            if not nxt or nxt in visited:
                return None, None
            if first_alias is None:
                first_alias = current
            visited.add(nxt)
            current = nxt

    @staticmethod
    def _body_snippet(info: Mapping[str, str | None]) -> str:
        """非 FTS 通道（向量命中 / 重定向）的摘要：body 前缀截断。"""
        body = str(info.get("body") or "")
        return body[:80] + ("…" if len(body) > 80 else "")

    # ---- 生命周期 ----

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> SearchIndex:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def wiki_search(
    query: str,
    k: int = 5,
    *,
    store: WikiStore,
    embedding: EmbeddingProvider | None = None,
    tokenizer: str = "auto",
    half_life_days: Mapping[str, float] | None = None,
    clock: Callable[[], datetime] | None = None,
) -> list[SearchMatch]:
    """便捷检索入口：面向 Distiller / MCP 层。

    注意（MVP 一致性策略）：本函数【不】自动同步 md → 索引。首次使用或 md
    变更后，由调用方执行 ``SearchIndex(store.root).rebuild(store)`` 或逐条
    ``index_note(note)``；否则查的是上次索引的快照。
    """
    with SearchIndex(
        store.root,
        embedding=embedding,
        tokenizer=tokenizer,
        half_life_days=half_life_days,
        clock=clock,
    ) as index:
        return index.search(query, k=k)
