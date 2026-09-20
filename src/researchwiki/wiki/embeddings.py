"""嵌入 Provider 抽象、OpenAI 兼容实现、确定性 Mock 与 SQLite 本地缓存。

skeleton-first（与 llm 路由同一哲学）：config 的 [embedding] 段 base_url/model
留空时，工厂直接返回 MockEmbeddingProvider——零 key、零网络、全链路可测。

本地缓存选择与检索索引同库（wiki-data/index.db，表 embedding_cache），键为
(model, sha256(text))：少管理一个文件，且缓存命中判定天然区分模型；命中不重算。
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

import httpx

from researchwiki.tools.fs import DEFAULT_WIKI_DATA_ROOT

DEFAULT_EMBEDDING_CACHE = DEFAULT_WIKI_DATA_ROOT / "index.db"


class EmbeddingError(Exception):
    """嵌入调用失败：非 200 响应或响应结构不符合 OpenAI 格式。"""


class EmbeddingProvider(Protocol):
    """嵌入 Provider 接口：批量文本 → 等长向量列表（顺序一一对应）。"""

    model: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _bigram_vector(text: str, dim: int) -> list[float]:
    """确定性特征向量：字符 unigram + bigram 的带符号特征哈希 + L2 归一化。

    用 hashlib（而非内建 hash——它按进程加盐，不可复现）。带符号哈希降低
    碰撞偏置，使相似中文串有非零余弦相似度、不相似串近似正交。
    """
    vec = [0.0] * dim
    grams = list(text)
    grams.extend(text[i : i + 2] for i in range(len(text) - 1))
    for gram in grams:
        digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        vec[value % dim] += 1.0 if (value >> 63) & 1 == 0 else -1.0
    norm = sum(x * x for x in vec) ** 0.5
    return [x / norm for x in vec] if norm else vec


class MockEmbeddingProvider:
    """确定性 Mock：bigram 特征哈希 + L2 归一化，无网络、跨进程可复现。"""

    model = "mock-embedding"

    def __init__(self, dim: int = 128) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [_bigram_vector(t, self.dim) for t in texts]


class OpenAICompatibleEmbedding:
    """OpenAI 兼容 /embeddings 端点（智谱、SiliconFlow 等均为此格式）。

    transport 参数用于注入 httpx.MockTransport（测试零真实网络）。
    api_key 优先；未给时读 api_key_env 指定的环境变量。
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        api_key_env: str | None = None,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        if api_key is None:
            api_key = os.environ.get(api_key_env, "") if api_key_env else ""
        self.api_key = api_key
        self.transport = transport
        self.timeout = timeout

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        with httpx.Client(transport=self.transport, timeout=self.timeout) as client:
            resp = client.post(
                f"{self.base_url}/embeddings",
                json={"model": self.model, "input": list(texts)},
                headers=headers,
            )
        if resp.status_code != 200:
            raise EmbeddingError(f"embeddings HTTP {resp.status_code}: {resp.text[:200]}")
        payload = resp.json()
        data = payload.get("data")
        if not isinstance(data, list) or len(data) != len(texts):
            raise EmbeddingError("embeddings 响应缺少 data 或长度不匹配")
        vectors: list[list[float]] = []
        for item in sorted(data, key=lambda d: d.get("index", 0)):
            vec = item.get("embedding")
            if not isinstance(vec, list) or not vec:
                raise EmbeddingError("embeddings 响应缺 embedding 向量")
            vectors.append([float(x) for x in vec])
        return vectors


class CachedEmbeddingProvider:
    """SQLite 缓存装饰器：keyed (model, sha256(text))，命中不重算、不触发 transport。

    缓存与检索索引同库（wiki-data/index.db），表 embedding_cache 独立建表。
    """

    def __init__(self, inner: EmbeddingProvider, cache_path: str | Path) -> None:
        self.inner = inner
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.cache_path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS embedding_cache ("
            "model TEXT NOT NULL, key TEXT NOT NULL, dim INTEGER NOT NULL, vec BLOB NOT NULL, "
            "PRIMARY KEY (model, key))"
        )
        self._conn.commit()

    def _lookup(self, key: str) -> list[float] | None:
        row = self._conn.execute(
            "SELECT dim, vec FROM embedding_cache WHERE model = ? AND key = ?",
            (self.inner.model, key),
        ).fetchone()
        if row is None:
            return None
        return list(struct.unpack(f"{row[0]}f", row[1]))

    def _store(self, key: str, vec: list[float]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO embedding_cache (model, key, dim, vec) VALUES (?, ?, ?, ?)",
            (self.inner.model, key, len(vec), struct.pack(f"{len(vec)}f", *vec)),
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        results: dict[int, list[float]] = {}
        missing: list[tuple[int, str, str]] = []
        for i, text in enumerate(texts):
            key = hashlib.sha256(text.encode("utf-8")).hexdigest()
            hit = self._lookup(key)
            if hit is None:
                missing.append((i, text, key))
            else:
                results[i] = hit
        if missing:
            fresh = self.inner.embed([text for _, text, _ in missing])
            for (i, _, key), vec in zip(missing, fresh, strict=True):
                self._store(key, vec)
                results[i] = vec
        return [results[i] for i in range(len(texts))]

    def close(self) -> None:
        self._conn.close()


def get_embedding_provider(
    config: Mapping[str, object] | None,
    *,
    cache_path: str | Path | None = DEFAULT_EMBEDDING_CACHE,
) -> EmbeddingProvider:
    """工厂：读 config 的 [embedding] 段。

    base_url 或 model 留空 → MockEmbeddingProvider（skeleton-first，无需 key）；
    否则 OpenAICompatibleEmbedding（api_key_env 指定的环境变量取 key），
    cache_path 非 None 时再包一层 SQLite 缓存（默认与索引同库 index.db）。
    """
    cfg_raw: object = (config or {}).get("embedding") or {}
    cfg: Mapping[str, object] = cfg_raw if isinstance(cfg_raw, Mapping) else {}
    model = str(cfg.get("model") or "").strip()
    base_url = str(cfg.get("base_url") or "").strip()
    if not base_url or not model:
        return MockEmbeddingProvider()
    inner: EmbeddingProvider = OpenAICompatibleEmbedding(
        base_url=base_url,
        model=model,
        api_key_env=str(cfg.get("api_key_env") or "") or None,
    )
    if cache_path is not None:
        return CachedEmbeddingProvider(inner, cache_path)
    return inner


__all__ = [
    "DEFAULT_EMBEDDING_CACHE",
    "CachedEmbeddingProvider",
    "EmbeddingError",
    "EmbeddingProvider",
    "MockEmbeddingProvider",
    "OpenAICompatibleEmbedding",
    "get_embedding_provider",
]
