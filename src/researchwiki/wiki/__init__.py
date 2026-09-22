"""wiki 子系统：frontmatter / 实体注册表 / 存储层 / 嵌入 / 检索索引。"""

from researchwiki.wiki.embeddings import (
    CachedEmbeddingProvider,
    EmbeddingError,
    EmbeddingProvider,
    MockEmbeddingProvider,
    OpenAICompatibleEmbedding,
    get_embedding_provider,
)
from researchwiki.wiki.entities import Entity, EntityRegistry, slugify
from researchwiki.wiki.frontmatter import NoteMeta, SourceRef, dump, parse
from researchwiki.wiki.index import (
    SearchIndex,
    SearchMatch,
    WikiSettings,
    freshness_factor,
    rrf_fuse,
    wiki_search,
    wiki_settings,
)
from researchwiki.wiki.prior import (
    PRIOR_CONTEXT_LABEL,
    PriorContext,
    PriorHit,
    ensure_index_fresh,
    format_prior_context,
    retrieve_priors,
)
from researchwiki.wiki.store import Conflict, Note, Page, WikiStore

__all__ = [
    "CachedEmbeddingProvider",
    "Conflict",
    "EmbeddingError",
    "EmbeddingProvider",
    "Entity",
    "EntityRegistry",
    "MockEmbeddingProvider",
    "Note",
    "NoteMeta",
    "OpenAICompatibleEmbedding",
    "PRIOR_CONTEXT_LABEL",
    "Page",
    "PriorContext",
    "PriorHit",
    "SearchIndex",
    "SearchMatch",
    "SourceRef",
    "WikiSettings",
    "WikiStore",
    "dump",
    "ensure_index_fresh",
    "format_prior_context",
    "freshness_factor",
    "get_embedding_provider",
    "parse",
    "retrieve_priors",
    "rrf_fuse",
    "slugify",
    "wiki_search",
    "wiki_settings",
]
