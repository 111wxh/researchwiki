"""入库去重（轻量 merge）：候选笔记写库前与既有 active 笔记查重，命中即合并 + redirect 留痕。

判定「重复」的两个条件（同时满足才算）：

1. 向量余弦相似度 ≥ ``similarity_threshold``（默认 0.9，config ``[wiki].dedup_similarity``）；
2. 实体重叠数 ≥ ``entity_overlap_min``（默认 1，config ``[wiki].dedup_entity_overlap``），
   实体名按 ``slugify`` 归一后比较（"GLM-5.3" 与 "glm-5-3" 视为同一实体）。

合并策略（**规范 ID 不变**：旧 ID 永不消失，引用一律指向规范 ID）：

- 命中：既有 active 笔记保持原 ID / created / 正文（简化实现保留原文），把新证据
  （sources 并集）、新实体并入其 frontmatter，把留痕 id 记进 ``merged_from``；
  若注入了 provider，则用一次 LLM 调用把两条同义断言重写成一句（失败即回退原文）。
  同时**新分配一个编号**承载候选内容，写成 ``status=merged`` + ``redirect_to=<规范 ID>``
  的留痕记录——候选的原始表述与来源因此可追溯，检索/引用旧 ID 会跟随到规范 ID。
- 未命中：按编号续接新建 active 笔记。

``add()`` 返回的 ``IngestResult.note`` 一定是规范 ID 那条，调用方（事件层/引用层）
必须用它。
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from researchwiki.llm.accounting import TokenAccountant
from researchwiki.llm.provider import Provider, TokenUsage
from researchwiki.wiki.distiller import CandidateNote, call_text, coerce_enum
from researchwiki.wiki.embeddings import EmbeddingProvider, MockEmbeddingProvider
from researchwiki.wiki.entities import slugify
from researchwiki.wiki.frontmatter import CONFIDENCE_LEVELS, VOLATILITY_LEVELS, SourceRef

if TYPE_CHECKING:  # 仅类型标注用：避免 wiki.store → loop.notes → loop.agent_loop 循环导入
    from researchwiki.wiki.store import Note, WikiStore

DEFAULT_SIMILARITY = 0.9
DEFAULT_ENTITY_OVERLAP = 1
MAX_MERGED_BODY_CHARS = 300

MERGE_SYSTEM = (
    "你是 wiki 笔记合并器。给定两条同义或高度重叠的原子笔记，合并成一条更完整的断言。\n"
    "规则：只输出合并后的正文（一行，不要解释、不要标注来源、不要加引号），不超过 80 字；"
    "保留双方不冲突的信息；细节冲突时以更具体、更新的表述为准；无法合并时原样输出第一条。"
)


@dataclass
class DedupSettings:
    """入库去重阈值（[wiki] 段可覆盖，见 config.toml 注释）。"""

    similarity: float = DEFAULT_SIMILARITY
    entity_overlap_min: int = DEFAULT_ENTITY_OVERLAP


def dedup_settings(config: Mapping[str, Any] | None = None) -> DedupSettings:
    """读去重阈值：接受完整 config（取 [wiki] 段）或直接给 [wiki] 段；非法值回默认。"""
    section: Mapping[str, Any] = {}
    if isinstance(config, Mapping):
        raw = config.get("wiki")
        section = raw if isinstance(raw, Mapping) else config
    similarity = DEFAULT_SIMILARITY
    value = section.get("dedup_similarity")
    if value is not None:
        try:
            similarity = min(1.0, max(0.0, float(value)))
        except (TypeError, ValueError):
            similarity = DEFAULT_SIMILARITY
    overlap = DEFAULT_ENTITY_OVERLAP
    raw_overlap = section.get("dedup_entity_overlap")
    if raw_overlap is not None:
        try:
            overlap = max(0, int(raw_overlap))
        except (TypeError, ValueError):
            overlap = DEFAULT_ENTITY_OVERLAP
    return DedupSettings(similarity=similarity, entity_overlap_min=overlap)


@dataclass
class IngestResult:
    """一次入库的结果：note 恒为规范 ID 的笔记。"""

    note: Note
    action: str  # "created" | "merged"
    merged_into: str | None = None  # 命中合并时的规范 ID
    merged_from: str | None = None  # 合并留痕记录（status=merged）的 id
    similarity: float = 0.0  # 命中时为与规范笔记的余弦相似度；未命中恒为 0.0


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """余弦相似度；维度不一致或零向量返回 0.0（调用方按"不相似"处理）。"""
    if len(a) != len(b) or not a:
        return 0.0
    num = float(sum(x * y for x, y in zip(a, b, strict=True)))
    norm_a = float(sum(x * x for x in a)) ** 0.5
    norm_b = float(sum(y * y for y in b)) ** 0.5
    if not norm_a or not norm_b:
        return 0.0
    return num / (norm_a * norm_b)


def entity_keys(names: Sequence[str]) -> set[str]:
    """实体名归一键集合：slugify 后 casefold（跨写法统一到同一稳定 ID）。"""
    return {slugify(str(name)).casefold() for name in names if str(name).strip()}


def _stronger_confidence(left: str, right: str) -> str:
    """置信度取更强的一方（high > medium > low；未知值按 medium）。"""
    ranks = {level: i for i, level in enumerate(CONFIDENCE_LEVELS)}
    left_rank = ranks.get(coerce_enum(left, CONFIDENCE_LEVELS, "medium"), 1)
    right_rank = ranks.get(coerce_enum(right, CONFIDENCE_LEVELS, "medium"), 1)
    return left if left_rank <= right_rank else right


def _more_volatile(left: str, right: str) -> str:
    """volatility 取更易变的一方（volatile > drifting > stable）——保守起见宁可多刷新。"""
    ranks = {level: i for i, level in enumerate(VOLATILITY_LEVELS)}
    left_rank = ranks.get(coerce_enum(left, VOLATILITY_LEVELS, "stable"), 0)
    right_rank = ranks.get(coerce_enum(right, VOLATILITY_LEVELS, "stable"), 0)
    return left if left_rank >= right_rank else right


def _merge_refs(left: Sequence[SourceRef], right: Sequence[SourceRef]) -> list[SourceRef]:
    """来源并集（按 url + content_hash 去重，保序）。"""
    out: list[SourceRef] = []
    for ref in [*left, *right]:
        if ref.url and ref not in out:
            out.append(ref)
    return out


class Ingestor:
    """候选笔记入库门面：查重 → 合并（含留痕）/ 新建。

    embedding 缺省为 MockEmbeddingProvider（确定性 bigram 特征哈希，零网络）；
    真模型场景传 get_embedding_provider(config) 的结果即可。provider 注入后才会用
    LLM 重写合并正文；不注入则走简化合并（保留原文 + 合并 sources），
    适合 run 内联（不额外增加模型调用）。
    """

    def __init__(
        self,
        store: WikiStore,
        *,
        embedding: EmbeddingProvider | None = None,
        similarity_threshold: float = DEFAULT_SIMILARITY,
        entity_overlap_min: int = DEFAULT_ENTITY_OVERLAP,
        provider: Provider | None = None,
        accountant: TokenAccountant | None = None,
        trace_id: str = "",
        clock: Callable[[], float] = time.perf_counter,
        on_usage: Callable[[TokenUsage], None] | None = None,
    ) -> None:
        self.store = store
        self.embedding = embedding if embedding is not None else MockEmbeddingProvider()
        self.similarity_threshold = similarity_threshold
        self.entity_overlap_min = entity_overlap_min
        self.provider = provider
        self.accountant = accountant
        self.trace_id = trace_id
        self.clock = clock
        self.on_usage = on_usage
        self.degradations: list[str] = []
        self.merged_bodies = 0  # 用 LLM 重写过正文的合并次数（stat 用）

    # ---- 主入口 --------------------------------------------------------

    def add(self, candidate: CandidateNote, *, trace_id: str = "") -> IngestResult:
        """把一条候选笔记入库；返回的 note 恒为可对外引用的规范 ID 笔记。"""
        text = str(candidate.text or "").strip()
        if not text:
            raise ValueError("候选笔记正文为空，不能入库")
        refs = list(candidate.source_refs)
        existing = self.store.list_notes()  # 只与 active 笔记比对
        match, similarity = self._best_match(text, candidate.entities, existing)
        if match is None:
            note = self.store.save_note(
                text,
                entities=list(candidate.entities),
                confidence=candidate.confidence,
                status="active",
                volatility=candidate.volatility,
                trace_id=trace_id or self.trace_id,
                sources=refs,
            )
            return IngestResult(note=note, action="created", similarity=similarity)
        merged = self._merge(match, candidate, similarity, refs, trace_id=trace_id)
        return merged

    # ---- 查重 ----------------------------------------------------------

    def _best_match(
        self, text: str, entities: Sequence[str], existing: Sequence[Note]
    ) -> tuple[Note | None, float]:
        """挑出最相似的 active 笔记；不满足阈值或实体重叠要求则返回 (None, 最高分)。"""
        if not existing:
            return None, 0.0
        try:
            vectors = self.embedding.embed([text, *(note.body.strip() for note in existing)])
        except Exception as exc:  # noqa: BLE001 -- 嵌入失败降级为"这轮不去重"，笔记照写
            self.degradations.append(f"嵌入调用失败，跳过查重：{type(exc).__name__}: {exc}")
            return None, 0.0
        query_vec = vectors[0]
        keys = entity_keys(entities)
        best: Note | None = None
        best_similarity = 0.0
        for note, vector in zip(existing, vectors[1:], strict=True):
            similarity = cosine(query_vec, vector)
            if similarity < self.similarity_threshold:
                continue
            if len(keys & entity_keys(note.entities)) < self.entity_overlap_min:
                continue
            if best is None or similarity > best_similarity or (
                similarity == best_similarity and note.id < best.id
            ):
                best, best_similarity = note, similarity
        return best, best_similarity

    # ---- 合并 ----------------------------------------------------------

    def _merge(
        self,
        canonical: Note,
        candidate: CandidateNote,
        similarity: float,
        refs: Sequence[SourceRef],
        *,
        trace_id: str,
    ) -> IngestResult:
        """把候选并入规范笔记，并写一条 merged 留痕记录（新编号）。"""
        extra: dict[str, Any] = dict(canonical.meta.extra)
        body = self._merged_body(canonical.body, candidate.text)
        # 留痕先落盘：候选的原始表述 + 来源进 merged 记录（旧 ID 永不消失）
        trail_extra: dict[str, Any] = {
            "merged_into": canonical.id,
            "merged_similarity": round(similarity, 4),
        }
        trail = self.store.save_note(
            candidate.text,
            entities=list(candidate.entities),
            confidence=candidate.confidence,
            status="merged",
            redirect_to=canonical.id,
            volatility=candidate.volatility,
            trace_id=trace_id or self.trace_id,
            sources=list(refs),
            extra=trail_extra,
        )
        history = [str(item) for item in (extra.get("merged_from") or []) if str(item)]
        if trail.id not in history:
            history.append(trail.id)
        extra["merged_from"] = history
        note = self.store.save_note(
            body,
            note_id=canonical.id,
            title=canonical.title,
            entities=_merge_names(canonical.entities, candidate.entities),
            confidence=_stronger_confidence(canonical.meta.confidence, candidate.confidence),
            status="active",
            volatility=_more_volatile(canonical.meta.volatility, candidate.volatility),
            observed_at=canonical.meta.observed_at,
            created=canonical.meta.created,
            trace_id=canonical.meta.trace_id or trace_id or self.trace_id,
            sources=_merge_refs(canonical.meta.sources, refs),
            extra=extra,
        )
        return IngestResult(
            note=note,
            action="merged",
            merged_into=note.id,
            merged_from=trail.id,
            similarity=similarity,
        )

    def _merged_body(self, canonical_body: str, candidate_text: str) -> str:
        """合并后的正文：有 provider 就请 LLM 重写一句，否则保留既有表述。"""
        if self.provider is None:
            return canonical_body
        raw = call_text(
            self.provider,
            system=MERGE_SYSTEM,
            user=f"第一条：{canonical_body.strip()}\n第二条：{candidate_text.strip()}",
            step="ingest:merge",
            accountant=self.accountant,
            trace_id=self.trace_id,
            clock=self.clock,
            on_usage=self.on_usage,
        )
        merged = raw.strip().strip('"').strip()
        if not merged or len(merged) > MAX_MERGED_BODY_CHARS or "\n" in merged:
            self.degradations.append("合并正文不可用，保留既有表述")
            return canonical_body
        self.merged_bodies += 1
        return merged


def _merge_names(left: Sequence[str], right: Sequence[str]) -> list[str]:
    """实体名并集：按 slugify 归一（与查重同一套身份定义）去重，保序、左侧优先。

    "GLM-5.3" 与 "glm-5-3" 视为同一实体，只保留既有写法（避免同实体在 frontmatter
    里堆积多种写法——实体注册表才是写法的权威）。
    """
    out: list[str] = []
    seen: set[str] = set()
    for name in [*left, *right]:
        key = slugify(str(name)).casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(str(name).strip())
    return out


def ingest_candidates(
    ingestor: Ingestor, candidates: Sequence[CandidateNote], *, trace_id: str = ""
) -> list[IngestResult]:
    """批量入库的便捷入口（保序）；空正文候选直接跳过，不抛异常。"""
    results: list[IngestResult] = []
    for candidate in candidates:
        if not str(candidate.text or "").strip():
            continue
        results.append(ingestor.add(candidate, trace_id=trace_id))
    return results


__all__ = [
    "DEFAULT_ENTITY_OVERLAP",
    "DEFAULT_SIMILARITY",
    "MERGE_SYSTEM",
    "DedupSettings",
    "IngestResult",
    "Ingestor",
    "cosine",
    "dedup_settings",
    "entity_keys",
    "ingest_candidates",
]
