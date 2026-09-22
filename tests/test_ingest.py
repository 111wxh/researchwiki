"""入库去重单测：FixedEmbedding 精确控制余弦、ScriptedProvider 控制合并正文、tmp_path 隔离。

覆盖：未命中新建（编号续接）、命中合并（规范 ID 不变 + merged 留痕 + redirect 跟随）、
阈值边界、实体重叠门槛、LLM 重写正文与降级、嵌入失败降级、config 阈值解析。

P2-F 修复轮 I-1 追加：墓碑不入查重全集、不被合并清标记（`meta.replace` 透传纪律）。
"""

import math
from pathlib import Path

import pytest

from researchwiki.llm.provider import ScriptedProvider, StreamEvent
from researchwiki.wiki.distiller import CandidateNote
from researchwiki.wiki.frontmatter import NoteMeta, SourceRef
from researchwiki.wiki.ingest import (
    DedupSettings,
    Ingestor,
    cosine,
    dedup_settings,
    entity_keys,
    ingest_candidates,
)
from researchwiki.wiki.store import WikiStore

BASE = "GLM-5.3 支持 1M 上下文窗口"
REPHRASED = "GLM-5.3 的上下文窗口为 1M tokens"
UNRELATED = "向量检索用 sqlite-vec"


def unit(cos_angle: float) -> list[float]:
    """与 [1, 0] 余弦恰为 cos_angle 的二维单位向量。"""
    return [cos_angle, math.sqrt(max(0.0, 1 - cos_angle * cos_angle))]


class FixedEmbedding:
    """测试替身：查表返回预置向量（未登记的文本直接抛错，测试里不允许静默跑偏）。"""

    model = "fixed"
    dim = 2

    def __init__(self, table: dict[str, list[float]]) -> None:
        self.table = table

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(self.table[text]) for text in texts]


class BrokenEmbedding:
    model = "broken"

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embedding backend down")


def text_turn(text: str) -> list[StreamEvent]:
    return [StreamEvent(type="text_delta", delta=text)]


@pytest.fixture()
def store(tmp_path: Path) -> WikiStore:
    return WikiStore(tmp_path / "wiki-data")


# ---- 新建 --------------------------------------------------------------------


def test_add_creates_note_and_continues_numbering(store: WikiStore):
    store.save_note("既有笔记", entities=["实体A"], sources=[])  # N-0001
    ingestor = Ingestor(
        store, embedding=FixedEmbedding({"既有笔记": [0.0, 1.0], "全新断言": [1.0, 0.0]})
    )
    result = ingestor.add(CandidateNote(text="全新断言", entities=["实体B"], confidence="high"))

    assert result.action == "created" and result.merged_into is None
    assert result.note.id == "N-0002" and result.note.status == "active"
    assert result.note.confidence == "high"
    assert store.list_notes(status=None)[-1].id == "N-0002"


def test_add_rejects_empty_text(store: WikiStore):
    ingestor = Ingestor(store)
    with pytest.raises(ValueError, match="正文为空"):
        ingestor.add(CandidateNote(text="   "))


def test_embedding_failure_degrades_to_create(store: WikiStore):
    store.save_note(BASE, entities=["实体A"])
    ingestor = Ingestor(store, embedding=BrokenEmbedding())
    result = ingestor.add(CandidateNote(text=REPHRASED, entities=["实体A"]))
    # 嵌入不可用时宁可多写一条，也不能丢笔记
    assert result.action == "created"
    assert any("嵌入调用失败" in item for item in ingestor.degradations)


# ---- 命中合并 ----------------------------------------------------------------


def test_merge_keeps_canonical_id_and_writes_trail(store: WikiStore):
    store.save_note(
        BASE,
        entities=["GLM-5.3"],
        confidence="low",
        sources=[SourceRef(url="https://a", content_hash="a" * 64)],
    )
    ingestor = Ingestor(
        store, embedding=FixedEmbedding({BASE: [1.0, 0.0], REPHRASED: unit(0.95)})
    )
    result = ingestor.add(
        CandidateNote(
            text=REPHRASED,
            entities=["glm-5-3"],  # slug 归一后与 "GLM-5.3" 同一实体
            confidence="high",
            volatility="volatile",
            source_refs=[SourceRef(url="https://b", content_hash="b" * 64)],
        )
    )

    # 返回的 note 是规范 ID（引用一律指向它）
    assert result.action == "merged"
    assert result.note.id == "N-0001" and result.merged_into == "N-0001"
    assert result.merged_from == "N-0002"
    assert result.similarity == pytest.approx(0.95)

    canonical = store.get_note("N-0001")
    assert canonical.status == "active"
    assert canonical.body == BASE  # 简化合并：正文保留原文，新表述进留痕记录
    assert [s.url for s in canonical.meta.sources] == ["https://a", "https://b"]
    assert canonical.meta.extra["merged_from"] == ["N-0002"]
    assert canonical.confidence == "high"  # 取更强置信度
    assert canonical.volatility == "volatile"  # 取更易变档位（保守：宁可多刷新）
    assert canonical.entities == ["GLM-5.3"]  # 实体并集，写法保留既有
    assert canonical.meta.created == store.get_note("N-0001").meta.created

    trail = store.get_note("N-0002")
    assert trail.status == "merged" and trail.meta.redirect_to == "N-0001"
    assert trail.body == REPHRASED and trail.entities == ["glm-5-3"]
    assert trail.meta.extra["merged_into"] == "N-0001"
    assert trail.meta.extra["merged_similarity"] == pytest.approx(0.95)
    assert store.follow_redirect("N-0002").id == "N-0001"


def test_merge_accumulates_multiple_trails(store: WikiStore):
    store.save_note(BASE, entities=["实体A"])
    table = {BASE: [1.0, 0.0], REPHRASED: unit(0.95), "GLM-5.3 窗口 1M（又一次改写）": unit(0.93)}
    ingestor = Ingestor(store, embedding=FixedEmbedding(table))
    ingestor.add(CandidateNote(text=REPHRASED, entities=["实体A"]))
    second = ingestor.add(CandidateNote(text="GLM-5.3 窗口 1M（又一次改写）", entities=["实体A"]))

    assert second.note.id == "N-0001" and second.merged_from == "N-0003"
    assert second.note.meta.extra["merged_from"] == ["N-0002", "N-0003"]
    assert [n.id for n in store.list_notes()] == ["N-0001"]  # active 只有规范笔记


def test_merge_body_rewritten_when_provider_injected(store: WikiStore):
    store.save_note(BASE, entities=["实体A"])
    provider = ScriptedProvider([text_turn("GLM-5.3 支持 1M 上下文窗口（多来源印证）")])
    ingestor = Ingestor(
        store,
        embedding=FixedEmbedding({BASE: [1.0, 0.0], REPHRASED: unit(0.95)}),
        provider=provider,
    )
    result = ingestor.add(CandidateNote(text=REPHRASED, entities=["实体A"]))

    assert result.note.body == "GLM-5.3 支持 1M 上下文窗口（多来源印证）"
    assert ingestor.merged_bodies == 1
    assert [len(call) for call in provider.calls] == [1]
    # 留痕仍保留候选原始表述
    assert store.get_note(result.merged_from).body == REPHRASED


def test_merge_body_keeps_original_when_provider_output_unusable(store: WikiStore):
    store.save_note(BASE, entities=["实体A"])
    provider = ScriptedProvider([text_turn("第一行\n第二行\n第三行")])
    ingestor = Ingestor(
        store,
        embedding=FixedEmbedding({BASE: [1.0, 0.0], REPHRASED: unit(0.95)}),
        provider=provider,
    )
    result = ingestor.add(CandidateNote(text=REPHRASED, entities=["实体A"]))
    assert result.note.body == BASE
    assert any("合并正文不可用" in item for item in ingestor.degradations)


# ---- 阈值与实体门槛 ----------------------------------------------------------


def test_similarity_threshold_boundary(store: WikiStore):
    store.save_note(BASE, entities=["实体A"])
    table = {BASE: [1.0, 0.0], "恰好等于阈值": unit(0.9), "略低于阈值": unit(0.89)}
    ingestor = Ingestor(store, embedding=FixedEmbedding(table))

    # 等于阈值算命中（≥ 阈值即判重）
    assert ingestor.add(CandidateNote(text="恰好等于阈值", entities=["实体A"])).action == "merged"
    # 低于阈值新建
    assert ingestor.add(CandidateNote(text="略低于阈值", entities=["实体A"])).action == "created"
    # 阈值可下调到 0.85 → 0.89 的候选改为合并
    lenient = Ingestor(store, embedding=FixedEmbedding(table), similarity_threshold=0.85)
    assert lenient.add(CandidateNote(text="略低于阈值", entities=["实体A"])).action == "merged"


def test_entity_overlap_required_by_default(store: WikiStore):
    store.save_note(BASE, entities=["实体A"])
    table = {BASE: [1.0, 0.0], REPHRASED: [1.0, 0.0], UNRELATED: [0.0, 1.0]}
    ingestor = Ingestor(store, embedding=FixedEmbedding(table))

    # 余弦 1.0 但实体无重叠 → 默认不合并（避免"只是说法像"被误并）
    assert ingestor.add(CandidateNote(text=REPHRASED, entities=["实体B"])).action == "created"
    # 余弦为 0 → 即使实体相同也不合并
    assert ingestor.add(CandidateNote(text=UNRELATED, entities=["实体A"])).action == "created"
    # dedup_entity_overlap=0 → 允许纯向量判重
    vector_only = Ingestor(store, embedding=FixedEmbedding(table), entity_overlap_min=0)
    assert vector_only.add(CandidateNote(text=REPHRASED, entities=["实体C"])).action == "merged"


def test_missing_existing_note_body_still_matches(store: WikiStore):
    """既有笔记正文为空（历史轻量笔记）时不该炸：向量比对降级为不相似。"""
    store.save_note("", entities=["实体A"])
    ingestor = Ingestor(store, embedding=FixedEmbedding({"": [0.0, 1.0], REPHRASED: [1.0, 0.0]}))
    assert ingestor.add(CandidateNote(text=REPHRASED, entities=["实体A"])).action == "created"


# ---- 墓碑：不入查重全集、不被合并清标记（P2-F 修复轮 I-1）----------------------


def _tombstone_meta(**overrides) -> NoteMeta:
    """一条"全字段非默认"的 NoteMeta：任何字段被静默重置都会被对比断言抓住。"""
    fields = {
        "id": "N-0001",
        "title": "上下文压缩（旧）",
        "entities": ["GLM-5.3"],
        "confidence": "high",
        "status": "active",
        "volatility": "drifting",
        "kind": "experience",
        "importance": 0.7,
        "tombstone": True,
        "observed_at": "2026-05-01T00:00:00+00:00",
        "reviewed_at": "2026-05-02T00:00:00+00:00",
        "valid_from": "2026-05-01T00:00:00+00:00",
        "valid_until": "2026-06-01T00:00:00+00:00",
        "source_changed_at": "2026-05-03T00:00:00+00:00",
        "created": "2026-04-30T00:00:00+00:00",
        "trace_id": "run-7",
        "sources": [SourceRef(url="https://a", content_hash="a" * 64)],
        "extra": {"keep": "me"},
    }
    fields.update(overrides)
    return NoteMeta(**fields)


def test_tombstone_excluded_from_dedup_candidates(store: WikiStore):
    """候选与墓碑正文高度相似（余弦 1.0 + 实体重叠）也不合并：墓碑不参与查重。"""
    meta = _tombstone_meta(tombstone=True)
    store.save_meta(meta, BASE)
    ingestor = Ingestor(
        store, embedding=FixedEmbedding({BASE: [1.0, 0.0], REPHRASED: [1.0, 0.0]})
    )
    result = ingestor.add(CandidateNote(text=REPHRASED, entities=["GLM-5.3"]))

    # 不命中墓碑 → 新建（而不是把候选并进失效裁定里）
    assert result.action == "created"
    assert result.note.id == "N-0002"
    # 墓碑原样不动：标记在、正文在、没有任何 merged 留痕
    tombstone = store.get_note("N-0001")
    assert tombstone.tombstone is True
    assert tombstone.body == BASE
    assert tombstone.meta.extra == {"keep": "me"}


def test_merge_does_not_lose_tombstone_or_semantic_fields(store: WikiStore):
    """即便合并命中（此处绕过查重直接调 _merge），标记与其余字段也不能被清掉。

    这是修复轮 I-1 的"第二道防线"：``_merge`` 用 ``meta.replace`` 只点名要改的
    字段，其余逐字段保留——旧写法逐参数透传 ``save_note`` 时，kind / importance /
    tombstone 会被默认值静默覆盖。
    """
    meta = _tombstone_meta(tombstone=True)
    canonical = store.save_meta(meta, BASE)
    ingestor = Ingestor(store, embedding=FixedEmbedding({BASE: [1.0, 0.0]}))
    result = ingestor._merge(  # noqa: SLF001 -- 直接驱动"命中合并"分支（查重已单测覆盖）
        canonical,
        CandidateNote(
            text=REPHRASED,
            entities=["glm-5-3"],
            confidence="low",
            volatility="volatile",
            source_refs=[SourceRef(url="https://b", content_hash="b" * 64)],
        ),
        similarity=0.95,
        refs=[SourceRef(url="https://b", content_hash="b" * 64)],
        trace_id="run-9",
    )

    merged = result.note
    assert merged.id == "N-0001"
    # 预期变更的字段
    assert [s.url for s in merged.meta.sources] == ["https://a", "https://b"]
    assert merged.confidence == "high"  # 取更强
    assert merged.volatility == "volatile"  # 取更易变
    assert merged.meta.extra["merged_from"] == ["N-0002"]
    assert merged.meta.trace_id == "run-7"  # 既有 trace_id 优先
    # 除预期变更外**逐字段不变**（含 tombstone：漏传会把它变回可召回的正常笔记）
    expected = meta.replace(
        entities=["GLM-5.3"],  # 实体并集按 slugify 归一，保留既有写法
        confidence="high",
        volatility="volatile",
        sources=[SourceRef(url="https://a", content_hash="a" * 64),
                 SourceRef(url="https://b", content_hash="b" * 64)],
        extra={"keep": "me", "merged_from": ["N-0002"]},
    )
    assert merged.meta == expected
    assert merged.meta.tombstone is True


def test_created_path_keeps_candidate_fields(store: WikiStore):
    """新建路径不受修复影响：候选字段照旧写入（回归护栏）。"""
    ingestor = Ingestor(store, embedding=FixedEmbedding({"全新断言": [1.0, 0.0]}))
    result = ingestor.add(CandidateNote(text="全新断言", entities=["实体B"], confidence="high"))
    assert result.action == "created" and result.note.tombstone is False


# ---- 批量与配置 --------------------------------------------------------------


def test_ingest_candidates_batch_skips_empty(store: WikiStore):
    ingestor = Ingestor(store)  # 默认 MockEmbedding：同文本余弦 1.0
    results = ingest_candidates(
        ingestor,
        [
            CandidateNote(text="同一条断言", entities=["实体A"]),
            CandidateNote(text="   ", entities=["实体A"]),
            CandidateNote(text="同一条断言", entities=["实体A"]),
        ],
        trace_id="run-9",
    )
    assert [result.action for result in results] == ["created", "merged"]
    assert {result.note.id for result in results} == {"N-0001"}
    assert results[0].note.meta.trace_id == "run-9"


def test_dedup_settings_from_config():
    assert dedup_settings(None) == DedupSettings()
    assert dedup_settings({}) == DedupSettings()
    config = {"wiki": {"dedup_similarity": 0.8, "dedup_entity_overlap": 2}}
    assert dedup_settings(config) == DedupSettings(similarity=0.8, entity_overlap_min=2)
    # 直接给 [wiki] 段也接受
    assert dedup_settings({"dedup_similarity": 0.75}) == DedupSettings(similarity=0.75)
    # 非法值回默认；相似度夹紧到 [0, 1]
    assert dedup_settings({"wiki": {"dedup_similarity": "bad", "dedup_entity_overlap": None}}) == (
        DedupSettings()
    )
    assert dedup_settings({"wiki": {"dedup_similarity": 1.5}}) == DedupSettings(similarity=1.0)
    assert dedup_settings({"wiki": {"dedup_entity_overlap": -3}}).entity_overlap_min == 0


def test_cosine_and_entity_keys_helpers():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine([1.0, 0.0], [1.0, 0.0, 0.0]) == 0.0  # 维度不一致
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0  # 零向量
    assert cosine([], []) == 0.0
    # 各种写法（大小写/空白/点与横线）归一成同一个稳定 ID，空串被丢弃
    assert entity_keys(["GLM-5.3", " glm-5-3 ", "", "GLM 5.3"]) == {"glm-5-3"}


def test_config_toml_documents_dedup_defaults():
    import tomllib

    from researchwiki.wiki.ingest import DEFAULT_ENTITY_OVERLAP, DEFAULT_SIMILARITY

    root = Path(__file__).resolve().parent.parent
    with open(root / "config.toml", "rb") as handle:
        config = tomllib.load(handle)
    assert config["wiki"]["dedup_similarity"] == DEFAULT_SIMILARITY
    assert config["wiki"]["dedup_entity_overlap"] == DEFAULT_ENTITY_OVERLAP
    assert dedup_settings(config) == DedupSettings()
