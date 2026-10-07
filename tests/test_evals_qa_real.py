"""真实层题集与冻结语料测试（评测支线第二层 task-1）。

被测交付物（真实网络语料，非受控合成）：

- evals/qa/ai-frameworks-real.jsonl：30 题中文 QA（四题型）；
- evals/fixtures/ai-frameworks-real/manifest.json：冻结语料清单
  （langchain-ai/langchain @ 两个 pinned SHA，含冻结永久链接与许可证）；
- evals/fixtures/ai-frameworks-real/docs/*.md：英文原文语料（含 3 对同名 v1/v2 文档）。

覆盖契约：

- schema 逐字复用 load_qa：qid 唯一、qtype 白名单、gold_points 口径（无答案题空）；
- 题型分布硬口径：single_hop 10–12 / multi_hop 8–10 / temporal 4–6 / unanswerable 4–6，共 30；
- 时效题引用的 v1/v2 文档对在 manifest 与磁盘上真实存在（3 对全覆盖）；
- unanswerable 题：notes 记录"缺席词面"，逐词确认语料零命中（词面缺席抽查）；
- 每题 gold 可判定性 sanity：normalize_answer + point_hit 下至少一条 gold_point
  能命中其对应版本文档原文（temporal 对 @v2 文档、其余按 notes 来源 doc）。

全部零网络、零模型调用——数据在仓库内冻结，可复现。
"""

import json
import re
from pathlib import Path

import pytest

from researchwiki.evals.metrics import normalize_answer, point_hit
from researchwiki.evals.qa import load_qa, qa_type_counts

REPO_ROOT = Path(__file__).resolve().parents[1]
EVALS_DIR = REPO_ROOT / "evals"
QA_PATH = EVALS_DIR / "qa" / "ai-frameworks-real.jsonl"
FIXTURE_DIR = EVALS_DIR / "fixtures" / "ai-frameworks-real"
DOCS_DIR = FIXTURE_DIR / "docs"
MANIFEST_PATH = FIXTURE_DIR / "manifest.json"

#: 题量硬口径：30 题，四题型区间（简报：single 10–12 / multi 8–10 / temporal 5±1 / 无答案 4–6）
TOTAL_QUESTIONS = 30
EXPECTED_RANGES = {
    "single_hop": (10, 12),
    "multi_hop": (8, 10),
    "temporal": (4, 6),
    "unanswerable": (4, 6),
}

#: 冻结语料来源仓库与两个 pinned SHA（manifest sources 必须与此一致）
REPO = "langchain-ai/langchain"
SHA_A = "4cf62a51a8849d4baea15071c5b0e10bf7ea31c8"  # langchain==0.3.30
SHA_B = "4a65e827f7d7fd8139a4232f408a005f704dc71b"  # langchain==1.0.0


def doc_text(doc_id: str) -> str:
    """按 doc_id 读语料正文（docs/<doc_id>.md）。"""
    return (DOCS_DIR / f"{doc_id}.md").read_text(encoding="utf-8")


def sources_of(notes: str) -> list[str]:
    """从 notes 的「来源：」段解析 doc_id 列表（顿号分隔）。"""
    match = re.search(r"来源：([^\n]+)", notes)
    if not match:
        return []
    return [part.strip() for part in match.group(1).split("、") if part.strip()]


def absent_tokens_of(notes: str) -> list[str]:
    """从无答案题 notes 的「缺席词面：」段解析词面列表（顿号分隔）。"""
    match = re.search(r"缺席词面：([^\n]+)", notes)
    if not match:
        return []
    return [part.strip() for part in match.group(1).split("、") if part.strip()]


def union_text(doc_ids: list[str]) -> str:
    """多篇语料合并正文（多跳题跨文档命中用）。"""
    return "\n".join(doc_text(doc_id) for doc_id in doc_ids)


# ---------------------------------------------------------------------------
# 题集：schema、题量、qid、题型分布
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qa_items() -> list:
    return load_qa(QA_PATH)


def test_real_qa_has_30_items(qa_items) -> None:
    """共 30 题，qid 为 RQ001–RQ030 连续编号且唯一（load_qa 已保证唯一）。"""
    assert len(qa_items) == TOTAL_QUESTIONS
    assert [item.qid for item in qa_items] == [f"RQ{i:03d}" for i in range(1, 31)]


def test_real_qa_type_distribution(qa_items) -> None:
    """四题型分布落在简报区间内，总数对齐。"""
    counts = qa_type_counts(qa_items)
    for qtype, (low, high) in EXPECTED_RANGES.items():
        assert low <= counts[qtype] <= high, (qtype, counts[qtype])
    assert sum(counts.values()) == TOTAL_QUESTIONS


def test_real_qa_wellformed_fields(qa_items) -> None:
    """有答案题 2–4 条要点；无答案题空要点 + 注明无答案；全部标注来源。"""
    for item in qa_items:
        assert item.entities, item.qid
        if item.qtype == "unanswerable":
            # 无答案题：空要点 + 注明无答案 + 给出缺席词面（不标来源文档）。
            assert item.gold_points == [], item.qid
            assert "无答案题" in item.notes, item.qid
            assert absent_tokens_of(item.notes), item.qid
        else:
            # 有答案题：标注来源文档（时效题可用约定的时效前缀标注）。
            assert sources_of(item.notes) or "时效题" in item.notes, item.qid
            assert 2 <= len(item.gold_points) <= 4, item.qid


def test_real_qa_questions_are_chinese(qa_items) -> None:
    """问题为中文（含至少一个 CJK 字符）——语料英文原文、问题中文的显式口径。"""
    cjk = re.compile(r"[\u4e00-\u9fff]")
    for item in qa_items:
        assert cjk.search(item.question), item.qid


def test_real_qa_multi_hop_chains_entities(qa_items) -> None:
    """多跳题 entities 至少 2 个（跨文档串联实体）。"""
    multi = [item for item in qa_items if item.qtype == "multi_hop"]
    assert EXPECTED_RANGES["multi_hop"][0] <= len(multi)
    for item in multi:
        assert len(item.entities) >= 2, item.qid


# ---------------------------------------------------------------------------
# 语料 manifest：来源、冻结永久链接、时效对
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_manifest_declares_real_corpus(manifest) -> None:
    """domain 与 provider_note 声明 real-corpus 口径（真实来源冻结语料）。"""
    assert manifest["domain"] == "ai-frameworks-real"
    assert "real-corpus" in manifest["provider_note"]
    assert "中文" in manifest["provider_note"]


def test_manifest_sources_pin_two_shas(manifest) -> None:
    """sources 记录仓库、两个 pinned SHA 与 MIT 许可证。"""
    sources = manifest["sources"]
    assert len(sources) == 1
    source = sources[0]
    assert source["repo"] == REPO
    for key in ("sha_a", "sha_b"):
        assert re.fullmatch(r"[0-9a-f]{40}", source[key]), key
    assert source["sha_a"] == SHA_A
    assert source["sha_b"] == SHA_B
    assert source["license"] == "MIT"


def test_manifest_docs_frozen_permalinks(manifest) -> None:
    """每条 doc：url 是指向 pinned SHA 的冻结永久链接；磁盘上有同名 md。"""
    docs = manifest["docs"]
    assert 12 <= len(docs) <= 16
    doc_ids = [d["doc_id"] for d in docs]
    assert len(doc_ids) == len(set(doc_ids))
    prefix = f"https://github.com/{REPO}/blob/"
    for doc in docs:
        assert doc["url"].startswith(prefix), doc["doc_id"]
        sha = doc["url"].removeprefix(prefix).split("/", 1)[0]
        assert sha in (SHA_A, SHA_B), doc["doc_id"]
        assert doc["title"]
        assert doc["version"] in ("v1", "v2")
        path = DOCS_DIR / f"{doc['doc_id']}.md"
        assert path.exists(), doc["doc_id"]
        first_line = path.read_text(encoding="utf-8").splitlines()[0]
        assert first_line.startswith("# "), doc["doc_id"]


def test_docs_length_bounds(manifest) -> None:
    """每篇语料 300–900 字（去空白计数）——相关章节截取，与受控层口径一致。"""
    for doc in manifest["docs"]:
        text = doc_text(doc["doc_id"])
        n_chars = len("".join(text.split()))
        assert 300 <= n_chars <= 900, (doc["doc_id"], n_chars)


def test_temporal_pairs_exist_in_manifest(manifest) -> None:
    """3 对同名 v1/v2 文档成对存在：旧 <name>.md（v1）+ 新 <name>@v2.md（v2）。"""
    v2_ids = {d["doc_id"] for d in manifest["docs"] if d["version"] == "v2" and d["doc_id"].endswith("@v2")}
    assert v2_ids == {"root-readme@v2", "libs-langchain-readme@v2", "libs-core-readme@v2"}
    for v2_id in v2_ids:
        base = v2_id.removesuffix("@v2")
        base_entry = next(d for d in manifest["docs"] if d["doc_id"] == base)
        assert base_entry["version"] == "v1"
        # 同一路径的两个版本分别冻结在 SHA-A / SHA-B
        for doc in (base_entry, next(d for d in manifest["docs"] if d["doc_id"] == v2_id)):
            sha = doc["url"].split("/blob/")[1].split("/", 1)[0]
            assert sha == (SHA_A if doc["version"] == "v1" else SHA_B), doc["doc_id"]


# ---------------------------------------------------------------------------
# 逐题可判定性 sanity（gold 可在对应版本文档原文命中）
# ---------------------------------------------------------------------------


def test_single_and_multi_hop_gold_hits_source_docs(qa_items) -> None:
    """单跳/多跳题：至少一条 gold_point 在其来源文档（notes 标注）原文命中。"""
    checked = 0
    for item in qa_items:
        if item.qtype not in ("single_hop", "multi_hop"):
            continue
        doc_ids = sources_of(item.notes)
        assert doc_ids, item.qid
        for doc_id in doc_ids:
            assert (DOCS_DIR / f"{doc_id}.md").exists(), (item.qid, doc_id)
        text = union_text(doc_ids)
        assert any(point_hit(text, gold) for gold in item.gold_points), item.qid
        checked += 1
    assert checked >= 18


def test_temporal_gold_hits_v2_doc(qa_items) -> None:
    """时效题：gold 按 v2 口径，至少一条 gold_point 命中对应 @v2 文档原文。"""
    pair_bases = {"root-readme", "libs-langchain-readme", "libs-core-readme"}
    referenced: set[str] = set()
    for item in (q for q in qa_items if q.qtype == "temporal"):
        # notes 约定格式："时效题：依赖 <base> 的 v1/v2 对，gold 按 v2 口径。"
        assert item.notes.startswith("时效题：依赖 "), item.qid
        base = item.notes.split("：依赖 ", 1)[1].split(" 的 v1/v2", 1)[0]
        assert base in pair_bases, item.qid
        referenced.add(base)
        for suffix in (".md", "@v2.md"):
            assert (DOCS_DIR / f"{base}{suffix}").exists(), (item.qid, base)
        v2_text = doc_text(f"{base}@v2")
        assert any(point_hit(v2_text, gold) for gold in item.gold_points), item.qid
    assert referenced == pair_bases


# ---------------------------------------------------------------------------
# unanswerable：词面缺席抽查（语料零命中）
# ---------------------------------------------------------------------------


def test_unanswerable_tokens_absent_from_corpus(qa_items) -> None:
    """无答案题：notes 中的缺席词面在全部语料（归一化后）零命中。"""
    corpus = "\n".join(doc_text(d["doc_id"]) for d in manifest_docs())
    corpus_norm = normalize_answer(corpus)
    for item in (q for q in qa_items if q.qtype == "unanswerable"):
        tokens = absent_tokens_of(item.notes)
        assert tokens, item.qid
        for token in tokens:
            assert normalize_answer(token), item.qid
            assert normalize_answer(token) not in corpus_norm, (item.qid, token)


def manifest_docs() -> list:
    """读取 manifest 的 docs 列表（独立于 module fixture，便于逐文档扫描）。"""
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["docs"]
