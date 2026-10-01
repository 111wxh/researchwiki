"""评测题集 loader 与 schema 校验测试（评测支线 task-1）。

覆盖 task-1 简报的三块契约：

- QaItem frozen dataclass：qid/question/qtype/gold_points/entities/notes=""；
- load_qa 逐行校验：qid 唯一、qtype 白名单、gold_points 非空（unanswerable 必须为空）、
  entities 非空、缺字段报错、多余 JSON 键忽略、坏行 ValueError 带物理行号；
- qa_type_counts 题型计数；
- 真实题集 evals/qa/ai-frameworks.jsonl：30 题、占比 {12, 9, 5, 4}、
  5 道时效题全部引用真实存在的 v1/v2 语料对；
- 真实语料 evals/fixtures/ai-frameworks：manifest 自洽、每篇 400–700 字、
  首行 `# 标题`、3 对时效文档文件成对存在。

全部零网络、零模型调用。
"""

import json
from pathlib import Path

import pytest

from researchwiki.evals import QaItem, qa_type_counts
from researchwiki.evals.qa import load_qa

REPO_ROOT = Path(__file__).resolve().parents[1]
EVALS_DIR = REPO_ROOT / "evals"
QA_PATH = EVALS_DIR / "qa" / "ai-frameworks.jsonl"
FIXTURE_DIR = EVALS_DIR / "fixtures" / "ai-frameworks"
DOCS_DIR = FIXTURE_DIR / "docs"

#: MVP 占比硬口径：30 题 = 12 单跳 + 9 多跳 + 5 时效 + 4 无答案
EXPECTED_COUNTS = {"single_hop": 12, "multi_hop": 9, "temporal": 5, "unanswerable": 4}


def valid_line(**overrides: object) -> str:
    """构造一行合法 JSONL（可按字段覆盖），供校验器单测使用。"""
    base: dict[str, object] = {
        "qid": "Q001",
        "question": "LangChain 的默认重试次数是多少？",
        "qtype": "single_hop",
        "gold_points": ["默认重试 3 次", "通过 RunnableRetry 实现"],
        "entities": ["LangChain"],
        "notes": "",
    }
    base.update(overrides)
    return json.dumps(base, ensure_ascii=False)


def write_qa(tmp_path: Path, *lines: str) -> Path:
    path = tmp_path / "qa.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# QaItem 数据类
# ---------------------------------------------------------------------------


def test_qa_item_is_frozen_and_defaults() -> None:
    """frozen：禁止改字段；notes 缺省为空串。"""
    item = QaItem(qid="Q001", question="q", qtype="single_hop", gold_points=["g"], entities=["e"])
    assert item.notes == ""
    with pytest.raises(Exception):
        item.qid = "Q002"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# load_qa schema 校验（临时 jsonl）
# ---------------------------------------------------------------------------


def test_load_qa_valid_minimal(tmp_path: Path) -> None:
    """合法最小行（notes 省略）可加载，多余 JSON 键被忽略。"""
    line = json.dumps(
        {
            "qid": "Q001",
            "question": "问题？",
            "qtype": "temporal",
            "gold_points": ["要点一", "要点二"],
            "entities": ["实体"],
            "extra_key": "多余键应被忽略",
        },
        ensure_ascii=False,
    )
    items = load_qa(write_qa(tmp_path, line))
    assert len(items) == 1
    assert items[0].qid == "Q001"
    assert items[0].notes == ""
    assert items[0].gold_points == ["要点一", "要点二"]


def test_load_qa_skips_blank_lines_and_counts_physical_lineno(tmp_path: Path) -> None:
    """空行跳过；坏行报错用物理行号（含空行计数）。"""
    path = write_qa(tmp_path, valid_line(), "", "{not json", valid_line(qid="Q002"))
    with pytest.raises(ValueError, match="第 3 行"):
        load_qa(path)


def test_load_qa_rejects_duplicate_qid(tmp_path: Path) -> None:
    """qid 重复报错，且指出首次出现行。"""
    path = write_qa(tmp_path, valid_line(), valid_line())
    with pytest.raises(ValueError, match="qid 重复") as exc:
        load_qa(path)
    assert "第 2 行" in str(exc.value)


def test_load_qa_rejects_unknown_qtype(tmp_path: Path) -> None:
    """qtype 不在白名单报错。"""
    path = write_qa(tmp_path, valid_line(qtype="tricky"))
    with pytest.raises(ValueError, match="第 1 行"):
        load_qa(path)


@pytest.mark.parametrize("bad", ["", ["仅一条但整体不是要点列表也行", 2], "要点", []])
def test_load_qa_rejects_bad_gold_points(tmp_path: Path, bad: object) -> None:
    """gold_points 必须是非空字符串列表；单跳题传空列表也报错。"""
    path = write_qa(tmp_path, valid_line(gold_points=bad))
    with pytest.raises(ValueError, match="gold_points"):
        load_qa(path)


def test_load_qa_unanswerable_requires_empty_gold_points(tmp_path: Path) -> None:
    """无答案题 gold_points 必须为空数组（有内容视为坏行）。"""
    ok = write_qa(tmp_path, valid_line(qtype="unanswerable", gold_points=[]))
    assert load_qa(ok)[0].qtype == "unanswerable"
    bad = write_qa(tmp_path, valid_line(qtype="unanswerable", gold_points=["居然有要点"]))
    with pytest.raises(ValueError, match="unanswerable"):
        load_qa(bad)


@pytest.mark.parametrize("missing", ["qid", "question", "qtype", "gold_points", "entities"])
def test_load_qa_rejects_missing_field(tmp_path: Path, missing: str) -> None:
    """缺任一必填字段报错并带行号。"""
    base = {
        "qid": "Q001",
        "question": "问题？",
        "qtype": "single_hop",
        "gold_points": ["要点"],
        "entities": ["实体"],
    }
    base.pop(missing)
    path = write_qa(tmp_path, json.dumps(base, ensure_ascii=False))
    with pytest.raises(ValueError, match=missing):
        load_qa(path)


@pytest.mark.parametrize(
    "overrides",
    [{"entities": []}, {"entities": "实体"}, {"qid": ""}, {"question": "  "}, {"notes": 3}],
)
def test_load_qa_rejects_bad_field_values(tmp_path: Path, overrides: dict) -> None:
    """entities 非空列表、qid/question 非空、notes 必须是字符串。"""
    path = write_qa(tmp_path, valid_line(**overrides))
    with pytest.raises(ValueError, match="第 1 行"):
        load_qa(path)


def test_qa_type_counts() -> None:
    """qa_type_counts 按题型计数，四种题型键齐全。"""
    items = [
        QaItem("Q1", "q", "single_hop", ["g"], ["e"]),
        QaItem("Q2", "q", "single_hop", ["g"], ["e"]),
        QaItem("Q3", "q", "temporal", ["g"], ["e"]),
    ]
    counts = qa_type_counts(items)
    assert counts == {"single_hop": 2, "multi_hop": 0, "temporal": 1, "unanswerable": 0}


# ---------------------------------------------------------------------------
# 真实题集：30 题、占比、时效题引用真实 v1/v2 对
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qa_items() -> list:
    return load_qa(QA_PATH)


def test_real_qa_has_30_items_with_expected_ids(qa_items) -> None:
    """30 题、qid 唯一且为 Q001–Q030 连续编号。"""
    assert len(qa_items) == 30
    assert [item.qid for item in qa_items] == [f"Q{i:03d}" for i in range(1, 31)]


def test_real_qa_type_distribution(qa_items) -> None:
    """占比硬口径：{12, 9, 5, 4} ± 0。"""
    assert qa_type_counts(qa_items) == EXPECTED_COUNTS


def test_real_qa_gold_points_and_entities_wellformed(qa_items) -> None:
    """有答案题 2–4 条要点；无答案题空要点 + 注明无答案；各题 entities 非空。"""
    for item in qa_items:
        assert item.entities, item.qid
        if item.qtype == "unanswerable":
            assert item.gold_points == []
            assert "无答案题" in item.notes
        else:
            assert 2 <= len(item.gold_points) <= 4, item.qid


def test_real_qa_multi_hop_chains_entities(qa_items) -> None:
    """多跳题 entities 至少 2 个（跨文档串联）。"""
    multi = [item for item in qa_items if item.qtype == "multi_hop"]
    assert len(multi) == 9
    for item in multi:
        assert len(item.entities) >= 2, item.qid


def test_real_qa_temporal_references_existing_pairs(qa_items) -> None:
    """5 道时效题都引用真实存在的 v1/v2 文档对，且 3 对全覆盖；gold 按 v2 口径。"""
    pair_bases = {
        p.name.removesuffix("@v2.md")
        for p in DOCS_DIR.glob("*@v2.md")
    }
    assert pair_bases == {"langchain-core", "openai-python-sdk", "milvus"}
    referenced: set[str] = set()
    for item in (q for q in qa_items if q.qtype == "temporal"):
        # notes 约定格式："时效题：依赖 <base> 的 v1/v2 对，gold 按 v2 口径。..."
        assert item.notes.startswith("时效题：依赖 "), item.qid
        base = item.notes.split("：依赖 ", 1)[1].split(" 的 v1/v2", 1)[0]
        assert base in pair_bases, item.qid
        referenced.add(base)
        for suffix in (".md", "@v2.md"):
            assert (DOCS_DIR / f"{base}{suffix}").exists(), (item.qid, base)
    assert referenced == pair_bases


# ---------------------------------------------------------------------------
# 真实语料：manifest 自洽、篇幅、标题行、时效对成对
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads((FIXTURE_DIR / "manifest.json").read_text(encoding="utf-8"))


def test_manifest_declares_fixture_corpus(manifest) -> None:
    """口径元数据：受控合成语料、provider 记录为 fixture-corpus。"""
    assert manifest["domain"] == "ai-frameworks"
    assert "fixture-corpus" in manifest["provider_note"]


def test_manifest_docs_are_self_consistent(manifest) -> None:
    """21 条目（18 基础 + 3 v2）：url 指向存在的 md 文件，version 标记正确。"""
    docs = manifest["docs"]
    assert len(docs) == 21
    doc_ids = [d["doc_id"] for d in docs]
    assert len(doc_ids) == len(set(doc_ids))
    v2_ids = {d["doc_id"] for d in docs if d["version"] == "v2"}
    assert len(v2_ids) == 3
    for doc in docs:
        assert doc["url"] == f"fixture://ai-frameworks/{doc['doc_id']}"
        assert doc["title"]
        path = DOCS_DIR / f"{doc['doc_id']}.md"
        assert path.exists(), doc["doc_id"]
        first_line = path.read_text(encoding="utf-8").splitlines()[0]
        assert first_line.startswith("# "), doc["doc_id"]
    # 每个 v2 条目都有对应 v1 基础文档
    for doc_id in v2_ids:
        base = doc_id.removesuffix("@v2")
        assert f"{base}@v2" in doc_id  # 命名约定自检
        entry = next(d for d in docs if d["doc_id"] == base)
        assert entry["version"] == "v1"


def test_docs_length_and_revision_note(manifest) -> None:
    """每篇 400–700 字（去空白计数）；v2 文末有修订说明行。"""
    for doc in manifest["docs"]:
        text = (DOCS_DIR / f"{doc['doc_id']}.md").read_text(encoding="utf-8")
        n_chars = len("".join(text.split()))
        assert 400 <= n_chars <= 700, (doc["doc_id"], n_chars)
        if doc["version"] == "v2":
            assert "修订说明：" in text.splitlines()[-1], doc["doc_id"]


def test_docs_cover_four_topics_groups(manifest) -> None:
    """主题分布：LangChain 系 6、OpenAI SDK 系 5、向量库/嵌入 5、跨域综述 2（按 v1 计）。"""
    groups = {
        "langchain": {"langchain-core", "langchain-agents", "langchain-memory",
                      "langchain-retrievers", "langgraph", "langchain-versions"},
        "openai": {"openai-python-sdk", "openai-assistants", "openai-structured-outputs",
                   "openai-embeddings-api", "openai-versions"},
        "vector": {"faiss", "chroma", "milvus", "embedding-models", "rag-architecture"},
        "cross": {"framework-comparison", "memory-systems"},
    }
    v1_ids = {d["doc_id"] for d in manifest["docs"] if d["version"] == "v1"}
    assert len(v1_ids) == 18
    for name, expected in groups.items():
        assert expected <= v1_ids, name
    assert sum(len(g) for g in groups.values()) == 18
