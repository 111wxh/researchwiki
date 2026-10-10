"""序列实验事件流 schema 与校验器测试（复用序列实验 task-1）。

被测交付物：

- src/researchwiki/evals/sequence.py：SeqEvent / load_scenario / clusters_from_manifest；
- evals/qa/ai-frameworks-seq.jsonl：序列题集（簇问 12 + 时效 3 + 无答案 2 = 17 题）；
- evals/sequence/scenario.jsonl：事件流（t0 ingest → t1–t4 四簇查询块 → t5 update →
  t6 时效/无答案/更新后重复）。

覆盖契约：

- load_scenario 逐条校验规则（每条至少 1 个坏例，错误带物理行号）：
  query 的 qid ∈ 题集；qid 恰出现 1 或 2 次；出现两次必须"分居 update 两侧"
  （更新后重复对）或"同在 update 前"（簇内重复对），同在 update 后非法；
  update 恰 1 个且在全部时效题之前；ingest/update 的 doc_ids ⊆ 语料 manifest；
  query 题目来源文档（notes「来源：」段）必须在首个引用它的 query 之前 ingest
  或 update；event_id 从 0 连续递增；kind 白名单；query 不带 doc_ids；
  ingest/update 不带 qid；非无答案题必须有可解析的来源段；重复对两次事件
  note 必须写明重复对类型；
- SeqEvent：frozen dataclass，doc_ids 归一化为 tuple；
- clusters_from_manifest：15 篇语料每篇恰归入一簇、v1/v2 对同簇；
- 真实场景文件 + 真实题集全过校验器；题集占比硬口径断言
  （簇问 12–16、时效 3、无答案 2、重复对 4–6，查询事件 ≈22）；
- 序列题集与 real 题集的复用一致性：notes 标「复用 RQxxx」的行，question/
  qtype/gold_points/entities 与 real 题集逐字一致；标「新题」的行必须带
  「盲答验证 PASS」留痕标注；每道非无答案题至少一条 gold 要点能命中其
  来源文档原文（可判定性 sanity）。

全部零网络、零模型调用——数据在仓库内冻结，可复现。
"""

import json
import re
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from researchwiki.evals.metrics import point_hit
from researchwiki.evals.qa import load_qa
from researchwiki.evals.sequence import SeqEvent, clusters_from_manifest, load_scenario

REPO_ROOT = Path(__file__).resolve().parents[1]
EVALS_DIR = REPO_ROOT / "evals"
SEQ_QA_PATH = EVALS_DIR / "qa" / "ai-frameworks-seq.jsonl"
SCENARIO_PATH = EVALS_DIR / "sequence" / "scenario.jsonl"
REAL_QA_PATH = EVALS_DIR / "qa" / "ai-frameworks-real.jsonl"
MANIFEST_PATH = EVALS_DIR / "fixtures" / "ai-frameworks-real" / "manifest.json"
DOCS_DIR = EVALS_DIR / "fixtures" / "ai-frameworks-real" / "docs"


# ---------------------------------------------------------------------------
# 合成题集 / manifest / 场景的小型构造器（坏例单测用，与真实文件解耦）
# ---------------------------------------------------------------------------

def write_jsonl(path: Path, rows: list[dict]) -> None:
    """逐行写 JSONL（UTF-8、ensure_ascii=False，与真实数据同格式）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(text + "\n", encoding="utf-8")


def make_qa_rows() -> list[dict]:
    """小型题集：SEQ001 单跳（来源 doc-a）、SEQ002 时效（来源 doc-a@v2）、
    SEQ003 无答案、SEQ004 单跳（来源 doc-b）——覆盖四类校验分支的最小集合。"""
    return [
        {
            "qid": "SEQ001",
            "question": "doc-a 的关键方法是什么？",
            "qtype": "single_hop",
            "gold_points": ["method_a", "doc-a 定义关键方法"],
            "entities": ["doc-a"],
            "notes": "来源：doc-a。簇问。",
        },
        {
            "qid": "SEQ002",
            "question": "更新后 doc-a 的文档入口是什么？",
            "qtype": "temporal",
            "gold_points": ["docs.example.com", "文档入口已迁移到新域名"],
            "entities": ["doc-a"],
            "notes": "时效题：依赖 doc-a 的 v1/v2 对，gold 按 v2 口径。来源：doc-a@v2。",
        },
        {
            "qid": "SEQ003",
            "question": "语料外的信息是什么？",
            "qtype": "unanswerable",
            "gold_points": [],
            "entities": ["x"],
            "notes": "无答案题：语料不含此信息。",
        },
        {
            "qid": "SEQ004",
            "question": "doc-b 的安装命令是什么？",
            "qtype": "single_hop",
            "gold_points": ["pip install doc-b", "doc-b 提供安装命令"],
            "entities": ["doc-b"],
            "notes": "来源：doc-b。簇问。",
        },
    ]


def make_manifest_docs() -> list[dict]:
    """小型 manifest 文档表：一对 v1/v2 + 一篇普通文档。"""
    return [
        {"doc_id": "doc-a", "version": "v1"},
        {"doc_id": "doc-a@v2", "version": "v2"},
        {"doc_id": "doc-b", "version": "v1"},
    ]


def make_qa_file(tmp_path: Path, rows: list[dict] | None = None) -> Path:
    path = tmp_path / "qa.jsonl"
    write_jsonl(path, rows if rows is not None else make_qa_rows())
    return path


def make_manifest_file(tmp_path: Path, docs: list[dict] | None = None) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps({"domain": "tiny", "docs": docs if docs is not None else make_manifest_docs()},
                   ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def base_scenario_rows() -> list[dict]:
    """合法基线场景：t0 ingest 全部 → SEQ001 → SEQ004 → SEQ001（簇内重复对）
    → update（v2 到达）→ SEQ002（时效，update 后）。"""
    return [
        {"event_id": 0, "kind": "ingest", "doc_ids": ["doc-a", "doc-b"], "note": "全量 ingest"},
        {"event_id": 1, "kind": "query", "qid": "SEQ001", "note": "簇内重复对第 1 次"},
        {"event_id": 2, "kind": "query", "qid": "SEQ004", "note": "相邻问"},
        {"event_id": 3, "kind": "query", "qid": "SEQ001",
         "note": "簇内重复对第 2 次（两次均在 update 前）"},
        {"event_id": 4, "kind": "update", "doc_ids": ["doc-a@v2"], "note": "v2 替换 v1"},
        {"event_id": 5, "kind": "query", "qid": "SEQ002", "note": "时效问（update 后）"},
    ]


def make_scenario_file(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "scenario.jsonl"
    write_jsonl(path, rows)
    return path


def load_fail_case(tmp_path: Path, rows: list[dict], qa_rows: list[dict] | None = None,
                   manifest_docs: list[dict] | None = None) -> ValueError:
    """写坏例场景并断言 load_scenario 抛 ValueError，返回异常供消息断言。"""
    with pytest.raises(ValueError) as excinfo:
        load_scenario(
            make_scenario_file(tmp_path, rows),
            make_qa_file(tmp_path, qa_rows),
            make_manifest_file(tmp_path, manifest_docs),
        )
    return excinfo.value


# ---------------------------------------------------------------------------
# SeqEvent 基本契约
# ---------------------------------------------------------------------------

def test_seqevent_is_frozen() -> None:
    """frozen dataclass：字段赋值必须抛 FrozenInstanceError。"""
    event = SeqEvent(event_id=0, kind="ingest", doc_ids=("doc-a",), note="n")
    with pytest.raises(FrozenInstanceError):
        event.kind = "query"  # type: ignore[misc]


def test_seqevent_normalizes_doc_ids_to_tuple() -> None:
    """doc_ids 传入 list 时归一化为 tuple（frozen 数据不可变哈希）。"""
    event = SeqEvent(event_id=1, kind="ingest", doc_ids=["doc-a", "doc-b"])
    assert event.doc_ids == ("doc-a", "doc-b")
    assert isinstance(event.doc_ids, tuple)


def test_seqevent_defaults() -> None:
    """doc_ids/qid/note 缺省值：空 tuple / 空串。"""
    event = SeqEvent(event_id=2, kind="query", qid="SEQ001")
    assert event.doc_ids == ()
    assert event.note == ""


# ---------------------------------------------------------------------------
# load_scenario 坏例（每条校验规则至少 1 例；错误带物理行号）
# ---------------------------------------------------------------------------

def test_rejects_query_with_unknown_qid(tmp_path) -> None:
    """query 的 qid 不在题集 → ValueError 且带行号。"""
    rows = base_scenario_rows()
    rows[1]["qid"] = "SEQ999"
    err = load_fail_case(tmp_path, rows)
    assert "SEQ999" in str(err)
    assert "第 2 行" in str(err)


def test_rejects_query_with_empty_qid(tmp_path) -> None:
    """query 事件缺 qid → ValueError。"""
    rows = base_scenario_rows()
    del rows[1]["qid"]
    err = load_fail_case(tmp_path, rows)
    assert "第 2 行" in str(err)


def test_rejects_qid_appearing_three_times(tmp_path) -> None:
    """qid 出现 3 次（超过重复对上限 2）→ ValueError。"""
    rows = base_scenario_rows()
    rows.append({"event_id": 6, "kind": "query", "qid": "SEQ001", "note": "第三次"})
    err = load_fail_case(tmp_path, rows)
    assert "SEQ001" in str(err)
    assert "2 次" in str(err) or "两次" in str(err)


def test_rejects_repeated_pair_both_after_update(tmp_path) -> None:
    """重复对两次都在 update 之后 → 非法（合法形态只有簇内重复对与更新后重复对）。"""
    rows = [
        {"event_id": 0, "kind": "ingest", "doc_ids": ["doc-a", "doc-b"], "note": "ingest"},
        {"event_id": 1, "kind": "update", "doc_ids": ["doc-a@v2"], "note": "v2 替换 v1"},
        {"event_id": 2, "kind": "query", "qid": "SEQ002", "note": "时效问"},
        {"event_id": 3, "kind": "query", "qid": "SEQ001", "note": "重复对第 1 次"},
        {"event_id": 4, "kind": "query", "qid": "SEQ001", "note": "重复对第 2 次"},
    ]
    err = load_fail_case(tmp_path, rows)
    assert "SEQ001" in str(err)
    assert "update" in str(err)


def test_rejects_missing_update(tmp_path) -> None:
    """场景缺 update 事件 → ValueError（update 必须恰 1 个）。"""
    rows = [row for row in base_scenario_rows() if row["kind"] != "update"]
    # 时效题在无 update 的场景里同样前置依赖失败——先把时效题拿掉，隔离验证 update 规则
    rows = [row for row in rows if row.get("qid") != "SEQ002"]
    err = load_fail_case(tmp_path, rows)
    assert "update" in str(err)


def test_rejects_two_updates(tmp_path) -> None:
    """两个 update 事件 → ValueError。"""
    rows = base_scenario_rows()
    rows.insert(5, {"event_id": 6, "kind": "update", "doc_ids": ["doc-a@v2"], "note": "第二个"})
    # 重新编号让 event_id 连续，隔离"update 恰 1"规则
    for i, row in enumerate(rows):
        row["event_id"] = i
    err = load_fail_case(tmp_path, rows)
    assert "update" in str(err)


def test_rejects_temporal_query_before_update(tmp_path) -> None:
    """时效题的 query 事件在 update 之前 → ValueError。"""
    rows = [
        {"event_id": 0, "kind": "ingest", "doc_ids": ["doc-a", "doc-b"], "note": "ingest"},
        {"event_id": 1, "kind": "query", "qid": "SEQ002", "note": "时效问（过早）"},
        {"event_id": 2, "kind": "update", "doc_ids": ["doc-a@v2"], "note": "v2 替换 v1"},
    ]
    err = load_fail_case(tmp_path, rows)
    assert "时效" in str(err)
    assert "第 2 行" in str(err)


def test_rejects_ingest_doc_outside_manifest(tmp_path) -> None:
    """ingest 的 doc_ids 不在语料 manifest → ValueError 带行号。"""
    rows = base_scenario_rows()
    rows[0]["doc_ids"] = ["doc-a", "doc-ghost"]
    err = load_fail_case(tmp_path, rows)
    assert "doc-ghost" in str(err)
    assert "第 1 行" in str(err)


def test_rejects_update_doc_outside_manifest(tmp_path) -> None:
    """update 的 doc_ids 不在语料 manifest → ValueError。"""
    rows = base_scenario_rows()
    rows[4]["doc_ids"] = ["doc-b@v2"]
    err = load_fail_case(tmp_path, rows)
    assert "doc-b@v2" in str(err)
    assert "第 5 行" in str(err)


def test_rejects_query_citing_uningested_doc(tmp_path) -> None:
    """query 题目来源文档在其之前未被 ingest/update → ValueError 带行号。"""
    rows = [
        {"event_id": 0, "kind": "ingest", "doc_ids": ["doc-a"], "note": "只 ingest doc-a"},
        {"event_id": 1, "kind": "query", "qid": "SEQ004", "note": "引用 doc-b 但尚未 ingest"},
        {"event_id": 2, "kind": "update", "doc_ids": ["doc-a@v2"], "note": "v2 替换 v1"},
    ]
    err = load_fail_case(tmp_path, rows)
    assert "doc-b" in str(err)
    assert "第 2 行" in str(err)


def test_rejects_source_doc_outside_manifest(tmp_path) -> None:
    """题集 notes 的来源文档不在 manifest（拼写错）→ ValueError。"""
    qa_rows = make_qa_rows()
    qa_rows[0]["notes"] = "来源：doc-typo。簇问。"
    rows = base_scenario_rows()
    err = load_fail_case(tmp_path, rows, qa_rows=qa_rows)
    assert "doc-typo" in str(err)


def test_rejects_answerable_question_without_source_segment(tmp_path) -> None:
    """非无答案题的 notes 缺「来源：」段 → 无法做依赖自查，ValueError。"""
    qa_rows = make_qa_rows()
    qa_rows[0]["notes"] = "没有来源段。"
    rows = base_scenario_rows()
    err = load_fail_case(tmp_path, rows, qa_rows=qa_rows)
    assert "SEQ001" in str(err)
    assert "来源" in str(err)


def test_rejects_unknown_kind(tmp_path) -> None:
    """未知事件类型 → ValueError 带行号。"""
    rows = base_scenario_rows()
    rows[2]["kind"] = "delete"
    err = load_fail_case(tmp_path, rows)
    assert "第 3 行" in str(err)


def test_rejects_query_carrying_doc_ids(tmp_path) -> None:
    """query 事件携带 doc_ids → ValueError（文档集只属于 ingest/update）。"""
    rows = base_scenario_rows()
    rows[1]["doc_ids"] = ["doc-a"]
    err = load_fail_case(tmp_path, rows)
    assert "第 2 行" in str(err)


def test_rejects_ingest_carrying_qid(tmp_path) -> None:
    """ingest/update 事件携带 qid → ValueError。"""
    rows = base_scenario_rows()
    rows[0]["qid"] = "SEQ001"
    err = load_fail_case(tmp_path, rows)
    assert "第 1 行" in str(err)


def test_rejects_non_continuous_event_ids(tmp_path) -> None:
    """event_id 不从 0 连续递增 → ValueError（时间线顺序即文件行序）。"""
    rows = base_scenario_rows()
    rows[3]["event_id"] = 99
    err = load_fail_case(tmp_path, rows)
    assert "event_id" in str(err)
    assert "第 4 行" in str(err)


def test_rejects_repeated_pair_with_empty_note(tmp_path) -> None:
    """重复对两次事件的 note 为空 → 无法判别重复对类型，ValueError。"""
    rows = base_scenario_rows()
    rows[3]["note"] = ""
    err = load_fail_case(tmp_path, rows)
    assert "SEQ001" in str(err)
    assert "note" in str(err)


def test_rejects_malformed_json_line(tmp_path) -> None:
    """非 JSON 行 → ValueError 带行号。"""
    path = tmp_path / "scenario.jsonl"
    lines = [json.dumps(row, ensure_ascii=False) for row in base_scenario_rows()]
    lines.insert(2, "{这不是JSON")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        load_scenario(path, make_qa_file(tmp_path), make_manifest_file(tmp_path))
    assert "第 3 行" in str(excinfo.value)


# ---------------------------------------------------------------------------
# load_scenario 合法形态（两种重复对）
# ---------------------------------------------------------------------------

def test_accepts_intra_cluster_pair_before_update(tmp_path) -> None:
    """簇内重复对：同一 qid 两次都在 update 前 → 合法。"""
    events = load_scenario(
        make_scenario_file(tmp_path, base_scenario_rows()),
        make_qa_file(tmp_path),
        make_manifest_file(tmp_path),
    )
    assert [e.kind for e in events] == [
        "ingest", "query", "query", "query", "update", "query"
    ]
    qids = [e.qid for e in events if e.kind == "query"]
    assert qids == ["SEQ001", "SEQ004", "SEQ001", "SEQ002"]


def test_accepts_straddle_pair_around_update(tmp_path) -> None:
    """更新后重复对：一前一后分居 update 两侧 → 合法。"""
    rows = [
        {"event_id": 0, "kind": "ingest", "doc_ids": ["doc-a", "doc-b"], "note": "ingest"},
        {"event_id": 1, "kind": "query", "qid": "SEQ004",
         "note": "更新后重复对第 1 次（update 前问）"},
        {"event_id": 2, "kind": "update", "doc_ids": ["doc-a@v2"], "note": "v2 替换 v1"},
        {"event_id": 3, "kind": "query", "qid": "SEQ002", "note": "时效问（update 后）"},
        {"event_id": 4, "kind": "query", "qid": "SEQ004",
         "note": "更新后重复对第 2 次（update 后重问测演化）"},
    ]
    events = load_scenario(make_scenario_file(tmp_path, rows), make_qa_file(tmp_path),
                           make_manifest_file(tmp_path))
    straddle = [e for e in events if e.kind == "query" and e.qid == "SEQ004"]
    assert len(straddle) == 2
    assert straddle[0].event_id < 2 < straddle[1].event_id


def test_update_makes_v2_doc_available_for_temporal_query(tmp_path) -> None:
    """时效题来源是 @v2 文档：由 update 事件（而非 ingest）提供，update 后可问。"""
    rows = base_scenario_rows()
    events = load_scenario(make_scenario_file(tmp_path, rows), make_qa_file(tmp_path),
                           make_manifest_file(tmp_path))
    temporal = [e for e in events if e.kind == "query" and e.qid == "SEQ002"]
    update = [e for e in events if e.kind == "update"]
    assert temporal[0].event_id > update[0].event_id


# ---------------------------------------------------------------------------
# clusters_from_manifest（真实 manifest）
# ---------------------------------------------------------------------------

def test_clusters_cover_all_manifest_docs_exactly_once() -> None:
    """15 篇语料每篇恰归入一簇；共 4 簇。"""
    clusters = clusters_from_manifest(MANIFEST_PATH)
    assert len(clusters) == 4
    doc_ids = [doc["doc_id"] for doc in json.loads(MANIFEST_PATH.read_text("utf-8"))["docs"]]
    flat = [d for docs in clusters.values() for d in docs]
    assert sorted(flat) == sorted(doc_ids)
    assert len(flat) == len(set(flat))


def test_clusters_keep_version_pairs_together() -> None:
    """v1/v2 时效对必须同簇（update 后仍属同一主题簇）。"""
    clusters = clusters_from_manifest(MANIFEST_PATH)
    doc_to_cluster = {d: name for name, docs in clusters.items() for d in docs}
    for base in ("root-readme", "libs-langchain-readme", "libs-core-readme"):
        assert f"{base}@v2" in doc_to_cluster, base
        assert doc_to_cluster[base] == doc_to_cluster[f"{base}@v2"]


def test_clusters_from_manifest_accepts_loaded_dict() -> None:
    """manifest 传入已加载 dict 与传路径等价。"""
    manifest = json.loads(MANIFEST_PATH.read_text("utf-8"))
    assert clusters_from_manifest(manifest) == clusters_from_manifest(MANIFEST_PATH)


def test_clusters_from_manifest_rejects_unknown_doc() -> None:
    """manifest 含未知 doc_id（不在簇定义里）→ ValueError。"""
    with pytest.raises(ValueError):
        clusters_from_manifest({"docs": [{"doc_id": "not-in-any-cluster"}]})


# ---------------------------------------------------------------------------
# 真实交付物：序列题集 + 事件流全过校验器 + 占比硬口径
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_qa_items():
    return load_qa(SEQ_QA_PATH)


@pytest.fixture(scope="module")
def real_events():
    return load_scenario(SCENARIO_PATH, SEQ_QA_PATH, MANIFEST_PATH)


def test_real_scenario_passes_validation(real_events) -> None:
    """真实事件流过全部校验规则。"""
    assert real_events, "事件流非空"
    kinds = [e.kind for e in real_events]
    assert kinds[0] == "ingest"
    assert kinds.count("update") == 1


def test_real_scenario_query_event_budget(real_events) -> None:
    """查询事件总数 22（≈22 问口径）：16 簇问事件 + 3 时效 + 2 无答案 + 1 更新后重复。"""
    query_events = [e for e in real_events if e.kind == "query"]
    assert len(query_events) == 22


def test_real_scenario_all_qids_queried(real_events, real_qa_items) -> None:
    """题集 17 题全部进入事件流（无死题）。"""
    queried = {e.qid for e in real_events if e.kind == "query"}
    assert queried == {q.qid for q in real_qa_items}


def test_real_qa_type_proportions(real_qa_items) -> None:
    """题集占比硬口径：簇问 12–16、时效 3、无答案 2，共 17。"""
    counts: dict[str, int] = {}
    for item in real_qa_items:
        counts[item.qtype] = counts.get(item.qtype, 0) + 1
    cluster_count = counts.get("single_hop", 0) + counts.get("multi_hop", 0)
    assert 12 <= cluster_count <= 16
    assert counts.get("temporal", 0) == 3
    assert counts.get("unanswerable", 0) == 2
    assert len(real_qa_items) == cluster_count + 3 + 2


def test_real_qa_qids_sequential(real_qa_items) -> None:
    """qid 从 SEQ001 连续编号、无重复。"""
    qids = [q.qid for q in real_qa_items]
    assert qids == [f"SEQ{i:03d}" for i in range(1, len(qids) + 1)]


def test_real_scenario_repeat_pair_structure(real_events) -> None:
    """重复对 4–6：4 个簇内重复对（两次均在 update 前）+ 1 个更新后重复对。"""
    update_id = next(e.event_id for e in real_events if e.kind == "update")
    by_qid: dict[str, list[int]] = {}
    for e in real_events:
        if e.kind == "query":
            by_qid.setdefault(e.qid, []).append(e.event_id)
    pairs = {qid: ids for qid, ids in by_qid.items() if len(ids) == 2}
    assert 4 <= len(pairs) <= 6
    intra = [qid for qid, ids in pairs.items() if ids[1] < update_id]
    straddle = [qid for qid, ids in pairs.items()
                if ids[0] < update_id < ids[1]]
    assert len(intra) == 4
    assert len(straddle) == 1


def test_real_scenario_temporal_queries_after_update(real_events, real_qa_items) -> None:
    """时效题 query 事件全部在 update 之后（显式断言，报告口径）。"""
    update_id = next(e.event_id for e in real_events if e.kind == "update")
    temporal_qids = {q.qid for q in real_qa_items if q.qtype == "temporal"}
    for e in real_events:
        if e.kind == "query" and e.qid in temporal_qids:
            assert e.event_id > update_id


def test_real_scenario_ingest_covers_all_t0_docs(real_events) -> None:
    """t0 ingest 恰 1 个事件，携带 12 篇 t0 文档（3 对 @v2 由 update 提供）。"""
    ingests = [e for e in real_events if e.kind == "ingest"]
    assert len(ingests) == 1
    updates = [e for e in real_events if e.kind == "update"]
    assert len(updates[0].doc_ids) == 3
    assert all(d.endswith("@v2") for d in updates[0].doc_ids)
    assert not any(d.endswith("@v2") for d in ingests[0].doc_ids)
    assert len(ingests[0].doc_ids) == 12


# ---------------------------------------------------------------------------
# 序列题集与 real 题集的复用一致性
# ---------------------------------------------------------------------------

def _real_items_by_qid() -> dict:
    return {q.qid: q for q in load_qa(REAL_QA_PATH)}


def test_reused_rows_match_real_gold_verbatim(real_qa_items) -> None:
    """notes 标「复用 RQxxx」的行：question/qtype/gold_points/entities 与
    real 题集逐字一致（gold 零漂移）。"""
    real = _real_items_by_qid()
    reused = 0
    for item in real_qa_items:
        match = re.search(r"复用 (RQ\d+)", item.notes)
        if match is None:
            continue
        rqid = match.group(1)
        assert rqid in real, f"{item.qid} 声称复用不存在的 {rqid}"
        src = real[rqid]
        assert item.question == src.question, item.qid
        assert item.qtype == src.qtype, item.qid
        assert item.gold_points == src.gold_points, item.qid
        assert item.entities == src.entities, item.qid
        reused += 1
    assert reused >= 14, f"复用题应占多数，实际 {reused}"


def test_new_rows_carry_blind_validation_pass_note(real_qa_items) -> None:
    """标「新题」的行必须带「盲答验证 PASS」标注（真实 cheap 调用留痕）。"""
    new_rows = [q for q in real_qa_items if "新题" in q.notes]
    assert 2 <= len(new_rows) <= 8, f"新综合问数应≈4–8 区间内，实际 {len(new_rows)}"
    for item in new_rows:
        assert "盲答验证 PASS" in item.notes, item.qid
        assert "复用 RQ" not in item.notes, item.qid


def test_every_non_unanswerable_gold_hits_source_doc_text(real_qa_items) -> None:
    """可判定性 sanity：每道非无答案题至少一条 gold 要点命中其来源文档原文。"""
    for item in real_qa_items:
        if item.qtype == "unanswerable":
            continue
        sources = _parse_sources(item.notes)
        assert sources, f"{item.qid} 缺来源段"
        corpus_text = "\n".join(
            (DOCS_DIR / f"{doc_id}.md").read_text(encoding="utf-8") for doc_id in sources
        )
        assert any(point_hit(corpus_text, gp) for gp in item.gold_points), item.qid


def _parse_sources(notes: str) -> list[str]:
    """从 notes 的「来源：」段解析 doc_id 列表（顿号分隔，句号终止）——
    与 sequence.py 的依赖自查同口径。"""
    match = re.search(r"来源：(.+)", notes)
    if not match:
        return []
    segment = match.group(1).split("。")[0]
    return [part.strip() for part in segment.split("、") if part.strip()]


def test_blind_trace_file_exists_for_new_questions(real_qa_items) -> None:
    """新题的盲答验证留痕 JSONL 存在且每道新题至少一条 PASS 记录。"""
    new_rows = [q for q in real_qa_items if "新题" in q.notes]
    trace_path = EVALS_DIR / "qa" / "ai-frameworks-seq.blind.jsonl"
    assert trace_path.exists(), "缺少盲答验证留痕文件"
    records = [json.loads(line) for line in
               trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    passed_qids = {r["qid"] for r in records if r.get("pass") is True}
    for item in new_rows:
        assert item.qid in passed_qids, f"{item.qid} 无盲答 PASS 记录"
