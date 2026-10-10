"""scripts/run_sequence.py 的测试：mock 模式小事件流×四条件端到端 + 对账。

脚本不是包，用 importlib 按路径加载（与 tests/test_run_eval.py 同款）。
端到端跑真实 AgentLoop 与 run_rag（ScriptedProvider + FixtureSearch 受控语料 +
httpx.MockTransport 本地回放语料 markdown），零网络、零 key；断言落在
events.jsonl 行契约、四条件事件重放语义（c4 持久记忆 / c1 每问一次性 root /
c2/c3 持久索引 update 换 chunk / gold_override diff-gold）与 token 对账上。

小场景（合成语料 2 簇 5 文档 + 3 题 7 事件）：
  e0 ingest [doc-a, doc-b, doc-c, doc-d]
  e1 query SEQ001（簇内重复对第 1 次）
  e2 query SEQ001（簇内重复对第 2 次）
  e3 query SEQ002（更新后重复对第 1 次）
  e4 update [doc-a@v2]
  e5 query SEQ003（时效问，gold=v2 口径）
  e6 query SEQ002（更新后重复对第 2 次，gold_override=v2 口径）
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from researchwiki.evals.sequence import SeqEvent, load_scenario
from researchwiki.evals.qa import QaItem
from researchwiki.loop.metrics import sum_tokens_from_jsonl
from researchwiki.wiki.embeddings import MockEmbeddingProvider

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str) -> ModuleType:
    """按路径加载脚本；先注册进 sys.modules，否则 dataclass 解析注解会失败。"""
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


run_sequence = _load("run_sequence", "scripts/run_sequence.py")

# ---- 合成语料 / 题集 / 事件流（tmp_path 内构建，零网络零仓库污染）--------------


def _make_corpus(root: Path) -> Path:
    """合成 5 文档语料：doc-a/v2 对（切分策略 v1→v2）+ doc-b/c/d（簇 2）。"""
    docs = {
        "doc-a": "# 文档切分策略（v1）\n\n文本切分是 RAG 的重要预处理步骤。最直观的策略是 "
        "Length-based 切分：基于长度的简单而有效的方法，确保每个块不超过指定大小限制。\n",
        "doc-a@v2": "# 文档切分策略（v2）\n\n1.0 版本起，文档切分推荐语义切分（semantic "
        "chunking）。v2 口径：语义切分按句子边界与嵌入相似度切分，取代固定长度切分。\n",
        "doc-b": "# 嵌入模型（v1）\n\n嵌入模型把文本转换为向量表示，用于语义检索与相似度计算。\n",
        "doc-c": "# 向量存储（v1）\n\n向量存储通过 similarity_search 方法实现基于语义相似性的检索，"
        "是 RAG 系统检索功能的基础。\n",
        "doc-d": "# 聊天历史（v1）\n\n聊天历史记录是用户和聊天模型之间对话的记录，每条消息都与特定角色相关联。\n",
    }
    fixture = root / "fixtures"
    (fixture / "docs").mkdir(parents=True)
    manifest_docs = []
    for doc_id, text in docs.items():
        (fixture / "docs" / f"{doc_id}.md").write_text(text, encoding="utf-8")
        manifest_docs.append(
            {
                "doc_id": doc_id,
                "title": text.splitlines()[0].lstrip("# "),
                "url": f"fixture://seqtest/{doc_id}",
            }
        )
    manifest = {
        "domain": "seqtest",
        "provider_note": "合成测试语料（零网络）",
        "docs": manifest_docs,
    }
    (fixture / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return fixture


def _make_qa(root: Path) -> Path:
    qa = root / "qa-seq.jsonl"
    rows = [
        {
            "qid": "SEQ001",
            "question": "文档切分中最直观的策略是什么？",
            "qtype": "single_hop",
            "gold_points": ["Length-based", "基于长度的简单而有效的方法"],
            "entities": ["Length-based"],
            "notes": "来源：doc-a。簇内重复对。",
        },
        {
            "qid": "SEQ002",
            "question": "向量存储通过哪个方法实现相似性检索？",
            "qtype": "single_hop",
            "gold_points": ["similarity_search", "基于语义相似性"],
            "entities": ["similarity_search"],
            "notes": "来源：doc-c。更新后重复对。",
        },
        {
            "qid": "SEQ003",
            "question": "1.0 版本起文档切分推荐什么策略？",
            "qtype": "temporal",
            "gold_points": ["语义切分", "semantic chunking"],
            "entities": ["语义切分"],
            "notes": "来源：doc-a@v2。时效题，gold 按 v2 口径。",
        },
    ]
    qa.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    return qa


def _make_scenario(root: Path) -> Path:
    scenario = root / "scenario.jsonl"
    rows = [
        {
            "event_id": 0,
            "kind": "ingest",
            "doc_ids": ["doc-a", "doc-b", "doc-c", "doc-d"],
            "note": "t0 全量 ingest。",
        },
        {"event_id": 1, "kind": "query", "qid": "SEQ001", "note": "簇内重复对第 1 次。"},
        {"event_id": 2, "kind": "query", "qid": "SEQ001", "note": "簇内重复对第 2 次（均在 update 前）。"},
        {"event_id": 3, "kind": "query", "qid": "SEQ002", "note": "更新后重复对第 1 次。"},
        {
            "event_id": 4,
            "kind": "update",
            "doc_ids": ["doc-a@v2"],
            "note": "t5 update：v2 替换 v1。",
        },
        {"event_id": 5, "kind": "query", "qid": "SEQ003", "note": "时效问（update 后）。"},
        {
            "event_id": 6,
            "kind": "query",
            "qid": "SEQ002",
            "note": "更新后重复对第 2 次，gold_override=v2 口径。",
            "gold_override": ["语义切分", "semantic chunking"],
        },
    ]
    scenario.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    return scenario


@pytest.fixture()
def seq_env(tmp_path: Path) -> dict[str, Path]:
    """一套合成场景：返回 {fixtures, qa, scenario} 路径。"""
    return {
        "fixtures": _make_corpus(tmp_path),
        "qa": _make_qa(tmp_path),
        "scenario": _make_scenario(tmp_path),
    }


def _base_args(env: dict[str, Path], out_root: Path) -> list[str]:
    return [
        "--provider",
        "mock",
        "--scenario",
        str(env["scenario"]),
        "--qa",
        str(env["qa"]),
        "--fixtures",
        str(env["fixtures"]),
        "--out",
        str(out_root),
        "--env-file",
        "",
    ]


@pytest.fixture(autouse=True)
def _no_search_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """零网络保证：清掉搜索相关环境变量，兜住一切环境巧合（与 test_run_eval 同约定）。"""
    for var in ("SEARCH_PROVIDER", "TAVILY_API_KEY", "BOCHA_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def _read_rows(out_dir: Path) -> list[dict[str, Any]]:
    lines = (out_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def _run_metrics(row: dict[str, Any]) -> dict[str, Any]:
    return json.loads((Path(row["run_dir"]) / "run-metrics.json").read_text(encoding="utf-8"))


# ---- 行契约基键 ----------------------------------------------------------------

BASE_ROW_KEYS = {
    "condition",
    "event_id",
    "event_kind",
    "qid",
    "in_tok",
    "out_tok",
    "latency_ms",
    "em",
    "point_hits",
    "refusal",
    "citation_coverage",
    "judge",
    "gold_source",
    "trace_id",
}


def test_mock_sequence_end_to_end(tmp_path: Path, seq_env: dict[str, Path], capsys) -> None:
    out_root = tmp_path / "results"
    rc = run_sequence.main(_base_args(seq_env, out_root))

    assert rc == 0
    run_dirs = list(out_root.glob("sequence_mock_*"))
    assert len(run_dirs) == 1
    out_dir = run_dirs[0]
    rows = _read_rows(out_dir)

    # 事件序驱动：e0 ingest→3 行（c4 study + c2/c3 索引）；e1–e3、e5–e6 query→各 4 行；
    # e4 update→3 行；共 26 行。event_id 单调不减，query 事件内条件按 c1→c4 排布。
    assert len(rows) == 26
    event_ids = [r["event_id"] for r in rows]
    assert event_ids == sorted(event_ids)
    by_event: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_event.setdefault(row["event_id"], []).append(row)
    assert sorted(r["condition"] for r in by_event[0]) == ["c2", "c3", "c4"]
    assert sorted(r["condition"] for r in by_event[4]) == ["c2", "c3", "c4"]
    for ev in (1, 2, 3, 5, 6):
        assert [r["condition"] for r in by_event[ev]] == ["c1", "c2", "c3", "c4"]
        assert all(r["event_kind"] == "query" for r in by_event[ev])

    # 字段契约：基键齐全；loop 行（c1/c4）带 fresh_search_count/run_dir/wiki_root，
    # c4 loop 行再带 policy_mode；RAG 行带 hit_urls、无 fresh/run_dir/policy_mode。
    for row in rows:
        assert BASE_ROW_KEYS <= set(row), f"缺基键：{sorted(BASE_ROW_KEYS - set(row))}"
        if row["condition"] in ("c1", "c4"):
            assert "fresh_search_count" in row and "run_dir" in row and "wiki_root" in row
        else:
            assert "fresh_search_count" not in row
            assert "run_dir" not in row and "wiki_root" not in row
            if row["event_kind"] == "query":
                assert "hit_urls" in row and row["hit_urls"]
        if row["condition"] == "c4":
            assert "policy_mode" in row and row["policy_mode"] in ("simple", "update", "deep")

    # ingest/update 事件行语义：c4 记 study run 成本（qid=""，judge/em 均不评分），
    # c2/c3 记索引构建（embedding_calls>0、零 LLM 成本 in_tok==0），c1 无行。
    for ev in (0, 4):
        c4_row = next(r for r in by_event[ev] if r["condition"] == "c4")
        assert c4_row["event_kind"] in ("ingest", "update")
        assert c4_row["qid"] == "" and c4_row["in_tok"] > 0
        assert c4_row["judge"] is None and c4_row["em"] is None
        assert c4_row["gold_source"] == "base"
        for cond in ("c2", "c3"):
            idx_row = next(r for r in by_event[ev] if r["condition"] == cond)
            assert idx_row["embedding_calls"] > 0
            assert idx_row["in_tok"] == 0 and idx_row["out_tok"] == 0
            assert idx_row["judge"] is None
        assert not any(r["condition"] == "c1" for r in by_event[ev])

    # c4 持久 root 贯穿全时间线：所有 c4 行同一 wiki_root；第二次 SEQ001 查询
    # （e2）prior 命中 >0——记忆在 ingest 事件沉淀、跨 query 复用。
    c4_rows = [r for r in rows if r["condition"] == "c4"]
    assert len({r["wiki_root"] for r in c4_rows}) == 1
    e2_c4 = next(r for r in by_event[2] if r["condition"] == "c4")
    assert _run_metrics(e2_c4)["prior_hit_count"] > 0

    # c1 每问一次性 root：每问（5 个 query 事件）独立目录，prior 从未命中。
    c1_rows = [r for r in rows if r["condition"] == "c1"]
    assert len({r["wiki_root"] for r in c1_rows}) == len(c1_rows) == 5
    for row in c1_rows:
        assert _run_metrics(row)["prior_hit_count"] == 0

    # RAG 行语料时间线：update 前 v2 不在索引（hit_urls 无 @v2）；update 后
    # 时效问（SEQ003）的检索命中 v2 chunk（v2 替换 v1，取对 RAG 有利解释）。
    for ev in (1, 2, 3):
        for row in by_event[ev]:
            if row["condition"] in ("c2", "c3"):
                assert all("@v2" not in u for u in row["hit_urls"])
    for ev in (5, 6):
        for row in by_event[ev]:
            if row["condition"] in ("c2", "c3"):
                assert any("doc-a@v2" in u for u in row["hit_urls"])

    # gold_override diff-gold：e6（带 override 的 query）全条件 gold_source=override，
    # 其余 query 行为 base。
    for row in by_event[6]:
        assert row["gold_source"] == "override"
    for ev in (1, 2, 3, 5):
        for row in by_event[ev]:
            assert row["gold_source"] == "base"

    # judge：每个 query 行恰一次评审（mock 占位 verdict 三维分齐全）。
    for row in rows:
        if row["event_kind"] == "query":
            judge = row["judge"]
            assert isinstance(judge, dict)
            for field in ("coverage", "citation", "temporal"):
                assert 1 <= judge[field] <= 5

    # token 对账：loop 行按 <wiki_root>/tokens.jsonl + trace_id 复算一致（ingest/update
    # 的 study run 与 judge 用独立 trace，不混入主 trace）；RAG 行按 <out>/tokens.jsonl。
    for row in c1_rows + c4_rows:
        tok_in, tok_out = sum_tokens_from_jsonl(
            Path(row["wiki_root"]) / "tokens.jsonl", row["trace_id"]
        )
        assert tok_in > 0, "对账不应空转"
        assert (row["in_tok"], row["out_tok"]) == (tok_in, tok_out)
    for row in rows:
        if row["condition"] in ("c2", "c3") and row["event_kind"] == "query":
            tok_in, tok_out = sum_tokens_from_jsonl(out_dir / "tokens.jsonl", row["trace_id"])
            assert tok_in > 0
            assert (row["in_tok"], row["out_tok"]) == (tok_in, tok_out)

    # manifest provenance：scenario/qa sha256、语料时间线、裁定与预注册判定原文。
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["gate"] == "run_sequence"
    assert manifest["scenario_sha256"] == hashlib.sha256(
        seq_env["scenario"].read_bytes()
    ).hexdigest()
    assert manifest["qa_sha256"] == hashlib.sha256(seq_env["qa"].read_bytes()).hexdigest()
    timeline = {item["event_id"]: item for item in manifest["corpus_timeline"]}
    assert set(timeline[0]["doc_set_after"]) == {"doc-a", "doc-b", "doc-c", "doc-d"}
    assert "doc-a@v2" not in timeline[0]["doc_set_after"]
    assert "doc-a@v2" in timeline[4]["doc_set_after"]
    assert "doc-a" not in timeline[4]["doc_set_after"]
    assert "经济性" in json.dumps(manifest["pre_registered"], ensure_ascii=False)
    rulings = manifest["rulings"]
    assert rulings["query_readonly"] is True
    assert rulings["index_full_rebuild"] is True
    assert "c1" in rulings["search_corpus_switches_with_update"]

    # stdout 汇总带防误读声明与落盘位置
    printed = capsys.readouterr().out
    assert "小样本" in printed
    assert "events.jsonl" in printed


def test_gold_override_semantics() -> None:
    """gold_for：带 gold_override 的 query 事件用 override 判分（gold_source=override），
    其余用题集 gold（base）——EM 单列演化口径的纯函数钉子。"""
    item = QaItem(
        qid="SEQ002",
        question="q",
        qtype="single_hop",
        gold_points=["v1 口径要点", "另一要点"],
        entities=["x"],
    )
    base_event = SeqEvent(event_id=3, kind="query", qid="SEQ002")
    assert run_sequence.gold_for(item, base_event) == (["v1 口径要点", "另一要点"], "base")
    override_event = SeqEvent(
        event_id=6, kind="query", qid="SEQ002", gold_override=("v2 口径要点",)
    )
    gold, source = run_sequence.gold_for(item, override_event)
    assert gold == ["v2 口径要点"] and source == "override"


def test_corpus_index_update_replaces_v1(tmp_path: Path) -> None:
    """索引更新函数：update 后 v1 chunk 从索引消失、v2 chunk 可被检索命中——
    与 harness 在 update 事件用的是同一函数。"""
    fixture = _make_corpus(tmp_path)
    embedding = MockEmbeddingProvider(dim=512)
    doc_ids = ["doc-a", "doc-b", "doc-c", "doc-d"]
    index = run_sequence.build_corpus_index(fixture, doc_ids, embedding)
    assert all(c.doc_id != "doc-a@v2" for c in index.chunks)

    new_state = run_sequence.apply_corpus_change(doc_ids, ["doc-a@v2"])
    assert new_state == ["doc-b", "doc-c", "doc-d", "doc-a@v2"]
    index2 = run_sequence.build_corpus_index(fixture, new_state, embedding)
    assert all(c.doc_id != "doc-a" for c in index2.chunks)
    hits = index2.search("1.0 版本 语义切分 策略", max_results=5, mode="hybrid")
    assert hits and "doc-a@v2" in hits[0].url


def test_scenario_validation_failure_aborts_before_any_run(tmp_path: Path, capsys) -> None:
    """事件流校验失败（时效题在 update 前）→ 干净报错退出，零落盘。"""
    env = {
        "fixtures": _make_corpus(tmp_path / "c"),
        "qa": _make_qa(tmp_path / "c"),
        "scenario": tmp_path / "bad-scenario.jsonl",
    }
    (tmp_path / "c").mkdir(parents=True, exist_ok=True)
    rows = [
        {"event_id": 0, "kind": "ingest", "doc_ids": ["doc-a"], "note": ""},
        {"event_id": 1, "kind": "query", "qid": "SEQ003", "note": "时效问"},
        {"event_id": 2, "kind": "update", "doc_ids": ["doc-a@v2"], "note": ""},
        {"event_id": 3, "kind": "query", "qid": "SEQ001", "note": ""},
    ]
    env["scenario"].write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    out_root = tmp_path / "results"
    rc = run_sequence.main(_base_args(env, out_root))
    assert rc == run_sequence.EXIT_RUN_ERROR
    assert not out_root.exists(), "校验失败必须在创建输出目录之前退出（零落盘）"
    assert "时效" in capsys.readouterr().out


def test_real_mode_rejects_mock_judge(tmp_path: Path, seq_env: dict[str, Path]) -> None:
    """real + [llm.judge] 缺失/空 base_url → 启动即退出码 2，不产生任何输出目录。"""
    out_root = tmp_path / "results"
    rc = run_sequence.main(
        [
            "--provider",
            "real",
            "--scenario",
            str(seq_env["scenario"]),
            "--qa",
            str(seq_env["qa"]),
            "--fixtures",
            str(seq_env["fixtures"]),
            "--out",
            str(out_root),
            "--env-file",
            "",
            "--config",
            str(tmp_path / "missing.toml"),
        ]
    )
    assert rc == run_sequence.EXIT_RUN_ERROR
    assert not out_root.exists(), "校验失败必须在创建输出目录之前退出（零落盘）"


def test_make_corpus_fetch_transport_replays_manifest_urls(tmp_path: Path) -> None:
    """fetch 回放 transport：manifest 的 URL（及 Jina 前缀变体）回放本地语料；
    未登记 URL 返回 404——real 语料（github URL）零网络 fetch 的机制钉子。"""
    import httpx

    fixture = _make_corpus(tmp_path)
    transport = run_sequence.make_corpus_fetch_transport(fixture)
    manifest = json.loads((fixture / "manifest.json").read_text(encoding="utf-8"))
    url = manifest["docs"][0]["url"]  # fixture://seqtest/doc-a
    expected = (fixture / "docs" / "doc-a.md").read_text(encoding="utf-8")

    client = httpx.Client(transport=transport)
    resp = client.get(url)
    assert resp.status_code == 200 and resp.text == expected
    assert "text/markdown" in resp.headers["content-type"]
    # Jina Reader 降级路径同样回放
    resp2 = client.get(f"https://r.jina.ai/{url}")
    assert resp2.status_code == 200 and resp2.text == expected
    # 未登记 URL → 404（fetch_url 如实报 FetchError）
    assert client.get("https://example.com/unknown").status_code == 404


def test_load_scenario_wires_harness_defaults(tmp_path: Path, seq_env: dict[str, Path]) -> None:
    """harness 的事件流加载：load_scenario(SeqEvent) 经默认路径参数可跑通，
    校验器与 harness 消费同一 SeqEvent 契约。"""
    events = load_scenario(seq_env["scenario"], seq_env["qa"], seq_env["fixtures"] / "manifest.json")
    assert [e.kind for e in events] == [
        "ingest", "query", "query", "query", "update", "query", "query",
    ]
    assert events[6].gold_override == ("语义切分", "semantic chunking")
