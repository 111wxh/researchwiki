"""人工仲裁 CLI 与 Prior 待解决冲突注入（P4a Task 18 / T4）测试。

覆盖 task-18 简报的验收清单（六组）：

① arbitrate list：open/resolved/all 过滤（默认 open）与 --json；
② arbitrate resolve：conflict 变 resolved + resolution/resolved_at 落盘（store
   既有行为的接线验证）+ 产生裁决留痕 kind=experience 笔记（extra 平铺
   ledger=arbitration / conflict_id / adopted_note_id，body 引用双方与结论）；
③ resolve 传不存在 / 非 active-merged（superseded、墓碑）的笔记 ID → 退出 2 且
   conflict 保持 open、无留痕笔记；merged 笔记合法；
④ arbitrate show 全文（open 与 resolved）+ --json + 不存在的 id 退出 2；
⑤ Prior 注入：问题实体与 open 冲突实体相交 → 注入（置于护栏块之后、Prior 块
   之前）；无相交且相似度不达标 → 不注入；相似度达 [verification].similarity_floor
   也可注入；resolved 后同问题不再注入；[prior].enabled=false 不注入；
⑥ 仲裁留痕笔记可被 memory.search（WikiService.search，双通道）检索到——它就是
   普通记忆。

全部零网络、零模型调用（AgentLoop 用 ScriptedProvider；相似度复用
wiki/verification.py 的确定性 bigram 特征哈希度量——同一把尺）。
"""

import json
from pathlib import Path

import pytest

from researchwiki import cli
from researchwiki.llm.accounting import TokenAccountant
from researchwiki.llm.provider import ScriptedProvider, StreamEvent
from researchwiki.loop.agent_loop import AgentLoop
from researchwiki.mcp_server.service import WikiService
from researchwiki.tools import MockSearch
from researchwiki.wiki.embeddings import MockEmbeddingProvider
from researchwiki.wiki.ledger import REASON_NO_NOTES, record_unanswered
from researchwiki.wiki.prior import (
    CONFLICT_GUARD_HEADER,
    PRIOR_CONTEXT_LABEL,
    UNANSWERED_GUARD_HEADER,
    ConflictGuard,
    format_conflict_guards,
    retrieve_conflict_guards,
)
from researchwiki.wiki.store import WikiStore

# ---- 固定素材 ----------------------------------------------------------------

QUESTION = "GLM-5.3 的上下文窗口是多少？"
ENTITY = "GLM-5.3"
# 与 tests/test_maintenance.py 同一组冲突对（同实体、同槽位取值不同）
CONFLICT_A = "GLM-5.3 的上下文窗口是 128k。"
CONFLICT_B = "GLM-5.3 的上下文窗口是 64k。"
VERDICT = "上下文窗口以 64k 为准（2026-10 官方文档）"
SLOT = {
    "slot": "number:k:上下文窗口是",
    "kind": "number",
    "context": "上下文窗口是",
    "prior_value": "128",
    "evidence_value": "64",
}
DISSIMILAR_QUESTION = "量子纠缠的贝尔不等式实验结论"
SIMILAR_BODY = "分层摘要是长上下文记忆压缩的主要手段。"
MCP_CONFIG = {"wiki": {"fts_tokenizer": "trigram"}}


# ---- 脚手架（store 侧与 tests/test_maintenance.py 同构，loop 侧与 test_ledger.py 同构）


def make_store(tmp_path: Path) -> WikiStore:
    return WikiStore(tmp_path / "wiki-data")


def save(store: WikiStore, note_id: str, body: str, *, entities: list[str] | None = None, **kwargs):
    """按测试默认值写一条笔记（字段显式给出，不吃 save_note 的静默默认）。"""
    return store.save_note(
        body,
        note_id=note_id,
        title=f"t-{note_id}",
        entities=list(entities or []),
        **kwargs,
    )


def claim(note_id: str, role: str, *, slots: list[dict] | None = None, excerpt: str = "") -> dict:
    """构造与 T17 执行器 _exec_conflict 同形态的 claim（记录字段，注入侧只读不重比）。"""
    return {
        "note_id": note_id,
        "title": f"t-{note_id}",
        "excerpt": excerpt,
        "role": role,
        "slots": list(slots or []),
        "similarity": 0.8473,
        "idempotency_key": f"conflict:{note_id}",
    }


def seed_conflict_pair(store: WikiStore, *, conflict_id: str = "C-0001") -> None:
    """一对同实体矛盾笔记 + 一条 open 冲突台账（claim 形态同 T17 执行器产出）。"""
    save(store, "N-0001", CONFLICT_A, entities=[ENTITY])
    save(store, "N-0002", CONFLICT_B, entities=[ENTITY])
    store.save_conflict(
        f"{conflict_id}：GLM-5.3（N-0001）与 GLM-5.3（N-0002）"
        "同实体槽位取值冲突（consolidation 判定 conflicting），待人工裁决",
        claim("N-0001", "prior", slots=[SLOT], excerpt=CONFLICT_A),
        claim("N-0002", "evidence", slots=[SLOT], excerpt=CONFLICT_B),
        conflict_id=conflict_id,
    )


def text_turn(text: str) -> list[StreamEvent]:
    return [StreamEvent(type="text_delta", delta=text)]


def zero_output_turns() -> list[list[StreamEvent]]:
    """零产出 run 的脚本回合（与 tests/test_ledger.py 同款：plan → 空手 → 蒸馏空 → 报告）。"""
    plan_text = "- 任务一：检索上下文窗口参数\n- 任务二：蒸馏笔记\n"
    empty_distill = json.dumps({"notes": [], "conflicts": []}, ensure_ascii=False)
    report = "## 研究报告\n\n本轮如上。\n"
    return [
        text_turn(plan_text),
        text_turn("没有检索到可用来源，研究结束。"),
        text_turn(empty_distill),
        text_turn(report),
    ]


class FakeRouter:
    def __init__(self, strong: ScriptedProvider) -> None:
        self._providers = {"strong": strong}

    def get(self, tier: str) -> ScriptedProvider:
        if tier not in self._providers:
            raise AssertionError(f"测试未配置 {tier} 档 Provider，却被请求了")
        return self._providers[tier]


def make_loop(
    tmp_path: Path,
    *,
    turns: list[list[StreamEvent]],
    question: str = QUESTION,
    **kwargs,
) -> AgentLoop:
    return AgentLoop(
        question,
        router=FakeRouter(ScriptedProvider(turns, model="mock-strong")),
        llm_config={"strong": {"base_url": "https://mock"}, "cheap": {"base_url": ""}},
        accountant=TokenAccountant(tmp_path / "tokens.jsonl"),
        search_provider=MockSearch(),
        wiki_root=tmp_path / "wiki-data",
        run_dir=tmp_path / "run",
        embedding=MockEmbeddingProvider(dim=512),
        wiki_config={"fts_tokenizer": "trigram"},
        **kwargs,
    )


# ---- ① arbitrate list：状态过滤与 --json -------------------------------------


def test_cli_list_filters_by_status(tmp_path: Path, capsys) -> None:
    store = make_store(tmp_path)
    seed_conflict_pair(store, conflict_id="C-0001")
    seed_conflict_pair(store, conflict_id="C-0002")
    store.resolve_conflict("C-0002", verdict="以 N-0002 为准", resolved_with="N-0002")
    root = str(store.root)

    assert cli.main(["arbitrate", "list", "--status", "open", "--root", root]) == 0
    out = capsys.readouterr().out
    assert "C-0001" in out and "C-0002" not in out

    assert cli.main(["arbitrate", "list", "--status", "resolved", "--root", root]) == 0
    out = capsys.readouterr().out
    assert "C-0002" in out and "C-0001" not in out
    assert "N-0002" in out  # 采纳方留痕

    assert cli.main(["arbitrate", "list", "--status", "all", "--root", root]) == 0
    out = capsys.readouterr().out
    assert "C-0001" in out and "C-0002" in out

    # 缺省 --status = open（待裁决工作台）
    assert cli.main(["arbitrate", "list", "--root", root]) == 0
    out = capsys.readouterr().out
    assert "C-0001" in out and "C-0002" not in out


def test_cli_list_json_payload(tmp_path: Path, capsys) -> None:
    store = make_store(tmp_path)
    seed_conflict_pair(store, conflict_id="C-0001")
    seed_conflict_pair(store, conflict_id="C-0002")
    store.resolve_conflict("C-0002", verdict="以 N-0002 为准", resolved_with="N-0002")

    assert (
        cli.main(["arbitrate", "list", "--status", "all", "--root", str(store.root), "--json"])
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert [c["id"] for c in payload] == ["C-0001", "C-0002"]
    assert payload[0]["status"] == "open"
    assert payload[0]["claim_a"]["note_id"] == "N-0001"
    assert payload[0]["claim_b"]["role"] == "evidence"
    resolved = payload[1]
    assert resolved["status"] == "resolved"
    assert resolved["resolution"] == {"verdict": "以 N-0002 为准", "resolved_with": "N-0002"}
    assert resolved["resolved_at"]


def test_cli_list_empty_ledger(tmp_path: Path, capsys) -> None:
    assert cli.main(["arbitrate", "list", "--root", str(make_store(tmp_path).root)]) == 0
    assert "共 0 条" in capsys.readouterr().out


def resolve_argv(root: str, conflict_id: str, note_id: str, *extra: str) -> list[str]:
    """arbitrate resolve 的 argv（VERDICT 固定素材，降低用例噪音）。"""
    return [
        "arbitrate",
        "resolve",
        conflict_id,
        "--verdict",
        VERDICT,
        "--with",
        note_id,
        "--root",
        root,
        *extra,
    ]


# ---- ② arbitrate resolve：已决落盘 + 裁决留痕笔记 ------------------------------


def test_cli_resolve_marks_resolved_and_writes_trace_note(tmp_path: Path, capsys) -> None:
    store = make_store(tmp_path)
    seed_conflict_pair(store)
    root = str(store.root)

    assert cli.main(resolve_argv(root, "C-0001", "N-0002")) == 0
    out = capsys.readouterr().out
    assert "C-0001" in out and "N-0002" in out

    # 已决记录：resolution + resolved_at 落盘（store.resolve_conflict 既有行为的接线验证）
    conflict = store.get_conflict("C-0001")
    assert conflict.status == "resolved"
    assert conflict.resolution == {"verdict": VERDICT, "resolved_with": "N-0002"}
    assert conflict.resolved_at

    # 裁决留痕笔记：kind=experience + extra 平铺（store 既有惯例）
    experience = [n for n in store.list_notes(status="active") if n.kind == "experience"]
    assert len(experience) == 1
    note = experience[0]
    assert note.meta.extra["ledger"] == "arbitration"
    assert note.meta.extra["conflict_id"] == "C-0001"
    assert note.meta.extra["adopted_note_id"] == "N-0002"
    # body 引用冲突双方与结论
    assert "N-0001" in note.body and "N-0002" in note.body
    assert VERDICT in note.body


def test_cli_resolve_json_payload(tmp_path: Path, capsys) -> None:
    store = make_store(tmp_path)
    seed_conflict_pair(store)

    assert cli.main(resolve_argv(str(store.root), "C-0001", "N-0002", "--json")) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["trace_note_id"].startswith("N-")
    assert payload["conflict"]["status"] == "resolved"
    assert payload["conflict"]["resolution"]["resolved_with"] == "N-0002"


# ---- ③ arbitrate resolve 的 --with 校验：违规退出 2 且 conflict 保持 open ------


def test_cli_resolve_with_unknown_note_exits_2_and_keeps_open(tmp_path: Path, capsys) -> None:
    store = make_store(tmp_path)
    seed_conflict_pair(store)

    code = cli.main(resolve_argv(str(store.root), "C-0001", "N-9999"))
    assert code == 2
    assert "N-9999" in capsys.readouterr().err

    conflict = store.get_conflict("C-0001")
    assert conflict.status == "open"  # conflict 保持 open
    assert conflict.resolution == {} and conflict.resolved_at == ""
    assert [n for n in store.list_notes() if n.kind == "experience"] == []  # 无留痕


def test_cli_resolve_rejects_superseded_note(tmp_path: Path, capsys) -> None:
    store = make_store(tmp_path)
    seed_conflict_pair(store)
    store.save_note("新的现行口径。", note_id="N-0003", title="t-N-0003")
    meta = store.get_note("N-0002").meta.replace(status="superseded", superseded_by="N-0003")
    store.save_meta(meta, store.get_note("N-0002").body)

    code = cli.main(resolve_argv(str(store.root), "C-0001", "N-0002"))
    assert code == 2
    assert store.get_conflict("C-0001").status == "open"


def test_cli_resolve_accepts_merged_note(tmp_path: Path, capsys) -> None:
    """--with 允许 merged（采纳链上的合并笔记按其 id 留痕，不擅自跟随 redirect）。"""
    store = make_store(tmp_path)
    seed_conflict_pair(store)
    store.save_note("合并后的现行口径。", note_id="N-0003", title="t-N-0003")
    meta = store.get_note("N-0002").meta.replace(status="merged", redirect_to="N-0003")
    store.save_meta(meta, store.get_note("N-0002").body)

    assert cli.main(resolve_argv(str(store.root), "C-0001", "N-0002")) == 0
    assert store.get_conflict("C-0001").resolution["resolved_with"] == "N-0002"
    experience = [n for n in store.list_notes(status="active") if n.kind == "experience"]
    assert experience[0].meta.extra["adopted_note_id"] == "N-0002"


def test_cli_resolve_missing_conflict_exits_2(tmp_path: Path, capsys) -> None:
    root = str(make_store(tmp_path).root)
    code = cli.main(resolve_argv(root, "C-0042", "N-0001"))
    assert code == 2
    assert "C-0042" in capsys.readouterr().err


# ---- ④ arbitrate show：全文与 --json -----------------------------------------


def test_cli_show_prints_full_conflict(tmp_path: Path, capsys) -> None:
    store = make_store(tmp_path)
    seed_conflict_pair(store)

    assert cli.main(["arbitrate", "show", "C-0001", "--root", str(store.root)]) == 0
    out = capsys.readouterr().out
    assert "C-0001" in out and "open" in out
    assert "N-0001" in out and "N-0002" in out  # 双方
    assert "上下文窗口是 128" in out and "上下文窗口是 64" in out  # 断言对（槽位渲染）
    assert CONFLICT_A in out and CONFLICT_B in out  # 双方摘录
    assert "prior" in out and "evidence" in out  # role
    assert "128" in out and "64" in out  # 槽位取值


def test_cli_show_resolved_includes_resolution(tmp_path: Path, capsys) -> None:
    store = make_store(tmp_path)
    seed_conflict_pair(store)
    store.resolve_conflict("C-0001", verdict=VERDICT, resolved_with="N-0002")

    assert cli.main(["arbitrate", "show", "C-0001", "--root", str(store.root)]) == 0
    out = capsys.readouterr().out
    assert VERDICT in out and "N-0002" in out
    assert store.get_conflict("C-0001").resolved_at in out


def test_cli_show_json_and_missing_id(tmp_path: Path, capsys) -> None:
    store = make_store(tmp_path)
    seed_conflict_pair(store)

    assert cli.main(["arbitrate", "show", "C-0001", "--root", str(store.root), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["id"] == "C-0001"
    assert payload["claim_a"]["slots"] == [SLOT]

    assert cli.main(["arbitrate", "show", "C-9999", "--root", str(store.root)]) == 2
    assert "C-9999" in capsys.readouterr().err


# ---- ⑤ Prior 注入：待解决冲突块 ----------------------------------------------


def test_retrieve_conflict_guards_entity_intersection(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    seed_conflict_pair(store)

    guards = retrieve_conflict_guards(QUESTION, store)
    assert len(guards) == 1
    guard = guards[0]
    assert guard.conflict_id == "C-0001"
    assert guard.entity == ENTITY
    assert guard.assertion_a == "上下文窗口是 128"
    assert guard.assertion_b == "上下文窗口是 64"
    assert guard.note_a == "N-0001" and guard.note_b == "N-0002"
    # 无实体相交且相似度不达标 → 不注入
    assert retrieve_conflict_guards(DISSIMILAR_QUESTION, store) == []


def test_retrieve_conflict_guards_similarity_floor(tmp_path: Path) -> None:
    """无实体相交，但冲突笔记与问题相似度 ≥ [verification].similarity_floor → 注入。"""
    store = make_store(tmp_path)
    save(store, "N-0001", SIMILAR_BODY)  # 无实体（实体路径不可用）
    save(store, "N-0002", SIMILAR_BODY + "这是旧版表述。")
    conflict = store.save_conflict(
        "槽位取值冲突，待人工裁决",
        claim("N-0001", "prior", excerpt=SIMILAR_BODY),
        claim("N-0002", "evidence", excerpt=SIMILAR_BODY + "这是旧版表述。"),
    )

    guards = retrieve_conflict_guards(SIMILAR_BODY, store)
    assert [g.conflict_id for g in guards] == [conflict.id]
    # 实体与相似度都不达标 → 不注入
    assert retrieve_conflict_guards(DISSIMILAR_QUESTION, store) == []


def test_retrieve_conflict_guards_excludes_resolved(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    seed_conflict_pair(store)
    store.resolve_conflict("C-0001", verdict=VERDICT, resolved_with="N-0002")
    assert retrieve_conflict_guards(QUESTION, store) == []


def test_retrieve_conflict_guards_top_k(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    seed_conflict_pair(store, conflict_id="C-0001")
    seed_conflict_pair(store, conflict_id="C-0002")
    seed_conflict_pair(store, conflict_id="C-0003")
    assert [g.conflict_id for g in retrieve_conflict_guards(QUESTION, store)] == [
        "C-0001",
        "C-0002",
        "C-0003",
    ]
    assert [g.conflict_id for g in retrieve_conflict_guards(QUESTION, store, top_k=2)] == [
        "C-0001",
        "C-0002",
    ]


def test_format_conflict_guards_block_format() -> None:
    guard = ConflictGuard(
        conflict_id="C-0001",
        entity=ENTITY,
        assertion_a="上下文窗口是 128",
        assertion_b="上下文窗口是 64",
        note_a="N-0001",
        note_b="N-0002",
    )
    block = format_conflict_guards([guard])
    # 文案形态逐字（task-18 简报）：块头 + 台账行 + 来源与指令行
    assert block.startswith(CONFLICT_GUARD_HEADER + "\n")
    assert CONFLICT_GUARD_HEADER == "### 待解决冲突（优先核验）"
    assert "C-0001 实体「GLM-5.3」存在矛盾断言：「上下文窗口是 128」vs「上下文窗口是 64」" in block
    assert "（来源 N-0001 / N-0002）。请优先核实哪一方是现行事实，并在报告中说明。" in block
    assert format_conflict_guards([]) == ""


def test_loop_injects_conflict_block_between_guard_and_prior(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    seed_conflict_pair(store)
    record_unanswered(
        store,
        question=QUESTION,
        reason=REASON_NO_NOTES,
        trace_id="t-old",
        sources_used=0,
        notes_created=0,
        asked_at="2026-10-11T08:00:00+00:00",
    )
    # 一条普通笔记让 Prior 块非空（三块齐备才能断言相对位置）；显式给独立 id，
    # 避免覆盖台账笔记的自动编号（record_unanswered 会占掉下一个编号）
    save(store, "N-0009", "GLM-5.3 的上下文窗口官方口径分档公布。")

    loop = make_loop(tmp_path, turns=zero_output_turns(), unanswered_config={"enabled": True})
    events = list(loop.events())
    assert events[-1]["type"] == "finish"
    plan_message = loop.provider.calls[0][0].content
    guard_pos = plan_message.index(UNANSWERED_GUARD_HEADER)
    conflict_pos = plan_message.index(CONFLICT_GUARD_HEADER)
    prior_pos = plan_message.index(PRIOR_CONTEXT_LABEL)
    # R3 顺序裁定：护栏块 → 冲突块 → Prior 笔记
    assert guard_pos < conflict_pos < prior_pos
    assert "C-0001 实体「GLM-5.3」存在矛盾断言" in plan_message


def test_loop_conflict_block_absent_for_unrelated_question(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    seed_conflict_pair(store)
    loop = make_loop(
        tmp_path,
        turns=zero_output_turns(),
        question=DISSIMILAR_QUESTION,
        unanswered_config={"enabled": True},
    )
    events = list(loop.events())
    assert events[-1]["type"] == "finish"
    plan_message = loop.provider.calls[0][0].content
    assert CONFLICT_GUARD_HEADER not in plan_message


def test_loop_conflict_block_not_injected_after_resolve(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    seed_conflict_pair(store)
    store.resolve_conflict("C-0001", verdict=VERDICT, resolved_with="N-0002")

    loop = make_loop(tmp_path, turns=zero_output_turns())
    events = list(loop.events())
    assert events[-1]["type"] == "finish"
    assert CONFLICT_GUARD_HEADER not in loop.provider.calls[0][0].content


def test_loop_conflict_block_not_injected_when_prior_disabled(tmp_path: Path) -> None:
    """[prior].enabled=false 逃生阀：冲突块不注入（装配入口在 enabled 检查之内）。"""
    store = make_store(tmp_path)
    seed_conflict_pair(store)
    loop = make_loop(
        tmp_path, turns=zero_output_turns(), prior_config={"enabled": False}
    )
    events = list(loop.events())
    assert events[-1]["type"] == "finish"
    assert CONFLICT_GUARD_HEADER not in loop.provider.calls[0][0].content


# ---- ⑥ 裁决留痕是普通记忆：memory.search 可检索 -------------------------------


def test_arbitration_trace_note_retrievable_via_memory_search(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    seed_conflict_pair(store)

    assert cli.main(resolve_argv(str(store.root), "C-0001", "N-0002")) == 0
    trace = [
        n
        for n in store.list_notes(status=None)
        if n.meta.extra.get("ledger") == "arbitration"
    ]
    assert len(trace) == 1

    # memory.search（MCP memory_search 的逻辑层）：留痕笔记是普通记忆，双通道可召回
    service = WikiService(store.root, config=MCP_CONFIG)
    payload = service.search("冲突仲裁留痕")
    assert trace[0].id in [r["note_id"] for r in payload["results"]]


@pytest.mark.parametrize(
    "question", ["GLM-5.3 支持多大的上下文窗口？", "GLM-5-3 的上下文窗口多大"]
)
def test_question_entity_key_matching_tolerates_spelling(question: str, tmp_path: Path) -> None:
    """实体匹配键是 slugify 归一键：写法变体（空格/点差）仍相交；与 ingest 同身份定义。"""
    store = make_store(tmp_path)
    seed_conflict_pair(store)
    assert len(retrieve_conflict_guards(question, store)) == 1
