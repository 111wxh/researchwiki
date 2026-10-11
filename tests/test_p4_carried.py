"""P2 移交清账（task-19）测试：写入 id 形态护栏 / SearchIndex 连接关闭 / Prior P- 编号。

三组用例对应 task-19 简报的三个遗留项（台账 progress.md P2 节移交）：

① F7 写入 id 形态护栏：WikiStore.save_note / save_meta 的 note.id 必须匹配
   ``^N-\\d{4,}$``（save_conflict 的 conflict id 同理 ``^C-\\d{4,}$``），违规抛
   ValueError；``{4,}`` 允许 5+ 位防编号溢出，内部生成（N-%04d / C-%04d）与
   外部传入同守。
② SearchIndex 连接关闭：AgentLoop.events() 的 run 完成路径（含异常、生成器被
   close）经 try/finally 关闭 Prior 索引与模式探测索引的 SQLite 连接——server
   通路与直接调用通路共用 events()，一处收尾两路覆盖。
③ Prior 条目编号 P- 前缀：Prior 条目从 ``### [n]`` 改为 ``### [P-n]``，消除与
   报告来源引用 ``[n]`` 的形态碰撞。

全部零网络、零模型调用（ScriptedProvider + MockSearch + MockEmbeddingProvider，
脚手架与 tests/test_ledger.py 同构）。
"""

import json
import re
import sqlite3
from pathlib import Path

import pytest

from researchwiki.llm.accounting import TokenAccountant
from researchwiki.llm.provider import ScriptedProvider, StreamEvent
from researchwiki.loop.agent_loop import AgentLoop
from researchwiki.tools import MockSearch
from researchwiki.wiki.embeddings import MockEmbeddingProvider
from researchwiki.wiki.frontmatter import NoteMeta
from researchwiki.wiki.index import SearchIndex
from researchwiki.wiki.prior import retrieve_priors
from researchwiki.wiki.store import WikiStore

QUESTION = "agent 记忆方案对比"
PLAN_TEXT = "- 任务一：检索主流方案\n- 任务二：蒸馏笔记\n"
REPORT_TEXT = "## 研究报告\n\n本轮证据不足，无法给出结论。\n"
EMPTY_DISTILL_JSON = json.dumps({"notes": [], "conflicts": []}, ensure_ascii=False)


# ---- 脚手架（与 tests/test_ledger.py 同构）----------------------------------


def text_turn(text: str) -> list[StreamEvent]:
    return [StreamEvent(type="text_delta", delta=text)]


class FakeRouter:
    def __init__(self, strong: ScriptedProvider) -> None:
        self._providers = {"strong": strong}

    def get(self, tier: str) -> ScriptedProvider:
        if tier not in self._providers:
            raise AssertionError(f"测试未配置 {tier} 档 Provider，却被请求了")
        return self._providers[tier]


def zero_output_turns() -> list[list[StreamEvent]]:
    """零产出 run：无工具调用、蒸馏 0 条——最短闭环，专注收尾行为。"""
    return [
        text_turn(PLAN_TEXT),
        text_turn("没有检索到可用来源，研究结束。"),
        text_turn(EMPTY_DISTILL_JSON),
        text_turn(REPORT_TEXT),
    ]


def make_loop(tmp_path: Path, *, turns: list[list[StreamEvent]], **kwargs) -> AgentLoop:
    return AgentLoop(
        QUESTION,
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


def make_store(tmp_path: Path) -> WikiStore:
    return WikiStore(tmp_path / "wiki-data")


def conn_closed(index: SearchIndex) -> bool:
    """连接状态断言：已关闭的 sqlite3 连接执行任何语句都会抛 ProgrammingError。"""
    try:
        index._conn.execute("SELECT 1")
    except sqlite3.ProgrammingError:
        return True
    return False


# ---- ① 写入 id 形态护栏（F7）------------------------------------------------


def test_save_meta_rejects_illegal_note_ids(tmp_path: Path) -> None:
    """save_meta 传非法形态 id（note-1 / N1 / 空串 / 三位数）必须 ValueError。"""
    store = make_store(tmp_path)
    for bad in ("note-1", "N1", "", "N-123", "n-0001", "N-0001x"):
        with pytest.raises(ValueError, match="id 形态非法"):
            store.save_meta(NoteMeta(id=bad, title="t"), "正文")
    # 非法 id 零落盘（不存在 notes/.md 之类的畸形文件）
    assert list(store.notes_dir.glob("*")) == []


def test_save_note_rejects_illegal_note_ids(tmp_path: Path) -> None:
    """save_note 显式传非法 note_id 同样拒绝（内部生成与外部传入同守）。"""
    store = make_store(tmp_path)
    for bad in ("note-1", "N1", "N-123"):
        with pytest.raises(ValueError, match="id 形态非法"):
            store.save_note("正文", note_id=bad, title="t")
    assert not (store.notes_dir / "note-1.md").exists()
    assert not (store.notes_dir / "N-123.md").exists()


def test_save_note_accepts_valid_and_generated_ids(tmp_path: Path) -> None:
    """合法形态照常写入：自动生成（N-0001）、显式四位、5 位溢出形态。"""
    store = make_store(tmp_path)
    auto = store.save_note("正文一", title="t")  # 自动生成：N-%04d
    assert auto.id == "N-0001"
    explicit = store.save_note("正文二", note_id="N-0002", title="t")
    assert explicit.id == "N-0002"
    # {4,} 允许 5+ 位：编号溢出万位后 %04d 自然变宽，护栏不挡
    wide = store.save_note("正文三", note_id="N-12345", title="t")
    assert wide.id == "N-12345"


def test_save_conflict_rejects_illegal_conflict_ids(tmp_path: Path) -> None:
    """save_conflict 的 conflict id 同守 ^C-\\d{4,}$：非法拒绝、合法与自动生成照常。"""
    store = make_store(tmp_path)
    for bad in ("C-1", "C-123", "conflict-1", "C-0001x"):
        with pytest.raises(ValueError, match="id 形态非法"):
            store.save_conflict("问题", {"a": 1}, {"b": 2}, conflict_id=bad)
    auto = store.save_conflict("问题", {"a": 1}, {"b": 2})
    assert auto.id == "C-0001"  # 自动生成（C-%04d）天然合法
    explicit = store.save_conflict("问题", {"a": 1}, {"b": 2}, conflict_id="C-0009")
    assert explicit.id == "C-0009"


# ---- ② run 结束关闭 SearchIndex 连接 ----------------------------------------


def test_agent_loop_run_closes_prior_index(tmp_path: Path) -> None:
    """run 正常完成：Prior 索引的 SQLite 连接必须已关闭（直接调用通路）。"""
    loop = make_loop(tmp_path, turns=zero_output_turns())
    index = loop.prior_index
    assert index is not None
    assert not conn_closed(index)
    list(loop.events())
    assert conn_closed(index), "run 完成后 Prior 索引连接必须已关闭"


def test_agent_loop_run_closes_probe_index_without_prior(tmp_path: Path) -> None:
    """Prior 禁用 + 模式判定启用：探测索引（_policy_index）也要被关闭。"""
    loop = make_loop(
        tmp_path,
        turns=zero_output_turns(),
        prior_config={"enabled": False},
        retrieval_config={"enabled": True},
    )
    assert loop.prior_index is None
    list(loop.events())
    assert loop._policy_index is not None
    assert conn_closed(loop._policy_index), "run 完成后探测索引连接必须已关闭"


def test_agent_loop_exception_path_still_closes_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run 中途异常：异常照常抛出，索引连接仍被 finally 关闭。"""
    loop = make_loop(tmp_path, turns=zero_output_turns())
    index = loop.prior_index
    assert index is not None

    def boom(self: AgentLoop) -> None:
        raise RuntimeError("Prior 检索爆炸")

    monkeypatch.setattr(AgentLoop, "_retrieve_priors", boom)
    with pytest.raises(RuntimeError):
        list(loop.events())
    assert conn_closed(index), "异常关闭路径同样必须释放索引连接"


def test_agent_loop_generator_close_closes_index(tmp_path: Path) -> None:
    """生成器被调用方提前 close（server 客户端断连的等价路径）：连接也要关闭。"""
    loop = make_loop(tmp_path, turns=zero_output_turns())
    index = loop.prior_index
    assert index is not None
    gen = loop.events()
    next(gen)  # start
    next(gen)  # 进入 try（_events 的首个 yield）
    gen.close()
    assert conn_closed(index), "生成器被 close 时必须释放索引连接"


# ---- ③ Prior 条目编号 P- 前缀 ------------------------------------------------


def test_prior_block_uses_p_prefix_numbering(tmp_path: Path) -> None:
    """Prior 条目编号为 [P-n]：不再与报告来源引用 [n] 同形态。"""
    store = make_store(tmp_path)
    store.save_note(
        "agent 记忆方案对比：Letta 用后台 subagent 整理记忆。",
        note_id="N-0001",
        title="Letta（历史）",
    )
    index = SearchIndex(
        tmp_path / "wiki-data",
        embedding=MockEmbeddingProvider(dim=512),
        tokenizer="trigram",
    )
    index.rebuild(store)
    ctx = retrieve_priors(QUESTION, store, index)
    text = ctx.format()
    assert "### [P-1] N-0001 Letta（历史）" in text
    # 条目行不得再是 "### [数字]" 形态（报告来源引用 [n] 的碰撞消除）
    assert not re.search(r"### \[\d", text)


def test_agent_loop_plan_message_uses_p_prefix(tmp_path: Path) -> None:
    """loop 通路：注入 plan 的 Prior 块条目行同样为 [P-n] 形态。"""
    store = make_store(tmp_path)
    store.save_note(
        "agent 记忆方案对比：MemGPT 用分页管理上下文。",
        note_id="N-0001",
        title="MemGPT（历史）",
    )
    loop = make_loop(tmp_path, turns=zero_output_turns())
    list(loop.events())
    plan_message = loop.provider.calls[0][0].content
    assert "### [P-1] N-0001 MemGPT（历史）" in plan_message
