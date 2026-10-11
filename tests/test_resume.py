"""P4b 支撑设施 T7（task-21）loop 级测试：resume 跳过已完成 step、笔记幂等、
keys_seen、kill_resume 冒烟。

对应 task-21 简报 Step 1 的用例 ③④⑤⑥
（①② 的单元部分在 tests/test_checkpoint.py）：

③ resume 跳过 completed_step_ids：恢复进程的 provider 调用只覆盖未完成 step
   （计数断言），history 由侧车文件重建、来源池从工具结果复原，budget 基线衔接
   （checkpoint 基线 + 恢复新增 == 一次性基线；恢复 run 的 metrics 只记新消耗）；
④ 同 run_step_key 笔记不重复创建：checkpoint 滞后一拍（笔记已入库、checkpoint 未记
   distill）时，恢复 run 重跑蒸馏调用但不重复入库（写-崩溃窗口兜底）；
⑤ llm_keys_seen 幂等：已完成调用的请求哈希不重复出现（mock provider 调用计数断言）；
⑥ kill_resume_smoke 四闸在脚本内绿（subprocess 用例跑通短版本，真实进程 kill）。

进程内"kill"模拟手法：消费事件流直到目标 step 的 checkpoint 落盘后 close 生成器
（finally 收尾照跑）；真实 kill -9 / taskkill /F 的进程级证据由 ⑥ 的 subprocess 冒烟覆盖。
全部零网络、零真实模型（ScriptedProvider + MockSearch + MockEmbeddingProvider）。
"""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx

from researchwiki.llm.accounting import TokenAccountant
from researchwiki.llm.provider import ScriptedProvider, StreamEvent
from researchwiki.llm.replay import request_hash
from researchwiki.loop import checkpoint
from researchwiki.loop.agent_loop import AgentLoop
from researchwiki.tools import MockSearch
from researchwiki.wiki.store import WikiStore

QUESTION = "agent 长期记忆的主流方案"
PLAN_TEXT = "- 任务一：检索主流方案\n- 任务二：把结论蒸馏成笔记\n"
SEARCH_SUMMARY = "已检索到主流方案，信息足够，研究结束。"
REPORT_TEXT = "## 研究报告\n\nLetta 的后台 subagent 方案[1] 值得优先试点。\n"
FETCH_URL = "https://example.com/article"
FETCH_PAGE = "<html><body><h1>文章标题</h1>" + "正文内容。" * 60 + "</body></html>"
DISTILL_NOTE_URL = "https://github.com/letta-ai/letta"
DISTILL_JSON = json.dumps(
    {
        "notes": [
            {
                "text": "关于「agent 长期记忆」的研究结论：Letta 用后台 subagent "
                "在会话空闲期整理长期记忆，跨会话保留长期事实。",
                "entities": ["Letta"],
                "confidence": "high",
                "source_urls": [DISTILL_NOTE_URL],
            }
        ],
        "conflicts": [],
    },
    ensure_ascii=False,
)
PER_CALL_TOKENS_IN = 1200  # ScriptedProvider 默认 usage（每次调用恒定，预算衔接可精确复算）
PER_CALL_TOKENS_OUT = 180


# ---- 脚手架 ------------------------------------------------------------------


def text_turn(text: str) -> list[StreamEvent]:
    return [StreamEvent(type="text_delta", delta=text)]


def tool_turn(calls: list[dict]) -> list[StreamEvent]:
    return [StreamEvent(type="tool_calls", tool_calls=calls)]


def call(name: str, arguments: dict, *, id: str) -> dict:
    return {
        "id": id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
    }


def strong_turns() -> list[list[StreamEvent]]:
    """strong 五连剧本：plan → act:1(search) → act:2(fetch) → act:3(文本收尾) → report。"""
    return [
        text_turn(PLAN_TEXT),
        tool_turn([call("web_search", {"query": "agent 长期记忆"}, id="s1")]),
        tool_turn([call("fetch_url", {"url": FETCH_URL}, id="f1")]),
        text_turn(SEARCH_SUMMARY),
        text_turn(REPORT_TEXT),
    ]


def distill_turn() -> list[StreamEvent]:
    return [text_turn(DISTILL_JSON)]


_CHECKPOINT_ON: dict[str, Any] = {"enabled": True, "every_step": True}


class FakeRouter:
    def __init__(self, strong: ScriptedProvider, cheap: ScriptedProvider) -> None:
        self._providers = {"strong": strong, "cheap": cheap}

    def get(self, tier: str) -> ScriptedProvider:
        return self._providers[tier]


class RecordingProvider(ScriptedProvider):
    """记录每次调用收到的 system 实参（恢复 run 的来源池复原断言用）。"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.systems: list[str | None] = []

    def stream(self, messages, *, system=None, tools=None):
        self.systems.append(system)
        yield from super().stream(messages, system=system, tools=tools)


def make_loop(
    root: Path,
    *,
    strong: ScriptedProvider,
    cheap: ScriptedProvider,
    run_dir: Path | None = None,
    resume_from: Path | None = None,
    checkpoint_config: Any = _CHECKPOINT_ON,
) -> AgentLoop:
    return AgentLoop(
        QUESTION,
        router=FakeRouter(strong, cheap),
        llm_config={
            "strong": {"base_url": "https://mock"},
            "cheap": {"base_url": "https://mock-cheap"},
        },
        accountant=TokenAccountant(root / "tokens.jsonl"),
        search_provider=MockSearch(),
        wiki_root=root / "wiki-data",
        run_dir=run_dir if run_dir is not None else root / "run",
        fetch_transport=httpx.MockTransport(
            lambda request: httpx.Response(200, html=FETCH_PAGE)
        ),
        formation_config={"enabled": True},
        checkpoint_config=checkpoint_config,
        resume_from=resume_from,
    )


def read_metrics(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "run-metrics.json").read_text(encoding="utf-8"))


def _kill_and_resume(tmp_path: Path) -> dict[str, Any]:
    """基线一次性 run → 受害 run 在 act:1 checkpoint 落盘后"停止" → 恢复 run 续跑至完成。

    返回基线/受害/恢复三方的可断言产物（provider 调用、checkpoint、metrics、笔记）。
    """
    # 基线：同剧本一次性跑完（every gate 的对照）
    base_root = tmp_path / "base"
    strong0 = RecordingProvider(strong_turns(), model="mock-strong")
    cheap0 = ScriptedProvider(distill_turn(), tier="cheap", model="mock-cheap")
    list(make_loop(base_root, strong=strong0, cheap=cheap0).events())
    base_metrics = read_metrics(base_root / "run")
    base_notes = WikiStore(base_root / "wiki-data").list_notes()

    # 受害：跑到 act:1 的 checkpoint 落盘后 close 生成器（进程内 kill 模拟）
    root = tmp_path / "killed"
    run_dir = root / "run"
    strong_a = RecordingProvider(strong_turns(), model="mock-strong")
    cheap_a = ScriptedProvider(distill_turn(), tier="cheap", model="mock-cheap")
    loop_a = make_loop(root, strong=strong_a, cheap=cheap_a, run_dir=run_dir)
    victim_tools_schema = loop_a.tools_schema
    gen = loop_a.events()
    for _event in gen:
        ck = checkpoint.load(run_dir)
        if ck and "act:1" in ck["completed_step_ids"]:
            break
    gen.close()
    ck_kill = checkpoint.load(run_dir)
    assert ck_kill is not None
    assert ck_kill["completed_step_ids"] == ["plan", "act:1"]

    # 恢复：同 run_dir、resume_from 指向 checkpoint；剧本从第一个未完成 step（act:2）起
    strong_b = RecordingProvider(strong_turns()[2:], model="mock-strong")
    cheap_b = ScriptedProvider(distill_turn(), tier="cheap", model="mock-cheap")
    loop_b = make_loop(root, strong=strong_b, cheap=cheap_b, run_dir=run_dir, resume_from=run_dir)
    events_b = list(loop_b.events())
    return {
        "base_metrics": base_metrics,
        "base_notes": base_notes,
        "root": root,
        "run_dir": run_dir,
        "ck_kill": ck_kill,
        "strong_a": strong_a,
        "strong_b": strong_b,
        "cheap_b": cheap_b,
        "victim_tools_schema": victim_tools_schema,
        "events_b": events_b,
        "final_ck": checkpoint.load(run_dir),
    }


# ---- ③ resume 跳过已完成 step（计数断言 + budget 衔接）--------------------------


def test_resume_skips_completed_steps_and_reconciles_budget(tmp_path: Path) -> None:
    """恢复进程只执行未完成 step；history/来源池重建；budget 基线衔接、计量诚实。"""
    ctx = _kill_and_resume(tmp_path)
    run_dir = ctx["run_dir"]
    assert ctx["events_b"][-1]["type"] == "finish"

    # 闸2（进程内版）：已完成 step 不重复执行——恢复进程只调 act:2 / act:3 / report
    assert len(ctx["strong_b"].calls) == 3
    assert len(ctx["cheap_b"].calls) == 1  # 蒸馏未完成 → 恢复进程补跑（调用 1 次）

    # history 由侧车重建：恢复进程第一次调用收到断点消息态（问题/计划/act:1 轮）
    first_call = ctx["strong_b"].calls[0]
    assert [m.role for m in first_call] == ["user", "assistant", "assistant", "tool"]
    assert first_call[0].content == QUESTION
    # 来源池从工具结果复原：报告 system 提示带恢复出的来源（否则蒸馏候选会因缺来源被拒）
    assert DISTILL_NOTE_URL in ctx["strong_b"].systems[-1]

    # 笔记无重复（闸4 进程内版）：与一次性基线一致，且带 run_step_key 幂等键
    notes = WikiStore(ctx["root"] / "wiki-data").list_notes()
    assert len(notes) == len(ctx["base_notes"]) == 1
    assert notes[0].body.strip() == ctx["base_notes"][0].body.strip()
    assert notes[0].meta.extra.get("run_step_key") == checkpoint.run_step_key("run", "distill")

    # budget 衔接：kill 时 checkpoint 记 2 次调用（plan + act:1）；恢复后累计 == 一次性基线
    assert ctx["ck_kill"]["budget_used"]["tokens_in"] == 2 * PER_CALL_TOKENS_IN
    final_ck = ctx["final_ck"]
    assert final_ck is not None
    assert final_ck["budget_used"]["tokens_in"] == ctx["base_metrics"]["input_tokens"]
    assert final_ck["budget_used"]["tokens_out"] == ctx["base_metrics"]["output_tokens"]
    # 诚实计量：恢复 run 的 run-metrics 只记恢复后新消耗（14 字段契约不变，口径见报告）
    metrics_b = read_metrics(run_dir)
    assert metrics_b["input_tokens"] == (
        ctx["base_metrics"]["input_tokens"] - ctx["ck_kill"]["budget_used"]["tokens_in"]
    )
    assert metrics_b["output_tokens"] == (
        ctx["base_metrics"]["output_tokens"] - ctx["ck_kill"]["budget_used"]["tokens_out"]
    )

    # 步进走完全程：completed 以 finalize 收尾，契约字段齐全
    assert final_ck["completed_step_ids"] == [
        "plan", "act:1", "act:2", "act:3", "distill", "report", "finalize",
    ]
    assert final_ck["version"] == 1
    assert final_ck["state_file_version"] == 1
    assert final_ck["run_id"] == "run"
    assert final_ck["history_summary_ref"] == "checkpoint-history.json"
    # state.md 带恢复行（滚动状态可审计）
    state = (run_dir / "state.md").read_text(encoding="utf-8")
    assert "恢复" in state and "plan" in state


# ---- ④ 同 run_step_key 笔记不重复创建（checkpoint 滞后一拍）---------------------


def test_resume_skips_note_creation_when_step_key_note_exists(tmp_path: Path, capsys) -> None:
    """写-崩溃窗口兜底：store 已有同 run_step_key 笔记而 checkpoint 未记 distill →
    恢复 run 跳过整个蒸馏阶段（同键笔记存在即 kill 前的抽取与冲突登记都已完成：
    零重复笔记、零重复冲突、零重复扣费）。"""
    root = tmp_path
    run_dir = root / "run"
    store = WikiStore(root / "wiki-data")
    key = checkpoint.run_step_key("run", "distill")
    store.save_note("已入库的研究结论笔记正文足够长", extra={"run_step_key": key})

    strong = ScriptedProvider(strong_turns(), model="mock-strong")
    cheap = ScriptedProvider(distill_turn(), tier="cheap", model="mock-cheap")
    # checkpoint 文件不存在（滞后/丢失），但 resume_from 显式给出 → 蒸馏前查同键笔记
    loop = make_loop(root, strong=strong, cheap=cheap, run_dir=run_dir, resume_from=run_dir)
    events = list(loop.events())

    assert events[-1]["type"] == "finish"
    # 蒸馏阶段整体跳过：抽取的模型调用也不发生（零重复扣费；那次调用在 kill 前
    # 已完成，只是 checkpoint 滞后一拍没记上）
    assert len(cheap.calls) == 0
    assert len(strong.calls) == 5  # plan/act/report 照常（store 无断点可跳）
    # 但已完成写入不重复创建：store 仍只有 1 条，且不发 data-note 事件
    notes = store.list_notes()
    assert len(notes) == 1
    assert notes[0].meta.extra.get("run_step_key") == key
    assert not [e for e in events if e["type"] == "data-note"]
    # 告警可审计；distill 记入 completed（后续 save 携带）
    assert "run_step_key" in capsys.readouterr().err
    final_ck = checkpoint.load(run_dir)
    assert final_ck is not None
    assert "distill" in final_ck["completed_step_ids"]
    assert final_ck["notes_created"] == [notes[0].id]


# ---- ⑤ llm_keys_seen 幂等（纵深防御）-------------------------------------------


def test_resume_llm_keys_seen_no_completed_call_repeated(tmp_path: Path) -> None:
    """keys_seen 采集已发调用的请求哈希（与 replay.request_hash 同约定）；恢复后
    新调用的哈希不与已完成调用重复（同一请求不重复扣费的纵深证据）。"""
    ctx = _kill_and_resume(tmp_path)
    ck_kill = ctx["ck_kill"]
    final_ck = ctx["final_ck"]
    assert final_ck is not None

    pre_keys = ck_kill["llm_keys_seen"]
    assert len(pre_keys) == 2  # plan + act:1
    # 哈希值可复算：受害进程前两次调用的请求哈希
    assert pre_keys[0] == request_hash(
        "mock-strong", ctx["strong_a"].calls[0], system=ctx["strong_a"].systems[0], tools=None
    )
    assert pre_keys[1] == request_hash(
        "mock-strong",
        ctx["strong_a"].calls[1],
        system=ctx["strong_a"].systems[1],
        tools=ctx["victim_tools_schema"],
    )

    final_keys = final_ck["llm_keys_seen"]
    # 已完成调用的哈希原样保留在表头，恢复新增追加其后
    assert final_keys[: len(pre_keys)] == pre_keys
    new_keys = final_keys[len(pre_keys):]
    # 恢复进程的每次调用（strong 3 + cheap 1）各记一条，且与已完成调用零交集
    assert len(new_keys) == len(ctx["strong_b"].calls) + len(ctx["cheap_b"].calls)
    assert not (set(new_keys) & set(pre_keys))
    assert len(final_keys) == len(set(final_keys))  # 无重复记账


# ---- ⑥ kill_resume_smoke 四闸（subprocess，真实进程 kill）-----------------------


ROOT = Path(__file__).resolve().parents[1]


def _load_smoke() -> ModuleType:
    """按路径加载 scripts/kill_resume_smoke.py（tests/test_cold_warm_smoke.py 同款）。"""
    spec = importlib.util.spec_from_file_location(
        "kill_resume_smoke", ROOT / "scripts/kill_resume_smoke.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["kill_resume_smoke"] = module
    spec.loader.exec_module(module)
    return module


def test_kill_resume_smoke_four_gates_subprocess(tmp_path: Path) -> None:
    """子进程起 run → 真实 kill → 新进程 resume_from 续跑完成；脚本内四道硬闸全绿。"""
    smoke = _load_smoke()
    json_path = tmp_path / "summary.json"
    rc = smoke.main(["--root", str(tmp_path / "smoke"), "--json", str(json_path)])
    assert rc == smoke.EXIT_PASS
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert set(payload["gates"]) == {
        "task_completed",
        "no_step_redone",
        "notes_not_duplicated",
        "zero_api_cost",
    }
    assert all(payload["gates"].values())
    # kill 证据：受害进程的 run-metrics 未落盘（finally 未执行 = 硬 kill）
    assert payload["victim"]["run_metrics_missing"] is True
    assert payload["victim"]["killed_at_step"] in payload["victim"]["completed_step_ids"]
    # 四闸计数证据：受害 + 恢复 ≤ 一次性基线（闸2）
    for tier in ("strong_calls", "cheap_calls"):
        assert (
            payload["victim"][tier] + payload["resume"][tier] <= payload["baseline"][tier]
        )


# ---- 补充：逃生阀 / 不接线 / 损坏兜底 -------------------------------------------


def test_checkpoint_disabled_writes_nothing(tmp_path: Path) -> None:
    """[checkpoint] 段缺失（None = 不接线）：整场 run 零 checkpoint 留痕，行为与接入前一致。"""
    root = tmp_path
    strong = ScriptedProvider(strong_turns(), model="mock-strong")
    cheap = ScriptedProvider(distill_turn(), tier="cheap", model="mock-cheap")
    loop = make_loop(root, strong=strong, cheap=cheap, checkpoint_config=None)
    events = list(loop.events())
    assert events[-1]["type"] == "finish"
    run_dir = root / "run"
    assert not (run_dir / "checkpoint.json").exists()
    assert not (run_dir / "checkpoint-history.json").exists()
    assert len(strong.calls) == 5


def test_every_step_false_saves_only_phase_boundaries(tmp_path: Path) -> None:
    """every_step=false：act 步不落 checkpoint，只记阶段边界（plan/distill/report/finalize）。"""
    root = tmp_path
    strong = ScriptedProvider(strong_turns(), model="mock-strong")
    cheap = ScriptedProvider(distill_turn(), tier="cheap", model="mock-cheap")
    loop = make_loop(
        root, strong=strong, cheap=cheap, checkpoint_config={"enabled": True, "every_step": False}
    )
    list(loop.events())
    final_ck = checkpoint.load(root / "run")
    assert final_ck is not None
    assert final_ck["completed_step_ids"] == ["plan", "distill", "report", "finalize"]


def test_resume_with_corrupt_checkpoint_reruns_full(tmp_path: Path, capsys) -> None:
    """checkpoint 损坏 → 宁可重跑不可带病恢复：全流程照跑至完成 + stderr 告警。"""
    root = tmp_path
    run_dir = root / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "checkpoint.json").write_text("{corrupted", encoding="utf-8")
    strong = ScriptedProvider(strong_turns(), model="mock-strong")
    cheap = ScriptedProvider(distill_turn(), tier="cheap", model="mock-cheap")
    loop = make_loop(root, strong=strong, cheap=cheap, run_dir=run_dir, resume_from=run_dir)
    events = list(loop.events())
    assert events[-1]["type"] == "finish"
    assert len(strong.calls) == 5  # 从头跑：plan/act:1/act:2/act:3/report
    assert len(WikiStore(root / "wiki-data").list_notes()) == 1
    assert "checkpoint" in capsys.readouterr().err


def test_resume_killed_between_terminal_act_and_distill(tmp_path: Path) -> None:
    """在最后一个 act step（文本收尾）之后、蒸馏之前被 kill：恢复时不再重入 act 循环，
    直接从蒸馏继续（终端总结从 history 末条识别）。"""
    root = tmp_path / "killed"
    run_dir = root / "run"
    strong_a = RecordingProvider(strong_turns(), model="mock-strong")
    cheap_a = ScriptedProvider(distill_turn(), tier="cheap", model="mock-cheap")
    loop_a = make_loop(root, strong=strong_a, cheap=cheap_a, run_dir=run_dir)
    gen = loop_a.events()
    for _event in gen:
        ck = checkpoint.load(run_dir)
        if ck and "act:3" in ck["completed_step_ids"]:
            break
    gen.close()

    strong_b = RecordingProvider(strong_turns()[4:], model="mock-strong")  # 只剩 report
    cheap_b = ScriptedProvider(distill_turn(), tier="cheap", model="mock-cheap")
    loop_b = make_loop(root, strong=strong_b, cheap=cheap_b, run_dir=run_dir, resume_from=run_dir)
    events = list(loop_b.events())
    assert events[-1]["type"] == "finish"
    # act 循环零重入：恢复进程只调 report + 蒸馏
    assert len(strong_b.calls) == 1
    assert len(cheap_b.calls) == 1
    final_ck = checkpoint.load(run_dir)
    assert final_ck is not None
    assert final_ck["completed_step_ids"] == [
        "plan", "act:1", "act:2", "act:3", "distill", "report", "finalize",
    ]


# ---- dispatch 场景恢复：子 agent 候选从侧车重建（评审 Important 修复回归）--------


def test_resume_rebuilds_pending_candidates_from_dispatch(tmp_path: Path) -> None:
    """含 dispatch_research 的 act step 完成后被 kill：恢复 run 从侧车 dispatch 返回体
    （notes 键）重建 pending_candidates（与 _run_subagent 同口径）——恢复 run 的蒸馏
    候选与一次性基线一致，子 agent 笔记不再靠 _tool_digest 重推导近似还原（静默少记）。

    蒸馏脚本给空抽取结果：候选唯一来源是子 agent 笔记，重建缺失必然使笔记集不一致。
    """
    sub_json = json.dumps(
        {
            "findings": "子问题结论：Letta 的记忆固化走后台 subagent。",
            "notes": [
                {
                    "text": "子 agent 笔记：Letta 用后台 subagent 整理长期记忆。",
                    "entities": ["Letta"],
                    "confidence": "high",
                }
            ],
            "sources": [{"url": DISTILL_NOTE_URL, "title": "Letta (MemGPT)"}],
        },
        ensure_ascii=False,
    )
    empty_distill = text_turn(json.dumps({"notes": [], "conflicts": []}, ensure_ascii=False))
    strong_script = [
        text_turn(PLAN_TEXT),
        tool_turn([call("dispatch_research", {"topic": "上下文压缩"}, id="c1")]),
        tool_turn([call("web_search", {"query": "agent 长期记忆"}, id="s2")]),
        text_turn(SEARCH_SUMMARY),
        text_turn(REPORT_TEXT),
    ]
    cheap_script = [text_turn(sub_json), distill_turn()]  # 子 agent 单步 + 蒸馏

    # 基线：一次性跑完；笔记集 = 子 agent 笔记（蒸馏零抽取）
    base_root = tmp_path / "base"
    base_loop = make_loop(
        base_root,
        strong=ScriptedProvider(strong_script, model="mock-strong"),
        cheap=ScriptedProvider(cheap_script, tier="cheap", model="mock-cheap"),
    )
    list(base_loop.events())
    base_notes = WikiStore(base_root / "wiki-data").list_notes()
    assert len(base_notes) == 1
    assert base_notes[0].body.strip().startswith("子 agent 笔记：")

    # 受害：act:1（dispatch 步）checkpoint 落盘后中断（手法同 _kill_and_resume）。
    # kill 落在 act:1 与 act:2 之间：蒸馏从未执行，候选丢失只能在恢复时暴露。
    root = tmp_path / "killed"
    run_dir = root / "run"
    strong_a = ScriptedProvider(strong_script, model="mock-strong")
    cheap_a = ScriptedProvider(cheap_script, tier="cheap", model="mock-cheap")
    loop_a = make_loop(root, strong=strong_a, cheap=cheap_a, run_dir=run_dir)
    gen = loop_a.events()
    for _event in gen:
        ck = checkpoint.load(run_dir)
        if ck and "act:1" in ck["completed_step_ids"]:
            break
    gen.close()
    ck_kill = checkpoint.load(run_dir)
    assert ck_kill is not None
    assert ck_kill["completed_step_ids"] == ["plan", "act:1"]
    # 子 agent 在 act:1 内部跑过（cheap 第 1 轮），蒸馏未到（第 2 轮未消费）
    assert len(cheap_a.calls) == 1
    assert len(loop_a.pending_candidates) == 1  # kill 前候选在内存里，随进程消失

    # 恢复：剧本从 act:2 起；蒸馏空抽取 → 候选唯一来源是重建的 pending_candidates
    strong_b = ScriptedProvider(strong_script[2:], model="mock-strong")
    cheap_b = ScriptedProvider([empty_distill], tier="cheap", model="mock-cheap")
    loop_b = make_loop(root, strong=strong_b, cheap=cheap_b, run_dir=run_dir, resume_from=run_dir)
    events_b = list(loop_b.events())

    assert events_b[-1]["type"] == "finish"
    # pending_candidates 从侧车 dispatch 返回体重建（与 _run_subagent 同口径）
    assert len(loop_b.pending_candidates) == 1
    rebuilt = loop_b.pending_candidates[0]
    assert rebuilt.text.startswith("子 agent 笔记：")
    assert rebuilt.source_urls == [DISTILL_NOTE_URL]
    # 子 agent 不重跑（dispatch 步已跳过）：cheap 只有蒸馏 1 次调用
    assert len(cheap_b.calls) == 1
    assert len(strong_b.calls) == 3  # act:2 / act:3 / report
    # 恢复 run 的笔记集与一次性基线一致（静默少记的核心断言）
    notes_b = WikiStore(root / "wiki-data").list_notes()
    assert [n.body.strip() for n in notes_b] == [n.body.strip() for n in base_notes]
    assert [e["data"]["text"] for e in events_b if e["type"] == "data-note"] == [
        "子 agent 笔记：Letta 用后台 subagent 整理长期记忆。"
    ]
