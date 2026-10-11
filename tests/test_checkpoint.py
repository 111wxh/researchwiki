"""P4b 支撑设施 T7（task-21）单元测试：checkpoint 原子读写 + 幂等键 + 请求哈希。

六组用例对应 task-21 简报 Step 1（③④⑤⑥ 的 loop 级部分在 tests/test_resume.py）：

① checkpoint 原子写：save 后 load 返回原值；写中断残留的 .tmp 文件不影响 load；
② load 损坏文件返回 None 并向 stderr 告警（缺文件静默返回 None——宁可重跑不可带病恢复）；
③ run_step_key 步幂等键：确定性 + 同键笔记查找（status=None 全量扫描，含 merged/superseded）；
④ KeyRecordingProvider：调用请求哈希（sha256(model+messages+tools)，与 llm/replay.py 的
   request_hash 同一约定）采集进 sink，透传流式事件与调用记录；
⑤ history 侧车文件：Message 序列化/反序列化往返；损坏形态返回 None；
⑥ [checkpoint] 段解析：段缺失 = 不接线（None）；enabled=false 逃生阀；every_step 开关。

全部零网络、零真实模型。
"""

import json
from pathlib import Path

from researchwiki.llm.provider import Message, ScriptedProvider, StreamEvent
from researchwiki.loop import checkpoint
from researchwiki.wiki.store import WikiStore

# ---- ① 原子写：save/load 往返 + tmp 残留不影响 load -----------------------------


def _payload(**overrides) -> dict:
    """最小合法 payload（字段契约见 checkpoint 模块 docstring）。"""
    base = {
        "version": 1,
        "run_id": "run",
        "mode": "",
        "completed_step_ids": ["plan", "act:1"],
        "completed_subtask_ids": [],
        "notes_created": [],
        "llm_keys_seen": ["a" * 64, "b" * 64],
        "budget_used": {"tokens_in": 2400, "tokens_out": 360},
        "history_summary_ref": "checkpoint-history.json",
        "state_file_version": 1,
        "updated_at": "2026-10-11T00:00:00+00:00",
    }
    base.update(overrides)
    return base


def test_save_then_load_roundtrip(tmp_path: Path) -> None:
    """save → load 原值往返；checkpoint.json 落在 run_dir 下。"""
    run_dir = tmp_path / "run"
    payload = _payload()
    path = checkpoint.save(run_dir, payload)
    assert path == run_dir / "checkpoint.json"
    assert path.is_file()
    assert checkpoint.load(run_dir) == payload


def test_leftover_tmp_file_does_not_affect_load(tmp_path: Path) -> None:
    """写中断模拟：目录里残留半个 .tmp 文件（原子写的临时产物）不影响 load。"""
    run_dir = tmp_path / "run"
    checkpoint.save(run_dir, _payload())
    # 模拟中断：rename 前进程死掉，留下写了一半的 tmp 文件
    (run_dir / ".checkpoint.json.deadbeef.tmp").write_text('{"version": 99, "ru', encoding="utf-8")
    assert checkpoint.load(run_dir) == _payload()
    # 同理：history 侧车的 tmp 残留不影响读取
    checkpoint.save_history(run_dir, ["plan"], [Message(role="user", content="q")])
    (run_dir / ".checkpoint-history.zzz.tmp").write_text("{bad json", encoding="utf-8")
    record = checkpoint.load_history(run_dir)
    assert record is not None and record["completed_step_ids"] == ["plan"]


# ---- ② 损坏返回 None + 告警；缺文件静默 ----------------------------------------


def test_load_corrupt_returns_none_with_warning(tmp_path: Path, capsys) -> None:
    """损坏（解析失败 / 非 dict）→ None + stderr 告警，不 crash。"""
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "checkpoint.json").write_text("{broken json", encoding="utf-8")
    assert checkpoint.load(run_dir) is None
    assert "checkpoint" in capsys.readouterr().err

    # 合法 JSON 但不是对象（如被截断成数组）同样视为损坏
    (run_dir / "checkpoint.json").write_text("[1, 2]", encoding="utf-8")
    assert checkpoint.load(run_dir) is None
    assert "checkpoint" in capsys.readouterr().err


def test_load_missing_file_is_silent_none(tmp_path: Path, capsys) -> None:
    """缺文件 → None 且零告警（"没有 checkpoint" 是正常形态，不是故障）。"""
    assert checkpoint.load(tmp_path / "missing") is None
    assert capsys.readouterr().err == ""


def test_load_history_corrupt_returns_none_with_warning(tmp_path: Path, capsys) -> None:
    """history 侧车损坏 → None + 告警；缺文件 → None 静默。"""
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "checkpoint-history.json").write_text('{"version": 1, "mess', encoding="utf-8")
    assert checkpoint.load_history(run_dir) is None
    assert "checkpoint-history" in capsys.readouterr().err
    assert checkpoint.load_history(tmp_path / "missing") is None
    assert capsys.readouterr().err == ""


# ---- ③ run_step_key 步幂等键 ----------------------------------------------------


def test_run_step_key_deterministic() -> None:
    """同 (run_id, step_id) 恒同键；不同 run/step 键不同。"""
    assert checkpoint.run_step_key("run", "distill") == checkpoint.run_step_key("run", "distill")
    assert checkpoint.run_step_key("run", "distill") != checkpoint.run_step_key("run", "report")
    assert checkpoint.run_step_key("run", "distill") != checkpoint.run_step_key("run2", "distill")
    assert "distill" in checkpoint.run_step_key("run", "distill")


def test_find_step_key_note_ids_scans_all_statuses(tmp_path: Path) -> None:
    """同键笔记查找：跨 status 全量扫描（含 superseded 留痕），无 extra / 异键不命中。"""
    store = WikiStore(tmp_path / "wiki")
    hit_active = store.save_note(
        "研究结论甲的正文足够长", extra={"run_step_key": "run::distill"}
    )
    store.save_note("研究结论乙的正文足够长", extra={"run_step_key": "other::distill"})
    store.save_note("研究结论丙的正文足够长")  # 无 extra
    hit_superseded = store.save_note(
        "研究结论丁的正文足够长",
        status="superseded",
        superseded_by=hit_active.id,
        extra={"run_step_key": "run::distill"},
    )
    found = checkpoint.find_step_key_note_ids(store, "run::distill")
    assert set(found) == {hit_active.id, hit_superseded.id}
    assert checkpoint.find_step_key_note_ids(store, "run::missing") == []


# ---- ④ KeyRecordingProvider：请求哈希采集 ---------------------------------------


def text_turn(text: str) -> list[StreamEvent]:
    return [StreamEvent(type="text_delta", delta=text)]


def test_key_recording_provider_records_request_hash() -> None:
    """每次 stream 调用把请求哈希记进 sink（与 replay.request_hash 同约定）。"""
    from researchwiki.llm.replay import request_hash

    inner = ScriptedProvider([text_turn("回复")], model="mock-strong")
    sink: list[str] = []
    provider = checkpoint.KeyRecordingProvider(inner, sink)
    messages = [Message(role="user", content="问题")]
    tools = [{"type": "function", "function": {"name": "web_search", "parameters": {}}}]
    events = list(provider.stream(messages, system="系统提示", tools=tools))

    assert [e.type for e in events] == ["text_delta", "usage"]  # 事件透传（含自动补的 usage）
    assert len(sink) == 1
    assert sink[0] == request_hash("mock-strong", messages, system="系统提示", tools=tools)
    # Protocol 兼容：model/tier 透传，调用记录透传
    assert provider.model == "mock-strong"
    assert provider.tier == "strong"
    assert len(inner.calls) == 1

    # 同请求再发一次 → 两条哈希（重复调用在 keys_seen 里可被识别）
    list(provider.stream(messages, system="系统提示", tools=tools))
    assert len(sink) == 2 and sink[0] == sink[1]


# ---- ⑤ history 侧车：序列化往返 + 损坏形态 ---------------------------------------


def _sidecar_messages() -> list[Message]:
    return [
        Message(role="user", content="研究问题"),
        Message(role="assistant", content="计划"),
        Message(
            role="assistant",
            content="",
            tool_calls=[
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": "{}"},
                }
            ],
        ),
        Message(role="tool", tool_call_id="c1", name="web_search", content="结果"),
    ]


def test_history_sidecar_roundtrip(tmp_path: Path) -> None:
    """save_history → load_history → deserialize_history 还原出等价 Message 列表。"""
    run_dir = tmp_path / "run"
    messages = _sidecar_messages()
    checkpoint.save_history(run_dir, ["plan", "act:1"], messages)
    record = checkpoint.load_history(run_dir)
    assert record is not None
    assert record["completed_step_ids"] == ["plan", "act:1"]
    assert record["version"] == 1
    assert record["updated_at"]
    assert checkpoint.deserialize_history(record["messages"]) == messages


def test_deserialize_history_rejects_malformed() -> None:
    """畸形形态（非 dict / 缺 content / role 非法 / tool_calls 非列表）→ None，不抛异常。"""
    assert checkpoint.deserialize_history([]) == []
    assert checkpoint.deserialize_history(["nope"]) is None
    assert checkpoint.deserialize_history([{"role": "user"}]) is None  # 缺 content
    assert checkpoint.deserialize_history([{"role": "wizard", "content": "x"}]) is None
    assert (
        checkpoint.deserialize_history(
            [{"role": "assistant", "content": "", "tool_calls": "not-a-list"}]
        )
        is None
    )


# ---- ⑥ [checkpoint] 段解析 ------------------------------------------------------


def test_checkpoint_settings_from_config() -> None:
    """段缺失（None）= 不接线；缺省键 = 开 + 默认；逃生阀与 every_step 开关生效。"""
    assert checkpoint.checkpoint_settings_from_config(None) is None
    settings = checkpoint.checkpoint_settings_from_config({})
    assert settings is not None and settings.enabled is True and settings.every_step is True
    settings = checkpoint.checkpoint_settings_from_config({"enabled": False})
    assert settings is not None and settings.enabled is False
    settings = checkpoint.checkpoint_settings_from_config({"every_step": False})
    assert settings is not None and settings.every_step is False


def test_resume_state_holds_reconstruction_inputs() -> None:
    """ResumeState 只是重建输入的载体：字段齐全、可等价构造。"""
    state = checkpoint.ResumeState(
        completed_step_ids=["plan"],
        history=[Message(role="user", content="q")],
        budget_tokens_in=1200,
        budget_tokens_out=180,
        notes_created=["N-0001"],
        llm_keys_seen=["k"],
    )
    assert state.completed_step_ids == ["plan"]
    assert state.budget_tokens_in == 1200
    assert state.notes_created == ["N-0001"]


def test_checkpoint_json_contract_fields(tmp_path: Path) -> None:
    """checkpoint.json 的字段清单与简报逐字一致（version=1）。"""
    run_dir = tmp_path / "run"
    checkpoint.save(run_dir, _payload())
    on_disk = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
    assert set(on_disk) == {
        "version",
        "run_id",
        "mode",
        "completed_step_ids",
        "completed_subtask_ids",
        "notes_created",
        "llm_keys_seen",
        "budget_used",
        "history_summary_ref",
        "state_file_version",
        "updated_at",
    }
