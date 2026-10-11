"""checkpoint 与恢复（P4b 支撑设施 T7）：进程被 kill 后从断点继续。

每个研究 step（plan、每个 act 步、distill、report、finalize）完成后，把断点状态
原子落盘到 ``run_dir/checkpoint.json``；进程被 kill 后用同一 run_dir 新建 AgentLoop
并传 ``resume_from``，即可从断点继续——已完成 step 不重跑（模型调用自然不重复 →
不重复扣费）、已完成写入不重复创建（笔记带 run_step_key 幂等键，见下）。

run_dir/checkpoint.json 字段契约（version=1，简报逐字固定）::

    {version, run_id, mode, completed_step_ids, completed_subtask_ids,
     notes_created, llm_keys_seen, budget_used: {tokens_in, tokens_out},
     history_summary_ref, state_file_version, updated_at}

设计取舍（逐条写明）：

- **双文件形态（R4 裁定：内联 vs 摘要引用留给实现者，验收=四闸）**：checkpoint.json
  只存索引态（字段契约固定、体量小）；history 消息内联写入侧车文件
  ``checkpoint-history.json``，``history_summary_ref`` 指向它。每次保存先写侧车、
  后写 checkpoint.json（各自 tmp+rename 原子写）；侧车自带 completed_step_ids
  （自描述"这段 history 是这些 step 完成时的消息态"），恢复时两侧 completed 取并集
  ——进程死在两次写之间的窗口由并集兜底（侧车超前 = 该 step 确已完成）。
- **resume 语义 = 跳过已完成 step**：completed 的 plan/act:N/distill/report/finalize
  不再执行（step 内的模型调用自然不发生 → 不重复扣费）。恢复所需的 history 从侧车
  重建；来源池从 history 的工具结果按原顺序复原（报告 [n] 编号与蒸馏候选的来源
  过滤都依赖它）；budget_used 作为已耗基线带入预算熔断。
- **llm_keys_seen 是纵深防御，不是恢复机制**：记录**已发**调用的请求哈希
  （sha256(model+messages+tools)，与 llm/replay.py 的 request_hash 同一约定——
  两者可组合：给 provider 包一层 ReplayProvider 即得真"重放不扣费"）。恢复后若新
  调用的哈希已在表内，说明某个已完成 step 被重复执行——打印告警（不伪造响应，
  宁可重跑不可带病恢复）；主防线始终是"跳过已完成 step"。
- **宁可重跑不可带病恢复**：checkpoint 缺失/损坏、或已记录了完成 step 但 history
  侧车读不出来 → 不 resume，从头跑完整流程（stderr 告警）。损坏 = 解析失败即
  返回 None，绝不抛异常打断调用方。
- **写-崩溃窗口兜底**：蒸馏入库的笔记 extra 带 ``run_step_key``（产生它的蒸馏 step
  幂等键，``{run_id}::{step_id}``）；恢复 run 入库前先查同键笔记——存在即说明 kill
  前抽取与冲突登记都已完成 → 跳过整个蒸馏阶段（零重复笔记、零重复冲突、零重复
  扣费）。"已完成写入不重复创建"在 checkpoint 滞后一拍时仍成立。
- **budget_used 是本 run 谱系的累计计量**（kill 前消耗 + 恢复基线），恢复时作为
  基线带入**预算熔断**（恢复 run 不重置预算）；恢复 run 的 run-metrics 只记恢复后
  新消耗（诚实计量，14 字段契约不动）。全谱系计量 = checkpoint.budget_used +
  恢复 run 的 run-metrics 值，可复算。
- **恢复续用原 run_dir**：``resume_from`` 传入且未显式给 ``run_dir`` 时，run_dir 即
  resume_from 目录——research-plan.md / state.md / report.md 与 checkpoint 同目录，
  断点产物不被拆散；恢复 run 覆写 run-metrics.json（只含新消耗口径，见上）。
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from researchwiki.llm.provider import Message
from researchwiki.tools.fs import atomic_write_text

CHECKPOINT_VERSION = 1
STATE_FILE_VERSION = 1
CHECKPOINT_FILE = "checkpoint.json"
HISTORY_SIDECAR = "checkpoint-history.json"

_MESSAGE_ROLES = {"system", "user", "assistant", "tool"}


@dataclass
class CheckpointSettings:
    """[checkpoint] 段的解析结果（见 config.toml 注释与 checkpoint_settings_from_config）。

    enabled=false 是逃生阀：零写盘、零恢复检测；every_step=false 时 act 步不落
    checkpoint（只在阶段边界保存，kill 后的重做窗口相应变大）。
    """

    enabled: bool = True
    every_step: bool = True


@dataclass
class ResumeState:
    """从 checkpoint + history 侧车重建出的恢复输入（AgentLoop 消费）。"""

    completed_step_ids: list[str]
    history: list[Message]
    budget_tokens_in: int
    budget_tokens_out: int
    notes_created: list[str] = field(default_factory=list)
    llm_keys_seen: list[str] = field(default_factory=list)


def checkpoint_settings_from_config(
    checkpoint_config: dict[str, Any] | None,
) -> CheckpointSettings | None:
    """解析 [checkpoint] 段 → CheckpointSettings；段缺失（None）= 不接线，返回 None。"""
    if checkpoint_config is None:
        return None
    return CheckpointSettings(
        enabled=bool(checkpoint_config.get("enabled", True)),
        every_step=bool(checkpoint_config.get("every_step", True)),
    )


# ---- checkpoint.json（索引态）-------------------------------------------------


def save(run_dir: str | Path, payload: dict[str, Any]) -> Path:
    """checkpoint 原子写（tmp + rename）；返回落盘路径。"""
    path = Path(run_dir) / CHECKPOINT_FILE
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return path


def load(run_dir: str | Path) -> dict[str, Any] | None:
    """读 checkpoint；缺失返回 None（静默），损坏返回 None 并向 stderr 告警。

    "损坏" = JSON 解析失败或顶层不是对象。绝不抛异常、绝不带病恢复——调用方
    （AgentLoop）拿到 None 即按全新 run 处理（宁可重跑）。
    """
    path = Path(run_dir) / CHECKPOINT_FILE
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        print(
            f"[checkpoint] checkpoint.json 损坏，按无断点处理（宁可重跑不可带病恢复）："
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return None
    if not isinstance(payload, dict):
        print(
            "[checkpoint] checkpoint.json 顶层不是对象，按无断点处理（宁可重跑不可带病恢复）",
            file=sys.stderr,
        )
        return None
    return payload


# ---- history 侧车（checkpoint-history.json）-----------------------------------


def save_history(
    run_dir: str | Path,
    completed_step_ids: list[str],
    messages: list[Message],
) -> Path:
    """history 侧车原子写：自描述"这段 history 是这些 step 完成时的消息态"。"""
    payload = {
        "version": CHECKPOINT_VERSION,
        "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "completed_step_ids": list(completed_step_ids),
        "messages": serialize_history(messages),
    }
    path = Path(run_dir) / HISTORY_SIDECAR
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return path


def load_history(run_dir: str | Path) -> dict[str, Any] | None:
    """读 history 侧车；缺失返回 None（静默），损坏返回 None 并向 stderr 告警。"""
    path = Path(run_dir) / HISTORY_SIDECAR
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        print(
            f"[checkpoint] checkpoint-history.json 损坏，无法重建断点历史"
            f"（宁可重跑不可带病恢复）：{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return None
    if not isinstance(payload, dict):
        print(
            "[checkpoint] checkpoint-history.json 顶层不是对象，无法重建断点历史"
            "（宁可重跑不可带病恢复）",
            file=sys.stderr,
        )
        return None
    return payload


def serialize_history(messages: list[Message]) -> list[dict[str, Any]]:
    """Message 列表 → JSON 可落盘的 dict 列表（字段全量保留，None 显式写出）。"""
    return [
        {
            "role": m.role,
            "content": m.content,
            "tool_call_id": m.tool_call_id,
            "name": m.name,
            "tool_calls": m.tool_calls,
        }
        for m in messages
    ]


def deserialize_history(items: Any) -> list[Message] | None:
    """侧车 messages → Message 列表；任何畸形形态返回 None（调用方按损坏处理）。"""
    if not isinstance(items, list):
        return None
    messages: list[Message] = []
    for item in items:
        if not isinstance(item, dict):
            return None
        role = item.get("role")
        content = item.get("content")
        tool_calls = item.get("tool_calls")
        if role not in _MESSAGE_ROLES or not isinstance(content, str):
            return None
        if tool_calls is not None and not isinstance(tool_calls, list):
            return None
        tool_call_id = item.get("tool_call_id")
        name = item.get("name")
        messages.append(
            Message(
                role=role,
                content=content,
                tool_call_id=tool_call_id if isinstance(tool_call_id, str) else None,
                name=name if isinstance(name, str) else None,
                tool_calls=tool_calls,
            )
        )
    return messages


# ---- 步幂等键与同键笔记查找（写-崩溃窗口兜底）----------------------------------


def run_step_key(run_id: str, step_id: str) -> str:
    """step 幂等键：``{run_id}::{step_id}``（确定性、可复算，写进入库笔记 extra）。"""
    return f"{run_id}::{step_id}"


def find_step_key_note_ids(store: Any, key: str) -> list[str]:
    """在 store 全量笔记里找 extra["run_step_key"] == key 的笔记 id（升序去重）。

    status=None 全量扫描（含 merged 留痕 / superseded）：kill 恰好落在"入库完成、
    标注未完成"之间时，笔记可能处于任意状态，漏扫会把已完成写入判成未完成。
    store 是 WikiStore（鸭子类型：只用 list_notes 与 meta.extra，避免模块级
    「loop → wiki」导入成环）。
    """
    found: set[str] = set()
    for note in store.list_notes(status=None):
        if isinstance(note.meta.extra, dict) and note.meta.extra.get("run_step_key") == key:
            found.add(note.id)
    return sorted(found)


# ---- llm_keys_seen 采集（纵深防御）---------------------------------------------


def _request_key(
    model: str,
    messages: list[Message],
    *,
    system: str | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> str:
    """调用请求哈希 = sha256(model + messages(规范化 json) + tools)。

    与 llm/replay.py 的 request_hash 同一约定（system 折叠进消息参与规范化），
    这里独立实现以避免依赖 replay 的缓存语义——哈希口径一致，两者可组合。
    """
    msgs: list[dict[str, Any]] = []
    if system:
        msgs.append({"role": "system", "content": system})
    for m in messages:
        entry: dict[str, Any] = {"role": m.role, "content": m.content}
        if m.tool_call_id is not None:
            entry["tool_call_id"] = m.tool_call_id
        if m.name is not None:
            entry["name"] = m.name
        if m.tool_calls is not None:
            entry["tool_calls"] = m.tool_calls
        msgs.append(entry)
    payload = {"model": model, "messages": msgs, "tools": tools}
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class KeyRecordingProvider:
    """记录每次调用请求哈希的 Provider 包装（llm_keys_seen 的采集点）。

    仅在 [checkpoint] 接线且 enabled 时由 AgentLoop 安装；请求哈希在 stream() 发起
    时记入共享 sink（与 ReplayProvider 同一 request_hash 口径，见 _request_key）。
    透传事件流与 model/tier，对调用方零行为变化。
    """

    def __init__(self, inner: Any, sink: list[str]) -> None:
        self.inner = inner
        self.model = inner.model
        self.tier = inner.tier
        self._sink = sink

    def stream(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        self._sink.append(_request_key(self.model, messages, system=system, tools=tools))
        yield from self.inner.stream(messages, system=system, tools=tools)
