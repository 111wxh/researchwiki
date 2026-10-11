"""滚动压缩（P4b 支撑设施 T6）：上下文预算触发的旧轮次压缩。

长程运行（连续问题流）下主循环 history 无界增长会撑爆模型窗口。本模块在每次
observe（工具结果入史）之后、下一次模型调用之前，对 history 做确定性 token
估算（llm/accounting.py 的 ``estimate_*``，**估算非计量**），估算达到
``compaction_threshold × max_context_tokens`` 时把旧轮次压缩成一条「研究进展
摘要」消息，保留最近 ``keep_recent_turns`` 轮原文与计划里的未解决任务清单。

设计取舍（逐条写明）：

- **触发估算只看消息列表（后缀区）**：system prompt 与工具 schema 属稳定前缀，
  在 run 内恒定、不随轮次增长（见 agent_loop 装配处的分区注释），阈值的语义
  针对「会增长的部分」——``maybe_compact`` 只收 messages，估算即消息列表。
- **整轮切分**（``segment_turns``）：一条 assistant 消息开启一轮，其后的 tool
  结果随同一轮。整轮保留才能保住 OpenAI 协议的 ``tool_calls → tool`` 配对——
  半轮切割会让下一次模型调用直接被 API 拒绝。
- **未解决任务清单**：run 内没有结构化任务状态（state.md 只记阶段/步数/token
  等滚动计数，无任务清单字段），压缩时刻无法判定哪些计划任务已完成——从旧轮次
  中的计划文本（plan 阶段产物，history 里第一条 assistant 消息）提取任务列表
  行，**原文保留**进摘要消息，不替模型宣告完成。显式 ``- [ ]`` 复选框优先，
  没有复选框时退回全部「- 」列表项（计划提示词约定的任务拆分形态）。
- **诚实呈现**：摘要消息带「【研究进展摘要·已压缩 N 轮】」标记前缀，让后续
  模型知道上下文被压缩过，不伪装成原文；role 用 user（OpenAI 协议下 system
  角色出现在对话中段不合规）。
- **保守降级**：summarizer 抛异常 → 不压缩（原文一个字不丢），结果标 failed，
  事件照记——压缩失败绝不能变成上下文损坏。
- **审计**：每次压缩（含 failed）追加一行到 ``run_dir/compactions.jsonl``，
  读取按字节读、逐行 decode（``errors="replace"``）——手法照抄 T17 的
  jobs.jsonl（真实断电可能把字节流截在 UTF-8 序列中间，整文件 decode 会在
  此时抛 UnicodeDecodeError 拖垮读取方）。估算/压缩数字只进这本账本，不改
  run-metrics 的 14 字段契约。
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from researchwiki.llm.accounting import estimate_message_tokens
from researchwiki.llm.provider import Message

# summarizer 注入契约：接收待压缩的旧轮次消息列表，返回摘要文本。
# AgentLoop 侧包一层 cheap 档模型调用（未配置复用 strong，与蒸馏/子 agent 同
# 口径）；测试注入 fake 实现零网络。
Summarizer = Callable[[list[Message]], str]

DEFAULT_MAX_CONTEXT_TOKENS = 131072
DEFAULT_COMPACTION_THRESHOLD = 0.7
DEFAULT_KEEP_RECENT_TURNS = 4

# 摘要消息的标记前缀：诚实呈现压缩事实（N = 被压缩的轮数）
SUMMARY_MARKER = "【研究进展摘要·已压缩 {n} 轮】"

# 摘要生成的 system 提示词（AgentLoop 侧 cheap 模型调用使用）
COMPACTION_SYSTEM = (
    "你是研究助理，负责把一段被裁剪的研究对话历史压缩成一份研究进展摘要。"
    "要求：保留研究问题、已确认的关键事实与来源线索、尚未完成的任务；"
    "不要编造未出现的信息；直接输出摘要正文（300 字以内）。"
)

# 未解决任务清单的标题（摘要消息内的一节）
_OPEN_TASKS_HEADER = "未解决任务清单（计划原文保留）："


@dataclass
class ContextSettings:
    """[context] 段的解析结果（见 config.toml 注释与 context_settings_from_config）。"""

    enabled: bool = True
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS
    compaction_threshold: float = DEFAULT_COMPACTION_THRESHOLD
    keep_recent_turns: int = DEFAULT_KEEP_RECENT_TURNS


@dataclass
class CompactionResult:
    """一次 maybe_compact 的裁决结果（简报契约字段 + 降级状态）。

    status：ok（含"未触发"的正常路径）/ failed（summarizer 异常，保守不压缩）。
    error：failed 时的异常摘要（"类型: 消息"），只进 compactions.jsonl。
    """

    compacted: bool = False
    pre_tokens: int = 0
    post_tokens: int = 0
    kept_turns: int = 0
    summary: str = ""
    status: str = "ok"
    error: str = ""


def context_settings_from_config(
    context_config: Mapping[str, Any] | None,
    runtime_config: Mapping[str, Any] | None = None,
) -> ContextSettings | None:
    """解析 [context] 段 → ContextSettings；段缺失（None）= 不接线，返回 None。

    compaction_threshold 的读取顺序（迁移路径，见 config.toml 注释）：
    [context].compaction_threshold（新家）→ [runtime].compaction_threshold
    （旧键回退，已部署配置可能只在 [runtime] 调过比例）→ 模块默认 0.7。
    """
    if context_config is None:
        return None  # 段缺失 = 不接线 = 零行为变化
    threshold = context_config.get("compaction_threshold")
    if threshold is None and runtime_config is not None:
        threshold = runtime_config.get("compaction_threshold")
    return ContextSettings(
        enabled=bool(context_config.get("enabled", True)),
        max_context_tokens=int(
            context_config.get("max_context_tokens") or DEFAULT_MAX_CONTEXT_TOKENS
        ),
        compaction_threshold=float(
            threshold if threshold is not None else DEFAULT_COMPACTION_THRESHOLD
        ),
        keep_recent_turns=int(
            context_config.get("keep_recent_turns") or DEFAULT_KEEP_RECENT_TURNS
        ),
    )


def segment_turns(messages: list[Message]) -> list[list[Message]]:
    """按「轮」切分 history：一条 assistant 消息开启一轮，其后紧随的 tool 结果
    随同一轮；history 开头的 user 消息（研究问题）并入随后的计划轮，不单独成轮。

    整轮是保留的最小单位——assistant(tool_calls) 与其 tool 结果永不分离，
    否则下一次模型调用会因协议配对缺失被拒绝。
    """
    turns: list[list[Message]] = []
    for m in messages:
        if m.role == "tool" and turns:
            turns[-1].append(m)  # 工具结果随其触发的 assistant 轮
        elif m.role == "assistant" and turns and _is_bare_user_turn(turns[-1]):
            # 首条 user（研究问题）+ 计划 assistant 同属一轮
            turns[-1].append(m)
        elif m.role == "assistant" or not turns:
            turns.append([m])
        else:
            turns[-1].append(m)  # 中段 user 消息（如报告请求）：随当前轮
    return turns


def _is_bare_user_turn(turn: list[Message]) -> bool:
    """尚未配对 assistant 的 user 开头轮（研究问题轮等摘要前的孤立 user）。"""
    return turn[0].role == "user" and len(turn) == 1


def extract_open_tasks(plan_text: str) -> list[str]:
    """从计划文本提取未解决任务清单（原文行）。

    取舍见模块 docstring：压缩时刻无法判定任务完成状态，计划的任务拆分一律
    视为未解决、原文保留。显式 ``- [ ]`` 复选框优先；没有复选框时退回全部
    「- 」列表项（计划提示词约定的任务拆分形态）。
    """
    lines = [line.strip() for line in plan_text.splitlines()]
    checkboxes = [line for line in lines if re.match(r"^- \[ \]\s+\S", line)]
    if checkboxes:
        return checkboxes
    return [line for line in lines if re.match(r"^-\s+\S", line)]


def build_summary_message(
    summary_text: str,
    *,
    turns_compacted: int,
    open_tasks: Sequence[str] = (),
) -> Message:
    """构造压缩摘要消息：标记前缀（诚实呈现）+ 摘要正文 + 未解决任务清单原文。

    role=user：OpenAI 协议下 system 角色出现在对话中段不合规，user 协议安全。
    """
    lines = [SUMMARY_MARKER.format(n=turns_compacted)]
    body = summary_text.strip()
    lines.append(body if body else "（摘要为空：模型未返回内容）")
    if open_tasks:
        lines.append("")
        lines.append(_OPEN_TASKS_HEADER)
        lines.extend(open_tasks)
    return Message(role="user", content="\n".join(lines))


def maybe_compact(
    messages: list[Message],
    settings: ContextSettings | None,
    *,
    summarizer: Summarizer,
) -> CompactionResult:
    """估算 history，达到阈值把旧轮次压缩成研究进展摘要（**原地改写** messages）。

    - 逃生阀：settings 为 None（不接线）或 enabled=False → 直接返回、零动作
      （AgentLoop 侧已短路，这里再守一道）；
    - 触发条件：``estimate(messages) >= compaction_threshold × max_context_tokens``
      （估算只看消息列表，取舍见模块 docstring）；
    - 未跨阈值、或旧轮次不足（轮数 ≤ keep_recent_turns，压无可压）→ compacted
      =False、messages 原样；
    - 触发时：最近 keep_recent_turns 轮整轮原文保留，其余轮次交 summarizer 压缩，
      摘要消息（带标记前缀 + 未解决任务清单原文）插在保留区之前；
    - summarizer 抛异常 → 保守降级：messages 原样、status="failed"（不压缩）。
    """
    if settings is None or not settings.enabled:
        return CompactionResult()  # 逃生阀 / 未接线：零估算零动作

    pre_tokens = sum(estimate_message_tokens(m) for m in messages)
    threshold_tokens = int(settings.compaction_threshold * settings.max_context_tokens)
    if pre_tokens < threshold_tokens:
        return CompactionResult(pre_tokens=pre_tokens, post_tokens=pre_tokens)

    keep = max(0, settings.keep_recent_turns)
    turns = segment_turns(messages)
    # 轮数不足：全保留（不能把历史压到只剩摘要——摘要无原文可依）
    if len(turns) <= keep:
        return CompactionResult(pre_tokens=pre_tokens, post_tokens=pre_tokens)

    # 注意 Python 切片陷阱：turns[:-0] == turns[:0]，keep=0 必须单独取整表
    old_turns = turns if keep == 0 else turns[:-keep]
    kept_turns_msgs = [] if keep == 0 else turns[-keep:]
    old_messages = [m for turn in old_turns for m in turn]
    kept_messages = [m for turn in kept_turns_msgs for m in turn]

    # 未解决任务清单：旧轮次里第一条 assistant 消息即计划文本（plan 阶段产物）。
    # 计划已在早前压缩中进入摘要时（重复压缩），第一条 assistant 是 act 消息，
    # 提取结果为空——任务的留存职责交给上一条摘要，不重复提取。
    plan_text = next(
        (m.content for m in old_messages if m.role == "assistant" and m.content.strip()),
        "",
    )
    open_tasks = extract_open_tasks(plan_text)

    try:
        summary_text = summarizer(old_messages)
    except Exception as exc:  # noqa: BLE001 -- 保守降级：摘要失败绝不丢原文
        return CompactionResult(
            pre_tokens=pre_tokens,
            post_tokens=pre_tokens,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )

    summary = build_summary_message(
        summary_text, turns_compacted=len(old_turns), open_tasks=open_tasks
    )
    messages[:] = [summary, *kept_messages]
    post_tokens = sum(estimate_message_tokens(m) for m in messages)
    return CompactionResult(
        compacted=True,
        pre_tokens=pre_tokens,
        post_tokens=post_tokens,
        kept_turns=len(kept_turns_msgs),
        summary=summary_text,
    )


# ---- 压缩事件审计（run_dir/compactions.jsonl）--------------------------------


def record_compaction_event(
    run_dir: str | Path, *, index: int, result: CompactionResult
) -> dict[str, Any]:
    """压缩事件追加一行到 ``run_dir/compactions.jsonl``，返回写入的事件 dict。

    字段：index（run 内压缩序号，1 起）/ at / pre_tokens / post_tokens /
    kept_turns / summary_chars；summarizer 失败的事件再加 status="failed" 与
    error。断电容错照抄 T17 jobs.jsonl 手法：追加前检查末行是否缺换行，
    保证新记录不与截断行粘连成不可解析的坏数据。
    """
    event: dict[str, Any] = {
        "index": index,
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
        "pre_tokens": result.pre_tokens,
        "post_tokens": result.post_tokens,
        "kept_turns": result.kept_turns,
        "summary_chars": len(result.summary),
    }
    if result.status == "failed":
        event["status"] = "failed"
        event["error"] = result.error
    path = Path(run_dir) / "compactions.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_newline = False
    if path.is_file():
        with path.open("rb") as f:
            f.seek(0, 2)  # 末尾
            if f.tell() > 0:  # 空文件（外部 touch/损坏）无从谈"末行缺换行"
                f.seek(-1, 2)
                needs_newline = f.read(1) != b"\n"
    with path.open("a", encoding="utf-8") as f:
        if needs_newline:
            f.write("\n")
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    return event


def read_compaction_events(run_dir: str | Path) -> list[dict[str, Any]]:
    """读取压缩事件（追加账本，坏行跳过）。

    按字节读、逐行 decode（``errors="replace"``）：账本行含中文（摘要 error），
    真实断电可能把字节流截在 UTF-8 序列中间——整文件 decode 会在此时抛
    UnicodeDecodeError；替换后的坏行交由 json.loads 拒绝。读取侧的承诺与
    T17 jobs.jsonl 一致："最多损坏最后一行"。
    """
    path = Path(run_dir) / "compactions.jsonl"
    if not path.is_file():
        return []
    events: list[dict[str, Any]] = []
    for raw_line in path.read_bytes().split(b"\n"):
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue  # 截断/畸形行：跳过，不让一条坏行拖垮整个账本
        if isinstance(data, dict):
            events.append(data)
    return events
