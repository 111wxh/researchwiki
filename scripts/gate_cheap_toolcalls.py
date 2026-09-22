#!/usr/bin/env python
"""闸 1：cheap 档模型（config.toml [llm.cheap]）工具调用稳定性测量。

── 这个闸测什么 ────────────────────────────────────────────────────────
便宜档模型是整个 agent loop 的地基：它必须能稳定地按 OpenAI 协议发起工具调用
（工具名不幻觉、arguments 是合法 JSON、必填字段齐全），并且在不需要工具的时候
不硬调。本脚本用**项目真实工具面**（复用 loop.agent_loop.build_registry 出的
ToolRegistry：web_search / fetch_url / fs_read / fs_write / fs_list /
dispatch_research）与**真实 system 提示词**（ACT_SYSTEM）设计 20 个工具调用回合，
逐条判定、汇总，并在达标线不满足时以非零退出码结束（CI / 闸门可用）。

回合类别（各类 4 个，共 20；--rounds 可按类别轮转裁剪）：
  单工具   简单检索，期望恰好一次 web_search；
  多参数   期望带 URL / path+content 等必填参数的调用；
  多轮     第一轮工具结果以 role="tool" 回灌后再问一次，考察它不重复调用、
           不把工具结果当幻觉继续编；
  不该调   直接回答即可的提问（靠 ACT_SYSTEM 自带规则，不额外加"别调工具"提示）；
  中文     中文指令下的工具选择（项目就是中文场景）。

── 怎么判（阈值）───────────────────────────────────────────────────────
每条回合给出一个判定（VERDICTS，见脚本常量）：
  ok                 工具名命中期望、参数合法、（多轮回合）未原样重复上一次调用；
  bad_arguments      工具名对但 arguments 不是合法 JSON / 缺必填字段 / 基础类型不符；
  no_call            期望调用却没调（该调没调）；
  unwanted_call      不该调的回合却调了（不该调乱调）；
  wrong_tool         调了注册工具但都不是本回合期望的；
  repeated_call      多轮回合第二轮原样重复了上一轮的调用（不含新增信息）；
  name_hallucinated  调用了未注册的工具名（工具名幻觉）；
  provider_error     网络 / 协议 / 认证异常（该回合计为失败，但不足以据此判定模型）。
判定优先级：provider_error > name_hallucinated > no_call / unwanted_call /
wrong_tool > bad_arguments > repeated_call > ok（取最先命中的那条）。

可用线（--min-ok，默认 90，百分数）：
  · 工具名正确率 ≥ min-ok 且 参数合法率 ≥ min-ok 且 总体合格率 ≥ min-ok → 退出码 0；
  · 任一项低于 min-ok → 退出码 1；
  · 环境问题（认证被拒 / 网络不通 / 连续异常）→ 退出码 2，本次测量不作数。
口径（写死在 summary 里，便于复现）：
  · 工具名正确率 = 判定 ∈ {ok, bad_arguments, repeated_call} 的"期望调用回合"数
    ÷ 期望调用回合数（分母 16，不含 4 个"不该调"回合）；
  · 参数合法率  = 期望调用回合里"参数合法"的回合数 ÷ 期望调用回合数（该调没调 / 工具名
    幻觉的回合都没有可校验的 schema，一律不记参数合法——没调用就没有合法参数）；
  · 总体合格率  = 判定为 ok 的回合数 ÷ 总回合数（分母 20，含"不该调"回合）。
  · 参数合法性按 schema 判：required 字段齐全且非空、properties 声明的基础类型
    一致（string/integer/number/boolean/array/object）；未声明的额外字段不判错。

── 怎么跑 ─────────────────────────────────────────────────────────────
  # 自证：Mock 造出已知结果的回合，验证统计与退出码逻辑（零 key 零网络）
  uv run --no-sync python scripts/gate_cheap_toolcalls.py --provider mock
  uv run --no-sync python scripts/gate_cheap_toolcalls.py --provider mock --mock-profile flaky
  # 真实测量：走 config.toml [llm.cheap]，key 从 .env 读（脚本内零硬编码）
  uv run --no-sync python scripts/gate_cheap_toolcalls.py --provider config
  uv run --no-sync python scripts/gate_cheap_toolcalls.py --provider config --json gate1.json
  # 快速冒烟：只跑 5 个回合（每类 1 个）
  uv run --no-sync python scripts/gate_cheap_toolcalls.py --provider config --rounds 5

── 结果怎么解读 ────────────────────────────────────────────────────────
逐回合一行：回合号 | 类别 | 期望 | 实际 | 判定 | 参数合法 | 延迟 ms | in/out tokens。
末段汇总：判定分布、三项比率与达标标记、p50/p95 延迟、总 token、可选费用估算。
若未达标，先看失败回合的实际文案：中文指令下"该调没调"多数靠提示词收紧；
"工具名幻觉 / JSON 破损"通常只能换档或加 few-shot 约束。
注意：单次 20 回合采样的方差不可忽略——判定为"未达标"时建议重跑一次再下结论。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from researchwiki.env import load_env_file
from researchwiki.llm.openai_provider import ProviderError
from researchwiki.llm.provider import Message, ScriptedProvider, StreamEvent, TokenUsage
from researchwiki.llm.router import ModelRouter
from researchwiki.loop.agent_loop import ACT_SYSTEM, RunContext, build_registry
from researchwiki.loop.registry import Tool, ToolRegistry
from researchwiki.tools import atomic_write_text, get_search_provider

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config.toml"
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"
DEFAULT_WORKDIR = PROJECT_ROOT / "wiki-data" / "gate" / "toolcalls"

# 判定标签（JSON 用英文键，终端显示中文）
VERDICT_LABELS: dict[str, str] = {
    "ok": "合格",
    "bad_arguments": "参数非法或 JSON 破损",
    "no_call": "该调没调",
    "unwanted_call": "不该调乱调",
    "wrong_tool": "调错工具",
    "repeated_call": "重复调用",
    "name_hallucinated": "工具名幻觉",
    "provider_error": "网络或协议异常",
}
# 这些判定说明"工具名正确"（参数或时序另有问题）
NAME_OK_VERDICTS = frozenset({"ok", "bad_arguments", "repeated_call"})
# 认证失败的特征：智谱新端点回 HTTP 401 "令牌已过期或验证不正确"，
# 旧端点回 {"error":{"code":"1000","message":"身份验证失败。"}}（两种都识别）
AUTH_HINTS = (
    "HTTP 401",
    "HTTP 403",
    "身份验证失败",
    "令牌已过期",
    "验证不正确",
    "authentication",
    "invalid api key",
)
AUTH_CODES = ("1000", "1001", "1002", "1003")

# 关键退出码
EXIT_PASS = 0
EXIT_BELOW_LINE = 1
EXIT_ENVIRONMENT = 2


# ---- 回合定义 ---------------------------------------------------------------


@dataclass(frozen=True)
class Round:
    """一个工具调用回合。

    expected 为空元组 = 期望"不调用任何工具"；expected 有多个 = 这些工具名都算对。
    follow_up=True = 第一轮的工具结果回灌 role="tool" 后再问一次（多轮回合）。
    sample 为按 schema 造参时的字段覆盖（保证 mock 参数安全、不误写真实文件）。
    """

    idx: int
    category: str
    prompt: str
    expected: tuple[str, ...]
    follow_up: bool = False
    sample: Mapping[str, Any] = field(default_factory=dict)


def build_rounds() -> list[Round]:
    """20 个回合：单工具 / 多参数 / 多轮 / 不该调 / 中文，各 4 个。"""
    rounds: list[Round] = []

    def add(category: str, prompt: str, expected: tuple[str, ...], **kw: Any) -> None:
        rounds.append(Round(idx=len(rounds) + 1, category=category, prompt=prompt,
                            expected=expected, **kw))

    # 1) 简单单工具调用
    add("单工具", "帮我搜一下「上下文压缩」的工程实践。", ("web_search",))
    add("单工具", "搜索最近关于 agent 长程记忆的进展。", ("web_search",))
    add("单工具", "查一下 sqlite-vec 这个向量扩展怎么用。", ("web_search",))
    add("单工具", "检索一下个人知识库场景里的混合检索方案。", ("web_search",))

    # 2) 需要多参数的调用
    add("多参数", "读取 notes/N-0001.md 的内容。", ("fs_read",),
        sample={"path": "notes/N-0001.md"})
    add("多参数", "把「记忆分层：短期工作记忆 + 长期沉淀」写进 notes/gate-draft.md。",
        ("fs_write",),
        sample={"path": "notes/gate-draft.md",
                "content": "# 记忆分层\n短期工作记忆 + 长期沉淀，按需召回。"})
    add("多参数", "抓取 https://example.com/context-compaction 的正文。", ("fetch_url",),
        sample={"url": "https://example.com/context-compaction"})
    add("多参数", "列出 notes/ 目录下有哪些文件。", ("fs_list",), sample={"path": "."})

    # 3) 多轮：先给工具结果，再让它继续决策（不重复调用同一工具）
    add("多轮", "搜一下 prompt caching 的断点怎么设置。", ("web_search",), follow_up=True)
    add("多轮", "检索一下中文分词在 FTS5 里的主流做法。", ("web_search",), follow_up=True)
    add("多轮", "读取 notes/N-0001.md，然后告诉我它的主题。", ("fs_read",), follow_up=True,
        sample={"path": "notes/N-0001.md"})
    add("多轮", "搜一下 Anthropic 关于上下文工程的文章。", ("web_search",), follow_up=True)

    # 4) 不该调工具：直接回答即可（依靠 ACT_SYSTEM 自带规则，不额外提示）
    add("不该调", "1+1 等于几？直接回答。", ())
    add("不该调", "用一句话解释什么是 RRF 融合。", ())
    add("不该调", "把「检索」翻译成英文，只回译词。", ())
    add("不该调", "3 的平方是多少？", ())

    # 5) 中文指令
    add("中文", "请先检索「上下文窗口管理」相关的中文资料，再告诉我你用的检索词。", ("web_search",))
    add("中文", "中文检索的分词方案有哪些主流选择？搜一下。", ("web_search",))
    add("中文", "把「向量检索」与「向量数据库」的区别整理成 notes/vector-vs-db.md。", ("fs_write",),
        sample={"path": "notes/vector-vs-db.md",
                "content": "# 向量检索 vs 向量数据库\n检索是算法，数据库是存储与索引设施。"})
    add("中文", "读取 notes/N-0001.md 并总结它讲了什么。", ("fs_read",),
        sample={"path": "notes/N-0001.md"})
    return rounds


def select_rounds(rounds: Sequence[Round], limit: int) -> list[Round]:
    """按类别轮转裁剪回合：保证 --rounds 缩小时各类别都被覆盖（纯函数）。"""
    if limit >= len(rounds):
        return list(rounds)
    buckets: dict[str, list[Round]] = {}
    for rd in rounds:
        buckets.setdefault(rd.category, []).append(rd)
    picked: list[Round] = []
    position = 0
    while len(picked) < limit:
        progressed = False
        for bucket in buckets.values():
            if position < len(bucket) and len(picked) < limit:
                picked.append(bucket[position])
                progressed = True
        if not progressed:
            break
        position += 1
    picked.sort(key=lambda r: r.idx)
    return picked


# ---- 工具调用解析与判定（纯函数，供单测）--------------------------------------


@dataclass
class ParsedCall:
    """一条模型发出的工具调用：name + 原始 arguments 字符串 + 解析结果。"""

    name: str
    raw_arguments: str
    args: dict[str, Any] | None
    parse_error: str = ""


def parse_tool_calls(tool_calls: Sequence[Mapping[str, Any]] | None) -> list[ParsedCall]:
    """OpenAI 格式工具调用 → ParsedCall 列表（arguments 解析失败不抛错，记 parse_error）。"""
    parsed: list[ParsedCall] = []
    for tc in tool_calls or []:
        function = tc.get("function") or {}
        name = str(function.get("name") or "")
        raw = function.get("arguments")
        if isinstance(raw, Mapping):
            as_dict = dict(raw)
            parsed.append(
                ParsedCall(name=name, raw_arguments=json.dumps(as_dict), args=as_dict)
            )
            continue
        text = str(raw or "")
        try:
            value = json.loads(text) if text.strip() else {}
        except json.JSONDecodeError as exc:
            parsed.append(
                ParsedCall(
                    name=name, raw_arguments=text, args=None, parse_error=f"JSON 破损: {exc}"
                )
            )
            continue
        if not isinstance(value, dict):
            parsed.append(
                ParsedCall(
                    name=name,
                    raw_arguments=text,
                    args=None,
                    parse_error=f"arguments 不是 JSON object（实得 {type(value).__name__}）",
                )
            )
            continue
        parsed.append(ParsedCall(name=name, raw_arguments=text, args=value))
    return parsed


def _type_ok(value: Any, want: str) -> bool:
    """基础 JSON 类型校验；未知类型声明一律放行。"""
    if want == "string":
        return isinstance(value, str)
    if want == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if want == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if want == "boolean":
        return isinstance(value, bool)
    if want == "array":
        return isinstance(value, list)
    if want == "object":
        return isinstance(value, dict)
    return True


def validate_arguments(args: Mapping[str, Any], tool: Tool) -> tuple[bool, str]:
    """按工具 schema 校验参数：required 齐全非空 + 已声明字段类型一致。"""
    schema = tool.parameters or {}
    required = [str(k) for k in (schema.get("required") or [])]
    missing = [k for k in required if k not in args or args[k] in ("", None)]
    if missing:
        return False, f"缺必填字段 {missing}"
    properties = schema.get("properties") or {}
    for key, value in args.items():
        spec = properties.get(key)
        if not isinstance(spec, Mapping):
            continue  # 未声明的额外字段：不判错
        want = str(spec.get("type") or "")
        if want and not _type_ok(value, want):
            return False, f"字段 {key} 类型应为 {want}，实得 {type(value).__name__}"
    return True, "参数合法"


@dataclass
class TurnOutcome:
    """一次 stream 调用的观测结果。"""

    calls: list[ParsedCall] = field(default_factory=list)
    text: str = ""
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    error: str = ""
    error_kind: str = ""  # auth | network | protocol（空 = 无错）


def classify_error(exc: BaseException) -> str:
    """异常 → 错误类别：auth（key 被拒）/ network（连不上）/ protocol（其他）。"""
    status = getattr(exc, "status_code", None)
    text = str(exc)
    lowered = text.lower()
    if status in (401, 403) or any(hint in text for hint in AUTH_HINTS) or any(
        f'"code":"{code}"' in text or f'"code": "{code}"' in text for code in AUTH_CODES
    ):
        return "auth"
    if isinstance(exc, ProviderError) and "transport error" in lowered:
        return "network"
    if isinstance(exc, ProviderError) and "重试" in text:
        return "network"
    if "transport error" in lowered or "connect" in lowered or "timeout" in lowered:
        return "network"
    return "protocol"


def _same_call(a: ParsedCall, b: ParsedCall) -> bool:
    """两条调用是否等价（名字相同且参数等价）。"""
    if a.name != b.name:
        return False
    if a.args is None or b.args is None:
        return a.raw_arguments.strip() == b.raw_arguments.strip()
    return a.args == b.args


def classify_round(
    rd: Round, turns: Sequence[TurnOutcome], *, known: Mapping[str, Tool]
) -> tuple[str, bool, str]:
    """判定一个回合 → (verdict, args_ok, detail)。纯函数，不触碰网络。"""
    for turn in turns:
        if turn.error:
            return "provider_error", False, turn.error[:160]
    calls = [call for turn in turns for call in turn.calls]

    if not rd.expected:  # 不该调的回合
        if calls:
            # 参数合法性照实报（该回合不进参数合法率分母，也不该把"乱调"说成"参数非法"）
            return "unwanted_call", all(call.args is not None for call in calls), (
                f"期望不调用，实际调了 {calls[0].name}"
            )
        return "ok", True, "未调用工具（符合预期）"

    if not calls:
        return "no_call", False, f"期望 {'/'.join(rd.expected)}，实际无工具调用"

    hit = [call for call in calls if call.name in rd.expected]
    if not hit:
        if any(call.name not in known for call in calls):
            bad = next(call for call in calls if call.name not in known)
            return "name_hallucinated", False, f"未注册工具名 {bad.name!r}"
        return "wrong_tool", False, f"期望 {'/'.join(rd.expected)}，实际 {calls[0].name}"
    if any(call.name not in known for call in calls):
        bad = next(call for call in calls if call.name not in known)
        return "name_hallucinated", False, f"未注册工具名 {bad.name!r}"

    primary = hit[0]
    tool = known[primary.name]
    if primary.args is None:
        return "bad_arguments", False, primary.parse_error or "arguments 无法解析"
    ok, reason = validate_arguments(primary.args, tool)
    if not ok:
        return "bad_arguments", False, reason

    if rd.follow_up and len(turns) >= 2 and turns[1].calls:
        if _same_call(turns[1].calls[0], primary):
            return "repeated_call", True, "第二轮原样重复了上一轮的调用"
    return "ok", True, f"{primary.name} 参数合法"


# ---- 统计（纯函数，供单测）---------------------------------------------------


def percentile(values: Sequence[float], p: float) -> float:
    """线性插值分位数（p ∈ [0,100]）；空输入返回 0.0。"""
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (p / 100.0) * (len(ordered) - 1)
    low, high = math.floor(rank), math.ceil(rank)
    if low == high:
        return ordered[int(rank)]
    frac = rank - low
    return ordered[low] * (1 - frac) + ordered[high] * frac


@dataclass
class Summary:
    """汇总指标：三项比率 + 判定分布 + 延迟/token。比率均为"分之几"的原分数对。"""

    rounds: int
    expect_calls: int
    name_ok: int
    args_ok: int
    ok: int
    verdict_counts: dict[str, int]
    latency_p50: float
    latency_p95: float
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cost: float | None

    @property
    def name_rate(self) -> float:
        return self.name_ok / self.expect_calls if self.expect_calls else 0.0

    @property
    def args_rate(self) -> float:
        return self.args_ok / self.expect_calls if self.expect_calls else 0.0

    @property
    def ok_rate(self) -> float:
        return self.ok / self.rounds if self.rounds else 0.0

    def passed(self, min_ok: float) -> bool:
        return (
            self.name_rate * 100 >= min_ok
            and self.args_rate * 100 >= min_ok
            and self.ok_rate * 100 >= min_ok
        )


def compute_summary(
    results: Sequence[tuple[Round, str, bool, float, int, int, int]],
    *,
    price_in: float | None = None,
    price_out: float | None = None,
) -> Summary:
    """按回合结果算汇总。每条结果 = (Round, verdict, args_ok, latency_ms, in, out, cache)。

    price_in / price_out 为每百万 token 单价（同一货币）；都给才估算费用，否则 None。
    """
    verdict_counts: dict[str, int] = {}
    expect_calls = name_ok = args_ok = ok_count = 0
    latencies: list[float] = []
    tok_in = tok_out = tok_cache = 0
    for rd, verdict, args_valid, latency, in_tok, out_tok, cache_tok in results:
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        latencies.append(float(latency))
        tok_in += int(in_tok)
        tok_out += int(out_tok)
        tok_cache += int(cache_tok)
        if rd.expected:
            expect_calls += 1
            if verdict in NAME_OK_VERDICTS:
                name_ok += 1
            if args_valid:
                args_ok += 1
        if verdict == "ok":
            ok_count += 1
    cost: float | None = None
    if price_in is not None and price_out is not None:
        cost = tok_in / 1_000_000 * price_in + tok_out / 1_000_000 * price_out
    return Summary(
        rounds=len(results),
        expect_calls=expect_calls,
        name_ok=name_ok,
        args_ok=args_ok,
        ok=ok_count,
        verdict_counts=verdict_counts,
        latency_p50=percentile(latencies, 50),
        latency_p95=percentile(latencies, 95),
        input_tokens=tok_in,
        output_tokens=tok_out,
        cache_read_tokens=tok_cache,
        cost=cost,
    )


def fingerprint(secret: str) -> str:
    """key 指纹：前 6 位 + 长度；空值明确标注（绝不打印完整 key）。"""
    if not secret:
        return "未设置"
    return f"{secret[:6]}…(len={len(secret)})"


# ---- Mock 回合脚本（自证统计逻辑用）------------------------------------------


_SAMPLE_VALUES: dict[str, Any] = {
    "query": "上下文压缩 工程实践",
    "max_results": 5,
    "url": "https://example.com/context-compaction",
    "path": "notes/N-0001.md",
    "content": "# 记忆分层\n短期工作记忆 + 长期沉淀，按需召回。",
    "topic": "prompt caching 的断点设置",
    "brief": "聚焦工程实践",
}

# flaky 档：按回合号（idx）指定一处"坏行为"，其余照常，用来验证统计与退出码
_MOCK_FLAKY: dict[int, str] = {
    2: "hallucinated_name",  # 工具名幻觉
    5: "bad_json",           # arguments JSON 破损
    9: "no_call",            # 该调没调
    11: "repeat",            # 多轮原样重复
    13: "unwanted_call",     # 不该调却调了
}
MOCK_PROFILES = ("good", "flaky")


def sample_arguments(tool: Tool, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """按 schema required 造一组"像样"的参数（mock 用；overrides 优先）。"""
    overrides = overrides or {}
    schema = tool.parameters or {}
    properties = schema.get("properties") or {}
    out: dict[str, Any] = {}
    for key in [str(k) for k in (schema.get("required") or [])]:
        if key in overrides:
            out[key] = overrides[key]
            continue
        if key in _SAMPLE_VALUES:
            out[key] = _SAMPLE_VALUES[key]
            continue
        want = str((properties.get(key) or {}).get("type") or "string")
        out[key] = {
            "integer": 1,
            "number": 1.0,
            "boolean": True,
            "array": [],
            "object": {},
        }.get(want, "示例")
    return out


def _call_events(name: str, raw_arguments: str, *, call_id: str) -> list[StreamEvent]:
    payload = [
        {
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": raw_arguments},
        }
    ]
    return [
        StreamEvent(type="text_delta", delta=""),
        StreamEvent(type="tool_calls", tool_calls=payload),
    ]


def _text_events(text: str) -> list[StreamEvent]:
    return [StreamEvent(type="text_delta", delta=text)]


def build_mock_turns(
    rounds: Sequence[Round], registry: ToolRegistry, profile: str
) -> list[list[StreamEvent]]:
    """按回合定义造 ScriptedProvider 的事件脚本（一次 stream 调用消费一轮）。

    顺序必须与 runner 的调用顺序一致：每个回合 1 次 stream，多轮回合 2 次。
    """
    flaky = _MOCK_FLAKY if profile == "flaky" else {}
    turns: list[list[StreamEvent]] = []
    for rd in rounds:
        behavior = flaky.get(rd.idx, "")
        first_name = rd.expected[0] if rd.expected else "web_search"
        if behavior == "unwanted_call" and not rd.expected:
            args = sample_arguments(registry.get("web_search") or _fallback_tool())
            raw = json.dumps(args, ensure_ascii=False)
            turns.append(_call_events("web_search", raw, call_id=f"mock-{rd.idx}-a"))
            continue
        if not rd.expected:
            turns.append(_text_events("RRF 融合就是把多路召回按名次倒数加权求和。"))
            continue
        tool = registry.get(first_name) or _fallback_tool()
        args = sample_arguments(tool, rd.sample)
        raw = json.dumps(args, ensure_ascii=False)
        if behavior == "hallucinated_name":
            turns.append(_call_events(first_name + "_v2", raw, call_id=f"mock-{rd.idx}-a"))
            continue
        if behavior == "bad_json":
            broken = '{"' + first_name + '": ,}'
            turns.append(_call_events(first_name, broken, call_id=f"mock-{rd.idx}-a"))
            continue
        if behavior == "no_call":
            turns.append(_text_events("我先直接回答，不调用工具。"))
            continue
        turns.append(_call_events(first_name, raw, call_id=f"mock-{rd.idx}-a"))
        if rd.follow_up:
            if behavior == "repeat":
                turns.append(_call_events(first_name, raw, call_id=f"mock-{rd.idx}-b"))
            else:
                turns.append(_text_events("基于工具结果：这是本轮研究的简短结论。"))
    return turns


def _fallback_tool() -> Tool:
    """mock 兜底工具（仅用于造脚本，不进 registry）。"""
    return Tool(
        name="web_search",
        description="mock",
        parameters={"type": "object", "properties": {"query": {"type": "string"}},
                    "required": ["query"]},
        handler=lambda args: "{}",
    )


# ---- 运行 -------------------------------------------------------------------


def run_turn(
    provider: Any,
    messages: list[Message],
    *,
    tools: list[dict[str, Any]] | None,
) -> TurnOutcome:
    """跑一次 stream，聚合成 TurnOutcome（异常折算成 error，不抛出）。"""
    outcome = TurnOutcome()
    started = time.perf_counter()
    try:
        for event in provider.stream(messages, system=ACT_SYSTEM, tools=tools):
            if event.type == "text_delta":
                outcome.text += event.delta
            elif event.type == "tool_calls" and event.tool_calls:
                outcome.calls.extend(parse_tool_calls(event.tool_calls))
            elif event.type == "usage" and event.usage is not None:
                outcome.input_tokens += event.usage.input_tokens
                outcome.output_tokens += event.usage.output_tokens
                outcome.cache_read_tokens += event.usage.cache_read_tokens
    except Exception as exc:  # noqa: BLE001 -- 单回合异常折算为判定，不炸整轮
        outcome.error = f"{type(exc).__name__}: {exc}"
        outcome.error_kind = classify_error(exc)
    outcome.latency_ms = (time.perf_counter() - started) * 1000.0
    return outcome


FOLLOW_UP_PROMPT = (
    "以上是工具返回的结果。请继续下一步：信息已足够就直接给出简短结论，"
    "否则换一个工具继续。"
)


def run_round(provider: Any, registry: ToolRegistry, rd: Round) -> tuple[TurnOutcome, ...]:
    """跑一个回合：多轮回合会执行一次真实工具并把结果回灌 role="tool"。"""
    messages: list[Message] = [Message(role="user", content=rd.prompt)]
    first = run_turn(provider, messages, tools=registry.schemas())
    turns: list[TurnOutcome] = [first]
    if not rd.follow_up or first.error or not first.calls:
        return tuple(turns)
    call = first.calls[0]
    messages.append(
        Message(role="assistant", content=first.text, tool_calls=[
            {"id": "planned", "type": "function",
             "function": {"name": call.name, "arguments": call.raw_arguments}}
        ])
    )
    result = registry.dispatch(call.name, call.raw_arguments or "{}")
    messages.append(
        Message(role="tool", content=result, tool_call_id="planned", name=call.name)
    )
    messages.append(Message(role="user", content=FOLLOW_UP_PROMPT))
    turns.append(run_turn(provider, messages, tools=registry.schemas()))
    return tuple(turns)


def build_context(workdir: Path, config: Mapping[str, Any]) -> RunContext:
    """构造真实工具接线所需的 RunContext（沙箱指向 workdir，绝不碰真实 wiki）。"""
    workdir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        workdir / "notes" / "N-0001.md",
        "---\nid: N-0001\ntitle: 上下文压缩的工程实践\nstatus: active\n---\n"
        "滚动压缩：窗口占用超阈值时把早期轮次折叠进状态文件。\n",
    )
    return RunContext(
        search_provider=get_search_provider(config),
        sources_dir=workdir / "sources",
        wiki_root=workdir,
        fetch_timeout=8.0,
    )


@dataclass
class RunPlan:
    """一次运行的全部输入（provider 与工具面已就绪）。"""

    provider: Any
    registry: ToolRegistry
    rounds: list[Round]
    provider_info: dict[str, Any]


def build_plan(args: argparse.Namespace, *, config: Mapping[str, Any]) -> RunPlan:
    """按 --provider 组装 provider / 工具面 / 回合。"""
    rounds = select_rounds(build_rounds(), args.rounds)
    ctx = build_context(Path(args.workdir), config)
    registry = build_registry(ctx)
    llm_cfg = config.get("llm") or {}
    if args.provider == "mock":
        tier_cfg = (llm_cfg.get(args.tier) or {}) if isinstance(llm_cfg, Mapping) else {}
        turns = build_mock_turns(rounds, registry, args.mock_profile)
        provider = ScriptedProvider(
            turns,
            tier=args.tier,
            model=f"mock-{args.tier}",
            usage=TokenUsage(input_tokens=900, output_tokens=70, cache_read_tokens=120),
        )
        info = {
            "mode": "mock",
            "tier": args.tier,
            "model": provider.model,
            "base_url": "",
            "api_key": "未使用",
            "mock_profile": args.mock_profile,
            "configured_model": str(tier_cfg.get("model") or ""),
        }
        return RunPlan(provider=provider, registry=registry, rounds=rounds, provider_info=info)

    router = ModelRouter(dict(llm_cfg))
    provider = router.get(args.tier)
    tier_cfg = (llm_cfg.get(args.tier) or {}) if isinstance(llm_cfg, Mapping) else {}
    env_name = str(tier_cfg.get("api_key_env") or "")
    info = {
        "mode": "config",
        "tier": args.tier,
        "model": getattr(provider, "model", ""),
        "base_url": str(getattr(provider, "base_url", "") or ""),
        "api_key": fingerprint(os.environ.get(env_name, "") if env_name else ""),
        "api_key_env": env_name,
    }
    return RunPlan(provider=provider, registry=registry, rounds=rounds, provider_info=info)


def auth_guidance(info: Mapping[str, Any]) -> str:
    """认证失败时的可执行提示（含 key 指纹与 base_url，绝不含 key 本体）。"""
    env_name = str(info.get("api_key_env") or "RESEARCHWIKI_CHEAP_API_KEY")
    return "\n".join(
        [
            "✗ 认证失败：key 被服务端拒绝（HTTP 401/403）。本次测量不作数，退出码 2。",
            f"  · 当前 base_url = {info.get('base_url') or '(空)'}，"
            f"model = {info.get('model') or '(空)'}",
            f"  · 当前 key 指纹 = {info.get('api_key')}（来源环境变量 {env_name}）",
            f"  · 检查 1：.env 里 {env_name} 是否有效 / 未过期 / 无多余空白或引号；",
            f"  · 检查 2：{env_name} 已存在于真实环境变量时以真实值为准，.env 不覆盖；",
            "  · 检查 3：若用的是中转/代理服务，把 config.toml 的",
            f"    [llm.{info.get('tier')}].base_url 指向该服务（不含 /chat/completions）；",
            "  · 检查 4：确认 model 名在该服务上真实存在（如 glm-4.5-air）。",
        ]
    )


def print_header(plan: RunPlan, args: argparse.Namespace, min_ok: float) -> None:
    info = plan.provider_info
    print("=== 闸 1：cheap 档工具调用稳定性 ===")
    print(f"provider   : {info['mode']}（--provider {args.provider}）")
    print(f"tier/model : {info['tier']} / {info['model']}")
    if info.get("base_url"):
        print(f"base_url   : {info['base_url']}")
    print(f"api_key    : {info['api_key']}")
    if info.get("mock_profile"):
        print(f"mock 档    : {info['mock_profile']}")
    print(f"工具面     : {', '.join(plan.registry.names())}")
    print(f"回合数     : {len(plan.rounds)}   可用线：{min_ok:g}%")
    print("-" * 96)


def describe_expected(rd: Round) -> str:
    if not rd.expected:
        return "不调用工具"
    suffix = "（多轮）" if rd.follow_up else ""
    return "/".join(rd.expected) + suffix


def describe_actual(turns: Sequence[TurnOutcome]) -> str:
    parts: list[str] = []
    for turn in turns:
        if turn.error:
            parts.append("异常")
        elif not turn.calls:
            parts.append("无调用")
        else:
            parts.append("+".join(call.name for call in turn.calls))
    return " → ".join(parts)


def print_round(
    rd: Round, verdict: str, args_valid: bool, detail: str, turns: Sequence[TurnOutcome]
) -> None:
    in_tok = sum(t.input_tokens for t in turns)
    out_tok = sum(t.output_tokens for t in turns)
    latency = sum(t.latency_ms for t in turns)
    label = VERDICT_LABELS.get(verdict, verdict)
    print(
        f"#{rd.idx:02d} | {rd.category} | 期望={describe_expected(rd)}"
        f" | 实际={describe_actual(turns)} | 判定={label}"
        f" | 参数={'合法' if args_valid else '不合法'} | {latency:.0f}ms"
        f" | in/out={in_tok}/{out_tok}"
    )
    if verdict != "ok":
        print(f"      └ {detail}")


def print_summary(summary: Summary, min_ok: float) -> None:
    print("-" * 96)
    print("── 判定分布 ──")
    for verdict, count in sorted(summary.verdict_counts.items(), key=lambda kv: -kv[1]):
        share = count / summary.rounds * 100 if summary.rounds else 0.0
        print(f"  {VERDICT_LABELS.get(verdict, verdict):<16}{count:>3}  ({share:5.1f}%)")
    print("── 指标 ──")

    def line(name: str, num: int, den: int) -> str:
        rate = (num / den * 100) if den else 0.0
        flag = "达标" if rate >= min_ok else "未达标"
        return f"  {name:<14}{num}/{den} = {rate:5.1f}%   [{flag} 阈值 {min_ok:g}%]"

    print(line("工具名正确率", summary.name_ok, summary.expect_calls))
    print(line("参数合法率", summary.args_ok, summary.expect_calls))
    print(line("总体合格率", summary.ok, summary.rounds))
    print(f"  延迟 p50/p95   {summary.latency_p50:.0f}ms / {summary.latency_p95:.0f}ms")
    cache = f"（cache_read {summary.cache_read_tokens}）" if summary.cache_read_tokens else ""
    print(f"  token 总计     in {summary.input_tokens} / out {summary.output_tokens}{cache}")
    if summary.cost is None:
        print("  费用估算       TODO（未提供 --price-in/--price-out，单位为每百万 token）")
    else:
        print(f"  费用估算       {summary.cost:.4f}")
    print("-" * 96)


def build_conclusion(summary: Summary, min_ok: float, plan: RunPlan) -> str:
    if summary.passed(min_ok):
        return (
            f"{summary.rounds} 回合中 {summary.name_ok}/{summary.expect_calls} 次工具名正确"
            f"（{summary.name_rate * 100:.1f}%）、参数合法率 {summary.args_rate * 100:.1f}%、"
            f"总体合格率 {summary.ok_rate * 100:.1f}% —— 达到 loop 可用线（阈值 {min_ok:g}%）。"
        )
    reasons: list[str] = []
    if summary.name_rate * 100 < min_ok:
        reasons.append(f"工具名正确率 {summary.name_rate * 100:.1f}%")
    if summary.args_rate * 100 < min_ok:
        reasons.append(f"参数合法率 {summary.args_rate * 100:.1f}%")
    if summary.ok_rate * 100 < min_ok:
        reasons.append(f"总体合格率 {summary.ok_rate * 100:.1f}%")
    worst = sorted(summary.verdict_counts.items(), key=lambda kv: -kv[1])
    top_bad = [f"{VERDICT_LABELS.get(k, k)}×{v}" for k, v in worst if k != "ok"][:3]
    return (
        f"{summary.rounds} 回合中工具名正确 {summary.name_ok}/{summary.expect_calls}、"
        f"参数合法 {summary.args_ok}/{summary.expect_calls}、合格 {summary.ok}/{summary.rounds}"
        f" —— 未达 loop 可用线（阈值 {min_ok:g}%）："
        + "、".join(reasons)
        + ("；主要问题：" + "、".join(top_bad) if top_bad else "")
    )


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n"
    path.write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    if args.env_file:
        load_env_file(args.env_file)
    config = load_config(Path(args.config)) if args.config else {}
    plan = build_plan(args, config=config)
    min_ok = float(args.min_ok)
    print_header(plan, args, min_ok)

    known: dict[str, Tool] = {}
    for name in plan.registry.names():
        tool = plan.registry.get(name)
        if tool is not None:
            known[name] = tool
    results: list[tuple[Round, str, bool, float, int, int, int]] = []
    round_rows: list[dict[str, Any]] = []
    consecutive_errors = 0

    for rd in plan.rounds:
        turns = run_round(plan.provider, plan.registry, rd)
        verdict, args_valid, detail = classify_round(rd, turns, known=known)
        print_round(rd, verdict, args_valid, detail, turns)
        latency = sum(t.latency_ms for t in turns)
        in_tok = sum(t.input_tokens for t in turns)
        out_tok = sum(t.output_tokens for t in turns)
        cache_tok = sum(t.cache_read_tokens for t in turns)
        results.append((rd, verdict, args_valid, latency, in_tok, out_tok, cache_tok))
        round_rows.append(
            {
                "round": rd.idx,
                "category": rd.category,
                "prompt": rd.prompt,
                "expected": list(rd.expected),
                "follow_up": rd.follow_up,
                "actual": [
                    {
                        "calls": [{"name": c.name, "arguments": c.raw_arguments} for c in t.calls],
                        "text": t.text[:200],
                        "error": t.error,
                    }
                    for t in turns
                ],
                "verdict": verdict,
                "args_valid": args_valid,
                "detail": detail,
                "latency_ms": round(latency, 1),
                "input_tokens": in_tok,
                "output_tokens": out_tok,
            }
        )
        if verdict == "provider_error":
            consecutive_errors += 1
            kind = turns[0].error_kind if turns else "protocol"
            if kind == "auth":
                print()
                print(auth_guidance(plan.provider_info))
                return EXIT_ENVIRONMENT
            if consecutive_errors >= 3:
                print()
                print(
                    "✗ 连续 3 个回合都是网络/协议异常，本次测量不作数（退出码 2）。\n"
                    f"  最后一条错误：{turns[0].error[:200] if turns else '(空)'}\n"
                    "  · 检查网络 / 代理（github 需代理时，LLM 端点通常直连即可）；\n"
                    f"  · 检查 base_url={plan.provider_info.get('base_url')} 是否可达。"
                )
                return EXIT_ENVIRONMENT
        else:
            consecutive_errors = 0

    summary = compute_summary(results, price_in=args.price_in, price_out=args.price_out)
    print_summary(summary, min_ok)
    conclusion = build_conclusion(summary, min_ok, plan)
    print("结论：" + conclusion)

    if args.json:
        write_json(
            Path(args.json),
            {
                "gate": "cheap_toolcalls",
                "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "provider": plan.provider_info,
                "tools": plan.registry.names(),
                "min_ok_percent": min_ok,
                "conclusion": conclusion,
                "passed": summary.passed(min_ok),
                "summary": {
                    "rounds": summary.rounds,
                    "expect_calls": summary.expect_calls,
                    "name_ok": summary.name_ok,
                    "args_ok": summary.args_ok,
                    "ok": summary.ok,
                    "name_rate": round(summary.name_rate, 4),
                    "args_rate": round(summary.args_rate, 4),
                    "ok_rate": round(summary.ok_rate, 4),
                    "verdict_counts": summary.verdict_counts,
                    "latency_p50_ms": round(summary.latency_p50, 1),
                    "latency_p95_ms": round(summary.latency_p95, 1),
                    "input_tokens": summary.input_tokens,
                    "output_tokens": summary.output_tokens,
                    "cache_read_tokens": summary.cache_read_tokens,
                    "cost_estimate": summary.cost,
                },
                "rounds": round_rows,
            },
        )
        print(f"JSON 已写入 {args.json}")

    return EXIT_PASS if summary.passed(min_ok) else EXIT_BELOW_LINE


def load_config(path: Path) -> dict[str, Any]:
    """读 config.toml；缺失/损坏按空配置处理（等价全 mock）。"""
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except (FileNotFoundError, tomllib.TOMLDecodeError):
        return {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="闸 1：cheap 档模型工具调用稳定性测量（判定标准见文件头 docstring）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--provider", choices=("mock", "config"), default="config",
                        help="mock=自证统计逻辑（零网络）；config=走 config.toml 真实档位（默认）")
    parser.add_argument("--tier", choices=("cheap", "strong"), default="cheap",
                        help="测哪一档（默认 cheap；--tier strong 可做对照）")
    parser.add_argument("--rounds", type=int, default=20,
                        help="回合数上限（默认 20，按类别轮转裁剪）")
    parser.add_argument("--min-ok", type=float, default=90.0,
                        help="可用线百分比（默认 90，三项比率都要达标）")
    parser.add_argument("--mock-profile", choices=MOCK_PROFILES, default="good",
                        help="mock 行为档：good=全部符合预期；flaky=故意混入 5 类失败")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE),
                        help=".env 路径（真实环境变量优先）")
    parser.add_argument("--workdir", default=str(DEFAULT_WORKDIR),
                        help="工具沙箱目录（默认 wiki-data/gate/toolcalls，不碰真实 wiki）")
    parser.add_argument("--json", default="", help="把结果另存为 JSON（不含 key，只有指纹）")
    parser.add_argument("--price-in", type=float, default=None,
                        help="每百万 input token 单价（可选）")
    parser.add_argument("--price-out", type=float, default=None,
                        help="每百万 output token 单价（可选）")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="config.toml 路径")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001 -- 老环境/被替换的 stdout 上忽略
        pass
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
