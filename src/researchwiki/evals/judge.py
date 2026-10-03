"""LLM 评审（judge）：用强模型按 rubric 给候选回答打三维分。

评测支线（PLAN §7.2）的模型判官：对每道题的候选回答做**一次** judge 调用
（成本口径：每题一次，一次打齐三维分），输出严格 JSON——

- coverage：回答对 gold 要点的覆盖与正确程度（无答案题正确拒答得高分、编造得低分）；
- citation：回答中 [n] 引用与其声称内容的相符程度（非无答案题却无引用应扣分）；
- temporal：时间敏感事实是否采用了最新口径。

分数各 1–5；解析失败 / 字段缺失 / 类型错 / 非有限分数（NaN/Infinity，json.loads
默认接受其字面量）返回 None 并记 warning，**绝不编造分数**——缺评审分的题由
上层聚合口径决定怎么计。
调用复用 :func:`researchwiki.wiki.distiller.call_text`，token 记账与蒸馏同一条链路
（step="evals:judge"）。
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from researchwiki.llm.accounting import TokenAccountant
from researchwiki.llm.openai_provider import OpenAICompatibleProvider
from researchwiki.llm.provider import MockProvider, Provider
from researchwiki.loop.subagent import extract_json
from researchwiki.wiki.distiller import call_text

logger = logging.getLogger(__name__)

#: 评审 rubric（system prompt）：说明输入、三个维度的口径与严格 JSON 输出格式
JUDGE_SYSTEM = (
    "你是「自进化研究 Wiki」的评测判官，负责给一段候选回答打分。\n"
    "输入包含：问题、gold 参考要点列表、候选回答原文、来源清单（[n] 编号）。\n"
    "按以下三个维度各打 1–5 的整数分：\n"
    "- coverage：回答对 gold 要点的覆盖与正确程度；对无答案题（gold 为空），"
    "正确的拒答得高分，编造答案得低分；\n"
    "- citation：回答中 [n] 引用与其声称内容的相符程度；非无答案题却完全没有引用应扣分；\n"
    "- temporal：时间敏感事实是否采用了最新口径（拿旧版本口径回答新版本问题应扣分）。\n"
    "只输出一个 JSON 对象，不要输出任何其他文字、不要加代码围栏以外的说明：\n"
    '{"coverage": <1-5 整数>, "citation": <1-5 整数>, "temporal": <1-5 整数>, '
    '"reasons": "<三个维度的简短中文理由>"}'
)

#: 三个分数维度（顺序即输出语义：覆盖 / 引用 / 时效）
_SCORE_FIELDS: tuple[str, str, str] = ("coverage", "citation", "temporal")

#: mock judge 的占位回复：本身就是可解析 JSON，管线测试能走通全链路
_MOCK_JUDGE_SCRIPT = (
    "（mock judge）未配置真实评审模型，以下是占位评分——仅用于管线测试，分数无意义：\n"
    '{"coverage": 3, "citation": 3, "temporal": 3, "reasons": "mock judge 占位评分"}'
)


@dataclass(frozen=True)
class JudgeVerdict:
    """一次评审的三维评分（各 1–5，clamp 后的整数）。

    raw 保留模型原始输出，供审计与调参；解析失败时 :func:`judge_answer`
    返回 None，不会用编造的分数填充本类。
    """

    coverage: int
    citation: int
    temporal: int
    reasons: str
    raw: str


# ---------------------------------------------------------------------------
# prompt 构造
# ---------------------------------------------------------------------------


def _format_source_entry(n: int, item: object) -> str:
    """把单个来源格式化成 `[n] url — title`；条目本身为空则返回空串。"""
    if isinstance(item, Mapping):
        url = str(item.get("url") or "").strip()
        title = str(item.get("title") or "").strip()
        if url and title:
            return f"[{n}] {url} — {title}"
        return f"[{n}] {url or title}".rstrip()
    text = str(item).strip()
    return f"[{n}] {text}" if text else ""


def _format_sources(sources: Sequence[object]) -> str:
    """来源清单编号列表（str 或 {url, title} 映射均可），空清单返回空串。"""
    lines = [_format_source_entry(n, item) for n, item in enumerate(sources, start=1)]
    return "\n".join(line for line in lines if line)


def _build_user_prompt(
    question: str, gold_points: Sequence[str], answer: str, sources: Sequence[object]
) -> str:
    """评审输入：问题 / gold 逐条编号 / 回答原文 / 来源清单 [n] 编号。"""
    lines = [f"问题：\n{question}", "", "gold 参考要点："]
    if gold_points:
        lines.extend(f"{i}. {point}" for i, point in enumerate(gold_points, start=1))
    else:
        # 语料契约：gold_points 为空 ⟺ unanswerable 题，提示判官按拒答质量评 coverage
        lines.append("（无——本题是无答案题，正确的拒答应得高分，编造答案应得低分）")
    lines.extend(["", "候选回答：", answer])
    formatted = _format_sources(sources)
    if formatted:
        lines.extend(["", "来源清单：", formatted])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 解析（容忍围栏与杂白，clamp 1–5，不编造）
# ---------------------------------------------------------------------------


def _clamp_score(value: float) -> int:
    """越界分数夹回 1–5（6 → 5、0 → 1），round 后取整。"""
    return max(1, min(5, int(round(value))))


def _parse_verdict(raw: str) -> JudgeVerdict | None:
    """从模型输出提取第一个 {...} 块并校验字段；任何一步不合法都返回 None。

    - 分数必须是数字（bool 不算——json 的 true/false 不是评分）；
    - 分数必须有限：NaN/Infinity 在 clamp 前按解析失败处理（json.loads 默认
      接受这些字面量，得到 float('nan')/float('inf')——clamp 会把它们硬夹成
      合法分数，等于把坏输出洗成假数据，绝不允许）；
    - reasons 必须是字符串；缺失或类型错一律 None，不猜默认值。
    """
    obj = extract_json(raw)
    if obj is None:
        return None
    scores: dict[str, int] = {}
    for field in _SCORE_FIELDS:
        value = obj.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        if not math.isfinite(value):
            return None
        scores[field] = _clamp_score(value)
    reasons = obj.get("reasons")
    if not isinstance(reasons, str):
        return None
    return JudgeVerdict(
        coverage=scores["coverage"],
        citation=scores["citation"],
        temporal=scores["temporal"],
        reasons=reasons,
        raw=raw,
    )


# ---------------------------------------------------------------------------
# 对外入口
# ---------------------------------------------------------------------------


def judge_answer(
    provider: Provider,
    *,
    question: str,
    gold_points: Sequence[str],
    answer: str,
    sources: Sequence[object] = (),
    accountant: TokenAccountant | None = None,
    trace_id: str = "",
    clock: Callable[[], float] = time.perf_counter,
) -> JudgeVerdict | None:
    """对一道题的候选回答做一次评审调用，返回三维评分；解析失败返回 None。

    与 Distiller 逐字同款的调用链：:func:`researchwiki.wiki.distiller.call_text`
    消费整段流并记账（step="evals:judge"，每题恰一次 judge 调用）。
    解析失败只记一行 warning，不抛异常、不编造分数。
    """
    raw = call_text(
        provider,
        system=JUDGE_SYSTEM,
        user=_build_user_prompt(question, gold_points, answer, sources),
        step="evals:judge",
        accountant=accountant,
        trace_id=trace_id,
        clock=clock,
    )
    verdict = _parse_verdict(raw)
    if verdict is None:
        logger.warning("judge 输出无法解析为合法评分 JSON，该题计为无评审分（不编造分数）")
    return verdict


def build_judge_provider(config: Mapping[str, Any] | None) -> Provider:
    """构建 judge 档 Provider。

    注意传入的是**整份 config**（tomllib 解析 config.toml 的全量字典），
    不是 [llm] 段——本函数自己读 ``config["llm"]["judge"]``。

    - 缺 [llm] / [llm.judge] 段、或 base_url 为空 → 返回 tier="judge" 的
      MockProvider（model 带 mock 标记）。**mock judge 仅用于管线测试，
      分数无意义**——真实评测必须配置强模型，且评审必须强于被测模型；
    - 否则返回 OpenAICompatibleProvider(tier="judge")。

    real run 想要可复现，可在外层再包一层 ReplayProvider（本模块不强制）。
    """
    judge_cfg: Mapping[str, Any] = {}
    if config is not None:
        llm_cfg = config.get("llm")
        if isinstance(llm_cfg, Mapping):
            candidate = llm_cfg.get("judge")
            if isinstance(candidate, Mapping):
                judge_cfg = candidate
    model = str(judge_cfg.get("model") or "").strip()
    base_url = str(judge_cfg.get("base_url") or "").strip()
    api_key_env = str(judge_cfg.get("api_key_env") or "").strip()
    if not base_url:
        return MockProvider(_MOCK_JUDGE_SCRIPT, tier="judge", model=model or "mock-judge")
    return OpenAICompatibleProvider(
        base_url=base_url,
        model=model,
        tier="judge",
        api_key_env=api_key_env or None,
    )
