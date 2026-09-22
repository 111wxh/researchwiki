"""Memory Formation MVP（RQ1：什么时候应该记住）——确定性的入库判定策略。

职责（PLAN v2 §3.2 P1 收尾 / §0.4 RQ1）：在 run 的自动入库路径上，对每条蒸馏
候选回答"该不该记住"：不达标的不入库（防无差别堆积），达标的赋 importance /
confidence 并给出一句可读理由。**零模型调用、零网络**——全部特征从候选本体
测得（正文字符数、实体数、是否带来源、是否含具体事实要素、与既有 active 记忆
的最大相似度），同一输入恒得同一判定，可单测、可复算、可解释。

判定规则（按序短路，命中即拒绝；理由必须引用具体信号值）：

1. 正文过短：``body_chars < min_body_chars``；
2. knowledge 无来源：``require_source_for_knowledge`` 且候选没有任何来源
   （kind 恒按 knowledge 判定，见下）；
3. 重要性不足：``importance < min_importance_to_persist``；
4. 近乎重复：``similarity_max >= near_duplicate_similarity``（与既有 active
   记忆的最大相似度；``similarity_max is None`` 表示无嵌入上下文，跳过本条）。

importance 权重表（各项相加后夹到 [0.0, 1.0]，保留 2 位小数；测试按表断言）：

===========================  =====================================  =========
信号                          条件                                   加分
===========================  =====================================  =========
基础分                        恒定                                   +0.15
来源                          has_source=True                        +0.25
具体事实要素                  has_specifics=True                     +0.25
实体                          每个实体 +0.10，最多计 3 个（封顶）     +0.30 max
正文信息量（完整档）          body_chars >= 80（BODY_FULL_CHARS）    +0.10
正文信息量（达标档）          min_body_chars <= body_chars < 80      +0.05
===========================  =====================================  =========

confidence（证据质量的确定性口径）：有来源且含具体事实要素 → high；有来源 →
medium；其余 → low。

kind 恒为 ``"knowledge"``：user / experience 两类记忆（用户画像、经验教训）
由显式写入路径负责（MCP ``memory.store``，不变量 ⑧ 的"或"分支），不经自动
formation 判定。

配置：``from_config`` 读 ``[formation]`` 段（也接受完整 config，自动取段）；
缺省 / None 回退模块默认值，非法值宽容回退（与 dedup_settings / prior_settings
同风格）。``enabled = false`` 是逃生阀：调用方跳过判定、全部照旧入库。
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from researchwiki.wiki.frontmatter import SourceRef

# ---- 默认阈值（[formation] 段可覆盖）---------------------------------------

DEFAULT_MIN_BODY_CHARS = 20
DEFAULT_MIN_IMPORTANCE = 0.3
DEFAULT_NEAR_DUPLICATE = 0.95

# ---- importance 权重表（见模块 docstring；测试按这些常量断言）----------------

BASE_IMPORTANCE = 0.15
WEIGHT_SOURCE = 0.25
WEIGHT_SPECIFICS = 0.25
WEIGHT_PER_ENTITY = 0.10
MAX_ENTITY_BONUS_COUNT = 3  # 实体加分封顶：3 × 0.10 = 0.30
BODY_FULL_CHARS = 80  # 正文信息量完整档的门槛
WEIGHT_BODY_FULL = 0.10
WEIGHT_BODY_MIN = 0.05

# 具体事实要素（has_specifics）的确定性规则——命中任一即算"含具体要素"：
# 1. 阿拉伯数字：数量、日期、版本号、百分比、榜单名次等的载体；
# 2. 拉丁字母词元：专名 / 技术名词 / 型号（Letta、MCP、GLM-4.5）的载体；
# 3. 引号包裹的术语：中文单双书名引号或西文引号内的被定义概念。
_SPECIFIC_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\d"),
    re.compile(r"[A-Za-z]"),
    re.compile(r"[「『“\"'][^」』”\"']+[」』”\"']"),
)


def _has_specifics(body: str) -> bool:
    """正文是否含具体事实要素（规则见模块常量 _SPECIFIC_PATTERNS，确定性可复算）。"""
    return any(pattern.search(body) for pattern in _SPECIFIC_PATTERNS)


# ---- 数据结构 ---------------------------------------------------------------


@dataclass
class FormationSignals:
    """从候选记忆测得的特征（全部可解释、可复算）。"""

    body_chars: int  # 正文字符数（strip 后）
    entity_count: int  # 实体数（去空白后的非空实体名个数）
    has_source: bool  # 是否带至少一个来源
    has_specifics: bool  # 是否含具体事实要素（规则见 _SPECIFIC_PATTERNS）
    similarity_max: float | None  # 与既有 active 记忆的最大相似度（无嵌入上下文时 None）


@dataclass
class FormationDecision:
    """一次 formation 判定的完整结论（拒绝理由 / 接受依据都可读、可审计）。"""

    persist: bool
    kind: str  # MVP 恒 "knowledge"（user/experience 走显式 memory.store，见 docstring）
    importance: float  # 0.0–1.0（权重表加权，夹取 + 保留 2 位小数）
    confidence: str  # high / medium / low（证据质量的确定性口径）
    reason: str  # 一句人话解释（入库后进笔记 extra["formation_reason"]）
    signals: FormationSignals


@dataclass
class FormationSettings:
    """``[formation]`` 段的解析结果：逃生阀与全部判定阈值（集中可配）。"""

    enabled: bool = True
    min_body_chars: int = DEFAULT_MIN_BODY_CHARS
    require_source_for_knowledge: bool = True
    min_importance_to_persist: float = DEFAULT_MIN_IMPORTANCE
    near_duplicate_similarity: float = DEFAULT_NEAR_DUPLICATE


def from_config(config: Mapping[str, Any] | None = None) -> FormationSettings:
    """解析 formation 配置：接受完整 config（取 [formation] 段）或直接给段。

    缺省 / None = 全默认（enabled=true + 默认阈值）；非法值宽容回退默认
    （与 dedup_settings / prior_settings 同风格，不让配置错误打断研究链路）。
    """
    section: Mapping[str, Any] = {}
    if isinstance(config, Mapping):
        raw = config.get("formation")
        section = raw if isinstance(raw, Mapping) else config
    settings = FormationSettings()

    enabled = section.get("enabled")
    if enabled is not None:
        settings.enabled = bool(enabled)

    value = section.get("min_body_chars")
    if value is not None:
        try:
            settings.min_body_chars = max(0, int(value))
        except (TypeError, ValueError):
            settings.min_body_chars = DEFAULT_MIN_BODY_CHARS

    value = section.get("require_source_for_knowledge")
    if value is not None:
        settings.require_source_for_knowledge = bool(value)

    value = section.get("min_importance_to_persist")
    if value is not None:
        try:
            settings.min_importance_to_persist = min(1.0, max(0.0, float(value)))
        except (TypeError, ValueError):
            settings.min_importance_to_persist = DEFAULT_MIN_IMPORTANCE

    value = section.get("near_duplicate_similarity")
    if value is not None:
        try:
            settings.near_duplicate_similarity = min(1.0, max(0.0, float(value)))
        except (TypeError, ValueError):
            settings.near_duplicate_similarity = DEFAULT_NEAR_DUPLICATE
    return settings


# ---- importance 权重表 ------------------------------------------------------


def score_importance(
    signals: FormationSignals, settings: FormationSettings | None = None
) -> float:
    """按模块 docstring 的权重表给 importance 打分（0.0–1.0，保留 2 位小数）。

    各项相加：基础分 + 来源 + 具体事实要素 + 实体（封顶 3 个）+ 正文档位，
    再夹到 [0.0, 1.0]——全要素候选（0.15+0.25+0.25+0.30+0.10 = 1.05）封顶 1.0。
    """
    cfg = settings if settings is not None else FormationSettings()
    score = BASE_IMPORTANCE
    if signals.has_source:
        score += WEIGHT_SOURCE
    if signals.has_specifics:
        score += WEIGHT_SPECIFICS
    score += WEIGHT_PER_ENTITY * min(signals.entity_count, MAX_ENTITY_BONUS_COUNT)
    if signals.body_chars >= BODY_FULL_CHARS:
        score += WEIGHT_BODY_FULL
    elif signals.body_chars >= cfg.min_body_chars:
        score += WEIGHT_BODY_MIN
    return round(min(1.0, max(0.0, score)), 2)


# ---- 主入口 -----------------------------------------------------------------


def evaluate_candidate(
    body: str,
    *,
    entities: Sequence[str] = (),
    sources: Sequence[SourceRef] = (),
    similarity_max: float | None = None,
    settings: FormationSettings | None = None,
) -> FormationDecision:
    """对一条候选记忆做确定性 formation 判定（规则与权重表见模块 docstring）。

    判定按拒绝规则 1→4 短路：第一个命中的规则即给出含信号值的拒绝理由；
    全部通过时给出含全部信号值的接受依据。同一输入恒得同一结论。
    """
    cfg = settings if settings is not None else FormationSettings()
    text = str(body or "")
    signals = FormationSignals(
        body_chars=len(text.strip()),
        entity_count=sum(1 for name in entities if str(name).strip()),
        has_source=any(str(source.url or "").strip() for source in sources),
        has_specifics=_has_specifics(text),
        similarity_max=similarity_max,
    )
    importance = score_importance(signals, cfg)
    confidence = (
        "high"
        if signals.has_source and signals.has_specifics
        else "medium"
        if signals.has_source
        else "low"
    )

    def reject(reason: str) -> FormationDecision:
        return FormationDecision(
            persist=False,
            kind="knowledge",
            importance=importance,
            confidence=confidence,
            reason=reason,
            signals=signals,
        )

    # 拒绝规则 1：正文过短
    if signals.body_chars < cfg.min_body_chars:
        return reject(f"拒绝：正文 {signals.body_chars} 字 < 阈值 {cfg.min_body_chars} 字")
    # 拒绝规则 2：knowledge 无来源（kind 恒 knowledge，见模块 docstring）
    if cfg.require_source_for_knowledge and not signals.has_source:
        return reject(
            "拒绝：knowledge 类候选无来源"
            "（has_source=False，require_source_for_knowledge=True）"
        )
    # 拒绝规则 3：重要性不足
    if importance < cfg.min_importance_to_persist:
        return reject(
            f"拒绝：importance {importance:.2f} 低于阈值 "
            f"{cfg.min_importance_to_persist:.2f}"
        )
    # 拒绝规则 4：与既有 active 记忆近乎重复（None = 无嵌入上下文，跳过）
    if (
        signals.similarity_max is not None
        and signals.similarity_max >= cfg.near_duplicate_similarity
    ):
        return reject(
            f"拒绝：与既有记忆最大相似度 {signals.similarity_max:.3f} ≥ "
            f"{cfg.near_duplicate_similarity}，近乎重复"
        )

    evidence = (
        f"来源{'有' if signals.has_source else '无'}、"
        f"具体要素{'有' if signals.has_specifics else '无'}"
    )
    reason = (
        f"接受：正文 {signals.body_chars} 字、实体 {signals.entity_count} 个、"
        f"{evidence}，importance {importance:.2f} ≥ 阈值 "
        f"{cfg.min_importance_to_persist:.2f}"
    )
    return FormationDecision(
        persist=True,
        kind="knowledge",
        importance=importance,
        confidence=confidence,
        reason=reason,
        signals=signals,
    )


__all__ = [
    "BASE_IMPORTANCE",
    "BODY_FULL_CHARS",
    "DEFAULT_MIN_BODY_CHARS",
    "DEFAULT_MIN_IMPORTANCE",
    "DEFAULT_NEAR_DUPLICATE",
    "MAX_ENTITY_BONUS_COUNT",
    "WEIGHT_BODY_FULL",
    "WEIGHT_BODY_MIN",
    "WEIGHT_PER_ENTITY",
    "WEIGHT_SOURCE",
    "WEIGHT_SPECIFICS",
    "FormationDecision",
    "FormationSettings",
    "FormationSignals",
    "evaluate_candidate",
    "from_config",
    "score_importance",
]
