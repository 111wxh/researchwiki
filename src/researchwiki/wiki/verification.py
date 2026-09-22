"""旧记忆与新证据的结构化比较（RQ2 核心判定，P2-C）。

把"新证据出现了，旧记忆该怎么变"变成**结构化、可审计、可复算**的判定：输入一条
``Note``（旧记忆）与一条 ``EvidenceItem``（本轮新证据），输出一个
``EvidenceComparison``（verdict + 引用具体值的 reasons + suggested_action）。
本模块只**判定与建议**，不改笔记、不写台账、不写盘（写入动作在 P2-D 由显式通道
完成）——"不静默覆盖"是设计约束，不是实现细节。

判定阶梯（按执行顺序短路，命中即定 verdict）
--------------------------------------------

0. **相似度地板（先决门）**：``token_similarity < similarity_floor`` → ``uncertain``
   / ``none``，**且不调用 judge**。简报把它写作规则 7，但它的语义是"避免把无关
   新证据拿去比较"，所以实际执行在全部比较之前；地板短路路径也刻意不交给语义
   判官——地板存在的意义就是让无关证据不消耗任何比较器（含模型）的 token。
1. **冲突门**：抽取双方的数值/日期/版本/章节令牌，**同一槽位出现不同值** →
   ``conflicting`` / ``open_conflict``。简报把冲突检测列为规则 3（低于时间规则
   2），这里把它提到时间与具体度之上，理由有两条：简报规则 1 自己就写明"若冲突
   检测命中 → CONFLICTING"；PLAN §3.3 要求"冲突独立落在 conflicts/，不把冲突
   伪装成普通 merge"，而"更晚但自相矛盾"的证据若判 newer/supersede，冲突就被
   supersede 静默掩盖了。
   **实体护栏（全局前置，修复轮 2/5）**：两侧实体集合都非空且不交时**整条短路**
   ——不判冲突、不判 newer/consistent/更具体，直接进兜底落 ``uncertain`` /
   ``none``（judge 仍可改判），理由行写明"无法确认主体同一性 → 不做更新判定"。
   为什么必须全局而不是只拦冲突门：槽位键里不含主语，只拦冲突门时**其他规则会
   接着用同一个不可信的前提**——`模型 A 参数 70B`(GLM-5.3) 对 `模型 B 参数 128B`
   (Qwen-3) 在证据时间更晚时会从"开冲突台账"变成"建议 supersede"（覆盖一条可能
   不是同一主语的记忆），而 `GLM-5.3` 对 `GLM-4.5` 这种同一家族版本号（``entity_keys``
   是精确 slug，家族内不同版本也判不交）还会把两条真冲突整条压掉。方向选择：
   实体不交时**不做任何更新/一致判定**，只标记需复核——这与"不静默覆盖"是同一个
   不可退让方向：宁可少一次自动判定，不可多一次静默覆盖。剩余边界：``entity_keys``
   是精确 slug（``GLM-5.3`` 与 ``GLM5.3`` 判不交），拼写差异会被当成"不同主语"，
   代价是这类证据落 uncertain（不做动作、不写台账），方向安全。
2. **来源变化**：``evidence.source_url`` 命中旧记忆某条 ``SourceRef``——
   - ``content_hash`` **相同**（两侧都非空）→ 来源未变，**不早退**，只追加审计行
     后继续走 3/4/5（同一快照的重述 → consistent；同一快照里更完整的事实 →
     more_specific）。
   - ``content_hash`` **不同**、且 ``prior.meta.source_changed_at`` 存在且可解析
     → ``newer`` / ``supersede``，理由给出两个 hash 的前 8 位与 source_changed_at。
   - ``content_hash`` 不同但未声明 ``source_changed_at``（或声明了但不是合法
     ISO）→ **不据此判定**，追加审计行后继续（来源标记由 P2-B 的检测器写入，
     本模块不替它背判定）。
   - ``source_url`` 命中但证据未给 ``content_hash`` → 记一行"无法判定是否换版"。
3. **一致**：``相似度 ≥ consistent_similarity``、证据没有新增事实令牌、**没有
   被跳过的槽位**、**且槽位级取值逐一对得上** → ``consistent`` /
   ``refresh_reviewed_at``（PLAN §3.3"一致则更新 reviewed_at"）。放在时间规则之前：
   时间更晚但断言未变 = 只是重新观察，不该制造无意义的新版本。
   **"槽位级"是修复轮 1/5 的关键修正**：一致必须逐槽位比（旧记忆的每个可比槽位在
   证据中有同一槽位且取值相同、证据也不多出可比槽位），不能只看"证据用到的
   ``(kind, value)`` 有没有在旧文里出现过"——集合级检查会把**取值错位互换**判成
   仍然成立（旧 ``上下文窗口 128k，参数量 70B`` 对证据 ``上下文窗口 70B，
   参数量 128k``：取值集合完全相同、相似度 0.946、无冲突，但事实已经错位），
   而这正是来源换版时"表格列序/标签错位"的真实形态，方向恰是"不静默覆盖"最不能
   出错的一侧。被跳过的槽位（单侧多值）同样让"一致"退回 uncertain——"还成立"的
   判定必须建立在**所有可比槽位都真的比过且对得上**之上：漏判一致性只是多一次复核，
   误判一致性等于把可能已经变化的事实当成仍然成立。**实体不交由规则 1 的全局护栏
   短路**（同上）：主体同一性都确认不了时，"一致"是最不该给出的结论。
4. **时间**：``evidence.observed_at`` 晚于旧记忆基准时间（``observed_at`` →
   ``created``，**与 P2-A freshness 同口径**，直接复用其公开的
   ``freshness.resolve_base_time`` / ``freshness.parse_ts``，不写第二份）→
   ``newer`` / ``supersede``；相等或更早、旧记忆缺基准、证据时间
   不可解析 → **不因时间判更新**（各写一行理由）。
5. **具体度**：证据的事实令牌数 **多于**旧记忆且无冲突 → ``more_specific`` /
   ``merge``。建议动作取 ``merge`` 而非 ``supersede``：证据是旧记忆事实集合的
   超集，恰当地处置是把新事实并入旧记忆（保住规范 ID），而不是换掉旧 ID；简报
   允许"supersede 或 merge"，这里选后者并留痕。**实体护栏**：两侧实体集合都非空
   且不交时，不判更具体（不同主语的长句不该被判成同一条记忆的更具体版本）。
6. **兜底**：以上都不命中 → ``uncertain`` / ``none``，理由写明"无确定性判据，
   建议人工或语义判官复核"。``judge`` 非 None 时**只在此处**调用。

judge 注入约束（逐条可测）
--------------------------

- 签名 ``Callable[[Note, EvidenceItem], str | None]``，返回五类 verdict 之一
  （大小写与空白宽松归一）或 ``None``。
- **只在确定性判定落到 uncertain 且未命中相似度地板时调用**：已确定的判定绝不
  交给模型覆写（PLAN v2 P3 立场：确定性策略先行，规则明显误判才引入分类模型，
  且其 token 成本必须计入总成本）。
- 返回 ``None``、返回非法值、或调用抛异常 → 一律**保持 uncertain**，并把原因写进
  reasons；异常被吞掉，确定性链路不受影响（判定模块不因模型侧故障而失败）。
- ``EvidenceComparison.judge_used`` 记录是否调用过（研究用成本核算）。

判官越权拦截（修复轮 3/5，结构性不变量）
--------------------------------------

**问题**：``judge`` 原先的唯一拒绝条件是 ``verdict not in VERDICTS``，而
``VERDICT_ACTIONS`` 把 ``newer→supersede``、``more_specific→merge``——即判官可以
改判出**确定性护栏刚刚拒绝过的覆盖动作**（护栏触发的场景下，判官一句
``newer`` 就能让它变成 "建议 supersede"）。更糟的是模块只暴露 ``judge_used``，
接线层无法机器可读地知道"护栏已触发"，只能去匹配中文 reasons（脆弱）。

**不变量**（本模块内约束，不靠接线层自觉）：

- ``EvidenceComparison.guard``：机器可读的护栏标记（当前只有
  ``"entities_disjoint"``；无护栏为 ``None``）。
- ``EvidenceComparison.judge_verdict``：判官给出的**原始** verdict（未调用判官或
  判官未给出合法值时为 ``None``）——即使该 verdict 被护栏拒绝，原始值仍可读，
  接线层与评测都不必去匹配中文理由。
- 护栏触发时 ``_finish_uncertain`` 把 allowlist 收窄到 ``CONSERVATIVE_VERDICTS``
  （``conflicting`` / ``uncertain``）：判官返回 ``newer`` / ``more_specific`` /
  ``consistent`` 一律**不采纳**（最终 verdict 保持 ``uncertain``、动作 ``none``），
  reasons 写明"判官建议被护栏拒绝"。无护栏时判官行为逐字段不变（回归）。
  方向选择与护栏本身一致：主体同一性未确认时，宁可少一次自动判定，不可多一次
  静默覆盖。

冲突槽位识别规则（纯函数，逐条可测）
------------------------------------

1. **令牌抽取**：按优先级 ``date > chapter > version > percent > number`` 扫描，
   先到先得、重叠丢弃（``2026-09`` 是日期而不是数字；``1.2.3`` 是版本而不是
   "1.2"）。开关：``extract_dates`` / ``extract_chapters`` / ``extract_versions``
   / ``extract_numbers``（percent 与 number 同属数值开关）。
2. **槽位键**：``kind + ':' + 上下文锚点``；数值令牌再加单位
   （``128k`` 的槽位形如 ``number:k:上下文窗口是``）。**上下文锚点** = 令牌紧邻
   的那个"词"——从令牌往前扫到最近的分词断点（空白 / 数字 / 标点，**连字符不算
   断点**），断点之后到令牌之间的字符即锚点，超过 ``context_chars``（默认 6）时取
   末 N 个字符。于是 ``GLM-5.3`` 的锚点是 ``GLM``（品牌进槽位）、
   ``共 3 个 共 5 个`` 里两个数字的锚点都是 ``共``。这与简报举例的"同一句内、同一
   量词/单位上下文"同构：量词进单位、紧邻词进锚点——用"紧邻词"而不是"固定字符
   窗口"，是为了让重复出现的同类取值**落进同一个槽位**（否则"单侧多值即跳过"这条
   护栏永远触发不到，见第 4 条）。
3. **可比性**：数值令牌必须有**单位或非空上下文**才参与比较（裸数字槽位无区分
   度，判冲突只会制造假阳性）；日期/版本/章节/百分比不看这条。
4. **单侧多值即跳过**：同一槽位在旧记忆或证据**单侧**出现多个不同取值（如
   "版本 3.5 版本 4.0 都支持"）→ 无法确定被比较的是哪一个，**跳过该槽位**并把原因
   写进 ``SlotScan.skipped``。同理，同一 ``(kind, value, slot)`` 的重复令牌只保留
   首次出现，避免重复计数影响具体度比较。
5. 抽取不到可比令牌 → **不判冲突**（简报规则 3 的原话："若抽取不到可比令牌 → 不判
   冲突"），由后续规则继续判。

其它约定
--------

- **零模型调用、零网络、全确定性**：同一 ``(prior, evidence, settings)`` 恒得同一
  结果；相似度用 ``MockEmbeddingProvider``（确定性 bigram 特征哈希）配
  ``ingest.cosine``——与 ``Ingestor`` 去重同一套度量，只是阈值尺度不同（去重
  0.9 = 近乎重复；本模块 0.8 = 断言重述，0.3 = 还值得比较）。
- **不写盘**：本模块不 import 任何写接口（``store.save_conflict`` 由 P2-D 调用）。
- 相似度与令牌都用纯函数实现，可单测；``scan_slots`` 暴露完整扫描细节供审计。
- 时间解析与基准时间直接复用 ``freshness.parse_ts`` / ``freshness.resolve_base_time``
  （修复轮 1/5 把这两个名字从私有提升为公开：口径只有一份，freshness 侧内部重构
  不再能静默改变判定结果；旧私有名保留为别名）。

与 freshness 规则 5 的取舍差异（**声明式取舍，P2-D 接线的决策输入**）
---------------------------------------------------------------------

``freshness._apply_source_change_floor`` 的口径是"``source_changed_at`` 晚于
``reviewed_at``（或没复核过）才把记忆降到至少 ``review_due``"，即**看过就不再降级**。
本模块规则 2 **不看 ``reviewed_at``**：只要 ``source_changed_at`` 存在且可解析、
证据 URL 命中旧来源、证据 ``content_hash`` 与记录不同，就建议 ``newer``（即使
``reviewed_at`` 已经晚于 ``source_changed_at``——此时会在 reasons 里额外留一行
"注：source_changed_at 不晚于 reviewed_at …本模块仍按来源变化处理"）。

为什么两侧口径可以不同（这是刻意取舍，不是疏漏）：

- 两边回答的是不同的问题。freshness 答"这条记忆**该不该复核**"，复核是一个幂等的
  用户/系统动作，看过一次就不该被同一个标记反复拉起来；本模块答"**本轮新证据相对
  旧记忆该怎么处置**"，是一个一次性判定，输出的是建议动作。
- 本模块的触发条件里有**独立于该标记的新事实**：证据侧给出了一个与旧记录**不同**
  的 ``content_hash``。哪怕旧标记已经复核过，来源内容与记忆里记录的快照仍然不同
  （P2-B 的 ``mark_source_changed`` 不改 ``sources.content_hash``），此时按 newer
  建议 supersede 是把记录追上现实。
- 若 P2-D 想采用"复核过就忽略旧标记"的口径，需要在**证据侧**加判据（例如
  "已复核过的来源变化且证据 hash 与旧记录相同"），而不是直接沿用 freshness 的
  floor——那会把上一条的新事实一并抹掉。该决策留给 P2-D（本模块的 reasons 已把
  ``reviewed_at`` 与 ``source_changed_at`` 的具体值都写出来，接线时可回放）。
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from researchwiki.wiki.embeddings import MockEmbeddingProvider
from researchwiki.wiki.freshness import parse_ts, resolve_base_time
from researchwiki.wiki.frontmatter import SourceRef
from researchwiki.wiki.ingest import cosine, entity_keys
from researchwiki.wiki.store import Note

# ---- verdict / action 常量 ---------------------------------------------------

VERDICT_CONSISTENT = "consistent"  # 一致 → 只刷新 reviewed_at
VERDICT_NEWER = "newer"  # 更新（更晚/来源换版）→ 建议 supersede
VERDICT_MORE_SPECIFIC = "more_specific"  # 更具体 → 建议并入（merge）
VERDICT_CONFLICTING = "conflicting"  # 无法同时成立 → 冲突台账
VERDICT_UNCERTAIN = "uncertain"  # 信息不足 → 只标记需复核
VERDICTS: tuple[str, ...] = (
    VERDICT_CONSISTENT,
    VERDICT_NEWER,
    VERDICT_MORE_SPECIFIC,
    VERDICT_CONFLICTING,
    VERDICT_UNCERTAIN,
)
# 护栏触发时**判官**允许产出的保守判定（修复轮 3/5 的判官越权拦截）：
# conflicting / uncertain 都不产生覆盖动作（开台账 / 不做动作），是主体同一性未
# 确认时唯一可放行的两类；newer / more_specific / consistent 都会被拒（前者建议
# supersede / merge，后者宣称"仍然成立"——都是需要主体同一性的判定）。
CONSERVATIVE_VERDICTS: tuple[str, ...] = (VERDICT_CONFLICTING, VERDICT_UNCERTAIN)

# 护栏名（EvidenceComparison.guard）：实体护栏是当前唯一的护栏，取值即"护栏名"本身，
# 供接线层机器可读地复核（memory_update.py 按 guard 非 None 拒绝 supersede/merge）。
GUARD_ENTITIES_DISJOINT = "entities_disjoint"

ACTION_REFRESH_REVIEWED_AT = "refresh_reviewed_at"
ACTION_SUPERSEDE = "supersede"
ACTION_MERGE = "merge"
ACTION_OPEN_CONFLICT = "open_conflict"
ACTION_NONE = "none"

# verdict → 建议动作（P2-D 按此分派；"建议"不等于"执行"，写入由显式通道完成）
VERDICT_ACTIONS: Mapping[str, str] = {
    VERDICT_CONSISTENT: ACTION_REFRESH_REVIEWED_AT,
    VERDICT_NEWER: ACTION_SUPERSEDE,
    VERDICT_MORE_SPECIFIC: ACTION_MERGE,
    VERDICT_CONFLICTING: ACTION_OPEN_CONFLICT,
    VERDICT_UNCERTAIN: ACTION_NONE,
}

# ---- 缺省参数（[verification] 段可覆盖）-------------------------------------

DEFAULT_SIMILARITY_FLOOR = 0.3
DEFAULT_CONSISTENT_SIMILARITY = 0.8
# 与 MockEmbeddingProvider 的默认维度同值（tests 逐值断言锁死，防单侧漂移）
DEFAULT_SIMILARITY_DIM = 128
DEFAULT_CONTEXT_CHARS = 6

# ---- 令牌抽取 ---------------------------------------------------------------

_DATE_RE = re.compile(
    r"\d{4}-\d{2}(?:-\d{2})?|\d{4}\s*年\s*\d{1,2}\s*月(?:\s*\d{1,2}\s*日)?"
)
_CHAPTER_RE = re.compile(r"第\s*\d+\s*[章节条款]|(?:[Cc]hapter|CH)\s*\d+")
_VERSION_RE = re.compile(r"[vV]?\d+(?:\.\d+)+(?:[-.][A-Za-z][\w.]*)?")
_PERCENT_RE = re.compile(r"\d+(?:\.\d+)?\s*%")
_NUMBER_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:([A-Za-z%]{1,4}|[年月日天周次个条块倍位万亿人]))?")

# 抽取优先级：数值越小越先占用字符区间（日期 > 章节 > 版本 > 百分比 > 普通数值）
_KIND_PRIORITY: Mapping[str, int] = {
    "date": 0,
    "chapter": 1,
    "version": 2,
    "percent": 3,
    "number": 4,
}
# (kind, 正则, settings 开关字段名)
_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("date", _DATE_RE, "extract_dates"),
    ("chapter", _CHAPTER_RE, "extract_chapters"),
    ("version", _VERSION_RE, "extract_versions"),
    ("percent", _PERCENT_RE, "extract_numbers"),
    ("number", _NUMBER_RE, "extract_numbers"),
)

_STRIP_CHARS = "，。；、：:,.!?！？()（）「」【】[]\"'`*#>-_"
_CJK_DATE_RE = re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月(?:\s*(\d{1,2})\s*日)?")
# 分词断点：空白、数字、标点（**不含连字符**——"GLM-5.3" 的紧邻词要保住 "GLM"）
_RUN_BREAK_RE = re.compile(r"[\s\d，。；、：:,.!?！？()（）「」【】\[\]\"'`*#>_+/=~|&]")


# ---- 数据结构 ---------------------------------------------------------------


@dataclass
class EvidenceItem:
    """一条本轮新证据：正文 + 可选时间/来源/实体。

    - ``observed_at``：这条证据的观察时间（ISO 字符串，可缺省）；缺省时时间规则
      一律不生效（绝不拿"无时间的证据"去判谁更新）。
    - ``source_url`` / ``content_hash``：来源定位。只有**同时**给出可解析的
      source_url（命中旧记忆的 SourceRef）与 content_hash 才可能触发来源变化判定。
    - ``entities``：本条证据的实体名（与旧记忆实体按 slugify 归一边比较，仅用于
      具体度判定的护栏）。
    """

    text: str
    observed_at: str | None = None
    source_url: str = ""
    content_hash: str = ""
    entities: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class FactToken:
    """一个可抽取的事实令牌：数值/日期/版本/章节/百分比 + 它所在的槽位。"""

    kind: str
    value: str  # 归一后的取值（比较用）
    raw: str  # 原文片段（理由引用用）
    slot: str  # 槽位键 = kind[:unit] + 上下文锚点
    context: str  # 上下文锚点（数字已在锚点里归一为 #）
    unit: str = ""  # 量词/单位（仅数值令牌可能有）

    def label(self) -> str:
        """简短标签（理由文案与新增令牌列表用）。"""
        return f"{self.kind}:{self.value}"


@dataclass(frozen=True)
class SlotConflict:
    """一个槽位上的取值冲突：旧记忆与证据在同一槽位给出不同值。"""

    slot: str
    kind: str
    context: str
    prior_value: str
    evidence_value: str

    def describe(self) -> str:
        """人类可读描述（引用槽位、上下文与两侧具体取值）。"""
        anchor = self.context or "无"
        return (
            f"槽位 {self.slot}（{self.kind}，上下文锚点「{anchor}」）"
            f"旧={self.prior_value} 新={self.evidence_value}"
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "slot": self.slot,
            "kind": self.kind,
            "context": self.context,
            "prior_value": self.prior_value,
            "evidence_value": self.evidence_value,
        }


@dataclass(frozen=True)
class SourceComparison:
    """来源侧的结构化留痕：命中的来源 URL 与**完整**新旧 content_hash。

    reasons 里的 hash 只有前 8 位（人类可读），字面截断在哈希碰撞时无法区分；本结构
    带全量 hash，供 P2-D 写台账/复算时逐字比对（``EvidenceComparison.source``）。
    """

    url: str
    prior_content_hash: str
    evidence_content_hash: str
    source_changed_at: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "url": self.url,
            "prior_content_hash": self.prior_content_hash,
            "evidence_content_hash": self.evidence_content_hash,
            "source_changed_at": self.source_changed_at,
        }


@dataclass
class SlotScan:
    """一次冲突扫描的完整细节（供审计与单测；``detect_slot_conflicts`` 只回 conflicts）。

    ``slots_agree``：旧记忆的**每个可比槽位**在证据中都有同一槽位且取值相同，**且**
    证据没有多余的可比槽位——即两侧的可比"槽位 → 取值"映射完全相等。判定"一致"
    必须用它（槽位级），不能用"证据的每个 (kind,value) 是否在旧文任何位置出现过"
    （集合级）：后者会把"取值错位互换"（如旧 `上下文窗口 128k，参数量 70B` 对新
    `上下文窗口 70B，参数量 128k`）判成仍然成立。缺省 False = 不宣称一致（方向安全）。
    """

    conflicts: list[SlotConflict] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # 单侧多值等跳过说明
    prior_tokens: list[FactToken] = field(default_factory=list)
    evidence_tokens: list[FactToken] = field(default_factory=list)
    new_tokens: list[FactToken] = field(default_factory=list)
    slots_agree: bool = False


@dataclass
class EvidenceComparison:
    """一条旧记忆 vs 一条新证据的判定结论（六个简报字段 + 六个审计字段）。

    - ``verdict``：五类之一（VERDICTS）。
    - ``reasons``：人类可读判定依据，每条引用具体值（时间戳、hash 前 8 位、令牌
      取值、相似度数值），供评测阶段逐条复查。
    - ``prior_note_id`` / ``evidence_index``：旧记忆 id 与该证据在输入列表中的下标
      （回填留痕用，``compare_batch`` 保证下标与输入一一对应）。
    - ``suggested_action``：VERDICT_ACTIONS 给出的建议动作，**只是建议**。
    - ``similarity`` / ``conflicts`` / ``judge_used`` / ``source``：审计字段（超出简报
      字段清单但只增不改语义）——相似度数值、命中的冲突槽位（P2-D 写台账时引用具体
      取值）、是否调用过语义判官（成本核算）、来源侧全量 hash 留痕（理由里只有 8 位
      截断，碰撞时字面相同）。
    - ``guard`` / ``judge_verdict``（修复轮 3/5）：机器可读的安全不变量字段——
      前者是触发的护栏名（``None`` = 无护栏），后者是判官给出的**原始** verdict
      （``None`` = 未调用或未给出合法值）。护栏触发时判官只能产出保守判定，
      被拒的原始值仍从这里可读（详见模块 docstring「判官越权拦截」）。
    """

    verdict: str
    reasons: list[str]
    prior_note_id: str
    evidence_index: int
    suggested_action: str
    similarity: float = 0.0
    conflicts: list[SlotConflict] = field(default_factory=list)
    judge_used: bool = False
    source: SourceComparison | None = None
    guard: str | None = None
    judge_verdict: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """序列化（留痕 / lint --json / MCP 复用；reasons 原样保留）。

        ``source`` 带**完整**新旧 content_hash（理由里只有前 8 位截断），供台账复算；
        ``guard`` / ``judge_verdict`` 是护栏与判官的机器可读留痕（接线层按它们
        复核动作，不必匹配中文 reasons）。
        """
        return {
            "prior_note_id": self.prior_note_id,
            "evidence_index": self.evidence_index,
            "verdict": self.verdict,
            "suggested_action": self.suggested_action,
            "similarity": self.similarity,
            "judge_used": self.judge_used,
            "guard": self.guard,
            "judge_verdict": self.judge_verdict,
            "reasons": list(self.reasons),
            "conflicts": [c.to_dict() for c in self.conflicts],
            "source": self.source.to_dict() if self.source is not None else None,
        }


@dataclass
class VerificationSettings:
    """比较判定的全部参数（集中可配；默认值见模块常量）。

    - ``similarity_floor``：相似度地板（规则 0）；低于它直接 uncertain，不比较、
      不调用 judge。
    - ``consistent_similarity``：判定"断言重述"的相似度阈值（规则 3）；小于地板时
      被夹到地板（否则规则 3 不可能命中，判定退化）。
    - ``similarity_dim``：bigram 特征哈希向量的维度（8–4096，缺省与
      MockEmbeddingProvider 一致）。
    - ``context_chars``：槽位上下文锚点长度（1–24 字符）。
    - ``extract_dates`` / ``extract_chapters`` / ``extract_versions`` /
      ``extract_numbers``：令牌抽取开关（数值开关同时管百分比）。

    本类在 ``__post_init__`` 里做宽容归一：比率夹到 [0.0, 1.0]、维度/长度夹到合法
    区间、开关只认"真/假"语义的值，非法值回退默认且**不抛异常**（与
    formation / freshness 的宽容风格一致：配置错误不打断研究链路）。
    """

    similarity_floor: float = DEFAULT_SIMILARITY_FLOOR
    consistent_similarity: float = DEFAULT_CONSISTENT_SIMILARITY
    similarity_dim: int = DEFAULT_SIMILARITY_DIM
    context_chars: int = DEFAULT_CONTEXT_CHARS
    extract_dates: bool = True
    extract_chapters: bool = True
    extract_versions: bool = True
    extract_numbers: bool = True

    def __post_init__(self) -> None:
        self.similarity_floor = _ratio(self.similarity_floor, DEFAULT_SIMILARITY_FLOOR)
        self.consistent_similarity = max(
            _ratio(self.consistent_similarity, DEFAULT_CONSISTENT_SIMILARITY),
            self.similarity_floor,
        )
        self.similarity_dim = _int_in_range(
            self.similarity_dim, DEFAULT_SIMILARITY_DIM, low=8, high=4096
        )
        self.context_chars = _int_in_range(
            self.context_chars, DEFAULT_CONTEXT_CHARS, low=1, high=24
        )
        self.extract_dates = _flag(self.extract_dates, True)
        self.extract_chapters = _flag(self.extract_chapters, True)
        self.extract_versions = _flag(self.extract_versions, True)
        self.extract_numbers = _flag(self.extract_numbers, True)

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None = None) -> VerificationSettings:
        """解析配置：接受完整 config（取 ``[verification]`` 段）或直接给段。

        支持形态（缺省 / None = 全默认；非法值宽容回退，不抛异常）::

            [verification]
            similarity_floor = 0.3
            consistent_similarity = 0.8
            similarity_dim = 128
            context_chars = 6
            extract_dates = true
            extract_numbers = true

        映射型非已知键一律忽略（本模块没有按 kind 分段的语义）；布尔开关接受
        ``true/false``、``1/0``、``yes/no``、``on/off``（字符串宽松归一）。
        """
        section: Mapping[str, Any] = {}
        if isinstance(config, Mapping):
            raw = config.get("verification")
            section = raw if isinstance(raw, Mapping) else config
        return cls(
            similarity_floor=_ratio(
                section.get("similarity_floor"), DEFAULT_SIMILARITY_FLOOR
            ),
            consistent_similarity=_ratio(
                section.get("consistent_similarity"), DEFAULT_CONSISTENT_SIMILARITY
            ),
            similarity_dim=_int_in_range(
                section.get("similarity_dim"), DEFAULT_SIMILARITY_DIM, low=8, high=4096
            ),
            context_chars=_int_in_range(
                section.get("context_chars"), DEFAULT_CONTEXT_CHARS, low=1, high=24
            ),
            extract_dates=_flag(section.get("extract_dates"), True),
            extract_chapters=_flag(section.get("extract_chapters"), True),
            extract_versions=_flag(section.get("extract_versions"), True),
            extract_numbers=_flag(section.get("extract_numbers"), True),
        )


def _ratio(value: Any, default: float) -> float:
    """比率宽容归一：None / 非法 / bool / NaN 回退 default，其余夹到 [0.0, 1.0]。"""
    if value is None or isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:  # NaN
        return default
    return min(1.0, max(0.0, number))


def _int_in_range(value: Any, default: int, *, low: int, high: int) -> int:
    """整数宽容归一：None / 非法 / bool 回退 default，越界夹到 [low, high]。"""
    if value is None or isinstance(value, bool):
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return min(high, max(low, number))


_TRUE_WORDS = ("1", "true", "yes", "on", "y", "t")
_FALSE_WORDS = ("0", "false", "no", "off", "n", "f")


def _flag(value: Any, default: bool) -> bool:
    """布尔开关宽容归一：真 bool 直接认，字符串按真/假词表认，其余回 default。"""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, int | float):
        return bool(value)
    text = str(value).strip().lower()
    if text in _TRUE_WORDS:
        return True
    if text in _FALSE_WORDS:
        return False
    return default


# ---- 令牌抽取与槽位（纯函数）------------------------------------------------


def _context_anchor(text: str, start: int, context_chars: int) -> str:
    """令牌紧邻的上下文锚点：向前取到最近的分词断点（空白/数字/标点）之后的**词**。

    断点集合刻意不含连字符——``GLM-5.3`` 的锚点是 ``GLM``；``共 3 个 共 5 个``
    里两个数字的锚点都是 ``共``（于是同一槽位出现两个取值 → 走"单侧多值即跳过"，
    不误判冲突）。锚点长于 ``context_chars`` 时取末 N 个字符。
    """
    prefix = str(text or "")[: max(0, start)].rstrip()
    cut = 0
    for index in range(len(prefix) - 1, -1, -1):
        if _RUN_BREAK_RE.match(prefix[index]):
            cut = index + 1
            break
    return prefix[cut:].strip(_STRIP_CHARS)[-context_chars:]


def _normalize_value(kind: str, raw: str) -> str:
    """取值归一（比较用）：日期补零、中文日期转 ISO、版本去前缀、数值去多余 0。

    单位不进取值（它在槽位键里）——只比"数值本身"，``3 个`` 与 ``3个`` 才不会被
    判成两个不同的值。
    """
    if kind == "date":
        match = _CJK_DATE_RE.search(raw)
        if match is not None:
            year, month, day = match.group(1), match.group(2), match.group(3)
            iso = f"{int(year):04d}-{int(month):02d}"
            return f"{iso}-{int(day):02d}" if day else iso
        return raw.strip()
    if kind == "chapter":
        digits = re.search(r"\d+", raw)
        number = str(int(digits.group(0))) if digits else raw.strip()
        lowered = raw.lower()
        if "章" in raw or "chapter" in lowered or "ch" in lowered:
            unit_char = "章"
        elif "节" in raw:
            unit_char = "节"
        else:
            unit_char = "条" if "条" in raw else "款"
        return f"{number}{unit_char}"
    if kind == "version":
        return raw.lstrip("vV").strip().rstrip(".-").lower()
    if kind == "percent":
        return raw.replace(" ", "")
    try:
        return f"{float(raw):g}"
    except (TypeError, ValueError):
        return raw.strip()


def extract_fact_tokens(
    text: str, *, settings: VerificationSettings | None = None
) -> list[FactToken]:
    """抽取文本里的事实令牌（纯函数；顺序即出现顺序，同一 (kind,value,slot) 去重）。

    抽取优先级与槽位规则见模块 docstring；重叠片段按优先级先到先得
    （``2026-09`` 归日期、``1.2.3`` 归版本、``128k`` 归数值）。开关关闭的类别不抽。
    """
    cfg = settings if settings is not None else VerificationSettings()
    raw_text = str(text or "")
    candidates: list[tuple[int, int, str, str, str]] = []
    for kind, pattern, flag_name in _PATTERNS:
        if not getattr(cfg, flag_name):
            continue
        for match in pattern.finditer(raw_text):
            unit = ""
            raw_token = match.group(0)
            if kind == "number":
                # 数值本体取 group(1)：group(0) 含空格与量词（"3 个"），带进取值会让
                # "3 个" 与 "3个" 判成不同值，制造假冲突
                found = match.group(2)
                unit = str(found).strip().lower() if found else ""
                raw_token = match.group(1)
            candidates.append((match.start(), match.end(), kind, raw_token, unit))
    candidates.sort(key=lambda item: (item[0], _KIND_PRIORITY[item[2]], -item[1]))

    tokens: list[FactToken] = []
    seen: set[tuple[str, str, str]] = set()
    last_end = -1
    for start, end, kind, raw, unit in candidates:
        if start < last_end:  # 与已占用区间重叠 → 丢弃（优先级已保证先到先得）
            continue
        last_end = end
        context = _context_anchor(raw_text, start, cfg.context_chars)
        value = _normalize_value(kind, raw)
        slot = f"{kind}:{unit}:{context}" if kind == "number" else f"{kind}:{context}"
        key = (kind, value, slot)
        if key in seen:
            continue
        seen.add(key)
        tokens.append(
            FactToken(kind=kind, value=value, raw=raw, slot=slot, context=context, unit=unit)
        )
    return tokens


def _is_comparable(token: FactToken) -> bool:
    """数值令牌必须有单位或非空上下文才参与冲突比较（裸数字不比较）。"""
    if token.kind != "number":
        return True
    return bool(token.unit) or bool(token.context)


def detect_slot_conflicts(
    prior_text: str, evidence_text: str, *, settings: VerificationSettings | None = None
) -> list[SlotConflict]:
    """纯函数：抽取双方令牌，返回**同槽位不同取值**的冲突列表（按槽位排序）。

    不判冲突的情形（逐条见模块 docstring）：抽取不到可比令牌、数值令牌无单位无
    上下文、同一槽位在单侧有多个不同取值（无法确定比较对象）。
    """
    return scan_slots(prior_text, evidence_text, settings=settings).conflicts


def scan_slots(
    prior_text: str, evidence_text: str, *, settings: VerificationSettings | None = None
) -> SlotScan:
    """完整扫描：冲突 + 跳过说明 + 双方令牌 + 证据新增令牌（纯函数、确定性）。"""
    cfg = settings if settings is not None else VerificationSettings()
    prior_tokens = extract_fact_tokens(prior_text, settings=cfg)
    evidence_tokens = extract_fact_tokens(evidence_text, settings=cfg)

    prior_slots = _slot_values(prior_tokens)
    evidence_slots = _slot_values(evidence_tokens)
    conflicts: list[SlotConflict] = []
    skipped: list[str] = []
    for slot in sorted(set(prior_slots) & set(evidence_slots)):
        left = prior_slots[slot]
        right = evidence_slots[slot]
        left_multi = len(left) > 1
        right_multi = len(right) > 1
        if left_multi or right_multi:
            sides = "与".join(
                name for name, multi in (("旧记忆", left_multi), ("证据", right_multi)) if multi
            )
            skipped.append(
                f"槽位 {slot} 在{sides}有多个取值"
                f"（旧={_join_values(left)}；新={_join_values(right)}），"
                "无法确定比较对象 → 跳过该槽位"
            )
            continue
        if left[0] != right[0]:
            conflicts.append(
                SlotConflict(
                    slot=slot,
                    kind=_slot_kind(slot),
                    context=_slot_context(slot),
                    prior_value=left[0],
                    evidence_value=right[0],
                )
            )
    prior_labels = {(t.kind, t.value) for t in prior_tokens}
    return SlotScan(
        conflicts=conflicts,
        skipped=skipped,
        prior_tokens=prior_tokens,
        evidence_tokens=evidence_tokens,
        new_tokens=[t for t in evidence_tokens if (t.kind, t.value) not in prior_labels],
        slots_agree=_slots_agree(prior_slots, evidence_slots),
    )


def _slots_agree(left: Mapping[str, list[str]], right: Mapping[str, list[str]]) -> bool:
    """两侧的"可比槽位 → 取值"映射是否完全相等（槽位级一致判据，见 SlotScan 注释）。

    要求：键集合相同、每个槽位两侧都只有一个取值且相等。任一槽位在单侧有多个取值
    （无法确定比较对象）即判不成立——"仍然成立"必须建立在两边**逐槽位**对得上之上。
    两个空映射（两侧都抽不到可比令牌）恒等，返回 True（无可比事实即无可反驳）。
    """
    if set(left) != set(right):
        return False
    for slot, values in left.items():
        if len(values) != 1 or right[slot] != values:
            return False
    return True


def _slot_summary(slots: Mapping[str, list[str]], *, limit: int = 4) -> str:
    """槽位映射的可读摘要（理由文案用）：``槽位键=取值`` 按槽位排序、超限截断。"""
    rendered = [f"{slot}={_join_values(slots[slot])}" for slot in sorted(slots)]
    if not rendered:
        return "无"
    shown = "、".join(rendered[:limit])
    return shown + ("…" if len(rendered) > limit else "")


def _slot_kind(slot: str) -> str:
    """从槽位键取 kind（``number:k:锚点`` 与 ``date:锚点`` 都取第一段）。"""
    return slot.split(":", 1)[0]


def _slot_context(slot: str) -> str:
    """从槽位键取上下文锚点（末段）。"""
    return slot.rsplit(":", 1)[-1]


def _join_values(values: Sequence[str]) -> str:
    """取值列表的可读拼接（空列表写"无"，理由文案不留空白）。"""
    return "、".join(values) if values else "无"


def _slot_values(tokens: Sequence[FactToken]) -> dict[str, list[str]]:
    """按槽位聚合可比令牌的取值（保序、去重；不可比的数值令牌被排除）。"""
    out: dict[str, list[str]] = {}
    for token in tokens:
        if not _is_comparable(token):
            continue
        values = out.setdefault(token.slot, [])
        if token.value not in values:
            values.append(token.value)
    return out


# ---- 相似度（复用 MockEmbeddingProvider + ingest.cosine）---------------------


@lru_cache(maxsize=8)
def _embedder(dim: int) -> MockEmbeddingProvider:
    """确定性 bigram 特征哈希嵌入器（零网络、跨进程可复现），按维度缓存。"""
    return MockEmbeddingProvider(dim=dim)


def token_similarity(
    left: str, right: str, *, settings: VerificationSettings | None = None
) -> float:
    """轻量文本相似度：bigram 特征哈希向量的余弦（口径同 Ingestor 的去重打分）。

    - 复用 ``MockEmbeddingProvider``（确定性、零网络）与 ``ingest.cosine``，不新造
      轮子；任一侧为空串（零向量）→ 0.0。
    - 维度由 ``settings.similarity_dim`` 决定；结果恒在 [0.0, 1.0]。
    """
    cfg = settings if settings is not None else VerificationSettings()
    left_text = str(left or "")
    right_text = str(right or "")
    if not left_text.strip() or not right_text.strip():
        return 0.0
    vectors = _embedder(cfg.similarity_dim).embed([left_text, right_text])
    return min(1.0, max(0.0, cosine(vectors[0], vectors[1])))


# ---- 主入口 -----------------------------------------------------------------


def compare_prior_and_evidence(
    prior: Note,
    evidence: EvidenceItem,
    *,
    settings: VerificationSettings | None = None,
    judge: Callable[[Note, EvidenceItem], str | None] | None = None,
    evidence_index: int = 0,
) -> EvidenceComparison:
    """比较一条旧记忆与一条新证据，返回结构化判定（规则 0–6 见模块 docstring）。

    ``judge`` 只在确定性判定落到 uncertain（且未命中相似度地板）时被调用；
    本函数不改笔记、不写台账、不访问网络，同一入参恒得同一结果。
    """
    cfg = settings if settings is not None else VerificationSettings()
    reasons: list[str] = []
    # 来源解析与实体护栏都只读、且被多条规则共用，先算好（保证每条返回路径都能带上
    # 结构化留痕；判定口径本身不受此顺序影响——它们都不写 reasons 以外的状态）。
    ref = _match_source_ref(prior, evidence)
    source_trace = SourceComparison(
        url=ref.url,
        prior_content_hash=ref.content_hash,
        evidence_content_hash=evidence.content_hash,
        source_changed_at=str(prior.meta.source_changed_at or ""),
    ) if ref is not None else None
    entities_disjoint, prior_entity_keys, evidence_entity_keys = _entities_disjoint(prior, evidence)

    # 规则 0：相似度地板（先决门；命中即 uncertain，且不调用 judge）
    similarity = token_similarity(prior.body, evidence.text, settings=cfg)
    reasons.append(
        f"相似度 {similarity:.3f}（bigram 特征哈希余弦 dim={cfg.similarity_dim}；"
        f"地板 {cfg.similarity_floor:.2f}、一致阈值 {cfg.consistent_similarity:.2f}）"
    )
    if similarity < cfg.similarity_floor:
        reasons.append(
            f"相似度 {similarity:.3f} 低于地板 {cfg.similarity_floor:.2f}"
            "：证据与旧记忆不构成同一断言，不进入确定性比较"
            "（也不调用语义判官——地板就是为了不让无关证据消耗比较器）→ uncertain"
        )
        return _make(
            prior,
            evidence_index,
            VERDICT_UNCERTAIN,
            reasons,
            similarity=similarity,
            source=source_trace,
        )

    # 冲突门（规则 1）+ 审计行
    scan = scan_slots(prior.body, evidence.text, settings=cfg)
    reasons.append(_token_summary(scan))
    reasons.extend(scan.skipped)
    if entities_disjoint:
        # 实体护栏（修复轮 2/5 提升为**全局前置**）：两侧实体集合都非空且不交 =
        # 无法确认主体同一性。此时槽位键里不含主语，任何"更新/一致/更具体"结论都
        # 站不住：判 conflicting 会误开冲突台账，判 newer 会建议覆盖一个可能不是
        # 同一主语的记忆，判 consistent 会把"假一致"当成仍然成立——三者都违背
        # "不静默覆盖"。故这里一次短路掉规则 2/3/4/5，直接进兜底（uncertain/none，
        # judge 仍可改判）。
        boundary = (
            f"两侧实体不交（旧={'、'.join(prior_entity_keys)}；"
            f"新={'、'.join(evidence_entity_keys)}）"
        )
        reasons.append(
            f"{boundary}：槽位键里不含主语 → 跳过冲突判定"
            "（主语不同时同槽位不同值不构成对同一条记忆的反驳）"
        )
        return _finish_uncertain(
            prior,
            evidence,
            evidence_index,
            reasons,
            similarity=similarity,
            source=source_trace,
            judge=judge,
            blockers="实体不交，无法确认主体同一性 → 不做更新判定"
            "（不判 newer / consistent / 更具体，也不据此开冲突台账）",
            guard=GUARD_ENTITIES_DISJOINT,
        )
    if scan.conflicts:
        reasons.append(
            f"冲突检测命中 {len(scan.conflicts)} 个槽位："
            + "；".join(conflict.describe() for conflict in scan.conflicts)
        )
        reasons.append(
            "同槽位取值无法同时成立 → conflicting（进冲突台账，不静默覆盖；"
            "优先级高于来源变化与时间规则）"
        )
        return _make(
            prior,
            evidence_index,
            VERDICT_CONFLICTING,
            reasons,
            similarity=similarity,
            conflicts=scan.conflicts,
            source=source_trace,
        )

    # 规则 2：来源变化（url 命中旧来源的分支）
    if ref is not None:
        if not evidence.content_hash:
            reasons.append(
                f"来源 URL 命中旧来源 {ref.url}，但证据未给 content_hash，"
                "无法判定是否换版 → 不据此判定"
            )
        elif evidence.content_hash == ref.content_hash:
            reasons.append(
                f"来源未变：url={ref.url}，content_hash={_short_hash(evidence.content_hash)} "
                "与旧记忆记录一致（同一快照）→ 不判更新，继续比较表述与事实"
            )
        else:
            changed = parse_ts(prior.meta.source_changed_at)
            if changed is not None:
                reasons.append(
                    f"来源变化：url={ref.url} 命中旧来源，content_hash "
                    f"旧={_short_hash(ref.content_hash)} 新={_short_hash(evidence.content_hash)}"
                    f"（source_changed_at={changed.isoformat()}）"
                    " → 新证据更新（建议 supersede）"
                )
                reviewed = parse_ts(prior.meta.reviewed_at)
                if reviewed is not None and changed <= reviewed:
                    # 与 freshness 规则 5 的取舍不同，显式留痕（见模块 docstring
                    # 「与 freshness 规则 5 的取舍差异」）：本模块判"新证据怎么处置"，
                    # 触发条件里还有证据侧的新 hash，不看复核时间。
                    reasons.append(
                        f"注：source_changed_at 不晚于 reviewed_at="
                        f"{reviewed.isoformat()}（旧记忆已复核过），本模块仍按来源变化"
                        "处理——本模块判的是「新证据怎么处置」，且证据侧 hash 与记录不同"
                        "本身即新事实（取舍见模块 docstring）"
                    )
                reasons.extend(_new_token_notes(scan))
                return _make(
                    prior,
                    evidence_index,
                    VERDICT_NEWER,
                    reasons,
                    similarity=similarity,
                    source=source_trace,
                )
            if prior.meta.source_changed_at:
                reasons.append(
                    f"source_changed_at={prior.meta.source_changed_at} 不是合法 ISO 时间，"
                    "按未声明处理（不参与判定）"
                )
            reasons.append(
                f"来源哈希不同（旧={_short_hash(ref.content_hash)} "
                f"新={_short_hash(evidence.content_hash)}），但旧记忆未声明 "
                "source_changed_at → 本轮不据此判定"
            )

    # 规则 3：一致（高相似重述、无新增令牌且**槽位级取值对得上**）→ 只刷新 reviewed_at
    if similarity >= cfg.consistent_similarity and not scan.new_tokens:
        blockers = _consistency_blockers(scan)
        if not blockers:
            reasons.append(
                f"一致：相似度 {similarity:.3f} ≥ 一致阈值 {cfg.consistent_similarity:.2f}，"
                "且证据未引入新的事实令牌、无冲突、槽位级取值逐一对得上"
                " → consistent（只需刷新 reviewed_at）"
            )
            return _make(
                prior,
                evidence_index,
                VERDICT_CONSISTENT,
                reasons,
                similarity=similarity,
                source=source_trace,
            )
        reasons.append(
            "一致判定被保留：" + "；".join(blockers) + " → 不判 consistent，交后续规则或人工复核"
        )

    # 规则 4：时间先后（基准口径与 P2-A freshness 一致：observed_at → created）
    evidence_ts = parse_ts(evidence.observed_at)
    if evidence.observed_at and evidence_ts is None:
        reasons.append(
            f"证据 observed_at={evidence.observed_at} 不是合法 ISO 时间，按未提供处理"
            "（不参与判定）"
        )
    base_field, base_ts = resolve_base_time(prior.meta)
    if evidence_ts is not None and base_ts is not None:
        if evidence_ts > base_ts:
            reasons.append(
                f"时间：证据 observed_at={evidence_ts.isoformat()} 晚于旧记忆 "
                f"{base_field}={base_ts.isoformat()} → 新证据更新（建议 supersede）"
            )
            reasons.extend(_new_token_notes(scan))
            return _make(
                prior,
                evidence_index,
                VERDICT_NEWER,
                reasons,
                similarity=similarity,
                source=source_trace,
            )
        reasons.append(
            f"时间：证据 observed_at={evidence_ts.isoformat()} 不晚于旧记忆 "
            f"{base_field}={base_ts.isoformat()}（相等或更早）→ 不因时间判更新"
        )
    elif evidence_ts is not None:
        reasons.append(
            "时间：旧记忆无时间基准（observed_at 与 created 均缺失或不可解析），"
            "无法比较时间先后 → 不因时间判更新"
        )
    else:
        reasons.append("时间：证据未提供可解析的 observed_at → 不因时间判更新")

    # 规则 5：具体度（事实令牌更多）。实体护栏已在上方全局短路，这里不必再判实体
    if len(scan.evidence_tokens) > len(scan.prior_tokens):
        fresh = "、".join(t.label() for t in scan.new_tokens) or "无"
        reasons.append(
            f"具体度：证据事实令牌 {len(scan.evidence_tokens)} 个 > 旧记忆 "
            f"{len(scan.prior_tokens)} 个（{fresh}）"
            " 且无冲突 → 更具体（建议 merge 并入旧记忆，保留规范 ID）"
        )
        return _make(
            prior,
            evidence_index,
            VERDICT_MORE_SPECIFIC,
            reasons,
            similarity=similarity,
            source=source_trace,
        )
    reasons.append(
        f"具体度：证据事实令牌 {len(scan.evidence_tokens)} 个 不多于 旧记忆 "
        f"{len(scan.prior_tokens)} 个 → 不判更具体"
    )

    # 规则 6：兜底（judge 只在此处注入）
    return _finish_uncertain(
        prior,
        evidence,
        evidence_index,
        reasons,
        similarity=similarity,
        source=source_trace,
        judge=judge,
        blockers=(
            "无确定性判据：相似度高于地板、无冲突，但时间不更新、也无新增事实令牌可用，"
            "既不足以判「一致」也不足以判「更具体」"
        ),
    )


def compare_batch(
    prior: Note,
    evidence: Sequence[EvidenceItem],
    *,
    settings: VerificationSettings | None = None,
    judge: Callable[[Note, EvidenceItem], str | None] | None = None,
) -> list[EvidenceComparison]:
    """批量比较（保序）：**每条证据恰好产出一条结论**，``evidence_index`` 与输入下标对齐。

    空正文的证据不跳过——留痕的下标必须能回填到输入列表，跳过会让下游错位；
    空正文自然因相似度 0.0 落到 uncertain / none。
    """
    return [
        compare_prior_and_evidence(
            prior, item, settings=settings, judge=judge, evidence_index=index
        )
        for index, item in enumerate(evidence)
    ]


# ---- 内部工具 ---------------------------------------------------------------


def _make(
    prior: Note,
    evidence_index: int,
    verdict: str,
    reasons: list[str],
    *,
    similarity: float = 0.0,
    conflicts: Sequence[SlotConflict] = (),
    judge_used: bool = False,
    source: SourceComparison | None = None,
    guard: str | None = None,
    judge_verdict: str | None = None,
) -> EvidenceComparison:
    """按 verdict 组装结论（动作查 VERDICT_ACTIONS 表；reasons 就地共享同一列表）。

    ``guard`` / ``judge_verdict`` 是机器可读的安全留痕（见模块 docstring
    「判官越权拦截」）：前者记录触发的护栏名，后者记录判官给出的**原始** verdict
    （即使它被护栏拒绝也照样可读）。
    """
    return EvidenceComparison(
        verdict=verdict,
        reasons=reasons,
        prior_note_id=prior.id,
        evidence_index=evidence_index,
        suggested_action=VERDICT_ACTIONS[verdict],
        similarity=similarity,
        conflicts=list(conflicts),
        judge_used=judge_used,
        source=source,
        guard=guard,
        judge_verdict=judge_verdict,
    )


def _consistency_blockers(scan: SlotScan) -> list[str]:
    """列出"一致"为何被保留（空列表 = 可以判 consistent）。

    两条护栏，都指向同一件事：**必须逐槽位比过**才能说"仍然成立"。
    - 有槽位单侧多值 → 无法确定比较对象（可能藏着取值差异）；
    - 槽位级取值对不上（旧记忆的某个可比槽位在证据里没有对应槽位、或取值不同）
      → 典型形态是"取值错位互换"（表格列序/标签错位），集合级检查看不见它
      （旧文出现过的取值在新文里都还在，只是换了槽位）。
    """
    blockers: list[str] = []
    if scan.skipped:
        blockers.append("有槽位因单侧多值无法比较（可能藏着取值差异）")
    if not scan.slots_agree:
        prior_slots = _slot_summary(_slot_values(scan.prior_tokens))
        evidence_slots = _slot_summary(_slot_values(scan.evidence_tokens))
        blockers.append(f"槽位级取值对不上（旧={prior_slots}；新={evidence_slots}）")
    return blockers


def _finish_uncertain(
    prior: Note,
    evidence: EvidenceItem,
    evidence_index: int,
    reasons: list[str],
    *,
    similarity: float,
    source: SourceComparison | None,
    judge: Callable[[Note, EvidenceItem], str | None] | None,
    blockers: str,
    guard: str | None = None,
) -> EvidenceComparison:
    """兜底出口（规则 6 与"实体不交"全局短路共用）：默认 uncertain，judge 可改判。

    ``blockers`` 是"为什么确定性判据不足"的一句（引用具体值）；结句按**最终**判定
    生成——判官改判后不能再留一句"→ uncertain"与 verdict 自相牵制（修复轮 1/5 Minor c）。
    judge=None 时本函数是纯确定性路径（uncertain / none）。

    ``guard`` 非 None 时判官的 allowlist 收窄到 ``CONSERVATIVE_VERDICTS``（修复轮
    3/5 的判官越权拦截）：判官给出的覆盖动作（newer / more_specific / consistent）
    不采纳，最终 verdict 保持 uncertain / none，但 ``judge_verdict`` 仍记录原始值
    ——接线层不必匹配中文 reasons 就能看出"判官想覆盖、护栏拦住了"。
    """
    judge_used = judge is not None
    verdict = VERDICT_UNCERTAIN
    judge_verdict: str | None = None
    if judge is not None:
        allowed = CONSERVATIVE_VERDICTS if guard is not None else VERDICTS
        judged, accepted = _apply_judge(
            judge,
            prior,
            evidence,
            reasons,
            allowed_verdicts=allowed,
            guard=guard,
        )
        judge_verdict = judged
        if accepted and judged is not None:
            verdict = judged
    if verdict == VERDICT_UNCERTAIN:
        reasons.append(f"{blockers} → uncertain（建议人工或语义判官复核，本轮不做动作）")
    else:
        reasons.append(
            f"{blockers} → 最终判定 {verdict}"
            f"（由语义判官给出，建议动作 {VERDICT_ACTIONS[verdict]}）"
        )
    return _make(
        prior,
        evidence_index,
        verdict,
        reasons,
        similarity=similarity,
        judge_used=judge_used,
        source=source,
        guard=guard,
        judge_verdict=judge_verdict,
    )


def _apply_judge(
    judge: Callable[[Note, EvidenceItem], str | None],
    prior: Note,
    evidence: EvidenceItem,
    reasons: list[str],
    *,
    allowed_verdicts: Sequence[str] = VERDICTS,
    guard: str | None = None,
) -> tuple[str | None, bool]:
    """调用语义判官（**仅在 uncertain 路径**）；返回 ``(原始 verdict | None, 是否采纳)``。

    返回原始 verdict 而不是"采纳后的判定"，是修复轮 3/5 的接口选择：护栏拒绝判官
    建议时，调用方仍需要把**判官说了什么**记进 ``judge_verdict``，否则接线层只能去
    匹配中文 reasons 才知道护栏拦过一次（脆弱）。第二项 False 表示"判官给了合法
    verdict 但不被允许采纳"（护栏触发），与"判官没给（None/非法/异常）"区分开。
    两种情况下 reasons 都写明了原因，确定性链路都不受影响。
    """
    try:
        raw = judge(prior, evidence)
    except Exception as exc:
        reasons.append(
            f"语义判官调用失败（{type(exc).__name__}: {exc}）→ 保持 uncertain"
            "（确定性判定不受模型侧故障影响）"
        )
        return None, False
    if raw is None:
        reasons.append("语义判官未给出判定（返回 None）→ 保持 uncertain")
        return None, False
    verdict = str(raw).strip().lower()
    if verdict not in VERDICTS:
        reasons.append(
            f"语义判官返回「{raw}」不是合法判定（合法值：{'/'.join(VERDICTS)}）→ 保持 uncertain"
        )
        return None, False
    if verdict not in allowed_verdicts:
        # 判官越权拦截：护栏触发时只允许保守判定（模块 docstring「判官越权拦截」）。
        # 不采纳、不落可疑判定，但把原始 verdict 原样返回给调用方留痕。
        reasons.append(
            f"语义判官判定 {verdict}（建议动作 {VERDICT_ACTIONS[verdict]}），"
            f"但护栏「{guard}」触发时只允许保守判定"
            f"（{'/'.join(allowed_verdicts)}）→ 判官建议被护栏拒绝，保持 uncertain"
            "（主体同一性未确认时不得放行覆盖/一致判定）"
        )
        return verdict, False
    reasons.append(
        f"语义判官判定 {verdict}（仅在确定性判据不足时调用；"
        f"建议动作 {VERDICT_ACTIONS[verdict]}）"
    )
    return verdict, True


def _match_source_ref(prior: Note, evidence: EvidenceItem) -> SourceRef | None:
    """证据的 source_url 命中旧记忆的哪条 SourceRef（精确匹配，取首条；无命中返回 None）。"""
    url = str(evidence.source_url or "")
    if not url:
        return None
    for ref in prior.meta.sources:
        if ref.url == url:
            return ref
    return None


def _entities_disjoint(
    prior: Note, evidence: EvidenceItem
) -> tuple[bool, list[str], list[str]]:
    """两侧实体是否不交（任一侧为空 = 无从比较，返回 False）；同时回两侧归一键。"""
    prior_keys = sorted(entity_keys(prior.entities))
    evidence_keys = sorted(entity_keys(evidence.entities))
    if not prior_keys or not evidence_keys:
        return False, prior_keys, evidence_keys
    return not (set(prior_keys) & set(evidence_keys)), prior_keys, evidence_keys


def _short_hash(value: str) -> str:
    """hash 前 8 位（理由引用用）；空值写成"(未记录)"，避免理由出现空白。"""
    text = str(value or "")
    return text[:8] if text else "(未记录)"


def _token_summary(scan: SlotScan) -> str:
    """facts 令牌摘要行：两侧数量 + 证据新增令牌（最多列 6 个）。"""
    labels = [token.label() for token in scan.new_tokens]
    shown = "、".join(labels[:6]) + ("…" if len(labels) > 6 else "")
    return (
        f"事实令牌：旧记忆 {len(scan.prior_tokens)} 个、证据 {len(scan.evidence_tokens)} 个"
        f"（证据新增 {len(labels)} 个{('：' + shown) if shown else ''}）"
    )


def _new_token_notes(scan: SlotScan) -> list[str]:
    """来源/时间判更新时追加一行审计：证据多出的事实令牌（供 P2-D 保留）。"""
    if not scan.new_tokens:
        return []
    labels = "、".join(token.label() for token in scan.new_tokens[:6])
    return [f"另：证据新增事实令牌 {labels}（supersede 时建议一并保留）"]


__all__ = [
    "ACTION_MERGE",
    "ACTION_NONE",
    "ACTION_OPEN_CONFLICT",
    "ACTION_REFRESH_REVIEWED_AT",
    "ACTION_SUPERSEDE",
    "CONSERVATIVE_VERDICTS",
    "DEFAULT_CONSISTENT_SIMILARITY",
    "DEFAULT_CONTEXT_CHARS",
    "DEFAULT_SIMILARITY_DIM",
    "DEFAULT_SIMILARITY_FLOOR",
    "GUARD_ENTITIES_DISJOINT",
    "VERDICTS",
    "VERDICT_ACTIONS",
    "VERDICT_CONFLICTING",
    "VERDICT_CONSISTENT",
    "VERDICT_MORE_SPECIFIC",
    "VERDICT_NEWER",
    "VERDICT_UNCERTAIN",
    "EvidenceComparison",
    "EvidenceItem",
    "FactToken",
    "SlotConflict",
    "SlotScan",
    "SourceComparison",
    "VerificationSettings",
    "compare_batch",
    "compare_prior_and_evidence",
    "detect_slot_conflicts",
    "extract_fact_tokens",
    "scan_slots",
    "token_similarity",
]
