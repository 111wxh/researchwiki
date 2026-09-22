#!/usr/bin/env python
"""闸 2：中文检索栈召回测量（关键词 fts / 向量 vector / 融合 hybrid 三通道对比）。

── 这个闸测什么 ────────────────────────────────────────────────────────
量化中文查询下项目检索栈的真实召回质量。语料是脚本内置的 24 条中文夹具笔记
（研究 wiki 口吻，覆盖记忆分层 / 上下文压缩 / 向量检索 / 向量库 / 融合 /
分词 / 编排 / 蒸馏 / 冲突 / 新鲜度等主题），查询 12 条中文（含 2 字短查询、
近义改写、跨实体组合、带"不应命中"约束的干扰查询）。每条笔记与"期望被哪些
查询命中"是人工写死的 ground truth（见 FIXTURE_NOTES / QUERIES）。

三条通道都走项目真实实现（wiki.index.SearchIndex）：
  fts    关键词通道：FTS5 短语查询（trigram 时是**字面子串**匹配）+ 短查询 LIKE 兜底；
  vector 向量通道：embedding 余弦近邻（vec0 KNN，扩展不可用时纯 Python 暴力扫描）；
  hybrid 项目默认：RRF 倒数排名融合 + 置信度/新鲜度权重（index.search() 全链路）。

── 怎么判（指标与阈值）─────────────────────────────────────────────────
对每条查询分别取三条通道的排序，算：
  Recall@3 / Recall@5 = top-k 覆盖了 ground truth 集合的比例（多目标笔记时按比例给分）；
  MRR@5               = 第一个命中名次的倒数（top-5 内无命中记 0）；
  forbidden@5         = 带"不应命中"约束的查询里，禁用笔记出现在 top-5 的次数。
可用线（--min-recall5，默认 0.80）：**hybrid 通道**的平均 Recall@5 ≥ 阈值 → 退出码 0，
否则 1；环境问题（embedding 认证被拒 / 网络不通）→ 退出码 2，本次测量不作数。
说明：0.80 是先验工程起点（"八成查询能在 top5 内找齐期望笔记"），不是从真实
工作负载回归出来的数字——语料扩到几百条后应当重新校准。

── 怎么跑 ─────────────────────────────────────────────────────────────
  # 默认：Mock 向量（确定性 bigram 特征哈希，零 key 零网络，可重复）
  uv run --no-sync python scripts/gate_retrieval.py
  # 限定 tokenizer / 落盘 JSON
  uv run --no-sync python scripts/gate_retrieval.py --tokenizer trigram --json gate2.json
  # 真实向量：走 config.toml [embedding]；该段留空时用 --embedding-model/--embedding-base-url
  uv run --no-sync python scripts/gate_retrieval.py --embedding config \
      --embedding-model embedding-3 --embedding-base-url https://open.bigmodel.cn/api/paas/v4
  # 只验证真实 embedding 通道可用性（维度 / 延迟 / 缓存命中）
  uv run --no-sync python scripts/gate_retrieval.py --check-embedding

── 结果怎么解读 ────────────────────────────────────────────────────────
逐查询分组打印：期望命中、三条通道各自的 top5 与命中位置、是否触发禁用笔记。
末尾给三通道指标对比表、失败查询清单与结论。两条已知的栈特性会影响读数：
  1) fts_tokenizer=auto 在本机解析为 trigram（vendor 的 jieba 词库路径含中文，
     探测直接放弃 simple）；trigram 的短语查询是**字面子串**匹配，所以口语化
     长查询（"……怎么办"）在 fts 通道必然全灭——这是通道特性，不是夹具问题；
  2) MockEmbeddingProvider 是确定性 bigram 特征哈希，只适合验证链路与统计，
     它的召回质量不代表真实 embedding 模型；真实质量看 --embedding config。
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

import httpx

from researchwiki.env import load_env_file
from researchwiki.wiki.embeddings import (
    CachedEmbeddingProvider,
    EmbeddingError,
    EmbeddingProvider,
    MockEmbeddingProvider,
    OpenAICompatibleEmbedding,
    get_embedding_provider,
)
from researchwiki.wiki.index import SearchIndex
from researchwiki.wiki.store import WikiStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config.toml"
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"
DEFAULT_WORKDIR = PROJECT_ROOT / "wiki-data" / "gate" / "retrieval"
CHANNELS = ("fts", "vector", "hybrid")
RECALL_KS = (3, 5)
MRR_K = 5
# 夹具笔记统一的元数据：stable + high 让置信/新鲜度权重恒为 1.0，
# 三条通道的差异就只剩"检索本身"，而不是排序权重
FIXTURE_CONFIDENCE = "high"
FIXTURE_VOLATILITY = "stable"
FIXTURE_CREATED = "2026-01-15T00:00:00+00:00"

EXIT_PASS = 0
EXIT_BELOW_LINE = 1
EXIT_ENVIRONMENT = 2

AUTH_HINTS = (
    "HTTP 401",
    "HTTP 403",
    "身份验证失败",
    "令牌已过期",
    "验证不正确",
    "authentication",
    "invalid api key",
)


# ---- 夹具：24 条中文原子笔记 -------------------------------------------------


@dataclass(frozen=True)
class FixtureNote:
    """一条夹具笔记（note_id 固定，便于 ground truth 引用）。"""

    note_id: str
    title: str
    body: str
    entities: tuple[str, ...] = ()


FIXTURE_NOTES: tuple[FixtureNote, ...] = (
    FixtureNote(
        "N-0001",
        "上下文压缩（compaction）的工程实践",
        "上下文压缩（compaction）在窗口占用超过阈值时触发：把早期轮次折叠成状态摘要，"
        "只保留最近几轮的原文。\n"
        "工程要点：摘要要写成可继续推理的状态（目标 / 已完成 / 未决），而不是复述对话；"
        "折叠后前缀会变，前缀缓存命中率下降，所以按固定比例滚动触发更划算。\n"
        "经验：阈值取 0.7 左右、状态文件独立落盘，长任务的中断率明显下降。",
        ("上下文压缩", "compaction"),
    ),
    FixtureNote(
        "N-0002",
        "上下文窗口与 token 预算",
        "上下文窗口是单次请求的输入上限，token 预算则是给一次研究 run 分配的额度。\n"
        "窗口上限决定一步能塞多少工具结果，预算决定整个 run 能跑多少步。两者都要"
        "显式记账：input tokens 累计超过预算就强制收尾。\n"
        "常见误区：把窗口上限当成预算，结果一步就吃满额度。",
        ("上下文窗口", "token 预算"),
    ),
    FixtureNote(
        "N-0003",
        "Prompt Caching 的前缀稳定与断点",
        "Prompt Caching 靠前缀复用降低单价：OpenAI 兼容端点自动按最长公共前缀命中，"
        "Anthropic 需要显式设置 cache_control 断点。\n"
        "工程约束：系统提示词与工具 schema 必须放在最前且保持字节稳定，否则前缀每次都变，"
        "缓存永远不命中。\n"
        "命中率可以从响应 usage 里的 cached_tokens 观测。",
        ("Prompt Caching", "前缀缓存"),
    ),
    FixtureNote(
        "N-0004",
        "长程记忆的分层：工作记忆与长期沉淀",
        "把记忆分成两层：工作记忆是当前任务的上下文（易失、容量小），"
        "长期沉淀是落到 wiki 的原子笔记（持久、可检索）。\n"
        "分层的价值在于把写入成本与召回成本解耦：交互期只读写工作记忆，"
        "会话结束后再异步整理进长期层。",
        ("记忆分层", "工作记忆"),
    ),
    FixtureNote(
        "N-0005",
        "记忆中间件 Mem0：add / update / merge",
        "记忆中间件是可以直接插进应用的库：Mem0 的写入流水线包含 add / update / merge "
        "三类操作，写入前先用相似度查重。\n"
        "中间件的取舍：接入成本低、默认策略够用，但记忆结构由库决定，"
        "难以表达领域特有的冲突裁决规则。",
        ("Mem0", "记忆中间件"),
    ),
    FixtureNote(
        "N-0006",
        "agent 记忆方案横向对比：Letta 与 A-MEM",
        "两种记忆方案的定位差异：Letta（原 MemGPT）把记忆当操作系统来管，"
        "后台子 agent 在空闲时整理（sleep-time compute）；A-MEM 走自动链路组织。\n"
        "选型看两点：记忆维护是否要移出交互窗口、结构是否要可审计。",
        ("Letta", "MemGPT"),
    ),
    FixtureNote(
        "N-0007",
        "记忆召回时机与相关性阈值",
        "召回不是越多越好：无关记忆会挤占窗口并带偏推理。\n"
        "常见触发条件——话题切换、用户显式引用历史、召回得分超过阈值。\n"
        "阈值要按语料调：太高漏召回，太低噪声大；可以用召回后是否被引用做反馈信号。",
        ("记忆召回", "相关性阈值"),
    ),
    FixtureNote(
        "N-0008",
        "向量检索：embedding 与余弦相似度",
        "向量检索把文本映射到稠密向量，再用余弦相似度找近邻。\n"
        "要点：查询与文档必须用同一模型、同一归一化方式；相似度阈值要显式设置，"
        "否则正交（相似度 0）的条目也会被排在前面。\n"
        "个人语料规模下纯 Python 暴力扫描就够用，不必先上专门的存储设施。",
        ("向量检索", "余弦相似度"),
    ),
    FixtureNote(
        "N-0009",
        "向量数据库选型：sqlite-vec 与 pgvector",
        "向量数据库负责向量的持久化与近邻索引。个人 wiki 规模（万级以下）用 sqlite-vec "
        "最省事：单文件、零运维、与 FTS5 同库。\n"
        "多用户或亿级规模再考虑 pgvector / Milvus。选型看四点：规模、元数据预筛能力、"
        "运维成本、与现有存储的耦合度。",
        ("向量数据库", "sqlite-vec"),
    ),
    FixtureNote(
        "N-0010",
        "混合检索与 RRF 倒数排名融合",
        "混合检索并行跑关键词与向量两路，再用 RRF 融合：score = Σ 1/(k + rank)。\n"
        "为什么用 RRF：它只依赖名次、不依赖分数尺度，两路分数量纲不同也能合。\n"
        "实践细节：k 取 60 是常见默认；融合后仍要按置信度与新鲜度做二次调节。",
        ("混合检索", "RRF"),
    ),
    FixtureNote(
        "N-0011",
        "中文分词与 FTS5 trigram",
        "中文没有空格，FTS5 的默认 unicode61 分词器几乎按单字切，召回质量差。\n"
        "三条路线：simple 扩展（带 jieba 词库）做词级切分；trigram 用 3-gram 子串匹配、"
        "无需词库；或者先用外部分词器切好再入库。\n"
        "注意：trigram 无法命中短于 3 个字符的查询，需要补一条 LIKE 兜底。",
        ("中文分词", "trigram"),
    ),
    FixtureNote(
        "N-0012",
        "BM25 与关键词检索的局限",
        "BM25 按词频与逆文档频率打分，优点是快、可解释、无需训练。\n"
        "局限：同义改写命中不了（用户说「窗口装不下」时检索不到讲压缩的笔记），"
        "长文档词频偏置明显，中文还要先解决切词。\n"
        "结论：关键词通道保底，语义通道补召回，两者互补而不是替代。",
        ("BM25", "关键词检索"),
    ),
    FixtureNote(
        "N-0013",
        "重排序 rerank：交叉编码器",
        "重排序在召回之后加一道精排：把查询与候选拼接后过交叉编码器，直接输出相关性分数。\n"
        "代价是每个候选一次前向计算，只适合对少量候选做。\n"
        "工程折中：召回 50 条、精排取前 5 条，延迟与质量都还可控。",
        ("重排序", "交叉编码器"),
    ),
    FixtureNote(
        "N-0014",
        "查询改写：同义改写、子查询与 HyDE",
        "查询改写在检索前把用户问题变成更适合检索的形式：同义改写、拆成子查询，"
        "或先生成一段假想答案再拿它检索（HyDE）。\n"
        "中文场景收益尤其明显：口语提问与笔记的书面表述往往对不上词。\n"
        "风险是改写引入漂移，最好保留原始查询一起召回。",
        ("查询改写", "HyDE"),
    ),
    FixtureNote(
        "N-0015",
        "Agent 主循环编排：plan-act-observe",
        "主循环把一次研究拆成规划、执行（工具调用）、观察、蒸馏、报告五个阶段。\n"
        "工程要点：熔断（步数上限与 token 预算）、每阶段状态落盘、工具结果统一截断。\n"
        "编排的目标是让长任务在预算内跑完，并留下可复盘的轨迹。",
        ("agent 编排", "plan-act-observe"),
    ),
    FixtureNote(
        "N-0016",
        "子 agent 与上下文隔离",
        "子 agent 用独立上下文与预算深入研究子问题，只把结论与来源带回主循环。\n"
        "隔离的价值：主循环的窗口不被原始网页占满，长任务的上下文压力可控。\n"
        "代价：子 agent 看不到主循环的既有结论，派发时要写清任务简报，避免重复劳动。",
        ("子 agent", "上下文隔离"),
    ),
    FixtureNote(
        "N-0017",
        "工具调用协议与 schema 校验",
        "OpenAI function calling 的协议面：assistant.tool_calls 里带工具名与 arguments"
        "（JSON 字符串），执行后以 role=tool + tool_call_id 回传结果。\n"
        "工程要点：arguments 必须容忍破损（解析失败折算成 error 结果回灌模型）；"
        "工具名要按注册表校验，未注册的名字不能直接执行。",
        ("function calling", "schema 校验"),
    ),
    FixtureNote(
        "N-0018",
        "蒸馏：从报告抽取原子笔记",
        "蒸馏把一次研究的报告压成若干原子笔记：一条笔记一个可独立引用的断言。\n"
        "抽取契约：标题、正文、实体、置信度、波动性、来源 URL。\n"
        "为什么要原子化：合并与冲突裁决都以笔记为最小单位，粒度太粗就无法增量更新。",
        ("蒸馏", "原子笔记"),
    ),
    FixtureNote(
        "N-0019",
        "去重与合并：相似度阈值",
        "入库前先查重：候选笔记与既有笔记向量余弦超过阈值、且实体重叠时判为重复。\n"
        "重复时合并而不是新增：保留既有编号，新证据并入来源列表。\n"
        "阈值太高会漏合并，太低会把仅仅主题相近的两条合并掉；相似度不够时应当保留两条"
        "并登记一条冲突记录，而不是硬合。",
        ("去重", "合并"),
    ),
    FixtureNote(
        "N-0020",
        "冲突台账与裁决",
        "同一实体的矛盾断言进冲突台账：登记双方证据、提出时间与状态。\n"
        "裁决要写清采纳哪一方、依据是什么，并把结论回写到笔记。\n"
        "台账的价值是可追溯：三个月后仍能看出当时的判断依据。",
        ("冲突台账", "裁决"),
    ),
    FixtureNote(
        "N-0021",
        "新鲜度衰减与半衰期",
        "知识会过期：给每条笔记标波动性，volatile 按 30 天、drifting 按 90 天做指数衰减，"
        "stable 不衰减。\n"
        "检索排序里乘上衰减因子，让旧结论自然靠后。\n"
        "注意观察时间优先于创建时间，否则长期维护的笔记会被误判为陈旧。",
        ("新鲜度", "半衰期"),
    ),
    FixtureNote(
        "N-0022",
        "token 记账与成本核算",
        "每次模型调用的 usage 都要落盘：input / output / cached tokens 与延迟。\n"
        "记账的用途：算单次研究的成本、定位哪一步在烧钱、给预算熔断提供输入。\n"
        "口径要统一：流式响应里 usage 常常在最后一个 chunk 才给。",
        ("token 记账", "成本核算"),
    ),
    FixtureNote(
        "N-0023",
        "重试与指数退避",
        "上游限流（HTTP 429）与服务端错误（5xx）要重试，4xx 不要重试。\n"
        "退避用指数加抖动：base * 2^attempt + random，避免所有客户端同时重试打爆上游。\n"
        "重试只发生在拿到首个流式事件之前，中途断流只能交给上层决定。",
        ("重试", "指数退避"),
    ),
    FixtureNote(
        "N-0024",
        "沙箱文件读写与原子落盘",
        "工具写文件必须限制在沙箱根内：路径解析后不在根内一律拒绝（防路径逃逸）。\n"
        "写入用临时文件加原子替换，进程崩溃不会留下半个文件。\n"
        "读路径同样要校验，否则 ../ 就能读到沙箱外的内容。",
        ("沙箱", "原子写"),
    ),
)

assert len(FIXTURE_NOTES) == 24, "夹具笔记数应为 24"


# ---- 查询与 ground truth（人工标注）------------------------------------------


@dataclass(frozen=True)
class Query:
    """一条评测查询。

    truth  = 期望命中的笔记 id 集合（人工判定"这条查询真正想问的笔记"）；
    forbid = 不应出现在 top-5 的笔记 id（易混词对里的另一侧，用来量误召）；
    kind   = 查询类型，便于按类型看通道强弱。
    """

    qid: str
    text: str
    truth: tuple[str, ...]
    kind: str
    forbid: tuple[str, ...] = ()


QUERIES: tuple[Query, ...] = (
    Query("Q01", "压缩", ("N-0001",), "短查询"),
    Query("Q02", "上下文压缩", ("N-0001",), "术语", ("N-0002",)),
    Query("Q03", "上下文窗口装不下时怎么处理", ("N-0001",), "近义改写"),
    Query("Q04", "向量数据库怎么选型", ("N-0009",), "术语", ("N-0008",)),
    Query("Q05", "向量检索的相似度怎么算", ("N-0008",), "术语"),
    Query("Q06", "怎么让检索结果排序更准", ("N-0010", "N-0013"), "跨实体"),
    Query("Q07", "中文分词有哪些方案", ("N-0011",), "术语"),
    Query("Q08", "记忆中间件能做什么", ("N-0005",), "术语", ("N-0006",)),
    Query("Q09", "长期记忆和短期记忆怎么分层", ("N-0004",), "近义改写"),
    Query("Q10", "什么时候该召回历史记忆", ("N-0007",), "近义改写"),
    Query("Q11", "冲突了怎么办", ("N-0020",), "干扰", ("N-0019",)),
    Query("Q12", "HyDE 查询改写能解决什么问题", ("N-0014",), "术语"),
)


# ---- 指标（纯函数，供单测）---------------------------------------------------


def recall_at_k(ranked: Sequence[str], truth: Sequence[str], k: int) -> float:
    """Recall@k：top-k 覆盖 truth 的比例；truth 为空视为 1.0（无遗漏）。"""
    if not truth:
        return 1.0
    top = set(ranked[:k])
    return len(top & set(truth)) / len(truth)


def reciprocal_rank(ranked: Sequence[str], truth: Sequence[str], k: int = MRR_K) -> float:
    """第一个命中的名次倒数（只看 top-k）；无命中 0.0。"""
    wanted = set(truth)
    for position, note_id in enumerate(ranked[:k], start=1):
        if note_id in wanted:
            return 1.0 / position
    return 0.0


def mean(values: Sequence[float]) -> float:
    """算术平均；空序列返回 0.0。"""
    return sum(values) / len(values) if values else 0.0


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
class EvalRow:
    """一条查询在三通道上的观测（全部是纯数据，便于单测聚合逻辑）。"""

    qid: str
    query: str
    kind: str
    truth: tuple[str, ...]
    forbid: tuple[str, ...]
    ranked: dict[str, list[str]]


@dataclass
class ChannelMetrics:
    """一条通道的汇总指标（分母只算有 ground truth 的查询）。"""

    channel: str
    scored_queries: int
    recall3: float
    recall5: float
    mrr: float
    forbidden_hits: int
    forbidden_checks: int
    failures: list[str] = field(default_factory=list)

    @property
    def forbidden_rate(self) -> float:
        return self.forbidden_hits / self.forbidden_checks if self.forbidden_checks else 0.0


def evaluate_row(row: EvalRow, channel: str) -> dict[str, Any]:
    """一条查询在一条通道上的明细（top5 / 命中位置 / 指标 / 误召）。"""
    ranked = row.ranked.get(channel, [])
    top = ranked[:MRR_K]
    hits = [note_id for note_id in top if note_id in set(row.truth)]
    positions = {note_id: rank for rank, note_id in enumerate(top, start=1)}
    return {
        "channel": channel,
        "top5": top,
        "hit_positions": {note_id: positions[note_id] for note_id in hits},
        "missed": [note_id for note_id in row.truth if note_id not in positions],
        "recall3": recall_at_k(ranked, row.truth, 3),
        "recall5": recall_at_k(ranked, row.truth, 5),
        "rr": reciprocal_rank(ranked, row.truth, MRR_K),
        "forbidden_hits": [note_id for note_id in top if note_id in set(row.forbid)],
    }


def channel_metrics(channel: str, rows: Sequence[EvalRow]) -> ChannelMetrics:
    """聚合一条通道：指标只在对"有 truth 的查询"求平均，误召单独统计。"""
    scored = [row for row in rows if row.truth]
    recall3 = [recall_at_k(row.ranked.get(channel, []), row.truth, 3) for row in scored]
    recall5 = [recall_at_k(row.ranked.get(channel, []), row.truth, 5) for row in scored]
    rrs = [reciprocal_rank(row.ranked.get(channel, []), row.truth, MRR_K) for row in scored]
    metrics = ChannelMetrics(
        channel=channel,
        scored_queries=len(scored),
        recall3=mean(recall3),
        recall5=mean(recall5),
        mrr=mean(rrs),
        forbidden_hits=0,
        forbidden_checks=0,
    )
    for row in rows:
        if not row.forbid:
            continue
        metrics.forbidden_checks += 1
        detail = evaluate_row(row, channel)
        if detail["forbidden_hits"]:
            metrics.forbidden_hits += 1
            metrics.failures.append(
                f"{row.qid} 「{row.query}」误召 {', '.join(detail['forbidden_hits'])}"
                f"（top5: {' '.join(detail['top5']) or '空'}）"
            )
    for row in scored:
        detail = evaluate_row(row, channel)
        if detail["missed"]:
            metrics.failures.append(
                f"{row.qid} 「{row.query}」漏 {', '.join(detail['missed'])}"
                f"（top5: {' '.join(detail['top5']) or '空'}）"
            )
    return metrics


# ---- 检索通道 ---------------------------------------------------------------


# 刻意复用 SearchIndex 的内部通道方法：测的必须是项目真实实现的排序
# （tokenizer 选择、短语构造、LIKE 兜底、KNN/暴力扫描、重定向跟随都在里面）。
# 它们改名即显式报错，绝不静默降级成"脚本自己重写一套"。
_INTERNAL_NAMES = ("_load_meta_map", "_resolve_redirected", "_fts_candidates", "_vector_candidates")


def channel_ranking(index: SearchIndex, query: str, channel: str, *, pool: int) -> list[str]:
    """单通道排序（note_id 列表，按相关度降序）。channel ∈ {fts, vector}。"""
    missing = [name for name in _INTERNAL_NAMES if not hasattr(index, name)]
    if missing:
        raise RuntimeError(
            "SearchIndex 内部接口已变化，闸 2 需要同步更新：缺少 "
            + ", ".join(missing)
            + "。本脚本刻意复用项目真实通道实现，避免自己重写一份 tokenizer / KNN。"
        )
    meta_map = index._load_meta_map()
    raw = (
        index._fts_candidates(query, pool)
        if channel == "fts"
        else index._vector_candidates(query, pool)
    )
    return list(index._resolve_redirected(raw, meta_map))


def collect_rows(index: SearchIndex, queries: Sequence[Query], *, pool: int) -> list[EvalRow]:
    """跑完所有查询，收集三通道排序。hybrid 走公开的 index.search() 全链路。"""
    rows: list[EvalRow] = []
    for query in queries:
        ranked = {
            "fts": channel_ranking(index, query.text, "fts", pool=pool),
            "vector": channel_ranking(index, query.text, "vector", pool=pool),
            "hybrid": [m.note_id for m in index.search(query.text, k=MRR_K)],
        }
        rows.append(
            EvalRow(
                qid=query.qid,
                query=query.text,
                kind=query.kind,
                truth=query.truth,
                forbid=query.forbid,
                ranked=ranked,
            )
        )
    return rows


# ---- 夹具与索引 -------------------------------------------------------------


def seed_fixture(workdir: Path) -> WikiStore:
    """把内置夹具写进 workdir/notes（固定 id，可重复覆盖），返回 WikiStore。"""
    store = WikiStore(workdir)
    for note in FIXTURE_NOTES:
        store.save_note(
            note.body,
            note_id=note.note_id,
            title=note.title,
            entities=list(note.entities),
            confidence=FIXTURE_CONFIDENCE,
            volatility=FIXTURE_VOLATILITY,
            observed_at=FIXTURE_CREATED,
            created=FIXTURE_CREATED,
        )
    return store


def fingerprint(secret: str) -> str:
    """key 指纹：前 6 位 + 长度；空值明确标注（绝不打印完整 key）。"""
    if not secret:
        return "未设置"
    return f"{secret[:6]}…(len={len(secret)})"


def embedding_settings(
    config: Mapping[str, Any], *, base_url: str, model: str
) -> tuple[str, str, str]:
    """返回 (base_url, model, api_key_env)；CLI 覆盖优先于 config.toml [embedding]。"""
    section: object = (config or {}).get("embedding") or {}
    cfg: Mapping[str, Any] = section if isinstance(section, Mapping) else {}
    resolved_url = base_url.strip() or str(cfg.get("base_url") or "").strip()
    resolved_model = model.strip() or str(cfg.get("model") or "").strip()
    return resolved_url, resolved_model, str(cfg.get("api_key_env") or "")


def resolve_embedding(
    config: Mapping[str, Any],
    args: argparse.Namespace,
    *,
    cache_path: Path,
) -> tuple[EmbeddingProvider, dict[str, Any]]:
    """按 --embedding 组装向量 provider，并给出可打印的描述。"""
    base_url, model, key_env = embedding_settings(
        config, base_url=args.embedding_base_url, model=args.embedding_model
    )
    if args.embedding == "mock" or not base_url or not model:
        provider = MockEmbeddingProvider()
        if args.embedding == "mock":
            why = "显式 --embedding mock"
        else:
            why = "[embedding] 段 base_url/model 为空"
        return provider, {
            "kind": "mock",
            "model": provider.model,
            "dim": getattr(provider, "dim", None),
            "base_url": "",
            "api_key": "未使用",
            "note": f"实际使用 MockEmbeddingProvider（{why}）：确定性 bigram 特征哈希，"
            "零网络；召回质量不代表真实 embedding 模型。",
        }
    secret = os.environ.get(key_env, "") if key_env else ""
    inner = OpenAICompatibleEmbedding(
        base_url=base_url, model=model, api_key_env=key_env or None
    )
    provider = CachedEmbeddingProvider(inner, cache_path)
    return provider, {
        "kind": "openai-compatible",
        "model": model,
        "dim": None,  # 首次嵌入后才知道
        "base_url": base_url,
        "api_key": fingerprint(secret),
        "api_key_env": key_env,
        "cache": str(cache_path),
    }


class CountingTransport(httpx.BaseTransport):
    """统计真实 transport 请求次数：用来证明第二次嵌入命中缓存、未触发网络。"""

    def __init__(self, delegate: httpx.BaseTransport | None = None) -> None:
        self.delegate = delegate or httpx.HTTPTransport()
        self.calls = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        return self.delegate.handle_request(request)


def auth_guidance(base_url: str, model: str, key_env: str, secret: str) -> str:
    """embedding 认证失败时的可执行提示（含 key 指纹，绝不含 key 本体）。"""
    env_name = key_env or "RESEARCHWIKI_EMBEDDING_API_KEY"
    return "\n".join(
        [
            "✗ embedding 认证失败：key 被服务端拒绝（HTTP 401/403）。本次测量不作数，退出码 2。",
            f"  · 当前 base_url = {base_url or '(空)'}，model = {model or '(空)'}",
            f"  · 当前 key 指纹 = {fingerprint(secret)}（来源环境变量 {env_name}）",
            f"  · 检查 1：.env 里 {env_name} 是否有效 / 未过期 / 无多余空白或引号；",
            f"  · 检查 2：.env 与真实环境变量同时存在时，以 {env_name} 的真实环境变量值为准；",
            "  · 检查 3：若用的是中转/代理服务，把 base_url 指向该服务（不含 /embeddings）；",
            "  · 检查 4：确认 model 名在该服务上真实存在（如智谱 embedding-3）；",
            "  · 检查 5：只想先验证链路可改用 --embedding mock（免 key 零网络）。",
        ]
    )


def classify_embedding_error(exc: BaseException) -> str:
    """embedding 异常 → auth / network / protocol。"""
    text = str(exc)
    lowered = text.lower()
    if any(hint in text for hint in AUTH_HINTS) or "401" in text[:40] or "403" in text[:40]:
        return "auth"
    if isinstance(exc, EmbeddingError):
        return "protocol"
    if "connect" in lowered or "timeout" in lowered or "transport" in lowered:
        return "network"
    return "protocol"


def check_embedding(config: Mapping[str, Any], args: argparse.Namespace, workdir: Path) -> int:
    """真实 embedding 通道可用性探针：维度 / 延迟 / 第二次是否命中缓存。

    刻意自己拼装 provider（与 get_embedding_provider 同样的两层结构）以便注入
    计数 transport——不这么做就数不出"第二次调用有没有走网络"。
    """
    print("── 真实 embedding 通道探针（--check-embedding）──")
    base_url, model, key_env = embedding_settings(
        config, base_url=args.embedding_base_url, model=args.embedding_model
    )
    secret = os.environ.get(key_env, "") if key_env else ""
    if not base_url or not model:
        provider = get_embedding_provider(config, cache_path=None)
        probe = provider.embed(["中文向量通道探针文本"])
        print(f"provider : {type(provider).__name__}（[embedding] 段留空 → Mock）")
        print(f"dim      : {len(probe[0])}（Mock 的 bigram 哈希维度，不代表真实模型）")
        print("结论：本次没有真实 embedding 可验证——请在 config.toml [embedding] 填好")
        print("      model / base_url（或用 --embedding-model / --embedding-base-url），再重跑。")
        return EXIT_PASS

    cache_path = workdir / "probe-cache.db"
    cache_path.unlink(missing_ok=True)  # 独立冷启动缓存，保证"第二次才该命中"
    transport = CountingTransport()
    inner = OpenAICompatibleEmbedding(
        base_url=base_url, model=model, api_key_env=key_env or None, transport=transport
    )
    provider = CachedEmbeddingProvider(inner, cache_path)
    print("provider : OpenAICompatibleEmbedding + CachedEmbeddingProvider")
    print(f"model    : {model}")
    print(f"base_url : {base_url}")
    print(f"api_key  : {fingerprint(secret)}（来源 {key_env or 'RESEARCHWIKI_EMBEDDING_API_KEY'}）")
    print(f"cache    : {cache_path}（每次探针清空，保证第一次是冷启动）")
    probe_text = "中文向量通道探针：检索栈可用性检查"
    try:
        started = time.perf_counter()
        first = provider.embed([probe_text])[0]
        first_ms = (time.perf_counter() - started) * 1000
        first_calls = transport.calls
        started = time.perf_counter()
        second = provider.embed([probe_text])[0]
        second_ms = (time.perf_counter() - started) * 1000
        second_calls = transport.calls - first_calls
    except Exception as exc:  # noqa: BLE001 -- 探针失败要给人话，不是 traceback
        kind = classify_embedding_error(exc)
        print()
        if kind == "auth":
            print(auth_guidance(base_url, model, key_env, secret))
        else:
            print(f"✗ embedding 调用失败（{kind}）：{type(exc).__name__}: {str(exc)[:200]}")
            print(f"  · 检查 base_url={base_url} 是否可达、model={model} 是否存在。")
        return EXIT_ENVIRONMENT
    same = first == second
    # 缓存按 float32 落库，二次读回的是量化值：不能要求逐位相等，比最大偏差
    max_diff = max((abs(a - b) for a, b in zip(first, second, strict=True)), default=0.0)
    print(f"第一次调用：dim={len(first)}  延迟 {first_ms:.0f}ms  transport 请求 {first_calls} 次")
    print(
        f"第二次调用：dim={len(second)}  延迟 {second_ms:.0f}ms  "
        f"transport 请求 {second_calls} 次（0 = 命中缓存，未触发网络）"
    )
    print(f"向量一致性：逐位相等={same}  最大偏差={max_diff:.2e}（缓存按 float32 落库）")
    ok = first_calls == 1 and second_calls == 0 and len(first) > 0 and max_diff < 1e-5
    print(
        "结论：真实 embedding 通道可用，缓存（model + sha256(text)）按预期命中。"
        if ok
        else "结论：探针未完全符合预期（首次应恰好 1 次 transport、二次应 0 次且向量一致），"
        "请核对上面的计数。"
    )
    return EXIT_PASS if ok else EXIT_BELOW_LINE


# ---- 打印 -------------------------------------------------------------------


def print_header(
    *,
    embedding_info: Mapping[str, Any],
    tokenizer: str,
    configured_tokenizer: str,
    index: SearchIndex,
    note_count: int,
) -> None:
    print("=== 闸 2：中文检索栈召回 ===")
    print(f"embedding  : {embedding_info['kind']} / {embedding_info['model']}")
    if embedding_info.get("dim"):
        print(f"dim        : {embedding_info['dim']}")
    if embedding_info.get("base_url"):
        print(f"             {embedding_info['base_url']}  api_key={embedding_info['api_key']}")
    print(f"tokenizer  : {tokenizer}（配置 {configured_tokenizer}）")
    print(f"索引       : {index.db_path}（夹具笔记 {note_count} 条，每次运行全量重建）")
    print(f"查询       : {len(QUERIES)} 条，指标窗口 top-3 / top-5 / MRR@5")
    if embedding_info.get("note"):
        print(f"注意       : {embedding_info['note']}")
    print("-" * 96)


def print_rows(rows: Sequence[EvalRow]) -> None:
    for row in rows:
        forbid = f"  禁用={','.join(row.forbid)}" if row.forbid else ""
        print(f"{row.qid} [{row.kind}] 「{row.query}」  期望={','.join(row.truth)}{forbid}")
        for channel in CHANNELS:
            detail = evaluate_row(row, channel)
            marks: list[str] = []
            for note_id in row.truth:
                position = detail["hit_positions"].get(note_id)
                marks.append(f"{note_id}@{position}" if position else f"{note_id}✗")
            bad = detail["forbidden_hits"]
            top5 = " ".join(detail["top5"]) or "(空)"
            print(
                f"   {channel:<6} top5={top5:<28} 命中={' '.join(marks):<20}"
                f" R@3={detail['recall3']:.2f} R@5={detail['recall5']:.2f}"
                + (f"  误召={','.join(bad)}" if bad else "")
            )
    print("-" * 96)


def print_channel_table(metrics: Sequence[ChannelMetrics], min_recall5: float) -> None:
    print("── 三通道指标对比（对 12 条查询里带 ground truth 的求平均）──")
    print(f"  {'通道':<8}{'R@3':>7}{'R@5':>7}{'MRR@5':>8}{'误召':>8}   失败查询")
    for item in metrics:
        forbidden = (
            f"{item.forbidden_hits}/{item.forbidden_checks}" if item.forbidden_checks else "-"
        )
        flag = ""
        if item.channel == "hybrid":
            flag = "  达标" if item.recall5 >= min_recall5 else "  未达标"
        print(
            f"  {item.channel:<8}{item.recall3:>7.3f}{item.recall5:>7.3f}{item.mrr:>8.3f}"
            f"{forbidden:>8}   {len(item.failures)}{flag}"
        )
    print(f"  可用线：hybrid 的平均 Recall@5 ≥ {min_recall5:.2f}")


def print_failures(metrics: Sequence[ChannelMetrics]) -> None:
    any_failure = any(item.failures for item in metrics)
    print("── 失败查询清单（漏召回 / 误召）──")
    if not any_failure:
        print("  无")
        return
    for item in metrics:
        for line in item.failures:
            print(f"  [{item.channel}] {line}")


def build_conclusion(metrics: Sequence[ChannelMetrics], min_recall5: float) -> str:
    by_channel = {item.channel: item for item in metrics}
    hybrid = by_channel.get("hybrid")
    if hybrid is None:
        return "未采集到 hybrid 指标。"
    best_single = max(
        (item for item in metrics if item.channel != "hybrid"),
        key=lambda item: item.recall5,
        default=None,
    )
    gain = ""
    if best_single is not None:
        diff = hybrid.recall5 - best_single.recall5
        gain = (
            f"相对最强的单通道（{best_single.channel} {best_single.recall5:.3f}）"
            f"{'高' if diff >= 0 else '低'} {abs(diff):.3f}"
        )
    verdict = "达到" if hybrid.recall5 >= min_recall5 else "未达到"
    return (
        f"项目默认 hybrid 通道 Recall@5 = {hybrid.recall5:.3f}、MRR@5 = {hybrid.mrr:.3f}，"
        f"{gain}，{verdict}可用线（{min_recall5:.2f}）。"
    )


# ---- 主流程 -----------------------------------------------------------------


def load_config(path: Path) -> dict[str, Any]:
    """读 config.toml；缺失/损坏按空配置处理（等价全 mock）。"""
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except (FileNotFoundError, tomllib.TOMLDecodeError):
        return {}


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n"
    path.write_text(text, encoding="utf-8")


def report_embedding_failure(exc: BaseException, config: Mapping[str, Any],
                             args: argparse.Namespace) -> int:
    """把 embedding 异常翻译成人话（认证失败给可执行清单），返回环境错误退出码。"""
    kind = classify_embedding_error(exc)
    base_url, model, key_env = embedding_settings(
        config, base_url=args.embedding_base_url, model=args.embedding_model
    )
    print()
    if kind == "auth":
        secret = os.environ.get(key_env, "") if key_env else ""
        print(auth_guidance(base_url, model, key_env, secret))
    elif kind == "network":
        print(f"✗ embedding 网络不可达（{type(exc).__name__}）：{str(exc)[:200]}")
        print(f"  · 检查 base_url={base_url} 是否可达、是否需要代理；")
        print("  · 只想先跑通链路可加 --embedding mock（免 key 零网络）。")
    else:
        print(f"✗ embedding 调用失败：{type(exc).__name__}: {str(exc)[:200]}")
        print(f"  · base_url={base_url}，model={model}；请核对响应格式是否符合 OpenAI 兼容。")
    return EXIT_ENVIRONMENT


def run(args: argparse.Namespace) -> int:
    if args.env_file:
        load_env_file(args.env_file)
    config = load_config(Path(args.config)) if args.config else {}
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    if args.check_embedding or args.only_check_embedding:
        code = check_embedding(config, args, workdir)
        if code != EXIT_PASS or args.only_check_embedding:
            return code

    provider, embedding_info = resolve_embedding(
        config, args, cache_path=workdir / "embedding-cache.db"
    )
    if embedding_info["kind"] != "mock" and not embedding_info.get("dim"):
        # 先探一次维度（用第一条夹具笔记的同一段文本，顺手把它的缓存热上）
        head = FIXTURE_NOTES[0]
        try:
            embedding_info["dim"] = len(provider.embed([f"{head.title}\n{head.body}"])[0])
        except Exception as exc:  # noqa: BLE001 -- 无 key / key 被拒要给人话，不是 traceback
            return report_embedding_failure(exc, config, args)
    store = seed_fixture(workdir)
    resolved_tokenizer = args.tokenizer
    try:
        with SearchIndex(workdir, embedding=provider, tokenizer=args.tokenizer) as index:
            indexed = index.rebuild(store)
            resolved_tokenizer = index.tokenizer
            print_header(
                embedding_info=embedding_info,
                tokenizer=resolved_tokenizer,
                configured_tokenizer=args.tokenizer,
                index=index,
                note_count=indexed,
            )
            rows = collect_rows(index, QUERIES, pool=max(MRR_K * 4, 16))
    except Exception as exc:  # noqa: BLE001 -- 建索引/检索期的上游故障也要给人话
        return report_embedding_failure(exc, config, args)

    metrics = [channel_metrics(channel, rows) for channel in CHANNELS]
    print_rows(rows)
    print_channel_table(metrics, float(args.min_recall5))
    print_failures(metrics)
    conclusion = build_conclusion(metrics, float(args.min_recall5))
    hybrid = next(item for item in metrics if item.channel == "hybrid")
    print("结论：" + conclusion)
    if hybrid.recall5 < float(args.min_recall5):
        print("建议：先看失败清单属于哪一类——")
        print("  · 术语型查询漏召 → 通道问题（换 tokenizer / 上真实 embedding）；")
        print("  · 口语化长查询在 fts 全灭 → trigram 是字面子串匹配，属已知通道特性；")
        print("  · 误召（禁用笔记进 top5）→ 需要精排或提高向量相似度阈值。")

    if args.json:
        write_json(
            Path(args.json),
            {
                "gate": "retrieval",
                "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "embedding": embedding_info,
                "tokenizer": resolved_tokenizer,
                "configured_tokenizer": args.tokenizer,
                "notes": len(FIXTURE_NOTES),
                "queries": [
                    {"qid": row.qid, "query": row.query, "kind": row.kind,
                     "truth": list(row.truth), "forbid": list(row.forbid)}
                    for row in rows
                ],
                "per_query": [
                    {
                        "qid": row.qid,
                        "query": row.query,
                        "kind": row.kind,
                        "channels": {channel: evaluate_row(row, channel) for channel in CHANNELS},
                    }
                    for row in rows
                ],
                "metrics": {
                    item.channel: {
                        "scored_queries": item.scored_queries,
                        "recall3": round(item.recall3, 4),
                        "recall5": round(item.recall5, 4),
                        "mrr": round(item.mrr, 4),
                        "forbidden_hits": item.forbidden_hits,
                        "forbidden_checks": item.forbidden_checks,
                        "failures": item.failures,
                    }
                    for item in metrics
                },
                "min_recall5": float(args.min_recall5),
                "passed": hybrid.recall5 >= float(args.min_recall5),
                "conclusion": conclusion,
            },
        )
        print(f"JSON 已写入 {args.json}")

    return EXIT_PASS if hybrid.recall5 >= float(args.min_recall5) else EXIT_BELOW_LINE


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="闸 2：中文检索栈召回测量（三通道对比；判定标准见文件头 docstring）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--embedding", choices=("mock", "config"), default="mock",
                        help="mock=确定性 Mock 向量（默认，零 key）；config=走 [embedding] 段")
    parser.add_argument("--embedding-model", default="",
                        help="覆盖 [embedding].model（config 模式常用）")
    parser.add_argument("--embedding-base-url", default="",
                        help="覆盖 [embedding].base_url（不含 /embeddings）")
    parser.add_argument("--tokenizer", choices=("auto", "trigram", "simple"), default="auto",
                        help="FTS tokenizer（默认 auto；simple 需 vendor 的 simple.dll 与词库）")
    parser.add_argument("--min-recall5", type=float, default=0.80,
                        help="hybrid 平均 Recall@5 可用线（默认 0.80）")
    parser.add_argument("--check-embedding", action="store_true",
                        help="跑真实 embedding 探针（维度 / 延迟 / 缓存命中）")
    parser.add_argument("--only-check-embedding", action="store_true",
                        help="只跑探针，不做检索对比")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="config.toml 路径")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE),
                        help=".env 路径（真实环境变量优先）")
    parser.add_argument("--workdir", default=str(DEFAULT_WORKDIR),
                        help="夹具与索引目录（默认 wiki-data/gate/retrieval，不碰真实 wiki）")
    parser.add_argument("--json", default="", help="把结果另存为 JSON（不含 key，只有指纹）")
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
