"""Prior Reader：把历史 Wiki 的 active notes 以"仅供核验的 Prior"形式检索出来。

职责（PLAN §4.2/§4.3 的 wiki 层部分，供后续任务接入 AgentLoop）：

- ``retrieve_priors``：对问题检索 active notes。命中 merged/superseded 时返回
  最终 active note，并保留重定向链信息（哪些旧 ID 重定向到了它）；同一最终
  note 被多个旧 ID 命中时合并 ``redirected_from`` 列表。**墓碑（``meta.tombstone``）
  默认不进 Prior**——它是审计记录不是当前知识，被当作 active 知识注入 plan 在
  RQ2 语义下是错的（P2-F 裁定一）；要审计时才显式传给 ``SearchIndex.search``。
- ``format_prior_context``：格式化成人类可读的 markdown 块。**必须包含标签行**
  ``历史 Prior，仅供核验，不是本轮 fresh evidence``（原文一字不差，后续任务
  的测试会断言它），提醒下游 Prior 不能直接当答案、其 URL 不得混入本轮
  SourcePool。
- ``ensure_index_fresh``：run 开始前检查索引是否落后于 store（笔记集合、status
  或**索引指纹**任一不一致即视为落后），落后则整体 ``index.rebuild(store)``。
  判定本身委托给 ``index.index_drift``——**与 MCP 路径共用同一份判据**（P2-F
  裁定二），本函数只负责"落后了怎么办"。MVP 只做"检测落后 → 整体 rebuild"，
  等收益实验跑通后再优化增量同步。
- ``retrieve_unanswered_guards`` / ``format_unanswered_guards``（P4a Task 15）：
  从活跃的未答问题台账（wiki/ledger.py，kind=experience 反幻觉记忆）里挑出与
  当前问题相似度达标的条目，格式化成**独立护栏块**——调用方（AgentLoop）把它
  插在 Prior 块**之前**，不占用 ``[prior].top_k`` 配额：同题再到来时先看到
  "此问题此前已判定证据不足，禁止编造"。
- ``retrieve_conflict_guards`` / ``format_conflict_guards``（P4a Task 18）：从
  open 冲突台账（conflicts/，T17 维护执行器照单开出）里挑出与当前问题相关的
  条目，格式化成**独立冲突块**，插在护栏块之后、Prior 块之前（R3 顺序裁定：
  护栏 → 冲突 → Prior 笔记）。匹配是"实体相交 **或** 笔记相似度达标"两条满足
  其一；注入内容只读台账记录字段（claim 槽位/笔记实体），**不重新比较**（T16
  裁定）。``conflict_entity_names`` / ``conflict_assertion_pair`` 等展示字段
  助手同时供 CLI（``arbitrate show`` 与仲裁留痕笔记）复用——单一实现防漂移。

注入内容每条带：note_id、title、置信度、volatility、observed_at（缺省用
created）、来源 URL 列表、正文（可截断）、以及若有重定向链则列出
``redirected_from`` 旧 ID。检索排序直接沿用 ``SearchIndex.search`` 的
置信度/新鲜度调权，本模块不改打分逻辑。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from researchwiki.wiki.entities import slugify
from researchwiki.wiki.index import SearchIndex, SearchMatch, index_drift
from researchwiki.wiki.ledger import DEFAULT_GUARD_SIMILARITY, DEFAULT_GUARD_TOP_K
from researchwiki.wiki.store import Conflict, Note, WikiStore
from researchwiki.wiki.verification import (
    DEFAULT_SIMILARITY_DIM,
    DEFAULT_SIMILARITY_FLOOR,
    VerificationSettings,
    token_similarity,
)

# 标签行原文（PLAN §4.2 一字不差；后续任务在测试里断言它）
PRIOR_CONTEXT_LABEL = "历史 Prior，仅供核验，不是本轮 fresh evidence"

DEFAULT_K = 5
DEFAULT_MAX_CHARS = 4000
# 单条 Prior 正文的默认截断长度（retrieve 时先粗截，预算不够时再细截）
_MAX_BODY_CHARS = 600
# 预算再紧，单条正文也至少保留的长度（低于此不如整条丢弃）
_MIN_BODY_CHARS = 80
# 重定向链防御性跟随的最大步数（自行走链同样防环，双保险）
_MAX_CHAIN_STEPS = 10


# ---- 数据结构 ---------------------------------------------------------------


@dataclass
class PriorHit:
    """一条可注入的 Prior：最终 active note 的引用 + 重定向链 + 检索分数。

    - ``body``：注入用正文（retrieve 时按预算截断，可能带省略号）。
    - ``observed_at``：断言观察时间，meta.observed_at 缺省时回退 created。
    - ``redirected_from``：重定向到本条的旧 ID 列表（按链条顺序）；直接命中
      active note 时为空列表。
    - ``match_type`` / ``score``：透传自 SearchMatch，供调试与指标记录。
    """

    note_id: str
    title: str
    body: str
    score: float
    confidence: str
    volatility: str
    observed_at: str
    source_urls: list[str] = field(default_factory=list)
    redirected_from: list[str] = field(default_factory=list)
    match_type: str = ""


@dataclass
class PriorContext:
    """一次 Prior 检索的结果：保留下来的 hits + 格式化后总字符数 + 格式化方法。

    ``context_chars`` 恒等于 ``len(self.format())``；空结果（空 Wiki 或预算
    内一条也放不下）时 hits 为空、``format()`` 返回空字符串。
    """

    hits: list[PriorHit]
    context_chars: int

    def format(self) -> str:
        """格式化为 markdown 块（含标签行）；空结果返回空字符串。"""
        return format_prior_context(self.hits)


# ---- 配置 -------------------------------------------------------------------


@dataclass
class PriorSettings:
    """``[prior]`` 段的解析结果：注入开关与检索预算（缺省 = 开启 + 默认值）。"""

    enabled: bool = True
    k: int = DEFAULT_K
    max_chars: int = DEFAULT_MAX_CHARS


def prior_settings(config: Mapping[str, Any] | None) -> PriorSettings:
    """解析 ``[prior]`` 段（enabled / top_k / max_chars）。

    入参就是 [prior] 段本身（AgentLoop.prior_config，server 传
    ``config.get("prior")``）；缺省或 None = enabled=true + 默认值（模块内
    默认，不强制用户配置）；非法值回退默认（与 wiki_settings / dedup_settings
    同风格）。
    """
    settings = PriorSettings()
    section: Mapping[str, Any] = config if isinstance(config, Mapping) else {}
    enabled = section.get("enabled")
    if enabled is not None:
        settings.enabled = bool(enabled)
    top_k = section.get("top_k")
    if top_k is not None:
        try:
            settings.k = max(1, int(top_k))
        except (TypeError, ValueError):
            settings.k = DEFAULT_K
    max_chars = section.get("max_chars")
    if max_chars is not None:
        try:
            settings.max_chars = max(0, int(max_chars))
        except (TypeError, ValueError):
            settings.max_chars = DEFAULT_MAX_CHARS
    return settings


# ---- 检索 -------------------------------------------------------------------


def retrieve_priors(
    question: str,
    store: WikiStore,
    index: SearchIndex,
    *,
    k: int = DEFAULT_K,
    max_chars: int = DEFAULT_MAX_CHARS,
    include_tombstones: bool = False,
) -> PriorContext:
    """对问题检索 Prior：检索 → 跟随重定向 → 按 k 截取 → 预算内格式化。

    - 检索排序沿用 ``index.search``（置信度/新鲜度调权），本层不重排；
      命中 merged/superseded 时取最终 active note（``SearchMatch.redirected_from``
      指向被命中的旧 ID；若拿到的 note 仍是 merged/superseded，则防御性地
      自行走链到最终 active——有界防环、不抛异常，走不到就丢弃该命中）。
    - ``include_tombstones`` 缺省 False：墓碑（失效裁定记录）不进 Prior 注入。
      过滤在检索层（``SearchIndex.search``）完成，含"命中旧 ID → 重定向落到
      墓碑"的情况；这里透传参数只是把"注入什么"的口径显式写在本层签名上，供
      审计场景显式打开。
    - 总上下文不超过 ``max_chars``：按分数从高到低装入，放不下的整条丢弃，
      末位命中可压缩正文到最低保留长度；被保留的 hits 才进入 ``PriorContext``。
    - 空 Wiki / 无命中 / 预算内一条也放不下 → 空 PriorContext，不抛异常。
    """
    matches = index.search(question, k=k, include_tombstones=include_tombstones)
    hits = _collect_hits(matches, store)
    kept = _fit_within_budget(hits, max_chars)
    context_chars = len(format_prior_context(kept))
    return PriorContext(hits=kept, context_chars=context_chars)


def _collect_hits(matches: Sequence[SearchMatch], store: WikiStore) -> list[PriorHit]:
    """SearchMatch 列表 → PriorHit 列表：跟随重定向、按最终 note 去重合并别名。

    search 已按最终 note 去重，这里再按最终 note_id 去重一次属防御：同一最终
    note 出现多条时合并 redirected_from，内容取首条（分数更高）。
    """
    hits: list[PriorHit] = []
    by_id: dict[str, PriorHit] = {}
    for m in matches:
        note = store.get_note(m.note_id)
        if note is None:
            continue
        aliases: list[str] = []
        if m.redirected_from:
            aliases.extend(_redirect_chain(store, m.redirected_from, note.id))
        if note.status != "active":
            # 防御：索引快照过期等导致命中结果仍指向非 active 笔记时，自行走链
            # 到最终 active。不调 store.follow_redirect——它在链成环时会 raise
            # ValueError，这里保持"检索不抛异常"口径（走不到就丢弃该命中）。
            final = _follow_to_active(store, note.id)
            if final is None:
                continue
            if final.id != note.id:
                aliases.extend(_redirect_chain(store, note.id, final.id))
            note = final
        existing = by_id.get(note.id)
        if existing is not None:
            for alias in aliases:
                if alias != existing.note_id and alias not in existing.redirected_from:
                    existing.redirected_from.append(alias)
            continue
        body = note.body.strip()
        if len(body) > _MAX_BODY_CHARS:
            body = body[:_MAX_BODY_CHARS] + "…"
        hit = PriorHit(
            note_id=note.id,
            title=note.title,
            body=body,
            score=m.score,
            confidence=note.meta.confidence,
            volatility=note.meta.volatility,
            observed_at=note.meta.observed_at or note.meta.created,
            source_urls=[s.url for s in note.meta.sources],
            redirected_from=aliases,
            match_type=m.match_type,
        )
        hits.append(hit)
        by_id[note.id] = hit
    return hits


def _redirect_chain(store: WikiStore, alias_id: str, final_id: str) -> list[str]:
    """alias → final 之间的旧 ID 列表（含 alias 自身，不含 final），防环有界。"""
    chain: list[str] = []
    visited: set[str] = set()
    current = alias_id
    for _ in range(_MAX_CHAIN_STEPS):
        if current == final_id or current in visited:
            break
        visited.add(current)
        chain.append(current)
        note = store.get_note(current)
        if note is None:
            break
        target = note.meta.redirect_to if note.status == "merged" else note.meta.superseded_by
        if not target:
            break
        current = target
    return chain


def _follow_to_active(store: WikiStore, note_id: str) -> Note | None:
    """沿 redirect 链走到最终 active 笔记（有界防环）；走不到返回 None，不抛异常。

    与 ``_redirect_chain`` 同一套防御口径：visited 去重 + 步数上限；断裂、成环、
    落点非 active 一律返回 None（调用方丢弃该命中）。与 ``store.follow_redirect``
    的差别只在失败语义：这里返回 None 而不是 raise。
    """
    visited: set[str] = set()
    current = note_id
    for _ in range(_MAX_CHAIN_STEPS):
        if current in visited:
            return None  # 成环
        visited.add(current)
        note = store.get_note(current)
        if note is None:
            return None  # 断裂
        if note.status == "active":
            return note
        target = note.meta.redirect_to if note.status == "merged" else note.meta.superseded_by
        if not target:
            return None  # 状态非 active 又没有跳转目标
        current = target
    return None  # 超过步数上限：按走不到处理


def _fit_within_budget(hits: list[PriorHit], max_chars: int) -> list[PriorHit]:
    """按分数从高到低把 hits 装进 max_chars 预算（header 按最大条数预留）。

    放不下的整条丢弃；末位命中先尝试压缩正文（保底 _MIN_BODY_CHARS，压不动
    才放弃）。会就地改写被压缩命中 ``body``，保证 ``format_prior_context``
    重放结果与预算一致。
    """
    if not hits:
        return []
    # header 长度随条数非降，用初始条数预留最保守
    budget = max_chars - len(_header(len(hits)))
    if budget <= 0:
        return []
    kept: list[PriorHit] = []
    for hit in hits:  # hits 已按分数从高到低（search 排序透传）
        block = _format_block(len(kept) + 1, hit)
        if len(block) <= budget:
            kept.append(hit)
            budget -= len(block)
            continue
        empty_block = len(_format_block(len(kept) + 1, replace(hit, body="")))
        room = budget - empty_block
        fitted = False
        for cap in (room - 8, room // 2):
            if cap < _MIN_BODY_CHARS or cap >= len(hit.body):
                continue  # cap 不够保底，或压不动（原块已超预算）
            hit.body = hit.body[:cap] + "…"
            block = _format_block(len(kept) + 1, hit)
            if len(block) <= budget:
                kept.append(hit)
                budget -= len(block)
                fitted = True
                break
        if not fitted:
            break  # 预算耗尽：其余分数更低，整条丢弃
    return kept


# ---- 格式化 -----------------------------------------------------------------


def format_prior_context(hits: Sequence[PriorHit]) -> str:
    """把 PriorHit 列表格式化成人类可读的 markdown 块。

    首行为标签行 ``历史 Prior，仅供核验，不是本轮 fresh evidence``（原文一字
    不差）；空列表返回空字符串（空 Wiki 时上下文为空，不注入标签行）。
    """
    if not hits:
        return ""
    return _header(len(hits)) + "".join(
        _format_block(pos, hit) for pos, hit in enumerate(hits, start=1)
    )


def _header(count: int) -> str:
    return (
        f"## {PRIOR_CONTEXT_LABEL}\n\n"
        f"以下 {count} 条来自历史 Wiki，按检索分数从高到低排列，供本轮核验比对：\n\n"
    )


def _format_block(pos: int, hit: PriorHit) -> str:
    """单条 Prior 的 markdown 块（含全部注入字段；redirected_from 仅在链存在时列出）。"""
    lines = [
        f"### [{pos}] {hit.note_id} {hit.title}".rstrip(),
        f"- note_id: {hit.note_id}",
        f"- title: {hit.title}",
        f"- 置信度: {hit.confidence}",
        f"- volatility: {hit.volatility}",
        f"- observed_at: {hit.observed_at or '（缺省）'}",
    ]
    if hit.source_urls:
        lines.append("- 来源 URL:")
        lines.extend(f"  - {url}" for url in hit.source_urls)
    else:
        lines.append("- 来源 URL: （无）")
    if hit.redirected_from:
        lines.append(f"- redirected_from: {'、'.join(hit.redirected_from)}")
    lines.append("- 正文:")
    if hit.body:
        lines.extend(f"  {line}" for line in hit.body.splitlines())
    else:
        lines.append("  （空）")
    return "\n".join(lines) + "\n\n"


# ---- 未答问题护栏（P4a Task 15：反幻觉记忆注入）-----------------------------


UNANSWERED_GUARD_HEADER = "### 未答问题护栏（不可编造）"


@dataclass(frozen=True)
class UnansweredGuard:
    """一条可注入的未答问题护栏：活跃台账条目 + 与当前问题的相似度。

    由 ``retrieve_unanswered_guards`` 产生；文案由 ``format_unanswered_guards``
    按简报契约渲染（每条一个 ``###`` 小节，问题原文 / 判定日期 / 原因逐字引用）。
    """

    note_id: str
    question: str
    asked_date: str
    reason: str
    similarity: float


def retrieve_unanswered_guards(
    question: str,
    store: WikiStore,
    *,
    guard_similarity: float = DEFAULT_GUARD_SIMILARITY,
    top_k: int = DEFAULT_GUARD_TOP_K,
    dim: int = DEFAULT_SIMILARITY_DIM,
) -> list[UnansweredGuard]:
    """从活跃台账里挑出与当前问题相似度达 ``guard_similarity`` 的条目。

    - 匹配口径：``ledger.question_similarity``（verification 尺度的确定性
      bigram 特征哈希余弦，零网络、可复算）≥ ``guard_similarity``；
    - 排序：相似度降序，同分按 note_id 升序（确定性），最多 ``top_k`` 条；
    - 正文形态不符的台账条目（``parse_unanswered_entry`` 返回 None）跳过；
    - 纯读 store，零写盘、零模型调用，同一入参恒得同一结果。
    """
    from researchwiki.wiki.ledger import (
        iter_active_unanswered,
        parse_unanswered_entry,
        question_similarity,
    )

    scored: list[UnansweredGuard] = []
    for note in iter_active_unanswered(store):
        entry = parse_unanswered_entry(note)
        if entry is None:
            continue
        similarity = question_similarity(entry.question, question, dim=dim)
        if similarity >= guard_similarity:
            scored.append(
                UnansweredGuard(
                    note_id=note.id,
                    question=entry.question,
                    asked_date=entry.asked_date,
                    reason=entry.reason,
                    similarity=similarity,
                )
            )
    scored.sort(key=lambda guard: (-guard.similarity, guard.note_id))
    return scored[: max(0, top_k)]


def format_unanswered_guards(guards: Sequence[UnansweredGuard]) -> str:
    """把护栏条目格式化成 markdown 独立块（每条一个 ``###`` 小节）。

    文案契约（简报逐字）：``### 未答问题护栏（不可编造）`` + 一行判定与禁令。
    空列表返回空字符串；多条时各小节以空行分隔，块以空行结尾（与 Prior 块
    无缝拼接，调用方直接前置到 Prior 块之前）。
    """
    if not guards:
        return ""
    blocks = [
        f"{UNANSWERED_GUARD_HEADER}\n"
        f"问题「{guard.question}」于 {guard.asked_date} 被判定证据不足"
        f"（原因 {guard.reason}）。本次如仍无法证实，必须明说\"证据不足\"，禁止编造。"
        for guard in guards
    ]
    return "\n\n".join(blocks) + "\n\n"


# ---- 待解决冲突（P4a Task 18：open 冲突优先注入，展示字段助手供 CLI 复用）----


# 冲突块头（task-18 简报文案逐字）
CONFLICT_GUARD_HEADER = "### 待解决冲突（优先核验）"
# 冲突注入条数上限：open 冲突是高价值信号但量级远小于笔记，缺省 3 条兜底防失控
# （台账由 consolidation 按 max_actions 开出，常态只有个位数）
DEFAULT_CONFLICT_TOP_K = 3


@dataclass(frozen=True)
class ConflictGuard:
    """一条可注入的待解决冲突：台账记录字段 + 按槽位渲染的一对矛盾断言。

    由 ``retrieve_conflict_guards`` 产生；文案由 ``format_conflict_guards`` 按
    task-18 简报契约渲染（``### 待解决冲突（优先核验）`` + 台账行 + 来源与指令行）。
    """

    conflict_id: str
    entity: str
    assertion_a: str
    assertion_b: str
    note_a: str
    note_b: str


def conflict_claim_notes(
    store: WikiStore, conflict: Conflict
) -> tuple[Note | None, Note | None]:
    """conflict 双方 claim 的笔记对象（读不到返回 None——台账记录可能悬空）。"""

    def _load(claim_side: Mapping[str, object]) -> Note | None:
        note_id = str(claim_side.get("note_id") or "")
        return store.get_note(note_id) if note_id else None

    return _load(conflict.claim_a), _load(conflict.claim_b)


def conflict_entity_names(store: WikiStore, conflict: Conflict) -> list[str]:
    """conflict 的实体名列表（注入匹配与 CLI 展示/留痕共用，确定性排序）。

    实体身份 = slugify 归一键（"GLM-5.3" 与 "glm-5-3" 同一实体，与 ingest 查重
    同一套身份定义）。优先取**双方公共实体**——consolidation 只对同实体对开
    台账，公共实体就是冲突主体；无公共实体（手工登记 / 一方笔记已缺失）回退
    双方并集（护栏宁可多注入，不因记录不完整而漏掉待解决问题）。
    """

    def _keys(note: Note | None) -> dict[str, str]:
        # 归一键 → 原名（先到先得，保留笔记里登记的原始写法）
        keys: dict[str, str] = {}
        if note is None:
            return keys
        for name in note.entities:
            key = slugify(str(name))
            if key and key not in keys:
                keys[key] = str(name)
        return keys

    note_a, note_b = conflict_claim_notes(store, conflict)
    keys_a, keys_b = _keys(note_a), _keys(note_b)
    shared = sorted(set(keys_a) & set(keys_b))
    if shared:
        return [keys_a[key] for key in shared]
    merged: dict[str, str] = {**keys_b, **keys_a}  # 并集时 a 侧写法优先
    return [merged[key] for key in sorted(merged)]


def conflict_assertion_pair(conflict: Conflict) -> tuple[str, str]:
    """conflict 的一对矛盾断言文案（只读台账记录，不重新比较——T16 裁定）。

    T17 执行器把比较槽位原样写进双方 claim（``slots`` 列表 + ``role``），本函数
    从记录取数渲染成「{context} {value}」形态。断言取**第一个槽位**（注入块是
    摘要，全量槽位走 ``arbitrate show``）；槽位缺失时回退 claim.excerpt。两侧
    取值键按 claim.role 对齐（prior → prior_value / evidence → evidence_value），
    role 缺失时按 T16 方向兜底（claim_a = prior）。
    """

    def _render(claim_side: Mapping[str, object], default_key: str) -> str:
        role = str(claim_side.get("role") or "")
        if role in ("prior", "evidence"):
            value_key = "evidence_value" if role == "evidence" else "prior_value"
        else:
            value_key = default_key
        slots = claim_side.get("slots")
        if isinstance(slots, list):
            for slot in slots:
                if not isinstance(slot, Mapping):
                    continue
                context = str(slot.get("context") or "").strip()
                value = str(slot.get(value_key) or "").strip()
                if context or value:
                    return f"{context} {value}".strip()
        return str(claim_side.get("excerpt") or "").strip()

    return _render(conflict.claim_a, "prior_value"), _render(conflict.claim_b, "evidence_value")


def _claim_label(claim_side: Mapping[str, object]) -> str:
    """claim 的来源标签：note_id 优先，缺失回退 title（手工台账可能无 note_id）。"""
    return str(claim_side.get("note_id") or claim_side.get("title") or "未知")


def _note_question_similarity(note: Note | None, question: str, *, dim: int) -> float:
    """冲突笔记与问题的相似度（verification 尺度）。

    文本形态与 research_policy 的全库门控同款（``title + body[:400]``）——同一
    系统对"笔记与问题像不像"用同一把尺、同一种取材。
    """
    if note is None:
        return 0.0
    text = f"{note.title} {note.body[:400]}".strip()
    if not text:
        return 0.0
    return token_similarity(text, question, settings=VerificationSettings(similarity_dim=dim))


def retrieve_conflict_guards(
    question: str,
    store: WikiStore,
    *,
    similarity_floor: float = DEFAULT_SIMILARITY_FLOOR,
    top_k: int = DEFAULT_CONFLICT_TOP_K,
    dim: int = DEFAULT_SIMILARITY_DIM,
) -> list[ConflictGuard]:
    """挑出与当前问题相关的 open 冲突（实体相交 **或** 笔记相似度达标）。

    - 匹配口径（两条满足其一即注入）：
      1. **实体相交**——冲突实体名的 slugify 键出现在问题文本的 slugify 键里
         （``GLM-5.3`` 的键 ``glm-5-3`` 是问题 slug 的子串；写法变体同键归一）；
      2. **笔记相似度**——任一方冲突笔记与问题的 ``token_similarity`` ≥
         ``similarity_floor``（[verification].similarity_floor 同源同尺，维度
         随 ``dim``——与护栏/记忆更新判定同一把尺）；
    - resolved 冲突天然排除（只扫 open）；排序 = conflict_id 升序
      （``list_conflicts`` 的落盘序，确定性），最多 ``top_k`` 条；
    - 双方断言与实体都取不出的退化台账（无可展示内容）跳过；
    - 纯读 store，零模型调用，同一入参恒得同一结果。
    """
    question_key = slugify(str(question))
    guards: list[ConflictGuard] = []
    for conflict in store.list_conflicts(status="open"):
        note_a, note_b = conflict_claim_notes(store, conflict)
        entity_names = conflict_entity_names(store, conflict)
        entity_hit = any(
            (key := slugify(name)) and key in question_key for name in entity_names
        )
        similar = max(
            _note_question_similarity(note_a, question, dim=dim),
            _note_question_similarity(note_b, question, dim=dim),
        ) >= similarity_floor
        if not (entity_hit or similar):
            continue
        assertion_a, assertion_b = conflict_assertion_pair(conflict)
        entity = entity_names[0] if entity_names else str(conflict.claim_a.get("title") or "")
        if not (entity or assertion_a or assertion_b):
            continue  # 无任何可展示内容的退化台账：注入等于噪音
        guards.append(
            ConflictGuard(
                conflict_id=conflict.id,
                entity=entity,
                assertion_a=assertion_a,
                assertion_b=assertion_b,
                note_a=_claim_label(conflict.claim_a),
                note_b=_claim_label(conflict.claim_b),
            )
        )
    return guards[: max(0, top_k)]


def format_conflict_guards(guards: Sequence[ConflictGuard]) -> str:
    """把待解决冲突格式化成 markdown 独立块（task-18 简报文案逐字）。

    块头一行 ``### 待解决冲突（优先核验）``，每条冲突两行：台账行（id + 实体 +
    一对矛盾断言）与来源/指令行（双方笔记 + 优先核验指令）。空列表返回空字符串；
    块以空行结尾（与护栏块/Prior 块无缝拼接，调用方按 护栏 → 冲突 → Prior 顺序
    前置）。
    """
    if not guards:
        return ""
    lines = [CONFLICT_GUARD_HEADER]
    for guard in guards:
        lines.append(
            f"{guard.conflict_id} 实体「{guard.entity}」存在矛盾断言："
            f"「{guard.assertion_a}」vs「{guard.assertion_b}」"
        )
        lines.append(
            f"（来源 {guard.note_a} / {guard.note_b}）。"
            "请优先核实哪一方是现行事实，并在报告中说明。"
        )
    return "\n".join(lines) + "\n\n"


# ---- 索引一致性 -------------------------------------------------------------


def ensure_index_fresh(store: WikiStore, index: SearchIndex) -> tuple[bool, str]:
    """run 开始前检查索引是否落后于 store；落后则整体 rebuild。

    判定**不在本模块**：``index.index_drift`` 是 loop 与 MCP 两条路径共用的唯一
    判据（P2-F 裁定二——同一系统对"索引是否落后"必须只有一个答案），本函数只
    负责"落后了怎么办"：落后 → ``index.rebuild(store)``，返回 ``(True, 原因)``；
    新鲜 → 不写任何东西，返回 ``(False, 原因)``。返回文案与抽判据之前逐字一致
    （陈旧态 ``索引落后于 store：…；已执行 rebuild（索引 N 条笔记）``，新鲜态
    ``索引与 store 一致（…），无需 rebuild``），P2-B 的调用方与测试不受影响。

    三维口径（id 集合 / status / 索引指纹，含墓碑标记与空指纹）见 ``index_drift``
    的 docstring；本函数只读 store + 写索引，不动 md。
    """
    stale, reason = index_drift(store, index)
    if not stale:
        return False, f"{reason}，无需 rebuild"
    count = index.rebuild(store)
    return True, f"{reason}；已执行 rebuild（索引 {count} 条笔记）"
