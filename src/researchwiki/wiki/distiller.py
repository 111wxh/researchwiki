"""蒸馏层：研究报告 → 原子笔记（LLM 抽取）→ 实体/主题页（LLM 聚合 + 稳定 ID 双链）。

三段职责：

- ``extract_notes``：一次 cheap 档 LLM 调用把报告拆成自包含的原子笔记，顺带收集
  来源之间明显矛盾的事实。**降级优先**——非法 JSON、缺字段、超量、越界 URL 一律
  截断/丢弃并记入 ``degradations``，绝不抛异常打断整轮研究（研究可以少沉淀，
  不能因为一次抽取失败全盘作废）。
- ``build_pages``：按实体聚合笔记，LLM 生成实体/主题页正文；生成结果再做**确定性
  兜底修复**（断言行补笔记 ID 标注、补齐 ``[[entity:<slug>|显示名]]`` 双链），
  保证页面格式对 ``researchwiki lint`` 友好，模型偶发偷懒也不至于让 wiki 变脏。
- ``annotate_citations`` / ``ensure_entity_links``：可单独测试的纯函数。

格式契约（与 wiki/lint.py 共享同一套"断言行"定义，见 ``lint.iter_assertion_lines``）：

- 页面里每个断言行行内必须出现笔记 ID（``N-XXXX``），多条依据写作 ``（N-0003、N-0007）``；
- 实体一律用稳定 ID 双链 ``[[entity:glm-5-3|GLM.5.3]]``——**链接用 ID（不可变），
  显示名可变**，slug 来自 EntityRegistry（``slugify``）。

提示词全部写在模块里、system 措辞固定（KV-cache 友好：同一 run 内多次调用与
跨 run 复用同一前缀）。
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from researchwiki.llm.accounting import TokenAccountant
from researchwiki.llm.provider import Message, Provider, TokenUsage
from researchwiki.loop.subagent import extract_json
from researchwiki.wiki.entities import Entity, EntityRegistry
from researchwiki.wiki.frontmatter import CONFIDENCE_LEVELS, VOLATILITY_LEVELS, SourceRef
from researchwiki.wiki.lint import ENTITY_LINK_RE, NOTE_ID_RE, is_assertion_line

if TYPE_CHECKING:  # 仅类型标注用：避免 wiki.store → loop.notes → loop.agent_loop 循环导入
    from researchwiki.wiki.store import Note, WikiStore

MAX_REPORT_CHARS = 12_000
MAX_NOTE_CHARS = 240
MAX_ENTITIES_PER_NOTE = 6
MAX_CONFLICTS = 5

# 双链（带显示名捕获组）：[[entity:glm-5-3|GLM-5.3]] / [[entity:letta]]
ENTITY_LINK_DISPLAY_RE = re.compile(r"\[\[entity:([^\]|]+?)(?:\|([^\]]*))?\]\]")


# ---- 提示词（中文、措辞稳定，KV-cache 友好）--------------------------------


EXTRACT_SYSTEM = (
    "你是「自进化研究 Wiki」的信息蒸馏器。从研究报告里抽取原子笔记，"
    "并指出来源之间明显矛盾的事实。\n"
    "只输出严格 JSON（不要输出任何其他文本，不要用代码围栏）：\n"
    '{"notes": [{"text": "单一事实、自包含、不超过 80 字", "entities": ["实体名"], '
    '"confidence": "high|medium|low", "volatility": "stable|drifting|volatile", '
    '"source_urls": ["https://..."]}], '
    '"conflicts": [{"summary": "矛盾描述", "action": "建议动作"}]}\n'
    "规则：\n"
    "- 一条笔记只写一个事实，不要照抄报告整段；不要重复同一条事实；\n"
    "- entities 填 1-3 个规范实体名，同一实体在不同笔记里保持同一写法；\n"
    "- confidence：多来源互相印证 high，单一来源 medium，存疑 low；\n"
    "- volatility：版本号/价格/榜单等会变的事实 volatile，方法与原理等稳定事实 stable，"
    "介于两者之间 drifting；\n"
    "- source_urls 只能取下方来源列表里的 URL，没有对应来源就填空数组；\n"
    "- 来源之间明显矛盾的事实写入 conflicts，没有就输出空数组。"
)

PAGE_SYSTEM = (
    "你是「自进化研究 Wiki」的页面编者：把若干原子笔记整理成一个实体/主题页。\n"
    "输出 Markdown 正文，必须遵守以下格式（健康度检查按此判定）：\n"
    "- 第一行是 `# 页面标题`，随后按主题分 2-4 个小节（`## 小节名`）；\n"
    "- 每个断言行（段落或列表项）末尾用括号标注依据的笔记 ID，例如 `（N-0003）`，"
    "多条依据写作 `（N-0003、N-0007）`；只能引用给出的笔记 ID，不得编造；\n"
    "- 提到其它实体时用稳定 ID 双链：`[[entity:glm-5-3|GLM-5.3]]`"
    "（链接目标用给定的稳定 ID，显示名可读）；\n"
    "- 最后一节固定为 `## 相关实体`，逐行列出 `- [[entity:<稳定 ID>|<显示名>]]`；\n"
    "- 不要输出 frontmatter，不要编造笔记里没有的事实，不要重复同一断言。"
)


# ---- 数据结构 --------------------------------------------------------------


@dataclass
class CandidateNote:
    """蒸馏出的候选笔记（尚未入库；id 由入库层分配）。

    - source_urls 是抽取出的原始 URL 列表（已过滤到本次来源池内）；
    - source_refs 是与来源快照对上的引用（url + content_hash），入库时写进 frontmatter。
    """

    text: str
    entities: list[str] = field(default_factory=list)
    confidence: str = "medium"
    volatility: str = "stable"
    source_urls: list[str] = field(default_factory=list)
    source_refs: list[SourceRef] = field(default_factory=list)

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any], *, sources: Sequence[SourceRef] = ()
    ) -> CandidateNote:
        """从 LLM 返回项 / 子 agent 笔记 dict 构造候选；字段缺失走默认值，类型漂移容错。"""
        raw_urls = _url_list(data.get("source_urls") or data.get("sources"))
        refs = match_source_refs(raw_urls, sources)
        urls = [ref.url for ref in refs] or [
            url for url in raw_urls if any(s.url == url for s in sources)
        ]
        return cls(
            text=str(data.get("text") or "").strip(),
            entities=coerce_entities(data.get("entities")),
            confidence=coerce_enum(data.get("confidence"), CONFIDENCE_LEVELS, "medium"),
            volatility=coerce_enum(data.get("volatility"), VOLATILITY_LEVELS, "stable"),
            source_urls=urls,
            source_refs=refs,
        )


@dataclass
class CandidateConflict:
    """一次抽取顺带发现的矛盾（已落 conflicts/ 台账，id 为台账编号）。"""

    id: str
    summary: str
    action: str = ""


@dataclass
class PageDraft:
    """一个实体/主题页草稿（未落盘；由调用方决定何时 save_page）。"""

    slug: str
    title: str
    body: str
    entity_ids: list[str] = field(default_factory=list)
    note_ids: list[str] = field(default_factory=list)


# ---- 纯函数辅助 ------------------------------------------------------------


def coerce_enum(value: Any, allowed: tuple[str, ...], default: str) -> str:
    """枚举字段归一：小写后落在白名单内才认，否则回默认值。"""
    text = str(value or "").strip().lower()
    return text if text in allowed else default


def coerce_entities(value: Any, *, limit: int = MAX_ENTITIES_PER_NOTE) -> list[str]:
    """实体列表归一：接受列表或「、,;」分隔的字符串，去重保序并截断。"""
    if isinstance(value, str):
        raw: list[str] = [part for part in re.split(r"[，,、;；/|]+", value)]
    elif isinstance(value, list | tuple):
        raw = [str(item) for item in value]
    else:
        raw = []
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        name = item.strip()
        key = name.casefold()
        if not name or key in seen:
            continue
        seen.add(key)
        out.append(name)
        if len(out) >= limit:
            break
    return out


def _url_list(value: Any) -> list[str]:
    """URL 列表归一：接受 ["https://..."] 或 [{"url": "..."}]，去重保序。"""
    items = value if isinstance(value, list | tuple) else []
    out: list[str] = []
    for item in items:
        url = ""
        if isinstance(item, str):
            url = item.strip()
        elif isinstance(item, Mapping):
            url = str(item.get("url") or "").strip()
        if url and url not in out:
            out.append(url)
    return out


def match_source_refs(
    urls: Sequence[str], sources: Sequence[SourceRef]
) -> list[SourceRef]:
    """把 URL 列表对上来源池里的 SourceRef（带上 content_hash）；来源池外的 URL 丢弃。"""
    by_url = {source.url: source for source in sources}
    refs: list[SourceRef] = []
    for url in urls:
        ref = by_url.get(url)
        if ref is not None and ref not in refs:
            refs.append(ref)
    return refs


def call_text(
    provider: Provider,
    *,
    system: str,
    user: str,
    step: str = "",
    accountant: TokenAccountant | None = None,
    trace_id: str = "",
    clock: Callable[[], float] = time.perf_counter,
    on_usage: Callable[[TokenUsage], None] | None = None,
) -> str:
    """整段消费一次 LLM 调用并拼出文本（wiki 层的蒸馏/合并共用；不向事件流转发）。

    on_usage 让调用方（agent loop）把这次调用的 token 计入自己的滚动状态——
    loop 的 input_tokens 必须覆盖全部主循环调用（含蒸馏），否则预算判断会漏账。
    """
    t0 = clock()
    parts: list[str] = []
    usage: TokenUsage | None = None
    for event in provider.stream(
        [Message(role="user", content=user)], system=system, tools=None
    ):
        if event.type == "text_delta":
            parts.append(event.delta)
        elif event.type == "usage" and event.usage is not None:
            usage = event.usage
    if accountant is not None:
        accountant.record(
            trace_id=trace_id,
            step=step or "wiki",
            model=provider.model,
            usage=usage or TokenUsage(),
            latency_ms=(clock() - t0) * 1000.0,
        )
    if on_usage is not None and usage is not None:
        on_usage(usage)
    return "".join(parts)


def _bigrams(text: str) -> set[str]:
    """字符 bigram 集合（去空白）；长度 1 的串退化为 unigram。"""
    compact = "".join(ch for ch in text if not ch.isspace())
    if len(compact) < 2:
        return {compact} if compact else set()
    return {compact[i : i + 2] for i in range(len(compact) - 1)}


def best_matching_note(line: str, note_texts: Mapping[str, str]) -> str:
    """挑出与某一行最像的笔记 ID（bigram 重合度，取包含度最高者；无重合返回 ""）。"""
    line_grams = _bigrams(line)
    best_id = ""
    best_score = 0.0
    for note_id, text in note_texts.items():
        grams = _bigrams(text)
        if not line_grams or not grams:
            continue
        overlap = len(line_grams & grams)
        if not overlap:
            continue
        score = overlap / min(len(line_grams), len(grams))
        if score > best_score:
            best_id, best_score = note_id, score
    return best_id


def annotate_citations(body: str, note_texts: Mapping[str, str]) -> str:
    """给缺笔记 ID 标注的断言行补上引用（确定性兜底，保证 lint 引用覆盖率）。

    行内已有 ``N-XXXX`` 的行不动；缺标注的行按"与哪条笔记最像"补一条依据，
    完全无重合时回退到第一条笔记。``note_texts`` 为空时原样返回。
    """
    if not note_texts:
        return body
    fallback = next(iter(note_texts))
    trailing = "\n" if body.endswith("\n") else ""
    out: list[str] = []
    for line in body.splitlines():
        if _contains_note_id(line) or not is_assertion_line(line):
            out.append(line)
            continue
        note_id = best_matching_note(line, note_texts) or fallback
        out.append(f"{line.rstrip()}（{note_id}）")
    return "\n".join(out) + trailing


def _contains_note_id(line: str) -> bool:
    return NOTE_ID_RE.search(line) is not None


def ensure_entity_links(body: str, entities: Sequence[Entity]) -> str:
    """保证页面含各实体的稳定 ID 双链；缺失的补进末尾（或既有）「相关实体」小节。"""
    present = {match.group(1).strip().casefold() for match in ENTITY_LINK_RE.finditer(body)}
    missing = [entity for entity in entities if entity.id.casefold() not in present]
    if not missing:
        return body
    bullets = "\n".join(f"- [[entity:{e.id}|{e.name}]]" for e in missing)
    text = body if body.endswith("\n") else body + "\n"
    heading = re.search(r"^##\s*相关实体[^\n]*$", text, re.MULTILINE)
    if heading is not None:
        return f"{text[: heading.end()]}\n{bullets}{text[heading.end() :]}"
    return f"{text}\n## 相关实体\n\n{bullets}\n"


def restrict_entity_links(body: str, known: set[str]) -> str:
    """把指向未注册实体的双链降级成纯文本（页面不允许产生断链）。

    known 是允许的稳定 ID 集合（本页相关实体 + 注册表已有实体）。模型偶尔会
    自行发明实体名，这里直接在生成侧兜住——比事后靠 lint 发现更省事。
    """
    def replace(match: re.Match[str]) -> str:
        target = match.group(1).strip()
        if target.casefold() in known:
            return match.group(0)
        return (match.group(2) or target).strip()

    return ENTITY_LINK_DISPLAY_RE.sub(replace, body)


def skeleton_page(entity: Entity, note_texts: Mapping[str, str]) -> str:
    """模型输出不可用时的确定性骨架页：标题 + 笔记逐条列出（已带 ID 标注）。"""
    lines = [f"# {entity.name}", "", "## 关键事实", ""]
    lines += [f"- {text}（{note_id}）" for note_id, text in note_texts.items()]
    return "\n".join(lines) + "\n"


def strip_code_fence(text: str) -> str:
    """剥掉模型爱加的 ```markdown 围栏（只处理整体包裹的情况）。"""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    body = re.sub(r"^```[a-zA-Z]*\n?", "", stripped)
    body = re.sub(r"\n?```\s*$", "", body)
    return body.strip()


# ---- 蒸馏器 ----------------------------------------------------------------


class Distiller:
    """报告 → 原子笔记 → 实体/主题页；LLM 由调用方注入（测试用 ScriptedProvider）。"""

    def __init__(
        self,
        store: WikiStore,
        *,
        provider: Provider,
        entity_registry: EntityRegistry | None = None,
        accountant: TokenAccountant | None = None,
        trace_id: str = "",
        clock: Callable[[], float] = time.perf_counter,
        step: str = "distill",
        max_report_chars: int = MAX_REPORT_CHARS,
        on_usage: Callable[[TokenUsage], None] | None = None,
    ) -> None:
        self.store = store
        self.provider = provider
        self.entity_registry = (
            entity_registry if entity_registry is not None else EntityRegistry(store.root)
        )
        self.accountant = accountant
        self.trace_id = trace_id
        self.clock = clock
        self.step = step
        self.max_report_chars = max_report_chars
        self.on_usage = on_usage
        # 最近一次抽取的降级记录（截断/丢弃原因，中文，供 state.md 与测试读取）
        self.degradations: list[str] = []
        # 最近一次抽取顺带发现的矛盾（已落台账）；事件层用它的 id 发 data-conflict
        self.last_conflicts: list[CandidateConflict] = []

    # ---- 抽取 ----------------------------------------------------------

    def extract_notes(
        self,
        report: str,
        *,
        question: str = "",
        sources: Sequence[SourceRef] = (),
        max_notes: int = 8,
    ) -> list[CandidateNote]:
        """从报告抽取原子笔记；任何异常/畸形输出都降级为空列表 + degradations 记录。"""
        self.degradations = []
        self.last_conflicts = []
        text = str(report or "").strip()
        if not text:
            self.degradations.append("蒸馏输入为空，跳过抽取")
            return []
        if len(text) > self.max_report_chars:
            text = text[: self.max_report_chars]
            self.degradations.append(f"报告超长，截断到 {self.max_report_chars} 字符")
        source_list = list(sources)
        try:
            raw = call_text(
                self.provider,
                system=EXTRACT_SYSTEM,
                user=build_extract_prompt(question, text, source_list, max_notes),
                step=self.step,
                accountant=self.accountant,
                trace_id=self.trace_id,
                clock=self.clock,
                on_usage=self.on_usage,
            )
            parsed = extract_json(raw)
            if not isinstance(parsed, Mapping):
                self.degradations.append("抽取输出不是合法 JSON，本轮不抽取笔记")
                return []
            notes = self._collect_notes(parsed.get("notes"), source_list, max_notes=max_notes)
            self.last_conflicts = self._collect_conflicts(parsed.get("conflicts"), question)
        except Exception as exc:  # noqa: BLE001 -- 降级优先：任何异常都不得打断整轮研究
            self.degradations.append(
                f"抽取阶段异常，本轮不抽取笔记：{type(exc).__name__}: {exc}"
            )
            self.last_conflicts = []
            return []
        return notes

    def _collect_notes(
        self, items: Any, sources: Sequence[SourceRef], *, max_notes: int
    ) -> list[CandidateNote]:
        """逐项校验候选笔记：缺正文/结构非法丢弃，越界 URL 与超长正文截断并记录。"""
        if not isinstance(items, list):
            self.degradations.append("抽取输出缺少 notes 数组，本轮不抽取笔记")
            return []
        known_urls = {source.url for source in sources}
        notes: list[CandidateNote] = []
        dropped = 0
        for item in items:
            if not isinstance(item, Mapping):
                dropped += 1
                continue
            candidate = CandidateNote.from_dict(item, sources=sources)
            if not candidate.text:
                dropped += 1
                continue
            if len(candidate.text) > MAX_NOTE_CHARS:
                candidate.text = candidate.text[:MAX_NOTE_CHARS]
                self.degradations.append(f"笔记正文超长，截断到 {MAX_NOTE_CHARS} 字符")
            outside = [
                url
                for url in _url_list(item.get("source_urls") or item.get("sources"))
                if url not in known_urls
            ]
            if outside:
                self.degradations.append(f"丢弃来源列表之外的 URL：{outside[0]}")
            for name in candidate.entities:
                self.entity_registry.get_or_create(name)
            notes.append(candidate)
        if dropped:
            self.degradations.append(f"丢弃 {dropped} 条结构非法的笔记项（缺 text 或非对象）")
        if len(notes) > max_notes:
            self.degradations.append(
                f"抽取 {len(notes)} 条超过上限 {max_notes}，截断保留前 {max_notes} 条"
            )
            notes = notes[:max_notes]
        return notes

    def _collect_conflicts(self, items: Any, question: str) -> list[CandidateConflict]:
        """矛盾点落 conflicts/ 台账（question 取矛盾描述），返回可直接发事件的条目。"""
        if not isinstance(items, list):
            return []
        out: list[CandidateConflict] = []
        for item in items:
            if not isinstance(item, Mapping):
                continue
            summary = str(item.get("summary") or "").strip()
            if not summary:
                continue
            action = str(item.get("action") or "").strip()
            record = self.store.save_conflict(
                summary,
                {"summary": summary, "question": question},
                {"action": action},
                trace_id=self.trace_id,
            )
            out.append(CandidateConflict(id=record.id, summary=summary, action=action))
            if len(out) >= MAX_CONFLICTS:
                self.degradations.append(
                    f"矛盾点超过 {MAX_CONFLICTS} 条，只登记前 {MAX_CONFLICTS} 条"
                )
                break
        return out

    # ---- 建页 ----------------------------------------------------------

    def build_pages(self, notes: Sequence[Note], *, max_pages: int = 3) -> list[PageDraft]:
        """按实体聚合笔记并生成页面草稿（不落盘，由调用方 save_page）。

        notes 是**已入库**的笔记（需要规范 ID 才能做行内标注）；同一笔记出现在多个
        实体下时按实体各建一页，页内按规范 ID 去重。
        """
        groups: dict[str, list[Note]] = {}
        entities: dict[str, Entity] = {}

        def entity_for(name: str) -> Entity:
            entity = self.entity_registry.get_or_create(str(name))
            entities[entity.id] = entity
            return entity

        for note in notes:
            note_id = str(getattr(note, "id", "") or "")
            body = str(getattr(note, "body", "") or "").strip()
            if not note_id or not body:
                self.degradations.append("页面聚合跳过缺 id/正文的笔记")
                continue
            for name in note.entities:
                entity = entity_for(name)
                bucket = groups.setdefault(entity.id, [])
                if all(existing.id != note_id for existing in bucket):
                    bucket.append(note)
        if not groups:
            return []
        ordered = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))[:max_pages]
        drafts: list[PageDraft] = []
        for entity_id, group in ordered:
            # 相关实体 = 本组笔记涉及的全部实体（含本页实体自身，作为 canonical 锚点）
            related: list[Entity] = []
            for note in group:
                for name in note.entities:
                    member = entity_for(name)
                    if all(member.id != item.id for item in related):
                        related.append(member)
            related.sort(key=lambda item: item.id)
            drafts.append(self._build_page(entities[entity_id], group, related))
        return drafts

    def _build_page(
        self, entity: Entity, group: Sequence[Note], related: Sequence[Entity]
    ) -> PageDraft:
        note_texts = {note.id: note.body.strip() for note in group}
        raw = call_text(
            self.provider,
            system=PAGE_SYSTEM,
            user=build_page_prompt(entity, group, related),
            step=f"{self.step}:pages",
            accountant=self.accountant,
            trace_id=self.trace_id,
            clock=self.clock,
            on_usage=self.on_usage,
        )
        body = strip_code_fence(raw)
        if len(body) < 20:
            self.degradations.append(f"页面 {entity.id} 的模型输出不可用，改用骨架页")
            body = skeleton_page(entity, note_texts)
        body = _ensure_heading(body, entity.name)
        known = {item.id.casefold() for item in related} | {
            item.id.casefold() for item in self.entity_registry.list_entities()
        }
        restricted = restrict_entity_links(body, known)
        if restricted != body:
            self.degradations.append(f"页面 {entity.id} 的未注册实体双链已降级为纯文本")
            body = restricted
        body = annotate_citations(body, note_texts)
        body = ensure_entity_links(body, related)
        return PageDraft(
            slug=entity.id,
            title=entity.name,
            body=body,
            entity_ids=[item.id for item in related],
            note_ids=list(note_texts),
        )


def _ensure_heading(body: str, title: str) -> str:
    """正文缺一级标题时补上（lint 会把标题行排除在断言行之外）。"""
    text = body if body.endswith("\n") else body + "\n"
    lines = text.splitlines()
    if lines and lines[0].lstrip().startswith("# "):
        return text
    return f"# {title}\n\n{text}"


# ---- 提示词拼装 ------------------------------------------------------------


def build_extract_prompt(
    question: str, report: str, sources: Sequence[SourceRef], max_notes: int
) -> str:
    """抽取阶段的 user 提示：研究问题 + 编号来源列表 + 报告正文。"""
    lines: list[str] = []
    if question.strip():
        lines.append(f"研究问题：{question.strip()}")
    if sources:
        lines.append("来源列表：")
        lines += [f"[{i}] {source.url}" for i, source in enumerate(sources, start=1)]
    lines.append("研究报告：")
    lines.append(report)
    lines.append(f"请抽取原子笔记（最多 {max_notes} 条）与矛盾点，输出严格 JSON。")
    return "\n".join(lines)


def build_page_prompt(
    entity: Entity, notes: Sequence[Note], related: Sequence[Entity]
) -> str:
    """建页阶段的 user 提示：页面实体 + 可用双链目标 + 笔记清单（带 ID）。"""
    lines = [
        f"页面实体：{entity.name}（稳定 ID：{entity.id}）",
        "相关实体（双链只能指向这些稳定 ID）：",
    ]
    lines += [f"- {item.id} → {item.name}" for item in related]
    lines.append("笔记列表（每行一条，格式为「ID：正文」）：")
    lines += [f"- {note.id}：{note.body.strip()}" for note in notes]
    lines.append(
        f"请输出「{entity.name}」页面正文（Markdown），"
        "每个断言行标注笔记 ID，相关实体用稳定 ID 双链。"
    )
    return "\n".join(lines)
