"""wiki 健康度检查（``researchwiki lint``）：引用覆盖 / 断链 / merged 跟随 / 孤立笔记。

指标定义（生成侧与检查侧共用同一套"断言行"定义，见 iter_assertion_lines，
Distiller 生成页面时会用同一套规则做确定性兜底标注）：

- **断言行**：正文里非空、非标题、非代码块、非纯链接行（``- [[entity:x|X]]`` 这类
  导航/双链行不算断言，否则"相关实体"清单会拖垮覆盖率）；
- **引用覆盖**：页面正文的断言行里出现笔记 ID（``N-XXXX``）的比例——行内标注即可，
  不要求 [n] 式来源引用（那是研究报告的约定，页面用笔记 ID 溯源）。只统计页面正文：
  笔记正文本身就是原子断言，不要求（也不该）在正文里标注自己的 ID；
- **断链**：``[[entity:<slug>]]`` 指向未注册实体；笔记/页面里引用的笔记 ID 不存在；
- **merged 跟随**：引用的是 merged/superseded 笔记时，沿 redirect_to / superseded_by
  走到最终 active 笔记；判定通过但记入 merged_chains，提示改写成规范 ID
  （引用应指向规范条目，但历史 ID 永不消失，故"引用旧 ID"不是错误）；
- **孤立笔记**：没有任何页面引用的 active 笔记（页面引用的 merged 记录会跟随到规范 ID）。

退出码约定（给 CI 用）：**有断链**，或"存在断言行且引用覆盖率为 0" → 1，否则 0。
没有任何断言行（空 wiki）时覆盖率取 1.0（真空真），避免空库把 CI 判红。
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from researchwiki.wiki.entities import EntityRegistry

if TYPE_CHECKING:  # 仅类型标注用：避免 wiki.index → store → loop.notes 的循环导入
    from researchwiki.wiki.index import SearchIndex
    from researchwiki.wiki.store import WikiStore

# 笔记 ID：N-0001 这类编号（frontmatter 之外的正文里出现即算引用）
NOTE_ID_RE = re.compile(r"\bN-\d+\b")
# 稳定 ID 双链：[[entity:glm-5-3|GLM-5.3]]（显示名可省略）
ENTITY_LINK_RE = re.compile(r"\[\[entity:([^\]|]+?)(?:\|[^\]]*)?\]\]")
# 笔记双链写法 [[note:N-0001|...]]（其中 N-0001 也会被 NOTE_ID_RE 命中）
NOTE_LINK_RE = re.compile(r"\[\[note:([^\]|]+?)(?:\|[^\]]*)?\]\]")

_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_HEADING_RE = re.compile(r"^#{1,6}\s")
_TABLE_ROW_RE = re.compile(r"^\|.*\|\s*$")
# 去掉链接与标点/空白后仍为空 → 纯链接行或纯符号行（不算断言）
_LINK_STRIP_RE = re.compile(r"\[\[[^\]]*\]\]")
_DECOR_RE = re.compile(r"[\s\-*+>·、,，;；:：.。!！?？|/\\()（）\[\]【】{}<>《》“”\"'`#~=_]+")


def is_assertion_line(line: str) -> bool:
    """判断一行是不是"断言行"（需要笔记 ID 标注的内容行）。

    标题、空行、代码围栏标记、纯双链/纯符号行（如 ``- [[entity:letta|Letta]]``）返回 False。
    表格行按内容判定：表格里写断言同样需要标注，故不特殊排除。
    """
    text = line.strip()
    if not text or _FENCE_RE.match(text) or _HEADING_RE.match(text):
        return False
    return bool(_DECOR_RE.sub("", _LINK_STRIP_RE.sub("", text)))


def iter_assertion_lines(body: str) -> Iterator[tuple[int, str]]:
    """遍历正文里的断言行，产出 (行号, 行内容)；代码块（``` / ~~~ 围栏）内一律跳过。"""
    in_fence = False
    for lineno, line in enumerate(body.splitlines(), start=1):
        stripped = line.strip()
        if _FENCE_RE.match(stripped):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if is_assertion_line(stripped):
            yield lineno, stripped


def referenced_note_ids(body: str) -> list[str]:
    """正文里出现的笔记 ID 引用（按出现顺序去重）。"""
    seen: list[str] = []
    for match in NOTE_ID_RE.finditer(body):
        if match.group(0) not in seen:
            seen.append(match.group(0))
    return seen


@dataclass
class BrokenLink:
    """一处断链：来源（页面/笔记）+ 目标（实体 slug 或笔记 ID）+ 人话原因。"""

    kind: str  # "entity" | "note"
    source: str  # "page:letta" / "note:N-0001"
    target: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class MergedRef:
    """引用 merged/superseded 笔记：跟随 redirect 通过，但建议改写为规范 ID。"""

    source: str
    referenced: str
    canonical: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class LintReport:
    """一次 lint 的完整结果（字段即对外契约，CLI 的 --json 直接序列化）。"""

    citation_coverage: float = 1.0
    orphan_notes: list[str] = field(default_factory=list)
    broken_links: list[BrokenLink] = field(default_factory=list)
    merged_chains: list[MergedRef] = field(default_factory=list)
    notes_total: int = 0
    pages_total: int = 0
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def healthy(self) -> bool:
        """是否健康：无断链且引用覆盖率非 0（空 wiki 视为 1.0，真空真）。"""
        return not self.broken_links and self.citation_coverage > 0.0

    def exit_code(self) -> int:
        """CI 退出码：有断链或引用覆盖率为 0 → 1。"""
        return 0 if self.healthy else 1

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["healthy"] = self.healthy
        data["exit_code"] = self.exit_code()
        return data

    def format_text(self, *, root: str = "") -> str:
        """人类可读的中文报告（stdout 输出用）。"""
        lines: list[str] = []
        title = f"wiki 健康度报告（{root}）" if root else "wiki 健康度报告"
        lines.append(title)
        lines.append("=" * 46)
        lines.append(
            f"- 笔记：{self.notes_total} 条（active）· 页面：{self.pages_total} 个"
        )
        cited = int(self.details.get("cited_lines") or 0)
        total = int(self.details.get("assertion_lines") or 0)
        lines.append(
            f"- 引用覆盖：{self.citation_coverage * 100:.1f}%"
            f"（{cited}/{total} 行断言带笔记 ID 标注）"
        )
        if self.orphan_notes:
            shown = "、".join(self.orphan_notes[:12])
            more = f" 等 {len(self.orphan_notes)} 条" if len(self.orphan_notes) > 12 else ""
            lines.append(f"- 孤立笔记（无页面引用）：{shown}{more}")
        else:
            lines.append("- 孤立笔记：无")
        if self.broken_links:
            lines.append(f"- 断链：{len(self.broken_links)} 处")
            for link in self.broken_links[:12]:
                lines.append(f"  - [{link.source}] {link.target} —— {link.message}")
        else:
            lines.append("- 断链：无")
        if self.merged_chains:
            count = len(self.merged_chains)
            lines.append(f"- 引用了 merged 笔记：{count} 处（已跟随 redirect 通过）")
            for ref in self.merged_chains[:12]:
                lines.append(f"  - [{ref.source}] {ref.referenced} → 规范 ID {ref.canonical}")
        uncited = self.details.get("uncited_lines") or []
        if uncited:
            lines.append("- 缺标注的断言行示例：")
            for item in uncited[:5]:
                lines.append(f"  - [{item['source']}:{item['line']}] {item['text'][:60]}")
        lines.append(f"- 结论：{'健康' if self.healthy else '不健康'}（退出码 {self.exit_code()}）")
        return "\n".join(lines)


def lint_wiki(store: WikiStore, *, index: SearchIndex | None = None) -> LintReport:
    """体检一个 wiki：引用覆盖、断链、merged 跟随、孤立笔记。

    index 可选：传入检索索引时顺带做一次"索引是否落后于 md"的粗检查
    （best-effort，读不到索引内部表就跳过，不影响其它指标）。
    """
    registry = EntityRegistry(store.root)
    all_notes = store.list_notes(status=None)
    active_notes = [n for n in all_notes if n.status == "active"]
    pages = store.list_pages()

    broken: list[BrokenLink] = []
    merged_refs: list[MergedRef] = []
    cited_lines = 0
    assertion_lines = 0
    uncited: list[dict[str, Any]] = []
    cited_canonical_in_pages: set[str] = set()

    # 1) 页面：引用覆盖 + 页面里的实体/笔记引用
    for page in pages:
        source = f"page:{page.id}"
        for lineno, line in iter_assertion_lines(page.body):
            assertion_lines += 1
            if NOTE_ID_RE.search(line):
                cited_lines += 1
            elif len(uncited) < 20:
                uncited.append({"source": source, "line": lineno, "text": line})
        _check_entity_links(page.body, source=source, registry=registry, broken=broken)
        for note_id in referenced_note_ids(page.body):
            canonical = _resolve_reference(store, note_id, source=source, broken=broken)
            if canonical is not None:
                cited_canonical_in_pages.add(canonical)
                if canonical != note_id:
                    merged_refs.append(
                        MergedRef(source=source, referenced=note_id, canonical=canonical)
                    )

    # 2) 笔记：笔记正文里的实体/笔记引用（笔记之间也会互相引用）
    for note in all_notes:
        source = f"note:{note.id}"
        _check_entity_links(note.body, source=source, registry=registry, broken=broken)
        for note_id in referenced_note_ids(note.body):
            if note_id == note.id:
                continue  # 自引用（记 ID 的写法）不算引用
            canonical = _resolve_reference(store, note_id, source=source, broken=broken)
            if canonical is not None and canonical != note_id:
                merged_refs.append(
                    MergedRef(source=source, referenced=note_id, canonical=canonical)
                )

    orphans = sorted(n.id for n in active_notes if n.id not in cited_canonical_in_pages)
    coverage = cited_lines / assertion_lines if assertion_lines else 1.0
    details: dict[str, Any] = {
        "assertion_lines": assertion_lines,
        "cited_lines": cited_lines,
        "uncited_lines": uncited,
        "active_notes": len(active_notes),
        "records_total": len(all_notes),
    }
    stale = _index_lag(index, {n.id for n in all_notes})
    if stale is not None:
        details["index_checked"] = True
        details["index_stale"] = stale
    return LintReport(
        citation_coverage=coverage,
        orphan_notes=orphans,
        broken_links=broken,
        merged_chains=merged_refs,
        notes_total=len(active_notes),
        pages_total=len(pages),
        details=details,
    )


def _check_entity_links(
    body: str, *, source: str, registry: EntityRegistry, broken: list[BrokenLink]
) -> None:
    """正文里的 [[entity:...]] 必须指向注册表存在的实体。"""
    for match in ENTITY_LINK_RE.finditer(body):
        entity_id = match.group(1).strip()
        if entity_id and registry.resolve(entity_id) is None:
            broken.append(
                BrokenLink(
                    kind="entity",
                    source=source,
                    target=f"[[entity:{entity_id}]]",
                    message=f"实体 {entity_id!r} 未注册（需先在 entities/ 建条目）",
                )
            )


def _resolve_reference(
    store: WikiStore, note_id: str, *, source: str, broken: list[BrokenLink]
) -> str | None:
    """把正文里的笔记 ID 解析成规范 ID；断链就地登记并返回 None。"""
    try:
        target = store.follow_redirect(note_id)
    except ValueError as exc:  # redirect 环路
        broken.append(
            BrokenLink(kind="note", source=source, target=note_id, message=f"redirect 环路：{exc}")
        )
        return None
    if target is None:
        exists = store.get_note(note_id) is not None
        reason = "redirect 链断裂（指向不存在的笔记）" if exists else "笔记不存在"
        broken.append(BrokenLink(kind="note", source=source, target=note_id, message=reason))
        return None
    return target.id


def _index_lag(index: SearchIndex | None, note_ids: set[str]) -> list[str] | None:
    """索引落后于 md 的笔记 id 列表；读不到索引内部表时返回 None（跳过检查）。

    MVP 的 best-effort：SearchIndex 没有公开的"列出已索引 id"接口，
    这里直接读它持有的连接（缺失就跳过，绝不因索引不可用让 lint 失败）。
    """
    conn = getattr(index, "_conn", None)
    if conn is None:
        return None
    try:
        rows = conn.execute("SELECT note_id FROM note_meta").fetchall()
    except Exception:  # noqa: BLE001 -- 索引库结构异常时跳过这项附加检查
        return None
    indexed = {str(row[0]) for row in rows}
    return sorted(note_ids - indexed)
