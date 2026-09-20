"""wiki 存储层：原子笔记 notes/、聚合页 pages/、冲突台账 conflicts/。

目录布局（root 即 wiki-data，与 loop 层共用）：
- notes/N-XXXX.md      原子笔记，编号续接既有最大值（复用 loop.notes 的扫描器，
                       保证与 loop 层 NoteStore 写入的轻量笔记无缝续号、互读兼容）
- pages/{slug}.md      聚合页（人工/蒸馏整理的主题页）
- conflicts/C-XXXX.md  冲突台账：同一实体的矛盾断言双方证据 + 裁决记录

所有写盘走 atomic_write_text（tmp + os.replace）；列表默认只返回 status=active
的笔记；follow_redirect 链式跟随 merged/superseded 并防环。
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from researchwiki.loop.notes import scan_max_note_id
from researchwiki.tools.fs import atomic_write_text
from researchwiki.wiki.frontmatter import NoteMeta, SourceRef, dump, parse

CONFLICT_ID_RE = re.compile(r"^C-(\d+)$")


def source_snapshot_path(sources_root: str | Path, url: str, content_hash: str) -> Path:
    """纯函数：{url, content_hash} → sources 快照正文路径（与 tools.fetch 落盘布局对齐）。

    不校验文件是否存在——笔记先于快照被清洗/迁移时返回的路径可能悬空。
    """
    url_key = hashlib.sha1(url.encode("utf-8")).hexdigest()
    return Path(sources_root) / url_key / content_hash / "content.md"


@dataclass
class Note:
    """一条笔记：元数据（NoteMeta）+ 正文 + 落盘路径。"""

    id: str
    title: str
    body: str
    meta: NoteMeta
    path: Path | None = None

    @property
    def status(self) -> str:
        return self.meta.status

    @property
    def entities(self) -> list[str]:
        return self.meta.entities

    @property
    def confidence(self) -> str:
        return self.meta.confidence

    @property
    def volatility(self) -> str:
        return self.meta.volatility


@dataclass
class Page:
    """聚合页：slug 稳定，title/body 可迭代更新。"""

    id: str
    title: str
    body: str
    created: str
    updated: str


@dataclass
class Conflict:
    """冲突台账条目：双方证据（claim_a/claim_b 为任意证据 dict）+ 裁决。"""

    id: str
    question: str
    claim_a: dict[str, object]
    claim_b: dict[str, object]
    status: str  # open | resolved
    resolution: dict[str, object] = field(default_factory=dict)
    created: str = ""
    resolved_at: str = ""


class WikiStore:
    """wiki 三层存储的门面：notes / pages / conflicts / entities。"""

    def __init__(self, root: str | Path = "wiki-data") -> None:
        self.root = Path(root)
        self.notes_dir = self.root / "notes"
        self.pages_dir = self.root / "pages"
        self.conflicts_dir = self.root / "conflicts"
        self.sources_dir = self.root / "sources"

    # ---- notes ----------------------------------------------------------

    def _scan_conflict_max_id(self) -> int:
        if not self.conflicts_dir.is_dir():
            return 0
        ids = (
            int(m.group(1))
            for p in self.conflicts_dir.iterdir()
            if (m := CONFLICT_ID_RE.match(p.stem))
        )
        return max(ids, default=0)

    def next_note_id(self) -> str:
        """下一个笔记编号（续接既有最大值，跨 loop 层兼容）。"""
        return f"N-{scan_max_note_id(self.notes_dir) + 1:04d}"

    def save_note(
        self,
        body: str,
        *,
        note_id: str | None = None,
        title: str = "",
        entities: list[str] | None = None,
        confidence: str = "medium",
        status: str = "active",
        redirect_to: str | None = None,
        superseded_by: str | None = None,
        volatility: str = "stable",
        observed_at: str | None = None,
        reviewed_at: str | None = None,
        trace_id: str = "",
        sources: list[SourceRef] | None = None,
        extra: Mapping[str, object] | None = None,
        created: str | None = None,
    ) -> Note:
        """写入/覆写一条笔记（frontmatter 全字段 + 原子落盘），返回 Note。

        note_id 缺省时自动分配（续接编号）；传入已有 id 即原地更新，
        created 会保留原值（除非显式传入）。
        """
        meta = NoteMeta(
            id=note_id or self.next_note_id(),
            title=title,
            entities=list(entities or []),
            confidence=confidence,
            status=status,
            redirect_to=redirect_to,
            superseded_by=superseded_by,
            volatility=volatility,
            observed_at=observed_at,
            reviewed_at=reviewed_at,
            created=created or datetime.now(UTC).isoformat(timespec="seconds"),
            trace_id=trace_id,
            sources=list(sources or []),
            extra=dict(extra or {}),
        )
        return self._write_note(meta, body)

    def _write_note(self, meta: NoteMeta, body: str) -> Note:
        path = self.notes_dir / f"{meta.id}.md"
        atomic_write_text(path, dump(meta.to_dict(), body))
        return Note(id=meta.id, title=meta.title, body=body, meta=meta, path=path)

    def get_note(self, note_id: str) -> Note | None:
        """按 id 读笔记（兼容 loop 层轻量 frontmatter）；不存在返回 None。"""
        path = self.notes_dir / f"{note_id}.md"
        if not path.is_file():
            return None
        meta_dict, body = parse(path.read_text(encoding="utf-8"))
        meta = NoteMeta.from_dict(meta_dict)
        meta.id = meta.id or note_id
        return Note(id=meta.id, title=meta.title, body=body, meta=meta, path=path)

    def list_notes(self, *, status: str | None = "active") -> list[Note]:
        """列出笔记；status="active"（默认）只回 active，status=None 回全部。"""
        notes: list[Note] = []
        if not self.notes_dir.is_dir():
            return notes
        for path in sorted(self.notes_dir.glob("N-*.md")):
            meta_dict, body = parse(path.read_text(encoding="utf-8"))
            meta = NoteMeta.from_dict(meta_dict)
            if not meta.id:
                meta.id = path.stem
            if status is not None and meta.status != status:
                continue
            notes.append(Note(id=meta.id, title=meta.title, body=body, meta=meta, path=path))
        return notes

    def follow_redirect(self, note_id: str) -> Note | None:
        """链式跟随 merged → redirect_to / superseded → superseded_by，返回最终 active 笔记。

        环路抛 ValueError；起点不存在或链条断裂返回 None。
        """
        visited: set[str] = set()
        current_id = note_id
        while True:
            if current_id in visited:
                raise ValueError(f"redirect cycle detected at {current_id!r} (start: {note_id})")
            visited.add(current_id)
            note = self.get_note(current_id)
            if note is None:
                return None
            if note.meta.status == "active":
                return note
            if note.meta.status == "merged":
                target = note.meta.redirect_to
            elif note.meta.status == "superseded":
                target = note.meta.superseded_by
            else:
                return None
            if not target:
                return None
            current_id = target

    def note_snapshot_paths(self, note: Note) -> list[Path]:
        """笔记 frontmatter.sources → sources 快照正文路径（不校验存在）。"""
        return [
            source_snapshot_path(self.sources_dir, s.url, s.content_hash) for s in note.meta.sources
        ]

    # ---- pages ----------------------------------------------------------

    def save_page(self, slug: str, title: str, body: str) -> Page:
        """写入/更新聚合页（upsert；更新时保留 created、刷新 updated）。"""
        existing = self.get_page(slug)
        now = datetime.now(UTC).isoformat(timespec="seconds")
        page = Page(
            id=slug,
            title=title,
            body=body,
            created=existing.created if existing else now,
            updated=now,
        )
        meta = {
            "id": page.id,
            "title": page.title,
            "created": page.created,
            "updated": page.updated,
        }
        atomic_write_text(self.pages_dir / f"{slug}.md", dump(meta, body))
        return page

    def get_page(self, slug: str) -> Page | None:
        path = self.pages_dir / f"{slug}.md"
        if not path.is_file():
            return None
        meta, body = parse(path.read_text(encoding="utf-8"))
        return Page(
            id=str(meta.get("id") or slug),
            title=str(meta.get("title") or ""),
            body=body,
            created=str(meta.get("created") or ""),
            updated=str(meta.get("updated") or ""),
        )

    def list_pages(self) -> list[Page]:
        if not self.pages_dir.is_dir():
            return []
        pages = []
        for slug_path in sorted(self.pages_dir.glob("*.md")):
            page = self.get_page(slug_path.stem)
            if page is not None:
                pages.append(page)
        return pages

    # ---- conflicts ------------------------------------------------------

    def next_conflict_id(self) -> str:
        return f"C-{self._scan_conflict_max_id() + 1:04d}"

    def save_conflict(
        self,
        question: str,
        claim_a: Mapping[str, object],
        claim_b: Mapping[str, object],
        *,
        conflict_id: str | None = None,
        trace_id: str = "",
    ) -> Conflict:
        """登记一条冲突（status=open），双方证据任意结构（note_id + 摘录 + 来源等）。"""
        conflict = Conflict(
            id=conflict_id or self.next_conflict_id(),
            question=question,
            claim_a=dict(claim_a),
            claim_b=dict(claim_b),
            status="open",
            created=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        self._write_conflict(conflict, trace_id=trace_id)
        return conflict

    def _write_conflict(self, conflict: Conflict, *, trace_id: str = "") -> None:
        meta: dict[str, object] = {
            "id": conflict.id,
            "status": conflict.status,
            "question": conflict.question,
            "claim_a": conflict.claim_a,
            "claim_b": conflict.claim_b,
            "created": conflict.created,
        }
        if trace_id:
            meta["trace_id"] = trace_id
        if conflict.status == "resolved":
            meta["resolution"] = conflict.resolution
            meta["resolved_at"] = conflict.resolved_at
        atomic_write_text(self.conflicts_dir / f"{conflict.id}.md", dump(meta, ""))

    def get_conflict(self, conflict_id: str) -> Conflict | None:
        path = self.conflicts_dir / f"{conflict_id}.md"
        if not path.is_file():
            return None
        meta, _ = parse(path.read_text(encoding="utf-8"))
        return Conflict(
            id=str(meta.get("id") or conflict_id),
            question=str(meta.get("question") or ""),
            claim_a=dict(meta.get("claim_a") or {}),
            claim_b=dict(meta.get("claim_b") or {}),
            status=str(meta.get("status") or "open"),
            resolution=dict(meta.get("resolution") or {}),
            created=str(meta.get("created") or ""),
            resolved_at=str(meta.get("resolved_at") or ""),
        )

    def list_conflicts(self, *, status: str | None = "open") -> list[Conflict]:
        """列出冲突；status="open"（默认）只回未裁决，status=None 回全部。"""
        if not self.conflicts_dir.is_dir():
            return []
        out: list[Conflict] = []
        for path in sorted(self.conflicts_dir.glob("C-*.md")):
            conflict = self.get_conflict(path.stem)
            if conflict is None:
                continue
            if status is not None and conflict.status != status:
                continue
            out.append(conflict)
        return out

    def resolve_conflict(self, conflict_id: str, *, verdict: str, resolved_with: str) -> Conflict:
        """裁决冲突：verdict 为结论说明，resolved_with 为采纳的笔记/页面 id。"""
        conflict = self.get_conflict(conflict_id)
        if conflict is None:
            raise FileNotFoundError(f"conflict not found: {conflict_id}")
        conflict.status = "resolved"
        conflict.resolution = {"verdict": verdict, "resolved_with": resolved_with}
        conflict.resolved_at = datetime.now(UTC).isoformat(timespec="seconds")
        self._write_conflict(conflict)
        return conflict
