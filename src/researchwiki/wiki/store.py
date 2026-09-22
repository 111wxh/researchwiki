"""wiki 存储层：原子笔记 notes/、聚合页 pages/、冲突台账 conflicts/。

目录布局（root 即 wiki-data，与 loop 层共用）：
- notes/N-XXXX.md      原子笔记，编号续接既有最大值（复用 loop.notes 的扫描器，
                       保证与 loop 层 NoteStore 写入的轻量笔记无缝续号、互读兼容）
- pages/{slug}.md      聚合页（人工/蒸馏整理的主题页）
- conflicts/C-XXXX.md  冲突台账：同一实体的矛盾断言双方证据 + 裁决记录

所有写盘走 atomic_write_text（tmp + os.replace）；列表默认只返回 status=active
的笔记；follow_redirect 链式跟随 merged/superseded 并防环。

来源溯源（P2-B）：``notes_by_source`` / ``notes_depending_on`` / ``missing_snapshots``
三个纯读查询回答"这条记忆的证据来自哪个快照、快照是否还在"；``mark_source_changed``
是唯一的写侧动作——只给受影响的 active 笔记打 ``source_changed_at`` 时间戳（不改
sources 里的哈希、不抓网络），复核判定交给 wiki/freshness.py。
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

# mark_source_changed 的去重记账键：``{url: 最近一次为该 URL 标记过的 new_content_hash}``。
# 存在 note extra 里（不新增类型化 frontmatter 字段，与 formation_reason /
# update_reasons 同处 extra，round-trip 不丢）；**按 URL 分槽**——笔记常引用多个
# 来源，单槽会被不同 URL 的标记互相覆盖，导致下一周期把已记过的变化判成"新变化"
# 再推时间戳（详见该方法的 docstring）。用途与兼容语义见 mark_source_changed。
SOURCE_CHANGED_HASH_KEY = "source_changed_hash"


def source_snapshot_path(sources_root: str | Path, url: str, content_hash: str) -> Path:
    """纯函数：{url, content_hash} → sources 快照正文路径（与 tools.fetch 落盘布局对齐）。

    不校验文件是否存在——笔记先于快照被清洗/迁移时返回的路径可能悬空。
    """
    url_key = hashlib.sha1(url.encode("utf-8")).hexdigest()
    return Path(sources_root) / url_key / content_hash / "content.md"


def _references_url(note: Note, url: str) -> bool:
    """笔记的 sources 里是否记录了该 URL（精确字符串匹配，不做归一）。"""
    return any(s.url == url for s in note.meta.sources)


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

    @property
    def kind(self) -> str:
        return self.meta.kind

    @property
    def importance(self) -> float | None:
        return self.meta.importance

    # 时效字段（P2 freshness）：只读透传，与 volatility/kind/importance 同风格；
    # 写入仍走 meta（frontmatter）或 save_note 的 valid_from / valid_until 参数
    @property
    def valid_from(self) -> str | None:
        return self.meta.valid_from

    @property
    def valid_until(self) -> str | None:
        return self.meta.valid_until

    @property
    def source_changed_at(self) -> str | None:
        return self.meta.source_changed_at

    @property
    def tombstone(self) -> bool:
        """是否墓碑笔记（审计记录，默认不参与检索与 Prior 注入；见 NoteMeta.tombstone）。"""
        return self.meta.tombstone


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
        kind: str = "knowledge",
        importance: float | None = None,
        tombstone: bool = False,
        observed_at: str | None = None,
        reviewed_at: str | None = None,
        valid_from: str | None = None,
        valid_until: str | None = None,
        source_changed_at: str | None = None,
        trace_id: str = "",
        sources: list[SourceRef] | None = None,
        extra: Mapping[str, object] | None = None,
        created: str | None = None,
    ) -> Note:
        """写入/覆写一条笔记（frontmatter 全字段 + 原子落盘），返回 Note。

        note_id 缺省时自动分配（续接编号）；传入已有 id 即原地更新，
        created 会保留原值（除非显式传入）。

        ``tombstone`` 是墓碑标记（P2-F 裁定一）：True 表示这条笔记是"已失效记忆"
        的审计记录（``memory_invalidate`` 生成），检索与 Prior 注入默认排除它。

        **重建既有笔记别用本方法**：这里的参数默认值会静默丢字段（kind /
        importance / tombstone 都栽过）。要"只改几个字段、其余保留"，用
        ``meta.replace(改动的字段=...)`` 拿到新 meta 再交给 ``save_meta``——
        本方法只适合**新建**（字段由调用方全新给出）。
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
            kind=kind,
            importance=importance,
            tombstone=bool(tombstone),
            observed_at=observed_at,
            reviewed_at=reviewed_at,
            valid_from=valid_from,
            valid_until=valid_until,
            source_changed_at=source_changed_at,
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

    def save_meta(self, meta: NoteMeta, body: str) -> Note:
        """以**给定的完整 meta** 覆写一条笔记（不做字段补默认、不做归一）。

        **重建既有笔记时的推荐入口**（P2-F 修复轮 I-2）：调用方先用
        ``meta.replace(改动的字段=...)`` 得到新 meta（未点名的字段逐字段保留），
        再交给本方法落盘——这样"漏传字段"在语法上就不存在了。``save_note`` 的
        参数默认值会静默丢字段，是 ingest merge / formation 标注 / source_changed
        标记把 kind / importance / tombstone 悄悄清掉的共同根因。

        ``meta.id`` 决定落盘文件名（不存在即新建）；元数据按 ``NoteMeta.to_dict``
        序列化，未知字段仍在 ``extra`` 里 round-trip 不丢。事实校验（title 非空、
        枚举白名单等）是**调用方**的职责：本方法不猜调用方意图。
        """
        return self._write_note(meta, body)

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

    # ---- 来源溯源（P2-B：证据来自哪个快照 / 那个快照是否还在）--------------
    #
    # 三个查询都是纯读、零副作用、不做 URL 归一（url 按字符串精确匹配，
    # 与 frontmatter 里记录的写法逐字符一致——调用方要拿 fetch 时的原 URL）。
    # 它们只回答"谁引用了什么"，不判断该不该复核（判定在 freshness.py）。

    def notes_by_source(self, url: str) -> list[Note]:
        """所有（含 merged/superseded）引用了该 URL 的笔记，按 note_id 升序。

        status=None 全量扫描是刻意的：证据链审计要能看到已被合并/废止的历史
        版本（它们同样引用过该来源），只看 active 会漏掉"这条 URL 被谁用过"。
        """
        return [n for n in self.list_notes(status=None) if _references_url(n, url)]

    def notes_depending_on(self, url: str, content_hash: str) -> list[Note]:
        """精确依赖某快照版本的笔记（url + content_hash 双匹配），按 note_id 升序。

        与 ``notes_by_source`` 的差别：同一个 URL 的不同快照版本是不同证据，
        换版后只有旧 ``content_hash`` 的那批笔记需要复核。
        """
        return [
            n
            for n in self.list_notes(status=None)
            if any(s.url == url and s.content_hash == content_hash for s in n.meta.sources)
        ]

    def missing_snapshots(self, *, status: str | None = "active") -> list[tuple[Note, Path]]:
        """笔记声明的快照文件不存在的证据（悬空证据），按 (note_id, path) 升序。

        - ``status`` 默认只看 active（现役记忆的悬空证据才需要处理）；
          ``status=None`` 检查全部（含历史版本）。
        - 空 content_hash 也一并报出：没有哈希就根本无法定位快照，
          与"快照文件已被清理"同属悬空（宁可多报，也不静默放过证据链断裂）。
        - 同一 (note, path) 只出现一次（笔记里重复写同一来源时不重复计数）。
        """
        out: list[tuple[Note, Path]] = []
        seen: set[tuple[str, Path]] = set()
        for note in self.list_notes(status=status):
            for path in self.note_snapshot_paths(note):
                if path.is_file():
                    continue
                key = (note.id, path)
                if key in seen:
                    continue
                seen.add(key)
                out.append((note, path))
        out.sort(key=lambda pair: (pair[0].id, str(pair[1])))
        return out

    def mark_source_changed(
        self, url: str, new_content_hash: str, *, now: str | None = None
    ) -> list[str]:
        """标记"引用了该 URL 的记忆，其来源内容已变化"；返回受影响的 note_id 列表。

        语义（P2-B 的来源变化最小落地，**零网络**——新 hash 由调用方提供，
        抓取属 P4 refresh 的职责）：

        - 只处理 **active** 笔记（merged/superseded 已退役，复核无意义）；
        - 命中条件：引用了该 URL **且** 记录里的 content_hash ≠ ``new_content_hash``
          **且** 该笔记还没有为这个 URL 的这个新哈希标记过（去重见下）；
        - 写入 ``source_changed_at = now``（缺省当前 UTC 秒级 ISO），**只记"何时
          发现变化"**，不改 sources 里的 content_hash——证据换版要等复核后由
          supersede / memory_update 决定（下一包），这样来源变化的检测与证据的
          修订是两个可分别审计的动作；
        - 同时把被检测到的 ``new_content_hash`` 记进
          ``extra["source_changed_hash"]``（不新增类型化 frontmatter 字段，与
          formation_reason / update_reasons 同处 extra，round-trip 不丢），
          形态是 ``{url: hash}`` 的映射；
        - 笔记的其余字段（含 reviewed_at / valid_* / 其余 extra 键）原样保留，
          只这两个写入项变化；返回值按 note_id 升序，便于调用方直接展示。

        **去重（幂等，按 URL 分槽）**：同一次变化反复检测是 no-op —— 若
        ``extra["source_changed_hash"][url] == new_content_hash``，直接跳过
        （返回空列表、时间戳不动）。没有这条记账时，例行检测（P4 refresh 每周期
        调用）会把 ``source_changed_at`` 一路往后推，于是"标记 → 复核（reviewed_at
        晚于 source_changed_at → 恢复 fresh）→ 下一周期又被标记推后"，
        已复核的笔记被永久钉在 review_due；记账后的行为：同一 URL 的同一次变化
        只标记一次，**哈希确实再次变化时**（该 URL 的新 new_content_hash）正常
        标记并推后时间戳。

        为什么按 URL 分槽而不是只记一个哈希：笔记引用多个来源是成熟 wiki 的常态，
        调用方按 URL 循环标记时，单槽会被不同 URL 互相覆盖（mark(url1,h1) 记 h1 →
        mark(url2,h2) 覆盖成 h2 → 下一周期 mark(url1,h1) 又被判成"新变化"），
        对这类多来源笔记，"永久 review_due"照旧发生。分槽后每条 URL 的记账互不
        干扰，槽位数上界 = 该笔记引用过的 URL 数（不随检测次数增长）。

        **兼容语义**：旧版本（P2-B 首版）记的是单值字符串，无法归属到具体 URL，
        读取时**一律忽略**（不当作任何 URL 的去重依据——宁可多标一次也不漏标），
        写入时用 ``{url: hash}`` 映射覆盖它（frontmatter 归一为新形态，旧值不再
        保留）。因此升级后首次标记某 URL 会重新推后一次时间戳，此后即按 URL 去重。
        退化情形：若整个记账键被外部清掉（含经 ``save_note`` 不带 extra 的重写），
        同一变化会被再标记一次——可接受，宁可多标一次也不漏标。
        """
        stamp = now or datetime.now(UTC).isoformat(timespec="seconds")
        affected: list[str] = []
        for note in self.list_notes(status="active"):
            if not _references_url(note, url):
                continue
            if all(
                s.content_hash == new_content_hash
                for s in note.meta.sources
                if s.url == url
            ):
                continue  # 记录哈希与新哈希一致：证据未变
            meta = note.meta
            extra = dict(meta.extra)
            recorded = extra.get(SOURCE_CHANGED_HASH_KEY)
            if isinstance(recorded, Mapping) and recorded.get(url) == new_content_hash:
                continue  # 该 URL 的同一次变化已标记过：no-op（防"永久钉在 review_due"）
            # 旧形态（单值字符串）或其它非映射值：忽略，只保留可归属的映射槽
            slots: dict[str, str] = {}
            if isinstance(recorded, Mapping):
                slots = {str(k): str(v) for k, v in recorded.items()}
            slots[url] = new_content_hash
            extra[SOURCE_CHANGED_HASH_KEY] = slots
            # 只改两个字段，其余逐字段保留（replace + save_meta，见 NoteMeta.replace：
            # 旧写法逐参数透传 save_note，漏传一个就静默丢一个字段——本方法曾漏过
            # tombstone / kind / importance 这一组"记忆语义"字段）。
            self.save_meta(meta.replace(source_changed_at=stamp, extra=extra), note.body)
            affected.append(note.id)
        return sorted(affected)

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
