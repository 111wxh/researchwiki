"""原子笔记落盘：编号从现有 wiki 最大编号 +1 递增，Markdown + 轻量 frontmatter。

阶段 2 的最小实现：笔记写 wiki-data/notes/N-XXXX.md，编号跨 run 持久（读目录取最大值）；
frontmatter 结构与阶段 3 的三层存储对齐，届时无需迁移格式。
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_NOTE_ID_RE = re.compile(r"^N-(\d+)$")


def scan_max_note_id(notes_dir: str | Path) -> int:
    """扫描 notes 目录下 N-<数字>.md 的最大编号；目录不存在或为空返回 0。"""
    root = Path(notes_dir)
    if not root.is_dir():
        return 0
    max_id = 0
    for p in root.iterdir():
        m = _NOTE_ID_RE.match(p.stem)
        if m and p.suffix == ".md":
            max_id = max(max_id, int(m.group(1)))
    return max_id


class NoteStore:
    """为一次 run 分配连续笔记编号并原子落盘；编号起点取自现有 wiki。"""

    def __init__(self, notes_dir: str | Path, *, trace_id: str = "") -> None:
        self.notes_dir = Path(notes_dir)
        self.trace_id = trace_id
        self._next = scan_max_note_id(notes_dir) + 1
        self._written = 0

    def progress(self) -> int:
        """本次 run 已写入的笔记条数（state.md 用）。"""
        return self._written

    def next_id(self) -> str:
        """分配下一个笔记 id（不落盘），如 N-0001、N-0002……编号超过 9999 自然加宽。"""
        note_id = f"N-{self._next:04d}"
        self._next += 1
        return note_id

    def save(self, note: dict[str, Any]) -> dict[str, Any]:
        """补全 id、写入 Markdown 文件，返回带 id 的完整笔记（data-note 载荷同构）。"""
        note_id = self.next_id()
        text = str(note.get("text") or "").strip()
        entities = list(note.get("entities") or [])
        confidence = str(note.get("confidence") or "medium")
        front = {
            "id": note_id,
            "entities": entities,
            "confidence": confidence,
            "created": datetime.now(UTC).isoformat(timespec="seconds"),
            "trace_id": self.trace_id,
        }
        body = (
            "---\n"
            + "\n".join(
                f"{k}: {json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v}"
                for k, v in front.items()
            )
            + "\n---\n\n"
            + text
            + "\n"
        )
        self.notes_dir.mkdir(parents=True, exist_ok=True)
        (self.notes_dir / f"{note_id}.md").write_text(body, encoding="utf-8")
        self._written += 1
        return {
            "id": note_id,
            "text": text,
            "entities": entities,
            "confidence": confidence,
        }
