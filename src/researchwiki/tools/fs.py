"""沙箱文件读写：所有路径必须落在 wiki_data_root 内，越界抛 SandboxError。

约定：path 按进程 cwd 解析（或直接传绝对路径），解析后必须位于 root 之内。
写操作统一 tmp + os.replace 原子落盘，进程崩溃不会留下半个文件。
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

DEFAULT_WIKI_DATA_ROOT = Path("wiki-data")


class SandboxError(PermissionError):
    """路径逃逸出 wiki_data_root 沙箱。"""


def _resolve_root(root: str | Path | None) -> Path:
    return Path(root).resolve() if root is not None else DEFAULT_WIKI_DATA_ROOT.resolve()


def _ensure_inside(path: str | Path, root: Path) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(root):
        raise SandboxError(f"path escapes wiki-data sandbox: {path!r} -> {resolved} (root: {root})")
    return resolved


def atomic_write_text(path: str | Path, content: str) -> None:
    """tmp + os.replace 原子写（tmp 与目标同目录，保证同一文件系统）。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


def safe_read(path: str | Path, *, root: str | Path | None = None) -> str:
    """沙箱内读文件；路径越界抛 SandboxError，文件缺失抛 FileNotFoundError。"""
    resolved = _ensure_inside(path, _resolve_root(root))
    return resolved.read_text(encoding="utf-8")


def safe_write(path: str | Path, content: str, *, root: str | Path | None = None) -> Path:
    """沙箱内原子写文件，自动创建父目录；返回写入路径。"""
    resolved = _ensure_inside(path, _resolve_root(root))
    atomic_write_text(resolved, content)
    return resolved


def safe_list(dir_path: str | Path, *, root: str | Path | None = None) -> list[Path]:
    """沙箱内列出目录条目（按名字排序）；目录缺失抛 FileNotFoundError。"""
    resolved = _ensure_inside(dir_path, _resolve_root(root))
    return sorted(resolved.iterdir(), key=lambda p: p.name)
