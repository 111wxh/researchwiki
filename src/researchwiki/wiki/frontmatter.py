"""YAML frontmatter 读写与类型化笔记元数据。

字段一次定齐（防后续迁移）：id / title / entities / confidence / status /
redirect_to / superseded_by / volatility / observed_at / reviewed_at /
created / trace_id / sources。

设计约定：
- 未知字段一律收进 ``extra``，round-trip 不丢数据（向后兼容未来扩展）。
- loop 层写的轻量 frontmatter（只有 id/entities/confidence/created/trace_id）
  必须能直接读：缺省字段走 dataclass 默认值。
- YAML 时间戳会被 PyYAML 解析成 datetime，统一转回 ISO 字符串，保持
  "frontmatter 里只有标量与列表"的简单心智模型。
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from datetime import date, datetime
from typing import Any

import yaml

CONFIDENCE_LEVELS = ("high", "medium", "low")
STATUS_LEVELS = ("active", "merged", "superseded")
VOLATILITY_LEVELS = ("stable", "drifting", "volatile")

# 匹配文件头部的第一个 frontmatter 块：--- \n ... \n --- \n 正文
_FRONTMATTER_RE = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*\r?\n?(.*)\Z", re.DOTALL)


def parse(text: str) -> tuple[dict[str, Any], str]:
    """解析 ``--- 包裹的 YAML frontmatter + 正文``。

    无 frontmatter 时返回 ``({}, 原文)``；frontmatter 块为空时 meta 为空 dict。
    frontmatter 不是映射（畸形文件）时抛 ValueError。
    """
    m = _FRONTMATTER_RE.match(text)
    if m is None:
        return {}, text
    meta = yaml.safe_load(m.group(1))
    if meta is None:
        meta = {}
    if not isinstance(meta, dict):
        raise ValueError(f"frontmatter 必须是 YAML 映射，得到 {type(meta).__name__}")
    return meta, m.group(2)


def dump(meta: Mapping[str, Any], body: str) -> str:
    """把 meta dict + 正文序列化回 frontmatter 文本（yaml.safe_dump，保留中文）。"""
    text = yaml.safe_dump(dict(meta), allow_unicode=True, sort_keys=False, default_flow_style=False)
    return f"---\n{text}---\n{body}"


@dataclass
class SourceRef:
    """笔记引用的网页快照定位：url + 内容哈希，对应 sources/{sha1(url)}/{content_hash}/。"""

    url: str
    content_hash: str


@dataclass
class NoteMeta:
    """类型化笔记元数据（一次定齐，后续阶段不再迁移格式）。

    - status: active | merged | superseded；merged 必须给 redirect_to，
      superseded 必须给 superseded_by（本层不强校验，由调用方保证）。
    - volatility: stable（不衰减）| drifting（90 天半衰）| volatile（30 天半衰）。
    - observed_at: 断言的观察时间（ISO 字符串）；缺省时新鲜度回退 created。
    - extra: 未知字段原样保留，round-trip 不丢。
    """

    id: str
    title: str = ""
    entities: list[str] = field(default_factory=list)
    confidence: str = "medium"
    status: str = "active"
    redirect_to: str | None = None
    superseded_by: str | None = None
    volatility: str = "stable"
    observed_at: str | None = None
    reviewed_at: str | None = None
    created: str = ""
    trace_id: str = ""
    sources: list[SourceRef] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """序列化回 frontmatter dict；None 的可选字段省略，extra 平铺在最后。"""
        out: dict[str, Any] = {}
        for f in fields(self):
            if f.name == "extra":
                continue
            value = getattr(self, f.name)
            optional = ("redirect_to", "superseded_by", "observed_at", "reviewed_at")
            if value is None and f.name in optional:
                continue
            if f.name == "sources":
                out[f.name] = [{"url": s.url, "content_hash": s.content_hash} for s in self.sources]
            else:
                out[f.name] = value
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> NoteMeta:
        """从 frontmatter dict 构造；宽容处理 loop 层轻量格式与类型漂移。"""
        known = {f.name for f in fields(cls)} - {"extra"}
        conf = _enum_str(data.get("confidence"), CONFIDENCE_LEVELS, "medium")
        status = _enum_str(data.get("status"), STATUS_LEVELS, "active")
        volatility = _enum_str(data.get("volatility"), VOLATILITY_LEVELS, "stable")
        sources: list[SourceRef] = []
        for item in data.get("sources") or []:
            if isinstance(item, Mapping):
                sources.append(
                    SourceRef(
                        url=str(item.get("url") or ""),
                        content_hash=str(item.get("content_hash") or ""),
                    )
                )
        extra = {k: v for k, v in data.items() if k not in known}
        return cls(
            id=_scalar_str(data.get("id")) or "",
            title=_scalar_str(data.get("title")) or "",
            entities=[str(e) for e in (data.get("entities") or []) if str(e)],
            confidence=conf,
            status=status,
            redirect_to=_scalar_str(data.get("redirect_to")),
            superseded_by=_scalar_str(data.get("superseded_by")),
            volatility=volatility,
            observed_at=_scalar_str(data.get("observed_at")),
            reviewed_at=_scalar_str(data.get("reviewed_at")),
            created=_scalar_str(data.get("created")) or "",
            trace_id=_scalar_str(data.get("trace_id")) or "",
            sources=sources,
            extra=extra,
        )


def _scalar_str(value: Any) -> str | None:
    """标量转字符串；None 原样返回；datetime/date 转 ISO 字符串。"""
    if value is None:
        return None
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, bool | int | float):
        return str(value)
    return str(value)


def _enum_str(value: Any, allowed: tuple[str, ...], default: str) -> str:
    """枚举字段归一：小写、白名单内才认，否则回默认值。"""
    text = _scalar_str(value)
    if text is None:
        return default
    lowered = text.strip().lower()
    return lowered if lowered in allowed else default
