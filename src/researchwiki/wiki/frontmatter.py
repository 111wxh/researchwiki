"""YAML frontmatter 读写与类型化笔记元数据。

字段一次定齐（防后续迁移）：id / title / entities / confidence / status /
redirect_to / superseded_by / volatility / observed_at / reviewed_at /
valid_from / valid_until / source_changed_at / created / trace_id / sources /
kind / importance / tombstone。

设计约定：
- 未知字段一律收进 ``extra``，round-trip 不丢数据（向后兼容未来扩展）。
- loop 层写的轻量 frontmatter（只有 id/entities/confidence/created/trace_id）
  必须能直接读：缺省字段走 dataclass 默认值。
- YAML 时间戳会被 PyYAML 解析成 datetime，统一转回 ISO 字符串，保持
  "frontmatter 里只有标量与列表"的简单心智模型。
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from datetime import date, datetime
from typing import Any

import yaml

CONFIDENCE_LEVELS = ("high", "medium", "low")
STATUS_LEVELS = ("active", "merged", "superseded")
VOLATILITY_LEVELS = ("stable", "drifting", "volatile")
# 记忆载体类型：knowledge（世界知识）/ user（用户画像与偏好）/ experience（经验教训）
KIND_LEVELS = ("knowledge", "user", "experience")
# importance 的合法区间（0.0–1.0），越界视为未填写
IMPORTANCE_MIN = 0.0
IMPORTANCE_MAX = 1.0

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
    - kind: knowledge（世界知识）| user（用户画像/偏好）| experience（经验教训）。
    - importance: 0.0–1.0 的主观重要性；None 表示未评估（序列化时省略）。
    - tombstone: True = 本笔记是**墓碑**（``memory_invalidate`` 生成的失效裁定
      记录），False = 普通笔记。墓碑仍是 status=active 的笔记（"superseded 必须
      沿链可达 active"的不变量不变），但语义上属**审计记录**而非当前知识：
      检索（``SearchIndex.search``）与 Prior 注入默认按"该标记为真 → 排除"过滤，
      要看到它需显式 ``include_tombstones=True``；``memory_read`` / ``timeline``
      / ``lint`` 照常可见（可读性不受影响）。False 是缺省值，序列化时省略，
      不往每篇普通笔记的 frontmatter 里写 ``tombstone: false``。
    - observed_at: 断言的观察时间（ISO 字符串）；缺省时新鲜度回退 created。
    - valid_from / valid_until: 断言的有效期窗口（ISO 字符串，均可缺省）。
      valid_until 为 None = 未声明显式失效时间，靠在 freshness.py 里按
      volatility 衰减判断；两者非 ISO 时不报错，原样保留、由计算侧按
      "不可解析"处理（见 wiki/freshness.py 规则 2）。
    - source_changed_at: 最近一次"该笔记引用的来源内容已变化"的检测时间
      （ISO 字符串，None = 未检测到变化）。由 ``WikiStore.mark_source_changed``
      写入（detector 提供新 content_hash，本字段只记"何时发现"）；它**不改**
      sources 里的 content_hash——证据的换版要等复核后由 supersede/update 决定。
      freshness 计算侧据此把记忆降级为至少 review_due（见 freshness.py 规则 5）。
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
    kind: str = "knowledge"
    importance: float | None = None
    tombstone: bool = False
    observed_at: str | None = None
    reviewed_at: str | None = None
    valid_from: str | None = None
    valid_until: str | None = None
    source_changed_at: str | None = None
    created: str = ""
    trace_id: str = ""
    sources: list[SourceRef] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """序列化回 frontmatter dict：None 的可选字段与 False 的 tombstone 省略，extra 平铺在后。"""
        out: dict[str, Any] = {}
        for f in fields(self):
            if f.name == "extra":
                continue
            value = getattr(self, f.name)
            optional = (
                "redirect_to",
                "superseded_by",
                "observed_at",
                "reviewed_at",
                "valid_from",
                "valid_until",
                "source_changed_at",
                "importance",
            )
            if value is None and f.name in optional:
                continue
            if f.name == "tombstone" and not value:
                continue  # False 是缺省：不把 tombstone: false 写进每篇普通笔记
            if f.name == "sources":
                out[f.name] = [{"url": s.url, "content_hash": s.content_hash} for s in self.sources]
            else:
                out[f.name] = value
        out.update(self.extra)
        return out

    def replace(self, **changes: Any) -> NoteMeta:
        """返回一份"只改了指定字段"的新 meta（其余字段逐字段原样保留）。

        存在的理由（P2-F 修复轮 I-2）：``WikiStore.save_note`` 的参数默认值会
        **静默丢字段**——任何"以既有 meta 重建笔记"的路径只要漏传一个参数，
        该字段就被悄悄重置（kind / importance / tombstone 都栽过：ingest merge
        把墓碑变回可召回的正常笔记、formation 标注再清一次）。把重建点改成
        ``meta.replace(...) + store.save_meta(...)`` 之后，"漏传"在语法上就不
        存在了：没点名的字段必然保留。

        实现是 ``dataclasses.replace`` 语义（浅拷贝 + 覆盖点名字段），但
        **列表/字典字段各拷一份**（entities / sources 元素与 extra 的值仍共享），
        避免调用方在原地改新 meta 的列表时把旧 meta 一起改了；未知字段名抛
        ``TypeError``（与 dataclasses.replace 一致，早失败好过静默忽略）。
        """
        copied = dataclasses.replace(self, **changes)
        copied.entities = list(copied.entities)
        copied.sources = list(copied.sources)
        copied.extra = dict(copied.extra)
        return copied

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> NoteMeta:
        """从 frontmatter dict 构造；宽容处理 loop 层轻量格式与类型漂移。"""
        known = {f.name for f in fields(cls)} - {"extra"}
        conf = _enum_str(data.get("confidence"), CONFIDENCE_LEVELS, "medium")
        status = _enum_str(data.get("status"), STATUS_LEVELS, "active")
        volatility = _enum_str(data.get("volatility"), VOLATILITY_LEVELS, "stable")
        kind = _enum_str(data.get("kind"), KIND_LEVELS, "knowledge")
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
            kind=kind,
            importance=_importance(data.get("importance")),
            tombstone=_flag(data.get("tombstone")),
            observed_at=_scalar_str(data.get("observed_at")),
            reviewed_at=_scalar_str(data.get("reviewed_at")),
            # 有效期窗口：宽容透传（非 ISO 字符串原样保留，由 freshness 计算侧按不可解析处理）
            valid_from=_scalar_str(data.get("valid_from")),
            valid_until=_scalar_str(data.get("valid_until")),
            # 来源变化检测时间：与有效期字段同口径（宽容透传，非 ISO 由计算侧按未声明处理）
            source_changed_at=_scalar_str(data.get("source_changed_at")),
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


def _importance(value: Any) -> float | None:
    """importance 宽容解析：仅认数值（bool 除外）且落在 [0.0, 1.0]，否则 None。

    注意 _scalar_str 会把数字转成字符串，所以这里必须在原始值上判断类型；
    非法值（字符串 / bool / 越界）一律视为"未评估"→ None，to_dict 随即省略。
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    if IMPORTANCE_MIN <= number <= IMPORTANCE_MAX:
        return number
    return None


def _flag(value: Any) -> bool:
    """布尔标记字段的宽容解析（tombstone；手工编辑 / 旧数据 / YAML 类型漂移都能读）。

    真值集合：``True``、非零数值、``"true"/"yes"/"on"/"1"``（忽略大小写与两侧空白）；
    其余（``None`` / False / 0 / 空串 / 无法识别的字符串）一律 False。默认 False 是
    "普通笔记"，所以无法识别时**不做**墓碑过滤——失败的代价只是多召回一条审计
    记录，比"正常记忆因为标记没解析出来就从检索里消失"安全得多。
    """
    if value is None or isinstance(value, bool):
        return bool(value)
    if isinstance(value, int | float):
        return value != 0
    return str(value).strip().lower() in ("true", "yes", "on", "1")
