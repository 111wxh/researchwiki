"""实体注册表：实体稳定 ID（slug）+ 别名归一，持久化为 wiki-data/entities/{slug}.md。

双链格式约定：``[[entity:glm-5-3|GLM-5.3]]`` —— 链接用稳定 ID（slug，不可变），
显示名可变。别名归一规则：精确匹配 name / aliases，忽略大小写与首尾空白。

frontmatter 字段：id / name / aliases / created（沿用 frontmatter.dump 序列化）。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from researchwiki.tools.fs import atomic_write_text
from researchwiki.wiki.frontmatter import dump, parse

# slug 保留 a-z0-9 与 CJK 统一表意文字，其余连续字符折叠成 '-'
_SLUG_KEEP_RE = re.compile(r"[^a-z0-9\u4e00-\u9fff]+")


def slugify(name: str) -> str:
    """实体名转稳定 slug：如 "GLM-5.3" → "glm-5-3"，中文名原样保留。"""
    slug = _SLUG_KEEP_RE.sub("-", name.strip().lower()).strip("-")
    if not slug:
        # 全部字符被剥光（如纯符号名）时用哈希兜底，保证 id 唯一可写
        slug = "e-" + hashlib.sha1(name.strip().encode("utf-8")).hexdigest()[:8]
    return slug


def _norm(text: str) -> str:
    """别名归一键：去首尾空白 + casefold。"""
    return text.strip().casefold()


@dataclass
class Entity:
    """一个实体：稳定 ID（slug）+ 当前名 + 别名表。"""

    id: str
    name: str
    aliases: list[str] = field(default_factory=list)
    created: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "aliases": list(self.aliases),
            "created": self.created,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> Entity:
        return cls(
            id=str(data.get("id") or ""),
            name=str(data.get("name") or ""),
            aliases=[str(a) for a in (data.get("aliases") or []) if str(a)],
            created=str(data.get("created") or ""),
        )


class EntityRegistry:
    """实体注册表。

    内存索引（归一名 → Entity）惰性构建；进程外的文件改动需调用 reload()
    重新加载。get_or_create 的归并语义：
    - 名字或别名已存在 → 返回既有实体，并把新传入的别名并入（去重）；
    - 名字不存在但 slug 与既有实体冲突（如 "GLM-5.3" 与 "glm-5-3"）→
      把新名字并入既有实体的别名（同名实体天然归一）。
    """

    def __init__(self, root: str | Path = "wiki-data") -> None:
        self.entities_dir = Path(root) / "entities"
        self._index: dict[str, Entity] | None = None

    # ---- 索引管理 -------------------------------------------------------

    def reload(self) -> None:
        """丢弃内存索引，下次访问时从磁盘重建。"""
        self._index = None

    def _ensure_index(self) -> dict[str, Entity]:
        if self._index is None:
            self._index = self._build_index()
        return self._index

    def _build_index(self) -> dict[str, Entity]:
        index: dict[str, Entity] = {}
        if not self.entities_dir.is_dir():
            return index
        for path in sorted(self.entities_dir.glob("*.md")):
            meta, _ = parse(path.read_text(encoding="utf-8"))
            try:
                entity = Entity.from_dict(meta)
            except Exception:  # noqa: BLE001 -- 单个坏文件不拖垮整个注册表
                continue
            if entity.id:
                index[entity.id.casefold()] = entity
                index[_norm(entity.name)] = entity
                for alias in entity.aliases:
                    index[_norm(alias)] = entity
        return index

    def _put(self, entity: Entity) -> None:
        """写盘 + 更新内存索引（原子写，绝不留半个文件）。"""
        self.entities_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self.entities_dir / f"{entity.id}.md", dump(entity.to_dict(), ""))
        index = self._ensure_index()
        index[entity.id.casefold()] = entity
        index[_norm(entity.name)] = entity
        for alias in entity.aliases:
            index[_norm(alias)] = entity

    # ---- 公开 API -------------------------------------------------------

    def resolve(self, name: str) -> Entity | None:
        """按名字或别名解析实体；忽略大小写与首尾空白；未注册返回 None。"""
        return self._ensure_index().get(_norm(name))

    def get_or_create(self, name: str, aliases: list[str] | None = None) -> Entity:
        """解析或注册实体；已存在时把新别名并入既有实体并落盘。"""
        name = name.strip()
        existing = self.resolve(name)
        extra_aliases = [
            a.strip() for a in (aliases or []) if a.strip() and _norm(a) != _norm(name)
        ]
        if existing is not None:
            # 命中路径可能是"名字/别名"或"稳定 id"：后者的写法要补成别名，
            # 让 resolve() 纯靠名字归一也能命中；其余新别名照常并入。
            known = {_norm(existing.name), *(_norm(a) for a in existing.aliases)}
            changed = False
            for candidate in [name, *extra_aliases]:
                if _norm(candidate) not in known:
                    existing.aliases.append(candidate)
                    known.add(_norm(candidate))
                    changed = True
            if changed:
                self._put(existing)
            return existing

        # slug 冲突归并：文件已存在但名字没解析到，说明是新写法/新别名
        slug = slugify(name)
        collided = self._ensure_index().get(slug.casefold())
        if collided is not None:
            known = {_norm(collided.name), *(_norm(a) for a in collided.aliases)}
            for candidate in [name, *extra_aliases]:
                if _norm(candidate) not in known:
                    collided.aliases.append(candidate)
                    known.add(_norm(candidate))
            self._put(collided)
            return collided

        entity = Entity(
            id=slug,
            name=name,
            aliases=extra_aliases,
            created=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        self._put(entity)
        return entity

    def list_entities(self) -> list[Entity]:
        """全部实体，按稳定 ID 排序。"""
        entities = {e.id: e for e in self._ensure_index().values()}
        return [entities[eid] for eid in sorted(entities)]
