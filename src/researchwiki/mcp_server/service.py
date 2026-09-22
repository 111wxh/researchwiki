"""MCP 工具的 wiki 服务层：把 wiki 公开 API 包装成 JSON 友好的读写操作。

分层约定
--------
- 本模块只产出 JSON 可序列化的 dict：预期失败返回 ``{"ok": false, "error": {...}}``
  而不是抛异常——MCP 客户端拿到的是一条普通结果，不会看到协议错误或堆栈。
- ``server.py`` 里的 FastMCP 工具是薄适配层，函数体只有一行调用本模块。
  所以逻辑测试可以直接打服务层，协议层（参数 schema / 序列化）另用 in-process
  Client 覆盖，见 tests/test_mcp.py 的说明。

写保护三件套（wiki_write / memory_store）
------------------------------------------
1. frontmatter schema 校验：confidence / volatility / kind 白名单、title 非空、
   entities 必须是字符串列表、importance 必须落在 0.0–1.0……非法输入一律拒绝，
   并给出可读的中文原因（``NoteMeta.from_dict`` 是"宽容归一"语义，会把非法值
   静默改成默认值，所以这里必须显式校验，不能靠它兜底）。
2. 路径限制：笔记路径一律由 ``tools/fs`` 的沙箱函数读写（safe_read / safe_write），
   客户端传入的 note_id 先过白名单正则再拼路径；写入用的 id 由
   ``WikiStore.next_note_id()`` 机器生成，形态固定为 N-XXXX。修订/取代/失效
   复用库内既有 id 作为写路径分量，写前另做一次形态复核
   （``_assert_note_id_writable``：id 必须匹配白名单正则且与落盘文件名一致），
   库内污染 id（含 ``../`` 或与文件名不一致）一律拒绝写入。
3. 写前备份：把将被覆盖的原文件复制到 ``wiki-data/.backups/<UTC 时间戳>/``，
   新增（无原文件）时在该目录写一条 manifest 标记 action=created；
   备份任何一步失败都拒绝写入。

索引一致性
----------
写入后立即 ``SearchIndex.index_note``；检索前检查索引是否落后于 store——
**判据与 loop 路径共用** ``wiki.index.index_drift``（id 集合 / status / 索引指纹
三维），落后则整体重建一次。每次调用新建 SearchIndex（sqlite 连接不跨线程；
FastMCP 默认在线程池里跑同步工具）。

v1 的旧判据是"md 的 mtime 比 index.db 新，或索引条数与笔记数不符"，与 loop 路径
（``prior.ensure_index_fresh`` 的指纹判定）分叉：同一 store/index 状态两条路径会
给出不同答案，且任何"写 md 不写 index.db"的路径（``mark_source_changed`` 只改
frontmatter、``index_note`` 失败后的补偿等）都会让 MCP 路径必然判 stale 并白做一次
全量重建。P2-F 裁定二把这个判据抽成共享函数后，mtime 前置被删除：它只是"内容变过"
的弱代理（同秒改写漏判、只改元数据的原地改写完全看不出来），而共享判据要读的
``store.list_notes`` 本来就是 mtime 检查的必需步骤，删掉 mtime 不省任何 IO。

memory_* 记忆接口（P1-A）
------------------------
九个 memory_* 工具是"面向 Agent 的外置记忆"语义层：store（显式写入）/ search
（kind 过滤检索）/ recall（动态召回钩子位，MVP 为结构化透传）/ update（原地修订，
必须留 reason + 备份）/ supersede（新内容替代旧记忆，新旧 ID 沿链可达）/
invalidate（判定失效，自动生成墓碑笔记而非新增状态——"superseded 必须沿链可达
active"的不变量与状态机都留给 P2）/ timeline（版本史）/ conflicts（冲突台账）/
profile（User Memory 视图）。wiki_* 五工具保持原样作为兼容层。

墓碑的检索可见性（P2-F 裁定一）
------------------------------
``memory_invalidate`` 生成的墓碑带 ``tombstone: true`` 标记（类型化 frontmatter
字段 + ``note_meta.tombstone`` 列，进 ``note_index_hash`` 指纹）。它是**审计记录**
而非当前知识，所以 search / recall **默认排除**，要审计时显式
``include_tombstones=True``；``read`` / ``timeline`` / ``lint`` 照常可见。

证据链入口（P2 carried 项④）
---------------------------
supersede / update 接受可选 ``source_urls``（元素为 URL 字符串或
``{"url", "content_hash"}`` 映射，见 ``SourceInput``；校验与 ``write()`` 共用
``_normalize_sources``；**显式空列表视为非法参数**，省略才表示"不提供证据"）：
supersede 提供了来源 → 新笔记 sources=提供值（不继承旧来源），未提供 → sources
为空并在返回体标注 ``evidence: "none"``；update 是修订语义，来源按
``(url, content_hash)`` 去重后**追加**到既有 sources。两个工具的正确性都不依赖
调用方自觉：返回体的 ``evidence`` 字段就是"这条记忆有没有证据"的机器可读声明。

supersede 另有可选 ``confidence``（high/medium/low，缺省按"有来源 → medium、
两者都没给 → 继承旧值"），**从不为了新证据继承旧置信度**：显式给值即可表达
"有来源 + 具体事实要素 → high"（``wiki/formation.py`` 的确定性口径），
不必绕道 memory_store（那会留下两条 active 记忆且断掉 supersede 链）。

错误码词汇表（错误结构里的 ``error.code``）
------------------------------------------
invalid_argument 参数非法｜invalid_kind kind 不在 knowledge/user/experience 白名单｜
validation_failed frontmatter 校验未过｜not_found 笔记不存在｜
redirect_cycle redirect 成环｜broken_redirect redirect 断裂｜path_rejected 路径越界｜
backup_failed 写前备份失败（已拒绝写入）｜write_failed 落盘失败｜io_error 文件读写失败｜
internal 服务内部异常。
"""

from __future__ import annotations

import bisect
import functools
import json
import logging
import re
import sqlite3
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from researchwiki.tools.fs import SandboxError, safe_read, safe_write
from researchwiki.wiki.embeddings import get_embedding_provider
from researchwiki.wiki.entities import EntityRegistry
from researchwiki.wiki.frontmatter import (
    CONFIDENCE_LEVELS,
    IMPORTANCE_MAX,
    IMPORTANCE_MIN,
    KIND_LEVELS,
    VOLATILITY_LEVELS,
    NoteMeta,
    SourceRef,
    parse,
)
from researchwiki.wiki.index import SearchIndex, index_drift, wiki_settings
from researchwiki.wiki.store import Note, WikiStore

# 工具参数上限：防止客户端一次拉爆自己的上下文
MAX_K = 50
MAX_LIMIT = 200
MAX_BODY_CHARS = 200_000
BACKUP_DIR_NAME = ".backups"
# extra.update_reasons 的保留上限（终审 F6）：修订原因按序累积，但**只保留最近 N 条**
# ——无界追加会让 frontmatter 随修订次数无限膨胀（取舍同 store.mark_source_changed 的
# "上界 = 引用过的 URL 数"）。被裁掉的原因仍可从 `.backups/<时间戳>/` 的历史原稿追溯
# （每次修订前都备份），所以全量审计没有丢，只是不再随笔记长胖。与 loop 侧
# memory_update.MAX_RECORDED_KEYS 同值：两条写路径对同一字段保持同一口径。
MAX_UPDATE_REASONS = 50

logger = logging.getLogger(__name__)

# 客户端可传入的 note_id 白名单：字母开头，只含字母数字下划线连字符。
# 直接把 ".."、"a/b"、"a\\b"、绝对路径、".md" 之外的点号全部挡在拼路径之前。
NOTE_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
# WikiStore.next_note_id() 生成的 id 形态（机器生成，不接受客户端传入）
GENERATED_NOTE_ID_RE = re.compile(r"^N-\d{4,}$")

# 来源参数（sources / source_urls）的元素形态：URL 字符串，或
# {"url": ..., "content_hash": ...} 映射（content_hash 可省/可为 None）。
# server.py 的工具签名直接复用这个别名，保证 MCP schema 与校验一致。
SourceInput = str | Mapping[str, Any]


# ---- 错误结构 ---------------------------------------------------------------


class WikiToolError(Exception):
    """服务层的"预期失败"：由公共方法转换成结构化错误 dict，不外抛。"""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details

    def to_payload(self) -> dict[str, Any]:
        return error_payload(self.code, self.message, **self.details)


def error_payload(code: str, message: str, **details: Any) -> dict[str, Any]:
    """统一的错误返回结构：{"ok": false, "error": {code, message, details?}}。"""
    err: dict[str, Any] = {"code": code, "message": message}
    if details:
        err["details"] = details
    return {"ok": False, "error": err}


def _structured_errors(fn: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
    """把预期异常翻译成结构化错误；未预期异常留给上层（工具层再兜一层 internal）。"""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
        try:
            return fn(*args, **kwargs)
        except WikiToolError as exc:
            return exc.to_payload()
        except SandboxError as exc:  # 注意：SandboxError 是 PermissionError(=OSError) 子类
            return error_payload("path_rejected", f"路径越界，已拒绝：{exc}")
        except OSError as exc:
            return error_payload("io_error", f"文件读写失败（{type(exc).__name__}）：{exc}")

    return wrapper


# ---- 时间与 id 工具 ---------------------------------------------------------


def _parse_ts(text: str | None) -> datetime | None:
    """宽松解析 ISO 时间戳；无时区的按 UTC（与 index.freshness_factor 同约定）。"""
    if not text:
        return None
    raw = text.strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def normalize_note_id(raw: str) -> str:
    """归一客户端传来的 note_id：去空白、去 "wiki-data/"/"notes/" 前缀、去 .md 后缀。

    归一后必须是白名单形态（字母开头，仅字母数字下划线连字符），否则拒绝——
    这一步在拼路径之前完成，`../`、`notes/../x`、`../../etc/passwd` 都会在此被挡掉。
    """
    if not isinstance(raw, str):
        raise WikiToolError("invalid_argument", f"note_id 必须是字符串，得到 {type(raw).__name__}")
    text = raw.strip().replace("\\", "/")
    if text.endswith(".md"):
        text = text[:-3]
    for prefix in ("wiki-data/", "./"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    if text.startswith("notes/"):
        text = text[len("notes/") :]
    if not NOTE_ID_RE.match(text):
        raise WikiToolError(
            "path_rejected",
            f"note_id {raw!r} 形态非法：只允许字母开头、由字母数字下划线连字符组成"
            "（如 N-0001），不接受路径分隔符或 ..",
            note_id=raw,
        )
    return text


def _is_inside(path: Path, root: Path) -> bool:
    """包含性断言：与 tools/fs 沙箱同语义（resolve 后必须在 root 之内）。"""
    try:
        return Path(path).resolve().is_relative_to(Path(root).resolve())
    except OSError:  # pragma: no cover - resolve 失败（循环链接等）一律视为越界
        return False


def _assert_note_id_writable(note: Note) -> None:
    """写路径 id 形态复核（写保护第 2 件的写侧兜底，终审 Important 修复）。

    ``WikiStore._write_note`` 直接拼 ``notes_dir/{meta.id}.md`` 且无形态校验，
    而 ``_load`` 优先采用 frontmatter 里的 id（``meta.id = meta.id or path.stem``）——
    手工编辑 / 同步污染可让库内笔记带上任意形态的 id。凡以既有笔记的 id 作为
    写路径分量的操作（update_memory / supersede / invalidate），写前必须断言：

    1. id 匹配白名单正则（挡 ``../``、路径分隔符——否则 ``save_note`` 会写出沙箱外）；
    2. id 与落盘文件名一致（挡良性 id≠stem——旧行为会静默写出"新文件"而原文件
       不动，客户端却收到 ok）。

    不满足即抛 validation_failed 拒绝写入；修复方式是改正 frontmatter id 或文件名。
    """
    note_id = str(note.id)
    stem = note.path.stem
    if not NOTE_ID_RE.match(note_id) or note_id != stem:
        raise WikiToolError(
            "validation_failed",
            f"笔记 id 形态非法或与文件名不一致，已拒绝写入：id={note_id!r}、"
            f"文件名={stem}.md（id 必须匹配 {NOTE_ID_RE.pattern} 且与文件名一致；"
            "请先修复该笔记文件的 frontmatter id 或文件名，再重试本次操作）",
            note_id=note_id,
        )


def validate_kind(kind: Any) -> str | None:
    """kind 参数归一：None 原样透传（检索时表示不过滤）；非法值抛 invalid_kind。"""
    if kind is None:
        return None
    if not isinstance(kind, str):
        raise WikiToolError("invalid_kind", f"kind 必须是字符串，得到 {type(kind).__name__}")
    normalized = kind.strip().lower()
    if normalized not in KIND_LEVELS:
        raise WikiToolError(
            "invalid_kind",
            f"kind 非法：{kind!r}，只能是 {' / '.join(KIND_LEVELS)}"
            "（knowledge 世界知识 / user 用户画像与偏好 / experience 经验教训）",
        )
    return normalized


def derive_title(content: str, fallback: str = "未命名记忆") -> str:
    """从正文派生标题：首个非空行去掉 Markdown 行首记号后截 60 字符。

    memory_store / memory_supersede / 墓碑笔记的写入都不带 title 参数，
    由正文派生即可保证检索可命中；要精确控制标题请走 wiki_write。
    """
    for line in content.splitlines():
        text = line.strip().lstrip("#>*- ").strip()
        if text:
            return text[:60]
    return fallback


# ---- 写前备份 ---------------------------------------------------------------


def create_backup(
    root: str | Path,
    *,
    note_id: str,
    title: str,
    source_path: str | Path,
    existed: bool,
) -> dict[str, Any]:
    """把将被覆盖的原文件复制到 ``wiki-data/.backups/<UTC 时间戳>/``。

    - ``existed=True``：把原文件按相对 wiki-data 的结构复制过去（notes/N-0001.md）。
    - ``existed=False``：写一条 manifest 标记 action=created（新增没有原文件可备份）。
    两种情况下都写 manifest.json 记录本次操作（谁在什么时候写哪个位置）。

    读写都走 tools/fs 沙箱（safe_read / safe_write），任何失败都抛 OSError/SandboxError，
    由调用方转成 backup_failed 并拒绝写入。返回备份元信息（回填进工具响应，便于审计）。
    """
    root_path = Path(root)
    designator = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    backup_dir = root_path / BACKUP_DIR_NAME / designator
    relative = Path("notes") / f"{note_id}.md"
    manifest: dict[str, Any] = {
        "action": "overwrite" if existed else "created",
        "note_id": note_id,
        "title": title,
        "at": _now_iso(),
        "target": relative.as_posix(),
        "backup_file": None,
    }
    if existed:
        # 原文件 → 备份目录：读走沙箱，写也走沙箱，越界直接抛 SandboxError
        original = safe_read(source_path, root=root_path)
        destination = backup_dir / relative
        safe_write(destination, original, root=root_path)
        manifest["backup_file"] = (Path(BACKUP_DIR_NAME) / designator / relative).as_posix()
        manifest["bytes"] = len(original.encode("utf-8"))
    safe_write(
        backup_dir / "manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2),
        root=root_path,
    )
    return {
        "dir": (Path(BACKUP_DIR_NAME) / designator).as_posix(),
        "action": manifest["action"],
        "backup_file": manifest["backup_file"],
    }


# ---- 检索结果 ---------------------------------------------------------------


@dataclass
class _IndexSync:
    """一次索引同步的结果：是否重建、重建条数、错误信息。"""

    rebuilt: bool = False
    notes_indexed: int = 0
    error: str | None = None


class WikiService:
    """MCP 工具的 wiki 门面：检索 / 读取 / 写入 / 增量列表 / 健康检查。

    - ``root`` 即 wiki-data 沙箱根目录，也是 WikiStore / SearchIndex / EntityRegistry
      的根；所有文件读写都在它之内。
    - ``config`` 为 config.toml 的解析结果（缺省时全部走 mock：无 key、零网络）。
    - 写操作串行化（进程内 RLock）：并发调用不会交错进同一段"校验→备份→落盘"。
    """

    def __init__(self, root: str | Path = "wiki-data", *, config: Mapping[str, Any] | None = None):
        self.root = Path(root)
        self.config: Mapping[str, Any] = config or {}
        self.store = WikiStore(self.root)
        self.settings = wiki_settings(self.config)
        self._write_lock = threading.RLock()

    # ---- 路径与读取原语 ----

    def notes_path(self, note_id: str) -> Path:
        """note_id → notes/ 下的绝对路径；id 非法或路径越界一律拒绝。

        这里校验的正是 ``WikiStore`` 自己会拼出的那条路径（同为 notes_dir/id.md），
        保证沙箱断言与实际落盘位置一致。
        """
        normalized = normalize_note_id(note_id)
        target = self.store.notes_dir / f"{normalized}.md"
        if not _is_inside(target, self.root):
            raise WikiToolError(
                "path_rejected",
                f"路径越界：{target} 不在 wiki 沙箱 {self.root} 之内",
                note_id=note_id,
            )
        return target

    def _load(self, note_id: str) -> Note | None:
        """沙箱内读一条笔记（文件不存在返回 None）。

        与 ``WikiStore.get_note`` 同语义，区别是文件读取强制走 tools/fs 的 safe_read
        （路径越界抛 SandboxError），并把 FileNotFoundError 收敛成 None。
        """
        path = self.notes_path(note_id)
        try:
            text = safe_read(path, root=self.root)
        except FileNotFoundError:
            return None
        meta_dict, body = parse(text)
        meta = NoteMeta.from_dict(meta_dict)
        meta.id = meta.id or path.stem
        return Note(id=meta.id, title=meta.title, body=body, meta=meta, path=path)

    def _walk_redirects(self, note_id: str) -> tuple[Note | None, list[str]]:
        """沿 merged→redirect_to / superseded→superseded_by 走到最终 active 笔记。

        返回 ``(最终笔记, 链上被跳过的别名 id)``；起点不存在返回 (None, [])，
        成环/断裂抛 WikiToolError。这里自行走链（不直接调 store.follow_redirect）
        是为了同时拿到被跳过的 ID 列表，好在响应里向客户端交代来源。
        """
        origin = normalize_note_id(note_id)
        note = self._load(origin)
        if note is None:
            return None, []
        aliases: list[str] = []
        visited = {origin}
        while note.meta.status != "active":
            target = (
                note.meta.redirect_to
                if note.meta.status == "merged"
                else note.meta.superseded_by
            )
            if not target:
                link_field = "redirect_to" if note.meta.status == "merged" else "superseded_by"
                raise WikiToolError(
                    "broken_redirect",
                    f"笔记 {note.id} 状态为 {note.meta.status}，但缺少 {link_field} 字段",
                    note_id=note.id,
                )
            if target in visited:
                raise WikiToolError(
                    "redirect_cycle",
                    f"redirect 链成环：{' → '.join([*visited, target])}",
                    note_id=origin,
                    chain=[*aliases, note.id],
                )
            aliases.append(note.id)
            visited.add(target)
            try:
                nxt = self._load(target)
            except WikiToolError as exc:
                raise WikiToolError(
                    "broken_redirect",
                    f"笔记 {note.id} 跳转到 {target}，但该目标无法读取：{exc.message}",
                    note_id=origin,
                    missing=target,
                ) from exc
            if nxt is None:
                raise WikiToolError(
                    "broken_redirect",
                    f"笔记 {note.id} 跳转到 {target}，但目标笔记不存在",
                    note_id=origin,
                    missing=target,
                )
            note = nxt
        return note, aliases

    # ---- 检索 ----

    def _open_index(self) -> SearchIndex:
        return SearchIndex(
            self.root,
            embedding=get_embedding_provider(self.config, cache_path=self.root / "index.db"),
            tokenizer=self.settings.fts_tokenizer,
            half_life_days=self.settings.half_life_days,
        )

    def _index_is_stale(self, index: SearchIndex) -> str | None:
        """索引落后于 store 的原因（``None`` = 新鲜，无需重建）。

        **契约变更（P2-F 裁定二）**：v1 是"md 比 index.db 新，或索引条数与笔记数
        不符"的 mtime + 条数口径，且必须在打开 SearchIndex 之前调用（打开索引会写
        index.db，之后 mtime 判据永远失效）。现在判定委托给 loop 路径同一份
        ``wiki.index.index_drift``（id 集合 / status / **索引指纹**三维），于是：

        - 返回类型由 ``bool`` 变成 ``str | None``（落后原因，供 health/审计展示）；
        - 调用时点反转：必须在**打开索引之后**调用（判据要读 note_meta 快照），
          调用方先 ``_open_index()`` 再问"落后没有"；
        - mtime 前置被删除（理由见模块 docstring）：它能漏掉的正是指纹维度要治的
          那些原地改写，而共享判据本来就要全量扫一遍 store。

        与 loop 路径的差别只剩"落后了怎么办"：这里重建失败不阻断检索（降级为旧
        快照 + 结构化告警），``ensure_index_fresh`` 则直接抛给调用方。
        """
        stale, reason = index_drift(self.store, index)
        return reason if stale else None

    def _sync_index(self, index: SearchIndex, *, stale: bool) -> _IndexSync:
        """按需重建索引（落后于 store 时），失败不阻断检索（降级为旧快照）。

        ``stale`` 由调用方用 ``_index_is_stale`` 在打开索引之后算好（P2-F 起判据
        是共享的 ``index_drift``，见该方法的契约说明）。
        """
        if not stale:
            return _IndexSync()
        try:
            count = index.rebuild(self.store)
        except Exception as exc:  # noqa: BLE001 -- 索引坏掉不应让检索整体失败
            return _IndexSync(error=f"{type(exc).__name__}: {exc}")
        return _IndexSync(rebuilt=True, notes_indexed=count)

    @_structured_errors
    def search(
        self,
        query: str,
        k: int = 5,
        kind: str | None = None,
        include_tombstones: bool = False,
    ) -> dict[str, Any]:
        """双通道检索（FTS5 + 向量 + RRF），返回 active 笔记；跟随 redirect 并标注来源。

        ``kind`` 给定时（knowledge/user/experience）只返回该类型笔记；
        非法取值抛 invalid_kind。

        ``include_tombstones``（P2-F 裁定一）缺省 False = 墓碑（``memory_invalidate``
        生成的失效裁定记录）**不出现在结果里**；传 True 才把墓碑当普通命中返回
        （审计/排查用）。判定含重定向落点，见 ``SearchIndex.search``。
        """
        text = (query or "").strip()
        if not text:
            raise WikiToolError("invalid_argument", "query 不能为空")
        if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= MAX_K:
            raise WikiToolError("invalid_argument", f"k 必须是 1..{MAX_K} 的整数，得到 {k!r}")
        normalized_kind = validate_kind(kind)
        if not isinstance(include_tombstones, bool):
            raise WikiToolError(
                "invalid_argument",
                f"include_tombstones 必须是布尔值，得到 {include_tombstones!r}",
            )
        with self._open_index() as index:
            # 判据是共享的 index_drift（读 note_meta 快照），故先开索引再判落后
            stale_reason = self._index_is_stale(index)
            sync = self._sync_index(index, stale=stale_reason is not None)
            matches = index.search(
                text, k=k, kind=normalized_kind, include_tombstones=include_tombstones
            )
        results = [
            {
                "note_id": m.note_id,
                "title": m.title,
                "snippet": m.snippet,
                "score": round(m.score, 6),
                "match_type": m.match_type,
                "redirected_from": m.redirected_from,
                "redirect_note": (
                    f"{m.redirected_from} 已并入 {m.note_id}（merged/superseded），"
                    "该结果由重定向得到"
                    if m.redirected_from
                    else None
                ),
            }
            for m in matches
        ]
        payload: dict[str, Any] = {
            "ok": True,
            "query": text,
            "k": k,
            "count": len(results),
            "results": results,
        }
        if sync.rebuilt:
            payload["index_rebuilt"] = sync.notes_indexed
            payload["index_stale_reason"] = stale_reason
        if sync.error:
            payload["index_warning"] = f"索引重建失败，本次检索基于旧快照：{sync.error}"
        if not results:
            payload["hint"] = "没有命中；换个关键词，或先用 wiki_list_changes 看看 wiki 里有什么"
        return payload

    # ---- 读取 ----

    @_structured_errors
    def read(self, note_id: str) -> dict[str, Any]:
        """读一条笔记全文（frontmatter 关键字段 + 正文）；merged/superseded 跟随重定向。"""
        note, aliases = self._walk_redirects(note_id)
        if note is None:
            raise WikiToolError(
                "not_found",
                f"笔记 {note_id!r} 不存在（wiki 根目录：{self.root}）",
                note_id=note_id,
            )
        body = note.body
        truncated = len(body) > MAX_BODY_CHARS
        payload: dict[str, Any] = {
            "ok": True,
            "requested_id": note_id,
            "note": {
                "note_id": note.id,
                "title": note.title,
                "status": note.meta.status,
                "kind": note.meta.kind,
                "importance": note.meta.importance,
                # 墓碑标记：read 是审计可见性的出口（墓碑默认不在检索结果里，
                # 但要能读全文、能看出"这条是墓碑"）
                "tombstone": note.meta.tombstone,
                "confidence": note.meta.confidence,
                "volatility": note.meta.volatility,
                "entities": list(note.meta.entities),
                "created": note.meta.created,
                "observed_at": note.meta.observed_at,
                "reviewed_at": note.meta.reviewed_at,
                "trace_id": note.meta.trace_id,
                "sources": [
                    {"url": s.url, "content_hash": s.content_hash} for s in note.meta.sources
                ],
                "redirect_to": note.meta.redirect_to,
                "superseded_by": note.meta.superseded_by,
                "extra": dict(note.meta.extra),
            },
            "body": body[:MAX_BODY_CHARS] if truncated else body,
            "path": _relative_path(note.path, self.root),
        }
        if aliases:
            # 请求的是别名笔记：明确告诉客户端"你要的是 N-0001，内容来自 N-0002"
            payload["redirected"] = True
            payload["redirect_chain"] = aliases
            payload["source_note"] = aliases[0]
            payload["message"] = (
                f"{aliases[0]} 已{'合并' if len(aliases) == 1 else '经过重定向'}"
                f"到 {note.id}，以下内容来自 {note.id}"
                f"（来源链：{' → '.join([*aliases, note.id])}）"
            )
        else:
            payload["redirected"] = False
        if truncated:
            payload["truncated"] = True
            payload["message"] = (
                f"正文超过 {MAX_BODY_CHARS} 字符已截断，完整内容见 {payload['path']}"
            )
        return payload

    # ---- 写入（写保护三件套）----

    @_structured_errors
    def write(
        self,
        body: str,
        *,
        title: str = "",
        entities: Sequence[str] | None = None,
        confidence: str = "medium",
        volatility: str = "stable",
        sources: Sequence[SourceInput] | None = None,
        kind: str = "knowledge",
        importance: float | None = None,
        tombstone: bool = False,
    ) -> dict[str, Any]:
        """新增一条原子笔记：校验 → 备份 → 落盘 → 同步索引。

        ``kind``/``importance`` 是记忆载体字段（P1-A）：wiki_write 不暴露它们
        （工具签名不变，走默认值），memory_store 经此通道透传。
        ``sources`` 的元素可以是 URL 字符串或 ``{"url", "content_hash"}`` 映射。
        ``tombstone`` 是墓碑标记（P2-F 裁定一）：同样不暴露给 wiki_write，
        只由 ``invalidate_memory`` 生成墓碑时传 True——客户端无法自行把普通
        笔记标成墓碑（那是失效裁定，必须走 memory_invalidate 留 reason）。
        """
        fields = _validate_write_inputs(
            body=body,
            title=title,
            entities=entities,
            confidence=confidence,
            volatility=volatility,
            sources=sources,
            kind=kind,
            importance=importance,
            tombstone=tombstone,
        )
        warnings = list(fields.pop("warnings", []))
        with self._write_lock:
            note_id = self.store.next_note_id()
            if not GENERATED_NOTE_ID_RE.match(note_id):  # pragma: no cover - 防御性断言
                raise WikiToolError("internal", f"内部生成的 note_id 形态异常：{note_id!r}")
            target = self.notes_path(note_id)
            existed = target.is_file()
            try:
                backup = create_backup(
                    self.root,
                    note_id=note_id,
                    title=str(fields.get("title") or ""),
                    source_path=target,
                    existed=existed,
                )
            except (OSError, SandboxError) as exc:
                raise WikiToolError(
                    "backup_failed",
                    f"写前备份失败，已拒绝写入：{type(exc).__name__}: {exc}",
                    note_id=note_id,
                ) from exc
            try:
                note = self.store.save_note(body=body, note_id=note_id, **fields)
            except (OSError, ValueError) as exc:
                raise WikiToolError(
                    "write_failed",
                    f"笔记落盘失败：{type(exc).__name__}: {exc}",
                    note_id=note_id,
                ) from exc
            index_error = self._index_note(note)
        payload: dict[str, Any] = {
            "ok": True,
            "note_id": note.id,
            "title": note.title,
            "status": note.meta.status,
            "kind": note.meta.kind,
            "importance": note.meta.importance,
            "created": note.meta.created,
            "path": _relative_path(note.path, self.root),
            "backup": backup,
            "indexed": index_error is None,
            "message": f"已写入笔记 {note.id}（{note.title or '无标题'}）",
        }
        if index_error is not None:
            payload["index_warning"] = f"索引同步失败，检索暂不可见：{index_error}"
        if warnings:
            payload["warnings"] = warnings
        return payload

    def _index_note(self, note: Note) -> str | None:
        """增量同步索引；失败返回错误字符串（笔记已落盘，不回滚）。"""
        try:
            with self._open_index() as index:
                index.index_note(note)
        except Exception as exc:  # noqa: BLE001 -- 索引失败不能推翻已成功的写入
            return f"{type(exc).__name__}: {exc}"
        return None

    # ---- memory_*：面向 Agent 的外置记忆接口（P1-A）----------------------------
    #
    # 九个方法对应 server.py 的九个 memory_* 工具。写路径全部复用 write() 的
    # "校验 → 备份 → 落盘 → 索引"通道并持有 _write_lock；修订/取代/失效都强制
    # 留 reason 与备份，禁止静默覆盖。

    @_structured_errors
    def store_memory(
        self,
        content: str,
        *,
        kind: str = "knowledge",
        entities: Sequence[str] | None = None,
        importance: float | None = None,
        source_urls: Sequence[str] | None = None,
        confidence: str = "medium",
        volatility: str = "stable",
    ) -> dict[str, Any]:
        """显式写入一条记忆（Agent/用户主动存）：复用 write() 通道，透传 kind/importance。

        title 不作为参数：由正文首个非空行派生（见 derive_title）；
        需要精确控制标题的写入走 wiki_write。
        """
        if not isinstance(content, str) or not content.strip():
            raise WikiToolError("validation_failed", "content 不能为空：请写入记忆正文")
        normalized_kind = validate_kind(kind)
        return self.write(
            content,
            title=derive_title(content),
            entities=entities,
            confidence=confidence,
            volatility=volatility,
            sources=source_urls,
            kind=normalized_kind or "knowledge",
            importance=importance,
        )

    @_structured_errors
    def recall(
        self,
        query: str,
        k: int = 5,
        kind: str | None = None,
        include_tombstones: bool = False,
    ) -> dict[str, Any]:
        """动态召回入口（P3 的钩子位）：MVP = search + 重定向跟随 + 结构化透传。

        P3 将升级为 budget-aware 动态召回（按 token 预算与 importance 挑选记忆），
        当前为结构化透传：在 search 结果上为每条记忆标注 status / observed_at /
        kind（附 importance），让调用方按记忆状态自行取舍，不做预算裁剪。

        ``include_tombstones`` 与 ``search`` 同语义（缺省 False = 墓碑不进召回）。
        """
        payload = self.search(query, k=k, kind=kind, include_tombstones=include_tombstones)
        if not payload.get("ok"):
            return payload
        for item in payload["results"]:
            note = self._load(item["note_id"])
            if note is None:  # pragma: no cover - 结果刚被外部删除的竞态，标注缺省
                continue
            item["status"] = note.meta.status
            item["observed_at"] = note.meta.observed_at
            item["kind"] = note.meta.kind
            item["importance"] = note.meta.importance
        payload["mode"] = "passthrough"
        return payload

    @_structured_errors
    def update_memory(
        self,
        note_id: str,
        content: str,
        reason: str,
        *,
        source_urls: Sequence[SourceInput] | None = None,
    ) -> dict[str, Any]:
        """修订既有记忆正文（不新增 ID）：写前备份 + reason 落盘，禁止静默覆盖。

        - 备份：create_backup(existed=True)，原稿进 wiki-data/.backups/；
        - reason：**追加**进 frontmatter extra 的 ``update_reasons`` 列表（按修订顺序，
          每条 ``{"reason", "at"}``），按序累积、不覆盖，但只保留最近
          ``MAX_UPDATE_REASONS`` 条（终审 F6：防 frontmatter 无界膨胀；更早的原因
          仍可从 ``.backups/<时间戳>/`` 的历史原稿追溯）；同时刷新
          ``reviewed_at``（list_changes 里表现为 reviewed 变更）。兼容迁移：旧版
          本的单键 ``update_reason``/``update_reason_at`` 在首次追加时折叠为列表
          首条目，随后旧键移除（round-trip 归一，不丢历史原因）；
        - 只允许修订 active 笔记：merged/superseded 的内容归属最终版本，
          应先 wiki_read 跟随重定向，再用 memory_supersede 取代。

        证据链（carried 项④）
        --------------------
        ``source_urls`` 是可选的新证据入口，元素为 URL 字符串或
        ``{"url": ..., "content_hash": ...}`` 映射（hash 可省/可为空串）。
        **update 是修订而非替换，所以来源按 ``(url, content_hash)`` 去重后追加**
        到既有 sources（不替换、不丢旧证据）；不给该参数时 sources 原样不动。
        **空列表视为非法参数**（``validation_failed``）：要么给至少一个来源、
        要么省略该参数——省略才表示"本次不补充证据"。

        confidence 不受本参数影响（修订不改变原记忆的置信度）。要按新证据重标
        置信度请用 ``memory_supersede(source_urls=..., confidence=...)``（旧 ID
        沿 superseded_by 链可达新版本）；**不要**用 memory_store 代替——那会新建
        一条 active 记忆而旧 ID 不被退役，同一事实会留下两条 active 记忆。

        返回体带 ``evidence``：给了 ``source_urls`` 为 ``"provided"``（附
        ``sources_added`` = 去重后新增条数），没给为 ``"none"``。
        """
        if not isinstance(content, str) or not content.strip():
            raise WikiToolError("validation_failed", "content 不能为空：请提供修订后的记忆正文")
        if not isinstance(reason, str) or not reason.strip():
            raise WikiToolError(
                "validation_failed", "reason 不能为空：修订必须留下原因（审计留痕）"
            )
        # 证据链参数在锁外、在任何写入（含备份）之前校验：非法参数零落盘。
        provided = _evidence_sources(source_urls)
        reason_text = reason.strip()
        with self._write_lock:
            note = self._require_active(note_id)
            try:
                backup = create_backup(
                    self.root,
                    note_id=note.id,
                    title=note.title,
                    source_path=note.path,
                    existed=True,
                )
            except (OSError, SandboxError) as exc:
                raise WikiToolError(
                    "backup_failed",
                    f"写前备份失败，已拒绝修订：{type(exc).__name__}: {exc}",
                    note_id=note.id,
                ) from exc
            now = _now_iso()
            meta = note.meta
            merged_sources, sources_added = _merge_sources(meta.sources, provided)
            try:
                # 只点名真正要改的字段（reviewed_at / sources / extra），其余逐字段
                # 保留——replace + save_meta 让"漏传字段"在语法上不存在（修复轮 I-2）
                updated = self.store.save_meta(
                    meta.replace(
                        reviewed_at=now,
                        sources=merged_sources,
                        extra=_append_update_reason(meta.extra, reason_text, now),
                    ),
                    content,
                )
            except (OSError, ValueError) as exc:
                raise WikiToolError(
                    "write_failed",
                    f"修订落盘失败：{type(exc).__name__}: {exc}",
                    note_id=note.id,
                ) from exc
            index_error = self._index_note(updated)
        payload: dict[str, Any] = {
            "ok": True,
            "note_id": updated.id,
            "title": updated.title,
            "reason": reason_text,
            "reasons_so_far": len(updated.meta.extra.get("update_reasons") or []),
            "reviewed_at": now,
            "backup": backup,
            "indexed": index_error is None,
            "path": _relative_path(updated.path, self.root),
            "evidence": "provided" if provided else "none",
            "sources_added": sources_added,
            "message": (
                f"已修订记忆 {updated.id}（原因追加记入 update_reasons，"
                f"原稿备份于 {backup['dir']}）"
                + (
                    f"；本次追加来源 {sources_added} 条"
                    f"（按 (url, content_hash) 去重，现有 {len(merged_sources)} 条）"
                    if provided
                    else "；本次未提供来源（evidence=none，既有来源原样保留）"
                )
            ),
        }
        if index_error is not None:
            payload["index_warning"] = f"索引同步失败，检索暂不可见：{index_error}"
        return payload

    @_structured_errors
    def supersede_memory(
        self,
        note_id: str,
        new_content: str,
        reason: str,
        *,
        source_urls: Sequence[SourceInput] | None = None,
        confidence: str | None = None,
    ) -> dict[str, Any]:
        """用新内容替代旧记忆：先写新笔记，再把旧笔记标 superseded_by=新 ID。

        不带 ``source_urls`` 也不带 ``confidence`` 时（原 P1-A 行为，逐字段不变）：
        新笔记继承旧笔记的 entities/kind/confidence/volatility/importance（身份
        连续性），**不继承 sources**，正文与标题来自 new_content；旧笔记沿
        superseded_by 链可达新笔记（active），reason 记入 extra 的
        ``supersede_reason``。

        证据链（carried 项④）
        --------------------
        ``source_urls`` 是可选的新证据入口，元素为 URL 字符串或
        ``{"url": ..., "content_hash": ...}`` 映射（hash 可省/可为空串；同一
        ``(url, content_hash)`` 只保留一条，去重口径与 memory_update 一致）。
        **空列表视为非法参数**（``validation_failed``）：要么给至少一个来源、
        要么省略该参数——省略才表示"本次不提供证据"。

        - **提供了来源**：新笔记的 sources = 提供值，而不是旧笔记的来源（旧证据
          不该被伪装成新结论的证据）。
        - **未提供来源**：sources 保持为空（不继承旧来源），其余字段照旧继承，
          同时在返回体标注 ``evidence: "none"``——调用方能看出自己造了一条
          没有证据链的替代记忆，而不是静默以为证据跟着过来了。

        confidence（**从不继承旧值**，简报硬约束）
        ------------------------------------------
        旧记忆的 confidence 是给**旧证据**打的，把新证据接到它上面等于用旧证据的
        背书为新结论担保。取值优先级：

        1. 显式给了 ``confidence`` → 用该值（走与 ``write()`` 相同的白名单校验；
           非法/非字符串值在**锁外**以 ``validation_failed`` 拒绝——与
           ``source_urls`` 同一优先级，先于 note_id 查找，且不产生任何备份或落盘）；
        2. 没给但带了 ``source_urls`` → 缺省 ``"medium"``（= "未特别标定"）；
        3. 两者都没给 → 继承旧值（P1-A 兼容路径，仅此一种继承）。

        所以"有来源 + 具体事实要素 → high"（``wiki/formation.py`` 的确定性口径）
        在本工具里是可执行的：``memory_supersede(..., source_urls=[...],
        confidence="high")``。**不要**用 memory_store / wiki_write 来"补标定"：
        那会新建一条 active 记忆而旧 ID 不被退役，同一事实留下两条 active 记忆、
        supersede 链也断了。

        返回体的 ``evidence``：提供了来源为 ``"provided"``，未提供为 ``"none"``；
        ``confidence`` 是本次实际写入新笔记的值。
        """
        if not isinstance(new_content, str) or not new_content.strip():
            raise WikiToolError(
                "validation_failed", "new_content 不能为空：请提供替代旧记忆的新内容"
            )
        if not isinstance(reason, str) or not reason.strip():
            raise WikiToolError(
                "validation_failed", "reason 不能为空：取代必须留下原因（审计留痕）"
            )
        # 证据链参数在锁外、在任何写入（含新笔记落盘与旧笔记标记）之前校验：
        # 非法参数不会留下半成品（既无新笔记也无备份）。
        provided = _evidence_sources(source_urls)
        # confidence 同样在锁外提前校验（与 source_urls 同一优先级）：非法值不应
        # 因为"目标恰好不存在"就变成 not_found；也保证零落盘。
        explicit_confidence = _require_confidence(confidence) if confidence is not None else None
        # 去重口径与 update 一致（(url, content_hash) 只留一条），保证新笔记的
        # sources 恰好等于"去重后的提供值"；回传给 write() 的是映射形态，
        # 与 _normalize_sources 的归一结果等价（url 已 strip、hash 已归一）。
        deduped, _ = _merge_sources([], provided)
        evidence_input = [
            {"url": ref.url, "content_hash": ref.content_hash} for ref in deduped
        ]
        reason_text = reason.strip()
        with self._write_lock:
            old = self._require_active(note_id)
            now = _now_iso()
            # confidence 一律不继承旧值：显式给出就用给出的，带了来源则缺省
            # medium，只有"既没来源也没显式值"这条 P1-A 兼容路径才继承。
            resolved_confidence = (
                explicit_confidence
                if explicit_confidence is not None
                else ("medium" if provided else old.meta.confidence)
            )
            # 第一步：新笔记走完整写保护通道（校验/备份/落盘/索引）。
            # 备份失败会在这里整体失败，旧笔记尚未动——无半成品状态。
            written = self.write(
                new_content,
                title=derive_title(new_content, fallback=old.title or "替代笔记"),
                entities=list(old.meta.entities),
                confidence=resolved_confidence,
                volatility=old.meta.volatility,
                kind=old.meta.kind,
                importance=old.meta.importance,
                sources=evidence_input or None,
            )
            if not written.get("ok"):
                return written
            new_id = str(written["note_id"])
            # 第二步：旧笔记 → superseded，链到新笔记。
            marked = self._mark_superseded(old, superseded_by=new_id, now=now, reason=reason_text)
        # 落盘值已由 write() 归一（strip + lower），返回体报告的就是实际写入值
        confidence_used = str(resolved_confidence).strip().lower()
        if provided:
            tail = (
                f"；新笔记来源=本次提供的 {len(deduped)} 条"
                f"（按 (url, content_hash) 去重），confidence={confidence_used}"
                + (
                    "（显式指定）"
                    if explicit_confidence is not None
                    else "（缺省 medium，不继承旧证据的置信度）"
                )
            )
        else:
            tail = (
                "；本次未提供来源（evidence=none）：新笔记没有证据链，"
                f"confidence={confidence_used}"
                + ("（显式指定）" if explicit_confidence is not None else "（沿用旧值）")
            )
        payload: dict[str, Any] = {
            "ok": True,
            "old_note_id": old.id,
            "new_note_id": new_id,
            "superseded_by": new_id,
            "reason": reason_text,
            "backup": written.get("backup"),
            "path": _relative_path(marked.path, self.root),
            "evidence": "provided" if provided else "none",
            "confidence": confidence_used,
            "message": (
                f"旧记忆 {old.id} 已被 {new_id} 取代（status=superseded，"
                f"沿 superseded_by 可达；原因记入 supersede_reason）" + tail
            ),
        }
        if written.get("warnings"):
            payload["warnings"] = written["warnings"]
        return payload

    @_structured_errors
    def invalidate_memory(self, note_id: str, reason: str) -> dict[str, Any]:
        """判定记忆失效（无替代内容）：自动生成墓碑笔记，旧笔记沿链可达 active。

        裁定语义（不新增 invalidated 状态，状态机变更留给 P2）：墓碑是一条
        kind=knowledge 的 active 笔记（正文 = 失效原因 + 时间戳），旧笔记
        status=superseded 且 superseded_by 指向墓碑——"superseded 必须沿链
        可达 active"的不变量与审计留痕同时保住。

        墓碑另带 ``tombstone: true`` 标记（P2-F 裁定一）：它是审计记录而非当前
        知识，检索（``search``/``recall``）与 run 的 Prior 注入**默认排除**它，
        但 ``wiki_read`` / ``memory_timeline`` / ``lint`` 照常可见——对外返回体
        与链可达性（P1-A 契约）逐字段不变，变的只是"它不再被当作 active 知识
        召回"。需要把墓碑当普通命中取回时用 ``include_tombstones=True``。
        """
        if not isinstance(reason, str) or not reason.strip():
            raise WikiToolError(
                "validation_failed", "reason 不能为空：失效裁定必须留下原因（审计留痕）"
            )
        reason_text = reason.strip()
        with self._write_lock:
            old = self._require_active(note_id)
            now = _now_iso()
            tombstone_body = (
                f"记忆 {old.id}（{old.title or '无标题'}）已被裁定失效。\n"
                f"失效原因：{reason_text}\n"
                f"失效时间：{now}"
            )
            written = self.write(
                tombstone_body,
                title=derive_title(f"[已失效] {old.title}" if old.title else "[已失效] 记忆"),
                entities=list(old.meta.entities),
                kind="knowledge",
                tombstone=True,
            )
            if not written.get("ok"):
                return written
            tombstone_id = str(written["note_id"])
            marked = self._mark_superseded(
                old, superseded_by=tombstone_id, now=now, reason=reason_text, prefix="invalidate"
            )
        payload: dict[str, Any] = {
            "ok": True,
            "old_note_id": old.id,
            "tombstone_id": tombstone_id,
            "superseded_by": tombstone_id,
            "reason": reason_text,
            "backup": written.get("backup"),
            "path": _relative_path(marked.path, self.root),
            "message": (
                f"记忆 {old.id} 已失效：生成墓碑笔记 {tombstone_id}（active），"
                f"旧笔记沿 superseded_by 可达墓碑，原因记入 invalidate_reason"
            ),
        }
        if written.get("warnings"):
            payload["warnings"] = written["warnings"]
        return payload

    @_structured_errors
    def timeline(self, note_id: str) -> dict[str, Any]:
        """单条记忆的版本史：redirect 链拓扑事件 + list_changes 口径变更记录，合并输出。

        方向约定：**旧 → 新**（oldest_first）。两类来源按 note_id 合并：
        ① 沿 redirect_to/superseded_by 链的拓扑事件——从请求的笔记走到最终
        active 版本，并反向收集指向链上节点的更早版本（查链上任一 ID 都拿
        完整历史）；② 每条笔记按 list_changes 同口径拆出的变更记录——一条
        created（新建）、每次修订一条 reviewed（reason 取 extra.update_reasons，
        兼容旧单键 update_reason）、superseded/merged 一条 status（reason 取
        supersede_reason / invalidate_reason）。全部事件按时间升序输出并做
        全字段去重；同刻按链拓扑序与事件类型（created < reviewed < status）
        兜底排序，保证同一笔记内 created → reviewed… → status 的相对次序。
        """
        origin = normalize_note_id(note_id)
        final, aliases = self._walk_redirects(origin)
        if final is None:
            raise WikiToolError(
                "not_found",
                f"笔记 {note_id!r} 不存在（wiki 根目录：{self.root}）",
                note_id=note_id,
            )
        lineage: dict[str, Note] = {}
        for nid in [*aliases, final.id]:
            note = self._load(nid)
            if note is not None:
                lineage[nid] = note
        # 反向收集更早版本：redirect_to / superseded_by 指向链上节点的笔记
        all_notes = self.store.list_notes(status=None)
        frontier = [*aliases, final.id]
        while frontier:
            current = frontier.pop()
            for note in all_notes:
                if note.id in lineage:
                    continue
                if current in (note.meta.redirect_to, note.meta.superseded_by):
                    lineage[note.id] = note
                    frontier.append(note.id)
        # 两类来源合并：链拓扑序内逐笔记生成事件 → 时间升序 → 全字段去重
        keyed: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        for topo_idx, nid in enumerate(_topo_order(lineage)):
            keyed.extend(self._note_events(lineage[nid], topo_idx))
        keyed.sort(key=lambda pair: pair[0])
        seen: set[tuple[Any, ...]] = set()
        events: list[dict[str, Any]] = []
        for _, payload in keyed:
            fingerprint = tuple(sorted((k, str(v)) for k, v in payload.items()))
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            events.append(payload)
        return {
            "ok": True,
            "note_id": origin,
            "requested_id": note_id,
            "current": final.id,
            "direction": "oldest_first",
            "count": len(events),
            "events": events,
        }

    @_structured_errors
    def conflicts(self, status: str = "open") -> dict[str, Any]:
        """冲突台账查询：透传 WikiStore.list_conflicts。status 取 open/resolved/all。"""
        if not isinstance(status, str):
            raise WikiToolError(
                "invalid_argument", f"status 必须是字符串，得到 {type(status).__name__}"
            )
        normalized = status.strip().lower()
        if normalized == "all":
            store_status = None
        elif normalized in ("open", "resolved"):
            store_status = normalized
        else:
            raise WikiToolError(
                "invalid_argument", f"status 非法：{status!r}，只能是 open / resolved / all"
            )
        rows = self.store.list_conflicts(status=store_status)
        return {
            "ok": True,
            "status": normalized,
            "count": len(rows),
            "conflicts": [
                {
                    "conflict_id": c.id,
                    "question": c.question,
                    "status": c.status,
                    "claim_a": dict(c.claim_a),
                    "claim_b": dict(c.claim_b),
                    "resolution": dict(c.resolution) or None,
                    "created": c.created,
                    "resolved_at": c.resolved_at or None,
                }
                for c in rows
            ],
        }

    @_structured_errors
    def profile(self) -> dict[str, Any]:
        """User Memory 视图：平铺列出 kind=user 的 active 记忆（MVP 不做实体聚合）。"""
        memories = []
        for note in self.store.list_notes(status="active"):
            if note.meta.kind != "user":
                continue
            memories.append(
                {
                    "note_id": note.id,
                    "title": note.title,
                    "entities": list(note.meta.entities),
                    "confidence": note.meta.confidence,
                    "importance": note.meta.importance,
                    "created": note.meta.created,
                    "observed_at": note.meta.observed_at,
                    "snippet": note.body[:120] + ("…" if len(note.body) > 120 else ""),
                    "path": _relative_path(note.path, self.root),
                }
            )
        memories.sort(key=lambda m: str(m["note_id"]))
        return {"ok": True, "count": len(memories), "memories": memories}

    def _require_active(self, note_id: str) -> Note:
        """读一条笔记并断言存在且 active；merged/superseded 指引用 supersede。

        随后做写路径 id 形态复核：update / supersede / invalidate 三条写流程都
        以这里的 note.id 作为后续 save_note / 备份的路径分量，必须在任何写入
        （含写前备份）发生之前拦下形态非法或与文件名不一致的污染 id。
        """
        note = self._load(note_id)
        if note is None:
            raise WikiToolError(
                "not_found",
                f"笔记 {note_id!r} 不存在（wiki 根目录：{self.root}）",
                note_id=note_id,
            )
        _assert_note_id_writable(note)
        if note.meta.status != "active":
            raise WikiToolError(
                "invalid_argument",
                f"笔记 {note.id} 状态为 {note.meta.status}，不能执行该操作；"
                "请先 wiki_read 跟随重定向到当前版本，再对当前版本操作",
                note_id=note.id,
                status=note.meta.status,
            )
        return note

    def _mark_superseded(
        self, note: Note, *, superseded_by: str, now: str, reason: str, prefix: str = "supersede"
    ) -> Note:
        """把 active 笔记标记为 superseded 并链到目标（墓碑或新笔记）。

        本方法直接以 note.id 作为 save_meta 的写路径分量（meta.id 决定文件名），
        写前再复核一次 id 形态（入口处的 _require_active 已查过，这里是纵深防御）。
        只改 status / superseded_by / reviewed_at / extra，其余字段（含 tombstone：
        把墓碑本身标 superseded 时标记不能丢）逐字段保留。
        """
        _assert_note_id_writable(note)
        meta = note.meta
        try:
            marked = self.store.save_meta(
                meta.replace(
                    status="superseded",
                    superseded_by=superseded_by,
                    reviewed_at=now,
                    extra={
                        **meta.extra,
                        f"{prefix}_reason": reason,
                        f"{prefix}_reason_at": now,
                    },
                ),
                note.body,
            )
        except (OSError, ValueError) as exc:
            raise WikiToolError(
                "write_failed",
                f"旧笔记状态更新失败（{note.id} → superseded）：{type(exc).__name__}: {exc}",
                note_id=note.id,
            ) from exc
        index_error = self._index_note(marked)
        if index_error is not None:
            logger.warning("旧笔记索引同步失败（%s）：%s", note.id, index_error)
        return marked

    def _note_events(
        self, note: Note, topo_idx: int
    ) -> list[tuple[tuple[Any, ...], dict[str, Any]]]:
        """一条笔记的全部时间线事件（list_changes 同口径拆分），附排序键。

        事件类型与 list_changes 的 change_kind 对齐：created（新建）/ reviewed
        （每次修订一条，reason 取 update_reasons；无修订记录但 reviewed_at 晚于
        created 的 active 笔记按老口径补一条，兼容手工维护的笔记）/ status
        （superseded/merged 状态变更，reason 取 supersede_reason 或
        invalidate_reason）。排序键 (updated, topo_idx, kind_order, note_id)：
        同刻先按链拓扑序、再按 created < reviewed < status 的类型次序兜底，
        Python 稳定排序保证同刻同类的多条修订保持 frontmatter 里的追加顺序。
        """
        base: dict[str, Any] = {
            "note_id": note.id,
            "title": note.title,
            "kind": note.meta.kind,
            "status": note.meta.status,
            "confidence": note.meta.confidence,
            "importance": note.meta.importance,
            "entities": list(note.meta.entities),
            "created": note.meta.created,
            "reviewed_at": note.meta.reviewed_at,
            "observed_at": note.meta.observed_at,
            "redirect_to": note.meta.redirect_to,
            "superseded_by": note.meta.superseded_by,
            "path": _relative_path(note.path, self.root),
        }
        entries: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

        def add(change_kind: str, at: str, reason: str | None, kind_order: int) -> None:
            payload = {**base, "change_kind": change_kind, "updated": at, "reason": reason}
            entries.append(((at or "", topo_idx, kind_order, note.id), payload))

        created = _parse_ts(note.meta.created)
        if created is not None:
            add("created", note.meta.created, None, 0)
        revisions = _revision_records(note.meta.extra)
        if revisions:
            for record in revisions:
                at = record["at"] or note.meta.reviewed_at or note.meta.created or ""
                add("reviewed", at, record["reason"], 1)
        elif note.meta.status == "active":
            # 老口径兼容：没有修订记录但 reviewed_at 晚于 created → 一次复核事件
            reviewed = _parse_ts(note.meta.reviewed_at)
            if reviewed is not None and created is not None and reviewed > created:
                add("reviewed", note.meta.reviewed_at or "", None, 1)
        if note.meta.status != "active":
            change_time = _change_time(note)
            at = change_time.isoformat() if change_time else (note.meta.created or "")
            reason = (
                str(
                    note.meta.extra.get("supersede_reason")
                    or note.meta.extra.get("invalidate_reason")
                    or ""
                ).strip()
                or None
            )
            add("status", at, reason, 2)
        return entries

    # ---- 增量列表 ----

    @_structured_errors
    def list_changes(self, since: str | None = None, limit: int = 50) -> dict[str, Any]:
        """列出最近变更的笔记（含新建与状态变更），供客户端做增量同步。

        排序恒为"变更时间倒序"（最新在前）。超过 limit 时的取舍：
        - 没给 since（"看看最近变更"）：返回最新的一页，has_more 提示还有更早的没回；
        - 给了 since（增量同步）：返回 since 之后【最早】的一页，并给出 next_since 游标，
          客户端把 next_since 当作下次的 since 继续拉，就能从旧到新完整覆盖不漏条目。
          该页的边界对齐到完整时间戳分组：边界处同一时刻的条目全部纳入本页
          （同秒批量写入的笔记会被游标时间戳挡住，分组切开就会漏），
          因此本页条数可能略多于 limit（has_more 仍按 limit 判定）。
        """
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= MAX_LIMIT:
            raise WikiToolError(
                "invalid_argument", f"limit 必须是 1..{MAX_LIMIT} 的整数，得到 {limit!r}"
            )
        since_dt: datetime | None = None
        if since is not None:
            since_dt = _parse_ts(since)
            if since_dt is None:
                raise WikiToolError(
                    "invalid_argument",
                    f"since 必须是 ISO 时间字符串（如 2026-09-20T12:00:00+00:00），得到 {since!r}",
                )
        matched: list[dict[str, Any]] = []
        for note in self.store.list_notes(status=None):
            updated = _change_time(note)
            if since_dt is not None and (updated is None or updated <= since_dt):
                continue
            matched.append(_change_entry(note, updated, self.root))
        # 最近变更在前；同一时刻用 note_id 倒序兜底（编号越大越新）
        matched.sort(key=lambda e: (e["updated"] or "", e["note_id"]), reverse=True)

        next_since: str | None = None
        if len(matched) <= limit:
            page = matched
        elif since_dt is None:
            page = matched[:limit]
        else:
            # 同步场景取最早一页。游标只带时间戳、since 又是排他语义，所以左边界必须
            # 对齐到完整的时间戳分组：边界处同一时刻的更早条目（同一秒批量写入的笔记）
            # 若不并入本页，下次调用会因 "updated <= since" 被过滤掉，同步静默丢条目。
            start = len(matched) - limit
            boundary = matched[start]["updated"]
            while start > 0 and matched[start - 1]["updated"] == boundary:
                start -= 1
            page = matched[start:]
            next_since = page[0]["updated"]
        has_more = len(matched) > limit
        payload: dict[str, Any] = {
            "ok": True,
            "since": since,
            "total": len(matched),
            "count": len(page),
            "has_more": has_more,
            "next_since": next_since,
            "latest": matched[0]["updated"] if matched else None,
            "changes": page,
        }
        if has_more and next_since is not None:
            payload["hint"] = (
                f"共 {len(matched)} 条变更，本次返回 since 之后最早的一页（页内仍倒序）。"
                "把 next_since 作为下次调用的 since 继续拉取（since 为排他语义），"
                "直到 has_more=false 即可完整同步。"
            )
        elif has_more:
            payload["hint"] = (
                f"共 {len(matched)} 条变更（本页为最新的 {len(page)} 条，页内倒序），"
                "还有更早的未返回；要完整同步请带上 since 分批拉取。"
            )
        return payload

    # ---- 健康检查 ----

    @_structured_errors
    def health(self) -> dict[str, Any]:
        """wiki 自检：笔记/页面/实体/冲突计数 + 索引状态。

        ``index.stale`` 布尔语义不变，判据换成与 loop 路径共享的 ``index_drift``
        （P2-F 裁定二）；落后时另给 ``index.stale_reason``（点名缺失/变化的
        note_id），这是 v1 只有 ``stale: true/false`` 时拿不到的审计信息（新增
        字段为追加，既有键与含义不变）。
        """
        notes = self.store.list_notes(status=None)
        by_status: dict[str, int] = {"active": 0, "merged": 0, "superseded": 0}
        for note in notes:
            by_status[note.meta.status] = by_status.get(note.meta.status, 0) + 1
        conflicts = self.store.list_conflicts(status=None)
        db_path = self.root / "index.db"
        index_info: dict[str, Any] = {"path": _relative_path(db_path, self.root)}
        if db_path.is_file():
            index_info["exists"] = True
            index_info["indexed_notes"] = _count_indexed_notes(db_path)
            try:
                with self._open_index() as index:
                    index_info["tokenizer"] = index.tokenizer
                    # 与 loop 路径同一判据（index_drift）：stale 为布尔（既有契约），
                    # 落后原因另放 stale_reason，便于排查"为什么它判落后"。
                    reason = self._index_is_stale(index)
                    index_info["stale"] = reason is not None
                    if reason is not None:
                        index_info["stale_reason"] = reason
            except Exception as exc:  # noqa: BLE001 -- 索引坏了也要能报健康度
                index_info["error"] = f"{type(exc).__name__}: {exc}"
                # 索引打不开就没法比对指纹：退回"有笔记就算需重建"的保守判断
                index_info["stale"] = bool(notes)
        else:
            index_info["exists"] = False
            index_info["stale"] = bool(notes)
        entities = EntityRegistry(self.root).list_entities()
        return {
            "ok": True,
            "root": str(Path(self.root).resolve()),
            "notes": {
                "total": len(notes),
                "active": by_status.get("active", 0),
                "merged": by_status.get("merged", 0),
                "superseded": by_status.get("superseded", 0),
            },
            "pages": len(self.store.list_pages()),
            "entities": len(entities),
            "conflicts": {
                "total": len(conflicts),
                "open": sum(1 for c in conflicts if c.status == "open"),
                "resolved": sum(1 for c in conflicts if c.status == "resolved"),
            },
            "index": index_info,
        }


# ---- 纯函数：校验 / 时间 / 序列化辅助 ---------------------------------------


def _validate_write_inputs(
    *,
    body: str,
    title: str,
    entities: Sequence[str] | None,
    confidence: str,
    volatility: str,
    sources: Sequence[SourceInput] | None,
    kind: str = "knowledge",
    importance: float | None = None,
    tombstone: bool = False,
) -> dict[str, Any]:
    """wiki_write / memory_store 的输入校验（写保护第 1 件）。

    返回可直接喂给 store.save_note 的字段。所有错误一次性收集，
    让客户端一次就能改对，而不是挤牙膏式报错。
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(body, str) or not body.strip():
        errors.append("body 不能为空：请写入笔记正文（每条笔记一个事实 + 佐证）")
    if not isinstance(title, str) or not title.strip():
        errors.append("title 不能为空：请给笔记一个可检索的短标题")
    elif len(title.strip()) > 200:
        errors.append(f"title 过长（{len(title.strip())} 字符），请压到 200 字符以内")

    if entities is None:
        normalized_entities: list[str] = []
    elif isinstance(entities, str):
        errors.append(
            'entities 必须是字符串列表（如 ["glm-5-3", "上下文压缩"]），'
            "不要传逗号分隔的字符串或单个字符串"
        )
        normalized_entities = []
    elif isinstance(entities, Sequence):
        normalized_entities = []
        for item in entities:
            if not isinstance(item, str) or not item.strip():
                errors.append(f"entities 的元素必须是非空字符串，得到 {item!r}")
                continue
            normalized_entities.append(item.strip())
    else:
        errors.append(f"entities 必须是字符串列表，得到 {type(entities).__name__}")
        normalized_entities = []

    conf = _normalize_confidence(confidence, errors)
    vol = volatility.strip().lower() if isinstance(volatility, str) else volatility
    if vol not in VOLATILITY_LEVELS:
        errors.append(
            f"volatility 非法：{volatility!r}，只能是 {' / '.join(VOLATILITY_LEVELS)}"
            "（stable 不衰减 / drifting 90 天半衰 / volatile 30 天半衰）"
        )

    # kind/importance：记忆载体字段（P1-A）。write 收集错误语义 → 不立即抛，
    # 统一进 errors 让客户端一次改对（检索路径的 kind 校验见 validate_kind）。
    normalized_kind = "knowledge"
    if isinstance(kind, str):
        normalized_kind = kind.strip().lower()
        if normalized_kind not in KIND_LEVELS:
            errors.append(
                f"kind 非法：{kind!r}，只能是 {' / '.join(KIND_LEVELS)}"
                "（knowledge 世界知识 / user 用户画像与偏好 / experience 经验教训）"
            )
    else:
        errors.append(f"kind 必须是字符串，得到 {type(kind).__name__}")

    normalized_importance: float | None = None
    if importance is not None:
        if isinstance(importance, bool) or not isinstance(importance, int | float):
            errors.append(f"importance 必须是 0.0–1.0 之间的数值，得到 {importance!r}")
        elif not IMPORTANCE_MIN <= float(importance) <= IMPORTANCE_MAX:
            errors.append(
                f"importance 越界：{importance!r}，必须落在 {IMPORTANCE_MIN}–{IMPORTANCE_MAX}"
            )
        else:
            normalized_importance = float(importance)

    normalized_sources = _normalize_sources(sources, errors)
    if sources is None:
        warnings.append(
            "未提供 sources：笔记缺少来源 URL，检索与引用会失去证据链（建议补上）"
        )
    elif not normalized_sources:
        warnings.append("sources 解析后为空：笔记没有可用来源 URL")
    # tombstone 是布尔标记（P2-F 裁定一），只由 invalidate_memory 传 True；
    # 非布尔值一律拒绝，避免 "true"/1 这类形态在服务层被静默归一成另一层语义。
    if not isinstance(tombstone, bool):
        errors.append(f"tombstone 必须是布尔值，得到 {tombstone!r}")
    if errors:
        raise WikiToolError(
            "validation_failed",
            "笔记校验未通过：" + "；".join(errors),
            errors=errors,
        )
    return {
        "title": title.strip(),
        "entities": normalized_entities,
        "confidence": str(conf),
        "volatility": str(vol),
        "kind": normalized_kind,
        "importance": normalized_importance,
        "tombstone": bool(tombstone),
        "sources": normalized_sources,
        "warnings": warnings,
    }


def _normalize_sources(
    sources: Sequence[SourceInput] | str | None, errors: list[str], *, label: str = "sources"
) -> list[SourceRef]:
    """把客户端给的来源归一成 ``SourceRef`` 列表（write 与 memory 证据链参数共用）。

    接受两种形态（同一列表里可混用）：

    - URL 字符串：``"https://example.com/a"`` → ``SourceRef(url, content_hash="")``；
    - 映射：``{"url": "https://example.com/a", "content_hash": "sha1:..."}``，
      ``content_hash`` 缺省 / None / 空串统一归一为 ``""``（表示"未提供哈希"）。

    url 两侧空白去掉；空 URL、空字符串元素、非字符串非映射元素一律往 ``errors``
    追加一条可读中文错误（调用方决定是整体拒绝还是仅告警）。
    ``sources is None`` 返回空列表且不记错误（= 调用方没提供来源）。
    """
    normalized: list[SourceRef] = []
    if sources is None:
        return normalized
    if isinstance(sources, str):
        errors.append(f'{label} 必须是 URL 字符串列表（如 ["https://example.com/a"]）')
        return normalized
    if not isinstance(sources, Sequence):
        errors.append(f"{label} 必须是 URL 字符串列表，得到 {type(sources).__name__}")
        return normalized
    for item in sources:
        if isinstance(item, Mapping):
            url = str(item.get("url") or "").strip()
            if not url:
                errors.append(f"{label} 里的映射缺少 url 字段：{dict(item)!r}")
                continue
            normalized.append(SourceRef(url=url, content_hash=str(item.get("content_hash") or "")))
            continue
        if not isinstance(item, str) or not item.strip():
            errors.append(f"{label} 的元素必须是非空 URL 字符串，得到 {item!r}")
            continue
        normalized.append(SourceRef(url=item.strip(), content_hash=""))
    return normalized


def _evidence_sources(source_urls: Sequence[SourceInput] | str | None) -> list[SourceRef]:
    """memory_supersede / memory_update 的 ``source_urls`` 校验（写前完成，零落盘）。

    与 ``write()`` 走同一套 ``SourceRef`` 归一（``_normalize_sources``），所以
    URL 非空、hash 可为空串但会归一等约定完全一致；差别只有一处：显式给了
    **空列表** 视为参数错误（"要么给至少一个来源，要么省略该参数"），因为
    "我带了证据"与"我没带证据"是两种不同的语义，空列表让调用方意图不可辨。

    任何非法输入抛 ``validation_failed``（复用既有错误码），调用方在创建备份 /
    落盘之前调用本函数即可保证"非法参数不写盘"。
    """
    if source_urls is None:
        return []
    errors: list[str] = []
    normalized = _normalize_sources(source_urls, errors, label="source_urls")
    if not normalized and not errors:
        errors.append(
            "source_urls 是空列表：请给出至少一个来源 URL，或省略该参数"
            "（省略=不提供证据，返回体 evidence=none）"
        )
    if errors:
        raise WikiToolError(
            "validation_failed",
            "来源校验未通过：" + "；".join(errors),
            errors=errors,
        )
    return normalized


def _normalize_confidence(value: Any, errors: list[str]) -> Any:
    """confidence 白名单归一（``write()`` 与 ``memory_supersede`` 共用同一份规则）。

    非字符串、大小写/空白不规整、不在 high/medium/low 内 → 往 ``errors`` 追加一条
    可读中文错误（文案与历史行为逐字一致），否则返回归一后的字符串。
    """
    conf = value.strip().lower() if isinstance(value, str) else value
    if conf not in CONFIDENCE_LEVELS:
        errors.append(f"confidence 非法：{value!r}，只能是 {' / '.join(CONFIDENCE_LEVELS)}")
    return conf


def _require_confidence(value: Any) -> str:
    """memory_supersede 的 ``confidence`` 参数提前校验（锁外、任何写入之前）。

    用 ``write()`` 的同一套白名单（``_normalize_confidence``），只把"拒绝时机"
    提前——参数非法与目标是否存在无关，应当与 ``source_urls`` 同优先级返回
    ``validation_failed``，而不是先落到 not_found。write() 内的校验仍是最终裁决。
    """
    errors: list[str] = []
    normalized = _normalize_confidence(value, errors)
    if errors:
        raise WikiToolError(
            "validation_failed", "参数校验未通过：" + "；".join(errors), errors=errors
        )
    return str(normalized)


def _merge_sources(
    existing: Sequence[SourceRef], provided: Sequence[SourceRef]
) -> tuple[list[SourceRef], int]:
    """追加式合并来源（update 语义）：按 ``(url, content_hash)`` 去重，旧来源在前。

    返回 ``(合并后的列表, 本次新增条数)``；已存在的 (url, hash) 不重复追加。
    """
    merged = list(existing)
    seen = {(ref.url, ref.content_hash) for ref in merged}
    added = 0
    for ref in provided:
        key = (ref.url, ref.content_hash)
        if key in seen:
            continue
        seen.add(key)
        merged.append(ref)
        added += 1
    return merged, added


def _change_time(note: Note) -> datetime | None:
    """笔记的"最后变更时间" = max(created, reviewed_at)（均按 ISO 串解析，UTC 归一）。"""
    candidates = [t for t in (_parse_ts(note.meta.created), _parse_ts(note.meta.reviewed_at)) if t]
    return max(candidates) if candidates else None


def _append_update_reason(extra: Mapping[str, Any], reason: str, at: str) -> dict[str, Any]:
    """把一次修订原因**追加**进 extra 的 ``update_reasons``（按修订顺序累积）。

    条目形态 ``{"reason": ..., "at": ...}``；多次修订按序累积、不覆盖，但**只保留
    最近 ``MAX_UPDATE_REASONS`` 条**（终审 F6：无界追加会让 frontmatter 随修订次数
    无限膨胀）。被裁掉的原因仍可从 ``.backups/<时间戳>/`` 的历史原稿追溯——每次修订
    前都备份，所以这是"不在笔记里长胖"而不是"丢掉审计"。
    兼容迁移：旧版本把原因写在单键 ``update_reason``/``update_reason_at``，
    首次追加时先把它折叠为列表首条目，随后旧键移除——round-trip 归一，
    历史原因不丢。
    """
    merged = dict(extra)
    records: list[Any] = list(merged.pop("update_reasons", None) or [])
    legacy_reason = merged.pop("update_reason", None)
    legacy_at = merged.pop("update_reason_at", None)
    if not records and legacy_reason is not None:
        # YAML 会把时间戳解析成 datetime：归一回 ISO 字符串再入列表
        normalized_at = _parse_ts(str(legacy_at) if legacy_at is not None else "")
        records.append(
            {
                "reason": str(legacy_reason),
                "at": normalized_at.isoformat() if normalized_at else str(legacy_at or ""),
            }
        )
    records.append({"reason": reason, "at": at})
    merged["update_reasons"] = records[-MAX_UPDATE_REASONS:]
    return merged


def _revision_records(extra: Mapping[str, Any]) -> list[dict[str, str]]:
    """读出 extra.update_reasons 的修订记录（宽容解析手工编辑的条目形态）。"""
    out: list[dict[str, str]] = []
    for item in extra.get("update_reasons") or []:
        if isinstance(item, Mapping):
            out.append(
                {
                    "reason": str(item.get("reason") or ""),
                    "at": str(item.get("at") or ""),
                }
            )
        elif item:
            out.append({"reason": str(item), "at": ""})
    return out


def _topo_order(lineage: Mapping[str, Note]) -> list[str]:
    """版本谱系 → 旧 → 新的有序 id 列表（拓扑排序）。

    边 A → B 表示" B 直接接替 A"（A 的 superseded_by / redirect_to 指向 B）；
    同刻/无依赖时按 note_id 升序兜底。理论上谱系是无环 DAG（主链成环早已被
    _walk_redirects 挡下），万一出现环则回退为 note_id 排序，绝不静默丢版本。
    """
    edges: dict[str, list[str]] = {nid: [] for nid in lineage}
    indegree: dict[str, int] = {nid: 0 for nid in lineage}
    for nid, note in lineage.items():
        target = note.meta.superseded_by or note.meta.redirect_to
        if target and target != nid and target in lineage:
            edges[nid].append(target)
            indegree[target] += 1
    ready = sorted(nid for nid, deg in indegree.items() if deg == 0)
    ordered: list[str] = []
    while ready:
        nid = ready.pop(0)
        ordered.append(nid)
        for nxt in edges[nid]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                bisect.insort(ready, nxt)
    if len(ordered) != len(lineage):  # pragma: no cover - 防御：谱系意外成环
        return sorted(lineage)
    return ordered


def _change_kind(note: Note) -> str:
    """变更类型：状态变更（merged/superseded）> 复核更新 > 新建。"""
    if note.meta.status != "active":
        return "status"
    created = _parse_ts(note.meta.created)
    reviewed = _parse_ts(note.meta.reviewed_at)
    if created and reviewed and reviewed > created:
        return "reviewed"
    return "created"


def _change_entry(note: Note, updated: datetime | None, root: Path) -> dict[str, Any]:
    return {
        "note_id": note.id,
        "title": note.title,
        "status": note.meta.status,
        "change_kind": _change_kind(note),
        "confidence": note.meta.confidence,
        "volatility": note.meta.volatility,
        "entities": list(note.meta.entities),
        "created": note.meta.created,
        "reviewed_at": note.meta.reviewed_at,
        "observed_at": note.meta.observed_at,
        "updated": updated.isoformat() if updated else (note.meta.created or ""),
        "redirect_to": note.meta.redirect_to,
        "superseded_by": note.meta.superseded_by,
        "path": _relative_path(note.path, root),
    }


def _relative_path(path: Path | None, root: Path) -> str:
    """落盘路径 → 相对 wiki-data 的 posix 形式（不向客户端泄露本机绝对路径）。"""
    if path is None:
        return ""
    try:
        return path.resolve().relative_to(Path(root).resolve()).as_posix()
    except (ValueError, OSError):
        return Path(path).as_posix()


def _count_indexed_notes(db_path: Path) -> int | None:
    """索引里的笔记条数（只读连接）；读不到返回 None（调用方跳过该项一致性检查）。"""
    try:
        conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    except sqlite3.Error:
        return None
    try:
        row = conn.execute("SELECT COUNT(*) FROM note_meta").fetchone()
        return int(row[0]) if row else None
    except sqlite3.Error:
        return None
    finally:
        conn.close()
