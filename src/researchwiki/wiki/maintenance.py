"""维护执行器（P4a Task 17 / T3）：把 consolidation 选择器的计划逐动作执行。

``MaintenanceRunner`` 消费 T2 的 ``ConsolidationPlan``（merge / refresh / rejudge /
conflict 四类 ``PlannedAction``），带进程内 job queue：每个动作一条 job 记录（谁、
何时、用什么模型、花了多少 token、结果与错误），账本与详情双落盘——

- ``{root}/maintenance/jobs.jsonl``：追加式账本，一行一条 job 记录（append-only，
  断电最多损坏最后一行，读取侧容忍并跳过不可解析的行）；
- ``{root}/maintenance/jobs/{job_id}.json``：单 job 详情（tmp + os.replace 原子写）。

四类执行器语义（PLAN §3.5 逐条）：

- **merge**：absorbed 笔记 body 并入 canonical（保留规范 ID），sources / entities
  取并集；absorbed 置 ``status=merged + redirect_to`` 留痕（与入库去重同形态，
  落盘顺序同 ingest：留痕先落盘）。canonical 重写走 ``meta.replace + save_meta``
  ——``save_note`` 的参数默认值会静默丢字段，重建既有笔记一律走 ``save_meta``。
- **refresh**：``fetch_url(原 URL)`` 比对快照哈希——未变 → 仅刷新 reviewed_at
  （result 标 ``source_unchanged``）；变了 → ``mark_source_changed`` 留痕 + 新证据
  笔记（kind=knowledge、带新 content_hash 来源，来源可追溯），旧快照一律保留
  （快照版本化是 fetch_url 的内部事务，执行器不自写 sources/）；抓取异常 → job
  failed、原笔记零改动。``fetcher`` 注入缺省用真 ``fetch_url``（测试注入 fake）。
- **rejudge**：cheap 档模型判定（单条 user 消息 + 严格 JSON 输出解析，允许字段
  恰好为 {verdict, reason}，超字段即畸形）——confirm → 刷新 reviewed_at；
  supersede → 原笔记置 superseded（理由记 ``supersede_reason``，不删除历史、不创建
  替代笔记）；JSON 畸形 → job failed、原笔记零改动。token 记账：provider 回了
  usage 就如实计入 tokens_in/tokens_out，没回就记 0 并在 result 注明来源
  （诚实计量）；调用本身的幂等由 ReplayProvider 的请求哈希覆盖，job 记 model 与
  idempotency_key 留痕。
- **conflict**：把 T16 已判定的冲突候选落成台账（``store.save_conflict``），claim
  内容取 payload 的断言槽位——**绝不重新构造带来源/时间的比较**（判定方向是
  T16 选择器的承重裁定，执行器照单开台账），两笔记均保持 active，绝不自动 merge。

幂等与失败语义（Global Constraints 逐条落地）：

- **幂等重放**：``idempotency_key`` 已有 done job → 新记录 status=skipped
  （reason=idempotent），零重复副作用（不重写笔记、不重复开台账、不重复调模型）。
  job 记录附 ``payload``（计划动作参数），``retry_job`` 按原样重放，不从当前状态
  重新推导（conflict 重推导等于重新比较，破坏 T16 裁定）。
- **失败保原文**：任何写入/抓取/解析失败都把 job 标 failed，原笔记逐字节不动
  ——执行器只在成功路径上落盘，failed 的动作零副作用。
- **账本完整性**：JSONL 逐行独立可解析；上次进程可能在行中途被杀（尾行残缺），
  追加前补换行防止粘连，读取侧跳过不可解析的行，job 编号只认可解析记录。

状态词表（五值）：``pending``（已创建未执行）→ ``running``（执行中，进程内瞬态）
→ ``done`` / ``failed``（终态）；``skipped``（幂等重放或人工跳过）。``retry_job``
仅 failed 可重试；``skip_job`` 仅 pending/failed 可跳过。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any

from researchwiki.llm.provider import Message
from researchwiki.tools.fs import atomic_write_text
from researchwiki.wiki.consolidation import (
    ACTION_CONFLICT,
    ACTION_MERGE,
    ACTION_REFRESH,
    ACTION_REJUDGE,
    REJUDGE_TIER,
)
from researchwiki.wiki.entities import slugify
from researchwiki.wiki.frontmatter import SourceRef
from researchwiki.wiki.store import Note, WikiStore

# ---- 状态词表（简报给定五值）--------------------------------------------------

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"

# 台账 claim 摘录长度（与 loop/memory_update.py 的 EXCERPT_CHARS 同量级的留痕口径）
EXCERPT_CHARS = 200

# rejudge 输出的严格 schema：字段恰好这两个，多一个少一个都算畸形
_REJUDGE_KEYS = {"verdict", "reason"}
_REJUDGE_VERDICTS = ("confirm", "supersede")


def _now_iso() -> str:
    """当前 UTC 秒级 ISO 时间（与 store / memory_update 的落盘口径一致）。"""
    return datetime.now(UTC).isoformat(timespec="seconds")


# ---- 数据结构 ----------------------------------------------------------------


@dataclass
class JobRecord:
    """一条维护 job 的完整记录（审计与幂等的载体）。

    - ``job_id``：``J-xxxx`` 顺序号（只认可解析的账本记录续号）；
    - ``idempotency_key``：计划动作的幂等键（T2 公式），done 记录按它拦截重放；
    - ``payload``：计划动作参数（merge 的 canonical/absorbed、refresh 的 url、
      conflict 的断言槽位等）——``retry_job`` 按原样重放的依据，不从当前状态重新
      推导（对 conflict 而言重推导等于重新比较，会破坏 T16 的判定方向裁定）；
    - ``model`` / ``tokens_in`` / ``tokens_out``：rejudge 的模型与用量（其余动作
      恒为 None/0）；provider 没回 usage 时记 0，来源在 result["tokens_source"]
      注明（诚实计量）；
    - ``status``：pending | running | done | failed | skipped；
    - ``result``：执行产物（新笔记 ID、台账 ID、verdict、reason 等）；
    - ``error``：failed 时的错误摘要（``类型: 消息``）；
    - ``attempt``：执行次数（新 job 为 1，retry 递增）。
    """

    job_id: str
    action: str
    note_ids: list[str]
    idempotency_key: str
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    model: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    status: str = STATUS_PENDING
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    attempt: int = 1
    # payload 不在简报字段清单里，是本实现的补充：retry 需要按"计划当时的参数"
    # 原样重放（尤其 conflict——从当前笔记重新推导槽位等于重新比较）。
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """序列化（JSONL 行与单 job 详情 JSON 的形态）。"""
        return {
            "job_id": self.job_id,
            "action": self.action,
            "note_ids": list(self.note_ids),
            "idempotency_key": self.idempotency_key,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "model": self.model,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "status": self.status,
            "result": dict(self.result),
            "error": self.error,
            "attempt": self.attempt,
            "payload": dict(self.payload),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> JobRecord:
        """从账本行反序列化（宽容缺省：老记录 / 手工修补的行都能读）。"""
        return cls(
            job_id=str(data.get("job_id") or ""),
            action=str(data.get("action") or ""),
            note_ids=[str(n) for n in (data.get("note_ids") or [])],
            idempotency_key=str(data.get("idempotency_key") or ""),
            created_at=str(data.get("created_at") or ""),
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            model=data.get("model"),
            tokens_in=int(data.get("tokens_in") or 0),
            tokens_out=int(data.get("tokens_out") or 0),
            status=str(data.get("status") or STATUS_PENDING),
            result=dict(data.get("result") or {}),
            error=data.get("error"),
            attempt=int(data.get("attempt") or 1),
            payload=dict(data.get("payload") or {}),
        )


@dataclass
class MaintenanceSettings:
    """``[maintenance]`` 段的解析结果。

    - ``enabled = false`` 逃生阀：完全禁用（run 零动作零留痕）；
    - ``rejudge_model_tier``：rejudge 用的模型档（默认 cheap；payload 里的 tier
      优先——T16 选择器在计划里记了它认定的档位）。

    段缺失 = 不接线（``maintenance_settings`` 返回 None），与 [formation] /
    [memory_update] 的 guarded 模式逐字一致。
    """

    enabled: bool = True
    rejudge_model_tier: str = REJUDGE_TIER


_TIERS = ("strong", "cheap", "judge")


def maintenance_settings(config: Mapping[str, Any] | None) -> MaintenanceSettings | None:
    """解析 ``[maintenance]`` 段；**段缺失返回 None = 不接线**。

    与 [formation]/[memory_update]/[consolidation] 的 guarded 语义逐字一致：缺段时
    CLI 报"未配置"退出 1，不做任何执行。段存在时非法值宽容回退默认，不抛异常。
    """
    if not isinstance(config, Mapping):
        return None
    section = config.get("maintenance")
    if not isinstance(section, Mapping):
        return None
    enabled = section.get("enabled")
    if isinstance(enabled, bool):
        flag = enabled
    elif enabled is None:
        flag = True
    else:
        flag = str(enabled).strip().lower() in ("1", "true", "yes", "on")
    tier = str(section.get("rejudge_model_tier") or REJUDGE_TIER).strip().lower()
    if tier not in _TIERS:
        tier = REJUDGE_TIER
    return MaintenanceSettings(enabled=flag, rejudge_model_tier=tier)


# ---- 执行器 ------------------------------------------------------------------


class MaintenanceRunner:
    """维护计划的进程内执行器：job queue + 四类动作 + 幂等重试 + 失败保原文。

    ``fetcher`` 缺省 None 时用真 ``fetch_url``（sources_dir 接 store 的 sources/）；
    测试注入 ``Callable[[str], FetchResult]``。``router`` 只在 rejudge 时取用
    （``router.get(tier)``），其余动作不触碰。
    """

    def __init__(
        self,
        store: WikiStore,
        router: Any,
        settings: MaintenanceSettings,
        *,
        fetcher: Callable[[str], Any] | None = None,
    ) -> None:
        self.store = store
        self.router = router
        self.settings = settings
        self._fetcher = fetcher
        self.maintenance_dir = store.root / "maintenance"
        self.ledger_path = self.maintenance_dir / "jobs.jsonl"
        self.details_dir = self.maintenance_dir / "jobs"

    # ---- 对外入口 --------------------------------------------------------

    def run(self, plan: Any) -> list[JobRecord]:
        """执行计划：逐动作产出 job 记录（done / failed / skipped），全部落账本。

        - ``settings.enabled = False`` → 零动作零留痕（返回空表）；
        - ``idempotency_key`` 已有 done job → 新记录 status=skipped
          （reason=idempotent），零重复副作用；
        - 单个动作失败不中断后续动作（failed 记录错误后继续）。
        """
        if not self.settings.enabled:
            return []
        done_keys = {
            record.idempotency_key: record
            for record in self._load_records().values()
            if record.status == STATUS_DONE
        }
        records: list[JobRecord] = []
        for action in plan.actions:
            matched = done_keys.get(action.idempotency_key)
            if matched is not None:
                record = JobRecord(
                    job_id=self._next_job_id(),
                    action=action.action,
                    note_ids=list(action.note_ids),
                    idempotency_key=action.idempotency_key,
                    payload=dict(action.payload),
                    created_at=_now_iso(),
                    finished_at=_now_iso(),
                    status=STATUS_SKIPPED,
                    result={"reason": "idempotent", "matched_job_id": matched.job_id},
                )
                self._append(record)
                records.append(record)
                continue
            record = self._execute_action(
                action=action.action,
                note_ids=list(action.note_ids),
                payload=dict(action.payload),
                idempotency_key=action.idempotency_key,
                attempt=1,
                done_keys=done_keys,
            )
            self._append(record)
            records.append(record)
        return records

    def list_jobs(self, status: str | None = None) -> list[JobRecord]:
        """按 job_id 升序列出账本中的 job（同 job 多行取最后一行）；status 过滤可选。

        追加式账本 + 断电容忍：不可解析的行（只可能是被截断的尾行）跳过。
        """
        records = self._load_records()
        out = sorted(records.values(), key=lambda record: record.job_id)
        if status is not None:
            out = [record for record in out if record.status == status]
        return out

    def retry_job(self, job_id: str) -> JobRecord:
        """重试 failed 的 job：按原记录的 action/note_ids/payload 原样重放（attempt+1）。

        新 job 记录（新 job_id、旧记录保留作失败历史）。重放同样过幂等闸：若该
        idempotency_key 此刻已有 done job（例如全量 --run 已把它做成），产出
        skipped 记录而不是重复执行。仅 failed 可重试，其余状态抛 ValueError。
        """
        source = self._load_records().get(job_id)
        if source is None:
            raise ValueError(f"job 不存在：{job_id}")
        if source.status != STATUS_FAILED:
            raise ValueError(
                f"仅 failed 的 job 可重试（{job_id} 当前为 {source.status}）"
            )
        done_keys = {
            record.idempotency_key: record
            for record in self._load_records().values()
            if record.status == STATUS_DONE
        }
        record = self._execute_action(
            action=source.action,
            note_ids=list(source.note_ids),
            payload=dict(source.payload),
            idempotency_key=source.idempotency_key,
            attempt=source.attempt + 1,
            done_keys=done_keys,
        )
        self._append(record)
        return record

    def skip_job(self, job_id: str, reason: str) -> JobRecord:
        """跳过 pending/failed 的 job：reason 留痕进 result（人工处置的审计依据）。

        同一 job_id 追加一行 skipped 记录（账本 append-only，读取侧取最后一行）。
        仅 pending/failed 可跳过，其余状态抛 ValueError。
        """
        source = self._load_records().get(job_id)
        if source is None:
            raise ValueError(f"job 不存在：{job_id}")
        if source.status not in (STATUS_PENDING, STATUS_FAILED):
            raise ValueError(
                f"仅 pending/failed 的 job 可跳过（{job_id} 当前为 {source.status}）"
            )
        record = replace(
            source,
            status=STATUS_SKIPPED,
            finished_at=_now_iso(),
            result={**source.result, "reason": reason, "skipped_by": "operator"},
        )
        self._append(record)
        return record

    # ---- 执行分派 --------------------------------------------------------

    def _execute_action(
        self,
        *,
        action: str,
        note_ids: list[str],
        payload: dict[str, Any],
        idempotency_key: str,
        attempt: int,
        done_keys: Mapping[str, JobRecord],
    ) -> JobRecord:
        """执行单个动作并产出终态记录；执行器抛出的任何异常都降级为 failed。

        幂等闸先于执行：done 键命中（retry 路径可能撞上）→ skipped，零副作用。
        """
        job_id = self._next_job_id()
        record = JobRecord(
            job_id=job_id,
            action=action,
            note_ids=list(note_ids),
            idempotency_key=idempotency_key,
            payload=dict(payload),
            created_at=_now_iso(),
            started_at=_now_iso(),
            attempt=attempt,
        )
        matched = done_keys.get(idempotency_key)
        if matched is not None:
            record.status = STATUS_SKIPPED
            record.finished_at = _now_iso()
            record.result = {"reason": "idempotent", "matched_job_id": matched.job_id}
            return record
        record.status = STATUS_RUNNING
        try:
            if action == ACTION_MERGE:
                record.result = self._exec_merge(payload, record)
            elif action == ACTION_REFRESH:
                record.result = self._exec_refresh(note_ids, payload, record)
            elif action == ACTION_REJUDGE:
                record.result = self._exec_rejudge(note_ids, payload, record)
            elif action == ACTION_CONFLICT:
                record.result = self._exec_conflict(note_ids, payload, record)
            else:
                raise ValueError(f"未知维护动作：{action}")
            record.status = STATUS_DONE
        except Exception as exc:  # noqa: BLE001 -- 失败保原文：降级为 failed，不中断批
            record.status = STATUS_FAILED
            record.error = f"{type(exc).__name__}: {exc}"
        record.finished_at = _now_iso()
        return record

    # ---- merge -----------------------------------------------------------

    def _exec_merge(self, payload: Mapping[str, Any], record: JobRecord) -> dict[str, Any]:
        """absorbed 并入 canonical：body 并入、sources/entities 并集、留痕 merged。

        落盘顺序同 ingest（留痕先落盘）：先写 absorbed 的 merged 留痕，再重写
        canonical——第二步失败时 absorbed 已可达 canonical（follow_redirect 链路
        完整，内容不丢），重试会因状态漂移被拒、交人工。
        """
        canonical_id = str(payload.get("canonical_id") or "")
        absorbed_id = str(payload.get("absorbed_id") or "")
        if not canonical_id or not absorbed_id:
            raise ValueError("merge 动作缺少 canonical_id/absorbed_id")
        canonical = self._require_active(canonical_id)
        absorbed = self._require_active(absorbed_id)

        # 留痕先落盘：absorbed 原表述原样保留在 merged 记录里（旧 ID 永不消失）
        self.store.save_meta(
            absorbed.meta.replace(status="merged", redirect_to=canonical.id),
            absorbed.body,
        )
        body = f"{canonical.body.rstrip()}\n\n{absorbed.body.strip()}"
        extra = dict(canonical.meta.extra)
        history = [str(item) for item in (extra.get("merged_from") or []) if str(item)]
        if absorbed_id not in history:
            history.append(absorbed_id)
        extra["merged_from"] = history
        merged = self.store.save_meta(
            canonical.meta.replace(
                sources=self._merge_refs(canonical.meta.sources, absorbed.meta.sources),
                entities=self._merge_entities(canonical.entities, absorbed.entities),
                extra=extra,
            ),
            body,
        )
        return {
            "canonical_id": merged.id,
            "absorbed_id": absorbed_id,
            "sources": [
                {"url": ref.url, "content_hash": ref.content_hash}
                for ref in merged.meta.sources
            ],
        }

    @staticmethod
    def _merge_refs(left: list[SourceRef], right: list[SourceRef]) -> list[SourceRef]:
        """来源并集：按 (url, content_hash) 精确去重，canonical 顺序优先。"""
        out = list(left)
        seen = {(ref.url, ref.content_hash) for ref in out}
        for ref in right:
            if (ref.url, ref.content_hash) in seen:
                continue
            seen.add((ref.url, ref.content_hash))
            out.append(ref)
        return out

    @staticmethod
    def _merge_entities(left: list[str], right: list[str]) -> list[str]:
        """实体并集：按 slugify 归一去重（与入库查重同一套身份定义），保序。"""
        out: list[str] = []
        seen: set[str] = set()
        for name in [*left, *right]:
            key = slugify(str(name)).casefold()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(str(name))
        return out

    # ---- refresh ---------------------------------------------------------

    def _exec_refresh(
        self, note_ids: list[str], payload: Mapping[str, Any], record: JobRecord
    ) -> dict[str, Any]:
        """重抓来源并比对快照哈希：未变仅刷新 reviewed_at；变了留痕 + 落新证据笔记。"""
        if not note_ids:
            raise ValueError("refresh 动作缺少 note_ids")
        note = self._require_active(note_ids[0])
        url = str(payload.get("url") or "")
        if not url:
            raise ValueError("refresh 动作缺少 url")
        ref = next((s for s in note.meta.sources if s.url == url), None)
        if ref is None:
            raise ValueError(f"笔记 {note.id} 已不引用 {url}：状态漂移，拒绝执行")

        result = self._fetch(url)
        if result.content_hash == ref.content_hash:
            # 内容未变：仅刷新 reviewed_at（正文与来源哈希零改动）
            stamp = _now_iso()
            self.store.save_meta(note.meta.replace(reviewed_at=stamp), note.body)
            return {
                "source_unchanged": True,
                "url": url,
                "content_hash": result.content_hash,
                "reviewed_at": stamp,
            }

        # 内容变了：先 mark_source_changed（幂等记账），再落新证据笔记
        affected = self.store.mark_source_changed(url, result.content_hash)
        new_note = self.store.save_note(
            result.text,
            title=f"来源重抓：{note.title or note.id}",
            entities=list(note.meta.entities),
            kind="knowledge",
            sources=[SourceRef(url=url, content_hash=result.content_hash)],
            extra={"refresh_of": note.id, "maintenance_job": record.job_id},
        )
        return {
            "source_unchanged": False,
            "url": url,
            "old_content_hash": ref.content_hash,
            "new_content_hash": result.content_hash,
            "marked_notes": affected,
            "new_note_id": new_note.id,
        }

    def _fetch(self, url: str) -> Any:
        """抓取入口：注入的 fake 优先，缺省用真 fetch_url（快照版本化是它的内部事务）。"""
        if self._fetcher is not None:
            return self._fetcher(url)
        from researchwiki.tools.fetch import fetch_url  # 延迟导入：重 HTTP 依赖按需拉起

        return fetch_url(url, sources_dir=self.store.sources_dir)

    # ---- rejudge ---------------------------------------------------------

    def _exec_rejudge(
        self, note_ids: list[str], payload: Mapping[str, Any], record: JobRecord
    ) -> dict[str, Any]:
        """cheap 档模型重判：严格 JSON 输出（{verdict, reason}），畸形即 failed。

        confirm → 刷新 reviewed_at；supersede → 原笔记置 superseded（理由记
        supersede_reason，不删除历史、不创建替代笔记）。token 用量如实记账：
        provider 回了 usage 记实际值，没回记 0 并在 result 注明来源。
        """
        if not note_ids:
            raise ValueError("rejudge 动作缺少 note_ids")
        note = self._require_active(note_ids[0])
        tier = str(payload.get("tier") or self.settings.rejudge_model_tier or REJUDGE_TIER)
        provider = self.router.get(tier)
        record.model = provider.model

        text_parts: list[str] = []
        usage = None
        for event in provider.stream([Message(role="user", content=_rejudge_prompt(note))]):
            if event.type == "text_delta":
                text_parts.append(event.delta)
            elif event.type == "usage" and event.usage is not None:
                usage = event.usage
        raw = "".join(text_parts)
        verdict, reason = _parse_rejudge_output(raw)
        if usage is not None:
            record.tokens_in = usage.input_tokens
            record.tokens_out = usage.output_tokens
            tokens_source = "provider_usage"
        else:
            tokens_source = "none_reported"  # 没有计量就记 0 并注明，不硬造

        stamp = _now_iso()
        if verdict == "confirm":
            self.store.save_meta(note.meta.replace(reviewed_at=stamp), note.body)
        else:  # supersede：与 mcp_server._mark_superseded 同形态，只多 reason 两个 extra 槽
            self.store.save_meta(
                note.meta.replace(
                    status="superseded",
                    reviewed_at=stamp,
                    extra={
                        **note.meta.extra,
                        "supersede_reason": reason,
                        "supersede_reason_at": stamp,
                    },
                ),
                note.body,
            )
        return {
            "verdict": verdict,
            "reason": reason,
            "tier": tier,
            "raw_chars": len(raw),
            "tokens_source": tokens_source,
        }

    # ---- conflict --------------------------------------------------------

    def _exec_conflict(
        self, note_ids: list[str], payload: Mapping[str, Any], record: JobRecord
    ) -> dict[str, Any]:
        """把 T16 已判定的冲突候选落成台账（claim 取 payload 断言槽位），双方保持 active。

        **不重新比较**：比较方向（ID 小者为 prior、不含来源/时间的 EvidenceItem）
        是 T16 选择器的承重裁定，执行器照单开台账。台账侧幂等：claim 里带
        idempotency_key，已存在同键条目时复用、不重复开（job 账本丢失后重放安全）。
        """
        if len(note_ids) != 2:
            raise ValueError("conflict 动作必须是笔记对")
        prior = self._require_active(note_ids[0])  # ID 小者为 prior（T16 方向）
        evidence = self._require_active(note_ids[1])
        slots = payload.get("slots")
        if not isinstance(slots, list) or not slots:
            raise ValueError("conflict 动作缺少断言槽位（payload.slots）")
        similarity = payload.get("similarity")

        # 台账侧幂等：同 idempotency_key 的条目已存在 → 复用
        for existing in self.store.list_conflicts(status=None):
            keys = {
                existing.claim_a.get("idempotency_key"),
                existing.claim_b.get("idempotency_key"),
            }
            if record.idempotency_key in keys:
                return {
                    "conflict_id": existing.id,
                    "reused": True,
                    "slots": len(slots),
                }

        claim_a: dict[str, Any] = {
            "note_id": prior.id,
            "title": prior.title,
            "excerpt": prior.body.strip()[:EXCERPT_CHARS],
            "role": "prior",
            "slots": slots,
            "similarity": similarity,
            "idempotency_key": record.idempotency_key,
        }
        claim_b: dict[str, Any] = {
            "note_id": evidence.id,
            "title": evidence.title,
            "excerpt": evidence.body.strip()[:EXCERPT_CHARS],
            "role": "evidence",
            "slots": slots,
            "similarity": similarity,
            "idempotency_key": record.idempotency_key,
        }
        question = (
            f"{prior.title or prior.id}（{prior.id}）与 "
            f"{evidence.title or evidence.id}（{evidence.id}）"
            "同实体槽位取值冲突（consolidation 判定 conflicting），待人工裁决"
        )
        conflict = self.store.save_conflict(question, claim_a, claim_b)
        return {"conflict_id": conflict.id, "reused": False, "slots": len(slots)}

    # ---- 存储：账本 + 详情 ------------------------------------------------

    def _append(self, record: JobRecord) -> None:
        """账本追加一行 + 单 job 详情原子落盘（tmp + rename）。

        断电容忍：上次进程可能在行中途被杀（尾行残缺、缺换行），追加前补换行，
        保证新记录不与截断行粘连成一行不可解析的坏数据。
        """
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        needs_newline = False
        if self.ledger_path.is_file():
            with self.ledger_path.open("rb") as f:
                f.seek(0, 2)  # 末尾
                if f.tell() > 0:  # 空文件（外部 touch/损坏）无从谈"末行缺换行"
                    f.seek(-1, 2)
                    needs_newline = f.read(1) != b"\n"
        line = json.dumps(record.to_dict(), ensure_ascii=False)
        with self.ledger_path.open("a", encoding="utf-8") as f:
            if needs_newline:
                f.write("\n")
            f.write(line + "\n")
        detail = json.dumps(record.to_dict(), ensure_ascii=False, indent=2) + "\n"
        atomic_write_text(self.details_dir / f"{record.job_id}.json", detail)

    def _load_records(self) -> dict[str, JobRecord]:
        """读取账本并按 job_id 折叠（同 job 多行取最后一行）。

        不可解析的行跳过——append-only 语义下只可能是断电截断的尾行，
        既有记录不受影响。
        """
        records: dict[str, JobRecord] = {}
        if not self.ledger_path.is_file():
            return records
        for line in self.ledger_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                record = JobRecord.from_dict(data)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue  # 截断/畸形行：跳过，不让一条坏行拖垮整个账本
            if record.job_id:
                records[record.job_id] = record
        return records

    def _next_job_id(self) -> str:
        """下一个 job 编号（J-xxxx，续接账本里可解析记录的最大值）。"""
        numbers = [
            int(record.job_id.split("-")[1])
            for record in self._load_records().values()
            if record.job_id.startswith("J-") and record.job_id.split("-")[1].isdigit()
        ]
        return f"J-{max(numbers, default=0) + 1:04d}"

    def _require_active(self, note_id: str) -> Note:
        """取 active 笔记；不存在或状态漂移（已 merged/superseded）一律拒绝执行。"""
        note = self.store.get_note(note_id)
        if note is None:
            raise ValueError(f"笔记不存在：{note_id}")
        if note.meta.status != "active":
            raise ValueError(
                f"笔记 {note_id} 状态为 {note.meta.status}（非 active）：状态已漂移，拒绝执行"
            )
        return note


# ---- rejudge 输出解析与提示词 --------------------------------------------------


def _parse_rejudge_output(raw: str) -> tuple[str, str]:
    """严格解析 rejudge 输出：必须是 JSON 对象且字段恰好 {verdict, reason}。

    verdict ∈ {confirm, supersede}、reason 为非空字符串；任何偏差（非 JSON、非
    对象、超字段、缺字段、非法取值）都抛 ValueError → job failed、原笔记不动。
    """
    try:
        parsed = json.loads(raw.strip())
    except json.JSONDecodeError as exc:
        raise ValueError(f"rejudge 输出不是合法 JSON：{exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"rejudge 输出必须是 JSON 对象，得到 {type(parsed).__name__}")
    if set(parsed) != _REJUDGE_KEYS:
        raise ValueError(
            f"rejudge 输出字段必须恰好是 {sorted(_REJUDGE_KEYS)}，得到 {sorted(parsed)}"
        )
    verdict = parsed["verdict"]
    reason = parsed["reason"]
    if verdict not in _REJUDGE_VERDICTS:
        raise ValueError(
            f"rejudge verdict 必须是 {'/'.join(_REJUDGE_VERDICTS)}，得到 {verdict!r}"
        )
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("rejudge reason 必须是非空字符串")
    return verdict, reason


def _rejudge_prompt(note: Note) -> str:
    """rejudge 的单条 user 消息：笔记正文 + 来源摘要 + 严格 JSON 输出要求。"""
    urls = [ref.url for ref in note.meta.sources if ref.url]
    source_line = "；".join(urls) if urls else "（无来源记录）"
    return (
        "你是研究 wiki 的记忆复核员。请复核下面这条记忆是否仍然成立。\n\n"
        f"记忆（{note.id}）标题：{note.title or '（无标题）'}\n"
        f"正文：{note.body.strip()}\n\n"
        f"来源摘要：{source_line}\n\n"
        '只输出一个 JSON 对象，格式严格为 {"verdict": "confirm", "reason": "判定理由"}：'
        'verdict 只能取 "confirm"（仍然成立）或 "supersede"（已失效，应废止），'
        "reason 说明判定依据；不得输出 JSON 以外的任何文字。"
    )


__all__ = [
    "ACTION_CONFLICT",
    "ACTION_MERGE",
    "ACTION_REJUDGE",
    "ACTION_REFRESH",
    "REJUDGE_TIER",
    "STATUS_DONE",
    "STATUS_FAILED",
    "STATUS_PENDING",
    "STATUS_RUNNING",
    "STATUS_SKIPPED",
    "JobRecord",
    "MaintenanceRunner",
    "MaintenanceSettings",
    "maintenance_settings",
]
