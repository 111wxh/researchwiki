# P4 · Continual Memory 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 Memory 从"研究结束后写入"升级为"随新证据与新交互持续演化"的可审计闭环（未答问题台账 / merge / refresh / conflict / re-judge），并补齐长程运行的支撑设施（compaction / checkpoint / resume / 压测）。

**Architecture:** 维护动作为"先选择、后执行"两段式：`consolidation.py` 做确定性候选选择与 dry-run（不落盘），`maintenance.py` 做进程内 job queue 执行（幂等重试、失败保原文）；未答问题台账是数据驱动的首项（复用序列实验反幻觉证伪的直接回应）；支撑设施沿 AgentLoop 现有结构增量接入（估算→压缩→checkpoint→resume），不引入 RabbitMQ/Redis/分布式调度。

**Tech Stack:** Python 3.12 + uv、pytest、Markdown+YAML frontmatter 记忆库、SQLite FTS5/sqlite-vec 索引、OpenAI 兼容 Provider（skeleton-first：Mock 全链路零成本，真实证据走 smoke 脚本）。

**Spec:**
- `PLAN.md` §3.5 P4 · Continual Memory（四类维护动作、运行方式、支撑设施、幂等要求、验收退出条件——本计划逐条对应，不得缩水）
- `docs/agent-worklog.md` §10.8 复用序列实验 → **对 P4 的决策输入**：①演化价值成立 → 维护闭环方向被数据支持；②反幻觉证伪 → **未答问题台账（kind=experience）是数据驱动的首项**；③经济性叙事主打"vs 无记忆 + 质量维度"，不与 RAG 拼单次成本
- `.superpowers/sdd/PLAN/progress.md` 移交台账：F7 id 形态护栏、merge 独立下限、替换门与相似度带明文化（触发条件均为 P4）

---

## Global Constraints

- **不引入外部基础设施**：无 RabbitMQ、无 Redis、无分布式 scheduler；job queue 是进程内可重复执行 + CLI 驱动。
- **所有自动动作可审计、可暂停、可回滚**：每个动作有 job 记录（job_id、输入记忆 ID、执行时间、模型、token、结果、错误）；dry-run 不落盘。
- **失败绝不当作新事实**：写入失败、抓取失败、模型输出畸形都必须保留原记忆，job 标 failed，原文零改动。
- **幂等三件套**：LLM 调用幂等键 `hash(model + messages + tools)`；state/meta 写入 tmp + atomic rename；job 重试不得重复生成记忆、source snapshot 或 token 记账。
- **skeleton-first**：每个任务必须 Mock 全链路测试通过（零 API 成本）；真实模型证据只通过 smoke 脚本/压测脚本产出，数字必须来自真实 token JSONL。
- **逃生阀惯例**：每段配置 `enabled = false` 即完全禁用（零判定零留痕或按既有惯例留痕）；**配置段缺失 = 不接线 = 整段无操作**（与 `[formation]`/`[memory_update]` 的 guarded 模式逐字一致，见 `agent_loop.py:1353` `_apply_memory_updates_guarded`）。
- **不改 run-metrics 契约**（14 字段，P2 先例）：新阶段产物写 `run_dir/*.json` + `state.md` 一行。
- **诚实计量**：估算与计量分开标注（估算不得伪装成计量）；cache 命中率只在 provider 返回 cache 字段时报告。
- **测试基线 902 passed、ruff 全绿**；每任务交付后全量测试必须绿；commit 只 stage 显式路径；**推送需要本地代理 127.0.0.1:7897**。
- **`mcp-memory/` 是登记在案的分支线（主仓库零提交）**：任何任务不得提交、修改或 gitignore 它。
- 记忆 kind ∈ knowledge / user / experience；笔记 = Markdown + YAML frontmatter + 稳定 ID `N-xxxx`；代码注释/docstring 风格与现有中文注释一致。

## File Structure（P4 全景）

```
src/researchwiki/
  wiki/
    ledger.py          [T1 新建] 未答问题台账（kind=experience 反幻觉记忆）
    consolidation.py   [T2 新建] 维护动作选择器（merge/refresh/conflict/re-judge）+ dry-run
    maintenance.py     [T3 新建] job queue + 执行器（幂等/失败保原文/重试/跳过）
    prior.py           [T1 护栏注入][T4 冲突注入][T5 P-1 编号]
    store.py           [T5 id 形态护栏]
  loop/
    agent_loop.py      [T1 收尾钩子][T4/T5 接线][T6 前缀稳定+压缩触发][T7 checkpoint]
    compaction.py      [T6 新建] 滚动压缩
    checkpoint.py      [T7 新建] checkpoint/resume
    watchdog.py        [T8 新建] 看门狗（超时/失败账本/可读状态）
  llm/
    accounting.py      [T6 上下文估算]
  cli.py               [T2/T3 consolidate 实装][T4 arbitrate 实装]
config.toml            [T1][unanswered] [T2][consolidation] [T3][maintenance] [T6][context] [T7][checkpoint]
scripts/
  stress_run.py        [T8 新建] 连续问题流压测
  kill_resume_smoke.py [T7 新建] kill→resume 冒烟
tests/
  test_ledger.py [T1] test_consolidation.py [T2] test_maintenance.py [T3]
  test_arbitrate.py [T4] test_p4_carried.py [T5] test_compaction.py [T6]
  test_checkpoint.py [T7] test_resume.py [T7] test_stress.py [T8]
```

执行顺序 T1→T8 串行（T1/T4/T5 都改 prior.py、T6/T7 都改 agent_loop.py，并行会互相踩）。每任务：实现者交付 → 监察者评审 → 修复轮（如需）→ 全量测试绿 → 提交推送。任务简报派发时写入 `.superpowers/sdd/PLAN/task-N-brief.md`（N 续接现有编号，从 15 起）。

---

### Task 15 (T1): 未答问题台账 —— kind=experience 反幻觉记忆

**为什么是首项**：复用序列实验反幻觉维度**证伪**——无答案问题上 loop 条件编造完整报告（SEQ016 复现 RQ028，cov=1.0，RAG 反而干净拒绝）。本任务把"这个问题此前已判定证据不足"变成可检索的记忆，在同题再次到来时注入不可编造护栏。

**Files:**
- Create: `src/researchwiki/wiki/ledger.py`
- Modify: `src/researchwiki/loop/agent_loop.py`（run 收尾，紧邻 `_apply_memory_updates_guarded`，同款 guarded 模式）
- Modify: `src/researchwiki/wiki/prior.py`（护栏块注入）
- Modify: `config.toml`（新增 `[unanswered]` 段）
- Test: `tests/test_ledger.py`

**Interfaces:**
- Consumes: `WikiStore.save_note(kind="experience", ...)`（显式写入路径，不经 formation，与 MCP memory.store 同通道）；`wiki/verification.py` 的确定性 bigram 相似度度量（护栏匹配复用同一把尺，与 [verification] 注释"不是同一尺度"的告诫一致——本任务在 verification 尺度上工作）。
- Produces:
  - `ledger.record_unanswered(store, *, question, reason, trace_id, sources_used, notes_created, asked_at=None) -> Note`——写 kind=experience 台账笔记。
  - `ledger.iter_active_unanswered(store) -> list[Note]`——status=active 且 `extra.ledger == "unanswered_question"` 的台账（tombstone/superseded 自动排除）。
  - `ledger.question_similarity(a: str, b: str, dim: int = 128) -> float`——问题文本相似度（复用 verification 的度量实现）。
  - `unanswered_settings(config: Mapping | None) -> UnansweredSettings | None`——缺段返回 None（=不接线）。
- 台账笔记形态（契约，frontmatter extra 承载结构化字段）：
  ```markdown
  ---
  id: N-0107
  kind: experience
  status: active
  importance: 0.8          # 反幻觉护栏是高价值记忆，固定 0.8
  extra:
    ledger: unanswered_question
    reason: no_notes        # no_notes | below_min_fresh
    trace_id: 20261011T-abc
    sources_used: 0
    notes_created: 0
  ---
  # 未答问题台账

  问题「RQ028 的原问题文本」于 2026-10-11 被判定证据不足（原因 no_notes）。
  本次尝试：0 个来源、0 条笔记、trace 20261011T-abc。

  处置：后续相似问题注入反编造护栏；再次研究后若已可答，以 supersede 了结本条（保留历史）。
  ```
- 触发判定（run 收尾，确定性，无模型调用）：
  - `reason="no_notes"`：本轮 `notes_created == 0` 且 fresh 来源数 == 0；
  - `reason="below_min_fresh"`：本轮 fresh 来源数 < 当前模式（policy.json 的 mode）对应 `[retrieval.<mode>].min_fresh_sources`（mode 不可得时按 deep 段）。
  - 同一 run 内同题只记一条（trace_id 去重）。
- 护栏注入（prior 注入阶段）：活跃台账与当前问题 `question_similarity ≥ [unanswered].guard_similarity`（默认 0.4）时，在 Prior 块**之前**插入独立护栏块（最多 `guard_top_k=2` 条，按相似度降序）：
  ```
  ### 未答问题护栏（不可编造）
  问题「……」于 2026-10-11 被判定证据不足（原因 no_notes）。本次如仍无法证实，必须明说"证据不足"，禁止编造。
  ```

**[unanswered] 配置段：**
```toml
[unanswered]
# 未答问题台账（P4a 数据驱动首项，复用序列实验反幻觉证伪的回应）：run 收尾
# 确定性判定（无模型调用）——notes_created==0 且无 fresh 来源 → no_notes；
# fresh 来源数低于当前模式 min_fresh_sources → below_min_fresh。台账是
# kind=experience 的高 importance 记忆，Prior 阶段对相似问题注入"不可编造"护栏。
# 局限（诚实边界）：判定是确定性代理，"有部分证据但仍不足"的问题不在捕获范围
# （序列实验 SEQ016 形态恰好落在 no_notes，但不是所有形态都如此）。
# enabled=false 逃生阀；段缺失 = 不接线 = 收尾零动作、Prior 零护栏。
enabled = true
dry_run = false            # 影子模式：判定与留痕照跑，不写笔记、不注入护栏
guard_similarity = 0.4     # 护栏触发的问题相似度（verification 尺度，校准项）
guard_top_k = 2
```

- [ ] **Step 1: 写失败测试** `tests/test_ledger.py`——用例：①record 后 store 里出现 kind=experience、extra.ledger=="unanswered_question"、importance 0.8 的 active 笔记，body 含问题原文与原因；②同 trace_id 二次 record 不新增（幂等）；③`iter_active_unanswered` 排除 superseded/tombstone；④`no_notes`/`below_min_fresh` 两个触发路径（构造 AgentLoop mock run：一组零产出 run、一组 fresh 来源不足 run），段缺失时零动作、`enabled=false` 零动作、`dry_run=true` 零写盘但 run_dir/unanswered.json 留痕；⑤护栏注入：相似问题（构造 ≥0.4 相似度的问题对，如原问题与其同义改写）注入护栏块，不相似问题不注入，dry_run 台账不注入；⑥护栏块出现在 Prior 块之前且不计入 Prior 的 top_k 配额。
- [ ] **Step 2: 跑测试确认失败** `uv run pytest tests/test_ledger.py -v` → FAIL（模块不存在）。
- [ ] **Step 3: 实现** ledger.py + agent_loop 收尾钩子（在 `_apply_memory_updates_guarded()` 调用点之后追加 `_record_unanswered_guarded()`，None-settings 直接 return）+ prior.py 护栏注入 + config.toml 段。
- [ ] **Step 4: 全量测试 + ruff** `uv run pytest && uv run ruff check .` → 全绿（902 + 新增）。
- [ ] **Step 5: Commit** `git add`（显式路径）→ `feat(memory): 未答问题台账——反幻觉经验记忆与 Prior 护栏注入`

---

### Task 16 (T2): consolidation 选择器 + `consolidate --dry-run`

**Files:**
- Create: `src/researchwiki/wiki/consolidation.py`
- Modify: `src/researchwiki/cli.py`（`consolidate` 从桩变实装：本任务只接 `--dry-run`（默认）+ `--json` + `--root`；`--run` 留给 T3）
- Modify: `config.toml`（新增 `[consolidation]` 段）
- Test: `tests/test_consolidation.py`

**Interfaces:**
- Consumes: `WikiStore.list_notes(status="active")`、`wiki/freshness.py` 三态判定（review_due/stale）、`wiki/verification.py` 比较器（conflict 判定复用，含 similarity_floor/实体护栏全局前置）、`store.save_conflict`（T3 才真正调用）。
- Produces（T3 依赖）:
  - `@dataclass PlannedAction`: `action: str`（merge/refresh/rejudge/conflict）、`note_ids: list[str]`、`reason: str`、`idempotency_key: str`、`payload: dict`（merge 的 canonical_id / refresh 的 url / rejudge 的模型档位 / conflict 的双方断言槽位）。
  - `@dataclass ConsolidationPlan`: `actions: list[PlannedAction]`、`scanned: int`、`settings_snapshot: dict`。
  - `plan_consolidation(store, settings, *, embedding=None, now=None) -> ConsolidationPlan`——纯选择，零写盘。
  - `consolidation_settings(config) -> ConsolidationSettings | None`——缺段返回 None（CLI 报"未配置"退出 1，与旧桩语义衔接）。
- 四类候选规则（全确定性，顺序固定，产出可复算）：
  1. **merge**：active knowledge 笔记两两之间，实体交集 ≥1 且 verification 尺度相似度 ∈ `[merge_min_similarity, 0.95)`。canonical ID = created_at 更早者（并列取字典序更小 ID）；payload 记 canoncal/absorbed。**下限独立明文化（P2 移交裁定）**：merge_floor（默认 0.5）< supersede 门（0.6），因为 merge 是保守动作（双内容都保留在规范 ID 下）而 supersede 是覆盖动作；0.6–0.8 灰区的明文规定写进 `[consolidation]` 注释（supersede 门只约束覆盖，merge/conflict 不受它约束——与 [verification] 现有注释对齐）。校准锚点：真实 run 在 sim 0.31 曾并入正文——0.31 < 0.5 新地板，该形态不再自动 merge。
  2. **refresh**：freshness 三态 ∈ {review_due, stale} 且 ≥1 个来源 URL 的笔记 → 重抓候选；**无来源的 review_due/stale 笔记改派 rejudge**。
  3. **rejudge**：confidence=="low" 的 active 笔记 ∪（无来源且 review_due/stale 的笔记，来自规则 2 改派）。
  4. **conflict**：同实体 active knowledge 笔记对过 verification 比较（respect similarity_floor 与实体护栏），verdict=="conflicting" → 开台账候选。每计划最多扫 `max_conflict_pairs=50` 对（确定性排序后取前 N，防组合爆炸）。
  - 每类批量上限 `max_merges / max_refresh / max_rejudge / max_conflicts = 10`；动作排序 `(action, note_ids)` 字典序——**同一 store 状态必须产出逐字节相同的 plan**（可复算测试）。
  - `idempotency_key = sha256(action + sorted(note_ids) + 各笔记 body_hash + action 参数)`——T3 幂等重试的钥匙。
- CLI：`researchwiki consolidate [--root R] [--json] [--dry-run]`——dry-run 打印人读表或 JSON（`--json` 输出 plan 的 dict 形态），退出码 0；零写盘。

**[consolidation] 配置段：**
```toml
[consolidation]
# 后台维护选择器（P4a）：扫描 active 记忆产出四类候选（merge/refresh/rejudge/
# conflict）。纯选择零写盘；执行在 [maintenance]（consolidate --run）。
# merge_floor 明文化（P2 移交裁定）：merge 是保守动作（双内容保留在规范 ID 下），
# 独立下限 0.5，低于 supersede 门 0.6；supersede 门只约束覆盖类动作，merge 与
# conflict 不受它约束（与 [verification] 注释一致）。0.6–0.8 灰区：相似度在该带
# 内的新证据不 supersede（落 uncertain 人工看），但允许 merge（内容不丢）。
enabled = true
merge_min_similarity = 0.5
rejudge_stale_ratio = 0.25   # 等于 [freshness].stale_ratio；超龄笔记派 rejudge
max_conflict_pairs = 50
max_merges = 10
max_refresh = 10
max_rejudge = 10
max_conflicts = 10
```

- [ ] **Step 1: 写失败测试** `tests/test_consolidation.py`——用例：①构造相似度 ≥0.5 + 实体重叠的对 → merge 候选且 canonical=较早者；②构造 ~0.31 相似度对（校准锚点）→ **不**产生 merge；③review_due + 有来源 → refresh 候选（payload 带 url）；review_due + 无来源 → rejudge；④confidence=low → rejudge；⑤同实体矛盾对（slot 级取值冲突，参照 tests/test_verification.py 的构造法）→ conflict 候选；⑥max_* 截断生效；⑦**dry-run 零变异**：对临时 root 跑 plan 前后目录逐字节对比（含 mtime）；⑧同状态两次 plan 输出逐字节相同；⑨idempotency_key 对同输入稳定、对 body 变化敏感；⑩段缺失 → CLI 退出 1 提示未配置；⑪`--json` 形态可解析且含 settings_snapshot。
- [ ] **Step 2: 跑测试确认失败**。
- [ ] **Step 3: 实现** consolidation.py + cli.py consolidate 实装（--run 在本任务 print "执行器在 T3 接线" 返回 1）+ config 段。
- [ ] **Step 4: 全量测试 + ruff** → 全绿。
- [ ] **Step 5: Commit** → `feat(memory): consolidation 选择器与 consolidate --dry-run（merge 独立下限明文化）`

---

### Task 17 (T3): maintenance job queue + 执行器 + `consolidate --run`

**Files:**
- Create: `src/researchwiki/wiki/maintenance.py`
- Modify: `src/researchwiki/cli.py`（`--run` 接通执行器；`--jobs` / `--retry JOB_ID` / `--skip JOB_ID --reason TEXT`）
- Modify: `config.toml`（新增 `[maintenance]` 段）
- Test: `tests/test_maintenance.py`

**Interfaces:**
- Consumes: T2 的 `ConsolidationPlan/PlannedAction`；`fetch_url`（fetch.py，含快照版本化 `sources/{url_sha1}/{content_hash}/`）；`store.mark_source_changed`、`store.save_note`（refresh 产新证据笔记，kind=knowledge）；supersede 路径（P2 `_mark_superseded` 语义：status→superseded + 原文保留）；`llm/router.py ModelRouter.get("cheap")`（rejudge 模型档）；`llm/replay.py` 幂等键约定 `sha256(model+messages+tools)`。
- Produces:
  - `@dataclass JobRecord`: `job_id: str`（`J-xxxx` 顺序号）、`action`、`note_ids`、`idempotency_key`、`created_at/started_at/finished_at`、`model: str | None`、`tokens_in/tokens_out: int`、`status: pending|running|done|failed|skipped`、`result: dict`、`error: str | None`、`attempt: int`。
  - `MaintenanceRunner(store, router, settings, *, fetcher=None)`：
    - `run(plan) -> list[JobRecord]`——执行计划；idempotency_key 已有 done job → 新记录 status=skipped（reason=idempotent），**零重复副作用**。
    - `list_jobs(status=None) -> list[JobRecord]`、`retry_job(job_id) -> JobRecord`（仅 failed 可重试）、`skip_job(job_id, reason) -> JobRecord`（仅 pending/failed 可跳过）。
  - 存储布局：`wiki-data/maintenance/jobs.jsonl`（追加式账本，一行一记录）+ `wiki-data/maintenance/jobs/{job_id}.json`（单 job 详情，tmp+rename 原子写）。
- 三类执行器语义（PLAN §3.5 逐条）：
  - **merge**：absorbed 笔记 body 并入 canonical（保留规范 ID 与来源并集），absorbed 记 status=merged + redirect_to 留痕（与入库去重同形态）；canonical 的 sources 取并集。
  - **refresh**：`fetch_url(原 URL)` → content_hash 未变 → 仅刷新 reviewed_at，result 标 `source_unchanged`；变了 → `mark_source_changed` + 新证据笔记（kind=knowledge、走既有快照版本化，来源可追溯），**旧快照一律保留**；抓取异常 → job failed + 原笔记零改动。
  - **rejudge**：cheap 档模型判定（输入=笔记正文+来源摘要，输出=严格 JSON `{verdict: confirm|supersede, reason}`）；JSON 畸形/超字段 → job failed（PLAN：模型输出畸形必须保留原记忆）；confirm → reviewed_at 刷新；supersede → 原笔记置 superseded（带理由，**不删除历史**，不创建替代笔记）。
  - conflict 动作在本任务执行 = `store.save_conflict`（开台账），**绝不自动 merge 冲突对**（PLAN 验收明文）。
- token 记账：rejudge 的 usage 计入 job 记录（tokens_in/tokens_out），mock provider 返回的 usage 也如实记录。

**[maintenance] 配置段：**
```toml
[maintenance]
# 维护执行器（P4a）：consolidate --run 按 [consolidation] 的计划逐动作执行。
# 幂等：同 idempotency_key 的 done job 使重放变 skipped；失败保原文（抓取失败/
# 模型畸形绝不落新事实）；job 账本 jobs.jsonl 追加式 + 单 job JSON 原子写。
# 用户可查看（--jobs）、重试（--retry，仅 failed）、跳过（--skip）。
enabled = true
# rejudge_model_tier = "cheap"   # rejudge 用的模型档（默认 cheap）
```

- [ ] **Step 1: 写失败测试** `tests/test_maintenance.py`——用例：①merge 执行后 canonical body 含双方内容、absorbed status=merged+redirect_to、sources 并集；②refresh：Mock fetcher 返回相同内容 → 仅 reviewed_at 变、无新快照；返回不同内容 → 新证据笔记 + 旧快照文件仍在 + mark_source_changed 留痕；fetcher 抛异常 → job failed、原笔记 body/frontmatter 逐字节不变；③rejudge：Mock provider 返回合法 confirm/supersede JSON 两条路径；返回畸形 JSON → failed + 原笔记不变；usage 记进 job；④幂等：同 plan 跑两遍，第二遍全 skipped、笔记数/快照数/JSONL 行数不增（第二遍只追加 skipped 行）；⑤retry（failed→重试成功）、skip（含 reason 留痕）、list 过滤；⑥conflict 动作开台账且两笔记均保持 active；⑦JSONL 中途断电模拟（写入后 kill 进程的记录完整性：逐行 JSON 可解析）。
- [ ] **Step 2: 跑测试确认失败**。
- [ ] **Step 3: 实现** maintenance.py + cli.py（--run/--jobs/--retry/--skip）+ config 段。
- [ ] **Step 4: 全量测试 + ruff** → 全绿。
- [ ] **Step 5: Commit** → `feat(memory): maintenance job queue 与执行器（幂等重试、失败保原文）`

---

### Task 18 (T4): `arbitrate` 实装 + Prior 冲突注入

**Files:**
- Modify: `src/researchwiki/cli.py`（`arbitrate` 桩变实装：`list` / `show ID` / `resolve ID`）
- Modify: `src/researchwiki/wiki/prior.py`（open 冲突注入"待解决冲突"块）
- Modify: `src/researchwiki/loop/agent_loop.py`（Prior 装配处透传，如需）
- Test: `tests/test_arbitrate.py`

**Interfaces:**
- Consumes: `store.list_conflicts(status)`、`store.resolve_conflict(conflict_id, verdict=, resolved_with=)`（**已有**，含 resolved_at；不改动它）、`store.save_note(kind="experience")`（裁决留痕笔记）。
- Produces:
  - CLI：`researchwiki arbitrate list [--status open|resolved|all] [--root] [--json]`；`researchwiki arbitrate show ID [--root] [--json]`；`researchwiki arbitrate resolve ID --verdict "结论说明" --with N-xxxx [--root] [--json]`（`--with` 必须是存在且 active/merged 的笔记 ID，否则退出 2）。
  - `arbitrate resolve` 行为：`store.resolve_conflict(...)` + 写一条 kind=experience 裁决留痕笔记（extra: `ledger: arbitration`, `conflict_id`, `adopted_note_id`，body 引用冲突双方与结论），使裁决本身成为可检索记忆。
  - Prior 注入（prior.py）：`retrieve(...)` 增加冲突块——open 冲突的实体与当前问题实体相交（或冲突笔记与问题相似度 ≥ `[verification].similarity_floor`）时，在护栏块之后、Prior 笔记之前插入：
    ```
    ### 待解决冲突（优先核验）
    C-0003 实体「glm-5.3」存在矛盾断言：「上下文 200K」vs「上下文 256K」
    （来源 N-0012 / N-0031）。请优先核实哪一方是现行事实，并在报告中说明。
    ```
  - resolved 冲突不注入；`[prior].enabled=false` 时一切照旧不注入。

- [ ] **Step 1: 写失败测试** `tests/test_arbitrate.py`——用例：①list（open/resolved 过滤、--json）；②resolve：conflict 变 resolved + resolution/resolved_at 落盘（既有 store 行为的接线验证）+ 产生裁决留痕 experience 笔记；③resolve 传不存在的笔记 ID → 退出 2 且 conflict 仍 open；④show 全文；⑤Prior 注入：问题实体与 open 冲突实体相交 → 注入块出现（置于护栏块后）；无相交 → 不注入；resolve 后同问题不再注入；⑥仲裁留痕笔记可被 `memory.search` 检索到（它就是普通记忆）。
- [ ] **Step 2: 跑测试确认失败**。
- [ ] **Step 3: 实现**。
- [ ] **Step 4: 全量测试 + ruff** → 全绿。
- [ ] **Step 5: Commit** → `feat(memory): arbitrate 人工仲裁 CLI 与 Prior 待解决冲突注入`

---

### Task 19 (T5): P2 移交清账 —— F7 id 护栏 + SearchIndex 关闭 + Prior P-1 编号

**Files:**
- Modify: `src/researchwiki/wiki/store.py`（写入路径 id 形态护栏）
- Modify: `src/researchwiki/loop/agent_loop.py`（run 结束关闭 SearchIndex 连接）
- Modify: `src/researchwiki/wiki/prior.py`（Prior 条目编号 `### [n]` → `### [P-n]`，消除与报告引用 `[n]` 的碰撞）
- Test: `tests/test_p4_carried.py`

**触发理由**（.superpowers/sdd/PLAN/progress.md 台账）：P4 引入外部证据通道（maintenance 产新证据、仲裁留痕），F7 护栏触发条件成立；SearchIndex 泄漏与编号碰撞是 P1 遗留。

**Interfaces:**
- `WikiStore.save_note / save_meta`：`note.id` 必须匹配 `^N-\d{4,}$`（内部生成与外部传入同守），违规抛 `ValueError`；conflict id 同理 `^C-\d{4,}$`。**注意**：先全仓 grep 现有测试/ fixtures 用过的 id 形态，护栏宽度必须兼容现状（四位是现行 `next_note_id` 下限形态，`{4,}` 允许更多位防溢出）。
- `AgentLoop`：run 完成路径（含异常路径——`try/finally`）关闭索引连接；Server 通路与直接调用通路都要覆盖。
- prior 输出：`### [P-1] 标题` 形态；对应测试同步改名（grep 全仓 `[n]` 相关断言）。

- [ ] **Step 1: 写失败测试**——①`save_meta` 传 `id="note-1"` / `id="N1"` / 空串 → ValueError，合法 `N-0001` 通过；②AgentLoop mock run 后索引连接已关闭（监控 close 调用或 `conn` 状态）；③Prior 块编号为 `P-1..P-k` 且报告引用编号不与 P 前缀冲突。
- [ ] **Step 2: 跑失败**。**Step 3: 实现**。**Step 4: 全量 + ruff 绿**（此任务最可能碰旧测试断言，逐个订正并保持语义）。**Step 5: Commit** → `refactor(memory): P2 移交清账——写入 id 护栏、索引连接关闭、Prior P-编号`

---

### Task 20 (T6): 上下文预算 —— 估算 + 稳定前缀 + 滚动压缩

**Files:**
- Create: `src/researchwiki/loop/compaction.py`
- Modify: `src/researchwiki/llm/accounting.py`（上下文估算函数）
- Modify: `src/researchwiki/loop/agent_loop.py`（消息装配分区 + 压缩触发）
- Modify: `config.toml`（新增 `[context]` 段）
- Test: `tests/test_compaction.py`

**Interfaces:**
- Consumes: 现有 token 记账（accounting.py 的 JSONL 计量——估算与计量**分开**：计量来自 provider usage，估算用于决策）；`llm/router.py` cheap 档（摘要模型）。
- Produces:
  - `accounting.estimate_message_tokens(msg: dict) -> int`、`estimate_tools_tokens(tools: list[dict]) -> int`、`estimate_context_tokens(messages, tools) -> int`——确定性字符启发式（CJK 计 1 token/字、ASCII 1 token/4 字符，实现里写明这是**估算不是计量**）。
  - `compaction.maybe_compact(messages, settings, *, summarizer) -> CompactionResult`：估算 ≥ `threshold × max_context_tokens` 时触发；保留最近 `keep_recent_turns` 轮 + 未解决任务清单（从 state 文件读）原文，其余轮次压缩成一条"研究进展摘要"消息（summarizer=cheap 模型调用；测试注入 fake summarizer）；结果 `{compacted: bool, pre_tokens, post_tokens, kept_turns, summary}`。
  - 压缩事件审计：`run_dir/compactions.jsonl` 追加 `{index, at, pre_tokens, post_tokens, kept_turns, summary_chars}`。
  - `[context]` 段：`max_context_tokens`（默认 131072）、`compaction_threshold = 0.7`（`[runtime].compaction_threshold` 保留为回退读取，注释标迁移路径）、`keep_recent_turns = 4`、`enabled = true`。
- **稳定前缀不变量**（PLAN 支撑设施 3）：同一 run 内 system prompt 字节、工具 schema 顺序、格式化模板固定；Prior/检索/工具结果一律在后缀区。agent_loop 装配处加注释明示分区 + 测试固化：同配置两次构造 → system prompt 与 tools 列表逐字节相同。

- [ ] **Step 1: 写失败测试**——①估算函数对中英文样例的确定性与单调性（更长→更大）；②触发数学：构造 messages 使估算跨过阈值 → compacted=True，保留最近 K 轮原文，摘要消息在保留区之前；未跨阈值 → compacted=False 原样返回；③`enabled=false` 永不触发；④compactions.jsonl 事件字段完整；⑤前缀稳定性（两次构造逐字节相同）；⑥summarizer 抛异常 → 不压缩（保守降级）+ 事件记录 failed。
- [ ] **Step 2: 跑失败**。**Step 3: 实现**。**Step 4: 全量 + ruff 绿**。**Step 5: Commit** → `feat(loop): 上下文估算、稳定前缀分区与滚动压缩（compaction 审计）`

---

### Task 21 (T7): checkpoint + resume

**Files:**
- Create: `src/researchwiki/loop/checkpoint.py`
- Modify: `src/researchwiki/loop/agent_loop.py`（每步完成后落 checkpoint；`resume_from` 入口）
- Create: `scripts/kill_resume_smoke.py`
- Modify: `config.toml`（新增 `[checkpoint]` 段）
- Test: `tests/test_checkpoint.py`、`tests/test_resume.py`

**Interfaces:**
- Produces:
  - `run_dir/checkpoint.json`（tmp+rename 原子写，每步完成后更新）：`{version: 1, run_id, mode, completed_step_ids: [], completed_subtask_ids: [], notes_created: [], llm_keys_seen: [], budget_used: {tokens_in, tokens_out}, history_summary_ref, state_file_version, updated_at}`。
  - `checkpoint.save(run_dir, payload)`（原子写）、`checkpoint.load(run_dir) -> dict | None`（损坏返回 None 并告警，不 crash）。
  - `AgentLoop.run(..., resume_from: Path | None = None)`：从 checkpoint 重建状态——completed step 跳过、`llm_keys_seen` 命中的调用走 replay 缓存语义（不重复扣费）、`notes_created` 里已有的笔记**不重复创建**。
  - **写-崩溃窗口兜底**：蒸馏入库的笔记 extra 带 `run_step_key`（产生它的 step 幂等键）；恢复后入库前先查同 key 笔记存在 → 跳过创建（"已完成写入不重复创建"在 checkpoint 滞后一拍时仍成立）。
  - `[checkpoint]` 段：`enabled = true`、`every_step = true`（false = 仅阶段边界）。
- `scripts/kill_resume_smoke.py`（mock provider，零成本，四道硬闸仿 cold_warm_smoke）：①子进程起 run → 在第 N 步 kill -9 →②新进程 `resume_from` 续跑至完成 →③闸1：未完成任务继续执行完成；闸2：已完成 step 不重复执行（LLM 调用次数 ≤ 首次+续跑新增）；闸4：笔记无重复（store 里同题笔记数与一次性跑完的基线一致）；④全程零 API 成本。

- [ ] **Step 1: 写失败测试**——①checkpoint 原子写（写中断模拟：tmp 残留不影响 load）；②load 损坏文件返回 None；③resume 跳过 completed_step_ids（计数断言）；④同 run_step_key 笔记不重复创建；⑤llm_keys_seen 幂等（mock provider 调用计数）；⑥kill_resume_smoke 四闸在脚本内绿（mock，作为测试用 subprocess 用例跑通短版本）。
- [ ] **Step 2: 跑失败**。**Step 3: 实现**。**Step 4: 全量 + ruff 绿**。**Step 5: Commit** → `feat(loop): checkpoint/resume——幂等恢复与 kill_resume 冒烟`

---

### Task 22 (T8): watchdog + 压测脚本 + P4 验收总闸

**Files:**
- Create: `src/researchwiki/loop/watchdog.py`
- Create: `scripts/stress_run.py`
- Test: `tests/test_stress.py`（机制层，mock）
- 收尾：真实压测 + 验收证据 + worklog §11

**Interfaces:**
- `RunWatchdog(settings)`：逐问题 timeout（默认 600s）、连续失败上限（默认 3 → **暂停**不是崩溃）、失败账本（每条带 trace + 错误类型 + 所在 step）、`status_line()` 可读状态。**PLAN 原文**：连续问题流、超时、工具失败和单个 provider 失败都必须有可读状态。
- `scripts/stress_run.py --questions N --root R --provider mock|real --per-question-timeout S [--duration 2h]`：单持久 wiki root 上连续问题流（同时锻炼 compaction/checkpoint/maintenance）；每问题结束追加 `wiki-data/stress/{ts}/questions.jsonl`（问题、耗时、tokens、错误）；每 5 问拍一次 `lint --json` 快照（**验收闸**：断链/孤立/无来源指标不恶化）；结束打印汇总。
- **验收证据链**（对应 PLAN §3.5 退出条件，逐条落到脚本输出或测试断言）：
  1. dry-run 准确且不落盘 → T2 测试⑦ + T8 压测前手动 dry-run 抽查
  2. 同 job 重试零重复 → T3 测试④
  3. refresh 保留旧快照 + 新证据可追溯 → T3 测试②
  4. conflict 不被自动 merge、仲裁有 resolution/resolved_at → T3 测试⑥ + T4 测试②
  5. 连续运行后 lint 指标不恶化 → stress_run 断言
  6. 10+ 连续问题 / 2h 零未解释崩溃 → mock 2h 必跑；real 2h 跑前先报预算估算（air 单问题 ~7 万 input tok，2h 约 12–20 问 ≈ 1–3M tok），跑完出真实 token JSONL
  7. token 曲线在阈值内、compaction 次数/前后 token/保留摘要可审计 → stress 期间 compactions.jsonl 汇总
  8. kill 后恢复、不重复扣费/写入 → T7 冒烟四闸
  9. cache 命中率只在 provider 返回 cache 字段时报告（当前 air 不返回 → 报告里写"未报告：provider 无 cache 字段"）

- [ ] **Step 1: 写失败测试**——watchdog 超时/连续失败暂停/状态行；stress_run 的 `--questions 3 --provider mock` 快速路径全绿 + lint 快照断言。
- [ ] **Step 2: 跑失败**。**Step 3: 实现**。**Step 4: 全量 + ruff 绿**。**Step 5: Commit** → `feat(loop): 看门狗与压测脚本（P4 支撑设施收口）`
- [ ] **Step 6: mock 2h 压测**（后台跑，零成本）→ 证据落 `wiki-data/stress/`。
- [ ] **Step 7: real 压测**（glm-4.5-air，先在报告中写明预算估算）→ 证据 + token JSONL。
- [ ] **Step 8: 监察者终审**：PLAN §3.5 九条验收逐条对证据；worklog §11 记录（含成本、失败、诚实边界）；更新 auto-memory；全部推送。

---

## Self-Review 记录

1. **Spec 覆盖**：PLAN §3.5 四类动作（merge/refresh/conflict/re-judge → T2/T3）、dry-run 先行（T2）、job queue + job 字段（T3）、失败保原文（T3）、查看/重试/跳过/人工仲裁（T3/T4）、支撑设施六条（估算/压缩/前缀/checkpoint/resume/watchdog → T6/T7/T8）、幂等要求（T3/T7）、九条验收（T8 证据链）——全覆盖。§10.8 决策输入①②③ → T1 首项 + 叙事约束进 worklog。移交台账三项 → T2（merge 下限+明文化）+ T5（F7）。
2. **占位符**：无 TBD/TODO；所有接口给了签名与数据形态，所有测试给了具体断言内容。
3. **类型一致性**：`PlannedAction.idempotency_key` 在 T2 产出、T3 消费同名；`consolidation_settings`/`unanswered_settings` 缺段返回 None 的约定与 `memory_update_settings` 一致；job 状态词表五值（pending/running/done/failed/skipped）贯穿 T3/T8。
