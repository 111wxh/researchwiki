# 复用序列实验 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development。Spec: `docs/superpowers/specs/2026-10-10-reuse-sequence-eval-design.md`（预注册判定与口径以 spec 为权威）。

**Goal:** 四条件重放同一事件时间线（ingest→簇内查询→update→时效/无答案/更新后重复），逐事件计量，对照预注册判定输出记忆架构的经济性/复用/演化/反幻觉裁决。

**Tech Stack:** 复用 run_eval 条件实现（importlib）、judge、指标；Python 3.11+ / pytest / uv。

## Global Constraints

- Spec 为约束权威；预注册判定四条不得在实现中被弱化或改动。
- c4 持久 root 全时间线；查询策略 auto（retrieval_config 接 [retrieval] 段，不设 forced_mode）；c1 每问一次性 root。
- 信息权限对等：ingest 事件同批文档同时刻进两系统；update 事件 RAG 侧 v2 换 v1 chunk（取对 RAG 有利解释）。
- 新综合问盲答验证：不过修一次，再不过丢弃；复用既有 gold 优先。
- 注释中文；显式路径提交；mock 全链路先行；real run 由监察者在 mock 绿后直接执行（用户已授权，无中途关卡）。

---

### Task 1: 序列题集 + 事件流 schema + 校验器

**Files:** Create `evals/qa/ai-frameworks-seq.jsonl`、`evals/sequence/scenario.jsonl`、`src/researchwiki/evals/sequence.py`、`tests/test_evals_sequence.py`

**Interfaces:**
- Produces: `SeqEvent`（frozen：`event_id: int, kind: str  # ingest|query|update, doc_ids: tuple[str, ...] = (), qid: str = "", note: str = ""`）；`load_scenario(path, qa_path, corpus_manifest_path) -> list[SeqEvent]`（校验：query 的 qid ∈ 题集；qid 出现两次则必须为"同簇重复对（都在 update 前）"或"更新后重复对（一前一后）"；update 恰 1 个且在全部时效题之前；ingest 的 doc_ids ⊆ 语料 manifest；doc_ids 引用的文档必须在首个引用它的 query 之前 ingest）；`clusters_from_manifest(manifest) -> dict[str, list[str]]` 由实现者按语料主题分组并在 scenario.jsonl 的 note 字段披露分组依据。
- 题集：复用 `evals/qa/ai-frameworks-real.jsonl` 的 gold（按簇挑 4×3 道相邻/综合 + 时效 3 + 无答案 2）+ 新综合问（每簇 1–2 道，**先找真实层会话的盲答验证工具**（commits df94774/22002ac，grep scripts/ evals/）复用；无脚本则写 `scripts/blind_validate.py` 最小实现：语料全文+问题无 gold 交 cheap 档 → point_hit 判 gold）。
- TDD：schema 坏例（qid 不存在/重复对结构非法/update 缺失或多个/时效题在 update 前/doc 提前未 ingest）→ 实现 → 真实场景文件全过。
- Commit：`feat(evals): 复用序列题集与事件流 schema 校验器`

### Task 2: `scripts/run_sequence.py` 重放 harness

**Files:** Create `scripts/run_sequence.py`、`tests/test_run_sequence.py`

**Interfaces:**
- Consumes: importlib 加载 `scripts/run_eval.py` 复用其条件实现（读该文件确认可导入的函数边界；不可导入的薄改 run_eval 为可导入，不改行为）；`load_qa`/`load_scenario`；judge/metrics 全复用。
- 行为：按事件流驱动——c4 单持久 root（study run=一次 formation 完整 loop 吃下 ingest 文档集；query 事件按 auto 策略跑 warm loop，memory_update 开）；c1 每问一次性 root（loop 无记忆开关）；c2/c3 持久索引对象，update 事件换 chunk；judge 每问每条件一次（复用 `-judge` trace 分离）。
- 输出：`evals/results/sequence_<provider>_<ts>/events.jsonl`（每行：condition/event_id/event_kind/qid/policy_mode?/in_tok/out_tok/latency/fresh_search?/em/point_hits/refusal/citation_coverage/judge）+ manifest（含 spec 路径与预注册判定原文引用）。
- TDD：mock 两簇×2 问+1 update+1 时效问，断言事件序驱动正确（c4 第二次查询 prior 命中>0；c1 每问 prior_hit=0；RAG 行无 fresh 键；update 后 RAG 检索命中 v2 chunk；对账一致）。
- Commit：`feat(evals): 复用序列重放 harness（四条件同事件流，c4 持久记忆+auto 策略）`

### Task 3: `scripts/report_sequence.py` 累计曲线与预注册裁决

**Files:** Create `scripts/report_sequence.py`、`tests/test_report_sequence.py`

**Produces:** 报告四段：头部（spec 引用+预注册判定表原文）→ 累计成本曲线表（每条件逐事件累计 in-tok，c4 标注每次查询的 policy 模式）→ 摊销拐点（c4 累计 vs c2 累计的交叉点或"未交叉"）→ **预注册逐条裁决**（经济性/复用质量/演化价值/反幻觉，每条给成立|证伪|不可判定 + 数据引用）。手造 rows 单测覆盖三向。
- Commit：`feat(evals): 序列实验报告（累计成本曲线与预注册判定裁决）`

### Task 4: mock 全链路 + 场景自审 + real run + 判定（监察者执行）

- mock 全链路（全场景）→ 场景自审清单（盲答验证通过率、簇分组合理性、重复问结构、预注册条件未被弱化）入 worklog
- real run（≈1.7M in-tok @air + ≈90 judge；部分行落盘保护已内置）
- 判定报告入库（events.jsonl+manifest+报告 force-add 凭证）+ worklog §10.8 + push

## Self-Review

- Spec 覆盖：四条件/事件流/c4 auto/预注册四条/交付物路径全对上；范围外（P4 功能、多 update、100 题）在 spec §6。
- 接口一致性：SeqEvent 在 T1 定义、T2 消费；events.jsonl 行键在 T2 产出、T3 消费——键名以 T2 简报为准并在 T3 简报传递。
