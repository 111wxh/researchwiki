# ResearchWiki — 面向 Agent 的外置、时间感知、可演化长期记忆系统（v2）

> **本文档是当前项目的唯一产品与工程路线。** 更新日期：2026-09-22。
>
> **v2 方向决策（项目所有者，2026-09-22）**：研究问题从"研究知识系统"泛化为"Agent 外置记忆"。ResearchWiki（研究场景）降级为该 Memory 系统的**第一个应用/实验环境与参考实现**；已有全部工程机制保留，渐进迁移、不推倒重写——概念层先行换名（Wiki→Memory 的投影），代码路径后续再议，本版不改任何代码路径名。
>
> **推进方式：按阶段验收，不按周排期。** 每个阶段必须有可复跑的测试、原始指标和明确的退出条件；验收不过，不进入下一阶段。

---

## 0. 项目定位与核心研究问题

### 0.1 对外定位

> **本项目研究面向 Agent 的外置长期记忆系统（External Agent Memory），重点解决智能体在长期运行过程中面临的记忆存储、检索、更新、失效与时间一致性问题。ResearchWiki（研究场景）作为该 Memory Infrastructure 的第一个垂直应用与参考实现。**

### 0.2 三个现状问题

1. **历史信息的长期价值不同**：无差别累积会持续推高存储与上下文成本。
2. **事实具有时间属性**：过去正确的记忆会因用户状态、外部知识或任务环境变化而失效。
3. **传统方案依赖用户主动保存**：缺少"自动发现—记忆—检索—更新—失效"的闭环。

### 0.3 Memory 生命周期主图

```text
Interaction → Memory Candidate Detection → Memory Formation → External Memory
    → Temporal Validation → Dynamic Retrieval → Agent Context
    → New Evidence / New Interaction → Update / Supersede / Conflict / Invalidate
```

### 0.4 三个核心研究问题

| 编号 | 问题 | 对应生命周期环节 |
|---|---|---|
| **RQ1** | **什么时候应该记住？** Memory Formation：Interaction → Candidate → Importance → Persistence | 形成侧 |
| **RQ2** | **什么时候一个记忆不再应该被当成当前事实？** Temporal Validity：Aging → Stale Detection → Verification → Supersede/Conflict | 保鲜侧 |
| **RQ3** | **当前问题到底应该召回哪些记忆？** Budget-aware Dynamic Retrieval：relevance × temporal validity × confidence × importance × conflict × budget → Context | 检索侧 |

三个问题合起来一句话：**Remember → Maintain → Recall**。

### 0.5 设计立场（作为假设写进计划书，接受评测检验）

1. **Storage 与 Retrieval 分离**：存储层尽可能保存可追溯的状态与证据；检索层每次按 Query 动态组合各因子决定进 Context 的内容。**核心算法问题在检索层**，不在存储层。存储层字段基线：

   ```text
   content、source、created_at、observed_at、valid_from、valid_until、
   confidence、importance、volatility、status、relations
   ```

2. **不做固定分层记忆桶**：拒绝 short-term / long-term / episodic / semantic 固定分桶——同一信息可能同时"高重要性 + 高相关性 + 低新鲜度 + 高历史价值"，固定桶会强迫它只属于一层。因子化的检索决策取代桶归属。
3. **Wiki 是 Memory 的人类可读投影，不是 Memory 本体**：`pages/` 是面向人的视图层；`notes/` 才是记忆的载体。
4. **Memory 类型四分**：User Memory / Knowledge / Experience / Research Prior。其中 Prior 是**角色**而不是类型——任何一条 Knowledge 在被下一次 run 引用时都扮演 Prior 角色。

### 0.6 明确不做

- 不引入 LangChain、LangGraph 或重型 Agent 平台；核心 loop、工具注册、预算和状态管理保持自研。
- 不把项目改造成普通向量 RAG、普通聊天机器人或又一个 GraphRAG Demo。
- 不上固定分层记忆模型（见 0.5 立场 2）。
- 不把对话/交互**无差别全量写入** Memory；写入必须经过 formation 判定或显式 store。
- 不推倒重写现有代码；渐进迁移，wiki_* 接口保留为兼容层。
- 在收益没有数据之前，不上 Neo4j、复杂 Temporal Knowledge Graph 或分布式数据库。
- 不做多用户、账号、协作、计费和 SaaS 控制面；当前目标是单用户、本地优先、可复现。
- 不把没有 provider cache 字段的模型硬算成 KV-cache 命中；只报告 provider 实际返回的字段。

---

## 1. 已完成基线与阶段 1 证据（诚实记录，不再作为后续开发任务）

这一节只记录项目从哪里出发、已经证明了什么，不代表后续还要重新实现。

### 1.1 已具备的能力

- Web UI：Next.js + AI SDK，支持流式研究过程、思考折叠、任务卡片、冲突提示、报告文档视图、引用卡片和 Wiki 面板。
- 后端：FastAPI + SSE，输出 AI SDK UI Message Stream 协议。
- AgentLoop：`plan → act(tool calls) → observe → distill → report`。
- LLM：OpenAI-compatible Provider、strong/cheap 路由、流式输出、工具调用解析、重试、录制回放和 token JSONL 记账。
- 工具：内部 ToolRegistry 中的 `web_search`、`fetch_url`、`fs_read`、`fs_write`、`fs_list`、`dispatch_research`。
- 子 agent：独立上下文、独立 token budget、固定返回 schema，禁止递归派发。
- Memory 基础（Wiki 投影）：来源快照、Markdown + frontmatter 原子笔记（即记忆条目）、实体注册表、页面生成器、冲突台账、轻量去重合并。
- 检索：SQLite FTS5 + sqlite-vec，sqlite-vec 不可用时纯 Python 余弦扫描兜底；检索结果按置信度和新鲜度调节。
- MCP：当前仅暴露 Wiki 读写和健康检查能力：`wiki_search`、`wiki_read`、`wiki_write`、`wiki_list_changes`、`wiki_health`（P1 起新增 `memory_*` 接口面，wiki_* 保留为兼容层，见 §2.4）。MCP 不触发长时间 Research run。
- 部署：Mock 零 key Docker 模式、真实模型 Docker 模式、MIT 开源仓库。

### 1.2 基线证据与边界

| 能力 | 当前状态 | 必须诚实说明的边界 |
|---|---|---|
| Mock Web Demo | 已完成 | 这是演示链路，不证明真实模型效果 |
| 真模型 loop | 已接入并完成多次烟测 | 搜索未配置 key 时会回退 MockSearch；模型自报 URL 不等于全网检索 |
| Memory 三层基础（sources/notes/pages 投影） | 代码已具备 | 阶段验收尚未完成；页面聚合不是 Web 真实 run 的默认路径 |
| Memory 检索 | MCP 可检索 | 主 Agent 尚未自动把记忆接回下一次 Research 的完整决策回路 |
| **Wiki Prior 复用（RQ1 的第一块证据）** | **已完成（阶段 1，2026-09-22）** | Prior 注入、run-metrics、cold/warm smoke 全部可复跑；**真实收益结论待正式评测**（小样本不构成收益结论） |
| Compaction / checkpoint / resume | 尚未实现 | `state.md` 是原子重写状态，不等于完整断点续跑 |
| Consolidation / refresh / re-judge | 尚未实现 | `consolidate` / `arbitrate` CLI 仍是规划入口 |
| 工具调用闸 | 有小样本结果 | 当前落盘结果为 19/20=95.0%，工具名和参数合法率各 15/16=93.8%；不是最终评测 |
| 中文检索闸 | 有小样本结果 | hybrid 与 vector 当前同分，不能宣称 hybrid 已经优于 vector |

### 1.3 阶段 1（Prior MVP）实测证据

**完成范围**：Prior Reader 接入 AgentLoop、`run-metrics.json` 落盘、cold/warm 冒烟脚本，全部可复跑。

| 证据项 | 数字 | 出处 |
|---|---|---|
| 代码测试 | 251 → **306**（通过） | pytest，阶段 1 期间新增 prior/loop 测试 |
| 真模型 cold/warm 冒烟（glm-4.5-air，固定主题，样本数 2） | warm `prior_hit_count = 5` | `smoke_out/` 冒烟 JSONL |
| warm start 输入 token | 71831 → **53719（−25%）** | 同上，可由 `tokens.jsonl` 复算 |
| warm start 端到端时延 | 107.7s → **74.5s** | 同上 |
| 搜索边界 | 未配置搜索 key，回退 **MockSearch**；数字只说明链路可测，不代表真实搜索行为 | 冒烟配置记录 |

**闸门有效性的证明**：冒烟闸门曾抓到两个真实缺陷——嵌入缓存写入不 commit + 写事务内调用 embed 导致 SQLite 锁库。FAIL→FIX→PASS 证据链完整保留在 `smoke_out/` 与 git 历史（`8dd20e3` FAIL → `464138a` FIX → `9ee211e` PASS）。小样本不构成收益结论；真实收益判断留给 §7 正式评测。

### 1.4 基线完成标志

以下均已成立，后续不再重复排 Demo：

- 无 API key 时，Mock 模式可以在 Web UI 完整演示一次研究。
- 切换真实模型后，主 loop 能完成陌生主题的研究、报告和笔记蒸馏。
- 每次 LLM 调用能从 `tokens.jsonl` 复算输入/输出 token 与延迟。
- Memory 数据（Wiki 投影）可以通过 MCP 被外部客户端搜索、读取和写入。
- Prior 注入链路有 run-metrics 与 cold/warm 对照可复跑证据。
- **阶段 1 的四类证据（跨阶段评测闸门，见 §8）已齐**：代码测试（306 通过）、ScriptedProvider 行为 smoke、真模型小样本（glm-4.5-air，样本数 2）、原始数据（`smoke_out/`、`tokens.jsonl`、`run-metrics.json`）。

---

## 2. 目标架构

### 2.1 External Memory 总图

```text
Agent（ResearchWiki 等任意 Agent = 实验环境）
   │  Interaction / Query
   ▼
MCP（External Memory Interface）
   ▼
External Memory（本项目的 Memory Store）
   ├── Formation：Candidate Detection → Importance → Persistence   ← RQ1
   ├── Maintain：Temporal Validation → Update / Supersede / Conflict / Invalidate ← RQ2
   └── Retrieve：Budget-aware Dynamic Retrieval → Agent Context    ← RQ3
   ▼
Temporal Evolution（记忆随新证据/新交互持续演化，回到 Agent）
```

ResearchWiki 的 Research run 是第一个接入该 Memory 的 Agent 应用：研究过程中的笔记蒸馏即 Formation，freshness/supersede 即 Maintain，Prior 注入即 Retrieve。

### 2.2 Memory Engine 概念分层

```text
Memory Engine
  ├── Personal Memory（User Memory）
  ├── Research Memory（Knowledge / Research Prior）
  ├── Experience（任务经验）
  └── Wiki View（pages/ 等人类可读投影）
```

概念分层，不是物理分桶：四类共用同一套存储 schema 与检索因子（立场 0.5-2），只在 `kind` 字段上区分。

### 2.3 Storage schema（NoteMeta 现状与待加字段）

| 字段 | 含义 | 现状 |
|---|---|---|
| `content` | 记忆内容 | 已有（note 正文） |
| `source` | 来源与证据链 | 已有（`sources/` 快照 + 来源 frontmatter） |
| `created_at` | 创建时间 | 已有（`created`） |
| `observed_at` | 观察时间 | 已有 |
| `valid_from` | 何时开始有效 | **待加** |
| `valid_until` | 何时失效 | **待加** |
| `confidence` | 置信度 | 已有 |
| `importance` | 重要性 | **待加**（P1） |
| `volatility` | 变化速率 | 已有 |
| `status` | active / merged / superseded | 已有 |
| `relations` | 实体/记忆间关系 | **待加**（实体注册表已有雏形，需入 schema） |

### 2.4 Retrieval 因子表

| 因子 | 含义 | 现状 |
|---|---|---|
| semantic relevance | 语义相关性 | 已有（FTS5 + sqlite-vec） |
| temporal validity | 时间有效性 | 已有雏形（freshness_factor） |
| importance | 重要性 | **待加**（P1 落库，P3 进因子） |
| confidence | 置信度 | 已有 |
| conflict 状态 | 冲突/被取代标记 | 已有（`conflicts/` 台账 + status） |
| budget | token/时间预算 | **待加**（P3，属预算感知检索） |

### 2.5 MCP 定位升级：External Memory Interface

目标 API 面：

```text
memory.store / memory.search / memory.recall / memory.update
memory.supersede / memory.invalidate / memory.timeline
memory.profile / memory.conflicts
```

**渐进迁移**：现有 `wiki_*` 工具保留为兼容层，`memory_*` 为新接口面；两者在 P1 并存，外部客户端逐步切换。

### 2.6 Memory 数据生命周期

```text
网页 URL / Interaction
  ↓ fetch_url / candidate detection
sources/{url_sha1}/{content_hash}/
  ↓ formation 判定（RQ1：importance / confidence / persistence）
memory candidate → 正式记忆（notes/N-xxxx.md）
  ↓ search / prior / recall
历史记忆（不可直接当答案）
  ↓ fresh research / new interaction
新来源与新证据
  ↓ compare
active / merged / superseded / conflict / invalidated
  ↓ optional consolidation
pages/（人类可读投影） + lint health report
```

### 2.7 内部工具与 MCP 的边界

- **内部 Agent 工具**：由进程内 `ToolRegistry` 注册，给 AgentLoop 做搜索、抓取和文件操作，不经过 MCP。
- **Memory MCP 工具**：给 Claude Code 等外部客户端读写 External Memory，不负责启动 AgentLoop。
- **未来如需外部触发 Research**：必须等异步 job / checkpoint 模型稳定后再设计，不能让 MCP 请求同步阻塞数分钟。

### 2.8 不变量

1. `Source snapshot` 只追加，不覆盖旧快照。
2. `active` 是默认可检索状态；`merged` / `superseded` 必须沿 redirect 找到最终 active，找不到则报错。
3. Prior 的旧来源不能混入本轮 `SourcePool`，否则会把旧证据伪装成新检索结果。
4. 报告中的 `[n]` 只指向本轮真实来源池；历史记忆 ID 必须作为 Prior 或引用卡片明确标记。
5. 任何 supersede、merge、conflict 都必须留下可审计记录，不能静默覆盖。
6. 所有成本数字必须能由 `tokens.jsonl`、`run-metrics.json` 和评测结果 JSONL 重算。
7. **历史版本永不删除**：supersede / conflict / invalidate 全程可追溯（第 5 条在 Memory 语义下的泛化强调）。
8. **写入必须经过 formation 判定或显式 store**，不做无差别全量入库（立场 0.5-1 的写入侧保证）。

---

## 3. 路线总览：4 个 Phase

| Phase | 名称 | 目标 | 退出条件 |
|---|---|---|---|
| P1 | External Memory | Memory Store/Search/MCP 接口层 + Automatic Write 起步 | `memory_*` MCP 面可用；记忆类型（kind）落库；formation 判定 MVP 可测 |
| P2 | Temporal Memory | freshness / expiration / supersession / conflict / timeline | 旧记忆不会无条件被信任（继承 v1 阶段 2 验收 + `ensure_index_fresh` content_hash 比对补办） |
| P3 | Dynamic Retrieval | budget-aware 检索策略（确定性策略先行） | 同题不同模式下成本下降、质量不劣化（继承 v1 阶段 4 验收并泛化到 memory recall） |
| P4 | Continual Memory | 新证据 → 更新 → 经验 → 演化闭环（maintenance / compaction 并入） | 可审计运行（继承 v1 阶段 5 + 阶段 3 的 checkpoint/compaction 作为支撑设施并行） |
| 支线 | 评测与开源交付 | 30–50 题 QA，对比 **无 Memory / Vector RAG / 时间感知 External Memory**（对照条件全集见 §7.2） | 对照可复跑、数字可重算；主题域待用户确认 |

P1–P4 依次回答 RQ1（P1 Formation）→ RQ2（P2 Temporal）→ RQ3（P3 Retrieval）→ 闭环（P4 Evolution）；评测支线并行推进。

### 3.1 旧版 → v2 映射表

| v1 章节 | v2 位置 |
|---|---|
| v1 §0 定位与核心假设 | §0（泛化为 Agent 外置记忆三问） |
| v1 §1 可演示基线 | §1（保留，追加阶段 1 实测证据） |
| v1 §2 目标架构 | §2（Wiki→Memory 投影改写，新增 schema/因子表） |
| v1 §3 路线总览（6 阶段表） | §3（4 Phase + 评测支线） |
| v1 阶段 1 · Wiki Prior MVP | **已完成**；证据入 §1.3，收尾工作入 P1（§3.2） |
| v1 阶段 2 · Freshness / Verification | P2 · Temporal Memory（§3.3） |
| v1 阶段 3 · Compaction / Checkpoint / Resume | P4 的支撑设施，并行推进（§3.5） |
| v1 阶段 4 · Adaptive Research | P3 · Dynamic Retrieval（§3.4，泛化到 memory recall） |
| v1 阶段 5 · Background Maintenance | P4 · Continual Memory（§3.5） |
| v1 阶段 6 · 评测/产品化 | §7 评测与开源交付（独立章） |
| v1 §10–§14（闸门/风险/结构/执行顺序/成果） | §8 / §4 / §5 / §6 / §9 |

### 3.2 P1 · External Memory（进行中，主体已完成）

P1 的 Prior 注入、run-metrics、cold/warm smoke 已完成（§1.3）。收尾范围：

1. **NoteMeta 扩展**：增加 `kind`（user / knowledge / experience，缺省 knowledge，向后兼容）与 `importance` 字段。
2. **Memory MCP 接口面**：`memory.*` 工具（§2.5），`wiki_*` 兼容保留。
3. **Memory Formation MVP**（以 RQ1 为锚）：确定性策略先行——candidate 判定 + importance/confidence 赋值 + **拒绝无差别入库**（不变量 ⑧）。

退出条件：`memory_*` MCP 面可用；`kind` 落库；formation 判定 MVP 可测（有离线测试与 smoke）。

### 3.3 P2 · Temporal Memory（改写自 v1 阶段 2）

**目标**：让系统知道一条记忆"什么时候不再应该被当成当前事实"（RQ2），并在新证据与旧记忆不一致时留下可追溯的状态变化。

**设计**（现有 frontmatter 已有 `volatility`、`observed_at`、`reviewed_at`、`status`、来源快照和 redirect 字段，本阶段扩展现有模型，不新建数据库）：

- freshness 是计算状态：`fresh`、`review_due`、`stale`；expiration 由 `valid_from` / `valid_until`（待加字段）表达。
- 生命周期状态保持：`active`、`merged`、`superseded`。
- 冲突独立落在 `conflicts/`，不把"冲突"伪装成普通 merge。
- 来源 URL 的 `content_hash` 变化时，旧记忆标记为需要重新验证；旧快照保留。
- **carried 项（v1 阶段 2 补办）**：`ensure_index_fresh` 补 content_hash 比对——索引落后检测不只看条目数，还要比对来源内容哈希。
- 新研究/新交互比较旧记忆与新证据：一致则更新 `reviewed_at`；更具体/更新则 supersede；无法判断则进入 conflict ledger。
- `memory.timeline`：一条记忆的状态变迁历史可查询。
- 投影层（pages）中的断言必须能追溯到 active note；引用 merged note 时给出 redirect 警告。

**计划交付物**（代码路径不变）：

- `src/researchwiki/wiki/freshness.py`：半衰期计算、review_due/stale 分类和队列排序。
- `src/researchwiki/wiki/verification.py`：旧记忆与新证据的结构化比较结果。
- `src/researchwiki/wiki/frontmatter.py`、`store.py`：补齐 valid_from/valid_until/reviewed/evidence 状态读写，保持旧数据兼容。
- `src/researchwiki/wiki/ingest.py`：把验证结果转成 merge / supersede / conflict 留痕。
- `src/researchwiki/wiki/lint.py`：增加 stale、孤立、无来源和断链统计。
- `tests/test_freshness.py`、`tests/test_verification.py`、`tests/test_supersession.py`。

**验收（退出条件）**：

- stable、drifting、volatile 三类记忆在固定时钟下得到可预测的 freshness 状态。
- 来源 content hash 变化不会覆盖旧快照；能够定位"哪条记忆依赖哪个旧快照"。
- 新证据与旧记忆冲突时写入冲突台账，不静默覆盖。
- supersede 后旧 ID 仍可读取，并沿 `superseded_by` 找到当前 active note。
- `lint --json` 输出能统计 freshness 和冲突健康度。
- 总闸：**旧记忆不会无条件被信任**。

### 3.4 P3 · Dynamic Retrieval（改写自 v1 阶段 4）

**目标**：不是每个问题都召回同样多、同样深的记忆（RQ3）。budget-aware 检索策略决定召回什么、召回多少：

```text
Simple  → 直接回答或轻量检索
Update  → 读取旧记忆 + 少量 fresh verification
Deep    → 预算完整的研究 loop
```

**实现策略**：先做**确定性策略**，不要一开始再增加一个分类模型：

```text
coverage   = 旧记忆命中覆盖度
freshness  = 命中记忆的时间有效性
volatility = 命中实体的变化风险
uncertainty = low confidence / conflict / unanswered signals
budget     = 当前可用 token 与时间预算
```

检索层按 Query 动态组合各因子（relevance × temporal validity × confidence × importance × conflict × budget）决定进 Context 的内容；只有当规则策略在评测中出现明显误判，再引入 cheap classifier，并把 classifier 自己的 token 成本计入总成本。

**计划交付物**：

- `src/researchwiki/loop/research_policy.py`：策略输入、决策和解释（泛化为 recall 策略）。
- `src/researchwiki/loop/agent_loop.py`：不同模式的检索上限、报告要求和验证要求。
- `config.toml`：模式预算、freshness 阈值、最小 fresh source 数。
- `tests/test_research_policy.py`：覆盖 stable/fresh、volatile/stale、conflict、空记忆库、无答案问题。
- `scripts/adaptive_smoke.py`：同题不同模式的成本与质量对照。

**验收（退出条件）**：

- 每次模式决策都记录理由，不允许只有 `simple/update/deep` 字符串而没有输入特征。
- 简单问题的 p50 时延和 token 低于 deep baseline。
- stale、conflict、低置信旧记忆不得被错误路由为"无需搜索"。
- 在固定 QA 集上，整体质量不低于无路由 baseline；成本、时延和 fresh search 次数有可重复下降。
- 总闸：**同题不同模式下成本下降、质量不劣化**（验收口径继承 v1 阶段 4，泛化到 memory recall）。

### 3.5 P4 · Continual Memory（改写自 v1 阶段 5，并入 v1 阶段 3）

**主目标**：让 Memory 从"研究结束后写入"升级为"随新证据与新交互持续演化"的闭环（Update / Supersede / Conflict / Invalidate），但所有自动动作必须可审计、可暂停、可回滚。

**四类维护动作**：

1. **merge**：相似且实体重叠的 active 记忆合并，保留规范 ID 与来源并集。
2. **refresh**：按 volatility 半衰期把 review_due/stale 记忆放入刷新队列，重新抓取原 URL 或候选权威来源。
3. **conflict**：同一实体出现矛盾断言时生成台账，下一次 Research 优先注入待解决问题。
4. **re-judge**：低置信、超龄或冲突未解决的记忆由当前模型重新判断；无法确认时 supersede，不删除历史。

**运行方式**：

- 先实现 `researchwiki consolidate --dry-run` 和一次性 CLI。
- 再实现进程内可重复的 job queue；不先引入 RabbitMQ、Redis 或分布式 scheduler。
- 每个 job 有 `job_id`、输入记忆 ID、执行时间、模型、token、结果和错误。
- 写入失败、抓取失败、模型输出畸形都必须保留原记忆，不得把失败当作新事实。
- 用户可以查看、重试、跳过和人工仲裁 conflict。

**支撑设施（改写自 v1 阶段 3，与 P4 并行推进）**：长程稳定性与成本控制是 Continual Memory 的前提——

1. **Token/context 估算**：对 message、工具 schema 和工具结果统一估算；以模型窗口配置为上限。
2. **Rolling compaction**：达到配置阈值（默认 70%）时，把旧轮次压缩成研究进展摘要，保留最近 K 轮和未解决任务。
3. **稳定 prompt 前缀**：system prompt、工具顺序、格式化模板固定；召回的记忆和动态来源放在后缀区域。
4. **Checkpoint**：记录已完成 step、子任务 ID、history 摘要、预算、状态文件版本和幂等键。
5. **Resume**：进程被 kill 后从 checkpoint 继续；已完成调用不重复扣费，已完成写入不重复创建。
6. **Watchdog 与压力脚本**：连续问题流、超时、工具失败和单个 provider 失败都必须有可读状态。

幂等与写入要求：LLM 调用幂等键 `hash(model + messages + tools)`；`state.md`、`meta.json` 使用 tmp + atomic rename；checkpoint 跳过已完成子任务；写入必须在已有文件锁或事务边界内；恢复后 `notes_created`、token 记账和 source snapshot 不重复膨胀。

**计划交付物**：

- `src/researchwiki/wiki/consolidation.py`
- `src/researchwiki/wiki/maintenance.py`
- `src/researchwiki/cli.py`：实现 `consolidate`、`arbitrate`、`--dry-run`、`--json`。
- `src/researchwiki/loop/compaction.py`、`src/researchwiki/loop/checkpoint.py`
- `src/researchwiki/loop/agent_loop.py`、`src/researchwiki/llm/accounting.py`
- `config.toml` 的 context、checkpoint、resume 配置
- `wiki-data/maintenance/`：job log、队列和结果。
- `scripts/stress_run.py`、`scripts/kill_resume_smoke.py`
- `tests/test_consolidation.py`、`tests/test_maintenance.py`、`tests/test_arbitrate.py`、`tests/test_compaction.py`、`tests/test_checkpoint.py`、`tests/test_resume.py`

**验收（退出条件）**：

- dry-run 能准确列出将要 merge/refresh/conflict/re-judge 的对象，不修改 Memory。
- 正式运行可重复执行；同一 job 重试不会重复生成记忆或 source snapshot。
- refresh 发现来源变化时保留旧快照，并创建可追溯的新 evidence。
- conflict 不能被自动 merge 掉；人工仲裁后有 resolution 和 resolved_at。
- 连续运行后 `lint --json` 的断链、孤立和无来源指标不恶化。
- 10 个以上连续问题运行 2 小时，零未解释崩溃；所有失败有 trace 和错误类型。
- 上下文 token 曲线在阈值内，compaction 次数、压缩前后 token 和保留摘要可审计。
- 人为 kill 后恢复完成，至少一个未完成任务继续执行，已完成任务不重复写入。
- provider 支持 cache 字段时报告命中率；不支持时只报告稳定前缀策略，不填造数字。
- 总闸：**可审计运行**。

---

## 4. 风险与裁剪顺序

### 4.1 主要风险

| 风险 | 预案 |
|---|---|
| Prior 反而让模型过度相信旧知识 | 强制 Prior 标签；旧来源不进 SourcePool；加入无答案和过期题 |
| 记忆越用越乱（Wiki→Memory 语义下的同一风险） | merge 留痕、lint、冲突台账、dry-run consolidation，不静默覆盖 |
| 搜索 provider 不稳定或无 key | Replay/Mock 开发；真实 smoke 明确回退边界；评测记录 provider |
| embedding 成本高 | 本地缓存；固定模型；评测和 memory baseline 使用同一 embedding |
| 长程上下文爆炸 | 先做工具结果截断，再做 compaction；预算双熔断保留 |
| 恢复导致重复扣费/重复写入 | 幂等键、checkpoint、atomic write、写入锁；专门 kill/resume 测试 |
| Formation 拒写入过严/过松 | importance 策略确定性先行、可解释；用评测题集校准阈值 |
| 小样本数字被误读为结论 | 每个数字标样本数、来源 JSON 和置信区间；最终结论只来自正式 QA（阶段 1 的 −25% 即样本数 2 的冒烟数据） |
| 过早引入基础设施 | 单进程 + SQLite 先完成；只有 2h 和多任务瓶颈被证实后再评估 Redis/RabbitMQ |

### 4.2 时间不足时的裁剪顺序

必须保留：

1. Formation MVP 与 run metrics。
2. freshness / conflict 的最小验证回路。
3. 30–50 题 对照评测（无 Memory / Vector RAG / External Memory）。
4. 一次可复现的开源安装路径。

可以后移：

1. pages（人类可读投影）的自动聚合和复杂双链导航。
2. Recall classifier 的模型版本；先使用确定性策略。
3. refresh 的自动 scheduler；先保留手动 CLI。
4. re-judge 和复杂 conflict arbitration。
5. 2 小时压测可以先降为 1 小时，但 kill/resume 不删除。

不裁剪：

- 无答案题。
- token/cost 原始凭证。
- source snapshot 和 supersession 留痕。
- MCP 本地安全边界。

---

## 5. 目标仓库结构

**概念名与代码路径的渐进映射**：文档层使用 Memory 语义（Memory Store = `wiki-data/`、Memory 条目 = notes、Wiki View = pages）；代码路径一律不改名（`src/researchwiki/wiki/` 等保持原样），等 memory_* 接口面稳定后再议路径迁移。

```text
researchwiki/
├── README.md
├── PLAN.md
├── pyproject.toml
├── config.toml
├── src/researchwiki/
│   ├── llm/                 # provider、router、accounting、replay
│   ├── loop/                # agent_loop、prior 接入、compaction、checkpoint、policy
│   ├── tools/               # search、fetch、sandbox fs、内部工具 registry
│   ├── wiki/                # Memory Engine：store、index、prior、freshness、verification、consolidation、lint
│   ├── mcp_server/          # External Memory Interface（wiki_* 兼容层 + memory_* 新面）；不负责长任务调度
│   ├── server/              # FastAPI + SSE
│   └── cli.py               # serve、lint、consolidate、arbitrate、serve-mcp
├── tests/
├── scripts/
│   ├── cold_warm_smoke.py
│   ├── adaptive_smoke.py
│   ├── stress_run.py
│   ├── kill_resume_smoke.py
│   └── run_eval.py
├── evals/
│   ├── qa/
│   ├── results/
│   └── reports/
├── docs/
│   ├── architecture.md
│   └── evaluation.md
├── web/
├── Dockerfile
├── docker-compose.yml
└── docker-compose.full.yml
```

运行时目录（默认 gitignore）：

```text
wiki-data/                    # Memory Store 的物理载体
├── sources/
├── notes/                    # 记忆条目
├── pages/                    # 人类可读投影（Wiki View）
├── conflicts/
├── runs/<run-id>/
│   ├── research-plan.md
│   ├── state.md
│   ├── report.md
│   ├── run-metrics.json
│   └── checkpoint.json
├── maintenance/
├── index.db
└── tokens.jsonl
```

---

## 6. 当前执行顺序

1. **P1 收尾**：① NoteMeta 增加 `kind`（user/knowledge/experience，缺省 knowledge，向后兼容）与 `importance` 字段；② Memory MCP 接口面（`memory.*` 工具，`wiki_*` 兼容保留）；③ Memory Formation MVP（确定性策略先行：candidate 判定 + importance/confidence 赋值 + 拒绝无差别入库），以 RQ1 为锚。
2. **P2 Temporal Memory**（含 carried 项：`ensure_index_fresh` 补 content_hash 比对）。
3. **P3 Dynamic Retrieval**。
4. **P4 Continual Memory**（checkpoint/compaction 作为支撑设施并行）。
5. **评测支线与开源交付**（主题域待用户确认，见 §7）。

**当前唯一的下一项工程任务：P1 收尾 ①+②（Memory 接口层），随后 ③（Formation MVP）。** 在 P1 收尾完成前，不启动 Knowledge Graph、分布式基础设施或完整后台 scheduler。

---

## 7. 评测与开源交付（独立章，收自 v1 阶段 6）

这一章不是重新做 Demo，而是把各 Phase 的收益变成可信、可复跑、可传播的交付。评测支线与 P2–P4 并行推进。

### 7.1 QA 数据集

先做 30–50 题 MVP，验证评测管线后扩展到约 100 题；2–3 个互有关联的主题域（**主题域待用户确认**），每题人工校验。

| 题型 | 目标占比 | 用途 |
|---|---:|---|
| 单跳事实 | 40% | 基础检索和引用正确性 |
| 多跳综合 | 30% | 记忆召回、实体双链和跨条目综合 |
| 时效题 | 15% | freshness、refresh、supersede |
| 无答案题 | 15% | 幻觉抑制与"知道不知道" |

### 7.2 对照条件

至少保留以下条件，所有条件使用同一生成模型、同一 embedding、同一题集和相同回答长度约束：

1. **无 Memory baseline（Memoryless online）**：每题从零研究，不把它当准确率上界。
2. **Vector RAG over sources**。
3. **Hybrid RAG over sources**。
4. **External Memory warm**：记忆召回 + fresh verification + Memory update（对应 ResearchWiki warm start）。
5. **External Memory + maintenance**：在 P2/P4 的 freshness/maintenance 功能完成后增加。

### 7.3 指标

#### 成本与速度

- 输入 token、输出 token、总 token。
- 单次查询成本。
- Memory 构建一次性成本。
- 摊销成本：`build_cost / reuse_count + query_cost`。
- 端到端时延、p50、p95。
- fresh search 次数、fetch 次数、来源数。

#### 质量与可靠性

- normalized EM / 包含匹配。
- LLM judge：覆盖度、引用正确率、时效性，各 1–5 分。
- 人工抽检至少 20 题核对 judge。
- citation coverage、unsupported claim rate。
- freshness 状态正确率、conflict resolution 正确率。
- 无答案题的拒答/不确定表达率。

#### Harness

- compaction 次数和压缩前后 token。
- checkpoint 恢复成功率。
- 重复 LLM 调用率、重复 Memory 写入率。
- 2h 崩溃数、超时数、失败可恢复率。
- provider 支持时的 cache 字段和命中率。

### 7.4 结果产物

- `evals/qa/*.jsonl`：题集和人工标注。
- `evals/results/*.jsonl`：逐题原始结果。
- `evals/reports/*.md`：实验报告和统计方法。
- `scripts/run_eval.py`、`scripts/report_eval.py`：固定参数、可重跑。
- 所有结论都引用原始 `tokens.jsonl`、`run-metrics.json` 和逐题结果，不只提交一张手工表格。

### 7.5 开源交付

- README 改为当前定位：External Agent Memory、evidence-grounded、prior reuse、fresh verification，ResearchWiki 为参考实现。
- 增加一张真实架构图和一张 cold/warm 结果图。
- MCP `uvx` 安装说明和 Claude Code 配置片段（含 `memory_*` 接口面）。
- 提供脱敏小型 Memory fixture、Mock 零 key 复现和真实模型配置说明。
- GitHub Actions：pytest、ruff、前端 typecheck、最小 mock smoke。
- Docker mock/full 两档构建实测；真实模式明确搜索 provider、模型和 embedding 配置要求。
- 安全检查：API key 不入库、MCP 默认本地监听、远程抓取 SSRF 防护、限速、来源合规和示例数据脱敏。
- 发布前至少收集 1 个外部用户从安装到 MCP 读写或 Web 研究跑通的反馈。

---

## 8. 跨阶段评测闸门

每个阶段完成时都必须留下以下四类证据：

1. **代码测试**：对应模块的 pytest；不允许只用真模型手工截图作为验收。
2. **行为 smoke**：使用 ScriptedProvider / ReplayProvider 固定模型响应，验证状态机和边界。
3. **真实模型小样本**：只验证 provider、搜索和 embedding 的现实差异；明确样本数和失败样本。
4. **原始数据**：tokens、run metrics、逐题结果和命令行输出。

阶段完成报告至少回答：

```text
做了什么？
哪一条假设被验证或被否定？
成本/速度/质量改变了多少？
数字能否从仓库重算？
哪些情况仍然没有覆盖？
```

---

## 9. 预期最终成果

项目完成后，README 和求职材料应该能用真实数据回答：

- **RQ1（Remember）**：formation 判定能否在保持召回覆盖的同时，拒绝无差别入库并降低存储/上下文成本？
- **RQ2（Maintain）**：来源变化时，系统能否发现旧记忆 stale，并留下 supersede / conflict / invalidate 证据链？
- **RQ3（Recall）**：budget-aware 召回相比无差别召回 / Vector RAG / Hybrid RAG，成本降低多少、质量是否不劣化？
- **复用收益**：第二次相近研究相比冷启动减少了多少输入 token、搜索和时延？引用正确率、时效性和无答案题表现是否保持？
- **可恢复性**：长时间运行或进程被 kill 后，能否恢复而不重复扣费、不重复写入？
- **外部复现**：一个陌生用户能否用 Docker 或 MCP 在本地复现？

最终项目叙事：

> **我从一个可演示的 Research Agent 出发，把它收敛为一个外置、时间感知、可演化的 Agent 记忆系统，并在研究场景中验证了记忆的形成、保鲜、检索与演化如何共同构成一个可复算、可恢复、可持续演化的 External Memory Infrastructure。**
