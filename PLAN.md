# ResearchWiki — 自进化研究 Wiki 智能体 · 项目计划

> **一句话**：给它一个研究问题和一批信息源，它长程地搜索、阅读、综合，把结论沉淀成一份带交叉引用的个人 wiki；后续提问时优先复用 wiki，越用越快越准。
>
> **假设**：单人开发，**按阶段推进、不按周排期**（共 6 个阶段，阶段 1 已于 2026-09-17 完成；时间不够时按 §7 裁剪范围），预算敏感（长程任务 token 消耗大，分层用模型）。

---

## 0. 目标与边界

**核心目标（四件事）**

1. **长程 loop**：主 agent 规划、子 agent 隔离上下文执行搜索/阅读，稳定运行 2h+ 不崩。
2. **上下文工程**：滚动 compaction + 文件系统即上下文 + 稳定 prompt 前缀吃 KV-cache。
3. **记忆蒸馏**：原始来源 → 原子笔记 → wiki 条目三层管线，后台 consolidation 自我维护，并以 MCP server 形式可被别人装进 Claude Code。
4. **评测数字**：wiki 复用 vs 冷启动 / vs RAG 基线的准确率、token 成本对比。

**明确不做**（写进 README 的 Non-goals，防止 scope 蔓延）

- ❌ LangChain / LangGraph / dify —— 自研极简 loop 是本项目的核心设计
- ❌ 重型向量库服务（Milvus/Qdrant server）—— SQLite 嵌入式足够
- ❌ ~~Web UI~~ → ✅ **Web UI 是一等公民**（2026-09-17 用户决策：交互式 CLI 砍掉，Web 先做，"前端好看再往后做"，见 §1 前端层与阶段 1）
- ❌ 交互式 CLI——仅保留 `lint` / `consolidate` / `arbitrate` / `serve-mcp` 等开发用单命令入口
- ❌ 模型微调 —— 纯 prompt + harness 工程
- ❌ 多用户 / 账号 / 协作

---

## 1. 总体架构

```
   用户 ──►  Web UI (Next.js + Vercel AI SDK + AI Elements)
            │  流式输出 · Reasoning shimmer 折叠 · 过程时间线 · 三视图
            │  SSE (AI SDK Data Stream 协议)
                    ┌─────────────────────────────────────────┐
                    │           FastAPI Server (SSE)           │
                    └─────────────────────────────────────────┘
                    ┌─────────────────────────────────────────┐
                    │              Main Agent (Planner)        │
                    │  research-plan.md + state.md + 摘要       │
                    │  ← 上下文恒定小，靠文件系统存状态            │
                    └──────┬───────────────────────▲──────────┘
                           │ 派发任务简报             │ 结构化结论 JSON
                    ┌──────▼───────────────────────┴──────────┐
                    │      Research Sub-agents (一次性)         │
                    │  独立上下文：search → fetch → read         │
                    │  只回传 {findings, candidate_notes, ...}  │
                    └──────┬──────────────────────────────────┘
                           │
        ┌──────────────────▼───────────────────────────────────┐
        │  Tools (MCP): web_search · fetch_url · fs_read/write   │
        ├──────────────────────────────────────────────────────┤
        │  Wiki 三层存储: sources/ → notes/ → pages/  + index.db │
        │  Distiller(蒸馏) · Consolidator(后台合并/对账) · Linter │
        └──────────────────────────────────────────────────────┘
                           │
                    ┌──────▼───────────┐
                    │  Wiki MCP Server │ ← 可装进 Claude Code，天然传播点
                    └──────────────────┘
```

五个 harness 组件与对应的业界设计：

| 组件 | 对应业界概念 | 本项目的实现 |
|---|---|---|
| Agent loop | sub-agent architecture | 主 agent 只做规划调度；研究子 agent 用完即弃、只回传结论 |
| 上下文管理 | rolling compaction / KV-cache | 超 70% 窗口触发压缩，早期对话折叠进 state 文件；system prompt 与工具表保持稳定 |
| 记忆 | sleep-time compute | 后台 consolidation：合并重复笔记、刷新过期条目、矛盾进对账台账 |
| 工具层 | MCP | 搜索/抓取/文件系统 + wiki 读写全部走 MCP 协议 |
| 验证回路 | grounding / citation | 断言强制行内引用笔记 ID，笔记强制带 URL；冲突不静默覆盖，走对账 |

**前端选型（照抄现有成熟方案，不自己造轮子）**

- 底座：fork [vercel/ai-chatbot](https://github.com/vercel/ai-chatbot)（Next.js App Router + AI SDK `useChat` + shadcn/ui）——流式、会话管理、持久化开箱即用。
- 过程组件：Vercel **AI Elements**——`<Reasoning>` 组件原生支持"流式思考 + shimmer 动效 + 完成后自动折叠"，正是思考过程折叠的现成实现；`<Task>` / `<Tool>` 做子任务与工具调用过程可视化；`<Source>` 做引用来源展示。
- 对接：FastAPI 以 SSE 输出 **AI SDK Data Stream 协议**（JSON-lines，手写几十行），把 loop 事件（文本增量 / 思考 / 子任务开始结束 / 冲突提示 / 报告完成）映射为自定义 data parts。
- 备选：[assistant-ui](https://github.com/assistant-ui/assistant-ui)（shadcn 风格 React 组件，LangGraph runtime 集成深）；LobeChat（完整产品但太重，难剥离）。

**结果呈现：三视图（过程流 / 文档结果 / 知识库沉淀）**

1. **对话流 = 过程**：时间线展示研究计划 → 子 agent 任务卡片（运行中 shimmer、完成后折叠为一行摘要 + 发现数）→ 冲突/对账提示。参考 ChatGPT / Gemini deep research 的过程可视化。
2. **报告 = 文档视图**：最终报告**不塞进聊天流**——完成后出一张"报告卡片"入口，点开是类文档页面：正文 Markdown，断言处 `^N-0042` 上标引用，hover 弹出笔记卡（事实原文 + 来源 URL + 置信度），左侧栏显示双链相关条目可跳转。
3. **Wiki = 沉淀面板**：三层浏览（pages / notes / sources）+ 双链跳转；研究进行中实时显示"新增笔记流"，直观呈现"wiki 在生长"——这是本项目区别于普通 deep research 的演示点。

---

## 2. 技术选型

| 项 | 选择 | 理由 / 备选 |
|---|---|---|
| 语言 | Python 3.12 + uv | 评测生态最好；uv 管理依赖与脚本 |
| LLM 接入 | OpenAI 兼容协议统一接入；代码内做 **model router**（strong/cheap 两档）；**骨架先行**——先实现 Provider 接口 + MockProvider（录制回放），全部测试不依赖 API key，真模型接入只是配置问题 | 规划、蒸馏、judge 用强模型；子 agent 搜索阅读用便宜模型（GLM / DeepSeek / Kimi 均兼容 OpenAI 协议）。换 provider 只改配置 |
| MCP | `fastmcp`（Python 官方 SDK） | 几十行就能发布一个 MCP server |
| 搜索 | Tavily 或博查 API（国内可达优先） | 备选 Brave Search；封装成统一 `web_search` 工具 |
| 网页抓取 | httpx + trafilatura（正文提取） | 反爬失败时降级走 Jina Reader (`r.jina.ai`) |
| 索引 | SQLite FTS5（**wangfenjin/simple** 中文+拼音 tokenizer 优先，FTS5 trigram 兜底）+ sqlite-vec（向量） | FTS5 默认 unicode61 对中文几乎不切词；simple 是现成的中文分词扩展（jieba 词典 + 自动 query 组装），Day 1 验证；sqlite-vec 较新，**锁定版本**，fallback 为 numpy 暴力余弦（数据量小足够跑） |
| 嵌入 | **默认走 embedding API**（OpenAI 兼容，智谱 / SiliconFlow 等，config.toml 切换）；备选本地 bge-small-zh-v1.5（CPU） | 免运维、与 LLM 同一 provider 体系；嵌入结果**本地缓存**（SQLite 表，key=text+model），保证评测可复现且同文本不重复计费；wiki 与 RAG 基线必须用同一嵌入模型 |
| Wiki 存储 | 纯 Markdown + YAML frontmatter + `[[双链]]` | 人类可读可 diff、git 可版本化——这本身就是卖点 |
| 可观测性 | 每次 LLM 调用写 JSONL（trace_id / step / model / tokens / latency / error_type / cost，**cache 相关原始字段**——`cached_tokens`、`cache_read_input_tokens` 等，provider 返回什么记什么）；开发与评测用**录制回放缓存**（缓存 LLM 响应与网页抓取结果） | 复用收益"token 降低 X%"的原始凭证；长程调试靠 trace；压测与评测成本可控可重放 |
| 配置 | `config.toml`：模型路由、token 预算、阈值、路径、provider 缓存开关 | 换 provider / 调参不动代码 |
| 测试 | pytest；评测 harness 独立目录 | — |

**开源参考（借鉴设计与 prompt 结构，核心代码自研——§8 红线不变）**

| 项目 | 借鉴点 |
|---|---|
| [stanford-oval/storm](https://github.com/stanford-oval/storm) | 多视角提问 → 带引用的 Wikipedia 式报告管线；outline 驱动研究 |
| [assafelovic/gpt-researcher](https://github.com/assafelovic/gpt-researcher) | Researcher / Reporter 角色分工与来源管理 |
| [langchain-ai/open_deep_research](https://github.com/langchain-ai/open_deep_research) | 模型 / 搜索 / MCP 全可换的编排结构 |
| [huggingface/smolagents](https://github.com/huggingface/smolagents)（open-deep-research 示例） | 极简 loop 的参照实现，GAIA 打榜思路 |
| [letta-ai/letta](https://github.com/letta-ai/letta) + [sleep-time-compute](https://github.com/letta-ai/sleep-time-compute) | sleep-time consolidation 的工程化范式——与我们 Consolidator 的 merge/refresh/conflict 对齐 |
| [mem0ai/mem0](https://github.com/mem0ai/mem0) | 记忆 add/update/merge 的操作流水线 |
| [wangfenjin/simple](https://github.com/wangfenjin/simple) | SQLite FTS5 中文/拼音分词扩展，直接可用，省掉自写 tokenizer |
| [vercel/ai-chatbot](https://github.com/vercel/ai-chatbot) + [AI Elements](https://github.com/vercel/ai) | 前端底座直接 fork；`<Reasoning>` shimmer 折叠 / `<Task>` 过程可视化 / `<Source>` 引用组件开箱即用 |
| [Picrew/awesome-agent-harness](https://github.com/Picrew/awesome-agent-harness) | harness 工程资源索引，持续翻新 |

---

## 3. Wiki 三层数据模型（具体 schema）

```
wiki-data/
├── sources/                 # 第 1 层：原始来源（append-only，不可改）
│   └── {url_sha1}/
│       └── {content_hash}/  # 快照式：同一 URL 重抓后内容变了 → 新快照目录，旧快照永久保留
│           ├── meta.json    # {url, title, domain, fetched_at, content_hash, http_status}
│           └── content.md   # trafilatura 提取后的正文
├── notes/                   # 第 2 层：原子笔记（一条事实一条，可合并/废弃）
│   └── N-0042.md
├── pages/                   # 第 3 层：wiki 条目
│   ├── entities/GLM-5.3.md  # 实体页
│   └── topics/上下文压缩.md  # 主题页
├── conflicts/               # 对账台账
│   └── C-0007.md            # {冲突描述, 双方证据, 状态: open/resolved}
└── index.db                 # SQLite: FTS5 全文 + sqlite-vec 向量 + 实体注册表 + 链接表
                             # 笔记/页面引用 sources 时指向具体快照（url_sha1 + content_hash），refresh 的证据链由此而来
```

**原子笔记 frontmatter 规范**：

```markdown
---
id: N-0042
type: fact
entities: [GLM-5.3, Context-Compression]   # 指向实体注册表的规范名
sources:
  - url: https://example.com/xxx
    span: "para 3"                          # 源文内的定位
confidence: high
created_at: 2026-09-17T10:00:00Z
updated_at: 2026-09-17T10:00:00Z
status: active                             # active | merged | superseded
redirect_to: null                          # status=merged 时必填：指向合并目标 N-00xx；旧 ID 永不消失，引用自动跟随
---

GLM-5.3 的上下文窗口为 200K tokens。（正文只写这一条事实）
```

**Wiki 条目规范**：正文每个断言行内标注 `^N-0042`；相关条目用**稳定 ID 双链** `[[entity:glm-5.3|GLM-5.3]]`（链接用 ID、显示名可变），不用裸显示名——中文别名、重命名、笔记合并时裸名极易断链；frontmatter 记 `reviewed_at`（供 consolidation 判断过期）。

**实体注册表**（index.db 内一张表）：稳定 ID + 规范名 + 别名列表，解决 "GLM-5.3 / glm5.3 / 智谱GLM" 归一问题——双链质量的关键。

---

## 4. 核心机制设计要点

**① 子 agent 上下文隔离**
- 任务简报字段：`{question, search_budget(次数), token_budget, focus_entities, output_schema}`——**子 agent 有独立的 token_budget**，不只是次数限制。
- 返回固定 schema：`{findings[], candidate_notes[](符合 §3 frontmatter), sources[], open_questions[], tokens_used}`。
- 主 agent **只**接收该 JSON，不接收网页原文——这是上下文不爆炸的根本保证。
- 子 agent 内部同样防爆：单次 `fetch_url` 设最大字符截断；长文分块阅读 + 子 agent 内滚动摘要；工具结果原文落盘 `sources/`，上下文里只留截断预览，回传只带摘要与引用。

**② 滚动 compaction**
- 触发：上下文 > 模型窗口 70%。
- 动作：把早期轮次压成一段"研究进展摘要"追加进 `state.md`（已有发现 / 待解决问题 / 下一步），保留最近 K 轮原文。
- KV-cache 友好：system prompt 固定措辞、工具定义顺序固定、文件命名 append-only（`state.md` 不改名，只追加）。
- 缓存数据口径：并非所有 OpenAI 兼容 provider 都返回 `cached_tokens`（Anthropic 还需显式 `cache_control`）。`tokens.jsonl` 记录 provider 返回的原始缓存字段，**只对支持缓存的 provider 报命中率**；不支持的退化为"稳定前缀 + 固定工具序"的定性优化，不硬造数字。

**③ Consolidation（sleep-time compute）**
- 触发：每次研究 run 结束后自动跑一次，或 `researchwiki consolidate` 手动触发。
- 三个动作：**merge**（新笔记与旧笔记嵌入相似度 > 阈值 → 合并，旧笔记 status=merged + `redirect_to` 留痕，旧 ID 永不消失）；**refresh**（条目 `reviewed_at` 超过 N 天 → 重抓核对更新，证据链靠 sources 快照版本）；**conflict**（同实体出现矛盾断言 → 写入 `conflicts/` 台账，下轮研究优先核实，绝不静默覆盖）。
- 冲突检测**批量**做：run 结束后对新笔记按实体分组批量比对，不做逐条实时 LLM 比对（省 token）；配套 `researchwiki arbitrate` CLI 人工仲裁入口。

**④ 验证回路**
- 硬约束（写入时 lint）：页面断言必须行内引用**存在且 active** 的笔记 ID（引用到 merged 笔记时自动跟随 `redirect_to` 并提示改写）；笔记必须有 URL；双链必须指向注册表中的实体稳定 ID。
- `researchwiki lint` 输出健康度指标：引用覆盖率、孤立笔记率、断链数——这些指标本身可以进评测报告。

---

## 5. 里程碑拆解（分阶段）

> 不按周排期，按阶段推进；工作量占比仅供预估节奏。每个阶段有明确验收标准，验收不过不进下一阶段。复选框可直接当 TODO 用。

### 阶段 1 · 骨架与 Web 前端（✅ 已完成 2026-09-17，~15%）

- [x] 仓库初始化：uv + pyproject + 目录骨架 + ruff/pytest 本地全绿（CI 待 GitHub 建仓后补，移入阶段 6）
- [x] LLM 骨架：Provider 接口 + MockProvider（脚本回放、零 API key）+ token 记账 JSONL（含 cache 原始字段位）
- [x] FastAPI server：SSE 输出 AI SDK UI Message Stream 协议，研究 run 事件（思考 / 任务 / 笔记 / 冲突 / 报告 / 来源）映射为 data parts
- [x] Web 前端（超额完成，交互式 CLI 已砍）：Next.js + AI SDK——流式输出、思考折叠行（含持续秒数、正文淡色）、研究过程折叠块（ZCode 风格计数摘要 pill）、原子笔记流、冲突提示、报告文档视图（引用上标 chip + hover 悬浮笔记卡 + 来源列表）、Wiki 沉淀面板、智能滚动、中性深灰主题
- [x] Mock 冒烟：端到端演示通过（后端 5 测试 + tsc 零错误 + 浏览器实测）

**完成标志**：不接任何真模型，即可在 Web UI 完整演示一次研究的全过程与产出。

### 阶段 2 · 真实 loop + 工具 + 真模型接入（原阶段 1 核心剩余，~25%）

- [x] LLM 真实现：OpenAI 兼容 Provider（流式 + 工具调用解析）、model router（strong/cheap）、重试与退避、录制回放缓存（2026-09-17，16 个 MockTransport 测试）
- [x] 工具：`web_search`（Tavily / 博查统一封装 + mock 夹具）、`fetch_url`（trafilatura + Jina 降级 + 最大字符截断 + sources 快照落盘）、`fs_read/fs_write`（沙箱限制在 wiki-data/）（2026-09-17，14 个测试）
- [x] 极简 agent loop：plan → act(tools) → observe 循环，工具注册表，max-steps / token-budget 熔断（2026-09-17，AgentLoop + ToolRegistry + ScriptedProvider）
- [x] **最小 state 落盘 + 工具结果统一截断**（完整 compaction 留到阶段 4，但不做这两样一个复杂问题就会撑爆上下文）（2026-09-17，runs/{ts}-{trace}/research-plan.md + state.md + report.md，registry 统一截断）
- [x] Research 子 agent：独立上下文 + 独立 token_budget + 固定返回 schema（2026-09-17，ResearchSubagent，工具集无 dispatch_research 防递归）
- [x] 主 agent 的 `research-plan.md` / `state.md` 文件约定（2026-09-17，atomic_write_text 每步重写）
- [ ] 两道技术闸（需要 API key，见文末 Day 1 清单）：cheap 模型工具调用稳定性；中文检索栈召回
- [x] server 接真实 loop（config 开关，mock 保留为默认演示模式）（2026-09-17，config.toml `[server] mode`，默认 mock 已验证不变）
- [ ] 真模型冒烟：1 个陌生主题端到端 → 报告进 Web 文档视图，每条结论有来源 URL；`tokens.jsonl` 能算出本次总成本；全程无人工干预

**验收**：config 切到真模型后，同一 Web UI 跑通陌生主题研究；子 agent 全程 cheap 档；一次 run 的成本可从记账 JSONL 精确复算。

### 阶段 3 · Wiki 蒸馏 + MCP server（~20%）

**并行支线**：评测题集构建在本阶段启动（见 §6），每天人工校验 10 题，避免阶段 5 前集中赶工。

- [x] 三层存储实现（sources 快照式）+ frontmatter 读写 + 实体注册表（稳定 ID + 别名归一）。frontmatter 一次定齐防迁移字段：`status: active|merged|superseded`、`redirect_to`/`superseded_by`、`volatility: stable|drifting|volatile`（半衰期分级）、`observed_at`；检索默认只返回 active、跟随 redirect，排序乘置信度与新鲜度因子（2026-09-20，frontmatter.py + entities.py + store.py；sources 快照式在阶段 2 已由 fetch.py 落地）
- [x] Distiller：报告 → 原子笔记（LLM 抽取）→ 实体/主题页（LLM 聚合 + 稳定 ID 双链）（2026-09-21，wiki/distiller.py；提示词单一持有，异常/畸形输出一律降级为空列表并记 degradations，不打断整轮研究）
- [x] **入库去重（轻量 merge）**：候选笔记写库前与既有笔记查重，命中则合并 + redirect_to——验收的"重复笔记 <10%"由这一步保证（完整 consolidation 在阶段 4）（2026-09-21，wiki/ingest.py：向量余弦 ≥ 阈值 **且** 实体重叠才判重，保留既有规范 ID + sources 并集 + merged 留痕；反向用例确保阈值不是橡皮图章）
- [x] 索引：FTS5（wangfenjin/simple 中文 tokenizer，Day 1 已验证）+ sqlite-vec（锁定版本，fallback numpy 余弦）建库，`wiki_search`（关键词+向量混合，嵌入走 API + 本地缓存）（2026-09-20：sqlite-vec 0.1.9 实测可用，退路为**纯 Python 余弦暴力扫描**而非 numpy——省一个依赖且语义一致性有测试保证；tokenizer 为 auto 探测：simple 的 jieba 词库在**含中文的路径下会 C++ abort**，故本机走 trigram，探测保护已实现）
- [x] `researchwiki lint`（引用覆盖、断链——含 merged 笔记 redirect 跟随、孤立笔记）（2026-09-21，wiki/lint.py + CLI `lint` 子命令，支持 `--json` 与 CI 退出码）
- [x] MCP server：`wiki_search / wiki_read / wiki_write / wiki_list_changes` 四个工具；**写工具加保护**——frontmatter schema 校验、路径限制在 `wiki-data/` 内、写前备份（2026-09-21，fastmcp 实现，另加 `wiki_health` 共 5 个工具；备份失败即拒写；验证阶段修掉 list_changes 分页静默丢条目的缺陷）
- [ ] 在 Claude Code 里实测安装该 MCP 并读写 wiki（截图留档，README 素材）——**需用户在本地 Claude Code 实操**，README 已备好注册配置片段

**验收**：对同一主题域连续研究 5 个问题，wiki 条目互相双链、笔记去重率达标（重复笔记 < 10%）；Claude Code 能通过 MCP 查到 wiki 内容并新增一条笔记。

### 阶段 4 · Compaction + 长程稳定性（~15%）

- [ ] 滚动 compaction（70% 阈值 + state.md 折叠）
- [ ] 稳定 prompt 前缀改造 + **KV-cache 命中率统计**（只对支持缓存的 provider 报数，见 §4②）
- [ ] Consolidation 后台任务（merge / refresh / conflict 三动作 + 第四动作 re-judge：低置信/超龄 volatile 笔记用当前模型重审，次品标 superseded 留痕）；refresh 队列按 volatility 半衰期调度（volatile 30d / drifting 90d / stable 不主动刷新）
- [ ] 冲突对账流程：台账 → 下轮研究任务注入 → 解决后回写
- [ ] Checkpoint / 断点续跑 + watchdog（进程崩溃自动恢复），且**恢复幂等**：LLM 调用带幂等键（hash(model+messages+tools)，防恢复后重复扣费）；`state.md` / `meta.json` 用 tmp+rename 原子写；wiki 写入走 SQLite 事务或文件锁；checkpoint 记录已完成子任务 ID，恢复时跳过
- [ ] 压力测试：脚本驱动 10+ 问题连续流，跑满 2 小时

**验收**：连续 2h 运行零崩溃；上下文 token 曲线稳定在阈值内（贴进博文的图）；人为 kill 进程后能从 checkpoint 恢复继续。

### 阶段 5 · 评测（~15%）

- [ ] QA 题集构建（~100 题，见 §6 详细设计）
- [ ] 基线实现：纯向量 RAG、混合 RAG（FTS+向量），检索对象 = 原始 sources；**基线公平性**——chunk 大小、top-k、嵌入模型、重排器、生成模型与 wiki 方法对齐（同一嵌入模型、同一生成模型），并在报告里写明这些超参
- [ ] **成本口径**：分别报告单次查询成本、wiki 构建一次性成本、**摊销成本（构建成本/复用次数 + 查询成本）**——"token 降低 X%"必须同时给单次与摊销两个口径
- [ ] 实验一：wiki 整读 vs 向量 RAG vs 混合 RAG（准确率 + token + 时延）
- [ ] 实验二：冷启动 vs 热启动（复用收益）
- [ ] 实验三：报告质量（LLM judge rubric + 人工抽检 20 题）
- [ ] 稳定性数据整理：2h 压测的崩溃数、恢复成功率、token 曲线
- [ ] 出评测报告（图表 + 结论），所有数字有 `tokens.jsonl` 原始凭证

**验收**：拿到核心对比数字——"wiki 复用使重复问题 token 成本降低 X%、准确率变化 Y%"；对比图表可直接放进 README/博文。

### 阶段 6 · 打磨与开源（~10%，可压缩；Web 前端已在阶段 1 完成，不再是打磨项）

- [x] README：quickstart（陌生人 30 分钟内复现）、架构图、评测表（2026-09-20 首版：定位与差异点、免 key 快速开始、阶段进度、关键设计；**动图 demo 与评测数字待评测阶段回填**）
- [x] **Docker 一键部署（两档：mock 零 key 演示 / 完整接真模型 + wiki 持久化）**（2026-09-20，提前完成：Dockerfile + web/Dockerfile(standalone) + docker-compose.yml + docker-compose.full.yml + .env.example；本机未构建镜像验证，首次部署时需实测）
- [x] MIT 协议 + GitHub 公开仓库（2026-09-20：https://github.com/111wxh/researchwiki）
- [ ] MCP server 发布说明（`uvx` 一键安装进 Claude Code 的配置片段）
- [ ] 博文一篇：讲 harness 设计（compaction / 三层记忆 / 对账），带评测数字
- [ ] 示例 wiki 数据（脱敏小样例）、FAQ、CI（GitHub Actions 跑 pytest + ruff）
- [ ] 安全合规检查：抓取遵守 robots.txt 与限速；API key 走 .env 且不入库；示例数据脱敏；MCP server 仅监听本地

**验收**：发到社交平台/社区，附 demo；收到至少 1 个外部用户跑通反馈。

---

## 6. 评测方案细化

**题集构建（~100 题，2–3 个主题域，域内问题要互相关联才能形成 wiki 网络）**

| 题型 | 占比 | 目的 |
|---|---|---|
| 单跳事实 | 40% | 基础检索能力 |
| 多跳综合 | 30% | 考验跨条目双链/整合，wiki 应显著赢 |
| 时效题 | 15% | 答案随时间变化，考验 consolidation/对账 |
| **无答案题** | 15% | wiki 里没有 → 应答"不知道"，测幻觉抑制 |

出题方式：先人工选定域和 20–30 个种子来源，LLM 辅助出题。**先做 30–50 题 MVP 人工逐题校验，跑通评测管线后滚动扩到 100 题**——单人做 100 题校验 + 仲裁远不止 2 天，不要等题集齐了才开评测。判分：短答案 **normalized EM + 包含匹配**（中文 exact match 不友好）+ LLM judge 双判，分歧题人工仲裁。

**三组实验的变量控制**

1. **检索读法对比**：同一题集、同一模型，四种条件——(a) 全新研究（**无记忆在线研究基线**——注意它不是上界：准确率可能更高但成本也高，仅作参照）；(b) 向量 RAG over sources；(c) 混合 RAG over sources；(d) 读 wiki（pages + 按需展开 notes）。指标：准确率、输入/输出 token、端到端时延。
2. **复用收益**：第 1 次问主题 A（冷启动，全量研究并写 wiki）→ 第 2 次问 A 的近邻问题（热启动，wiki 已有内容）。指标：token、时延、准确率。**同时报告单次查询成本与含构建摊销的总成本**——只报单次成本会高估复用收益，只报摊销会掩盖查询本身的便宜，两个口径都给（**这是复用收益核心数字的来源**）。
3. **报告质量**：LLM judge 按 rubric 打 1–5 分（覆盖度 / 引用正确率 / 时效性），另人工抽 20 题核对 judge 可信度。

**附加指标**（体现 harness 深度）：KV-cache 命中率（compaction 前后对比，仅支持缓存的 provider）、崩溃恢复成功率、wiki 健康度（lint 指标随使用轮次的变化）。

---

## 7. 范围裁剪版（时间不够时的底线）

- 阶段 2、3 范围不变，但 consolidation 只做 merge，refresh/conflict 移出范围。
- 阶段 4 压缩：compaction 只做简化版，稳定性压测从 2h 降为 1h。
- 阶段 5：题集降到 60 题（无答案题保留——它是差异化亮点）；实验一只跑 wiki vs 向量 RAG 两条件。
- 阶段 6：砍掉博文之外的一切打磨（Web 前端阶段 1 已完成，不再是被裁对象）。

---

## 8. 风险与预案

| 风险 | 预案 |
|---|---|
| 长程跑 API 成本失控 | 每 run 设 token budget 熔断；子 agent 全用 cheap 档；token 记账每天看一次；开发与评测用录制回放缓存 |
| provider 不返回缓存字段，KV-cache 命中率拿不到 | tokens.jsonl 记原始字段；只对支持缓存的 provider 报命中率，其余退化为定性描述 |
| 跑 2h 崩溃 | 从阶段 2 起 state 落盘就是硬约定（文件系统即上下文），checkpoint 是自然产物而非补丁 |
| wiki 越用越乱（条目膨胀、重复、断链） | lint 健康度指标进阶段 3 验收；consolidation 定期回收；实体注册表防别名分裂 |
| 搜索/抓取被反爬 | Jina Reader 降级链路；限速；题集构建时人工筛过可达的来源 |
| 评测集构建拖期 | 出题作为阶段 3 的并行支线启动（每天校 10 题）；60 题是底线 |
| 双链质量差导致实验一对 wiki 不利 | 实体规范化先行（注册表 + 别名合并）；实验一里报告 wiki 健康度指标作为背景 |
| 自研 loop 遇到死角（工具调用不稳定） | 允许参考 Claude Agent SDK 的 prompt 结构，但代码自研——核心设计不能丢 |

---

## 9. 仓库结构（阶段 2 结束时应长这样）

```
researchwiki/
├── pyproject.toml
├── README.md
├── PLAN.md                      # 本文件
├── src/researchwiki/
│   ├── llm/                     # client, router(strong/cheap), token 记账
│   ├── loop/                    # 主循环, planner, subagent, compaction
│   ├── tools/                   # web_search, fetch_url, fs, wiki_io
│   ├── wiki/                    # 三层存储, index, distiller, consolidation, lint
│   ├── mcp_server/              # wiki MCP server 入口
│   ├── server/                  # FastAPI: SSE Data Stream 协议, 会话管理
│   ├── eval/                    # qa_builder, baselines, judge, report
│   └── cli.py                   # 开发用单命令: consolidate / lint / arbitrate / serve-mcp（非交互式）
├── wiki-data/                   # 运行时生成的个人 wiki（gitignore）
├── web/                         # Next.js 前端（fork vercel/ai-chatbot 改造）
├── evals/                       # 题集 jsonl + 结果
├── docs/                        # 设计笔记、博文草稿
└── tests/
```

---

## 10. 项目成果一览（评测完成后填入真实数字）

- 长程稳定性：子代理上下文隔离 + 滚动压缩，连续运行 2h+，KV-cache 命中率 Z%（仅支持缓存的 provider）；
- 复用收益：重复问题 token 成本降低 X%、准确率提升 Y%（对比向量 RAG 基线，100 题主题 QA 集）；
- 可复用交付：wiki 以 MCP server 形式开源，可一键装入 Claude Code。

---

## 附：Day 1 清单（阶段 2 的开工闸门）

1. [x] `uv init` + 目录骨架（§9）+ ruff/pytest 全绿（CI 待建仓后补）。
2. [x] `llm/`：Provider 接口 + **MockProvider** + token 记账 JSONL + 单测——**不接真模型，全部测试也能绿**。
3. [x] `web_search` / `fetch_url` 以协议事件形式 stub；真实实现与搜索 key 接入在阶段 2。
4. 骨架跑通后再做两个各半小时级的验证（这是接入真模型/真检索前必须过的闸）：
   - 20 行脚本验证"便宜模型 + OpenAI 兼容端点"的工具调用稳定性——这是阶段 2 最大的技术不确定性；
   - 20 条中文网页验证 FTS5（wangfenjin/simple 或 trigram）+ sqlite-vec + 嵌入 API 的召回效果——中文检索栈是第二大不确定性；不行就换 tokenizer 或加大向量权重，numpy 暴力余弦兜底。
