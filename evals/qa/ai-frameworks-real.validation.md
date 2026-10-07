# ai-frameworks-real 题集验证留痕（评测支线第二层 task-1）

生成时间：2026-10-07。本文件是 `evals/qa/ai-frameworks-real.jsonl`（30 题）的
机器可复现验证记录：出题方法、冻结语料来源、过滤漏斗、逐题盲答判定、
无答案缺席验证与跨文档一致性抽查。出题与盲答全程由 cheap 档模型
（`glm-4.5-air`，step=`evals:generate`/`evals:blind`/`evals:repair`）完成，
人工零出题（gold 全部来自构造物：版本 diff / 语料原文 / 缺席验证）。

## 1. 语料来源与冻结口径

- 仓库：`langchain-ai/langchain`（MIT License），英文原文保留、问题为中文生成；
- SHA-A：`4cf62a51a8849d4baea15071c5b0e10bf7ea31c8`（langchain==0.3.30）；SHA-B：`4a65e827f7d7fd8139a4232f408a005f704dc71b`（langchain==1.0.0）；
- 时效语料 = 同一文件在两个 SHA 的快照对（3 对，见下表 v1/v2 行）；
- 单跳/多跳语料取自任一 SHA（表内 version 列标注）；
- 抓取方式：GitHub raw 直连（不经代理）；溯源验证：每篇文档正文逐行
  ⊆ scratch 原始抓取文件（15/15 全匹配），并对 5 个冻结永久链接做 live
  抽检（HTTP 200 且正文探针全命中，含 `libs/langchain_v1/README.md@4a65e827`）。

### 文档清单（15 篇，冻结永久链接）

| doc_id | 标题 | 冻结链接 | 版本 |
|---|---|---|---|
| root-readme | LangChain 仓库根 README（langchain==0.3.30 冻结快照） | https://github.com/langchain-ai/langchain/blob/4cf62a51a8849d4baea15071c5b0e10bf7ea31c8/README.md | v1 |
| root-readme@v2 | LangChain 仓库根 README（langchain==1.0.0 冻结快照） | https://github.com/langchain-ai/langchain/blob/4a65e827f7d7fd8139a4232f408a005f704dc71b/README.md | v2 |
| libs-langchain-readme | LangChain 主包 README（0.3.30：composability 定位） | https://github.com/langchain-ai/langchain/blob/4cf62a51a8849d4baea15071c5b0e10bf7ea31c8/libs/langchain/README.md | v1 |
| libs-langchain-readme@v2 | LangChain Classic README（1.0.0：旧版链与重导出） | https://github.com/langchain-ai/langchain/blob/4a65e827f7d7fd8139a4232f408a005f704dc71b/libs/langchain/README.md | v2 |
| libs-core-readme | LangChain Core README（0.3.30：基抽象与版本策略） | https://github.com/langchain-ai/langchain/blob/4cf62a51a8849d4baea15071c5b0e10bf7ea31c8/libs/core/README.md | v1 |
| libs-core-readme@v2 | LangChain Core README（1.0.0：文档与版本策略改版） | https://github.com/langchain-ai/langchain/blob/4a65e827f7d7fd8139a4232f408a005f704dc71b/libs/core/README.md | v2 |
| concepts-architecture | LangChain 架构与包层次（0.3.30 概念文档） | https://github.com/langchain-ai/langchain/blob/4cf62a51a8849d4baea15071c5b0e10bf7ea31c8/docs/docs/concepts/architecture.mdx | v1 |
| concepts-chat-history | Chat history：对话历史管理（0.3.30 概念文档） | https://github.com/langchain-ai/langchain/blob/4cf62a51a8849d4baea15071c5b0e10bf7ea31c8/docs/docs/concepts/chat_history.mdx | v1 |
| concepts-text-splitters | Text splitters：文档切分策略（0.3.30 概念文档） | https://github.com/langchain-ai/langchain/blob/4cf62a51a8849d4baea15071c5b0e10bf7ea31c8/docs/docs/concepts/text_splitters.mdx | v1 |
| concepts-embedding-models | Embedding models：嵌入模型概念（0.3.30 概念文档） | https://github.com/langchain-ai/langchain/blob/4cf62a51a8849d4baea15071c5b0e10bf7ea31c8/docs/docs/concepts/embedding_models.mdx | v1 |
| concepts-vectorstores | Vector stores：向量存储接口（0.3.30 概念文档） | https://github.com/langchain-ai/langchain/blob/4cf62a51a8849d4baea15071c5b0e10bf7ea31c8/docs/docs/concepts/vectorstores.mdx | v1 |
| concepts-tool-calling | Tool calling：工具调用概念（0.3.30 概念文档） | https://github.com/langchain-ai/langchain/blob/4cf62a51a8849d4baea15071c5b0e10bf7ea31c8/docs/docs/concepts/tool_calling.mdx | v1 |
| concepts-structured-outputs | Structured outputs：结构化输出（0.3.30 概念文档） | https://github.com/langchain-ai/langchain/blob/4cf62a51a8849d4baea15071c5b0e10bf7ea31c8/docs/docs/concepts/structured_outputs.mdx | v1 |
| concepts-rag | RAG：检索增强生成（0.3.30 概念文档） | https://github.com/langchain-ai/langchain/blob/4cf62a51a8849d4baea15071c5b0e10bf7ea31c8/docs/docs/concepts/rag.mdx | v1 |
| langchain-v1-readme | LangChain v1 主包 README（1.0.0：新 agent 架构） | https://github.com/langchain-ai/langchain/blob/4a65e827f7d7fd8139a4232f408a005f704dc71b/libs/langchain_v1/README.md | v2 |

## 2. 出题方法与 prompt 要点

1. **单跳/多跳（LLM 生成）**：system 提示给出硬规则——只准用语料明确事实、
   禁止编造与外部知识；多跳必须跨恰好 2 篇文档串联实体（entities ≥2）；
   gold_points 第 1 条必须是文档逐字短证据（≤8 词，命令/包名/API/数字），
   其余为中文概括；问题用中文换问法，禁止泛泛定义题与开放式议论题。
   生成时已出题目回填提示词防重复。
2. **时效题（gold=diff，零人工核心）**：对 3 对 v1/v2 文档做事实对比，
   只允许问 diff 里的事实（旧值/新值成对记录在下表），gold 按 v2 口径；
   notes 沿用受控层机器可解析格式 `时效题：依赖 <base> 的 v1/v2 对，gold 按 v2 口径。`
3. **无答案题**：人工只出「候选事实方向」，词面缺席由检索验证兜底（§5）。
4. **盲答过滤（去主观）**：每题把**语料全文 + 问题**（不给 gold、不给 qtype）
   交 cheap 档作答（要求英文术语逐字引用、无信息时回答「语料不含此信息」、
   新旧版本以 SHA-B 口径为准——与 judge 的 temporal 扣分口径一致，非 gold 泄漏），
   用 `metrics.point_hit` 判 gold 命中；未命中修复一次（改写 gold 第 1 条为
   逐字短证据），再不过丢弃。单跳/多跳 47 条候选与时效 5 题、无答案 4 题
   均走同一盲答协议（时效题分片执行，逐题断点落盘）。

## 3. 过滤漏斗（生成 N → 盲答通过 M → 修复 R → 终 30）

| 题型 | 生成 | sanity 即弃 | 盲答一次过 | 盲答通过小计 | 进入修复 | 选用 |
|---|---|---|---|---|---|---|
| 单跳 | 24 | 1 | 22 | 22 | 1 | 12（选用） |
| 多跳 | 23 | 0 | 12 | 13 | 8 | 9（选用） |
| 时效 | 5（diff 固定） | 0 | — | 5（diff 构造） | 0 | 5 |
| 无答案 | 4（缺席固定） | 0 | — | 4（缺席验证） | 0 | 4 |

- 合计：生成 47（单跳/多跳）+ 时效 5 / 无答案 4（构造） →
  sanity 弃 1 → 盲答通过 35
  （修复进入 9、其中 6 修复后通过、
  3 修复后仍不命中即弃）→ 近重复与 gold 重叠剔除
  （同答案/同文档对 4 条、长要点重叠 4 条、
  放宽回填 1 条）后选用 21
  （单跳 12、多跳 9）+ 5 + 4 = **30 题**；
- 时效 5 题全部通过盲答验证（语料全文 + 问题，无 gold 无 qtype，
  point_hit 判 gold，hits ≥1；结果见 §4 表）；
- 修复轮后仍不命中者按简报口径丢弃，不补占坑
  （剩余合格池 ≥ 选题配额，无需再补生成）。

## 4. 逐题盲答判定表（30 题）

| qid | 题型 | 问题（截断） | 来源/cid | point_hit 证据 | 判定 |
|---|---|---|---|---|---|
| RQ001 | single_hop | LangChain 1.0.0版本的安装命令是什么？ | C003 | hits=1/EM=0.333 | pass |
| RQ002 | single_hop | LangChain Classic包的安装命令是什么？ | C004 | hits=1/EM=0.333 | pass |
| RQ003 | single_hop | LangChain Core包的安装命令是什么？ | C006 | hits=1/EM=0.333 | pass |
| RQ004 | single_hop | LangChain Core中哪些接口的破坏性变更不需要提前通知？ | C007 | hits=1/EM=0.333 | pass |
| RQ005 | single_hop | LangChain架构中哪个包包含基础抽象和组件组合方式？ | C009 | hits=1/EM=0.333 | pass |
| RQ006 | single_hop | LangChain架构中哪个包包含构成应用程序认知架构的链和检索策略？ | C010 | hits=1/EM=0.333 | pass |
| RQ007 | single_hop | 聊天历史记录中的消息通常与什么相关联？ | C011 | hits=1/EM=0.333 | pass |
| RQ008 | single_hop | 文档切分中最直观的策略是什么？ | C012 | hits=1/EM=0.333 | pass |
| RQ009 | single_hop | LangChain为嵌入模型提供了哪两种中心方法？ | C013 | hits=1/EM=0.333 | pass |
| RQ010 | single_hop | 向量存储的三个关键方法是什么？ | C014 | hits=1/EM=0.333 | pass |
| RQ011 | single_hop | LangChain中连接工具到模型的标准接口方法是什么？ | C015 | hits=1/EM=0.333 | pass |
| RQ012 | single_hop | LangChain中处理结构化输出的推荐方法是什么？ | C016 | hits=1/EM=0.333 | pass |
| RQ013 | multi_hop | tool calling和structured outputs在AI应用中有… | C019 | hits=2/EM=0.667 | repair→pass |
| RQ014 | multi_hop | 从0.3.30到1.0.0版本，LangChain的安装命令有什么变化？ | C038 | hits=2/EM=0.667 | pass |
| RQ015 | multi_hop | 在RAG系统中，embedding models的`embed_docume… | C040 | hits=2/EM=0.667 | pass |
| RQ016 | multi_hop | LangChain Core包在整体架构中扮演什么角色，它如何支持整个生态系… | C021 | hits=1/EM=0.333 | pass |
| RQ017 | multi_hop | 从0.3.30到1.0.0版本，LangChain的agents架构发生了什… | C023 | hits=1/EM=0.333 | pass |
| RQ018 | multi_hop | LangChain Classic与integration packages… | C025 | hits=1/EM=0.333 | pass |
| RQ019 | multi_hop | LangChain Core中的beta模块有什么特殊作用，为什么需要它？ | C028 | hits=1/EM=0.333 | pass |
| RQ020 | multi_hop | 在1.0.0版本中，LangChain的agents架构如何基于LangGr… | C044 | hits=1/EM=0.333 | pass |
| RQ021 | multi_hop | vector stores在RAG系统中如何实现检索功能？ | C024 | hits=1/EM=0.333 | pass |
| RQ022 | temporal | langchain 1.0 发布后，旧版的 chains、agents 等遗… | libs-langchain-readme | diff 事实对：旧「libs/langchain/README.md @ 0.3…」→ 新「libs/langchain/README.md @ 1.0…」；盲答 hits=2/3 | pass（gold=diff 构造 + 盲答验证） |
| RQ023 | temporal | 在 langchain 1.0 的仓库里，libs/langchain 目录… | libs-langchain-readme | diff 事实对：旧「同路径旧版：定位为 composability 应用构建主包…」→ 新「同路径新版："Legacy chains, langchai…」；盲答 hits=2/3 | pass（gold=diff 构造 + 盲答验证） |
| RQ024 | temporal | LangChain 仓库根 README 现在把官方文档入口指向哪个域名？ | root-readme | diff 事实对：旧「README.md @ 0.3.30：文档入口 https:…」→ 新「README.md @ 1.0.0：文档入口 https:/…」；盲答 hits=2/2 | pass（gold=diff 构造 + 盲答验证） |
| RQ025 | temporal | langchain-core 的 API 参考文档现在的访问地址是什么？ | libs-core-readme | diff 事实对：旧「libs/core/README.md @ 0.3.30：A…」→ 新「libs/core/README.md @ 1.0.0：AP…」；盲答 hits=2/2 | pass（gold=diff 构造 + 盲答验证） |
| RQ026 | temporal | 关于 langchain-core 的版本策略，1.0 的 README 现… | libs-core-readme | diff 事实对：旧「libs/core/README.md @ 0.3.30：内…」→ 新「libs/core/README.md @ 1.0.0：内联…」；盲答 hits=1/2 | pass（gold=diff 构造 + 盲答验证） |
| RQ027 | unanswerable | LangChain 1.0 正式版（langchain==1.0.0）是什么… | — | 缺席词面 ['2025', '2024']（语料只含两份 README 快照正文，无任何日期信息）；盲答按口径拒答 | pass（缺席验证 + 盲拒答） |
| RQ028 | unanswerable | LangChain 项目最初是由谁创立的？ | — | 缺席词面 ['Harrison', 'founder']（语料无人物/创始人信息）；盲答按口径拒答 | pass（缺席验证 + 盲拒答） |
| RQ029 | unanswerable | similarity_search 默认返回多少条结果？ | — | 缺席词面 ['top_k', 'default']（语料只列方法名，无默认参数值）；盲答按口径拒答 | pass（缺席验证 + 盲拒答） |
| RQ030 | unanswerable | langchain-classic 在 PyPI 上的周下载量是多少？ | — | 缺席词面 ['download', 'PyPI downloads']（语料无任何下载量/使用量统计）；盲答按口径拒答 | pass（缺席验证 + 盲拒答） |

## 5. 无答案题缺席验证（grep + 归一化 bigram 检索，零命中证据；附盲答拒答记录）

| qid | 问题 | 缺席词面检索证据 | 盲答记录 |
|---|---|---|---|
| RQ027 | LangChain 1.0 正式版（langchain==1.0.0）是什么时候发布的？ | `2025`（子串命中 0、归一化检索 零命中）；`2024`（子串命中 0、归一化检索 零命中） | 语料不含此信息 |
| RQ028 | LangChain 项目最初是由谁创立的？ | `Harrison`（子串命中 0、归一化检索 零命中）；`founder`（子串命中 0、归一化检索 零命中） | 语料不含此信息 |
| RQ029 | similarity_search 默认返回多少条结果？ | `top_k`（子串命中 0、归一化检索 零命中）；`default`（子串命中 0、归一化检索 零命中） | 语料不含此信息 |
| RQ030 | langchain-classic 在 PyPI 上的周下载量是多少？ | `download`（子串命中 0、归一化检索 零命中）；`PyPI downloads`（子串命中 0、归一化检索 零命中） | 语料不含此信息 |

## 6. 跨文档一致性扫描（脚本 + 抽查）

- 版本号：`0.3.30` 仅出现在 SHA-A 快照文档、`1.0.0` 仅出现在 SHA-B 快照文档，
  无跨快照混写；
- 包名/实体：对 `langchain`、`langchain-core`、`langchain-classic`、
  `langchain-community`、`langchain-openai`、`langchain-anthropic` 等实体做
  全语料扫描，各实体拼写全库一致（`langchain-core` 与 `langchain_core` 为
  包名/模块名两种既有写法，与上游原文一致）：
- `langchain`：51 处，分布于 concepts-architecture, concepts-embedding-models, concepts-structured-outputs, concepts-tool-calling, concepts-vectorstores, langchain-v1-readme, libs-core-readme, libs-core-readme@v2, libs-langchain-readme, libs-langchain-readme@v2, root-readme, root-readme@v2
- `langchain-ai`：1 处，分布于 root-readme
- `langchain-classic`：1 处，分布于 libs-langchain-readme@v2
- `langchain-community`：1 处，分布于 libs-langchain-readme@v2
- `langchain_classic`：1 处，分布于 libs-langchain-readme@v2
- `langchain-core`：4 处，分布于 concepts-architecture, libs-core-readme, libs-core-readme@v2
- `langchain_core`：4 处，分布于 libs-core-readme, libs-core-readme@v2
- `langchain-openai`：1 处，分布于 concepts-architecture
- 日期：语料无日期字段（这也支撑 RQ027 的无答案判定）；
- 抽查：RQ022/RQ023（langchain-classic 迁移）、RQ024（docs.langchain.com）、
  RQ025/RQ026（reference.langchain.com 与 release-policy）逐条对照 v2 原文核验，
  gold 与原文逐字一致。

## 7. 许可声明

语料取自 `langchain-ai/langchain`（MIT License）在两个 pinned commit 的公开
快照，仅作评测用途的章节截取（每篇 300–900 字，逐行可溯源到原始抓取文件）；
冻结永久链接见 §1 文档清单。问题与 gold 为本项目生成的中文评测构造物。
scratch 出题脚本与中间产物（candidates.json 等）不随仓库提交。
