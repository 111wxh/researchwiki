# 复用序列实验设计（阶段 A+B 合并）

> 决策来源：用户 2026-10-10 对 real run 负结果基线的判定——不开 P4，先做复用序列实验（阶段 A）+ 时间演化与无答案测试（阶段 B），生命周期同轨计价、信息权限对等、预注册证伪条件。用户指令：监察者全自主执行，事后审计替代事前审批。

## 1. 核心抽象：事件时间线

四条件重放**同一条事件流**，逐事件计量。事件两类：

- **ingest**：文档在时刻 t 可用。记忆侧走 formation study run（LLM 提取入库=架构真实构建成本，如实记账）；RAG 侧索引构建（零 LLM、embedding 计量入账）。信息权限对等：同批文档、同一时刻。
- **query**：条件在时刻 t 用其当时状态作答。

**update 事件**=特殊 ingest：3 对双 SHA 文档的 v2 版（langchain 0.3.30→1.0.0）到达。记忆侧走 P2 memory-update 机制（supersede/merge/conflict）；RAG 侧 v2 chunk 替换 v1（取对 RAG 最有利解释，避免沙包化）。

## 2. 条件语义（复用 run_eval 四条件，两处关键差异）

- **c1 无 Memory**：每问一次性 root，从零研究。
- **c2 Vector RAG / c3 Hybrid RAG**：查当前索引；update 事件换 chunk。
- **c4 External Memory**：**一个持久 root 贯穿全时间线**（记忆累积=被测对象）；查询用 **P3 自适应策略 auto**（retrieval_config 接 [retrieval]，forced_mode 不设）——简单复用路由 simple，过期/冲突信号路由 update；memory_update 开（update 事件触发 supersede）。
- judge glm-4.7 不变；生成模型 glm-4.5-air 不变。

与 real run 的口径差异（报告头部声明）：①c4 由 forced deep 改为 auto（修正"重复查询承担完整构建成本"）；②同一信息流分事件进入（修正信息权限不对称）。

## 3. 时间线与题集（≈22 问）

```text
t0  ingest 全部研究文档（4 簇 × 2–3 篇，真实冻结语料）
t1–t4  每簇 4 问（1 道重复问=簇内先问后重问测摊销；2 相邻问；1–2 综合问）
t5  update：3 对 v2 到达
t6  时效问×3（gold=v2 口径）+ 无答案问×2 + 更新后重复问×1–2（测演化）
```

- 簇=语料的主题分组；簇内问尽量复用既有 30 题 gold（已盲答验证）；新综合问走同一盲答验证协议（不过修一次，再不过丢弃——不迭代）。
- 无答案问不加任何反幻觉功能，纯基线测量。

## 4. 计量与预注册判定

每行：条件、事件序号、事件类型、qid、in/out tok、latency、fresh_search、c4 policy 模式（policy.json）、EM、judge 三维。累计成本曲线 + 摊销（build_cost/reuse_count + query_cost，PLAN §7.3）。

**预注册判定（报告头部先写死，跑完对表）**：

| 维度 | 成立需要 | 证伪条件 |
|---|---|---|
| 经济性 | c4-auto 累计成本在 ≤22 问内 ≤ c2 同期，或 judge cov 优势 ≥+0.5 | 全程 c4 累计 > c2 且无任何质量维度优势 |
| 复用质量 | 重复问第二次 judge cov ≥ 第一次且 simple 路由发生（率 >0） | simple 路由率为 0 或复用后质量下降 |
| 演化价值 | 更新后时效题 c4 优于 c2/c3 | c4 时效题无优势（supersede 未转化为答案正确性） |
| 反幻觉 | （测量，无成立条件） | c4 重演 RQ028 编造 |

经济性与演化价值**双双证伪** → 记忆架构在当前形态下未证明存在 RAG 之外的价值（写进结论，作为 P4 决策依据）。

## 5. 交付物

- `evals/qa/ai-frameworks-seq.jsonl`（序列题集）+ `evals/sequence/scenario.jsonl`（事件流：每问带 gold/题型/事件序号/文档集，机器可校验）
- `scripts/run_sequence.py`（重放 harness，importlib 复用 run_eval 条件实现；mock 全链路可跑）
- `scripts/report_sequence.py`（累计曲线表 + 摊销拐点 + 预注册逐条裁决）
- 产物：`evals/results/sequence_<provider>_<ts>/events.jsonl` + manifest；real 凭证入库

## 6. 流程

工程（TDD，SDD）→ mock 全链路 → 场景自审（对照预注册条件与盲答协议，问题清单入报告）→ real run（≈1.7M in-tok @air + ≈90 judge）→ 判定报告 + worklog。无用户中途关卡；全部产物事后可审。
