# 多智能体协作日志（2026-09-22 会话，2026-09-30 编译）

> 本文件记录 ResearchWiki 项目一次完整会话内的**全部操作**：监察者（主会话）的规划、派工、验收、裁定与推送，以及所有子智能体的分工、成果与产物。
>
> **协作模式**：用户要求监察者只负责任务验收与规划派分，按阶段交付。每个工程任务 = 1 个新派实现者 + 1 个评审者（规格+质量双审）；修复轮后 1 个范围化复审者；每个阶段收尾有 1 个终审者 + 一次性修复波 + 1 个复审者。所有裁定与任务状态记录在 `.superpowers/sdd/PLAN/progress.md`（SDD 台账）。
>
> **时间线**：阶段 1、方向收敛（PLAN v2）、P1 收尾、P2 主体于 2026-09-22 一天内完成；用户中途要求暂停，恢复后完成最后复审、文档订正与推送。测试从 **251 → 703** 全绿；产生提交 **41 个**（`a7fa903..5f30eee`），全部已推送至 `origin/main`。
>
> **分支线（与主线隔离）**：2026-09-30 起存在一条 mcp-memory 分支线（ResearchWiki 记忆思想 → DSH 插件小版）。它**不属于本主线**——不走 PLAN 阶段路线与 SDD 台账、主仓库零提交、不计入下文任何统计与测试口径；登记见文末第 8 节，详细记录与接手指引见 [agent-worklog-mcp-memory.md](./agent-worklog-mcp-memory.md)。

---

## 0. 会话准备（监察者）

- 读取用户更新后的 PLAN.md（v1.1，790 行改写）与仓库现状（git log、模块清单），确认"阶段 1 Prior MVP"尚未开工、仓库无 prior/metrics/cold_warm 相关文件。
- 加载 `superpowers:using-superpowers` 与 `superpowers:subagent-driven-development` 技能，建立 SDD 工作区 `.superpowers/sdd/PLAN/`，创建进度台账。
- 跑基线全量测试：**251 passed**；记录裁定：直接在 main 分支推进（沿用用户既有逐交付 push 工作流）；写冲突扫描表（任务对产出→消费关系）入台账。
- 撰写阶段 1 四个任务的简报（task-1..4-brief.md），从 PLAN §4 提取验收口径与已核实的现有接口签名。

---

## 1. 阶段 1 · Wiki Prior MVP（10 commits，测试 251→302）

### Task 1 — Prior Reader（`wiki/prior.py`）
- **实现者** `a6f41b98`：新建 Prior 检索模块——检索 active notes、merged/superseded 沿重定向取最终 active、"历史 Prior，仅供核验"标签、`ensure_index_fresh` 索引落后检测，配 16 个测试。成果：全量 251→267 全绿，ruff 通过。产物：`a7fa903`。
- **评审者** `ae44a5e7`：9 条规格逐条核对 + 实读仓库核实接口匹配，判 Approved；挑出 1 条 Important（prior.py 访问 `SearchIndex._conn` 私有连接）与 4 条 Minor。产物：评审报告（台账 Task 1 节）。
- **监察者裁定**：Important 项不当场修（简报禁改 index.py），作为偿还项带入 Task 3。

### Task 2 — run-metrics 指标模块（`loop/metrics.py`）
- **实现者** `ad622015`：新建 `RunMetrics`（与 PLAN §4.4 逐字一致的 14 字段契约，集合相等测试防漂移）、`sum_tokens_from_jsonl` 对账、`write_run_metrics` 落盘、`compute_citation_coverage`，配 18 个测试。成果：267→285 全绿。产物：`db175d3`。
- **评审者** `3d5155b1`：实读 provider.py/accounting.py 核对对账字段为真实契约，判 Approved；4 条 Minor（bool 未排、引用标记正则等）。

### Task 3 — AgentLoop 接入 Prior（核心集成）
- **实现者** `d0c2cfd1`（含修复轮 1）：Prior 注入 plan 步骤 user 消息（系统提示词一字不动）、Prior URL 不进 SourcePool、search/fetch 计数、metrics 落盘、`[prior]` 配置贯通 server；偿还 Task 1 遗留（`indexed_status()` 只读接口替代私有访问）。修复轮：子 agent token 回流主计数（修 dispatch 路径对账断裂）+ metrics try/finally 恰好写一次。成果：285→291→294。产物：`f3cefa5`、`4a9d574`、`c0989f9`。
- **评审者** `9481b32f`：12 条硬口径 11 满足，判 Needs fixes——Critical：`dispatch_research` 路径下 run-metrics 与 tokens.jsonl 永久对不上账。
- **复审者** `52ce5feb`：两条发现均 ADDRESSED；判定 test_loop.py 契约变更是加强非削弱；清白。

### Task 4 — cold/warm 冒烟脚本（`scripts/cold_warm_smoke.py`）
- **实现者** `0d6fcd80`（含修复轮 1）：mock/config 双模式冒烟脚本 + 四道硬断言闸（warm prior_hit>0、token 双向对账）+ 人类可读对照表；修复轮显式锁死 MockSearch（防环境变量泄漏发起真实网络）+ 补 cold 两道闸（prior_hit==0、notes_created>=1）防 `--wiki-root` 预填充假绿。成果：294→302，mock 自跑 EXIT=0。产物：`8e1e186`、`63e69a1`。
- **评审者** `37458634`：Approved 附 2 条 Important（零网络声明的环境条件性、验收闸假绿路径）。
- **复审者** `6b3f13a3`：两条 ADDRESSED，新测试可证伪，清白。

### 阶段终审 + 真模型证据（本阶段最有价值的发现）
- **终审者** `8122d33c`：全分支（7 commits，+51 测试）跨模块一致性核实，§4.5 五条 + §10 四类证据逐条判定，判"可合入"，附条件：缺真实模型小样本证据（§10 类型 3）。
- **真模型运行员** `62911f26`：跑 `--provider config` 两次均 FAIL——warm run 抛 `sqlite3.OperationalError: database is locked`；按指令诚实留档失败证据并定位根因（嵌入缓存 `_store()` 从不 commit + 缓存与检索索引共库形成锁环；mock 模式完全测不出）。产物：失败证据 commit `8dd20e3`。
- **缺陷修复+重跑员** `ccfde839`：修 embeddings.py 写入即时提交 + 双连接 busy_timeout；并挖出第二处根因——`index_note` 在写事务中途调 embed 成锁环，把 embed 移到事务前；4 条可证伪回归测试（200ms 短忙等钉住修复前失败模式）；重跑真模型冒烟 EXIT=0。成果：**warm prior_hit=5，input tokens 71831→53719（−25%），时延 107.7s→74.5s，token 双向对账一致**。产物：`464138a`、`9ee211e`（smoke_out/ FAIL→FIX→PASS 证据链）。
- **修复波复审者** `d54e7827`：锁环根因、回归可证伪性、全部构造点覆盖逐条核实；诚实性链完整（FAIL 留档在工作树被覆盖记 Minor）；清白。
- **监察者**：推送阶段 1 全部 11 commits；交付阶段验收报告给用户。

---

## 2. 方向收敛 → PLAN v2（2026-09-22 用户决策）

- **用户决策**：项目从"Research Knowledge System"收敛为"面向 Agent 的外置、时间感知、可演化长期记忆系统"，ResearchWiki 降为首个实验场景；工程层渐进迁移不推倒重写。
- **监察者**：更新项目持久记忆（方向决策 + 阶段 1 证据 + 三研究问题 RQ1-3 + 四 Phase 路线）；写 PLAN v2 改写简报（章节骨架、必须逐字采用的用户原话、必须保留的基线证据）。
- **重写者** `8b15394b`：整体重写 PLAN.md 为 v2（620 行）——RQ1-3、四 Phase 表、旧→新 11 行映射表、Storage/Retrieval 分离与"不做固定分层桶"立场、阶段 1 实测证据全保留；代码路径零改名。产物：`c6221b1`。
- **监察者**：亲自抽查关键章节验收（RQ/四 Phase/映射表/证据均落位），推送。

---

## 3. P1 收尾 · External Memory（5 commits，测试 306→366）

### P1-A — NoteMeta 扩展 + memory_* MCP 九工具
- **实现者** `54b2cede`（含修复轮 1）：NoteMeta 增加 `kind`（user/knowledge/experience）与 `importance`（0–1）、旧 index.db 在线迁移（ALTER+回填）、SearchIndex kind 过滤；MCP 新增 memory.store/search/recall/update/supersede/invalidate/timeline/conflicts/profile 九工具（wiki_* 五工具冻结为兼容层）。修复轮：update_reason 改追加语义（旧单键折叠迁移）+ timeline 合并 list_changes 变更记录。成果：306→339→342。产物：`ec0079f`、`0f9d45e`、`2277775`。
- **评审者** `007015ea`：2358 行 diff 全读，三条关键裁定语义（墓碑可达、reason+备份、沿链不变量）全部落实，判 Approved 附 2 条 Important（审计链完备性）。
- **复审者** `f506a953`：两条 ADDRESSED，迁移折叠与同刻排序稳定，既有用例改显式时间戳未削弱断言；清白。
- **监察者裁定**：memory.invalidate = supersede 到自动墓碑笔记（不新增状态机）。

### P1-B — Memory Formation MVP（RQ1）
- **实现者** `49fe815b`：新建 `wiki/formation.py` 确定性策略（四类拒绝规则 + importance 权重表三方一致：docstring/实现/测试）接入 run 自动入库路径，`formation_stats` 计数 + state.md 输出，`[formation]` 配置 + enabled 逃生阀。成果：342→361。产物：`8532c9b`。
- **监察者裁定**：接受实现者偏离简报字面——AgentLoop 层配置 None 即关闭（保冒烟脚本兼容），server 生产路径恒传配置（应用级默认开启）。
- **评审者** `228923ae`：首轮 Approved（无 Critical/Important）。

### P1 终审 + 修复波
- **终审者** `72495ab9`：P1 退出条件三条全达成，判可合入；附 1 项合入前小修——MCP 写路径未复核 note.id 形态，库内污染 id 可逃出路径沙箱；并更正台账一处误诊（"死代码"实为活代码）。
- **修复波实现者** `04165a2a`：`_assert_note_id_writable` 断言（放进 `_require_active` 先拒后写防半成品）+ merged+formation 端到端回归网 + formation 布尔宽容解析加固 + 索引同步注释。成果：361→366。产物：`ca88223`。
- **修复波复审者** `386cb31d`：4/4 ADDRESSED，正则一致性/只读路径/布尔兼容逐条核实；清白。
- **监察者**：推送 P1 全部 6 commits；交付 P1 验收报告。

---

## 4. 方向补充：两类时效性问题入 PLAN

- **补写者** `088f142a`：把用户原话的"两类时效性问题"（外部信息 / 用户记忆 + 总述一句）逐字写进 PLAN §0，并在 §3.3 P2 章节加引用锚点与 kind 区分待定项。产物：`524825a`。
- **监察者裁定**：freshness 的 user/knowledge 参数区分以 per_kind 覆盖接口预留、默认等价于不区分（数据之前不定死参数）。

---

## 5. P2 · Temporal Memory（25 commits，测试 366→703）

### P2-A — freshness 计算
- **实现者** `acc8ae8a`（含修复轮 1）：`wiki/freshness.py` 三态判定（基准时间顺序、显式有效期优先、半衰期衰减）、`valid_from`/`valid_until` 字段、per_kind 覆盖接口、decay 与 `freshness_factor` 防漂移测试、lint freshness 统计，68 个测试。修复轮：半衰期三级取值（DEFAULT→[wiki]→[freshness]，修配置态口径分叉）+ R1/R2 理由文案解耦。成果：366→434→441。产物：`cdb4e1b`、`bb1187b`。
- **评审者** `27635857`：15 条硬口径逐条核实（含防漂移测试确实跨实现比对、时区约定三处一致），Approved 附 2 条 Important。
- **复审者** `9268d9c3`：两条 ADDRESSED，配置态一致性测试真跨实现（12 点逐点比对）；清白。

### P2-B — 索引指纹 + 来源溯源
- **实现者** `3a1ba24f`（含修复轮 1、2）：`note_index_hash` 正文指纹 + `ensure_index_fresh` 三维判定（修掉 carried 缺陷：原地改正文此前对索引不可见）+ 溯源三查询（notes_by_source / notes_depending_on / missing_snapshots）+ `mark_source_changed`（来源变化标记）+ freshness 来源变化降级规则 + `[freshness]` 配置接线。修复轮 1：指纹扩至全部 9 个参与检索的字段（补 kind 漂移盲区）+ 去重记账；修复轮 2：去重键按 URL 分槽（复审发现多来源笔记仍会被永久钉在 review_due）。成果：441→466→474→552。产物：`d448d4a`、`76a8f60`、`ab68279`、`349ce0a`。
- **评审者** `8c6db0e2`：10 条规格全满足，2 条 Important（指纹覆盖面、幂等 docstring 失实）；独立评估 MCP 与 loop 判据分叉为"应排期的架构债"。
- **复审者** `4e832791`（轮 1）：两条 ADDRESSED，发现残留 Important（去重键缺 URL 维度）。
- **复审者** `04577e0b`（轮 2）：清白。

### P2-C — verification 判定模块（RQ2 核心）
- **实现者** `17be765f`（含修复轮 1、2、3）：`wiki/verification.py` 五类判定（consistent/newer/more_specific/conflicting/uncertain）+ 确定性冲突槽位检测 + judge 注入（judge=None 全绿），76 个测试。修复轮 1：修 **Critical 假一致**（一致判据集合级→槽位级，"取值错位互换"曾被判"仍然成立"）+ 冲突门实体护栏；修复轮 2：护栏提升为全局前置（实体不交不做任何更新判定）；修复轮 3（终审 fix wave 复审后）：门控分支不再输出自相矛盾假理由 + 门值上下界夹取 + save_meta 不补 created 的语义声明。成果：474→550→569→592→703。产物：`ee24443`、`a7d404d`、`3267d6a`、`5f30eee`。
- **评审者** `53d0667e`：实测复现 Critical（prior「上下文窗口 128k，参数量 70B」vs 证据互换取值 → 被判 consistent）与 Important（假冲突），Needs fixes。
- **复审者** `9beb65a2`（轮 1）：Critical/Important ADDRESSED，随修 a–f 全做；发现新 Important（护栏只拦冲突门，实体不交仍可被升级为 supersede）。
- **复审者** `98592fd9`（轮 2）：清白；确认 judge 可越权判 supersede 的问题移交 P2-E 结构性解决。
- **监察者裁定**：冲突门前置于时间比较（更晚但矛盾不得判 newer 掩盖冲突）；judge 可判保守动作（conflicting）但不可判覆盖动作（supersede）。

### P2-D — MCP 证据链参数
- **实现者** `a39d8413`（含修复轮 1）：supersede/update 携带 `source_urls`（防把旧证据伪装成新证据）、返回体 `evidence: provided/none` 标注、带来源时 confidence 不继承旧 high；修复轮：confidence 改可选参数（三级语义：显式值 > 带来源→medium > 都没给→继承）+ 空 URL 语义三处文档化。成果：569→583→592。产物：`bac7c79`、`2889305`。
- **评审者** `23833532`：Approved（防伪装有等值级硬断言），挑出 confidence 硬编码盲区。
- **复审者** `4e917301`：Important ADDRESSED（三级语义为结构性保证而非测试约定），清白。

### P2-E — 记忆更新闭环（RQ2 接线）
- **实现者** `4394a53d`（含修复轮 1、终审修复波、skip 留痕小修）：判官越权拦截做成**模块内结构性不变量**（`guard`/`judge_verdict` 机器可读字段 + allowlist，护栏触发时 judge 只能判保守动作）；run 收尾动作分发（refresh/supersede/merge/conflict/none）+ 幂等键双通道 + dry_run + `memory-update.json`，35 个测试。修复轮：幂等键生产不可达、永真断言假覆盖、`_apply_supersede` confidence 硬编码、同证据双写产生两条 active 记忆（采纳"复用本轮入库笔记 ID"）等 8 项。终审修复波（resume）：F1 replace 根治墓碑复活 / F2 替换门 0.6 / F3 声明如实 / F5 幂等键并集 / F6 留痕封顶。小修：skip 留痕补 similarity/reasons/skip_reason_kind（真模型 35 条比较中 31 条跳过记录从不可复算变为可复算）。成果：592→638→650→699。产物：`bf38a6a`、`3142cf7`、`ab0fc3e`、`6482c1b`、`c5d3cc6`。
- **评审者** `efb0b236`：4 Important + 4 Minor（含实测墓碑复活：tombstone True→False）。
- **复审者** `c9b9d410`：4 Important 全 ADDRESSED，6 条新 Minor 全部方向保守；清白。

### P2-F — 收尾（墓碑语义 + 判据统一 + lint 健康度）
- **实现者** `bd34b2e2`（含修复轮 1）：墓碑默认排除检索/Prior 注入（保留 read/timeline/lint 可见 + `include_tombstones` 审计出口）；索引落后判据收敛——`index_drift` 共享判据取代 loop 指纹 / MCP mtime / lint id 集合**三个答案**；lint 补冲突/墓碑/悬空证据/来源变化待复核四项健康度。修复轮：采纳根治方案——`NoteMeta.replace()` + `store.save_meta()` 统一全部 meta 重建点（从根上消灭"重建漏传字段"类缺陷，此前墓碑复活缺陷在两条路径各出现一次）+ 去重全集排除墓碑 + lint 收敛共享判据。成果：650→670→681。产物：`067461f`、`03941b1`、`73da7a4`、`4be2bed`、`2472cf2`、`41094f4`。
- **评审者** `12fcad0b`：三条裁定硬口径全部落地，Approved 附 3 条 Important（`_annotate_formation` 是第三个漏传点、只修 `_merge` 不足）。
- **复审者** `a40adb39`：I-1/2/3 ADDRESSED，replace 语义（可变字段真拷贝）/save_meta 原子性/墓碑纠错通道逐条核实；清白。

### P2 终审 + fix wave
- **终审者** `498bc97f`：20 commits 全阶段终审——P2 六条退出条件全达成、索引判据收敛为"本阶段最干净的架构收敛"、测试 603→681 无凑数；判"需修复后合入"（3 必修 + F2 校准裁定交监察者）。留档 F4/F7/F8/F9/F10/F11/F12 给后续阶段。
- **fix wave**（resume `4394a53d`）：见 P2-E 条目。产物：`6482c1b`（689 passed）。
- **fix wave 复审者** `04adb49f`：F1/F3/F5/F6 清白；发现 F2 门控分支落进"未声明"兜底输出两条假理由（Important）→ 转 P2-C 实现者修复（`5f30eee`）。
- **监察者裁定（当场）**：给替换类动作加独立高门 `supersede_min_similarity=0.6`——终审实测替换门实际退化成 0.3 相似度地板且实体为空时不设防，异 facet 更晚证据（sim 0.517）可退役有效旧记忆。

### P2-G — 真模型端到端证据（F9 补证）
- **实现者** `756fdb39`：冒烟脚本接入 formation/memory_update/verification 三段配置（mock 模式四道硬闸保持全绿）+ 真模型 cold/warm 运行一次通过 + 完整证据入库（JSONL/console/state.md/memory-update.json/run-metrics 副本）。成果：689→695；真模型 warm：prior_hit 5、**35 条比较 → 复核 2 / 合并 2 / 跳过 31**，两条 merge 真改写笔记正文。产物：`495a029`、`ca77206`、`smoke_out/cold_warm_real_p2.*`。
- **诚实边界（如实归因）**：supersede 与 conflicts 恒 0——来源变化检测器不在研究 run 链路 + 替换门 0.6 挡住重述对 + 无搜索 key 回退 MockSearch。

---

## 6. 暂停、恢复与推送（监察者）

- **用户暂停**（"先暂停，明天再说"）：立即停止在途复审，固化状态——确认工作区干净、703 passed、26 提交未推送；把恢复三步（重派合并复审→推送→P3 校准裁定）与 P2 诚实边界写入台账"暂停点"节。
- **用户恢复**（"继续"）：重派最后 4 commits 的合并范围化复审。
- **合并复审者** `899c5686`：P2-G 接线正确（参数名与 AgentLoop 一致、mock 四道硬闸照旧全绿、新键 additive）；**证据真实性独立复现**——用仓库模块重建 5 prior × 7 证据，35/35 判定与 4 个相似度值（0.826/0.310/0.326/0.920）逐条对上；判清白，唯报告两处数字与证据不符（纯文档）。
- **监察者**：依复审结论订正 `task-14-report.md`（相似度分桶 24/9 → **25/8/2**，并纠正"more_specific 被幂等闸拦下"的错误归因）与 `task-12-report.md`（"10 个调用点"→12 处），均标注订正来源；不改证据文件（保持"证据是当时产物"的诚实性）。更新台账与项目持久记忆。
- **推送受阻**：`git push` 失败——本地代理（127.0.0.1:7897）未启动，直连亦被重置；判定为环境问题（该项目 push 历来需此代理），报告用户。
- **用户启动代理后**：推送成功，`origin/main` = `5f30eee`（26 提交），无剩余未推送提交。清理台账过期"暂停点"段落，交付 P2 阶段报告。

---

## 7. 会话统计

| 项 | 数值 |
|---|---|
| 子智能体人次 | **48**（实现侧 18：含 5 次修复轮 resume 与 1 次终审修复波 resume；评审/复审/终审 30） |
| 修复轮 | 实现侧修复轮 8 次 + 终审/阶段修复波 3 次，全部经范围化复审验证 |
| 提交 | 41 个（`a7fa903..5f30eee`），全部已推送 |
| 测试 | 251 → **703** 全绿（阶段 1 +55、P1 +60、P2 +100 → 部分计入各阶段口径） |
| 评审抓到的典型缺陷 | 真模型闸门抓到 SQLite 锁环（mock 完全测不出）；"取值错位互换"被判假一致（Critical，评审实测复现）；墓碑复活字段丢失（两条路径各一次）；多处"报告声称有测试但实际没有"的诚实性缺口 |
| 关键裁定 | 记录于台账，含：替换门 0.6、NoteMeta.replace 根治方案、save_meta 不补 created、invalidate=墓碑、judge 只可判保守动作、复用入库笔记 ID 等 |

**未完事项（移交后续阶段）**：P3 开工前两个校准裁定（替换门与相似度带的关系明文化；merge 独立下限——真实 run 在 sim 0.31 即并入正文，且 merge 在接线路径结构性不可达）；F7 loop 写路径 id 形态护栏（触发条件：P4 引入外部证据通道）；其余 deferred Minor 见台账。

---

## 8. 分支线登记（2026-09-30，非主线会话）

2026-09-30 出现一条**与主线隔离的分支线**：mcp-memory——把本项目的外置记忆思想蒸馏成独立小版 MCP server，接入 DSH（DeepSeek Harness）Desktop。特此登记并划清边界：

- **不属性主线**：不走 PLAN v2 阶段路线与 SDD 台账，主仓库零提交（git 仅未跟踪目录 `?? mcp-memory/`），不计入本日志任何统计、测试口径或"未完事项"（第 7 节的移交清单与它无关）。
- **零耦合**：Dockerfile 选择性 COPY 不含该目录；主项目 uv venv / uv.lock 未动；其依赖（官方 `mcp` SDK 2.x，Store Python 3.11 用户 site-packages）与主线（uv 3.12 venv，`fastmcp`）分属不同环境。
- **详细记录、状态与接手指引**：见 [agent-worklog-mcp-memory.md](./agent-worklog-mcp-memory.md)（分支线专用日志，含 DSH 环境事实、验证步骤、移除方法、并入主线的路径）。
- 接手主线的人**无需**读分支线日志；接手分支线的人**无需**读本日志第 1–7 节。

---

## 9. P3 · Dynamic Retrieval 会话（2026-10-01）

> 新会话（与第 1–7 节的 2026-09-22 会话无共享上下文）。计划：`docs/superpowers/plans/2026-10-01-p3-dynamic-retrieval.md`（实现 PLAN §3.4），SDD 工作区 `.superpowers/sdd/2026-10-01-p3-dynamic-retrieval/`（gitignored，本节为其完整存档——工作区已按惯例清除，git 历史与本节为唯一记录）。子智能体全部继承会话模型（startplan GLM-5.3-Flash，用户指示）。测试 **703 → 747** 全绿；提交 17 个（`0149446..08a7f23`），全部已推送。

### 9.1 会话准备（监察者）

- 读取 PLAN §3.4（Dynamic Retrieval 规格）与 §2.4 因子表，确认 P3 无任何既有代码；核对 worklog §7 移交三项（①替换门明文化 ②merge 下限 ③F7 护栏）——均非 P3 阻塞，登记为范围外。
- 派侦察者 `b5b67e42`（Explore，只读）：产出 10 节接口报告——store/index 检索打分链（RRF×confidence×freshness，importance 未进因子）、`memory_recall` 的 "P3 钩子位"（passthrough）、AgentLoop 五个接线点与 `*_config=None=关闭` 约定、tokens.jsonl/run-metrics 14 字段契约、config 键位、cold_warm_smoke 冒烟范式、测试惯例（ScriptedProvider/MockEmbedding/注入时钟）、703 基线。
- 撰写实施计划（9 任务、逐步 TDD、全局约束、自审记录），落 `docs/superpowers/plans/2026-10-01-p3-dynamic-retrieval.md`（后随 worklog 一并入库 `08a7f23`）。
- 建 SDD 工作区与台账；预检裁定 R0（main 直推沿用逐交付 push 协议；子智能体不传 model 覆盖）+ 任务对产出→消费冲突扫描表（9 对，全一致）。

### 9.2 逐任务操作与产物

**Task 1 — 策略配置与模式限额表（`loop/research_policy.py` 骨架）**
- **实现者** `95194f91`：按 brief 逐字转写 `ModeLimits`/`PolicySettings`/`policy_settings_from_config`（缺省=simple(3,1200,0,3,brief,0)/update(5,4000,2,6,standard,1)/deep(5,4000,4,12,full,0) 阶梯；越界夹取；非法 forced_mode 拒绝）+ 3 测试。TDD RED（ModuleNotFoundError）→ 全量 706。产物：`7a0528b`。
- **评审者** `42327671`：Spec ✅（签名/默认值逐字核对）+ Approved；Minor 4（未用 import、两阈值无夹取、override 裸 TypeError、limits 可变 dict）。
- **备注**：push 时仓库代理（127.0.0.1:7897）不可用，一次性 `git -c http.proxy= push` 成功；本次会话后续 push 时好时坏（TLS 抖动），均留待补推并最终清零。

**Task 2 — 五因子特征采集 `collect_features`**
- **实现者** `83cb2434`：`PolicyFeatures`（11 字段+to_dict）+ `collect_features`（探测检索→freshness 三态计数→volatility 排名→低置信→冲突按 `token_similarity ≥ conflict_similarity` 全库扫描）+ 5 测试（PLAN 五类场景种子的特征层）。机械适配 4 处，关键是 **UTC-aware NOW**（naive 时钟会在有 observed_at 的笔记上 aware/naive 相减 TypeError）。706→711。产物：`5ba85e5`。
- **评审者** `c43f83d2`：Spec ✅ + Approved；4 处适配逐一核实（UTC 一项独立确认必要）；对源核验 `Conflict.question`/`SearchMatch`/`evaluate_freshness` 签名全部成立；Minor 3（hit_count 含 get_note=None 匹配、_VOLATILITY_RANK 词表外 KeyError 等）。

**Task 3 — `decide_mode` 确定性判定（2 修复轮 + 跨任务饱和修复）**
- **实现者** `a0c5f918`：`PolicyDecision`+`decide_mode`（守卫红线>覆盖度>预算降级>forced，理由全带数值）+ 9 测试。711→720。产物：`1b62586`。
- **主动申报**：预算降级可穿透守卫（stale 触发的 update 被降到 simple=零 fresh 搜索）→ 监察者裁定 **R1（守卫地板）** → **修复轮 1**：`guard_floored` 钉回 update+理由留痕 + 2 测试。720→722。产物：`421e0e5`。
- **评审者** `e7d12ce3`：Approved；**穷举守卫×覆盖度×预算×时间×forced 全组合矩阵**核验红线成立（含连续降级不可能落 simple 的证明）；挑出 1 条 Important（plan-mandated）："覆盖度不足"分支理由为静态字符串、缺具体数值 → 监察者裁定 **R2** → **修复轮 2**：f-string 嵌入数值与阈值 + 1 测试。722→723。产物：`f5e884c`。
- **复审者** `f4db1062`：ADDRESSED，无新破坏（并核实新测试的分支可达性与断言必然性）。
- **会话尾段（真实冒烟闸④抓到缺口，见 Task 9）**：同一实现者执行 **R3 修复**——守卫信号改全库相关性门控扫描（`list_notes(active)` 全量、排除窗口内 note_id、stale/review_due/低置信 × `token_similarity(title+body[:400])` 门控；docstring 记录真实教训与 PLAN §3.4 依据）+ 2 测试（拥挤窗口 RED 精确复现冒烟缺口：`hit_count=8, stale_hits=0`）。743→745。产物：`ee96667`。
- **复审者** `75dfffe6`：6 项裁定逐条 ADDRESSED；确认零模型、同一性去重、400 字截断有界、`decide_mode` 未动；O(笔记文件数) 磁盘 I/O 记 deferred（10k+ 优化目标）。

**Task 4 — importance 进检索排名因子（`wiki/index.py`）**
- **实现者** `07499d81`：`note_meta` 加 `importance REAL` 列 + `_migrate_note_meta` PRAGMA 探测迁移（存量 NULL→因子 0.8 中性）+ `note_index_hash` 扩展第 11 字段（旧行哈希失配→drift→rebuild，docstring 预留约定兑现）+ 打分改四因子全乘 + 3 测试（含按真实 13 列 DDL 复刻旧库）。**自发表扬点**：发现 `0.0` 与 `None` 因子不同（0.5 vs 0.8）若同指纹会互改逃过 drift——显式判 None 而非 `or ""`。723→726。产物：`1e86b9e`。
- **评审者** `b55d2fb1`：Approved；迁移-on-open、drift→rebuild、NULL 可检索三约束均有代码+测试证据；grep 确认 index.py 是 note_meta 唯一写入方；Minor 2。

**Task 5 — AgentLoop 接线（最重集成任务）**
- **实现者** `22d80e9e`：五处接线全部落地——①判定先于 Prior 注入（复用 ensure_index_fresh 后同源 store/index，探测幂等）②`run_dir/policy.json`（atomic write，outcome 报告后回填 fresh_source_count/min_fresh_sources）+ state.md `retrieval:` 行 ③prior k/max_chars 被模式限额覆盖 ④搜索帽（超帽返回 `search_budget_exhausted`，不调 provider 不计数）、子代理门（不注册 dispatch_research）、生效步数 ⑤brief 报告后缀 + 诚实缺口块（只落盘不进事件流）；`server/main.py` 两键接线。+ 6 行为测试（legacy 逐字段等价、留痕、搜索帽、stale 不落 simple、回填）。726→732。产物：`f0e24c0`。
- **评审者** `759cc505`：Approved（50KB diff 分遍审完）；两个聚焦风险核验安全（注册表重建与原构造严格同构不丢工具、ensure_index_fresh 幂等无双重建）；披露的"搜索帽同口径扩到子注册表"核验无双计数、不封堵合法深搜。
- **终审修复波**由同一实现者执行（见本节末"终审"段）。

**Task 6 — MCP `memory_recall` 升级 budget-aware**
- **实现者** `0692256f`：`recall` 从 passthrough 升级——特征（budget_remaining_ratio=1.0，docstring 写明 MCP 无 loop 账本）→判定→结果宽度 `min(count, limits.prior_k)` 收窄（results/count/k 同步）→`payload["mode"]`+`policy` 透传；passthrough 基线（缺段/enabled=false）在策略导入**之前**返回（防漂移）；工具签名不变 + 4 测试（含进程内 MCP client 协议面测试）。732→736。产物：`977a1d5`。
- **评审者** `0faae2b4`：Approved；三个具名风险核验（索引无重复重建、基线零漂移、k<prior_k 不过度收窄）；挑出 1 条 Important（plan-mandated）：simple 收窄断言是条件式（else 空洞通过，核心交付从未被断言）→ 监察者裁定 **R5** → **修复轮**：无条件断言 + 补 enabled=false 逃生阀测试。736→737。产物：`594fa3c`。
- **复审者** `d0d3b6f7`：ADDRESSED；并复核无条件断言的确定性依据（blake2b 嵌入不按进程加盐、trigram 确定性、top_score 0.0232 ≥ 0.010）。

**Task 7 — config.toml `[retrieval]` 段**
- **实现者** `8ead50ba`：TOML 段+中文注释（模式阶梯/守卫红线/留痕口径/enabled=false 逃生阀；可选键保持注释态）+ 默认值锁定测试（解析真实 config.toml）。如实记录 TDD 反转（缺段即过——测试的真实职责是锁"注释默认值==代码默认值"）；自检 TOML 与代码默认值全等（含 limits）。737→738。产物：`6561819`。
- **评审者** `3cb16161`：Approved；逐项对照 research_policy.py 全部 26 个值（四个表含边界）确认零行为漂移；Minor 2（锁定测试仅锁 2 值等）。

**Task 8 — `scripts/adaptive_smoke.py` 同题跨模式对照（2 修复轮）**
- **实现者** `39efb374`：播种（forced deep 产记忆 + 手工 stale 笔记）→ `copytree` 每模式独立副本（排除 index.db，Windows 锁；首查 ensure_index_fresh 重建）→ simple/update/deep/auto 四行矩阵 → 五道验收闸 + tokens.jsonl 对账（不变量 ⑥）。剧本适配：update 语义化为 plan+一次搜索+收尾+report（plan 轮 tool_calls 会被 loop 静默忽略——对照源码证实）、deep 补一次 MockTransport 离线 fetch 使成本阶梯严格单调；`run_mode_once` 自有实现，**cold_warm_smoke 零改动**（P2 证据脚本冻结）。738→741；mock 自跑 EXIT=0，证据落 `smoke_out/adaptive_mock_*.jsonl`。产物：`5d1827a`。
- **评审者** `16838bca`：Approved；五闸逐字核验（读 metrics 原文非重算；stale 种子固定 observed_at 非时敏）；三个剧本适配对照 loop/policy/freshness 源码逐一验证；留控制器标记：断言③真实模型下非确定。
- 监察者裁定 **R4（修复轮 1/2）**：③ config 留痕警示（`fresh_search_ladder_ok`，None=未判定不假绿）→742 `d4c4d86`；② 拆分（0 帽半边 mock/config 同闸硬；成本半边 `cost_gap_ok` config 软）→743 `cb6c592`。判定内核唯一（盖章与闸共用）。
- **复审者** `37d43319`：两裁定 ADDRESSED；核实盖章先于写盘、子集 run 的 None 语义不假绿不假红；1 条 Minor（mock 成本硬闸内联重算未共用内核，行为无漂移可能）。

**Task 9 — 终验 + 真实冒烟 + 交付记录（监察者执行）**
- 全量 743 复跑通过；授权真实模型对照（R6，5 个真实 run）。
- **第一次真实 run：闸④失败（退出码 1，闸门正确拦下）**——auto 误判 simple。取证 `policy.json`：`hit_count=8, stale_hits=0, top_score=0.0145`——seed deep 产的 8 条同题 fresh 笔记把手工 stale 笔记（N-0009）挤出 probe_k=8 窗口，策略没看见它。失败数字留档 `smoke_out/adaptive_config_20261001T080844Z.jsonl`（simple 9360/0、update 16492/2、deep 75059/4、auto(simple) 6899/0）。
- 裁定 **R3** 并派回 `a0c5f918` 修复（见 Task 3 段）→ **复跑五闸全过**：simple 9,296/0、update 16,271/2、deep 38,068/4、**auto=update**（stale_hits=1 被全库扫描抓到）18,163/2；auto vs deep 输入 token **−52.3%**、时延 −20.2%、citation_coverage 1.0 持平。证据 `adaptive_config_20261001T083359Z.jsonl`。
- 用户 09-30 遗留的 mcp-memory 文档登记单独成 commit `fa9a07a`（代码目录按边界声明保持未跟踪）；worklog 本节 + 计划文件入库 `08a7f23`。

**终审（全分支）与修复波**
- **终审者** `7b136d0b`（`0149446..ee96667` 全 178KB diff + 账本 13 条缓还分类）：**With fixes**——代码本身可合入（跨路径一致性、迁移安全、两条红线由本阶段自己的闸抓到并修复、真实证据对账相符）；2 条 Important（worklog 交付记录缺失——即本节；接线①④无行为断言）；Minor 若干含 1 条计划层观察（预算因子结构性惰性）。
- **修复波**（`22d80e9e` resume，一次打包）：`test_mode_prior_k_reaches_retrieve_priors`（4 条同题笔记，forced simple→prior_hit_count==3，deep 对照==4）+ `test_effective_max_steps_truncates_act_loop`（max_steps=3 熔断，记账序列 plan/act:1..3/distill/report，观测口径照抄 test_loop.py 先例）+ lint 3 条清零（UP035/E501/F401，零运行时影响）。745→747。产物：`6e26195`。
- **复审者** `a168958b`：3 项全 ADDRESSED；确认两测试走真实产物（run-metrics/tokens.jsonl/state.md/事件）而非 mock 接线，且真正有判别力（无截断时记账序列断言会失败）；修复波范围约束（"除此之外一行不动"）遵守。

### 9.3 交付物（对照 PLAN §3.4 五项）

| 计划交付物 | 产物 | 任务 |
|---|---|---|
| `loop/research_policy.py` 策略输入/决策/解释 | `PolicySettings`/`ModeLimits`、`PolicyFeatures`+`collect_features`、`PolicyDecision`+`decide_mode`（守卫红线>覆盖度>预算降级>forced，理由全带数值） | T1–T3 |
| `loop/agent_loop.py` 模式化检索上限/报告/验证要求 | 判定先于 Prior 注入；`run_dir/policy.json`（outcome 报告后回填）+ state.md 行；prior k/chars 覆盖、fresh 搜索帽、子代理门、生效步数、brief 报告样式、min_fresh_sources 诚实缺口块 | T5 |
| `config.toml` 模式预算/阈值/最小 fresh source 数 | `[retrieval]` 段（值==代码默认值，零行为漂移；enabled=false=评测 baseline） | T7 |
| `tests/test_research_policy.py` 五类用例 | 26 条（stable/fresh、volatile/stale、conflict、空库、无答案=零覆盖 + 守卫地板 + 饱和窗口等） | T1–T3+ |
| `scripts/adaptive_smoke.py` 同题跨模式对照 | simple/update/deep/auto 四行矩阵 + 五道验收闸（②③按 mock 硬/config 留痕分治） | T8 |

附加交付（PLAN §2.4 因子表落地）：`wiki/index.py` importance 进排名（RRF×confidence×freshness×importance；旧库 ALTER 迁移 + `note_index_hash` 扩展→drift→rebuild；0.0/None 指纹碰撞规避）；`mcp_server/service.py` `memory_recall` 从 passthrough 升级 budget-aware（"P3 钩子位"兑现）。

### 9.4 验收（对照 §3.4 退出条件，含诚实边界）

1. **决策留痕**：每次判定 policy.json 全量 features+reasons+limits；冒烟闸①硬性核对（不允许裸模式字符串）。
2. **simple 低于 deep baseline**：mock 硬闸（4800<5700<6600 input tokens 严格单调）；真实 run simple 9,296 vs deep 38,068（−75.6%），时延 45.3s vs 73.4s。
3. **守卫红线**：decide_mode 守卫 + 预算降级守卫地板 + **全库相关性门控扫描**（见 9.5 裁定 R3）；冒烟闸④ mock 硬闸，真实 run 曾失败并由此抓出真缺口（见下）。
4. **质量不劣化**：citation_coverage 1.0（auto/update/simple）vs deep 1.0（真实 n=1）；`enabled=false` 无路由 baseline 保留且经 legacy 等价性测试锁定；正式 QA 对照属评测支线（PLAN §7），不在本阶段口径内。
5. **总闸（同题不同模式成本下降、质量不劣化）**：真实 run auto（策略自然判定 update）vs deep = 18,163 vs 38,068 input tokens（**−52.3%**）、时延 −20.2%、引用覆盖持平。小样本（n=1），不构成收益结论（§11.1），可重复性留评测支线。

真实证据：`smoke_out/adaptive_config_20261001T080844Z.jsonl`（修复前，闸④失败留档）与 `adaptive_config_20261001T083359Z.jsonl`（修复后，五闸全过）；mock：`adaptive_mock_20261001T075951Z.jsonl`。

### 9.5 关键裁定（全录，工作区已清除，此处为唯一存档）

- **R0 流程**：直接在 main 实施+逐交付 push（用户既定协议）；全部子智能体继承会话模型（用户 2026-10-01 指示），无模型升级阶梯。
- **R1（Task 3 修复轮 1）**：预算/时间降级不得穿透守卫地板——计划参考代码允许 stale/低置信触发的 update 被降到 simple（零 fresh 搜索），撞 §3.4 红线；裁定任一守卫信号存在时降级下限 update。计划代码让位于计划全局约束。
- **R2（Task 3 修复轮 2）**："覆盖度不足"分支理由由静态字符串改为嵌入具体数值与阈值——计划参考代码违反计划自身"理由带具体数值"红线。
- **R3（跨任务修复，真实冒烟闸④抓到）**：**探测窗口饱和缺口**——collect_features 的守卫信号原本只来自 probe_k 探测窗口；真实 run 中 8 条同题 fresh 笔记（seed deep 产物）在 embedding 下把同题 stale 记忆挤出窗口（policy.json 实证：hit_count=8、stale_hits=0），auto 误判 simple。裁定：守卫信号（stale/review_due/低置信）改为**全库扫描 + 相关性门控**（token_similarity(title+body[:400]) ≥ conflict_similarity，排除窗口内已见 note_id），零模型、可复算，与 conflict 全库扫描同口径。修复后真实 run auto 正确路由 update（stale_hits=1）。volatile/hit_count/top_score 保持窗口口径（非免检红线信号）。
- **R4（Task 8 修复轮 1/2）**：验收闸②③在真实模型下按确定性拆分——0 帽半边（simple 零 fresh 搜索，结构性保证）mock/config 同闸硬；成本差半边与搜索阶梯半边（模型在帽内自选，非策略属性）mock 硬、config 留痕警示（`cost_gap_ok`/`fresh_search_ladder_ok`，None=未判定不假绿）。
- **R5（Task 6 修复轮）**：simple 收窄断言从条件式改无条件（brief 草稿的 if/else 使核心交付从未被断言）。
- **R6（Task 9）**：真实模型对照（5 run）授权执行（P2-G 惯例+质量优先政策+键已配）；用户 09-30 遗留的 mcp-memory 文档登记单独成 commit（`fa9a07a`），`mcp-memory/` 代码目录按其边界声明保持未跟踪。

### 9.6 会话统计

| 项 | 数值 |
|---|---|
| 子智能体 | **31**（实现侧 15：8 任务派工 + 7 次 resume 修复；评审侧 15：8 任务审 + 6 范围化复审 + 1 终审；前置侦察 1） |
| 修复轮 | 7 次（T3×2、T6×1、T8×2、跨任务饱和修复×1、终审修复波×1），全部经范围化复审验证 |
| 提交 | 17 个（`0149446..08a7f23`），全部已推送 origin/main |
| 测试 | 703 → **747**（策略 26 + loop 接线 8 + MCP 5 + index 3 + smoke 5 等） |
| 终审结论 | With fixes → 修复波（2 条接线行为断言 + lint 债清零）→ 复审 clean |

**Deferred minors（缓还清单，终审已分类：无阻塞项）**：config 校验加固（conflict_similarity/coverage_min_score 夹取、非 Mapping override 的 TypeError、_VOLATILITY_RANK .get 化）；time_budget_seconds 路径零覆盖（预算因子当前惰性，激活时一并补）；recall 双开索引与全库扫描 O(笔记文件数)（10k+ 笔记优化目标）；默认值锁定测试仅锁 2 值（可固化为全等断言）；杂项命名/重复表达式。**预算因子结构性惰性（判定先于一切模型调用，budget_remaining_ratio 恒 1.0）**为计划层面已知限制，评测期再议 mid-run 重判定。

**P2 移交三项（范围外确认）**：①替换门与相似度带明文化 ②merge 独立下限 ③F7 id 形态护栏——均未在本会话处理，继续悬置（②③触发条件在 P4）。
