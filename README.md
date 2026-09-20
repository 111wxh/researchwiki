# ResearchWiki · 自进化研究 Wiki 智能体

> A self-evolving research wiki agent — it researches, then remembers.

给它一个研究问题，它长程地搜索、阅读、综合，把结论沉淀成一份**带交叉引用的个人 wiki**；下一次提问优先复用已沉淀的知识，越用越快越准。

**和普通 deep research 的差别**：普通工具每次提问都从零开始，产出用完即弃；ResearchWiki 把每次研究的结论**蒸馏成原子笔记写进 wiki**，笔记之间用稳定 ID 双链，每条断言强制带来源 URL——下一轮研究在已有 wiki 上继续生长。

## 快速开始

### 方式一：Docker（推荐，两条命令）

**A. 演示模式** —— 零配置、零 API key、不联网、不花钱，数据全部由脚本化 mock 生成：

```bash
docker compose up --build
```

打开 http://localhost:3000 即可。适合快速看前端交互与完整事件流。

**B. 完整模式** —— 接真实模型，跑真实 agent loop：

```bash
cp .env.example .env        # 填入 key
# 并在 config.toml 里填好 [llm.strong] / [llm.cheap] 的 model 与 base_url
docker compose -f docker-compose.full.yml up --build
```

完整模式会把 `config.toml` 和 `wiki-data/` 挂进容器：前者用你自己的模型配置，后者让 wiki 沉淀的研究成果**重启不丢**。

部署到服务器时，把浏览器要访问的后端地址传进去（构建期注入）：

```bash
NEXT_PUBLIC_API_URL=http://<你的域名或IP>:8000 docker compose up --build
```

### 方式二：本地开发

```bash
# 后端（需要 uv 与 Python 3.12+）
uv sync
uv run researchwiki serve             # http://127.0.0.1:8000

# 前端（另开一个终端）
cd web && npm install && npm run dev  # http://localhost:3000
```

打开 http://localhost:3000，输入一个研究问题（例如「agent 记忆方案对比」），可以看到：研究计划（流式思考，完成后自动折叠为一行摘要）、子任务卡片、原子笔记实时入库、冲突提示、带行内引用上标的报告、以及右侧 wiki 沉淀面板。

接入真实模型只需改 `config.toml`（OpenAI 兼容协议，任何厂商均可）并把 key 放进 `.env`：

```toml
[server]
mode = "real"                          # 默认 mock

[llm.strong]
model = "glm-4.7"
base_url = "https://open.bigmodel.cn/api/paas/v4"
api_key_env = "RESEARCHWIKI_STRONG_API_KEY"

[llm.cheap]                            # 子 agent / 蒸馏走便宜档
model = "glm-4.7-flash"
base_url = "https://open.bigmodel.cn/api/paas/v4"
api_key_env = "RESEARCHWIKI_CHEAP_API_KEY"
```

## 当前进度

| 阶段 | 内容 | 状态 |
|---|---|---|
| 1 | 骨架 + Web 前端（Provider 接口、Mock 驱动全链路、SSE 协议、前端三视图） | ✅ 2026-09-17 |
| 2 | 真实 agent loop + 工具层 + 真模型接入 | ✅ 2026-09-17 |
| 3 | Wiki 蒸馏 + 三层存储 + 混合检索 + MCP server | 🚧 进行中 |
| 4 | Compaction + 长程稳定性（2h 压测、断点续跑） | 计划中 |
| 5 | 评测（~100 题 QA 集，冷启动 vs 暖启动对照实验） | 计划中 |
| 6 | 打磨与开源 | 计划中 |

测试：**102 passed**（全部离线：MockTransport 假 HTTP、脚本化 Provider、零真实网络），`ruff` 零告警。

## 架构

```
用户 ──► Web UI (Next.js + AI SDK)
              │  SSE（AI SDK UI Message Stream 协议）
              ▼
         FastAPI server
              │
              ▼
        AgentLoop（plan → act → observe）
         ├── llm/     Provider 抽象 · strong/cheap 路由 · 重试退避 · 录制回放 · token 记账
         ├── tools/   web_search · fetch_url（正文抽取 + 快照落盘）· 沙箱文件系统
         ├── loop/    研究编排 · ResearchSubagent（独立上下文与预算）
         └── wiki/    三层存储 · 蒸馏 · 混合检索（阶段 3）
```

研究过程会实时落盘到 `wiki-data/runs/<时间戳>/`：`research-plan.md`（计划）、`state.md`（滚动状态：步骤、token 消耗、预算占比、关键发现）、`report.md`（报告）。**文件系统即上下文**——崩溃后重启能接着跑，而不是从头再来。

## 关键设计

**三层记忆**：`sources`（网页快照）→ `notes`（原子笔记，每条一个事实 + 来源 URL）→ `pages`（实体/主题页，聚合断言并双链）。

**来源快照式存储**：每次抓取按 `sources/{url_sha1}/{content_hash}/` 落盘，内容变化产生新目录、旧快照永不覆盖——所以「这条结论当时依据的原文到底长什么样」永远可回溯，也为后续的过期核对提供了证据链。

**引用强制的防幻觉回路**：页面里每条断言行内标注笔记 ID，笔记必须带 URL。冲突不静默覆盖，写进冲突台账待下轮研究优先核实。

**稳定 ID 双链**：链接用 `[[entity:glm-5-3|GLM-5.3]]`（地址用 ID、显示名可变），笔记合并后旧 ID 不消失而是 `status: merged + redirect_to`，引用自动跟随——中文别名和重命名不会断链。

**成本可复算**：每次 LLM 调用写一行 JSONL（模型、输入/输出/cache token、延迟、错误类型），一次研究的总成本可以从记账文件精确复算，而不是估个大概。

**熔断与防爆**：主循环有 `max_steps` 与 token 预算双熔断，子 agent 另有独立预算，工具结果统一截断后入库——上下文撑爆是长程 agent 最常见的死法。

**自建 harness**：不依赖 LangChain / LangGraph，loop、工具注册表、路由、记账都是自己实现的——这是本项目想展示的核心能力。

## 中文检索：tokenizer 取舍

中文没有空格，SQLite 内置的 `unicode61` 分词器对中文基本不可用。默认路径用 **trigram**（SQLite ≥3.34 自带）：三元组子串索引，中文召回成立，`< 3` 字的短查询走 LIKE 兜底。

可选地，把项目放在**纯 ASCII 路径**下时，可以启用 [wangfenjin/simple](https://github.com/wangfenjin/simple)（Apache-2.0）做词级分词进一步提升召回：

```bash
# 下载对应平台的 release，解压到仓库根的 vendor/ 下
# 目录结构需为 vendor/libsimple-windows-x64/{simple.dll,dict/}
```

⚠️ 踩过的坑：simple 的 jieba 词库用 C++ 文件流读取，**路径含中文时 Windows 下会触发不可捕获的进程崩溃**（不是异常，是 `abort`）。代码因此做了探测保护——路径非 ASCII 或词库不全时自动回退 trigram，绝不盲目加载。这也是为什么本仓库默认不启用它。

`config.toml` 的 `[wiki].fts_tokenizer` 可选 `auto`（默认，探测后决定）/ `simple` / `trigram`。

向量检索用 [sqlite-vec](https://github.com/asg017/sqlite-vec)（可加载扩展）；扩展不可用时自动退回纯 Python 余弦暴力扫描，两条路径的过滤语义有测试保证一致——所以任何环境都能跑，不会因为装不上扩展就瘸一条腿。

## 项目结构

```
src/researchwiki/
├── llm/         Provider 抽象、OpenAI 兼容实现、模型路由、录制回放、token 记账
├── tools/       搜索、网页抓取与快照、沙箱文件系统
├── loop/        AgentLoop、工具注册表、ResearchSubagent、笔记存储
├── wiki/        三层存储、实体注册表、混合检索（FTS5 + sqlite-vec + RRF 融合）
├── server/      FastAPI + SSE
└── cli.py       开发用单命令入口（serve / lint / consolidate / arbitrate / serve-mcp）
web/             Next.js 前端（流式输出、思考折叠、过程可视化、报告文档视图、wiki 面板）
tests/           102 个离线测试
Dockerfile             后端镜像（uv + Python 3.12）
web/Dockerfile         前端镜像（next standalone 多阶段构建）
docker-compose.yml         演示模式（mock，零 key）
docker-compose.full.yml    完整模式（真实模型 + wiki 持久化）
PLAN.md          完整设计文档与分阶段验收标准
```

运行模式由 `config.toml` 的 `[server].mode` 决定，可被环境变量 `RESEARCHWIKI_MODE` 覆盖（容器部署用）。**mock 是默认值**——误配真模型配置也不会带着空 key 去联网，会安静地回退到 mock。

## 评测计划

不只是"跑起来好看"：计划构建 ~100 题 QA 集（含 15% **无答案题**专门检验幻觉守卫），做**冷启动 vs 暖启动的复用实验**——同一个问题，第一次研究 vs wiki 里有沉淀之后再研究，对比 token 成本与准确率。所有数字都由 token 记账 JSONL 支撑，并报告置信区间与逐题结果。

## License

[MIT](LICENSE)
