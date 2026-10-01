# 分支线工作日志 · mcp-memory（ResearchWiki 记忆 → DSH 插件小版）

> **这条线是什么**：把 ResearchWiki 的外置记忆思想蒸馏成一个**独立小版** MCP server（stdio），接入 DSH（DeepSeek Harness）Desktop，让 DSH 的 agent 跨会话具备持久记忆。它是一条**分支线**，不是主线 PLAN v2 四阶段路线的一部分。

## ⚠ 接手前先读：与主线的关系

- **不走**主线的 PLAN 阶段路线、SDD 台账（`.superpowers/sdd/`）、阶段验收流程；主线工作日志见 [agent-worklog.md](./agent-worklog.md)（其第 8 节只有指路登记）。
- **主仓库零提交**：产物全部在未跟踪目录 `mcp-memory/`、用户目录 `~/.researchwiki-memory/`（数据）与 `~/.dsh/profiles/desktop/cordis.patch.yml`（挂载点），git 状态仅 `?? mcp-memory/`。
- **与主项目零耦合**：Dockerfile 选择性 COPY 不含本目录；主项目 uv venv / uv.lock 未动；主项目声明的 `fastmcp>=4.0.5` 与本支线所装官方 `mcp` SDK 2.x 相互独立、分属不同环境（主项目 uv 3.12 venv，本支线 Store Python 3.11 用户 site-packages）。
- 改这条线只动 `mcp-memory/`；**不要**把本目录计进主线测试/构建口径；接手主线的人不需要读本文件，接手本支线的人不需要读主线第 1–7 节。

---

## 会话记录

### 会话 1（2026-09-30 约 18:20–19:50，单智能体，无子智能体派工）

**动机**：用户询问"项目弄成 DSH 的插件难不难"。裁定：原生 Cordis 插件中等偏难（要做成 npm 包进 DSH profile 工作区、对齐 0.2.0-rc.2 内部 tools/llm API，升级易碎）→ 暂不做；走官方一等公民路径 `@deepseek-ai/dsh-mcp-client` + stdio MCP server。

**过程**：

1. 摸底 DSH Desktop 2.0.17（装于 `D:\deepseek_harness`，Electron 壳本身就是 Cordis 插件，`resources/app` 未打包 asar 可直读）＋对照官方开源仓库文档（`docs/user/guide/mcp-memory.zh.md`、`docs/cookbook/extension-cookbook.zh.md` 等）确认 MCP 记忆接入格式与验证流程。
2. 新建 `mcp-memory/` 子项目（19:26–19:28）：`server.py`（JSONL 追加式存储，6 工具，supersede 失效链保留旧记录、检索只返回 active）；`smoke_test.py`（11 项断言，含**杀进程重启后召回**的持久性验证，全部通过；期间两处失败均为测试脚本自身抠字符串 bug，server 行为无缺陷）；`README.md`。
3. 挂载进 DSH（约 19:45）：`~/.dsh/profiles/desktop/cordis.patch.yml` 末尾追加 insert 条目（id `memory-researchwiki`，serverName `researchwiki`），原文件备份 `cordis.patch.yml.bak-20260930`；用 DSH 自带 yaml 包解析验证通过。

**产物**：

- `mcp-memory/server.py`、`mcp-memory/smoke_test.py`、`mcp-memory/README.md`
- `~/.dsh/profiles/desktop/cordis.patch.yml` 新增 insert 条目（含备份）
- `~/.researchwiki-memory/`（运行期数据目录）；ZCode 侧持久记忆 `researchwiki-dsh-memory-plugin.md`
- 本文件（分支线日志）

---

## 当前状态与接手点（截至 2026-09-30 21:58）

- **已验证**：server 冒烟测试 11/11；patch 文件语法经 DSH 自带 yaml 解析器验证。
- **待验证（下一个动作）**：用户**完全退出 DSH Desktop（含托盘）→ 重启 → 新会话**确认 `mcp__researchwiki__memory_*` 六工具出现 → 按官方流程验证（会话 A 写"记住我的验证饮品是正山小种-lapsang-xxx"→ 新建会话 B 问"我的验证饮品是什么？查记忆"→ 确认召回）。
- **未做**：git 提交/推送（整个目录未跟踪）；原生 Cordis 插件完整版；检索为关键词匹配（非语义/embedding）。
- **移除方法**：删 patch 条目（或恢复 `.bak-20260930`）并重启 DSH 即解挂；`~/.researchwiki-memory/` 可整目录删除。
- **若决定并入主线**：用主项目已声明的 `fastmcp>=4.0.5` 重写 server 统一技术栈；supersede/失效语义对齐 P1/P2 的生命周期机制（本小版是同一思想的最简实现）；数据落点与主线 store 的关系需先裁定。

## 环境事实（接手必读）

- DSH＝DeepSeek Harness，开源在 github.com/deepseek-ai/deepseek-harness，文档 deepseek-harness.github.io/deepseek-harness（有中文版）。
- `$DSH_HOME`＝`~/.dsh`；desktop profile 的用户 patch 层＝`~/.dsh/profiles/desktop/cordis.patch.yml`（YAML 数组，insert 条目加载插件）；DSH 插件＝Cordis 模块（`name` + `apply(ctx)`），agent 工具注册走 `ctx.tools.register()`；MCP 工具在 agent 里的名字形态为 `mcp__<serverName>__<tool>`。
- 本机 Python：`py` 启动器 → Store Python 3.11.9；`mcp` 2.x 已装其用户 site-packages；数据目录可用环境变量 `RESEARCHWIKI_MEMORY_DIR` 覆盖。
- 冒烟测试跑法：`cd mcp-memory && py smoke_test.py`。
