# ResearchWiki · 自进化研究 Wiki 智能体

给它一个研究问题和一批信息源，它长程地搜索、阅读、综合，把结论沉淀成一份带交叉引用的个人 wiki；后续提问优先复用 wiki，越用越快越准。

> 设计文档见 [PLAN.md](PLAN.md)（分 5 阶段推进，不按周排期）。

## 当前状态

- ✅ 阶段 1 · 骨架 + Web 前端（完成 2026-09-17）：Mock 驱动全流程演示
- 🚧 阶段 2 · 真实 loop + 工具 + 真模型接入（进行中）

## 阶段 1 已交付

- ✅ LLM Provider 接口 + MockProvider（零 API key、零成本跑通全流程）
- ✅ FastAPI server：SSE 输出 AI SDK UI Message Stream 协议
- ✅ Next.js 前端：流式输出、思考折叠（含持续秒数）、研究过程折叠摘要、原子笔记流、冲突提示、报告文档视图（引用悬浮卡 + 来源列表）、Wiki 沉淀面板、智能滚动

## Quickstart

```bash
# 后端（需要 uv）
uv sync
uv run researchwiki serve        # http://127.0.0.1:8000

# 前端（另开终端）
cd web
npm install
npm run dev                      # http://localhost:3000
```

打开 http://localhost:3000，输入一个研究问题（如"agent 记忆方案对比"），观察完整研究过程的流式演示。

## 架构

```
用户 ──► Web UI (Next.js + AI SDK)  ──SSE──►  FastAPI  ──►  Agent Loop
                                                            ├── llm/      Provider + Mock + 记账
                                                            ├── loop/     研究 run 编排
                                                            └── server/   SSE Data Stream
```

后续阶段：wiki 三层存储与蒸馏（阶段 2）、compaction 与长程稳定性（阶段 3）、评测（阶段 4）、开源打磨（阶段 5）。
