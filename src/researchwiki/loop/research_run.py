"""Mock 研究 run：按 AI SDK UI Message Stream 协议产出事件序列。

阶段 1 的演示场景，完整走一遍"规划 → 子 agent 搜索阅读 → 蒸馏笔记 → 冲突提示 → 报告"。
真模型接入后，由真实 loop 产生同样的事件序列，前端零改动。
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

from researchwiki.llm.accounting import TokenAccountant
from researchwiki.llm.provider import TokenUsage, chunk_text

# 事件间停顿（秒），让前端流式效果可感知；生产中由真实 LLM 延迟决定
DEMO_DELAY = 0.04


class ResearchRun:
    """一次研究 run 的事件编排器。Mock 阶段全部内容为脚本；接口即协议。"""

    def __init__(
        self,
        question: str,
        *,
        accountant: TokenAccountant | None = None,
        delay: float = DEMO_DELAY,
    ) -> None:
        self.question = question
        self.accountant = accountant
        self.delay = delay
        self.trace_id = accountant.new_trace_id() if accountant else "mock00000000"

    # ---- 事件发射辅助 ----------------------------------------------------

    def _emit(self, event: dict[str, Any]) -> dict[str, Any]:
        if self.delay:
            time.sleep(self.delay)
        return event

    def _reasoning(self, rid: str, text: str) -> Iterator[dict[str, Any]]:
        yield self._emit({"type": "reasoning-start", "id": rid})
        for chunk in chunk_text(text, 30):
            yield self._emit({"type": "reasoning-delta", "id": rid, "delta": chunk})
        yield self._emit({"type": "reasoning-end", "id": rid})
        self._record_usage(step=f"reasoning:{rid}")

    def _task(
        self, task_id: str, data: dict[str, Any]
    ) -> Iterator[dict[str, Any]]:
        yield self._emit({"type": "data-task", "id": task_id, "data": data})
        self._record_usage(step=f"task:{task_id}")

    def _text(self, tid: str, text: str) -> Iterator[dict[str, Any]]:
        yield self._emit({"type": "text-start", "id": tid})
        for chunk in chunk_text(text, 60):
            yield self._emit({"type": "text-delta", "id": tid, "delta": chunk})
        yield self._emit({"type": "text-end", "id": tid})

    def _record_usage(self, step: str) -> None:
        if self.accountant:
            self.accountant.record(
                trace_id=self.trace_id,
                step=step,
                model="mock-strong",
                usage=TokenUsage(input_tokens=900, output_tokens=120),
                latency_ms=self.delay * 1000,
            )

    # ---- 主流程 ----------------------------------------------------------

    def events(self) -> Iterator[dict[str, Any]]:
        q = self.question
        yield self._emit({"type": "start"})

        # 1. 规划（思考过程，前端 shimmer 折叠展示）
        yield from self._reasoning(
            "plan",
            f"分析研究问题「{q}」：拆成 4 个子任务——检索主流方案、"
            "阅读核心来源、蒸馏原子笔记并写入 wiki、对账冲突后撰写带引用报告。"
            "子 agent 用 cheap 档模型，规划与蒸馏用 strong 档。",
        )

        # 2. 子 agent 1：搜索
        yield from self._task(
            "t1",
            {
                "title": "搜索主流 agent 记忆方案",
                "status": "running",
                "detail": "检索中 3/10 次",
            },
        )
        yield from self._task(
            "t1",
            {
                "title": "搜索主流 agent 记忆方案",
                "status": "done",
                "detail": "10 次检索 · 命中 12 个来源",
            },
        )

        # 3. 子 agent 2：阅读
        yield from self._task(
            "t2",
            {
                "title": "阅读并抽取关键来源",
                "status": "running",
                "detail": "fetch_url ×4，长文分块阅读",
            },
        )
        yield from self._task(
            "t2",
            {
                "title": "阅读并抽取关键来源",
                "status": "done",
                "detail": "4 个来源 · 正文已落盘 sources/",
            },
        )

        # 4. 蒸馏：原子笔记写入 wiki（前端"wiki 在生长"演示点）
        yield from self._task(
            "t3", {"title": "蒸馏原子笔记 → wiki", "status": "running", "detail": "抽取事实中"}
        )
        notes = [
            {
                "id": "N-0001",
                "text": "Letta（MemGPT）通过后台 subagent 在会话间整理记忆，"
                "即 sleep-time compute。",
                "entities": ["Letta", "Sleep-time-Compute"],
                "confidence": "high",
            },
            {
                "id": "N-0002",
                "text": "Mem0 的记忆流水线包含 add / update / merge 操作，写入前做相似度查重。",
                "entities": ["Mem0"],
                "confidence": "high",
            },
            {
                "id": "N-0003",
                "text": "滚动 compaction 在上下文超过窗口 70% 时触发，早期轮次折叠进 state 文件。",
                "entities": ["Context-Compression"],
                "confidence": "high",
            },
            {
                "id": "N-0004",
                "text": "Anthropic 的 prompt caching 需要显式设置 cache_control 断点。",
                "entities": ["Prompt-Caching"],
                "confidence": "medium",
            },
            {
                "id": "N-0005",
                "text": "sqlite-vec 提供嵌入式向量检索，单文件零运维。",
                "entities": ["SQLite-vec"],
                "confidence": "high",
            },
        ]
        for i, note in enumerate(notes, 1):
            yield self._emit(
                {
                    "type": "data-note",
                    "id": f"note-{i}",
                    "data": note,
                }
            )
        yield from self._task(
            "t3",
            {
                "title": "蒸馏原子笔记 → wiki",
                "status": "done",
                "detail": "新增 5 条笔记 · 1 条合并去重",
            },
        )

        # 5. 冲突对账提示
        yield self._emit(
            {
                "type": "data-conflict",
                "id": "conflict-1",
                "data": {
                    "id": "C-0001",
                    "summary": "两个来源对 GLM 旗舰档上下文窗口的表述不一致（128K vs 200K）",
                    "action": "已写入 conflicts/ 台账，下轮研究优先核实",
                },
            }
        )

        # 6. 撰写报告（正文流式输出，行内引用）
        report = (
            "## 研究报告：Agent 记忆方案对比\n\n"
            "### 1. 背景与问题\n"
            f"围绕「{q}」，本次研究检索了 12 个来源、精读 4 篇，"
            "蒸馏出 5 条原子笔记并写入 wiki。\n\n"
            "### 2. 核心发现\n"
            "- **Sleep-time compute**：Letta 通过后台 subagent 在会话间整理记忆，"
            "把记忆维护成本移出交互窗口[1]。\n"
            "- **记忆写入流水线**：Mem0 在写入前做相似度查重，"
            "add/update/merge 三步操作[2]。\n"
            "- **上下文压缩**：主流做法是 70% 窗口阈值触发滚动压缩，"
            "早期轮次折叠进状态文件[3]。\n"
            "- **KV-cache**：Anthropic 需要显式 `cache_control` 断点，"
            "OpenAI 兼容端点则依赖前缀稳定[4]。\n\n"
            "### 3. 对本项目的启示\n"
            "三层数据模型（sources → notes → pages）与 consolidation 后台任务的设计，"
            "与上述方案的方向一致；差异点在于本项目的 wiki 是人类可读的 Markdown 双链网络。\n\n"
            "### 4. 未决问题\n"
            "- GLM 旗舰档的上下文窗口存在来源分歧，已进入冲突台账待核实。\n"
            "- 嵌入模型在中文长文本上的召回需实测（见 Day 1 清单）。\n"
        )
        yield from self._text("report", report)

        # 7. 引用来源
        sources = [
            {
                "sourceId": "s1",
                "url": "https://github.com/letta-ai/letta",
                "title": "Letta (MemGPT) — GitHub",
            },
            {
                "sourceId": "s2",
                "url": "https://github.com/mem0ai/mem0",
                "title": "Mem0 — GitHub",
            },
            {
                "sourceId": "s3",
                "url": "https://www.anthropic.com/engineering/claude-code-best-practices",
                "title": "Claude Code 最佳实践",
            },
            {
                "sourceId": "s4",
                "url": "https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching",
                "title": "Anthropic Prompt Caching 文档",
            },
        ]
        for s in sources:
            yield self._emit({"type": "source-url", **s})

        yield self._emit({"type": "finish"})
