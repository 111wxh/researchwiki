"""run-metrics.json 指标模块：一次 Research run 的复用/成本/质量指标。

本模块只做三件事，不依赖 AgentLoop（接入由 Task 3 完成，避免循环依赖）：

1. `RunMetrics` 数据结构：字段与 PLAN §4.4 的最小指标集逐字一致，这是后续
   评测复算的契约，不得改名/增删；`to_dict()` 产出的键即 run-metrics.json 的键。
2. 落盘与对账：`write_run_metrics` 写 `<run_dir>/run-metrics.json`；
   `sum_tokens_from_jsonl` 按 trace_id 汇总 tokens.jsonl，两者共同保证
   PLAN §4.5 的可复算性（run-metrics.json 与 tokens.jsonl 能复算本次 run 的成本）。
3. `compute_citation_coverage`：引用覆盖率的候选实现（计算口径最终由 Task 3 定）。

tokens.jsonl 行结构见 `researchwiki.llm.accounting.TokenAccountant.record`：
每行 JSON 含 ``trace_id`` 与 ``asdict(TokenUsage)`` 展开字段
（``input_tokens`` / ``output_tokens`` / ``cache_read_tokens`` / ``cache_creation_tokens``）。
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

# 报告正文中的行内引用标记，如 [1]、[12]
_CITATION_MARKER = re.compile(r"\[(\d+)\]")


@dataclass
class RunMetrics:
    """一次 run 的最小指标集（PLAN §4.4，逐字契约）。

    默认值即 §4.4 JSON 中的数值：计数为 0、列表为空、无法计算时
    ``citation_coverage`` 为 None。
    """

    trace_id: str = ""
    # ---- prior 复用 ------------------------------------------------------
    prior_hit_count: int = 0
    prior_note_ids: list[str] = field(default_factory=list)
    prior_context_chars: int = 0
    # ---- fresh 检索 ------------------------------------------------------
    fresh_search_count: int = 0
    fresh_fetch_count: int = 0
    source_count: int = 0
    # ---- wiki 写入动作 ---------------------------------------------------
    notes_created: int = 0
    notes_merged: int = 0
    notes_superseded: int = 0
    # ---- 成本 ------------------------------------------------------------
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    # ---- 质量 ------------------------------------------------------------
    citation_coverage: float | None = None

    def to_dict(self) -> dict:
        """产出与 PLAN §4.4 完全一致的键与值（run-metrics.json 的内容）。"""
        return asdict(self)

    @classmethod
    def from_loop(
        cls,
        *,
        trace_id: str,
        prior_hit_count: int = 0,
        prior_note_ids: list[str] | None = None,
        prior_context_chars: int = 0,
        fresh_search_count: int = 0,
        fresh_fetch_count: int = 0,
        source_count: int = 0,
        notes_created: int = 0,
        notes_merged: int = 0,
        notes_superseded: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency_ms: int = 0,
        citation_coverage: float | None = None,
    ) -> RunMetrics:
        """从 AgentLoop 的 run 状态装载指标（Task 3 在 run 收尾时调用）。

        参数全部是简单标量/列表：trace_id、prior 命中计数、fresh 工具调用计数、
        wiki store 动作结果（created/merged/superseded）、token 会计合计与总时延。
        不 import agent_loop、不接触其私有状态——Task 3 负责把这些值取出来传进来。
        token 合计可用 :func:`sum_tokens_from_jsonl` 从 tokens.jsonl 复算核对。
        """
        return cls(
            trace_id=trace_id,
            prior_hit_count=prior_hit_count,
            prior_note_ids=list(prior_note_ids) if prior_note_ids else [],
            prior_context_chars=prior_context_chars,
            fresh_search_count=fresh_search_count,
            fresh_fetch_count=fresh_fetch_count,
            source_count=source_count,
            notes_created=notes_created,
            notes_merged=notes_merged,
            notes_superseded=notes_superseded,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            citation_coverage=citation_coverage,
        )


def compute_citation_coverage(report_text: str, source_count: int) -> float | None:
    """按报告正文中的 ``[n]`` 引用标记计算引用覆盖率。

    口径：报告中出现的来源编号去重数 / source_count。编号只认
    ``1..source_count`` 范围内的（越界编号指向不存在的来源，不计入覆盖）。

    - ``source_count == 0``：无法计算，返回 None（与"零来源 run"语义区分）。
    - ``source_count > 0`` 且正文无任何引用标记：返回 0.0（有来源但零覆盖）。

    最终计算口径由 Task 3 决定，本函数是候选实现。
    """
    if source_count <= 0:
        return None
    covered = {
        int(m.group(1))
        for m in _CITATION_MARKER.finditer(report_text)
        if 1 <= int(m.group(1)) <= source_count
    }
    return len(covered) / source_count


def write_run_metrics(run_dir: str | Path, metrics: RunMetrics) -> Path:
    """把指标落盘为 ``<run_dir>/run-metrics.json``，返回文件路径。

    UTF-8、``ensure_ascii=False``、缩进 2；目录不存在则创建。
    """
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    out = run_path / "run-metrics.json"
    out.write_text(
        json.dumps(metrics.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return out


def sum_tokens_from_jsonl(path: str | Path, trace_id: str) -> tuple[int, int]:
    """从 tokens.jsonl 按 trace_id 汇总 token，返回 ``(input_tokens, output_tokens)``。

    行结构与 `researchwiki.llm.accounting.TokenAccountant.record` 写入的一致：
    每行 JSON 含 ``trace_id`` 与 TokenUsage 展开字段（input_tokens / output_tokens
    / cache_read_tokens / cache_creation_tokens）。

    容错与口径：

    - 文件不存在返回 ``(0, 0)``。
    - 空 file、损坏 JSON 行、缺字段行（缺 trace_id 或 token 字段）直接跳过，不抛异常。
    - ``error_type`` 非空（调用出错）的记录**计入**合计：错误调用同样消耗 token，
      复算成本时不能漏掉。
    - cache_read / cache_creation 字段不参与合计（input_tokens 已是 provider 报告值）。
    """
    tokens_path = Path(path)
    if not tokens_path.is_file():
        return (0, 0)
    total_in = 0
    total_out = 0
    with tokens_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # 损坏行：跳过
            if not isinstance(row, dict):
                continue
            if row.get("trace_id") != trace_id:
                continue
            input_tokens = row.get("input_tokens")
            output_tokens = row.get("output_tokens")
            if not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
                continue  # 缺 token 字段或类型异常：跳过
            total_in += input_tokens
            total_out += output_tokens
    return (total_in, total_out)
