"""确定性指标（evals/metrics.py）单测（评测支线 task-3）。

覆盖 task-3 简报的五块契约：

- normalize_answer：全角→半角、英文小写、去中英文标点、空白删除（规则见 docstring）；
- point_hit / normalized_em：要点级包含命中（gold 空串不命中；空要点列表 EM=0）；
- refusal_detected：拒答词典 + 归一后长度 < 80 双条件（常量可测）；
- mean / percentile：空序列 0.0、单元素、p50/p95 线性插值，且与
  scripts/gate_retrieval.py 的 percentile 逐用例同口径；
- aggregate_rows：按 condition 分组、em 缺失行不计 em 分母但 n 仍计、
  fresh_search_count 缺失组跳过字段、各组 n 正确。

全部零网络、零模型调用。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from researchwiki.evals.metrics import (
    REFUSAL_MAX_LEN,
    REFUSAL_PHRASES,
    aggregate_rows,
    mean,
    normalize_answer,
    normalized_em,
    percentile,
    point_hit,
    refusal_detected,
)

ROOT = Path(__file__).resolve().parents[1]


def _load_gate_retrieval() -> ModuleType:
    """按路径加载 scripts/gate_retrieval.py，用于钉住 percentile 同口径。"""
    name = "gate_retrieval_metrics_parity"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "gate_retrieval.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# normalize_answer：全角 / 大小写 / 标点 / 空白
# ---------------------------------------------------------------------------


def test_normalize_answer_rules() -> None:
    """四步规则：NFKC 全角→半角、英文小写、去中英文标点、删除全部空白。"""
    # 全角字母/数字/标点折算为半角后统一小写
    assert normalize_answer("ＬａｎｇＣｈａｉｎ Ｖｅｒｓｉｏｎ １．２") == "langchainversion12"
    # 英文统一小写（连字符等标点删除）
    assert normalize_answer("LangChain-Agent") == "langchainagent"
    # 去中英文标点：冒号/叹号/逗号/全角括号/小数点均删，保留汉字、数字、字母
    assert normalize_answer("默认重试：3次！") == "默认重试3次"
    assert normalize_answer("QPS 15,600（峰值）") == "qps15600峰值"
    # 等值内符号 % 保留（去标点不误伤百分数）
    assert normalize_answer("准确率 93.7%") == "准确率937%"
    # 空白删除：中文排版空格无语义，制表/换行/首尾空白一并处理
    assert normalize_answer("  多  空格\t与\n换行  ") == "多空格与换行"
    # 确定性：排版不同的同一内容归一后一致；下划线按字母数字类保留
    assert normalize_answer("A Ｂ，c") == normalize_answer("a b c") == "abc"
    assert normalize_answer("foo_bar") == "foo_bar"


# ---------------------------------------------------------------------------
# point_hit / normalized_em：要点级包含命中
# ---------------------------------------------------------------------------


def test_point_hit_rules() -> None:
    """归一后 gold 是 answer 子串即命中；gold 归一后为空不命中。"""
    # 正例：标点/空白/大小写差异不影响命中
    assert point_hit("默认重试 3 次，配合指数退避。", "重试：3次") is True
    assert point_hit("LangChain v0.2.20 released", "0.2.20") is True
    # 负例：要点未出现
    assert point_hit("答案是四个", "五个") is False
    # gold 空串 / 纯标点 → 归一后为空 → False（防空要点虚增 EM）
    assert point_hit("任意回答", "") is False
    assert point_hit("任意回答", "！？。") is False


def test_normalized_em_cases() -> None:
    """EM = 命中要点数 / 要点总数；空要点列表 → 0.0。"""
    assert normalized_em("默认重试3次，配合指数退避", ["重试 3 次", "指数退避"]) == 1.0
    assert normalized_em("默认重试3次", ["重试 3 次", "指数退避"]) == 0.5
    assert normalized_em("完全无关的回答", ["重试 3 次", "指数退避"]) == 0.0
    assert normalized_em("任意回答", []) == 0.0


# ---------------------------------------------------------------------------
# refusal_detected：拒答词典 + 长度双条件
# ---------------------------------------------------------------------------


def test_refusal_detected_rules() -> None:
    """命中拒答词典且归一后长度 < 80 才算拒答；任一条件不满足即 False。"""
    # 词典与阈值是模块常量，可直接钉住内容
    assert REFUSAL_MAX_LEN == 80
    assert {"无法确定", "我不知道", "资料未提及", "没有找到", "无法回答", "不确定"} <= set(
        REFUSAL_PHRASES
    )
    # 命中 + 短答案 → True
    assert refusal_detected("抱歉，资料未提及该信息。") is True
    assert refusal_detected("我不知道。") is True
    # 命中但归一后长度 ≥ 80 → False
    assert refusal_detected("我不知道。此外" + "详细" * 45) is False
    # 边界：归一后恰好 80 字 → False（严格小于）
    exactly = "我不知道" + "字" * 76
    assert len(normalize_answer(exactly)) == REFUSAL_MAX_LEN
    assert refusal_detected(exactly) is False
    # 不命中词典 → False（无论长短）
    assert refusal_detected("默认重试 3 次。") is False


# ---------------------------------------------------------------------------
# mean / percentile：与 gate_retrieval 同口径
# ---------------------------------------------------------------------------


def test_mean_and_percentile_basics() -> None:
    """空序列 0.0；单元素原样；p50 精确中位；p95 线性插值。"""
    assert mean([]) == 0.0
    assert mean([2.0, 4.0]) == 3.0
    assert percentile([], 50) == 0.0
    assert percentile([7.0], 95) == 7.0
    # 偶数个元素的 p50：rank=1.5 → 2 与 3 的线性插值
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    # p95 插值：10 个元素 rank=0.95*9=8.55 → 8 + 0.55*(9-8)
    assert percentile([float(i) for i in range(10)], 95) == pytest.approx(8.55)


def test_percentile_matches_gate_script() -> None:
    """与 scripts/gate_retrieval.py 的 percentile 逐用例一致（报告口径可比）。"""
    gate = _load_gate_retrieval()
    cases: list[tuple[list[float], float]] = [
        ([1.0, 2.0, 3.0, 4.0], 50),
        ([float(i) for i in range(10)], 95),
        ([5.0], 50),
        ([3.0, 1.0, 2.0], 50),
        ([], 50),
        ([100.0, 300.0], 95),
    ]
    for values, p in cases:
        assert percentile(values, p) == gate.percentile(values, p)
    assert percentile([100.0, 300.0], 95) == pytest.approx(290.0)


# ---------------------------------------------------------------------------
# aggregate_rows：分组聚合与缺失语义
# ---------------------------------------------------------------------------


def test_aggregate_rows_groups_two_conditions() -> None:
    """按 condition 分组：n / em_mean / refusal_rate / 分位 / token 均值。"""
    rows: list[dict[str, object]] = [
        {
            "condition": "baseline",
            "em": 1.0,
            "refusal": False,
            "latency_ms": 100,
            "input_tokens": 10,
            "output_tokens": 20,
        },
        {
            "condition": "baseline",
            "em": 0.0,
            "refusal": True,
            "latency_ms": 300,
            "input_tokens": 30,
            "output_tokens": 40,
        },
        {
            "condition": "loop",
            "em": 1.0,
            "refusal": False,
            "latency_ms": 200,
            "input_tokens": 50,
            "output_tokens": 60,
            "fresh_search_count": 2,
        },
    ]
    out = aggregate_rows(rows)
    assert set(out) == {"baseline", "loop"}
    base = out["baseline"]
    assert base["n"] == 2
    assert base["em_mean"] == 0.5
    assert base["refusal_rate"] == 0.5
    assert base["latency_p50"] == 200.0
    assert base["latency_p95"] == pytest.approx(290.0)
    assert base["input_tokens_mean"] == 20.0
    assert base["output_tokens_mean"] == 30.0
    # 整组都不携带 fresh_search_count → 跳过该字段
    assert "fresh_search_mean" not in base
    loop = out["loop"]
    assert loop["n"] == 1
    assert loop["em_mean"] == 1.0
    assert loop["fresh_search_mean"] == 2.0
    # 空输入 → 空 dict
    assert aggregate_rows([]) == {}


def test_aggregate_rows_missing_semantics() -> None:
    """em 缺失行不计 em 分母但 n 仍计；fresh_search 缺失按 0；refusal 缺失按 False。"""
    rows: list[dict[str, object]] = [
        {
            "condition": "loop",
            "em": 1.0,
            "refusal": False,
            "latency_ms": 100,
            "input_tokens": 10,
            "output_tokens": 5,
            "fresh_search_count": 3,
        },
        # em 与 fresh_search_count 都缺失：n 计入、em 分母不计
        {
            "condition": "loop",
            "refusal": True,
            "latency_ms": 200,
            "input_tokens": 20,
            "output_tokens": 15,
        },
        {
            "condition": "loop",
            "em": 0.0,
            "refusal": False,
            "latency_ms": 300,
            "input_tokens": 30,
            "output_tokens": 25,
            "fresh_search_count": 0,
        },
        # em 显式 None 视同缺失
        {
            "condition": "loop",
            "em": None,
            "refusal": False,
            "latency_ms": 400,
            "input_tokens": 40,
            "output_tokens": 35,
            "fresh_search_count": 1,
        },
    ]
    out = aggregate_rows(rows)["loop"]
    assert out["n"] == 4
    assert out["em_mean"] == 0.5  # (1.0 + 0.0) / 2，None 与缺失不计分母
    assert out["refusal_rate"] == 0.25  # 4 行中 1 行 True，缺失按 False
    assert out["latency_p50"] == 250.0
    assert out["input_tokens_mean"] == 25.0
    assert out["output_tokens_mean"] == 20.0
    # 组内至少一行携带 → 字段输出，缺失行按 0 计：(3 + 0 + 0 + 1) / 4
    assert out["fresh_search_mean"] == 1.0
