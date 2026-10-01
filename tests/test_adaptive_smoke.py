"""scripts/adaptive_smoke.py 的测试：mock 模式端到端矩阵 + 硬断言纯函数。

脚本不是包，用 importlib 按路径加载（与 tests/test_cold_warm_smoke.py 同款）。
端到端跑真实 AgentLoop（ScriptedProvider + MockSearch + MockEmbeddingProvider
+ httpx.MockTransport 的离线 fetch），零网络、零 key；矩阵 = 同题先播种
（forced deep + 手工 stale 记忆）再 copytree 出每模式独立副本跑
simple/update/deep/auto，断言落在 JSONL 行、policy 留痕与 token 对账上。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from researchwiki.loop.metrics import RunMetrics, sum_tokens_from_jsonl
from researchwiki.wiki.store import WikiStore

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str) -> ModuleType:
    """按路径加载脚本；先注册进 sys.modules，否则 dataclass 解析注解会失败。"""
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


adaptive = _load("adaptive_smoke", "scripts/adaptive_smoke.py")


@pytest.fixture(autouse=True)
def _no_search_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """零网络保证：mock 模式已显式锁定 MockSearch、fetch 走 MockTransport；
    这里再清掉搜索相关环境变量，兜住一切环境巧合（与 cold_warm 测试同约定）。"""
    for var in ("SEARCH_PROVIDER", "TAVILY_API_KEY", "BOCHA_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def _read_rows(out: Path) -> list[dict[str, Any]]:
    lines = out.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


EXPECTED_ROW_KEYS = {
    "mode",
    "question",
    "trace_id",
    "wiki_root",
    "run_dir",
    "provider_mode",
    "model",
    "metrics",
    "policy",
    "report_chars",
    "citation_coverage",
    "input_tokens_checked",
    "fresh_search_ladder_ok",
}


# ---- mock 模式端到端：完整四模式矩阵 -------------------------------------------


def test_mock_matrix_end_to_end(tmp_path, capsys) -> None:
    seed_root = tmp_path / "seed"
    out = tmp_path / "adaptive.jsonl"
    rc = adaptive.main(
        [
            "--provider",
            "mock",
            "--wiki-root",
            str(seed_root),
            "--out",
            str(out),
            "--env-file",
            "",
        ]
    )

    assert rc == adaptive.EXIT_PASS
    rows = _read_rows(out)

    # JSONL 恰好四行：simple / update / deep / auto，字段契约齐全
    assert [row["mode"] for row in rows] == ["simple", "update", "deep", "auto"]
    assert all(set(row) == EXPECTED_ROW_KEYS for row in rows)
    assert all(set(row["metrics"]) == set(RunMetrics().to_dict()) for row in rows)
    assert all(row["provider_mode"] == "mock" for row in rows)
    assert all(row["question"] == adaptive.DEFAULT_QUESTION for row in rows)

    # 硬断言 ①：每行 policy 含 features 与非空 reasons（不允许裸模式字符串），
    # 且与 run_dir/policy.json 的落盘原文一致（测的是落盘契约）
    for row in rows:
        policy = row["policy"]
        assert policy["features"], "policy.features 必须非空"
        assert any(str(r).strip() for r in policy["reasons"]), "policy.reasons 必须非空"
        on_disk = json.loads(
            (Path(row["run_dir"]) / "policy.json").read_text(encoding="utf-8")
        )
        assert on_disk["mode"] == policy["mode"]
        assert on_disk["features"] == policy["features"]
    # forced 标记：三个显式模式 forced=True，auto 自然判定 forced=False
    assert [row["policy"]["forced"] for row in rows] == [True, True, True, False]
    assert [row["policy"]["mode"] for row in rows] == [
        "simple",
        "update",
        "deep",
        "update",
    ]

    # 硬断言 ④（守卫红线复验）：播种含 stale 记忆时 auto 不落 simple；
    # stale_hits ≥ 1 证明判定真的看到了那条旧记忆
    auto = rows[-1]
    assert auto["policy"]["mode"] != "simple"
    assert auto["policy"]["mode"] == "update"
    assert auto["policy"]["features"]["stale_hits"] >= 1

    # 硬断言 ②：mock 下 simple 成本 < deep 且 simple 零 fresh 搜索
    by_mode = {row["mode"]: row for row in rows}
    simple_metrics = by_mode["simple"]["metrics"]
    deep_metrics = by_mode["deep"]["metrics"]
    assert simple_metrics["input_tokens"] < deep_metrics["input_tokens"]
    assert simple_metrics["fresh_search_count"] == 0

    # 硬断言 ③：fresh 搜索沿 simple ≤ update ≤ deep 单调（mock 下是硬闸）
    counts = [
        by_mode[mode]["metrics"]["fresh_search_count"]
        for mode in ("simple", "update", "deep")
    ]
    assert counts == sorted(counts)
    # ③ 的行级留痕：mock 矩阵阶梯单调 → 每行盖章 true（断裂时 verify_rows 会判死）
    assert all(row["fresh_search_ladder_ok"] is True for row in rows)

    # 每模式独立 wiki 根目录（warm 起跑、互不污染），且副本里确有 stale 种子笔记
    roots = [Path(row["wiki_root"]) for row in rows]
    assert len(set(roots)) == 4
    assert seed_root not in roots
    for root in roots:
        notes = WikiStore(root).list_notes()
        assert len(notes) >= 2, "副本必须同时含播种蒸馏笔记 + 手工 stale 记忆"
        assert any(
            note.meta.volatility == "volatile"
            and note.meta.observed_at == adaptive.STALE_OBSERVED_AT
            for note in notes
        ), "stale 种子笔记必须带旧 observed_at"

    # 硬断言 ⑤：token 可复算——metrics 与 tokens.jsonl 按 trace_id 复算一致，
    # 且行内 input_tokens_checked 就是这个对账值
    for row in rows:
        tok_in, tok_out = sum_tokens_from_jsonl(
            Path(row["wiki_root"]) / "tokens.jsonl", row["trace_id"]
        )
        assert tok_in > 0, "对账不应空转"
        assert row["metrics"]["input_tokens"] == tok_in
        assert row["input_tokens_checked"] == tok_in

    # 四次 run 是独立 AgentLoop 实例：独立 trace_id / run_dir，产物落盘齐全
    assert len({row["trace_id"] for row in rows}) == 4
    assert len({row["run_dir"] for row in rows}) == 4
    for row in rows:
        assert (Path(row["run_dir"]) / "run-metrics.json").is_file()
        assert (Path(row["run_dir"]) / "report.md").is_file()
        assert row["report_chars"] > 0

    # 汇总表可打印：模式列、成本/检索列、防误读声明
    printed = capsys.readouterr().out
    for field in ("simple", "update", "deep", "auto"):
        assert field in printed
    assert "fresh_search_count" in printed and "input_tokens" in printed
    assert "小样本冒烟" in printed
    assert f"JSONL 已写入 {out}" in printed

    # 脚本自带的硬断言全绿（与上面逐条断言互为印证）
    assert adaptive.verify_rows(rows) == []


def test_mock_matrix_modes_subset(tmp_path) -> None:
    """--modes 子集：只跑请求的模式；缺 update 时不做该档单调断言，硬闸仍全绿。"""
    out = tmp_path / "subset.jsonl"
    rc = adaptive.main(
        [
            "--provider",
            "mock",
            "--modes",
            "simple,deep",
            "--wiki-root",
            str(tmp_path / "seed"),
            "--out",
            str(out),
            "--env-file",
            "",
        ]
    )

    assert rc == adaptive.EXIT_PASS
    rows = _read_rows(out)
    assert [row["mode"] for row in rows] == ["simple", "deep"]
    assert adaptive.verify_rows(rows) == []


def test_ladder_assertion_mock_hard_config_soft(tmp_path) -> None:
    """断言 ③ 按模式分治（fix-round 裁定）：mock 非单调 → 硬闸；config 同数据 → 不判死。

    判定内核（ladder_breaks / fresh_search_ladder_ok）两模式共用一份：config 的
    断裂只算出来留痕（false 落盘 + main 打警示），绝不进失败清单——真实模型在帽内
    自选搜索次数，硬闸度量的是模型服从度而非策略正确性。
    """

    def make_rows(provider_mode: str, counts: tuple[int, ...]) -> list[dict[str, Any]]:
        modes = ("simple", "update", "deep")[: len(counts)]
        return [
            {
                "mode": mode,
                "provider_mode": provider_mode,
                "trace_id": f"t{index}",
                "wiki_root": str(tmp_path),
                "metrics": {"fresh_search_count": count, "input_tokens": 0,
                            "output_tokens": 0},
            }
            for index, (mode, count) in enumerate(zip(modes, counts, strict=True))
        ]

    broken_mock = make_rows("mock", (0, 2, 1))
    assert adaptive.fresh_search_ladder_ok(broken_mock) is False
    assert any("阶梯断裂" in failure for failure in adaptive.verify_rows(broken_mock))

    broken_config = make_rows("config", (0, 2, 1))
    assert adaptive.fresh_search_ladder_ok(broken_config) is False
    # 同样的断裂在 config 下不进失败清单（其余断言照常各自判定）
    assert not any("阶梯" in failure for failure in adaptive.verify_rows(broken_config))

    # 链上不足两个强制模式 → None（无可比较对，不把"未判定"伪装成"已通过"）
    assert adaptive.fresh_search_ladder_ok(make_rows("mock", (0,))) is None


def test_invalid_mode_rejected_by_argparse(tmp_path) -> None:
    """非法模式名在 argparse 层拒绝（exit code 2），不产生任何输出文件。"""
    out = tmp_path / "bad.jsonl"
    with pytest.raises(SystemExit) as excinfo:
        adaptive.main(
            [
                "--provider",
                "mock",
                "--modes",
                "simple,ultra",
                "--out",
                str(out),
                "--env-file",
                "",
            ]
        )
    assert excinfo.value.code != 0
    assert not out.exists()
