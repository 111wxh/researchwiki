"""scripts/cold_warm_smoke.py 的测试：mock 模式端到端 + 硬断言/汇总表纯函数。

脚本不是包，用 importlib 按路径加载（与 tests/test_gate_scripts.py 同款）。
端到端跑真实 AgentLoop（ScriptedProvider + MockSearch + MockEmbeddingProvider），
零网络、零 key；config 模式只测"真模型不可用时明确报失败"的路径
（config 读不到 → MockProvider 占位回复 → 蒸馏无产出 → warm prior 命中 0
→ 退出码 1，绝不静默通过）。

记忆演化三段（P1-B formation / P2-E memory_update / P2-C verification）的接线也被
本文件覆盖：脚本"传了配置"不算证据，断言落在**可观察产物**上——笔记 frontmatter
的 formation_reason、run_dir/memory-update.json、state.md 的记忆形成/记忆更新行，
以及"run_once 到底把哪三段传给了 AgentLoop"（spy 住 AgentLoop 记录 kwargs）。
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
from researchwiki.tools import MockSearch
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


smoke = _load("cold_warm_smoke", "scripts/cold_warm_smoke.py")


@pytest.fixture(autouse=True)
def _no_search_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """零网络保证：mock 模式已显式锁定 MockSearch；这里再清掉搜索相关环境变量，
    兜住 config 模式测试（空 config 时 get_search_provider 会回退读环境变量）。"""
    for var in ("SEARCH_PROVIDER", "TAVILY_API_KEY", "BOCHA_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def _run(tmp_path: Path, argv: list[str]) -> tuple[int, list[dict], Path]:
    """跑一次脚本 main，读回（退出码, JSONL 行, 输出路径）。"""
    out = tmp_path / "cold_warm.jsonl"
    rc = smoke.main([*argv, "--out", str(out)])
    lines = out.read_text(encoding="utf-8").splitlines()
    return rc, [json.loads(line) for line in lines], out


def _spy_agent_loop(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """把脚本里的 AgentLoop 换成子类，记录每次构造的 kwargs（wiring 证据）。

    子类是同一个真实 AgentLoop（行为逐字段不变），只是把 kwargs 抄一份出来——
    断言"三段配置真的传进去了"必须看构造点，而不是看结果差异（结果差异在
    formation/memory_update 关闭时会完全消失）。
    """
    seen: list[dict[str, Any]] = []
    real_loop = smoke.AgentLoop

    class SpyLoop(real_loop):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            seen.append(dict(kwargs))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(smoke, "AgentLoop", SpyLoop)
    return seen


# ---- mock 模式端到端（真实 AgentLoop 两次 run）--------------------------------


def test_mock_end_to_end_prior_hit_and_token_reconciliation(tmp_path, capsys) -> None:
    wiki_root = tmp_path / "wiki"
    rc, rows, out = _run(tmp_path, ["--provider", "mock", "--wiki-root", str(wiki_root)])

    assert rc == smoke.EXIT_PASS

    # JSONL 恰好两行：cold + warm，字段与 run-metrics 14 字段契约齐全
    assert [row["run"] for row in rows] == ["cold", "warm"]
    expected_keys = {
        "run",
        "question",
        "trace_id",
        "wiki_root",
        "run_dir",
        "provider_mode",
        "model",
        "metrics",
        # P2-G 新增键：记忆形成计数 + run 落盘的记忆更新报告（阶段未执行时 None）
        "formation",
        "memory_update",
    }
    assert all(set(row) == expected_keys for row in rows)
    assert all(set(row["metrics"]) == set(RunMetrics().to_dict()) for row in rows)
    assert all(row["provider_mode"] == "mock" for row in rows)
    assert all(row["question"] == smoke.DEFAULT_QUESTION for row in rows)
    assert all(Path(row["wiki_root"]) == wiki_root for row in rows)
    assert rows[0]["model"] == {
        "strong": {"model": "mock-strong", "base_url": ""},
        "cheap": {"model": "mock-cheap", "base_url": ""},
    }

    # 两次 run 是独立 AgentLoop 实例：独立 trace_id，各自 run-metrics.json 落盘
    assert rows[0]["trace_id"] != rows[1]["trace_id"]
    assert rows[0]["run_dir"] != rows[1]["run_dir"]
    for row in rows:
        assert (Path(row["run_dir"]) / "run-metrics.json").is_file()

    # 阶段验收口径：cold 空 Wiki 不命中；warm 必须命中 cold 沉淀的 active note
    assert rows[0]["metrics"]["prior_hit_count"] == 0
    assert rows[0]["metrics"]["prior_note_ids"] == []
    assert rows[1]["metrics"]["prior_hit_count"] > 0
    assert rows[1]["metrics"]["prior_note_ids"], "warm 命中必须带 note id"
    # cold 沉淀 ≥1 条 active note（warm 能命中的前提），且运行时硬断言全绿
    assert rows[0]["metrics"]["notes_created"] >= 1
    assert smoke.verify_rows(rows) == []

    # 同一份剧本的两次 run 行为一致（完全脚本化、可重复）
    assert rows[0]["metrics"]["fresh_search_count"] == rows[1]["metrics"]["fresh_search_count"]
    assert rows[0]["metrics"]["fresh_search_count"] == 1

    # token 可复算：两次 run 的 metrics 与 tokens.jsonl 按 trace_id 复算完全一致
    for row in rows:
        tok_in, tok_out = sum_tokens_from_jsonl(wiki_root / "tokens.jsonl", row["trace_id"])
        assert tok_in > 0, "对账不应空转"
        assert row["metrics"]["input_tokens"] == tok_in
        assert row["metrics"]["output_tokens"] == tok_out

    # 汇总表可打印：对照字段、差值列、防误读声明、记忆形成/记忆更新计数块
    printed = capsys.readouterr().out
    for field in (
        "fresh_search_count",
        "fresh_fetch_count",
        "source_count",
        "notes_created",
        "input_tokens",
        "output_tokens",
        "latency_ms",
        "citation_coverage",
    ):
        assert field in printed
    assert "差值" in printed
    assert "小样本冒烟" in printed
    assert f"JSONL 已写入 {out}" in printed
    for field in smoke.MEMORY_UPDATE_COUNT_KEYS:
        assert field in printed
    assert "记忆形成 / 记忆更新" in printed
    assert "features    : formation=enabled=True" in printed


def test_mock_mode_supports_custom_question(tmp_path) -> None:
    """自定义问题下剧本同样自洽：笔记正文嵌入问题原文，warm 依然命中。"""
    rc, rows, _out = _run(
        tmp_path,
        ["--provider", "mock", "--question", "上下文压缩的工程实践有哪些"],
    )
    assert rc == smoke.EXIT_PASS
    assert rows[0]["metrics"]["prior_hit_count"] == 0
    assert rows[1]["metrics"]["prior_hit_count"] > 0


def test_mock_mode_locks_mock_search_even_with_env_keys(monkeypatch, tmp_path) -> None:
    """环境里恰好有搜索 key 时，mock 模式仍显式锁定 MockSearch——零网络不靠环境巧合。"""
    monkeypatch.setenv("TAVILY_API_KEY", "smoke-test-not-a-real-key")
    stack = smoke.build_mock_stack(smoke.DEFAULT_QUESTION)
    assert isinstance(stack.search_provider, MockSearch)
    # 端到端仍然通过（不会向真实搜索发起请求）
    rc, _rows, _out = _run(
        tmp_path, ["--provider", "mock", "--wiki-root", str(tmp_path / "wiki")]
    )
    assert rc == smoke.EXIT_PASS


# ---- 记忆演化三段配置：接上了 + 真的生效（P1-B / P2-E / P2-C）------------------


def test_mock_mode_passes_three_memory_configs_to_agent_loop(monkeypatch, tmp_path) -> None:
    """wiring：run_once 把三段配置原样传给 AgentLoop（两个 run 各一次）。

    断言的是**构造点**：三层配置关掉时结果差异会完全消失（loop 的保守默认是
    "None = 零行为变化"），所以"生效"必须先证"传进去了"。
    """
    seen = _spy_agent_loop(monkeypatch)
    rc, rows, _out = _run(
        tmp_path, ["--provider", "mock", "--wiki-root", str(tmp_path / "wiki")]
    )

    assert rc == smoke.EXIT_PASS
    assert len(seen) == 2, "cold / warm 各构造一次 AgentLoop"
    for kwargs in seen:
        assert kwargs["formation_config"] == smoke.MOCK_FORMATION_CONFIG
        assert kwargs["memory_update_config"] == smoke.MOCK_MEMORY_UPDATE_CONFIG
        assert kwargs["verification_config"] == smoke.MOCK_VERIFICATION_CONFIG
    # 三段解析出的状态与脚本打印的一行一致（打印用的就是同一套 src 侧解析函数）
    stack = smoke.build_mock_stack(smoke.DEFAULT_QUESTION)
    for field in ("formation=enabled=True", "memory_update=enabled=True, dry_run=False"):
        assert field in smoke.describe_features(stack)
    assert "verification=已传" in smoke.describe_features(stack)
    assert all(row["memory_update"]["enabled"] is True for row in rows)


def test_mock_mode_formation_gates_ingest_and_annotates_note(tmp_path) -> None:
    """formation 不只是计数：冷启动笔记带 formation_reason/importance；warm 的近乎重复候选被拒。

    同一份剧本下 warm 的候选与 cold 沉淀的笔记正文完全相同 → 最大相似度 ≈ 1.0
    ≥ near_duplicate_similarity(0.95) → 拒绝入库。这是判定的预期语义（防无差别
    堆积），不是失败；代价是 warm 的"新证据"为空、memory_update 无动作可比
    （故 counts 全 0，动作语义由 tests/test_memory_update.py 覆盖）。
    """
    wiki_root = tmp_path / "wiki"
    rc, rows, _out = _run(tmp_path, ["--provider", "mock", "--wiki-root", str(wiki_root)])

    assert rc == smoke.EXIT_PASS
    cold, warm = rows
    assert cold["formation"] == {"candidates": 1, "persisted": 1, "rejected": 0}
    assert warm["formation"] == {"candidates": 1, "persisted": 0, "rejected": 1}

    # 判定结果落到了笔记 frontmatter（只判过才写：接受理由 + importance + kind）
    notes = WikiStore(wiki_root).list_notes()
    assert len(notes) == 1
    meta = notes[0].meta
    assert str(meta.extra.get("formation_reason", "")).startswith("接受：")
    assert meta.extra.get("formation_confidence") == "high"
    assert meta.importance is not None and meta.importance > 0.0
    assert meta.kind == "knowledge"
    assert meta.sources and meta.sources[0].url == smoke.MOCK_SOURCE_URL

    # warm 的候选被拒 → 没有第二条笔记（也没有新的 data-note 落库）
    assert warm["metrics"]["notes_created"] == 0
    assert warm["metrics"]["notes_merged"] == 0


def test_mock_mode_memory_update_report_is_readable_and_enabled(tmp_path) -> None:
    """memory_update 真的跑了：run_dir 有 memory-update.json，state.md 有记忆更新行。

    warm run 的报告必须能看到它命中的 Prior（prior_hit_count/prior_note_ids 非空）
    ——"阶段执行了并且看见了旧记忆"是这一步能给出的最强可观察证据；因为 warm 的
    候选被 formation 拒（近乎重复），evidence_count 为 0，counts 全 0（见上一个
    测试的说明），所以这里不断言任何写盘动作。
    """
    wiki_root = tmp_path / "wiki"
    rc, rows, _out = _run(tmp_path, ["--provider", "mock", "--wiki-root", str(wiki_root)])

    assert rc == smoke.EXIT_PASS
    cold, warm = rows
    for row in rows:
        payload = row["memory_update"]
        assert isinstance(payload, dict), "run_dir/memory-update.json 必须被读回"
        assert payload["trace_id"] == row["trace_id"]
        assert payload["enabled"] is True and payload["dry_run"] is False
        assert set(payload["counts"]) == set(smoke.MEMORY_UPDATE_COUNT_KEYS)
        # JSONL 里的对象就是盘子上的那份报告（落盘契约，不是脚本另算的副本）
        on_disk = json.loads(
            (Path(row["run_dir"]) / "memory-update.json").read_text(encoding="utf-8")
        )
        assert on_disk == payload

    assert cold["memory_update"]["prior_hit_count"] == 0
    assert warm["memory_update"]["prior_hit_count"] > 0
    assert warm["memory_update"]["prior_note_ids"]
    assert warm["memory_update"]["evidence_count"] == 0
    assert warm["memory_update"]["counts"] == dict.fromkeys(
        smoke.MEMORY_UPDATE_COUNT_KEYS, 0
    )

    # state.md：记忆形成行恒在；记忆更新行只在阶段启用时出现
    cold_state = (Path(cold["run_dir"]) / "state.md").read_text(encoding="utf-8")
    warm_state = (Path(warm["run_dir"]) / "state.md").read_text(encoding="utf-8")
    assert "记忆形成：候选 1，入库 1，拒绝 0" in cold_state
    assert "记忆更新：复核 0，替代 0，合并 0，冲突 0，跳过 0" in warm_state


def test_no_memory_update_flag_skips_stage_entirely(tmp_path, capsys) -> None:
    """逃生开关：--no-memory-update = 不传该段（整阶段不执行、逐字段零行为变化）。

    P1 的四道硬闸不受影响；memory-update.json 不落盘、state.md 没有记忆更新行、
    JSONL 里 memory_update 为 None（"没跑"与"跑了但零动作"必须可区分）。
    """
    wiki_root = tmp_path / "wiki"
    rc, rows, _out = _run(
        tmp_path,
        ["--provider", "mock", "--no-memory-update", "--wiki-root", str(wiki_root)],
    )

    assert rc == smoke.EXIT_PASS
    assert smoke.verify_rows(rows) == []
    for row in rows:
        assert row["memory_update"] is None
        assert not (Path(row["run_dir"]) / "memory-update.json").exists()
        state = (Path(row["run_dir"]) / "state.md").read_text(encoding="utf-8")
        assert "记忆更新" not in state
        # formation 不受逃生开关影响（照旧判定与计数）
        assert row["formation"]["candidates"] == 1
    # 汇总块里两行的 memory_update 列都是占位符（"没跑" != "跑了但零动作"）
    data_lines = [
        line
        for line in smoke.format_feature_summary(rows).splitlines()
        if line.strip().startswith(("cold", "warm"))
    ]
    assert len(data_lines) == 2 and all("—" in line for line in data_lines)
    assert "memory_update=未传（整阶段不执行）" in capsys.readouterr().out

def test_no_memory_update_flag_passes_none_to_agent_loop(monkeypatch, tmp_path) -> None:
    """逃生开关的 wiring：memory_update_config 是 None（loop 的"未配置"分支），
    另两段照旧传（逃生开关只关一个阶段，不整体降级）。"""
    seen = _spy_agent_loop(monkeypatch)
    rc, _rows, _out = _run(
        tmp_path,
        ["--provider", "mock", "--no-memory-update", "--wiki-root", str(tmp_path / "wiki")],
    )

    assert rc == smoke.EXIT_PASS
    assert len(seen) == 2
    assert all(kwargs["memory_update_config"] is None for kwargs in seen)
    assert all(kwargs["formation_config"] == smoke.MOCK_FORMATION_CONFIG for kwargs in seen)
    assert all(
        kwargs["verification_config"] == smoke.MOCK_VERIFICATION_CONFIG for kwargs in seen
    )


def test_config_mode_passes_sections_and_reports_feature_state(
    monkeypatch, tmp_path, capsys
) -> None:
    """config 模式按 config.toml 的段落传三段（传法同 server）+ 打印解析后的状态。

    模型不可用时（无 key / 空 base_url）研究链路照旧跑完但断言失败——本测试只
    关心配置接线：告警行、features 行、以及 spy 到的 kwargs 必须来自 temp config。
    """
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "[verification]\n"
        "similarity_floor = 0.45\n"
        "consistent_similarity = 0.85\n"
        "supersede_min_similarity = 0.65\n"
        "[formation]\n"
        "enabled = false\n"
        "[memory_update]\n"
        "enabled = true\n"
        "dry_run = true\n",
        encoding="utf-8",
    )
    seen = _spy_agent_loop(monkeypatch)
    rc, _rows, _out = _run(
        tmp_path,
        [
            "--provider",
            "config",
            "--config",
            str(config_path),
            "--env-file",
            "",
            "--wiki-root",
            str(tmp_path / "wiki"),
        ],
    )

    assert rc == smoke.EXIT_ASSERT_FAILED  # 无可用模型 → warm 命中 0，明确报失败
    assert seen and seen[0]["verification_config"] == {
        "similarity_floor": 0.45,
        "consistent_similarity": 0.85,
        "supersede_min_similarity": 0.65,
    }
    assert seen[0]["formation_config"] == {"enabled": False}
    assert seen[0]["memory_update_config"] == {"enabled": True, "dry_run": True}
    printed = capsys.readouterr().out
    assert "成本警告" in printed
    assert "formation=enabled=False" in printed
    assert "memory_update=enabled=True, dry_run=True" in printed
    assert "verification=已传（floor=0.45, consistent=0.85, supersede_min=0.65）" in printed




def test_config_mode_fails_loudly_without_usable_models(tmp_path, capsys) -> None:
    """config 读不到 → MockProvider 占位回复 → 蒸馏无产出 → warm prior 命中 0 → 退出码 1。

    同时验证：config 模式开始前打印成本警告、运行本身零网络（MockSearch 兜底）。
    """
    rc, rows, _out = _run(
        tmp_path,
        [
            "--provider",
            "config",
            "--config",
            str(tmp_path / "missing.toml"),
            "--env-file",
            "",  # 不读项目 .env（测试零 key 依赖）
            "--wiki-root",
            str(tmp_path / "wiki"),
        ],
    )
    assert rc == smoke.EXIT_ASSERT_FAILED
    assert [row["run"] for row in rows] == ["cold", "warm"]
    assert rows[1]["metrics"]["prior_hit_count"] == 0
    printed = capsys.readouterr().out
    assert "成本警告" in printed
    assert "断言失败" in printed and "prior_hit_count" in printed


# ---- 硬断言与汇总表：纯函数 ---------------------------------------------------


def test_verify_rows_flags_zero_prior_and_token_mismatch(tmp_path) -> None:
    (tmp_path / "tokens.jsonl").write_text(
        json.dumps({"trace_id": "t1", "input_tokens": 100, "output_tokens": 10}) + "\n",
        encoding="utf-8",
    )
    rows = [
        {"run": "cold", "trace_id": "t1", "wiki_root": str(tmp_path),
         "metrics": {"input_tokens": 100, "output_tokens": 10, "prior_hit_count": 0,
                     "notes_created": 1}},
        {"run": "warm", "trace_id": "t2", "wiki_root": str(tmp_path),
         "metrics": {"input_tokens": 5, "output_tokens": 5, "prior_hit_count": 0,
                     "notes_created": 0}},
    ]
    failures = smoke.verify_rows(rows)
    assert any("prior_hit_count" in failure for failure in failures)
    # 只有对不上的 warm 行被点名；cold 行与 tokens.jsonl 一致，不进失败列表
    mismatches = [failure for failure in failures if "对账失败" in failure]
    assert len(mismatches) == 1 and "warm" in mismatches[0]

    # warm 命中后仅剩的问题是对账失败
    rows[1]["metrics"]["prior_hit_count"] = 2
    assert all("prior_hit_count" not in failure for failure in smoke.verify_rows(rows))


def test_verify_rows_flags_dirty_cold_start(tmp_path) -> None:
    """cold 闸：--wiki-root 已含相关笔记（cold 直接命中）或蒸馏零产出时必须拦住假绿对照。"""
    rows = [
        {"run": "cold", "trace_id": "t1", "wiki_root": str(tmp_path),
         "metrics": {"input_tokens": 0, "output_tokens": 0, "prior_hit_count": 1,
                     "notes_created": 0}},
        {"run": "warm", "trace_id": "t2", "wiki_root": str(tmp_path),
         "metrics": {"input_tokens": 0, "output_tokens": 0, "prior_hit_count": 1,
                     "notes_created": 0}},
    ]
    failures = smoke.verify_rows(rows)
    assert any(
        "cold run prior_hit_count=1" in failure and "假绿" in failure
        for failure in failures
    )
    assert any("cold run notes_created=0" in failure for failure in failures)

    # 修正 cold 口径后这两条失败消失（warm 命中 1 保持合法）
    rows[0]["metrics"]["prior_hit_count"] = 0
    rows[0]["metrics"]["notes_created"] = 1
    remaining = smoke.verify_rows(rows)
    assert not any("cold run" in failure for failure in remaining)


def test_verify_rows_flags_missing_warm_row() -> None:
    failures = smoke.verify_rows([{"run": "cold", "trace_id": "t", "wiki_root": "x",
                                   "metrics": {}}])
    assert any("缺少 warm run" in failure for failure in failures)


def test_format_summary_shows_compare_fields_delta_and_disclaimer() -> None:
    rows = [
        {"run": "cold", "metrics": {"prior_hit_count": 0, "fresh_search_count": 1,
                                    "fresh_fetch_count": 0, "source_count": 3,
                                    "notes_created": 1, "input_tokens": 100,
                                    "output_tokens": 20, "latency_ms": 5,
                                    "citation_coverage": 0.3333}},
        {"run": "warm", "metrics": {"prior_hit_count": 1, "fresh_search_count": 1,
                                    "fresh_fetch_count": 0, "source_count": 3,
                                    "notes_created": 0, "input_tokens": 90,
                                    "output_tokens": 20, "latency_ms": 4,
                                    "citation_coverage": None}},
    ]
    text = smoke.format_summary(rows)
    assert "cold" in text and "warm" in text and "差值" in text
    assert "小样本冒烟，不构成收益结论" in text
    # citation_coverage 一侧为 None 时差值显示占位符，不硬算
    assert "—" in text
    for field in smoke.COMPARE_FIELDS:
        assert field in text
