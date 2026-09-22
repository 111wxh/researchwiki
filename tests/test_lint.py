"""lint 单测：引用覆盖率、断链检出、merged 跟随、孤立笔记、时效统计、CLI 退出码与 --json。

零网络、tmp_path 隔离；CLI 直接调 main() 并断言返回码（capsys 校验中文报告/JOSN）。
时效统计用注入的固定时钟（FIXED_NOW），断言精确到数值。
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from researchwiki.cli import main
from researchwiki.mcp_server.service import WikiService
from researchwiki.wiki.entities import EntityRegistry
from researchwiki.wiki.freshness import FreshnessSettings
from researchwiki.wiki.frontmatter import SourceRef
from researchwiki.wiki.index import SearchIndex
from researchwiki.wiki.lint import (
    LintReport,
    is_assertion_line,
    iter_assertion_lines,
    lint_wiki,
    referenced_note_ids,
)
from researchwiki.wiki.store import WikiStore

FIXED_NOW = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)


def make_store(tmp_path: Path) -> WikiStore:
    return WikiStore(tmp_path / "wiki-data")


# ---- 纯函数：断言行定义 ------------------------------------------------------


def test_is_assertion_line_excludes_heading_link_and_fence():
    assert is_assertion_line("Letta 用后台子 agent 整理记忆。")
    assert is_assertion_line("- 支持 1M 上下文（N-0001）。")
    assert is_assertion_line("| 模型 | 窗口 |")  # 表格行算内容行
    assert not is_assertion_line("")
    assert not is_assertion_line("   ")
    assert not is_assertion_line("## 小节标题")
    assert not is_assertion_line("```")
    assert not is_assertion_line("- [[entity:letta|Letta]]")
    assert not is_assertion_line("- [[entity:letta|Letta]]、[[entity:mem0|Mem0]]")
    assert not is_assertion_line("---")
    # 「纯链接 + 说明文字」是断言行（说明文字需要标注）
    assert is_assertion_line("- [[entity:letta|Letta]]：后台整理记忆。")


def test_iter_assertion_lines_skips_code_fences():
    body = "# T\n\n断言一。\n\n```python\nprint('nope')\n```\n\n断言二。\n"
    lines = [text for _, text in iter_assertion_lines(body)]
    assert lines == ["断言一。", "断言二。"]
    assert [lineno for lineno, _ in iter_assertion_lines(body)] == [3, 9]


def test_referenced_note_ids_dedupes_in_order():
    assert referenced_note_ids("见 N-0002 与 N-0001，另见 N-0002。") == ["N-0002", "N-0001"]
    assert referenced_note_ids("没有引用") == []


# ---- 指标 --------------------------------------------------------------------


def test_citation_coverage_and_orphans(tmp_path: Path):
    store = make_store(tmp_path)
    store.save_note("笔记一", entities=["实体A"])  # N-0001
    store.save_note("笔记二", entities=["实体B"])  # N-0002
    store.save_page("实体a", "实体A", "# 实体A\n\n断言一（N-0001）。\n断言二缺标注。\n")

    report = lint_wiki(store)
    assert report.notes_total == 2 and report.pages_total == 1
    assert report.citation_coverage == 0.5
    assert report.details["assertion_lines"] == 2 and report.details["cited_lines"] == 1
    assert report.details["uncited_lines"][0]["text"] == "断言二缺标注。"
    assert report.orphan_notes == ["N-0002"]
    assert report.broken_links == [] and report.merged_chains == []
    assert report.exit_code() == 0 and report.healthy
    text = report.format_text(root="wiki-data")
    assert "引用覆盖：50.0%" in text and "孤立笔记" in text and "N-0002" in text


def test_broken_entity_and_note_links(tmp_path: Path):
    store = make_store(tmp_path)
    store.save_note("笔记", note_id="N-0001", entities=["实体A"])
    EntityRegistry(store.root).get_or_create("实体A")
    store.save_page(
        "实体a",
        "实体A",
        "# 实体A\n\n[[entity:幽灵实体|幽灵]] 断言（N-0001）。\n引用不存在的 N-0099。\n",
    )

    report = lint_wiki(store)
    kinds = {(link.kind, link.target) for link in report.broken_links}
    assert ("entity", "[[entity:幽灵实体]]") in kinds
    assert ("note", "N-0099") in kinds
    assert report.exit_code() == 1 and not report.healthy
    assert "断链：2 处" in report.format_text()


def test_merged_reference_followed_and_recorded(tmp_path: Path):
    store = make_store(tmp_path)
    canonical = store.save_note("规范表述", note_id="N-0001", entities=["实体A"])
    store.save_note(
        "旧表述", note_id="N-0002", status="merged", redirect_to=canonical.id
    )
    store.save_note("孤零零的笔记", note_id="N-0003")
    EntityRegistry(store.root).get_or_create("实体A")
    store.save_page("实体a", "实体A", "# 实体A\n\n断言（N-0002）。\n")

    report = lint_wiki(store)
    # 引用 merged 笔记：跟随 redirect 通过，但提示改写成规范 ID
    assert report.broken_links == []
    assert [(ref.source, ref.referenced, ref.canonical) for ref in report.merged_chains] == [
        ("page:实体a", "N-0002", "N-0001")
    ]
    assert report.orphan_notes == ["N-0003"]
    assert report.exit_code() == 0
    assert "N-0002 → 规范 ID N-0001" in report.format_text()


def test_broken_redirect_chain_is_reported(tmp_path: Path):
    store = make_store(tmp_path)
    store.save_note("断链笔记", note_id="N-0001", status="merged", redirect_to="N-9999")
    store.save_page("p", "P", "# P\n\n断言（N-0001）。\n")

    report = lint_wiki(store)
    assert len(report.broken_links) == 1
    assert report.broken_links[0].kind == "note"
    assert "redirect 链断裂" in report.broken_links[0].message
    assert report.exit_code() == 1


def test_redirect_cycle_is_reported(tmp_path: Path):
    store = make_store(tmp_path)
    store.save_note("环一", note_id="N-0001", status="merged", redirect_to="N-0002")
    store.save_note("环二", note_id="N-0002", status="merged", redirect_to="N-0001")
    store.save_page("p", "P", "# P\n\n断言（N-0001）。\n")

    report = lint_wiki(store)
    assert [link.target for link in report.broken_links] == ["N-0001"]
    assert "redirect 环路" in report.broken_links[0].message


def test_note_body_references_are_checked(tmp_path: Path):
    store = make_store(tmp_path)
    store.save_note("引用不存在笔记 N-0042", note_id="N-0001")
    report = lint_wiki(store)
    assert [(link.source, link.target) for link in report.broken_links] == [
        ("note:N-0001", "N-0042")
    ]


def test_zero_coverage_fails(tmp_path: Path):
    store = make_store(tmp_path)
    store.save_note("笔记", note_id="N-0001")
    store.save_page("p", "P", "# P\n\n这条断言没有标注。\n")

    report = lint_wiki(store)
    assert report.citation_coverage == 0.0
    assert report.exit_code() == 1
    assert "不健康" in report.format_text()


def test_empty_wiki_is_healthy(tmp_path: Path):
    report = lint_wiki(make_store(tmp_path))
    assert report.notes_total == 0 and report.pages_total == 0
    assert report.orphan_notes == [] and report.broken_links == []
    assert report.merged_chains == [] and report.details["assertion_lines"] == 0
    assert report.citation_coverage == 1.0  # 无断言行 = 真空真，空库不该把 CI 判红
    assert report.exit_code() == 0
    assert report == LintReport(citation_coverage=1.0, details=report.details)
    assert "wiki 健康度报告" in report.format_text()
    # P2 新增观测字段：空库恒为零，且不影响既有等值断言（字段有默认值）
    assert report.freshness == {"fresh": 0, "review_due": 0, "stale": 0}
    assert report.conflicts_open == 0


# ---- 时效统计（P2 freshness）-------------------------------------------------


def make_freshness_store(tmp_path: Path) -> WikiStore:
    """四条 active 笔记覆盖三类状态 + 一条 merged（不应计入时效统计）+ 一条未裁决冲突。"""
    store = make_store(tmp_path)
    old = (FIXED_NOW - timedelta(days=60)).isoformat()
    drifted = (FIXED_NOW - timedelta(days=90)).isoformat()
    recent = (FIXED_NOW - timedelta(days=1)).isoformat()
    store.save_note("稳定断言。", note_id="N-0001", volatility="stable",
                    observed_at=old, created=old)
    store.save_note("新鲜断言。", note_id="N-0002", volatility="volatile",
                    observed_at=recent, created=recent)
    store.save_note("陈旧断言。", note_id="N-0003", volatility="volatile",
                    observed_at=old, created=old)
    store.save_note("待复核断言。", note_id="N-0004", volatility="drifting",
                    observed_at=drifted, created=drifted)
    store.save_note("已合并记录。", note_id="N-0005", status="merged", redirect_to="N-0001",
                    volatility="volatile", observed_at=old, created=old)
    store.save_conflict("矛盾 A", {"note_id": "N-0001"}, {"note_id": "N-0003"})
    return store


def test_freshness_stats_conflicts_and_queue(tmp_path: Path):
    store = make_freshness_store(tmp_path)
    report = lint_wiki(store, now=FIXED_NOW)

    # 只统计 active 笔记：merged 的 N-0005（同样陈旧）不计入
    assert report.notes_total == 4
    assert report.details["records_total"] == 5
    assert report.freshness == {"fresh": 2, "review_due": 1, "stale": 1}
    assert report.conflicts_open == 1
    # 队列：stale 在前、同级按年龄降序
    assert [item["note_id"] for item in report.details["freshness_queue"]] == ["N-0003", "N-0004"]
    assert report.details["freshness_queue_total"] == 2
    first = report.details["freshness_queue"][0]
    assert first["state"] == "stale" and first["age_days"] == pytest.approx(60.0)
    assert first["decay"] == pytest.approx(0.25)
    # 陈旧是"该复核"不是"不健康"：退出码契约不变
    assert report.exit_code() == 0 and report.healthy
    payload = report.to_dict()
    assert payload["freshness"] == {"fresh": 2, "review_due": 1, "stale": 1}
    assert payload["conflicts_open"] == 1
    assert payload["exit_code"] == 0


def test_freshness_text_summary_lines(tmp_path: Path):
    report = lint_wiki(make_freshness_store(tmp_path), now=FIXED_NOW)
    text = report.format_text(root=str(tmp_path))
    assert "- 时效（active）：fresh 2 · review_due 1 · stale 1" in text
    assert "待复核：N-0003（stale，年龄 60.0 天，decay 0.250）" in text
    assert "待复核：N-0004（review_due，年龄 90.0 天，decay 0.500）" in text
    assert "- 未裁决冲突：1 条" in text
    # P2-F 健康度摘要行（既有"未裁决冲突：N 条"文案保留为前缀，追加已裁决计数）
    assert "- 未裁决冲突：1 条（已裁决 0 条）" in text
    assert "- 墓碑（失效裁定记录，默认不参与检索）：0 条" in text
    assert "- 悬空证据：0 处" in text
    assert "- 待复核来源变化：0 条" in text


# ---- 健康度补齐（P2-F 裁定三：PLAN §3.3 验收第 5 条）---------------------------


def make_health_store(tmp_path: Path) -> WikiStore:
    """一份把四个新计数都点亮的库：墓碑 / 悬空证据 / 来源变化待复核 / 冲突两态。

    - N-0001：active、来源快照存在（sources/{sha1}/{hash}/content.md 真落盘）；
    - N-0002：active、引用了不存在的快照（悬空证据）且 source_changed_at 晚于
      reviewed_at（待复核队列）；另引用一个空 content_hash（同样算悬空，但同一
      笔记同一 path 只计一次 → N-0002 共 2 处悬空）；
    - N-0003：active、墓碑（tombstone=True，审计记录）；
    - N-0004：已裁决冲突的载体笔记；
    - 冲突：一条 open + 一条 resolved。
    """
    store = make_store(tmp_path)
    good = store.save_note(
        "证据完整的断言（N-0003）。",
        note_id="N-0001",
        title="有快照的记忆",
        sources=[SourceRef(url="https://example.com/ok", content_hash="a" * 64)],
    )
    snapshot = store.note_snapshot_paths(good)[0]
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_text("快照正文", encoding="utf-8")
    store.save_note(
        "来源已变化、尚未复核的断言。",
        note_id="N-0002",
        title="待复核记忆",
        reviewed_at="2026-05-01T00:00:00+00:00",
        source_changed_at="2026-05-02T00:00:00+00:00",
        sources=[
            SourceRef(url="https://example.com/gone", content_hash="b" * 64),
            SourceRef(url="https://example.com/nohash", content_hash=""),
        ],
    )
    store.save_note(
        "记忆 N-0001 已被裁定失效：结论不成立。",
        note_id="N-0003",
        title="[已失效] 有快照的记忆",
        tombstone=True,
    )
    store.save_note("冲突的另一方。", note_id="N-0004", title="另一方")
    open_conflict = store.save_conflict("哪个对？", {"note_id": "N-0001"}, {"note_id": "N-0004"})
    resolved = store.save_conflict("昨天的问题", {"note_id": "N-0001"}, {"note_id": "N-0004"})
    store.resolve_conflict(resolved.id, verdict="采纳 N-0001", resolved_with="N-0001")
    assert open_conflict.status == "open"
    return store


def test_health_metrics_counted_from_constructed_data(tmp_path: Path):
    store = make_health_store(tmp_path)
    report = lint_wiki(store, now=FIXED_NOW)

    assert report.conflicts == {"open": 1, "resolved": 1}
    assert report.conflicts_open == 1  # 既有字段语义不变（同源同值）
    assert report.tombstones == 1
    assert report.dangling_evidence == 2  # N-0002 的两处来源都没快照
    assert report.source_changed_pending == 1  # 只有 N-0002（N-0001 没被标记过）
    assert report.details["tombstone_note_ids"] == ["N-0003"]
    assert report.details["dangling_evidence_total"] == 2
    # 观测字段不影响结论与退出码（NEW: 全部是"该复核/该清理"，不是"wiki 坏了"）
    assert report.exit_code() == 0 and report.healthy
    payload = report.to_dict()
    assert payload["conflicts"] == {"open": 1, "resolved": 1}
    assert payload["tombstones"] == 1
    assert payload["dangling_evidence"] == 2
    assert payload["source_changed_pending"] == 1
    text = report.format_text()
    assert "- 墓碑（失效裁定记录，默认不参与检索）：1 条" in text
    assert "- 悬空证据：2 处" in text
    assert "- 待复核来源变化：1 条" in text


def test_health_metrics_are_zero_on_empty_wiki(tmp_path: Path):
    """空库 / 目录不存在：四个计数全 0 且不报错（新增字段必须有缺省值）。"""
    for root in (tmp_path / "wiki-data", tmp_path / "nope"):
        report = lint_wiki(WikiStore(root), now=FIXED_NOW)
        assert report.conflicts == {"open": 0, "resolved": 0}
        assert report.tombstones == 0
        assert report.dangling_evidence == 0
        assert report.source_changed_pending == 0
        assert report.exit_code() == 0


def test_health_metrics_ignore_unparsable_source_change(tmp_path: Path):
    """不可解析的 source_changed_at 按"未声明"处理（与 freshness 规则 5 同口径）。"""
    store = make_store(tmp_path)
    store.save_note(
        "坏时间戳。", note_id="N-0001", title="坏时间戳", source_changed_at="昨天下午"
    )
    report = lint_wiki(store, now=FIXED_NOW)
    assert report.source_changed_pending == 0


def test_tombstone_not_counted_as_source_change_or_dangling(tmp_path: Path):
    """墓碑没有 sources：不会污染悬空证据 / 待复核来源变化计数。"""
    store = make_store(tmp_path)
    store.save_note("记忆 N-0001 已失效。", note_id="N-0001", title="[已失效] x", tombstone=True)
    report = lint_wiki(store, now=FIXED_NOW)
    assert report.tombstones == 1
    assert report.dangling_evidence == 0
    assert report.source_changed_pending == 0


def test_freshness_settings_injection(tmp_path: Path):
    """freshness_settings 可注入（如 config 的 [freshness] 段）：判定随之改变。"""
    store = make_store(tmp_path)
    observed = (FIXED_NOW - timedelta(days=10)).isoformat()
    store.save_note("断言。", note_id="N-0001", volatility="volatile",
                    observed_at=observed, created=observed)
    assert lint_wiki(store, now=FIXED_NOW).freshness == {
        "fresh": 1, "review_due": 0, "stale": 0
    }
    strict = FreshnessSettings(review_due_ratio=0.95, stale_ratio=0.8)
    report = lint_wiki(store, now=FIXED_NOW, freshness_settings=strict)
    assert report.freshness == {"fresh": 0, "review_due": 0, "stale": 1}


def test_missing_root_is_healthy(tmp_path: Path):
    report = lint_wiki(WikiStore(tmp_path / "nope"))
    assert report.exit_code() == 0 and report.citation_coverage == 1.0


def test_healthy_wiki_reports_no_orphans(tmp_path: Path):
    store = make_store(tmp_path)
    store.save_note("断言一", note_id="N-0001", entities=["实体A"])
    EntityRegistry(store.root).get_or_create("实体A")
    store.save_page(
        "实体a",
        "实体A",
        "# 实体A\n\n## 关键事实\n\n断言一（N-0001）。\n\n## 相关实体\n\n- [[entity:实体a|实体A]]\n",
    )
    report = lint_wiki(store)
    assert report.citation_coverage == 1.0
    assert report.orphan_notes == [] and report.broken_links == []
    assert report.exit_code() == 0


def test_index_param_reports_stale_ids(tmp_path: Path):
    store = make_store(tmp_path)
    store.save_note("已索引", note_id="N-0001")
    with SearchIndex(store.root, tokenizer="trigram") as index:
        index.rebuild(store)
        store.save_note("索引之后新增", note_id="N-0002")  # 索引落后于 md
        report = lint_wiki(store, index=index)
        assert report.details["index_checked"] is True
        assert report.details["index_stale"] == ["N-0002"]
        assert "新增" in report.details["index_stale_reason"]
    # 不传 index 时跳过这项附加检查（不影响其它指标）
    assert "index_checked" not in lint_wiki(store).details


def test_index_stale_agrees_with_shared_judgement_on_metadata_drift(tmp_path: Path):
    """P2-F 修复轮 I-3：只改元数据（id 集不变）时 lint 与 health 必须同一结论。

    v1 的 lint 判据只比"store 有、索引没有"的 id 集，于是只改 kind 的漂移下
    ``health.index.stale=true`` 而 lint 的 ``details.index_stale=[]``——同一系统出现
    第三个答案。现在两边都走 ``index.index_drift_analysis``。
    """
    store = make_store(tmp_path)
    body = "观测记录\n旧机器：量子退火炉的初代读数甲。"
    store.save_note(body, note_id="N-0001", title="观测记录", kind="knowledge")
    with SearchIndex(store.root, tokenizer="trigram") as index:
        index.rebuild(store)
        # 只改 kind：正文/标题/status/id 集合全不变（旧口径下正是漏报的形态）
        store.save_note(body, note_id="N-0001", title="观测记录", kind="user")
        report = lint_wiki(store, index=index)
        assert report.details["index_checked"] is True
        assert report.details["index_stale"] == ["N-0001"]  # 不再是 []
        assert "索引字段变化" in report.details["index_stale_reason"]
        # lint 只读：不 rebuild（索引照旧落后）、不动结论与退出码
        assert report.exit_code() == 0 and report.healthy

    config = {"wiki": {"fts_tokenizer": "trigram"}}
    service = WikiService(store.root, config=config)
    index_info = service.health()["index"]
    assert index_info["stale"] is True  # 与上面的 lint 结论一致
    assert "索引字段变化" in index_info["stale_reason"]
    # 索引被 rebuild 之后两边都判新鲜（同一判据、同一方向）
    with SearchIndex(store.root, tokenizer="trigram") as index:
        index.rebuild(store)
        assert lint_wiki(store, index=index).details["index_stale"] == []
    assert service.health()["index"]["stale"] is False


def test_index_lag_skips_broken_or_foreign_index(tmp_path: Path):
    """索引不可用（替身缺接口 / 查询报错）时跳过附加检查，绝不让 lint 失败。"""
    store = make_store(tmp_path)
    store.save_note("断言", note_id="N-0001")

    class NoInterface:
        pass

    class Exploding:
        def indexed_fingerprints(self) -> dict[str, tuple[str, str]]:
            raise RuntimeError("索引坏了")

    for broken in (NoInterface(), Exploding()):
        report = lint_wiki(store, index=broken)  # type: ignore[arg-type]
        assert "index_checked" not in report.details
        assert report.exit_code() == 0


# ---- CLI ---------------------------------------------------------------------


def test_cli_lint_exit_code_and_text_output(tmp_path: Path, capsys):
    store = make_store(tmp_path)
    store.save_note("笔记", note_id="N-0001")
    store.save_page("p", "P", "# P\n\n断言（N-0001）。\n")

    assert main(["lint", "--root", str(tmp_path / "wiki-data")]) == 0
    out = capsys.readouterr().out
    assert "wiki 健康度报告" in out and "引用覆盖：100.0%" in out and "退出码 0" in out

    # 断链 → 退出码 1
    store.save_page("p", "P", "# P\n\n[[entity:不存在|X]] 断言（N-0099）。\n")
    assert main(["lint", "--root", str(tmp_path / "wiki-data")]) == 1


def test_cli_lint_json_output(tmp_path: Path, capsys):
    store = make_store(tmp_path)
    store.save_note("笔记", note_id="N-0001")
    store.save_page("p", "P", "# P\n\n断言缺标注。\n")

    assert main(["lint", "--root", str(tmp_path / "wiki-data"), "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["citation_coverage"] == 0.0
    assert payload["exit_code"] == 1 and payload["healthy"] is False
    assert payload["notes_total"] == 1 and payload["pages_total"] == 1
    assert payload["orphan_notes"] == ["N-0001"]


def test_cli_lint_json_includes_freshness_and_conflicts(tmp_path: Path, capsys):
    """CLI --json 暴露 freshness 分布与 conflicts_open；退出码不受时效影响。"""
    store = make_store(tmp_path)
    old = "2020-01-01T00:00:00+00:00"  # 远早于任何合理"现在"
    store.save_note("陈旧。", note_id="N-0001", volatility="volatile",
                    observed_at=old, created=old)
    store.save_note("稳定。", note_id="N-0002", volatility="stable",
                    observed_at=old, created=old)
    store.save_note("引用 N-0001 与 N-0002。", note_id="N-0003")
    store.save_page("p", "P", "# P\n\n断言（N-0001、N-0002）。\n")
    store.save_conflict("矛盾", {"note_id": "N-0001"}, {"note_id": "N-0002"})

    assert main(["lint", "--root", str(tmp_path / "wiki-data"), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["freshness"] == {"fresh": 2, "review_due": 0, "stale": 1}
    assert payload["conflicts_open"] == 1
    assert payload["exit_code"] == 0  # 陈旧不判红
    assert payload["details"]["freshness_queue"][0]["note_id"] == "N-0001"


def test_cli_lint_missing_root_is_healthy(tmp_path: Path, capsys):
    assert main(["lint", "--root", str(tmp_path / "nope")]) == 0
    assert "笔记：0 条" in capsys.readouterr().out


def test_cli_lint_json_includes_health_metrics(tmp_path: Path, capsys):
    """P2-F 裁定三：``lint --json`` 补齐 conflicts / tombstones / 两个队列长度。"""
    store = make_health_store(tmp_path)
    assert main(["lint", "--root", str(tmp_path / "wiki-data"), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["conflicts"] == {"open": 1, "resolved": 1}
    assert payload["conflicts_open"] == 1  # 既有字段保留（追加式扩展）
    assert payload["tombstones"] == 1
    assert payload["dangling_evidence"] == 2
    assert payload["source_changed_pending"] == 1
    assert payload["exit_code"] == 0  # 新字段全是观测项，不进退出码
    assert store.list_notes(status=None)  # 夹具真的建了库（防"空库恰好也全 0"）


def _lint_json(root: Path, capsys) -> dict:
    """跑一次 CLI lint --json 并解析输出（配置文件由调用方经 CONFIG_PATH 注入）。"""
    assert main(["lint", "--root", str(root), "--json"]) == 0
    return json.loads(capsys.readouterr().out)


def test_cli_lint_reads_freshness_and_wiki_config(tmp_path: Path, capsys, monkeypatch):
    """lint 读 [freshness] + [wiki] 段注入时效参数；缺配置时行为与默认一致。

    夹具取"年龄 1 天 + volatile"：默认参数下 decay≈0.977 判 fresh，收窄阈值后
    判 stale——同一批笔记、只换配置，即可证明参数确实从 config.toml 接了进来。
    """
    from researchwiki import cli

    store = make_store(tmp_path)
    now = datetime.now(UTC)
    observed = (now - timedelta(days=1)).isoformat()
    store.save_note("时效断言。", note_id="N-0001", volatility="volatile",
                    observed_at=observed, created=observed)
    root = tmp_path / "wiki-data"

    # ① 无配置文件（缺省路径不存在）→ 全默认参数，判定与接线前一致
    monkeypatch.setattr(cli, "CONFIG_PATH", tmp_path / "no-such-config.toml")
    assert _lint_json(root, capsys)["freshness"] == {"fresh": 1, "review_due": 0, "stale": 0}

    # ② [freshness] 段（阈值收窄到 decay 也算过期）→ 同一笔记改判 stale
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[freshness]\nreview_due_ratio = 0.99\nstale_ratio = 0.98\n', encoding="utf-8"
    )
    monkeypatch.setattr(cli, "CONFIG_PATH", config_path)
    assert _lint_json(root, capsys)["freshness"] == {"fresh": 0, "review_due": 0, "stale": 1}

    # ③ 只写 [wiki].half_life_days 时也生效（与检索层同一张表：半衰期回退读 [wiki]）
    config_path.write_text(
        '[wiki]\nhalf_life_days = { volatile = 0.01, drifting = 0.01 }\n', encoding="utf-8"
    )
    assert _lint_json(root, capsys)["freshness"] == {"fresh": 0, "review_due": 0, "stale": 1}

    # ④ 配置文件损坏 → 按空配置处理（回退默认，不崩）
    config_path.write_text("[freshness\n坏掉的 toml", encoding="utf-8")
    assert _lint_json(root, capsys)["freshness"] == {"fresh": 1, "review_due": 0, "stale": 0}


def test_cli_lint_tolerates_unreadable_config_path(tmp_path: Path, capsys, monkeypatch):
    """配置路径不可读（这里用"是目录"复现 IsADirectoryError）→ 回退默认，不崩。

    接线前 lint 从不读配置，故读配置的失败面必须全被吞掉：路径是目录 / 无读权限
    都不能让 lint 整体失败（宽恕面见 cli.load_config，捕获整个 OSError）。
    """
    from researchwiki import cli

    store = make_store(tmp_path)
    now = datetime.now(UTC)
    observed = (now - timedelta(days=1)).isoformat()
    store.save_note("时效断言。", note_id="N-0001", volatility="volatile",
                    observed_at=observed, created=observed)
    as_dir = tmp_path / "config-dir"
    as_dir.mkdir()
    monkeypatch.setattr(cli, "CONFIG_PATH", as_dir)
    assert cli.load_config() == {}
    assert _lint_json(tmp_path / "wiki-data", capsys)["freshness"] == {
        "fresh": 1, "review_due": 0, "stale": 0
    }
