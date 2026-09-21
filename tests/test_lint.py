"""lint 单测：引用覆盖率、断链检出、merged 跟随、孤立笔记、CLI 退出码与 --json 输出。

零网络、tmp_path 隔离；CLI 直接调 main() 并断言返回码（capsys 校验中文报告/JOSN）。
"""

import json
from pathlib import Path

from researchwiki.cli import main
from researchwiki.wiki.entities import EntityRegistry
from researchwiki.wiki.index import SearchIndex
from researchwiki.wiki.lint import (
    LintReport,
    is_assertion_line,
    iter_assertion_lines,
    lint_wiki,
    referenced_note_ids,
)
from researchwiki.wiki.store import WikiStore


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
    # 不传 index 时跳过这项附加检查（不影响其它指标）
    assert "index_checked" not in lint_wiki(store).details


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


def test_cli_lint_missing_root_is_healthy(tmp_path: Path, capsys):
    assert main(["lint", "--root", str(tmp_path / "nope")]) == 0
    assert "笔记：0 条" in capsys.readouterr().out
