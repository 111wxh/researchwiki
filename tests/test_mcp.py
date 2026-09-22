"""wiki MCP server 测试：tmp_path 隔离 wiki-data，零真实网络、零真实 stdio 进程。

测试客户端的选择（两层一起测，各取最稳的形态）
----------------------------------------------
1. **逻辑层**：直接调 ``WikiService``（即"底层函数 + 一层薄封装"里的底层）。
   写保护三件套、redirect 跟随、since 过滤这些分支全在这一层，同步、无事件循环，
   还能用 monkeypatch 注入备份失败。绝大多数断言放这里，跑得稳、失败信息直指代码。
2. **协议层**：用 FastMCP 自带的 in-process Client（``Client(build_server(...))``），
   同进程内存传输——不 spawn 子进程、不开端口、不碰 stdio，但确实走完整的
   MCP 报文往返，能验证工具注册、参数 schema、JSON 序列化，以及
   "结构化错误是普通结果（is_error=False）而不是 tool error"这一关键约定。
   用 ``asyncio.run`` 驱动：项目 dev 依赖里没有 pytest-asyncio，不为一个测试文件加依赖。

真实 stdio 进程的手工冒烟（initialize / tools/list / tools/call）不在测试里跑，
避免 CI 上起进程；启动方式见 mcp_server/server.py 的 docstring。
"""

from __future__ import annotations

import asyncio
import importlib
import json
from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client, FastMCP

from researchwiki.mcp_server import build_server, resolve_wiki_root
from researchwiki.mcp_server import service as service_module
from researchwiki.mcp_server.service import WikiService, create_backup, normalize_note_id
from researchwiki.wiki.frontmatter import parse
from researchwiki.wiki.store import WikiStore

CONFIG = {"wiki": {"fts_tokenizer": "trigram"}}  # trigram：确定性、不探测 vendor DLL


def _tool_call(server: FastMCP, name: str, arguments: dict[str, Any] | None = None) -> dict:
    """在独立事件循环里做一次 in-process MCP 工具调用，返回解析后的 JSON payload。"""

    async def run() -> Any:
        async with Client(server) as client:
            return await client.call_tool(name, arguments or {})

    result = asyncio.run(run())
    assert result.is_error is False, f"结构化错误不应变成 tool error：{result.content}"
    return json.loads(result.content[0].text)


@pytest.fixture()
def wiki_root(tmp_path: Path) -> Path:
    return tmp_path / "wiki-data"


@pytest.fixture()
def store(wiki_root: Path) -> WikiStore:
    return WikiStore(wiki_root)


@pytest.fixture()
def service(wiki_root: Path) -> WikiService:
    return WikiService(wiki_root, config=CONFIG)


@pytest.fixture()
def server(wiki_root: Path) -> FastMCP:
    return build_server(root=wiki_root, config=CONFIG)


def _write_note(service: WikiService, body: str = "正文", **kwargs: Any) -> dict:
    payload = service.write(body, **kwargs)
    assert payload["ok"] is True, payload
    return payload


def _seed_alias_chain(store: WikiStore) -> None:
    """N-0002 active（最终），N-0001 merged→N-0002，N-0003 superseded→N-0001。"""
    store.save_note(
        "上下文压缩把长对话压成摘要，保留关键决策。",
        note_id="N-0002",
        title="上下文压缩",
        entities=["上下文压缩"],
        confidence="high",
    )
    store.save_note(
        "量子退火在这里只是个只出现在旧笔记里的罕见词。",
        note_id="N-0001",
        title="上下文压缩（旧）",
        status="merged",
        redirect_to="N-0002",
    )
    store.save_note("更旧的压缩笔记", note_id="N-0003", title="压缩（更旧）", status="superseded",
                    superseded_by="N-0001")


# ---- 协议层：工具注册与端到端往返 -------------------------------------------


class TestProtocolLayer:
    def test_tool_catalog_and_chinese_descriptions(self, server: FastMCP) -> None:
        async def run() -> list[Any]:
            async with Client(server) as client:
                return await client.list_tools()

        tools = {t.name: t for t in asyncio.run(run())}
        assert set(tools) == {
            "wiki_search",
            "wiki_read",
            "wiki_write",
            "wiki_list_changes",
            "wiki_health",
            # memory_*（P1-A 记忆接口，wiki_* 保留为兼容层）
            "memory_store",
            "memory_search",
            "memory_recall",
            "memory_update",
            "memory_supersede",
            "memory_invalidate",
            "memory_timeline",
            "memory_conflicts",
            "memory_profile",
        }
        for tool in tools.values():
            # 描述是给模型看的：必须说清"什么时候该用"，且是中文
            assert "什么时候用" in (tool.description or ""), tool.name
            assert any("\u4e00" <= ch <= "\u9fff" for ch in tool.description or "")
        write_schema = tools["wiki_write"].input_schema
        assert write_schema["required"] == ["body"]
        assert write_schema["additionalProperties"] is False
        assert set(write_schema["properties"]) == {
            "body",
            "title",
            "entities",
            "confidence",
            "volatility",
            "sources",
        }, "wiki_* 兼容层签名不得改变"
        store_schema = tools["memory_store"].input_schema
        assert set(store_schema["properties"]) == {
            "content",
            "kind",
            "entities",
            "importance",
            "source_urls",
            "confidence",
            "volatility",
        }
        assert store_schema["required"] == ["content"]

    def test_write_read_search_roundtrip(self, server: FastMCP) -> None:
        written = _tool_call(
            server,
            "wiki_write",
            {
                "body": "GLM-5.3 支持 200k 上下文。",
                "title": "GLM-5.3 上下文长度",
                "entities": ["glm-5-3"],
                "confidence": "high",
                "volatility": "drifting",
                "sources": ["https://example.com/glm"],
            },
        )
        assert written["ok"] is True
        assert written["note_id"] == "N-0001"
        assert written["indexed"] is True
        assert written["path"] == "notes/N-0001.md"

        read = _tool_call(server, "wiki_read", {"note_id": "N-0001"})
        assert read["ok"] is True
        assert read["note"]["title"] == "GLM-5.3 上下文长度"
        assert read["note"]["confidence"] == "high"
        assert read["note"]["volatility"] == "drifting"
        assert read["note"]["sources"] == [
            {"url": "https://example.com/glm", "content_hash": ""}
        ]
        assert "200k" in read["body"]
        assert read["redirected"] is False

        found = _tool_call(server, "wiki_search", {"query": "GLM-5.3 上下文长度"})
        assert found["ok"] is True
        assert found["count"] >= 1
        hit = found["results"][0]
        assert hit["note_id"] == "N-0001"
        assert set(hit) == {
            "note_id",
            "title",
            "snippet",
            "score",
            "match_type",
            "redirected_from",
            "redirect_note",
        }
        assert hit["match_type"] in {"fts", "vector", "both"}

        changes = _tool_call(server, "wiki_list_changes", {})
        assert [c["note_id"] for c in changes["changes"]] == ["N-0001"]

        health = _tool_call(server, "wiki_health", {})
        assert health["ok"] is True
        assert health["notes"]["total"] == 1
        assert health["index"]["exists"] is True

    def test_not_found_is_structured_payload_not_tool_error(self, server: FastMCP) -> None:
        payload = _tool_call(server, "wiki_read", {"note_id": "N-9999"})
        assert payload["ok"] is False
        assert payload["error"]["code"] == "not_found"
        assert "不存在" in payload["error"]["message"]

    def test_path_traversal_rejected_over_wire(self, server: FastMCP) -> None:
        payload = _tool_call(server, "wiki_read", {"note_id": "../../../etc/passwd"})
        assert payload["ok"] is False
        assert payload["error"]["code"] == "path_rejected"
        assert payload["error"]["details"]["note_id"] == "../../../etc/passwd"

    def test_missing_body_rejected_by_schema(self, server: FastMCP) -> None:
        """body 是必填参数：协议层 schema 就会拦下来（is_error=True，带可读原因）。"""

        async def run() -> Any:
            async with Client(server) as client:
                return await client.call_tool(
                    "wiki_write", {"title": "缺正文"}, raise_on_error=False
                )

        result = asyncio.run(run())
        assert result.is_error is True
        assert "body" in result.content[0].text


# ---- 写保护 ①：frontmatter schema 校验 --------------------------------------


class TestWriteValidation:
    def test_invalid_confidence_rejected_with_readable_reason(
        self, service: WikiService, wiki_root: Path
    ) -> None:
        payload = service.write("正文", title="标题", confidence="very-high")
        assert payload["ok"] is False
        assert payload["error"]["code"] == "validation_failed"
        assert "confidence 非法" in payload["error"]["message"]
        assert "high / medium / low" in payload["error"]["message"]
        assert not (wiki_root / "notes").exists(), "校验失败不得落盘"
        assert not (wiki_root / ".backups").exists(), "校验失败不得产生备份"

    def test_invalid_volatility_rejected(self, service: WikiService) -> None:
        payload = service.write("正文", title="标题", volatility="fast")
        assert payload["ok"] is False
        assert payload["error"]["code"] == "validation_failed"
        assert "volatility 非法" in payload["error"]["message"]
        assert "stable" in payload["error"]["message"]

    def test_empty_title_rejected(self, service: WikiService) -> None:
        payload = service.write("正文", title="   ")
        assert payload["ok"] is False
        assert payload["error"]["code"] == "validation_failed"
        assert "title 不能为空" in payload["error"]["message"]

    def test_empty_body_rejected(self, service: WikiService) -> None:
        payload = service.write("", title="标题")
        assert payload["ok"] is False
        assert "body 不能为空" in payload["error"]["message"]

    def test_entities_must_be_string_list(self, service: WikiService) -> None:
        payload = service.write("正文", title="标题", entities="glm-5-3,上下文压缩")
        assert payload["ok"] is False
        assert payload["error"]["code"] == "validation_failed"
        assert "entities 必须是字符串列表" in payload["error"]["message"]

        payload = service.write("正文", title="标题", entities=[123])  # type: ignore[list-item]
        assert payload["ok"] is False
        assert "entities 的元素必须是非空字符串" in payload["error"]["message"]

    def test_all_errors_collected_at_once(self, service: WikiService) -> None:
        payload = service.write("", title="", confidence="x", volatility="y")
        errors = payload["error"]["details"]["errors"]
        assert len(errors) == 4
        assert any("body" in e for e in errors) and any("title" in e for e in errors)

    def test_case_and_space_normalized_but_garbage_rejected(self, service: WikiService) -> None:
        payload = _write_note(
            service, "正文", title="标题", confidence=" HIGH ", volatility=" Volatile "
        )
        assert payload["ok"] is True
        read = service.read(payload["note_id"])
        assert read["note"]["confidence"] == "high"
        assert read["note"]["volatility"] == "volatile"

    def test_written_file_is_valid_frontmatter(self, service: WikiService) -> None:
        payload = _write_note(
            service,
            "正文：一条事实。",
            title="标题：带冒号与 # 井号",
            entities=["glm-5-3", "上下文压缩"],
            confidence="high",
            volatility="drifting",
            sources=["https://example.com/a"],
        )
        raw = (service.root / "notes" / f"{payload['note_id']}.md").read_text(encoding="utf-8")
        meta, body = parse(raw)
        assert meta["id"] == payload["note_id"]
        assert meta["title"] == "标题：带冒号与 # 井号"
        assert meta["entities"] == ["glm-5-3", "上下文压缩"]
        assert meta["status"] == "active"
        assert meta["volatility"] == "drifting"
        assert meta["sources"] == [{"url": "https://example.com/a", "content_hash": ""}]
        assert body == "正文：一条事实。"
        assert meta["created"].startswith("20")  # 由 store 自动补的 ISO 时间戳

    def test_sources_missing_only_warns(self, service: WikiService) -> None:
        payload = _write_note(service, "正文", title="标题")
        assert any("sources" in w for w in payload.get("warnings", []))

    def test_huge_body_truncated_on_read(self, service: WikiService, wiki_root: Path) -> None:
        """超大笔记不整篇塞给客户端：截断到 MAX_BODY_CHARS 并显式标注。"""
        from researchwiki.mcp_server.service import MAX_BODY_CHARS

        wiki_root.mkdir(parents=True, exist_ok=True)
        body = "长" * (MAX_BODY_CHARS + 100)
        (wiki_root / "notes").mkdir(parents=True, exist_ok=True)
        (wiki_root / "notes" / "N-0001.md").write_text(
            f"---\nid: N-0001\ntitle: 巨型笔记\nstatus: active\n---\n{body}", encoding="utf-8"
        )
        payload = service.read("N-0001")
        assert payload["ok"] is True
        assert payload["truncated"] is True
        assert len(payload["body"]) == MAX_BODY_CHARS
        assert "截断" in payload["message"]

    def test_concurrent_writes_get_distinct_ids(self, service: WikiService) -> None:
        """并发写入串行化：4 个线程各写一条，ID 不冲突、索引全部可见。"""
        import threading

        results: list[dict] = []
        errors: list[BaseException] = []
        lock = threading.Lock()

        def worker(index: int) -> None:
            try:
                payload = service.write(f"并发笔记 {index}", title=f"并发 {index}")
            except BaseException as exc:  # noqa: BLE001 -- 线程里的异常要带回主线程断言
                with lock:
                    errors.append(exc)
                return
            with lock:
                results.append(payload)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
        assert errors == []
        assert all(r["ok"] is True for r in results), results
        ids = sorted(r["note_id"] for r in results)
        assert ids == ["N-0001", "N-0002", "N-0003", "N-0004"]
        assert len(list(service.root.glob("notes/N-*.md"))) == 4
        assert {r["note_id"] for r in service.search("并发笔记")["results"]} == set(ids)


# ---- 写保护 ②：沙箱路径限制 -------------------------------------------------


class TestPathGuard:
    @pytest.mark.parametrize(
        "bad_id",
        [
            "../../../etc/passwd",
            "..\\..\\windows\\system32",
            "notes/../../../x",
            "/etc/passwd",
            "a/b",
            "..",
            ".hidden",
            "N-0001/../../x",
        ],
    )
    def test_traversal_ids_rejected(self, service: WikiService, bad_id: str) -> None:
        payload = service.read(bad_id)
        assert payload["ok"] is False
        assert payload["error"]["code"] == "path_rejected"
        with pytest.raises(Exception, match="形态非法"):
            service.notes_path(bad_id)

    def test_normalization_accepts_path_forms(self, service: WikiService) -> None:
        _write_note(service, "正文", title="标题")
        assert normalize_note_id("notes/N-0001.md") == "N-0001"
        assert normalize_note_id("wiki-data/notes/N-0001.md") == "N-0001"
        assert normalize_note_id("  N-0001  ") == "N-0001"
        assert service.read("notes/N-0001.md")["note"]["note_id"] == "N-0001"
        assert service.read("N-0001")["ok"] is True

    def test_note_paths_stay_inside_wiki_root(self, service: WikiService, wiki_root: Path) -> None:
        _write_note(service, "正文", title="标题")
        path = service.notes_path("N-0001")
        assert path.resolve().is_relative_to(wiki_root.resolve())
        assert path.parent == (wiki_root / "notes").resolve()

    def test_write_never_accepts_client_supplied_path(self, service: WikiService) -> None:
        """写入路径由 store 分配 ID 决定，客户端无法指定落盘位置。"""
        payload = _write_note(service, "正文", title="标题")
        assert payload["note_id"] == "N-0001"
        assert payload["path"] == "notes/N-0001.md"
        assert (service.root / "notes" / "N-0001.md").is_file()


# ---- 写保护 ③：写前备份 -----------------------------------------------------


class TestWriteBackup:
    def test_backup_created_marker_for_new_note(
        self, service: WikiService, wiki_root: Path
    ) -> None:
        payload = _write_note(service, "正文", title="标题")
        backup = payload["backup"]
        assert backup["action"] == "created"
        assert backup["backup_file"] is None
        manifest_path = wiki_root / backup["dir"] / "manifest.json"
        assert manifest_path.is_file()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["action"] == "created"
        assert manifest["note_id"] == "N-0001"
        assert manifest["target"] == "notes/N-0001.md"

    def test_backup_copies_original_before_overwrite(self, wiki_root: Path, service) -> None:
        target = wiki_root / "notes" / "N-0001.md"
        target.parent.mkdir(parents=True)
        target.write_text("将被覆盖的原始内容", encoding="utf-8")
        info = create_backup(
            wiki_root, note_id="N-0001", title="原始", source_path=target, existed=True
        )
        assert info["action"] == "overwrite"
        copied = wiki_root / info["backup_file"]
        assert copied.read_text(encoding="utf-8") == "将被覆盖的原始内容"
        manifest = json.loads(
            (wiki_root / info["dir"] / "manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["action"] == "overwrite"
        assert manifest["backup_file"] == info["backup_file"]
        assert manifest["bytes"] == len("将被覆盖的原始内容".encode())

    def test_backup_dirs_are_unique_and_inside_root(self, service: WikiService) -> None:
        first = _write_note(service, "一", title="A")["backup"]["dir"]
        second = _write_note(service, "二", title="B")["backup"]["dir"]
        assert first != second, "同一秒内连续写入也不能覆盖彼此备份"
        assert (service.root / first).resolve().is_relative_to(service.root.resolve())

    def test_backup_failure_refuses_write(
        self, service: WikiService, wiki_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(*args: Any, **kwargs: Any) -> dict:
            raise OSError("模拟备份失败：目标不可写")

        monkeypatch.setattr(service_module, "create_backup", boom)
        payload = service.write("不该落盘的正文", title="禁止写入")
        assert payload["ok"] is False
        assert payload["error"]["code"] == "backup_failed"
        assert "已拒绝写入" in payload["error"]["message"]
        assert not (wiki_root / "notes").exists(), "备份失败必须拒绝写入"
        assert list(wiki_root.rglob("*.md")) == []

    def test_backup_sandbox_error_also_refuses_write(
        self, service: WikiService, wiki_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from researchwiki.tools.fs import SandboxError

        def boom(*args: Any, **kwargs: Any) -> dict:
            raise SandboxError("模拟越界")

        monkeypatch.setattr(service_module, "create_backup", boom)
        payload = service.write("不该落盘的正文", title="禁止写入")
        assert payload["ok"] is False
        assert payload["error"]["code"] == "backup_failed"
        assert not (wiki_root / "notes").exists()

    def test_write_backup_then_write_order(
        self, service: WikiService, wiki_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """备份必须发生在落盘之前：备份函数里能看到"笔记文件还不存在"。"""
        seen: dict[str, Any] = {}
        original = service_module.create_backup

        def spy(root: Any, **kwargs: Any) -> dict:
            seen["target_exists"] = Path(kwargs["source_path"]).exists()
            seen["note_id"] = kwargs["note_id"]
            return original(root, **kwargs)

        monkeypatch.setattr(service_module, "create_backup", spy)
        payload = _write_note(service, "正文", title="标题")
        assert seen == {"target_exists": False, "note_id": payload["note_id"]}


# ---- redirect 跟随 ----------------------------------------------------------


class TestRedirectRead:
    def test_merged_note_follows_redirect(
        self, service: WikiService, store: WikiStore
    ) -> None:
        _seed_alias_chain(store)
        payload = service.read("N-0001")
        assert payload["ok"] is True
        assert payload["note"]["note_id"] == "N-0002"
        assert payload["note"]["status"] == "active"
        assert payload["redirected"] is True
        assert payload["source_note"] == "N-0001"
        assert payload["redirect_chain"] == ["N-0001"]
        assert payload["requested_id"] == "N-0001"
        assert "N-0001" in payload["message"] and "N-0002" in payload["message"]
        assert "摘要" in payload["body"]

    def test_chained_superseded_follows_to_final(
        self, service: WikiService, store: WikiStore
    ) -> None:
        _seed_alias_chain(store)
        payload = service.read("N-0003")
        assert payload["note"]["note_id"] == "N-0002"
        assert payload["redirect_chain"] == ["N-0003", "N-0001"]
        assert "N-0003 → N-0001 → N-0002" in payload["message"]

    def test_active_note_not_marked_redirected(
        self, service: WikiService, store: WikiStore
    ) -> None:
        _seed_alias_chain(store)
        payload = service.read("N-0002")
        assert payload["redirected"] is False
        assert "message" not in payload
        assert payload["note"]["redirect_to"] is None

    def test_cycle_returns_structured_error(self, service: WikiService, store: WikiStore) -> None:
        store.save_note("a", note_id="N-0001", status="merged", redirect_to="N-0002")
        store.save_note("b", note_id="N-0002", status="merged", redirect_to="N-0001")
        payload = service.read("N-0001")
        assert payload["ok"] is False
        assert payload["error"]["code"] == "redirect_cycle"
        assert "成环" in payload["error"]["message"]

    def test_broken_redirect_missing_target(self, service: WikiService, store: WikiStore) -> None:
        store.save_note("a", note_id="N-0001", status="merged", redirect_to="N-9999")
        payload = service.read("N-0001")
        assert payload["ok"] is False
        assert payload["error"]["code"] == "broken_redirect"
        assert payload["error"]["details"]["missing"] == "N-9999"

    def test_broken_redirect_without_link_field(
        self, service: WikiService, store: WikiStore
    ) -> None:
        store.save_note("a", note_id="N-0001", status="merged")
        payload = service.read("N-0001")
        assert payload["ok"] is False
        assert payload["error"]["code"] == "broken_redirect"
        assert "redirect_to" in payload["error"]["message"]

    def test_search_hit_on_alias_annotates_redirected_from(
        self, service: WikiService, store: WikiStore
    ) -> None:
        _seed_alias_chain(store)
        payload = service.search("量子退火")
        assert payload["ok"] is True
        # 别名笔记不会作为独立结果出现：内容一律归到最终 active 笔记
        assert {r["note_id"] for r in payload["results"]} == {"N-0002"}

    def test_redirect_note_explained_in_payload(
        self, service: WikiService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """索引层报出 redirected_from 时，payload 里要有人话说明（含两个 ID）。"""
        from researchwiki.mcp_server import service as svc

        class FakeIndex:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            def __enter__(self) -> FakeIndex:
                return self

            def __exit__(self, *exc: object) -> None:
                return None

            def search(self, query: str, k: int = 5, kind: str | None = None) -> list[Any]:
                from researchwiki.wiki.index import SearchMatch

                return [
                    SearchMatch(
                        note_id="N-0002",
                        title="上下文压缩",
                        snippet="摘要",
                        score=0.03,
                        match_type="fts",
                        redirected_from="N-0001",
                    )
                ]

            def rebuild(self, store: Any) -> int:  # pragma: no cover - 不应被调用
                raise AssertionError("不该重建")

        monkeypatch.setattr(svc, "SearchIndex", FakeIndex)
        payload = service.search("上下文压缩")
        hit = payload["results"][0]
        assert hit["redirected_from"] == "N-0001"
        assert "N-0001" in hit["redirect_note"] and "N-0002" in hit["redirect_note"]
        assert "merged" in hit["redirect_note"]


# ---- 检索 -------------------------------------------------------------------


class TestSearch:
    def test_search_works_without_api_key(self, service: WikiService) -> None:
        """无 key（MockEmbeddingProvider）也能返回结果。"""
        _write_note(service, "上下文压缩技术把历史对话压成摘要。", title="上下文压缩")
        payload = service.search("上下文压缩")
        assert payload["ok"] is True
        assert payload["count"] >= 1
        assert payload["results"][0]["note_id"] == "N-0001"
        assert payload["results"][0]["snippet"]

    def test_search_rebuilds_index_after_external_write(
        self, service: WikiService, store: WikiStore
    ) -> None:
        """别的进程/loop 直接写 md（绕过 MCP），检索也应看到——按需自动重建索引。"""
        store.save_note("外部进程写入的事实：向量检索用 sqlite-vec。", note_id="N-0001",
                        title="向量检索")
        payload = service.search("sqlite-vec 向量检索")
        assert payload["ok"] is True
        assert payload["index_rebuilt"] >= 1
        assert [r["note_id"] for r in payload["results"]] == ["N-0001"]

    def test_search_empty_query_rejected(self, service: WikiService) -> None:
        payload = service.search("   ")
        assert payload["ok"] is False
        assert payload["error"]["code"] == "invalid_argument"

    @pytest.mark.parametrize("bad_k", [0, -1, 51, "5", True])
    def test_search_k_range_validated(self, service: WikiService, bad_k: Any) -> None:
        payload = service.search("查询", k=bad_k)
        assert payload["ok"] is False
        assert payload["error"]["code"] == "invalid_argument"

    def test_search_no_hit_returns_hint(self, service: WikiService) -> None:
        payload = service.search("完全不存在的词xyzzy")
        assert payload["ok"] is True
        assert payload["results"] == []
        assert payload["count"] == 0
        assert payload["hint"]


# ---- 增量列表 ---------------------------------------------------------------


class TestListChanges:
    def _seed(self, store: WikiStore) -> None:
        store.save_note("最旧", note_id="N-0001", title="最旧",
                        created="2026-09-01T00:00:00+00:00")
        store.save_note("中间", note_id="N-0002", title="中间",
                        created="2026-09-10T00:00:00+00:00",
                        reviewed_at="2026-09-12T00:00:00+00:00")
        store.save_note("最新", note_id="N-0003", title="最新",
                        created="2026-09-15T00:00:00+00:00",
                        status="merged", redirect_to="N-0002")

    def test_sorted_desc_by_change_time(self, service: WikiService, store: WikiStore) -> None:
        self._seed(store)
        payload = service.list_changes()
        assert payload["ok"] is True
        assert [c["note_id"] for c in payload["changes"]] == ["N-0003", "N-0002", "N-0001"]
        assert payload["latest"] == "2026-09-15T00:00:00+00:00"
        assert payload["total"] == 3
        assert payload["count"] == 3
        assert payload["has_more"] is False
        assert payload["next_since"] is None

    def test_change_kind_reports_new_reviewed_status(
        self, service: WikiService, store: WikiStore
    ) -> None:
        self._seed(store)
        kinds = {c["note_id"]: c["change_kind"] for c in service.list_changes()["changes"]}
        assert kinds == {"N-0001": "created", "N-0002": "reviewed", "N-0003": "status"}

    def test_since_filters_strictly_newer(self, service: WikiService, store: WikiStore) -> None:
        self._seed(store)
        payload = service.list_changes(since="2026-09-10T00:00:00+00:00")
        assert [c["note_id"] for c in payload["changes"]] == ["N-0003", "N-0002"]
        # reviewed_at 比 created 新时，按 reviewed_at 判定
        payload = service.list_changes(since="2026-09-11T00:00:00+00:00")
        assert [c["note_id"] for c in payload["changes"]] == ["N-0003", "N-0002"]
        payload = service.list_changes(since="2026-09-12T00:00:00+00:00")
        assert [c["note_id"] for c in payload["changes"]] == ["N-0003"]

    def test_truncation_without_since_returns_newest_page(
        self, service: WikiService, store: WikiStore
    ) -> None:
        self._seed(store)
        payload = service.list_changes(limit=2)
        assert [c["note_id"] for c in payload["changes"]] == ["N-0003", "N-0002"]
        assert payload["total"] == 3
        assert payload["has_more"] is True
        assert payload["next_since"] is None
        assert "完整同步" in payload["hint"]

    def test_paging_with_since_is_gapless(self, service: WikiService, store: WikiStore) -> None:
        """增量同步场景：给了 since 就返回最早一页 + next_since 前向游标，逐页覆盖不漏。"""
        for index, day in enumerate(["01", "05", "10", "15"], start=1):
            store.save_note(
                f"笔记{index}",
                note_id=f"N-{index:04d}",
                title=f"笔记{index}",
                created=f"2026-09-{day}T00:00:00+00:00",
            )
        page1 = service.list_changes(since="2026-08-31T00:00:00+00:00", limit=2)
        assert [c["note_id"] for c in page1["changes"]] == ["N-0002", "N-0001"]  # 最早一页
        assert page1["total"] == 4
        assert page1["has_more"] is True
        assert page1["next_since"] == "2026-09-05T00:00:00+00:00"
        page2 = service.list_changes(since=page1["next_since"], limit=2)
        assert [c["note_id"] for c in page2["changes"]] == ["N-0004", "N-0003"]
        assert page2["has_more"] is False
        covered = [c["note_id"] for c in page1["changes"] + page2["changes"]]
        assert sorted(covered) == ["N-0001", "N-0002", "N-0003", "N-0004"], "两页并集应无遗漏"
        assert len(covered) == len(set(covered)), "两页之间不应重复"

    def test_paging_keeps_same_timestamp_batch_together(
        self, service: WikiService, store: WikiStore
    ) -> None:
        """同一秒批量写入的笔记不得被页边界切开：游标只有时间戳，切开就会静默丢条目。"""
        created_by_id = {
            "N-0001": "2026-09-01T00:00:00+00:00",
            "N-0002": "2026-09-05T00:00:00+00:00",
            "N-0003": "2026-09-05T00:00:00+00:00",  # 与 N-0002 同一秒（批量入库的常见形态）
            "N-0004": "2026-09-10T00:00:00+00:00",
            "N-0005": "2026-09-10T00:00:00+00:00",
        }
        for note_id, created in created_by_id.items():
            store.save_note(f"笔记 {note_id}", note_id=note_id, created=created)
        page1 = service.list_changes(since="2026-08-31T00:00:00+00:00", limit=2)
        # 页边界落在 09-05 同秒分组里 → 该分组整体并入本页（条数超过 limit），游标不切开分组
        assert [c["note_id"] for c in page1["changes"]] == ["N-0003", "N-0002", "N-0001"]
        assert page1["next_since"] == "2026-09-05T00:00:00+00:00"
        page2 = service.list_changes(since=page1["next_since"], limit=2)
        assert [c["note_id"] for c in page2["changes"]] == ["N-0005", "N-0004"]
        assert page2["has_more"] is False
        covered = [c["note_id"] for c in page1["changes"] + page2["changes"]]
        assert sorted(covered) == ["N-0001", "N-0002", "N-0003", "N-0004", "N-0005"], "同秒分组不漏"
        assert len(covered) == len(set(covered)), "两页之间不应重复"

    def test_iso_z_suffix_accepted(self, service: WikiService, store: WikiStore) -> None:
        self._seed(store)
        payload = service.list_changes(since="2026-09-14T00:00:00Z")
        assert [c["note_id"] for c in payload["changes"]] == ["N-0003"]

    def test_invalid_since_rejected(self, service: WikiService) -> None:
        payload = service.list_changes(since="昨天")
        assert payload["ok"] is False
        assert payload["error"]["code"] == "invalid_argument"
        assert "ISO" in payload["error"]["message"]

    @pytest.mark.parametrize("bad_limit", [0, -3, 201, "50"])
    def test_invalid_limit_rejected(self, service: WikiService, bad_limit: Any) -> None:
        payload = service.list_changes(limit=bad_limit)
        assert payload["ok"] is False
        assert payload["error"]["code"] == "invalid_argument"

    def test_entries_carry_sync_fields(self, service: WikiService, store: WikiStore) -> None:
        self._seed(store)
        entry = service.list_changes()["changes"][0]
        assert set(entry) == {
            "note_id",
            "title",
            "status",
            "change_kind",
            "confidence",
            "volatility",
            "entities",
            "created",
            "reviewed_at",
            "observed_at",
            "updated",
            "redirect_to",
            "superseded_by",
            "path",
        }
        assert entry["status"] == "merged" and entry["redirect_to"] == "N-0002"
        assert entry["path"] == "notes/N-0003.md"


# ---- memory_*：面向 Agent 的外置记忆接口（P1-A）-----------------------------


class TestMemoryTools:
    """九个 memory_* 工具的服务层语义（协议层端到端见 TestMemoryProtocol）。"""

    def test_store_then_search_with_kind_filter(self, service: WikiService) -> None:
        stored = service.store_memory(
            "用户偏好深色主题的界面", kind="user", importance=0.9, entities=["用户偏好"]
        )
        assert stored["ok"] is True
        assert stored["note_id"] == "N-0001"
        assert stored["kind"] == "user"
        assert stored["importance"] == 0.9
        # 标题从正文派生，检索可命中
        raw = (service.root / "notes" / "N-0001.md").read_text(encoding="utf-8")
        meta, _ = parse(raw)
        assert meta["kind"] == "user" and meta["importance"] == 0.9
        assert meta["title"] == "用户偏好深色主题的界面"

        service.store_memory("上下文压缩技术降低长会话成本", kind="knowledge")
        found = service.search("用户偏好", kind="user")
        assert found["ok"] is True
        assert {r["note_id"] for r in found["results"]} == {"N-0001"}
        found_knowledge = service.search("上下文压缩", kind="knowledge")
        assert {r["note_id"] for r in found_knowledge["results"]} == {"N-0002"}
        # kind=None 不过滤
        assert service.search("压缩成本", k=5, kind=None)["count"] >= 1

    def test_store_defaults_and_derived_title(self, service: WikiService) -> None:
        payload = service.store_memory("GLM-5.3 支持 200k 上下文窗口。")
        assert payload["ok"] is True
        assert payload["kind"] == "knowledge"
        assert payload["importance"] is None
        read = service.read(payload["note_id"])
        assert read["note"]["kind"] == "knowledge"
        assert read["note"]["importance"] is None

    def test_store_empty_content_rejected(self, service: WikiService) -> None:
        payload = service.store_memory("   ")
        assert payload["ok"] is False
        assert payload["error"]["code"] == "validation_failed"
        assert not (service.root / "notes").exists()

    def test_store_invalid_kind_rejected(self, service: WikiService) -> None:
        payload = service.store_memory("正文", kind="diary")
        assert payload["ok"] is False
        assert payload["error"]["code"] == "invalid_kind"
        assert "knowledge / user / experience" in payload["error"]["message"]

    def test_store_invalid_importance_rejected(self, service: WikiService) -> None:
        for bad in (1.5, "0.8", True):
            payload = service.store_memory("正文", importance=bad)  # type: ignore[arg-type]
            assert payload["ok"] is False, bad
            assert payload["error"]["code"] == "validation_failed", bad
            assert "importance" in payload["error"]["message"]

    def test_recall_annotates_memory_state(self, service: WikiService) -> None:
        service.store_memory(
            "上下文压缩技术把历史对话压成摘要。", kind="knowledge", importance=0.6
        )
        payload = service.recall("上下文压缩")
        assert payload["ok"] is True
        assert payload["mode"] == "passthrough"
        hit = payload["results"][0]
        assert hit["note_id"] == "N-0001"
        assert hit["status"] == "active"
        assert hit["kind"] == "knowledge"
        assert hit["importance"] == 0.6
        assert hit["observed_at"] is None  # 未填 observed_at，透传 None

    def test_recall_invalid_kind(self, service: WikiService) -> None:
        payload = service.recall("查询", kind="diary")
        assert payload["ok"] is False
        assert payload["error"]["code"] == "invalid_kind"

    def test_search_invalid_kind(self, service: WikiService) -> None:
        payload = service.search("查询", kind="diary")
        assert payload["ok"] is False
        assert payload["error"]["code"] == "invalid_kind"

    def test_update_rewrites_body_with_backup_and_reason(
        self, service: WikiService, wiki_root: Path, store: WikiStore
    ) -> None:
        # created 给一个更早的时间戳：同秒内 updated 会让 reviewed==created，
        # list_changes 的 change_kind 判据（reviewed > created）就分不出 reviewed 了
        store.save_note(
            "GLM-5.3 支持 128k 上下文。", note_id="N-0001", title="上下文长度",
            created="2026-09-01T00:00:00+00:00",
        )
        old_reviewed = service.read("N-0001")["note"]["reviewed_at"]
        payload = service.update_memory(
            "N-0001", "GLM-5.3 支持 200k 上下文。", "官方文档更新为 200k"
        )
        assert payload["ok"] is True
        assert payload["note_id"] == "N-0001"  # 不新增 ID
        assert payload["backup"]["action"] == "overwrite"
        assert payload["backup"]["backup_file"] is not None

        # 正文已改、reason 追加落盘进 frontmatter extra、reviewed_at 刷新
        raw = (wiki_root / "notes" / "N-0001.md").read_text(encoding="utf-8")
        meta, body = parse(raw)
        assert "200k" in body and "128k" not in body
        assert meta["update_reasons"] == [{"reason": "官方文档更新为 200k",
                                          "at": payload["reviewed_at"]}]
        assert "update_reason" not in meta  # 单键形态不再使用
        assert meta["reviewed_at"] != old_reviewed

        # 原稿在备份里
        backup_file = wiki_root / payload["backup"]["backup_file"]
        assert "128k" in backup_file.read_text(encoding="utf-8")

        # 索引同步：旧关键词不再命中、新关键词命中
        assert service.search("128k")["count"] == 0
        assert service.search("200k")["count"] == 1
        # list_changes 里表现为 reviewed 变更
        changes = service.list_changes()["changes"]
        assert changes[0]["change_kind"] == "reviewed"

    def test_update_reasons_accumulate_across_revisions(
        self, service: WikiService, wiki_root: Path
    ) -> None:
        """多次修订：update_reasons 按序追加，全部原因保留（不整键覆盖）。"""
        _write_note(service, "v1 正文", title="标题")
        assert service.update_memory("N-0001", "v2 正文", "原因一")["ok"] is True
        assert service.update_memory("N-0001", "v3 正文", "原因二")["ok"] is True
        meta, _ = parse((wiki_root / "notes" / "N-0001.md").read_text(encoding="utf-8"))
        records = meta["update_reasons"]
        assert [r["reason"] for r in records] == ["原因一", "原因二"]
        assert all(str(r["at"]).startswith("20") for r in records)
        assert records[1]["at"] >= records[0]["at"]

    def test_update_reason_folds_legacy_single_key(
        self, service: WikiService, wiki_root: Path
    ) -> None:
        """迁移兼容：旧版本的单键 update_reason 首次追加时折叠为列表首条目。"""
        _write_note(service, "v1 正文", title="标题")
        path = wiki_root / "notes" / "N-0001.md"
        raw = path.read_text(encoding="utf-8")
        head, _, rest = raw.partition("---\n")
        assert head == "" and rest
        path.write_text(
            "---\n"
            "update_reason: 旧版原因\n"
            "update_reason_at: 2026-09-01T00:00:00+00:00\n"
            f"{rest}",
            encoding="utf-8",
        )
        payload = service.update_memory("N-0001", "v2 正文", "新版原因")
        assert payload["ok"] is True
        meta, _ = parse(path.read_text(encoding="utf-8"))
        assert [r["reason"] for r in meta["update_reasons"]] == ["旧版原因", "新版原因"]
        assert meta["update_reasons"][0]["at"] == "2026-09-01T00:00:00+00:00"
        assert "update_reason" not in meta and "update_reason_at" not in meta

    def test_update_validations(self, service: WikiService, store: WikiStore) -> None:
        _write_note(service, "正文", title="标题")
        assert service.update_memory("N-9999", "新正文", "原因")["error"]["code"] == "not_found"
        assert service.update_memory("N-0001", "  ", "原因")["error"]["code"] == (
            "validation_failed"
        )
        assert service.update_memory("N-0001", "新正文", "  ")["error"]["code"] == (
            "validation_failed"
        )
        # 非 active 笔记不能原地修订
        store.save_note("旧", note_id="N-0002", title="旧", status="superseded",
                        superseded_by="N-0001")
        payload = service.update_memory("N-0002", "新正文", "原因")
        assert payload["error"]["code"] == "invalid_argument"
        assert payload["error"]["details"]["status"] == "superseded"

    def test_update_rejects_write_on_validation_failure(
        self, service: WikiService, wiki_root: Path
    ) -> None:
        """修订校验失败时不得产生备份或改动。"""
        _write_note(service, "正文", title="标题")
        before = (wiki_root / "notes" / "N-0001.md").read_text(encoding="utf-8")
        payload = service.update_memory("N-0001", "", "原因")
        assert payload["ok"] is False
        assert (wiki_root / "notes" / "N-0001.md").read_text(encoding="utf-8") == before

    def test_supersede_chains_old_to_new(self, service: WikiService, wiki_root: Path) -> None:
        _write_note(
            service, "GLM-5.3 支持 128k 上下文。", title="上下文长度",
            entities=["glm-5-3"], confidence="high",
        )
        payload = service.supersede_memory("N-0001", "GLM-5.3 支持 200k 上下文。", "官方文档更新")
        assert payload["ok"] is True
        assert payload["old_note_id"] == "N-0001"
        assert payload["new_note_id"] == "N-0002"
        assert payload["superseded_by"] == "N-0002"

        # 旧笔记 status=superseded、superseded_by 指向新 ID、reason 落盘
        old_raw = (wiki_root / "notes" / "N-0001.md").read_text(encoding="utf-8")
        old_meta, _ = parse(old_raw)
        assert old_meta["status"] == "superseded"
        assert old_meta["superseded_by"] == "N-0002"
        assert old_meta["supersede_reason"] == "官方文档更新"

        # 沿链可达：读旧 ID 自动到新笔记；新笔记继承实体/kind
        read = service.read("N-0001")
        assert read["note"]["note_id"] == "N-0002"
        assert read["redirected"] is True
        assert read["note"]["entities"] == ["glm-5-3"]
        assert "200k" in read["body"]
        # 新笔记已入索引
        assert service.search("200k")["count"] == 1

    def test_supersede_validations(self, service: WikiService, store: WikiStore) -> None:
        _write_note(service, "正文", title="标题")
        assert service.supersede_memory("N-9999", "新内容", "原因")["error"]["code"] == (
            "not_found"
        )
        assert service.supersede_memory("N-0001", "  ", "原因")["error"]["code"] == (
            "validation_failed"
        )
        assert service.supersede_memory("N-0001", "新内容", "  ")["error"]["code"] == (
            "validation_failed"
        )

    def test_invalidate_creates_reachable_tombstone(
        self, service: WikiService, wiki_root: Path
    ) -> None:
        _write_note(service, "传闻：某模型下周发布。", title="某模型发布传闻")
        payload = service.invalidate_memory("N-0001", "官方辟谣，传闻不实")
        assert payload["ok"] is True
        assert payload["old_note_id"] == "N-0001"
        assert payload["tombstone_id"] == "N-0002"
        assert payload["superseded_by"] == "N-0002"

        # 墓碑是 kind=knowledge 的 active 笔记，正文含原因与时间戳
        tomb_raw = (wiki_root / "notes" / "N-0002.md").read_text(encoding="utf-8")
        tomb_meta, tomb_body = parse(tomb_raw)
        assert tomb_meta["status"] == "active"
        assert tomb_meta["kind"] == "knowledge"
        assert "官方辟谣，传闻不实" in tomb_body

        # 旧笔记沿链可达墓碑（不变量：superseded 必须沿链可达 active）
        old_meta, _ = parse((wiki_root / "notes" / "N-0001.md").read_text(encoding="utf-8"))
        assert old_meta["status"] == "superseded"
        assert old_meta["superseded_by"] == "N-0002"
        assert old_meta["invalidate_reason"] == "官方辟谣，传闻不实"
        read = service.read("N-0001")
        assert read["note"]["note_id"] == "N-0002"
        assert read["redirected"] is True

    def test_invalidate_requires_reason(self, service: WikiService) -> None:
        _write_note(service, "正文", title="标题")
        payload = service.invalidate_memory("N-0001", "  ")
        assert payload["ok"] is False
        assert payload["error"]["code"] == "validation_failed"

    def test_timeline_ordered_oldest_first(
        self, service: WikiService, store: WikiStore
    ) -> None:
        """链拓扑事件 + 变更记录合并，全时间线旧 → 新有序（显式时间戳保证确定性）。"""
        store.save_note(
            "v1", note_id="N-0001", title="一", status="superseded",
            superseded_by="N-0002", created="2026-09-01T00:00:00+00:00",
            reviewed_at="2026-09-02T00:00:00+00:00",
            extra={"supersede_reason": "官方文档更新"},
        )
        store.save_note(
            "v2", note_id="N-0002", title="二", status="superseded",
            superseded_by="N-0003", created="2026-09-02T00:00:00+00:00",
            reviewed_at="2026-09-03T00:00:00+00:00",
            extra={"supersede_reason": "表述修正"},
        )
        store.save_note("v3", note_id="N-0003", title="三",
                        created="2026-09-03T00:00:00+00:00")

        # 从最新版本查，也能拿到完整历史（含每条笔记的 created 与 status 事件）
        payload = service.timeline("N-0003")
        assert payload["ok"] is True
        assert payload["current"] == "N-0003"
        assert payload["direction"] == "oldest_first"
        events = payload["events"]
        assert [(e["note_id"], e["change_kind"]) for e in events] == [
            ("N-0001", "created"),
            ("N-0001", "status"),
            ("N-0002", "created"),
            ("N-0002", "status"),
            ("N-0003", "created"),
        ]
        assert [e["status"] for e in events] == [
            "superseded", "superseded", "superseded", "superseded", "active",
        ]
        assert events[1]["superseded_by"] == "N-0002"
        assert events[1]["reason"] == "官方文档更新"
        assert events[3]["reason"] == "表述修正"
        assert events[-1]["reason"] is None
        updated_list = [e["updated"] for e in events]
        assert updated_list == sorted(updated_list), "事件必须按时间升序"

        # 从链中/链首查，结果一致
        assert service.timeline("N-0001")["events"] == events
        assert service.timeline("N-0002")["current"] == "N-0003"

    def test_timeline_includes_predecessors_of_active_note(
        self, service: WikiService
    ) -> None:
        """对最新（active）版本查询时，反向收集到全部更早版本。"""
        _write_note(service, "第一版结论。", title="结论")
        service.supersede_memory("N-0001", "第二版结论。", "补充证据")
        payload = service.timeline("N-0002")
        kinds = [(e["note_id"], e["change_kind"]) for e in payload["events"]]
        assert len(kinds) == 3
        assert ("N-0001", "created") in kinds
        assert ("N-0001", "status") in kinds
        assert ("N-0002", "created") in kinds
        # 同一笔记内事件按时间有序（created 先于 status）
        assert kinds.index(("N-0001", "created")) < kinds.index(("N-0001", "status"))

    def test_timeline_active_note_multiple_revisions(self, service: WikiService) -> None:
        """同一 active 记忆多次修订：timeline 必须含两条 reviewed 事件（不折叠）。"""
        _write_note(service, "GLM-5.3 支持 128k。", title="上下文长度")
        assert service.update_memory("N-0001", "GLM-5.3 支持 192k。", "第一次修正")["ok"]
        assert service.update_memory("N-0001", "GLM-5.3 支持 200k。", "第二次修正")["ok"]
        payload = service.timeline("N-0001")
        assert payload["ok"] is True
        assert payload["current"] == "N-0001"
        events = payload["events"]
        assert [(e["note_id"], e["change_kind"]) for e in events] == [
            ("N-0001", "created"),
            ("N-0001", "reviewed"),
            ("N-0001", "reviewed"),
        ]
        # 修订事件按 update_reasons 追加顺序排列，原因各自保留
        assert [e["reason"] for e in events[1:]] == ["第一次修正", "第二次修正"]
        assert events[1]["status"] == "active"

    def test_timeline_not_found(self, service: WikiService) -> None:
        payload = service.timeline("N-9999")
        assert payload["ok"] is False
        assert payload["error"]["code"] == "not_found"

    def test_conflicts_passthrough(self, service: WikiService, store: WikiStore) -> None:
        assert service.conflicts()["count"] == 0
        store.save_conflict(
            "上下文窗口多大？", {"note_id": "N-0001", "excerpt": "128k"},
            {"note_id": "N-0002", "excerpt": "200k"},
        )
        open_payload = service.conflicts()
        assert open_payload["ok"] is True
        assert open_payload["status"] == "open"
        assert open_payload["count"] == 1
        entry = open_payload["conflicts"][0]
        assert entry["conflict_id"] == "C-0001"
        assert entry["claim_a"]["excerpt"] == "128k"
        assert entry["resolution"] is None

        store.resolve_conflict("C-0001", verdict="以官方文档为准", resolved_with="N-0002")
        assert service.conflicts()["count"] == 0
        resolved = service.conflicts(status="resolved")
        assert resolved["count"] == 1
        assert resolved["conflicts"][0]["resolution"]["resolved_with"] == "N-0002"
        assert service.conflicts(status="all")["count"] == 1

    def test_conflicts_invalid_status(self, service: WikiService) -> None:
        payload = service.conflicts(status="bogus")
        assert payload["ok"] is False
        assert payload["error"]["code"] == "invalid_argument"

    def test_profile_lists_user_memories_only(self, service: WikiService) -> None:
        service.store_memory("用户偏好深色主题。", kind="user", importance=0.8)
        service.store_memory("用户习惯早晨做研究。", kind="user")
        service.store_memory("上下文压缩技术降低成本。", kind="knowledge")
        payload = service.profile()
        assert payload["ok"] is True
        assert payload["count"] == 2
        assert [m["note_id"] for m in payload["memories"]] == ["N-0001", "N-0002"]
        entry = payload["memories"][0]
        assert entry["title"] == "用户偏好深色主题。"
        assert entry["importance"] == 0.8
        assert entry["snippet"]


class TestMemoryProtocol:
    """memory_* 协议层端到端：in-process Client 走完整 MCP 报文往返。"""

    def test_memory_lifecycle_over_wire(self, server: FastMCP) -> None:
        stored = _tool_call(
            server,
            "memory_store",
            {"content": "用户偏好简洁的中文回复。", "kind": "user", "importance": 0.7},
        )
        assert stored["ok"] is True
        assert stored["note_id"] == "N-0001"
        assert stored["kind"] == "user"

        recalled = _tool_call(server, "memory_recall", {"query": "用户偏好", "kind": "user"})
        assert recalled["ok"] is True
        assert recalled["mode"] == "passthrough"
        hit = recalled["results"][0]
        assert hit["note_id"] == "N-0001"
        assert hit["kind"] == "user" and hit["status"] == "active"

        updated = _tool_call(
            server,
            "memory_update",
            {
                "note_id": "N-0001",
                "content": "用户偏好简洁的中文回复，不要 emoji。",
                "reason": "补充约束",
            },
        )
        assert updated["ok"] is True

        superseded = _tool_call(
            server,
            "memory_supersede",
            {"note_id": "N-0001", "new_content": "用户偏好简体中文短回复。", "reason": "画像重构"},
        )
        assert superseded["ok"] is True
        assert superseded["new_note_id"] == "N-0002"

        profile = _tool_call(server, "memory_profile", {})
        assert profile["ok"] is True
        assert [m["note_id"] for m in profile["memories"]] == ["N-0002"]

        timeline = _tool_call(server, "memory_timeline", {"note_id": "N-0002"})
        kinds = [(e["note_id"], e["change_kind"]) for e in timeline["events"]]
        # N-0001 的 created + memory_update 留下的 reviewed + supersede 的 status，
        # 加上 N-0002 的 created——链拓扑事件与变更记录合并输出
        assert ("N-0001", "created") in kinds
        assert ("N-0001", "reviewed") in kinds
        assert ("N-0001", "status") in kinds
        assert ("N-0002", "created") in kinds
        assert len(kinds) == 4
        updated_list = [e["updated"] for e in timeline["events"]]
        assert updated_list == sorted(updated_list)

        invalidated = _tool_call(
            server,
            "memory_invalidate",
            {"note_id": "N-0002", "reason": "画像过期"},
        )
        assert invalidated["ok"] is True
        assert invalidated["tombstone_id"] == "N-0003"
        # 旧 ID 沿链可达墓碑
        read = _tool_call(server, "wiki_read", {"note_id": "N-0002"})
        assert read["note"]["note_id"] == "N-0003"

        conflicts = _tool_call(server, "memory_conflicts", {})
        assert conflicts == {"ok": True, "status": "open", "count": 0, "conflicts": []}

    def test_memory_store_invalid_kind_over_wire(self, server: FastMCP) -> None:
        payload = _tool_call(server, "memory_store", {"content": "正文", "kind": "diary"})
        assert payload["ok"] is False
        assert payload["error"]["code"] == "invalid_kind"

    def test_memory_update_not_found_over_wire(self, server: FastMCP) -> None:
        payload = _tool_call(
            server, "memory_update", {"note_id": "N-9999", "content": "正文", "reason": "原因"}
        )
        assert payload["ok"] is False
        assert payload["error"]["code"] == "not_found"


# ---- 健康检查 ---------------------------------------------------------------


class TestHealth:
    def test_counts(self, service: WikiService, store: WikiStore) -> None:
        store.save_note("active", note_id="N-0001", title="A")
        store.save_note("merged", note_id="N-0002", title="B", status="merged",
                        redirect_to="N-0001")
        store.save_note("superseded", note_id="N-0003", title="C", status="superseded",
                        superseded_by="N-0001")
        store.save_page("glm", "GLM 家族", "页面正文")
        store.save_conflict("哪个对？", {"note_id": "N-0001"}, {"note_id": "N-0002"})
        payload = service.health()
        assert payload["ok"] is True
        assert payload["notes"] == {"total": 3, "active": 1, "merged": 1, "superseded": 1}
        assert payload["pages"] == 1
        assert payload["conflicts"] == {"total": 1, "open": 1, "resolved": 0}
        assert payload["index"]["exists"] is False
        assert payload["root"] == str(service.root)

    def test_index_state_after_write(self, service: WikiService) -> None:
        _write_note(service, "正文", title="标题")
        index = service.health()["index"]
        assert index["exists"] is True
        assert index["indexed_notes"] == 1
        assert index["stale"] is False
        assert index["tokenizer"] == "trigram"

    def test_entity_count(self, service: WikiService, store: WikiStore) -> None:
        from researchwiki.wiki.entities import EntityRegistry

        registry = EntityRegistry(store.root)
        registry.get_or_create("GLM-5.3")
        registry.get_or_create("上下文压缩")
        assert service.health()["entities"] == 2


# ---- 模块/入口 --------------------------------------------------------------


class TestModuleSurface:
    def test_importable_and_exports(self) -> None:
        module = importlib.import_module("researchwiki.mcp_server")
        assert callable(module.build_server)
        assert callable(module.main)
        assert module.WikiService is WikiService

    def test_dunder_main_importable_without_running(self) -> None:
        """`python -m researchwiki.mcp_server` 的入口文件可导入（导入不等于启动）。"""
        module = importlib.import_module("researchwiki.mcp_server.__main__")
        assert callable(module.main)

    def test_build_server_defaults(self, tmp_path: Path) -> None:
        built = build_server(root=tmp_path / "w")
        assert isinstance(built, FastMCP)
        assert built.name == "researchwiki"

    def test_resolve_wiki_root_precedence(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("RESEARCHWIKI_WIKI_DATA", raising=False)
        explicit = tmp_path / "explicit"
        assert resolve_wiki_root(explicit, config={"server": {"wiki_data_dir": "cfg"}}) == explicit
        assert (
            resolve_wiki_root(None, config={"server": {"wiki_data_dir": str(tmp_path / "cfg")}})
            == tmp_path / "cfg"
        )
        assert resolve_wiki_root(None, config={}) == Path("wiki-data")
        monkeypatch.setenv("RESEARCHWIKI_WIKI_DATA", str(tmp_path / "env"))
        assert resolve_wiki_root(None, config={"server": {"wiki_data_dir": "cfg"}}) == (
            tmp_path / "env"
        )

    def test_parser_defaults(self) -> None:
        from researchwiki.mcp_server.server import _build_parser

        args = _build_parser().parse_args([])
        assert args.transport == "stdio"
        assert args.root is None
        assert args.port == 8765
