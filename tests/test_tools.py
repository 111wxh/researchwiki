"""tools 层单测：全部走 httpx.MockTransport + 本地夹具，零真实网络。"""

import hashlib
import json
from datetime import datetime

import httpx
import pytest

from researchwiki.tools.fetch import FetchError, fetch_url
from researchwiki.tools.fs import SandboxError, safe_list, safe_read, safe_write
from researchwiki.tools.search import (
    BochaSearch,
    MockSearch,
    SearchError,
    SearchHit,
    TavilySearch,
    get_search_provider,
)

# 正文约 620 字，超过 trafilatura 的最小提取阈值，nav/footer 应被丢弃
CHINESE_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="utf-8"><title>Agent 记忆方案综述</title></head>
<body>
<nav><a href="/">首页</a><a href="/about">关于</a></nav>
<article>
<h1>Agent 记忆方案综述</h1>
<p>智能体的长期记忆是当前 agent 工程的核心议题之一。Letta（原 MemGPT）提出 sleep-time compute
思路：由后台子代理在会话空闲期整理记忆，把记忆维护成本移出交互窗口，使主对话上下文始终保持精简，
同时保留跨会话可复用的长期事实。</p>
<p>Mem0 把记忆维护建模为一条流水线：新事实先经过嵌入相似度查重，
再执行 add、update、merge 三类操作之一。重复与矛盾的记忆在写入前
就被消化，历史版本通过 redirect 指针保留，引用可以自动跟随。</p>
<p>在上下文压缩方面，主流做法是滚动 compaction：当上下文占用超过
模型窗口的 70% 时触发压缩，早期轮次被折叠进状态文件，仅保留最近若干
轮原文。稳定的系统提示词与工具顺序还能显著提高 KV-cache 命中率。</p>
<p>检索层面，SQLite FTS5 配合中文分词扩展可以提供关键词召回，sqlite-vec 补充向量召回，
二者混合后在个人知识库规模上已经足够，无需部署重型向量库服务。</p>
<p>评测方面，复用收益必须同时报告单次查询成本与摊销成本：wiki 构建是一次性投入，随着复用次数
增加，摊销成本持续下降，这正是自进化 wiki 的核心价值假设。</p>
<p>最后，证据链的可追溯性决定了 wiki 的可信度：每条原子笔记都应指向具体的来源快照，
来源快照按 URL 与内容哈希双键存储，同一页面内容变化时产生新快照，而旧快照永不覆盖。</p>
</article>
<footer>版权所有 · 示例页面</footer>
</body>
</html>
"""

THIN_HTML = "<html><body><script>var x=1;</script></body></html>"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _url_key(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()


# ---- fetch_url -------------------------------------------------------------


def test_fetch_extracts_chinese_article_with_trafilatura(tmp_path):
    sources = tmp_path / "sources"

    def handler(request: httpx.Request) -> httpx.Response:
        assert "Mozilla" in request.headers["user-agent"]  # UA 伪装浏览器
        return httpx.Response(200, html=CHINESE_HTML)

    result = fetch_url(
        "https://example.com/memo", sources_dir=sources, transport=httpx.MockTransport(handler)
    )

    assert result.http_status == 200
    assert result.final_url == "https://example.com/memo"
    assert not result.truncated
    assert "sleep-time compute" in result.text
    assert "滚动 compaction" in result.text
    assert "首页" not in result.text and "版权所有" not in result.text  # nav/footer 被剔除
    assert result.content_hash == _sha256(result.text)

    # 快照落盘：sources/{sha1(url)}/{content_hash}/ + meta.json
    snap = sources / _url_key("https://example.com/memo") / result.content_hash
    assert (snap / "content.md").read_text(encoding="utf-8") == result.text
    meta = json.loads((snap / "meta.json").read_text(encoding="utf-8"))
    assert meta["url"] == "https://example.com/memo"
    assert meta["final_url"] == "https://example.com/memo"
    assert meta["content_hash"] == result.content_hash
    assert meta["http_status"] == 200
    datetime.fromisoformat(meta["fetched_at"])  # fetched_at 为可解析的 ISO 时间


def test_fetch_truncates_text_and_hash_covers_full_content(tmp_path):
    sources = tmp_path / "sources"
    transport = httpx.MockTransport(lambda request: httpx.Response(200, html=CHINESE_HTML))

    full = fetch_url("https://example.com/memo", sources_dir=sources, transport=transport)
    clipped = fetch_url(
        "https://example.com/memo", max_chars=50, sources_dir=sources, transport=transport
    )

    assert clipped.truncated
    assert len(clipped.text) == 50
    assert clipped.text == full.text[:50]
    assert clipped.content_hash == full.content_hash  # hash 基于截断前全文
    # 同 URL 同内容重抓：append-only，不产生第二个快照目录
    url_dir = sources / _url_key("https://example.com/memo")
    assert [p.name for p in url_dir.iterdir()] == [full.content_hash]


def test_fetch_falls_back_to_jina_reader(tmp_path):
    jina_body = "这是 Jina Reader 返回的纯文本正文。"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "r.jina.ai":
            return httpx.Response(200, text=jina_body)
        return httpx.Response(200, html=THIN_HTML)  # trafilatura 提取不到正文

    result = fetch_url(
        "https://example.com/thin",
        sources_dir=tmp_path / "sources",
        transport=httpx.MockTransport(handler),
    )

    assert result.text == jina_body
    assert not result.truncated
    assert result.content_hash == _sha256(jina_body)


def test_fetch_raises_fetch_error_when_all_paths_fail(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "r.jina.ai":
            return httpx.Response(503)
        return httpx.Response(404)

    with pytest.raises(FetchError) as excinfo:
        fetch_url(
            "https://example.com/gone",
            sources_dir=tmp_path / "sources",
            transport=httpx.MockTransport(handler),
        )
    assert excinfo.value.url == "https://example.com/gone"
    assert "404" in excinfo.value.reason and "503" in excinfo.value.reason


def test_snapshots_are_append_only_across_content_changes(tmp_path):
    sources = tmp_path / "sources"
    url = "https://example.com/evolving"
    bodies = [
        "第一版正文：agent 记忆分三层，来源快照是证据链的根。",
        "第二版正文：内容变了，新快照目录，旧快照永不覆盖。",
    ]
    counter = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "r.jina.ai":
            text = bodies[min(counter["n"], len(bodies) - 1)]
            counter["n"] += 1
            return httpx.Response(200, text=text)
        return httpx.Response(200, html=THIN_HTML)

    transport = httpx.MockTransport(handler)
    r1 = fetch_url(url, sources_dir=sources, transport=transport)
    r2 = fetch_url(url, sources_dir=sources, transport=transport)
    r3 = fetch_url(url, sources_dir=sources, transport=transport)  # 与 r2 同内容

    assert r1.content_hash != r2.content_hash
    url_dir = sources / _url_key(url)
    assert sorted(p.name for p in url_dir.iterdir()) == sorted([r1.content_hash, r2.content_hash])
    assert (
        url_dir / r1.content_hash / "content.md"
    ).read_text(encoding="utf-8") == bodies[0]  # 旧快照原样保留
    assert r3.content_hash == r2.content_hash
    assert len(list(url_dir.iterdir())) == 2  # 同内容重抓不新增快照


# ---- fs 沙箱 ---------------------------------------------------------------


def test_safe_write_read_list_roundtrip(tmp_path):
    root = tmp_path / "wiki-data"

    written = safe_write(root / "notes" / "N-0001.md", "# 笔记\n一条事实一条笔记。", root=root)
    assert written.exists()
    assert safe_read(root / "notes" / "N-0001.md", root=root) == "# 笔记\n一条事实一条笔记。"
    assert [p.name for p in safe_list(root, root=root)] == ["notes"]
    assert [p.name for p in safe_list(root / "notes", root=root)] == ["N-0001.md"]


def test_sandbox_rejects_escape(tmp_path):
    root = tmp_path / "wiki-data"
    outside = tmp_path / "outside.txt"

    with pytest.raises(SandboxError):
        safe_read(outside, root=root)
    with pytest.raises(SandboxError):
        safe_write(outside, "x", root=root)
    with pytest.raises(SandboxError):
        safe_list(outside, root=root)
    with pytest.raises(SandboxError):
        safe_read(root / ".." / "escape.md", root=root)  # .. 逃逸
    with pytest.raises(SandboxError):
        safe_write(tmp_path / "sub" / "x.md", "x", root=root)  # 绝对路径越界
    with pytest.raises(SandboxError):
        safe_read(tmp_path / "whatever.txt")  # 默认根（cwd/wiki-data）同样受限


# ---- web_search ------------------------------------------------------------


def test_tavily_search_parses_mock_response():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "api.tavily.com"
        body = json.loads(request.content)
        assert body["api_key"] == "tv-key"
        assert body["query"] == "agent 记忆" and body["max_results"] == 2
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "Letta",
                        "url": "https://github.com/letta-ai/letta",
                        "content": "sleep-time compute",
                    },
                    {
                        "title": "Mem0",
                        "url": "https://github.com/mem0ai/mem0",
                        "content": "记忆流水线",
                    },
                    {"title": "多余的一条", "url": "https://example.com/x", "content": "被截掉"},
                ]
            },
        )

    provider = TavilySearch(
        "tv-key", transport=httpx.MockTransport(handler), min_interval=0, backoff=0
    )
    hits = provider.search("agent 记忆", max_results=2)

    assert hits == [
        SearchHit(
            title="Letta", url="https://github.com/letta-ai/letta", snippet="sleep-time compute"
        ),
        SearchHit(title="Mem0", url="https://github.com/mem0ai/mem0", snippet="记忆流水线"),
    ]


def test_bocha_search_parses_mock_response():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "api.bochaai.com"
        assert request.headers["authorization"] == "Bearer bo-key"
        return httpx.Response(
            200,
            json={
                "code": 200,
                "data": {
                    "webPages": {
                        "value": [
                            {
                                "name": "博查结果一",
                                "url": "https://example.com/1",
                                "snippet": "摘要一",
                            },
                            {
                                "name": "博查结果二",
                                "url": "https://example.com/2",
                                "snippet": "摘要二",
                            },
                        ]
                    }
                },
            },
        )

    provider = BochaSearch(
        "bo-key", transport=httpx.MockTransport(handler), min_interval=0, backoff=0
    )
    assert provider.search("测试") == [
        SearchHit(title="博查结果一", url="https://example.com/1", snippet="摘要一"),
        SearchHit(title="博查结果二", url="https://example.com/2", snippet="摘要二"),
    ]


def test_search_retries_network_errors_then_succeeds():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.ConnectError("connection refused", request=request)
        return httpx.Response(
            200, json={"results": [{"title": "ok", "url": "https://e.com", "content": "c"}]}
        )

    provider = TavilySearch("k", transport=httpx.MockTransport(handler), min_interval=0, backoff=0)
    assert provider.search("q")[0].title == "ok"
    assert calls["n"] == 3  # 首次 + 重试 2 次


def test_search_raises_after_exhausting_retries():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ReadTimeout("timed out", request=request)

    provider = TavilySearch("k", transport=httpx.MockTransport(handler), min_interval=0, backoff=0)
    with pytest.raises(SearchError):
        provider.search("q")
    assert calls["n"] == 3


def test_mock_search_returns_chinese_fixtures():
    provider = MockSearch()
    hits = provider.search("agent 记忆", max_results=10)

    assert 3 <= len(hits) <= 5
    assert all(h.url.startswith("http") for h in hits)
    assert any("记忆" in h.title or "记忆" in h.snippet for h in hits)
    assert len(provider.search("q", max_results=2)) == 2  # 尊重 max_results


def test_factory_returns_mock_without_any_key(monkeypatch):
    for var in ("SEARCH_PROVIDER", "TAVILY_API_KEY", "BOCHA_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    assert isinstance(get_search_provider(None), MockSearch)
    assert isinstance(get_search_provider({}), MockSearch)
    # 指定了 provider 但没有 key：同样回退 MockSearch，不抛错
    assert isinstance(get_search_provider({"search": {"provider": "tavily"}}), MockSearch)


def test_factory_selects_provider_from_config_and_env(monkeypatch):
    provider = get_search_provider({"search": {"provider": "tavily", "tavily_api_key": "tvk"}})
    assert isinstance(provider, TavilySearch)
    assert provider.api_key == "tvk"

    monkeypatch.setenv("BOCHA_API_KEY", "bok")
    assert isinstance(get_search_provider(), BochaSearch)

    monkeypatch.setenv("SEARCH_PROVIDER", "bocha")
    assert isinstance(get_search_provider(), BochaSearch)
