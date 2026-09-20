"""fetch_url：抓取网页 -> trafilatura 提取正文 -> Jina Reader 降级 -> sources 快照落盘。

快照 append-only（PLAN.md §3）：sources/{sha1(url)}/{sha256(正文)}/，
同一 URL 重抓后内容变化产生新快照目录，旧快照永久保留、绝不覆盖。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
import trafilatura

from researchwiki.tools.fs import atomic_write_text

DEFAULT_SOURCES_DIR = Path("wiki-data/sources")
JINA_READER_PREFIX = "https://r.jina.ai/"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)


class FetchError(Exception):
    """抓取失败：主路径与 Jina Reader 降级均未取得正文。"""

    def __init__(self, url: str, reason: str) -> None:
        self.url = url
        self.reason = reason
        super().__init__(f"fetch failed for {url}: {reason}")


@dataclass
class FetchResult:
    url: str
    final_url: str
    text: str  # 截断后的正文（供上下文使用）
    content_hash: str  # sha256(截断前全文)，快照目录键
    http_status: int
    truncated: bool


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _extract_main_text(html: str) -> str:
    """trafilatura 正文提取；解析异常按空结果处理（交由 Jina 降级）。"""
    try:
        extracted = trafilatura.extract(html)
    except Exception:  # noqa: BLE001 -- trafilatura 对畸形页面偶发内部异常
        return ""
    return extracted or ""


def _fetch_via_jina(client: httpx.Client, url: str) -> tuple[str, int, str, str]:
    """Jina Reader 降级。返回 (text, http_status, final_url, reason)，成功时 reason 为空。"""
    try:
        resp = client.get(f"{JINA_READER_PREFIX}{url}", headers={"User-Agent": USER_AGENT})
    except httpx.HTTPError as exc:
        return "", 0, url, f"{type(exc).__name__}: {exc}"
    if resp.status_code != 200:
        return "", resp.status_code, str(resp.url), f"HTTP {resp.status_code}"
    body = resp.text.strip()
    if not body:
        return "", resp.status_code, str(resp.url), "jina reader 返回空内容"
    return body, resp.status_code, str(resp.url), ""


def _save_snapshot(
    sources_dir: Path,
    *,
    url: str,
    final_url: str,
    text: str,
    content_hash: str,
    http_status: int,
) -> Path:
    """快照落盘：目录已存在（同 URL 同内容）则跳过，append-only 绝不覆盖。"""
    url_key = hashlib.sha1(url.encode("utf-8")).hexdigest()
    snapshot_dir = sources_dir / url_key / content_hash
    if snapshot_dir.exists():
        return snapshot_dir
    meta = {
        "url": url,
        "final_url": final_url,
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "content_hash": content_hash,
        "http_status": http_status,
    }
    atomic_write_text(snapshot_dir / "content.md", text)
    atomic_write_text(
        snapshot_dir / "meta.json", json.dumps(meta, ensure_ascii=False, indent=2) + "\n"
    )
    return snapshot_dir


def fetch_url(
    url: str,
    *,
    max_chars: int = 8000,
    sources_dir: str | Path = DEFAULT_SOURCES_DIR,
    transport: httpx.BaseTransport | None = None,
    timeout: float = 15.0,
) -> FetchResult:
    """抓取 url 提取正文，失败依次降级 Jina Reader，最终抛 FetchError。

    返回的 text 已按 max_chars 截断；content_hash 始终基于截断前全文。
    成功后把完整正文快照写入 sources_dir/{sha1(url)}/{content_hash}/。
    """
    headers = {"User-Agent": USER_AGENT, "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"}
    text, status, final_url, primary_reason = "", 0, url, ""

    with httpx.Client(transport=transport, timeout=timeout, follow_redirects=True) as client:
        try:
            resp = client.get(url, headers=headers)
            status, final_url = resp.status_code, str(resp.url)
            if resp.status_code == 200:
                text = _extract_main_text(resp.text)
                if not text:
                    primary_reason = "trafilatura 未提取到正文"
            else:
                primary_reason = f"HTTP {resp.status_code}"
        except httpx.HTTPError as exc:
            primary_reason = f"{type(exc).__name__}: {exc}"

        if not text:
            jina_text, jina_status, jina_final, jina_reason = _fetch_via_jina(client, url)
            if not jina_text:
                raise FetchError(url, f"primary: {primary_reason}; jina: {jina_reason}")
            text, status, final_url = jina_text, jina_status, jina_final

    full_text = text
    content_hash = _sha256_hex(full_text)
    truncated = len(full_text) > max_chars
    _save_snapshot(
        Path(sources_dir),
        url=url,
        final_url=final_url,
        text=full_text,
        content_hash=content_hash,
        http_status=status,
    )
    return FetchResult(
        url=url,
        final_url=final_url,
        text=full_text[:max_chars],
        content_hash=content_hash,
        http_status=status,
        truncated=truncated,
    )
