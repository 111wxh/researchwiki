"""FastAPI server：以 SSE 输出 AI SDK UI Message Stream 协议，供 Next.js 前端消费。

/api/chat 按 config.toml 的 [server].mode 分流：
- "mock"（默认）：ResearchRun 脚本化演示，无 key 也能跑，行为与阶段 1 完全一致；
- "real"：真实 agent loop（AgentLoop），且要求 [llm.strong].base_url 非空，
  否则自动回退 mock——保证误配 mode 也不会带着空配置打网络。
"""

from __future__ import annotations

import json
import os
import tomllib
from collections.abc import Iterator
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from researchwiki.env import load_env_file
from researchwiki.llm.accounting import TokenAccountant
from researchwiki.llm.router import ModelRouter
from researchwiki.loop.agent_loop import AgentLoop
from researchwiki.loop.research_run import ResearchRun
from researchwiki.tools import get_search_provider
from researchwiki.wiki.embeddings import get_embedding_provider

# 早于任何读 key 的代码：把项目根的 .env 灌进环境（真实环境变量优先）
load_env_file()

app = FastAPI(title="ResearchWiki API")


def _cors_origins() -> list[str]:
    """允许的前端来源。默认只放本地；容器/远程部署用 RESEARCHWIKI_CORS_ORIGINS 覆盖
    （逗号分隔，填 `*` 表示不限制来源——本服务无鉴权、无 cookie，自托管场景可接受）。"""
    raw = os.environ.get(
        "RESEARCHWIKI_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    )
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    return origins or ["http://localhost:3000"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_methods=["*"],
    allow_headers=["*"],
)

CONFIG_PATH = Path(os.environ.get("RESEARCHWIKI_CONFIG", "config.toml"))


class ChatRequest(BaseModel):
    """AI SDK useChat 发来的消息体；阶段 1 只取最后一条 user 消息作为研究问题。"""

    messages: list[dict] = Field(default_factory=list)


def _last_user_question(messages: list[dict]) -> str:
    """AI SDK 发送 UI Message 格式（{role, parts:[{type:'text',...}]}），兼容 content 旧格式。"""
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        parts = msg.get("parts")
        if isinstance(parts, list):
            texts = [
                p.get("text", "")
                for p in parts
                if isinstance(p, dict) and p.get("type") == "text"
            ]
            joined = " ".join(t for t in texts if t).strip()
            if joined:
                return joined
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            texts = [p.get("text", "") for p in content if isinstance(p, dict)]
            joined = " ".join(t for t in texts if t).strip()
            if joined:
                return joined
    return "agent 记忆方案对比"


def _sse(events: Iterator[dict]) -> Iterator[str]:
    for ev in events:
        yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


def load_config() -> dict:
    """读取运行配置；文件缺失/损坏时按空配置处理（等价于全 mock，保证可启动）。"""
    try:
        with CONFIG_PATH.open("rb") as f:
            return tomllib.load(f)
    except (FileNotFoundError, tomllib.TOMLDecodeError):
        return {}


def real_loop_enabled(config: dict) -> bool:
    """mode="real" 且 strong 档 base_url 非空才走真实 loop，其余一律 mock。

    mode 可被环境变量 RESEARCHWIKI_MODE 覆盖——容器部署时不必改挂载的 config.toml。
    """
    server_cfg = config.get("server") or {}
    mode = os.environ.get("RESEARCHWIKI_MODE") or server_cfg.get("mode") or "mock"
    mode = str(mode).strip().lower()
    if mode != "real":
        return False
    strong = (config.get("llm") or {}).get("strong") or {}
    return bool(str(strong.get("base_url") or "").strip())


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    question = _last_user_question(req.messages)
    accountant = TokenAccountant()
    config = load_config()
    iterator: Iterator[dict]
    if real_loop_enabled(config):
        # 真实 agent loop：strong 驱动主循环，搜索 provider 与记账共用一套配置
        llm_cfg = config.get("llm") or {}
        server_cfg = config.get("server") or {}
        wiki_cfg = config.get("wiki") or {}
        wiki_root = Path(str(server_cfg.get("wiki_data_dir") or "wiki-data"))
        iterator = AgentLoop(
            question,
            router=ModelRouter(llm_cfg),
            llm_config=llm_cfg,
            accountant=accountant,
            search_provider=get_search_provider(config),
            wiki_root=wiki_root,
        # [wiki] / [embedding] 段的配置必须显式传入，否则去重阈值与嵌入模型会被静默忽略；
        # [prior] 段同理（缺省时 AgentLoop 内部按 enabled=true + 默认值处理）
        wiki_config=wiki_cfg,
        prior_config=config.get("prior"),
        formation_config=config.get("formation"),  # [formation] 段（None = 不做入库判定）
        embedding=get_embedding_provider(config, cache_path=wiki_root / "index.db"),
        ).events()
    else:
        iterator = ResearchRun(question, accountant=accountant).events()
    return StreamingResponse(
        _sse(iterator),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
