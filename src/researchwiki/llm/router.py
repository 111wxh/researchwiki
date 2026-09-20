"""模型档位路由：strong / cheap 两档 → 真实 Provider 或 MockProvider。

config 为 tomllib 解析 config.toml 后的 [llm] 段：
    {"strong": {"model": ..., "base_url": ..., "api_key_env": ...},
     "cheap":  {...}}
base_url 留空即 mock 模式——无 key 也能全链路跑通。
"""

from __future__ import annotations

from typing import Any, Literal

from researchwiki.llm.openai_provider import OpenAICompatibleProvider
from researchwiki.llm.provider import MockProvider, Provider

Tier = Literal["strong", "cheap"]

# mock 模式的占位回复脚本（无网络依赖，供全链路演示）
_MOCK_SCRIPT = "（mock 模式）未配置真实模型，这是一段占位回复。"


class ModelRouter:
    """按档位取 Provider：base_url 为空回退 MockProvider，否则走 OpenAI 兼容端点。"""

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config

    def get(self, tier: Tier) -> Provider:
        cfg = self._config.get(tier) or {}
        model = str(cfg.get("model") or "")
        base_url = str(cfg.get("base_url") or "").strip()
        api_key_env = str(cfg.get("api_key_env") or "").strip()
        if not base_url:
            return MockProvider(_MOCK_SCRIPT, tier=tier, model=model or f"mock-{tier}")
        return OpenAICompatibleProvider(
            base_url=base_url,
            model=model,
            tier=tier,
            api_key_env=api_key_env or None,
        )
