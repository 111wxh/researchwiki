"""LLM 评审（judge）模块测试（评测支线 task-4）。

覆盖 task-4 简报的契约：

- Tier widening："judge" 是合法档位，ModelRouter.get("judge") 走同一条
  base_url 空回退 MockProvider 的 duck-type 路径（纯注解加宽，运行时零逻辑改动）；
- judge_answer：ScriptedProvider 当 mock judge——合法 JSON 与 ```json 围栏 +
  前后杂文都能解析；畸形输出（无 JSON / 缺字段 / 类型错 / 非 JSON 对象）返回
  None 且不抛异常；越界分数 clamp 到 1–5（不编造、不报错）；
- build_judge_provider：config=None / 缺段 / base_url 空 → MockProvider
  （model 带 mock 标记）；有 base_url → OpenAICompatibleProvider 且 tier=="judge"；
- 真实 config.toml 含 [llm.judge] 段且三键与 [llm.strong] 同形。

全部零网络、零真实模型调用。
"""

import tomllib
from pathlib import Path

import pytest

from researchwiki.evals.judge import (
    JUDGE_SYSTEM,
    JudgeVerdict,
    build_judge_provider,
    judge_answer,
)
from researchwiki.llm.openai_provider import OpenAICompatibleProvider
from researchwiki.llm.provider import MockProvider, ScriptedProvider, StreamEvent
from researchwiki.llm.router import ModelRouter

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_TOML = REPO_ROOT / "config.toml"

VALID_JSON = (
    '{"coverage": 4, "citation": 3, "temporal": 5, "reasons": "覆盖了两个要点，引用基本相符"}'
)

#: judge_answer 的公共实参（一题一答一来源，最小编排）
JUDGE_KW = {
    "question": "LangChain 的默认重试次数是多少？",
    "gold_points": ["默认重试 3 次", "通过 RunnableRetry 实现"],
    "answer": "LangChain 默认重试 3 次 [1]。",
    "sources": ["fixture://ai-frameworks/langchain-core"],
}


def scripted(text: str) -> ScriptedProvider:
    """单回合脚本 Provider：只回一段文本，当 mock judge 用。"""
    return ScriptedProvider(
        [[StreamEvent(type="text_delta", delta=text)]], tier="judge", model="scripted-judge"
    )


# ---------------------------------------------------------------------------
# judge_answer：JSON 解析与 clamp
# ---------------------------------------------------------------------------


def test_judge_answer_parses_plain_json_and_builds_prompt() -> None:
    """合法 JSON → 三维分与理由逐字段落位；user prompt 含问题/要点/回答/编号来源。"""
    provider = scripted(VALID_JSON)
    verdict = judge_answer(provider, **JUDGE_KW)
    assert isinstance(verdict, JudgeVerdict)
    assert (verdict.coverage, verdict.citation, verdict.temporal) == (4, 3, 5)
    assert "覆盖了两个要点" in verdict.reasons
    assert verdict.raw == VALID_JSON  # raw 保留原始输出
    with pytest.raises(AttributeError):  # frozen：禁止改字段（FrozenInstanceError）
        verdict.coverage = 1  # type: ignore[misc]

    # prompt 构造：问题 / gold 逐条编号 / 回答原文 / 来源清单 [n] 编号
    [user_msg] = provider.calls[0]
    assert JUDGE_KW["question"] in user_msg.content
    assert "1. 默认重试 3 次" in user_msg.content
    assert JUDGE_KW["answer"] in user_msg.content
    assert "[1] fixture://ai-frameworks/langchain-core" in user_msg.content
    # rubric 里写明四个输出键
    for key in ("coverage", "citation", "temporal", "reasons"):
        assert key in JUDGE_SYSTEM


def test_judge_answer_parses_fenced_json_with_prose() -> None:
    """```json 围栏 + 前后杂文 → 仍能提取第一个 {...} 块解析。"""
    text = f"好的，以下是评审结果：\n```json\n{VALID_JSON}\n```\n以上。"
    verdict = judge_answer(scripted(text), **JUDGE_KW)
    assert verdict is not None
    assert (verdict.coverage, verdict.citation, verdict.temporal) == (4, 3, 5)
    assert verdict.raw == text


def test_judge_answer_malformed_returns_none() -> None:
    """畸形输出（无 JSON / 缺字段 / 类型错 / JSON 但非对象）→ None 且不抛异常。"""
    bad_outputs = [
        "这段回答没有覆盖 gold 要点，也无法给出评分。",  # 纯文本，无任何 {...}
        '{"coverage": 4, "citation": 3, "temporal": 5}',  # 缺 reasons
        '{"coverage": "4", "citation": 3, "temporal": 5, "reasons": "r"}',  # 分数是字符串
        '{"coverage": true, "citation": 3, "temporal": 5, "reasons": "r"}',  # bool 不算分数
        '["coverage", 4]',  # JSON 合法但不是对象
    ]
    for text in bad_outputs:
        assert judge_answer(scripted(text), **JUDGE_KW) is None, text


def test_judge_answer_clamps_out_of_range_scores() -> None:
    """越界分数 clamp 到 1–5：6 → 5、0 → 1，不抛错也不编造。"""
    text = '{"coverage": 6, "citation": 0, "temporal": 5, "reasons": "r"}'
    verdict = judge_answer(scripted(text), **JUDGE_KW)
    assert verdict is not None
    assert verdict.coverage == 5
    assert verdict.citation == 1
    assert verdict.temporal == 5


# ---------------------------------------------------------------------------
# build_judge_provider：mock 回退与真实端点
# ---------------------------------------------------------------------------


def test_build_judge_provider_falls_back_to_mock() -> None:
    """config=None / 缺 [llm] / 缺 [llm.judge] / base_url 空 → tier="judge" 的 MockProvider。"""
    configs = [
        None,
        {},
        {"llm": {}},
        {"llm": {"judge": {"model": "glm-4.7", "base_url": ""}}},
    ]
    for config in configs:
        provider = build_judge_provider(config)
        assert isinstance(provider, MockProvider), config
        assert provider.tier == "judge"
    assert "mock" in build_judge_provider(None).model


def test_build_judge_provider_real_endpoint() -> None:
    """配置了 base_url → OpenAICompatibleProvider，tier=="judge"，model / key 透传。"""
    provider = build_judge_provider(
        {
            "llm": {
                "judge": {
                    "model": "glm-4.7",
                    "base_url": "https://open.bigmodel.cn/api/paas/v4",
                    "api_key_env": "RESEARCHWIKI_JUDGE_API_KEY",
                }
            }
        }
    )
    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.tier == "judge"
    assert provider.model == "glm-4.7"
    assert provider.base_url == "https://open.bigmodel.cn/api/paas/v4"


# ---------------------------------------------------------------------------
# 真实 config.toml 与 router 的 judge 档
# ---------------------------------------------------------------------------


def test_config_toml_declares_judge_section() -> None:
    """config.toml 有 [llm.judge] 段：三键与 [llm.strong] 同形，model 固定 glm-4.7。"""
    with CONFIG_TOML.open("rb") as f:
        data = tomllib.load(f)
    llm = data["llm"]
    judge = llm["judge"]
    assert {"model", "base_url", "api_key_env"} <= set(judge)
    assert judge["model"] == "glm-4.7"
    assert judge["api_key_env"] == "RESEARCHWIKI_JUDGE_API_KEY"
    assert judge["base_url"] == llm["strong"]["base_url"]  # 同智谱端点


def test_model_router_accepts_judge_tier() -> None:
    """Tier widening 生效：ModelRouter.get("judge") 在 base_url 空的 config 下回 mock。"""
    router = ModelRouter(
        {"strong": {"model": "m", "base_url": "https://x/v1"}, "judge": {}}
    )
    provider = router.get("judge")
    assert isinstance(provider, MockProvider)
    assert provider.tier == "judge"
