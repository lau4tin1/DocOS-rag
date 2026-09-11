import pytest

from docrag.config import LLMConfig
from docrag.generate import llm


def test_build_prompt_includes_context_and_question():
    results = [(0.9, {"source": "a.md", "section": "S", "text": "这是片段内容"})]
    system, user = llm.build_prompt(results, "我的问题?")
    assert "这是片段内容" in user
    assert "我的问题?" in user
    assert "a.md" in user
    assert "S" in user
    assert "不要编造" in system  # 系统提示里有关键约束


def test_generate_routes_deepseek(monkeypatch):
    calls = []

    def fake(system, user, cfg, base_url, api_key_env):
        calls.append((base_url, api_key_env))
        return "OK"

    monkeypatch.setattr(llm, "_generate_openai_compatible", fake)
    cfg = LLMConfig(provider="deepseek")
    assert llm.generate("s", "u", cfg) == "OK"
    assert calls[0][0].rstrip("/").endswith("api.deepseek.com/v1")
    assert calls[0][1] == "DEEPSEEK_API_KEY"


def test_generate_unknown_provider_raises():
    with pytest.raises(ValueError):
        llm.generate("s", "u", LLMConfig(provider="nope"))
