from __future__ import annotations

import os

import httpx

from ..config import LLMConfig

SYSTEM_PROMPT = (
    "你是一个技术文档检索助手。请只根据下面提供的文档片段回答用户的问题。"
    "如果片段中没有足够的信息,请明确说“文档中没有找到相关信息”,不要编造内容。"
    "回答时尽量引用片段中的具体细节,并保持简洁。"
)


def build_prompt(results: list[tuple[float, dict]], question: str) -> tuple[str, str]:
    """把检索到的片段拼成上下文,返回 (system, user) 两条消息。"""
    blocks = []
    for i, (_, chunk) in enumerate(results, 1):
        blocks.append(
            f"[片段 {i}]\n"
            f"来源: {chunk.get('source')}\n"
            f"章节: {chunk.get('section')}\n"
            f"内容:\n{chunk.get('text')}"
        )
    context_text = "\n\n".join(blocks)

    user = (
        f"以下是检索到的技术文档片段:\n\n"
        f"{context_text}\n\n"
        f"请仅依据以上片段回答下面的问题。\n\n"
        f"问题: {question}"
    )
    return SYSTEM_PROMPT, user


def generate(system: str, user: str, cfg: LLMConfig) -> str:
    """根据 provider 分发到不同 API。以后想加本地模型,在这里加一个分支即可。"""
    if cfg.provider == "openai":
        return _generate_openai(system, user, cfg)
    if cfg.provider == "deepseek":
        return _generate_deepseek(system, user, cfg)
    if cfg.provider == "anthropic":
        return _generate_anthropic(system, user, cfg)
    raise ValueError(
        f"不支持的 provider: {cfg.provider!r}(可选 openai | deepseek | anthropic)"
    )


def _generate_openai(system: str, user: str, cfg: LLMConfig) -> str:
    return _generate_openai_compatible(
        system, user, cfg, "https://api.openai.com/v1", "OPENAI_API_KEY"
    )


def _generate_deepseek(system: str, user: str, cfg: LLMConfig) -> str:
    # DeepSeek 的 API 与 OpenAI 完全兼容,只差 base_url 和 key 环境变量
    return _generate_openai_compatible(
        system, user, cfg, "https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"
    )


def _generate_openai_compatible(
    system: str,
    user: str,
    cfg: LLMConfig,
    base_url: str,
    api_key_env: str,
) -> str:
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"请先设置环境变量 {api_key_env}")

    resp = httpx.post(
        base_url.rstrip("/") + "/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": cfg.model,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        },
        timeout=60.0,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _generate_anthropic(system: str, user: str, cfg: LLMConfig) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("请先设置环境变量 ANTHROPIC_API_KEY")

    resp = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": cfg.model,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        },
        timeout=60.0,
    )
    resp.raise_for_status()
    return resp.json()["content"][0]["text"]
