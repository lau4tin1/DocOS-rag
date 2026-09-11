from __future__ import annotations

import os

import httpx

from ..config import LLMConfig

SYSTEM_PROMPT = (
    "你是一个技术文档检索助手。请只根据下面提供的文档片段回答用户的问题。"
    "如果片段中没有足够的信息,请明确说“文档中没有找到相关信息”,不要编造内容。"
    "回答时尽量引用片段中的具体细节,并保持简洁。"
    "这是多轮对话,请结合之前的对话理解当前问题(尤其是代词、省略指代)。"
)


def _context_text(results: list[tuple[float, dict]]) -> str:
    blocks = []
    for i, (_, chunk) in enumerate(results, 1):
        blocks.append(
            f"[片段 {i}]\n"
            f"来源: {chunk.get('source')}\n"
            f"章节: {chunk.get('section')}\n"
            f"内容:\n{chunk.get('text')}"
        )
    return "\n\n".join(blocks)


def build_prompt(results: list[tuple[float, dict]], question: str) -> tuple[str, str]:
    """把检索到的片段拼成上下文,返回 (system, user) 两条消息(单轮)。"""
    context_text = _context_text(results)
    user = (
        f"以下是检索到的技术文档片段:\n\n"
        f"{context_text}\n\n"
        f"请仅依据以上片段回答下面的问题。\n\n"
        f"问题: {question}"
    )
    return SYSTEM_PROMPT, user


def build_chat_messages(
    results: list[tuple[float, dict]],
    question: str,
    history: list[dict],
) -> list[dict]:
    """构造多轮对话的消息列表。

    history: 之前的对话,每项 {"role": "user"|"assistant", "content": ...}。
    返回 [{"role":"system",...}, ...历史..., {"role":"user", "content": 上下文+问题}]
    """
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history:
        if turn.get("role") in ("user", "assistant"):
            messages.append({"role": turn["role"], "content": turn["content"]})

    context_text = _context_text(results)
    user = (
        f"以下是检索到的技术文档片段:\n\n"
        f"{context_text}\n\n"
        f"请仅依据以上片段回答下面的问题。\n\n"
        f"问题: {question}"
    )
    messages.append({"role": "user", "content": user})
    return messages


def generate(system: str, user: str, cfg: LLMConfig) -> str:
    """单轮生成(兼容旧接口)。"""
    return generate_messages(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        cfg,
    )


def generate_messages(messages: list[dict], cfg: LLMConfig) -> str:
    """按 provider 分发。messages 是完整消息列表(含 system / user / assistant)。"""
    if cfg.provider == "openai":
        return _openai_compat(messages, cfg, "https://api.openai.com/v1", "OPENAI_API_KEY")
    if cfg.provider == "deepseek":
        return _openai_compat(messages, cfg, "https://api.deepseek.com/v1", "DEEPSEEK_API_KEY")
    if cfg.provider == "anthropic":
        return _anthropic(messages, cfg)
    raise ValueError(
        f"不支持的 provider: {cfg.provider!r}(可选 openai | deepseek | anthropic)"
    )


def _openai_compat(
    messages: list[dict],
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
            "messages": messages,
        },
        timeout=60.0,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _anthropic(messages: list[dict], cfg: LLMConfig) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("请先设置环境变量 ANTHROPIC_API_KEY")

    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    chat = [m for m in messages if m["role"] != "system"]

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
            "messages": chat,
        },
        timeout=60.0,
    )
    resp.raise_for_status()
    return resp.json()["content"][0]["text"]
