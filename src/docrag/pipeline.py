from __future__ import annotations

from .config import Config
from .engine import RAGEngine


def build_index(cfg: Config) -> int:
    """离线阶段:读文档 -> 切分 -> 向量化 -> 存索引(增量)。返回片段数量。"""
    return RAGEngine(cfg).build_index()


def ask(question: str, cfg: Config) -> tuple[str, list[tuple[float, dict]]]:
    """在线阶段:检索 -> 拼 prompt -> 调 LLM。返回 (答案, 命中片段列表)。"""
    engine = RAGEngine(cfg)
    engine.reload()
    return engine.answer(question, [])
