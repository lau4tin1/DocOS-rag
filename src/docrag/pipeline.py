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


def list_documents(cfg: Config) -> list[dict]:
    """列出已索引的文件:{"name", "chunks"}。"""
    engine = RAGEngine(cfg)
    engine.reload()
    return engine.list_documents()


def delete_file(name: str, cfg: Config) -> dict:
    """删除一个文件及其 chunk/向量。返回 {"deleted", "found", "total_chunks"}。"""
    return RAGEngine(cfg).delete_file(name)
