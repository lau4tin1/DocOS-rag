from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# src/docrag/config.py -> 项目根目录是往上两级
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class EmbeddingConfig:
    model: str = "BAAI/bge-small-zh-v1.5"
    device: str = "cpu"
    # bge v1.5 检索查询建议加前缀,文档内容不加
    query_prefix: str = "为这个句子生成表示以用于检索相关文章："


@dataclass
class ChunkingConfig:
    # 单位是 token,不是字符 —— 与 embedding 模型的 tokenizer 口径一致
    chunk_tokens: int = 256
    overlap_tokens: int = 32


@dataclass
class RetrievalConfig:
    top_k: int = 4          # 最终返回给 LLM 的片段数
    hybrid: bool = True     # True=混合检索(BM25 关键词 + 向量语义);False=纯向量
    rrf_k: int = 60         # RRF 融合常数


@dataclass
class LLMConfig:
    provider: str = "openai"  # openai | anthropic
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_tokens: int = 1024


@dataclass
class PathsConfig:
    raw_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "raw")
    index_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "index")


@dataclass
class Config:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)


def _resolve_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def load_config(path: str | Path | None = None) -> Config:
    """从 YAML 加载配置;缺失的键回落到 dataclass 默认值。"""
    if path is None:
        path = PROJECT_ROOT / "config.yaml"

    cfg = Config()
    p = Path(path)
    if not p.exists():
        return cfg

    data: dict[str, Any] = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

    if "embedding" in data:
        cfg.embedding = EmbeddingConfig(**data["embedding"])
    if "chunking" in data:
        cfg.chunking = ChunkingConfig(**data["chunking"])
    if "retrieval" in data:
        cfg.retrieval = RetrievalConfig(**data["retrieval"])
    if "llm" in data:
        cfg.llm = LLMConfig(**data["llm"])
    if "paths" in data:
        pd = data["paths"]
        cfg.paths = PathsConfig(
            raw_dir=_resolve_path(pd.get("raw_dir", cfg.paths.raw_dir)),
            index_dir=_resolve_path(pd.get("index_dir", cfg.paths.index_dir)),
        )

    return cfg
