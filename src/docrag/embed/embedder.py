from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from ..config import EmbeddingConfig


class Embedder:
    """统一封装向量化,后续想换模型/换 API 只改这里。

    关键点:建索引和查询必须用同一个 Embedder 实例/配置,
    否则向量不在同一个空间里,相似度没有意义。
    """

    def __init__(self, cfg: EmbeddingConfig):
        self.cfg = cfg
        # normalize_embeddings=True 让向量模长为 1,之后余弦相似度 = 点积
        self.model = SentenceTransformer(cfg.model, device=cfg.device)

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype="float32")

    def embed_query(self, text: str) -> np.ndarray:
        """查询侧:给问题加前缀(部分模型需要),再向量化。"""
        query_text = self.cfg.query_prefix + text if self.cfg.query_prefix else text
        return self.embed([query_text])[0]
