from __future__ import annotations

from ..embed.embedder import Embedder
from ..store.index import NumpyIndex


class Retriever:
    """查询 -> 向量 -> top-k 片段。"""

    def __init__(self, embedder: Embedder, index: NumpyIndex):
        self.embedder = embedder
        self.index = index

    def retrieve(self, query: str, top_k: int) -> list[tuple[float, dict]]:
        query_vector = self.embedder.embed_query(query)
        return self.index.search(query_vector, top_k)
