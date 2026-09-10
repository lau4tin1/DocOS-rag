from __future__ import annotations

import numpy as np


class NumpyIndex:
    """内存中的精确最近邻检索。

    原理:先把所有向量归一化成单位向量,那么
        cos(A, B) = A·B / (|A||B|) = A·B
    于是余弦相似度退化成一个矩阵点积,一行代码就能算完。
    数据量大到几万片段时,再把它换成 faiss 即可(接口保持一致)。
    """

    def __init__(self) -> None:
        self._vectors: np.ndarray | None = None
        self._chunks: list[dict] = []

    def add(self, vectors: np.ndarray, chunks: list[dict]) -> None:
        vectors = np.asarray(vectors, dtype="float32")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0  # 防止除零
        self._vectors = vectors / norms
        self._chunks = list(chunks)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[float, dict]]:
        """返回按相似度降序的 [(score, chunk), ...],最多 top_k 个。"""
        if self._vectors is None or len(self._vectors) == 0:
            return []

        q = np.asarray(query_vector, dtype="float32")
        q = q / (np.linalg.norm(q) or 1.0)

        scores = self._vectors @ q
        top_k = min(top_k, len(scores))
        order = np.argsort(-scores)[:top_k]
        return [(float(scores[i]), self._chunks[i]) for i in order]

    def __len__(self) -> int:
        return 0 if self._vectors is None else len(self._vectors)
