from __future__ import annotations

from ..embed.embedder import Embedder
from ..store.index import NumpyIndex
from .bm25 import BM25Index


class HybridRetriever:
    """混合检索:向量语义 + BM25 关键词,用 RRF 融合。

    为什么需要混合:
      - 向量检索擅长"语义相近"(换个说法也能命中),但对精确关键词
        (报错码、函数名、ID)不敏感——这些 token 在预训练里没有语义信号;
      - BM25 擅长精确词匹配,但不懂"同义改写"。

    为什么用 RRF(Reciprocal Rank Fusion)融合:
        向量余弦分数在 [0,1],BM25 分数无上限,两者尺度不可比,直接相加没有意义。
        RRF 只依赖"排名"而非"分数":
            rrf(d) = Σ_methods 1 / (rrf_k + rank_method(d))
        k=60 是业界常用常数,rank 从 0 开始。分数可比、无需调权、鲁棒。
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_index: NumpyIndex,
        chunks: list[dict],
        rrf_k: int = 60,
    ):
        self.embedder = embedder
        self.vector_index = vector_index
        self.chunks = chunks
        self.rrf_k = rrf_k
        self.bm25 = BM25Index()
        self.bm25.fit([c["text"] for c in chunks])
        # chunk 对象 -> 下标;vector_index 里存的就是这些同一个对象
        self._pos = {id(c): i for i, c in enumerate(chunks)}

    def retrieve(self, query: str, top_k: int) -> list[tuple[float, dict]]:
        n = len(self.chunks)
        if n == 0:
            return []

        # 1) 向量排名:取全部文档,按余弦分数降序
        qv = self.embedder.embed_query(query)
        vec_ranked = self.vector_index.search(qv, n)  # [(score, chunk)]

        # 2) BM25 排名:按 BM25 分数降序
        bm25_ranked = self.bm25.search(query)         # [(score, doc_index)]

        # 转成 {doc_id: rank(从 0 开始)}
        vec_rank = {self._pos[id(c)]: i for i, (_, c) in enumerate(vec_ranked)}
        bm25_rank = {idx: i for i, (_, idx) in enumerate(bm25_ranked)}

        # 3) RRF 融合
        rrf: dict[int, float] = {}
        for doc_id in range(n):
            s = 0.0
            if doc_id in vec_rank:
                s += 1.0 / (self.rrf_k + vec_rank[doc_id])
            if doc_id in bm25_rank:
                s += 1.0 / (self.rrf_k + bm25_rank[doc_id])
            rrf[doc_id] = s

        order = sorted(range(n), key=lambda i: -rrf[i])[:top_k]
        return [(rrf[i], self.chunks[i]) for i in order]
