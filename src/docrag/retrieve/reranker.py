from __future__ import annotations

from sentence_transformers import CrossEncoder


class Reranker:
    """交叉编码器(cross-encoder)精排。

    与 embedding 用的"双编码器"(bi-encoder)不同:
      - 双编码器:query 和 doc 各自编码成向量,再算余弦。快(文档向量可预计算),
        但两者从不"互看",只能捕捉粗粒度的语义相近;
      - 交叉编码器:把 "[CLS] query [SEP] doc [SEP]" 拼成一条输入,一起过一遍
        Transformer,输出一个精确的"相关性"分数。准,但每个 (query, doc) 都要现算,
        无法预计算,所以只用在粗召回之后的小候选集上。

    这就是"两阶段检索":先粗召回(快但糙),再精排(慢但准)。
    """

    def __init__(self, model_name: str):
        self.model = CrossEncoder(model_name)

    def score(self, query: str, texts: list[str]) -> list[float]:
        """返回每个 (query, text) 对的相关性分数,分数越高越相关。"""
        pairs = [(query, t) for t in texts]
        scores = self.model.predict(pairs)
        return [float(s) for s in scores]

    def rerank(
        self,
        query: str,
        candidates: list[tuple[float, dict]],
        top_k: int,
    ) -> list[tuple[float, dict]]:
        """对候选片段精排,返回按相关度降序的 top_k 个 (score, chunk)。

        注意:返回的 score 是交叉编码器的相关性分(取代原来的检索分)。
        """
        if not candidates:
            return []
        texts = [chunk["text"] for _, chunk in candidates]
        scores = self.score(query, texts)
        ranked = sorted(zip(scores, candidates), key=lambda x: -x[0])
        return [(score, chunk) for score, (_, chunk) in ranked[:top_k]]
