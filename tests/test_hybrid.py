import numpy as np
import pytest

from docrag.retrieve.hybrid import HybridRetriever, rrf_fuse
from docrag.store.index import NumpyIndex


def test_rrf_fuse_combines_ranks():
    # 两个检索器:doc0 在第 1 个里排 0、第 2 个里排 1;doc1 反之
    ranks = [{0: 0, 1: 2}, {1: 0, 0: 1}]
    scores = rrf_fuse(ranks, k=60)
    assert scores[0] == pytest.approx(1 / 60 + 1 / 61)
    assert scores[1] == pytest.approx(1 / 62 + 1 / 60)
    assert scores[0] > scores[1]


class _FakeEmbedder:
    def __init__(self, qv):
        self.qv = qv

    def embed_query(self, text):
        return self.qv


def test_keyword_only_doc_is_rescued():
    # A/C 语义相近,B 只靠一个稀有词命中;向量检索会把 B 排到最后
    chunks = [
        {"section": "A", "text": "apple banana"},
        {"section": "B", "text": "unique_rare_token zzz"},
        {"section": "C", "text": "apple banana cherry"},
    ]
    vecs = np.array([[1, 0, 0], [0, 1, 0], [0.9, 0.1, 0]], dtype="float32")
    idx = NumpyIndex()
    idx.add(vecs, chunks)

    # 查询向量指向 A(向量只认 A),查询词只命中 B(BM25 只认 B)
    hyb = HybridRetriever(
        _FakeEmbedder(np.array([1, 0, 0], dtype="float32")), idx, chunks, rrf_k=60
    )
    results = hyb.retrieve("unique_rare_token", top_k=3)
    order = [c["section"] for _, c in results]
    # 纯向量: A, C, B;混合后 B 被 BM25 从末尾救到第二
    assert order == ["A", "B", "C"]
