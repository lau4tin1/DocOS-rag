import numpy as np

from docrag.store.index import NumpyIndex


def test_search_returns_nearest():
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((5, 16)).astype("float32")
    chunks = [{"text": f"doc{i}"} for i in range(5)]
    idx = NumpyIndex()
    idx.add(vecs, chunks)

    res = idx.search(vecs[3], 2)  # 用第 4 个向量自身当查询
    assert len(res) == 2
    assert res[0][1]["text"] == "doc3"
    assert abs(res[0][0] - 1.0) < 1e-4  # 自查询余弦 = 1.0


def test_top_k_respected():
    rng = np.random.default_rng(1)
    vecs = rng.standard_normal((10, 8)).astype("float32")
    idx = NumpyIndex()
    idx.add(vecs, [{"text": str(i)} for i in range(10)])
    assert len(idx.search(vecs[0], 4)) == 4
