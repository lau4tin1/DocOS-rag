from docrag.retrieve.reranker import Reranker


class _FakeModel:
    def predict(self, pairs):
        return [1.0 if "答案" in doc else 0.1 for _, doc in pairs]


def test_rerank_resorts_and_truncates():
    r = Reranker.__new__(Reranker)  # 跳过 __init__,避免加载真实交叉编码器模型
    r.model = _FakeModel()

    candidates = [
        (0.5, {"text": "无关内容"}),
        (0.4, {"text": "这是答案内容"}),
        (0.9, {"text": "另一条答案"}),
    ]
    out = r.rerank("问题", candidates, top_k=2)

    assert len(out) == 2
    assert out[0][1]["text"] == "这是答案内容"
    assert out[1][1]["text"] == "另一条答案"
    assert out[0][0] == 1.0
