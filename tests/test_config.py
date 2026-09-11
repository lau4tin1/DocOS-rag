from docrag.config import load_config


def test_load_config_overrides(tmp_path):
    y = tmp_path / "c.yaml"
    y.write_text(
        "chunking:\n  chunk_tokens: 100\n"
        "retrieval:\n  top_k: 7\n  rerank: false\n"
        "llm:\n  provider: anthropic\n",
        encoding="utf-8",
    )
    cfg = load_config(y)
    assert cfg.chunking.chunk_tokens == 100
    assert cfg.retrieval.top_k == 7
    assert cfg.retrieval.rerank is False
    assert cfg.llm.provider == "anthropic"
