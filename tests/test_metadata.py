from docrag.store import metadata


def test_manifest_roundtrip(tmp_path):
    m = {"a.md": {"hash": "abc", "config": {"model": "M", "chunk_tokens": 256}}}
    metadata.save_manifest(tmp_path, m)
    assert metadata.load_manifest(tmp_path) == m


def test_load_manifest_missing_returns_empty(tmp_path):
    assert metadata.load_manifest(tmp_path / "nonexistent") == {}


def test_clear_index(tmp_path):
    import numpy as np

    metadata.save_index(tmp_path, np.zeros((2, 8), dtype="float32"), [{"source": "a"}])
    metadata.save_manifest(tmp_path, {"a": {"hash": "x"}})
    metadata.clear_index(tmp_path)
    assert not (tmp_path / "vectors.npy").exists()
    assert not (tmp_path / "chunks.json").exists()
    assert not (tmp_path / "manifest.json").exists()
