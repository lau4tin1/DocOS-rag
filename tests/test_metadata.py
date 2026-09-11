from docrag.store import metadata


def test_manifest_roundtrip(tmp_path):
    m = {"a.md": {"hash": "abc", "config": {"model": "M", "chunk_tokens": 256}}}
    metadata.save_manifest(tmp_path, m)
    assert metadata.load_manifest(tmp_path) == m


def test_load_manifest_missing_returns_empty(tmp_path):
    assert metadata.load_manifest(tmp_path / "nonexistent") == {}
