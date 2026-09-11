from docrag.ingest.loader import hash_file


def test_hash_file_deterministic(tmp_path):
    f = tmp_path / "a.md"
    f.write_text("hello", encoding="utf-8")
    assert hash_file(f) == hash_file(f)


def test_hash_file_changes_with_content(tmp_path):
    f = tmp_path / "a.md"
    f.write_text("hello", encoding="utf-8")
    h1 = hash_file(f)
    f.write_text("world", encoding="utf-8")
    h2 = hash_file(f)
    assert h1 != h2
