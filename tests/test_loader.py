from docrag.ingest.loader import (
    hash_file,
    load_file,
    load_text_file,
    split_frontmatter,
)


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


def test_split_frontmatter():
    text = "---\ntitle: 测试文档\nauthor: 我\n---\n\n# 正文\n\n内容。\n"
    meta, body = split_frontmatter(text)
    assert meta == {"title": "测试文档", "author": "我"}
    assert "正文" in body
    assert "title:" not in body


def test_split_frontmatter_dots_end():
    text = "---\ntitle: X\n...\n\n正文内容\n"
    meta, body = split_frontmatter(text)
    assert meta == {"title": "X"}
    assert body.strip() == "正文内容"


def test_split_frontmatter_none():
    text = "# 没有 frontmatter\n\n内容\n"
    meta, body = split_frontmatter(text)
    assert meta == {}
    assert body == text


def test_load_text_file_uses_frontmatter_title(tmp_path):
    p = tmp_path / "a.md"
    p.write_text("---\ntitle: 我的标题\n---\n\n# 另一个 H1\n\n正文", encoding="utf-8")
    doc = load_text_file(p)
    assert doc["title"] == "我的标题"
    assert "title:" not in doc["text"]


def test_load_file_dispatch_txt(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("---\ntitle: T\n---\n正文", encoding="utf-8")
    doc = load_file(p)
    assert doc["title"] == "T"
