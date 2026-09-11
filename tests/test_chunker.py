from docrag.config import ChunkingConfig
from docrag.ingest.chunker import chunk_document


def _tokens(text: str) -> int:
    """测试用:1 字符 = 1 token,确定性、与模型无关(不加载真实 tokenizer)。"""
    return len(text)


DOC = """# 标题

## 第一节

第一段内容很长,需要被切分。这里写很多字,确保超过 chunk_tokens 的限制。
第二句继续写,让它变得更长。

## 第二节

```python
def f():
    return 1
```
"""


def test_records_section_path():
    cfg = ChunkingConfig(chunk_tokens=1000, overlap_tokens=0)
    chunks = chunk_document(
        {"source": "a.md", "title": "标题", "text": DOC}, cfg, _tokens
    )
    sections = [c["section"] for c in chunks]
    assert "标题 > 第一节" in sections
    assert "标题 > 第二节" in sections


def test_code_block_stays_intact():
    # chunk_tokens=40 足以容纳整个代码块(35 字符),但会切分旁边的长正文
    cfg = ChunkingConfig(chunk_tokens=40, overlap_tokens=0)
    chunks = chunk_document(
        {"source": "a.md", "title": "t", "text": DOC}, cfg, _tokens
    )
    code_chunks = [c for c in chunks if "```python" in c["text"]]
    assert len(code_chunks) == 1  # 代码块仍只出现在一个 chunk 里
    assert "def f():" in code_chunks[0]["text"]
    assert "return 1" in code_chunks[0]["text"]


def test_chunks_within_token_limit():
    cfg = ChunkingConfig(chunk_tokens=20, overlap_tokens=0)
    text = "一句话。" * 50
    chunks = chunk_document(
        {"source": "a.md", "title": "t", "text": text}, cfg, _tokens
    )
    assert len(chunks) > 1
    for c in chunks:
        assert _tokens(c["text"]) <= 20


def test_overlap_between_chunks():
    cfg = ChunkingConfig(chunk_tokens=20, overlap_tokens=10)
    text = "甲乙丙丁。" * 50  # 每句 5 字符
    chunks = chunk_document(
        {"source": "a.md", "title": "t", "text": text}, cfg, _tokens
    )
    assert len(chunks) > 1
    # 每句 5 字符,chunk 装 4 句(20),重叠保留最后 2 句(10)
    assert chunks[0]["text"][-10:] == chunks[1]["text"][:10]


def test_oversized_code_block_is_split():
    cfg = ChunkingConfig(chunk_tokens=20, overlap_tokens=0)
    code = "```python\n" + "\n".join(f"x = {i}" for i in range(20)) + "\n```"
    doc = {"source": "a.md", "title": "t", "text": f"# T\n\n{code}"}
    chunks = chunk_document(doc, cfg, _tokens)
    # 不再有超长 chunk:每个都不超过上限
    for c in chunks:
        assert _tokens(c["text"]) <= 20
    # 代码内容完整保留(所有 x = N 都在,单行不被切断)
    joined = "".join(c["text"] for c in chunks)
    for i in range(20):
        assert f"x = {i}" in joined


def test_english_sentences_split():
    from docrag.ingest.chunker import _split_sentences

    parts = _split_sentences("First sentence. Second sentence. Third sentence.")
    assert len(parts) == 3


def test_english_abbreviation_not_split():
    from docrag.ingest.chunker import _split_sentences

    parts = _split_sentences("Use e.g. a tool. Then another.")
    assert len(parts) == 2
    assert parts[0].strip().endswith("tool.")


def test_oversized_english_word_aware_split():
    from docrag.ingest.chunker import _split_by_chars

    unit = "alpha bravo charlie delta echo foxtrot golf"
    pieces = _split_by_chars(unit, chunk_tokens=12, count_tokens=lambda t: len(t))
    assert " ".join(pieces) == unit  # 单词不丢失、不被从中间劈开
    for p in pieces:
        assert len(p) <= 12
