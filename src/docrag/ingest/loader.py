from __future__ import annotations

import hashlib
import re
from pathlib import Path

_H1_RE = re.compile(r"^#\s+(.*)$", re.MULTILINE)


def load_markdown_file(path: str | Path) -> dict:
    """读取一个 Markdown 文件,返回 {source, title, text}。

    title 取第一个一级标题;没有则用文件名。
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    m = _H1_RE.search(text)
    title = m.group(1).strip() if m else p.stem
    return {"source": str(p), "title": title, "text": text}


def load_documents(raw_dir: str | Path) -> list[dict]:
    """递归读取目录下所有 .md 文件。"""
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        return []
    docs = []
    for p in sorted(raw_dir.rglob("*.md")):
        docs.append(load_markdown_file(p))
    return docs


def hash_file(path: str | Path) -> str:
    """返回文件内容的 SHA-256 十六进制哈希,用于增量索引判断文件是否变化。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
