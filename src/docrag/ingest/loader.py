from __future__ import annotations

import hashlib
import re
from pathlib import Path

import yaml

_H1_RE = re.compile(r"^#\s+(.*)$", re.MULTILINE)

# 支持的输入格式:统一先转成纯文本,再交给 chunker
_SUPPORTED_EXTS = {".md", ".markdown", ".txt", ".pdf"}


def split_frontmatter(text: str) -> tuple[dict, str]:
    """剥离开头的 YAML frontmatter,返回 (metadata, body)。

    支持 --- ... --- 和 --- ... ... 两种闭合方式(Jekyll/Pandoc 惯例)。
    没有 frontmatter 时返回 ({}, 原文本)。
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text

    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() in ("---", "..."):
            end = i
            break
    if end is None:
        return {}, text  # 没有闭合分隔符,不当作 frontmatter

    raw = "\n".join(lines[1:end])
    try:
        metadata = yaml.safe_load(raw)
    except yaml.YAMLError:
        metadata = {}
    if not isinstance(metadata, dict):
        metadata = {}

    body = "\n".join(lines[end + 1:])
    return metadata, body.lstrip("\n")


def load_text_file(path: str | Path) -> dict:
    """读任意纯文本(.md / .txt 等),剥离 frontmatter,提取标题。"""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    meta, body = split_frontmatter(text)
    title = _title_from(meta, body, p)
    return {"source": str(p), "title": title, "text": body, "meta": meta}


def load_pdf_file(path: str | Path) -> dict:
    """抽取 PDF 的文本层。注意:扫描版 PDF(无文本层)需要 OCR,这里不支持。"""
    from pypdf import PdfReader

    p = Path(path)
    reader = PdfReader(str(p))
    pages = [page.extract_text() or "" for page in reader.pages]
    body = "\n\n".join(pages).strip()

    meta_title = None
    if reader.metadata:
        meta_title = getattr(reader.metadata, "title", None)
    title = meta_title or p.stem
    return {"source": str(p), "title": title, "text": body, "meta": {}}


def load_file(path: str | Path) -> dict:
    """按扩展名分发到对应的加载器。"""
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return load_pdf_file(path)
    if ext in {".md", ".markdown", ".txt"}:
        return load_text_file(path)
    raise ValueError(f"不支持的文件类型: {ext!r}")


def load_documents(raw_dir: str | Path) -> list[dict]:
    """递归读取目录下所有支持的文本/PDF 文件,统一转成 {source,title,text,meta}。"""
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        return []
    docs = []
    for p in sorted(raw_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTS:
            docs.append(load_file(p))
    return docs


def _title_from(meta: dict, body: str, p: Path) -> str:
    """标题优先级:frontmatter 的 title -> 正文第一个 H1 -> 文件名。"""
    if meta.get("title"):
        return str(meta["title"])
    m = _H1_RE.search(body)
    if m:
        return m.group(1).strip()
    return p.stem


def hash_file(path: str | Path) -> str:
    """返回文件内容的 SHA-256 十六进制哈希,用于增量索引判断文件是否变化。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
