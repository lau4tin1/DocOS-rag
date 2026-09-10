from __future__ import annotations

import re

from ..config import ChunkingConfig

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
# 在句末标点后切分(保留标点),适配中文和英文
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;])")


def chunk_document(doc: dict, cfg: ChunkingConfig) -> list[dict]:
    """把一个文档切成若干片段,按标题层级保留结构信息。

    doc: {"source": str, "title": str, "text": str}
    返回: [{"source","title","section","text","chunk_index"}, ...]

    思路:
      1. 按 Markdown 标题(## 等)把正文分到不同"小节"里;
      2. 每个小节如果太长,再按句子切成 chunk_size 左右、带 overlap 的片段;
      3. section 记录"一级标题 > 二级标题"这样的层级路径,供展示和过滤。
    """
    text = doc["text"]
    heading_stack: list[str] = []
    section_lines: list[str] = []
    chunks: list[dict] = []

    def flush() -> None:
        if not any(line.strip() for line in section_lines):
            return
        body = "\n".join(section_lines).strip()
        if not body:
            return
        section = " > ".join(heading_stack) if heading_stack else doc["title"]
        for i, piece in enumerate(_chunk_section(body, cfg)):
            chunks.append(
                {
                    "source": doc["source"],
                    "title": heading_stack[-1] if heading_stack else doc["title"],
                    "section": section,
                    "text": piece,
                    "chunk_index": i,
                }
            )

    for line in text.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            # 遇到新标题:先把上一个小节落盘
            flush()
            level = len(m.group(1))
            heading = m.group(2).strip()
            # 标题层级回退:只保留比当前层级更浅的祖先
            heading_stack = heading_stack[: level - 1] + [heading]
            section_lines = []
        else:
            section_lines.append(line)

    flush()
    return chunks


def _chunk_section(body: str, cfg: ChunkingConfig) -> list[str]:
    if len(body) <= cfg.chunk_size:
        return [body]

    sentences = _split_sentences(body)
    if not sentences:
        return [body]
    return _merge_sentences(sentences, cfg.chunk_size, cfg.chunk_overlap)


def _split_sentences(text: str) -> list[str]:
    # 折叠多余空格,保留换行(代码块等)
    text = re.sub(r"[ \t]+", " ", text)
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _merge_sentences(
    sentences: list[str], chunk_size: int, overlap: int
) -> list[str]:
    """按字符数贪心合并句子成 chunk,并让相邻 chunk 有 overlap 个字符的重叠。"""
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0

    for s in sentences:
        if cur and cur_len + len(s) > chunk_size:
            chunks.append("".join(cur))
            # 重叠:把当前 chunk 结尾不超过 overlap 的句子留作下一个 chunk 的开头
            kept: list[str] = []
            kept_len = 0
            for prev in reversed(cur):
                if kept_len + len(prev) <= overlap:
                    kept.insert(0, prev)
                    kept_len += len(prev)
                else:
                    break
            cur = kept
            cur_len = kept_len
        cur.append(s)
        cur_len += len(s)

    if cur:
        chunks.append("".join(cur))
    return [c for c in chunks if c.strip()]
