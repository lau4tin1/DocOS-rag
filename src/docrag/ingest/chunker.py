from __future__ import annotations

import re

from ..config import ChunkingConfig

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
# 在句末标点后切分(保留标点),适配中文和英文
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;])")
# Markdown 代码围栏:行首(可有缩进)的 ```
_FENCE_RE = re.compile(r"^\s*```")


def chunk_document(doc: dict, cfg: ChunkingConfig) -> list[dict]:
    """把一个文档切成若干片段,按标题层级保留结构信息。

    doc: {"source": str, "title": str, "text": str}
    返回: [{"source","title","section","text","chunk_index"}, ...]

    思路:
      1. 按 Markdown 标题(## 等)把正文分到不同"小节"里;
      2. 每个小节如果太长,再按"原子单元"切成 chunk_size 左右、带 overlap 的片段;
      3. 原子单元里,普通正文按句子切,而 ``` 代码块作为一个整体、永不切开;
      4. section 记录"一级标题 > 二级标题"这样的层级路径,供展示和过滤。
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

    units = _split_into_units(body)
    if not units:
        return [body]
    return _merge_units(units, cfg.chunk_size, cfg.chunk_overlap)


def _split_into_units(text: str) -> list[str]:
    """把正文切成"不可再分的原子单元"。

    普通正文 -> 按句子切;``` 围起来的代码 -> 整段作为一个单元(带前后换行,
    便于拼接时分隔)。这样技术文档里的代码示例永远不会被截断。
    """
    units: list[str] = []
    in_code = False
    code_lines: list[str] = []
    prose_lines: list[str] = []

    def flush_prose() -> None:
        if prose_lines:
            block = "\n".join(prose_lines)
            units.extend(_split_sentences(block))
            prose_lines.clear()

    for line in text.splitlines():
        if _FENCE_RE.match(line):
            flush_prose()          # 先把之前的普通正文落盘
            code_lines.append(line)
            if in_code:
                # 这是结束围栏:整段代码作为一个原子单元
                units.append("\n" + "\n".join(code_lines) + "\n")
                code_lines.clear()
                in_code = False
            else:
                in_code = True     # 进入代码块
        else:
            if in_code:
                code_lines.append(line)
            else:
                prose_lines.append(line)

    flush_prose()
    if code_lines:  # 文档末尾代码块没闭合,也当作一个整体
        units.append("\n" + "\n".join(code_lines) + "\n")
    return [u for u in units if u.strip()]


def _split_sentences(text: str) -> list[str]:
    # 折叠多余空格,保留换行
    text = re.sub(r"[ \t]+", " ", text)
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _merge_units(units: list[str], chunk_size: int, overlap: int) -> list[str]:
    """按字符数贪心合并单元成 chunk,相邻 chunk 重叠 overlap 个字符。

    代码块不参与重叠回退(避免把整段代码复制两遍),只有普通正文句子才回退。
    """
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0

    for u in units:
        if cur and cur_len + len(u) > chunk_size:
            chunks.append("".join(cur))
            # 重叠:把当前 chunk 结尾不超过 overlap 的正文句子留作下一个 chunk 的开头
            kept: list[str] = []
            kept_len = 0
            for prev in reversed(cur):
                if _is_code(prev) or kept_len + len(prev) > overlap:
                    break
                kept.insert(0, prev)
                kept_len += len(prev)
            cur = kept
            cur_len = kept_len
        cur.append(u)
        cur_len += len(u)

    if cur:
        chunks.append("".join(cur))
    return [c for c in chunks if c.strip()]


def _is_code(unit: str) -> bool:
    return unit.lstrip().startswith("```")
