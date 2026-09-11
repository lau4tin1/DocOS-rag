from __future__ import annotations

import re
from typing import Callable

from ..config import ChunkingConfig

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
# 句末切分:中文/ASCII 标点直接切;英文句号 . 后跟空白+大写/数字时才切
# (避免误切 "e.g."、"3.14" 这类)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;])|(?<=\.)[ \t\n]+(?=[A-Z0-9])")
# Markdown 代码围栏:行首(可有缩进)的 ```
_FENCE_RE = re.compile(r"^\s*```")
# 切分逻辑版本:改了 chunker 的切分规则就 +1,增量索引会据此自动全量重建
CHUNK_VERSION = 3


def chunk_document(
    doc: dict,
    cfg: ChunkingConfig,
    count_tokens: Callable[[str], int],
) -> list[dict]:
    """把一个文档切成若干片段,按标题层级保留结构信息。

    doc: {"source": str, "title": str, "text": str}
    count_tokens: 统计一段文本 token 数的函数(必须用 embedding 模型自己的
        tokenizer,这样"片段长度"才和模型的 512 token 上限口径一致)。
    返回: [{"source","title","section","text","chunk_index"}, ...]

    思路:
      1. 按 Markdown 标题(## 等)把正文分到不同"小节"里;
      2. 每个小节如果太长(按 token 计),再切成 chunk_tokens 左右、带 overlap 的片段;
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
        for i, piece in enumerate(_chunk_section(body, cfg, count_tokens)):
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


def _chunk_section(
    body: str,
    cfg: ChunkingConfig,
    count_tokens: Callable[[str], int],
) -> list[str]:
    if count_tokens(body) <= cfg.chunk_tokens:
        return [body]

    units = _split_into_units(body)
    if not units:
        return [body]
    return _merge_units(units, cfg, count_tokens)


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


def _merge_units(
    units: list[str],
    cfg: ChunkingConfig,
    count_tokens: Callable[[str], int],
) -> list[str]:
    """按 token 数贪心合并单元成 chunk,相邻 chunk 重叠 overlap_tokens 个 token。

    代码块不参与重叠回退(避免把整段代码复制两遍),只有普通正文句子才回退。
    单个单元(通常是超长代码块)本身超过 chunk_tokens 时,做硬切兜底。
    """
    chunks: list[str] = []
    cur: list[str] = []
    cur_tokens = 0

    for u in units:
        t = count_tokens(u)
        if t > cfg.chunk_tokens:
            # 兜底:单个原子单元本身就超过上限,只能硬切。
            if cur:
                chunks.append("".join(cur))
                cur = []
                cur_tokens = 0
            chunks.extend(_split_oversized(u, cfg.chunk_tokens, count_tokens))
            continue
        if cur and cur_tokens + t > cfg.chunk_tokens:
            chunks.append("".join(cur))
            # 重叠:把当前 chunk 结尾不超过 overlap_tokens 的正文句子留作下一个 chunk 开头
            kept: list[str] = []
            kept_tokens = 0
            for prev in reversed(cur):
                if _is_code(prev):
                    break
                pt = count_tokens(prev)
                if kept_tokens + pt > cfg.overlap_tokens:
                    break
                kept.insert(0, prev)
                kept_tokens += pt
            cur = kept
            cur_tokens = kept_tokens
        cur.append(u)
        cur_tokens += t

    if cur:
        chunks.append("".join(cur))
    return [c for c in chunks if c.strip()]


def _split_oversized(
    unit: str,
    chunk_tokens: int,
    count_tokens: Callable[[str], int],
) -> list[str]:
    """把一个超过 chunk_tokens 的单元硬切成若干 <= chunk_tokens 的片段。

    代码块按行切(保持每行完整),普通正文按字符二分切。
    """
    if _is_code(unit):
        return _split_code_block(unit, chunk_tokens, count_tokens)
    return _split_by_chars(unit, chunk_tokens, count_tokens)


def _split_code_block(
    unit: str,
    chunk_tokens: int,
    count_tokens: Callable[[str], int],
) -> list[str]:
    """按行切分超长代码块,保证每一片 <= chunk_tokens、且不切断单行。"""
    lines = unit.split("\n")
    pieces: list[str] = []
    cur: list[str] = []
    cur_tokens = 0
    for line in lines:
        lt = count_tokens(line) + 1  # +1 是换行符
        if cur and cur_tokens + lt > chunk_tokens:
            pieces.append("\n".join(cur))
            cur = []
            cur_tokens = 0
        cur.append(line)
        cur_tokens += lt
    if cur:
        pieces.append("\n".join(cur))
    return [p for p in pieces if p.strip()]


def _split_by_chars(
    unit: str,
    chunk_tokens: int,
    count_tokens: Callable[[str], int],
) -> list[str]:
    """把超长单元按词/字符硬切,保证每片 <= chunk_tokens。

    有空格(英文等)时按"词"边界切,避免把单词劈成两半;无空格(中文)按字符二分。
    """
    if " " in unit:
        return _split_by_words(unit, chunk_tokens, count_tokens)

    pieces: list[str] = []
    rest = unit
    while rest:
        lo, hi, best = 1, len(rest), 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if count_tokens(rest[:mid]) <= chunk_tokens:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        if best == 0:
            best = 1  # 极端情况下至少前进 1 字符,避免死循环
        pieces.append(rest[:best])
        rest = rest[best:]
    return pieces


def _split_by_words(
    unit: str,
    chunk_tokens: int,
    count_tokens: Callable[[str], int],
) -> list[str]:
    """按空格分词,贪心聚合成 <= chunk_tokens 的片段(不切断单词)。"""
    words = unit.split(" ")
    pieces: list[str] = []
    cur: list[str] = []
    cur_tokens = 0
    for w in words:
        wt = count_tokens(w) + 1  # +1 是空格
        if cur and cur_tokens + wt > chunk_tokens:
            pieces.append(" ".join(cur))
            cur = []
            cur_tokens = 0
        cur.append(w)
        cur_tokens += wt
    if cur:
        pieces.append(" ".join(cur))
    return [p for p in pieces if p.strip()]


def _is_code(unit: str) -> bool:
    return unit.lstrip().startswith("```")
