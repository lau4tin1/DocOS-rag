from __future__ import annotations

import math
import re
from collections import Counter

# 简单分词:英文/数字/下划线按连续串切;中文按单个字切。
# (生产级可用 jieba 做中文分词,这里保持零依赖、便于理解原理。)
_TOKEN_RE = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """把一段文本切成词元(token)列表,统一小写。"""
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """从零实现的 BM25 关键词检索。

    BM25 是信息检索里最经典的"词法匹配"排序公式,核心思想:
      - 一个词在"少数文档"里出现,区分度高,权重(idf)就大;
      - 一个词在本文档里出现越多次,相关度越高,但有饱和(不会无限加分);
      - 长文档天然包含更多词,用平均文档长度做归一化(b)。
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1          # 词频饱和参数
        self.b = b            # 文档长度归一化参数
        self.corpus_size = 0
        self.doc_freqs: Counter = Counter()      # 词 -> 出现在多少个文档里
        self.doc_lengths: list[int] = []         # 每个文档的总 token 数
        self.doc_term_freqs: list[Counter] = []  # 每个文档里各词的词频
        self.avgdl = 0.0                         # 平均文档长度

    def fit(self, corpus: list[str]) -> None:
        """根据语料建立统计量。"""
        self.corpus_size = len(corpus)
        self.doc_lengths = []
        self.doc_term_freqs = []
        self.doc_freqs = Counter()

        for text in corpus:
            terms = tokenize(text)
            tf = Counter(terms)
            self.doc_lengths.append(len(terms))
            self.doc_term_freqs.append(tf)
            for term in tf:
                self.doc_freqs[term] += 1

        total = sum(self.doc_lengths)
        self.avgdl = total / self.corpus_size if self.corpus_size else 0.0

    def _idf(self, term: str) -> float:
        df = self.doc_freqs.get(term, 0)
        # 出现越多 -> idf 越小;这里用平滑版 idf
        return math.log(1 + (self.corpus_size - df + 0.5) / (df + 0.5))

    def score(self, query: str, doc_index: int) -> float:
        tf = self.doc_term_freqs[doc_index]
        dl = self.doc_lengths[doc_index]
        total = 0.0
        for term in set(tokenize(query)):  # 查询词去重,每个词只算一次
            f = tf.get(term, 0)
            if f == 0:
                continue
            denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            total += self._idf(term) * (f * (self.k1 + 1)) / denom
        return total

    def search(self, query: str) -> list[tuple[float, int]]:
        """返回 [(score, doc_index), ...],按分数降序,包含全部文档。"""
        scored = [(self.score(query, i), i) for i in range(self.corpus_size)]
        scored.sort(key=lambda x: -x[0])
        return scored
