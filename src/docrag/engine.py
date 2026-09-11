from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from .config import Config
from .embed.embedder import Embedder
from .generate import llm
from .ingest.chunker import CHUNK_VERSION, chunk_document
from .ingest.loader import hash_file, load_documents
from .retrieve.hybrid import HybridRetriever
from .retrieve.reranker import Reranker
from .retrieve.retriever import Retriever
from .store import metadata
from .store.index import NumpyIndex


class RAGEngine:
    """状态化的 RAG 引擎:常驻持有模型与索引,供 web 服务复用。

    与 CLI 的"一次性"不同,这里模型只加载一次、索引常驻内存,
    避免每次问答都重新加载 embedding / reranker 模型(那要几秒)。
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.embedder: Embedder | None = None
        self.reranker: Reranker | None = None
        self.index = NumpyIndex()
        self.chunks: list[dict] = []

    # ---------- 模型(懒加载) ----------

    def _embedder(self) -> Embedder:
        if self.embedder is None:
            self.embedder = Embedder(self.cfg.embedding)
        return self.embedder

    def _reranker(self) -> Reranker:
        if self.reranker is None:
            self.reranker = Reranker(self.cfg.retrieval.rerank_model)
        return self.reranker

    # ---------- 索引 ----------

    def reload(self) -> int:
        """从磁盘加载索引到内存;不存在则置空。返回片段数。"""
        try:
            vectors, chunks = metadata.load_index(self.cfg.paths.index_dir)
        except FileNotFoundError:
            self.index = NumpyIndex()
            self.chunks = []
            return 0
        self.index.add(vectors, chunks)
        self.chunks = chunks
        return len(chunks)

    def build_index(self) -> int:
        """增量建索引:只重跑变更的文件。完成后同步更新内存中的索引。"""
        docs = load_documents(self.cfg.paths.raw_dir)
        if not docs:
            return 0

        config_sig = {
            "model": self.cfg.embedding.model,
            "chunk_tokens": self.cfg.chunking.chunk_tokens,
            "overlap_tokens": self.cfg.chunking.overlap_tokens,
            "chunker": CHUNK_VERSION,
        }
        manifest = metadata.load_manifest(self.cfg.paths.index_dir)

        old_by_source: dict[str, list[tuple[np.ndarray, dict]]] = {}
        old_vectors = None
        old_chunks: list[dict] = []
        try:
            old_vectors, old_chunks = metadata.load_index(self.cfg.paths.index_dir)
            for v, c in zip(old_vectors, old_chunks):
                old_by_source.setdefault(c["source"], []).append((v, c))
        except FileNotFoundError:
            pass

        new_manifest: dict = {}
        to_embed: set[str] = set()
        for doc in docs:
            source = doc["source"]
            h = hash_file(source)
            entry = manifest.get(source, {})
            unchanged = (
                old_vectors is not None
                and entry.get("hash") == h
                and entry.get("config") == config_sig
                and source in old_by_source
            )
            new_manifest[source] = {"hash": h, "config": config_sig}
            if not unchanged:
                to_embed.add(source)

        old_sources = set(old_by_source.keys())
        current_sources = {doc["source"] for doc in docs}
        deleted = old_sources - current_sources

        if not to_embed and not deleted:
            self.reload()
            return len(self.chunks)

        if to_embed:
            embedder = self._embedder()
            msg = (
                f"增量索引:复用 {len(docs) - len(to_embed)} 个文件,"
                f"重新向量化 {len(to_embed)} 个文件"
            )
            if deleted:
                msg += f",移除 {len(deleted)} 个已删除文件"
            print(msg)

            new_chunks: list[dict] = []
            new_vectors: list[np.ndarray] = []
            for doc in docs:
                source = doc["source"]
                if source in to_embed:
                    chunks = chunk_document(doc, self.cfg.chunking, embedder.count_tokens)
                    vecs = embedder.embed([c["text"] for c in chunks])
                else:
                    chunks = [c for _, c in old_by_source[source]]
                    vecs = [v for v, _ in old_by_source[source]]
                new_chunks.extend(chunks)
                new_vectors.extend(vecs)
        else:
            # 只有删除,无需向量化
            print(f"检测到 {len(deleted)} 个文件被删除,更新索引(无需重新向量化)。")
            new_chunks = []
            new_vectors = []
            for doc in docs:
                source = doc["source"]
                new_chunks.extend(c for _, c in old_by_source[source])
                new_vectors.extend(v for v, _ in old_by_source[source])

        if not new_chunks:
            metadata.save_index(self.cfg.paths.index_dir, np.zeros((0, 512), dtype="float32"), [])
            metadata.save_manifest(self.cfg.paths.index_dir, new_manifest)
            self.index = NumpyIndex()
            self.chunks = []
            return 0

        matrix = np.vstack(new_vectors)
        metadata.save_index(self.cfg.paths.index_dir, matrix, new_chunks)
        metadata.save_manifest(self.cfg.paths.index_dir, new_manifest)
        self.index.add(matrix, new_chunks)
        self.chunks = new_chunks
        return len(new_chunks)

    # ---------- 检索与生成 ----------

    def retrieve(self, question: str) -> list[tuple[float, dict]]:
        """两阶段检索:粗召回(向量/混合)-> 精排(rerank)。返回 (score, chunk) 列表。"""
        n = len(self.chunks)
        if n == 0:
            return []

        embedder = self._embedder()
        if self.cfg.retrieval.hybrid:
            retriever = HybridRetriever(
                embedder, self.index, self.chunks, rrf_k=self.cfg.retrieval.rrf_k
            )
        else:
            retriever = Retriever(embedder, self.index)

        recall_n = (
            self.cfg.retrieval.rerank_top_n
            if self.cfg.retrieval.rerank
            else self.cfg.retrieval.top_k
        )
        candidates = retriever.retrieve(question, recall_n)

        if self.cfg.retrieval.rerank and len(candidates) > self.cfg.retrieval.top_k:
            reranker = self._reranker()
            return reranker.rerank(question, candidates, self.cfg.retrieval.top_k)
        return candidates[: self.cfg.retrieval.top_k]

    def answer(
        self,
        question: str,
        history: list[dict],
    ) -> tuple[str, list[tuple[float, dict]]]:
        """给定问题(和之前的对话),返回 (答案, 命中的片段列表)。"""
        results = self.retrieve(question)
        if not results:
            return "还没有索引任何文档,请先上传 .md / .txt / .pdf 文件。", []
        messages = llm.build_chat_messages(results, question, history)
        answer = llm.generate_messages(messages, self.cfg.llm)
        return answer, results

    def list_documents(self) -> list[dict]:
        """返回已索引文件:名称 + 片段数。"""
        counts = Counter(c["source"] for c in self.chunks)
        return [
            {"name": Path(source).name, "chunks": n}
            for source, n in sorted(counts.items())
        ]
