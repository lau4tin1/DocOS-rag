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


def _find_renamed_source(manifest: dict, content_hash: str, exclude: str) -> str | None:
    """在 manifest 里找另一个 source,其内容哈希 == content_hash(改名检测)。"""
    for src, entry in manifest.items():
        if src != exclude and entry.get("hash") == content_hash:
            return src
    return None


def plan_index_update(
    docs: list[dict],
    manifest: dict,
    config_sig: dict,
) -> tuple[dict, set[str], set[str], dict]:
    """规划增量索引:把每个文件分为 未变/改名/需重跑,并算新 manifest 与被删文件。

    manifest 是"上次索引过哪些文件"的权威记录(含 0 片段的空文件)。
    返回 (reuse_map, to_embed, deleted, new_manifest):
      - reuse_map: source -> 复用来源(等于自己 = 未变;不等于 = 改名)
      - to_embed: 需要重新向量化的 source 集合
      - deleted: 真正被删除(未被改名复用)的旧 source 集合
      - new_manifest: 新的 {source: {hash, config}}
    """
    new_manifest: dict = {}
    reuse_map: dict = {}
    to_embed: set[str] = set()

    for doc in docs:
        source = doc["source"]
        h = hash_file(source)
        entry = manifest.get(source, {})
        new_manifest[source] = {"hash": h, "config": config_sig}

        if entry.get("hash") == h and entry.get("config") == config_sig:
            reuse_map[source] = source  # 未变化(即使 0 片段)
        else:
            renamed_from = _find_renamed_source(manifest, h, source)
            if renamed_from:
                reuse_map[source] = renamed_from  # 改名
            else:
                to_embed.add(source)  # 新文件或内容变化

    current = {d["source"] for d in docs}
    reused_old = set(reuse_map.values())
    deleted = set(manifest.keys()) - current - reused_old
    return reuse_map, to_embed, deleted, new_manifest


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
        """增量建索引:只重跑变更的文件;支持改名检测。完成后同步更新内存索引。"""
        docs = load_documents(self.cfg.paths.raw_dir)

        config_sig = {
            "model": self.cfg.embedding.model,
            "chunk_tokens": self.cfg.chunking.chunk_tokens,
            "overlap_tokens": self.cfg.chunking.overlap_tokens,
            "chunker": CHUNK_VERSION,
        }
        manifest = metadata.load_manifest(self.cfg.paths.index_dir)

        old_by_source: dict[str, list[tuple[np.ndarray, dict]]] = {}
        try:
            old_vectors, old_chunks = metadata.load_index(self.cfg.paths.index_dir)
            for v, c in zip(old_vectors, old_chunks):
                old_by_source.setdefault(c["source"], []).append((v, c))
        except FileNotFoundError:
            pass

        if not docs:
            # 目录为空/无支持格式:清空索引
            if manifest:
                metadata.clear_index(self.cfg.paths.index_dir)
                self.index = NumpyIndex()
                self.chunks = []
            return 0

        reuse_map, to_embed, deleted, new_manifest = plan_index_update(
            docs, manifest, config_sig
        )
        renamed = {src: dst for src, dst in reuse_map.items() if dst != src}

        if not to_embed and not deleted and not renamed:
            self.reload()
            return len(self.chunks)

        if to_embed:
            embedder = self._embedder()
            msg = (
                f"增量索引:复用 {len(reuse_map)} 个文件,"
                f"重新向量化 {len(to_embed)} 个文件"
            )
            if renamed:
                msg += f",检测到 {len(renamed)} 个改名"
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
                    old_src = reuse_map[source]
                    chunks = [dict(c, source=source) for _, c in old_by_source.get(old_src, [])]
                    vecs = [v for v, _ in old_by_source.get(old_src, [])]
                new_chunks.extend(chunks)
                new_vectors.extend(vecs)
        else:
            # 只有删除/改名,无需向量化
            print(f"检测到 {len(deleted)} 个删除 / {len(renamed)} 个改名,更新索引(无需重新向量化)。")
            new_chunks = []
            new_vectors = []
            for doc in docs:
                source = doc["source"]
                old_src = reuse_map[source]
                new_chunks.extend(dict(c, source=source) for _, c in old_by_source.get(old_src, []))
                new_vectors.extend(v for v, _ in old_by_source.get(old_src, []))

        if not new_chunks:
            metadata.clear_index(self.cfg.paths.index_dir)
            self.index = NumpyIndex()
            self.chunks = []
            return 0

        matrix = np.vstack(new_vectors)
        metadata.save_index(self.cfg.paths.index_dir, matrix, new_chunks)
        metadata.save_manifest(self.cfg.paths.index_dir, new_manifest)
        self.index.add(matrix, new_chunks)
        self.chunks = new_chunks
        return len(new_chunks)

    def delete_file(self, name: str) -> dict:
        """删除一个文件(连同其 chunk 与向量)。name 为文件名。"""
        target = Path(self.cfg.paths.raw_dir) / name
        if target.exists():
            target.unlink()
        n = self.build_index()  # 增量索引会自动移除该文件的 chunk/向量
        return {"deleted": name, "total_chunks": n}

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

    def _rewrite(self, question: str, history: list[dict]) -> str:
        """把追问改写成独立查询(用于检索);无需改写则回退原问题。"""
        if history and self.cfg.retrieval.rewrite_query:
            return llm.rewrite_query(question, history, self.cfg.llm)
        return question

    def answer(
        self,
        question: str,
        history: list[dict],
    ) -> tuple[str, list[tuple[float, dict]]]:
        """给定问题(和之前的对话),返回 (答案, 命中的片段列表)。

        检索用"改写后的独立查询",生成仍用原始问题(结合历史理解本意)。
        """
        query = self._rewrite(question, history)
        results = self.retrieve(query)
        if not results:
            return "还没有索引任何文档,请先上传 .md / .txt / .pdf 文件。", []
        messages = llm.build_chat_messages(results, question, history)
        answer = llm.generate_messages(messages, self.cfg.llm)
        return answer, results

    def stream_answer(self, question: str, history: list[dict]):
        """流式版本:先发 sources 事件,再逐段发 delta,最后发 done。

        事件为 dict:{"type": "sources"|"delta"|"done", ...}。
        sources 事件里带 query(实际用于检索的改写后查询),便于调试。
        """
        query = self._rewrite(question, history)
        results = self.retrieve(query)
        sources = [
            {
                "score": round(score, 4),
                "source": chunk.get("source", ""),
                "section": chunk.get("section", ""),
                "text": chunk.get("text", "")[:300],
            }
            for score, chunk in results
        ]
        yield {"type": "sources", "sources": sources, "query": query}

        if not results:
            yield {"type": "delta", "text": "还没有索引任何文档,请先上传 .md / .txt / .pdf 文件。"}
            yield {"type": "done"}
            return

        messages = llm.build_chat_messages(results, question, history)
        try:
            for delta in llm.stream_messages(messages, self.cfg.llm):
                yield {"type": "delta", "text": delta}
        except Exception as e:  # 例如没配 key、网络失败
            yield {"type": "delta", "text": f"\n[生成失败:{e}]"}
        yield {"type": "done"}

    def list_documents(self) -> list[dict]:
        """返回已索引文件:名称 + 片段数。"""
        counts = Counter(c["source"] for c in self.chunks)
        return [
            {"name": Path(source).name, "chunks": n}
            for source, n in sorted(counts.items())
        ]
