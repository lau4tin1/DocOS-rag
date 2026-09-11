from __future__ import annotations

import numpy as np

from .config import Config
from .embed.embedder import Embedder
from .generate import llm
from .ingest.chunker import chunk_document
from .ingest.loader import hash_file, load_documents
from .retrieve.hybrid import HybridRetriever
from .retrieve.reranker import Reranker
from .retrieve.retriever import Retriever
from .store import metadata
from .store.index import NumpyIndex


def build_index(cfg: Config) -> int:
    """离线阶段:读文档 -> 切分 -> 向量化 -> 存索引。支持增量。

    返回片段总数。判断"文件是否变化"用的是内容哈希(SHA-256)+ 切分/模型配置:
      - 哈希相同且配置相同 -> 复用旧索引,不重新向量化;
      - 否则 -> 重新切分 + 向量化;
      - 已删除的文件会自动从索引里消失(只保留当前存在的文件)。
    """
    docs = load_documents(cfg.paths.raw_dir)
    if not docs:
        print(f"警告: {cfg.paths.raw_dir} 下没有找到 .md 文件。")
        return 0

    config_sig = {
        "model": cfg.embedding.model,
        "chunk_tokens": cfg.chunking.chunk_tokens,
        "overlap_tokens": cfg.chunking.overlap_tokens,
    }
    manifest = metadata.load_manifest(cfg.paths.index_dir)

    # 旧索引按 source 分组,便于复用
    old_by_source: dict[str, list[tuple[np.ndarray, dict]]] = {}
    old_vectors = None
    old_chunks: list[dict] = []
    try:
        old_vectors, old_chunks = metadata.load_index(cfg.paths.index_dir)
        for v, c in zip(old_vectors, old_chunks):
            old_by_source.setdefault(c["source"], []).append((v, c))
    except FileNotFoundError:
        pass  # 第一次建索引,没有旧数据

    # 第一遍:判断每个文件复用 or 重跑,并算出新的 manifest
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
        print(f"所有 {len(docs)} 个文件均未变化,索引已是最新,跳过重建。")
        return len(old_chunks)

    # 只有存在需要重新向量化的文件时才加载 embedding 模型
    if to_embed:
        embedder = Embedder(cfg.embedding)
        msg = (
            f"增量索引:复用 {len(docs) - len(to_embed)} 个文件,"
            f"重新向量化 {len(to_embed)} 个文件"
        )
        if deleted:
            msg += f",移除 {len(deleted)} 个已删除文件"
        print(msg + f"(模型 {cfg.embedding.model})...")
    else:
        print(f"检测到 {len(deleted)} 个文件被删除,更新索引(无需重新向量化)...")

    # 第二遍:按文档顺序组装(复用旧的 / 嵌入新的;已删除的文件自然被丢弃)
    new_chunks: list[dict] = []
    new_vectors: list[np.ndarray] = []
    for doc in docs:
        source = doc["source"]
        if source in to_embed:
            chunks = chunk_document(doc, cfg.chunking, embedder.count_tokens)
            vecs = embedder.embed([c["text"] for c in chunks])
        else:
            chunks = [c for _, c in old_by_source[source]]
            vecs = [v for v, _ in old_by_source[source]]
        new_chunks.extend(chunks)
        new_vectors.extend(vecs)

    if not new_chunks:
        print("警告: 没有产生任何片段,请检查文档内容。")
        return 0

    matrix = np.vstack(new_vectors)
    metadata.save_index(cfg.paths.index_dir, matrix, new_chunks)
    metadata.save_manifest(cfg.paths.index_dir, new_manifest)
    return len(new_chunks)


def ask(question: str, cfg: Config) -> tuple[str, list[tuple[float, dict]]]:
    """在线阶段:查询向量化 -> 检索 top-k -> 拼 prompt -> 调 LLM。"""
    vectors, chunks = metadata.load_index(cfg.paths.index_dir)

    index = NumpyIndex()
    index.add(vectors, chunks)

    embedder = Embedder(cfg.embedding)
    if cfg.retrieval.hybrid:
        retriever = HybridRetriever(
            embedder, index, chunks, rrf_k=cfg.retrieval.rrf_k
        )
    else:
        retriever = Retriever(embedder, index)

    # 两阶段检索:粗召回(快但糙)-> 交叉编码器精排(慢但准)
    recall_n = (
        cfg.retrieval.rerank_top_n if cfg.retrieval.rerank else cfg.retrieval.top_k
    )
    candidates = retriever.retrieve(question, recall_n)

    if cfg.retrieval.rerank and len(candidates) > cfg.retrieval.top_k:
        reranker = Reranker(cfg.retrieval.rerank_model)
        results = reranker.rerank(question, candidates, cfg.retrieval.top_k)
    else:
        results = candidates[: cfg.retrieval.top_k]

    system, user = llm.build_prompt(results, question)
    answer = llm.generate(system, user, cfg.llm)
    return answer, results
