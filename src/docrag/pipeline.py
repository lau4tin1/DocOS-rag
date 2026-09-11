from __future__ import annotations

from .config import Config
from .embed.embedder import Embedder
from .generate import llm
from .ingest.chunker import chunk_document
from .ingest.loader import load_documents
from .retrieve.hybrid import HybridRetriever
from .retrieve.reranker import Reranker
from .retrieve.retriever import Retriever
from .store import metadata
from .store.index import NumpyIndex


def build_index(cfg: Config) -> int:
    """离线阶段:读文档 -> 切分 -> 向量化 -> 存索引。返回片段数量。"""
    docs = load_documents(cfg.paths.raw_dir)
    if not docs:
        print(f"警告: {cfg.paths.raw_dir} 下没有找到 .md 文件。")
        return 0

    embedder = Embedder(cfg.embedding)

    all_chunks: list[dict] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, cfg.chunking, embedder.count_tokens))
    if not all_chunks:
        print("警告: 没有产生任何片段,请检查文档内容。")
        return 0

    texts = [c["text"] for c in all_chunks]
    print(f"正在向量化 {len(texts)} 个片段(模型 {cfg.embedding.model})...")
    vectors = embedder.embed(texts)

    metadata.save_index(cfg.paths.index_dir, vectors, all_chunks)
    return len(all_chunks)


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
