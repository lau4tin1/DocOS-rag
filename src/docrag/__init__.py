"""DocRAG: 从零构建的技术文档检索助手。

模块分层(对应 RAG 流水线):
    ingest     -> 读文件、切分片段
    embed      -> 文本转向量
    store      -> 向量索引与持久化
    retrieve   -> 相似度检索 top-k
    generate   -> 组装 prompt 并调用 LLM
    pipeline   -> 把上面串成 index / ask 两个入口
"""

__version__ = "0.1.0"
