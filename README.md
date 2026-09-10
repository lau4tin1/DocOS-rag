# DocRAG — 从零构建的技术文档检索助手

不依赖 LangChain 等编排框架,自己写每一层,便于学习 RAG 的原理。

## 流水线

```
建索引(离线):  Markdown 文档 -> Loader -> Chunker -> Embedder -> 向量索引(index/)
问答(在线):    用户提问 -> Embedder -> 向量检索 top-k -> 拼 prompt -> LLM -> 答案
```

对应代码分层:

| 层 | 文件 | 职责 |
|---|---|---|
| ingest | `src/docrag/ingest/loader.py` / `chunker.py` | 读文件、按标题结构切分片段 |
| embed | `src/docrag/embed/embedder.py` | 文本 -> 向量(本地 BGE 模型) |
| store | `src/docrag/store/index.py` / `metadata.py` | numpy 余弦检索 + 持久化 |
| retrieve | `src/docrag/retrieve/retriever.py` | 查询向量 -> top-k 片段 |
| generate | `src/docrag/generate/llm.py` | 组装 prompt + 调 LLM API |
| pipeline | `src/docrag/pipeline.py` | 串起 index / ask 两个入口 |

## 安装

```bash
# 建议使用虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 方式一:可编辑安装(会生成 docrag 命令)
pip install -e .

# 方式二:只装依赖
pip install -r requirements.txt
```

> `sentence-transformers` 会拉取 PyTorch,体积较大;首次运行会从 HuggingFace
> 下载 `BAAI/bge-small-zh-v1.5` 模型。若下载慢,可先设置镜像:
> `export HF_ENDPOINT=https://hf-mirror.com`
>
> Apple 芯片(M1/M2/M3)已把 `config.yaml` 里的 `device` 设成 `mps` 以用 GPU 加速;
> 非 Apple 芯片请改回 `cpu`。

## 使用

### 1. 放文档

把 Markdown 文档放进 `data/raw/`(已内置一个示例 `DocOS-安装指南.md`)。

### 2. 建索引

```bash
docrag index
# 或未安装时:python cli.py index
```

### 3. 提问

先设置 LLM 的 API Key。默认用 DeepSeek(OpenAI 兼容接口):

```bash
export DEEPSEEK_API_KEY=sk-...    # provider 选 deepseek 时(默认)
# export OPENAI_API_KEY=sk-...    # provider 选 openai 时
# export ANTHROPIC_API_KEY=...    # provider 选 anthropic 时
```

```bash
docrag ask "如何安装 DocOS?"
docrag ask "安装时报权限错误怎么办?" --show-sources
```

`--show-sources` 会打印命中的片段来源和内容预览,方便你观察检索效果。

> 说明:DeepSeek 目前**没有 embedding 接口**,所以向量化仍然用本地 BGE 模型,
> 只有"生成答案"这一步走 DeepSeek API。这个分工正好符合我们的设计。

## 配置

所有可调参数在 `config.yaml`(embedding 模型、chunk 的 token 数、top-k、LLM 等)。

## 下一步的改进方向

- 切分:超长单句/超长代码块的硬切、剥离 YAML frontmatter
- 检索:BM25 关键词检索 + 向量检索混合、rerank 重排、按目录/版本做元数据过滤
- 索引:numpy 换成 faiss 以支持大规模片段
- 生成:流式输出、多轮对话
