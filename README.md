# DocRAG

## 安装

```bash
cd DocOS-rag
python3 -m venv .venv
source .venv/bin/activate
pip install -e .            # 生成 docrag / docrag-web 命令
```

> 首次运行会下载两个模型:`bge-small-zh-v1.5`(embedding,~100MB)、
> `bge-reranker-base`(精排,~1.1GB)。下载慢可设 `export HF_ENDPOINT=https://hf-mirror.com`。

## 配置

编辑 `config.yaml`。主要改动是 LLM:

```yaml
llm:
  provider: deepseek        # deepseek | openai | anthropic
  model: deepseek-chat
```

常用配置项:

| 配置项 | 默认 | 说明 |
|---|---|---|
| `embedding.device` | `mps` | Apple 用 mps;NVIDIA 用 cuda;其余 cpu |
| `chunking.chunk_tokens` | `256` | 每片段最大 token 数(上限 512) |
| `chunking.overlap_tokens` | `32` | 相邻片段重叠 token 数 |
| `retrieval.top_k` | `4` | 最终送给 LLM 的片段数 |
| `retrieval.hybrid` | `true` | 是否混合检索(BM25 + 向量) |
| `retrieval.rerank` | `true` | 是否交叉编码器精排 |
| `llm.provider` / `llm.model` | `deepseek` / `deepseek-chat` | 生成模型 |

## 使用

### 放文档

把文档放进 `data/raw/`(支持子目录),支持 `.md / .markdown / .txt / .pdf`。`.pdf` 只抽文本层,扫描版需 OCR。

### 设置 API Key

```bash
export DEEPSEEK_API_KEY=sk-...   # 或 OPENAI_API_KEY / ANTHROPIC_API_KEY
```

### 命令行

```bash
docrag index                              # 建索引(增量,只跑新/改文件)
docrag list                               # 列出已索引的文件
docrag delete 某个文件.pdf                 # 删除文件及其索引
docrag ask "如何安装 DocOS?"              # 提问
docrag ask "..." --show-sources           # 附带命中的来源
```

### Web 界面

```bash
docrag-web                                # 打开 http://127.0.0.1:8000
```

拖拽上传、多轮对话、流式输出。

## 测试

```bash
pip install -e ".[dev]"
pytest
```

## 设计原理

见 [docs/DESIGN.md](docs/DESIGN.md)。
