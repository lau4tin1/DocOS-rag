# DocRAG — 从零构建的技术文档检索助手

本地 RAG(检索增强生成)助手,不依赖 LangChain 等框架,每一层自己实现,边用边学原理。

## 特性

- 多格式:`.md` / `.markdown` / `.txt` / `.pdf`(文本层),自动剥离 YAML frontmatter
- 结构感知切分:按标题分节、按 token 计长、代码块不切开、正文带重叠
- 两阶段检索:BM25 + 向量混合召回(RRF)→ 交叉编码器精排
- 多轮对话:查询改写 + 流式输出(Web 界面)
- 增量索引:按内容哈希复用,只重跑变更文件;支持**改名检测**与**文件删除**
- 本地 embedding + 云端 LLM(DeepSeek / OpenAI / Anthropic)

## 快速开始

### 1. 安装

```bash
cd DocOS-rag
python3 -m venv .venv
source .venv/bin/activate
pip install -e .            # 生成 docrag / docrag-web 两个命令
```

> 首次运行会下载两个模型:`bge-small-zh-v1.5`(embedding,~100MB)、
> `bge-reranker-base`(精排,~1.1GB)。下载慢可设 `export HF_ENDPOINT=https://hf-mirror.com`。
> Apple 芯片默认用 `mps` 加速(见 `config.yaml`)。

### 2. 配置

编辑 `config.yaml`,主要是:

```yaml
llm:
  provider: deepseek        # deepseek | openai | anthropic
  model: deepseek-chat
```

其余参数(chunk 大小、top-k、是否 rerank 等)都有默认值,一般不用改。

### 3. 放文档

把文档放进 `data/raw/`(支持子目录)。`.pdf` 只能抽文本层,扫描版需 OCR(不支持)。

### 4. 设置 API Key

```bash
export DEEPSEEK_API_KEY=sk-...   # 或 OPENAI_API_KEY / ANTHROPIC_API_KEY
```

### 5. 使用

命令行:

```bash
docrag index                              # 建索引(增量,只跑新/改文件)
docrag list                               # 列出已索引的文件
docrag delete 某个文件.pdf                 # 删除文件及其 chunk/向量
docrag ask "如何安装 DocOS?"              # 提问
docrag ask "..." --show-sources           # 附带命中的来源
```

Web 界面(拖拽上传 + 多轮对话 + 流式):

```bash
docrag-web                                # 打开 http://127.0.0.1:8000
```

## 常用配置

| 配置项 | 默认 | 说明 |
|---|---|---|
| `embedding.model` | `BAAI/bge-small-zh-v1.5` | 本地向量模型 |
| `embedding.device` | `mps` | Apple 用 mps;NVIDIA 用 cuda;其余 cpu |
| `chunking.chunk_tokens` | `256` | 每片段最大 token 数(上限 512,留余量) |
| `chunking.overlap_tokens` | `32` | 相邻正文片段重叠 token 数 |
| `retrieval.top_k` | `4` | 最终送给 LLM 的片段数 |
| `retrieval.hybrid` | `true` | 是否混合检索(BM25 + 向量) |
| `retrieval.rerank` | `true` | 是否交叉编码器精排 |
| `retrieval.rewrite_query` | `true` | 多轮追问时是否先改写为独立查询 |
| `llm.provider` | `deepseek` | `deepseek` / `openai` / `anthropic` |
| `llm.model` | `deepseek-chat` | 生成模型 |

## 测试

```bash
pip install -e ".[dev]"
pytest
```

## 设计原理

为什么这样切分、为什么混合检索、为什么增量索引……详见 [docs/DESIGN.md](docs/DESIGN.md)。

## 常见问题

- **需要 Docker 吗?** 不需要,`.venv` 已提供依赖隔离;部署成服务/上向量数据库时才值得引入。
- **需要数据库吗?** 目前不需要,文件系统 + 内存即可。
- **8G 内存的 M1 能跑吗?** 能,本地只跑 embedding 小模型,LLM 在云端。
- **DeepSeek 能做 embedding 吗?** 不能,所以向量化本地跑、生成走 API。
