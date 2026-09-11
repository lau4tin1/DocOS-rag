# DocRAG — 从零构建的技术文档检索助手

一个不依赖 LangChain 等编排框架的 RAG(检索增强生成)助手:文档 → 切分 → 向量化 → 检索 → 生成答案。每一层都自己实现,目标是**边用边看懂 RAG 的原理**。

核心能力:

- 多格式输入:`.md` / `.markdown` / `.txt` / `.pdf`(文本层),自动剥离 YAML frontmatter
- 结构感知切分:按标题层级切、按 token 计长、代码块不切开、正文带重叠
- 两阶段检索:BM25 + 向量混合召回(RRF 融合)→ 交叉编码器精排
- 增量索引:按内容哈希复用,只重跑变更的文件
- 本地 embedding + 云端 LLM(DeepSeek / OpenAI / Anthropic 可切换)

---

## 目录

- [一、结构设计](#一结构设计)
- [二、方法与问题(为什么这样设计)](#二方法与问题为什么这样设计)
- [三、分块说明](#三分块说明)
- [四、部署使用](#四部署使用)

---

## 一、结构设计

整体是两条流水线:**离线建索引**和**在线问答**。

```
离线(建索引):                         在线(问答):
  data/raw/ 下的文档                    用户提问
        │                                  │
        ▼                                  ▼
 ① Loader 读文件、剥 frontmatter       ① Embedder 查询向量化
        │                                  │
        ▼                                  ▼
 ② Chunker 结构感知切分                ② 粗召回:向量 + BM25 → RRF 融合
        │                                  │  (取出 rerank_top_n 个候选)
        ▼                                  ▼
 ③ Embedder 文本 → 向量                ③ 精排:交叉编码器打分
        │                                  │  (取 top_k)
        ▼                                  ▼
 ④ 存盘 index/ (向量 + 片段 + manifest)  ④ 拼 prompt(片段作为上下文)
                                           │
                                           ▼
                                        ⑤ LLM 生成答案
```

### 目录结构

```
DocOS-rag/
├── pyproject.toml            # 项目配置与依赖
├── requirements.txt          # 依赖清单
├── config.yaml               # 所有可调参数(模型、切分、检索、LLM)
├── cli.py                    # 免安装入口:python cli.py index/ask
├── data/raw/                 # 原始文档放这里
├── index/                    # 生成的索引(可随时删除重建,已被 gitignore)
├── src/docrag/
│   ├── config.py             # 读取 config.yaml 到结构化配置
│   ├── ingest/
│   │   ├── loader.py         # 多格式加载 + frontmatter 剥离
│   │   └── chunker.py        # 结构感知切分
│   ├── embed/
│   │   └── embedder.py       # 文本→向量;数 token
│   ├── store/
│   │   ├── index.py          # numpy 余弦检索
│   │   └── metadata.py       # 索引持久化 + 增量 manifest
│   ├── retrieve/
│   │   ├── retriever.py      # 纯向量检索
│   │   ├── bm25.py           # BM25 关键词检索
│   │   ├── hybrid.py         # 向量+BM25 的 RRF 融合
│   │   └── reranker.py       # 交叉编码器精排
│   ├── generate/
│   │   └── llm.py            # prompt 组装 + 调用 LLM
│   ├── pipeline.py           # 把上面串成 build_index / ask
│   └── cli.py                # 命令行入口
└── tests/                    # 单元测试(26 个,pytest)
```

### 分层原则

1. **接口隔离点只有两个**:`embedder.py`(向量化)和 `llm.py`(生成)。想换向量模型或换 LLM,只改这两个文件,其余层不动。
2. **索引是"可重建的副产品"**:原始文档是唯一真相,`index/` 随时能删掉重建,不应被当作数据源。
3. **依赖注入**:`chunker` 需要的 `count_tokens`、`HybridRetriever` 需要的 embedder,都通过参数传入而非内部硬编码。这既让"切分和向量化用同一个 tokenizer"成为强制约束,也让测试能用假对象替代重模型。

---

## 二、方法与问题(为什么这样设计)

按流水线顺序,每一条都是"遇到什么问题 → 用什么方法 → 为什么"。

### 2.1 为什么要把文档切分成 chunk

**问题**:不能把整篇文档直接发给模型。三个原因:① embedding 模型有硬上限(本项目的 BGE 是 **512 token**);② 一个向量只能表达一个主题,整篇文档的语义会被稀释;③ 检索粒度越细,越能精确命中"你需要的那一段"。

**方法**:把文档切成几百 token 的片段(chunk),每个 chunk 独立向量化、独立检索。

### 2.2 为什么按"标题结构"切,而不是等宽硬切

**问题**:等宽硬切会把不同主题糊进同一个片段,也可能把一句话从中间劈断。

**方法**:先用 Markdown 标题(`#`/`##`/`###`)把正文分到不同"小节",记录层级路径 `一级 > 二级 > 三级` 存进 chunk 的 `section` 字段;小节仍超长时再按句子切。标题是天然的主题边界。

### 2.3 为什么用 token 数,而不是字符数

**问题**:模型的限制是 **token**,不是字符。两者不是一回事:中文一个词可能多个字打包成一个 token,英文一个长单词往往只占 1 个 token。用"512 字符"当上限,中文可能溢出被静默截断,英文又填不满浪费。

**方法**:用**模型自己的 tokenizer** 数 token(通过 `embedder.count_tokens` 注入到 chunker)。`chunk_tokens=256`,给 512 上限留足余量(还要给 `[CLS]`/`[SEP]` 两个固定 token 腾位置)。

### 2.4 为什么 overlap,且只给正文、不给代码

**问题**:按句子切分可能在边界把"一个完整的意思"劈成两半——后半句在一个 chunk,前半句在另一个,命中一半却丢了另一半语境。

**方法**:相邻 chunk 保留 `overlap_tokens` 的重叠(把上一个 chunk 结尾的句子复制到下个 chunk 开头)。

**但代码不参与重叠**:代码是"按行"切的,一行代码本身就是完整单元,不存在"劈断"问题;且代码靠精确 token(函数名/错误码)检索,复制重叠行既浪费上下文又可能让 LLM 看到重复代码。所以重叠回退时遇到代码块就停。

### 2.5 为什么要保护代码块 + 超长硬切

**问题**:技术文档的核心是代码。按句号切分会把 `def f():` 这样的代码劈得面目全非。

**方法**:` ``` ` 围栏内的代码作为"原子单元",永不按句子切。

**兜底**:当单个代码块本身就超过 `chunk_tokens`(比如 900 token 的巨型示例),再坚持"不切开"就会被模型静默截断。此时按**行**硬切,保证每片 ≤ 上限且不切断单行。

### 2.6 为什么要给向量归一化

**问题**:余弦相似度公式是 `cos(a,b) = a·b / (|a||b|)`,直接算要做两次模长和一次除法。

**方法**:把向量 L2 归一化成单位向量(|a|=|b|=1),分母变成 1,于是 **余弦 = 点积**。检索就退化成一行矩阵乘法 `scores = V @ q`,既快又简洁。

### 2.7 为什么混合检索(BM25 + 向量),而不是只用向量

**问题**:向量擅长"语义相近"(换个说法也能命中),但对**精确关键词**(错误码 `ERR_...`、函数名、ID)不敏感——这些 token 在预训练里没有语义信号;BM25 正好相反,擅长精确词匹配、不懂同义改写。二者互补。

**方法**:两个都算,用 **RRF(Reciprocal Rank Fusion)** 融合:

```
rrf(d) = 1/(k + rank_向量(d)) + 1/(k + rank_BM25(d))
```

**为什么 RRF 而不是加权相加**:向量余弦分数在 `[0,1]`、BM25 无上限,尺度不可比,直接相加没意义。RRF 只依赖"排名",天然规避尺度问题,无需调权。

### 2.8 为什么要两阶段 rerank

**问题**:embedding 是"双编码器"——query 和文档各自编码成向量再算余弦,两者从不"互看",只能捕捉粗粒度语义相近。它快(文档向量可预计算),但不够准。

**方法**:先粗召回 `rerank_top_n` 个候选,再用**交叉编码器**(把 `[CLS] query [SEP] doc` 拼成一条一起过 Transformer)对候选逐一精排,取 top_k。交叉编码器准(能判断"是否真的回答了这个具体问题"),但每对都要现算、无法预计算,所以只用在小的候选集上——这就是"两阶段检索"。

### 2.9 为什么要增量索引

**问题**:文档一多,每次 `docrag index` 全量重跑太慢。

**方法**:给每个文件算 **SHA-256 内容哈希** + 记录当时的**切分/模型配置**,存进 `index/manifest.json`。下次建索引时:哈希和配置都没变 → 复用旧向量;变了 → 只重跑该文件;被删的文件自动移除。配置变化(如改 `chunk_tokens`、换模型、改切分代码版本)会自动触发全量重建。

### 2.10 为什么本地 embedding + 云端 LLM

**问题**:embedding 用小模型即可,本地跑免费、离线、可控;LLM 需要大模型,本地 8G 内存跑不动。

**方法**:embedding 用本地 `bge-small-zh`(MPS 加速),生成走 DeepSeek 等 API。这个分工也正好匹配现状:**DeepSeek 目前没有 embedding 接口**,只有生成接口。

---

## 三、分块说明

| 模块 | 文件 | 职责与要点 |
|---|---|---|
| 配置 | `config.py` | 读 `config.yaml` 到 dataclass;相对路径解析到项目根目录 |
| 加载 | `ingest/loader.py` | `.md/.txt` 读文本+剥 frontmatter+提标题;`.pdf` 用 pypdf 抽文本层;`hash_file` 供增量用 |
| 切分 | `ingest/chunker.py` | 标题分节 → 句子切分 → token 贪心合并(正文带重叠、代码不切、超长硬切);`CHUNK_VERSION` 供增量感知 |
| 向量化 | `embed/embedder.py` | `SentenceTransformer` 封装;`embed` 归一化向量化;`embed_query` 加查询前缀;`count_tokens` 数 token |
| 存储 | `store/index.py` | `NumpyIndex`:归一化后点积 = 余弦,取 top-k |
| 存储 | `store/metadata.py` | `save/load_index`(npy+json);`save/load_manifest`(增量清单) |
| 检索 | `retrieve/bm25.py` | 从零实现 BM25:分词、idf、词频饱和、文档长度归一化 |
| 检索 | `retrieve/retriever.py` | 纯向量检索(查询→向量→top-k) |
| 检索 | `retrieve/hybrid.py` | `rrf_fuse` 纯函数 + `HybridRetriever` 组合向量与 BM25 |
| 检索 | `retrieve/reranker.py` | 交叉编码器精排(两阶段检索的第二阶段) |
| 生成 | `generate/llm.py` | `build_prompt` 拼上下文;`generate` 按 provider 分发到 OpenAI/DeepSeek/Anthropic |
| 编排 | `pipeline.py` | `build_index`(加载→切分→向量化→存盘,增量);`ask`(召回→精排→拼 prompt→生成) |
| 命令行 | `cli.py` | `docrag index` / `docrag ask "..." [--show-sources]` |
| 测试 | `tests/` | 26 个单元测试,覆盖切分/BM25/索引/混合/rerank/llm/loader/config,秒级、不加载模型 |

---

## 四、部署使用

### 4.1 环境要求

- Python ≥ 3.10
- 内存 ≥ 8G(本地只跑 embedding 小模型,足够;LLM 在云端)
- Apple 芯片(M1/M2/M3)可用 MPS 加速;NVIDIA 用 CUDA;其余用 CPU

### 4.2 安装

```bash
cd DocOS-rag
python3 -m venv .venv
source .venv/bin/activate

pip install -e .            # 安装依赖 + 生成 docrag 命令
# pip install -e ".[dev]"   # 连测试依赖一起装
```

> 首次运行会从 HuggingFace 下载两个模型:`bge-small-zh-v1.5`(embedding,约 100MB)
> 和 `bge-reranker-base`(精排,约 1.1GB)。下载慢可设镜像:
> `export HF_ENDPOINT=https://hf-mirror.com`

### 4.3 配置

所有参数在 `config.yaml`:

| 配置项 | 默认 | 说明 |
|---|---|---|
| `embedding.model` | `BAAI/bge-small-zh-v1.5` | 本地向量模型 |
| `embedding.device` | `mps` | Apple 芯片用 mps;NVIDIA 用 cuda;其余 cpu |
| `chunking.chunk_tokens` | `256` | 每个片段的最大 token 数(上限 512,留余量) |
| `chunking.overlap_tokens` | `32` | 相邻正文片段的重叠 token 数 |
| `retrieval.top_k` | `4` | 最终送给 LLM 的片段数 |
| `retrieval.hybrid` | `true` | 是否混合检索(BM25+向量) |
| `retrieval.rrf_k` | `60` | RRF 融合常数 |
| `retrieval.rerank` | `true` | 是否交叉编码器精排 |
| `retrieval.rerank_top_n` | `12` | 粗召回候选数,精排后取 top_k |
| `llm.provider` | `deepseek` | `deepseek` / `openai` / `anthropic` |
| `llm.model` | `deepseek-chat` | 生成模型 |
| `paths.raw_dir` | `data/raw` | 文档目录 |
| `paths.index_dir` | `index` | 索引目录 |

### 4.4 放文档

把文档丢进 `data/raw/`(支持 `.md` / `.markdown` / `.txt` / `.pdf`),会递归扫描子目录。

> `.pdf` 只能抽**文本层**;扫描版 PDF(纯图片)需要 OCR,不在支持范围。

### 4.5 建索引

```bash
docrag index
# 或未安装时:python cli.py index
```

- 首次运行会全量构建;
- 之后是**增量**:只重跑新增/修改的文件,未变化的秒过,删除的文件自动移除;
- `index/` 可随时删除重建。

### 4.6 提问

先设置 LLM 的 API Key(默认 DeepSeek):

```bash
export DEEPSEEK_API_KEY=sk-...     # provider=deepseek(默认)
# export OPENAI_API_KEY=sk-...     # provider=openai
# export ANTHROPIC_API_KEY=...     # provider=anthropic
```

```bash
docrag ask "如何安装 DocOS?"
docrag ask "安装时报权限错误怎么办?" --show-sources
```

`--show-sources` 会打印命中的片段来源、章节、得分和内容预览,方便观察检索质量。

### 4.7 测试

```bash
pip install -e ".[dev]"
pytest
```

测试只用少量合成输入,不加载模型、不联网,秒级完成。重依赖(embedding/交叉编码器)在测试里用假对象注入。

### 4.8 常见问题

- **需要 Docker 吗?** 不需要。`.venv` 已提供依赖隔离;只有当你要部署成常驻服务或引入向量数据库时,才值得引入 Docker。
- **能跑在 8G 内存的 M1 上吗?** 能。本地只跑 embedding 小模型,LLM 在云端。
- **DeepSeek 能做 embedding 吗?** 不能,DeepSeek 没有 embedding 接口,所以向量化本地跑、生成走 API。
- **改模型/切分参数后需要手动重建吗?** 不需要,配置签名变了会自动全量重建。
