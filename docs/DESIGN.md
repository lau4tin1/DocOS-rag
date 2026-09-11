# 设计原理

本文说明 DocRAG 的架构设计、每个设计决策背后的"为什么",以及各模块职责。快速上手见 [README](../README.md)。

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
├── config.yaml               # 所有可调参数(模型、切分、检索、LLM)
├── cli.py                    # 免安装入口:python cli.py index/ask
├── data/raw/                 # 原始文档放这里
├── index/                    # 生成的索引(可随时删除重建,已被 gitignore)
├── src/docrag/
│   ├── config.py             # 读取 config.yaml 到结构化配置
│   ├── ingest/
│   │   ├── loader.py         # 多格式加载 + frontmatter 剥离
│   │   └── chunker.py        # 结构感知切分
│   ├── embed/embedder.py     # 文本→向量;数 token
│   ├── store/
│   │   ├── index.py          # numpy 余弦检索
│   │   └── metadata.py       # 索引持久化 + 增量 manifest
│   ├── retrieve/
│   │   ├── retriever.py      # 纯向量检索
│   │   ├── bm25.py           # BM25 关键词检索
│   │   ├── hybrid.py         # 向量+BM25 的 RRF 融合
│   │   └── reranker.py       # 交叉编码器精排
│   ├── generate/llm.py       # prompt 组装 + 调用 LLM(多轮/流式)
│   ├── engine.py             # RAGEngine:常驻模型 + 索引 + 检索/生成
│   ├── server.py             # FastAPI:上传/聊天/文档列表/静态托管
│   ├── pipeline.py           # CLI 入口,委托 engine
│   └── cli.py                # 命令行入口
├── frontend/                 # Web 界面(原生 HTML/CSS/JS)
└── tests/                    # 单元测试(pytest)
```

### 分层原则

1. **接口隔离点只有两个**:`embedder.py`(向量化)和 `llm.py`(生成)。换向量模型或换 LLM 只改这两个文件。
2. **索引是"可重建的副产品"**:原始文档是唯一真相,`index/` 随时能删掉重建。
3. **依赖注入**:`chunker` 需要的 `count_tokens`、`HybridRetriever` 需要的 embedder 都通过参数传入,而非内部硬编码——既让"切分和向量化用同一个 tokenizer"成为强制约束,也让测试能用假对象替代重模型。

## 二、方法与问题(为什么这样设计)

按流水线顺序,每一条都是"遇到什么问题 → 用什么方法 → 为什么"。

### 2.1 为什么切分成 chunk

**问题**:不能把整篇文档直接发给模型。① embedding 模型有硬上限(BGE 是 512 token);② 一个向量只能表达一个主题,整篇文档语义会被稀释;③ 检索粒度越细越能精确命中。

**方法**:切成几百 token 的片段,每个 chunk 独立向量化、独立检索。

### 2.2 为什么按"标题结构"切

**问题**:等宽硬切会把不同主题糊进同一个片段,也可能把一句话劈断。

**方法**:先用 Markdown 标题分到不同"小节",记层级路径 `一级 > 二级 > 三级` 存进 chunk 的 `section`;小节仍超长再按句子切。

### 2.3 为什么用 token 而非字符

**问题**:模型限制是 token,不是字符。中文一个词可能多字打包成一 token,英文一个长单词往往只占 1 token。用"512 字符"当上限,中文会溢出截断、英文又填不满。

**方法**:用模型自己的 tokenizer 数 token(`embedder.count_tokens` 注入 chunker)。`chunk_tokens=256` 给 512 上限留余量(还要给 `[CLS]`/`[SEP]` 腾位置)。

### 2.4 为什么 overlap,且只给正文

**问题**:按句切分可能在边界把"一个完整意思"劈成两半,命中一半丢一半语境。

**方法**:相邻 chunk 保留 `overlap_tokens` 重叠(上一 chunk 结尾的句子复制到下一 chunk 开头)。

**但代码不参与**:代码按行切、行本身完整,不存在劈断;且代码靠精确 token 检索,复制重叠行既浪费又可能让 LLM 看到重复代码。

### 2.5 为什么保护代码块 + 超长硬切

**问题**:技术文档核心是代码,按句号切会劈得面目全非。

**方法**:` ``` ` 围栏内代码作为"原子单元",永不按句子切。

**兜底**:单个代码块本身超 `chunk_tokens` 时按行硬切,保证每片 ≤ 上限且不切断单行。此外,英文句子切分用"句号 + 空白 + 大写/数字"规则(避免误切 `e.g.`、`3.14`);超长正文单元按词/字符硬切。

### 2.6 为什么向量归一化

**问题**:余弦公式 `cos(a,b)=a·b/(|a||b|)`,直接算要做两次模长一次除法。

**方法**:L2 归一化后 |a|=|b|=1,余弦退化成点积,检索变成一行矩阵乘法 `scores = V @ q`。

### 2.7 为什么混合检索 + RRF

**问题**:向量擅长语义相近,但对精确关键词(错误码/函数名/ID)不敏感;BM25 相反。二者互补。

**方法**:两个都算,用 RRF 融合 `rrf(d)=1/(k+rank_向量(d))+1/(k+rank_BM25(d))`。

**为什么 RRF 而非加权相加**:向量余弦 ∈[0,1]、BM25 无上限,尺度不可比;RRF 只依赖排名,天然规避尺度问题、无需调权。

### 2.8 为什么两阶段 rerank

**问题**:embedding 是"双编码器",query 和文档从不互看,只能抓粗粒度语义;快但不够准。

**方法**:先粗召回 `rerank_top_n` 个候选,再用交叉编码器(把 `[CLS] query [SEP] doc` 拼一起过 Transformer)逐一精排取 top_k。交叉编码器准但每对要现算,所以只用在候选集上。

### 2.9 为什么增量索引 + 改名检测

**问题**:文档一多,每次全量重跑太慢。

**方法**:给每个文件算 SHA-256 内容哈希 + 记录切分/模型配置,存 `index/manifest.json`。哈希和配置都没变 → 复用旧向量;变了 → 只重跑该文件;删了 → 自动移除;**改名**(路径变、哈希不变)→ 复用旧向量、只更新 source。配置或切分代码版本变化会自动触发全量重建。

### 2.10 为什么本地 embedding + 云端 LLM

**问题**:embedding 用小模型即可,本地跑免费离线;LLM 要大模型,8G 内存跑不动。

**方法**:embedding 用本地 `bge-small-zh`(MPS),生成走 DeepSeek 等 API。也正好匹配现状:DeepSeek 无 embedding 接口,只有生成接口。

## 三、分块说明

| 模块 | 文件 | 职责 |
|---|---|---|
| 配置 | `config.py` | 读 config.yaml 到 dataclass |
| 加载 | `ingest/loader.py` | .md/.txt 读文本+剥 frontmatter+提标题;.pdf 抽文本层;hash_file |
| 切分 | `ingest/chunker.py` | 标题分节→句子切分→token 贪心合并(带重叠/代码不切/超长硬切) |
| 向量化 | `embed/embedder.py` | SentenceTransformer 封装;embed/embed_query/count_tokens |
| 存储 | `store/index.py` | NumpyIndex:归一化点积=余弦 |
| 存储 | `store/metadata.py` | save/load_index;save/load_manifest;clear_index |
| 检索 | `retrieve/bm25.py` | 从零实现 BM25 |
| 检索 | `retrieve/retriever.py` | 纯向量检索 |
| 检索 | `retrieve/hybrid.py` | rrf_fuse + HybridRetriever |
| 检索 | `retrieve/reranker.py` | 交叉编码器精排 |
| 生成 | `generate/llm.py` | build_prompt/build_chat_messages;generate/stream;rewrite_query |
| 编排 | `engine.py` | RAGEngine:常驻模型+索引+增量索引+检索+生成;plan_index_update |
| 服务 | `server.py` | FastAPI:/api/upload、/api/chat、/api/documents、DELETE、静态托管 |
| 命令行 | `cli.py` | docrag index / ask;docrag-web 起服务 |
| 测试 | `tests/` | 单元测试,秒级、不加载模型 |
