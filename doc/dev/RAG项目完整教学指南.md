# BuyPilot-AI 后端与 RAG 源码学习指南

> 文档定位：以当前源码和配置为事实来源，沿真实请求链路学习项目，并支持比赛答辩和后端面试追问。
>
> 阅读原则：先看“谁调用谁”，再理解单个函数；设计目标、当前代码真实行为、已知缺陷必须分开表述。

## 1. 阅读方法与系统地图

### 1.1 一句话定位

BuyPilot-AI 是一个使用 FastAPI、PostgreSQL、pgvector 和多模型服务实现的多品类智能导购后端。它把用户的自然语言购物需求转换成受约束的 Criteria DSL，通过混合检索找到商品和证据，再以 SSE 事件流返回商品卡片、推荐解释、多轮反馈和决策结果。

### 1.2 分层架构

```text
Android Client
    ↓ HTTP / SSE contract
API：HTTP、鉴权、上传、SSE 序列化
    ↓
Runtime：单轮生命周期、意图路由、Handler 编排、事件顺序
    ↓
Service：Criteria、检索、推荐、反馈、购物车、模型调用
    ↓
Repo：PostgreSQL、pgvector、会话与审计持久化
    ↓
Config / Types：配置、领域词表、Pydantic/SSE 契约
```

依赖方向必须保持 `API → Runtime → Service → Repo`。Runtime 负责组织调用顺序，不应直接调用模型 SDK 或拼 SQL；Repo 只负责持久化，不反向依赖业务层。

### 1.3 一次在线推荐请求的主调用链

```text
POST /chat/stream
→ api.chat.stream_chat()
→ runtime.pipeline.chat_stream()
→ runtime.pipeline._run_chat_turn()
→ runtime.pipeline._prepare_pipeline_body()          # 可选图片预分析
→ runtime.pipeline._resolve_intent()
→ runtime.pipeline._missing_slots()
→ runtime.stages.slot_checker.check_required_slots()
→ runtime.pipeline._dispatch_intent_handler()
→ runtime.handlers.handle_recommendation()
→ runtime.stages.criteria.run_criteria()
→ runtime.stages.recommendation.run_retrieval()
→ services.retriever.retrieve_with_evidence()
→ runtime.handlers.continue_recommendation_from_criteria()
→ services.llm_client.stream_recommendation()
→ StreamContext 生成 SSEEvent
→ api.chat.stream_chat() 序列化为 SSE 文本
```

这条链路是全文主轴。其他功能不是孤立模块，而是在以下位置分叉：

| 分叉时机 | 条件 | 下游 |
|---|---|---|
| Pipeline 预处理 | 请求携带 `image_url` | VLM 图片理解，随后仍回到 Intent/Criteria 主线 |
| Intent 路由 | `compare` | `runtime.compare_handlers.handle_compare()` |
| Intent 路由 | 购物车或结算意图 | `runtime.cart_handlers` 对应 Handler |
| 推荐 Handler 内 | 判断为场景化选购 | `services.shopping_strategy.build_shopping_strategy_plan()` |
| 推荐 Handler 内 | 请求携带图片 | 图片 Embedding 与文本检索并行 |
| 多轮继续 | `continue` | 复用上一轮 Criteria/牌组，并依据反馈决定重新检索或最终决策 |

### 1.4 四组最重要的数据结构

| 数据结构 | 源码 | 作用与流转位置 |
|---|---|---|
| `ChatStreamRequest`、`IntentResult` | [`schemas.py`](../../backend/src/types/schemas.py) | API 请求进入 Pipeline；IntentResult 交给槽位检查和 Handler 路由 |
| `Constraints`、`CriteriaPayload` | [`sse_events.py`](../../backend/src/types/sse_events.py) | Intent 之后生成；作为检索、推荐卡片和多轮合并的统一标准 |
| `ProductPayload`、`EvidencePayload` | [`sse_events.py`](../../backend/src/types/sse_events.py) | 检索输出；分别承载商品事实和推荐证据 |
| `StreamContext`、各类 `SSEEvent` | [`streaming.py`](../../backend/src/runtime/streaming.py)、[`sse_events.py`](../../backend/src/types/sse_events.py) | 管理 turn、seq、取消和事件信封，最终由 API 层序列化 |

### 1.5 推荐阅读顺序

1. 第 3 章：先理解 HTTP、SSE 和 Pipeline 入口。
2. 第 4～7 章：顺着 Intent → Criteria → Retrieval → Recommendation 阅读主业务。
3. 第 8～10 章：理解多轮、多模态和其他 Handler 分支。
4. 第 2、11 章：补齐离线数据、启动、缓存、可观测性和测试。
5. 第 12、13 章：用于答辩复习和缺陷追问。

## 2. 应用启动与离线数据准备

应用启动决定配置、数据库和索引是否可用；离线数据准备决定在线检索能够搜索哪些 Chunk。二者都发生在用户请求主线之外，但共同构成 RAG 的前置条件。

### 本章调用关系卡

| 维度 | 说明 |
|---|---|
| 上游调用者 | `api.app.lifespan()` 启动初始化；Seed 命令或自动 Seed 调用商品入库 |
| 调用时机 | 进程启动时建表/建索引；首次或数据变化时生成商品、Chunk、文本/图片向量；BM25 在自动 Seed 后或首次检索时构建 |
| 核心输入 | 环境变量、LLM Profile、`data/raw/ecommerce_agent_dataset/` 商品 JSON |
| 核心输出 | PostgreSQL 表、pgvector/HNSW 索引、`products`、`product_chunks`、图片向量；BM25 是当前进程的内存索引 |
| 在线消费者 | `retrieve_with_evidence()` 查询这些结构；推荐与证据模块读取商品和 Chunk |

### 2.1 启动、配置与资源生命周期

这一阶段不再讨论“推荐结果怎么算”，而是回答后端工程中的另一类问题：

- 应用依赖哪些外部资源才能启动？
- 配置在哪里加载和校验？
- 数据库、HTTP Client 等资源何时创建和释放？
- 哪些故障必须阻止启动，哪些故障可以降级？
- 健康检查能证明系统健康到什么程度？

对于 Java/Spring 开发者，可以把这一节理解为：

```text
FastAPI lifespan       ≈ Spring Bean 生命周期 / ApplicationRunner / @PreDestroy
Settings               ≈ @ConfigurationProperties 配置对象
get_settings()         ≈ 容器中的单例配置 Bean
initialize_database()  ≈ 启动初始化器
create_async_engine()  ≈ 创建 DataSource / 连接池
```

#### 2.1.1 配置加载发生在什么时候

应用导入 [`app.py`](../../backend/src/api/app.py) 时，会立即执行模块级代码：

```python
settings = get_settings()
```

调用链为：

```text
导入 app.py
→ get_settings()
→ 第一次调用时创建 Settings
→ 读取进程环境变量和项目根目录 .env
→ 校验 DATABASE_URL
→ 加载 LLM task/profile 映射
→ 缓存 Settings 实例
→ 创建 FastAPI app、注册中间件和 Router
```

相关源码：

- [`_load_env_file()`](../../backend/src/config/settings.py)：读取项目根目录 `.env`，但不会覆盖进程中已经存在的环境变量。
- [`Settings.__init__()`](../../backend/src/config/settings.py)：集中解析数据库、运行模式、自动 Seed、数据集目录、上传目录、观测和检索缓存等配置。
- [`get_settings()`](../../backend/src/config/settings.py)：延迟创建并复用同一个 `Settings` 对象。
- [`load_llm_profiles()`](../../backend/src/config/settings.py)：使用 `@lru_cache(maxsize=1)` 缓存 YAML 配置解析结果。

这意味着进程环境变量的优先级高于 `.env`：

```text
操作系统 / Docker 注入的环境变量
        ↓ 已存在时不覆盖
项目根目录 .env
        ↓
Settings
```

业务代码应通过 `get_settings()` 访问配置，而不是在各个模块中散落 `os.getenv()`。这样可以统一处理默认值、类型转换、路径解析、必填校验和测试替换。

当前实现不是使用 Pydantic `BaseSettings`，而是自己实现了一个轻量配置单例。它与 Spring 单例 Bean 的相似点是“进程内共享同一个配置对象”，但它不具备 Spring 容器完整的依赖注入和配置绑定能力。

#### 2.1.2 为什么 DATABASE_URL 配置错误会直接退出

[`_resolve_database_url()`](../../backend/src/config/settings.py) 对数据库配置执行硬校验：

```text
DATABASE_URL 缺失
→ raise SystemExit

DATABASE_URL 不是 PostgreSQL
→ raise SystemExit
```

当前项目明确只支持 PostgreSQL + pgvector。商品、会话、反馈、检索 Chunk、向量索引和证据记录都依赖它，所以数据库不是可选增强能力。

这里采用的是 **fail-fast（快速失败）**：

> 核心依赖配置错误时阻止服务启动，避免应用表面启动成功，却在收到第一条推荐请求后才失败。

相对地，[`_check_api_key_configuration()`](../../backend/src/services/startup.py) 发现 `BAILIAN_API_KEY` 缺失时只记录警告，不阻止进程启动。因此当前代码真实行为是：

```text
数据库配置错误 → 应用不能启动
模型 Key 缺失   → 应用仍能启动，但 RAG 模型调用不可用
```

这是一种比赛项目中的可用性取舍：即使模型能力暂时不可用，健康检查、静态资源和部分非模型接口仍可以启动；但不能据此认为完整推荐链路可用。

#### 2.1.3 FastAPI lifespan 如何管理启动和关闭

[`lifespan()`](../../backend/src/api/app.py) 是应用生命周期入口：

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    await _initialize_database()
    try:
        yield
    finally:
        await drain_observability_tasks(timeout_seconds=5.0)
        await close_http_client()
```

`yield` 是生命周期分界线：

```text
yield 之前：应用启动阶段
yield 时：   FastAPI 正式接收请求
yield 之后：应用关闭阶段
```

完整启动链路为：

```text
FastAPI 启动
→ lifespan()
→ app._initialize_database()
→ services.startup.initialize_database()
→ repos.database.create_db_and_tables()
→ 确保 pgvector extension
→ 创建 SQLModel 数据表
→ 补充当前模型需要的字段
→ 创建普通业务索引
→ 创建文本和图片向量 HNSW 索引
→ 根据配置决定是否自动导入商品数据
→ yield
→ 开始处理 HTTP/SSE 请求
```

关闭时会：

```text
等待异步 LLM 观测记录任务，最多 5 秒
→ 关闭进程共享的 HTTP Client
```

与 Spring 类比：

```text
yield 之前  ≈ ApplicationRunner / @PostConstruct 后的启动初始化
yield 之后  ≈ @PreDestroy / DisposableBean 的资源回收
```

#### 2.1.4 数据库结构初始化和商品 Seed 必须区分

[`initialize_database()`](../../backend/src/services/startup.py) 中存在两层行为。

第一层每次启动都执行：

```text
create_db_and_tables()
├─ ensure_pgvector_extension()
├─ SQLModel.metadata.create_all()
├─ _ensure_model_columns_sync()
├─ ensure_runtime_indexes()
└─ ensure_pgvector_indexes()
```

它负责让数据库具备运行所需的表、字段和索引。

第二层仅在 `AUTO_SEED_ON_STARTUP=1` 时执行：

```text
seed_products_if_needed()
→ 导入商品、Chunk 和文本向量

seed_image_embeddings_if_needed()
→ 导入商品图片向量

bm25_index.build_from_db()
→ 从数据库 Chunk 构建当前进程的 BM25 内存索引
```

因此：

```text
数据库初始化 ≠ 商品数据初始化
建表和建索引 ≠ 生成商品 Embedding
pgvector 索引 ≠ BM25 内存索引
```

BM25 构建被 `try/except` 包围，失败后只记录警告。这是因为 BM25 在当前混合检索中是增强排序能力；数据库和 pgvector 才是核心召回依赖。BM25 失败时可以退回向量召回链路，不需要让整个服务启动失败。

#### 2.1.5 AsyncEngine 为什么要缓存

[`get_async_engine()`](../../backend/src/repos/database.py) 使用模块级 `_ASYNC_ENGINE_CACHE` 保存 `AsyncEngine`：

```text
第一次访问
→ 根据 DATABASE_URL 创建 AsyncEngine
→ 放入缓存

后续访问且 DATABASE_URL 相同
→ 复用已有 AsyncEngine
```

它可以类比 Spring 中由容器管理的单例 `DataSource`：不能每个请求都重新创建数据库 Engine 和连接池，否则会造成连接建立成本、连接数膨胀和资源泄漏。

项目还提供了 [`dispose_async_engine()`](../../backend/src/repos/database.py)，用于主动关闭 Engine；但当前 [`lifespan()`](../../backend/src/api/app.py) 关闭流程没有调用它。因此当前实现主要依赖进程退出时由运行环境回收数据库资源，优雅停机还可以进一步完善。

#### 2.1.6 `/health` 能证明什么

[`health()`](../../backend/src/api/app.py) 当前返回：

```text
status
service
strict_runtime
fallback_policy
active_turns
```

它能证明：

- FastAPI 进程还活着；
- Router 能处理请求；
- 当前运行模式可以读取；
- 可以查看进程内活跃聊天轮数。

它不能证明：

- PostgreSQL 当前仍然可连接；
- pgvector 查询能够正常执行；
- 商品和 Chunk 已完成 Seed；
- LLM、Embedding、Rerank Provider 可用；
- 一次完整 RAG 请求一定能成功。

因此它更接近 **liveness（存活检查）**，而不是完整的 **readiness（就绪检查）**。生产化时可以拆分：

```text
/health/live
→ 只检查进程是否存活

/health/ready
→ 检查数据库、必要数据、关键外部依赖
```

#### 2.1.7 设计目标、真实行为与工程缺陷

| 方面 | 设计目标 | 当前代码真实行为 | 可改进点 |
|---|---|---|---|
| 配置管理 | 配置集中读取 | 大部分配置经过 `Settings` | `startup.py` 仍直接读取 `BAILIAN_API_KEY` |
| 核心依赖 | 数据库异常时快速失败 | 缺少或使用非 PostgreSQL URL 会直接退出 | 可增加连接可用性和 pgvector 版本检查 |
| 数据库升级 | 启动后结构可用 | `create_all` 加手写补列、建索引 | 正式环境使用 Alembic，类似 Flyway/Liquibase |
| 自动 Seed | 开发和 Demo 可快速准备数据 | 由环境变量控制，关闭时不会自动导入 | 部署流程中将迁移和 Seed 分离 |
| BM25 | 提供关键词排序增强 | 只有自动 Seed 路径会在启动时构建，失败可降级 | 多实例分别构建，且需明确非 Seed 启动时的初始化策略 |
| 健康检查 | 暴露服务状态 | 只进行浅层进程检查 | 增加 readiness 和依赖状态 |
| 资源关闭 | 优雅释放共享资源 | 关闭观测任务和 HTTP Client | 同时调用 `dispose_async_engine()` |

答辩表达：

> 项目使用 FastAPI lifespan 管理应用资源生命周期。模块加载时通过集中式 Settings 读取配置，并对 PostgreSQL 配置执行 fail-fast；应用启动阶段初始化 pgvector、数据表和检索索引，再根据环境变量决定是否自动导入商品、文本向量和图片向量。BM25 属于可降级的增强能力，构建失败不会阻止服务启动。应用关闭时会等待观测任务并释放共享 HTTP Client。当前实现满足比赛 Demo，但健康检查仍偏 liveness，数据库结构升级也主要依赖 create_all 和补列逻辑，生产化应补充 readiness、Alembic 迁移以及 AsyncEngine 的显式释放。

自测问题：

1. 为什么 `get_settings()` 可以类比 Spring 的单例配置 Bean？
2. `.env` 和进程环境变量同时存在时，哪一个优先？
3. 为什么 `DATABASE_URL` 错误会阻止启动，而 BM25 构建失败不会？
4. `lifespan()` 中 `yield` 前后分别对应什么阶段？
5. 数据库建表、pgvector 建索引和商品 Seed 有什么区别？
6. 为什么不能为每个请求重新创建 `AsyncEngine`？
7. `/health` 返回 `ok` 为什么不能证明 RAG 链路已经就绪？
8. 当前启动和关闭流程还有哪些生产化改进点？

### 2.2 商品入库、知识包、Chunk 与 Embedding

#### 2.2.1 当前真正使用的数据管道

项目的商品事实源是 [`data/raw/ecommerce_agent_dataset/`](../../data/raw/ecommerce_agent_dataset/)，包含 100 条脱敏电商商品，分为美妆护肤、数码电子、服饰运动和食品生活四类。每件商品主要包含：

```text
product_id、title、brand、category、sub_category、base_price
image_path、skus
rag_knowledge
├─ marketing_description
├─ official_faq[]
└─ user_reviews[]
```

当前运行时 RAG 索引链路是：

```text
官方 raw JSON
    ↓
规则化预处理与知识加工
    ↓
构建 knowledge_package
    ↓
生成 typed semantic chunks
    ↓
每个 chunk 生成 Embedding
    ↓
写入 products / product_chunks
    ↓
构建 BM25 内存索引
```

[`load_raw_products()`](../../backend/src/repos/products.py) 和 [`list_raw_products()`](../../backend/src/repos/products.py) 直接读取官方 JSON；[`seed_products()`](../../backend/src/services/product_ingest.py) 是当前商品、chunk 和文本向量写入数据库的真实入口。

仓库还保留了 [`data/scripts/process_data.py`](../../data/scripts/process_data.py)，它会生成 `data/processed/products.json` 和 `chunks.json`。这是较早的离线导出脚本和派生产物，当前 `product_ingest.py` 不读取这些文件，不能把它当成运行时 RAG 索引链路。

#### 2.2.2 “数据清洗”实际做了什么

这里更准确地说是**规则化预处理与知识加工**，而不是完整意义上的数据质量清洗。源码会做格式标准化、属性提取和证据分类，但没有实现价格异常修正、重复商品合并、缺失值插补或人工事实纠错。

主要处理包括：

1. [`clean_text()`](../../backend/src/services/chunking.py) 将换行、Tab 和连续空白压缩成单空格，并按最大字符数截断；空文本不生成 chunk。
2. `normalize_category()` 将原始品类归一化到系统定义的四个顶层品类。
3. [`_raw_text()`](../../backend/src/services/chunking.py) 汇总标题、品牌、品类、营销描述、FAQ 和评价，作为确定性属性提取的输入。
4. 通过领域词表提取肤质、使用场景、成分词和风险词。
5. [`_skin_types()`](../../backend/src/services/chunking.py) 会检查否定前缀，避免把“不适合敏感肌”误提取为“适合敏感肌”。
6. [`_category_aliases()`](../../backend/src/services/chunking.py) 为部分子品类扩展同义表达，例如“洁面”扩展为“洗面奶、洁面乳、日常清洁、控油洁面”。
7. [`_dedup()`](../../backend/src/services/chunking.py) 对提取值执行清洗、去空和保持顺序的去重。
8. 评价按评分做简单极性划分：`rating <= 2` 为负面，`rating >= 4` 为正面。

需要注意一个当前实现边界：[`_add_review_chunks()`](../../backend/src/services/chunking.py) 只区分“负面”和“其他”，所以 3 分评价会被命名为 `positive_review`，但 [`_summary_sentences()`](../../backend/src/services/chunking.py) 不会把 3 分评价加入正面或负面摘要。这更准确地说是“非负面分支”，当前 `positive_review` 命名不够严谨。

#### 2.2.3 确定性知识包

[`build_product_knowledge_package()`](../../backend/src/services/chunking.py) 会从一件官方商品构造规则化知识包：

```text
basic
├─ product_id、name、category、source_category
├─ sub_category、brand、price

attributes
├─ skin_type
├─ use_scenario
├─ ingredient_terms
└─ risk_terms

retrieval_alias
├─ positive_alias
└─ risk_terms

evidence_summary
├─ why_buy
├─ risk
├─ compare_axis
└─ not_suitable_for
```

这个知识包不是 LLM 生成的，而是依据领域词表、评分和警告标记确定性构建。它既保存在商品 metadata 中，也用于生成 profile、warning 和 compare 等 chunk。

#### 2.2.4 Typed Semantic Chunking

当前真实入口是 [`build_product_chunks()`](../../backend/src/services/chunking.py)。它不是机械地每隔固定字符切一刀，而是优先按照商品知识的业务语义和证据用途分块。

| Chunk 类型 | 文本来源 | `retrieval_role` | `evidence_kind` | 主要用途 |
|---|---|---|---|---|
| `profile` | 名称、品牌、品类、价格、肤质、场景、别名 | `primary` | `profile` | 商品身份和基础属性召回 |
| `marketing` | 营销描述 | `primary` | `why_buy` | 推荐理由 |
| `faq` | 每条官方问答 | `primary` 或 `risk` | `faq` 或 `risk` | 常见问题或风险提示 |
| `positive_review` | 非低分评价 | `evidence` | `why_buy` | 用户体验证据 |
| `negative_review` | 1～2 分评价 | `risk` | `risk` | 差评和风险 |
| `warning` | 风险摘要句 | `risk` | `risk` | 集中风险提示 |
| `compare` | 品类对应的比较维度 | `evidence` | `compare` | 多商品比较 |

每件商品固定尝试生成一个 profile chunk：

```text
名称 | 品牌 | 品类 | 子品类 | 价格
| 适用肤质 | 使用场景 | 检索别名
```

营销描述由 [`_split_long_text()`](../../backend/src/services/chunking.py) 先按 `。！？!?` 划分句子，再尽量组合成不超过 420 字的 marketing chunk。每条 FAQ 和每条评价独立生成一个 chunk；如果 FAQ 包含警告标记，则归入 risk 路径。

风险摘要由营销描述、FAQ 回答和低分评价中的风险句构建；compare chunk 则根据顶层品类生成固定比较轴，例如美妆护肤使用“肤质匹配、成分风险、使用场景、价格”。

所有 chunk 最终还会经过 `clean_text(..., SEMANTIC_CHUNK_MAX_CHARS)`，当前统一上限是 480 字。一个 [`SemanticChunk`](../../backend/src/services/chunking.py) 包含：

```text
chunk_text
chunk_index
metadata
```

metadata 会记录：

```text
source、category、source_category、sub_category、brand、price
chunk_type、retrieval_role、evidence_kind、section_index
以及 rating、nickname、question 等类型专属字段
```

#### 2.2.5 一件商品如何被拆分

以一件包含 1 段营销描述、3 条 FAQ 和 5 条评价的精华商品为例，可能产生：

```text
chunk 0：profile
chunk 1：marketing
chunk 2：普通 FAQ
chunk 3：普通 FAQ
chunk 4：敏感肌注意事项 FAQ，risk
chunk 5：1 分评价，negative_review / risk
chunk 6：2 分评价，negative_review / risk
chunk 7：5 分评价，positive_review / why_buy
chunk 8：2 分评价，negative_review / risk
chunk 9：1 分评价，negative_review / risk
chunk 10：warning 风险摘要
chunk 11：compare 比较维度
```

这只是根据字段内容推导的示例，实际数量取决于空字段、警告词命中以及营销描述是否因长度被拆成多段。

不同类型分开存储的原因是：

- 局部语义不会被整件商品的大文本稀释。
- 每条评价可以拥有独立向量。
- 推荐理由、FAQ、差评风险和比较信息可以分别消费。
- 检索命中的原始 chunk 可以直接作为可解释证据。

#### 2.2.6 Embedding 与数据库入库

[`seed_products()`](../../backend/src/services/product_ingest.py) 的执行流程是：

```text
list_raw_products()
    ↓
_build_chunk_rows()
    ↓
build_product_chunks()
    ↓
_embed_chunks()，每批 10 个 chunk
    ↓
校验 Embedding 维度
    ↓
更新 Product，并删除该商品旧的 ProductChunk
    ↓
写入新的 ProductChunk
    ↓
保存 dataset_hash、chunking_version 和 embedding_model
```

每个 chunk ID 使用确定性格式：

```text
{product_id}:{chunk_index}
```

例如 `p_beauty_001:0`。写入 [`ProductChunk`](../../backend/src/repos/models.py) 的内容包括 chunk 文本、所属商品、序号、Embedding 和完整 metadata。严格重建路径要求文本向量为 1024 维。

系统还计算：

- `source_hash`：单件商品原始 JSON 的 SHA-256。
- `dataset_hash`：全部商品 ID 与 source hash 的有序摘要。
- `chunking_version`：当前为 `semantic_v1`。
- `embedding_model`：当前配置的 Embedding 模型名称。

[`seed_products_if_needed()`](../../backend/src/services/product_ingest.py) 只有在数据集 hash 相同、数据库已有 chunk 且向量维度符合要求时才跳过重建。旧数据库如果还没有保存 hash，即使已有 chunk，也会重新 seed，以写入新的索引元数据。

应用启动时，[`initialize_database()`](../../backend/src/services/startup.py) 仅在开启自动 seed 时执行这条链路；文本索引完成后再构建 [`BM25Index`](../../backend/src/services/bm25_recall.py)。也可以通过 [`reindex_embeddings.py`](../../backend/src/scripts/reindex_embeddings.py) 显式重建文本 chunk 和 Embedding。

#### 2.2.7 本阶段答辩表达

> 项目以官方商品 JSON 为事实源，先通过确定性规则完成文本空白清理、品类归一化、领域属性提取、别名扩展、评价极性划分和风险摘要，再按 profile、营销描述、FAQ、评价、风险和比较维度生成 typed semantic chunks。每个 chunk 独立生成 1024 维 Embedding，并与 product_id、证据类型和来源 metadata 一起写入 PostgreSQL。系统通过商品和数据集 hash 判断是否需要重建索引。这样既提高了局部语义召回能力，也支持商品聚合和推荐、FAQ、风险等不同类型的证据绑定。

自测问题：

1. 为什么当前处理更适合叫“规则化预处理”，而不是完整的数据质量清洗？
2. `data/scripts/process_data.py` 和当前运行时 `product_ingest.py` 有什么区别？
3. 为什么营销描述、FAQ 和每条评价不应该合成一个大 chunk？
4. `chunk_type`、`retrieval_role` 和 `evidence_kind` 分别表达什么？
5. 3 分评价当前会进入哪种 chunk，它为什么是一个命名边界？
6. `source_hash`、`dataset_hash` 和 `chunking_version` 分别解决什么问题？

---

## 3. 在线入口：HTTP、SSE 与 Pipeline 生命周期

### 本章调用关系卡

| 维度 | 说明 |
|---|---|
| 上游调用者 | Android 调用 `POST /chat/stream` |
| 入口方法 | `api.chat.stream_chat()` |
| 生命周期 Owner | `runtime.pipeline.chat_stream()` |
| 业务编排 | `runtime.pipeline._run_chat_turn()` |
| 下游调用 | 图片预分析 → Intent → 槽位检查 → Handler → SSEEvent |
| 输出 | HTTP 200 的 `text/event-stream`，流中包含 thinking/card/text/error/done 等事件 |

**为什么需要这一步？**
用户发送消息后，如果必须等待整条链路结束才看到内容，体验会很差。流式推送允许系统在不同阶段逐步发送 thinking、商品卡片和文本增量；实际首个事件延迟取决于命中的路由和外部服务响应，不能脱离运行数据承诺固定数值。

### 3.1 从 HTTP 请求到 SSE 事件流

核心调用链如下：

```text
Android
  → POST /chat/stream
  → ChatStreamRequest 参数校验
  → API：stream_chat()
  → Runtime：chat_stream()
  → Runtime：_run_chat_turn()
  → 逐个产生 SSEEvent
  → format_sse()
  → Android 接收事件
```

各组件的职责：

| 组件 | 职责 |
|------|------|
| `ChatStreamRequest` | 请求反序列化与参数校验 |
| `stream_chat()` | 处理 HTTP/SSE 协议边界 |
| `chat_stream()` | 管理单轮对话的生命周期、异常、取消、审计与资源清理 |
| `_run_chat_turn()` | 编排具体的对话业务流程 |
| Handler | 执行推荐、对比、购物车等具体用例 |
| `format_sse()` | 将事件对象序列化成标准 SSE 文本 |

Android 发送的 JSON 会由 FastAPI 和 Pydantic 自动反序列化为 `ChatStreamRequest`。关键校验包括：

- `message` 最长 2000 字符。
- `message` 和 `image_url` 至少有一个非空。
- 字段类型必须符合模型定义。
- 校验失败时直接返回 HTTP 422，不会进入 Pipeline。

Java 中可以将其理解为 Controller DTO 加 Bean Validation。

### 3.2 为什么使用异步生成器

如果返回 `List[SSEEvent]`，服务端必须等待所有计算结束，才能一次性返回完整列表。用户需要等到意图识别、检索和模型生成全部完成后才能看到结果。

异步生成器则可以边处理边返回：

```python
yield thinking
await llm_call()
yield product_card
yield done
```

每次执行 `yield`，当前事件都可以立即写入 HTTP 长连接，不需要等待后续阶段完成。

Java WebFlux 类比：

```text
List<SSEEvent>  → 全部完成后一次性返回
Flux<SSEEvent>  → 每产生一个事件就 emit 一个
```

### 3.3 SSE 文本格式

Pipeline 产生的是 `SSEEventBase` 的子类对象，API 层通过 `format_sse()` 将其转换为：

```text
event: thinking
data: {"session_id":"sess_001","turn_id":"turn_001","seq":1,...}

```

其中：

- `event` 表示事件类型。
- `data` 携带 JSON 数据。
- 最后的空行表示当前 SSE 事件结束。
- HTTP `Content-Type` 为 `text/event-stream`。

### 3.4 session_id、turn_id 与 seq

`session_id` 标识一段完整的多轮会话：

```text
session_001
├── turn_001：推荐洗面奶
├── turn_002：预算降低
└── turn_003：加入购物车
```

它用于关联历史标准、反馈、购物车和会话记录。

`turn_id` 标识一次“用户输入 + Agent 回复”，用于取消生成、审计、耗时统计和区分不同轮次。

`seq` 表示同一个 turn 内第几个 SSE 事件，用于保证顺序、检测事件丢失以及客户端去重。

记忆方式：

```text
session_id：哪段会话
turn_id：哪轮问答
seq：本轮第几个事件
```

`event`、`stage` 和 `seq` 容易混淆：

- `event`：事件类型，决定客户端如何解析和渲染，例如 `thinking`、`product_card`。
- `stage`：Pipeline 当前正在处理的业务阶段，主要存在于 `thinking` 事件中。
- `seq`：事件发送顺序，不包含业务语义。

例如：

```text
event=thinking, stage=retrieving, seq=3
event=thinking, stage=retrieving, seq=4
event=product_card, seq=5
```

前两个事件属于同一个检索阶段，但拥有不同的事件序号。

### 3.5 两种错误边界

请求参数不合法时：

```text
请求
→ Pydantic 校验失败
→ HTTP 422
```

此时 Pipeline 尚未执行，SSE 长连接也没有正常建立，客户端收到的是普通 HTTP 错误。

如果 LLM、数据库或业务代码在流式执行过程中发生异常：

```text
HTTP 200 已提交
→ thinking 已发送
→ Pipeline 异常
→ error
→ done(error)
```

HTTP 状态码位于响应头中，只能发送一次。SSE 开始发送后，`200 OK` 已经提交，因此不能再改为 HTTP 500，只能通过 `error` 和 `done(error)` SSE 事件表达失败。

真实异常堆栈只记录在服务端日志中，客户端收到的是脱敏错误信息，避免泄露数据库连接、API Key 和内部文件路径。

### 3.6 取消机制

```text
Android 点击停止
→ POST /chat/cancel
→ 根据 session_id + turn_id 定位当前任务
→ 设置 cancel_token
→ Pipeline 在心跳或活跃检查点发现取消状态
→ 抛出 StreamCancelled
→ 取消仍在运行的后台任务
→ 写入 chat.turn_cancelled 审计事件
→ 返回 done(cancelled)
→ finally 清理 active turn 和 cancel token
```

0.8 秒是心跳和取消状态的检查周期，不是 LLM 的超时时间。这是协作式取消：任务在检查点观察到取消状态后退出，而不是由操作系统强制杀死线程。

### 3.7 为什么 stream_chat() 与 chat_stream() 不合并

`stream_chat()` 属于 API 层，负责接收 HTTP 请求、确定标识、设置 SSE 响应头并序列化事件。

`chat_stream()` 属于 Runtime 层，负责注册取消令牌和 active turn、创建 `StreamContext`、写入审计事件、处理异常并清理资源。

拆分后，HTTP 协议细节与对话生命周期相互隔离，Runtime 可以脱离 FastAPI 进行测试，也避免 Controller 堆积业务编排、审计和异常处理逻辑。

源码位置：

| 模块 | 文件 | 关键位置 |
|------|------|---------|
| 请求模型 | [schemas.py](../../backend/src/types/schemas.py) | `ChatStreamRequest` |
| HTTP/SSE 入口 | [chat.py](../../backend/src/api/chat.py) | `stream_chat()` |
| Turn 生命周期 | [pipeline.py](../../backend/src/runtime/pipeline.py) | `chat_stream()` |
| 业务编排 | [pipeline.py](../../backend/src/runtime/pipeline.py) | `_run_chat_turn()` |
| SSE 事件模型 | [sse_events.py](../../backend/src/types/sse_events.py) | `SSEEventBase`、`EventSeq` |
| SSE 序列化 | [sse_events.py](../../backend/src/types/sse_events.py) | `format_sse()` |

面试概括：

> Android 调用 `/chat/stream` 后，FastAPI 先通过 Pydantic 完成请求反序列化和校验。API 层返回一个基于异步生成器的 `StreamingResponse`，并消费 Runtime 层产生的 SSE 事件。Pipeline 每生成一个事件，API 就将其序列化成标准 SSE 文本并立即发送。`session_id` 标识多轮会话，`turn_id` 标识一次问答，`seq` 保证本轮事件顺序。流建立前的参数错误使用 HTTP 422；流建立后的异常由于 HTTP 200 已经提交，只能通过 `error` 和 `done(error)` 事件表达。用户取消时则通过取消令牌，在 Pipeline 检查点终止任务并返回 `done(cancelled)`。

自测问题：

1. 为什么不能直接返回 `List[SSEEvent]`？
2. `session_id`、`turn_id`、`seq` 分别解决什么问题？
3. 参数校验失败和 Pipeline 执行失败，客户端看到的结果有什么区别？
4. 为什么已经发送 `thinking` 后不能再返回 HTTP 500？
5. `event`、`stage`、`seq` 有什么区别？
6. 用户点击停止后，系统经过哪些步骤？
7. `stream_chat()` 和 `chat_stream()` 为什么需要分开？

### 3.8 SSE 事件协议

**SSE 事件协议（10 种事件类型）：**

定义在 [backend/src/types/sse_events.py](../../backend/src/types/sse_events.py) 和 [contracts/sse-events.schema.json](../../contracts/sse-events.schema.json)：

| 事件类型 | 含义 | Android 端渲染方式 |
|----------|------|-------------------|
| `thinking` | 系统正在思考 | 骨架屏动画 |
| `clarification` | 需要用户补充信息 | 反问卡片 |
| `criteria_card` | 购买标准已生成 | 标准摘要卡 |
| `text_delta` | 流式文本片段 | 逐字打字动画 |
| `product_card` | 推荐商品 | 商品卡片 |
| `final_decision` | 最终决策 | 决策总结卡 |
| `cart_action` | 购物车操作 | 加购成功/失败提示 |
| `done` | 本轮结束 | 停止 loading |
| `error` | 出错 | 错误提示 |
| `compare_card` | 商品对比 | 对比卡片 |

**流式编排的核心机制（阶段级心跳）：**

心跳的核心行为确实很简单：**某个耗时业务阶段等待 0.8 秒仍未返回结果，就向前端发送一个 `thinking` 事件。**

但它不是一个从请求开始一直运行到请求结束的全局定时器，而是由每个耗时阶段分别调用 `run_with_heartbeat()`。意图识别结束后，该阶段的心跳循环随即结束；后续进入标准生成、检索或决策阶段时，再启动一个新的心跳循环。因此，相邻两次心跳不一定永远严格间隔 0.8 秒。

```text
意图识别阶段：
0.0s  启动意图识别
0.8s  未完成 → thinking(understanding)
1.6s  未完成 → thinking(understanding)
2.0s  完成   → 返回 IntentResult，本阶段心跳结束

中间产生真实业务事件：
      criteria_card / text_delta 等事件本身就在持续向前端传输

商品检索阶段：
0.0s  启动检索，同时重新启动该阶段的心跳循环
0.8s  未完成 → thinking(searching)
1.3s  完成   → 返回 RetrievalResult，本阶段心跳结束
```

之所以采用“阶段级心跳”，是因为真实业务事件已经能证明流仍在工作，不需要再插入无意义的心跳；同时，每个阶段可以携带准确的 `stage` 和提示文案，避免在生成推荐文本时仍显示“正在检索商品”。

#### 心跳循环到底做了什么

核心实现位于 [streaming.py](../../backend/src/runtime/streaming.py) 的 `run_with_heartbeat()` 和 `run_timed_task_with_heartbeat()`：

```python
# 接近源码行为的伪代码
task = start_stage_task(awaitable)  # 真正的 LLM、检索等任务开始执行

while True:
    await ensure_active(ctx)  # 等待前检查：用户是否已取消

    done, _ = await asyncio.wait(
        {task},
        timeout=ctx.heartbeat_interval_seconds,  # 当前为 0.8 秒
    )

    await ensure_active(ctx)  # 等待后再检查一次取消

    if task in done:
        yield StageResult(task.result())  # 业务结果只交给 Pipeline，不直接发给 Android
        return                            # 当前阶段的心跳结束

    yield ctx.thinking(stage, message)    # 0.8 秒仍未完成，发送 SSE thinking
```

这里不是“无条件睡眠 0.8 秒”。`asyncio.wait(..., timeout=0.8)` 表示：**最多等 0.8 秒**。如果任务在 0.3 秒时完成，代码会立即取出结果，不会为了凑够 0.8 秒而等待，也不会多发一次心跳。

#### 前端收到心跳后会怎样

`ctx.thinking()` 创建的事件包含当前 `stage`、展示文案、递增的 `seq`，以及同一轮固定的 `node_id="thinking_{turn_id}"`。事件经过 Pipeline、FastAPI SSE 接口传到 Android 后：

1. `SseEventParser` 将它解析为 `ThinkingPayload`；
2. `ChatReducer` 将界面标记为 `Streaming`；
3. 创建或更新当前轮的 `ThinkingNode`；
4. UI 显示“正在理解需求”“正在检索商品”等加载动画；
5. 后续文本、卡片或完成事件到达时，临时 Thinking 节点会被移除或替换。

由于同一轮使用相同的 Thinking 节点标识，连续心跳主要是更新或维持同一个加载节点，不会每 0.8 秒在聊天列表里新增一条消息。

#### 如果前端没有收到心跳会怎样

当前 Android 客户端没有实现“超过 0.8 秒没收到心跳就判定后端离线”的严格心跳超时协议，而且 OkHttp 的 SSE 客户端配置了 `readTimeout(0)`。因此，漏掉一两个心跳不会立即触发报错或重新请求。

如果后端长时间不发送任何事件，客户端可能仍保持 SSE 连接并停留在原来的加载状态，用户难以区分“仍在计算”和“后端卡住”。只有连接真正失败时才会进入 `onFailure()`。所以本项目的 `thinking` 心跳更准确地说是：

> **耗时阶段的进度占位事件 + 周期性取消检查点 + 一定程度的连接保活。**

它不是客户端和服务端互相探测存活状态的严格双向心跳协议。

#### 心跳与取消为什么放在一起

每次等待任务前后，`ensure_active()` 都会检查：

- 当前进程内的 `cancel_token`；
- 数据库中是否存在跨进程取消请求。

发现取消后会设置取消令牌、抛出 `StreamCancelled`，并取消尚未完成的阶段任务。因此，心跳循环不仅负责“告诉前端还在运行”，也为协作式取消提供了周期性检查点。

可以把整个机制记成一句话：

> **启动耗时任务 → 最多等待 0.8 秒 → 检查取消 → 完成则返回业务结果，未完成则发送 `thinking` → 继续循环。**

#### 心跳相关的 4 个重要代码位置

| 作用 | 文件 | 重点 |
|------|------|------|
| 定义 0.8 秒间隔并创建流上下文 | [pipeline.py](../../backend/src/runtime/pipeline.py) | `HEARTBEAT_INTERVAL_SECONDS`、`StreamContext(...)` |
| 实现等待、取消检查和心跳循环 | [streaming.py](../../backend/src/runtime/streaming.py) | `run_with_heartbeat()`、`run_timed_task_with_heartbeat()`、`ensure_active()` |
| 在标准生成、检索、决策等阶段调用心跳 | [handlers.py](../../backend/src/runtime/handlers.py) | 多处 `run_with_heartbeat(...)` |
| Android 收到心跳后更新界面状态 | [ChatReducer.kt](../../android/feature/chat/src/main/java/com/buypilot/feature/chat/state/ChatReducer.kt) | `AgentEventType.Thinking`、`upsertThinkingAtTail()` |

> **术语卡：async/await** — Python 的异步编程关键字。`async def` 定义一个"协程"（可以在等待期间暂停的函数），`await` 表示"在这里等结果，但等待期间 CPU 可以去干别的"。和 `def`（同步函数，调用时一口气跑完、不能中途让位）的区别是：同步函数在等 LLM 回复的 3 秒里什么也做不了；协程可以在这 3 秒里发心跳、检查取消信号、启动别的任务。类比：同步 = 你在电话里等客服转接，不能挂；异步 = 客服说"稍等"，你可以一边等一边回微信。

**并行化执行 —— 投机检索 + 标准生成重叠**

代码在 [backend/src/runtime/handlers.py](../../backend/src/runtime/handlers.py) 的 `handle_recommendation()` 函数中。实际并行模式是：

```
阶段 A: 启动 DB 预读（反馈记录 + 历史商品ID）——后台任务
阶段 B: 用 IntentResult 快速构造 spec_criteria（纯计算，瞬时完成）
阶段 C: 流式输出 intro text（打字机动画，用户开始看到文字）
阶段 D: ★ 投机检索（spec_criteria）后台启动 + criteria 生成（LLM）并发
         ┌─ retrieval_task = start_stage_task(run_retrieval(spec_criteria))  ← 后台
         └─ criteria = await run_criteria(ctx, body, intent)                  ← 前台+心跳
阶段 E: await retrieval_task → 投机检索已提前完成（因为 spec_criteria 比 LLM criteria 早启动）
阶段 F: 用完整 criteria 对投机检索结果做后过滤（post-filter，O(n) 极快）
阶段 G: 若投机检索无结果 → 用完整 criteria 重新串行检索（fallback）
```

核心思想：**不等 LLM 生成完完整标准才开始检索**。用 intent 快速构造一个"简版标准"（spec_criteria），提前启动检索，让检索的 I/O 时间（embedding + SQL + pgvector）和 LLM 的推理时间（criteria 生成）重叠。当 LLM 的完整标准产出时，检索结果大概率已经在内存里等着了，只需过一个轻量的后过滤。如果投机检索命中了，用户感知的延迟 ≈ LLM criteria 时间；如果没命中（标准变化太大），走串行 fallback 也不会崩溃。

### 3.9 后端开发必须掌握的 SSE 契约边界

负责后端不需要深入 Compose 页面、动画、导航或 ViewModel 的具体写法，但必须理解客户端依赖哪些协议字段，否则即使后端 JSON 能成功发送，客户端也可能无法正确聚合、去重或结束一轮状态。

#### 三端契约关系

```text
contracts/sse-events.schema.json
          ↓
Python：types/sse_events.py
          ↓
Android：AgentPayload.kt + SseEventParser.kt + ChatReducer.kt
```

各部分职责：

| 位置 | 职责 |
|---|---|
| [`contracts/sse-events.schema.json`](../../contracts/sse-events.schema.json) | SSE 协议的 Source of Truth |
| [`types/sse_events.py`](../../backend/src/types/sse_events.py) | Python 事件模型、校验、序列化和反序列化 |
| [`AgentPayload.kt`](../../android/core/model/src/main/java/com/buypilot/core/model/AgentPayload.kt) | Android 对每种事件业务字段的反序列化模型 |
| [`SseEventParser.kt`](../../android/core/network/src/main/java/com/buypilot/core/network/SseEventParser.kt) | 根据 `event` 选择具体 Payload，并解析公共信封 |
| [`ChatReducer.kt`](../../android/feature/chat/src/main/java/com/buypilot/feature/chat/state/ChatReducer.kt) | 根据事件类型更新客户端状态 |

修改事件类型或字段时应按顺序同步：

```text
JSON Schema
→ Python Event Model
→ Android Payload / Parser / Reducer
→ Python 契约测试与 Android Parser/Reducer 测试
```

只修改 Python 端属于不完整的协议变更。

#### 公共信封字段分别解决什么问题

所有事件继承 [`SSEEventBase`](../../backend/src/types/sse_events.py)：

| 字段 | 后端必须保证的语义 | 客户端用途 |
|---|---|---|
| `schema_version` | 标识协议版本，当前为 `2026-05-20` | 兼容性判断 |
| `event` | 必须属于十种封闭事件之一 | 决定 Payload 类型和状态处理分支 |
| `session_id` | 同一段多轮会话保持一致 | 关联历史、反馈、购物车 |
| `turn_id` | 同一次用户请求的所有事件保持一致 | 隔离不同轮的流式状态 |
| `seq` | 同一 turn 内从 1 开始严格递增 | 判断顺序、排查丢事件 |
| `event_id` | 当前使用 `{turn_id}:{seq}`，一条事件一个唯一 ID | 事件去重 |
| `node_id` | 表示事件对应的逻辑 UI 节点 | 更新已有节点而不是重复创建 |
| `deck_id` | 同一批商品卡共享同一牌组 ID | 商品聚合、反馈与后续收敛决策 |
| `display_mode` | 表达建议展示语义 | 选择卡片、文本或牌组等呈现方式 |
| `created_at_ms` | 事件创建时间 | 追踪和展示辅助信息 |

这些 ID 不能互相替代：

```text
session_id：哪段多轮会话
turn_id：哪一轮请求
seq：本轮第几个事件
event_id：这条事件的唯一身份
node_id：更新哪个逻辑节点
deck_id：属于哪组候选商品
```

当前 Android Parser 对缺失的 `turn_id`、`seq`、`event_id`、`node_id` 和部分 `deck_id` 提供 fallback，但这只是兼容防线。后端不能依赖 fallback 省略 Schema 必填字段，因为自动生成的 ID 可能无法保持真实牌组和节点语义。

#### `message_id` 与 `turn_id` 的区别

[`TextDeltaEvent`](../../backend/src/types/sse_events.py) 使用 `message_id` 聚合同一段流式文本：

```text
text_delta(message_id="msg_1", delta="第一款")
text_delta(message_id="msg_1", delta="更适合油性肤质")
text_delta(message_id="msg_1", delta="。", done=true)
```

一轮请求中可能出现多段文本：

```text
intro_{turn_id}
budget_warning_{turn_id}
msg_{turn_id}
followup_{turn_id}
```

因此：

```text
turn_id：这些事件属于同一轮
message_id：这些 delta 属于同一段文本
```

客户端不能把同一 turn 的全部 `text_delta` 无条件拼成一个消息。

#### `deck_id` 为什么是业务字段

[`ProductCardEvent`](../../backend/src/types/sse_events.py) 的 `deck_id` 是必填字段。同一批候选商品必须共享它：

```text
deck_001
├─ product_card A
├─ product_card B
└─ product_card C
```

它不仅用于前端聚合商品卡，还会被反馈和多轮决策链路使用：

```text
商品卡交互
→ POST /feedback(deck_id, product_id, action)
→ 按 deck 聚合反馈
→ continue / converge
→ 在原牌组内做最终决策
```

如果后端错误复用或丢失 `deck_id`，可能把反馈关联到错误的一批商品。

#### `done` 才是 turn 的终止符

HTTP 连接关闭不能替代业务级 `done`。[`DoneEvent.finish_reason`](../../backend/src/types/sse_events.py) 包含：

| `finish_reason` | 后端业务含义 |
|---|---|
| `awaiting_criteria_confirmation` | 等待用户确认标准 |
| `awaiting_criteria_adjustment` | 等待修改无结果的标准 |
| `awaiting_product_feedback` | 商品牌组已完成，等待用户反馈 |
| `completed` | 当前业务流程完成 |
| `cancelled` | 用户取消本轮 |
| `error` | 本轮异常结束 |

收到 `done` 只表示当前 turn 的 SSE 生命周期结束，不代表整个 session 结束。例如：

```text
done(awaiting_product_feedback)
```

表示客户端应停止 loading，但保留牌组交互，用户反馈后再开启新 turn。

异常分支必须发送：

```text
error(code, message, retryable)
→ done(error)
```

`error` 提供用户可见错误信息；`done(error)` 负责关闭本轮流式生命周期。如果只有 `error`，客户端无法可靠判断后端是否还会继续发送事件。

#### 客户端按事件类型驱动，而不是依赖固定顺序

不同业务分支的事件序列不同：

```text
槽位不足：
thinking → text_delta → clarification → done

多商品推荐：
thinking → product_card × N → text_delta → criteria_card
→ done(awaiting_product_feedback)

单商品推荐：
product_card → criteria_card → thinking(decision)
→ final_decision → done(completed)

异常：
已发送的任意事件 → error → done(error)
```

所以后端需要保证每个分支中的局部顺序和字段语义，而不能要求客户端按照一份固定事件模板消费。

#### 后端只需了解的 Android 消费流程

后端掌握到以下程度即可：

```text
OkHttp EventSource
→ SseEventParser
→ AgentUiEnvelope<AgentPayload>
→ ChatReducer
→ ChatUiState
```

- Parser 根据 `event` 选择 `ThinkingPayload`、`ProductCardPayload` 等类型。
- Reducer 按 `node_id` 更新节点、按 `message_id` 拼文本、按 `deck_id` 聚合商品。
- `done` 将 `isStreaming` 设为 false。
- `error` 将输入状态切换为 Error。

不需要继续学习 Compose 组件布局、动画、页面导航或 ViewModel 的业务实现。

#### 后端接口契约变更检查单

修改 SSE 前至少检查：

1. 新字段是否真有稳定业务语义，而不是临时 UI 需求。
2. 字段是必填还是可选，默认值和空值分别代表什么。
3. 是否改变 `event`、`finish_reason` 或枚举的封闭集合。
4. 是否影响 `node_id`、`deck_id`、`message_id` 的聚合语义。
5. 是否同步 Schema、Python、Android 和 golden trace。
6. 旧客户端遇到新字段或新枚举时是否还能安全降级。
7. 是否补充乱序、重复事件、错误结束和取消分支测试。

答辩表达：

> 后端通过封闭 SSE 契约与 Android 通信。每个事件携带 session、turn、seq、event_id 和 node_id：turn_id 隔离一轮请求，seq 保证本轮顺序，event_id 用于去重，node_id 用于更新逻辑节点；商品卡通过 deck_id 聚合并关联反馈，文本增量通过 message_id 聚合。客户端按事件类型驱动状态，而不是依赖固定事件顺序；done 是每个 turn 的终止符，finish_reason 决定后续是等待标准调整、商品反馈还是已经完成。Android 虽然有兼容 fallback，但后端仍必须严格遵守 JSON Schema。

---

## 4. Intent、槽位检查与 Handler 路由

### 本章调用关系卡

| 维度 | 说明 |
|---|---|
| 上游调用者 | `_run_chat_turn()` 调用 `_resolve_intent()` |
| 调用时机 | 请求预处理完成后、Criteria 和检索之前 |
| 核心输入 | 当前消息、history、上一轮商品/Criteria、图片分析文本 |
| 核心输出 | `IntentResult`；缺槽位时直接产生 Clarification 事件 |
| 下游调用 | `_dispatch_intent_handler()` 从 `INTENT_HANDLERS` 查表调用具体 Handler |

**为什么需要这一步？**

用户输入可能是“推荐洗面奶”“把第一个加入购物车”“换一组”“对比第一款和第三款”，也可能只是闲聊。系统需要先把自然语言收敛成一个稳定的业务命令，才能决定是否进入 RAG 检索、购物车或商品对比流程。

这一阶段不只是“调用 LLM 做分类”，而是一个混合路由系统：

```text
确定性规则负责明确、高频的操作判断
LLM 负责复杂语言与实体理解
确定性后处理负责补全、归一化与纠错
跨轮合并负责继承历史条件
槽位检查负责判断当前信息是否足够
Handler 负责执行具体业务
```

### 4.1 在完整 Pipeline 中的位置

意图识别并不是用户消息进入 Pipeline 后的第一行代码。真实顺序是：

```text
图片预处理（可选）
→ 商业声明安全拦截
→ _resolve_intent() 识别和精炼意图
→ _merge_followup_context() 合并历史条件
→ 检查商品类型是否受支持
→ 槽位检查
→ INTENT_HANDLERS 分发
```

“有货吗”“包邮吗”“有优惠券吗”等问题会在意图识别前被拦截，返回固定安全文案。因为系统没有真实库存、物流和优惠数据，不能让 LLM 根据语言习惯编造答案。

### 4.2 统一输出：IntentResult

无论结果来自规则还是 LLM，最终都必须收敛为 `IntentResult`：

```python
class IntentResult(BaseModel):
    intent: Literal[
        "recommend", "clarify", "continue", "feedback", "compare",
        "add_to_cart", "remove_from_cart", "update_cart_quantity",
        "view_cart", "checkout_preview", "checkout_confirm",
        "checkout_cancel", "chitchat",
    ]
    confidence: float = 1.0
    category: str | None = None
    extracted_constraints: dict[str, Any] = Field(default_factory=dict)
    soft_preferences: list[str] = Field(default_factory=list)
    target_product_id: str | None = None
    target_product_name: str | None = None
    compare_product_ids: list[str | int] = Field(default_factory=list)
```

主要字段的职责：

| 字段 | 作用 |
|------|------|
| `intent` | Handler 路由键，决定执行推荐、加购、对比还是闲聊 |
| `category` | 商品一级品类，如美妆护肤、数码电子 |
| `extracted_constraints` | 意图阶段提取的初步预算、肤质、商品类型等 |
| `target_product_id/name` | 购物车操作的目标商品 |
| `compare_product_ids` | 待对比的商品引用 |

Java 类比：

```text
IntentResult ≈ Command DTO
```

它用于表达“用户要执行什么命令”，但还不是正式的检索条件。正式检索条件由下一阶段 `CriteriaPayload` 生成。

### 4.3 规则优先，但规则无法确定时不猜

核心入口是 [pipeline.py](../../backend/src/runtime/pipeline.py) 的 `_resolve_intent()`。规则按业务优先级执行：

```text
1. 预算修改
2. 前端 converge 收敛信号
3. 结算 / 购物车 / 短推荐 / 闲聊快速规则
4. 加购目标是否需要 LLM 补实体
5. 换一组
6. 商品对比
7. 全部无法确定时调用 LLM
```

规则函数的返回约定：

```text
返回 IntentResult → 当前规则给出了确定结论
返回 None         → 当前规则无法确定，继续下一条规则或 LLM
```

`None` 不是业务意图，也不是执行失败。

结算、购物车和闲聊规则使用 Python 的短路求值：

```python
determined = (
    maybe_checkout_intent(message)
    or maybe_cart_intent(message)
    or maybe_shopping_intent(message)
)
```

它会从左到右寻找第一个非 `None` 的结果。Java 中可以理解为：

```java
IntentResult result = maybeCheckout(message);
if (result == null) result = maybeCart(message);
if (result == null) result = maybeShopping(message);
```

需要特别注意：`maybe_shopping_intent()` 是历史命名，当前主要用于识别短问候和能力询问并返回 `chitchat`；短推荐请求主要在 `maybe_cart_intent()` 中识别。

当请求携带 `image_url` 或 `criteria_patch` 时，这组快速规则会被跳过。图片语义和标准卡修改包含更多上下文，需要走更完整的语义处理链路。

### 4.4 主要确定性规则

**预算修改**

用户说“预算降到 200”“再便宜一点”时，系统读取上一轮 Criteria，把预算变化写入 `criteria_patch`，并构造 `recommend` 意图。后续 Criteria 阶段负责将补丁合并到正式标准。

**前端收敛信号**

当 Android 传入 `converge=true`，表示用户已经提供足够反馈，希望系统基于当前候选做收敛决策。Pipeline 构造 `continue` 意图，后续 `handle_continue()` 会决定继续推荐还是进入最终决策。

**购物车与短推荐**

明确的加购、删除、查看购物车以及较短的“推荐/帮我找”表达可以直接分类。例如：

```text
加入购物车       → add_to_cart
这个不要了       → remove_from_cart
打开购物车       → view_cart
推荐一款洗面奶   → recommend
```

规则还要区分：

```text
把这个去掉       → remove_from_cart
去掉含酒精的     → recommend + 排除条件
```

**换一组**

“换一组”“还有别的吗”等表达会继承上一轮 Criteria，构造新的 `recommend` 意图。真正排除已展示商品的动作会在推荐 Handler 中结合上一轮商品 ID 和反馈上下文完成。

**商品对比**

对比目标有两种来源：

- Android 直接传入 `compare_product_ids`；
- 用户说“对比第一个和第三个”，系统把序数映射到上一轮商品 ID。

最多接受 4 个对比商品。无法得到至少两个有效商品时，意图会降级；Pipeline 随后可能提示用户先获得足够候选，而不一定重新执行完整推荐。

### 4.5 规则和 LLM 如何协作处理加购

购物车动作与目标商品解析是两个不同问题：

```text
意图规则：判断用户是否要加购
目标解析：判断用户具体指向哪个商品
```

对于“把第一个加入购物车”：

```text
规则识别 add_to_cart
→ Pipeline 发现存在可解析的序数，不调用 LLM
→ handle_add_to_cart()
→ cart_rules.referenced_product_id()
→ “第一个”映射为上一轮 product_ids[0]
```

对于“把理肤泉加入购物车”：

```text
规则识别 add_to_cart
→ 规则无法直接得到 product_id
→ 暂时进入 LLM 路径，提取 target_product_name="理肤泉"
→ 代码恢复规则确定的 add_to_cart
→ referenced_product_id() 用商品名匹配商品目录
```

因此不是“整条请求都交给 LLM”，而是：

```text
规则确定做什么
LLM 补充对哪个商品做
代码重新锁定业务操作
```

`referenced_product_id()` 的真实解析优先级是：

```text
明确 target_product_id
→ 消息中直接携带的 product_id
→ LLM 提取的 target_product_name
→ “第一个/第二个”等序数
→ “这个/这款”等历史指代
→ 无法解析则发送 clarification，绝不猜测
```

并非所有“加进去”表达都能命中快速关键词。未命中的近义表达会直接进入 LLM 兜底，这是规则覆盖范围的正常边界。

### 4.6 LLM 兜底路径

规则都无法确定时：

```text
_resolve_intent()
→ stages.intent.run_intent()
→ services.llm_client.analyze_intent()
→ LLM Gateway（primary / fallback）
→ JSON 解析
→ Pydantic 校验
→ IntentResult
```

模型输入包含：

- 当前用户消息；
- Android 传来的对话历史；
- 图片信息；
- 服务端保存的会话摘要。

等待模型期间，`run_with_heartbeat()` 会发送 `thinking` 事件并检查取消状态。模型返回的 JSON 必须通过 `IntentResult` 校验；模型不能发明新的意图类型。

### 4.7 两层确定性后处理

LLM 输出不会直接进入 Handler。后处理分为两层。

**第一层：约束补充与领域纠错**

[intent_resolution.py](../../backend/src/services/intent_resolution.py) 的 `resolve_intent_constraints()` 会：

- 提取“再便宜一点”“不要太甜”等自然语言调整；
- 根据商品数据和同义词表补充品牌偏好；
- 将明显购物请求从错误的 `chitchat/feedback` 修正为 `clarify`；
- 根据用户文本补充缺失的一级品类；
- 补充 LLM 遗漏的 `product_type`；
- 根据商品类型纠正冲突的 `category`；
- 合并当前轮次中多个提取来源产生的列表约束并去重。

例如：

```text
LLM: category=数码电子, product_type=跑鞋
领域映射: 跑鞋 → 服饰运动
最终: category=服饰运动
```

这里的列表合并是“当前轮多个来源”的合并，例如：

```text
LLM 提取 + 调整规则提取 + 品牌规则提取
```

它不等同于跨轮历史继承。

**第二层：操作可执行性检查**

`pipeline.py` 继续检查：

- 是否需要恢复规则已经确定的 `add_to_cart`；
- 没有目标商品且没有历史候选时，加购是否应降级；
- 对比目标是否至少有两个；
- 场景化购物请求是否应该跳过普通品类槽位检查。

### 4.8 跨轮上下文合并

当前轮后处理完成后，`_merge_followup_context()` 会读取上一轮保存的 Criteria：

```text
上一轮：油性肤质 + 洗面奶
本轮：预算改为 200
最终：油性肤质 + 洗面奶 + 预算 200
```

合并原则：

- 本轮明确说出的信息优先；
- `clarify/continue/compare` 等延续性意图继承历史；
- 用户明确切换新品类时，不应把旧商品类型和品牌偏好带过去；
- 预算等仍有意义的通用条件可以按规则保留。

需要区分：

```text
_merge_constraints()       → 当前轮多个提取来源的约束合并
_merge_followup_context()  → 不同对话轮次之间的历史继承
```

### 4.9 槽位检查与 Handler 分发

得到最终意图后，并不会立即执行 Handler。Pipeline 会先检查是否缺少必要信息，目前最重要的是商品品类。

例如：

```text
用户：“帮我推荐一个”
IntentResult(intent="recommend", category=None)
→ 缺少 category
→ clarification 事件
→ done
→ 等待用户下一轮补充
```

如果槽位完整，则通过注册表分发：

```python
INTENT_HANDLERS = {
    "recommend": handle_recommendation,
    "clarify": handle_recommendation,
    "continue": handle_continue,
    "compare": handle_compare,
    "add_to_cart": handle_add_to_cart,
    "view_cart": handle_view_cart,
    "chitchat": handle_chitchat,
}
```

它体现的是“策略模式 + 注册表分发”：

- Pipeline 只负责查表和调用；
- Handler 负责具体用例；
- 避免大型 `if/elif`；
- 每个 Handler 可以独立测试。

`clarify` 映射到 `handle_recommendation` 并不意味着缺信息时不再澄清。真正缺槽位的请求已经在分发前发出 `clarification` 并提前结束；能进入注册表的 `clarify` 通常已经具备继续推荐所需的信息。

### 4.10 Intent 与 Criteria 的职责区别

Intent 回答：

> 用户要做什么？

例如：

```text
recommend / compare / add_to_cart
```

Criteria 回答：

> 如果要推荐，应该按照什么正式标准检索？

例如：

```text
product_type=洗面奶
skin_type=油性
budget_max=200
ingredient_avoid=[酒精]
```

Java 类比：

```text
IntentResult    ≈ Command DTO
CriteriaPayload ≈ Query DTO
```

快速规则只需要低成本确定业务路由，不负责完整理解所有购买条件。Intent 中的初步约束可以先构造投机 Criteria 并提前启动检索；正式 Criteria 随后负责生成封闭 DSL，并对投机结果重新过滤。

### 4.11 源码索引

| 职责 | 文件 | 关键函数/结构 |
|------|------|--------------|
| 意图流程入口 | [pipeline.py](../../backend/src/runtime/pipeline.py) | `_resolve_intent()` |
| 意图数据结构 | [schemas.py](../../backend/src/types/schemas.py) | `IntentResult` |
| 文本确定性规则 | [message_rules.py](../../backend/src/services/message_rules.py) | `maybe_*`、对比与调整规则 |
| LLM Stage | [intent.py](../../backend/src/runtime/stages/intent.py) | `run_intent()` |
| LLM 任务接口 | [llm_client.py](../../backend/src/services/llm_client.py) | `analyze_intent()` |
| Prompt | [intent_analysis.md](../../backend/prompts/intent_analysis.md) | 意图 JSON 契约 |
| 约束后处理 | [intent_resolution.py](../../backend/src/services/intent_resolution.py) | `resolve_intent_constraints()` |
| 跨轮合并 | [pipeline.py](../../backend/src/runtime/pipeline.py) | `_merge_followup_context()` |
| 购物车目标解析 | [cart_rules.py](../../backend/src/runtime/cart_rules.py) | `referenced_product_id()` |
| Handler 注册表 | [handlers.py](../../backend/src/runtime/handlers.py) | `INTENT_HANDLERS` |

### 4.12 自测问题

1. 为什么明确操作优先使用规则，而不是全部交给 LLM？
2. 规则返回 `None` 表示什么？
3. “把第一个加入购物车”和“把理肤泉加入购物车”的路径有什么区别？
4. 为什么 LLM 返回 `IntentResult` 后还要做确定性后处理？
5. 当前轮列表约束合并和跨轮历史继承有什么区别？
6. 为什么得到 `recommend` 后还需要 Criteria 阶段？
7. Handler 分发前为什么还需要槽位检查？

---

## 5. Criteria：从 Intent 到封闭检索 DSL

### 本章调用关系卡

| 维度 | 说明 |
|---|---|
| 上游调用者 | `handle_recommendation()` 调用 `run_criteria()`；同时可用 `criteria_from_intent()` 启动投机检索 |
| 调用时机 | Intent/槽位确认后、正式检索条件确定前 |
| 核心输入 | `IntentResult`、上一轮 Criteria、feedback、会话摘要、`criteria_patch` |
| 核心输出 | `CriteriaPayload` |
| 下游调用 | Criteria Card、`retrieve_with_evidence()`、推荐 Prompt、下一轮历史合并 |

### 5.1 Criteria 是什么，什么时候调用

Intent 回答“用户想执行什么动作”，Criteria 回答“推荐商品必须参考哪些购买条件”。

例如用户输入“推荐一款 200 元以内、适合油皮的洗面奶”：

```text
IntentResult
├─ intent = recommend
├─ category = 美妆护肤
└─ extracted_constraints = {budget_max: 200, skin_type: 油性, product_type: 洗面奶}

CriteriaPayload
├─ category = 美妆护肤
├─ constraints = Constraints(...)
├─ chips = ["美妆护肤", "油性肌肤", "200元内", "洗面奶"]
├─ summary = "美妆护肤，油性肌肤，200元内，洗面奶"
└─ field_sources = {...}
```

推荐请求进入 [`handle_recommendation()`](../../backend/src/runtime/handlers.py) 后，会调用 [`run_criteria()`](../../backend/src/runtime/stages/criteria.py) 生成本轮正式 Criteria。`continue` 分支是例外：[`handle_continue()`](../../backend/src/runtime/handlers.py) 会优先复用上一轮 Criteria，不一定重新调用 LLM。

可以用 Java 分层类比：

```text
IntentResult   ≈ Command / 路由 DTO
CriteriaPayload ≈ 经过校验和归一化的 Query DTO
Constraints    ≈ 字段固定的查询条件对象
```

### 5.2 为什么叫“封闭 DSL”

[`Constraints`](../../backend/src/types/sse_events.py) 用 Pydantic 显式声明允许出现的字段，例如预算、品牌排除、产品类型、肤质、成分、存储容量、运动类型和饮食偏好，而不是接收任意的 `dict[str, Any]`。

“封闭”的含义是：

- LLM 不能随意发明检索字段，未知字段会在 [`_sanitize_constraints()`](../../backend/src/services/llm_task_payloads.py) 中被丢弃。
- 每个字段都有固定类型，字符串预算等常见错误会先尝试转换，再交给 Pydantic 校验。
- 后续检索、排序、推荐理由和前端展示都围绕同一个结构消费数据，避免同一语义出现多套表示。

需要特别注意：**封闭 DSL 不等于所有字段都会成为硬过滤条件。** 当前 [`_FILTER_CHECKS`](../../backend/src/services/retriever.py) 直接硬过滤的是已反馈商品、品类、预算、排除品牌、排除产地、产品类型和规避特质等；`skin_type`、`use_scenario`、`storage`、`season` 等字段还可能用于查询文本、排序或推荐理由，而不是全部直接执行 `WHERE` 式过滤。

### 5.3 CriteriaPayload 的六个字段

数据模型定义在 [`CriteriaPayload`](../../backend/src/types/sse_events.py)：

| 字段 | 含义 | 主要消费者 |
|---|---|---|
| `criteria_id` | 本次标准的标识 | 会话状态、前端卡片 |
| `category` | 顶层商品品类 | 检索品类过滤、展示 |
| `constraints` | 机器可执行的封闭 DSL | 检索、排序、推荐理由 |
| `chips` | 前端可直接展示的条件标签 | `criteria_card` |
| `summary` | 面向用户的标准摘要 | 前端展示、上下文 |
| `field_sources` | 字段来自 `user`、`history` 还是 `inferred` | 来源标记、后处理审计 |

`field_sources` 的类型定义为 [`CriteriaFieldSource`](../../backend/src/types/sse_events.py)。它主要服务于来源追踪和确定性后处理，不应理解为前端必须依据它执行检索逻辑。

### 5.4 两条生成路径：投机 Criteria 与正式 Criteria

推荐 Handler 并不是先等待正式 Criteria，再开始检索，而是让“生成标准”和“检索”并发执行。

```text
IntentResult
   ├─ criteria_from_intent() ──→ spec_criteria ──→ 后台投机检索
   │
   └─ run_criteria() ──────────→ formal criteria
                                      │
投机结果 ─────────────────────────────┤
                                      ↓
                         用正式 Criteria 再过滤
                                      │
                           为空时执行正式串行检索
```

1. [`criteria_from_intent()`](../../backend/src/runtime/stages/criteria.py) 是纯计算函数，不调用 LLM，也不访问数据库。它只把 Intent 已提取出的合法字段快速转换为临时 Criteria。
2. [`handle_recommendation()`](../../backend/src/runtime/handlers.py) 立即用临时 Criteria 启动 `top_n=12` 的投机检索，同时调用正式的 `run_criteria()`。
3. 正式 Criteria 返回后，[`_post_filter_retrieval()`](../../backend/src/runtime/handlers.py) 会用完整标准重新筛选投机结果。
4. 如果投机结果全部被过滤，后续 [`continue_recommendation_from_criteria()`](../../backend/src/runtime/handlers.py) 会触发基于正式 Criteria 的串行检索。
5. 如果购物场景策略修改了 Criteria，原投机结果会被直接丢弃，因为它已经不再对应最终标准。

这不是“提前生成正式标准”，而是用一个可能不完整的临时查询换取 LLM 推理与检索 I/O 的时间重叠。它提高的是潜在延迟表现，不保证每次都命中；真实收益应通过 trace 中的阶段耗时与 fallback 比例评估。

### 5.5 正式 Criteria 的执行流程

正式入口是 [`run_criteria()`](../../backend/src/runtime/stages/criteria.py)：

```text
并发读取 previous_criteria、feedback_context、conversation_summary
                         │
              请求是否携带 criteria_patch？
                 ┌───────┴────────┐
                是                否
                │                 │
   apply_criteria_patch()   generate_criteria()
                │                 │
                │        JSON 解析、类型转换、历史合并
                │                 │
                └───────┬─────────┘
                        ↓
             来源标注与确定性约束清理
                        ↓
              product_type 最终兜底
                        ↓
               上下文丢失诊断与返回
```

`run_criteria()` 会先并发发起三个互不依赖的读取，减少串行等待：

- [`get_previous_criteria()`](../../backend/src/services/conversation_state.py)：上一轮正式标准。
- [`get_feedback_context()`](../../backend/src/services/feedback.py)：喜欢、排除商品和特质等反馈。
- [`get_conversation_summary()`](../../backend/src/services/conversation_state.py)：压缩后的会话上下文。

如果请求带有 `criteria_patch`，只需要上一轮标准，因此反馈和摘要任务会被取消，不调用 LLM。

### 5.6 `criteria_patch`：确定性的局部修改

前端标准卡快捷操作最终进入 [`apply_criteria_patch()`](../../backend/src/runtime/stages/criteria.py)。

合并规则是：

- patch 可以直接放约束字段，也可以放在 `constraints` 对象中。
- 不属于 `Constraints` 的字段会被忽略。
- 标量字段直接覆盖，例如 `budget_max: 200`。
- 列表字段执行“旧值 + 新值 + 去重”，例如原来排除酒精，再 patch 香精，最终保留二者。
- patch 不负责切换 `category`，只修改已有 Criteria 的约束。
- 修改后的字段来源标记为 `user`，并重新构建 `chips` 和 `summary`。
- `product_type` 会按当前品类归一化，无效值会被清空。

这条路径类似 Java 中对已有查询对象执行受控的 Patch Command：规则明确、无需模型、结果可预测。

### 5.7 LLM 路径：生成、解析与历史合并

没有 patch 时，[`generate_criteria()`](../../backend/src/services/llm_client.py) 会通过任务化 LLM 接口调用模型：

1. [`criteria_messages()`](../../backend/src/services/llm_task_payloads.py) 把当前消息、Intent、反馈、上一轮 Criteria、会话摘要和请求历史组织成 Prompt 输入。
2. Prompt 模板位于 [`backend/prompts/criteria_generation.md`](../../backend/prompts/criteria_generation.md)，要求模型输出 JSON。
3. [`criteria_from_live_payload()`](../../backend/src/services/llm_task_payloads.py) 负责解析和防御性校验，而不是让 Runtime 直接消费模型原始字符串。

历史合并规则必须区分两种情况：

- **同品类：** 以上一轮 `constraints` 为基础，再用 LLM 本轮返回字段覆盖。这里是字典覆盖语义，因此 LLM 返回的列表字段会整体替换旧列表。
- **切换品类：** 不继承旧约束，从空 `Constraints` 开始，避免把“敏感肌”“无糖”等条件泄漏到新品类。

因此，不能笼统地说“所有列表约束都会追加”：只有 `criteria_patch` 路径会追加并去重；LLM 正式结果在同品类合并时是本轮值覆盖历史值。

### 5.8 确定性后处理：为什么不能完全相信 LLM

模型生成结构化 JSON 后，还要执行代码规则兜底。

**第一层：类型与字段净化**

[`_sanitize_constraints()`](../../backend/src/services/llm_task_payloads.py) 会删除未知字段，转换数字和列表等常见类型；Pydantic 随后进行最终结构校验。

**第二层：`product_type` 与品类一致性**

[`sanitize_product_type_constraint()`](../../backend/src/services/criteria_sanitizer.py) 和 [`sanitize_criteria_product_type()`](../../backend/src/services/criteria_sanitizer.py) 会将产品类型归一化，并清除与品类不匹配的值。

**第三层：字段来源标注与危险约束清理**

[`annotate_criteria_sources()`](../../backend/src/runtime/stages/criteria.py) 按以下优先级处理每个字段：

```text
本轮 Intent 明确提取 → user
与上一轮值相同       → history
模型补充的普通字段   → inferred
模型擅自补充的硬约束 → 清空
```

当前被禁止仅凭模型推断的硬约束包括 `budget_min`、`budget_max`、`brand_avoid`、`origin_avoid` 和 `ingredient_avoid`。这些字段会直接缩小候选集，如果用户和历史都没有提供，代码会清除它们，防止模型替用户“做主”。

发生品类切换时，所有非本轮明确提供的约束都会被清除。这个规则比普通的硬约束清理更严格。

**第四层：Intent 的 `product_type` 最终优先**

在 [`run_criteria()`](../../backend/src/runtime/stages/criteria.py) 末尾，如果 Intent 已通过规则或确定性解析得到明确 `product_type`，而 LLM 仍保留旧值，则用 Intent 值覆盖，解决“从手机切换到电脑但模型还沿用手机”的问题。

**第五层：只诊断，不自动修复**

[`diagnose_criteria_context()`](../../backend/src/runtime/stages/criteria.py) 检查预算修改或排除条件是否在多轮合并中丢失，并写入审计事件。它只负责发现问题，不会再次修改 Criteria。

### 5.9 当前实现需要知道的两个细节

以下是基于当前源码的实现边界，答辩时应区分“设计目标”和“现状”：

1. `criteria_from_intent()` 会把投机 Criteria 中的 Intent 字段统一标成 `user`。但 Intent 内的部分值可能来自规则补全或推断，因此这个标记更准确地说是“本轮 Intent 已确定”，不一定全部都是用户逐字表达。
2. `run_criteria()` 最后的 `product_type` 强制覆盖会更新 `constraints` 和 `chips`，但当前没有同步重建 `summary` 与 `field_sources`。主检索约束仍会使用新值，不过展示摘要和来源元数据存在短暂不一致的可能。这是当前实现可继续完善的点，不应描述成已经完全同步。

### 5.10 本阶段面试表达

可以这样概括：

> 我们没有让 LLM 直接拼检索条件，而是先把自然语言转换为 Pydantic 定义的封闭 Criteria DSL。推荐链路会用 Intent 快速构造投机 Criteria，让检索与正式 Criteria 的 LLM 生成并发；正式结果返回后再执行字段净化、历史合并、来源标注、危险硬约束清理和二次过滤，投机结果不满足完整标准时再回退到正式检索。这样兼顾了性能、可控性和多轮一致性。

自测问题：

1. Intent 和 Criteria 分别负责什么？
2. 为什么封闭 DSL 不等于所有字段都是硬过滤？
3. 投机 Criteria 为什么可能失败，失败后如何兜底？
4. `criteria_patch` 与 LLM 正式结果对列表字段的合并语义有什么不同？
5. 为什么预算、排除品牌等字段不能只由 LLM 推断？
6. 品类切换时为什么必须清理旧约束？
7. `diagnose_criteria_context()` 是修复逻辑还是审计逻辑？

---

## 6. 混合检索与证据绑定

### 本章调用关系卡

| 维度 | 说明 |
|---|---|
| 上游调用者 | `runtime.stages.recommendation.run_retrieval()` |
| 核心入口 | `services.retriever.retrieve_with_evidence()` |
| 调用时机 | 正式或投机 Criteria 可用后；多轮换一组、反馈重推时也会再次进入 |
| 核心输入 | `CriteriaPayload`、`top_n`、feedback、可选 `image_embedding` |
| 核心输出 | `RetrievalOutput(products, evidence_by_product, trace_details)` |
| 下游调用 | 推荐 Handler 发送 Product Card、构建推荐 Prompt、记录 Retrieval Trace/Evidence Link |

### 6.1 它在什么时候调用，负责什么

真实运行顺序是：

```text
Intent 意图识别
    ↓
槽位检查
    ├─ 信息不足 → clarification + done，本轮不进入检索
    └─ 信息足够
          ↓
      Handler 分发
          ↓
      Criteria 生成
          ↓
      混合检索
          ↓
      推荐文案生成
```

因此，槽位检查确实在正式 Criteria 和混合检索之前。前面按模块分阶段学习，并不代表模块名称就是完整运行顺序。

Criteria 解决“要找什么”，例如：

```text
品类 = 美妆护肤
产品类型 = 洗面奶
肤质 = 油性
预算上限 = 200
```

混合检索解决“商品知识库中哪些商品符合条件、与需求相关，并且有哪些原始文本可以支持推荐”。推荐请求进入 [`handle_recommendation()`](../../backend/src/runtime/handlers.py) 后，会通过 [`run_retrieval()`](../../backend/src/runtime/stages/recommendation.py) 调用 [`retrieve_with_evidence()`](../../backend/src/services/retriever.py)。

检索阶段返回结构化结果，而不是最终自然语言：

```python
RetrievalResult(
    products=[...],                 # 最终候选商品
    evidence_by_product={...},      # product_id → 证据片段
    trace_details={...},            # 过滤、放宽、排名等追踪信息
)
```

推荐链路还会用 [`criteria_from_intent()`](../../backend/src/runtime/stages/criteria.py) 构造临时 Criteria，让投机检索与正式 Criteria 的 LLM 生成并行。正式 Criteria 返回后，[`_post_filter_retrieval()`](../../backend/src/runtime/handlers.py) 会重新筛选投机结果；结果为空时再使用正式 Criteria 串行检索。

### 6.2 Chunking、Embedding 与 pgvector

这三个概念不能混淆：

| 概念 | 发生时间 | 作用 |
|---|---|---|
| Chunking | 商品数据入库时 | 把完整商品知识拆成多个语义相对独立的文本块 |
| Embedding | 入库和查询时 | 把文本转换成固定维度的数字向量 |
| pgvector | 查询时 | 计算查询向量与 chunk 向量的距离，寻找近邻 |

商品入库流程：

```text
商品原始数据
    ↓ build_product_chunks()
profile / marketing / FAQ / review / risk / compare 等 chunk
    ↓ embed_texts()
每个 chunk 得到一个 1024 维向量
    ↓
文本、metadata、product_id、embedding 一起存入 product_chunks
```

Chunk 的构建逻辑在 [`build_product_chunks()`](../../backend/src/services/chunking.py)，持久化模型是 [`ProductChunk`](../../backend/src/repos/models.py)，Embedding 门面是 [`embed_texts()`](../../backend/src/services/embedding.py)。

查询流程：

```text
CriteriaPayload
    ↓ criteria_query_text()
"美妆护肤 油性肌肤 洗面奶 200元内"
    ↓ embed_text()
1024 维查询向量
    ↓ pgvector
与数据库中的每个可召回 chunk 向量计算余弦距离
    ↓
距离最小的 Top-K chunk
```

Embedding 不负责“根据相似度把数据存到某个位置”。它只生成可比较的语义坐标；真正计算距离和查找近邻的是 pgvector。

### 6.3 为什么检索单位是 Chunk，而不是整件商品

一件商品可以包含多种语义完全不同的内容：

```text
商品 A
├─ profile：名称、品牌、品类和基础属性
├─ marketing：商品卖点
├─ FAQ：常见问题
├─ positive_review：正面评价
├─ negative_review / warning：差评或风险
└─ compare：比较维度
```

如果把所有内容拼成一个大文本只生成一个向量，营销、评价、FAQ 和风险会相互稀释。拆成 chunk 后，系统可以准确召回与当前需求最相关的局部内容，也能说明推荐依据来自哪段原始数据。

数据库召回的 [`VectorChunkHit`](../../backend/src/repos/documents.py) 包含 `ChunkDocument + distance`。随后 [`_vector_recall_from_pgvector()`](../../backend/src/services/retriever.py) 使用 `chunk.product_id` 调用 `get_product()`，把 chunk 还原成完整 [`ProductPayload`](../../backend/src/types/sse_events.py)。

同一商品可能有多个 chunk 命中，因此 Rerank 后还要按 `product_id` 去重：

```text
商品 A 的 chunk A1 ┐
商品 A 的 chunk A2 ├→ 商品 A → 完整商品卡 + 选中的证据
商品 A 的 chunk A3 ┘
```

客户端最终需要的是商品卡，不是零散文本；但命中的 chunk 不能丢弃，因为它们是推荐的证据来源。

### 6.4 完整混合检索链路

当前设计链路由 [`retrieve_with_evidence()`](../../backend/src/services/retriever.py) 编排：

```text
① 检查 RetrievalCache
    ↓ 未命中
② Criteria → 查询文本 → Embedding 查询向量
    ↓
③ SQL 结构化预过滤 + pgvector chunk 召回
    ↓
④ chunk.product_id 还原商品 + Python 硬过滤
    ↓ 严格结果为空
⑤ 渐进式放宽后重新执行向量召回
    ↓
⑥ BM25 关键词排名 + RRF 排名融合
    ↓
⑦ 合并可选的图片向量结果和品牌偏好结果
    ↓
⑧ 粗排并截取 top_n × 8 个候选 chunk
    ↓
⑨ qwen3-rerank 精排
    ↓
⑩ 按 product_id 去重为最终商品
    ↓
⑪ 绑定证据、记录 trace、写入缓存
```

需要注意，当前真实代码是“严格向量召回 → 必要时渐进式放宽 → BM25/RRF”，不是先执行 BM25 再放宽。

### 6.5 SQL 预过滤与 Python 硬过滤

只使用向量检索无法可靠满足预算、品类和排除项等刚性条件。例如“300 元洗面奶”和“200 元以内的洗面奶”语义仍然相近，纯向量检索可能返回超预算商品。

SQL 预过滤由 [`VectorSearchFilters`](../../backend/src/repos/documents.py) 和 [`_pgvector_sql_filters()`](../../backend/src/repos/documents.py) 表达，设计目标是在执行向量距离排序前缩小搜索空间：

```sql
SELECT pc.*, pc.embedding <=> :query_embedding AS distance
FROM product_chunks pc
JOIN products p ON p.id = pc.product_id
WHERE pc.embedding IS NOT NULL
  AND category = :category
  AND price <= :budget_max
  AND ...
ORDER BY pc.embedding <=> :query_embedding
LIMIT 200;
```

`<=>` 返回余弦距离，**越小越相关**。`LIMIT 200` 指最多召回 200 个 chunk，不是 200 个商品；上限来自 [`PGVECTOR_RECALL_LIMIT`](../../backend/src/config/tuning.py)。

召回后，[`_passes_hard_filters()`](../../backend/src/services/retriever.py) 还会串联七个纯函数复核：

```python
_FILTER_CHECKS = (
    _passes_feedback_product_filter,
    _passes_category_filter,
    _passes_budget_filter,
    _passes_brand_filter,
    _passes_origin_filter,
    _passes_product_type_filter,
    _passes_avoid_trait_filter,
)
```

两层过滤的职责不同：

| 层次 | 主要目的 | 适合处理 |
|---|---|---|
| SQL 预过滤 | 尽早缩小搜索空间，减少向量计算 | 品类、预算、商品类型、排除 ID 等结构化条件 |
| Python 硬过滤 | 保证完整业务正确性 | 品牌别名、类型归一化、规避特质文本匹配等复杂规则 |

答辩时可以概括为：SQL 过滤主要解决性能，Python 过滤主要解决业务正确性。

### 6.6 BM25、RRF 与 Rerank

Embedding/pgvector 擅长语义召回，但对品牌、型号等精确词可能不够敏感。[`BM25Index`](../../backend/src/services/bm25_recall.py) 使用分词和关键词统计生成另一份 chunk 排名；[`rrf_merge()`](../../backend/src/services/rrf_merge.py) 使用排名位置而不是原始分数融合两路结果：

```text
RRF score(chunk) = Σ 1 / (k + rank_i)
```

但当前实现将 RRF 结果映射回 `chunk_by_id` 时，`chunk_by_id` 只包含向量召回结果。因此：

> BM25 会独立搜索和计算排名，但 BM25-only 的 chunk 不会进入候选集；当前作用更准确地说是“用关键词排名信号重排向量候选”，还没有真正扩大召回集合。

粗排后的候选数量由 `top_n × RETRIEVAL_CANDIDATE_MULTIPLIER` 控制。默认返回 5 个商品、倍数为 8，因此最多约 40 个 chunk 进入 [`_rerank_chunk_hits()`](../../backend/src/services/retriever.py)。

Embedding 与 Rerank 的区别：

| Embedding 召回 | Rerank 精排 |
|---|---|
| 查询和文档分别编码 | 模型同时阅读查询和候选文档 |
| 适合快速处理大量 chunk | 只处理较小候选集 |
| 负责提高召回范围 | 负责提高候选排序精度 |
| 理解相对粗、成本较低 | 理解更细、延迟和成本更高 |

Rerank 输入不是只有商品名称，而是 [`product_document_text()`](../../backend/src/services/retrieval_features.py) 生成的“商品信息 + 当前命中 chunk 文本”。模型返回候选索引的新顺序，代码随后按 `product_id` 去重为最终商品。

### 6.7 渐进式放宽、反馈、图片与缓存

严格检索结果为空时，[`_relaxation_attempts()`](../../backend/src/services/retriever.py) 依次尝试：

1. 预算上限扩大到原来的 1.3 倍。
2. 预算上限扩大到原来的 1.5 倍。
3. 移除预算上限。
4. 移除反馈中的 `avoid_traits`。

品类和 `product_type` 在这组放宽中不会被移除。预算比例来自 [`BUDGET_RELAXATION_STEPS`](../../backend/src/config/tuning.py)，属于实验调参，不是通用最优值。

用户反馈通过 [`_retrieval_filters()`](../../backend/src/services/retriever.py) 转换为：

- `avoid_products`：排除已经不喜欢或要求换掉的商品。
- `avoid_traits`：排除不喜欢的特质。

请求携带图片向量时，文本向量召回与 [`list_products_by_image_similarity()`](../../backend/src/repos/documents.py) 会并行执行，失败时降级为纯文本检索。图片查询不使用检索缓存；普通文本查询会使用进程内 [`RetrievalCache`](../../backend/src/services/retrieval_cache.py)，缓存键由完整 Criteria 和 feedback 共同计算。

### 6.8 证据绑定与“检索不全”问题

[`_evidence_by_product()`](../../backend/src/services/retriever.py) 会从 Rerank 后的全部 chunk 中，为最终商品按以下类型选择证据：

```text
why_buy → faq → risk → compare
```

每种类型最多保留一个排名最高的 chunk。

但是，向量 Top-K 只保证找到“与当前查询最相关”的 chunk，不保证找全某件商品的所有信息。假设用户搜索“适合油皮的洗面奶”，系统可能召回控油卖点和油皮好评，却不一定召回“敏感肌使用刺痛”或“包装容易漏液”等语义较远的评价。

当前普通 pgvector 查询还显式排除了 `retrieval_role = risk` 的 chunk：

```sql
AND COALESCE(pc."metadata"->>'retrieval_role', '') <> 'risk'
```

这样可以避免差评或警告文本因为语义相似而把商品当作正向候选召回，但也意味着普通候选召回不能承担完整风险收集。项目另外提供了 [`risk_chunks_for_products()`](../../backend/src/repos/documents.py)，并在 [`recommendation_reasons.py`](../../backend/src/services/recommendation_reasons.py) 中按最终商品 ID 补查风险 chunk。

更稳健的通用设计应明确分成两步：

```text
第一步：使用正向 chunk 召回候选商品
第二步：确定最终商品后，按 product_id 补查 FAQ、评价和风险 chunk
```

这叫“候选召回”和“证据补全”分离。当前项目已经具备风险补查路径和 chunk 类型划分，但主检索中的证据覆盖仍不是“商品全部信息”，答辩时不能宣称一次向量查询就能检索完整。

### 6.9 当前源码真实缺陷

本次源码审计发现以下调用契约漂移，尚未修复：

1. [`_sql_filters_for_recall()`](../../backend/src/services/retriever.py) 构造 `VectorSearchFilters` 时传入 `product_type`、`avoid_brands`，而模型字段实际是 `budget_max`、`product_type_aliases`、`brand_avoid`、`avoid_product_ids`。当前还没有正确传入 `budget_max`，正常召回会在 SQL 查询前因参数不匹配失败；即使修正构造参数，[`_sql_filter_payload()`](../../backend/src/services/retriever.py) 仍在读取旧字段名，也必须同步修复。
2. [`_post_filter_retrieval()`](../../backend/src/runtime/handlers.py) 调用 `filter_products(..., max_products=...)`，但当前 [`filter_products()`](../../backend/src/services/retriever.py) 不接收 `max_products`。
3. 品牌偏好直取路径调用 `list_products(filters=...)`，但当前 [`list_products()`](../../backend/src/repos/products.py) 不接收 `filters`。
4. `relaxation_steps` 当前写在 `trace_details` 顶层，而 [`_budget_was_relaxed()`](../../backend/src/runtime/handlers.py) 从 `filters_applied.relaxation_steps` 读取，可能无法识别预算放宽。
5. `retriever.py` 中部分“Top-50”注释已过时；当前配置上限是 200 个 chunk。

这些属于“设计目标基本完整，但部分增强路径接口未同步”的实现问题。修复时应以 `VectorSearchFilters` 和公开函数签名为唯一契约，并补充 SQL 过滤构造测试、投机检索集成测试和品牌偏好路径测试。

检索测试需要 PostgreSQL + pgvector。当前本地检查因未配置 `DATABASE_URL` 未能执行真实检索测试，因此不能把静态阅读等同于运行验证。

### 6.10 本阶段答辩表达

可以用下面这段概括：

> 系统先把 Criteria 转换成查询文本并生成 Embedding，通过 SQL 结构化条件缩小搜索空间，再使用 pgvector 执行 chunk 级语义召回。召回 chunk 通过 product_id 还原为商品，并经过 Python 硬过滤复核预算、品类和排除条件；随后使用 BM25/RRF 引入关键词排名信号，再通过 Rerank 对小候选集精排，最终按商品去重并绑定命中 chunk 作为推荐证据。Chunking、Embedding 和检索分别负责文本切分、语义向量化和近邻查询。由于 Top-K 向量召回不能保证找全商品的所有评价，系统应将候选召回与风险、FAQ 等证据补全分开处理。

面试追问：

1. 为什么不能只使用向量检索？
2. Chunking、Embedding 和 pgvector 分别负责什么？
3. SQL 预过滤和 Python 硬过滤为什么要同时存在？
4. 为什么先召回 chunk，再按 `product_id` 聚合为商品？
5. Embedding 召回与 Rerank 精排有什么区别？
6. 当前 BM25 是否真正扩大了召回集合？
7. 为什么一次 Top-K 向量检索不能保证找全商品的所有评价？
8. 为什么 risk chunk 不应直接参与普通正向商品召回？
9. 如何设计“候选召回”和“证据补全”两阶段流程？

---

## 7. 推荐生成、商品卡与防幻觉

### 本章调用关系卡

| 维度 | 说明 |
|---|---|
| 上游调用者 | `handle_recommendation()` 或 `handle_continue()` 在检索完成后调用 |
| 核心编排 | `continue_recommendation_from_criteria()` |
| 调用时机 | 候选商品和 Evidence 已确定后 |
| 核心输入 | Criteria、商品候选、Evidence、feedback、场景策略 |
| 核心输出 | Product Card、流式推荐正文、可选最终决策与 `done` |
| 下游调用 | `llm_client.stream_recommendation()`、Grounding Guard、会话持久化和 Trace |

### 7.1 这一阶段负责什么

混合检索返回：

```python
RetrievalResult(
    products=[...],
    evidence_by_product={...},
    trace_details={...},
)
```

推荐阶段负责把这些结构化结果转换成用户可见内容：

```text
候选商品
    ↓
商品卡片、确定性推荐理由、风险提示
    ↓
多商品推荐正文或单商品最终决策
    ↓
输出检查、状态保存与下一步操作
```

核心边界是：

> 检索和确定性代码决定“允许推荐哪些商品、事实是什么”；LLM 只负责“怎样解释这些候选”，不能自行扩充候选集。

主入口是 [`continue_recommendation_from_criteria()`](../../backend/src/runtime/handlers.py)。它由以下路径调用：

- [`handle_recommendation()`](../../backend/src/runtime/handlers.py)：首次推荐、反馈后重推、通过槽位检查的 `clarify` 等分支。
- [`handle_continue()`](../../backend/src/runtime/handlers.py)：需要基于上一轮 Criteria 再次推荐时。

### 7.2 按调用时机理解完整流程

| 业务逻辑 | 什么时候调用 | 被谁调用 | 输出或作用 |
|---|---|---|---|
| 获取检索结果 | 进入推荐续跑且没有投机预计算结果时 | `continue_recommendation_from_criteria()` 调 `stages.run_retrieval()` | `RetrievalResult` |
| 无结果兜底 | 首次检索没有商品时 | `continue_recommendation_from_criteria()` | 可移除 `product_type` 再检索，或返回调整条件提示 |
| 关键词最终重排 | 检索有商品，且 Criteria 有 chips 或 `product_type` 时 | `continue_recommendation_from_criteria()` 调 `keyword_boost_score()` | 调整商品展示顺序 |
| 预算放宽提示 | trace 表示检索放宽了预算时 | `continue_recommendation_from_criteria()` | 在商品卡前发送超预算风险文本 |
| 商品卡生成 | 有至少一个候选商品时 | `continue_recommendation_from_criteria()` 调 `_product_card_events()` | 多个 `product_card` SSE 事件 |
| 风险 chunk 补查 | 开始生成一组商品卡时 | `_product_card_events()` 调 `fetch_risk_notes_for_products()` | `product_id → risk_notes` |
| Evidence 补查 | 某商品没有检索证据时 | `_product_card_events()` 调 `get_evidence()` | 商品证据列表 |
| Reason Atom 构建 | 每张商品卡生成时 | `_product_card_events()` 调 `build_reason_atoms()` | 最多 4 个结构化理由 |
| 商品卡短理由 | 没有场景策略统一理由时 | `_product_card_events()` 调 `reason_from_atoms()` | 使用前两个 atom 拼成短文案 |
| 单商品决策 | 最终只有一个候选时 | `continue_recommendation_from_criteria()` 调 `_run_decision_with_context()` | `final_decision`，不生成普通推荐正文 |
| 多商品流式正文 | 最终有两个及以上候选时 | `continue_recommendation_from_criteria()` 调 `run_recommendation_text_stream()` | `text_delta` SSE 流 |
| 流式转 SSE | LLM 持续返回文本 delta 时 | `_stream_recommendation_text_events()` | 缓冲文本并生成带 `message_id` 的 `TextDeltaEvent` |
| 非流式推荐降级 | 流式调用在首个 delta 前失败时 | `stream_recommendation()` 调 `generate_recommendation()` | 经校验的 `text_chunks` |
| 非流式输出校验 | `generate_recommendation()` 得到 JSON 后 | `generate_recommendation()` 调 `_validate_recommendation_chunks()` | 拒绝候选外商品和不受支持商业声明 |
| 流式价格检查 | 多商品正文已经全部发出后 | `continue_recommendation_from_criteria()` 调 `validate_price_claims()` | 必要时追加价格纠正文本 |

需要特别记住两个分支：

```text
0 个商品
→ 尝试放宽 product_type 或提示用户修改标准

1 个商品
→ 商品卡 → 确定性评分 → Decision LLM → final_decision
→ 跳过普通推荐正文

2 个及以上商品
→ 商品卡 → 流式推荐正文 → 价格检查
→ 等待用户反馈，不立即发送 final_decision
```

### 7.3 商品卡为什么先于 LLM 推荐正文

[`_product_card_events()`](../../backend/src/runtime/handlers.py) 在多商品流式正文之前执行。每张 [`ProductCardEvent`](../../backend/src/types/sse_events.py) 包含：

```text
rank
product
reason
reason_atoms
risk_notes
evidence
actions
```

其中商品名称、ID、品牌、价格、SKU 和图片来自 [`ProductPayload`](../../backend/src/types/sse_events.py)，推荐理由原子由代码生成，风险来自数据库 chunk，操作按钮也由后端固定构造。

这样即使推荐正文 LLM 失败，客户端仍然已经拿到可展示、可操作的商品卡。流式推荐异常会被记录为 fallback，但不会撤回已发送的商品卡。

### 7.4 Reason Atom：代码生成的事实原子

[`build_reason_atoms()`](../../backend/src/services/recommendation_reasons.py) 在两个时机被调用：

1. `_product_card_events()` 为每张商品卡生成结构化推荐理由。
2. `generate_recommendation()` 或 `stream_recommendation()` 在构造 LLM Prompt 时重新生成一份 `reason_atoms_by_product`，限制模型可使用的事实。

当前按顺序尝试构建：

```text
肤质匹配
产品类型匹配
预算匹配
使用场景匹配
偏好成分
规避成分
品牌规避
存储容量
饮食偏好
```

最多保留前四个；一个都没有时，兜底生成“类目匹配”。

例如：

```json
{
  "dimension": "budget",
  "value": "159",
  "text": "159元符合200元预算",
  "evidence_id": "product_field:price"
}
```

Reason Atom 与 LLM 正文的职责区别：

| Reason Atom | LLM 推荐正文 |
|---|---|
| 表达单个可验证事实 | 组织多商品的自然语言解释 |
| 由业务代码生成 | 由模型生成 |
| 可关联商品字段或 evidence ID | 只能使用传入候选、atoms 和 evidence |
| 可直接用于商品卡 | 用于较长的比较与阅读体验 |

Java 类比：

```text
ReasonAtomPayload ≈ 后端规则生成的 RecommendationReason DTO
LLM               ≈ 将这些 DTO 组织成文案的 View Renderer
```

### 7.5 风险信息为什么要单独补查

普通正向向量召回会排除 `retrieval_role = risk` 的 chunk，避免“敏感肌用了刺痛”等差评因为与“敏感肌”语义接近，反而把不合适商品召回。

候选商品确定后，[`_product_card_events()`](../../backend/src/runtime/handlers.py) 调用：

```text
fetch_risk_notes_for_products(product_ids)
    ↓
risk_chunks_for_products(product_ids)
    ↓
build_risk_notes(chunks)
```

它会按 `product_id` 查询负面评价、warning 和风险 FAQ，并为每件商品最多生成 3 条、每条最多约 180 字的风险提示。

因此当前设计是：

```text
正向 chunk → 召回候选商品
risk chunk → 候选确定后补充风险
```

这实现了“候选召回”和“风险证据补全”分离。

如果检索阶段没有为某商品提供 evidence，[`get_evidence()`](../../backend/src/services/evidence.py) 会在商品卡生成时补查普通证据；因此风险补查与普通 evidence 补查是两条不同路径。

### 7.6 多商品流式推荐正文

最终候选不少于两个时，[`continue_recommendation_from_criteria()`](../../backend/src/runtime/handlers.py) 获取会话摘要，然后调用：

```text
stages.run_recommendation_text_stream()
    ↓
runtime/stages/recommendation.py
    ↓
services.llm_client.stream_recommendation()
    ↓
recommendation_stream_messages()
    ↓
LLM Provider 流式输出
```

Prompt 模板是 [`backend/prompts/recommendation_stream.md`](../../backend/prompts/recommendation_stream.md)，输入包含：

- 正式 `CriteriaPayload`。
- 已检索并排序的候选商品。
- `evidence_by_product` 中的证据片段。
- `reason_atoms_by_product`。
- 会话摘要。

[`_format_evidence_context()`](../../backend/src/services/llm_task_payloads.py) 对每件商品最多放入 3 条 evidence，每条截取前 150 字。Prompt 要求模型：

- 只解释传入候选商品。
- 商品名称必须与候选完全一致。
- 不得编造价格、优惠、库存、物流和购买链接。
- 优先使用已校验 Reason Atoms。
- 证据不足时少说，不要硬推。

[`_stream_recommendation_text_events()`](../../backend/src/runtime/handlers.py) 把 Service 返回的字符串 delta 转换为 SSE：

```text
LLM delta
→ 后台 pump task 写入 asyncio.Queue
→ 缓冲与切块
→ TextDeltaEvent(message_id, delta, done)
```

它还会在等待 delta 时检查取消状态，并记录 recommendation 阶段耗时。

### 7.7 流式失败与非流式降级

[`stream_recommendation()`](../../backend/src/services/llm_client.py) 区分失败时机：

- **首个 delta 之前失败：** 调用 [`generate_recommendation()`](../../backend/src/services/llm_client.py) 获取非流式 JSON，再逐段 yield。
- **已经发送过 delta 后失败：** 继续抛出异常，避免将另一份完整文案拼接到半段流式文案后面。

上层 `continue_recommendation_from_criteria()` 捕获推荐正文异常后只记录 fallback 并继续流程。商品卡此前已经发送，因此用户仍能完成查看证据、反馈和加购。

[`run_recommendation_text()`](../../backend/src/runtime/stages/recommendation.py) 是非流式 Runtime 包装；[`run_recommendation()`](../../backend/src/runtime/stages/recommendation.py) 则是“检索 + 非流式推荐”的组合接口。主聊天多商品路径当前优先使用流式接口。

### 7.8 非流式输出校验

[`generate_recommendation()`](../../backend/src/services/llm_client.py) 要求模型返回：

```json
{
  "text_chunks": ["段落1", "段落2"]
}
```

最多保留 4 段，然后调用 [`_validate_recommendation_chunks()`](../../backend/src/services/llm_client.py) 检查：

1. 文本不能为空。
2. 出现的 `product_id` 必须属于本轮候选。
3. 不能提到商品库中存在、但本轮未入选的商品名称。
4. 不能包含系统禁止的商业声明。

校验失败会抛出异常，而不是把未经验证的非流式结果继续发送。

这组检查只直接覆盖非流式 `text_chunks`。主聊天使用的流式正文不会在发送前经过同一套完整校验。

### 7.9 流式价格 Grounding Guard

多商品正文发送完成后，[`validate_price_claims()`](../../backend/src/services/grounding_guard.py) 扫描正文中的：

```text
数字 + 元 / 块 / RMB
```

允许出现的价格来自：

- 候选商品基础价格。
- 候选商品 SKU 价格。
- 用户预算上下限。

如果模型输出的价格不在白名单中，系统会追加：

```text
该价格未在商品库中记录，请以商品卡片价格为准
```

调用时机是**流式正文已经发送之后**，所以这是事后纠正，而不是发送前拦截。SSE 已发送的内容无法撤回。

当前边界包括：

- 只匹配“元、块、RMB”，不检查 `¥`、美元等写法。
- 使用 `round()` 比较，存在小数价格精度折叠。
- 只能追加纠正，用户可能已经看到错误价格。
- 不检查价格之外的全部功效、库存或物流声明。

更严格的实现可以选择：

```text
完整缓冲模型输出 → 校验 → 再发送
```

或者：

```text
按句缓冲流式文本 → 句级校验 → 校验后释放
```

前者会失去真正的流式首字延迟，后者实现复杂度更高。

### 7.10 单商品与多商品分支

**单商品分支**

调用条件：`len(products) == 1`。

```text
商品卡
→ score_candidates() 确定性评分
→ decision_confidence()
→ _run_decision_with_context()
→ final_decision
→ done(completed)
```

该分支不调用普通 `stream_recommendation()`，推荐解释位于 Decision 的 `summary / why / not_for`。

**多商品分支**

调用条件：`len(products) >= 2`。

```text
商品卡
→ stream_recommendation()
→ text_delta
→ validate_price_claims()
→ criteria_card 与后续引导
→ done(awaiting_product_feedback)
```

多候选场景不会立即替用户选出最终商品，需要等待反馈或后续明确收敛请求。这是推荐解释阶段与最终决策阶段的职责边界。

### 7.11 防幻觉分层

| 防护层 | 触发时机 | 实现 |
|---|---|---|
| 候选集约束 | 检索阶段 | LLM 只能接收检索返回的商品 |
| 结构化商品事实 | 商品卡生成时 | `ProductPayload` 直接来自商品数据 |
| 确定性理由 | 商品卡与 Prompt 构建时 | `build_reason_atoms()` |
| 风险补查 | 商品卡生成前 | `fetch_risk_notes_for_products()` |
| Prompt 约束 | 调用推荐 LLM 时 | 只解释候选，优先 atoms/evidence |
| 非流式校验 | JSON 推荐结果返回后 | `_validate_recommendation_chunks()` |
| 流式价格检查 | 正文发送完成后 | `validate_price_claims()` 追加纠正 |

因此，当前系统是分层降低幻觉风险，而不是保证完全无幻觉。尤其是流式正文无法在发送前完成完整校验。

### 7.12 本阶段答辩表达

> 推荐阶段采用“结构化事实优先，LLM 负责表达”的设计。候选商品由检索确定，商品卡中的名称、价格、推荐理由原子、风险提示和证据都由后端代码基于数据库生成。多商品场景再把 Criteria、候选、Reason Atoms、证据和会话摘要交给 LLM 生成流式解释；单商品场景则直接进入确定性评分和最终决策。非流式输出会检查候选外商品和商业声明，流式正文完成后会校验价格，但由于 SSE 内容已经发送，当前只能追加纠正提示，这是现有防幻觉机制的边界。

自测问题：

1. `continue_recommendation_from_criteria()` 在 0、1、多个商品时分别走什么分支？
2. 为什么商品卡要在推荐正文之前发送？
3. `build_reason_atoms()` 分别在商品卡和 LLM Prompt 构造的什么时机调用？
4. 普通 evidence 补查和 risk chunk 补查有什么区别？
5. 流式推荐在首个 delta 前失败和发送部分 delta 后失败，降级行为为什么不同？
6. 非流式推荐比流式推荐多了哪些发送前校验？
7. 为什么 `validate_price_claims()` 只能算事后纠正？

---

## 8. 多轮反馈、换一组与最终决策

### 本章调用关系卡

| 维度 | 说明 |
|---|---|
| 上游入口 | `POST /feedback` 或下一轮聊天中的 feedback/continue/换一组表达 |
| 状态来源 | `Conversation` 快照 + `FeedbackRecord` 原子事件 |
| 调用时机 | 上一轮已经产生候选牌组后 |
| 核心处理 | 聚合 avoid/like/prefer → 决定重新检索、临时排除上一牌组或在当前牌组评分 |
| 下游输出 | 新一轮推荐、Clarification、FinalDecision 或等待更多反馈 |

### 8.1 两类状态：Conversation 快照与 Feedback 事件

多轮导购不是把全部历史聊天直接交给 LLM，而是同时维护两类后端状态：

| 状态 | 数据表 | 保存内容 | 作用 |
|---|---|---|---|
| 会话快照 | [`Conversation`](../../backend/src/repos/models.py) | 本轮消息、Criteria、deck_id、候选商品、商品卡、决策和正文 | 回放“系统当时展示了什么” |
| 用户行为事件 | [`Feedback`](../../backend/src/repos/models.py) | session、deck、商品、action、reason、时间 | 记录“用户后来做了什么” |

推荐完成后，[`_persist_recommendation()`](../../backend/src/runtime/handlers.py) 调用 [`save_recommendation_turn()`](../../backend/src/services/conversation_state.py)，把本轮 Criteria、商品牌组和输出保存到 `conversations`。

反馈不会直接把 `avoid_products` 等聚合状态写入数据库，而是先保存原子事件，读取时再动态聚合。这类似事件表与聚合快照的组合：

```text
原子事件：
用户对 p001 执行 not_interested，原因是“酒精味太重”

派生状态：
avoid_products = [p001]
avoid_traits = [酒精]
```

### 8.2 反馈的两个入口及调用时机

**入口一：前端按钮反馈**

当用户点击商品卡上的“不感兴趣”、喜欢或查看证据等操作时，Android 调用：

```text
POST /feedback
→ api.feedback.submit_feedback()
→ services.feedback.submit_feedback_request()
→ record_feedback()
→ repos.feedbacks.add_feedback()
→ feedbacks 表
```

请求模型是 [`FeedbackRequest`](../../backend/src/types/schemas.py)，包含 `session_id`、`deck_id`、`product_id`、`feedback_type/action` 和可选原因。

**入口二：自然语言反馈**

当用户说“第一款不喜欢，太刺激”并被识别为 `feedback` 意图时：

```text
INTENT_HANDLERS["feedback"]
→ handle_recommendation()
→ _record_feedback_intent()
→ referenced_product_id()
→ record_feedback()
→ feedbacks 表
→ 继续 Criteria / 检索 / 推荐链路
```

[`_record_feedback_intent()`](../../backend/src/runtime/handlers.py) 会解析商品指代、读取上一轮 `deck_id`，保存 `not_interested` 或普通 feedback，并记录审计事件。

### 8.3 Feedback 如何转换成下一轮上下文

需要读取反馈时调用：

```text
get_feedback_context()
→ extract_feedback_from_session()
→ get_session_feedbacks()
→ extract_feedback_context()
```

[`extract_feedback_context()`](../../backend/src/repos/feedbacks.py) 输出：

```python
{
    "avoid_products": [],
    "avoid_traits": [],
    "prefer_traits": [],
    "liked_products": [],
    "add_to_cart_products": [],
    "viewed_products": [],
}
```

转换规则：

- 负面 action 且带商品 ID → `avoid_products`。
- `like/right_swipe` → `liked_products`。
- `add_to_cart` → `add_to_cart_products`。
- `view_detail/open_evidence` → `viewed_products`。
- 负面 reason → [`extract_feedback_avoid_terms()`](../../backend/src/config/domain_terms.py) 提取 `avoid_traits`。
- 非负面 reason → `prefer_traits`。

当前混合检索的 [`_retrieval_filters()`](../../backend/src/services/retriever.py) 只直接消费 `avoid_products` 和 `avoid_traits`；其他行为主要进入会话摘要、候选评分或后续决策。

### 8.4 “不喜欢”和“换一组”为什么不同

“不喜欢第一款”表达稳定偏好，应持久化：

```text
反馈落库
→ 下一轮 avoid_products 排除该商品
→ reason 还可能提取为 avoid_traits
```

“换一组”只表示想看其他选择，不代表讨厌上一组全部商品。因此 Pipeline 通过 [`is_replace_deck_phrase()`](../../backend/src/services/message_rules.py) 确定性识别后：

```text
继承上一轮 Criteria
→ handle_recommendation()
→ get_previous_product_ids()
→ _feedback_with_avoided_products()
→ 本轮临时把上一牌组加入 avoid_products
→ 重新检索
```

[`_is_replace_deck_request()`](../../backend/src/runtime/handlers.py) 同时支持自然语言“换一组”和标准卡传入的 `criteria_patch.replace_deck=true`。

| 操作 | 是否作为负面反馈持久化 | 排除范围 |
|---|---|---|
| 不喜欢某商品 | 是 | 指定商品及可能的负面特质 |
| 换一组 | 否 | 仅本轮临时排除上一牌组 |

### 8.5 `continue` 何时重新检索，何时直接决策

[`handle_continue()`](../../backend/src/runtime/handlers.py) 先读取上一轮 Criteria 和商品 ID。

如果没有可执行反馈：

```text
continue
→ continue_recommendation_from_criteria()
→ 复用上一轮 Criteria
→ 再次检索和推荐
```

如果存在 `converge=true`、排除商品、排除特质、偏好特质或喜欢商品：

```text
continue
→ continue_decision_from_current_deck()
→ 复用当前 deck
→ 不重新检索
→ 在当前候选中收敛决策
```

因此 `continue` 不是固定等于“再检索一次”：

```text
反馈不足 → 继续推荐
反馈充分 → 当前牌组内决策
```

### 8.6 最终赢家由确定性评分决定

[`continue_decision_from_current_deck()`](../../backend/src/runtime/handlers.py) 的执行顺序是：

```text
读取当前 deck 和反馈
→ 移除 avoid_products
→ 为剩余商品补充 evidence
→ score_candidates()
→ decision_confidence()
→ 得到确定性 winner
→ _run_decision_with_context()
→ LLM 解释结果
→ final_decision
```

赢家由 [`score_candidates()`](../../backend/src/services/decision_scoring.py) 决定。调用 Decision LLM 时会传入 `locked_winner_product_id`；即使模型返回其他商品，最终结果仍锁定为评分算法选出的赢家。

职责边界是：

```text
确定性评分：决定谁赢
Decision LLM：解释为什么赢、适合谁、不适合谁
```

如果用户排除了当前牌组全部商品，系统返回：

```text
decision_status = no_suitable_winner
next_step = replace_deck
```

而不是强行选择一个商品。

### 8.7 多轮闭环总览

```text
第一轮推荐
→ 保存 Conversation 和 deck_id
→ 用户点击或输入反馈
→ 保存 Feedback 原子事件
→ 聚合 avoid / prefer / liked 上下文
→ 再次推荐，或在当前 deck 中收敛
→ 确定性评分选择赢家
→ LLM 解释决策
→ final_decision
→ 用户可继续对比、调整标准或加购
```

答辩表达：

> 多轮状态由 Conversation 快照和 Feedback 事件共同维护。Conversation 保存每轮 Criteria、候选牌组和输出，Feedback 保存用户的原子行为。负面反馈动态聚合为商品和特质排除条件；“换一组”只临时排除上一牌组，不会误认为用户讨厌全部商品。反馈足够时，系统不重新检索，而是在当前牌组中通过确定性评分选择赢家，LLM 只负责解释最终决策。

---

## 9. 多模态图片理解与图文召回

### 本章调用关系卡

| 维度 | 说明 |
|---|---|
| 上游入口 | `POST /upload/image` 保存图片；随后 `/chat/stream` 携带 `image_url` |
| 第一条支路 | `_prepare_pipeline_body()` → VLM 分析 → 将可见语义追加到消息 |
| 第二条支路 | `handle_recommendation()` → 图片 Embedding → pgvector 图片相似召回 |
| 汇合位置 | VLM 文本回到 Intent/Criteria；视觉候选在 `retrieve_with_evidence()` 与文本候选合并 |
| 最终输出 | 完整商品卡仍来自商品库；解释证据应来自文本 Chunk |

### 9.1 两条不同的图片能力

图片请求会使用两个相互独立的模型任务：

| 能力 | 回答的问题 | 输出 | 调用位置 |
|---|---|---|---|
| VLM 图片理解 | 图片里是什么、有哪些可见特征 | 结构化文本 | Pipeline 预处理阶段 |
| VL Embedding | 商品库中哪些图片长得相似 | 1024 维图片向量 | 推荐 Handler |

它们不是同一个调用，也不存在“先由 VLM 生成 Embedding”的关系：

```text
图片
├─ analyze_image() → category / description / visible traits
└─ embed_image()   → visual embedding
```

### 9.2 从上传到聊天请求

Android 先调用：

```text
POST /upload/image
→ upload_image()
→ handle_upload_image()
→ parse_multipart_image()
→ save_uploaded_image()
→ ImageUploadResponse
```

上传逻辑位于 [`api/upload.py`](../../backend/src/api/upload.py) 和 [`services/image_upload.py`](../../backend/src/services/image_upload.py)。当前支持 JPEG、PNG 和 WebP，最大 5 MB，请求必须使用 `multipart/form-data`。

图片以随机名称保存到 `UPLOAD_DIR`，例如：

```text
upload_{uuid}.jpg
```

并返回 `/uploads/{file}`。FastAPI 在 [`api/app.py`](../../backend/src/api/app.py) 中将该目录挂载为静态资源。

上传接口只负责校验和保存，不执行 VLM。客户端随后将返回的 URL 放进聊天请求：

```json
{
  "message": "这个适合敏感肌吗？",
  "image_url": "/uploads/upload_xxx.jpg"
}
```

### 9.3 VLM 图片预分析何时调用

当 `ChatStreamRequest.image_url` 非空时：

```text
_run_chat_turn()
→ _prepare_pipeline_body()
→ run_with_heartbeat(run_multimodal())
→ analyze_image()
```

等待模型时会发送 `thinking(stage="analyzing_image")`。本地 `/uploads` 或 `/assets/products` URL 会通过 [`image_url_to_provider_url()`](../../backend/src/services/image_upload.py) 转换为 Base64 data URI，供外部模型读取。

[`analyze_image()`](../../backend/src/services/llm_client.py) 要求模型输出：

```json
{
  "category_hint": "美妆护肤",
  "description": "一瓶包装上标注舒缓保湿的护肤品",
  "visible_traits": ["敏感肌", "保湿", "舒缓"]
}
```

第一次输出不是合法 JSON 时，会额外调用一次 JSON 修复；仍失败时 [`run_multimodal()`](../../backend/src/runtime/stages/multimodal.py) 返回 `None`，保留原消息继续主流程。

代码最终最多保留 5 个 visible traits；虽然当前 Prompt 写的是最多 8 个，但真实行为以代码规范化为准。

### 9.4 图片语义如何进入 Intent 和 Criteria

VLM 结果不会直接成为最终 Criteria。[`message_with_image_context()`](../../backend/src/services/message_rules.py) 会将其转换为文本上下文：

```text
这个适合敏感肌吗？

图片分析：
品类=美妆护肤；
描述=一瓶主打舒缓保湿的护肤品；
可见特征=敏感肌，保湿，舒缓；
检索条件=category=美妆护肤，skin_type=敏感
```

[`image_analysis_to_retrieval_constraints()`](../../backend/src/services/message_rules.py) 还会尝试确定性提取 `category`、`product_type`、`skin_type`、`dietary` 和 `use_scenario`。

处理后的请求再进入正常链路：

```text
Intent
→ 槽位检查
→ Criteria 封闭 DSL
→ 检索
```

这样图片语义不会绕过现有校验和 Criteria 契约。

### 9.5 图片 Embedding 何时调用

请求通过 Intent 和槽位检查并进入 [`handle_recommendation()`](../../backend/src/runtime/handlers.py) 后：

```text
image_url
→ run_image_embedding()
→ image_url_to_provider_url()
→ embed_image()
→ qwen3-vl-embedding
→ 图片查询向量
```

图片 Embedding 会作为后台任务启动，与反馈读取、开场文本等操作重叠。它发生在推荐 Handler 中，因此当前不是与 Pipeline 的 VLM 预分析从最开始并行。

任意异常都会使 [`run_image_embedding()`](../../backend/src/runtime/stages/multimodal.py) 返回 `None`，后续降级为纯文本检索。

### 9.6 商品图片向量如何预先入库

应用自动 seed 时：

```text
initialize_database()
→ seed_image_embeddings_if_needed()
→ 读取商品图片
→ 计算 image_hash
→ embed_image()
→ 校验 1024 维
→ 写入 product_image_embeddings
```

[`seed_image_embeddings_if_needed()`](../../backend/src/services/product_ingest.py) 使用图片内容 SHA-256 判断是否需要重新向量化。表中保存 `product_id`、图片路径、向量、模型、维度、内容 hash 和索引时间。

### 9.7 图文候选如何合并

检索收到 `image_embedding` 后，文本和视觉查询并行：

```text
Criteria → 文本 Embedding → product_chunks 向量召回 ─┐
                                                    ├→ 候选合并
用户图片 → 图片 Embedding → product_image_embeddings ─┘
```

[`list_products_by_image_similarity()`](../../backend/src/repos/documents.py) 的设计目标是使用 pgvector 对商品图片向量按余弦距离排序，最多返回 10 个商品，并应用品类、预算、产品类型、排除品牌和排除商品等 SQL 条件。但当前调用在进入查询前会受到 `VectorSearchFilters` 参数漂移影响，不能把这段设计描述成已经稳定运行。

视觉候选还会经过 Python 硬过滤，然后由 [`_merge_text_and_visual()`](../../backend/src/services/retriever.py) 合并：

```text
保留文本候选原顺序
→ 追加尚未出现的视觉候选
```

所以当前是**候选去重合并**，不是按 `文本分 × 权重 + 图片分 × 权重` 计算的多模态分数融合；同一商品两路都命中时，也不会因为图片命中额外加分。

### 9.8 纯图片命中补充解释证据的设计与真实行为

图片召回只有商品 ID 和距离，没有文本 chunk。设计上，最终视觉商品缺少 evidence 时应执行：

```text
_supplement_visual_evidence()
→ evidence_for_product()
→ 按 product_id 查询文本 chunk
→ 补充商品证据
```

设计意图是：

```text
视觉向量负责找到长得相似的商品
文本 chunk 负责解释为什么推荐
```

但当前 [`_supplement_visual_evidence()`](../../backend/src/services/retriever.py) 调用异步 `evidence_for_product()` 时既没有 `await`，又传入了 `product_id` 字符串而不是 `ProductPayload`。因此这条补证据支路目前不能视为正确实现，只能作为待修复的设计目标。

### 9.9 降级策略与当前缺陷

降级策略：

- VLM 分析失败 → 不追加图片语义，保留原文字消息。
- 图片 Embedding 失败 → 不执行视觉相似召回。
- 图片数据库查询失败 → 记录错误并继续使用文本候选。
- 纯视觉商品缺少 evidence → 设计上应补查文本证据，但当前异步调用和参数类型均有错误。

按当前源码，图片链路还存在以下边界：

1. [`ImageSimilarityHit`](../../backend/src/repos/documents.py) 字段名是 `distance`，但 [`_build_visual_hits()`](../../backend/src/services/retriever.py) 读取不存在的 `similarity`，会触发异常并降级为文本检索。正确实现应将距离转换为相似度分数。
2. `_sql_filters_for_recall()` 的 `VectorSearchFilters` 参数漂移同时影响文本和图片查询。
3. `_supplement_visual_evidence()` 未 `await evidence_for_product()`，且参数类型错误，纯视觉候选的证据补查不正确。
4. 上传格式主要依据客户端提供的 MIME 类型，没有使用图片解码库完整验证并重新编码文件。
5. VLM Prompt 声明最多 8 个特征，代码和修复 Prompt 实际最多保留 5 个。
6. 本地上传目录没有定期清理、用户配额、内容 hash 去重、病毒扫描和多实例共享机制。
7. 当前图文只是候选合并，不是真正的图文分数融合。

答辩表达：

> 多模态链路分为图片理解和视觉相似召回两条路径。图片上传后，VLM 先提取品类、描述和可见特征，并将这些线索转成文本上下文，复用现有 Intent、Criteria 和 RAG 流程；进入推荐 Handler 后，再通过视觉 Embedding 生成图片向量，与预先入库的商品图片向量执行 pgvector 相似查询。设计上，文本和图片候选去重合并后，纯图片命中的商品还要补充文本证据；但当前视觉结果字段、过滤 DTO 和异步证据补查都存在接口错误，所以真实运行会优先降级到文本链路。即使修复这些问题，当前实现仍只是候选合并，不是图文分数融合。

自测问题：

1. VLM 图片理解和 VL Embedding 分别解决什么问题？
2. 为什么上传接口不直接执行 VLM？
3. 图片分析结果为什么要先变成文本，再进入 Intent 和 Criteria？
4. 商品图片向量何时建立，如何避免重复计算？
5. 当前图文召回为什么不能称为分数融合？
6. 纯图片命中的商品为什么还要补查文本 evidence？
7. 图片链路有哪些降级路径和当前缺陷？

---

## 10. 其他业务分支：策略、对比、购物车与结算

这一章补齐主推荐链之外的业务分支。它们都由 Intent 路由进入，但并不都调用 RAG。

### 10.1 Handler 注册表是所有分支的汇合点

源码：[`INTENT_HANDLERS`](../../backend/src/runtime/handlers.py)。

```text
_dispatch_intent_handler(ctx, body, intent)
→ INTENT_HANDLERS[intent.intent]
→ 调用对应 Handler
→ Handler 继续 yield SSEEvent
→ Pipeline 只转发事件，不理解分支内部业务
```

| Intent | Handler | 是否进入推荐检索 |
|---|---|---|
| `recommend`、`clarify`、`feedback` | `handle_recommendation()` | 是 |
| `continue` | `handle_continue()` | 视反馈与牌组状态决定 |
| `compare` | `compare_handlers.handle_compare()` | 不重新执行普通推荐检索，读取目标商品构建对比 |
| `add_to_cart`、`remove_from_cart`、`update_cart_quantity`、`view_cart` | `cart_handlers` 对应方法 | 否 |
| `checkout_preview`、`checkout_confirm`、`checkout_cancel` | `cart_handlers` 对应方法 | 否，不执行真实支付 |
| `chitchat` | `handle_chitchat()` | 否 |

### 10.2 场景化 Shopping Strategy 在推荐 Handler 内部调用

源码位置：

- [`is_likely_shopping_strategy_request()`](../../backend/src/services/shopping_strategy.py)：Pipeline 槽位阶段识别“旅行、送礼、宿舍”等场景型需求。
- [`_try_build_shopping_strategy_plan()`](../../backend/src/runtime/handlers.py)：正式 Criteria 产生后尝试构建策略。
- [`build_shopping_strategy_plan()`](../../backend/src/services/shopping_strategy.py)：生成结构化方向、障碍、组合 Criteria 和兜底文案。

调用链：

```text
Intent 为 recommend/clarify
→ Pipeline 判断是否为场景化选购
→ 场景请求可跳过普通商品类型槽位追问
→ handle_recommendation()
→ run_criteria()
→ _try_build_shopping_strategy_plan()
→ build_shopping_strategy_plan()
    ├─ 构造场景策略
    ├─ 必要时生成多个 combo_criteria
    └─ LLM 文案失败时使用模板兜底
→ _run_combo_retrieval() 或普通检索
→ ShoppingStrategyPayload 随 criteria_card 返回
```

它不是独立 Agent，也不是 LLM 自主选择工具；仍然是推荐 Handler 中的一条确定性分支。

### 10.3 商品对比分支

核心入口：[`handle_compare()`](../../backend/src/runtime/compare_handlers.py)。

调用时机：Intent 被解析为 `compare`，或者客户端直接传入 `compare_product_ids`。

```text
Pipeline 合并客户端 ID、LLM ID、上一牌组序数指代
→ handle_compare()
→ _resolve_compare_product_ids()
    ├─ 优先使用 Intent/客户端明确 ID
    ├─ 解析“第一个和第三个”
    └─ 只有对比词但无序数时默认上一牌组前两个
→ services.compare.build_comparison()
→ 发送 compare_card
→ 流式生成对比说明和结论
→ done
```

不足两个有效商品时不会强行比较，而是返回 Clarification。当前最多处理 4 个对比目标，以控制卡片复杂度和模型上下文。

### 10.4 购物车 CRUD 与轻量结算闭环

核心入口：[`cart_handlers.py`](../../backend/src/runtime/cart_handlers.py)。

```text
Intent 路由
→ 解析 product_id / 上一牌组指代 / 数量
→ services.cart
→ repos.cart_items 持久化
→ 记录 Audit Event
→ 返回 cart_action SSE
→ done
```

| Handler | 调用条件 | 行为 |
|---|---|---|
| `handle_add_to_cart()` | 有明确商品 ID 或可解析上一商品 | 增加数量并返回最新购物车 |
| `handle_remove_from_cart()` | 购物车非空且目标明确 | 删除商品 |
| `handle_update_cart_quantity()` | 目标与数量明确 | 更新数量 |
| `handle_view_cart()` | 查看购物车 | 返回购物车摘要 |
| `handle_checkout_preview()` | 查看结算信息 | 只展示当前购物车 |
| `handle_checkout_confirm()` | 确认购买意向 | 记录确认，不调用支付 |
| `handle_checkout_cancel()` | 取消购买 | 记录取消，不清空或退款 |

项目明确不实现真实订单、库存扣减、支付和退款。因此“checkout”是比赛要求中的购买意向闭环，不是电商交易系统。

### 10.5 Chitchat 与 Clarification

- [`handle_chitchat()`](../../backend/src/runtime/handlers.py)：用于问候和能力说明，不访问商品检索。
- [`handle_clarification()`](../../backend/src/runtime/handlers.py)：缺必要槽位时输出追问。
- `clarify` Intent 如果已经具备执行条件，会映射回 `handle_recommendation()`；真正缺槽位的请求通常在 Pipeline 分发前就被拦截。

## 11. 后端工程化：缓存、容错、观测、测试与安全

### 本章调用关系卡

| 维度 | 说明 |
|---|---|
| 覆盖范围 | 围绕主请求链提供缓存、Provider 容错、请求关联、审计、Trace、测试与安全边界 |
| 调用时机 | 缓存包围检索；LLM Gateway 包围模型调用；Middleware 包围 HTTP；Trace/Audit 在业务完成点写入 |
| 核心目标 | 降低重复成本、控制失败、能够解释一轮请求、避免架构和契约回归 |
| 重要边界 | 当前多项能力是单进程/比赛型实现，不等于完整生产基础设施 |

### 11.1 检索缓存何时调用

检索缓存位于混合检索主入口 [`retrieve_with_evidence()`](../../backend/src/services/retriever.py) 的最外层，用于复用已经完成精排和证据绑定的检索结果。

一次无图片检索的调用顺序为：

```text
retrieve_with_evidence(criteria, top_n, feedback)
→ get_retrieval_cache()
→ cache.get(criteria, feedback)
    ├─ 命中：直接返回 RetrievalOutput
    └─ 未命中：继续执行完整检索
        → 查询文本 Embedding
        → pgvector Chunk 召回
        → 渐进式放宽
        → BM25 / RRF
        → Rerank
        → 商品聚合与证据绑定
        → cache.set(criteria, feedback, result)
        → 返回 RetrievalOutput
```

因此缓存命中后，能够跳过 Embedding Provider、数据库向量查询、BM25/RRF、Rerank Provider 和证据重新组装。它缓存的不是某一个中间分数，而是完整的 [`RetrievalOutput`](../../backend/src/services/retriever.py)：

```text
RetrievalOutput
├─ products
├─ evidence_by_product
└─ trace_details
```

图片检索的调用分支不同：

```python
if image_embedding is None:
    cached = cache.get(criteria, feedback)
```

只有 `image_embedding is None` 时才读写缓存。当前 Key 不包含图片内容或图片向量，如果图片请求复用文本缓存，那么两张不同图片在 Criteria 相同的情况下可能错误地返回同一组商品。因此当前实现选择让所有图片查询绕过缓存。

### 11.2 缓存 Key 为什么必须包含 Criteria 和反馈

[`RetrievalCache._make_key()`](../../backend/src/services/retrieval_cache.py) 当前使用：

```text
CriteriaPayload 的完整 JSON
        +
feedback 字典 JSON
        ↓
SHA-256
        ↓
缓存 Key
```

反馈必须参与 Key。以下两次检索虽然购买标准相同，但业务语义不同：

```text
第一次：200 元以内的洗面奶，没有反馈
第二次：200 元以内的洗面奶，排除商品 p_001
```

如果 Key 只有 Criteria，第二次就可能命中第一次的结果，让用户明确排除的商品再次出现。

缓存值来自查询当时的 Criteria 和反馈，因此缓存主要复用“相同条件下的完整检索结果”，而不是保存某个用户永久不变的推荐列表。

### 11.3 TTL、热门 Key 与淘汰策略

[`RetrievalCache`](../../backend/src/services/retrieval_cache.py) 是一个进程内 TTL 缓存，默认参数为：

```text
最大 Key 数：128
普通 TTL：300 秒，即 5 分钟
热门阈值：同一个 Key 命中 3 次
热门 TTL：600 秒，即 10 分钟
```

每次缓存命中时：

```text
hits += 1
entry.hit_count += 1
更新 last_accessed_at
若 hit_count >= 3
→ TTL 延长到 10 分钟
→ 重新计算过期起点
```

达到容量上限后，[`_evict_lowest_hit()`](../../backend/src/services/retrieval_cache.py) 按以下顺序淘汰：

```text
先批量删除全部过期项
        ↓ 如果没有过期项
删除 hit_count 最低的项
        ↓ 命中次数相同时
删除 last_accessed_at 最早的项
```

所以它不是严格的 LRU，而是：

> 先处理 TTL，再以 LFU 的命中次数为主，以 LRU 的最近访问时间为兜底。

Java 中可以把它理解成一个带 TTL 和简单 LFU/LRU 淘汰逻辑的本地 Map。当前实现没有使用 Redis，也没有使用 Caffeine。

### 11.4 为什么多实例之间不能共享

全局缓存实例定义在 Python 模块中：

```python
_retrieval_cache = RetrievalCache()
```

它只存在于当前 Python 进程的内存中：

```text
FastAPI Worker A
└─ RetrievalCache A

FastAPI Worker B
└─ RetrievalCache B
```

因此多进程或多实例部署时：

- 不同 Worker 不共享缓存内容；
- 每个 Worker 分别统计命中率；
- 新 Worker 启动时是冷缓存；
- 进程重启后缓存全部消失；
- [`GET /observability/cache`](../../backend/src/api/observability.py) 只能看到当前实例的缓存统计。

这种设计适合单实例比赛 Demo：实现简单、访问延迟低、无需额外部署 Redis。但如果横向扩容，缓存命中率会被分散，并且不同实例可能在不同时间持有不同版本的商品结果。

### 11.5 当前缓存的正确性问题

#### 问题一：Key 没有包含 `top_n`

[`retrieve_with_evidence()`](../../backend/src/services/retriever.py) 接收 `top_n`，但 `cache.get()` 和 `cache.set()` 只传入 Criteria 与 feedback：

```text
先请求 top_n=3
→ 缓存只包含 3 件商品的 RetrievalOutput

再用相同 Criteria、feedback 请求 top_n=12
→ 命中相同 Key
→ 仍然只返回先前缓存的 3 件商品
```

这是当前代码真实存在的缓存正确性缺陷。

可选改进：

1. 把 `top_n` 加入缓存 Key；
2. 缓存固定数量的较大候选集，命中后再按 `top_n` 切片；
3. 分离“召回候选缓存”和“最终结果缓存”。

第二种方案还需要同步裁剪 evidence 和 trace，不能只裁剪 products。

#### 问题二：语义相同的 Criteria 可能无法命中

当前 Key 序列化整个 `CriteriaPayload`，其中可能包含 `criteria_id`、summary、chips、field_sources 等不直接影响检索结果的字段。

```text
两份 Criteria 的约束完全相同
但 criteria_id 或展示字段不同
→ JSON 不同
→ Hash 不同
→ 缓存未命中
```

更合理的方案是提取真正参与检索的规范化字段构造 Key。

#### 问题三：反馈列表没有排序归一化

`json.dumps(..., sort_keys=True)` 只排序字典的 Key，不排序列表：

```json
{"avoid_products": ["p1", "p2"]}
```

与：

```json
{"avoid_products": ["p2", "p1"]}
```

业务语义相同，但可能生成不同缓存 Key。构造 Key 前应对无顺序语义的列表执行排序和去重。

#### 问题四：商品和模型更新不会主动失效

当前缓存没有加入：

- 商品数据版本；
- 商品价格更新时间；
- Chunk 或向量版本；
- Embedding 模型版本；
- Rerank 模型版本；
- 检索算法版本。

商品价格或知识数据更新后，旧结果不会立即失效，只会等待 TTL 到期。普通 Key 最长约 5 分钟，已变成热门 Key 的结果最长约 10 分钟。

比赛数据基本静态，这个风险较小；真实电商场景中，价格、库存和上下架信息通常不能只依赖短 TTL 保证一致性。

#### 问题五：无结果查询没有负缓存

当所有检索路径都没有候选时，[`retrieve_with_evidence()`](../../backend/src/services/retriever.py) 会提前返回空结果，不会执行末尾的 `cache.set()`。

因此重复的无结果查询仍会反复调用 Embedding 和数据库检索。可以为无结果设置更短的负缓存 TTL，但必须在商品数据更新时及时失效，避免新商品已经入库后仍持续返回空结果。

#### 问题六：缓存值是可变对象引用

`cache.get()` 直接返回保存的 `RetrievalOutput` 对象。如果后续调用者原地修改其中的 products、evidence 或 trace，修改会留在缓存中并影响下一次请求。

可以通过以下方式改进：

- 约束缓存对象不可变；
- 命中时返回深拷贝；
- 缓存序列化数据，命中后重新反序列化；
- 明确规定调用方不得修改缓存返回值。

### 11.6 缓存监控与生产化方案

[`GET /observability/cache`](../../backend/src/api/observability.py) 会调用 `RetrievalCache.stats()`，返回：

```text
total_keys
hits
misses
hit_rate
hot_keys
```

其中 `hot_keys` 只暴露 SHA-256 的前 16 位，避免直接输出完整查询内容。

当前指标能够观察基础命中率，但缺少：

- 过期数量与淘汰数量；
- 命中后节省的平均耗时；
- Key 对应的品类和查询类型；
- 缓存值占用的内存；
- 多实例聚合结果；
- 数据版本和失效原因。

生产化时可以采用：

```text
请求
→ 规范化检索条件
→ 条件 + feedback + top_n + 数据版本 + 模型版本
→ 生成缓存 Key
→ 查询 Redis / 分布式缓存
→ 未命中时执行检索
→ 写入带 TTL 的结果
```

但 Redis 不是天然正确答案。只有在多实例共享、跨进程命中率和统一失效确实有价值时，才值得引入额外网络访问、序列化成本和缓存运维。

答辩表达：

> 检索入口使用进程内 TTL 缓存，以 Criteria 和反馈上下文的哈希作为 Key，缓存完成向量召回、Rerank 和证据绑定后的 RetrievalOutput。普通结果缓存 5 分钟，同一 Key 命中三次后延长到 10 分钟，淘汰时优先删除过期项，再按命中次数和最近访问时间选择。图片检索因为 Key 不包含图片语义，所以主动绕过缓存。该方案适合单实例比赛 Demo，但不支持跨 Worker 共享和数据更新主动失效；当前 Key 还没有包含 top_n，可能把先前的少量结果错误复用于需要更多结果的请求。生产化应规范化 Key，并加入数量、数据版本和模型版本，再根据部署规模考虑 Redis。

自测问题及参考答案：

1. **为什么 feedback 必须进入缓存 Key？**

   因为反馈会改变排除商品和规避特征；如果不加入 Key，用户明确排除的商品可能通过旧缓存再次出现。

2. **为什么图片检索不能直接复用文本检索缓存？**

   当前 Key 不包含图片内容或图片向量，不同图片可能拥有相同 Criteria，却需要不同视觉结果。

3. **先缓存 `top_n=3`，再请求 `top_n=12` 会发生什么？**

   当前会命中相同 Key，错误地只返回缓存中的 3 件商品。

4. **两个 FastAPI Worker 是否共享缓存？**

   不共享。缓存是每个 Python 进程自己的模块级内存对象。

5. **商品价格更新后缓存会立即失效吗？**

   不会。当前没有事件或版本驱动的主动失效，只能等待普通 5 分钟或热门 10 分钟 TTL 到期。

### 11.7 模型超时、快速重试与 Provider Fallback

聊天类 LLM 任务通过 [`llm_profiles.yaml`](../../backend/src/config/llm_profiles.yaml) 配置任务与模型 Profile 的映射，再由 [`llm_gateway.py`](../../backend/src/services/llm_gateway.py) 统一处理 Provider 调用：

```text
Intent / Criteria / Recommendation / Decision 等聊天任务
→ task_profile_names(task)
→ Primary Profile：Qwen
    ├─ 成功：返回结果
    └─ 失败：记录失败与 fallback 事件
→ Fallback Profile：Doubao
    ├─ 成功：返回备用结果
    └─ 失败：抛出 LiveLLMUnavailable
→ Pipeline 根据流是否已经建立转换成 HTTP 或 SSE 错误
```

容错分为两层。

第一层是同一个 Profile 内的快速重试。[`_retry_chat_completion()`](../../backend/src/services/llm_gateway.py) 只重试以下网络瞬态异常：

```text
ConnectError
RemoteProtocolError
ReadError
```

调用策略为：

```text
第一次调用失败
→ 等待 200ms
→ 第二次调用失败
→ 等待 400ms
→ 第三次调用
```

最大尝试次数为 3 次，即首次调用加两次重试。`TimeoutException` 不在快速重试列表中，因为一次调用可能已经等待完整的 20～40 秒；继续使用同一 Provider 重试会严重增加端到端延迟，所以超时交给外层尝试下一个 Profile。

第二层是 Profile Fallback：

```text
Qwen 配置缺失、网络异常、HTTP 异常或输出不可用
→ 记录 LLM Observation 和 fallback 原因
→ 尝试该任务配置的 Doubao Profile
```

需要特别区分：**当前并不是所有 AI 能力都有双 Provider 兜底。**

| 能力 | 当前 Profile 策略 |
|---|---|
| Intent、Criteria、Recommendation、Decision、对比或策略文案 | Qwen Primary + Doubao Fallback |
| VLM 图片理解 | Qwen-VL Primary，无 Fallback |
| 文本 Embedding | Qwen Embedding Primary，无 Fallback |
| Rerank | GTE/Qwen Rerank Primary，无 Fallback |
| 图片 Embedding | Qwen VL Embedding Primary，无 Fallback |

因此答辩时应表述为“聊天生成类任务具备双 Provider Fallback”，不能说“所有模型服务都有双路兜底”。

流式生成还有一个重要边界。[`_stream_chat_task()`](../../backend/src/services/llm_gateway.py) 在尚未向客户端输出任何 delta 时失败，可以尝试下一个 Profile；一旦已经输出部分文本，`yielded=True`，中途失败会直接抛出异常：

```text
尚未发送文本时失败
→ 可以切换备用 Provider

已经发送部分文本后失败
→ 不能重新生成并拼接另一套答案
→ 抛出异常并由 SSE 错误链路结束
```

这是为了避免客户端收到两套模型答案的重复或语义冲突。

### 11.8 Request Log、Audit、Retrieval Trace 与 LLM Observation

项目的可观测数据分成四类，职责不同。

#### Request Log：一次 HTTP 请求发生了什么

[`RequestContextMiddleware`](../../backend/src/middleware/request_context.py) 在请求进入时建立关联上下文：

```text
优先读取 X-Request-ID，否则生成 request_id
→ 优先读取 X-Trace-ID / X-Client-Trace-ID / traceparent
→ 没有 trace_id 时使用 request_id
→ 写入 RequestContext
→ 执行请求
→ 在响应头返回 X-Request-ID 和 X-Trace-ID
→ 记录状态码、耗时、请求/响应摘要和异常类型
```

请求体和响应体最多捕获 256KB，超出部分会标记为截断。静态图片路径和部分管理接口不会进入普通 Request Log，避免无意义开销或观测接口自我记录。

#### Audit Event：业务上发生了什么

[`record_audit_event()`](../../backend/src/services/audit.py) 记录业务动作，例如：

```text
feedback.created
image.uploaded
chat.cancel_requested
cart.checkout_cancelled
chat.context_diagnostic
```

Request Log 关注 HTTP 技术结果，Audit 关注业务动作与副作用。

#### Retrieval Trace：为什么召回和推荐这些商品

推荐 Handler 调用 [`record_retrieval_trace()`](../../backend/src/services/trace_recorder.py)，最终由 [`repos/traces.py`](../../backend/src/repos/traces.py) 持久化：

- 实际使用的过滤条件；
- 向量候选与排序信息；
- 渐进式放宽步骤；
- Rerank 或最终商品 ID；
- 商品与 Evidence Link 的关系。

它是解释 RAG 检索决策的核心记录。

#### LLM Observation：模型调用发生了什么

[`observability_llm.py`](../../backend/src/services/observability_llm.py) 记录：

- task、profile、model、provider；
- success、failed 或 fallback；
- 调用耗时；
- Prompt Hash 和有限长度预览；
- Response 预览；
- 输出校验错误；
- fallback 来源和异常类型。

默认情况下不会保存完整 Prompt/Response，只保存由 `OBSERVABILITY_PREVIEW_CHARS` 控制的预览；开启 `OBSERVABILITY_CAPTURE_FULL_PAYLOAD=1` 后才保存完整内容。因此生产环境开启完整捕获前，需要评估用户隐私和数据保留风险。

四类关联 ID 的语义为：

```text
session_id：一段多轮会话
└─ turn_id：其中一轮业务对话
   └─ request_id：一次具体 HTTP 请求

trace_id：跨客户端、HTTP、业务和外部调用的关联标识
```

[`get_turn_debug_bundle()`](../../backend/src/services/observability.py) 可以按 `turn_id` 聚合 Request Log、Audit Event、LLM 调用、SSE Event、Retrieval Trace、Evidence Link 和上下文诊断，形成一轮对话的调试包。

### 11.9 测试分层与默认测试边界

项目测试可以按以下层次理解。

| 测试层 | 主要验证内容 | 典型源码 |
|---|---|---|
| 纯函数单元测试 | 硬过滤、Criteria 合并、Chunk、RRF、评分、缓存淘汰 | [`backend/tests/`](../../backend/tests/) 中对应模块测试 |
| 契约测试 | SSE Schema、Python Event、Android Parser/ViewModel、Golden Trace | [`test_sse_events.py`](../../backend/tests/test_sse_events.py)、[`test_viewmodel_contract.py`](../../backend/tests/test_viewmodel_contract.py) |
| 数据库与集成测试 | Repo、PostgreSQL、pgvector、会话、反馈、Pipeline、API | [`conftest.py`](../../backend/tests/conftest.py) 及 DB/API 测试 |
| 架构测试 | 分层依赖和 Mock 边界 | [`test_architecture_layers.py`](../../backend/tests/test_architecture_layers.py) |
| Demo Smoke | 推荐、多轮、图片、对比、加购和 SSE 必需事件 | [`demo_smoke.py`](../../backend/src/scripts/demo_smoke.py)、[`test_demo_smoke.py`](../../backend/tests/test_demo_smoke.py) |

测试数据库默认是独立的 PostgreSQL `buypilot_test`。测试夹具在创建或清理数据库前检查数据库名称中必须包含 `test`，避免误操作开发库或生产库。

必须注意当前默认测试行为：

```text
uv run pytest
→ 为了快速反馈，默认跳过一批 DB/API/Pipeline 集成测试

RUN_FULL_TESTS=1 uv run pytest
→ 才运行被隔离的完整集成集合
```

因此“默认 pytest 通过”不能证明所有真实链路都通过。发布或答辩前，应在具备 PostgreSQL + pgvector 的环境中运行完整测试和 Demo Smoke。

架构测试的价值类似 Java 的 ArchUnit：它不是验证某个业务输入，而是防止代码逐渐破坏 `API → Runtime → Service → Repo` 的依赖方向。

### 11.10 当前安全边界与生产化缺口

当前已有的安全措施包括：

- SQLModel/SQLAlchemy 参数化访问数据库，用户消息不直接拼接 SQL；
- Pydantic 校验 HTTP 请求和模型结构化输出；
- Prompt 输出后继续执行封闭 DSL、白名单和确定性后处理；
- 图片上传限制为 JPEG、PNG、WebP，文件大小不超过 5MB；
- 上传文件使用随机 UUID 文件名，避免直接使用用户路径；
- 管理与观测接口通过 [`require_admin_key()`](../../backend/src/api/admin_auth.py) 校验；
- 未配置管理密钥时，管理接口返回 404，避免直接暴露接口存在；
- 错误通过统一 HTTP/SSE 边界返回，避免把完整内部异常直接交给客户端。

但当前防护仍有以下边界：

1. **Prompt Injection 防护有限。**

   [`_sanitize_user_message()`](../../backend/src/services/llm_task_payloads.py) 主要删除 ASCII 控制字符，并不能识别“忽略系统指令”等语义攻击。真正有效的防线是封闭 Criteria DSL、Pydantic 校验、候选白名单和确定性后处理。

2. **图片真实性校验不完整。**

   上传链路主要校验声明的 MIME 类型和部分文件头/尺寸，没有通过成熟图片库完整解码、重新编码，也没有病毒扫描和内容审核。

3. **缺少完整 Rate Limit。**

   当前没有面向用户或 IP 的统一限流，恶意或误操作请求可能消耗大量模型配额和数据库资源。

4. **管理 Token 支持 Query 参数。**

   `?token=...` 可能进入浏览器历史、代理日志或监控系统；生产环境应只接受 Authorization Header，并使用更完善的身份认证和权限控制。

5. **观测数据可能包含用户输入。**

   Request Log 和 LLM Prompt/Response 预览可能携带用户消息，需要增加字段脱敏、访问控制、保留周期和删除机制。

6. **本地上传目录缺少治理。**

   当前没有过期清理、用户配额、内容 Hash 去重、恶意文件扫描、多实例共享和对象存储。

答辩表达：

> 工程容错分为 Profile 内快速重试和聊天类任务的跨 Provider Fallback。网络瞬态错误最多尝试三次，超时不在同一 Provider 反复重试，而是尽快切换备用模型；流式输出一旦已经发送部分文本则不能再切模型。可观测性上，Request Log 记录 HTTP，Audit 记录业务动作，Retrieval Trace 解释检索决策，LLM Observation 记录模型调用，它们通过 request_id、trace_id、session_id 和 turn_id 关联。测试覆盖单元、契约、集成、架构和 Demo Smoke，但默认测试会隔离部分 PostgreSQL 集成用例，发布前必须显式运行完整集合。安全上已有结构化校验、上传限制和管理接口鉴权，但生产化还需要限流、隐私治理、完整图片校验和共享存储。

---

## 12. 答辩与面试串讲

本章用于复习和口头表达。事实细节仍以前面章节与源码为准。

这一节不是新的运行模块，而是把前面源码组织成一条可在答辩中完整讲述的请求链路。

### 12.1 一次推荐请求如何从 HTTP 走到商品卡片

以“推荐适合油皮的洗面奶，200 元以内”为例：

```text
Android POST /chat/stream
→ RequestContextMiddleware 建立 request_id / trace_id
→ api.chat.stream_chat() 管理 HTTP 与 SSE 边界
→ pipeline.chat_stream() 建立 session / turn 生命周期
→ _run_chat_turn() 执行业务编排
→ Intent：规则优先，模糊表达调用 LLM，再做确定性后处理
→ Slot Checker：判断当前动作是否具备必要槽位
→ Handler Router：根据 Intent 选择推荐 Handler
→ Criteria：自然语言转成封闭购买标准 DSL
→ 投机 Criteria 检索与正式 Criteria LLM 并行
→ 正式 Criteria 对投机结果二次过滤，必要时重新检索
→ Criteria 文本生成查询 Embedding
→ SQL 结构化预过滤 + pgvector Chunk 召回
→ Python 硬过滤复核
→ BM25 排序 + RRF 融合
→ Rerank 精排
→ Chunk 结果聚合为商品结果
→ 商品绑定 FAQ / 描述 / 评论证据
→ Recommendation LLM 在候选和证据范围内生成解释
→ Grounding 与确定性规则校验商品、价格和商业声明
→ SSE 发送 criteria_card / product_card / text_delta / done
→ Android 按 turn_id、deck_id、message_id 归并界面状态
→ Conversation 先落库，再并发持久化 Retrieval Trace / Audit / Evidence Link
```

关键源码入口：

- HTTP/SSE：[chat.py](../../backend/src/api/chat.py)；
- 单轮生命周期：[pipeline.py](../../backend/src/runtime/pipeline.py)；
- 推荐编排：[handlers.py](../../backend/src/runtime/handlers.py)；
- Intent：[stages/intent.py](../../backend/src/runtime/stages/intent.py)；
- Criteria：[stages/criteria.py](../../backend/src/runtime/stages/criteria.py)；
- 检索：[retriever.py](../../backend/src/services/retriever.py)；
- 推荐生成：[stages/recommendation.py](../../backend/src/runtime/stages/recommendation.py)；
- SSE 契约：[sse_events.py](../../backend/src/types/sse_events.py) 与 [`contracts/sse-events.schema.json`](../../contracts/sse-events.schema.json)。

### 12.2 答辩时如何解释核心设计

#### 为什么 Intent 和 Criteria 要分开

> Intent 表示用户想执行什么动作，用于选择 Handler；Criteria 表示推荐按什么标准检索，是受约束的查询 DSL。分开后，路由逻辑不会与检索条件耦合，同一个推荐 Handler 也可以处理不同品类和约束。

#### 为什么硬条件不能只交给向量相似度

> 预算、排除品牌和排除商品属于必须满足的业务约束，而向量相似度只能表达语义接近，不能保证价格和排除条件。因此先尽量下推 SQL，再由 Python 复核；Embedding、BM25 和 Rerank 只负责召回与排序。

#### 为什么按 Chunk 召回而不是整件商品

> 一件商品同时包含营销描述、FAQ 和多条评价，不同文本表达不同语义。拆成 Chunk 后，查询可以命中最相关的局部证据；之后再按 product_id 聚合成客户端需要的完整商品。

#### 为什么要做投机检索

> 正式 Criteria 需要等待 LLM。系统先用 Intent 构造便宜的临时 Criteria，让检索 I/O 与 Criteria 推理并行；正式标准完成后再二次过滤，不满足时回退到完整标准重新检索。这是延迟优化，不改变正式 Criteria 的最终约束权。

#### 如何降低 LLM 幻觉

> 商品候选、价格和结构化事实来自数据库，推荐解释必须基于绑定的 Chunk 证据；模型输出后还会进行候选 ID、价格和禁止声明校验。因此这是“数据库事实 + 检索证据 + 结构化约束 + 生成后校验”的分层防护，而不是只依赖 Prompt。

#### 为什么选择确定性 Workflow 而不是完全自主 Agent

> 电商推荐涉及预算、排除条件、价格和加购等确定性业务约束。当前系统允许 LLM 处理语义理解和自然语言生成，但路由、过滤、状态转移和副作用由代码控制，更容易测试、审计和保证稳定性。

### 12.3 项目最值得讲的五个亮点

1. **Intent 与 Criteria 分离**：动作路由与检索标准职责清晰。
2. **硬条件与语义相关性分离**：SQL/Python 保证业务约束，向量和 Rerank 负责相关性。
3. **投机检索并行化**：重叠 Criteria LLM 与检索 I/O。
4. **证据绑定与分层防幻觉**：商品事实、证据、候选和生成文本具有明确边界。
5. **可观测的多轮闭环**：反馈、检索 Trace、Evidence、LLM 调用和 SSE 可以按 turn 关联。

### 12.4 必须能够主动承认的缺陷

答辩时不应声称项目已经达到完整生产级，可以从下面选择最熟悉的三到五项主动说明：

- `_sql_filters_for_recall()` 与 `VectorSearchFilters` 存在字段参数漂移；
- 检索缓存 Key 没有包含 `top_n`；
- BM25 当前不能把向量召回之外的 Chunk 独立补入候选；
- 图文召回只是候选去重合并，不是统一分数融合；
- 流式文本发送后无法对所有内容执行发送前校验；
- `/health` 更接近 liveness，没有完整 readiness；
- 默认 pytest 隔离了大量数据库和 Pipeline 集成测试；
- 缓存、BM25 和上传目录都是进程或本地状态，多实例不共享；
- 图片验证、限流和敏感数据治理仍不完整；
- 数据库升级依赖 `create_all` 和补列逻辑，没有正式迁移工具。

推荐表达结构：

```text
为什么比赛阶段这样选择
→ 当前代码真实行为是什么
→ 边界或缺陷在哪里
→ 生产化准备怎样改
```

例如：

> 比赛阶段为了减少 Redis 等额外依赖，检索结果使用进程内 TTL 缓存，能够降低重复 Embedding、pgvector 和 Rerank 的成本。但它不支持多 Worker 共享，当前 Key 还遗漏 top_n。生产化时我会先规范化查询 Key并加入数据和模型版本，再根据部署规模使用 Redis，并通过商品更新事件主动失效。

### 12.5 一分钟项目陈述

> BuyPilot-AI 是一个基于 RAG 的多品类智能导购后端。它先通过规则与 LLM 识别用户动作，再把自然语言需求转换为封闭的 Criteria DSL。推荐链路使用 SQL 预过滤、pgvector Chunk 召回、Python 硬过滤、BM25/RRF 和 Rerank，最终把 Chunk 聚合成商品并绑定原始证据。LLM 只在候选和证据范围内生成推荐解释，再经过价格、商品 ID 和商业声明校验。系统通过 SSE 流式返回标准卡片、商品卡片和文本，并利用 session、turn、feedback 完成多轮排除、换一组和最终决策。工程上还实现了聊天模型 Provider Fallback、请求与检索 Trace、契约测试和 Demo Smoke。当前主要生产化缺口是多实例共享状态、缓存失效、数据库迁移、限流和更完整的内容安全。

### 12.6 高频面试追问

1. **如果商品量从 100 条增长到百万级，最先要改什么？**

   把 Seed 与在线服务分离；完善数据库迁移和批量异步入库；按品类和结构化字段建立索引；评估 pgvector HNSW 参数、分区和独立向量服务；缓存只保存规范化且可失效的数据；通过离线评测确定召回数量和 Rerank 成本。

2. **为什么还要 Python 硬过滤，SQL 不是已经过滤了吗？**

   SQL 下推减少候选规模，Python 复核用于统一处理尚未下推或字段表达复杂的约束，并防止 Repo 查询参数漂移导致漏过滤。但理想状态是可结构化条件尽量下推，Python 作为防御性复核。

3. **Rerank Provider 挂了怎么办？**

   当前 Rerank 没有备用 Provider，`_rerank_chunk_hits()` 也没有捕获 `RerankUnavailable` 并退回粗排，因此异常会继续抛到本轮推荐的统一错误边界。生产化应明确降级到向量/RRF 粗排结果并记录 fallback，而不是让整个推荐失败。

4. **如何评价 RAG 效果？**

   离线看硬约束通过率、召回率、证据覆盖率、排序指标和 LLM Judge；在线看无结果率、反馈变化率、用户排除率、点击或加购信号、延迟和 Provider 失败率。不能只用“回答看起来不错”评价。

5. **为什么不直接让大模型调用所有工具？**

   当前场景的路由和副作用有限且需要稳定保证。确定性 Workflow 更容易控制预算约束、事件顺序、加购副作用和测试覆盖；未来只有当工具种类和开放任务明显增加时，才值得引入更自主的 Agent Planner。

到此，后端源码学习主线完成。后续重点应从“继续增加章节”切换为“根据这套链路进行口头复述、源码定位和模拟追问”。

---

## 13. 源码审计：当前缺陷、边界与改进顺序

本章把分散在各业务章节中的问题集中起来。除非明确要求修复，否则这些内容用于学习和答辩，不代表本次已修改业务代码。

### 13.1 已确认的运行时接口错误

| 问题 | 当前代码真实行为 | 影响 | 源码位置 |
|---|---|---|---|
| SQL 过滤 DTO 与 Trace 字段漂移 | `_sql_filters_for_recall()` 传入不存在的 `product_type`、`avoid_brands`，而 `VectorSearchFilters` 接受 `budget_max`、`product_type_aliases`、`brand_avoid`、`avoid_product_ids`；`_sql_filter_payload()` 也继续读取旧字段名 | 当前正常文本/图片召回会在 SQL 查询前直接 `TypeError`；即使只修构造参数，Trace 序列化仍会继续报错；预算也未下推 | [`retriever.py`](../../backend/src/services/retriever.py)、[`documents.py`](../../backend/src/repos/documents.py) |
| 投机结果后过滤签名不匹配 | `_post_filter_retrieval()` 调用 `filter_products(..., max_products=...)`，但公开函数没有该参数 | 投机检索二次过滤路径可能报错 | [`handlers.py`](../../backend/src/runtime/handlers.py)、[`retriever.py`](../../backend/src/services/retriever.py) |
| 品牌偏好直取调用错误 | `_fetch_brand_preference_products()` 调用 `list_products(filters=...)`，而 Repo 函数不接受 `filters` | `brand_prefer` 增强路径可能报错 | [`retriever.py`](../../backend/src/services/retriever.py)、[`products.py`](../../backend/src/repos/products.py) |
| 图片命中字段错误 | `ImageSimilarityHit` 只有 `distance`，`_build_visual_hits()` 却读取 `similarity` | 视觉召回进入转换阶段后异常并降级为纯文本 | [`retriever.py`](../../backend/src/services/retriever.py)、[`documents.py`](../../backend/src/repos/documents.py) |
| 视觉证据补查异步调用错误 | `_supplement_visual_evidence()` 没有 `await evidence_for_product(...)`，且传入 product_id 字符串而不是 `ProductPayload` | 即使修复视觉命中字段，纯视觉商品的证据补查仍不正确 | [`retriever.py`](../../backend/src/services/retriever.py)、[`documents.py`](../../backend/src/repos/documents.py) |
| 预算放宽 Trace 读取位置不一致 | Retriever 把 `relaxation_steps` 放在 trace 顶层，`_budget_was_relaxed()` 却从 `filters_applied` 内读取 | 推荐文案可能不知道预算已经放宽 | [`retriever.py`](../../backend/src/services/retriever.py)、[`handlers.py`](../../backend/src/runtime/handlers.py) |

### 13.2 算法实现与设计目标之间的差距

| 方面 | 当前实现 | 边界或风险 |
|---|---|---|
| BM25 + RRF | BM25 对全量 Chunk 独立排名，RRF 会生成并集排名，但之后只从原向量 `chunk_hits` 取对象 | BM25 独有 Chunk 被丢弃，当前效果更接近“辅助重排向量候选”，不是完整双路召回 |
| 图文融合 | `_merge_text_and_visual()` 保留文本顺序并追加未出现的视觉候选 | 不是统一分数融合；同一商品两路命中不会获得组合加分 |
| Rerank 降级 | Rerank 没有备用 Profile，调用链未捕获 `RerankUnavailable` 回退粗排 | Provider 失败可能终止整轮推荐 |
| 流式 Grounding | 价格在正文发送后校验并可能追加纠正 | 无法撤回已经发送的错误文本；其他功效声明也未全部覆盖 |
| 风险证据 | 正向召回排除风险 Chunk，最终商品再按 ID 补查风险 | 设计方向合理，但商品卡 Evidence 仍不是商品全部知识 |
| 渐进式放宽 | 预算依次 ×1.3、×1.5、移除上限 | 最后一步可能偏离用户预算，需要明确标记或设置最大放宽范围 |

### 13.3 状态、缓存与部署边界

- 检索缓存 Key 没有包含 `top_n`，不同返回数量可能错误复用。
- 缓存 Key 序列化完整 Criteria，`criteria_id` 和无序列表可能降低命中率。
- 缓存、BM25 索引和取消注册表都是进程内状态，多 Worker 不共享。
- 商品、模型或 Prompt 更新没有主动缓存失效版本。
- `/health` 只接近 liveness，没有检查数据库、数据 Seed 和 Provider readiness。
- 数据库升级依赖 `create_all` 和补列逻辑，没有 Alembic/Flyway 式迁移历史。
- 应用关闭没有显式调用已经存在的 `dispose_async_engine()`。

### 13.4 数据规模与安全边界

- 当前官方数据只有约 100 件商品，不能据此证明百万级召回率、延迟和成本。
- 图片主要依据 MIME 和有限文件头信息验证，没有完整解码重编码、病毒扫描和内容审核。
- 缺少统一 Rate Limit、用户配额和上传清理。
- 管理 Token 支持 Query 参数，生产环境存在日志和浏览器历史泄露风险。
- Request Log 与 LLM 预览可能包含用户输入，需要脱敏、访问控制和保留周期。
- 默认 pytest 会隔离大量 PostgreSQL/API/Pipeline 集成测试，完整验证必须显式设置 `RUN_FULL_TESTS=1`。

### 13.5 建议修复优先级

```text
P0：修复会直接导致运行时异常的 DTO/函数签名/async 调用问题
→ 补充对应单元和集成测试

P1：为 Rerank、视觉链路、缓存 Key 和 Trace 结构建立明确降级契约
→ 加入 readiness 与完整测试门禁

P2：实现真正双路召回/图文分数融合、分布式缓存、迁移系统和安全治理
→ 再通过离线评测与线上指标决定参数
```

## 附录 A：核心源码索引

| 层 | 文件或类 | 被谁调用 | 主要职责 |
|---|---|---|---|
| API | [`api/app.py`](../../backend/src/api/app.py) | ASGI Server | 应用装配、lifespan、Router、静态资源和 health |
| API | [`api/chat.py::stream_chat`](../../backend/src/api/chat.py) | Android `/chat/stream` | HTTP 校验、SSE 序列化和响应头 |
| Runtime | [`pipeline.py::chat_stream`](../../backend/src/runtime/pipeline.py) | `stream_chat()` | 一轮 turn 生命周期、取消、异常和清理 |
| Runtime | [`pipeline.py::_run_chat_turn`](../../backend/src/runtime/pipeline.py) | `chat_stream()` | 预处理、Intent、槽位和 Handler 分发 |
| Runtime | [`handlers.py::INTENT_HANDLERS`](../../backend/src/runtime/handlers.py) | `_dispatch_intent_handler()` | Intent 到业务 Handler 的注册表 |
| Runtime | [`handlers.py::handle_recommendation`](../../backend/src/runtime/handlers.py) | recommend/clarify/feedback | Criteria、投机检索、策略、推荐事件编排 |
| Runtime | [`streaming.py::StreamContext`](../../backend/src/runtime/streaming.py) | Pipeline/Handler | seq、event_id、thinking、done、取消检查 |
| Service | [`llm_client.py`](../../backend/src/services/llm_client.py) | Runtime Stage/Handler | 面向任务的 LLM 门面 |
| Service | [`llm_gateway.py`](../../backend/src/services/llm_gateway.py) | `llm_client.py` | Provider Profile、重试、Fallback 和流式传输 |
| Service | [`retriever.py::retrieve_with_evidence`](../../backend/src/services/retriever.py) | Recommendation Stage | 混合检索、放宽、Rerank、聚合和证据绑定 |
| Service | [`chunking.py::build_product_chunks`](../../backend/src/services/chunking.py) | Product Ingest | Typed Semantic Chunking |
| Service | [`conversation_state.py`](../../backend/src/services/conversation_state.py) | Criteria/Pipeline/Handler | 多轮 Criteria、候选和摘要 |
| Service | [`feedback.py`](../../backend/src/services/feedback.py) | API/Handler/Criteria | 反馈写入与上下文聚合 |
| Repo | [`documents.py`](../../backend/src/repos/documents.py) | Retriever/Evidence | pgvector、Chunk 和 Evidence 查询 |
| Repo | [`models.py`](../../backend/src/repos/models.py) | 所有 Repo | SQLModel 表结构 |
| Types | [`schemas.py`](../../backend/src/types/schemas.py) | API/Pipeline | HTTP DTO 与 IntentResult |
| Types | [`sse_events.py`](../../backend/src/types/sse_events.py) | Runtime/API/Client Contract | Criteria、Product、Evidence 和 SSE Event |
| Contract | [`sse-events.schema.json`](../../contracts/sse-events.schema.json) | Python/Android 契约测试 | SSE JSON Schema 真相源 |

## 附录 B：复习检查单

复习时应能够不看文档回答：

1. `stream_chat()`、`chat_stream()`、`_run_chat_turn()` 分别负责什么？
2. Intent、Slot、Criteria 为什么必须分层？
3. 投机检索怎样与正式 Criteria 并行，失败后怎样回退？
4. SQL 过滤、Python 硬过滤、Embedding、BM25、RRF、Rerank 分别解决什么问题？
5. Chunk 如何聚合成商品，Evidence 如何绑定？
6. 为什么商品卡先发、推荐正文后发？
7. 流建立后为什么只能发送 SSE `error/done`？
8. feedback、continue、换一组和最终决策有什么不同？
9. 图片理解和图片向量召回为何是两条支路？
10. 哪些模型有 Provider Fallback，哪些没有？
11. Request Log、Audit、Retrieval Trace、LLM Observation 各自记录什么？
12. 当前最优先修复的运行时缺陷是什么？
