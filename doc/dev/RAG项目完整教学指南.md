# BuyPilot-AI RAG 项目完整教学指南

---

## 1. 项目概览

### 一句话定位

**这是一个"电商智能导购 Agent"：用户用模糊的自然语言描述购物需求 → 系统把需求解析成可量化的购买标准 → 在商品库中精准检索 → 生成带证据支撑的推荐理由，全程流式推送到手机 App。**

> 类比：它不是"会聊天的京东搜索框"，而是"一个懂商品的导购员"——它不只是搜商品，而是帮你理清"我到底该买什么、为什么买这个"。

### 项目的 RAG 完整链路（从输入到输出）

```
用户输入 "推荐适合油皮的洗面奶，200以内"
    │
    ▼
① 意图识别         ──→  判断用户想做什么（推荐/加购/对比/闲聊）
    │                    代码: backend/src/runtime/pipeline.py:682-929
    │                          backend/src/runtime/stages/intent.py
    │                          backend/prompts/intent_analysis.md
    │
    ▼
② 购买标准生成     ──→  将模糊需求转为结构化约束
    │                    "油皮"→skin_type="油性", "200以内"→budget_max=200
    │                    代码: backend/src/runtime/stages/criteria.py
    │                          backend/prompts/criteria_generation.md
    │
    ▼
③ 混合检索         ──→  SQL硬过滤 + 向量召回 + BM25 + RRF融合 + Rerank精排
    │                    代码: backend/src/services/retriever.py:125-296
    │
    ▼
④ 推荐文案生成     ──→  为每个商品生成推荐理由 + 风险提示
    │                    代码: backend/src/services/llm_client.py:184-263
    │
    ▼
⑤ 流式 SSE 推送    ──→  thinking→criteria_card→product_card→final_decision→done
    │                    代码: backend/src/api/chat.py (event_generator / stream_chat)
    │                          backend/src/runtime/pipeline.py:267-415 (chat_stream)
    │
    ▼
Android 端渲染     ──→  商品卡片 + 推荐理由 + 可交互按钮
```

---

## 2. 从整体到细节：每一步是怎么做的

### 2.1 数据准备 —— 商品入库与文本分块

**为什么需要这一步？**
LLM（大语言模型）不认识你数据库里的商品信息。你需要先把商品信息转成"向量"（一串数字），存到向量数据库里，后续检索时才能用"语义相似度"找到相关商品。

**本项目的数据特点：**
- 100 条脱敏电商数据，4 个品类 × 25 条（美妆护肤/数码电子/服饰运动/食品生活）
- 每条数据包含：商品名、品牌、价格、SKU、营销文案、FAQ、用户评价
- 数据源位置：`data/raw/ecommerce_agent_dataset/`

**这一步在代码中的位置：**

| 子步骤 | 文件 | 关键函数 |
|--------|------|---------|
| 商品 JSON 读取 | [backend/src/repos/products.py](backend/src/repos/products.py) | `list_products()` |
| 知识包构建（提取肤质/成分/场景等） | [backend/src/services/chunking.py:85-135](backend/src/services/chunking.py#L85-L135) | `build_product_knowledge_package()` |
| 语义分块（profile/marketing/FAQ/review） | [backend/src/services/chunking.py:138-172](backend/src/services/chunking.py#L138-L172) | `build_product_chunks()` |
| 文本向量化（调 text-embedding-v3 API） | [backend/src/services/embedding.py:31-59](backend/src/services/embedding.py#L31-L59) | `embed_text()` / `embed_texts()` |
| pgvector 向量查询（相似度搜索） | [backend/src/repos/documents.py](backend/src/repos/documents.py) | `list_vector_chunks_by_similarity()` |
| 数据入库脚本（向量写入的实际入口） | [backend/src/scripts/reindex_embeddings.py](backend/src/scripts/reindex_embeddings.py) | 重建 chunks + embedding，分批写入 pgvector |

**用到的关键库/技术：**
- **pgvector**：PostgreSQL 的向量扩展，可以在 SQL 里做 `cosine_distance` 相似度计算
- **text-embedding-v3**（百炼平台）：把文本转为 1024 维向量的 API 服务
- **SQLModel**：Python ORM，Pydantic + SQLAlchemy 的结合体

**分块策略的设计思想（[chunking.py:138-172](backend/src/services/chunking.py#L138-L172)）：**

```python
# 每条商品不是变成一个 chunk，而是按"证据类型"拆成多个 chunk：
#   profile chunk    → "XX品牌洗面奶 | 适合油性肤质 | 含水杨酸"（商品身份信息）
#   marketing chunk  → 营销文案的句子（告诉用户为什么值得买）
#   FAQ chunk        → Q&A 对（解答常见疑问）
#   review chunk     → 用户评价（好评用于 why_buy，差评用于 risk 提示）
#   warning chunk    → 风险汇总（敏感肌慎用、含酒精等）
```

为什么这么分？因为不同类型的 chunk 在推荐时有不同作用——营销文案适合当"推荐理由"的证据，差评适合当"风险提示"的来源。分类型存储让证据绑定更精准。

---

### 2.2 意图识别 —— 用户到底想干什么

**为什么需要这一步？**
用户输入千变万化："推荐洗面奶"、"这个加购物车"、"对比第一个和第二个"、"今天天气怎么样"……系统必须先搞清楚用户的意图类型，才能走对应的处理流程。如果用户只是闲聊，就不应该消耗昂贵的 LLM token 去检索商品。

**本项目最精彩的设计：规则优先，LLM 兜底**

代码在 [backend/src/runtime/pipeline.py:682-929](backend/src/runtime/pipeline.py#L682-L929) 的 `_resolve_intent()` 函数，7 类确定性规则按优先级排列（规则 3/4/5 通过 Python `or` 短路求值合并为一组）：

```
规则1: 预算修改检测         → "再便宜点" → 修改 budget_max
规则2: 前端收敛信号         → converge=true → 直接决策
规则3/4/5 (短路求值组):     → maybe_checkout_intent or maybe_cart_intent or maybe_shopping_intent
        "买单"             → checkout_confirm
        "加入购物车"        → add_to_cart
        "推荐/帮我选"       → recommend
规则6: 换一组              → "换一批" → recommend + 排除已看商品
规则7: 对比意图            → "对比第一个和第二个" → compare
兜底: LLM 意图识别         → 只有以上全部没命中时才调 LLM
```

此外还有一个"规则与 LLM 协作"机制（[pipeline.py:757-775](backend/src/runtime/pipeline.py#L757-L775)）：当规则识别出 `add_to_cart` 但无法解析目标商品名（如"把理肤泉的加进去"——没有序数也没有代词指代），会故意把规则结果清掉让 LLM 提取商品名，LLM 跑完后再强制恢复 `add_to_cart` 意图。这是规则和 LLM "分工合作"的经典案例。

**为什么这样设计？**
- **成本**：80%+ 的用户消息是固定模式，用正则就能识别，不需要 LLM 调用
- **速度**：规则匹配 < 1ms，LLM 调用 ~1.5s
- **准确性**：对于"加入购物车"这种明确意图，规则比 LLM 更可靠（LLM 可能产生幻觉）

**对应文件：**
- 规则检测函数：[backend/src/services/message_rules.py](backend/src/services/message_rules.py) — `maybe_cart_intent()`, `maybe_shopping_intent()` 等
- LLM 兜底：[backend/src/runtime/stages/intent.py](backend/src/runtime/stages/intent.py) → [backend/src/services/llm_client.py:129-152](backend/src/services/llm_client.py#L129-L152) 的 `analyze_intent()`
- Prompt 模板：[backend/prompts/intent_analysis.md](backend/prompts/intent_analysis.md)

**输出数据结构（IntentResult）：**
```python
# 定义在 backend/src/types/schemas.py
class IntentResult(BaseModel):
    intent: str          # "recommend" | "add_to_cart" | "compare" | ...
    confidence: float    # 0.0~1.0
    category: str | None # "美妆护肤" | "数码电子" | ...
    extracted_constraints: dict  # {budget_max: 200, skin_type: "油性", ...}
```

---

### 2.3 购买标准生成 —— 从"模糊需求"到"结构化约束"

**为什么需要这一步？**
这是本项目区别于"普通 RAG 聊天机器人"的核心。用户说的是自然语言（"适合油皮的洗面奶，别太贵"），但检索系统需要的是结构化条件（`category=美妆护肤, skin_type=油性, budget_max=200`）。这一步做"翻译"。

**核心数据结构：Constraints 封闭 DSL**

定义在 [backend/src/types/sse_events.py:90-123](backend/src/types/sse_events.py#L90-L123)：

```python
class Constraints(BaseModel):
    # 通用约束
    budget_min: float | None = None
    budget_max: float | None = None
    use_scenario: str | None = None       # 使用场景
    brand_avoid: list[str] = []           # 排除的品牌
    brand_prefer: list[str] = []          # 偏好的品牌
    product_type: str | None = None       # 产品子类型

    # 美妆护肤专属
    skin_type: str | None = None          # 油性/干性/敏感肌
    ingredient_avoid: list[str] = []      # 避开的成分（酒精/香精）
    ingredient_prefer: list[str] = []     # 偏好成分（玻尿酸/烟酰胺）

    # 数码电子专属
    storage: str | None = None            # 存储容量
    screen_size: str | None = None        # 屏幕尺寸

    # 服饰运动专属 / 食品生活专属 ...
```

所有约束维度显式枚举，禁止 `dict[str, Any]`。这保证了检索逻辑的确定性。

**对应文件：**
- Stage 入口：[backend/src/runtime/stages/criteria.py](backend/src/runtime/stages/criteria.py) 的 `run_criteria()`
- LLM 调用：[backend/src/services/llm_client.py:155-181](backend/src/services/llm_client.py#L155-L181) 的 `generate_criteria()`
- Prompt 模板：[backend/prompts/criteria_generation.md](backend/prompts/criteria_generation.md)
- 后处理净化：[backend/src/services/criteria_sanitizer.py](backend/src/services/criteria_sanitizer.py) — LLM 生成的 criteria 会经过 `sanitize_criteria_product_type()` 做跨品类归一化（如 LLM 把'手机'的 `product_type` 错归到错误品类时自动修正），并清理无效约束字段。

---

### 2.4 混合检索 —— 整个 RAG 链路的核心

**为什么需要这一步？**
纯向量检索有致命缺陷：它只能做"语义相似"，不能做"精确过滤"。如果你要"200 元以内的洗面奶"，向量检索可能返回一个 300 元的——因为"300 元的高端洗面奶"和"200 元以内的洗面奶"在语义上确实很相关。

**混合检索 = SQL 硬过滤 + 向量召回 + BM25 关键词 + RRF 融合 + Rerank 精排**

完整链路在 [backend/src/services/retriever.py:125-296](backend/src/services/retriever.py#L125-L296) 的 `retrieve_with_evidence()`：

```
步骤① 缓存检查
    ↓
步骤②+③ 向量召回（含 SQL 硬过滤）→ pgvector <=> 余弦距离 + WHERE 硬过滤并行嵌入，Top-50 chunk
            + 图片向量召回          → asyncio.create_task 并行，文本和图片两路同时跑
步骤④ BM25 + RRF 融合             → 关键词召回 + 排名融合
步骤⑤ 渐进式放宽                  → 结果为 0 时逐步放松预算约束
步骤⑥ 品牌偏好直取                → brand_prefer 时显式拉取
步骤⑦ Rerank 精排                 → qwen3-rerank 对候选重排序
步骤⑧ 证据绑定                    → 每个商品挂上命中的 chunk 作为推荐理由来源
```

> **注意**：SQL 硬过滤不是独立步骤，而是作为 SQL WHERE 条件嵌入在 pgvector 查询内部（`list_vector_chunks_by_similarity(filters=sql_filters)`），与向量相似度搜索合并为一次数据库请求。真正"并行"的是文本向量召回和图片向量召回（`asyncio.create_task` 两路同时跑）。

**每个子步骤的设计理由：**

**步骤② 硬过滤（SQL WHERE）** — 刚性条件必须精确匹配，不能靠语义"猜"。价格 > 预算 → 直接排除。代码：[retriever.py:513-612](backend/src/services/retriever.py#L513-L612)

```python
# 7 个硬过滤检查函数，串联执行
_FILTER_CHECKS = (
    _passes_feedback_product_filter,   # 用户"不喜欢"的商品
    _passes_category_filter,           # 品类必须匹配
    _passes_budget_filter,             # 价格必须在预算内
    _passes_brand_filter,              # 排除的品牌
    _passes_origin_filter,             # 排除的产地
    _passes_product_type_filter,       # 产品类型（洗面奶/手机/跑鞋）
    _passes_avoid_trait_filter,        # 规避成分（酒精/香精）
)
```

**步骤③ 向量召回（pgvector）** — 用一个 1024 维向量做余弦相似度搜索，找到语义最相关的 chunk。

> **术语卡：余弦相似度** — 衡量两个向量"方向"有多接近的指标，值域 [-1, 1]。两段文本语义越相关 → 它们的向量在空间中指向越一致 → 余弦值越接近 1。和"距离"不同（距离衡量远近），余弦只看方向不看长短。
>
> **术语卡：pgvector** — PostgreSQL 的一款扩展插件，让普通的 SQL 数据库支持向量存储和相似度检索。好处是：一张表里既有结构化字段（价格/品类）可以做 SQL WHERE 过滤，也有向量字段可以做 `<=>` 余弦距离排序——不需要两套数据库。

代码：[retriever.py:310-347](backend/src/services/retriever.py#L310-L347)

**步骤④ BM25 + RRF 融合** — 向量检索擅长语义但可能漏掉精确关键词匹配。

> **术语卡：BM25** — 一种经典的关键词检索打分算法，核心思想是：如果一个词在少数文档中出现频率高，它就是这个文档的"关键词"，应得高分。比简单的关键词匹配智能，但不如向量检索懂语义。可以理解为"语法级别的搜索引擎"。
>
> **术语卡：RRF (Reciprocal Rank Fusion)** — 一种"排名融合"算法，输入是多路检索各自排好的结果列表，输出是一份综合排名。公式很简单：一个 item 的得分 = 它在各路排名中的倒数之和。不需要知道各路打分函数的内部细节，只需要知道"排第几"。

代码：[retriever.py:206-231](backend/src/services/retriever.py#L206-L231)

**步骤⑦ Rerank 精排（qwen3-rerank）** — 粗排后的候选（~15个）送入专门的 Rerank 模型，它对 query-document 相关性打分比 embedding 的余弦相似度更精准。代码：[retriever.py:454-495](backend/src/services/retriever.py#L454-L495) + [reranker.py:48-65](backend/src/services/reranker.py#L48-L65)

**步骤⑧ 证据绑定** — 每个被推荐的商品，从命中的 chunk 中选一条"为什么推荐"的证据（来自营销文案/FAQ/好评），供前端展示。代码：[retriever.py:382-413](backend/src/services/retriever.py#L382-L413)

**渐进式放宽（[retriever.py:420-441](backend/src/services/retriever.py#L420-L441)）：**
如果严格条件筛选后结果为 0，系统不会直接说"没找到"，而是：
1. 预算上限 × 1.3 → 再试
2. 预算上限 × 1.5 → 再试
3. 预算无上限 → 再试
4. 移除反馈中的 avoid_traits → 再试

> 这些倍数的选择（1.3/1.5）来自 [config/tuning.py](backend/src/config/tuning.py) 的 `BUDGET_RELAXATION_STEPS`，属于实验调参。关于放宽策略的局限性讨论见第 7 节局限 3。

---

### 2.5 推荐文案生成 —— 把检索结果变成自然语言

**为什么需要这一步？**
检索返回的是商品数据 + chunk 片段，需要 LLM 把它们组织成用户能读懂的推荐文案，并绑定证据来源。

对应文件：[backend/src/services/llm_client.py:184-263](backend/src/services/llm_client.py#L184-L263) 的 `generate_recommendation()` 和 `stream_recommendation()`

**本项目做了一件很聪明的事：** 先把"推荐理由原子"用代码计算好（肤质匹配 +20 分、预算内 +10 分...），再交给 LLM 组织语言。这样 LLM 只需要做"翻译"，不需要做"判断"——大大降低了 LLM 编造幻觉的可能性。

代码：[backend/src/services/recommendation_reasons.py](backend/src/services/recommendation_reasons.py) 的 `build_reason_atoms()`

---

### 2.6 流式 SSE 推送 —— 让用户感觉"秒回"

**为什么需要这一步？**
用户发送消息后，如果等 15 秒才看到完整回复，体验很差。流式推送让首字延迟降到 < 100ms（立即发 thinking 事件），后续事件逐个推送。

**SSE 事件协议（10 种事件类型）：**

定义在 [backend/src/types/sse_events.py:178-272](backend/src/types/sse_events.py#L178-L272) 和 [contracts/sse-events.schema.json](contracts/sse-events.schema.json)：

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

**流式编排的核心机制（心跳填充）：**

在等 LLM 返回期间（通常 1-4 秒），每 0.8 秒自动发一个 `thinking` 事件更新进度文案，让用户感知系统在工作。代码在 [backend/src/runtime/streaming.py:146-186](backend/src/runtime/streaming.py#L146-L186) 的 `run_with_heartbeat()`：

```python
# 伪代码理解：
# async def 定义的是"协程函数"，它可以在等待 I/O 时让出 CPU 给其他任务
# "异步"的精髓：不是"同时做多件事"，而是"在等一件事的时候不闲着"
async def run_with_heartbeat(ctx, awaitable, stage, message):
    task = start_task(awaitable)   # 启动 LLM 调用（异步，不阻塞当前线程）
    while not task.done():
        await asyncio.sleep(0.8)    # 每 0.8 秒，"await" = 让出控制权
        yield ctx.thinking(stage, message)  # "yield" = 向上游推送一个事件
    yield StageResult(task.result())  # LLM 返回了，输出结果
```

> **术语卡：async/await** — Python 的异步编程关键字。`async def` 定义一个"协程"（可以在等待期间暂停的函数），`await` 表示"在这里等结果，但等待期间 CPU 可以去干别的"。和 `def`（同步函数，调用时一口气跑完、不能中途让位）的区别是：同步函数在等 LLM 回复的 3 秒里什么也做不了；协程可以在这 3 秒里发心跳、检查取消信号、启动别的任务。类比：同步 = 你在电话里等客服转接，不能挂；异步 = 客服说"稍等"，你可以一边等一边回微信。

**并行化执行 —— 投机检索 + 标准生成重叠**

代码在 [backend/src/runtime/handlers.py:342-470](backend/src/runtime/handlers.py#L342-L470) 的 `handle_recommendation()` 函数中。实际并行模式是：

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

---

## 3. 核心数据流详解

### 一个完整的用户请求（"推荐适合油皮的洗面奶，200以内"）的流转过程

**第 0 步：HTTP 请求到达**

```
Android App → POST /chat/stream
    Body: {"message": "推荐适合油皮的洗面奶，200以内", "session_id": "sess_abc"}
```

入口：[backend/src/api/chat.py](backend/src/api/chat.py) → 调 `pipeline.chat_stream()`

**第 1 步：Pipeline 初始化（~0ms）**

```
chat_stream() [pipeline.py:269]
  → 生成 turn_id = "turn_a1b2c3d4"
  → 创建 StreamContext（携带 seq/cancel_token/stages）
  → register_cancel_token（用于用户取消）
  → 调 _run_chat_turn()
```

**第 2 步：意图识别（~0-1.5s）**

```
_resolve_intent() [pipeline.py:682]
  规则1: maybe_intercept_budget_patch → 不命中（没有上一轮标准可修改）
  规则2: converge → 不命中
  规则3: maybe_checkout_intent → 不命中
  规则4: maybe_cart_intent → 不命中
  规则5: maybe_shopping_intent("推荐适合油皮的洗面奶...")
         → 命中！→ IntentResult(intent="recommend", category="美妆护肤")
  ★ 一次 LLM 调用都没有！纯代码规则完成意图识别。
```

**第 3 步：槽位检查（~0ms）**

```
_missing_slots() [pipeline.py:936]
  → check_required_slots("推荐适合油皮的洗面奶，200以内", intent)
  → category="美妆护肤" 已存在 → 不需要澄清 → 返回 []
```

**第 4 步：意图分发 → handle_recommendation()**

```
_dispatch_intent_handler() [pipeline.py:1007]
  → INTENT_HANDLERS["recommend"] = handle_recommendation
  → handler 内部（[handlers.py:342-470](backend/src/runtime/handlers.py#L342-L470)）：
      a) 后台启动 DB 预读（feedback + previous_product_ids）
      b) 用 intent 快速构造 spec_criteria（简版标准，纯计算）
      c) 流式输出 intro text（打字机动画，用户开始看到文字）
      d) ★ 投机检索（spec_criteria）后台启动 + criteria(LLM) 并发运行
      e) await 投机检索结果 + 用完整 criteria 后过滤
      f) 发 criteria_card 事件
      g) 对每个商品：发 product_card 事件
      h) 推荐文案流式输出 → text_delta 事件
      i) 发 final_decision 事件
      j) 发 done 事件
```

**第 5 步：SSE 序列化与发送**

```
chat.py 的 event_generator():
  每个 SSEEventBase → format_sse(event) [sse_events.py:424]
  → "event: product_card\ndata: {json}\n\n"  (标准 SSE 格式)
  → 通过 HTTP 响应流推给 Android 客户端
```

---

## 4. 设计亮点解读

### 亮点 1：规则优先 + LLM 兜底的意图识别

**为什么这样设计？** 降低延迟、降低成本、提高准确率。80% 的用户消息是固定模式，正则匹配 < 1ms，LLM 调用 ~1500ms，差别是 1500 倍。

**如果不这样做？** 每次都调 LLM → 1.5s 延迟 + 每次消耗 token → 又慢又贵，且 LLM 可能把"加入购物车"识别成"recommend"。

**代码位置：** [pipeline.py:682-929](backend/src/runtime/pipeline.py#L682-L929)

### 亮点 2：混合检索（Hard + Soft 分离）

**为什么这样设计？** 硬约束（价格、品类、排除品牌）必须精确满足，不能靠向量语义"大概匹配"。把硬过滤放在 SQL 层，既准确又快。

**如果不这样做？** 只用向量检索 → 用户要 200 元以内的，系统可能推荐 300 元的（因为"高端洗面奶"和用户查询语义相关），造成幻觉式推荐。

**代码位置：** [retriever.py:513-612](backend/src/services/retriever.py#L513-L612)（7 个硬过滤函数），[retriever.py:125-296](backend/src/services/retriever.py#L125-L296)（完整检索链路）

**被放弃的替代方案**：FAISS（纯向量库，无原生 SQL 过滤能力，需要两套查询合并）、Elasticsearch（功能强大但运维复杂度远高于 pgvector 的"一个 PostgreSQL 搞定一切"）、Qdrant/Milvus（专用向量数据库，但本项目数据量仅 100 条，专用库属于过度设计）。详见 [design-decisions.md](design-decisions.md)。

### 亮点 3：Pipeline 并行化 + 心跳机制

**为什么这样设计？** 首字延迟从 1-2s 降到 < 100ms。在等 LLM 响应的空隙里，把能提前做的工作（embedding、初步 SQL 查询）并行完成。心跳事件每 0.8s 更新一次 UI 文案，用户不会觉得"卡住了"。

**代码位置：** [streaming.py:146-186](backend/src/runtime/streaming.py#L146-L186)（心跳实现），[handlers.py:342-470](backend/src/runtime/handlers.py#L342-L470)（`handle_recommendation()` 中的投机检索+标准生成并行编排）。handler 内部还会根据选购策略（shopping_strategy）自动走场景化推荐分支——见亮点 7。

### 亮点 4：CriteriaPayload 封闭 DSL

**为什么这样设计？** 所有约束维度在代码中显式枚举（`skin_type: str | None`, `budget_max: float | None`, ...），检索逻辑不需要猜"这个字段是什么意思"。新增约束维度 = 先改 Schema，再改检索代码。

**如果不这样做？** 用 `dict[str, Any]` 存约束 → 检索代码需要 `if "skin_type" in constraints` 这种动态检查 → 拼写错误只有在运行时才暴露 → 熵增失控。

**代码位置：** [sse_events.py:90-123](backend/src/types/sse_events.py#L90-L123) 的 `Constraints` 类

### 亮点 5：渐进式放宽（Relaxation）

**为什么这样设计？** 严格筛选 0 结果时，不要直接说"没找到"，而是逐步放松约束（预算 ×1.3 → ×1.5 → 无上限），给用户一个"接近你预算的选择"。

**代码位置：** [retriever.py:420-441](backend/src/services/retriever.py#L420-L441)

### 亮点 6：证据绑定 + 防幻觉

**为什么这样设计？** 每个推荐理由必须能从商品数据中找到原文出处（marketing description/FAQ/review chunk）。如果 LLM 在生成推荐文案时引用了不存在的商品名或编造了"库存/优惠"信息，系统会在后处理中检测并替换（[llm_client.py:52-76](backend/src/services/llm_client.py#L52-L76) 的 `_FORBIDDEN_COMMERCIAL_TERMS`）。

### 亮点 7：场景化选购策略（Shopping Strategy）

**为什么这样设计？** 用户不总是带着明确的品类需求来购物。"送妈妈的礼物""旅行必备""最近想运动"——这些场景没有指定品类，但暗示了购物方向。普通导购系统会直接问"你想要什么品类？"，而这种反问打断了用户的自然表达。

**本项目如何做？** [handlers.py:429-454](backend/src/runtime/handlers.py#L429-L454) 中，在标准推荐流程启动前，会先尝试 `_try_build_shopping_strategy_plan()`。如果检测到是场景化请求（送礼/旅行/兴趣探索/风险敏感），[shopping_strategy.py](backend/src/services/shopping_strategy.py) 会：

1. 分类场景类型（gift / travel / interest / usage / risk_sensitive / goal_oriented）
2. 识别决策障碍（害怕选错？预算敏感？选择过载？）
3. 生成"选购方向"——一个跨品类的检索策略和推荐解释框架
4. 如果涉及多品类（如"旅行必备"→同时需要防晒+运动鞋+充电宝），走 combo 多品类并行检索

**如果不这样做？** 用户说"送妈妈礼物"→意图识别提取不出品类→触发澄清反问"你想要哪个品类？"→用户："不知道啊，有什么推荐吗？"→死循环。场景策略把这个循环打断，用跨品类视野给出建议。

**代码位置：** [shopping_strategy.py](backend/src/services/shopping_strategy.py)（场景分类+策略生成），[handlers.py:429-454](backend/src/runtime/handlers.py#L429-L454)（handler 中的策略分支）

---

## 5. 手把手：从零搭建最小 RAG 系统

> **在学习本节之前，先读这一段：本节教程与项目实际技术栈的对比**
>
> | 维度 | 本节教程（入门） | 项目实际（生产） |
> |------|-----------------|-----------------|
> | 向量数据库 | ChromaDB（零配置，数据存本地文件） | PostgreSQL + pgvector（需要 Docker） |
> | Embedding 模型 | text-embedding-3-small（OpenAI 兼容） | text-embedding-v3（百炼平台，1024维） |
> | LLM 调用方式 | `openai.OpenAI()` 直接调用 | `llm_gateway.py` → `llm_client.py` → `llm_task_payloads.py` 三层抽象 + profile 驱动 |
> | 检索方式 | 纯向量相似度搜索 | 混合检索（SQL硬过滤 + 向量 + BM25 + RRF + Rerank） |
>
> **学习建议**：先跟着本节跑通最小 RAG（理解"为什么"），然后对比阅读项目中的 `retriever.py` 和 `llm_client.py`（理解"怎么做生产级"）。两者原理相通，API 不同而已。
>
> 本节所有代码文件 **都是需要新建的练习文件，项目中不存在**，你可以在 `d:\tmp\rag_demo\` 下创建它们。

### 5.0 环境准备

**Python 版本**：需要 Python 3.10+（项目本身用 `uv` 管理依赖，但本教程用 `pip` 简化起步）。

**获取 API Key**（以百炼平台为例）：
1. 访问 [阿里云百炼平台](https://bailian.console.aliyun.com/) 注册/登录
2. 进入"模型广场" → 选择 `qwen-turbo`（或 `text-embedding-v3`）→ 开通服务
3. 在"API-KEY 管理"创建 API Key
4. 创建 `.env` 文件（放在 `d:\tmp\rag_demo\.env`）：

```bash
# .env 文件模板
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# 百炼兼容 OpenAI 格式的 base_url
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

> **重要**：永远不要将 API Key 硬编码写在 .py 文件中（提交到 git 会泄露）。使用 `os.getenv()` 从环境变量或 `.env` 文件读取。

### 5.1 安装依赖

```bash
pip install fastapi uvicorn openai httpx pydantic python-dotenv chromadb
```

> **Windows 用户注意**：`chromadb` 依赖 Microsoft Visual C++ Build Tools。如果 `pip install chromadb` 失败，可以去 [visualstudio.microsoft.com/downloads](https://visualstudio.microsoft.com/downloads/) 下载 "Build Tools for Visual Studio"，安装时勾选"Desktop development with C++"。

### 5.2 准备数据（模拟商品库）

```python
# 文件: seed_data.py
products = [
    {
        "id": "p_001",
        "name": "清爽控油洗面奶",
        "brand": "理肤泉",
        "category": "美妆护肤",
        "price": 89.0,
        "description": "专为油性肌肤设计，含水杨酸成分，有效控油不紧绷。适合日常使用。",
        "faq": [
            {"q": "敏感肌能用吗？", "a": "含微量水杨酸，敏感肌建议先做局部测试。"},
        ],
        "reviews": [
            {"user": "小明", "rating": 5, "text": "控油效果很好，洗完不紧绷，推荐！"},
        ],
    },
    # ... 更多商品
]
```

### 5.3 Embedding：把文本变成向量

```python
# 文件: embedding_demo.py （需新建，项目中无此文件）
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # 从 .env 文件加载环境变量

# 1. 初始化客户端（百炼 / OpenAI 兼容 API）
# ★ 用 os.getenv() 读 API Key，不要硬编码！硬编码提交到 git 会泄露
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
)

# 2. 调用 Embedding API
def get_embedding(text: str) -> list[float]:
    """
    把一段文本转成一个 1024 维的浮点数列表（向量）。
    每一维数字代表文本在某个"语义维度"上的强度。
    两段相似文本 → 两个向量在空间中距离很近（余弦相似度接近 1）。
    """
    try:
        response = client.embeddings.create(
            model="text-embedding-v3",    # 模型名（百炼）
            input=text,                    # 要向量化的文本
            dimensions=1024,               # 输出向量维度
        )
        return response.data[0].embedding  # [0.0123, -0.0456, ...]
    except Exception as e:
        # 常见错误见下文"常见错误与解决"表格
        print(f"Embedding 失败: {e}")
        raise

# 测试
if __name__ == "__main__":
    vec = get_embedding("适合油皮的洗面奶")
    print(f"向量维度: {len(vec)}")  # 1024
    print(f"前 5 维: {vec[:5]}")    # [0.0123, -0.0456, 0.0789, ...]
```

> **项目中的对应代码**：[backend/src/services/embedding.py:31-59](backend/src/services/embedding.py#L31-L59) — 同样是调 OpenAI 兼容 API，但增加了 profile 机制（从 `llm_profiles.yaml` 读取模型名/base_url/api_key，支持多 provider 切换）和批量处理（一次最多 10 条，这是百炼 `text-embedding-v3` 的限制）。

**参数说明：**
- `model`: 选择哪个 Embedding 模型。不同模型输出的向量维度和语义空间不同
- `dimensions`: 输出向量的维度。越高保留的语义信息越多，但存储和计算成本越高
- `input`: 可以是单个字符串，也可以是字符串列表（批量处理更高效）

### 5.4 向量数据库：用 Chroma 做最简单的入门

```python
# 文件: chroma_demo.py （需新建，项目中无此文件）
import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

# 1. 创建 Chroma 客户端（数据存在本地磁盘）
client = chromadb.PersistentClient(path="./chroma_db")

# 2. 创建一个 collection（类似数据库的"表"）
#    embedding_function 指定用什么模型把文本转向量
collection = client.create_collection(
    name="products",
    embedding_function=embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        api_base=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        model_name="text-embedding-v3",
    ),
)

# 3. 把商品数据写入向量库
collection.add(
    documents=[
        "清爽控油洗面奶，含水杨酸，适合油性肌肤，89元",
        "温和保湿面霜，含玻尿酸，适合干性肌肤，159元",
        "防晒霜SPF50，清爽不油腻，适合户外运动，129元",
    ],
    metadatas=[
        {"product_id": "p_001", "price": 89, "category": "洗面奶"},
        {"product_id": "p_002", "price": 159, "category": "面霜"},
        {"product_id": "p_003", "price": 129, "category": "防晒霜"},
    ],
    ids=["p_001", "p_002", "p_003"],  # 唯一 ID
)

# 4. 检索：找到和用户查询最相似的商品
results = collection.query(
    query_texts=["油性皮肤适合用什么洗面奶？"],  # 用户问题
    n_results=2,                                    # 返回 top-2
)

for i, doc in enumerate(results["documents"][0]):
    meta = results["metadatas"][0][i]
    distance = results["distances"][0][i]
    print(f"排名 {i+1}: {doc}")
    print(f"  价格: {meta['price']}元, 距离: {distance:.4f}")

# 5. ★ 硬过滤演示：在向量检索的基础上，用 Python 代码过滤预算
print("\n--- 加上硬过滤：只保留 price <= 130 ---")
budget = 130
for i, doc in enumerate(results["documents"][0]):
    meta = results["metadatas"][0][i]
    if meta["price"] <= budget:
        print(f"✓ {meta['product_id']}: {doc[:30]}... 价格: {meta['price']}元")
    else:
        print(f"✗ {meta['product_id']}: 超出预算（{meta['price']}元 > {budget}元）")
# 输出示例：
# ✓ p_001: 清爽控油洗面奶... 价格: 89元
# ✗ p_002: 超出预算（159元 > 130元）
# ✓ p_003: 防晒霜SPF50... 价格: 129元

# 6. ★ 处理检索结果为 0 的情况（渐进式放宽的思路）
found = [m for m in results["metadatas"][0] if m["price"] <= 50]
if not found:
    print("\n预算 50 元无结果 → 放宽到 100 元:")
    found = [m for m in results["metadatas"][0] if m["price"] <= 100]
    for m in found:
        print(f"  {m['product_id']}: {m['price']}元")

# 注：项目中用 SQL WHERE price <= budget 一步完成过滤，不需要 O(n) 遍历。
# 上面的 Python 循环是教学简化版。
```

> **Chroma 版本兼容性**：如果 `api_base` 参数报错，说明你的 Chroma 版本较新，可能需要改用 `OpenAIEmbeddingFunction(api_key=..., base_url=...)`（参数名有变化）。请查看 [Chroma 官方文档](https://docs.trychroma.com/) 确认最新 API。

> **项目中的对应代码**：[backend/src/services/retriever.py:513-612](backend/src/services/retriever.py#L513-L612) — 7 个 `_passes_*` 函数做硬过滤，在 SQL 层直接 WHERE 过滤（比上面的 Python 演示更高效）；[retriever.py:420-441](backend/src/services/retriever.py#L420-L441) — `_relaxation_attempts()` 做渐进式放宽。

### 5.4+ 进阶：模拟 Rerank（精排重排序）

向量召回的 Top-N 结果按"语义相似度"排序，但不一定按"购买决策相关性"排序。Rerank 的作用是对粗排结果做二次精排——这是本项目混合检索区别于简单 RAG 的关键一步。

下面用 Python 模拟 Rerank 的核心思想（不调真实 API，只用代码演示逻辑）：

```python
# 文件: rerank_demo.py （需新建，项目中无此文件）
# ★ Rerank 的本质：对粗排结果，按更多维度重新打分排序

def simulate_rerank(query: str, candidates: list[dict]) -> list[dict]:
    """
    模拟 Rerank 逻辑：对每个候选计算一个综合分数。
    真实项目中这是调 qwen3-rerank API：输入 query + documents，返回排序后的 index。
    """
    scored = []
    for item in candidates:
        score = 0.0
        # 维度1：关键词命中（query 中的词是否在商品名中出现）
        query_words = set(query.replace(" ", ""))
        name_words = set(item["name"].replace(" ", ""))
        keyword_hit = len(query_words & name_words) / max(len(query_words), 1)
        score += keyword_hit * 3.0  # 关键词匹配权重最高

        # 维度2：品类匹配（精准匹配得分高）
        if item.get("category") == query:
            score += 2.0

        # 维度3：价格越低分越高（模拟"性价比"偏好）
        price = item.get("price", 999)
        score += max(0, (200 - price) / 200) * 1.0

        scored.append({**item, "rerank_score": round(score, 2)})

    # 按综合分降序排列——这就是 Rerank 的核心效果
    return sorted(scored, key=lambda x: x["rerank_score"], reverse=True)


# 测试：模拟从 Chroma 粗排拿到的 3 个候选
candidates = [
    {"name": "清爽控油洗面奶", "category": "洗面奶", "price": 89},
    {"name": "温和保湿面霜", "category": "面霜", "price": 159},
    {"name": "控油洁面泡沫", "category": "洗面奶", "price": 149},
]
reranked = simulate_rerank("洗面奶", candidates)
for i, item in enumerate(reranked):
    print(f"Rerank 排名 {i+1}: {item['name']} (综合分: {item['rerank_score']})")
```

> **项目中的真实 Rerank**：[backend/src/services/reranker.py:48-65](backend/src/services/reranker.py#L48-L65) — `rerank_texts()` 调用 qwen3-rerank API，输入是 query + 商品描述文本列表，返回按相关性排序的索引（index）。与上面的模拟不同，真实 Rerank 用的是 Cross-Encoder 模型，它会同时读 query 和 document，比 Embedding 的"分别编码后算余弦"更精准。

### 5.5 Prompt 构造 + LLM 生成回答

```python
# 文件: rag_query.py （需新建，项目中无此文件）
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
)

def ask_rag(question: str, retrieved_docs: list[str]) -> str:
    """
    把检索到的文档作为"参考资料"注入 Prompt，让 LLM 基于它们回答。
    """

    # 1. 把检索结果拼成一段"参考资料"
    context = "\n\n".join(
        f"【参考资料 {i+1}】\n{doc}"
        for i, doc in enumerate(retrieved_docs)
    )

    # 2. 构造 Prompt（System Prompt + 用户问题 + 参考资料）
    system_prompt = """你是一个电商导购助手。请根据参考资料回答用户问题。
如果参考资料中没有相关信息，请如实说"我不确定"。
不要编造参考资料中没有的商品信息或价格。"""
# 注：实际项目中的 prompt 见 backend/prompts/recommendation.md，约 80 行，
# 包含详细的商品信息格式规范、证据引用要求和禁止编造条款。这里是教学简化版。

    user_message = f"""用户问题：{question}

参考资料：
{context}

请基于以上参考资料回答："""

    # 3. 调 LLM
    try:
        response = client.chat.completions.create(
            model="qwen-turbo",   # 模型名
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,       # 0=确定, 1=创意（导购场景偏低，减少胡说）
            max_tokens=500,        # 限制回答长度
        )
        return response.choices[0].message.content
    except Exception as e:
        # 常见错误见下方表格
        return f"生成回答失败: {e}"


# 完整流程
if __name__ == "__main__":
    question = "适合油皮的洗面奶，200元以内"
    docs = [
        "清爽控油洗面奶，含水杨酸，适合油性肌肤，89元",
        "控油洁面泡沫，含茶树精油，适合油性和混合性肌肤，149元",
    ]
    answer = ask_rag(question, docs)
    print(answer)
```

> **项目中的对应代码**：[backend/src/services/llm_client.py:129-152](backend/src/services/llm_client.py#L129-L152) — 与本教程最大的不同：项目不直接调 `openai.OpenAI()`，而是通过 `llm_gateway._call_chat_task()` → `llm_profiles.task_profile_names()` → `llm_task_payloads.intent_messages()` 三层抽象来切换 provider 和构造 prompt。这样做的原因是支持"百炼主力 + Doubao 兜底"的双轨模型策略。作为初学者，直接调 SDK 足够了；等你需要切换模型版本时，自然会理解 profile 模式的价值。

**参数解释：**

| 参数 | 作用 | 建议 |
|------|------|------|
| `model` | 选择哪个 LLM | qwen-turbo 快且便宜，适合简单任务 |
| `messages` | 对话上下文 | system=角色设定，user=实际问题 |
| `temperature` | 控制随机性（0-2） | 导购场景 0.1-0.3，创意场景 0.7-1.0 |
| `max_tokens` | 输出最大长度 | 防止回答过长，消耗过多 token |
| `stream` | 是否流式输出 | True=逐字推送, False=全部完成后一次返回 |

**常见错误与解决：**

| 错误 | 原因 | 解决 |
|------|------|------|
| `AuthenticationError` | API Key 错误 | 检查 key 是否正确、是否过期 |
| `RateLimitError` | 调用频率超限 | 加 `time.sleep()` 或降低并发 |
| `APITimeoutError` | 网络延迟或服务端过载 | 增加 `timeout` 参数值（如 `timeout=60`），或加重试逻辑 |
| `ConnectionError` | 网络问题 | 检查代理/VPN、检查 base_url |
| `context_length_exceeded` | Prompt 太长 | 减少检索文档数量或截断文档 |

---

## 6. 新手学 Agent 的进阶路径

### 建议学习顺序

**第一步：理解 Function Calling（工具调用）**

这是 Agent 的"手"。让 LLM 不仅能说话，还能调用外部函数。

```python
# 文件: function_calling_demo.py （需新建，完整的可运行代码）
import os, json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
)

# 模拟的商品搜索函数（实际项目里是 retriever.retrieve_with_evidence()）
def search_products(query: str, max_price: float | None = None) -> list[dict]:
    print(f"[执行搜索] query={query}, max_price={max_price}")
    products = [
        {"name": "清爽控油洗面奶", "price": 89, "reason": "含水杨酸，适合油性肌肤"},
        {"name": "控油洁面泡沫", "price": 149, "reason": "含茶树精油，温和控油"},
    ]
    if max_price is not None:
        products = [p for p in products if p["price"] <= max_price]
    return products

# 定义工具
tools = [{
    "type": "function",
    "function": {
        "name": "search_products",
        "description": "在商品数据库中搜索，返回匹配的商品列表",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"},
                "max_price": {"type": "number", "description": "最高预算"},
            },
            "required": ["query"],
        },
    },
}]

# 发起对话
messages = [{"role": "user", "content": "找200以内的洗面奶"}]
response = client.chat.completions.create(
    model="qwen-turbo",
    messages=messages,
    tools=tools,
)

# 处理工具调用
msg = response.choices[0].message
if msg.tool_calls:
    tool_call = msg.tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    result = search_products(**args)  # 实际执行工具

    # 把工具执行结果发回给 LLM，让它基于结果生成自然语言回答
    messages.append(msg)  # LLM 的工具调用请求
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": json.dumps(result, ensure_ascii=False),
    })
    final = client.chat.completions.create(model="qwen-turbo", messages=messages)
    print(final.choices[0].message.content)
```

**练习：** 给上面的 RAG 系统加一个 `search_products` 工具，让 LLM 自主决定什么时候搜商品。

**第二步：理解 ReAct 模式（推理+行动循环）**

ReAct = Reasoning（推理）+ Acting（行动）。每轮：思考 → 行动 → 观察 → 再思考 → ...

```python
# 文件: react_demo.py （需新建，可运行的极简演示）
# 模拟一个 ReAct Agent：用户问"200以内洗面奶哪个好？"

# 模拟工具函数
def search_products(query: str, max_price: float | None = None) -> list[str]:
    """模拟搜索商品库"""
    print(f"  [Action] search_products('{query}', max_price={max_price})")
    if "洗面奶" in query and (max_price is None or max_price >= 89):
        return ["p_001: 控油洗面奶 89元", "p_004: 温和洁面 149元"]
    return []

def compare_products(ids: list[str]) -> str:
    """模拟对比商品"""
    print(f"  [Action] compare_products({ids})")
    return "p_001 性价比更高，但 p_004 适合敏感肌"

# 模拟 LLM 的 Think → Act 循环（实际项目中 LLM 决定每一步做什么）
task = "200以内洗面奶哪个好？"
step = 0
while step < 5:
    step += 1
    if step == 1:
        # Thought: 需要先搜商品
        print(f"[Thought] 第{step}步: 需要搜索匹配的商品")
        result = search_products("洗面奶", max_price=200)
        print(f"  [Observation] 找到 {len(result)} 个商品: {result}")
    elif step == 2:
        # Thought: 有两个可选，对比一下
        print(f"[Thought] 第{step}步: 有两个候选，对比一下")
        comparison = compare_products(["p_001", "p_004"])
        print(f"  [Observation] {comparison}")
    elif step == 3:
        # Thought: 信息够了，做决策
        print(f"[Thought] 第{step}步: 信息足够，推荐 p_001，给出理由")
        print("  [Final Answer] 推荐控油洗面奶(89元)，性价比高，适合日常使用")
        break  # 任务完成
```

运行输出：
```
[Thought] 第1步: 需要搜索匹配的商品
  [Action] search_products('洗面奶', max_price=200)
  [Observation] 找到 2 个商品: ['p_001: 控油洗面奶 89元', 'p_004: 温和洁面 149元']
[Thought] 第2步: 有两个候选，对比一下
  [Action] compare_products(['p_001', 'p_004'])
  [Observation] p_001 性价比更高，但 p_004 适合敏感肌
[Thought] 第3步: 信息足够，推荐 p_001，给出理由
  [Final Answer] 推荐控油洗面奶(89元)，性价比高，适合日常使用
```

> 本项目的 `handle_recommendation()` 本质上就是一个硬编码的 ReAct 循环：它不需要 LLM 决定下一步是什么（意图已经确定了），直接按固定编排执行：检索 → 比对 → 推荐。真正的 ReAct Agent 会更灵活——LLM 自主决定"搜一次不够，换关键词再搜"或"先对比再搜"。理解了上面的模拟代码，你就理解了 Agent 的核心循环。

**第三步：理解记忆模块**

- **短期记忆**：当前对话的上下文（messages 数组）
- **长期记忆**：用户偏好、历史购买记录（本项目用 PostgreSQL conversations/feedbacks 表实现）
- 本项目代码：[backend/src/services/conversation_state.py](backend/src/services/conversation_state.py) — 获取/保存历史标准

**第四步：将 RAG 作为 Agent 的一个 Tool**

```python
# 设计思路
agent_tools = {
    "search_products": search_products,       # 商品检索（RAG）
    "compare_products": compare_products,     # 商品对比
    "add_to_cart": add_to_cart,              # 加购
}
# LLM 根据用户意图自动选择调用哪个工具
```

### 推荐学习资源

1. **OpenAI Function Calling 文档**：[platform.openai.com/docs/guides/function-calling](https://platform.openai.com/docs/guides/function-calling) — 本项目使用的是 OpenAI 兼容 API（百炼/Doubao 都兼容此格式），tool calling 的 JSON Schema 定义和消息格式以此为规范。注意不要去看 Anthropic 的 tool_use 文档，它用的是完全不同的格式。
2. **LangChain 的 RAG 概念文档**：[python.langchain.com/docs/concepts/rag](https://python.langchain.com/docs/concepts/rag) — 本项目 **不使用** LangChain，但它的概念文档对理解 RAG 的抽象层级（Document Loaders → Text Splitters → Embeddings → Vector Stores → Retrievers）有帮助，尤其适合面试前建立全局认知。
3. **动手练习**：从 [retriever.py](backend/src/services/retriever.py) 抽出 `filter_products()` + `_rank_hits()` + `_evidence_by_product()` 三个纯函数（不需要数据库连接，不需要 LLM），配合第 5 节的 Chroma 教程，把纯向量搜索升级为"向量召回 + 硬过滤 + 证据绑定"版本。然后加 5.4+ 节的模拟 Rerank，你就有了一个完整的"迷你混合检索"管道。
4. **阅读本项目的核心文件顺序**：
   - [sse_events.py](backend/src/types/sse_events.py) → 理解数据模型
   - [pipeline.py](backend/src/runtime/pipeline.py) → 理解编排逻辑
   - [retriever.py](backend/src/services/retriever.py) → 理解检索链路
   - [llm_client.py](backend/src/services/llm_client.py) → 理解 LLM 调用方式

---

## 7. 已知局限性与设计取舍

没有完美的架构，只有针对特定场景的取舍。以下列出本项目当前设计的几个已知 trade-off：

### 局限 1：100 条商品数据下的过拟合风险

**问题**：混合检索（硬过滤 + 向量 + BM25 + RRF + Rerank）在 100 条商品上效果很好，因为候选集小、Rerank 可以精细比较。但如果数据量扩大到 10 万条，pgvector 的 HNSW 索引会退化为近似搜索（有精度损失），Rerank 的昂贵 API 调用也会成为瓶颈。

**当前应对**：比赛场景下 100 条是给定的硬约束，所以这不是问题。但如果你要扩展这个架构，需要考虑：1) 引入 Elasticsearch 做大规模倒排索引；2) Rerank 只对粗排 Top-50 做，而不是全量。

### 局限 2：规则优先系统对未见表达式的覆盖盲区

**问题**：意图识别的"规则优先"策略依赖 `message_rules.py` 中的关键词词表。如果用户说了一个不在词表中的近义表达（如"弄进篮子里"而非"加入购物车"），规则不会命中，会走 LLM 兜底——这本身不是 bug（LLM 能识别），但延迟会从 < 1ms 变成 ~1.5s。此外，如果词表加太多规则，维护成本会线性增长，规则之间还可能出现冲突。

**当前应对**：LLM 兜底是覆盖盲区的安全网。词表范围由实际的 4 条 Demo 路径和评测样本驱动，不盲目扩张。

### 局限 3：渐进式放宽的预算跨度问题

**问题**：预算从原始值 ×1.3 → ×1.5 → 直接无上限，跨度很大。如果用户说"200 以内"，×1.5=300 元的商品还在合理范围，但无上限意味着可能推荐 500 元以上的商品——这和用户预期差距很大。

**改进方向**：[config/tuning.py](backend/src/config/tuning.py) 中 `BUDGET_RELAXATION_STEPS` 可以调整为更细粒度的步长（如 ×1.2 → ×1.4 → ×1.6 → ×2.0 → 无上限），或增加"放宽后结果打标"机制（告知用户"以下商品略超预算"）。

### 局限 4：投机检索的 spec_criteria 与 LLM 标准不一致

**问题**：[handlers.py:342-470](backend/src/runtime/handlers.py#L342-L470) 中，投机检索用的是 `criteria_from_intent()` 快速构造的简版标准（如只有 category + budget），而 LLM 后续可能追加 `brand_avoid`、`ingredient_avoid`、`skin_type` 等约束。如果完整标准比简版严格得多，投机检索的结果可能在 post-filter 阶段被全部过滤掉，只能走串行 fallback，相当于投机检索白做了。

**当前应对**：post-filter 是 O(n) 的纯内存操作，代价可忽略。投机检索命中时收益大（延迟 -2s），未命中时损失小（+0ms 额外延迟），期望收益为正。

### 局限 5：无流式检索——等待全量结果

**问题**：当前检索是"等全量结果返回后才开始推荐"，而不是"召回一个返回一个"的流式检索。嵌入、向量搜索、Rerank 都是 batch 操作，不支持 streaming。用户会感觉 thinking 阶段较长（~2-4s），看不到"商品在慢慢浮现"的过程。

**改进方向**：可以先立刻返回硬过滤的前 2 个商品（无需 embedding/Rerank，< 200ms），作为"快速预览"，再在后台继续完整检索。

### 局限 6：证据绑定粒度粗

**问题**：当前证据绑定只从每个商品的 chunk 中选一条 why_buy + 一条 risk（如果存在）。但一个商品可能有 3 条正面评价和 2 条负面 FAQ——每条都是有用的信号，被丢弃是信息损失。

**改进方向**：实现多证据摘要——把同一类型的所有证据 chunk 合并后做一次轻量摘要（"多数用户认为控油效果好，少数提到包装简陋"），而不是只选一条。

---

## 附录：项目架构速查

### 分层架构

```
UI (Android) → API (FastAPI) → Runtime (编排) → Service (业务) → Repo (数据库)
```

依赖只能自上而下流动，禁止反向/横向依赖。

### 核心文件索引

| 层 | 文件 | 一句话职责 |
|----|------|-----------|
| API | [backend/src/api/chat.py](backend/src/api/chat.py) | HTTP/SSE 边界 |
| Runtime | [backend/src/runtime/pipeline.py](backend/src/runtime/pipeline.py) | 单轮对话编排总控 |
| Runtime | [backend/src/runtime/handlers.py](backend/src/runtime/handlers.py) | 按意图分发到具体处理函数 |
| Runtime | [backend/src/runtime/streaming.py](backend/src/runtime/streaming.py) | 心跳机制 + 取消检查 |
| Service | [backend/src/services/llm_client.py](backend/src/services/llm_client.py) | LLM 任务门面 |
| Service | [backend/src/services/retriever.py](backend/src/services/retriever.py) | 混合检索核心 |
| Service | [backend/src/services/embedding.py](backend/src/services/embedding.py) | 文本/图片向量化 |
| Service | [backend/src/services/reranker.py](backend/src/services/reranker.py) | Rerank 精排 |
| Service | [backend/src/services/chunking.py](backend/src/services/chunking.py) | 商品文本分块 |
| Types | [backend/src/types/sse_events.py](backend/src/types/sse_events.py) | SSE 事件 + Constraints DSL |
| Prompts | [backend/prompts/](backend/prompts/) | Prompt 模板（12个） |
| Repo | [backend/src/repos/models.py](backend/src/repos/models.py) | 数据库表模型 |
| Config | [backend/src/config/settings.py](backend/src/config/settings.py) | 环境变量集中管理 |
