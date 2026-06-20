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
    │                    代码: backend/src/runtime/pipeline.py
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
③ 混合检索         ──→  结构化预过滤 + 向量召回 + BM25辅助排序 + RRF + Rerank
    │                    代码: backend/src/services/retriever.py
    │
    ▼
④ 推荐文案生成     ──→  为每个商品生成推荐理由 + 风险提示
    │                    代码: backend/src/services/llm_client.py
    │
    ▼
⑤ 流式 SSE 推送    ──→  按分支发送 thinking / product_card / criteria_card / final_decision / done 等事件
    │                    代码: backend/src/api/chat.py (event_generator / stream_chat)
    │                          backend/src/runtime/pipeline.py (chat_stream)
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
| 商品 JSON 读取 | [backend/src/repos/products.py](../../backend/src/repos/products.py) | `list_products()` |
| 知识包构建（提取肤质/成分/场景等） | [backend/src/services/chunking.py](../../backend/src/services/chunking.py) | `build_product_knowledge_package()` |
| 语义分块（profile/marketing/FAQ/review） | [backend/src/services/chunking.py](../../backend/src/services/chunking.py) | `build_product_chunks()` |
| 文本向量化（调 text-embedding-v3 API） | [backend/src/services/embedding.py](../../backend/src/services/embedding.py) | `embed_text()` / `embed_texts()` |
| pgvector 向量查询（相似度搜索） | [backend/src/repos/documents.py](../../backend/src/repos/documents.py) | `list_vector_chunks_by_similarity()` |
| 数据入库脚本（向量写入的实际入口） | [backend/src/scripts/reindex_embeddings.py](../../backend/src/scripts/reindex_embeddings.py) | 重建 chunks + embedding，分批写入 pgvector |

**用到的关键库/技术：**
- **pgvector**：PostgreSQL 的向量扩展，可以在 SQL 里做 `cosine_distance` 相似度计算
- **text-embedding-v3**（百炼平台）：把文本转为 1024 维向量的 API 服务
- **SQLModel**：Python ORM，Pydantic + SQLAlchemy 的结合体

**分块策略的设计思想（[chunking.py](../../backend/src/services/chunking.py)）：**

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

#### 2.2.1 在完整 Pipeline 中的位置

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

#### 2.2.2 统一输出：IntentResult

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

#### 2.2.3 规则优先，但规则无法确定时不猜

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

#### 2.2.4 主要确定性规则

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

#### 2.2.5 规则和 LLM 如何协作处理加购

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

#### 2.2.6 LLM 兜底路径

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

#### 2.2.7 两层确定性后处理

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

#### 2.2.8 跨轮上下文合并

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

#### 2.2.9 槽位检查与 Handler 分发

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

#### 2.2.10 Intent 与 Criteria 的职责区别

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

#### 2.2.11 源码索引

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

#### 2.2.12 自测问题

1. 为什么明确操作优先使用规则，而不是全部交给 LLM？
2. 规则返回 `None` 表示什么？
3. “把第一个加入购物车”和“把理肤泉加入购物车”的路径有什么区别？
4. 为什么 LLM 返回 `IntentResult` 后还要做确定性后处理？
5. 当前轮列表约束合并和跨轮历史继承有什么区别？
6. 为什么得到 `recommend` 后还需要 Criteria 阶段？
7. Handler 分发前为什么还需要槽位检查？

---

### 2.3 购买标准 Criteria —— 从自然语言到封闭 DSL

#### 2.3.1 Criteria 是什么，什么时候调用

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

#### 2.3.2 为什么叫“封闭 DSL”

[`Constraints`](../../backend/src/types/sse_events.py) 用 Pydantic 显式声明允许出现的字段，例如预算、品牌排除、产品类型、肤质、成分、存储容量、运动类型和饮食偏好，而不是接收任意的 `dict[str, Any]`。

“封闭”的含义是：

- LLM 不能随意发明检索字段，未知字段会在 [`_sanitize_constraints()`](../../backend/src/services/llm_task_payloads.py) 中被丢弃。
- 每个字段都有固定类型，字符串预算等常见错误会先尝试转换，再交给 Pydantic 校验。
- 后续检索、排序、推荐理由和前端展示都围绕同一个结构消费数据，避免同一语义出现多套表示。

需要特别注意：**封闭 DSL 不等于所有字段都会成为硬过滤条件。** 当前 [`_FILTER_CHECKS`](../../backend/src/services/retriever.py) 直接硬过滤的是已反馈商品、品类、预算、排除品牌、排除产地、产品类型和规避特质等；`skin_type`、`use_scenario`、`storage`、`season` 等字段还可能用于查询文本、排序或推荐理由，而不是全部直接执行 `WHERE` 式过滤。

#### 2.3.3 CriteriaPayload 的六个字段

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

#### 2.3.4 两条生成路径：投机 Criteria 与正式 Criteria

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

#### 2.3.5 正式 Criteria 的执行流程

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

#### 2.3.6 `criteria_patch`：确定性的局部修改

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

#### 2.3.7 LLM 路径：生成、解析与历史合并

没有 patch 时，[`generate_criteria()`](../../backend/src/services/llm_client.py) 会通过任务化 LLM 接口调用模型：

1. [`criteria_messages()`](../../backend/src/services/llm_task_payloads.py) 把当前消息、Intent、反馈、上一轮 Criteria、会话摘要和请求历史组织成 Prompt 输入。
2. Prompt 模板位于 [`backend/prompts/criteria_generation.md`](../../backend/prompts/criteria_generation.md)，要求模型输出 JSON。
3. [`criteria_from_live_payload()`](../../backend/src/services/llm_task_payloads.py) 负责解析和防御性校验，而不是让 Runtime 直接消费模型原始字符串。

历史合并规则必须区分两种情况：

- **同品类：** 以上一轮 `constraints` 为基础，再用 LLM 本轮返回字段覆盖。这里是字典覆盖语义，因此 LLM 返回的列表字段会整体替换旧列表。
- **切换品类：** 不继承旧约束，从空 `Constraints` 开始，避免把“敏感肌”“无糖”等条件泄漏到新品类。

因此，不能笼统地说“所有列表约束都会追加”：只有 `criteria_patch` 路径会追加并去重；LLM 正式结果在同品类合并时是本轮值覆盖历史值。

#### 2.3.8 确定性后处理：为什么不能完全相信 LLM

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

#### 2.3.9 当前实现需要知道的两个细节

以下是基于当前源码的实现边界，答辩时应区分“设计目标”和“现状”：

1. `criteria_from_intent()` 会把投机 Criteria 中的 Intent 字段统一标成 `user`。但 Intent 内的部分值可能来自规则补全或推断，因此这个标记更准确地说是“本轮 Intent 已确定”，不一定全部都是用户逐字表达。
2. `run_criteria()` 最后的 `product_type` 强制覆盖会更新 `constraints` 和 `chips`，但当前没有同步重建 `summary` 与 `field_sources`。主检索约束仍会使用新值，不过展示摘要和来源元数据存在短暂不一致的可能。这是当前实现可继续完善的点，不应描述成已经完全同步。

#### 2.3.10 本阶段面试表达

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

### 2.4 混合检索 —— 整个 RAG 链路的核心

**为什么需要这一步？**
纯向量检索有致命缺陷：它只能做"语义相似"，不能做"精确过滤"。如果你要"200 元以内的洗面奶"，向量检索可能返回一个 300 元的——因为"300 元的高端洗面奶"和"200 元以内的洗面奶"在语义上确实很相关。

**混合检索 = 结构化预过滤 + 向量召回 + BM25 排名信号 + RRF 融合 + Rerank 精排**

完整链路在 [backend/src/services/retriever.py](../../backend/src/services/retriever.py) 的 `retrieve_with_evidence()`：

```
步骤① 缓存检查
    ↓
步骤②+③ 向量召回（含 SQL 预过滤）→ pgvector <=> 余弦距离 + WHERE 过滤，最多召回 200 个 chunk
            + 图片向量召回          → asyncio.create_task 并行，文本和图片两路同时跑
步骤④ BM25 + RRF 融合             → 用 BM25 排名信号重排向量候选
步骤⑤ 渐进式放宽                  → 结果为 0 时逐步放松预算约束
步骤⑥ 品牌偏好直取                → brand_prefer 时显式拉取
步骤⑦ Rerank 精排                 → qwen3-rerank 对候选重排序
步骤⑧ 证据绑定                    → 每个商品挂上命中的 chunk 作为推荐理由来源
```

> **注意**：SQL 预过滤作为 WHERE 条件嵌入 pgvector 查询，主要缩小品类、预算、商品类型、排除品牌和排除商品范围。召回结果随后还会经过 `_passes_hard_filters()` 的 Python 业务规则复核。真正并行的是文本向量召回和图片向量召回。
>
> **当前源码审计提示（2026-06-20）**：`retriever._sql_filters_for_recall()` 构造 `VectorSearchFilters` 时使用了 `product_type`、`avoid_brands` 等参数名，但 Repository 模型实际字段是 `budget_max`、`product_type_aliases`、`brand_avoid`。这会在运行到该构造逻辑时产生参数不匹配，需要在正式演示前修复并补测试。上文描述的是该模块的设计目标。

**每个子步骤的设计理由：**

**步骤② 硬过滤** — 能表达为结构化字段的条件应尽量在 SQL 查询中提前缩小范围，召回后再通过 7 个纯函数复核预算、品类、品牌、产地、商品类型和规避特质。刚性条件不能仅依赖向量语义“猜测”。

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

代码：[retriever.py](../../backend/src/services/retriever.py)

**步骤④ BM25 + RRF 融合** — 向量检索擅长语义，但对品牌、型号等精确词不够敏感。当前实现计算向量排名和 BM25 排名的 RRF 分数，然后用该顺序重排已经进入向量候选集的 chunk；BM25-only 的 chunk 当前不会被追加到候选集，因此它更准确地说是“关键词辅助重排”，而不是独立扩大召回集合。

> **术语卡：BM25** — 一种经典的关键词检索打分算法，核心思想是：如果一个词在少数文档中出现频率高，它就是这个文档的"关键词"，应得高分。比简单的关键词匹配智能，但不如向量检索懂语义。可以理解为"语法级别的搜索引擎"。
>
> **术语卡：RRF (Reciprocal Rank Fusion)** — 一种"排名融合"算法，输入是多路检索各自排好的结果列表，输出是一份综合排名。公式很简单：一个 item 的得分 = 它在各路排名中的倒数之和。不需要知道各路打分函数的内部细节，只需要知道"排第几"。

代码：[retriever.py](../../backend/src/services/retriever.py)

**步骤⑦ Rerank 精排（qwen3-rerank）** — 粗排后最多取 `top_n × 8` 个 chunk 候选送入 Rerank；默认最终返回 5 个商品时，上限是 40 个候选。Rerank 同时读取 query 和 document，比 embedding 分别编码后计算余弦距离更精细。

**步骤⑧ 证据绑定** — 对每个最终商品，按 `why_buy → faq → risk → compare` 的优先级，每种证据类型最多选取一个排名最靠前的 chunk，供推荐理由、风险提示和前端证据页使用。

**渐进式放宽（[retriever.py](../../backend/src/services/retriever.py)）：**
如果严格条件筛选后结果为 0，系统不会直接说"没找到"，而是：
1. 预算上限 × 1.3 → 再试
2. 预算上限 × 1.5 → 再试
3. 预算无上限 → 再试
4. 移除反馈中的 avoid_traits → 再试

> 这些倍数的选择（1.3/1.5）来自 [config/tuning.py](../../backend/src/config/tuning.py) 的 `BUDGET_RELAXATION_STEPS`，属于实验调参。关于放宽策略的局限性讨论见第 7 节局限 3。

---

### 2.5 推荐文案生成 —— 把检索结果变成自然语言

**为什么需要这一步？**
检索返回的是商品数据 + chunk 片段，需要 LLM 把它们组织成用户能读懂的推荐文案，并绑定证据来源。

对应文件：[backend/src/services/llm_client.py](../../backend/src/services/llm_client.py) 的 `generate_recommendation()` 和 `stream_recommendation()`

**推荐理由原子：** 系统先用代码从商品字段、Criteria 和证据中构建最多 4 个结构化理由，例如“油性肤质匹配”“89 元符合 200 元预算”“已避开酒精”。商品卡可以直接使用这些确定性理由；LLM 主要负责组织较长的推荐解释，从而减少模型自由发挥的空间。

代码：[backend/src/services/recommendation_reasons.py](../../backend/src/services/recommendation_reasons.py) 的 `build_reason_atoms()`

---

### 2.6 流式 SSE 推送 —— 让用户感觉"秒回"

**为什么需要这一步？**
用户发送消息后，如果必须等待整条链路结束才看到内容，体验会很差。流式推送允许系统在不同阶段逐步发送 thinking、商品卡片和文本增量；实际首个事件延迟取决于命中的路由和外部服务响应，不能脱离运行数据承诺固定数值。

#### 2.6.1 从 HTTP 请求到 SSE 事件流

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

#### 2.6.2 为什么使用异步生成器

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

#### 2.6.3 SSE 文本格式

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

#### 2.6.4 session_id、turn_id 与 seq

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

#### 2.6.5 两种错误边界

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

#### 2.6.6 取消机制

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

#### 2.6.7 为什么 stream_chat() 与 chat_stream() 不合并

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

#### 2.6.8 SSE 事件协议

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

##### 心跳循环到底做了什么

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

##### 前端收到心跳后会怎样

`ctx.thinking()` 创建的事件包含当前 `stage`、展示文案、递增的 `seq`，以及同一轮固定的 `node_id="thinking_{turn_id}"`。事件经过 Pipeline、FastAPI SSE 接口传到 Android 后：

1. `SseEventParser` 将它解析为 `ThinkingPayload`；
2. `ChatReducer` 将界面标记为 `Streaming`；
3. 创建或更新当前轮的 `ThinkingNode`；
4. UI 显示“正在理解需求”“正在检索商品”等加载动画；
5. 后续文本、卡片或完成事件到达时，临时 Thinking 节点会被移除或替换。

由于同一轮使用相同的 Thinking 节点标识，连续心跳主要是更新或维持同一个加载节点，不会每 0.8 秒在聊天列表里新增一条消息。

##### 如果前端没有收到心跳会怎样

当前 Android 客户端没有实现“超过 0.8 秒没收到心跳就判定后端离线”的严格心跳超时协议，而且 OkHttp 的 SSE 客户端配置了 `readTimeout(0)`。因此，漏掉一两个心跳不会立即触发报错或重新请求。

如果后端长时间不发送任何事件，客户端可能仍保持 SSE 连接并停留在原来的加载状态，用户难以区分“仍在计算”和“后端卡住”。只有连接真正失败时才会进入 `onFailure()`。所以本项目的 `thinking` 心跳更准确地说是：

> **耗时阶段的进度占位事件 + 周期性取消检查点 + 一定程度的连接保活。**

它不是客户端和服务端互相探测存活状态的严格双向心跳协议。

##### 心跳与取消为什么放在一起

每次等待任务前后，`ensure_active()` 都会检查：

- 当前进程内的 `cancel_token`；
- 数据库中是否存在跨进程取消请求。

发现取消后会设置取消令牌、抛出 `StreamCancelled`，并取消尚未完成的阶段任务。因此，心跳循环不仅负责“告诉前端还在运行”，也为协作式取消提供了周期性检查点。

可以把整个机制记成一句话：

> **启动耗时任务 → 最多等待 0.8 秒 → 检查取消 → 完成则返回业务结果，未完成则发送 `thinking` → 继续循环。**

##### 心跳相关的 4 个重要代码位置

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

---

## 3. 核心数据流详解

### 一个完整的用户请求（"推荐适合油皮的洗面奶，200以内"）的流转过程

**第 0 步：HTTP 请求到达**

```
Android App → POST /chat/stream
    Body: {"message": "推荐适合油皮的洗面奶，200以内", "session_id": "sess_abc"}
```

入口：[backend/src/api/chat.py](../../backend/src/api/chat.py) → 调 `pipeline.chat_stream()`

**第 1 步：Pipeline 初始化**

```
chat_stream() [pipeline.py]
  → 生成 turn_id = "turn_a1b2c3d4"
  → 创建 StreamContext（携带 seq/cancel_token/stages）
  → register_turn（创建进程内取消令牌）
  → register_chat_turn（记录数据库 active turn）
  → 调 _run_chat_turn()
```

**第 2 步：意图识别**

```
_resolve_intent() [pipeline.py]
  规则1: maybe_intercept_budget_patch → 不命中（没有上一轮标准可修改）
  规则2: converge → 不命中
  规则3: maybe_checkout_intent → 不命中
  规则4: maybe_cart_intent("推荐适合油皮的洗面奶...")
         → 命中！→ IntentResult(intent="recommend")
  后处理: 从消息补充 category="美妆护肤"、product_type="洗面奶" 等初步约束
  ★ 一次 LLM 调用都没有！纯代码规则完成意图识别。
```

**第 3 步：槽位检查**

```
_missing_slots() [pipeline.py]
  → check_required_slots("推荐适合油皮的洗面奶，200以内", intent)
  → category="美妆护肤" 已存在 → 不需要澄清 → 返回 []
```

**第 4 步：意图分发 → handle_recommendation()**

```
_dispatch_intent_handler() [pipeline.py]
  → INTENT_HANDLERS["recommend"] = handle_recommendation
  → handler 内部（[handlers.py](../../backend/src/runtime/handlers.py)）：
      a) 后台启动 DB 预读（feedback + previous_product_ids）
      b) 用 intent 快速构造 spec_criteria（简版标准，纯计算）
      c) 流式输出 intro text（打字机动画，用户开始看到文字）
      d) ★ 投机检索（spec_criteria）后台启动 + criteria(LLM) 并发运行
      e) await 投机检索结果 + 用完整 criteria 后过滤
      f) 对每个商品发 product_card 事件
      g) 若只有 1 个候选：发 criteria_card → final_decision → done(completed)
      h) 若有 2 个及以上候选：流式推荐解释 → criteria_card → done(awaiting_product_feedback)
         多候选首轮不会直接发 final_decision，要等待用户反馈或显式 converge
```

**第 5 步：SSE 序列化与发送**

```
chat.py 的 event_generator():
  每个 SSEEventBase → format_sse(event) [sse_events.py]
  → "event: product_card\ndata: {json}\n\n"  (标准 SSE 格式)
  → 通过 HTTP 响应流推给 Android 客户端
```

---

## 4. 设计亮点解读

### 亮点 1：规则优先 + LLM 兜底的意图识别

**为什么这样设计？** 明确的结算、购物车、换一组和序数对比可以直接由代码判断，从而减少不必要的模型调用、降低延迟与成本，并避免概率性模型改变明确业务操作。具体命中率和延迟应以评测或观测数据为准，不能仅凭经验数字断言。

**如果不这样做？** 每次都调用 LLM 会增加一次远程请求和 Token 消耗，而且模型可能把明确的“加入购物车”误分类为推荐。

**代码位置：** [pipeline.py](../../backend/src/runtime/pipeline.py)

### 亮点 2：混合检索（Hard + Soft 分离）

**为什么这样设计？** 硬约束（价格、品类、排除品牌）必须精确满足，不能靠向量语义“大概匹配”。系统将部分结构化条件下推 SQL，并在召回后再次执行确定性硬过滤。

**如果不这样做？** 只用向量检索 → 用户要 200 元以内的，系统可能推荐 300 元的（因为"高端洗面奶"和用户查询语义相关），造成幻觉式推荐。

**代码位置：** [retriever.py](../../backend/src/services/retriever.py)（7 个硬过滤函数），[retriever.py](../../backend/src/services/retriever.py)（完整检索链路）

**被放弃的替代方案**：FAISS（纯向量库，无原生 SQL 过滤能力，需要两套查询合并）、Elasticsearch（功能强大但运维复杂度远高于 pgvector 的"一个 PostgreSQL 搞定一切"）、Qdrant/Milvus（专用向量数据库，但本项目数据量仅 100 条，专用库属于过度设计）。详见 [design-decisions.md](../../design-decisions.md)。

### 亮点 3：Pipeline 并行化 + 心跳机制

**为什么这样设计？** 在等待 Criteria LLM 时提前执行投机检索，可以重叠模型推理和检索 I/O；阶段级心跳则让客户端知道系统仍在工作。优化收益需要通过 trace 中的阶段耗时实际验证。

**代码位置：** [streaming.py](../../backend/src/runtime/streaming.py)（心跳实现），[handlers.py](../../backend/src/runtime/handlers.py)（`handle_recommendation()` 中的投机检索+标准生成并行编排）。handler 内部还会根据选购策略（shopping_strategy）自动走场景化推荐分支——见亮点 7。

### 亮点 4：CriteriaPayload 封闭 DSL

**为什么这样设计？** 所有约束维度在代码中显式枚举（`skin_type: str | None`, `budget_max: float | None`, ...），检索逻辑不需要猜"这个字段是什么意思"。新增约束维度 = 先改 Schema，再改检索代码。

**如果不这样做？** 用 `dict[str, Any]` 存约束 → 检索代码需要 `if "skin_type" in constraints` 这种动态检查 → 拼写错误只有在运行时才暴露 → 熵增失控。

**代码位置：** [sse_events.py](../../backend/src/types/sse_events.py) 的 `Constraints` 类

### 亮点 5：渐进式放宽（Relaxation）

**为什么这样设计？** 严格筛选 0 结果时，不要直接说"没找到"，而是逐步放松约束（预算 ×1.3 → ×1.5 → 无上限），给用户一个"接近你预算的选择"。

**代码位置：** [retriever.py](../../backend/src/services/retriever.py)

### 亮点 6：证据绑定 + 防幻觉

**为什么这样设计？** 商品卡片、价格和候选集合来自数据库，检索结果还会绑定原始 chunk 作为证据。商业声明问题会在 Pipeline 前置拦截；非流式推荐结果会拒绝候选集外商品和禁止商业词；决策文本会替换不安全声明；流式推荐中的价格则在发送后由 GroundingGuard 校验并追加纠正提示。需要注意：当前流式文本不是所有 token 都能在发送前完成禁止词校验，因此这是分层防护，不是绝对消除幻觉。

### 亮点 7：场景化选购策略（Shopping Strategy）

**为什么这样设计？** 用户不总是带着明确的品类需求来购物。"送妈妈的礼物""旅行必备""最近想运动"——这些场景没有指定品类，但暗示了购物方向。普通导购系统会直接问"你想要什么品类？"，而这种反问打断了用户的自然表达。

**本项目如何做？** [handlers.py](../../backend/src/runtime/handlers.py) 中，在标准推荐流程启动前，会先尝试 `_try_build_shopping_strategy_plan()`。如果检测到送礼、旅行或兴趣探索场景，[shopping_strategy.py](../../backend/src/services/shopping_strategy.py) 会：

1. 当前实际路由的场景类型是 gift / travel / interest
2. 识别决策障碍（害怕选错？预算敏感？选择过载？）
3. 生成"选购方向"——一个跨品类的检索策略和推荐解释框架
4. 如果涉及多品类（如"旅行必备"→同时需要防晒+运动鞋+充电宝），走 combo 多品类并行检索

**如果不这样做？** 用户说"送妈妈礼物"→意图识别提取不出品类→触发澄清反问"你想要哪个品类？"→用户："不知道啊，有什么推荐吗？"→死循环。场景策略把这个循环打断，用跨品类视野给出建议。

**代码位置：** [shopping_strategy.py](../../backend/src/services/shopping_strategy.py)（场景分类+策略生成），[handlers.py](../../backend/src/runtime/handlers.py)（handler 中的策略分支）

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

> **项目中的对应代码**：[backend/src/services/embedding.py](../../backend/src/services/embedding.py) — 同样是调 OpenAI 兼容 API，但增加了 profile 机制（从 `llm_profiles.yaml` 读取模型名/base_url/api_key，支持多 provider 切换）和批量处理（一次最多 10 条，这是百炼 `text-embedding-v3` 的限制）。

**参数说明：**
- `model`: 选择哪个 Embedding 模型。不同模型输出的向量维度和语义空间不同
- `dimensions`: 输出向量维度。必须选择模型支持的值；更高维通常意味着更高存储与计算成本，但不应简单理解为维度越高效果一定越好
- `input`: 可以是单个字符串，也可以是字符串列表（批量处理更高效）

### 5.4 向量数据库：用 Chroma 做最简单的入门

```python
# 文件: chroma_demo.py （需新建，项目中无此文件）
import os
from dotenv import load_dotenv
import chromadb
from embedding_demo import get_embedding

load_dotenv()

# 1. 创建 Chroma 客户端（数据存在本地磁盘）
client = chromadb.PersistentClient(path="./chroma_db")

# 2. 创建一个 collection（类似数据库的"表"）
# get_or_create_collection 允许脚本重复运行。
# 向量由上一节的 get_embedding() 显式生成，避免依赖 Chroma 对不同 Provider 参数的封装差异。
collection = client.get_or_create_collection(name="products")

# 3. 把商品数据写入向量库
collection.upsert(
    embeddings=[
        get_embedding("清爽控油洗面奶，含水杨酸，适合油性肌肤，89元"),
        get_embedding("温和保湿面霜，含玻尿酸，适合干性肌肤，159元"),
        get_embedding("防晒霜SPF50，清爽不油腻，适合户外运动，129元"),
    ],
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
    query_embeddings=[get_embedding("油性皮肤适合用什么洗面奶？")],
    n_results=3,
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

# 注：真实项目会尽量把结构化条件下推到 SQL，并在召回后再执行 Python 硬过滤复核。
# 上面的循环只是为了单独演示“硬条件不能依赖向量相似度”。
```

> **项目中的对应代码**：[retriever.py](../../backend/src/services/retriever.py) 的 7 个 `_passes_*` 函数负责召回后的确定性复核；Repository 层会把部分结构化条件下推到 SQL；`_relaxation_attempts()` 负责渐进式放宽。

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

> **项目中的真实 Rerank**：[backend/src/services/reranker.py](../../backend/src/services/reranker.py) — `rerank_texts()` 调用 qwen3-rerank API，输入是 query + 商品描述文本列表，返回按相关性排序的索引（index）。与上面的模拟不同，真实 Rerank 用的是 Cross-Encoder 模型，它会同时读 query 和 document，比 Embedding 的"分别编码后算余弦"更精准。

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

> **项目中的对应代码**：[backend/src/services/llm_client.py](../../backend/src/services/llm_client.py) — 与本教程最大的不同：项目不直接调 `openai.OpenAI()`，而是通过 `llm_gateway._call_chat_task()` → `llm_profiles.task_profile_names()` → `llm_task_payloads.intent_messages()` 三层抽象来切换 provider 和构造 prompt。这样做的原因是支持"百炼主力 + Doubao 兜底"的双轨模型策略。作为初学者，直接调 SDK 足够了；等你需要切换模型版本时，自然会理解 profile 模式的价值。

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

# 模拟 LLM 的 Think → Act 循环（用于学习通用 ReAct；当前项目并不让 LLM 自主选择每一步）
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

> 上面的代码是用于理解通用 ReAct 思想的独立教学示例。当前项目的 `handle_recommendation()` 是确定性工作流编排：代码预先规定 Criteria、检索、商品卡和推荐解释等步骤，LLM 不会在运行时自主选择下一项工具。因此它更接近 workflow-style Agent，而不是 ReAct Agent。

**第三步：理解记忆模块**

- **请求历史**：Android 传入的最近消息（messages/history）
- **会话状态**：`conversations` 保存当前 session 的 Criteria、候选商品和摘要
- **会话反馈**：`feedbacks` 保存喜欢、排除商品和规避特质
- 当前实现属于 session 级上下文与反馈闭环，不是跨账号、长期存在的用户画像或购买历史系统
- 本项目代码：[backend/src/services/conversation_state.py](../../backend/src/services/conversation_state.py) — 获取/保存会话标准

**第四步：将 RAG 作为 Agent 的一个 Tool**

```python
# 设计思路
agent_tools = {
    "search_products": search_products,       # 商品检索（RAG）
    "compare_products": compare_products,     # 商品对比
    "add_to_cart": add_to_cart,              # 加购
}
# 这是可选的未来架构：让 LLM 根据语义选择工具。当前项目使用规则/LLM分类后由 Handler 注册表确定调用路径。
```

### 推荐学习资源

1. **Function Calling / Tool Calling 概念**：可用于学习“模型选择工具”的通用模式。当前项目虽然通过 OpenAI-compatible HTTP 接口调用百炼和 Doubao，但没有使用模型原生 tool calling；它采用确定性路由和 Handler 注册表。
2. **LangChain 的 RAG 概念文档**：[python.langchain.com/docs/concepts/rag](https://python.langchain.com/docs/concepts/rag) — 本项目 **不使用** LangChain，但它的概念文档对理解 RAG 的抽象层级（Document Loaders → Text Splitters → Embeddings → Vector Stores → Retrievers）有帮助，尤其适合面试前建立全局认知。
3. **动手练习**：从 [retriever.py](../../backend/src/services/retriever.py) 抽出 `filter_products()` + `_rank_hits()` + `_evidence_by_product()` 三个纯函数（不需要数据库连接，不需要 LLM），配合第 5 节的 Chroma 教程，把纯向量搜索升级为"向量召回 + 硬过滤 + 证据绑定"版本。然后加 5.4+ 节的模拟 Rerank，你就有了一个完整的"迷你混合检索"管道。
4. **阅读本项目的核心文件顺序**：
   - [sse_events.py](../../backend/src/types/sse_events.py) → 理解数据模型
   - [pipeline.py](../../backend/src/runtime/pipeline.py) → 理解编排逻辑
   - [retriever.py](../../backend/src/services/retriever.py) → 理解检索链路
   - [llm_client.py](../../backend/src/services/llm_client.py) → 理解 LLM 调用方式

---

## 7. 已知局限性与设计取舍

没有完美的架构，只有针对特定场景的取舍。以下列出本项目当前设计的几个已知 trade-off：

### 局限 1：100 条商品数据下的过拟合风险

**问题**：当前数据只有 100 条商品，无法充分证明混合检索在十万级商品上的召回、延迟和索引维护成本。HNSW 本身就是近似最近邻索引，扩大数据规模后需要重新评估召回率、索引参数、过滤选择性和更新成本；Rerank API 的候选规模与费用也会成为约束。

**当前应对**：比赛场景使用给定的 100 条数据。扩展时可以评估 Elasticsearch/OpenSearch 等倒排系统，并继续严格限制进入 Rerank 的候选集。当前实现已经只对粗排候选执行 Rerank，并非对全量商品精排。

### 局限 2：规则优先系统对未见表达式的覆盖盲区

**问题**：意图识别的规则优先策略依赖关键词词表。如果用户使用未覆盖的近义表达，例如“弄进篮子里”，规则不会命中，会转入 LLM 兜底。这样会增加一次远程模型调用；如果词表无限扩张，规则之间也可能产生冲突。

**当前应对**：LLM 兜底是覆盖盲区的安全网。词表范围由实际的 4 条 Demo 路径和评测样本驱动，不盲目扩张。

### 局限 3：渐进式放宽的预算跨度问题

**问题**：预算从原始值 ×1.3 → ×1.5 → 直接无上限，跨度很大。如果用户说"200 以内"，×1.5=300 元的商品还在合理范围，但无上限意味着可能推荐 500 元以上的商品——这和用户预期差距很大。

**改进方向**：[config/tuning.py](../../backend/src/config/tuning.py) 中 `BUDGET_RELAXATION_STEPS` 可以调整为更细粒度的步长（如 ×1.2 → ×1.4 → ×1.6 → ×2.0 → 无上限），或增加"放宽后结果打标"机制（告知用户"以下商品略超预算"）。

### 局限 4：投机检索的 spec_criteria 与 LLM 标准不一致

**问题**：[handlers.py](../../backend/src/runtime/handlers.py) 中，投机检索用的是 `criteria_from_intent()` 快速构造的简版标准（如只有 category + budget），而 LLM 后续可能追加 `brand_avoid`、`ingredient_avoid`、`skin_type` 等约束。如果完整标准比简版严格得多，投机检索的结果可能在 post-filter 阶段被全部过滤掉，只能走串行 fallback，相当于投机检索白做了。

**当前应对**：post-filter 只处理有限候选，计算成本较低。如果投机结果全部被过滤，系统会用完整 Criteria 重新检索；这种情况下先前检索会产生额外模型/API/数据库消耗，并增加串行重试时间，所以收益必须通过实际命中率和阶段耗时评估。

### 局限 5：无流式检索——等待全量结果

**问题**：当前检索需要完成 embedding、向量召回、融合和 Rerank 后，才开始逐张发送商品卡片，而不是“召回一个返回一个”的流式检索。等待时间受外部模型和数据库状态影响。

**改进方向**：可以探索分阶段候选协议，例如先返回明确标记为“预览”的结构化候选，再用完整检索结果替换；但必须处理排序变化、重复卡片和用户已操作旧候选等一致性问题，不能承诺未经测量的固定延迟。

### 局限 6：证据绑定粒度粗

**问题**：当前证据绑定按 `why_buy / faq / risk / compare` 分类，每种类型最多保留一个排名最高的 chunk。一个商品可能还有多条有效评价和 FAQ，这些信息不会全部进入商品卡。

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
| API | [backend/src/api/chat.py](../../backend/src/api/chat.py) | HTTP/SSE 边界 |
| Runtime | [backend/src/runtime/pipeline.py](../../backend/src/runtime/pipeline.py) | 单轮对话编排总控 |
| Runtime | [backend/src/runtime/handlers.py](../../backend/src/runtime/handlers.py) | 按意图分发到具体处理函数 |
| Runtime | [backend/src/runtime/streaming.py](../../backend/src/runtime/streaming.py) | 心跳机制 + 取消检查 |
| Service | [backend/src/services/llm_client.py](../../backend/src/services/llm_client.py) | LLM 任务门面 |
| Service | [backend/src/services/retriever.py](../../backend/src/services/retriever.py) | 混合检索核心 |
| Service | [backend/src/services/embedding.py](../../backend/src/services/embedding.py) | 文本/图片向量化 |
| Service | [backend/src/services/reranker.py](../../backend/src/services/reranker.py) | Rerank 精排 |
| Service | [backend/src/services/chunking.py](../../backend/src/services/chunking.py) | 商品文本分块 |
| Types | [backend/src/types/sse_events.py](../../backend/src/types/sse_events.py) | SSE 事件 + Constraints DSL |
| Prompts | [backend/prompts/](../../backend/prompts/) | Prompt 模板（12个） |
| Repo | [backend/src/repos/models.py](../../backend/src/repos/models.py) | 数据库表模型 |
| Config | [backend/src/config/settings.py](../../backend/src/config/settings.py) | 环境变量集中管理 |
