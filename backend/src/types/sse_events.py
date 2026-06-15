from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Literal, Union

from pydantic import BaseModel, Field

# ============================================================================
# 常量定义
# ============================================================================

# SSE 协议版本号 —— 前后端契约的版本标识
SCHEMA_VERSION = "2026-05-20"

# 购买标准的每个字段是谁提供的信息：用户明确说的 / 系统推断的 / 历史对话继承的
CriteriaFieldSource = Literal["user", "inferred", "history"]

# done 事件的结束原因：
#   awaiting_criteria_confirmation  → 等待用户确认购买标准
#   awaiting_criteria_adjustment    → 等待用户调整标准
#   awaiting_product_feedback       → 等待用户反馈商品
#   completed                       → 正常完成
#   cancelled                       → 用户取消
#   error                           → 出错
DoneFinishReason = Literal[
    "awaiting_criteria_confirmation",
    "awaiting_criteria_adjustment",
    "awaiting_product_feedback",
    "completed",
    "cancelled",
    "error",
]

# 前端展示模式：每个事件告诉 Android 应该以什么 UI 形态渲染
DisplayMode = Literal[
    "inline_thinking",   # 思考中骨架屏
    "inline_card",       # 内嵌卡片
    "inline_text",       # 内嵌文本
    "summary_card",      # 总结卡片
    "swipe_deck_item",   # 滑动卡片组中的一张
    "none",              # 不渲染
]


# ============================================================================
# 基础设施
# ============================================================================

class SSEEventBase(BaseModel):
    """所有 SSE 事件的基类 —— 每个事件必带的公共字段"""
    schema_version: str = SCHEMA_VERSION   # 协议版本
    event: str                              # 事件类型名（thinking/clarification/...）
    session_id: str                         # 会话 ID（用户一次完整对话）
    turn_id: str                            # 轮次 ID（用户一次发送）
    seq: int                                # 同一 turn 内的递增序号（保证顺序）
    event_id: str                          # 全局唯一事件 ID
    node_id: str                           # 节点 ID（用于前端 diff 渲染）
    deck_id: str | None = None             # 商品牌组 ID（同一批推荐共享）
    display_mode: DisplayMode | None = None # 前端渲染模式
    created_at_ms: int | None = None       # 事件创建时间戳（毫秒）


class EventSeq:
    """同一 turn_id 内的序号递增计数器，用于保证事件顺序"""

    def __init__(self, turn_id: str) -> None:
        self.turn_id = turn_id
        self._seq = 0       # 内部计数器从 1 开始

    def next(self) -> int:
        self._seq += 1
        return self._seq

    def event_id(self) -> str:
        """生成全局唯一的事件 ID：{turn_id}:{序号}"""
        return f"{self.turn_id}:{self._seq}"


def now_ms() -> int:
    """获取当前毫秒级时间戳"""
    return int(time.time() * 1000)


# ============================================================================
# 核心数据结构
# ============================================================================

class Constraints(BaseModel):
    """
    封闭 DSL —— 所有允许的约束维度显式枚举
    这就是"购买标准"的数据本体，每一条都会用于检索硬过滤

    铁律 2 要求：只能有显式字段，禁止 dict[str, Any]
    """

    # ── 通用字段（所有品类共享）──
    budget_min: float | None = None      # 最低预算
    budget_max: float | None = None      # 最高预算
    use_scenario: str | None = None      # 使用场景（日常护肤 / 商务办公 / 户外运动...）

    # ── 排除与偏好 ──
    brand_avoid: list[str] = Field(default_factory=list)     # 要排除的品牌
    brand_prefer: list[str] = Field(default_factory=list)    # 偏好的品牌
    origin_avoid: list[str] = Field(default_factory=list)    # 要排除的产地
    product_type: str | None = None                          # 具体产品类型（洗面奶/手机/跑鞋...）

    # ── 美妆护肤专属 ──
    skin_type: str | None = None                             # 肤质（油性/干性/敏感肌...）
    ingredient_avoid: list[str] = Field(default_factory=list) # 要避开的成分（酒精/香精...）
    ingredient_prefer: list[str] = Field(default_factory=list) # 偏好的成分（玻尿酸/烟酰胺...）

    # ── 数码电子专属 ──
    storage: str | None = None         # 存储容量（256GB/512GB...）
    screen_size: str | None = None     # 屏幕尺寸

    # ── 服饰运动专属 ──
    sport_type: str | None = None      # 运动类型（跑步/篮球/瑜伽...）
    season: str | None = None          # 季节（春夏/秋冬）

    # ── 食品生活专属 ──
    dietary: list[str] = Field(default_factory=list)  # 饮食偏好（低糖/无麸质/清真...）


class CriteriaPayload(BaseModel):
    """购买标准 —— 发给前端的完整约束信息"""
    criteria_id: str = ""                                         # 标准 ID
    category: str = ""                                            # 品类（美妆护肤/数码电子/...）
    summary: str = ""                                             # 人类可读的总结
    chips: list[str] = Field(default_factory=list)                # 前端标签（如 ["油性肤质", "200以内"]）
    constraints: Constraints = Field(default_factory=Constraints) # 结构化约束（用于检索硬过滤）
    field_sources: dict[str, CriteriaFieldSource] = Field(default_factory=dict)  # 每个字段的来源


class ProductPayload(BaseModel):
    """商品信息 —— 发给前端的商品卡片数据"""
    product_id: str                               # 商品唯一 ID（如 p_beauty_001）
    name: str                                      # 商品名称
    price: float | None = None                     # 价格
    currency: str | None = None                    # 货币单位（CNY）
    image_url: str | None = None                   # 商品图片 URL
    category: str = ""                             # 所属品类
    sub_category: str | None = None                # 子品类
    brand: str | None = None                       # 品牌
    skin_type_match: list[str] = Field(default_factory=list)   # 适合肤质（美妆品类）
    ingredient_tags: list[str] = Field(default_factory=list)   # 成分标签
    ingredient_avoid: list[str] = Field(default_factory=list)  # 需注意的成分
    use_scenario: str | None = None                # 使用场景
    sku_options: list[dict[str, Any]] | None = None # SKU 规格选项（容量/颜色等）


class EvidencePayload(BaseModel):
    """证据片段 —— 推荐理由的来源"""
    source_type: str              # 来源类型（marketing/faq/review）
    snippet: str                  # 证据文本片段
    source_id: str | None = None  # 来源 chunk ID


class ReasonAtomPayload(BaseModel):
    """推荐理由原子 —— 一个维度的推荐理由"""
    dimension: str                # 维度名（肤质匹配/预算匹配/成分...）
    value: str                    # 维度值
    text: str                     # 可读的理由文本
    evidence_id: str | None = None # 对应的证据 ID


class AlternativePayload(BaseModel):
    """备选商品 —— 最终决策中非首选的商品"""
    product_id: str
    name: str


# ============================================================================
# 10 种 SSE 事件类型
# ============================================================================

class ThinkingEvent(SSEEventBase):
    """思考中 —— 告诉前端显示骨架屏/loading 动画"""
    event: Literal["thinking"] = "thinking"
    display_mode: DisplayMode = "inline_thinking"
    stage: str           # 当前阶段（understanding / analyzing_image / retrieving / recommending / deciding）
    message: str         # 展示给用户的文字（"正在理解您的需求..."）


class ClarificationEvent(SSEEventBase):
    """澄清问句 —— 信息不足时反问用户"""
    event: Literal["clarification"] = "clarification"
    display_mode: DisplayMode = "inline_card"
    question: str                                    # 问句（"您想找哪个品类的商品？"）
    required_slots: list[str] = Field(default_factory=list)     # 需要用户补充的槽位
    suggested_options: list[str] = Field(default_factory=list)  # 建议选项


class CriteriaCardEvent(SSEEventBase):
    """购买标准卡片 —— 展示系统理解的需求摘要"""
    event: Literal["criteria_card"] = "criteria_card"
    display_mode: DisplayMode = "summary_card"
    editable: bool = True                                       # 是否允许用户修改
    criteria: CriteriaPayload = Field(default_factory=CriteriaPayload)     # 购买标准
    shopping_strategy: ShoppingStrategyPayload | None = None    # 选购策略
    quick_actions: list[QuickActionPayload] = Field(default_factory=list)  # 快捷操作按钮


class TextDeltaEvent(SSEEventBase):
    """流式文本增量 —— ChatGPT 式逐字输出"""
    event: Literal["text_delta"] = "text_delta"
    display_mode: DisplayMode = "inline_text"
    message_id: str    # 消息 ID（同一个消息的多个 delta 共享）
    delta: str         # 增量文本片段
    done: bool = False # 这个消息是否结束


class ProductCardEvent(SSEEventBase):
    """商品卡片 —— 一个推荐商品"""
    event: Literal["product_card"] = "product_card"
    display_mode: DisplayMode = "swipe_deck_item"
    deck_id: str                                          # 牌组 ID（必须）
    rank: int                                             # 排名
    product: ProductPayload                                # 商品信息
    reason: str                                           # 推荐理由文案
    reason_atoms: list[ReasonAtomPayload] = Field(default_factory=list)  # 推荐理由原子
    risk_notes: list[str] = Field(default_factory=list)                # 风险提示
    evidence: list[EvidencePayload] = Field(default_factory=list)      # 证据来源
    actions: list[QuickActionPayload] = Field(default_factory=list)    # 操作按钮


class CartActionEvent(SSEEventBase):
    """购物车操作事件 —— 加购/查看/删除/修改数量"""
    event: Literal["cart_action"] = "cart_action"
    display_mode: DisplayMode = "inline_card"
    action: str                           # 操作类型（add/remove/update_quantity/view）
    product_id: str                       # 操作的商品
    quantity: int = 1                     # 数量
    status: str = "success"               # 状态（success/failed）
    cart: CartSummaryPayload | None = None # 操作后的购物车摘要


class FinalDecisionEvent(SSEEventBase):
    """最终决策 —— 综合评分后的推荐结果"""
    event: Literal["final_decision"] = "final_decision"
    display_mode: DisplayMode = "summary_card"
    winner_product_id: str                                 # 胜出商品 ID
    summary: str                                           # 决策总结
    why: list[str] = Field(default_factory=list)           # 为什么推荐
    not_for: list[str] = Field(default_factory=list)       # 不适合的原因
    alternatives: list[AlternativePayload] = Field(default_factory=list)  # 备选
    next_actions: list[QuickActionPayload] = Field(default_factory=list)  # 下一步操作
    decision_status: Literal["selected", "no_match", "no_suitable_winner", "needs_more_signal"] | None = None
    confidence: Literal["high", "medium", "low"] | None = None  # 置信度
    next_step: Literal["adjust_criteria", "replace_deck", "continue_current_deck", "accept_recommendation"] | None = None
    score_breakdown: dict[str, Any] | None = None           # 评分明细


class DoneEvent(SSEEventBase):
    """结束标记 —— 一个 turn 的终止符"""
    event: Literal["done"] = "done"
    display_mode: DisplayMode = "none"
    finish_reason: DoneFinishReason = "completed"


class ErrorEvent(SSEEventBase):
    """错误事件 —— 返回给用户的错误信息"""
    event: Literal["error"] = "error"
    display_mode: DisplayMode = "inline_card"
    code: str              # 错误码（PIPELINE_ERROR / VALIDATION_ERROR / ...）
    message: str           # 用户可见的错误信息（脱敏后）
    retryable: bool = True # 是否可重试


class CompareCardEvent(SSEEventBase):
    """多商品对比卡片 —— 两个以上商品的维度对比"""
    event: Literal["compare_card"] = "compare_card"
    display_mode: DisplayMode = "summary_card"
    compare_id: str                                   # 对比 ID
    source_deck_id: str | None = None                 # 来源牌组
    mode: Literal["exploratory", "decision"]          # 对比模式
    focus: str | None = None                          # 对比焦点
    products: list[ProductPayload] = Field(default_factory=list)        # 参与对比的商品
    axes: list[CompareAxisPayload] = Field(default_factory=list)        # 对比维度
    winner_product_id: str | None = None              # 优胜商品
    winner_reason: str | None = None                  # 优胜理由
    tradeoffs: list[str] = Field(default_factory=list)                 # 权衡说明
    risk_notes: list[CompareRiskNotePayload] = Field(default_factory=list)  # 风险提示
    confidence: Literal["high", "medium", "low"] | None = None  # 置信度


# ============================================================================
# 辅助 Payload 类型（被上面的事件类型引用）
# ============================================================================

class CartItemEventPayload(BaseModel):
    """购物车商品项"""
    product_id: str
    name: str
    price: float | None = None
    quantity: int = 1
    added_at: str | None = None
    product: ProductPayload | None = None  # 关联的完整商品信息


class CartSummaryPayload(BaseModel):
    """购物车摘要"""
    items: list[CartItemEventPayload] = Field(default_factory=list)
    total_items: int = 0
    total_price: float = 0.0


class CompareAxisValuePayload(BaseModel):
    """对比维度中单个商品的得分"""
    product_id: str
    score: float | None = None
    label: str | None = None
    detail: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)


class CompareAxisPayload(BaseModel):
    """对比维度"""
    name: str
    values: list[CompareAxisValuePayload] = Field(default_factory=list)


class CompareRiskNotePayload(BaseModel):
    """对比中的风险提示"""
    product_id: str
    note: str


class QuickActionPayload(BaseModel):
    """快捷操作按钮 —— 前端在卡片上渲染的可点击按钮"""
    action_id: str
    label: str                         # 按钮文字（"加入购物车" / "不喜欢"）
    action: str                        # 动作类型（criteria_patch / feedback / add_to_cart / ...）
    feedback_type: str | None = None   # 反馈类型
    criteria_patch: dict[str, Any] | None = None  # 如果是修改标准，这里带修改值


class DecisionBarrierPayload(BaseModel):
    """决策障碍 —— 用户为什么犹豫"""
    barrier_type: Literal["fear_wrong_choice", "value_uncertainty", "fit_uncertainty",
                          "trust_uncertainty", "price_sensitive", "choice_overload"]
    label: str
    reason: str = ""
    conversion_strategy: str = ""  # 转化策略


class ShoppingStrategyPayload(BaseModel):
    """选购策略 —— 针对用户场景的选购建议"""
    strategy_id: str
    scene_type: Literal["gift", "interest", "travel", "usage", "risk_sensitive", "goal_oriented"]
    scene_summary: str = ""
    user_problem: str = ""
    decision_barrier: DecisionBarrierPayload | None = None
    primary_direction: PrimaryDirectionPayload     # 主要推荐方向
    avoid_risks: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"] = "medium"


class SearchStrategyPayload(BaseModel):
    """检索策略"""
    category: str | None = None
    product_type: str | None = None
    use_scenario: str | None = None


class PrimaryDirectionPayload(BaseModel):
    """主要推荐方向"""
    title: str
    summary: str = ""
    why: str = ""
    search_strategy: SearchStrategyPayload = Field(default_factory=SearchStrategyPayload)
    available_in_catalog: bool = False
    supporting_product_count: int = 0


# ============================================================================
# 事件类型联合 + 序列化
# ============================================================================

# 10 种事件类型的联合（TypeScript 的 Union 等价物）
SSEEvent = Union[
    ThinkingEvent,
    ClarificationEvent,
    CriteriaCardEvent,
    TextDeltaEvent,
    ProductCardEvent,
    CartActionEvent,
    FinalDecisionEvent,
    DoneEvent,
    ErrorEvent,
    CompareCardEvent,
]

# 事件名 → Pydantic 类的映射表（用于反序列化时路由到正确的类）
EVENT_TAG_MAP: dict[str, type[SSEEvent]] = {
    "thinking": ThinkingEvent,
    "clarification": ClarificationEvent,
    "criteria_card": CriteriaCardEvent,
    "text_delta": TextDeltaEvent,
    "product_card": ProductCardEvent,
    "cart_action": CartActionEvent,
    "final_decision": FinalDecisionEvent,
    "done": DoneEvent,
    "error": ErrorEvent,
    "compare_card": CompareCardEvent,
}


def parse_sse_event(data: str) -> SSEEvent | None:
    """
    反序列化：SSE 文本 → Pydantic 对象
    根据 event 字段自动选择对应的事件类
    """
    obj = json.loads(data)
    tag = obj.get("event")
    model_cls = EVENT_TAG_MAP.get(tag)
    if model_cls is None:
        return None
    return model_cls.model_validate(obj)


def format_sse(event: SSEEventBase) -> str:
    """
    序列化：Pydantic 对象 → SSE 文本格式

    输出格式：
        event: product_card
        data: {"event":"product_card","product":{...},...}
        <空行>

    这是 SSE 协议的标准格式，前端按此解析
    """
    tag = event.event
    data = event.model_dump_json()  # Pydantic → JSON 字符串
    return f"event: {tag}\ndata: {data}\n\n"


# ============================================================================
# 铁律 1 守卫：导入时校验 Python 端与 JSON Schema 端一致
#
# 如果两边不一致，这个模块导入就会报 ImportError，uvicorn 启动失败。
# 相当于 Rust 的编译期检查 —— 协议不一致的代码跑不起来。
# ============================================================================

def _load_schema_event_types() -> frozenset[str]:
    """从 contracts/sse-events.schema.json 中提取所有事件类型名"""
    schema_path = Path(__file__).resolve().parents[3] / "contracts" / "sse-events.schema.json"
    if not schema_path.exists():
        return frozenset()  # 测试环境可能没有 schema 文件，跳过检查
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    types: list[str] = []
    for name, defn in schema.get("$defs", {}).items():
        for part in defn.get("allOf", []):
            event_const = part.get("properties", {}).get("event", {}).get("const")
            if event_const:
                types.append(event_const)
    return frozenset(types)


def _verify_protocol_consistency() -> None:
    """
    对比 JSON Schema 和 Python EVENT_TAG_MAP 的事件类型列表
    不一致就抛出 ImportError，阻止启动
    """
    schema_types = _load_schema_event_types()
    if not schema_types:
        return

    python_types = frozenset(EVENT_TAG_MAP.keys())

    in_schema_not_python = schema_types - python_types  # Schema 有，Python 没有
    in_python_not_schema = python_types - schema_types  # Python 有，Schema 没有

    errors: list[str] = []
    if in_schema_not_python:
        errors.append(
            f"JSON Schema 定义了 Python 缺失的事件类型: {sorted(in_schema_not_python)}. "
            f"需要在 sse_events.py 添加对应的 Pydantic 类 + EVENT_TAG_MAP 条目."
        )
    if in_python_not_schema:
        errors.append(
            f"Python 定义了 Schema 未包含的事件类型: {sorted(in_python_not_schema)}. "
            f"必须先更新 contracts/sse-events.schema.json（铁律1: Schema 是第一真相源）."
        )
    if errors:
        raise ImportError(
            "SSE 协议不一致 —— 服务器无法启动.\n"
            + "\n".join(errors)
            + "\n\n修复方案: 更新不一致的那一端，然后重启."
        )


# 模块导入时自动执行校验
_verify_protocol_consistency()
