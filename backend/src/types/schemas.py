"""HTTP request and response contracts for the BuyPilot backend.

API modules import these models instead of defining request shapes inline.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from src.types.sse_events import CriteriaPayload, EvidencePayload, ProductPayload


# ═══════════════════════════════════════════════════════════════════════════════
# 对话历史中的单条消息（轻量版）
#
# 前端把当前会话的最近 N 条对话记录放在 ChatStreamRequest.history 里传给后端，
# 后端注入到 LLM prompt 中，让模型知道"前面聊了什么"。
#
# 为什么不用完整的 Message 对象？
#   多轮对话只需要 role + content 两个字段就够了。
#   不需要 timestamp / message_id / metadata 等额外信息，减少传输量。
# ═══════════════════════════════════════════════════════════════════════════════
class MessageLite(BaseModel):
    # 消息角色 —— 谁说的这句话
    #   "user"      : 用户说的话（"推荐油皮洗面奶"）
    #   "assistant" : AI 的回复（"为您找到了以下商品..."）
    #   "system"    : 系统消息（一般不出现在 history 中，而是由后端注入到 prompt）
    role: Literal["user", "assistant", "system"]

    # 消息正文 —— 对话的文本内容
    content: str


# ═══════════════════════════════════════════════════════════════════════════════
# POST /chat/stream 的请求体
#
# 这是 Android 客户端每次发起聊天请求时发送的 JSON 对象。
# FastAPI 会自动把 JSON 反序列化 + 校验为该 Pydantic model。
# 如果 JSON 字段类型不对或缺少必填字段 → FastAPI 返回 422，不会进入业务代码。
#
# Java 等价物：
#   public record ChatStreamRequest(
#       @Size(max=2000) String message,
#       String sessionId,
#       List<MessageLite> history,
#       ...
#   ) {}
#
# 数据流：
#   Android OkHttp POST JSON body
#     → FastAPI 自动反序列化为 ChatStreamRequest
#     → chat.py 的 stream_chat(body)
#     → pipeline.chat_stream(sid, body)  —— body 一路传到 pipeline
# ═══════════════════════════════════════════════════════════════════════════════
class ChatStreamRequest(BaseModel):
    # ── 核心输入 ──────────────────────────────────────────────────────────────
    # 用户输入的文本消息。
    # 允许空字符串 "" —— 因为用户可能只传图片不传文字。
    # max_length=2000：防止恶意超长输入（实际上用户不会打 2000 字以上的需求）。
    # 如果 message 和 image_url 都为空 → model_validator 在校验时 reject。
    message: str = Field(default="", min_length=0, max_length=2000)

    # ── 会话标识 ──────────────────────────────────────────────────────────────
    # 会话 ID，关联多轮对话。
    # 第一轮对话时客户端传 null（还没创建会话），服务端在 chat.py 中生成 "sess_{uuid}"。
    # 后续轮次客户端把上一轮返回的 session_id 原样带回，服务端就能关联到同一会话。
    # 如果客户端不传 → 每轮都是新会话，丢失多轮上下文。
    session_id: str | None = None

    # ── 对话历史 ──────────────────────────────────────────────────────────────
    # 当前会话的最近 N 条消息记录，每条是 MessageLite（role + content）。
    # 后端注入到 LLM prompt 中，让 LLM 知道对话上下文。
    # 为什么由客户端传而不由服务端从数据库查？
    #   1. 减少数据库查询（服务端不需要每次请求都查 conversations 表）
    #   2. 客户端最清楚自己显示了哪些消息
    #   3. 离线场景下客户端可以先缓存历史，不依赖服务端存储
    # default_factory=list：如果不传 → 空列表，不会报错
    history: list[MessageLite] = Field(default_factory=list)

    # ── 多模态输入 ────────────────────────────────────────────────────────────
    # 图片的 URL 或 Base64 data URI。
    # 用户拍照找货时，Android 先调 POST /upload 上传图片拿到 URL，
    # 然后把这个 URL 放在这里传给聊天接口。
    # 也可以直接传 Base64 编码的图片数据（data:image/jpeg;base64,...），
    #   但推荐用 URL 方式（减少请求体大小）。
    image_url: str | None = None

    # ── 购买标准修改 ──────────────────────────────────────────────────────────
    # 用户在 criteria_card 上点击 quick_action 修改购买标准时，
    # 前端把修改的字段打包成 dict 放在这里。
    # 例如用户把预算从 200 改成 300：
    #   { "budget_max": 300 }
    # 后端 pipeline 把这个 patch merge 到现有的 CriteriaPayload.constraints 中。
    # 用 dict 而不是强类型模型 —— 因为修改的字段不确定（可能是任何 constraint 维度），
    #   强类型会导致大量 Optional 字段，dict 更灵活，且在 pipeline 中有专门的解包逻辑。
    criteria_patch: dict[str, Any] | None = None

    # ── 阶段跳过 ──────────────────────────────────────────────────────────────
    # 开发/调试用 —— 跳过 pipeline 中的某些阶段，加速响应。
    # 例如 ["criteria", "rerank"] 表示跳过购买标准生成和重排序。
    # 生产环境中通常为空列表。
    skip_stages: list[str] = Field(default_factory=list)

    # ── 客户端生成的 turn 标识 ────────────────────────────────────────────────
    # 客户端为每轮对话生成的唯一 ID。
    # 如果客户端不传 → chat.py 自动生成 "turn_{uuid[:8]}"。
    # 客户端传的好处：客户端在请求发出前就已经知道 turn_id，
    #   可以在 UI 层用这个 ID 做去重（避免同一个 turn 被处理两次）。
    client_turn_id: str | None = None

    # ── 全链路追踪标识 ────────────────────────────────────────────────────────
    # 客户端生成的全链路追踪 ID（贯穿 Android → 后端 → LLM API → 返回全过程）。
    # 如果出 Bug，用户把这个 ID 发给开发者，开发者可以在日志系统中
    #   grep 到从请求进入到 SSE 返回的所有日志。
    # 不传不影响功能，只是少了一个调试手段。
    client_trace_id: str | None = None

    # ── 决策收敛模式 ──────────────────────────────────────────────────────────
    # 当用户在多轮对话中逐步缩小范围后，客户端设置 converge=true，
    # 告诉后端"用户已经提供了足够信息，直接做最终决策，不要再反问/推荐更多商品"。
    # pipeline.py 中看到 converge=true → 跳过部分追问逻辑 → 直接进入 decision 阶段。
    converge: bool = False

    # ── 多商品对比 ─────────────────────────────────────────────────────────────
    # 用户想对比的商品 ID 列表（"帮我对比一下这两个"）。
    # 支持 str 和 int 两种类型 —— 因为历史数据中 product_id 可能是数字格式。
    # pipeline 检测到非空后 → 进入 compare handler → 发 compare_card 事件。
    compare_product_ids: list[str | int] | None = None

    # ═══════════════════════════════════════════════════════════════════════════
    # Pydantic model_validator —— 跨字段校验
    #
    # 单个字段的校验（如 max_length=2000）由 Field() 完成。
    # 但 "message 和 image_url 至少有一个非空" 是跨字段规则，
    # 需要用 model_validator。
    #
    # mode="after"：先做字段级校验（非空、类型），再做这个跨字段校验。
    #   如果字段都非法，Pydantic 直接返回 422，不会进入这里。
    # ═══════════════════════════════════════════════════════════════════════════
    @model_validator(mode="after")
    def _require_message_or_image(self) -> "ChatStreamRequest":
        # .strip() 去掉纯空白字符 —— "   " 视为无效输入
        if not self.message.strip() and not (self.image_url and self.image_url.strip()):
            raise ValueError("message 和 image_url 至少需要一个")
        # 校验通过 → 返回 self（Pydantic 会用返回值替换原始对象）
        return self


class CancelRequest(BaseModel):
    session_id: str
    turn_id: str


class CancelResponse(BaseModel):
    session_id: str
    turn_id: str
    canceled: bool = True


class ImageUploadResponse(BaseModel):
    image_url: str
    width: int | None = None
    height: int | None = None
    mime_type: str = "image/jpeg"
    ocr_text: str | None = None
    analysis: dict[str, Any] = Field(default_factory=dict)


class FeedbackRequest(BaseModel):
    session_id: str
    deck_id: str | None = None
    feedback_type: str | None = None
    action: str | None = None
    product_id: str | None = None
    reason: str | None = None


class FeedbackResponse(BaseModel):
    status: str = "received"
    session_id: str
    feedback_type: str | None = None
    action: str | None = None


class CartItemPayload(BaseModel):
    product_id: str
    name: str
    price: float | None = None
    quantity: int = 1
    added_at: str | None = None
    product: ProductPayload | None = None


class CartResponse(BaseModel):
    items: list[CartItemPayload] = Field(default_factory=list)
    total_items: int = 0
    total_price: float = 0.0


class CartMutationRequest(BaseModel):
    quantity: int = Field(default=1, ge=0)


class IntentResult(BaseModel):
    intent: Literal[
        "recommend",
        "clarify",
        "continue",
        "feedback",
        "compare",
        "add_to_cart",
        "remove_from_cart",
        "update_cart_quantity",
        "view_cart",
        "checkout_preview",
        "checkout_confirm",
        "checkout_cancel",
        "chitchat",
    ]
    confidence: float = 1.0
    category: str | None = None
    extracted_constraints: dict[str, Any] = Field(default_factory=dict)
    soft_preferences: list[str] = Field(default_factory=list)
    target_product_id: str | None = None
    target_product_name: str | None = None
    compare_product_ids: list[str | int] = Field(default_factory=list)


class RecommendationResult(BaseModel):
    text_chunks: list[str] = Field(default_factory=list)
    products: list[ProductPayload] = Field(default_factory=list)
    evidence_by_product: dict[str, list[EvidencePayload]] = Field(default_factory=dict)


DecisionStatus = Literal["selected", "no_match", "no_suitable_winner", "needs_more_signal"]
DecisionConfidence = Literal["high", "medium", "low"]
DecisionNextStep = Literal["adjust_criteria", "replace_deck", "continue_current_deck", "accept_recommendation"]


class DecisionResult(BaseModel):
    winner_product_id: str
    summary: str
    why: list[str] = Field(default_factory=list)
    not_for: list[str] = Field(default_factory=list)
    decision_status: DecisionStatus | None = None
    confidence: DecisionConfidence | None = None
    next_step: DecisionNextStep | None = None


class SessionState(BaseModel):
    session_id: str
    last_criteria: CriteriaPayload | None = None
    last_product_ids: list[str] = Field(default_factory=list)


class FaqItem(BaseModel):
    question: str
    answer: str


class ReviewItem(BaseModel):
    nickname: str
    rating: int
    content: str


class ProductDetailResponse(BaseModel):
    product: ProductPayload
    marketing_description: str | None = None
    highlights: list[str] = Field(default_factory=list)
    faqs: list[FaqItem] = Field(default_factory=list)
    reviews: list[ReviewItem] = Field(default_factory=list)


class EvalRunRequest(BaseModel):
    """Request to trigger an evaluation run."""

    strategy_tag: str = "baseline"
    run_name: str | None = None
    prompt_version: str | None = None
