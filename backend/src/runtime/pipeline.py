"""
主流程编排 —— 单轮对话的总 owner（负责人）

这是整个后端的"指挥中心"。pipeline 自己不干业务活（不调 LLM、不查数据库、
不拼 prompt），它只决定一件事：先调谁、后调谁、出错了怎么办。

Java 对比：这是一个 @Service 的 orchestration 方法，它按顺序调用
IntentService → CriteriaService → RetrievalService → RecommendationService → DecisionService，
每个阶段的结果通过 yield 逐个发回给 Controller。

一个完整请求的 5 个阶段（按顺序执行）：
  1. _prepare_pipeline_body   → 图片预分析（有图时调 Qwen-VL-Plus 理解图片内容）
  2. _resolve_intent ★重点   → 意图识别（8 条确定性规则优先 → LLM 兜底）
  3. _missing_slots           → 槽位检查（缺品类就问用户）
  4. _dispatch_intent_handler → 按意图分发到具体 handler（推荐/加购/对比/…）
  5. handler 内部              → 购买标准 → 检索 → 推荐 → 决策 → 产出 SSE 事件

数据流（从 HTTP 请求到 SSE 事件）：
  chat.py 的 stream_chat() 调用
    → chat_stream(session_id, body)          ← 入口 + 异常处理壳
      → _run_chat_turn(ctx, body)            ← 真正的编排逻辑
        → _prepare_pipeline_body             ← 阶段1: 图片分析
        → 商业声明检查                        ← "有货吗？"→固定回复，不调LLM
        → _resolve_intent                    ← 阶段2: 意图识别 ★最复杂
        → _merge_followup_context            ← 多轮上下文合并
        → _unsupported_product_type 检查     ← 不支持的产品类型拦截
        → 对比不足检查                        ← "对比"但没有候选商品
        → _missing_slots                     ← 阶段3: 槽位检查
        → _emit_clarification                ← 反问用户缺什么
        → _dispatch_intent_handler           ← 阶段4: 意图分发到handler
          → handler 内部完成推荐/加购/对比等逻辑
"""

# __future__ 导入：让类型注解中的 forward reference（前向引用）可以不用引号包裹
# 例如 async def foo() -> AsyncGenerator[SSEEventBase, None] 不需要写成 "SSEEventBase"
from __future__ import annotations

import logging       # 日志记录（logger.info/logger.warning/logger.exception）
import time          # 性能计时（time.perf_counter() 高精度计时）
import uuid          # 生成唯一 ID（uuid4().hex 生成 32 位随机字符串）
import weakref       # 弱引用（用于 _TurnGuard 的 GC 自动清理机制）

# typing 模块的类型提示工具
# AsyncGenerator: 异步生成器的类型标注（yield 产出的值 → 外部发给它的值）
# Awaitable: 可等待对象（async/await 的对象）
# Callable: 可调用对象（函数）
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass   # @dataclass 装饰器：自动生成 __init__/__repr__/__eq__ 等方法
from typing import Any              # Any 表示任意类型（类似 Java 的 Object）

# ── Runtime 层 ──
# StreamCancelled: 用户取消时抛出的异常类型
# register_turn: 注册一个 turn 到取消注册表（返回 cancel_token）
# unregister_turn: 从取消注册表中移除 turn
from src.runtime.cancel_registry import StreamCancelled, register_turn, unregister_turn

# INTENT_HANDLERS: 意图 → handler 函数的映射字典
#   例如 {"recommend": handle_recommend, "add_to_cart": handle_add_to_cart, ...}
# handle_clarification: 处理澄清意图，构造 ClarificationEvent（问句+选项）并发送
# _criteria_card_event: 构造 CriteriaCardEvent 对象（购买标准卡片）
from src.runtime.handlers import INTENT_HANDLERS, handle_clarification, _criteria_card_event

# ── Service 层：消息规则 ──
# ★ 这些函数全部是确定性代码（正则/字典/关键词匹配），不调 LLM
# 它们是铁律 3（"路由逻辑必须代码化"）的数据基础
from src.services.message_rules import (
    COMMERCIAL_CLAIM_REPLY,           # 预定义的固定回复文案："建议查看商品详情页了解库存和物流信息"
    is_compare_phrase,               # 检测消息是否包含对比意图（"对比/比较/哪个更好"）
    is_commercial_claim_question,     # 检测消息是否是商业声明类问题（"有货吗/包邮吗"）
    is_replace_deck_phrase,           # 检测消息是否要求换一组推荐（"换一组/换一批/再看看"）
    maybe_cart_intent,                # 检测消息是否是加购意图 → 返回 IntentResult 或 None
    maybe_checkout_intent,            # 检测消息是否是结算意图 → 返回 IntentResult 或 None
    maybe_intercept_budget_patch,     # 检测消息是否是预算修改（"便宜点/200以内"）→ 修改 criteria_patch
    maybe_shopping_intent,            # 检测消息是否是购物意图（"推荐/帮我选"）→ 返回 IntentResult 或 None
    message_with_image_context,       # 把 VL 图片分析结果拼接到用户消息文本末尾
    resolve_compare_ids_mixed,        # 混合解析对比目标ID（支持序数、商品名、product_id 三种格式）
    resolve_compare_targets,          # 从消息文本中解析对比目标的 product_id 列表
)

# ── Service 层：意图解析 ──
from src.services.intent_resolution import (
    has_context_value,               # 判断一个值是否"有意义"（非空字符串/非 None/非空列表/非零）
    resolve_intent_constraints,       # 对 LLM 返回的约束做后处理：品牌别名→规范名、品类归一化
)

# ── Runtime Stages：5 个独立阶段，每个阶段封装一个 LLM 任务调用 ──
# criteria_from_intent: 从 IntentResult 快速构造初步 CriteriaPayload（用于澄清/投机检索时）
# run_criteria: 完整的购买标准生成（内部调 llm_client.generate_criteria）
from src.runtime.stages.criteria import criteria_from_intent, run_criteria
# run_decision: 最终决策生成（内部调 llm_client.generate_decision）
from src.runtime.stages.decision import run_decision
# run_intent: 意图识别（内部调 llm_client.analyze_intent）
from src.runtime.stages.intent import run_intent
# run_multimodal: 图片理解（内部调 llm_client.analyze_image）
# run_image_embedding: 图片向量化（内部调 embedding.embed_image）
from src.runtime.stages.multimodal import run_image_embedding, run_multimodal
# run_retrieval: 混合检索（内部调 retriever.retrieve_with_evidence）
# run_recommendation_text: 推荐文案一次性生成（内部调 llm_client.generate_recommendation）
# run_recommendation_text_stream: 推荐文案流式生成（逐 token 产出）
from src.runtime.stages.recommendation import run_recommendation_text, run_recommendation_text_stream, run_retrieval
# check_required_slots: 纯代码规则，检查品类等必填槽位是否缺失
from src.runtime.stages.slot_checker import check_required_slots

# ── Config ──
# msg: 用户可见的文案模板（集中管理在 config/user_messages.py，修改不需要搜代码）
from src.config import user_messages as msg
from src.config.domain_terms import (
    KNOWN_CATEGORIES,              # 系统支持的品类列表：["美妆护肤", "数码电子", "服饰运动", "食品生活"]
    is_known_brand_or_synonym,     # 判断一个字符串是否是已知品牌名或其别称（如"苹果"→True，既是水果也是品牌）
    is_supported_product_type,     # 判断产品类型是否在商品库中存在
    normalize_category,            # 品类别名归一化："美妆" → "美妆护肤"
)

# ── Runtime 基础设施 ──
# ordinal_index: 从消息文本中提取序数（"第一个"→0, "第二个"→1, "第三个"→2）
# message_refers_to_previous_product: 检测消息是否用代词指代了之前的商品（"那个/这个/它"）
from src.runtime.cart_rules import message_refers_to_previous_product, ordinal_index

# StreamContext: 流上下文 dataclass，承载整个 turn 生命周期中的状态
#   - session_id/turn_id/deck_id：标识信息
#   - seq: EventSeq 对象（事件序号自增器）
#   - cancel_token: 取消令牌（用户点取消时触发）
#   - stages: PipelineStages 对象（5 个阶段函数集合）
#   - background_tasks: 后台任务集合（用于追踪异步任务）
#   - stage_timings_ms: 各阶段耗时记录
# StageResult: 简单的 dataclass wrapper，标记"阶段完成了，这是结果"
# cancel_background_tasks: 取消所有在 background_tasks 中注册的后台任务
# run_with_heartbeat: ★关键函数——在等待 LLM 响应期间，每 0.8s 发一次 thinking 心跳事件，
#   同时监听 cancel_token，用户取消时立即中断
from src.runtime.streaming import (
    RunRetrieval, StageResult, StreamContext,
    cancel_background_tasks, run_with_heartbeat,
)

# ── Service 层：横切能力 ──
# record_audit_event: 向 audit_events 表写入一条审计日志（谁在什么时候做了什么）
from src.services.audit import record_audit_event
# clear_chat_turn: 清理 turn 注册记录（从 active_chat_turns 表中删除）
# register_chat_turn: 注册 turn 到 active_chat_turns 表（用于取消机制）
from src.services.cancellation import clear_chat_turn, register_chat_turn
# get_session_cart: 查询当前会话的购物车状态（返回 CartSummaryPayload）
from src.services.cart import get_session_cart
# get_previous_criteria: 从 DB 取上一轮对话保存的购买标准（多轮约束继承用）
# get_previous_product_ids: 从 DB 取上一轮推荐的商品 ID 列表（序数指代解析用："第一个""第二个"）
# save_recommendation_turn: 保存本轮的标准和商品 ID 到 DB（供下一轮读取）
from src.services.conversation_state import (
    get_previous_criteria, get_previous_product_ids, save_recommendation_turn,
)
# reset_fallback_events: 清空 fallback 事件计数器（每个 turn 独立统计降级次数）
from src.services.fallbacks import reset_fallback_events
# get_feedback_context: 取用户在当前会话中的反馈上下文（反选/排除的商品ID和成分）
from src.services.feedback import get_feedback_context
# update_request_context: 更新全局请求上下文（ContextVar），注入 trace_id/session_id/turn_id
from src.services.request_context import update_request_context

# ── Types ──
from src.types.schemas import ChatStreamRequest, DecisionResult, IntentResult, RecommendationResult
from src.types.sse_events import (
    CriteriaPayload, ErrorEvent, EventSeq, EvidencePayload,
    ProductPayload, SSEEventBase, TextDeltaEvent, now_ms,
)

HEARTBEAT_INTERVAL_SECONDS = 0.8  # 心跳间隔：等待 LLM 时每 0.8 秒发一个 thinking 事件
PUBLIC_PIPELINE_ERROR_MESSAGE = msg.PIPELINE_ERROR  # 用户看到错误文案的常量引用

logger = logging.getLogger(__name__)  # 本模块的日志记录器


# ═══════════════════════════════════════════════════════════════════════════════
# 基础设施类（不是业务逻辑，是工程保障）
# ═══════════════════════════════════════════════════════════════════════════════

class _TurnGuard:
    """
    取消令牌的自动清理器（防止资源泄漏的安全网）。

    为什么需要这个？
    chat_stream() 是一个 async generator（异步生成器）。
    如果客户端在 generator 运行到一半时断开 TCP 连接（用户关App、Nginx超时等），
    generator 不会被正常迭代完，finally 块可能不执行，cancel_token 就泄漏了。

    这个类利用 Python 的 weakref.finalize（弱引用终结器）机制：
    当 _TurnGuard 对象被 GC（垃圾回收）时，自动调用 unregister_turn() 清理令牌。
    这是"双重保险"——正常路径在 finally 中清理，异常路径靠 GC 兜底。
    """

    def __init__(self, session_id: str, turn_id: str) -> None:
        self.session_id = session_id
        self.turn_id = turn_id
        # ★ weakref.finalize(obj, func, *args)
        #   当 obj（self）被垃圾回收时，自动调用 func(*args)
        #   这里：self 被 GC 时 → 自动调用 unregister_turn(session_id, turn_id)
        self._finalizer = weakref.finalize(self, unregister_turn, session_id, turn_id)

    def detach(self) -> None:
        """
        告诉 finalizer："已经手动清理过了，不用自动清理了"。
        在 finally 块中调用，因为那里已经执行了 unregister_turn。
        如果不调用 detach()，GC 时会重复清理（虽然 unregister_turn 是幂等的，但避免无意义调用）。
        """
        self._finalizer.detach()


@dataclass(frozen=True)
# @dataclass 是 Python 3.7+ 的装饰器，自动生成 __init__ / __repr__ / __eq__ 等方法
# frozen=True 表示实例创建后不可修改（类似 Java 的 record / Kotlin 的 data class val）
class PipelineStages:
    """
    5 个阶段函数 + 检索函数的集合容器。

    为什么要包装在 dataclass 里而不是直接 import 调用？
    单元测试时可以只替换某一个阶段函数（monkeypatch），其他阶段不受影响。
    例如：把 run_intent 替换成返回固定 IntentResult 的 mock 函数，
    其他 4 个阶段仍然走真实逻辑。
    """
    run_multimodal: Callable[[str | None], Awaitable[dict[str, Any] | None]]
    # ↑ 图片理解函数：接收 image_url（或 None），异步返回分析结果 dict（或 None）

    run_image_embedding: Callable[[str | None], Awaitable[list[float] | None]]
    # ↑ 图片向量化函数：接收 image_url（或 None），异步返回 embedding 向量（或 None）

    run_intent: Callable[[str, ChatStreamRequest], Awaitable[IntentResult]]
    # ↑ 意图识别函数：接收 session_id + 请求体，异步返回意图识别结果

    run_criteria: Callable[[str, ChatStreamRequest, IntentResult], Awaitable[CriteriaPayload]]
    # ↑ 购买标准生成函数：接收 session_id + 请求体 + 意图，异步返回购买标准

    run_retrieval: RunRetrieval
    # ↑ 混合检索函数：接收 CriteriaPayload + top_n + feedback + image_embedding，返回 RetrievalOutput

    run_recommendation_text: Callable[
        [CriteriaPayload, list[ProductPayload], dict[str, list[EvidencePayload]] | None],
        Awaitable[RecommendationResult],
    ]
    # ↑ 推荐文案生成函数（一次性模式）：接收标准 + 商品列表 + 证据，返回推荐结果

    run_recommendation_text_stream: Callable[
        [CriteriaPayload, list[ProductPayload], dict[str, list[EvidencePayload]] | None],
        AsyncGenerator[str, None],
    ]
    # ↑ 推荐文案生成函数（流式模式）：同上参数，但是 async generator，逐 token 产出文本

    run_decision: Callable[..., Awaitable[DecisionResult]]
    # ↑ 最终决策生成函数：接收多个参数，返回决策结果


@dataclass(frozen=True)
class _ResolvedIntent:
    """意图识别阶段的产物 —— 一个已确定意图的包装"""
    body: ChatStreamRequest          # 可能已被修改的请求体（图片分析结果已注入消息文本）
    intent: IntentResult             # 确定后的意图（可能是规则结果，也可能是 LLM 结果）
    skip_slot_check: bool = False    # True = 跳过槽位检查（场景化请求不需要品类）


# ═══════════════════════════════════════════════════════════════════════════════
# ★ 第一层：chat_stream() —— 入口函数 + try-catch-finally 异常处理壳
#
# 这是 api/chat.py 直接调用的函数。
# 它自己不干编排的活，只做三件事：
#   1. 初始化 StreamContext（流上下文）和 cancel_token（取消令牌）
#   2. 调用 _run_chat_turn()，把产出的事件逐个透传给 chat.py
#   3. 处理三种结束方式（正常 / 取消 / 异常），各自做不同的清理和审计
#
# 为什么把 try-catch-finally 放在这里而不是 _run_chat_turn 里？
# → "正常路径"和"异常路径"分离。_run_chat_turn 只关心正常流程的编排，
#   不用在自己的代码里混杂异常处理逻辑。这是经典的两层架构模式。
# ═══════════════════════════════════════════════════════════════════════════════

async def chat_stream(session_id: str, body: ChatStreamRequest) -> AsyncGenerator[SSEEventBase, None]:
    # ↑ async def 定义异步函数（协程函数），调用它返回一个协程对象
    # ↑ AsyncGenerator[SSEEventBase, None] 类型注解：
    #     第一个参数 SSEEventBase 是 yield 产出的值的类型（给外部的）
    #     第二个参数 None 是外部通过 async_generator.asend() 发进来的值的类型（本项目不用，填 None）
    """
    单轮对话的完整生命周期入口。

    参数：
        session_id: 会话 ID（多轮共享，由 chat.py 生成或客户端传入）
        body:       请求体（包含 message, history, image_url, criteria_patch 等字段）

    产出：
        SSEEventBase 的事件流（ThinkingEvent / CriteriaCardEvent / ProductCardEvent 等），
        逐个被 chat.py 的 event_generator() 序列化成 "event:xxx\ndata:{json}\n\n" 并发送给客户端
    """

    # 清空上一轮的降级事件统计计数器（每个 turn 独立的统计数据）
    # reset_fallback_events 内部把 fallback 计数器归零
    reset_fallback_events()

    # 确定 turn_id（本轮的标识）：
    #   优先用客户端传的 client_turn_id（客户端可以做去重——同一个 turn_id 不重复处理）
    #   如果客户端没传，服务端自己生成一个 "turn_" + uuid 前 8 位
    turn_id = body.client_turn_id or f"turn_{uuid.uuid4().hex[:8]}"
    # uuid4().hex 示例: "a1b2c3d4e5f6..."（32位不带横线的十六进制字符串）

    # 把 session_id / turn_id / trace_id 注入到全局请求上下文（ContextVar）
    # 之后在任何深度的函数调用中，通过 get_request_context() 都能拿到这些值
    # 不需要函数参数层层传递！这是 Python 的"隐式上下文传递"机制
    update_request_context(trace_id=body.client_trace_id, session_id=session_id, turn_id=turn_id)

    # 注册取消令牌：
    #   register_turn 在 cancel_registry 中创建一个条目，返回一个 cancel_token（asyncio.Event 对象）
    #   用户发 POST /cancel 时，cancel_registry 会 set 这个 event
    #   后续代码中 ctx.ensure_active() 和 run_with_heartbeat 会检查这个 event 是否被 set
    cancel_token = register_turn(session_id, turn_id)

    # 创建 _TurnGuard 实例（GC 兜底：如果 generator 异常退出，weakref.finalize 自动清理令牌）
    guard = _TurnGuard(session_id, turn_id)

    # 创建流上下文 StreamContext 对象，承载整个 turn 的生命周期状态
    ctx = StreamContext(
        session_id=session_id,
        turn_id=turn_id,
        deck_id=f"deck_{turn_id}",            # 牌组ID：同一批推荐的 product_card 共享同一个 deck_id
        seq=EventSeq(turn_id),                # 事件序号自增器：从 1 开始，每个事件调用 seq.next() 获得一个递增序号
        cancel_token=cancel_token,            # 取消令牌
        stages=_current_stages(),             # 组装 5 个阶段函数（调用 _current_stages() 创建 PipelineStages 实例）
        heartbeat_interval_seconds=HEARTBEAT_INTERVAL_SECONDS,  # 心跳间隔 0.8 秒
    )

    try:
        # 注册 turn 到数据库（active_chat_turns 表），用于取消机制的状态追踪
        await register_chat_turn(session_id, turn_id, trace_id=body.client_trace_id)

        # 记录审计事件：turn 开始（写入 audit_events 表）
        # metadata 参数是附加的 JSON 数据，这里记录了是否有图片和消息长度
        await record_audit_event(
            "chat.turn_started", session_id=session_id, turn_id=turn_id,
            trace_id=body.client_trace_id, resource_type="chat_turn",
            resource_id=turn_id,
            metadata={"has_image": bool(body.image_url), "message_chars": len(body.message)},
        )

        # ★ 核心调用：
        # _run_chat_turn(ctx, body) 返回一个 async generator（异步生成器）
        # async for 逐个消费它产出的 SSE 事件：
        #   _run_chat_turn 内部 yield ThinkingEvent → 这里 yield ThinkingEvent → chat.py 序列化发送
        #   _run_chat_turn 内部 yield CriteriaCardEvent → 这里 yield CriteriaCardEvent → chat.py 序列化发送
        #   ...
        # 这就是 SSE 流式传输的本质：pipeline 产出一个，这里透传一个，客户端收到一个
        async for event in _run_chat_turn(ctx, body):
            yield event  # yield 关键字：产出值给 async for 的调用方（chat.py 的 event_generator）

        # turn 正常完成 → 记录审计事件
        # ctx.stage_timings_ms 是一个 dict，记录了每个阶段的耗时（如 {"image_analysis": 1234, "intent": 567}）
        await record_audit_event(
            "chat.turn_completed", session_id=session_id, turn_id=turn_id,
            trace_id=body.client_trace_id, resource_type="chat_turn",
            resource_id=turn_id,
            metadata={"stage_timings_ms": ctx.stage_timings_ms},
        )

    except StreamCancelled:
        # ★ 异常分支 1：用户主动取消
        # StreamCancelled 是自定义异常，在 run_with_heartbeat 中检测到 cancel_token 被 set 后抛出

        # 停掉所有还在运行的后台任务（如正在进行的 embedding/LLM 调用）
        cancel_background_tasks(ctx.background_tasks)

        # 记录审计：turn 被取消
        await record_audit_event(
            "chat.turn_cancelled", session_id=session_id, turn_id=turn_id,
            trace_id=body.client_trace_id, resource_type="chat_turn",
            resource_id=turn_id,
            metadata={"stage_timings_ms": ctx.stage_timings_ms},
        )

        # ctx.done("cancelled") 构造一个 DoneEvent(finish_reason="cancelled")
        # 前端收到这个事件后知道本轮被取消了，清理 UI 状态
        yield ctx.done("cancelled")

    except Exception as exc:
        # ★ 异常分支 2：未知异常（LLM 调用失败、DB 写入失败、代码 Bug 等）

        # 停掉所有后台任务
        cancel_background_tasks(ctx.background_tasks)

        # 记录完整的异常堆栈到日志（开发调试用，包含真实的 traceback）
        # logger.exception 会自动附加上 exc_info（异常类型 + 堆栈跟踪）
        logger.exception("chat_stream failed: session_id=%s turn_id=%s", session_id, turn_id)

        # 记录审计：turn 失败
        await record_audit_event(
            "chat.turn_failed", session_id=session_id, turn_id=turn_id,
            trace_id=body.client_trace_id, resource_type="chat_turn",
            resource_id=turn_id,
            metadata={"error_type": type(exc).__name__, "stage_timings_ms": ctx.stage_timings_ms},
        )

        # ★ 构造 ErrorEvent 发给用户
        # 注意：message 是脱敏的通用文案（"系统处理请求时遇到问题"），不包含真实的 traceback
        # 为什么要脱敏？因为 traceback 中可能包含数据库连接串、API Key、内部文件路径等敏感信息
        yield ErrorEvent(
            session_id=session_id, turn_id=turn_id,
            seq=ctx.seq.next(),                          # 分配新的递增序号
            event_id=ctx.seq.event_id(),                 # 生成全局唯一事件ID
            node_id=f"error_{turn_id}",                  # 前端路由用的节点 ID
            created_at_ms=now_ms(),                      # 毫秒级时间戳
            code="PIPELINE_ERROR",                       # 错误码（前端可以用来做 switch-case 跳转）
            message=f"{PUBLIC_PIPELINE_ERROR_MESSAGE} trace_id={turn_id}",  # 脱敏后的用户可见信息 + trace_id 便于客服定位
            retryable=True,                              # 告诉前端这个错误可以重试
        )

        # 发 done(finish_reason="error") 标记本轮异常结束
        yield ctx.done("error")

    finally:
        # ★ finally 块：无论如何都会执行（正常/取消/异常）
        # 1. guard.detach(): 通知 _TurnGuard "已手动清理，GC 时不用自动清理了"
        # 2. clear_chat_turn: 从 active_chat_turns 表中删除本 turn 的记录
        # 3. unregister_turn: 从取消注册表中移除本 turn
        guard.detach()
        await clear_chat_turn(session_id, turn_id)
        unregister_turn(session_id, turn_id)


# ═══════════════════════════════════════════════════════════════════════════════
# ★ 第二层：_run_chat_turn() —— 真正的编排逻辑
#
# 这个函数按顺序走过 8 个决策点，其中 4 个可能提前 return（提前结束本轮）。
# 只有全部通过才会进入最后的意图分发阶段（handler 干活）。
#
# 为什么有些检查排在前面？
# → "尽早失败"（Fail Fast）原则：
#   能明确拒绝的请求（不支持的产品类型、无意义的对比对话），
#   在调检索和 LLM 之前就拦截掉，不浪费算力和 token。
# ═══════════════════════════════════════════════════════════════════════════════

async def _run_chat_turn(ctx: StreamContext, body: ChatStreamRequest) -> AsyncGenerator[SSEEventBase, None]:
    """
    完整的一轮对话编排。返回前一定会 yield 一个 done 事件。

    参数：
        ctx:  StreamContext 流上下文（携带 session_id/turn_id/deck_id/seq/cancel_token/stages）
        body: ChatStreamRequest 请求体（message, history, image_url, ...）
    """

    # 记录总开始时间（perf_counter 返回高精度秒数，用于性能基准测试）
    t_total_start = time.perf_counter()
    # 打印日志：session_id / turn_id / 消息前 50 个字符
    logger.info("[timing] session=%s turn=%s msg=%s...", ctx.session_id, ctx.turn_id, body.message[:50])

    # ═══════════════════════════════════════════════════════════════════════════
    # 阶段 1：图片预分析
    # ═══════════════════════════════════════════════════════════════════════════
    pipeline_body = None  # pipeline_body 将持有可能被修改过的请求体
    t_prep_start = time.perf_counter()

    # async for 迭代 _prepare_pipeline_body 产出的每个 item
    # 这个 stage 可能产出两种东西：
    #   - StageResult(ChatStreamRequest) → 阶段完成了，Item.value 包含（可能已修改的）body
    #   - ThinkingEvent → 心跳事件，发给前端显示"正在分析图片"
    async for item in _prepare_pipeline_body(ctx, body):
        if isinstance(item, StageResult):
            # isinstance 检查 item 是否是 StageResult 类型（类似 Java 的 instanceof）
            pipeline_body = item.value  # StageResult 的 value 字段携带 ChatStreamRequest
        else:
            yield item  # 不是 StageResult 就是 ThinkingEvent，直接发给客户端

    # 防御性检查：_prepare_pipeline_body 必须产出 StageResult
    if pipeline_body is None:
        raise RuntimeError("pipeline body stage completed without a result.")
    # 记录图片分析耗时
    logger.info("[timing] image_analysis=%.3fs", time.perf_counter() - t_prep_start)

    # ═══════════════════════════════════════════════════════════════════════════
    # ★ 商业声明拦截 —— 防 LLM 幻觉的关键防线
    # ═══════════════════════════════════════════════════════════════════════════
    # is_commercial_claim_question: 检测消息是否是商业承诺类问题
    #   （"有货吗/包邮吗/支持货到付款吗/有优惠券吗"）
    # maybe_checkout_intent: 检测是否包含结算意图（"结算/买单/去付款"）
    #
    # 为什么这样处理？
    # LLM 不知道真实的库存/物流/优惠状态。如果让它回答"有货吗"，
    # 它会根据训练数据中的电商对话模式编造答案（"仓库有货，一般3-5天到"）——这是纯粹的幻觉。
    # 所以直接返回固定话术，不调 LLM，不调检索。
    #
    # 例外：如果同时有结算意图（"结算"），说明用户在正常使用购物车流程，
    # 不拦截，走正常的 checkout 流程。
    if is_commercial_claim_question(pipeline_body.message) and maybe_checkout_intent(pipeline_body.message) is None:
        async for event in _emit_commercial_claim_reply(ctx):
            yield event
        return  # ← 直接结束本轮，后续所有阶段跳过

    # ═══════════════════════════════════════════════════════════════════════════
    # 阶段 2：意图识别 ★ 全链路最复杂的阶段
    # ═══════════════════════════════════════════════════════════════════════════
    resolved = None  # 将持有 _ResolvedIntent 实例
    t_intent_start = time.perf_counter()

    async for item in _resolve_intent(ctx, pipeline_body):
        if isinstance(item, StageResult):
            resolved = item.value  # 提取 _ResolvedIntent
        else:
            yield item  # ThinkingEvent 心跳

    if resolved is None:
        raise RuntimeError("intent stage completed without a result.")
    # 记录意图识别耗时 + 识别出的意图类型
    logger.info("[timing] intent=%.3fs intent=%s", time.perf_counter() - t_intent_start, resolved.intent.intent)

    # 多轮上下文合并：
    #   把上一轮的购买标准（skin_type/budget_max/brand_prefer 等）继承到本轮
    #   这样用户说"预算控制在200"时，之前设置的"油性肤质"不会丢失
    resolved = await _merge_followup_context(ctx.session_id, resolved)

    # ═══════════════════════════════════════════════════════════════════════════
    # 不支持的产品类型检查
    # ═══════════════════════════════════════════════════════════════════════════
    # _unsupported_product_type: 检查用户想买的品类/子品类是否在商品库中存在
    #   例如用户说"推荐一架飞机" → category="航空器" 不在 KNOWN_CATEGORIES 中 → 不支持
    if _unsupported_product_type(resolved.intent):
        # 提取 product_type（可能是 LLM 返回的或规则提取的）
        _pt = (resolved.intent.extracted_constraints or {}).get("product_type")

        # ★ 特殊例外：product_type 实际上是一个品牌名
        # 例如用户说"推荐苹果"，LLM 可能把"苹果"识别为 product_type
        # 但实际上"苹果"是品牌 Apple 的别称，不是产品类型
        # is_known_brand_or_synonym("苹果") → True
        # 此时不应该拦截，而是清除错误的 product_type，让品牌匹配器去处理
        if _pt and is_known_brand_or_synonym(str(_pt)):
            # 从 constraints 中移除 product_type（它不是产品类型，是品牌名）
            constraints = dict(resolved.intent.extracted_constraints or {})
            constraints["product_type"] = None  # 设为 None 表示"没有 product_type 约束"
            # model_copy(update=...) 是 Pydantic 的方法：创建新对象，部分字段更新
            resolved = _ResolvedIntent(
                body=resolved.body,
                intent=resolved.intent.model_copy(update={"extracted_constraints": constraints}),
            )
        else:
            # 确实不支持 → 发提示文本 + done，提前结束
            # _emit_unsupported_product_type 内部发 TextDeltaEvent("暂不支持XX品类") + done
            async for event in _emit_unsupported_product_type(ctx, resolved.intent):
                yield event
            return

    # ═══════════════════════════════════════════════════════════════════════════
    # 对比请求但候选不足
    # ═══════════════════════════════════════════════════════════════════════════
    # 用户说了"对比"但意图是 recommend（说明还没有可对比的商品）
    # 且消息中包含对比短语（"对比一下"） → 提示用户先浏览商品
    if resolved.intent.intent == "recommend" and is_compare_phrase(pipeline_body.message):
        async for event in _emit_compare_insufficient(ctx):
            yield event
        return

    # ═══════════════════════════════════════════════════════════════════════════
    # 阶段 3：槽位检查
    # ═══════════════════════════════════════════════════════════════════════════
    # skip_slot_check = True 时跳过（场景化请求/已明确约束等情况）
    if not resolved.skip_slot_check:
        # _missing_slots: 返回缺失的必填槽位列表，主要是 category（品类）
        missing_slots = _missing_slots(resolved.body, resolved.intent)
        if missing_slots:
            # 有缺失槽位 → 发澄清问句（ClarificationEvent），让用户补充信息
            async for event in _emit_clarification(ctx, resolved.body, resolved.intent, missing_slots):
                yield event
            return  # 发完问句本轮结束，等用户回复后下一轮继续

    # ═══════════════════════════════════════════════════════════════════════════
    # 阶段 4：意图分发 ★ 从这里开始真正干活
    # ═══════════════════════════════════════════════════════════════════════════
    t_handler_start = time.perf_counter()
    first_product_card_logged = False  # 标记是否已经记录了第一张商品卡的到达时间

    # _dispatch_intent_handler: 从 INTENT_HANDLERS 字典取对应的 handler 函数，调它
    async for event in _dispatch_intent_handler(ctx, resolved.body, resolved.intent):
        # 性能监控：记录第一张 product_card 的到达时间
        # getattr(event, "event", None) 安全获取 event 属性（如果没有则返回 None）
        if not first_product_card_logged and getattr(event, "event", None) == "product_card":
            first_product_card_logged = True
            # 打印端到端首屏延迟（从请求进入到第一张商品卡产出）
            logger.info("[timing] first_product_card=%.3fs", time.perf_counter() - t_total_start)
        yield event

    # 记录 handler 总耗时 + 整个 turn 的总耗时
    logger.info(
        "[timing] handler=%.3fs total=%.3fs",
        time.perf_counter() - t_handler_start,
        time.perf_counter() - t_total_start,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 阶段 1：图片预分析
# ═══════════════════════════════════════════════════════════════════════════════

async def _prepare_pipeline_body(
    ctx: StreamContext, body: ChatStreamRequest
) -> AsyncGenerator[SSEEventBase | StageResult[ChatStreamRequest], None]:
    # ↑ 返回类型：AsyncGenerator 的产出类型是 SSEEventBase | StageResult[ChatStreamRequest]
    #   用 | 表示"或"（Union 类型的新写法，Python 3.10+ 支持）
    #   这意味着这个 generator 产出的 item 要么是 SSE 事件，要么是 StageResult 包裹的 body
    """
    有图片时调 Qwen-VL-Plus 分析 → 把分析结果拼接到用户消息文本末尾。
    没有图片时什么都不做，直接透传 body。

    为什么图片分析要放在最前面（在意图识别之前）？
    图片分析结果中包含品类/品牌/肤质/成分等信息。
    先分析再识别，LLM 看到了图片上下文，意图识别的准确度会显著提升。

    例：
      用户上传护肤品照片 + 文字"这个适合敏感肌吗？"
      → Qwen-VL 返回: "这是XX品牌的控油洗面奶，适合油性肤质，含水杨酸"
      → message_with_image_context 拼接后:
        "这个适合敏感肌吗？[图片分析: XX品牌控油洗面奶，适合油性肤质，含水杨酸]"
      → 后续意图识别时，LLM 看到完整的上下文，能正确判断：
        category=美妆护肤, product_type=洗面奶, skin_type=油性, ingredient=水杨酸
    """
    if body.image_url:  # 只有带图片的请求才需要分析
        # ctx.thinking() 创建一个 ThinkingEvent
        #   参数："analyzing_image"=阶段标识(前端用来显示不同文案)
        #         msg.THINKING_ANALYZING_IMAGE=展示给用户的文字(如"正在分析图片...")
        yield ctx.thinking("analyzing_image", msg.THINKING_ANALYZING_IMAGE)

        image_analysis: dict[str, Any] | None = None  # 将持有 VL 模型返回的分析结果

        # ctx.ensure_active() 检查当前 turn 是否已被用户取消
        #   如果 cancel_token 已被 set → 抛出 StreamCancelled 异常 → 被 chat_stream 的 except 捕获
        ctx.ensure_active()

        # ★ run_with_heartbeat 是一个包装函数，它做三件事：
        #   1. 启动被包装的协程（ctx.stages.run_multimodal(body.image_url)）
        #   2. 在等待期间每 0.8 秒自动 yield 一个 thinking 心跳事件（前端不卡死）
        #   3. 监听 cancel_token，用户取消时立即中断
        #
        # ctx.stages.run_multimodal(body.image_url) 最终调用：
        #   stages/multimodal.py 的 run_multimodal() → llm_client.analyze_image()
        async for image_item in run_with_heartbeat(
            ctx,
            ctx.stages.run_multimodal(body.image_url),  # 要执行的协程
            "analyzing_image",                            # 阶段标识
            msg.THINKING_ANALYZING_IMAGE,                 # 心跳文案
            timing_key="image_analysis",                  # 计时键（用于 stage_timings_ms）
        ):
            if isinstance(image_item, StageResult):
                # StageResult 表示阶段完成了，value 携带结果
                image_analysis = image_item.value  # VL 返回的 dict，包含类别/品牌/肤质等分析
            else:
                # 不是 StageResult 就是 ThinkingEvent（心跳事件）
                yield image_item  # 直接发给客户端

        # 把 VL 分析结果注入到消息体中
        # Pydantic 的 model_copy(update={...}) 方法：
        #   创建原对象的一个新副本，同时更新指定字段。原对象不被修改。
        # message_with_image_context 函数把原始消息和图片分析结果拼接成新的消息文本
        body = body.model_copy(update={
            "message": message_with_image_context(body.message, image_analysis)
        })

    # StageResult(body) 包装阶段结果 —— 告诉调用方"这个阶段完成了，这是产出"
    yield StageResult(body)


# ═══════════════════════════════════════════════════════════════════════════════
# ★ 阶段 2：意图识别 —— 铁律 3 的现场演示
#
# 铁律 3："路由逻辑必须代码化，Prompt 只能做语言理解/生成"
#
# 设计原理：80%+ 的用户消息是固定模式（"推荐XX"、"加购物车"、"换一组"）。
# 这些模式用几句正则/关键词匹配就能准确识别，不需要 LLM。
# LLM 只在真正模糊的输入（"我想要那种……你懂的"）时才上场。
#
# 8 条确定性规则按优先级从高到低排列：
#   规则 1: maybe_intercept_budget_patch  → "再便宜点/预算设定为200"
#   规则 2: converge 信号                  → 前端"帮我选一个"
#   规则 3: maybe_checkout_intent         → "结算/买单"
#   规则 4: maybe_cart_intent             → "加入购物车/放入购物篮"
#   规则 5: maybe_shopping_intent         → "推荐/帮我选/有什么好用的"
#   规则 6: is_replace_deck_phrase        → "换一组/换一批"
#   规则 7: is_compare_phrase + 上下文     → "对比第一个和第三个"
#   兜底:   run_intent()                   → ★ 调 LLM analyze_intent()
#
# 为什么是 7 条（不是 8 条）？
# 规则 3/4/5 在一个 if 块中用 or 串联检测，因为它们互斥（一条消息不可能
# 同时是"结算"和"加购物车"），可以一次性检测。
#
# 规则 1 排在第一是因为它可能修改 body（补 budget patch），
# 后续规则需要基于修改后的 body 做判断。
# ═══════════════════════════════════════════════════════════════════════════════

async def _resolve_intent(
    ctx: StreamContext, pipeline_body: ChatStreamRequest
) -> AsyncGenerator[SSEEventBase | StageResult[_ResolvedIntent], None]:
    """
    意图识别 —— 规则优先，LLM 兜底。

    参数：
        ctx:           流上下文
        pipeline_body: 经过图片预分析处理后的请求体

    产出：
        StageResult(_ResolvedIntent) —— 确定的意图 + 是否跳过槽位检查
    """

    # ── 规则 1：预算修改拦截 ────────────────────────────────────────────────────
    # "再便宜一点 / 预算设定为 200 / 价格降到 300 以内"
    # maybe_intercept_budget_patch 内部逻辑：
    #   1. 检测用户消息是否包含预算修改意图
    #   2. 提取数字（"200"→200, "以内"→budget_max）
    #   3. 检查上一轮是否有购买标准可以修改
    #   4. 返回 (修改后的 body, continue 意图的 IntentResult) 或 (原 body, None)
    # 这个函数可能同时返回两个值（元组解包），synthetic_intent 非 None 表示规则命中
    pipeline_body, synthetic_intent = await maybe_intercept_budget_patch(ctx.session_id, pipeline_body)

    # ── 规则 2：前端收敛信号 ────────────────────────────────────────────────────
    # converge=true 是前端在用户多轮缩小范围后设置的标志
    # 含义："信息够了，直接决策，不要再推荐更多商品让我选了"
    # synthetic_intent is None 表示规则 1 没命中，继续尝试规则 2
    if synthetic_intent is None and pipeline_body.converge:
        # 从 DB 取上一轮的购买标准（品类至少不能丢）
        prev = await get_previous_criteria(ctx.session_id)
        synthetic_intent = IntentResult(
            intent="continue",       # continue 意图 → handler 复用推荐流（不重新生成标准）
            confidence=1.0,          # 确定性规则，置信度 100%（不是 LLM 的概率值）
            category=prev.category or None if prev else None,  # 继承上一轮的品类
        )

    # ── 规则 3/4/5：购物意图确定性检测 ───────────────────────────────────────────
    # 三个检测函数内部各自维护了中文购物相关的关键词/短语列表，
    # 通过正则和字典匹配。匹配成功返回 IntentResult，失败返回 None。
    #
    # 跳过条件（不进规则，交给 LLM）：
    #   - 有 image_url：图片场景比较复杂，LLM 更擅长
    #   - 有 criteria_patch：用户在有标准卡片的基础上修改，涉及槽位变更
    if synthetic_intent is None and not pipeline_body.image_url and not pipeline_body.criteria_patch:
        # Python 的 or 短路求值：从左到右依次执行，第一个非 None 结果被赋值
        determined = (
            maybe_checkout_intent(pipeline_body.message)     # "结算/买单/去付款" → checkout_confirm
            or maybe_cart_intent(pipeline_body.message)      # "加入购物车/放进去" → add_to_cart
            or maybe_shopping_intent(pipeline_body.message)  # "推荐/有什么好的" → recommend
        )
        # 防御性检查：如果三个规则中某个命中了 checkout_confirm，
        # 但购物车实际是空的 → 用户可能在说"结算"但从未加购过 → 退回 shopping 意图
        if determined is not None and determined.intent == "checkout_confirm":
            cart = await get_session_cart(ctx.session_id)  # 查购物车状态
            if cart.total_items == 0:  # 购物车为空
                # 重新用 shopping 意图的模式去匹配
                determined = maybe_shopping_intent(pipeline_body.message)
        if determined is not None:
            synthetic_intent = determined  # 规则命中，跳过 LLM

    # ── ★ 规则与 LLM 协作：加购中的商品名提取 ───────────────────────────────────
    # 这是规则和 LLM "分工合作"的经典案例：
    #
    # 规则能做的事：识别"加购物车" → add_to_cart（意图分类）
    # 规则不能做的事：提取"理肤泉B5修复霜" → product_id（实体链接）
    #
    # 当用户说"把理肤泉的加进去"而不是"把第一个加进去"时：
    #   规则：intent=add_to_cart, target_product_id=None, target_product_name=None
    #   序数检测：ordinal_index("把理肤泉的加进去") → None（没有"第一个/第二个"）
    #   指代检测：message_refers_to_previous_product("把理肤泉的加进去") → False（没有"那个/这个"）
    #   → 规则无法解析目标商品 → 故意把 synthetic_intent 清掉 → 让 LLM 提取商品名
    #
    # LLM 收到"把理肤泉的加进去"→ 返回 IntentResult(target_product_name="理肤泉")
    # 后处理 1 中把 intent 恢复为 add_to_cart → 后续 handler 根据商品名去检索匹配
    _force_cart_intent = False  # 标记：是否需要在 LLM 之后恢复 add_to_cart 意图
    if (
        synthetic_intent is not None                              # 规则命中了
        and synthetic_intent.intent == "add_to_cart"              # 识别为加购意图
        and not synthetic_intent.target_product_id                # 但没有目标商品 ID
    ):
        # ordinal_index 从消息中提取序数："第一个"→0, "第二个"→1, None→没找到
        has_ordinal = ordinal_index(pipeline_body.message) is not None

        # 检查是否有一轮推荐过商品（"第一个"只能在有历史商品时才有意义）
        has_previous = bool(await get_previous_product_ids(ctx.session_id))

        # 检查用户是否用代词指代了之前的商品（"那个/这个/它/这个商品"）
        refers_to_previous = message_refers_to_previous_product(pipeline_body.message)

        if not has_ordinal and not (has_previous and refers_to_previous):
            # 既不是序数指代，也不是代词指代 → 需要 LLM 提取商品名
            _force_cart_intent = True     # 标记：LLM 跑完后要恢复 add_to_cart
            synthetic_intent = None       # 清掉规则结果，走 LLM 流程

    # ── 规则 6：换一组 ──────────────────────────────────────────────────────────
    # "换一组 / 换一批 / 再看看其他的 / 不太喜欢这些"
    # 规则命中后复用上一轮的购买标准，但排除已展示过的商品
    # 这样用户能看到一批新的推荐结果，而不是同样的商品
    if synthetic_intent is None and is_replace_deck_phrase(pipeline_body.message):
        prev = await get_previous_criteria(ctx.session_id)  # 取上一轮的标准
        if prev is not None:
            # 深拷贝上一轮的 constraints，只保留有意义的字段
            # has_context_value 过滤掉 None/空字符串/空列表
            constraints = {
                key: value for key, value in prev.constraints.model_dump().items()
                if has_context_value(value)
            }
            synthetic_intent = IntentResult(
                intent="recommend",
                category=prev.category or None,
                extracted_constraints=constraints,
            )
        else:
            # 没有历史标准 → 只能给一个空壳 recommend 意图
            synthetic_intent = IntentResult(intent="recommend")

    # ── 规则 7：对比 ────────────────────────────────────────────────────────────
    if synthetic_intent is None:
        # 路径 A：前端直接把 compare_product_ids 放在了请求体里
        # 场景：用户在 UI 上手动选中了两个商品，点击"对比"按钮
        client_ids = pipeline_body.compare_product_ids or []
        if len(client_ids) >= 2:
            prev = await get_previous_criteria(ctx.session_id)
            synthetic_intent = IntentResult(
                intent="compare",
                category=prev.category or None if prev else None,
                compare_product_ids=client_ids[:4],  # 最多对比 4 个，防止 token 爆炸
            )
        # 路径 B：用户通过文本消息表达对比意图
        elif is_compare_phrase(pipeline_body.message):
            # "对比第一个和第三个" / "帮我比较一下第一个和第二个"
            # resolve_compare_targets: 从消息文本中解析序数指代（"第一个"→previous_ids[0]）
            previous_ids = await get_previous_product_ids(ctx.session_id)  # 取上一轮推荐的商品列表
            if previous_ids:
                resolved = resolve_compare_targets(pipeline_body.message, previous_ids)
                if len(resolved) >= 2:  # 至少能解析出 2 个商品才触发对比
                    prev = await get_previous_criteria(ctx.session_id)
                    synthetic_intent = IntentResult(
                        intent="compare",
                        category=prev.category or None if prev else None,
                        compare_product_ids=resolved,
                    )

    # ── ★ 分支点：7 条规则的结果 ────────────────────────────────────────────────
    if synthetic_intent is not None:
        # 规则命中了 → 直接使用规则结果（0 次 LLM 调用，极快）
        intent = synthetic_intent
    else:
        # 7 条规则全没命中 → LLM 上场
        # ctx.stages.run_intent 最终调用 stages/intent.py → llm_client.analyze_intent()
        # 这个 LLM 调用会：
        #   1. 加载 backend/prompts/intent.md 作为 system prompt
        #   2. 把用户消息 + 对话历史拼成 messages
        #   3. POST 到百炼/Doubao API
        #   4. 解析返回的 JSON，构造 IntentResult

        # 先发一个 thinking 事件给前端（"正在理解您的需求..."）
        yield ctx.thinking("understanding", msg.THINKING_UNDERSTANDING)

        intent = None
        # 再次检查取消信号（因为上面的 yield 交出了 CPU 控制权，用户可能在此期间取消了）
        ctx.ensure_active()

        # run_with_heartbeat: 在等 LLM 响应时每 0.8s 发 thinking 心跳
        async for intent_item in run_with_heartbeat(
            ctx,
            ctx.stages.run_intent(ctx.session_id, pipeline_body),  # 调意图识别 LLM
            "understanding",
            msg.THINKING_UNDERSTANDING,
            timing_key="intent",
        ):
            if isinstance(intent_item, StageResult):
                intent = intent_item.value       # LLM 返回的 IntentResult
            else:
                yield intent_item                 # 心跳 ThinkingEvent

        if intent is None:
            raise RuntimeError("intent stage completed without a result.")

    # ── 后处理 1：force cart intent（恢复加购意图）──────────────────────────────
    # 规则阶段故意把 synthetic_intent 清掉让 LLM 提取商品名。
    # LLM 现在跑完了（它可能识别为 recommend 或 clarify，因为不知道我们的加购规则），
    # 但我们需要它的 target_product_name（商品名提取结果）。
    # 把 intent 强制改回 add_to_cart，保留 LLM 提取的商品名。
    # Pydantic 的 model_copy(update=...) 创建新对象并更新字段，原对象不变。
    if _force_cart_intent:
        intent = intent.model_copy(update={
            "intent": "add_to_cart",
            "target_product_name": intent.target_product_name,  # LLM 提取的商品名
        })

    # ── 后处理 2：约束精炼 ──────────────────────────────────────────────────────
    # resolve_intent_constraints 对 LLM 返回的约束做确定性后处理：
    #   - 品牌别名 → 规范品牌名（"小黑瓶"→"兰蔻"，通过 domain_terms 映射表）
    #   - 品类别名 → 标准品类名（"洗面"→"洗面奶"）
    #   - 跨品类 product_type 归一化
    # 跳过场景："换一组"消息不需要精炼约束（它用的是上一轮的标准，不是 LLM 生成的）
    if not is_replace_deck_phrase(pipeline_body.message):
        intent = resolve_intent_constraints(intent, pipeline_body.message)

    # ── 后处理 3：无法执行的 add_to_cart → 降级为 recommend ─────────────────────
    # 场景：用户说"加购物车"，但之前没推荐过任何商品，也不知道要加什么
    # → 没东西可加，把这个意图降级为 recommend（相当于用户说"推荐点什么"）
    skip_slot_check = False  # 默认不跳过槽位检查
    if intent.intent == "add_to_cart" and not intent.target_product_id:
        previous_ids = await get_previous_product_ids(ctx.session_id)
        if not previous_ids:  # 上一轮没有推荐过商品
            logger.info("Reclassified add_to_cart → recommend (no product reference)")
            intent = intent.model_copy(update={"intent": "recommend"})
            skip_slot_check = True  # 降级后跳过槽位检查（避免不必要的反问）

    # ── 后处理 4：场景化请求跳过槽位检查 ─────────────────────────────────────────
    # 送礼/旅行/兴趣探索 类请求不需要指定品类（品类由场景策略决定）
    if intent.intent in {"recommend", "clarify"} and not skip_slot_check:
        from src.services.shopping_strategy import is_likely_shopping_strategy_request
        # is_likely_shopping_strategy_request: 检测是否是场景化选购请求（"送妈妈的礼物""旅行必备"）
        if is_likely_shopping_strategy_request(pipeline_body, intent):
            logger.info("Skipping slot check for scenario-based request")
            skip_slot_check = True

    # ── 后处理 5：对比意图的 ID 解析与降级 ───────────────────────────────────────
    # 把混合格式的 ID 统一解析为确定的 product_id 列表。
    # resolve_compare_ids_mixed 支持三种格式：
    #   - 序数："第一个" → previous_ids[0]
    #   - 数字：1 → "p_beauty_001"
    #   - product_id：直接使用
    # 解析后少于 2 个 → 无法对比 → 降级为 recommend
    if intent.intent == "compare":
        previous_ids = await get_previous_product_ids(ctx.session_id)
        client_ids = pipeline_body.compare_product_ids or []  # 前端传的
        llm_ids = intent.compare_product_ids or []             # LLM 返回的
        # 优先用前端传的（更准确），其次用 LLM 的
        combined_ids = client_ids if client_ids else llm_ids

        if len(previous_ids) >= 2 and combined_ids:
            resolved = resolve_compare_ids_mixed(combined_ids, pipeline_body.message, previous_ids)
            if len(resolved) >= 2:
                intent = intent.model_copy(update={"compare_product_ids": resolved})
            else:
                intent = intent.model_copy(update={"intent": "recommend"})  # 解析后不够 2 个 → 降级
        elif len(previous_ids) < 2:
            intent = intent.model_copy(update={"intent": "recommend"})  # 没历史商品 → 降级
        else:
            intent = intent.model_copy(update={"intent": "recommend"})  # 其他异常 → 降级

    # 产出阶段结果：_ResolvedIntent 包装了 body + intent + skip_slot_check
    yield StageResult(_ResolvedIntent(body=pipeline_body, intent=intent, skip_slot_check=skip_slot_check))


# ═══════════════════════════════════════════════════════════════════════════════
# 阶段 3：槽位检查
# ═══════════════════════════════════════════════════════════════════════════════

def _missing_slots(pipeline_body: ChatStreamRequest, intent: IntentResult) -> list[str]:
    # ↑ 这是一个普通函数（非 async），因为 check_required_slots 是纯代码规则，不涉及 I/O
    """
    检查是否缺少必填槽位（目前主要是 category 品类）。

    返回：缺失槽位的列表，如 ["category"]（前端会据此决定问什么问题）
    返回空列表 [] 表示槽位齐全。

    三种情况跳过检查：
    - 对比意图：对比的是已有商品，品类在上一轮已经确定
    - criteria_patch：用户在有标准卡片的基础上修改参数，品类已知
    - 有图片：图片分析结果中可能已包含品类信息，不需要再问
    """
    if intent.intent == "compare":
        return []  # 对比不需要品类
    if pipeline_body.criteria_patch or pipeline_body.image_url:
        return []  # 有标准补丁或图片 → 品类信息可能已包含
    # check_required_slots 在 slot_checker.py 中定义，纯代码规则（不调 LLM）
    return check_required_slots(pipeline_body.message, intent)


async def _emit_clarification(
    ctx: StreamContext, pipeline_body: ChatStreamRequest,
    intent: IntentResult, missing_slots: list[str],
) -> AsyncGenerator[SSEEventBase, None]:
    """
    反问用户缺失的槽位信息 + 同时跑投机检索。

    投机检索是什么？
    趁用户还没回复澄清问句的时间，用当前已收集到的部分标准
    （如只知道"美妆护肤"但不知道具体子品类）提前检索一批候选商品。
    用户回复后下一轮可以直接用这批候选，加速响应。

    即使投机检索失败（DB 挂了/检索超时），也不影响主流程的澄清问句发出。
    """
    # 从 IntentResult 快速构造一个初步的 CriteriaPayload
    # criteria_from_intent 在 stages/criteria.py 中定义
    partial = _intent_to_partial_criteria(intent, pipeline_body.message)

    # 如果品类已知（即使用户还缺其他槽位），先发一张部分标准卡片
    # 让用户看到系统已经理解了什么（"哦，系统知道我要美妆护肤类的"）
    if _should_emit_partial_criteria(missing_slots, partial):
        # _criteria_card_event 构造 CriteriaCardEvent 对象
        yield _criteria_card_event(ctx, partial)

    # ★ 投机检索：用部分标准预检索候选商品
    product_ids: list[str] = []
    try:
        # get_feedback_context 获取用户在当前会话中的反选/排除记录
        feedback = await get_feedback_context(ctx.session_id)
        # run_retrieval 调 retriever.retrieve_with_evidence()
        retrieval = await run_retrieval(partial, top_n=5, feedback=feedback)
        # 提取商品 ID 列表（用于下一轮的序数指代解析）
        product_ids = [p.product_id for p in (retrieval.products or [])]
    except Exception:
        # 投机检索失败 → 静默跳过（不打乱主流程）
        logger.warning("Pre-retrieval in clarification failed", exc_info=True)

    # handle_clarification 在 handlers.py 中定义
    # 它根据 missing_slots 构造对应的 ClarificationEvent（问句 + suggested_options）
    async for event in handle_clarification(ctx, missing_slots, intent.category):
        yield event

    # 保存本轮的部分标准和候选商品到 DB（供下一轮 _merge_followup_context 使用）
    await save_recommendation_turn(ctx.session_id, partial, product_ids, user_message=pipeline_body.message)


# ═══════════════════════════════════════════════════════════════════════════════
# 阶段 4：意图分发 —— 根据意图选择对应的 handler
# ═══════════════════════════════════════════════════════════════════════════════

async def _dispatch_intent_handler(
    ctx: StreamContext, pipeline_body: ChatStreamRequest, intent: IntentResult,
) -> AsyncGenerator[SSEEventBase, None]:
    """
    从 INTENT_HANDLERS 注册表取对应 handler → 调它 → 转发事件。

    INTENT_HANDLERS 在 handlers.py 中定义，是一个 dict[str, callable]：
        意图                 handler                           产出的 SSE 事件序列
        ─────────────────── ───────────────────────────────── ────────────────────────────────────────────
        "recommend"          handle_recommend()                thinking→criteria_card→text_delta*→product_card*→final_decision→done
        "clarify"            handle_clarification()            clarification→done
        "add_to_cart"        handle_add_to_cart()              cart_action(add)→done
        "view_cart"          handle_view_cart()                cart_action(detail)→done
        "remove_from_cart"   handle_remove_from_cart()         cart_action(remove)→done
        "update_cart_qty"    handle_update_cart_quantity()     cart_action(update)→done
        "checkout_preview"   handle_checkout_preview()         text_delta→done
        "checkout_confirm"   handle_checkout_confirm()         text_delta→done
        "checkout_cancel"    handle_checkout_cancel()          text_delta→done
        "compare"            handle_compare()                  compare_card→done
        "continue"           handle_recommend()                ← 复用推荐事件流
        "feedback"           handle_recommend()                ← 复用推荐事件流（用户给反馈后重新推荐）

    pipeline 不关心每个 handler 的内部实现（criteria → retrieval → recommendation → decision）。
    它只负责：取 handler → 调它 → 转发事件。
    这符合"编排层只做调度，不做业务"的原则。
    """

    # dict.get(key) 方法：如果 key 存在返回 value，不存在返回 None（不会抛 KeyError）
    handler = INTENT_HANDLERS.get(intent.intent)
    if handler is None:
        # 理论上不会走到这里。如果走到了 → 意图被识别但没有注册对应的 handler
        logger.warning("Unhandled intent '%s' — falling back to clarification.", intent.intent)
        # fallback 到澄清流程（至少不会崩溃）
        async for event in handle_clarification(ctx, ["category"]):
            yield event
        return

    # 调 handler，逐个转发它产出的事件
    # handler 签名：async def handler(ctx, pipeline_body, intent) -> AsyncGenerator[SSEEventBase, None]
    async for event in handler(ctx, pipeline_body, intent):
        yield event  # 透传给上一层（_run_chat_turn）


# ═══════════════════════════════════════════════════════════════════════════════
# ★ 多轮上下文合并 —— 多轮对话"记住上下文"的核心机制
#
# 例：
#   第 1 轮："推荐适合油皮的洗面奶"
#       → 系统保存：constraints = {skin_type:"油性", product_type:"洗面奶"}
#   第 2 轮："预算控制在 200 以内"  ← 没有重提肤质
#       → 合并上一轮：constraints = {skin_type:"油性", product_type:"洗面奶", budget_max:200}
#       → 用户不需要每轮都重新说"我是油皮"
#
# 话题切换检测：
#   第 1 轮："推荐适合油皮的洗面奶"
#   第 2 轮："推荐手机"  ← 完全不同的品类
#   → 检测到话题切换：不继承护肤品相关的品牌/产品约束
#   → 但通用约束（如预算）可以保留
# ═══════════════════════════════════════════════════════════════════════════════

async def _merge_followup_context(session_id: str, resolved: _ResolvedIntent) -> _ResolvedIntent:
    """
    把上一轮的购买标准继承到本轮。

    合并规则：
    - clarify/continue/compare 意图：总是继承（补充信息类场景）
    - recommend 意图 + 未指定新品类：继承（延续同品类推荐）
    - recommend 意图 + 指定了新品类：不继承（视为全新搜索）
    - 话题切换时：通用约束（预算/肤质/场景）继承，品牌/产品不继承
    - 本轮已设置的字段不覆盖（用户刚说的优先）
    """

    intent = resolved.intent
    # _should_merge_previous_context 判断是否应该合并
    if not _should_merge_previous_context(intent):
        return resolved  # 不合并，直接返回原值

    # get_previous_criteria 从 DB（conversations 表）取上一轮保存的购买标准
    previous = await get_previous_criteria(session_id)
    if previous is None:
        return resolved  # 没有历史标准，不需要合并

    # 把本轮 LLM/规则提取的约束转为 dict（方便后续逐个字段处理）
    constraints = dict(intent.extracted_constraints or {})

    # 检测话题是否切换了（洗面奶 → 手机？）
    topic_switched = _is_topic_switch(intent, previous)

    # 遍历上一轮的所有约束字段
    # model_dump() 把 Pydantic model 转为 dict，如 {"budget_max": 200, "skin_type": "油性", ...}
    for key, value in previous.constraints.model_dump().items():
        # 条件 1：本轮还没有设置这个字段（key not in constraints）
        #   为什么？本轮设置的优先，"用户刚说的"比"历史记住的"重要
        # 条件 2：上一轮的这个值有意义（非空/非 None/非空列表）
        if key not in constraints and has_context_value(value):
            # 话题切换时：品牌偏好/排除/产品类型不继承
            #   用户从"洗面奶"切换到"手机"，不应该把护肤品的品牌偏好带过去
            #   但预算/肤质等通用条件仍然可以保留
            if topic_switched and key in ("brand_prefer", "brand_avoid", "product_type"):
                continue  # 跳过这个字段，不继承
            constraints[key] = value  # 继承！

    # 品类优先用本轮指定的，其次用上一轮的
    category = intent.category or previous.category or None

    # 返回更新后的 _ResolvedIntent
    return _ResolvedIntent(
        body=resolved.body,
        intent=intent.model_copy(update={"category": category, "extracted_constraints": constraints}),
    )


def _is_topic_switch(intent: IntentResult, previous: CriteriaPayload) -> bool:
    """
    检测用户是否切换了话题。

    两种情况判定为话题切换：
    1. product_type 明确变了：
       "洗面奶"（经 normalize_product_type 后）≠ "手机"（经 normalize_product_type 后）
    2. 本轮指定了新品牌但没有指定 product_type：
       "推荐 Nike"（之前聊的护肤品）→ 品牌暗示了新的产品类别，按话题切换处理
    """
    # 提取本轮的产品类型
    current_pt = (intent.extracted_constraints or {}).get("product_type")
    # 提取上一轮的产品类型
    previous_pt = previous.constraints.product_type

    # 情况 1：两轮都有明确的 product_type，比较是否一致
    if current_pt and previous_pt:
        from src.config.domain_terms import normalize_product_type
        # normalize_product_type 把同义表达归一化（"洁面乳"→"洗面奶"）
        return normalize_product_type(current_pt) != normalize_product_type(previous_pt)

    # 情况 2：本轮有品牌偏好但没有 product_type
    current_bp = (intent.extracted_constraints or {}).get("brand_prefer")
    if current_bp and not current_pt and previous_pt:
        return True  # "推荐Nike"在护肤品上下文中 → 话题切换

    return False


def _should_merge_previous_context(intent: IntentResult) -> bool:
    """
    判断本轮是否应该继承上一轮的购买标准。

    逻辑：
    - clarify/continue/compare 意图 → 总是合并（补充信息类）
    - recommend 意图 + 未指定新品类 → 合并（延续推荐）
    - recommend 意图 + 指定了新品类 → 不合并（新搜索）
    - 非 recommend 意图（加购/结算等）→ 不合并
    """
    if intent.intent in {"clarify", "continue", "compare"}:
        return True
    if intent.intent != "recommend":
        return False
    if intent.category:
        return False  # 用户指定了新品类 → 这是新搜索，不继承旧标准
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# 辅助函数（简单的工具函数和事件生成器）
# ═══════════════════════════════════════════════════════════════════════════════

def _unsupported_product_type(intent: IntentResult) -> bool:
    """
    检查是否是系统不支持的产品类型（商品库里没有的品类/子品类）。

    判断流程：
    1. 只检查 recommend/clarify 意图（加购/结算不需要检查品类）
    2. category 不在已知品类列表 → 不支持
    3. product_type 经品类归一化后在已知品类中 → 支持（如"手机"属于"数码电子"）
    4. product_type 不在 is_supported_product_type 的结果中 → 不支持
    """
    # 只有推荐和澄清意图才需要检查品类（加购/结算不需要）
    if intent.intent not in {"recommend", "clarify"}:
        return False

    # category 是系统定义的四大品类之一（美妆护肤/数码电子/服饰运动/食品生活）
    # 如果不在列表中 → 不支持
    if intent.category and intent.category not in KNOWN_CATEGORIES:
        return True

    # 提取 product_type（子品类，如"洗面奶""手机""跑鞋"）
    product_type = (intent.extracted_constraints or {}).get("product_type")
    # normalize_category 把子品类尝试归一化为大类（如果它本身就是一个品类名的话）
    if normalize_category(product_type) in KNOWN_CATEGORIES:
        return False  # product_type 实际是一个品类名 → 支持
    # is_supported_product_type 检查商品库中是否存在该产品类型
    return bool(product_type and not is_supported_product_type(str(product_type)))


def _should_emit_partial_criteria(missing_slots: list[str], criteria: CriteriaPayload) -> bool:
    """品类已知时才发部分标准卡片（否则卡片上没有有意义的信息可展示）"""
    # category 不在 missing_slots 中 = 品类已知
    # criteria.category 非空 = 确实有品类信息
    return "category" not in missing_slots and bool(criteria.category)


async def _emit_unsupported_product_type(
    ctx: StreamContext, intent: IntentResult
) -> AsyncGenerator[SSEEventBase, None]:
    """
    发出"暂不支持该产品类型"的提示文本 + done 事件。

    这个函数不调 LLM，不调检索，只发一个 TextDeltaEvent。
    为什么要独立成一个函数？
    因为逻辑简单但步骤固定（构造文本 → 发事件 → 发 done），
    抽出来避免 _run_chat_turn 中散落太多事件构造细节。
    """
    # 提取用户想要的产品类型（优先用 product_type，其次用 category）
    raw_pt = (
        (intent.extracted_constraints or {}).get("product_type")
        or intent.category
        or msg.UNSUPPORTED_PRODUCT_TYPE_FALLBACK  # fallback 文案（如"该品类"）
    )
    # 清理标点符号
    product_type = str(raw_pt).strip("，。！？；：、,.!?;:")

    # 构造 TextDeltaEvent：一次性发完（done=True），不需要流式
    yield TextDeltaEvent(
        session_id=ctx.session_id, turn_id=ctx.turn_id,
        seq=ctx.seq.next(),                          # 递增序号
        event_id=ctx.seq.event_id(),                 # 事件唯一 ID
        node_id=f"unsupported_{ctx.turn_id}",        # 前端节点 ID
        created_at_ms=now_ms(),                      # 时间戳
        message_id=f"unsupported_{ctx.turn_id}",     # 消息 ID（用于前端去重）
        delta=msg.UNSUPPORTED_PRODUCT_TYPE_TEMPLATE.format(product_type=product_type),  # 格式化文案模板
        done=True,                                   # 一次性发完
    )
    # 发 done 事件标记本轮结束
    yield ctx.done()


async def _emit_compare_insufficient(ctx: StreamContext) -> AsyncGenerator[SSEEventBase, None]:
    """对比请求但没有足够候选 → 提示用户先浏览商品，不调 LLM"""
    yield ctx.thinking("understanding", msg.THINKING_PROCESSING)

    # 构造 TextDeltaEvent：提示用户先浏览商品再对比
    yield TextDeltaEvent(
        session_id=ctx.session_id, turn_id=ctx.turn_id,
        seq=ctx.seq.next(), event_id=ctx.seq.event_id(),
        node_id=f"compare_insufficient_{ctx.turn_id}", created_at_ms=now_ms(),
        message_id=f"compare_insufficient_{ctx.turn_id}",
        delta=msg.COMPARE_INSUFFICIENT,  # 预定义文案（在 config/user_messages.py 中）
        done=True,
    )
    yield ctx.done()


def _current_stages() -> PipelineStages:
    """
    组装 5 个阶段函数为 PipelineStages 实例。

    用 module 属性引用（如 run_intent）而不是在这里直接写 lambda，
    好处是单元测试可以 monkeypatch 模块属性（如 pipeline.run_intent = mock_func），
    所有调用自动走 mock。
    """
    return PipelineStages(
        run_multimodal=run_multimodal,              # → stages/multimodal.py → llm_client.analyze_image()
        run_image_embedding=run_image_embedding,    # → stages/multimodal.py → embedding.embed_image()
        run_intent=run_intent,                      # → stages/intent.py → llm_client.analyze_intent()
        run_criteria=run_criteria,                  # → stages/criteria.py → llm_client.generate_criteria()
        run_retrieval=run_retrieval,                # → stages/recommendation.py → retriever.retrieve_with_evidence()
        run_recommendation_text=run_recommendation_text,              # → llm_client.generate_recommendation()
        run_recommendation_text_stream=run_recommendation_text_stream,  # → llm_client.stream_recommendation()
        run_decision=run_decision,                  # → stages/decision.py → llm_client.generate_decision()
    )


async def _emit_commercial_claim_reply(ctx: StreamContext) -> AsyncGenerator[SSEEventBase, None]:
    """
    ★ 商业声明拦截：固定话术回复，不调 LLM。

    为什么这样做？
    "有货吗/包邮吗/支持优惠券吗" —— LLM 不知道真实的库存和物流状态。
    如果让它回答，它会根据训练数据中的电商对话模式编造答案
    （"有货的，一般 3-5 天到货"）——这是纯粹的幻觉，会对用户造成误导。

    所以我们直接返回预定义的固定文案
    （"建议点击商品查看详情页了解库存和配送信息"），
    不做任何 LLM 调用，不做检索。
    """
    # 发 thinking 事件
    yield ctx.thinking("understanding", msg.THINKING_PROCESSING)

    # 构造 TextDeltaEvent：固定话术 + 一次性发完
    yield TextDeltaEvent(
        session_id=ctx.session_id, turn_id=ctx.turn_id,
        seq=ctx.seq.next(), event_id=ctx.seq.event_id(),
        node_id=f"ai_text_{ctx.turn_id}", created_at_ms=now_ms(),
        message_id=f"msg_{ctx.turn_id}",
        delta=COMMERCIAL_CLAIM_REPLY,  # 预定义的固定回复文案
        done=True,                     # 一次性发完，不需要流式
    )

    yield ctx.done()


def _intent_to_partial_criteria(intent: IntentResult, message: str) -> CriteriaPayload:
    """
    从 IntentResult 快速构造一个初步的 CriteriaPayload。

    这个"初步"标准用于澄清阶段——在用户还没有提供完整信息时，
    先把系统已经理解的部分（品类、已知约束）展示给用户。

    完整的购买标准生成在 stages/criteria.py 的 run_criteria() 中，
    那个版本会调 LLM 做 deep reasoning。
    """
    return criteria_from_intent(intent)  # → stages/criteria.py 的 criteria_from_intent 函数
