"""
HTTP 入口层 —— POST /chat/stream

职责：接受 HTTP 请求 → 交给 pipeline 处理 → 把事件流转发给客户端
不做任何业务逻辑。这是整个后端的"接线员"。

Java 等价物：
    @PostMapping("/stream")
    public Flux<ServerSentEvent<String>> streamChat(@RequestBody ChatStreamRequest body)

分层位置：API 层（最外层），只依赖 Runtime 层和 Types 层
调用链：Android OkHttp → POST /chat/stream → 本文件 → pipeline.chat_stream() → ... → SSE events 流回

学习要点：
  1. SSE 是怎么实现的？—— async generator + StreamingResponse
  2. 为什么 chat.py 只有 140 行？—— 业务逻辑全在 pipeline/service 里，这里只是"接线"
  3. 请求上下文的生命周期 —— request body → session_id/turn_id → 全局上下文 → SSE headers
"""

import hashlib  # 对 text_delta 做 SHA256 摘要，用于可观测性日志（取 delta 内容的 hash 避免存全量文本）
import uuid      # 生成 session_id / turn_id（当客户端没传时）
from typing import Any  # _extract_sse_event_fields 的返回类型

from fastapi import APIRouter                     # FastAPI 的路由注册器，相当于 Spring 的 @RestController
from fastapi.responses import StreamingResponse   # SSE 的基础 —— HTTP 长连接 + 流式 body

from src.runtime.pipeline import chat_stream        # 核心编排函数（下一站要读的文件）
                                                     # 这是 Runtime 层暴露给 API 层的唯一入口
from src.services.observability_llm import schedule_sse_event_recording  # 异步记录 SSE 事件到观测表（fire-and-forget）
from src.services.request_context import set_request_context, update_request_context  # 请求上下文（trace_id/session_id）的全局存储
from src.types.schemas import ChatStreamRequest      # HTTP 请求体的 Pydantic DTO（message, session_id, image_url, client_trace_id...）
from src.types.sse_events import SSEEventBase, format_sse  # SSEEventBase: 所有事件的基类；format_sse: Pydantic → "event:xxx\ndata:{json}\n\n"


# ═══════════════════════════════════════════════════════════════════════════════
# 路由注册
# APIRouter 相当于 Spring 的 @RestController 注解
# tags=["chat"] 让 Swagger UI 把 /chat/stream 归类到 "chat" 分组下
# ═══════════════════════════════════════════════════════════════════════════════
chat_router = APIRouter(tags=["chat"])


def _extract_sse_event_fields(event: SSEEventBase) -> dict[str, Any]:
    """
    从 SSE 事件中提取关键字段，用于写入 observability 数据库表。

    这是横切关注点（cross-cutting concern）—— 不影响主业务流程，
    只是"顺手"把关键信息记下来，方便后续调试和监控。

    为什么要单独写这个函数而不是直接序列化整个 event？
      - event 可能很大（product_card 包含完整商品信息 + 证据文本）
      - 只提取关键字段，减少数据库写入量和查询复杂度
      - 不同类型的 event 有不同关键字段（多态提取）

    Java 等价物：AOP aspect + DTO projection
    """
    # 所有事件共有的基本信息
    fields: dict = {
        "event_type": event.event,   # "thinking" / "product_card" / "text_delta" ...
        "seq": event.seq,            # 同一 turn 内的递增序号（用于排顺序、检测丢事件）
        "node_id": getattr(event, "node_id", None),   # 前端路由用的节点 ID
        "deck_id": getattr(event, "deck_id", None),   # 商品牌组 ID（同一批推荐的 product_card 共享 deck_id）
    }

    # ── 按事件类型提取特有字段 ──
    # 每个 getattr 都用 hasattr 先检查，因为不同类型的 event 字段不同
    # 这相当于 instanceof + 类型转换

    # criteria_card 事件：记录购买标准的 criteria_id
    # event.criteria 是 CriteriaPayload 对象，包含 criteria_id / category / constraints / chips
    criteria = getattr(event, "criteria", None)
    if criteria and hasattr(criteria, "criteria_id"):
        fields["criteria_id"] = criteria.criteria_id

    # product_card 事件：记录推荐了哪个商品
    # event.product 是 ProductPayload 对象
    product = getattr(event, "product", None)
    if product and hasattr(product, "product_id"):
        fields["product_ids"] = [product.product_id]

    # cart_action 事件：event.product_id 是直接挂在 event 上的字符串，不是嵌套对象
    # 注意：cart_action 的 product_id 路径和 product_card 不同！
    cart_product_id = getattr(event, "product_id", None)
    if cart_product_id and event.event == "cart_action":
        fields["product_ids"] = [cart_product_id]

    # final_decision 事件：记录最终推荐的"冠军"商品
    winner = getattr(event, "winner_product_id", None)
    if winner and event.event == "final_decision":
        fields["product_ids"] = [winner]

    # text_delta 事件：记录流式文本的 message_id + delta 内容预览
    # message_id 用于把属于同一段话的多个 delta 聚合在一起
    message_id = getattr(event, "message_id", None)
    if message_id:
        fields["message_id"] = message_id

    # delta 是增量文本片段（如 "为您" / "找到了" / "以下"）
    # 只存前 100 字符的预览 + SHA256 hash，不存全量（全量太长且已经在前端页面显示了）
    delta = getattr(event, "delta", None)
    if delta:
        fields["delta_preview"] = delta[:100]                    # 前 100 字符预览，够用了
        fields["delta_hash"] = hashlib.sha256(delta.encode()).hexdigest()  # SHA256 hash 用于精确去重/比对

    # done 事件：记录结束原因（completed / error / cancelled / awaiting_product_feedback...）
    finish_reason = getattr(event, "finish_reason", None)
    if finish_reason:
        fields["finish_reason"] = finish_reason

    return fields


# ═══════════════════════════════════════════════════════════════════════════════
# 主端点：POST /chat/stream
#
# 这是 Android 客户端调用的唯一聊天入口。
# OkHttp 发起 POST 请求，读 SSE 事件流。
# ═══════════════════════════════════════════════════════════════════════════════
@chat_router.post("/stream")
async def stream_chat(body: ChatStreamRequest):
    """
    接收用户消息（文字/图片），返回 SSE 事件流。

    参数说明：
      body: ChatStreamRequest（Pydantic model，FastAPI 自动校验 + 反序列化）
        - message: str                  用户输入的文本
        - session_id: str | None        会话 ID（多轮对话共享，不传则自动生成）
        - client_turn_id: str | None    客户端生成的 turn ID（不传则服务端生成）
        - image_source: str | None      图片 URL 或 Base64（拍照找货场景）
        - client_trace_id: str | None   全链路追踪 ID（调试/日志用）

    返回：
      StreamingResponse(media_type="text/event-stream")
      事件流格式：event: {type}\ndata: {json}\n\n
      前端/OkHttp 按 SSE 标准协议解析

    异常处理（不在本文件，在 pipeline.py 中）：
      - LLM 调用失败 → pipeline 发 error 事件 → 前端显示错误提示
      - 检索无结果 → 渐进式放松预算 → 仍无结果则发 clarification 事件
      - 图片理解失败 → 降级为纯文本模式
    """

    # ═══════════════════════════════════════════════════════════════════════════
    # 步骤 1：确定 session_id 和 turn_id
    #
    # session_id: 一次完整对话的标识（可能跨多个 turn）
    # turn_id:    单次用户发送 + 系统回复的标识
    #
    # 关系：一个 session 包含多个 turn
    #   session_abc:
    #     turn_001: "推荐油皮洗面奶"      → 系统返回 3 个商品
    #     turn_002: "第二个不错，详细介绍" → 系统返回详细解说
    #     turn_003: "加入购物车"          → 系统操作购物车
    #
    # uuid.uuid4().hex 生成不带横线的 32 位随机字符串
    # 如 "a1b2c3d4e5f6..." —— 纯字母数字，适合做 ID
    # ═══════════════════════════════════════════════════════════════════════════
    sid = body.session_id or f"sess_{uuid.uuid4().hex}"       # 优先用客户端传的，否则生成新的
    turn_id = body.client_turn_id or f"turn_{uuid.uuid4().hex[:8]}"  # turn_id 只取前 8 位，够用了

    # 如果客户端没带 client_turn_id，把服务端生成的回填到 body 里
    # model_copy(update=...) 是 Pydantic 的不可变更新方法 —— 返回新对象，不修改原对象
    # 这样后续 pipeline 拿到的 body 里一定包含 turn_id
    stream_body = body if body.client_turn_id else body.model_copy(update={"client_turn_id": turn_id})

    # 更新全局请求上下文（Thread-local / ContextVar）
    # 之后在 pipeline → service → retriever 的任何深度，
    # 都可以通过 get_request_context() 拿到 trace_id / session_id / turn_id，
    # 用于日志、审计、trace 记录。不需要函数参数层层传递。
    stream_context = update_request_context(
        trace_id=body.client_trace_id,  # 客户端传入的全链路追踪 ID
        session_id=sid,
        turn_id=turn_id,
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # 步骤 2：定义异步生成器（async generator）
    #
    # async generator 是 Python 3.6+ 的协程特性：
    #   - 用 async def + yield 定义
    #   - async for 消费它时，每次 yield 交出控制权，等消费者处理完再继续
    #   - 这就是 SSE 流式传输的底层机制 —— 不是一次性 return 全部数据，
    #     而是一边生成一边发送，客户端立即收到
    #
    # Java 对比：Flux<ServerSentEvent<String>> 的 emitter
    # Kotlin 对比：Flow<Event>.collect { emit(it) }
    # ═══════════════════════════════════════════════════════════════════════════
    async def event_generator():
        # 把请求上下文注入到当前协程的 ContextVar 中
        # 这样在这个 generator 内部的所有子调用都能读取 trace_id/session_id/turn_id
        set_request_context(stream_context)

        # ═══════════════════════════════════════════════════════════════════════
        # 核心调用！
        #
        # chat_stream(sid, stream_body) 是整个 RAG 链路的入口函数。
        # 它返回一个 AsyncGenerator[SSEEventBase] —— 逐个产出事件对象。
        #
        # chat_stream 内部做的事（下一站 pipeline.py 详细看）：
        #   1. 图片预分析（如果有图片 → Qwen-VL-Plus 理解图片内容）
        #   2. 意图识别（Qwen-Turbo: "用户想买什么？有什么条件？"）
        #   3. 槽位检查（信息够不够？不够就发 clarification 反问）
        #   4. 购买标准生成（结构化 Constraints）
        #   5. 投机检索（提前检索，不等 LLM 全生成完）
        #   6. 推荐文案生成（基于检索结果 + 证据）
        #   7. 最终决策
        #   8. 逐个产出 ThinkingEvent → CriteriaCardEvent → TextDeltaEvent →
        #      ProductCardEvent → FinalDecisionEvent → DoneEvent
        #
        # async for 的工作方式：
        #   pipeline 产出一个 event → 本循环体立即处理（写日志 + 序列化 + yield）
        #   → FastAPI 把序列化后的字符串发给 HTTP 客户端
        #   → 客户端立即收到这个事件，不等后续事件
        #   → 然后本循环继续等下一个 event
        #
        # 这种模式：延迟 ≈ 处理时间，不用等所有商品都检索完才开始发
        # ═══════════════════════════════════════════════════════════════════════
        async for event in chat_stream(sid, stream_body):
            # 每个事件产出后做两件事：

            # ① 记录到数据库（可观测性）
            # schedule_sse_event_recording 是异步调度函数，
            # 把写入任务提交到后台执行，不阻塞主流程。
            # 即使数据库写入失败，也不影响 SSE 事件正常发送给客户端。
            fields = _extract_sse_event_fields(event)
            schedule_sse_event_recording(**fields)

            # ② 序列化并发送
            # format_sse(event) 把 Pydantic 对象变成字符串：
            #   "event: product_card\n"
            #   "data: {\"event\":\"product_card\",\"product\":{...},\"reason\":\"...\"}\n"
            #   "\n"
            # 最后那个空行是 SSE 协议规定的分隔符
            yield format_sse(event)

    # ═══════════════════════════════════════════════════════════════════════════
    # 步骤 3：构建 StreamingResponse 并返回
    #
    # FastAPI 看到 StreamingResponse 后：
    #   1. 设置 HTTP 状态码为 200
    #   2. 设置 Content-Type: text/event-stream
    #   3. 开始迭代 event_generator()，每次 yield 就 flush 到 TCP socket
    #   4. 客户端通过 OkHttp SSE 解析器逐行读取
    #
    # headers 的作用：
    #   Cache-Control: no-cache     → 中间代理（CDN/反向代理）不要缓存，SSE 必须实时
    #   Connection: keep-alive      → HTTP/1.1 长连接，一个 turn 可能持续数秒
    #   X-Accel-Buffering: no       → 如果前面有 Nginx，告诉它不要缓冲 SSE 流（直通模式）
    #                                  没有这个 header，Nginx 会等积累了几 KB 才转发，
    #                                  导致用户看到卡顿
    #   X-Request-ID / X-Trace-ID   → 调试用，出问题时可以在日志中 grep 到对应请求的全链路
    # ═══════════════════════════════════════════════════════════════════════════
    return StreamingResponse(
        event_generator(),                    # 异步生成器 —— 数据源
        media_type="text/event-stream",       # SSE 协议标准 MIME 类型
                                              # 浏览器/OkHttp 识别这个类型后自动进入 SSE 解析模式
        headers={
            "Cache-Control": "no-cache",      # 禁用所有缓存（CDN / 浏览器 / 中间代理）
            "Connection": "keep-alive",       # HTTP 长连接（避免频繁 TCP 握手）
            "X-Accel-Buffering": "no",        # 禁用 Nginx 缓冲（关键！不加可能导致 5-10 秒的延迟）
            "X-Request-ID": stream_context.request_id,  # 请求唯一 ID，方便在日志系统中检索
            "X-Trace-ID": stream_context.trace_id or "", # 全链路追踪 ID（跨服务追踪）
        },
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 文件总结 —— 学完这个文件你应该能回答：
#
# 1. chat.py 做了哪些事？
#    → 三件事：① 生成 session_id/turn_id ② 调用 pipeline.chat_stream()
#       ③ 把 pipeline 产出的事件序列化成 SSE 文本并发送
#
# 2. chat.py 没做哪些事？
#    → 意图识别、检索、推荐生成、LLM 调用、数据库查询 —— 全部不在这一层
#    → 这一层的原则是"只接线，不干活"
#
# 3. async generator 为什么适合 SSE？
#    → yield 一次，客户端收到一次。不需要等所有数据都准备好，延迟极低
#
# 4. StreamingResponse 的 media_type 为什么是 "text/event-stream"？
#    → 这是 SSE 协议的规范，浏览器和 OkHttp 都按这个类型识别并启动 SSE 解析器
#
# 5. 错误处理去哪了？
#    → pipeline.py 内部 try/except，出错时 yield ErrorEvent，不会抛到这里
#    → 如果 pipeline 抛了未捕获的异常，FastAPI 的全局异常处理会兜底
# ═══════════════════════════════════════════════════════════════════════════════
