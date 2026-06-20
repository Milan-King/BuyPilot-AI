"""意图识别 Stage：连接 Runtime 编排层与 LLM Service 层。

本文件故意保持很薄：

- Pipeline 只依赖 ``run_intent`` 这个 task-oriented interface；
- 本 Stage 负责准备会话上下文；
- 真正的模型调用与 JSON 校验由 ``services.llm_client`` 完成。

Java 类比：它接近 Application Service/Facade，而不是 Controller。
"""

from __future__ import annotations

from src.services.conversation_state import get_conversation_summary
from src.services.llm_client import analyze_intent
from src.types.schemas import ChatStreamRequest, IntentResult


async def run_intent(session_id: str, body: ChatStreamRequest) -> IntentResult:
    """调用 LLM 完成语义意图识别。

    注意：只有 Pipeline 的确定性规则无法给出结论时才会调用这里。
    """
    # 服务端摘要来自 conversations 表，用来补充客户端 history 之外的多轮状态。
    ctx_summary = await get_conversation_summary(session_id)

    # 将 Pydantic MessageLite 转成普通 dict，作为 Prompt messages 的输入。
    # model_dump() 类似 Java DTO 序列化为 Map。
    return await analyze_intent(
        body.message,
        history=[item.model_dump() for item in body.history],
        # 图片场景允许模型结合图片 URL/分析信息判断用户意图。
        image_url=body.image_url,
        # 压缩后的服务端上下文控制 Prompt 长度，避免无限携带完整历史。
        conversation_context=ctx_summary,
    )
