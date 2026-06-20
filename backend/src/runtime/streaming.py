"""Small helpers shared by chat stream handlers."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Awaitable, Callable, Generic, Mapping, Protocol, TypeVar

from src.runtime.cancel_registry import CancellationToken
from src.runtime.stages.recommendation import RetrievalResult
from src.services.cancellation import is_chat_turn_cancellation_requested
from src.types.schemas import ChatStreamRequest, DecisionResult, IntentResult, RecommendationResult
from src.types.sse_events import (
    CriteriaPayload,
    DoneFinishReason,
    DoneEvent,
    EventSeq,
    EvidencePayload,
    ProductPayload,
    SSEEventBase,
    ThinkingEvent,
    now_ms,
)

T = TypeVar("T")


class RunRetrieval(Protocol):
    def __call__(
        self,
        criteria: CriteriaPayload,
        top_n: int = 5,
        feedback: Mapping[str, list[str]] | None = None,
        image_embedding: list[float] | None = None,
    ) -> Awaitable[RetrievalResult]: ...


class StageBundle(Protocol):
    @property
    def run_multimodal(self) -> Callable[[str | None], Awaitable[dict[str, Any] | None]]: ...

    @property
    def run_image_embedding(self) -> Callable[[str | None], Awaitable[list[float] | None]]: ...

    @property
    def run_intent(self) -> Callable[[str, ChatStreamRequest], Awaitable[IntentResult]]: ...

    @property
    def run_criteria(self) -> Callable[[str, ChatStreamRequest, IntentResult], Awaitable[CriteriaPayload]]: ...

    @property
    def run_retrieval(self) -> RunRetrieval: ...

    @property
    def run_recommendation_text(
        self,
    ) -> Callable[
        [CriteriaPayload, list[ProductPayload], dict[str, list[EvidencePayload]] | None],
        Awaitable[RecommendationResult],
    ]: ...

    @property
    def run_recommendation_text_stream(
        self,
    ) -> Callable[
        [CriteriaPayload, list[ProductPayload], dict[str, list[EvidencePayload]] | None],
        AsyncGenerator[str, None],
    ]: ...

    @property
    def run_decision(
        self,
    ) -> Callable[..., Awaitable[DecisionResult]]: ...


@dataclass(frozen=True)
class StageResult(Generic[T]):
    value: T


@dataclass(frozen=True)
class TimedTask(Generic[T]):
    task: asyncio.Future[T]
    started_at: float


@dataclass
class StreamContext:
    session_id: str
    turn_id: str
    deck_id: str
    seq: EventSeq
    cancel_token: CancellationToken
    stages: StageBundle
    heartbeat_interval_seconds: float
    stage_timings_ms: dict[str, float] = field(default_factory=dict)
    background_tasks: list[TimedTask[Any]] = field(default_factory=list)

    def ensure_active(self) -> None:
        # CancellationToken 内部保存当前 turn 是否已取消；已取消时会抛出 StreamCancelled。
        self.cancel_token.raise_if_cancelled()

    def thinking(self, stage: str, message: str) -> ThinkingEvent:
        # 创建事件前再次检查取消，避免取消后仍向客户端发送过期的 loading 状态。
        self.ensure_active()

        # 每个 thinking 都是标准 SSE 事件，包含所属会话、轮次、顺序和展示内容。
        return ThinkingEvent(
            session_id=self.session_id,
            turn_id=self.turn_id,
            # seq.next() 为同一 turn 分配递增序号，客户端可据此保持事件顺序。
            seq=self.seq.next(),
            event_id=self.seq.event_id(),
            # 同一 turn 的心跳使用相同 node_id，Android 会更新同一个 ThinkingNode。
            node_id=f"thinking_{self.turn_id}",
            created_at_ms=now_ms(),
            # stage 表示当前业务阶段；message 是展示给用户的状态文案。
            stage=stage,
            message=message,
        )

    def done(self, finish_reason: DoneFinishReason = "completed", deck_id: str | None = None) -> DoneEvent:
        return DoneEvent(
            session_id=self.session_id,
            turn_id=self.turn_id,
            seq=self.seq.next(),
            event_id=self.seq.event_id(),
            node_id=f"done_{self.turn_id}",
            deck_id=deck_id,
            created_at_ms=now_ms(),
            finish_reason=finish_reason,
        )


def start_stage_task(
    ctx: StreamContext,
    awaitable: Awaitable[T],
    timing_key: str | None = None,
    background: bool = False,
) -> TimedTask[T]:
    """
    启动一个阶段任务，并保存它的开始时间。

    ``awaitable`` 通常是尚未等待的协程对象；``ensure_future`` 会把它交给
    asyncio 事件循环调度，使心跳循环可以一边等待任务，一边产出事件。
    """
    # 已取消的 turn 不再启动新的 LLM、检索或图片分析任务。
    ctx.ensure_active()

    # 使用高精度单调时钟记录开始时间，只用于计算耗时。
    started_at = time.perf_counter()

    # 将协程包装为 Future/Task。调用后任务即可由事件循环调度执行。
    task = asyncio.ensure_future(awaitable)

    if timing_key:
        # 任务完成时自动记录一次耗时。setdefault 会防止后续重复覆盖。
        task.add_done_callback(lambda _: _record_stage_timing(ctx.stage_timings_ms, timing_key, started_at))

    # 将任务和开始时间放在同一个对象中，供心跳循环使用。
    timed_task = TimedTask(task=task, started_at=started_at)

    if background:
        # 后台任务不一定会立即等待，因此登记到上下文中，取消 turn 时统一清理。
        ctx.background_tasks.append(timed_task)

    return timed_task


async def run_with_heartbeat(
    ctx: StreamContext,
    awaitable: Awaitable[T],
    stage: str,
    message: str,
    timing_key: str | None = None,
) -> AsyncGenerator[SSEEventBase | StageResult[T], None]:
    """
    执行一个耗时的异步操作，并在等待期间定时产出 thinking 心跳。

    这个函数本身不直接得到最终结果，而是一个异步生成器：
    - 任务尚未完成时，可能多次 yield ThinkingEvent；
    - 任务完成时，yield 一次 StageResult[T]，然后结束。

    调用方通过 ``async for`` 同时接收这两类值。
    """
    # 把协程转换成已经开始调度的 asyncio Task，并记录开始时间。
    # 例如 awaitable 可能是一次 LLM 调用或商品检索。
    timed_task = start_stage_task(ctx, awaitable, timing_key=timing_key)

    # 具体的“等待 0.8 秒、检查结果、发送心跳”循环由下层函数负责。
    async for item in run_timed_task_with_heartbeat(
        ctx,
        timed_task,
        stage,
        message,
        timing_key=timing_key,
    ):
        # 当前函数只是把下层产生的 ThinkingEvent 或 StageResult 原样向上传递。
        yield item


async def run_timed_task_with_heartbeat(
    ctx: StreamContext,
    timed_task: TimedTask[T],
    stage: str,
    message: str,
    timing_key: str | None = None,
) -> AsyncGenerator[SSEEventBase | StageResult[T], None]:
    """
    等待一个已经启动的任务，并在每次等待超时后产出 thinking 事件。

    ``heartbeat_interval_seconds`` 不是任务超时时间，而是单次等待的最长时间。
    任务可以执行很多个心跳周期，直到完成、被取消或抛出异常。
    """
    # TimedTask 同时保存 asyncio Task 和任务的开始时间。
    task = timed_task.task

    try:
        # 每轮循环最多等待一个心跳周期，直到任务完成。
        while True:
            # 等待前检查一次取消状态，避免已取消的任务继续调用外部服务。
            await ensure_active(ctx)

            # asyncio.wait 会等待 task 完成，但最多只等一个心跳周期。
            # 如果任务提前完成（例如 0.3 秒），这里会立即返回；
            # 如果等待满 0.8 秒仍未完成，done 中不会包含 task。
            done, _ = await asyncio.wait({task}, timeout=ctx.heartbeat_interval_seconds)

            # 等待期间用户可能点击了停止，因此返回后再检查一次取消状态。
            await ensure_active(ctx)

            if task in done:
                # task 已经结束，不再发送心跳。
                if timing_key:
                    # 将该阶段耗时写入 ctx.stage_timings_ms，供日志和审计使用。
                    _record_stage_timing(ctx.stage_timings_ms, timing_key, timed_task.started_at)

                # task.result() 取出真正的业务结果。
                # StageResult 是 Runtime 内部包装，不会直接作为 SSE 发给 Android。
                yield StageResult(task.result())
                return

            # 等满一个心跳周期任务仍未完成，生成 thinking 事件。
            # 调用方会继续把它传给 FastAPI SSE 接口和 Android 客户端。
            yield ctx.thinking(stage, message)

    except BaseException:
        # 心跳循环因取消或异常退出时，不能把底层 LLM/检索任务留在后台空跑。
        if not task.done():
            task.cancel()

        # 保留原始异常，让外层 chat_stream 统一处理取消、错误事件和资源清理。
        raise


def cancel_background_tasks(timed_tasks: list[TimedTask[Any]]) -> None:
    """取消当前 turn 中仍未结束的后台任务。"""
    for timed_task in timed_tasks:
        # 已经完成的 Task 不能也不需要再次取消。
        if not timed_task.task.done():
            timed_task.task.cancel()


async def ensure_active(ctx: StreamContext) -> None:
    """
    检查当前对话轮次是否仍应继续执行。

    先检查当前进程内的取消令牌，再查询持久化取消状态，使其他后端进程
    接收到的取消请求也能被当前任务发现。
    """
    # 快速路径：同一进程内已经取消时，立即抛出 StreamCancelled。
    ctx.ensure_active()

    # 查询数据库中的取消请求，用于多进程或多实例部署场景。
    if await is_chat_turn_cancellation_requested(ctx.session_id, ctx.turn_id):
        # 将跨进程取消同步到当前进程的 CancellationToken。
        ctx.cancel_token.cancel()

    # 再检查一次；如果上一步设置了取消令牌，这里会抛出 StreamCancelled。
    ctx.ensure_active()


def _record_stage_timing(stage_timings_ms: dict[str, float], timing_key: str, started_at: float) -> None:
    stage_timings_ms.setdefault(timing_key, round((time.perf_counter() - started_at) * 1000, 2))
