"""Intent resolution business logic.

This module contains deterministic rules for refining LLM-generated intent
with post-processing: constraint merging, category inference, product lookup,
and validation.

Separated from runtime/pipeline.py to keep business rules testable without
the full SSE pipeline machinery.
"""

from __future__ import annotations

import logging
from typing import Any

from src.config.domain_terms import (
    category_from_text,
    infer_category_from_product_type,
    is_supported_product_type,
    normalize_category,
)
from src.services.message_rules import (
    extract_adjustment_hints,
    extract_brand_prefer_from_message,
    extract_product_lookup_hints,
    extract_product_type_hint,
    has_shopping_signal,
)
from src.types.schemas import IntentResult

logger = logging.getLogger(__name__)

# Constraint fields that hold list values — must be merged (appended) rather
# than overwritten when multiple extraction sources contribute values.
MERGE_LIST_FIELDS = frozenset({
    "ingredient_avoid",
    "ingredient_prefer",
    "brand_avoid",
    "brand_prefer",
    "origin_avoid",
    "dietary",
})

_AMBIGUOUS_PHOTO_SCENE_TERMS = ("拍照", "摄影", "拍摄", "自拍")
_CONCRETE_DIGITAL_PRODUCT_TERMS = (
    "手机",
    "相机",
    "耳机",
    "电脑",
    "笔记本",
    "平板",
    "微单",
    "单反",
    "镜头",
)


def has_context_value(value: object) -> bool:
    """判断历史约束是否值得继承。

    ``None`` 和空容器代表“没有信息”；0、False 等标量仍可能是有效业务值，
    因此不能简单使用 ``bool(value)``。这是多轮上下文合并的基础工具函数。
    """
    if value is None:
        return False
    if isinstance(value, list | tuple | set | dict):
        return bool(value)
    return True


def is_ambiguous_scene_only_category_inference(message: str, category: str) -> bool:
    """Avoid turning a photo-taking scene into a digital-product category too early.

    Returns True if the category inference is ambiguous and should be skipped.
    """
    if category != "数码电子":
        return False
    if not any(term in message for term in _AMBIGUOUS_PHOTO_SCENE_TERMS):
        return False
    if any(term in message for term in _CONCRETE_DIGITAL_PRODUCT_TERMS):
        return False
    return extract_product_type_hint(message) is None


def _merge_constraints(
    merged: dict[str, Any],
    updates: dict[str, Any],
) -> dict[str, Any]:
    """合并两批意图约束，并正确处理列表字段。

    Java 中可以理解为创建一个新的 Map，而不是原地修改调用方传入的 Map。
    对 ``brand_avoid`` 等多值字段采用追加去重；预算等标量字段采用新值覆盖旧值。
    """
    # 浅拷贝确保该纯函数不会修改调用方持有的原始字典。
    result = dict(merged)
    for key, value in updates.items():
        if key in MERGE_LIST_FIELDS and isinstance(value, list):
            # dict.fromkeys 保留插入顺序并去重。
            existing = result.get(key, []) or []
            result[key] = list(dict.fromkeys([*existing, *value]))
        else:
            # 标量值遵循“后来的信息优先”。
            result[key] = value
    return result


def resolve_intent_constraints(
    intent: IntentResult,
    message: str,
) -> IntentResult:
    """对规则或 LLM 生成的意图做确定性后处理。

    为什么不能直接相信 LLM 返回值：

    1. 模型可能漏掉“再便宜一点”等局部调整；
    2. 模型可能把明确购物请求误判为闲聊；
    3. product_type 与 category 可能互相矛盾；
    4. 品牌、商品类型等领域词用代码表归一化更稳定。

    本函数不做网络或数据库 I/O，便于直接单元测试。它使用
    ``model_copy(update=...)`` 创建新 Pydantic 对象，类似 Java record 的 copy/with 方法。
    """
    # updates 收集 IntentResult 顶层字段的修改，最后一次性生成新对象。
    updates: dict[str, Any] = {}

    # 复制原始约束，避免直接修改 intent 中的字典。
    merged = dict(intent.extracted_constraints or {})

    # 1. Merge adjustment hints
    # 从原始用户消息补提“预算再低”“不要太甜”等确定性调整。
    adjustment = extract_adjustment_hints(message)
    if adjustment and intent.intent in {"recommend", "clarify", "feedback"}:
        merged = _merge_constraints(merged, adjustment)

    # 2. Extract brand preference (铁律3)
    # 品牌映射属于业务规则，不放到 Prompt 中，避免不同模型输出不一致。
    brand_prefer = extract_brand_prefer_from_message(message)
    if brand_prefer and intent.intent in {"recommend", "clarify", "feedback"}:
        existing = merged.get("brand_prefer", [])
        if isinstance(existing, list):
            merged["brand_prefer"] = list(dict.fromkeys(existing + brand_prefer))
        else:
            merged["brand_prefer"] = brand_prefer

    # 3. Post-LLM safety net: override chitchat/feedback with shopping signal
    # 如果模型说是闲聊，但代码检测到明确商品/预算等购物信号，则纠正为 clarify。
    # clarify 表示“方向是购物，但信息可能还不够”，后续槽位检查决定是否反问。
    if intent.intent in {"chitchat", "feedback"}:
        if has_shopping_signal(message):
            inferred_category = intent.category or category_from_text(message)
            intent = IntentResult(
                intent="clarify",
                confidence=intent.confidence,
                category=inferred_category,
                extracted_constraints=intent.extracted_constraints or {},
            )
            merged = dict(intent.extracted_constraints or {})
            updates = {}

    # 4. Fallback category inference for shopping intents
    # 只有购物相关意图且模型没给品类时才补推断，避免污染购物车等操作意图。
    if intent.intent in {"recommend", "clarify"} and not intent.category:
        inferred = category_from_text(message)
        if inferred and not is_ambiguous_scene_only_category_inference(message, inferred):
            updates["category"] = inferred

    # 5. Product-lookup extraction: "有鼠标吗" → product_type="鼠标"
    # lookup 可能同时包含产品类型、品牌等字段，统一按列表/标量规则合并。
    lookup = extract_product_lookup_hints(message)
    if lookup and intent.intent in {"recommend", "clarify", "feedback"}:
        merged = _merge_constraints(merged, lookup)

    # 6. Deterministic product_type extraction
    # 再做一层产品类型兜底；仅当已有结果没给 product_type 时补值。
    product_hint = extract_product_type_hint(message)
    if product_hint and intent.intent in {"recommend", "clarify", "feedback"}:
        if "product_type" not in merged:
            merged["product_type"] = product_hint

        # 商品类型能够唯一映射一级品类时，以领域映射表纠正模型品类。
        # 例如“跑鞋”应属于“服饰运动”，不能保留模型误判的“数码电子”。
        expected_cat = infer_category_from_product_type(product_hint)
        if expected_cat and normalize_category(intent.category) != expected_cat:
            updates["category"] = expected_cat

    # Apply accumulated updates
    # 只有真正变化时才写 update，避免无意义地创建新对象。
    if merged != (intent.extracted_constraints or {}):
        updates["extracted_constraints"] = merged
    if updates:
        intent = intent.model_copy(update=updates)

    return intent
