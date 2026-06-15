"""
混合检索服务 —— 整个 RAG 链路中最核心的模块

检索分为 7 个步骤（顺序执行）：
  1. 硬过滤 SQL  —— WHERE category='美妆' AND price<=200 AND brand NOT IN (...)
  2. 向量召回     —— pgvector <=> 余弦相似度，召 Top-50 chunk
  3. BM25 关键词  —— 分词 + TF-IDF 评分，召 Top-50 chunk
  4. RRF 融合     —— 向量排名 + BM25 排名 → 合并去重
  5. 渐进式放宽   —— 如果结果为 0，逐步放松预算上限
  6. Rerank 精排  —— qwen3-rerank 对候选重排序
  7. 证据绑定     —— 每个商品挂上命中的 chunk 作为推荐理由来源

输入: CriteriaPayload（购买标准） + feedback（用户反馈）
输出: list[ProductPayload]（商品列表） + evidence_by_product（证据映射）

Java 对比：类似 Elasticsearch 的查询流程
  硬过滤 = bool query 的 filter 子句
  向量召回 = knn query
  BM25 = match query
  RRF = 多路召回融合
  Rerank = Learning to Rank
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Mapping

from src.config.domain_terms import (
    avoid_trait_aliases, avoid_trait_matches_text,
    normalize_category, normalize_product_type, product_type_aliases,
)
from src.config.tuning import (
    BUDGET_PENALTY_CAP, BUDGET_PENALTY_FACTOR, BUDGET_RELAXATION_STEPS,
    FILTER_SCORE_BRAND, FILTER_SCORE_BRAND_PREFER, FILTER_SCORE_BUDGET,
    FILTER_SCORE_CATEGORY, FILTER_SCORE_SCENARIO, FILTER_SCORE_SKIN_TYPE,
    PGVECTOR_RECALL_LIMIT, RETRIEVAL_CANDIDATE_MULTIPLIER,
)
from src.repos.documents import (
    ChunkDocument, ImageSimilarityHit, VectorSearchFilters,
    evidence_for_chunk, evidence_for_product,
    list_products_by_image_similarity, list_vector_chunks_by_similarity,
)
from src.repos.products import get_product, list_products
from src.services.embedding import embed_text                    # 文本 → 1024 维向量
from src.services.retrieval_cache import get_retrieval_cache     # TTL 缓存
from src.services.retrieval_features import criteria_query_text, product_document_text, product_match_score
from src.services.reranker import rerank_texts                   # qwen3-rerank 精排
from src.types.sse_events import CriteriaPayload, EvidencePayload, ProductPayload

logger = logging.getLogger(__name__)

TRACE_VECTOR_TOP_K_LIMIT = 50  # 追踪日志中保留的向量召回 Top-K 上限


# ============================================================================
# 数据结构
# ============================================================================

@dataclass(frozen=True)
class ProductHit:
    """一个检索命中 —— 包含商品信息 + 向量分 + 过滤分 + 命中的 chunk"""
    product: ProductPayload
    vector_score: float         # 向量相似度分数
    filter_score: float         # 硬过滤匹配分数
    chunk: ChunkDocument | None = None  # 命中的 chunk（用于证据绑定）


@dataclass(frozen=True)
class RetrievalFilters:
    """检索时的排除条件（来自用户反馈）"""
    avoid_products: frozenset[str] = frozenset()   # 用户"不喜欢"的商品 ID
    avoid_traits: tuple[str, ...] = ()             # 用户"不喜欢"的特质


@dataclass(frozen=True)
class RetrievalOutput:
    """检索输出"""
    products: list[ProductPayload]
    evidence_by_product: dict[str, list[EvidencePayload]]  # product_id → 证据列表
    trace_details: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class VectorRecallResult:
    """向量召回阶段的结果统计"""
    hits: list[ProductHit]
    sql_filters_applied: dict[str, object]
    pre_filter_count: int      # 向量召回前（仅 SQL 过滤）的候选数
    post_filter_count: int     # 向量召回后（经过硬过滤）的候选数


# ============================================================================
# 公开接口
# ============================================================================

async def retrieve(
    criteria: CriteriaPayload,
    top_n: int = 5,
    feedback: Mapping[str, list[str]] | None = None,
    image_embedding: list[float] | None = None,
) -> list[ProductPayload]:
    """简化版接口 —— 只返回商品列表，不返回证据"""
    return (await retrieve_with_evidence(criteria, top_n=top_n, feedback=feedback, image_embedding=image_embedding)).products


# ============================================================================
# 核心函数：retrieve_with_evidence()
#
# 这是整个检索链路的主入口，按顺序执行 7 个步骤：
#   ① 缓存检查        → 同样的标准+反馈，返回缓存结果（有 TTL）
#   ② 硬过滤 SQL       → WHERE 子句先框定安全子集
#   ③ 向量召回+并行     → pgvector 余弦相似度 + 可选图片向量
#   ④ BM25 + RRF 融合   → 关键词召回 + 排名融合
#   ⑤ 渐进式放宽        → 结果为 0 时逐步放松预算
#   ⑥ 品牌偏好直取      → brand_prefer 时显式拉取
#   ⑦ Rerank 精排       → qwen3-rerank 重排序
#   ⑧ 证据绑定          → 每个商品挂 chunk 来源
# ============================================================================

async def retrieve_with_evidence(
    criteria: CriteriaPayload,
    top_n: int = 5,
    feedback: Mapping[str, list[str]] | None = None,
    image_embedding: list[float] | None = None,
) -> RetrievalOutput:
    """
    混合检索主入口

    参数:
        criteria: 购买标准（category/budget/skin_type/brand_avoid/...）
        top_n: 最终返回的商品数量
        feedback: 用户反馈（avoid_products/avoid_traits）
        image_embedding: 图片向量（可选，拍照找货场景）

    返回:
        RetrievalOutput(products, evidence_by_product, trace_details)
    """

    # ── 步骤 ①：缓存检查 ──
    cache = get_retrieval_cache()
    if image_embedding is None:  # 图片查询不走缓存（每次图片不同）
        cached = cache.get(criteria, feedback)
        if cached is not None:
            return cached

    # 构造排除过滤条件（用户反馈的"不喜欢"）
    filters = _retrieval_filters(feedback)

    # ═══════════════════════════════════════════════════
    # 将购买标准文本向量化
    # criteria_query_text() 把 criteria 拼接成查询文本
    # embed_text() 调 text-embedding-v3 转成 1024 维向量
    # ═══════════════════════════════════════════════════
    query_embedding = await embed_text(criteria_query_text(criteria))

    # ── 并行：图片向量召回（如果有）──
    visual_task = None
    if image_embedding:
        sql_filters = _sql_filters_for_recall(criteria, filters)
        visual_task = asyncio.create_task(      # ← 和文本召回并行跑！
            list_products_by_image_similarity(image_embedding, limit=10, filters=sql_filters)
        )

    # ── 步骤 ② + ③：向量召回 ──
    # _vector_recall_from_db 内部：
    #   1) 构建 SQL 过滤条件（category/budget/brand/product_type...）
    #   2) pgvector 余弦相似度查询（TOP 50 chunk）
    #   3) 对召回结果再过一遍硬过滤（_passes_hard_filters）
    recall_result = await _vector_recall_from_db(criteria, query_embedding, filters)
    chunk_hits = recall_result.hits
    recall_criteria = criteria
    recall_filters = filters
    recall_stats = recall_result
    original_budget_max = criteria.constraints.budget_max

    # 记录第一轮（严格模式）的候选数量
    relaxation_steps: list[dict[str, object]] = [
        {"step": "strict", "reason": "category/budget/product_type/exclusion hard filters",
         "relaxed_fields": [], "candidate_count": len(chunk_hits)}
    ]

    # ── 步骤 ⑤：渐进式放宽 ──
    # 如果严格模式下没结果，逐步放松约束：
    #   预算上限 × 1.3 → × 1.5 → 无限
    #   移除 feedback 的 avoid_traits
    if not chunk_hits:
        for step, relaxed_criteria, relaxed_filters, relaxed_fields in _relaxation_attempts(criteria, filters):
            relaxed_result = await _vector_recall_from_db(relaxed_criteria, query_embedding, relaxed_filters)
            relaxed_hits = relaxed_result.hits
            relaxation_steps.append({
                "step": step, "reason": "放宽约束重新检索",
                "relaxed_fields": relaxed_fields, "candidate_count": len(relaxed_hits),
            })
            if relaxed_hits:
                chunk_hits = relaxed_hits
                recall_criteria = relaxed_criteria
                recall_filters = relaxed_filters
                recall_stats = relaxed_result
                break

    # ── 步骤 ④：BM25 + RRF 融合 ──
    from src.services.bm25_recall import bm25_index   # BM25 索引（lazy init）
    from src.services.rrf_merge import rrf_merge       # RRF 融合算法

    await bm25_index.ensure_ready()                    # 确保 BM25 索引已构建
    bm25_hits = bm25_index.search(criteria_query_text(criteria), limit=PGVECTOR_RECALL_LIMIT)

    if bm25_hits:
        # 提取向量召回的 chunk ID 排名列表
        vector_ranking = [h.chunk.id for h in chunk_hits if h.chunk is not None]
        # 提取 BM25 召回的 chunk ID 排名列表
        bm25_ranking = [h.chunk_id for h in bm25_hits]

        # RRF 融合：综合两路排名，取交集的排序结果
        # 算法: score(chunk) = SUM( 1/(k + rank_i) )  for each ranking list
        merged_ids = rrf_merge([vector_ranking, bm25_ranking])

        # 按 RRF 融合后的顺序重排 chunk_hits
        chunk_by_id = {h.chunk.id: h for h in chunk_hits if h.chunk is not None}
        rrf_ordered_hits = [chunk_by_id[cid] for cid in merged_ids if cid in chunk_by_id]
        # 追加不在 BM25 结果中的向量 hits（放在末尾）
        rrf_ids_set = set(merged_ids)
        for h in chunk_hits:
            if h.chunk is not None and h.chunk.id not in rrf_ids_set:
                rrf_ordered_hits.append(h)
        chunk_hits = rrf_ordered_hits

    # ── 合并图片向量召回结果 ──
    visual_hits: list[ProductHit] = []
    visual_recall_stats: dict[str, object] = {}
    if visual_task is not None:
        try:
            image_hits = await visual_task  # 等图片召回完成
            visual_hits, visual_recall_stats = _build_visual_hits(image_hits, criteria, filters)
        except Exception:
            # 图片召回失败不阻塞主流程，降级为纯文本
            logger.warning("Visual recall failed, degraded to text-only", exc_info=True)
            visual_recall_stats = {"error": "visual recall failed, degraded to text-only"}

    # 合并文本 + 图片召回
    merged_hits = _merge_text_and_visual(chunk_hits, visual_hits)

    # ── 步骤 ⑥：品牌偏好直取 ──
    # brand_prefer 只靠向量相似度不能保证出现，需要显式拉取
    brand_hits = _fetch_brand_preference_products(criteria, filters)
    if brand_hits:
        merged_hits = _merge_text_and_visual(merged_hits, brand_hits)

    # 如果所有路径都没结果，返回空
    if not merged_hits:
        return RetrievalOutput(products=[], evidence_by_product={},
                               trace_details=_visual_trace(visual_recall_stats))

    # ── 粗排（向量分 + 过滤分 + 预算惩罚）──
    # 取 top_n * 倍数 作为候选集，后面交给 Rerank 精排
    candidate_hits = _rank_hits(merged_hits, original_budget_max)[
        : max(top_n * RETRIEVAL_CANDIDATE_MULTIPLIER, top_n)
    ]

    # ── 步骤 ⑦：Rerank 精排 ──
    # 调 qwen3-rerank 对候选集重新排序
    ranked_hits, all_reranked_hits = await _rerank_chunk_hits(criteria, candidate_hits, top_n=top_n)

    # ── 品牌偏好优先（硬保证）──
    # brand_prefer 商品必须排在第一位，不管 Rerank 怎么打分
    ranked_hits = _elevate_brand_preference(criteria, ranked_hits, all_reranked_hits)

    # ── 步骤 ⑧：证据绑定 ──
    evidence_map = _evidence_by_product(ranked_hits, all_reranked_hits)
    await _supplement_visual_evidence(ranked_hits, evidence_map)

    # ── 构建追踪信息（用于可观测性和调试）──
    trace = _trace_details_for_chunk_retrieval(
        criteria=criteria, recall_criteria=recall_criteria, filters=recall_filters,
        chunk_hits=chunk_hits, candidate_hits=candidate_hits, ranked_hits=ranked_hits,
        relaxation_steps=relaxation_steps, recall_stats=recall_stats,
    )
    if visual_recall_stats:
        trace["visual_recall"] = visual_recall_stats

    result = RetrievalOutput(
        products=[hit.product for hit in ranked_hits],
        evidence_by_product=evidence_map,
        trace_details=trace,
    )

    # ── 写入缓存 ──
    if image_embedding is None:
        cache.set(criteria, feedback, result)

    return result


# ============================================================================
# 向量召回（pgvector）
# ============================================================================

async def _vector_recall_from_db(
    criteria: CriteriaPayload, query_embedding: list[float], filters: RetrievalFilters,
) -> VectorRecallResult:
    """pgvector 向量召回入口"""
    return await _vector_recall_from_pgvector(criteria, query_embedding, filters)


async def _vector_recall_from_pgvector(
    criteria: CriteriaPayload, query_embedding: list[float], filters: RetrievalFilters,
) -> VectorRecallResult:
    """
    pgvector 向量召回核心：
    1. 构建 SQL 硬过滤条件（category / budget / brand / product_type ...）
    2. pgvector <=> 余弦距离排序，取 Top-50
    3. 对召回结果再过一遍硬过滤（_passes_hard_filters）—— 双重保险
    """
    hits: list[ProductHit] = []
    sql_filters = _sql_filters_for_recall(criteria, filters)

    # pgvector 相似度查询（<=> 余弦距离越小越相似）
    vector_hits = await list_vector_chunks_by_similarity(
        query_embedding,
        limit=PGVECTOR_RECALL_LIMIT,  # 默认 50
        filters=sql_filters,
    )

    # 对召回结果再过一遍硬过滤（防止 pgvector 近似搜索漏掉一些约束）
    for vector_hit in vector_hits:
        chunk = vector_hit.document
        product = get_product(chunk.product_id)  # 从商品表还原完整商品信息
        if product is None or not _passes_hard_filters(criteria, product, filters):
            continue  # ← 二次过滤
        hits.append(ProductHit(
            product=product,
            vector_score=_distance_to_score(vector_hit.distance),
            filter_score=_filter_score(criteria, product),
            chunk=chunk,
        ))

    return VectorRecallResult(
        hits=hits,
        sql_filters_applied=_sql_filter_payload(sql_filters),
        pre_filter_count=len(vector_hits),
        post_filter_count=len(hits),
    )


# ============================================================================
# 粗排：向量分 + 过滤分 + 预算惩罚
# ============================================================================

def _rank_hits(
    hits: list[ProductHit],
    original_budget_max: float | None = None,
) -> list[ProductHit]:
    """多目标排序：过滤分 DESC → 向量分 DESC → 价格 ASC"""

    def _key(hit: ProductHit) -> tuple:
        price = hit.product.price or 0.0
        budget_penalty = 0.0
        # 超预算惩罚：超得越多扣越多，但有上限
        if original_budget_max and original_budget_max > 0 and price > original_budget_max:
            excess_pct = (price - original_budget_max) / original_budget_max
            budget_penalty = min(excess_pct * BUDGET_PENALTY_FACTOR, BUDGET_PENALTY_CAP)
        return (
            hit.filter_score - budget_penalty,  # 第一优先级
            hit.vector_score,                     # 第二优先级
            -(price),                             # 第三优先级（负值 = 价格低排前面）
        )

    return sorted(hits, key=_key, reverse=True)


# ============================================================================
# 证据绑定：为每个商品匹配证据 chunk
# ============================================================================

_EVIDENCE_KIND_PRIORITY = ("why_buy", "faq", "risk", "compare")  # 证据类型优先级

def _evidence_by_product(
    ranked_hits: list[ProductHit], all_reranked_hits: list[ProductHit],
) -> dict[str, list[EvidencePayload]]:
    """
    为每个最终排名的商品，从 Rerank 结果中选取最优证据 chunk
    按优先级选：why_buy > faq > risk > compare
    每个类型只取一个（多样性 > 数量）
    """
    evidence_by_product: dict[str, list[EvidencePayload]] = {}
    for hit in ranked_hits:
        product_id = hit.product.product_id
        # 按证据类型分组，每个类型保留排名最高的 chunk
        groups: dict[str, ChunkDocument] = {}
        seen_chunk_ids: set[str] = set()
        for candidate in all_reranked_hits:
            if candidate.product.product_id != product_id or candidate.chunk is None:
                continue
            if candidate.chunk.id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(candidate.chunk.id)
            kind = candidate.chunk.metadata.get("evidence_kind") or "other"
            if kind not in groups:
                groups[kind] = candidate.chunk

        # 按优先级选取证据
        selected_chunks = [groups[kind] for kind in _EVIDENCE_KIND_PRIORITY if kind in groups]
        if not selected_chunks and groups:
            selected_chunks = [next(iter(groups.values()))]

        evidence_by_product[product_id] = [evidence_for_chunk(chunk) for chunk in selected_chunks]

    return evidence_by_product


# ============================================================================
# 渐进式放宽
# ============================================================================

def _relaxation_attempts(
    criteria: CriteriaPayload, filters: RetrievalFilters,
) -> list[tuple[str, CriteriaPayload, RetrievalFilters, list[str]]]:
    """
    生成一系列放宽尝试：
    1. 预算上限 × 1.3 → × 1.5 → 无上限
    2. 移除 feedback 的 avoid_traits

    category 和 product_type 永远不会放宽 —— 品类不对比没结果更差
    """
    attempts: list[tuple[str, CriteriaPayload, RetrievalFilters, list[str]]] = []
    if criteria.constraints.budget_max is not None:
        budget = criteria.constraints.budget_max
        for ratio, label in BUDGET_RELAXATION_STEPS:
            relaxed_max = None if ratio is None else math.ceil(budget * ratio)
            attempts.append((label, _criteria_with_constraints(criteria, budget_max=relaxed_max),
                             filters, ["budget_max"]))
    if filters.avoid_traits:
        attempts.append(("without_feedback_avoid_traits", criteria,
                         RetrievalFilters(avoid_products=filters.avoid_products),
                         ["feedback.avoid_traits"]))
    return attempts


def _criteria_with_constraints(criteria: CriteriaPayload, **updates: object) -> CriteriaPayload:
    """复制 criteria 并修改部分 constraint 字段"""
    constraints = criteria.constraints.model_copy(update=updates)
    return criteria.model_copy(update={"constraints": constraints})


# ============================================================================
# Rerank 精排
# ============================================================================

async def _rerank_chunk_hits(
    criteria: CriteriaPayload, hits: list[ProductHit], top_n: int,
) -> tuple[list[ProductHit], list[ProductHit]]:
    """
    调 qwen3-rerank 对候选集重新排序

    输入: hits（粗排后的候选，可能 15-20 个）
    输出: (top_n 个去重商品, 全量 rerank 结果)

    每个候选的文本 = 商品信息 + chunk 文本
    """
    # 每个候选拼成一段完整文本
    documents = [_chunk_document_text(hit) for hit in hits]

    # 调 qwen3-rerank API：输入 query + documents，返回排序后的索引
    index_order = await rerank_texts(criteria, documents, top_n=len(hits))

    # 按 rerank 排序后的结果
    all_reranked_hits = [hits[index] for index in index_order if 0 <= index < len(hits)]

    # 取 top_n 个去重商品
    selected: list[ProductHit] = []
    seen_products: set[str] = set()
    for hit in all_reranked_hits:
        if hit.product.product_id in seen_products:
            continue
        seen_products.add(hit.product.product_id)
        selected.append(hit)
        if len(selected) >= top_n:
            break

    # 如果 rerank 后不够 top_n，从原始 hits 中补充
    if len(selected) < top_n:
        for hit in hits:
            if hit.product.product_id in seen_products:
                continue
            seen_products.add(hit.product.product_id)
            selected.append(hit)
            if len(selected) >= top_n:
                break

    return selected, all_reranked_hits


def _chunk_document_text(hit: ProductHit) -> str:
    """拼接商品信息 + chunk 文本，作为 Rerank 的输入文本"""
    chunk_text = hit.chunk.chunk_text if hit.chunk is not None else ""
    return product_document_text(hit.product, extra_text=chunk_text)


# ============================================================================
# 硬过滤函数族 —— 7 个纯函数，每个检查一个约束维度
#
# 所有这些函数签名一致: (criteria, product, filters) → bool
# 通过 _FILTER_CHECKS 元组串联，_passes_hard_filters 就是 all(checks)
# ═══════════════════════════════════════════════════════════════════

FilterCheck = Callable[[CriteriaPayload, ProductPayload, RetrievalFilters], bool]

def _passes_hard_filters(criteria: CriteriaPayload, product: ProductPayload, filters: RetrievalFilters) -> bool:
    """综合硬过滤：7 个检查全部通过才返回 True"""
    return all(check(criteria, product, filters) for check in _FILTER_CHECKS)


def _passes_feedback_product_filter(
    criteria: CriteriaPayload, product: ProductPayload, filters: RetrievalFilters,
) -> bool:
    """排除用户"不喜欢"的商品"""
    del criteria
    return product.product_id not in filters.avoid_products


def _passes_category_filter(
    criteria: CriteriaPayload, product: ProductPayload, filters: RetrievalFilters,
) -> bool:
    """品类过滤：商品品类必须匹配标准中的品类"""
    del filters
    criteria_category = normalize_category(criteria.category)
    product_category = normalize_category(product.category)
    return not criteria_category or product_category == criteria_category


def _passes_budget_filter(
    criteria: CriteriaPayload, product: ProductPayload, filters: RetrievalFilters,
) -> bool:
    """预算过滤：商品价格必须在 [budget_min, budget_max] 内"""
    del filters
    if product.price is None:
        return True  # 无价格信息的不拦截
    constraints = criteria.constraints
    if constraints.budget_max is not None and product.price > constraints.budget_max:
        return False
    if constraints.budget_min is not None and product.price < constraints.budget_min:
        return False
    return True


def _passes_brand_filter(
    criteria: CriteriaPayload, product: ProductPayload, filters: RetrievalFilters,
) -> bool:
    """品牌过滤：排除 brand_avoid 中的品牌（精确匹配 + 别名匹配）"""
    del filters
    constraints = criteria.constraints
    if not constraints.brand_avoid or not product.brand:
        return True
    product_brand_lower = product.brand.lower()
    if any(_brand_matches(brand, product_brand_lower) for brand in constraints.brand_avoid):
        return False
    # 别名匹配：例如 brand_avoid=["日系"] → 通过 avoid_trait_matches_text 扩展匹配
    return not any(avoid_trait_matches_text(brand, product.brand) for brand in constraints.brand_avoid)


def _passes_origin_filter(
    criteria: CriteriaPayload, product: ProductPayload, filters: RetrievalFilters,
) -> bool:
    """产源过滤：排除 origin_avoid 中的产地"""
    del filters
    constraints = criteria.constraints
    if not constraints.origin_avoid or not product.brand:
        return True
    return not any(avoid_trait_matches_text(origin, product.brand) for origin in constraints.origin_avoid)


def _passes_product_type_filter(
    criteria: CriteriaPayload, product: ProductPayload, filters: RetrievalFilters,
) -> bool:
    """产品类型过滤：精确匹配 + 包含匹配（"裤子" 匹配 "户外裤"）"""
    del filters
    criteria_product_type = normalize_product_type(criteria.constraints.product_type)
    product_type = normalize_product_type(product.sub_category)
    if not criteria_product_type:
        return True
    if product_type == criteria_product_type:
        return True
    if criteria_product_type in (product_type or ""):
        return True
    if (product_type or "") in criteria_product_type:
        return True
    return False


def _passes_avoid_trait_filter(
    criteria: CriteriaPayload, product: ProductPayload, filters: RetrievalFilters,
) -> bool:
    """规避特质过滤：排除 ingredient_avoid 中提到的成分/特质"""
    avoid_traits = tuple(criteria.constraints.ingredient_avoid) + filters.avoid_traits
    return not any(_matches_avoid_trait(product, token) for token in avoid_traits)


# 7 个过滤检查的组合（顺序无关，all() 短路）
_FILTER_CHECKS: tuple[FilterCheck, ...] = (
    _passes_feedback_product_filter,   # 排除用户不喜欢的商品
    _passes_category_filter,           # 品类匹配
    _passes_budget_filter,             # 预算范围
    _passes_brand_filter,              # 品牌排除
    _passes_origin_filter,             # 产地排除
    _passes_product_type_filter,       # 产品类型匹配
    _passes_avoid_trait_filter,        # 规避特质排除
)


# ============================================================================
# 辅助函数
# ============================================================================

def _brand_matches(avoid_brand: str, product_brand_lower: str) -> bool:
    """品牌精确匹配（忽略大小写）"""
    return avoid_brand.strip().lower() == product_brand_lower


def _matches_avoid_trait(product: ProductPayload, token: str) -> bool:
    """检查商品的文本字段中是否包含要规避的特质"""
    return avoid_trait_matches_text(token, _product_haystack(product))


def _product_haystack(product: ProductPayload) -> str:
    """把商品的所有文本字段拼接成一个字符串（用于特质匹配搜索）"""
    parts = [
        product.product_id, product.name, product.brand or "",
        product.category, product.sub_category or "",
        product.use_scenario or "",
        *product.ingredient_tags, *product.ingredient_avoid,
    ]
    return " ".join(part for part in parts if part)


def _unique_product_ids(hits: list[ProductHit]) -> list[str]:
    """去重提取商品 ID 列表（保持顺序）"""
    seen: set[str] = set()
    result: list[str] = []
    for hit in hits:
        if hit.product.product_id in seen:
            continue
        seen.add(hit.product.product_id)
        result.append(hit.product.product_id)
    return result


def _retrieval_filters(feedback: Mapping[str, list[str]] | None) -> RetrievalFilters:
    """从用户反馈中构造排除条件"""
    if not feedback:
        return RetrievalFilters()
    return RetrievalFilters(
        avoid_products=frozenset(feedback.get("avoid_products", [])),
        avoid_traits=tuple(feedback.get("avoid_traits", [])),
    )


def _sql_filters_for_recall(criteria: CriteriaPayload, filters: RetrievalFilters) -> VectorSearchFilters:
    """构建 pgvector 查询的 SQL 过滤条件（在向量搜索前缩小范围）"""
    return VectorSearchFilters(
        category=normalize_category(criteria.category),
        product_type=criteria.constraints.product_type,
        product_type_aliases=list(product_type_aliases(criteria.constraints.product_type)),
        avoid_product_ids=list(filters.avoid_products),
        avoid_brands=_brand_avoid_terms(criteria),
    )


def _brand_avoid_terms(criteria: CriteriaPayload) -> tuple[str, ...]:
    """获取要排除的品牌列表（含别名扩展）"""
    avoid = set(criteria.constraints.brand_avoid or [])
    for ab in criteria.constraints.brand_avoid or []:
        for alias in avoid_trait_aliases(ab):
            avoid.add(alias)
    return tuple(sorted(avoid))


def _sql_filter_payload(filters: VectorSearchFilters) -> dict[str, object]:
    """将 SQL 过滤条件转为可序列化的字典（用于追踪日志）"""
    return {
        "category": filters.category,
        "product_type": filters.product_type,
        "product_type_aliases": filters.product_type_aliases,
        "avoid_product_ids": sorted(filters.avoid_product_ids or []),
        "avoid_brands": sorted(filters.avoid_brands or []),
    }


# ============================================================================
# 公开工具函数
# ============================================================================

def filter_products(
    products: list[ProductPayload],
    criteria: CriteriaPayload,
    feedback: Mapping[str, list[str]] | None = None,
) -> list[ProductPayload]:
    """纯内存硬过滤 —— 用于不经过 pgvector 的场景（eval/trace 等）"""
    filters = _retrieval_filters(feedback)
    return [p for p in products if _passes_hard_filters(criteria, p, filters)]


def _filter_score(criteria: CriteriaPayload, product: ProductPayload) -> float:
    """计算商品的硬过滤匹配分（用于粗排）"""
    score = 0.0
    constraints = criteria.constraints
    if constraints.brand_prefer:
        p_brand = (product.brand or "").lower()
        for b in constraints.brand_prefer:
            if b.strip().lower() == p_brand:
                score += FILTER_SCORE_BRAND_PREFER
                break
    if not constraints.skin_type and not constraints.use_scenario:
        return score
    doc = product_document_text(product)
    if constraints.skin_type:
        if constraints.skin_type.strip().lower() in doc.lower():
            score += FILTER_SCORE_SKIN_TYPE
    if constraints.use_scenario:
        if constraints.use_scenario.strip().lower() in doc.lower():
            score += FILTER_SCORE_SCENARIO
    score += product_match_score(criteria, product)
    return round(score, 4)


def _distance_to_score(distance: float) -> float:
    """pgvector 余弦距离 → [0,1] 分数（距离越小分越高）"""
    return round(1.0 - min(distance / 2.0, 1.0), 4)


# ============================================================================
# 品牌偏好 + 图片召回 + 文本视觉合并
# ============================================================================

def _fetch_brand_preference_products(
    criteria: CriteriaPayload, filters: RetrievalFilters,
) -> list[ProductHit]:
    """
    品牌偏好直取：当 brand_prefer 设置后，直接从商品表拉取匹配品牌的所有商品
    向量相似度不能保证品牌偏好商品出现在召回中，这是硬保证
    """
    brand_prefer = criteria.constraints.brand_prefer
    if not brand_prefer:
        return []
    sql_filters = _sql_filters_for_recall(criteria, filters)
    all_products = list_products(filters=sql_filters)
    hits: list[ProductHit] = []
    for product in all_products:
        product_brand_lower = (product.brand or "").lower()
        for bp in brand_prefer:
            if bp.strip().lower() == product_brand_lower:
                if _passes_hard_filters(criteria, product, filters):
                    hits.append(ProductHit(
                        product=product,
                        vector_score=0.0,  # 品牌偏好不依赖向量分
                        filter_score=FILTER_SCORE_BRAND_PREFER,
                        chunk=None,
                    ))
    return hits


def _build_visual_hits(
    image_hits: list[ImageSimilarityHit], criteria: CriteriaPayload, filters: RetrievalFilters,
) -> tuple[list[ProductHit], dict[str, object]]:
    """将图片相似度结果转换为 ProductHit 列表"""
    hits: list[ProductHit] = []
    for ih in image_hits:
        product = get_product(ih.product_id)
        if product is None or not _passes_hard_filters(criteria, product, filters):
            continue
        hits.append(ProductHit(
            product=product,
            vector_score=ih.similarity,
            filter_score=_filter_score(criteria, product),
            chunk=None,  # 图片召回的没有文本 chunk
        ))
    return hits, {"visual_hits": len(hits)}


def _elevate_brand_preference(
    criteria: CriteriaPayload, ranked_hits: list[ProductHit], all_reranked_hits: list[ProductHit],
) -> list[ProductHit]:
    """
    硬保证：brand_prefer 商品必须排在第一位
    不管 Rerank 打了多少分，偏好品牌的优先级最高
    """
    brand_prefer = criteria.constraints.brand_prefer
    if not brand_prefer:
        return ranked_hits
    pref_brands = {bp.strip().lower() for bp in brand_prefer}
    pref_hits = [
        h for h in ranked_hits
        if (h.product.brand or "").lower() in pref_brands
    ]
    if not pref_hits:
        return ranked_hits
    # 把 brand_prefer 商品移到最前面，其余保持 rerank 原序
    pref_ids = {h.product.product_id for h in pref_hits}
    rest = [h for h in ranked_hits if h.product.product_id not in pref_ids]
    return pref_hits + rest


def inject_brand_preference_products(
    criteria: CriteriaPayload, hits: list[ProductHit], filters: RetrievalFilters,
) -> list[ProductHit]:
    """在已有候选集中注入品牌偏好商品（用于外部注入场景）"""
    brand_hits = _fetch_brand_preference_products(criteria, filters)
    if not brand_hits:
        return hits
    return _merge_text_and_visual(hits, brand_hits)


def _merge_text_and_visual(
    text_hits: list[ProductHit], visual_hits: list[ProductHit],
) -> list[ProductHit]:
    """合并文本召回的 hits 和图片/品牌召回的 hits（去重，文本优先）"""
    text_ids = {h.product.product_id for h in text_hits}
    extra = [h for h in visual_hits if h.product.product_id not in text_ids]
    return text_hits + extra  # 文本结果在前，额外结果追加


async def _supplement_visual_evidence(
    ranked_hits: list[ProductHit], evidence_map: dict[str, list[EvidencePayload]],
) -> None:
    """为纯图片召回（无 chunk）的商品补充证据"""
    for hit in ranked_hits:
        pid = hit.product.product_id
        if pid in evidence_map and evidence_map[pid]:
            continue
        ev = evidence_for_product(pid)
        if ev:
            evidence_map[pid] = ev


def _visual_trace(visual_recall_stats: dict[str, object]) -> dict[str, object]:
    """构建图片召回追踪信息"""
    return {"visual_recall": visual_recall_stats} if visual_recall_stats else {}


# ============================================================================
# 追踪信息构建
# ============================================================================

def _trace_details_for_chunk_retrieval(
    criteria: CriteriaPayload, recall_criteria: CriteriaPayload, filters: RetrievalFilters,
    chunk_hits: list[ProductHit], candidate_hits: list[ProductHit],
    ranked_hits: list[ProductHit], relaxation_steps: list[dict[str, object]],
    recall_stats: VectorRecallResult,
) -> dict[str, object]:
    """构建完整的检索追踪信息（写入 DB trace 表，用于调试和评测）"""
    vector_hits = sorted(chunk_hits, key=lambda hit: hit.vector_score, reverse=True)[:TRACE_VECTOR_TOP_K_LIMIT]
    return {
        "filters_applied": {
            "_candidate_product_ids": _unique_product_ids(candidate_hits),
            "category": criteria.category or None,
            "budget_max": criteria.constraints.budget_max,
            "product_type": criteria.constraints.product_type,
            "product_type_aliases": list(product_type_aliases(criteria.constraints.product_type)),
            "avoid_products": sorted(filters.avoid_products),
            "avoid_traits": list(filters.avoid_traits),
            "skin_type": criteria.constraints.skin_type,
            "brand_avoid": criteria.constraints.brand_avoid,
            "brand_prefer": criteria.constraints.brand_prefer,
            "origin_avoid": criteria.constraints.origin_avoid,
            "ingredient_avoid": criteria.constraints.ingredient_avoid,
            "use_scenario": criteria.constraints.use_scenario,
        },
        "relaxation_steps": relaxation_steps,
        "vector_hits": [_trace_hit_payload(hit, i + 1) for i, hit in enumerate(vector_hits)],
        "ranked_products": [hit.product.product_id for hit in ranked_hits],
        "evidence_by_product": {hit.product.product_id: [getattr(e, "source_type", "") for e in
                                  _evidence_by_product(ranked_hits, candidate_hits).get(hit.product.product_id, [])]
                                for hit in ranked_hits},
    }


def _trace_hit_payload(hit: ProductHit, rank: int) -> dict[str, object]:
    """单个检索命中的追踪信息"""
    payload: dict[str, object] = {
        "rank": rank,
        "product_id": hit.product.product_id,
        "vector_score": hit.vector_score,
        "filter_score": hit.filter_score,
    }
    if hit.chunk is not None:
        payload["chunk_id"] = hit.chunk.id
        payload["chunk_text"] = hit.chunk.chunk_text[:200]  # 截断长文本
        payload["evidence_kind"] = hit.chunk.metadata.get("evidence_kind") or "other"
    return payload
