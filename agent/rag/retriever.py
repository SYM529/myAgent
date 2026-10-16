"""统一检索入口：召回 → 精排 → 阈值过滤 → 生成带编号的上下文。

对外只暴露 retrieve()，返回的 Citation 带来源文件名与片段，
既供 LLM 生成答案时标注引用，也供前端展示"这句话出自哪里"。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from agent.rag.chunker import Chunk
from agent.rag.config import (
    CANDIDATE_MULTIPLIER,
    MAX_CITATION_CHARS,
    MIN_RERANK_SCORE,
    TOP_K,
)
from agent.rag.reranker import Reranker
from agent.rag.store import HybridStore

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """一条可核验的引用。"""

    index: int            # 上下文中的编号，如 [1]
    source: str           # 来源文件名
    chunk_index: int
    heading: str
    score: float
    text: str             # 截断后的片段，用于前端展示


@dataclass
class RetrievalResult:
    citations: list[Citation]
    context: str          # 编号后的上下文文本，可直接放入 prompt
    hit: bool             # 是否有可用结果


def _excerpt(text: str, limit: int = MAX_CITATION_CHARS) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + "…"


class Retriever:
    """把混合检索与精排串成一条链路。"""

    def __init__(
        self,
        store: HybridStore,
        reranker: Reranker | None = None,
        top_k: int = TOP_K,
    ):
        self.store = store
        self.reranker = reranker or Reranker()
        self.top_k = top_k

    def retrieve(
        self, query: str, *, scope: frozenset[str] | None = None
    ) -> RetrievalResult:
        """检索并组装引用与上下文。

        ``scope=None`` 表示调用方没有提供身份（MCP、脚本、未接线路径），
        此时按**仅 public** 的安全默认处理，绝不是"全库"。
        """
        if self.store.count() == 0:
            return RetrievalResult([], "", False)

        if scope is None:
            # 惰性导入：agent.rag.__init__ 会 import 本模块，顶层导入会成环
            from agent.rag import public_scope

            scope = public_scope(store=self.store)

        # 第一步：混合召回一个较大的候选集
        hits = self.store.search(
            query,
            top_k=self.top_k * CANDIDATE_MULTIPLIER,
            visible_doc_ids=scope,
        )
        if not hits:
            return RetrievalResult([], "", False)

        # 二次防御：即使存储层返回了范围外的命中（上游 bug / 脏数据），
        # 也不允许它进入精排、上下文与引用
        hits = [hit for hit in hits if hit.chunk.doc_id in scope]
        if not hits:
            return RetrievalResult([], "", False)

        # 第二步：cross-encoder 精排
        fused_scores = {h.chunk.chunk_id: h.score for h in hits}
        ranked = self.reranker.rerank(query, [h.chunk for h in hits], top_n=self.top_k)

        # 第三步：阈值过滤。宁可让 Agent 回答"知识库里没有"，
        # 也不要把明显不相关的内容塞进上下文诱发幻觉
        selected: list[tuple[Chunk, float]] = []
        for chunk, score in ranked:
            if score >= MIN_RERANK_SCORE:
                selected.append((chunk, score))
            else:
                logger.debug("丢弃低分候选 %.4f：%s", score, chunk.chunk_id)

        # 精排全部失败（降级返回 0 分）时，退回融合分排序，避免整条链路失效
        if not selected and all(score == 0.0 for _, score in ranked):
            selected = [(chunk, fused_scores.get(chunk.chunk_id, 0.0))
                        for chunk, _ in ranked]

        if not selected:
            return RetrievalResult([], "", False)

        citations = [
            Citation(
                index=i,
                source=chunk.source,
                chunk_index=chunk.chunk_index,
                heading=chunk.metadata.get("heading", ""),
                score=round(score, 4),
                text=_excerpt(chunk.text),
            )
            for i, (chunk, score) in enumerate(selected, start=1)
        ]
        return RetrievalResult(citations, self.build_context(citations), True)

    @staticmethod
    def build_context(citations: list[Citation]) -> str:
        """生成带编号的上下文。编号让模型能显式标注引用来源。"""
        blocks = []
        for citation in citations:
            location = f"来源：{citation.source}"
            if citation.heading:
                location += f" · {citation.heading}"
            blocks.append(f"[{citation.index}] （{location}）\n{citation.text}")
        return "\n\n".join(blocks)
