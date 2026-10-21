"""语义缓存：让重复提问不再付第二次钱。

为什么需要
----------
Agent 一次请求背后是多轮 LLM 调用，外加检索、精排、联网搜索这些外部 API。
同一个意思的问题再问一遍，答案是一样的，钱和延迟却要再付一次。
精确匹配（字符串相等）几乎命中不了——用户不会一字不差地复述问题，
于是有了按语义相似度匹配的方案。

最大的风险不是慢，而是**答错**
------------------------------
语义相近 ≠ 语义相同。"北京今天天气"和"北京明天天气"相似度极高，
一旦误命中，用户拿到的是**看起来很自信的错答案**——
这比多花几分钱严重得多。

所以这里设了四道闸，任一不满足都不复用：

1. **高阈值**：只有相似度超过 `CACHE_MIN_SIMILARITY` 才复用，
   阈值是用评测集标定的（见 README），不是凭感觉定的 0.8。
2. **数字集合一致**：「本金 5000/3.5%」与「10000/5%」语义几乎一样（相似度
   0.9257）但答案完全不同——数字变了必须 miss；正是这道闸把最大误命中
   相似度从 0.9257 压到 0.8763，阈值才不用被迫提到 0.95。
3. **能力签名**：只有当次请求的模型 / 工具集 / system prompt 与写入时一致才复用。
   改了提示词或增删了工具，旧答案就不再代表现在的行为。
4. **上下文指纹**：命中范围包含"本轮之前的历史"指纹。
   多轮对话里不能把上一轮语境下的答案拿来答这一轮。

另外，缓存按 user_id 分命名空间：一个人的缓存内容
绝不会出现在另一个人的回答里。
"""

from __future__ import annotations

import logging
import math
import os
import re
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# auto：配了 Embedding 才启用；off 强制关闭；on 强制尝试（缺 key 时会自动退化为不缓存）
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "auto").lower()
# 默认值由 `python -m evals.cache_calibrate` 标定得出，不是拍脑袋定的：
#   148 条评测集上，可缓存问题之间的最大相似度 0.8763
#   同义改写（可缓存）的最小相似度 0.5978
# 取 0.92：误命中 0 条，且留有 0.0437 的余量抵御数据集未覆盖到的相似配对。
CACHE_MIN_SIMILARITY = float(os.getenv("CACHE_MIN_SIMILARITY", "0.92"))
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))
CACHE_MAX_ENTRIES = int(os.getenv("CACHE_MAX_ENTRIES", "500"))


def available() -> bool:
    """缓存是否可用。缺 Embedding key 时不启用，其余功能不受影响。"""
    flag = CACHE_ENABLED
    if flag in ("0", "off", "false", "no"):
        return False
    if flag in ("1", "on", "true", "yes"):
        return True
    return bool(os.getenv("EMBEDDING_API_KEY") or os.getenv("SILICON_API_KEY"))


_get_embedder = None


def get_embedder():
    """复用 RAG 的 Embedding 客户端：同一套向量空间，不必再接一个模型服务。"""
    global _get_embedder
    if _get_embedder is None:
        from agent.rag.embeddings import EmbeddingClient

        client = EmbeddingClient()
        _get_embedder = client.embed_one
    return _get_embedder


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """余弦相似度。两个向量都已是单位向量的情形很常见，
    但这里仍然完整计算，避免依赖 embedding 的具体归一化行为。"""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


_NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?")


def number_signature(text: str) -> tuple[str, ...]:
    """提取问题里的数字集合。

    这不是锦上添花，而是标定数据逼出来的一道闸门。

    用评测集扫描时发现，相似度最高的一对"不同问题"是：
      「本金 5000 元，年利率 3.5%，一年的利息是多少元」
      「10000 元按年利率 5% 存一年，利息是多少」
    余弦相似度 0.9257，但答案是 175 元与 500 元——完全不同。
    这类"换了数字"的提问，向量模型很难分开：
    两句的语义结构确实几乎一样，只有数值不同。

    而数字集合比对是一行正则就能做到的事，且命中判定因此
    可以在**更低的阈值**上运行，把同义改写的覆盖率从 17% 拉回 50%。
    """
    return tuple(sorted(_NUMBER_PATTERN.findall(text or "")))


@dataclass
class CacheEntry:
    question: str
    answer: str
    vector: list[float]
    signature: str          # 写入时的能力签名
    context_id: str         # 写入时的上下文指纹
    created_at: float
    numbers: tuple[str, ...] = field(default_factory=tuple)
    hits: int = field(default=0)


class SemanticCache:
    """进程内的语义缓存。

    为什么是进程内而不是 Redis：
    相似度检索需要向量索引，Redis 默认没有向量能力；引入 RediSearch
    会显著抬高部署复杂度。这里与"并发闸门"保持一致的取舍——
    进程内简单可靠，代价是多副本时每个实例各有一份命中。
    """

    def __init__(self, threshold: float = CACHE_MIN_SIMILARITY,
                 ttl: float = CACHE_TTL, max_entries: int = CACHE_MAX_ENTRIES,
                 clock=time.time):
        self.threshold = threshold
        self.ttl = ttl
        self.max_entries = max_entries
        self._clock = clock
        self._store: dict[str, list[CacheEntry]] = {}
        self._hits = 0
        self._misses = 0

    # ---------- 查询与写入 ----------

    def lookup(
        self,
        user_id: int,
        vector: list[float],
        signature: str,
        context_id: str,
        numbers: tuple[str, ...] = (),
    ) -> CacheEntry | None:
        """找出可复用的答案。任一防错条件不满足都返回 None（宁可不省这笔钱）。"""
        now = self._clock()
        self._prune(user_id, now)

        best: CacheEntry | None = None
        best_score = 0.0
        for entry in self._store.get(str(user_id), ()):
            if now - entry.created_at > self.ttl:
                continue
            # 能力签名不同 → 提示词或工具变了，旧答案不再可信
            if entry.signature != signature:
                continue
            # 上下文指纹不同 → 这是另一个对话语境，不能直接套用
            if entry.context_id != context_id:
                continue
            # 数字集合不同 → 换了参数的同类问题，答案必然不同
            if entry.numbers != numbers:
                continue
            score = cosine_similarity(vector, entry.vector)
            if score > best_score:
                best_score, best = score, entry

        if best is None or best_score < self.threshold:
            self._misses += 1
            return None

        self._hits += 1
        best.hits += 1
        logger.info("语义缓存命中：score=%.4f question=%s", best_score, best.question[:40])
        return best

    def store(
        self,
        user_id: int,
        question: str,
        answer: str,
        vector: list[float],
        signature: str,
        context_id: str,
        numbers: tuple[str, ...] = (),
    ) -> None:
        now = self._clock()
        key = str(user_id)
        entries = self._store.setdefault(key, [])
        self._prune(user_id, now)

        if len(entries) >= self.max_entries:
            # 淘汰最久未命中、其次最旧的：简单但足以防止无限增长
            entries.sort(key=lambda e: (e.hits, e.created_at))
            entries.pop(0)

        entries.append(CacheEntry(
            question=question, answer=answer, vector=vector,
            signature=signature, context_id=context_id, created_at=now,
            numbers=numbers,
        ))

    # ---------- 维护 ----------

    def _prune(self, user_id: str | int, now: float) -> None:
        key = str(user_id)
        entries = self._store.get(key)
        if not entries:
            return
        alive = [e for e in entries if now - e.created_at <= self.ttl]
        if len(alive) != len(entries):
            self._store[key] = alive

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "entries": sum(len(v) for v in self._store.values()),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 4) if total else 0.0,
            "threshold": self.threshold,
        }

    def clear(self, user_id: int | None = None) -> None:
        if user_id is None:
            self._store.clear()
        else:
            self._store.pop(str(user_id), None)
