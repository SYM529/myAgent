"""语义缓存测试。

这组测试守的不是"能不能命中"，而是**会不会误命中**。

缓存出错的代价远高于缓存没命中：没命中只是多花几分钱，
误命中是给用户一个看起来很自信的错答案。所以下面这些
"看似不该命中"的用例才是重点，每一条都对应一个真实的风险场景。
"""

import pytest

from server.cache import (
    SemanticCache, cosine_similarity, number_signature,
)


class FakeClock:
    """可推进的时钟，用来验证过期而不需要真等一小时。"""

    def __init__(self, start: float = 1_000_000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_cache(**kwargs) -> SemanticCache:
    kwargs.setdefault("threshold", 0.9)
    kwargs.setdefault("ttl", 3600)
    kwargs.setdefault("max_entries", 10)
    return SemanticCache(**kwargs)


class TestCosine:
    def test_identical_vectors(self):
        assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    @pytest.mark.parametrize("a,b", [
        ([], []),
        ([1.0], [1.0, 2.0]),     # 维度不一致
        ([0.0, 0.0], [1.0, 1.0]),  # 零向量
    ])
    def test_degenerate_inputs_return_zero(self, a, b):
        assert cosine_similarity(a, b) == 0.0


class TestHitAndMiss:
    def test_miss_when_cache_empty(self):
        cache = make_cache()
        assert cache.lookup(1, [1.0, 0.0], "sig", "") is None

    def test_hit_on_identical_question(self):
        cache = make_cache()
        cache.store(1, "今天几号", "周二", [1.0, 0.0], "sig", "")

        entry = cache.lookup(1, [1.0, 0.0], "sig", "")

        assert entry is not None
        assert entry.answer == "周二"

    def test_miss_when_similarity_below_threshold(self):
        cache = make_cache(threshold=0.95)
        cache.store(1, "北京今天天气", "晴", [1.0, 0.0], "sig", "")

        # 余弦 0.8 < 0.95：相似但不够相似，宁可多花一次钱也不能给错答案
        assert cache.lookup(1, [0.8, 0.6], "sig", "") is None


class TestNumberSignature:
    """数字校验：这是被标定数据逼出来的一道闸门。"""

    def test_extracts_numbers_in_sorted_order(self):
        assert number_signature("本金 10000 元，利率 5%") == ("10000", "5")

    def test_ignores_text_without_numbers(self):
        assert number_signature("什么是 ReAct") == ()

    def test_supports_decimals(self):
        assert number_signature("0.1 加 0.2") == ("0.1", "0.2")

    def test_detects_swapped_parameters(self):
        # 标定数据里风险最高的一对：语义结构几乎相同，只有数字不同，
        # 余弦相似度 0.9257，但答案分别是 175 元与 500 元。
        a = number_signature("本金 5000 元，年利率 3.5%，一年的利息是多少元")
        b = number_signature("10000 元按年利率 5% 存一年，利息是多少")
        assert a != b


class TestAntiPoisoning:
    """防误命中的四道闸：能力签名、上下文指纹、数字集合、用户隔离。"""

    def test_swapped_numbers_never_hit_even_at_high_similarity(self):
        cache = make_cache(threshold=0.80)   # 刻意用低阈值
        cache.store(
            1, "本金 5000 元，年利率 3.5%", "175 元", [1.0, 0.0],
            "sig", "", numbers=number_signature("本金 5000 元，年利率 3.5%"),
        )

        # 即使向量几乎一致、且阈值放宽到 0.80，数字不同也必须拒绝
        assert cache.lookup(
            1, [1.0, 0.0], "sig", "",
            numbers=number_signature("10000 元按年利率 5%"),
        ) is None

    def test_same_numbers_are_allowed(self):
        cache = make_cache()
        cache.store(1, "123 * 456", "56088", [1.0, 0.0], "sig", "",
                    numbers=number_signature("123 * 456"))

        assert cache.lookup(
            1, [1.0, 0.0], "sig", "", numbers=number_signature("123 乘 456")
        ) is not None

    def test_different_capability_signature_never_hits(self):
        cache = make_cache()
        cache.store(1, "你会做什么", "我能算数", [1.0, 0.0], "sig-old", "")

        # 改了提示词或增删了工具，旧答案就不再代表现在的行为
        assert cache.lookup(1, [1.0, 0.0], "sig-new", "") is None

    def test_different_context_never_hits(self):
        cache = make_cache()
        cache.store(1, "它多少钱", "20 元", [1.0, 0.0], "sig", "ctx-a")

        # 同一句话在另一段对话语境下指向别的东西
        assert cache.lookup(1, [1.0, 0.0], "sig", "ctx-b") is None

    def test_other_users_cache_is_invisible(self):
        cache = make_cache()
        cache.store(1, "我的偏好是什么", "用中文回答", [1.0, 0.0], "sig", "")

        # 缓存按用户隔离：一个人的回答绝不能出现在另一个人的回复里
        assert cache.lookup(2, [1.0, 0.0], "sig", "") is None


class TestExpiryAndEviction:
    def test_expired_entry_is_not_served(self):
        clock = FakeClock()
        cache = make_cache(ttl=60, clock=clock)
        cache.store(1, "你好", "你好！", [1.0, 0.0], "sig", "")

        assert cache.lookup(1, [1.0, 0.0], "sig", "") is not None

        clock.advance(61)
        assert cache.lookup(1, [1.0, 0.0], "sig", "") is None

    def test_oldest_least_used_entry_is_evicted(self):
        clock = FakeClock()
        cache = make_cache(max_entries=2, clock=clock)
        cache.store(1, "q1", "a1", [1.0, 0.0], "sig", "ctx-1")
        clock.advance(1)
        cache.store(1, "q2", "a2", [0.0, 1.0], "sig", "ctx-2")

        # 命中过 q1，它就有一次使用记录，应优先淘汰从未命中的 q2
        cache.lookup(1, [1.0, 0.0], "sig", "ctx-1")
        clock.advance(1)
        cache.store(1, "q3", "a3", [0.0, -1.0], "sig", "ctx-3")

        assert cache.lookup(1, [0.0, 1.0], "sig", "ctx-2") is None
        assert cache.lookup(1, [1.0, 0.0], "sig", "ctx-1") is not None

    def test_clear_removes_only_requested_user(self):
        cache = make_cache()
        cache.store(1, "q", "a", [1.0, 0.0], "sig", "")
        cache.store(2, "q", "a", [1.0, 0.0], "sig", "")

        cache.clear(1)

        assert cache.lookup(1, [1.0, 0.0], "sig", "") is None
        assert cache.lookup(2, [1.0, 0.0], "sig", "") is not None


class TestStats:
    def test_hit_rate_is_reported(self):
        cache = make_cache()
        cache.store(1, "q", "a", [1.0, 0.0], "sig", "")

        cache.lookup(1, [1.0, 0.0], "sig", "")   # 命中
        cache.lookup(2, [1.0, 0.0], "sig", "")   # 未命中（另一个用户）

        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == pytest.approx(0.5)

    def test_hit_rate_is_zero_when_no_traffic(self):
        # 除零保护：没有请求时不该抛 ZeroDivisionError
        assert make_cache().stats()["hit_rate"] == 0.0
