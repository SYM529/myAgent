"""请求级 RAG Context 的测试。

验证两件事：

1. **身份读得到**：工作线程（``asyncio.to_thread`` 与
   ``run_in_executor(..., copy_context().run, ...)``——与 langchain / langgraph
   的真实执行方式一致）里能读到请求任务设置的身份；
2. **引用不串号**：citations 是父任务创建的对象，工具在工作线程里原地修改，
   两个并发请求各写各的，且不存在"默认共享容器"。

反向约束也在这里锁住：线程内的 ``set()`` **不会**回传父任务——这正是
引用必须靠"共享同一个 list 对象"而不是"重新绑定变量"传递的原因。
"""

import asyncio
import contextvars

import pytest

from agent.rag.context import (
    RagRequestContext,
    get_context,
    reset_context,
    set_context,
)


@pytest.fixture(autouse=True)
def _clean_context():
    reset_context()
    yield
    reset_context()


class TestDefaults:
    def test_without_context_get_returns_none(self):
        # 未设置时必须是 None：调用方据此走 public-only 的安全默认
        assert get_context() is None

    def test_reset_clears_context(self):
        set_context(RagRequestContext(user_id=6))

        reset_context()

        assert get_context() is None

    def test_fresh_context_has_no_identity_and_no_admin(self):
        context = RagRequestContext()

        assert context.user_id is None
        assert context.is_admin is False
        assert context.citations == []

    def test_contexts_do_not_share_citations_container(self):
        first, second = RagRequestContext(), RagRequestContext()

        first.citations.append({"source": "a"})

        assert second.citations == []
        assert first.citations is not second.citations


class TestWorkerThreadVisibility:
    def test_context_is_readable_inside_to_thread(self):
        async def scenario():
            set_context(RagRequestContext(user_id=6))
            return await asyncio.to_thread(lambda: get_context().user_id)

        assert asyncio.run(scenario()) == 6

    def test_context_is_readable_inside_run_in_executor_with_copied_context(self):
        """复刻 langchain run_in_executor 的实现（copy_context().run 提交线程池）。"""

        async def scenario():
            set_context(RagRequestContext(user_id=7, is_admin=True))
            loop = asyncio.get_running_loop()
            copied = contextvars.copy_context()
            return await loop.run_in_executor(
                None,
                copied.run,
                lambda: (get_context().user_id, get_context().is_admin),
            )

        assert asyncio.run(scenario()) == (7, True)

    def test_binding_set_inside_worker_thread_does_not_leak_to_parent(self):
        # 线程内 set() 只影响副本：引用不能靠"重新绑定"回传，必须原地修改对象
        async def scenario():
            set_context(RagRequestContext(user_id=6))
            await asyncio.to_thread(
                lambda: set_context(RagRequestContext(user_id=99))
            )
            return get_context().user_id

        assert asyncio.run(scenario()) == 6

    def test_in_place_mutation_inside_worker_thread_is_visible_to_parent(self):
        async def scenario():
            context = RagRequestContext(user_id=6)
            set_context(context)

            def write_in_worker():          # 模拟工具在线程里的写法
                context.citations.clear()
                context.citations.extend([{"source": "手册.md"}])

            await asyncio.to_thread(write_in_worker)
            return context.citations

        assert asyncio.run(scenario()) == [{"source": "手册.md"}]


class TestConcurrencyIsolation:
    def test_concurrent_requests_see_their_own_identity(self):
        async def worker(user_id, delay):
            set_context(RagRequestContext(user_id=user_id))
            await asyncio.sleep(delay)
            return await asyncio.to_thread(lambda: get_context().user_id)

        async def scenario():
            return await asyncio.gather(worker(6, 0.01), worker(7, 0.0))

        assert asyncio.run(scenario()) == [6, 7]

    def test_concurrent_requests_do_not_share_citations(self):
        async def worker(user_id, delay, source):
            context = RagRequestContext(user_id=user_id)
            set_context(context)
            await asyncio.sleep(delay)      # 交错执行

            def write_in_worker():
                context.citations.clear()
                context.citations.extend([{"source": source}])

            await asyncio.to_thread(write_in_worker)
            return context

        async def scenario():
            return await asyncio.gather(
                worker(6, 0.01, "A的文档"), worker(7, 0.0, "B的文档")
            )

        alice, bob = asyncio.run(scenario())

        assert alice.citations == [{"source": "A的文档"}]
        assert bob.citations == [{"source": "B的文档"}]
        assert alice.citations is not bob.citations

    def test_interleaved_misses_do_not_resurrect_old_citations(self):
        """未命中必须清空：A 先命中、随后同请求未命中，sink 里不能留旧引用。"""
        async def scenario():
            context = RagRequestContext(user_id=6)
            set_context(context)
            context.citations.extend([{"source": "旧引用"}])

            await asyncio.to_thread(context.citations.clear)   # 未命中的写法
            return context.citations

        assert asyncio.run(scenario()) == []
