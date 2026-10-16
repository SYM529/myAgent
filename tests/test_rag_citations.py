"""RAG 引用的请求级隔离测试（阶段 5）。

覆盖四层：

1. **工具层**：引用写进当前请求的 ``RagRequestContext.citations``（原地修改），
   未命中与异常都清空本请求引用，无上下文时不创建任何共享状态；
2. **并发层**：两个请求各自的引用互不串扰（工具级 + SSE 级）；
3. **graph 层**：继续通过 state 返回 citations，不写 ctx（避免与工具路径重复）；
4. **SSE 层**：直接调用路由协程（真实 ``event_generator`` 代码），验证下发的
   引用来自**本请求**，并且请求结束后上下文被清理。

隔离方式：fake store / fake reranker / 假 ``agent.astream``；
``semantic_cache`` 被置空以避免任何 embedding 网络调用；
不做真实 HTTP、不打开真实 Chroma、不调用模型。
``server.main`` 的导入沿用既有 API 测试的假环境变量（先于导入设置）。
"""

import asyncio
import atexit
import json
import os
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("LLM_API_KEY", "test-key-not-real")
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")

_TMP_DB_DIR = Path(tempfile.mkdtemp(prefix="rag-citations-test-"))
os.environ["RAG_ENABLED"] = "off"
os.environ["VISION_ENABLED"] = "off"
os.environ["JWT_SECRET"] = "test-secret-not-real"
os.environ["DATABASE_URL"] = f"sqlite:///{(_TMP_DB_DIR / 'test.db').as_posix()}"
os.environ["REDIS_URL"] = "redis://127.0.0.1:1/0"


def _cleanup_tmp_db() -> None:
    from server.database import engine

    engine.dispose()
    shutil.rmtree(_TMP_DB_DIR, ignore_errors=True)


atexit.register(_cleanup_tmp_db)

from langchain_core.messages import ToolMessage  # noqa: E402

import agent.rag.tool as tool_module  # noqa: E402
import server.main as main_module  # noqa: E402
from agent.graph.nodes import knowledge_expert  # noqa: E402
from agent.rag.chunker import Chunk  # noqa: E402
from agent.rag.context import (  # noqa: E402
    RagRequestContext,
    get_context,
    reset_context,
    set_context,
)
from agent.rag.retriever import Retriever  # noqa: E402
from agent.rag.store import RetrievalHit  # noqa: E402
from agent.rag.tool import search_knowledge_base  # noqa: E402

ALICE = 6
BOB = 7

PUBLIC_DOC = "aaaa1111aaaa1111"
ALICE_DOC = "bbbb2222bbbb2222"
BOB_DOC = "cccc3333cccc3333"

PUBLIC_SOURCE = "公共手册.md"
ALICE_SOURCE = "Alice笔记.md"
BOB_SOURCE = "Bob笔记.md"

PUBLIC_MARKER = "PUBLIC-CONTENT"
ALICE_MARKER = "ALICE-CONTENT"
BOB_MARKER = "BOB-CONTENT"

QUERY = "CONTENT 说明 笔记 文档"


def make_chunk(doc_id: str, source: str, text: str, **meta) -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}-0",
        doc_id=doc_id,
        source=source,
        chunk_index=0,
        text=text,
        metadata=dict(meta),
    )


def fixture_chunks() -> list[Chunk]:
    return [
        make_chunk(PUBLIC_DOC, PUBLIC_SOURCE, f"{PUBLIC_MARKER} 公共说明", visibility="public"),
        make_chunk(
            ALICE_DOC, ALICE_SOURCE, f"{ALICE_MARKER} Alice 私有",
            visibility="private", owner=str(ALICE),
        ),
        make_chunk(
            BOB_DOC, BOB_SOURCE, f"{BOB_MARKER} Bob 私有",
            visibility="private", owner=str(BOB),
        ),
    ]


class FakeStore:
    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.scopes: list = []
        self.hits_enabled = True
        self.failure: Exception | None = None
        self.leak_doc_ids: set[str] | None = None

    def all_chunks(self):
        return list(self._chunks)

    def count(self):
        return len(self._chunks)

    def search(self, query, top_k=20, *, visible_doc_ids):
        self.scopes.append(visible_doc_ids)
        if self.failure is not None:
            raise self.failure
        if not self.hits_enabled:
            return []
        allowed = set(self.leak_doc_ids) if self.leak_doc_ids is not None else set(visible_doc_ids)
        hits = [
            RetrievalHit(chunk=chunk, score=1.0)
            for chunk in self._chunks
            if chunk.doc_id in allowed
        ]
        return hits[:top_k]


class FakeReranker:
    def rerank(self, query, chunks, top_n):
        return [(chunk, 0.9 - 0.01 * index) for index, chunk in enumerate(chunks)]


@pytest.fixture(autouse=True)
def _clean_context():
    reset_context()
    yield
    reset_context()


@pytest.fixture
def store():
    return FakeStore(fixture_chunks())


@pytest.fixture
def wired(monkeypatch, store):
    monkeypatch.setattr("agent.rag.get_store", lambda: store)
    monkeypatch.setattr(
        "agent.rag.get_retriever",
        lambda: Retriever(store, reranker=FakeReranker(), top_k=5),
    )
    return store


def sources(citations) -> set[str]:
    return {c["source"] for c in citations}


class TestToolCitations:
    def test_hit_writes_citations_into_request_context(self, wired):
        context = RagRequestContext(user_id=ALICE)
        set_context(context)

        search_knowledge_base.invoke({"query": QUERY})

        assert sources(context.citations) == {PUBLIC_SOURCE, ALICE_SOURCE}
        assert all({"index", "source", "heading", "score", "text"} <= set(c) for c in context.citations)
        assert get_context() is context

    def test_worker_thread_writes_back_to_parent_context(self, wired):
        async def scenario():
            context = RagRequestContext(user_id=ALICE)
            set_context(context)
            await asyncio.to_thread(
                lambda: search_knowledge_base.invoke({"query": QUERY})
            )
            return context

        context = asyncio.run(scenario())

        assert sources(context.citations) == {PUBLIC_SOURCE, ALICE_SOURCE}

    def test_miss_clears_previous_citations(self, wired):
        context = RagRequestContext(user_id=ALICE)
        set_context(context)
        search_knowledge_base.invoke({"query": QUERY})
        assert context.citations          # 先制造残留

        wired.hits_enabled = False
        output = search_knowledge_base.invoke({"query": QUERY})

        assert "没有检索到" in output
        assert context.citations == []

    def test_failure_clears_citations_and_keeps_error_desensitized(self, wired):
        context = RagRequestContext(user_id=ALICE)
        set_context(context)
        search_knowledge_base.invoke({"query": QUERY})
        assert context.citations

        wired.failure = RuntimeError(f"boom SECRET-TOKEN-abc {ALICE_MARKER}")
        output = search_knowledge_base.invoke({"query": QUERY})

        assert "知识库检索失败" in output and "RuntimeError" in output
        assert "SECRET-TOKEN-abc" not in output
        assert ALICE_MARKER not in output
        assert context.citations == []          # 失败事件不能带上次的引用

    def test_failure_of_one_request_does_not_touch_another(self, wired):
        bob_context = RagRequestContext(user_id=BOB)
        bob_context.citations.append({"source": BOB_SOURCE})
        alice_context = RagRequestContext(user_id=ALICE)
        set_context(alice_context)
        wired.failure = RuntimeError("boom")

        search_knowledge_base.invoke({"query": QUERY})

        assert alice_context.citations == []
        assert sources(bob_context.citations) == {BOB_SOURCE}

    def test_without_context_creates_no_shared_state(self, wired):
        reset_context()

        output = search_knowledge_base.invoke({"query": QUERY})

        assert PUBLIC_MARKER in output                       # 仍按 public-only 检索
        assert get_context() is None
        assert not hasattr(tool_module, "_last_citations")
        assert not hasattr(tool_module, "take_citations")

    def test_out_of_scope_candidates_never_enter_citations(self, wired):
        wired.leak_doc_ids = {PUBLIC_DOC, ALICE_DOC, BOB_DOC}
        context = RagRequestContext(user_id=ALICE)
        set_context(context)

        search_knowledge_base.invoke({"query": QUERY})

        assert sources(context.citations) == {PUBLIC_SOURCE, ALICE_SOURCE}
        assert BOB_SOURCE not in sources(context.citations)


class TestConcurrentToolCitations:
    def test_two_concurrent_requests_do_not_cross_citations(self, wired):
        async def request(user_id, delay):
            context = RagRequestContext(user_id=user_id)
            set_context(context)
            await asyncio.sleep(delay)          # 交错执行
            await asyncio.to_thread(
                lambda: search_knowledge_base.invoke({"query": QUERY})
            )
            return context

        async def scenario():
            return await asyncio.gather(request(ALICE, 0.01), request(BOB, 0.0))

        alice_context, bob_context = asyncio.run(scenario())

        assert sources(alice_context.citations) == {PUBLIC_SOURCE, ALICE_SOURCE}
        assert sources(bob_context.citations) == {PUBLIC_SOURCE, BOB_SOURCE}
        assert alice_context.citations is not bob_context.citations
        assert ALICE_SOURCE not in sources(bob_context.citations)


class TestGraphPath:
    def test_graph_returns_state_citations_without_writing_context(self, wired):
        context = RagRequestContext(user_id=ALICE)
        set_context(context)

        result = knowledge_expert({"question": QUERY})

        assert sources(result["citations"]) == {PUBLIC_SOURCE, ALICE_SOURCE}
        # graph 的引用走 state（SSE 用 event["citations"] 回退分支），
        # 不写 ctx.citations，避免与工具路径重复下发
        assert context.citations == []


def sse_payloads(chunks) -> list[dict]:
    payloads = []
    for chunk in chunks:
        text = chunk if isinstance(chunk, str) else chunk.decode("utf-8")
        for line in text.splitlines():
            if line.startswith("data: ") and "[DONE]" not in line:
                payloads.append(json.loads(line[len("data: "):]))
    return payloads


def citations_from(chunks) -> list[dict]:
    found = [payload["citations"] for payload in sse_payloads(chunks) if payload.get("citations")]
    assert len(found) == 1, f"应恰好有一条带引用的 SSE 事件，实际 {len(found)}"
    return found[0]


class FakeAgent:
    """假 astream：在**当前任务**里真的调用一次工具，再吐一条工具消息。"""

    def astream(self, payload, stream_mode="values"):
        async def generator():
            text = await asyncio.to_thread(
                lambda: search_knowledge_base.invoke({"query": QUERY})
            )
            yield {
                "messages": [
                    ToolMessage(
                        content=text, name="search_knowledge_base", tool_call_id="call-1"
                    )
                ]
            }

        return generator()


def make_request(session_id: str):
    return main_module.ChatRequest(message=QUERY, session_id=session_id)


def make_user(user_id: int, username: str):
    return SimpleNamespace(id=user_id, username=username, is_admin=False)


@pytest.fixture
def sse_env(monkeypatch, wired):
    """把 SSE 路径上的外部依赖全部替换掉：假 agent、无缓存、不碰 Redis 内容。"""
    monkeypatch.setattr(main_module, "agent", FakeAgent())
    monkeypatch.setattr(main_module, "semantic_cache", None)
    monkeypatch.setattr(main_module, "cache_embed", None)
    monkeypatch.setattr(main_module, "get_history", lambda *a, **k: [])
    monkeypatch.setattr(main_module, "save_history", lambda *a, **k: None)
    return wired


class TestSseCitations:
    def test_sse_reads_current_request_citations(self, sse_env):
        async def scenario():
            response = await main_module.agent_stream(
                make_request("session-a"), make_user(ALICE, "alice")
            )
            chunks = [chunk async for chunk in response.body_iterator]
            return chunks, get_context()

        chunks, context_after = asyncio.run(scenario())

        assert sources(citations_from(chunks)) == {PUBLIC_SOURCE, ALICE_SOURCE}
        # 请求结束后上下文被 finally 清理（同一任务内验证）
        assert context_after is None

    def test_concurrent_sse_streams_do_not_cross_citations(self, sse_env):
        async def drain(response):
            return [chunk async for chunk in response.body_iterator]

        async def scenario():
            alice = await main_module.agent_stream(
                make_request("session-a"), make_user(ALICE, "alice")
            )
            bob = await main_module.agent_stream(
                make_request("session-b"), make_user(BOB, "bob")
            )
            return await asyncio.gather(drain(alice), drain(bob))

        alice_chunks, bob_chunks = asyncio.run(scenario())

        assert sources(citations_from(alice_chunks)) == {PUBLIC_SOURCE, ALICE_SOURCE}
        assert sources(citations_from(bob_chunks)) == {PUBLIC_SOURCE, BOB_SOURCE}
        assert ALICE_SOURCE not in sources(citations_from(bob_chunks))
