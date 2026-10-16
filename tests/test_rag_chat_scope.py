"""HTTP 聊天路径的检索范围接线测试（阶段 4）。

覆盖三件事：

1. ``scope_for_request()``：有效身份 → public + 本人 private；无上下文 / 无效
   ``user_id``（None、0、负数、非数字字符串、布尔值、其他类型）→ public-only；
2. ReAct 工具与 graph 节点**确实把该 scope 传给** ``Retriever.retrieve()``，
   因而越界内容在召回阶段就被挡住（而不是只过滤最终回答）；
3. 在 ``asyncio.to_thread``（langchain / langgraph 的真实执行方式）里，请求身份
   仍能正确形成 scope。

全部使用 fake store / fake reranker，不打开真实 Chroma、不调用模型、不读取 .env。
"""

import asyncio
import os

import pytest

os.environ.setdefault("LLM_API_KEY", "test-key-not-real")
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")

from agent.graph.nodes import knowledge_expert  # noqa: E402
from agent.rag import public_scope, scope_for_request, visible_doc_ids_for  # noqa: E402
from agent.rag.chunker import Chunk  # noqa: E402
from agent.rag.context import RagRequestContext, reset_context, set_context  # noqa: E402
from agent.rag.retriever import Retriever  # noqa: E402
from agent.rag.store import RetrievalHit  # noqa: E402
from agent.rag.tool import search_knowledge_base  # noqa: E402

ALICE = 6
BOB = 7
ADMIN = 1

PUBLIC_DOC = "aaaa1111aaaa1111"
ALICE_DOC = "bbbb2222bbbb2222"
BOB_DOC = "cccc3333cccc3333"
UNKNOWN_DOC = "dddd4444dddd4444"

PUBLIC_SOURCE = "公共手册.md"
ALICE_SOURCE = "Alice笔记.md"
BOB_SOURCE = "Bob笔记.md"
UNKNOWN_SOURCE = "历史文档.md"

PUBLIC_MARKER = "PUBLIC-CONTENT"
ALICE_MARKER = "ALICE-CONTENT"
BOB_MARKER = "BOB-CONTENT"
UNKNOWN_MARKER = "UNKNOWN-CONTENT"


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
            ALICE_DOC, ALICE_SOURCE, f"{ALICE_MARKER} Alice 的私有笔记",
            visibility="private", owner=str(ALICE),
        ),
        make_chunk(
            BOB_DOC, BOB_SOURCE, f"{BOB_MARKER} Bob 的私有笔记",
            visibility="private", owner=str(BOB),
        ),
        make_chunk(UNKNOWN_DOC, UNKNOWN_SOURCE, f"{UNKNOWN_MARKER} 历史无标记文档"),
    ]


class FakeStore:
    """记录每次检索收到的 scope；可注入越界候选，验证"召回前过滤"。"""

    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.scopes: list = []
        self.leak_doc_ids: set[str] | None = None

    def all_chunks(self):
        return list(self._chunks)

    def count(self):
        return len(self._chunks)

    def search(self, query, top_k=20, *, visible_doc_ids):
        self.scopes.append(visible_doc_ids)
        allowed = set(self.leak_doc_ids) if self.leak_doc_ids is not None else set(visible_doc_ids)
        hits = [
            RetrievalHit(chunk=chunk, score=1.0)
            for chunk in self._chunks
            if chunk.doc_id in allowed
        ]
        return hits[:top_k]


class FakeReranker:
    def __init__(self):
        self.received: list[Chunk] = []

    def rerank(self, query, chunks, top_n):
        self.received = list(chunks)
        return [(chunk, 0.9 - 0.01 * index) for index, chunk in enumerate(chunks)]


@pytest.fixture(autouse=True)
def _clean_context_and_globals():
    reset_context()
    yield
    reset_context()


@pytest.fixture
def store():
    return FakeStore(fixture_chunks())


@pytest.fixture
def reranker():
    return FakeReranker()


@pytest.fixture
def wired(monkeypatch, store, reranker):
    """让工具与节点里的 get_retriever() 返回使用 fake store 的 Retriever。"""
    # 同时替换 get_store：任何"默认取 store"的代码路径都不许碰到真实 rag_store
    monkeypatch.setattr("agent.rag.get_store", lambda: store)
    monkeypatch.setattr(
        "agent.rag.get_retriever",
        lambda: Retriever(store, reranker=reranker, top_k=5),
    )
    return store


def as_user(user_id, *, is_admin=False):
    set_context(RagRequestContext(user_id=user_id, is_admin=is_admin))


class TestScopeForRequest:
    def test_valid_user_gets_public_and_own_private(self, store):
        as_user(ALICE)

        assert scope_for_request(store=store) == frozenset({PUBLIC_DOC, ALICE_DOC})

    def test_other_users_private_is_excluded(self, store):
        as_user(BOB)

        scope = scope_for_request(store=store)
        assert scope == frozenset({PUBLIC_DOC, BOB_DOC})
        assert ALICE_DOC not in scope

    def test_admin_is_not_widened(self, store):
        as_user(ADMIN, is_admin=True)

        scope = scope_for_request(store=store)
        assert scope == frozenset({PUBLIC_DOC})
        assert ALICE_DOC not in scope and BOB_DOC not in scope

    def test_no_context_is_public_only(self, store):
        reset_context()

        assert scope_for_request(store=store) == public_scope(store=store)

    def test_unknown_document_is_never_in_scope(self, store):
        for user_id, is_admin in ((ALICE, False), (BOB, False), (ADMIN, True)):
            as_user(user_id, is_admin=is_admin)
            assert UNKNOWN_DOC not in scope_for_request(store=store)

    @pytest.mark.parametrize("bad_user_id", [
        None, 0, -1, "abc", True, False, 3.5, [], {}, "",
    ])
    def test_invalid_user_id_falls_back_to_public(self, store, bad_user_id):
        as_user(bad_user_id, is_admin=True)          # 连管理员也不放宽

        assert scope_for_request(store=store) == public_scope(store=store)

    def test_numeric_string_user_id_is_accepted(self, store):
        as_user(str(ALICE))

        assert scope_for_request(store=store) == frozenset({PUBLIC_DOC, ALICE_DOC})

    def test_missing_context_object_is_public_only(self, monkeypatch, store):
        # 上下文存在但 user_id 字段缺失（None）→ public
        set_context(RagRequestContext())

        assert scope_for_request(store=store) == public_scope(store=store)


class TestToolUsesScope:
    def test_tool_passes_scope_into_retrieve(self, wired, monkeypatch):
        captured: list = []
        original = Retriever.retrieve

        def spy(self, query, *, scope=None):
            captured.append(scope)
            return original(self, query, scope=scope)

        monkeypatch.setattr(Retriever, "retrieve", spy)
        as_user(ALICE)

        search_knowledge_base.invoke({"query": "CONTENT 说明 笔记 文档"})

        assert captured == [visible_doc_ids_for(ALICE, store=wired)]
        assert wired.scopes[-1] == visible_doc_ids_for(ALICE, store=wired)

    def test_alice_sees_public_and_own_private_only(self, wired):
        as_user(ALICE)

        output = search_knowledge_base.invoke({"query": "CONTENT 说明 笔记 文档"})

        assert PUBLIC_MARKER in output and ALICE_MARKER in output
        for leaked in (BOB_MARKER, UNKNOWN_MARKER, BOB_SOURCE, UNKNOWN_SOURCE):
            assert leaked not in output

    def test_bob_cannot_see_alices_private(self, wired):
        as_user(BOB)

        output = search_knowledge_base.invoke({"query": "CONTENT 说明 笔记 文档"})

        assert ALICE_MARKER not in output
        assert ALICE_SOURCE not in output
        assert BOB_MARKER in output

    def test_admin_cannot_see_others_private(self, wired):
        as_user(ADMIN, is_admin=True)

        output = search_knowledge_base.invoke({"query": "CONTENT 说明 笔记 文档"})

        assert PUBLIC_MARKER in output
        for leaked in (ALICE_MARKER, BOB_MARKER, UNKNOWN_MARKER):
            assert leaked not in output

    def test_without_context_tool_is_public_only(self, wired):
        reset_context()

        output = search_knowledge_base.invoke({"query": "CONTENT 说明 笔记 文档"})

        assert PUBLIC_MARKER in output
        for leaked in (ALICE_MARKER, BOB_MARKER, UNKNOWN_MARKER):
            assert leaked not in output

    def test_tool_scope_survives_worker_thread(self, wired):
        async def scenario():
            as_user(ALICE)
            return await asyncio.to_thread(
                lambda: search_knowledge_base.invoke({"query": "CONTENT 说明 笔记 文档"})
            )

        output = asyncio.run(scenario())

        assert wired.scopes[-1] == visible_doc_ids_for(ALICE, store=wired)
        assert PUBLIC_MARKER in output and ALICE_MARKER in output
        assert BOB_MARKER not in output

    def test_injected_out_of_scope_hits_never_reach_final_output(
        self, wired, reranker
    ):
        # 模拟上游 bug：store 返回了越界候选 → 精排前的二次过滤必须拦住
        wired.leak_doc_ids = {PUBLIC_DOC, ALICE_DOC, BOB_DOC, UNKNOWN_DOC}
        as_user(ALICE)

        output = search_knowledge_base.invoke({"query": "CONTENT 说明 笔记 文档"})

        assert {chunk.doc_id for chunk in reranker.received} <= {PUBLIC_DOC, ALICE_DOC}
        for leaked in (BOB_MARKER, UNKNOWN_MARKER, BOB_SOURCE, UNKNOWN_SOURCE):
            assert leaked not in output


class TestGraphNodeUsesScope:
    def test_node_passes_scope_into_retrieve(self, wired, monkeypatch):
        captured: list = []
        original = Retriever.retrieve

        def spy(self, query, *, scope=None):
            captured.append(scope)
            return original(self, query, scope=scope)

        monkeypatch.setattr(Retriever, "retrieve", spy)
        as_user(ALICE)

        knowledge_expert({"question": "CONTENT 说明 笔记 文档"})

        assert captured == [visible_doc_ids_for(ALICE, store=wired)]

    def test_node_output_stays_inside_scope(self, wired):
        as_user(ALICE)

        result = knowledge_expert({"question": "CONTENT 说明 笔记 文档"})

        assert PUBLIC_MARKER in result["knowledge_context"]
        assert ALICE_MARKER in result["knowledge_context"]
        for leaked in (BOB_MARKER, UNKNOWN_MARKER):
            assert leaked not in result["knowledge_context"]
        assert {c["source"] for c in result["citations"]} == {
            PUBLIC_SOURCE, ALICE_SOURCE
        }

    def test_node_admin_sees_only_public(self, wired):
        as_user(ADMIN, is_admin=True)

        result = knowledge_expert({"question": "CONTENT 说明 笔记 文档"})

        assert {c["source"] for c in result["citations"]} == {PUBLIC_SOURCE}

    def test_node_without_context_is_public_only(self, wired):
        reset_context()

        result = knowledge_expert({"question": "CONTENT 说明 笔记 文档"})

        assert {c["source"] for c in result["citations"]} == {PUBLIC_SOURCE}
        assert BOB_MARKER not in result["knowledge_context"]

    def test_node_with_injected_out_of_scope_hits(self, wired, reranker):
        wired.leak_doc_ids = {PUBLIC_DOC, BOB_DOC}
        as_user(ALICE)

        result = knowledge_expert({"question": "CONTENT 说明 笔记 文档"})

        assert {chunk.doc_id for chunk in reranker.received} <= {PUBLIC_DOC, ALICE_DOC}
        assert {c["source"] for c in result["citations"]} == {PUBLIC_SOURCE}
