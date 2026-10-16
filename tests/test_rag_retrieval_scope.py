"""`Retriever.retrieve()` 范围隔离与二次防御的测试。

用 fake store + fake reranker：既能断言"最终结果干净"，也能直接断言
"reranker 收到的输入里就没有越界 chunk"——后者才是真正的安全边界。

另外专门模拟"上游返回越界命中"（fake store 故意多返回），验证 Retriever
在精排之前的二次过滤确实生效，而不是依赖存储层永远正确。
"""

import pytest

from agent.rag import all_documents_scope, public_scope, visible_doc_ids_for
from agent.rag.chunker import Chunk
from agent.rag.retriever import Retriever
from agent.rag.store import RetrievalHit

ALICE = 6
BOB = 7
ADMIN = 1

PUBLIC = "aaaa1111aaaa1111"
ALICE_DOC = "bbbb2222bbbb2222"
BOB_DOC = "cccc3333cccc3333"
UNKNOWN = "dddd4444dddd4444"


def make_chunk(doc_id: str, index: int = 0, **meta) -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}-{index}",
        doc_id=doc_id,
        source=f"{doc_id}.md",
        chunk_index=index,
        text=f"{doc_id} 的内容片段 {index}",
        metadata=dict(meta),
    )


CHUNKS = (
    [make_chunk(PUBLIC, 0, visibility="public"), make_chunk(PUBLIC, 1, visibility="public")]
    + [make_chunk(ALICE_DOC, 0, visibility="private", owner=str(ALICE))]
    + [make_chunk(BOB_DOC, 0, visibility="private", owner=str(BOB))]
    + [make_chunk(UNKNOWN, 0)]          # 历史 unknown：无 owner、无 visibility
)


class FakeStore:
    """只实现检索所需接口，并记录收到的 scope。"""

    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.searches: list[dict] = []
        self.leak_doc_ids: set[str] | None = None   # 模拟上游返回越界命中

    def all_chunks(self):
        return list(self._chunks)

    def count(self):
        return len(self._chunks)

    def search(self, query, top_k=20, *, visible_doc_ids):
        self.searches.append(
            {"query": query, "top_k": top_k, "visible_doc_ids": visible_doc_ids}
        )
        allowed = set(self.leak_doc_ids) if self.leak_doc_ids is not None else set(visible_doc_ids)
        hits = [
            RetrievalHit(chunk=chunk, score=1.0)
            for chunk in self._chunks
            if chunk.doc_id in allowed
        ]
        return hits[:top_k]


class FakeReranker:
    def __init__(self, zero_scores: bool = False):
        self.received: list[Chunk] = []
        self.zero_scores = zero_scores

    def rerank(self, query, chunks, top_n):
        self.received = list(chunks)
        if self.zero_scores:
            return [(chunk, 0.0) for chunk in chunks]
        return [(chunk, 0.9 - 0.1 * index) for index, chunk in enumerate(chunks)]


@pytest.fixture
def store():
    return FakeStore(CHUNKS)


def retriever_for(store, reranker=None) -> Retriever:
    return Retriever(store, reranker=reranker or FakeReranker(), top_k=4)


def sources_of(result) -> set[str]:
    return {citation.source for citation in result.citations}


class TestScopeComputation:
    def test_alice_sees_public_and_own_private(self, store):
        assert visible_doc_ids_for(ALICE, store=store) == frozenset({PUBLIC, ALICE_DOC})

    def test_bob_cannot_see_alices_private(self, store):
        scope = visible_doc_ids_for(BOB, store=store)

        assert ALICE_DOC not in scope
        assert scope == frozenset({PUBLIC, BOB_DOC})

    def test_admin_sees_only_public(self, store):
        assert visible_doc_ids_for(ADMIN, is_admin=True, store=store) == frozenset({PUBLIC})

    def test_unknown_and_private_are_not_in_public_scope(self, store):
        assert public_scope(store=store) == frozenset({PUBLIC})

    def test_all_documents_scope_is_explicit_opt_in(self, store):
        everything = all_documents_scope(store=store)

        assert everything == frozenset({PUBLIC, ALICE_DOC, BOB_DOC, UNKNOWN})
        assert everything != public_scope(store=store)


class TestRetrievalScope:
    def test_alice_retrieves_public_and_own_private(self, store):
        reranker = FakeReranker()

        result = retriever_for(store, reranker).retrieve(
            "问题", scope=visible_doc_ids_for(ALICE, store=store)
        )

        assert result.hit
        assert sources_of(result) == {f"{PUBLIC}.md", f"{ALICE_DOC}.md"}
        assert {chunk.doc_id for chunk in reranker.received} == {PUBLIC, ALICE_DOC}

    def test_bob_cannot_retrieve_alices_private(self, store):
        result = retriever_for(store).retrieve(
            "问题", scope=visible_doc_ids_for(BOB, store=store)
        )

        assert f"{ALICE_DOC}.md" not in sources_of(result)
        assert ALICE_DOC not in result.context

    @pytest.mark.parametrize("user_id", [ALICE, BOB])
    def test_unknown_is_never_retrieved(self, store, user_id):
        result = retriever_for(store).retrieve(
            "问题", scope=visible_doc_ids_for(user_id, store=store)
        )

        assert UNKNOWN not in result.context
        assert f"{UNKNOWN}.md" not in sources_of(result)

    def test_reranker_input_has_no_out_of_scope_chunk(self, store):
        # 模拟上游 bug：store 返回了越界的 BOB 文档
        store.leak_doc_ids = {PUBLIC, ALICE_DOC, BOB_DOC}
        reranker = FakeReranker()

        result = retriever_for(store, reranker).retrieve(
            "问题", scope=visible_doc_ids_for(ALICE, store=store)
        )

        assert {chunk.doc_id for chunk in reranker.received} <= {PUBLIC, ALICE_DOC}
        assert f"{BOB_DOC}.md" not in sources_of(result)

    def test_context_and_citations_stay_inside_scope(self, store):
        store.leak_doc_ids = {PUBLIC, ALICE_DOC, BOB_DOC, UNKNOWN}

        result = retriever_for(store).retrieve(
            "问题", scope=visible_doc_ids_for(ALICE, store=store)
        )

        assert BOB_DOC not in result.context
        assert UNKNOWN not in result.context
        assert sources_of(result) == {f"{PUBLIC}.md", f"{ALICE_DOC}.md"}

    def test_zero_score_fallback_is_scope_safe(self, store):
        # 精排未配置/失败时全部返回 0，走的是"退回融合分"的降级分支
        store.leak_doc_ids = {PUBLIC, BOB_DOC}
        reranker = FakeReranker(zero_scores=True)

        result = retriever_for(store, reranker).retrieve(
            "问题", scope=visible_doc_ids_for(ALICE, store=store)
        )

        assert result.hit
        assert sources_of(result) == {f"{PUBLIC}.md"}

    def test_default_scope_is_public_only(self, store):
        result = retriever_for(store).retrieve("问题")

        assert store.searches[-1]["visible_doc_ids"] == public_scope(store=store)
        assert sources_of(result) == {f"{PUBLIC}.md"}

    def test_scope_is_passed_down_to_store(self, store):
        scope = visible_doc_ids_for(ALICE, store=store)

        retriever_for(store).retrieve("问题", scope=scope)

        assert store.searches[-1]["visible_doc_ids"] == scope
