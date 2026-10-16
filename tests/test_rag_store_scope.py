"""`HybridStore.search()` 检索范围隔离的测试。

全部使用 fake collection / fake embedding（`chromadb.PersistentClient` 被替换），
不打开真实 Chroma、不触碰真实 rag_store。断言的是"未过滤结果不可能进入候选"，
而不只是"最终返回看起来没问题"。
"""

import chromadb
import pytest

from agent.rag.chunker import Chunk
from agent.rag.store import HybridStore

PUBLIC_DOC = "aaaa1111aaaa1111"
PRIVATE_DOC = "bbbb2222bbbb2222"


class FakeEmbedding:
    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_one(self, text):
        self.calls += 1
        return [0.1, 0.2, 0.3]


class FakeCollection:
    """记录 query 的调用与 where 参数；upsert 按键合并（与真实 Chroma 一致）。"""

    def __init__(self, name):
        self.name = name
        self.rows: dict[str, dict] = {}
        self.query_calls: list[dict] = []
        self.query_result: dict = {"ids": [[]]}

    def upsert(self, ids, documents, embeddings, metadatas):
        for index, chunk_id in enumerate(ids):
            merged = dict(self.rows.get(chunk_id, {}))
            merged.update(metadatas[index])
            self.rows[chunk_id] = merged

    def query(self, query_embeddings=None, n_results=None, where=None):
        self.query_calls.append({"n_results": n_results, "where": where})
        return self.query_result

    def get(self, where=None, include=None):
        doc_id = (where or {}).get("doc_id")
        ids = [cid for cid, meta in self.rows.items() if meta.get("doc_id") == doc_id]
        return {"ids": ids, "metadatas": [self.rows[cid] for cid in ids]}

    def delete(self, ids):
        for chunk_id in ids:
            self.rows.pop(chunk_id, None)


class FakeChromaClient:
    def __init__(self):
        self.collections: dict[str, FakeCollection] = {}

    def get_or_create_collection(self, name, metadata=None):
        return self.collections.setdefault(name, FakeCollection(name))

    def delete_collection(self, name):
        self.collections.pop(name, None)


@pytest.fixture
def chroma(monkeypatch):
    client = FakeChromaClient()
    monkeypatch.setattr(chromadb, "PersistentClient", lambda **kwargs: client)
    return client


@pytest.fixture
def embedding():
    return FakeEmbedding()


@pytest.fixture
def store(tmp_path, chroma, embedding):
    store = HybridStore(persist_dir=tmp_path, embedding_client=embedding)
    store.add([
        chunk(PUBLIC_DOC, 0, text="公开文档 项目说明"),
        chunk(PUBLIC_DOC, 1, text="公开文档 部署说明"),
    ])
    store.add([
        chunk(PRIVATE_DOC, 0, text="私有文档 密钥配置 内部口令"),
    ])
    embedding.calls = 0          # 只统计检索阶段的调用
    return store


def chunk(doc_id, index, text="内容", **extra):
    metadata = {"visibility": "public"} if doc_id == PUBLIC_DOC else {
        "visibility": "private", "owner": "6"
    }
    metadata.update(extra)
    return Chunk(
        chunk_id=f"{doc_id}-{index}",
        doc_id=doc_id,
        source=f"{doc_id}.md",
        chunk_index=index,
        text=text,
        metadata=metadata,
    )


class TestEmptyScope:
    def test_empty_scope_returns_nothing_without_touching_backends(
        self, store, chroma, embedding
    ):
        assert store.search("项目说明", visible_doc_ids=frozenset()) == []
        assert embedding.calls == 0
        assert chroma.collections["knowledge_base"].query_calls == []

    def test_none_scope_is_fail_closed(self, store, chroma, embedding):
        # 显式传 None 也必须是"什么都搜不到"，而不是全库
        assert store.search("项目说明", visible_doc_ids=None) == []
        assert embedding.calls == 0
        assert chroma.collections["knowledge_base"].query_calls == []

    def test_scope_without_matching_chunks_skips_backends(self, store, chroma, embedding):
        assert store.search("项目说明", visible_doc_ids=frozenset({"9999ffff9999ffff"})) == []
        assert embedding.calls == 0
        assert chroma.collections["knowledge_base"].query_calls == []


class TestVectorFiltering:
    def test_where_pushes_down_the_allowed_doc_ids(self, store, chroma, embedding):
        store.search("项目说明", visible_doc_ids=frozenset({PUBLIC_DOC, PRIVATE_DOC}))

        call = chroma.collections["knowledge_base"].query_calls[-1]
        assert call["where"] == {
            "doc_id": {"$in": sorted([PUBLIC_DOC, PRIVATE_DOC])}
        }

    def test_out_of_scope_ids_from_vector_store_are_dropped(self, store, chroma, embedding):
        # 模拟向量库返回范围外 id（跨进程写入/脏数据）：不许进入排名
        chroma.collections["knowledge_base"].query_result = {
            "ids": [[f"{PRIVATE_DOC}-0", f"{PUBLIC_DOC}-0"]]
        }

        hits = store.search("项目说明", visible_doc_ids=frozenset({PUBLIC_DOC}))

        assert {h.chunk.doc_id for h in hits} == {PUBLIC_DOC}


class TestBm25Filtering:
    def test_invisible_high_score_chunk_does_not_take_quota(
        self, store, chroma, embedding
    ):
        # 让不可见文档成为 BM25 最高分（关键词只出现在它里面），向量侧返回空
        chroma.collections["knowledge_base"].query_result = {"ids": [[]]}

        hits = store.search("内部口令", top_k=2, visible_doc_ids=frozenset({PUBLIC_DOC}))

        assert hits
        assert {h.chunk.doc_id for h in hits} == {PUBLIC_DOC}
        assert all("内部口令" not in h.chunk.text for h in hits)

    def test_visible_chunks_fill_the_quota(self, store, chroma, embedding):
        chroma.collections["knowledge_base"].query_result = {"ids": [[]]}

        hits = store.search("公开文档", top_k=2, visible_doc_ids=frozenset({PUBLIC_DOC}))

        assert {h.chunk.doc_id for h in hits} == {PUBLIC_DOC}
        assert len(hits) == 2          # 配额没有被不可见文档占掉


class TestScopeContract:
    def test_missing_scope_raises_type_error(self, store):
        with pytest.raises(TypeError):
            store.search("项目说明")

    @pytest.mark.parametrize("scope", [frozenset({PUBLIC_DOC}), frozenset({PUBLIC_DOC, PRIVATE_DOC})])
    def test_all_hits_are_inside_scope(self, store, chroma, embedding, scope):
        chroma.collections["knowledge_base"].query_result = {
            "ids": [[f"{PRIVATE_DOC}-0", f"{PUBLIC_DOC}-0", f"{PUBLIC_DOC}-1"]]
        }

        hits = store.search("文档", top_k=5, visible_doc_ids=scope)

        assert hits
        assert {h.chunk.doc_id for h in hits} <= set(scope)
