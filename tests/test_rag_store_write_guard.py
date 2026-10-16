"""`HybridStore.add` 写入前两道闸门的测试。

守卫的意义是"错误归属根本进不来"，所以每个拒绝用例都要断言
embedding、Chroma upsert 与 chunks.json **都没有被动过**，
而不只是断言抛了异常。

两道闸门：
1. 逐 chunk 完整性——每个 chunk 都要带明确合法的 visibility，
   public 不许带 owner，private 必须每块带同一个有效 owner；
2. 不许静默覆盖——已存在同 doc_id 的归属（private/unknown）时，
   新批次不得改写它，两侧存储不一致也必须拒绝。

另外这里还验证真实的两条写入路径产出的 metadata 能顺利通过守卫——
否则守卫会把正常上传也挡在门外。
"""

import json
from pathlib import Path

import chromadb
import pytest

from agent.rag.chunker import Chunk
from agent.rag.store import HybridStore
from server import uploads


class FakeEmbedding:
    """不发网络请求的 embedding 假实现，同时记录调用次数。"""

    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_one(self, text):
        return [0.1, 0.2, 0.3]


class FakeCollection:
    """最小可用的 Chroma collection 假实现。

    upsert 刻意模拟真实 Chroma 的语义：命中已有 id 时**按键合并**——
    写到的键被覆盖，没写到的旧键会保留（依据：chromadb
    segment/impl/metadata/sqlite.py 的 _update_metadata 只删除
    "新值显式为 None" 的键）。这样测试才不会被假实现的替换语义骗过。
    """

    def __init__(self, name):
        self.name = name
        self.rows: dict[str, dict] = {}
        self.upsert_calls = 0

    def upsert(self, ids, documents, embeddings, metadatas):
        self.upsert_calls += 1
        for index, chunk_id in enumerate(ids):
            merged = dict(self.rows.get(chunk_id, {}))
            merged.update({k: v for k, v in metadatas[index].items() if v is not None})
            self.rows[chunk_id] = merged

    def get(self, where=None, include=None):
        doc_id = (where or {}).get("doc_id")
        ids = [cid for cid, meta in self.rows.items() if meta.get("doc_id") == doc_id]
        return {"ids": ids, "metadatas": [self.rows[cid] for cid in ids]}

    def delete(self, ids):
        for chunk_id in ids:
            self.rows.pop(chunk_id, None)


class FakeChromaClient:
    """替代 chromadb.PersistentClient：不建目录、不落 sqlite。"""

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
    return HybridStore(persist_dir=tmp_path, embedding_client=embedding)


def chunk(doc_id: str, index: int, **meta) -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}-{index}",
        doc_id=doc_id,
        source="手册.md",
        chunk_index=index,
        text=f"第 {index} 段",
        metadata=dict(meta),
    )


def write_doc(directory: Path, name: str = "笔记.md") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text("# 标题\n\n正文内容。\n", encoding="utf-8")
    return path


def seed_legacy(
    persist: Path,
    chroma,
    doc_id: str = "bbbb2222bbbb2222",
    metas: list | None = None,
    *,
    json_side: bool = True,
    chroma_side: bool = True,
    source: str = "历史文档.md",
) -> None:
    """直接落盘模拟历史数据（绕过写入闸门——这正是要复现的场景）。"""
    metas = metas if metas is not None else [{}, {}]
    payload = [
        {
            "chunk_id": f"{doc_id}-{index}",
            "doc_id": doc_id,
            "source": source,
            "chunk_index": index,
            "text": f"第 {index} 段",
            "metadata": dict(meta),
        }
        for index, meta in enumerate(metas)
    ]
    if json_side:
        persist.mkdir(parents=True, exist_ok=True)
        (persist / "chunks.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
    if chroma_side:
        collection = chroma.get_or_create_collection("knowledge_base")
        for item in payload:
            collection.rows[item["chunk_id"]] = {
                "source": item["source"],
                "doc_id": doc_id,
                "chunk_index": item["chunk_index"],
                **item["metadata"],
            }


class WriteProbe:
    """记录"写入面"：embedding 调用数、Chroma upsert 次数、chunks.json 字节。

    拒绝路径必须三个面都不动——只断言"抛了异常"是不够的。
    """

    def __init__(self, store, chroma, embedding, tmp_path):
        self._embedding = embedding
        self._collection = chroma.collections["knowledge_base"]
        self._path = Path(tmp_path) / "chunks.json"
        self.embed_calls = embedding.calls
        self.upsert_calls = self._collection.upsert_calls
        self.json_text = (
            self._path.read_text(encoding="utf-8") if self._path.exists() else None
        )

    def assert_unchanged(self):
        assert self._embedding.calls == self.embed_calls, "embedding 不应被调用"
        assert self._collection.upsert_calls == self.upsert_calls, "Chroma 不应被写入"
        current = (
            self._path.read_text(encoding="utf-8") if self._path.exists() else None
        )
        assert current == self.json_text, "chunks.json 不应变化"


def patch_chroma_get(chroma, response):
    """把 FakeCollection.get 换成固定返回/固定异常，模拟坏数据与查询失败。"""
    collection = chroma.get_or_create_collection("knowledge_base")

    if isinstance(response, Exception):
        def raising_get(where=None, include=None):
            raise response

        collection.get = raising_get
    else:
        collection.get = lambda where=None, include=None: response


class TestAddGuard:
    def test_private_document_is_accepted(self, store, chroma, tmp_path):
        written = store.add([chunk("aaaa1111aaaa1111", 0, owner="6", visibility="private")])

        assert written == 1
        assert store.count() == 1
        assert (tmp_path / "chunks.json").exists()
        assert chroma.collections["knowledge_base"].rows

    def test_public_document_is_accepted(self, store, chroma):
        written = store.add([chunk("aaaa1111aaaa1111", 0, visibility="public")])

        assert written == 1
        assert chroma.collections["knowledge_base"].rows

    def test_unmarked_document_is_rejected(self, store, chroma, tmp_path):
        with pytest.raises(ValueError, match="拒绝写入"):
            store.add([chunk("aaaa1111aaaa1111", 0)])

        assert store.count() == 0
        assert not (tmp_path / "chunks.json").exists()
        assert chroma.collections["knowledge_base"].rows == {}

    @pytest.mark.parametrize("meta", [
        {"visibility": "private"},                  # private 但没有 owner
        {"visibility": "shared"},                   # 不是合法标记
        {"owner": "6"},                             # 只有 owner：读路径兼容，写路径必须声明
        {"visibility": "private", "owner": "abc"},  # owner 写坏了
        {"visibility": "public", "owner": "6"},     # public 不该带 owner
    ])
    def test_invalid_metadata_is_rejected(self, store, chroma, tmp_path, meta):
        with pytest.raises(ValueError, match="拒绝写入"):
            store.add([chunk("aaaa1111aaaa1111", 0, **meta)])

        assert store.count() == 0
        assert not (tmp_path / "chunks.json").exists()
        assert chroma.collections["knowledge_base"].rows == {}

    def test_conflicting_markers_inside_one_document_are_rejected(self, store, chroma):
        # 不能因为"某个 chunk 合法"就放过整篇文档
        chunks = [
            chunk("aaaa1111aaaa1111", 0, visibility="public"),
            chunk("aaaa1111aaaa1111", 1, visibility="private", owner="6"),
        ]

        with pytest.raises(ValueError, match="拒绝写入"):
            store.add(chunks)

        assert chroma.collections["knowledge_base"].rows == {}

    def test_two_owners_inside_one_document_are_rejected(self, store, chroma):
        chunks = [
            chunk("aaaa1111aaaa1111", 0, visibility="private", owner="6"),
            chunk("aaaa1111aaaa1111", 1, visibility="private", owner="7"),
        ]

        with pytest.raises(ValueError, match="多个不同 owner"):
            store.add(chunks)

        assert chroma.collections["knowledge_base"].rows == {}

    def test_batch_is_validated_before_any_write(self, store, chroma, tmp_path):
        # 批量里混进一篇坏文档时不做"写一半"
        chunks = [
            chunk("aaaa1111aaaa1111", 0, visibility="public"),
            chunk("bbbb2222bbbb2222", 0),           # 坏文档
        ]

        with pytest.raises(ValueError, match="拒绝写入"):
            store.add(chunks)

        assert store.count() == 0
        assert not (tmp_path / "chunks.json").exists()
        assert chroma.collections["knowledge_base"].rows == {}

    def test_empty_batch_is_a_noop(self, store, chroma):
        assert store.add([]) == 0
        assert chroma.collections["knowledge_base"].rows == {}


class TestNoSilentOverwrite:
    """已有文档的归属不允许被后来者改写。"""

    DOC = "aaaa1111aaaa1111"

    def _seed_private(self, store, chunks: int = 2, owner: str = "6"):
        written = store.add([
            chunk(self.DOC, index, visibility="private", owner=owner)
            for index in range(chunks)
        ])
        assert written == chunks

    def test_public_import_cannot_overwrite_private_document(self, store, chroma, tmp_path):
        self._seed_private(store)
        before_json = (tmp_path / "chunks.json").read_text(encoding="utf-8")
        before_rows = dict(chroma.collections["knowledge_base"].rows)

        with pytest.raises(ValueError, match="不允许改写既有归属"):
            store.add([
                chunk(self.DOC, 0, visibility="public"),
                chunk(self.DOC, 1, visibility="public"),
            ])

        assert (tmp_path / "chunks.json").read_text(encoding="utf-8") == before_json
        assert chroma.collections["knowledge_base"].rows == before_rows
        assert store.count() == 2

    def test_public_import_cannot_overwrite_unknown_document(self, tmp_path, chroma, embedding):
        seed_legacy(tmp_path, chroma, doc_id=self.DOC)      # 历史文档：无 owner、无 visibility
        store = HybridStore(persist_dir=tmp_path, embedding_client=embedding)
        before_rows = dict(chroma.collections["knowledge_base"].rows)

        with pytest.raises(ValueError, match="既有记录无法判定归属"):
            store.add([
                chunk(self.DOC, 0, visibility="public"),
                chunk(self.DOC, 1, visibility="public"),
            ])

        assert chroma.collections["knowledge_base"].rows == before_rows
        assert embedding.calls == 0

    def test_confirmed_public_document_can_be_reimported(self, store, chroma):
        store.add([chunk(self.DOC, 0, visibility="public")])

        written = store.add([chunk(self.DOC, 0, visibility="public", heading="新标题")])

        assert written == 1
        assert chroma.collections["knowledge_base"].rows[f"{self.DOC}-0"]["heading"] == "新标题"

    def test_same_user_reupload_still_works(self, store, chroma):
        self._seed_private(store)

        written = store.add([
            chunk(self.DOC, 0, visibility="private", owner="6", file_id="u6/new.md"),
            chunk(self.DOC, 1, visibility="private", owner="6", file_id="u6/new.md"),
        ])

        assert written == 2
        assert {c.metadata["file_id"] for c in store.all_chunks()} == {"u6/new.md"}

    def test_owner_change_is_rejected(self, store, chroma):
        self._seed_private(store, owner="6")

        with pytest.raises(ValueError, match="不允许改写既有归属"):
            store.add([
                chunk(self.DOC, 0, visibility="private", owner="7"),
                chunk(self.DOC, 1, visibility="private", owner="7"),
            ])

    def test_two_stores_disagreeing_is_rejected(self, tmp_path, chroma, embedding):
        seed_legacy(
            tmp_path, chroma, doc_id=self.DOC,
            metas=[{"visibility": "private", "owner": "6"}] * 2,
        )
        # 只把 Chroma 侧改成另一个 owner，制造两侧不一致
        for meta in chroma.collections["knowledge_base"].rows.values():
            meta["owner"] = "7"
        store = HybridStore(persist_dir=tmp_path, embedding_client=embedding)
        before_rows = dict(chroma.collections["knowledge_base"].rows)

        with pytest.raises(ValueError, match="两侧既有记录不一致"):
            store.add([
                chunk(self.DOC, 0, visibility="private", owner="6"),
                chunk(self.DOC, 1, visibility="private", owner="6"),
            ])

        assert chroma.collections["knowledge_base"].rows == before_rows

    def test_record_present_only_in_json_is_rejected(self, tmp_path, chroma, embedding):
        seed_legacy(
            tmp_path, chroma, doc_id=self.DOC,
            metas=[{"visibility": "private", "owner": "6"}] * 2,
            chroma_side=False,
        )
        store = HybridStore(persist_dir=tmp_path, embedding_client=embedding)

        with pytest.raises(ValueError, match="Chroma 没有"):
            store.add([
                chunk(self.DOC, 0, visibility="private", owner="6"),
                chunk(self.DOC, 1, visibility="private", owner="6"),
            ])

    def test_record_present_only_in_chroma_is_rejected(self, tmp_path, chroma, embedding):
        seed_legacy(
            tmp_path, chroma, doc_id=self.DOC,
            metas=[{"visibility": "private", "owner": "6"}] * 2,
            json_side=False,
        )
        store = HybridStore(persist_dir=tmp_path, embedding_client=embedding)

        with pytest.raises(ValueError, match="chunks.json 没有"):
            store.add([
                chunk(self.DOC, 0, visibility="private", owner="6"),
                chunk(self.DOC, 1, visibility="private", owner="6"),
            ])

    def test_chunk_set_drift_is_rejected(self, store, chroma):
        self._seed_private(store, chunks=3)

        with pytest.raises(ValueError, match="分块结果与既有记录不一致"):
            store.add([chunk(self.DOC, 0, visibility="private", owner="6")])

    def test_brand_new_documents_in_one_batch_are_written(self, store, chroma):
        written = store.add([
            chunk("aaaa1111aaaa1111", 0, visibility="public"),
            chunk("bbbb2222bbbb2222", 0, visibility="private", owner="6"),
        ])

        assert written == 2
        assert store.count() == 2


class TestPerChunkConsistency:
    """闸门要求"每个 chunk 都完整一致"，不是"整篇能分类"。"""

    DOC = "dddd4444dddd4444"

    def test_chunk_without_visibility_is_rejected(self, store, chroma):
        with pytest.raises(ValueError, match="visibility 缺失或非法"):
            store.add([
                chunk(self.DOC, 0, visibility="public"),
                chunk(self.DOC, 1),                       # 静默 chunk
            ])

        assert chroma.collections["knowledge_base"].rows == {}

    def test_private_chunk_without_owner_is_rejected(self, store, chroma):
        with pytest.raises(ValueError, match="必须带有效 owner"):
            store.add([
                chunk(self.DOC, 0, visibility="private", owner="6"),
                chunk(self.DOC, 1, visibility="private"),  # 缺 owner
            ])

    def test_inconsistent_owners_are_rejected(self, store, chroma):
        with pytest.raises(ValueError, match="多个不同 owner"):
            store.add([
                chunk(self.DOC, 0, visibility="private", owner="6"),
                chunk(self.DOC, 1, visibility="private", owner="7"),
            ])

    def test_public_chunk_with_owner_is_rejected(self, store, chroma):
        with pytest.raises(ValueError, match="不允许声明 owner"):
            store.add([
                chunk(self.DOC, 0, visibility="public"),
                chunk(self.DOC, 1, visibility="public", owner="6"),
            ])

    @pytest.mark.parametrize("value", ["shared", 123, "", "   ", True])
    def test_unusable_visibility_values_are_rejected(self, store, chroma, value):
        with pytest.raises(ValueError, match="visibility 缺失或非法"):
            store.add([chunk(self.DOC, 0, visibility=value)])

    def test_failed_validation_writes_nothing(self, store, chroma, embedding, tmp_path):
        with pytest.raises(ValueError):
            store.add([chunk(self.DOC, 0)])              # 完全没有标记

        assert embedding.calls == 0                       # 没走到 embedding
        assert chroma.collections["knowledge_base"].upsert_calls == 0
        assert not (tmp_path / "chunks.json").exists()
        assert store.count() == 0

    def test_rejected_overwrite_writes_nothing(self, store, chroma, embedding, tmp_path):
        store.add([chunk(self.DOC, 0, visibility="private", owner="6")])
        before_json = (tmp_path / "chunks.json").read_text(encoding="utf-8")
        before_upserts = chroma.collections["knowledge_base"].upsert_calls

        with pytest.raises(ValueError):
            store.add([chunk(self.DOC, 0, visibility="public")])

        assert embedding.calls == 1                       # 只有种子那一次
        assert chroma.collections["knowledge_base"].upsert_calls == before_upserts
        assert (tmp_path / "chunks.json").read_text(encoding="utf-8") == before_json


class TestCrossStoreIdSets:
    """JSON 与 Chroma 的 chunk id 集合必须逐项相等，归属判定一致不够。"""

    DOC = "aaaa1111aaaa1111"

    def _seed_public(self, store, chroma, chunks: int = 2):
        written = store.add([
            chunk(self.DOC, index, visibility="public") for index in range(chunks)
        ])
        assert written == chunks

    def test_matching_id_sets_allow_idempotent_public_update(self, store, chroma):
        self._seed_public(store, chroma)
        json_ids = {c.chunk_id for c in store.all_chunks()}
        chroma_ids = set(chroma.collections["knowledge_base"].rows)
        assert json_ids == chroma_ids == {f"{self.DOC}-0", f"{self.DOC}-1"}

        written = store.add([
            chunk(self.DOC, 0, visibility="public"),
            chunk(self.DOC, 1, visibility="public"),
        ])

        assert written == 2
        assert {c.chunk_id for c in store.all_chunks()} == json_ids

    def test_extra_chunk_in_chroma_is_rejected(self, store, chroma, embedding, tmp_path):
        self._seed_public(store, chroma)
        # 伪造一个同 doc_id、归属标记一致、但 JSON 里没有的孤儿 chunk
        chroma.collections["knowledge_base"].rows[f"{self.DOC}-9"] = {
            "source": "手册.md",
            "doc_id": self.DOC,
            "chunk_index": 9,
            "visibility": "public",
        }
        probe = WriteProbe(store, chroma, embedding, tmp_path)

        with pytest.raises(ValueError, match="两侧 chunk 集合不一致"):
            store.add([
                chunk(self.DOC, 0, visibility="public"),
                chunk(self.DOC, 1, visibility="public"),
            ])

        probe.assert_unchanged()

    def test_missing_chunk_in_chroma_is_rejected(self, store, chroma, embedding, tmp_path):
        self._seed_public(store, chroma)
        chroma.collections["knowledge_base"].rows.pop(f"{self.DOC}-1")
        probe = WriteProbe(store, chroma, embedding, tmp_path)

        with pytest.raises(ValueError, match="两侧 chunk 集合不一致"):
            store.add([
                chunk(self.DOC, 0, visibility="public"),
                chunk(self.DOC, 1, visibility="public"),
            ])

        probe.assert_unchanged()

    def test_chroma_record_with_none_metadata_is_rejected(
        self, store, chroma, embedding, tmp_path
    ):
        self._seed_public(store, chroma, chunks=1)
        patch_chroma_get(chroma, {"ids": [f"{self.DOC}-0"], "metadatas": [None]})
        probe = WriteProbe(store, chroma, embedding, tmp_path)

        with pytest.raises(RuntimeError, match="缺少可用 metadata"):
            store.add([chunk(self.DOC, 0, visibility="public")])

        probe.assert_unchanged()

    @pytest.mark.parametrize("response", [
        {"ids": ["aaaa1111aaaa1111-0"]},                                   # metadatas 键缺失
        {"ids": ["aaaa1111aaaa1111-0"], "metadatas": None},                # metadatas 为 None
        {"ids": ["aaaa1111aaaa1111-0", "aaaa1111aaaa1111-1"],
         "metadatas": [{"doc_id": "aaaa1111aaaa1111", "visibility": "public"}]},  # 条数不匹配
        {"ids": ["aaaa1111aaaa1111-0"], "metadatas": [{}]},                # 空 metadata
    ])
    def test_unusable_metadata_shapes_are_rejected(
        self, store, chroma, embedding, tmp_path, response
    ):
        self._seed_public(store, chroma, chunks=1)
        patch_chroma_get(chroma, response)
        probe = WriteProbe(store, chroma, embedding, tmp_path)

        with pytest.raises(RuntimeError, match="拒绝写入"):
            store.add([chunk(self.DOC, 0, visibility="public")])

        probe.assert_unchanged()

    def test_chroma_query_failure_is_rejected(self, store, chroma, embedding, tmp_path):
        self._seed_public(store, chroma, chunks=1)
        patch_chroma_get(chroma, ValueError("simulated chroma read failure"))
        probe = WriteProbe(store, chroma, embedding, tmp_path)

        with pytest.raises(RuntimeError, match="无法读取 Chroma 既有记录"):
            store.add([chunk(self.DOC, 0, visibility="public")])

        probe.assert_unchanged()

    @pytest.mark.parametrize("response", [
        {"metadatas": [{"doc_id": "aaaa1111aaaa1111", "visibility": "public"}]},   # 缺 ids 键
        {"ids": None, "metadatas": []},                                            # ids=None
        {"ids": "abc", "metadatas": []},                                           # 错误类型（字符串）
        {"ids": [""], "metadatas": [{"doc_id": "aaaa1111aaaa1111"}]},              # 空字符串 ID
        {"ids": [None], "metadatas": [{"doc_id": "aaaa1111aaaa1111"}]},            # 非字符串 ID
        {"ids": [123], "metadatas": [{"doc_id": "aaaa1111aaaa1111"}]},             # 非字符串 ID（整数）
    ])
    def test_malformed_ids_are_rejected(
        self, store, chroma, embedding, tmp_path, response
    ):
        # 畸形结构必须 fail-closed：不能被解释成"没有既有记录"
        self._seed_public(store, chroma, chunks=1)
        patch_chroma_get(chroma, response)
        probe = WriteProbe(store, chroma, embedding, tmp_path)

        with pytest.raises(RuntimeError, match="Chroma 返回结构异常"):
            store.add([chunk(self.DOC, 0, visibility="public")])

        probe.assert_unchanged()

    def test_legal_empty_result_is_treated_as_no_record(self, store, chroma, embedding):
        # 合法空列表是唯一允许被判为"没有既有记录"的情况
        patch_chroma_get(chroma, {"ids": [], "metadatas": []})

        written = store.add([chunk("bbbb2222bbbb2222", 0, visibility="public")])

        assert written == 1
        assert embedding.calls == 1
        assert chroma.collections["knowledge_base"].upsert_calls == 1


class TestRealChromaSmoke:
    """用真实（但独立临时）的 Chroma 验证 `get()` 返回结构的假设。

    不 patch PersistentClient，也不碰项目里的 rag_store——库文件只落在 pytest
    的临时目录下。这一条用来兜住"fake 与真实实现漂移"的风险。
    """

    def test_read_chroma_document_against_real_chroma(self, tmp_path, embedding):
        store = HybridStore(
            persist_dir=tmp_path / "store", embedding_client=embedding
        )
        store.add([chunk("aaaa1111aaaa1111", 0, visibility="public")])

        state = store._read_chroma_document("aaaa1111aaaa1111")
        assert state is not None
        assert state.chunk_ids == {"aaaa1111aaaa1111-0"}
        assert state.access.is_public

        # 空集合必须返回 None 而不是抛错（真实 Chroma 返回空列表）
        assert store._read_chroma_document("bbbb2222bbbb2222") is None


class TestEmptyOwnerSemantics:
    """`owner=""` 一律视为"没声明"——public 允许，private 视为缺失。"""

    DOC = "eeee5555eeee5555"

    def test_empty_owner_is_allowed_for_public(self, store, chroma):
        written = store.add([
            chunk(self.DOC, 0, visibility="public", owner=""),
            chunk(self.DOC, 1, visibility="public", owner="   "),
        ])

        assert written == 2

    def test_empty_owner_is_missing_for_private(self, store, chroma):
        with pytest.raises(ValueError, match="必须带有效 owner"):
            store.add([
                chunk(self.DOC, 0, visibility="private", owner="6"),
                chunk(self.DOC, 1, visibility="private", owner=""),
            ])

    def test_whitespace_only_owner_is_missing_for_private(self, store, chroma):
        with pytest.raises(ValueError, match="必须带有效 owner"):
            store.add([chunk(self.DOC, 0, visibility="private", owner="   ")])


class TestRealWritePathsPassTheGuard:
    """真实写入路径产出的 metadata 必须能被守卫接受。"""

    def test_web_upload_metadata_is_accepted(self, store, chroma, tmp_path, monkeypatch):
        monkeypatch.setattr("agent.rag.get_store", lambda: store)
        path = write_doc(tmp_path / "uploads" / "u6")

        result = uploads.ingest_document_into_kb(
            path, display_name="我的笔记.md", owner=6, file_id="u6/abc.md"
        )

        assert result["ingested"] is True
        assert store.count() >= 1
        assert {c.metadata["visibility"] for c in store.all_chunks()} == {"private"}
        assert chroma.collections["knowledge_base"].rows

    def test_cli_import_metadata_is_accepted(self, store, chroma, tmp_path, monkeypatch):
        from agent.rag import ingest_directory

        monkeypatch.setattr("agent.rag.get_store", lambda: store)
        docs = tmp_path / "docs"
        write_doc(docs, "项目说明.md")

        stats = ingest_directory(docs, visibility="public")

        assert stats["documents"] == 1
        assert store.count() >= 1
        assert {c.metadata["visibility"] for c in store.all_chunks()} == {"public"}
        assert chroma.collections["knowledge_base"].rows
