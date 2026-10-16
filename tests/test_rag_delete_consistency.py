"""R6 删除一致性：行为特征化测试 + chunks.json 原子写验证。

本文件做两件事：

1. 把 ``HybridStore.delete_document`` 在**各失败点**的真实状态变化钉成基线
   （Chroma / 内存 ``_chunks`` / ``chunks.json`` / BM25 缓存各自变成什么样），
   供后续修复对照——其中"半删""重复删除"的断言描述的是**当前**行为，
   不是期望行为（删除顺序与 API 语义属阶段 8B 范围）；
2. 验证阶段 8A 的 ``_save_chunks`` **原子替换**：任何写入失败都不破坏原 JSON、
   不留残留临时文件。

隔离（静态可核对）：

- ``chromadb.PersistentClient`` 被替换成 fake，``persist_dir`` 指向 pytest 临时目录；
- 不调用 ``get_store()`` / ``get_retriever()`` / 任何 HTTP 路径；
- 不打开真实 Chroma、不读 ``rag_store``、不碰 ``.env``、不调模型。
"""

import builtins
import json
import os
import threading
import time
from pathlib import Path

import chromadb
import pytest

from agent.rag.chunker import Chunk
from agent.rag.store import HybridStore

DOC = "aaaa1111aaaa1111"
OTHER = "bbbb3333bbbb3333"
NEW_DOC = "cccc4444cccc4444"
OWNER = "6"

_MISSING_COLLECTION = "Collection [knowledge_base] does not exist"


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
    """最小 Chroma 替身：upsert 按键合并、get 按 doc_id 过滤、delete 可注入失败。"""

    def __init__(self, name):
        self.name = name
        self.rows: dict[str, dict] = {}
        self.delete_calls: list[list[str]] = []
        self.fail_next_delete: Exception | None = None

    def upsert(self, ids, documents, embeddings, metadatas):
        for index, chunk_id in enumerate(ids):
            merged = dict(self.rows.get(chunk_id, {}))
            merged.update(metadatas[index])
            self.rows[chunk_id] = merged

    def get(self, where=None, include=None):
        doc_id = (where or {}).get("doc_id")
        ids = [cid for cid, meta in self.rows.items() if meta.get("doc_id") == doc_id]
        return {"ids": ids, "metadatas": [self.rows[cid] for cid in ids]}

    def query(self, query_embeddings=None, n_results=None, where=None):
        return {"ids": [[]]}

    def delete(self, ids):
        self.delete_calls.append(list(ids))
        if self.fail_next_delete is not None:
            error, self.fail_next_delete = self.fail_next_delete, None
            raise error
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
def store(tmp_path, chroma):
    store = HybridStore(persist_dir=tmp_path, embedding_client=FakeEmbedding())
    store.add([
        chunk(DOC, 0, text="第一段"),
        chunk(DOC, 1, text="第二段"),
    ])
    return store


def chunk(doc_id: str, index: int, text: str = "内容") -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}-{index}",
        doc_id=doc_id,
        source="笔记.md",
        chunk_index=index,
        text=text,
        metadata={"visibility": "private", "owner": OWNER, "ingested_at": "2026-10-16T00:00:00"},
    )


def collection_of(chroma) -> FakeCollection:
    return chroma.collections["knowledge_base"]


def json_text(tmp_path) -> str:
    return (Path(tmp_path) / "chunks.json").read_text(encoding="utf-8")


def json_doc_ids(tmp_path) -> set[str]:
    return {item["doc_id"] for item in json.loads(json_text(tmp_path))}


class TestNormalDelete:
    def test_delete_keeps_all_three_stores_consistent(self, store, chroma, tmp_path):
        result = store.delete_document(DOC)

        assert result.status == "deleted"
        assert result.ok is True and result.partial is False
        assert result.chroma_deleted == 2 and result.chunks_removed == 2
        assert result.json_saved is True and result.bm25_invalidated is True
        assert result.error is None
        assert store.count() == 0                       # 内存
        assert collection_of(chroma).rows == {}         # Chroma
        assert json_doc_ids(tmp_path) == set()          # chunks.json

    def test_delete_invalidates_bm25_cache(self, store):
        assert store._get_bm25() is not None            # 先构建出缓存

        store.delete_document(DOC)

        assert store._bm25 is None                      # 删除成功后必须失效

    def test_delete_of_unknown_doc_is_noop(self, store, chroma, tmp_path):
        before = json_text(tmp_path)

        result = store.delete_document("ffff9999ffff9999")

        assert result.status == "already_absent"
        assert result.ok is True and result.partial is False
        assert result.chunks_removed == 0 and result.chroma_deleted == 0
        assert collection_of(chroma).delete_calls == []  # 没有发出 Chroma 删除
        assert json_text(tmp_path) == before


class TestChromaFailure:
    def test_chroma_delete_failure_changes_nothing(self, store, chroma, tmp_path):
        collection_of(chroma).fail_next_delete = RuntimeError("boom")
        before_json = json_text(tmp_path)
        before_rows = dict(collection_of(chroma).rows)

        with pytest.raises(RuntimeError, match="boom"):
            store.delete_document(DOC)

        # 现状：Chroma 删除是第一步，失败即整体中止，三处都保持原样
        assert store.count() == 2
        assert collection_of(chroma).rows == before_rows
        assert json_text(tmp_path) == before_json

    def test_stale_collection_handle_is_retried_once(self, store, chroma, tmp_path):
        collection_of(chroma).fail_next_delete = RuntimeError(_MISSING_COLLECTION)

        result = store.delete_document(DOC)

        assert result.status == "deleted"
        assert result.chunks_removed == 2
        assert len(collection_of(chroma).delete_calls) == 2      # 重试了一次
        assert collection_of(chroma).rows == {}
        assert json_doc_ids(tmp_path) == set()


class TestJsonSaveFailure:
    """元数据写入失败 → 结构化 partial：向量库已删，内存/JSON/BM25 一律保持原样。"""

    def test_json_failure_reports_partial_without_touching_memory_or_bm25(
        self, store, chroma, tmp_path, monkeypatch
    ):
        assert store._get_bm25() is not None            # 让 BM25 缓存存在
        before_json = json_text(tmp_path)

        def broken_save(chunks=None):
            raise OSError("disk full")

        monkeypatch.setattr(store, "_save_chunks", broken_save)

        result = store.delete_document(DOC)

        # 结构化结果：明确是"部分失败"，不是成功
        assert result.status == "partial"
        assert result.partial is True and result.ok is False
        assert result.chroma_deleted == 2 and result.chunks_removed == 0
        assert result.json_saved is False and result.bm25_invalidated is False
        assert "元数据写入失败" in (result.error or "")
        assert collection_of(chroma).rows == {}          # ① Chroma：已删
        assert store.count() == 2                        # ② 内存：**未**提前清理
        assert json_text(tmp_path) == before_json        # ③ JSON：仍是旧内容
        assert json_doc_ids(tmp_path) == {DOC}
        assert store._bm25 is not None                   # ④ BM25：**未**提前失效

    def test_partial_state_rejects_reupload_and_survives_restart(
        self, store, chroma, tmp_path
    ):
        """半删后：同一进程与重启后都由写闸门拒绝再次上传（一侧有记录）。"""
        with pytest.MonkeyPatch.context() as patcher:
            patcher.setattr(
                store, "_save_chunks",
                lambda chunks=None: (_ for _ in ()).throw(OSError("disk full")),
            )
            assert store.delete_document(DOC).partial is True

        # 同一进程内：内存仍有该文档、Chroma 已没有 → 写闸门拒绝
        with pytest.raises(ValueError, match="Chroma 没有"):
            store.add([chunk(DOC, 0), chunk(DOC, 1)])

        # 重启（同一 persist_dir 重新构造）后同样拒绝
        restarted = HybridStore(persist_dir=tmp_path, embedding_client=FakeEmbedding())
        assert restarted.count() == 2
        with pytest.raises(ValueError, match="Chroma 没有"):
            restarted.add([chunk(DOC, 0), chunk(DOC, 1)])

    def test_retrying_delete_after_partial_heals_the_state(
        self, store, chroma, tmp_path
    ):
        """partial 不会自动恢复，但人工重试一次删除即可修复（幂等 + 原子写）。"""
        with pytest.MonkeyPatch.context() as patcher:
            patcher.setattr(
                store, "_save_chunks",
                lambda chunks=None: (_ for _ in ()).throw(OSError("disk full")),
            )
            assert store.delete_document(DOC).partial is True

        result = store.delete_document(DOC)              # 重试（不再注入失败）

        assert result.status == "deleted"
        assert store.count() == 0
        assert collection_of(chroma).rows == {}
        assert json_doc_ids(tmp_path) == set()


class TestChromaResidue:
    """内存索引里没有该文档、但向量库仍有记录：必须报 partial，不能当成"不存在"。"""

    def test_memory_missing_but_vector_store_has_residue_is_partial(
        self, store, chroma, tmp_path
    ):
        before_json = json_text(tmp_path)
        # 只在内存里移除该文档，向量库保持原样（模拟两侧不一致）
        store._chunks = {
            cid: c for cid, c in store._chunks.items() if c.doc_id != DOC
        }

        result = store.delete_document(DOC)

        assert result.status == "partial"
        assert result.partial is True and result.ok is False
        assert result.chroma_deleted == 0 and result.chunks_removed == 0
        assert "向量库仍有记录" in (result.error or "")
        assert collection_of(chroma).delete_calls == []      # 不猜着删
        assert collection_of(chroma).rows                    # 残留仍在
        assert json_text(tmp_path) == before_json


class TestConcurrentDeletes:
    """并发写必须串行：两个删除都要落地，不能被"后写者覆盖前写者"。"""

    def test_concurrent_deletes_of_two_documents_keep_both_removals(
        self, store, chroma, tmp_path, monkeypatch
    ):
        store.add([chunk(OTHER, 0), chunk(OTHER, 1)])
        assert store._get_bm25() is not None            # 让 BM25 缓存存在，便于断言失效

        real_save = store._save_chunks

        def slow_save(chunks=None):
            # 放大写盘窗口：若没有写锁，两个线程会基于同一份陈旧快照互相覆盖
            time.sleep(0.05)
            return real_save(chunks)

        monkeypatch.setattr(store, "_save_chunks", slow_save)

        barrier = threading.Barrier(2)
        results: dict[str, object] = {}
        errors: list[BaseException] = []

        def worker(doc_id):
            barrier.wait(timeout=10)                    # 真正同时起跑
            try:
                results[doc_id] = store.delete_document(doc_id)
            except BaseException as e:                  # pragma: no cover - 失败时留线索
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=(doc_id,)) for doc_id in (DOC, OTHER)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=15)

        assert errors == []
        assert {r.status for r in results.values()} == {"deleted"}
        assert store.count() == 0                       # 内存：两份都删了
        assert json_doc_ids(tmp_path) == set()          # JSON：两份都删了
        assert collection_of(chroma).rows == {}         # Chroma：两份都删了
        assert store._bm25 is None                      # BM25 已失效


class TestConcurrentDeleteAndAdd:
    def test_delete_and_upload_concurrently_keep_both_outcomes(
        self, store, chroma, tmp_path, monkeypatch
    ):
        """一个线程删除、另一个线程上传：删除要生效，新文档也不能被覆盖丢掉。"""
        real_save = store._save_chunks

        def slow_delete_save(chunks=None):
            # 只延迟"带候选索引"的调用（删除传参、上传不传参），让删除的快照写盘
            # 确定性地落在上传之后——这正是没有写锁时的丢数据窗口
            if chunks is not None:
                time.sleep(0.05)
            return real_save(chunks)

        monkeypatch.setattr(store, "_save_chunks", slow_delete_save)

        barrier = threading.Barrier(2)
        outcome: dict[str, object] = {}
        errors: list[BaseException] = []

        def deleter():
            barrier.wait(timeout=10)
            try:
                outcome["delete"] = store.delete_document(DOC)
            except BaseException as e:                  # pragma: no cover
                errors.append(e)

        def uploader():
            barrier.wait(timeout=10)
            try:
                outcome["add"] = store.add([chunk(NEW_DOC, 0)])
            except BaseException as e:                  # pragma: no cover
                errors.append(e)

        threads = [threading.Thread(target=deleter), threading.Thread(target=uploader)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=15)

        assert errors == []
        assert outcome["delete"].status == "deleted"
        assert outcome["add"] == 1
        # 删除目标确实没了；新文档在内存、JSON、Chroma 三处都在（不分叉）
        assert {c.doc_id for c in store.all_chunks()} == {NEW_DOC}
        assert json_doc_ids(tmp_path) == {NEW_DOC}
        assert set(collection_of(chroma).rows) == {f"{NEW_DOC}-0"}


class TestAtomicSaveChunks:
    """阶段 8A：``_save_chunks`` 的原子替换（无半截文件、失败不破坏原文件）。"""

    @staticmethod
    def json_path(tmp_path) -> Path:
        return Path(tmp_path) / "chunks.json"

    @staticmethod
    def temp_files(tmp_path) -> list[Path]:
        return sorted(Path(tmp_path).glob("*.tmp"))

    def test_normal_save_writes_parseable_json_without_temp_leftovers(
        self, store, tmp_path
    ):
        store._save_chunks()

        payload = json.loads(json_text(tmp_path))
        assert {item["doc_id"] for item in payload} == {DOC}
        assert self.temp_files(tmp_path) == []

    def test_saved_file_is_readable_by_a_fresh_store(self, store, tmp_path, chroma):
        restarted = HybridStore(persist_dir=tmp_path, embedding_client=FakeEmbedding())

        assert restarted.count() == 2
        assert {c.doc_id for c in restarted.all_chunks()} == {DOC}

    def test_temp_write_failure_keeps_original_bytes(self, store, tmp_path, monkeypatch):
        before = self.json_path(tmp_path).read_bytes()
        real_open = builtins.open

        def failing_open(file, *args, **kwargs):
            if str(file).endswith(".tmp"):
                raise OSError("disk full")
            return real_open(file, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", failing_open)

        with pytest.raises(RuntimeError, match="原文件保持不变"):
            store._save_chunks()

        assert self.json_path(tmp_path).read_bytes() == before

    def test_temp_write_failure_leaves_no_temp_file(self, store, tmp_path, monkeypatch):
        real_open = builtins.open

        def failing_open(file, *args, **kwargs):
            if str(file).endswith(".tmp"):
                # 先创建出半截临时文件，再失败——验证失败路径会把它清掉
                real_open(file, "w", encoding="utf-8").write("半截")
                raise OSError("disk full")
            return real_open(file, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", failing_open)

        with pytest.raises(RuntimeError):
            store._save_chunks()

        assert self.temp_files(tmp_path) == []

    def test_replace_failure_keeps_original_bytes_and_cleans_temp(
        self, store, tmp_path, monkeypatch
    ):
        before = self.json_path(tmp_path).read_bytes()

        def failing_replace(src, dst):
            raise OSError("replace failed")

        monkeypatch.setattr(os, "replace", failing_replace)

        with pytest.raises(RuntimeError, match="原文件保持不变"):
            store._save_chunks()

        assert self.json_path(tmp_path).read_bytes() == before
        assert self.temp_files(tmp_path) == []

    def test_serialization_failure_keeps_original_and_reports_clearly(
        self, store, tmp_path
    ):
        before = self.json_path(tmp_path).read_bytes()
        store._chunks[f"{DOC}-0"].metadata["unserializable"] = object()

        with pytest.raises(RuntimeError, match="序列化失败"):
            store._save_chunks()

        assert self.json_path(tmp_path).read_bytes() == before
        assert self.temp_files(tmp_path) == []

    def test_consecutive_saves_end_with_expected_content(self, store, tmp_path):
        store.delete_document(DOC)            # 第二次保存：内容变空
        store.add([chunk(DOC, 0)])            # 第三次保存：只留一个 chunk

        payload = json.loads(json_text(tmp_path))
        assert {item["chunk_id"] for item in payload} == {f"{DOC}-0"}
        assert self.temp_files(tmp_path) == []

    def test_saving_a_candidate_mapping_does_not_switch_memory(self, store, tmp_path):
        """删除流程依赖这一点：先写盘成功，才允许切换内存。"""
        remaining = {
            cid: c for cid, c in store._chunks.items() if c.doc_id != DOC
        }

        store._save_chunks(remaining)

        assert json.loads(json_text(tmp_path)) == []      # 磁盘已是"删掉后"的内容
        assert store.count() == 2                         # 内存还没切换

    def test_stale_temp_file_from_a_previous_crash_is_not_ours(self, store, tmp_path):
        # 进程在写临时文件时被杀，会留下一个 .tmp；保存不应误删别人的文件，
        # 但也不能被它影响（正式文件照常写入）
        stale = Path(tmp_path) / "chunks.json.999-deadbeef.tmp"
        stale.write_text("崩溃残留", encoding="utf-8")

        store._save_chunks()

        assert json.loads(json_text(tmp_path))
        assert stale.exists()


class TestRepeatedDelete:
    def test_second_delete_reports_already_absent(self, store, chroma, tmp_path):
        first = store.delete_document(DOC)
        assert first.status == "deleted" and first.chunks_removed == 2
        before_json = json_text(tmp_path)

        # 幂等语义：第二次返回 already_absent（不再抛错、不再动任何存储）
        second = store.delete_document(DOC)
        assert second.status == "already_absent"
        assert second.ok is True and second.partial is False
        assert second.chunks_removed == 0
        assert json_text(tmp_path) == before_json
        # 第二次在 ids 为空时直接短路：只发生过一次 Chroma 删除调用
        assert len(collection_of(chroma).delete_calls) == 1


class TestReuploadAfterDelete:
    def test_reupload_after_full_delete_is_allowed(self, store, chroma, tmp_path):
        store.delete_document(DOC)

        written = store.add([chunk(DOC, 0), chunk(DOC, 1)])

        assert written == 2
        assert store.count() == 2
        assert json_doc_ids(tmp_path) == {DOC}
        assert set(collection_of(chroma).rows) == {f"{DOC}-0", f"{DOC}-1"}

    def test_reupload_when_chroma_lost_the_doc_is_rejected(self, store, chroma):
        # 模拟"Chroma 侧丢了这份文档"（例如跨进程/半删）：JSON 仍有 → 闸门拒绝
        for chunk_id in [cid for cid in collection_of(chroma).rows if cid.startswith(DOC)]:
            collection_of(chroma).rows.pop(chunk_id)

        with pytest.raises(ValueError, match="Chroma 没有"):
            store.add([chunk(DOC, 0), chunk(DOC, 1)])
