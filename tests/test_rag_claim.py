"""历史 unknown 文档"显式认领"工具（agent.rag.claim）的测试。

背景：历史文档缺少 owner/visibility，属于 unknown，对所有用户不可见。
直接把同名内容重新灌库会被写入保护拒绝（"既有记录是 unknown → 拒绝写入"），
所以需要一个**显式、可审计、只吃白名单 doc_id** 的认领工具。

这里全部用真实 HybridStore（临时目录 + fake Chroma 客户端）：
真实读取/写入 chunks.json 与真实的内存切换逻辑，只有 Chroma 是替身。
每个拒绝/失败用例都断言"零副作用"：chunks.json 字节不变、Chroma 未被触碰。
"""

import copy
import json
from pathlib import Path

import chromadb
import pytest

from agent.rag.audit import build_report
from agent.rag.claim import ClaimError, claim_documents
from agent.rag.claim import main as claim_main
from agent.rag.store import HybridStore


class FakeEmbedding:
    """不发网络请求的 embedding 假实现（认领流程本身不会用到它）。"""

    def embed(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_one(self, text):
        return [0.1, 0.2, 0.3]


class FakeCollection:
    """最小可用的 Chroma collection 假实现。

    对齐真实语义：
    - ``get(ids=...)`` 返回 ``{"ids": [...], "metadatas": [...]}``（只返回命中的）；
    - ``update`` 对 metadata 做按键合并（None 值表示删除该键）。
    """

    def __init__(self, name: str):
        self.name = name
        self.rows: dict[str, dict] = {}
        self.update_calls = 0
        self.fail_next_update = False

    def upsert(self, ids, documents, embeddings, metadatas):
        for index, chunk_id in enumerate(ids):
            merged = dict(self.rows.get(chunk_id, {}))
            merged.update({k: v for k, v in metadatas[index].items() if v is not None})
            self.rows[chunk_id] = merged

    def update(self, ids, metadatas=None):
        self.update_calls += 1
        if self.fail_next_update:
            self.fail_next_update = False
            raise RuntimeError("chroma update failed (simulated)")
        for index, chunk_id in enumerate(ids):
            if chunk_id not in self.rows:
                raise KeyError(f"unknown chunk id {chunk_id}")
            merged = dict(self.rows[chunk_id])
            for key, value in (metadatas[index] or {}).items():
                if value is None:
                    merged.pop(key, None)
                else:
                    merged[key] = value
            self.rows[chunk_id] = merged

    def get(self, ids=None, where=None, include=None):
        if ids is not None:
            found = [chunk_id for chunk_id in ids if chunk_id in self.rows]
        else:
            doc_id = (where or {}).get("doc_id")
            found = [
                chunk_id
                for chunk_id, meta in self.rows.items()
                if meta.get("doc_id") == doc_id
            ]
        return {"ids": found, "metadatas": [self.rows[chunk_id] for chunk_id in found]}

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


DOC_A = "aaaa1111aaaa1111"
DOC_B = "bbbb2222bbbb2222"
LEGACY_META = {"chars": 10, "heading": "第一节", "is_table": False, "suffix": ".md"}


def seed_legacy(
    persist_dir: Path,
    chroma: FakeChromaClient,
    doc_id: str = DOC_A,
    metas: list[dict] | None = None,
    source: str = "历史文档.md",
) -> list[dict]:
    """按"历史遗留"形态造数据：chunks.json + Chroma 行，元数据里没有归属字段。"""
    metas = metas or [
        dict(LEGACY_META, heading="第一节"),
        dict(LEGACY_META, heading="第二节"),
    ]
    items = []
    for index, meta in enumerate(metas):
        items.append(
            {
                "chunk_id": f"{doc_id}-{index}",
                "doc_id": doc_id,
                "source": source,
                "chunk_index": index,
                "text": f"{source} 第 {index} 段正文",
                "metadata": dict(meta),
            }
        )
    (persist_dir / "chunks.json").write_text(
        json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    collection = chroma.get_or_create_collection("knowledge_base", metadata=None)
    for item in items:
        collection.rows[item["chunk_id"]] = {
            **item["metadata"],
            "doc_id": doc_id,
        }
    return items


def make_store(persist_dir: Path) -> HybridStore:
    return HybridStore(persist_dir=persist_dir, embedding_client=FakeEmbedding())


def read_items(persist_dir: Path) -> list[dict]:
    return json.loads((persist_dir / "chunks.json").read_text(encoding="utf-8"))


class TestDryRun:
    def test_dry_run_reports_plan_without_writing(self, tmp_path, chroma):
        seed_legacy(tmp_path, chroma)
        store = make_store(tmp_path)
        before_bytes = (tmp_path / "chunks.json").read_bytes()
        before_rows = copy.deepcopy(chroma.collections["knowledge_base"].rows)

        result = claim_documents(store, [DOC_A], dry_run=True)

        entry = result["documents"][0]
        assert result["ok"] is True
        assert result["dry_run"] is True
        assert entry["doc_id"] == DOC_A
        assert entry["chunks"] == 2
        assert entry["before"] == "unknown"
        assert entry["after"] == "public"
        assert (tmp_path / "chunks.json").read_bytes() == before_bytes
        assert chroma.collections["knowledge_base"].rows == before_rows
        assert chroma.collections["knowledge_base"].update_calls == 0


class TestSuccessfulClaim:
    def test_claim_marks_every_chunk_public_and_keeps_other_docs_untouched(
        self, tmp_path, chroma
    ):
        seed_legacy(tmp_path, chroma, DOC_A)
        seed_legacy(tmp_path, chroma, DOC_B, metas=[dict(LEGACY_META)], source="另一篇.md")
        # 重新写 combined chunks.json（seed 会覆盖文件，这里合并两篇）
        items = json.loads((tmp_path / "chunks.json").read_text(encoding="utf-8"))
        items += [
            {
                "chunk_id": f"{DOC_A}-{index}",
                "doc_id": DOC_A,
                "source": "历史文档.md",
                "chunk_index": index,
                "text": f"历史文档 第 {index} 段正文",
                "metadata": dict(LEGACY_META),
            }
            for index in range(2)
        ]
        (tmp_path / "chunks.json").write_text(
            json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        store = make_store(tmp_path)
        b_before = [item for item in read_items(tmp_path) if item["doc_id"] == DOC_B]

        result = claim_documents(store, [DOC_A])

        assert result["ok"] is True
        assert result["changed_json"] is True
        assert result["changed_chroma"] is True

        after = read_items(tmp_path)
        a_metas = [item["metadata"] for item in after if item["doc_id"] == DOC_A]
        assert len(a_metas) == 2
        for meta in a_metas:
            assert meta["visibility"] == "public"
            assert "owner" not in meta

        b_after = [item for item in after if item["doc_id"] == DOC_B]
        assert b_after == b_before  # 其他文档一字未动

        rows = chroma.collections["knowledge_base"].rows
        for index in range(2):
            assert rows[f"{DOC_A}-{index}"]["visibility"] == "public"
        assert rows[f"{DOC_B}-0"].get("visibility") is None

        for chunk in store.all_chunks():
            if chunk.doc_id == DOC_A:
                assert chunk.metadata["visibility"] == "public"


class TestRejections:
    def _assert_rejected(self, store, tmp_path, chroma, doc_ids, match):
        before_bytes = (tmp_path / "chunks.json").read_bytes()
        before_rows = copy.deepcopy(chroma.collections["knowledge_base"].rows)
        with pytest.raises(ClaimError, match=match):
            claim_documents(store, doc_ids)
        assert (tmp_path / "chunks.json").read_bytes() == before_bytes
        assert chroma.collections["knowledge_base"].rows == before_rows
        assert chroma.collections["knowledge_base"].update_calls == 0

    def test_rejects_missing_doc_id(self, tmp_path, chroma):
        seed_legacy(tmp_path, chroma)
        store = make_store(tmp_path)
        self._assert_rejected(store, tmp_path, chroma, ["ffff9999ffff9999"], "不存在")

    def test_public_document_is_noop_not_error(self, tmp_path, chroma):
        seed_legacy(tmp_path, chroma, metas=[{"visibility": "public", "chars": 3}])
        store = make_store(tmp_path)
        before_bytes = (tmp_path / "chunks.json").read_bytes()

        result = claim_documents(store, [DOC_A])

        assert result["ok"] is True
        assert result["changed_json"] is False
        assert result["changed_chroma"] is False
        assert (tmp_path / "chunks.json").read_bytes() == before_bytes
        assert chroma.collections["knowledge_base"].update_calls == 0

    def test_rejects_document_with_owner(self, tmp_path, chroma):
        seed_legacy(tmp_path, chroma, metas=[{"owner": "3", "chars": 3}])
        store = make_store(tmp_path)
        self._assert_rejected(store, tmp_path, chroma, [DOC_A], "private|owner")

    def test_rejects_invalid_visibility_value(self, tmp_path, chroma):
        seed_legacy(tmp_path, chroma, metas=[{"visibility": "shared", "chars": 3}])
        store = make_store(tmp_path)
        self._assert_rejected(store, tmp_path, chroma, [DOC_A], "unknown|归属|非法")

    def test_rejects_conflicting_chunk_visibility(self, tmp_path, chroma):
        seed_legacy(
            tmp_path,
            chroma,
            metas=[
                {"visibility": "public", "chars": 3},
                {"visibility": "private", "owner": "3", "chars": 3},
            ],
        )
        store = make_store(tmp_path)
        self._assert_rejected(store, tmp_path, chroma, [DOC_A], "unknown|归属|非法")

    def test_rejects_unsupported_visibility_argument(self, tmp_path, chroma):
        seed_legacy(tmp_path, chroma)
        store = make_store(tmp_path)
        with pytest.raises(ClaimError, match="public"):
            claim_documents(store, [DOC_A], visibility="private")

    def test_rejects_empty_doc_id_list(self, tmp_path, chroma):
        seed_legacy(tmp_path, chroma)
        store = make_store(tmp_path)
        with pytest.raises(ClaimError, match="doc-id|doc_id"):
            claim_documents(store, [])


class TestFailurePaths:
    def test_json_save_failure_aborts_without_side_effects(
        self, tmp_path, chroma, monkeypatch
    ):
        seed_legacy(tmp_path, chroma)
        store = make_store(tmp_path)
        before_bytes = (tmp_path / "chunks.json").read_bytes()
        before_rows = copy.deepcopy(chroma.collections["knowledge_base"].rows)
        memory_before = {
            chunk_id: (chunk.doc_id, dict(chunk.metadata))
            for chunk_id, chunk in store._chunks.items()
        }

        def boom(chunks=None):
            raise RuntimeError("模拟写入失败")

        monkeypatch.setattr(store, "_save_chunks", boom)
        with pytest.raises(ClaimError, match="JSON 保存失败"):
            claim_documents(store, [DOC_A])

        assert (tmp_path / "chunks.json").read_bytes() == before_bytes
        assert chroma.collections["knowledge_base"].rows == before_rows
        assert chroma.collections["knowledge_base"].update_calls == 0
        memory_after = {
            chunk_id: (chunk.doc_id, dict(chunk.metadata))
            for chunk_id, chunk in store._chunks.items()
        }
        assert memory_after == memory_before  # 内存完全没有切换

    def test_chroma_failure_reports_incomplete_then_repair_finishes(
        self, tmp_path, chroma
    ):
        seed_legacy(tmp_path, chroma)
        store = make_store(tmp_path)
        collection = chroma.collections["knowledge_base"]
        collection.fail_next_update = True

        result = claim_documents(store, [DOC_A])

        assert result["ok"] is False
        assert result["chroma_incomplete"] is True
        assert "repair" in result["error"].lower() or "重试" in result["error"]
        # JSON/内存已按设计完成（与 JSON 一致），Chroma 尚缺 public
        a_metas = [item["metadata"] for item in read_items(tmp_path) if item["doc_id"] == DOC_A]
        assert all(meta["visibility"] == "public" for meta in a_metas)
        assert all(
            row.get("visibility") is None
            for row in collection.rows.values()
            if row.get("doc_id") == DOC_A
        )

        repaired = claim_documents(store, [DOC_A], repair_chroma_only=True)

        assert repaired["ok"] is True
        assert repaired["changed_chroma"] is True
        assert all(
            row["visibility"] == "public"
            for row in collection.rows.values()
            if row.get("doc_id") == DOC_A
        )

    def test_rerun_detects_json_public_but_chroma_missing_and_advises_repair(
        self, tmp_path, chroma
    ):
        seed_legacy(tmp_path, chroma)
        store = make_store(tmp_path)
        collection = chroma.collections["knowledge_base"]
        collection.fail_next_update = True
        first = claim_documents(store, [DOC_A])
        assert first["chroma_incomplete"] is True

        second = claim_documents(store, [DOC_A])

        assert second["ok"] is False
        assert second["chroma_incomplete"] is True
        assert "--repair-chroma-only" in second["error"]

    def test_second_run_is_idempotent_noop(self, tmp_path, chroma):
        seed_legacy(tmp_path, chroma)
        store = make_store(tmp_path)
        first = claim_documents(store, [DOC_A])
        assert first["ok"] is True
        updates_after_first = chroma.collections["knowledge_base"].update_calls

        second = claim_documents(store, [DOC_A])

        assert second["ok"] is True
        assert second["changed_json"] is False
        assert second["changed_chroma"] is False
        assert chroma.collections["knowledge_base"].update_calls == updates_after_first
        assert second["documents"][0]["before"] == "public"


class TestAuditIntegration:
    def test_audit_report_lists_claimed_docs_as_public(self, tmp_path, chroma):
        seed_legacy(tmp_path, chroma)
        store = make_store(tmp_path)
        claim_documents(store, [DOC_A])

        report = build_report(tmp_path)

        entry = next(doc for doc in report.documents if doc.doc_id == DOC_A)
        assert entry.status == "public"
        assert entry.needs_review is False
        assert report.public_documents == 1


class TestCli:
    def test_cli_dry_run_works_and_writes_nothing(self, tmp_path, chroma, capsys):
        seed_legacy(tmp_path, chroma)
        before_bytes = (tmp_path / "chunks.json").read_bytes()

        code = claim_main(
            ["--persist-dir", str(tmp_path), "--doc-id", DOC_A, "--dry-run"]
        )

        captured = capsys.readouterr()
        assert code == 0
        assert '"dry_run": true' in captured.out
        assert (tmp_path / "chunks.json").read_bytes() == before_bytes

    def test_cli_write_requires_yes_flag(self, tmp_path, chroma, capsys):
        seed_legacy(tmp_path, chroma)
        before_bytes = (tmp_path / "chunks.json").read_bytes()

        code = claim_main(["--persist-dir", str(tmp_path), "--doc-id", DOC_A])

        captured = capsys.readouterr()
        assert code == 2
        assert "--yes" in captured.out
        assert (tmp_path / "chunks.json").read_bytes() == before_bytes
