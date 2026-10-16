"""写入侧的可见性标记测试：Web 上传路径 + CLI 公共导入路径。

本轮的目标只有一件事——文档必须**带着明确、合法的可见性标记**进入存储。
所以这里全部用 fake store 抓取 chunk，不连接真实持久化：既不会误写 rag_store，
也让断言能直接落在"每个 chunk 的 metadata"上。

覆盖的核心边界：
1. Web 上传：所有 chunk 一致带 owner + visibility="private"；
2. CLI 导入：所有 chunk 带 visibility="public" 且没有 owner；
3. 可见性必须显式声明，缺省、非法值都不能蒙混过关；
4. 不同用户上传同一内容时 doc_id 不同（互不覆盖归属），同一用户重复上传幂等。
"""

from pathlib import Path

import pytest

from agent.rag import ingest_directory
from agent.rag.chunker import Chunk
from agent.rag.loader import load_document
from server import uploads


class FakeStore:
    """只记录写入的 chunk，不做任何持久化。"""

    def __init__(self):
        self.added: list[Chunk] = []

    def add(self, chunks):
        self.added.extend(chunks)
        return len(chunks)

    def count(self):
        return len(self.added)


@pytest.fixture
def fake_store(monkeypatch):
    store = FakeStore()
    # 两条写入路径都是"调用时才 from agent.rag import get_store"，替换包属性即可
    monkeypatch.setattr("agent.rag.get_store", lambda: store)
    return store


def write_doc(directory: Path, name: str = "笔记.md", text: str = "# 标题\n\n正文内容。\n") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


class TestWebUploadPath:
    def test_every_chunk_is_private_with_same_owner(self, tmp_path, fake_store):
        path = write_doc(tmp_path / "uploads" / "u6")

        result = uploads.ingest_document_into_kb(
            path, display_name="我的笔记.md", owner=6, file_id="u6/abc.md"
        )

        assert result["ingested"] is True
        assert fake_store.added
        for chunk in fake_store.added:
            assert chunk.metadata["visibility"] == "private"
            assert chunk.metadata["owner"] == "6"
            assert chunk.metadata["file_id"] == "u6/abc.md"
            assert chunk.metadata["ingested_at"]
            assert chunk.source == "我的笔记.md"

    def test_owner_is_required(self, tmp_path, fake_store):
        path = write_doc(tmp_path / "uploads" / "u6")

        with pytest.raises(ValueError, match="必须带 owner"):
            uploads.ingest_document_into_kb(path, display_name="笔记.md")

        assert fake_store.added == []          # 拒绝时不能有半份数据落库

    @pytest.mark.parametrize("owner", [0, -1, "abc", "", True])
    def test_invalid_owner_is_rejected(self, tmp_path, fake_store, owner):
        path = write_doc(tmp_path / "uploads" / "u6")

        with pytest.raises(ValueError, match="owner 非法"):
            uploads.ingest_document_into_kb(path, owner=owner)

        assert fake_store.added == []

    def test_doc_id_is_namespaced_per_user(self, tmp_path, fake_store):
        path = write_doc(tmp_path / "uploads" / "same", text="# 标题\n\n同一份内容。\n")

        uploads.ingest_document_into_kb(path, owner=6, file_id="u6/a.md")
        uploads.ingest_document_into_kb(path, owner=7, file_id="u7/b.md")

        by_owner: dict[str, set[str]] = {}
        for chunk in fake_store.added:
            by_owner.setdefault(chunk.metadata["owner"], set()).add(chunk.doc_id)

        assert set(by_owner) == {"6", "7"}
        assert len(by_owner["6"]) == 1 and len(by_owner["7"]) == 1
        assert by_owner["6"] != by_owner["7"]     # 同一内容，两人各得一份，互不覆盖

    def test_same_user_reupload_keeps_doc_id_and_updates_source(self, tmp_path, fake_store):
        path = write_doc(tmp_path / "uploads" / "u6", text="# 标题\n\n同一份内容。\n")

        uploads.ingest_document_into_kb(path, display_name="旧名字.md", owner=6)
        first = {c.doc_id for c in fake_store.added}
        fake_store.added.clear()

        uploads.ingest_document_into_kb(path, display_name="新名字.md", owner=6)
        second = {c.doc_id for c in fake_store.added}

        assert first == second                    # 幂等：upsert 而不是新增一份
        assert {c.source for c in fake_store.added} == {"新名字.md"}


class TestCliPublicImportPath:
    def test_every_chunk_is_public_without_owner(self, tmp_path, fake_store):
        docs = tmp_path / "docs"
        write_doc(docs, "项目说明.md")

        stats = ingest_directory(docs, visibility="public")

        assert stats["documents"] == 1
        assert stats["visibility"] == "public"
        assert fake_store.added
        for chunk in fake_store.added:
            assert chunk.metadata["visibility"] == "public"
            assert "owner" not in chunk.metadata
            assert "file_id" not in chunk.metadata

    def test_visibility_must_be_declared(self, tmp_path, fake_store):
        docs = tmp_path / "docs"
        write_doc(docs, "项目说明.md")

        with pytest.raises(TypeError):
            ingest_directory(docs, rebuild=False)

        assert fake_store.added == []

    def test_private_visibility_is_not_supported_yet(self, tmp_path, fake_store):
        docs = tmp_path / "docs"
        write_doc(docs, "项目说明.md")

        with pytest.raises(ValueError, match="仅支持"):
            ingest_directory(docs, visibility="private")

        assert fake_store.added == []

    def test_doc_id_matches_legacy_content_hash(self, tmp_path, fake_store):
        # 重新导入 ./docs 必须命中历史 doc_id，否则迁移会凭空复制出一份新文档
        docs = tmp_path / "docs"
        path = write_doc(docs, "项目说明.md")

        ingest_directory(docs, visibility="public")

        assert {c.doc_id for c in fake_store.added} == {load_document(path).doc_id}

    def test_explicit_public_wins_over_conflicting_metadata(
        self, tmp_path, fake_store, monkeypatch
    ):
        # 文档自带矛盾标记时以显式声明为准：public 且不带归属
        docs = tmp_path / "docs"
        path = write_doc(docs, "项目说明.md")
        document = load_document(path)
        document.metadata.update({"owner": "9", "visibility": "shared", "file_id": "u9/x.md"})
        monkeypatch.setattr("agent.rag.load_directory", lambda directory: [document])

        ingest_directory(docs, visibility="public")

        for chunk in fake_store.added:
            assert chunk.metadata["visibility"] == "public"
            assert "owner" not in chunk.metadata
            assert "file_id" not in chunk.metadata

    def test_empty_directory_writes_nothing(self, tmp_path, fake_store):
        docs = tmp_path / "docs"
        docs.mkdir()

        stats = ingest_directory(docs, visibility="public")

        assert stats == {"documents": 0, "chunks": 0, "total": 0}
        assert fake_store.added == []
