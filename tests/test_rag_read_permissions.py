"""读取/删除权限在数据层的测试。

这里刻意**不 mock 权限判断**：真实调用 ``classify_document``、``can_read``、
``can_delete``，只把存储层换成假 store（不实例化 HybridStore、不碰 Chroma、
不写 chunks.json）。要验证的是"规则本身在数据层被正确执行"，而不是"调用方
拿到了 mock 返回值"。

覆盖：public / private / unknown / 非法标记 / 冲突元数据，以及普通用户与
管理员的边界。
"""

import pytest

from agent.rag import document_access, list_documents_for
from agent.rag.chunker import Chunk
from agent.rag.visibility import can_delete, can_read

ALICE = 6
BOB = 7
ADMIN = 1


class FakeStore:
    """只回答 all_chunks()，别无其他能力。"""

    def __init__(self, chunks):
        self._chunks = list(chunks)

    def all_chunks(self):
        return list(self._chunks)


@pytest.fixture
def store(monkeypatch):
    """把 agent.rag.get_store 换成假 store（按用例再填充数据）。"""
    holder = {"chunks": []}

    def fake_get_store():
        return FakeStore(holder["chunks"])

    monkeypatch.setattr("agent.rag.get_store", fake_get_store)
    return holder


def doc(doc_id: str, source: str, count: int = 2, **meta) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"{doc_id}-{index}",
            doc_id=doc_id,
            source=source,
            chunk_index=index,
            text=f"{source} 第 {index} 段",
            metadata=dict(meta),
        )
        for index in range(count)
    ]


def ids_of(documents) -> set[str]:
    return {item["doc_id"] for item in documents}


PUBLIC = "aaaa1111aaaa1111"
ALICE_DOC = "bbbb2222bbbb2222"
BOB_DOC = "cccc3333cccc3333"
UNKNOWN = "dddd4444dddd4444"
BAD_MARKER = "eeee5555eeee5555"
CONFLICTED = "ffff6666ffff6666"


@pytest.fixture
def mixed_store(store):
    """六类文档：public、A 私有、B 私有、unknown、非法标记、冲突元数据。"""
    store["chunks"] = (
        doc(PUBLIC, "公共手册.md", visibility="public")
        + doc(ALICE_DOC, "Alice笔记.md", visibility="private", owner=str(ALICE))
        + doc(BOB_DOC, "Bob笔记.md", visibility="private", owner=str(BOB))
        + doc(UNKNOWN, "历史文档.md")                       # 无 owner、无 visibility
        + doc(BAD_MARKER, "坏标记.md", visibility="shared")  # 非法标记
        + doc(CONFLICTED, "矛盾.md", visibility="public", owner=str(ALICE))  # public 带 owner
    )
    return store


class TestListVisibility:
    def test_owner_sees_own_private_and_public(self, mixed_store):
        documents = list_documents_for(ALICE)

        assert ids_of(documents) == {PUBLIC, ALICE_DOC}

    def test_other_user_cannot_see_private_document(self, mixed_store):
        documents = list_documents_for(BOB)

        assert ALICE_DOC not in ids_of(documents)
        assert BOB_DOC in ids_of(documents)

    def test_admin_cannot_see_others_private_document(self, mixed_store):
        documents = list_documents_for(ADMIN, is_admin=True)

        assert ALICE_DOC not in ids_of(documents)
        assert BOB_DOC not in ids_of(documents)
        assert ids_of(documents) == {PUBLIC}          # 管理员也只多看到 public

    def test_public_is_visible_to_everyone(self, mixed_store):
        for user_id, is_admin in ((ALICE, False), (BOB, False), (ADMIN, True)):
            assert PUBLIC in ids_of(list_documents_for(user_id, is_admin=is_admin))

    def test_unknown_and_invalid_documents_are_hidden_from_all(self, mixed_store):
        for user_id, is_admin in ((ALICE, False), (BOB, False), (ADMIN, True)):
            visible = ids_of(list_documents_for(user_id, is_admin=is_admin))

            assert UNKNOWN not in visible
            assert BAD_MARKER not in visible
            assert CONFLICTED not in visible

    def test_filtering_happens_before_search_and_paging(self, mixed_store):
        # 列表函数只返回可见文档；搜索/分页在接口层做，看不到的文档根本不在结果里
        documents = list_documents_for(BOB)

        assert all(item["doc_id"] in {PUBLIC, BOB_DOC} for item in documents)
        assert all("Alice" not in (item["source"] or "") for item in documents)


class TestDocumentAccess:
    def test_missing_document_returns_none(self, mixed_store):
        assert document_access("1111222233334444") is None

    def test_legacy_owner_only_document_is_private(self, store):
        store["chunks"] = doc(ALICE_DOC, "旧文档.md", owner=str(ALICE))

        access = document_access(ALICE_DOC)

        assert access is not None and access.is_private
        assert access.owner == str(ALICE)

    @pytest.mark.parametrize("doc_id", [UNKNOWN, BAD_MARKER, CONFLICTED])
    def test_unusable_metadata_is_unknown(self, mixed_store, doc_id):
        access = document_access(doc_id)

        assert access is not None and access.is_unknown

    def test_private_visibility_is_row_scoped_not_chunk_scoped(self, mixed_store):
        # 归属写在后面的 chunk 上也必须算数（与 summarize_documents 一致）
        chunks = doc(ALICE_DOC, "笔记.md")
        chunks[1].metadata.update({"visibility": "private", "owner": str(ALICE)})
        mixed_store["chunks"] = chunks

        access = document_access(ALICE_DOC)

        assert access is not None and access.is_private and access.owner == str(ALICE)


class TestDeleteDecisions:
    """删除判定直接调用真实的 can_delete。"""

    def _access(self, doc_id):
        access = document_access(doc_id)
        assert access is not None
        return access

    def test_owner_can_delete_own_private(self, mixed_store):
        assert can_delete(self._access(ALICE_DOC), user_id=ALICE) is True

    def test_other_user_cannot_delete_private(self, mixed_store):
        assert can_delete(self._access(ALICE_DOC), user_id=BOB) is False

    def test_admin_cannot_delete_others_private(self, mixed_store):
        assert (
            can_delete(self._access(ALICE_DOC), user_id=ADMIN, is_admin=True) is False
        )

    def test_normal_user_cannot_delete_public(self, mixed_store):
        assert can_delete(self._access(PUBLIC), user_id=ALICE) is False

    def test_admin_can_delete_public(self, mixed_store):
        assert can_delete(self._access(PUBLIC), user_id=ADMIN, is_admin=True) is True

    @pytest.mark.parametrize("doc_id", [UNKNOWN, BAD_MARKER, CONFLICTED])
    def test_unknown_documents_cannot_be_deleted_by_anyone(self, mixed_store, doc_id):
        access = self._access(doc_id)

        assert can_delete(access, user_id=ALICE) is False
        assert can_delete(access, user_id=ADMIN, is_admin=True) is False

    @pytest.mark.parametrize("doc_id", [UNKNOWN, BAD_MARKER, CONFLICTED])
    def test_unknown_documents_cannot_be_read_by_anyone(self, mixed_store, doc_id):
        access = self._access(doc_id)

        assert can_read(access, user_id=ALICE) is False
        assert can_read(access, user_id=ADMIN, is_admin=True) is False
