"""知识库列表/预览/删除接口的权限测试（真实端点代码 + 假存储）。

只替换**存储层**（`agent.rag.get_store`），权限判定链路——`document_access` →
`classify_document` → `can_read` / `can_delete`——全部是真实代码。因此这里验证的是
"端点真的拒绝了"，而不是"mock 返回了什么"。

导入 `server.main` 之前设置了一组**假**环境变量，用来隔离副作用：

- `RAG_ENABLED=off`：导入期不去构造真实 Chroma（`is_ready()` → `get_store()`）；
- `DATABASE_URL` 指向临时目录：即使导入期建表也落在临时文件里；
- `JWT_SECRET` / `LLM_API_KEY`：给假值，避免读取真实配置；
- `REDIS_URL` 指向不可达端口：限流器按既有逻辑降级为进程内计数。

这些变量在 `load_dotenv()` 之前就已存在，python-dotenv 默认不覆盖，
因此进程里不会引入 `.env` 的真实值。
"""

import atexit
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

_TMP_DB_DIR = Path(tempfile.mkdtemp(prefix="rag-api-test-"))


def _cleanup_tmp_db() -> None:
    """跑完清掉临时库目录。

    必须先 dispose 引擎：Windows 上 SQLite 文件句柄还被连接池占着时，
    rmtree 会静默失败，留下一堆临时目录。
    """
    from server.database import engine

    engine.dispose()
    shutil.rmtree(_TMP_DB_DIR, ignore_errors=True)


atexit.register(_cleanup_tmp_db)
os.environ["RAG_ENABLED"] = "off"
os.environ["VISION_ENABLED"] = "off"
os.environ["JWT_SECRET"] = "test-secret-not-real"
os.environ["LLM_API_KEY"] = "test-key-not-real"
os.environ["DATABASE_URL"] = f"sqlite:///{(_TMP_DB_DIR / 'test.db').as_posix()}"
os.environ["REDIS_URL"] = "redis://127.0.0.1:1/0"

from starlette.testclient import TestClient  # noqa: E402

from agent.rag.chunker import Chunk  # noqa: E402
from agent.rag.store import DeleteResult  # noqa: E402
from server.auth import get_current_user  # noqa: E402
from server.main import app  # noqa: E402

ALICE = 6
BOB = 7
ADMIN = 1

ALICE_USER = SimpleNamespace(id=ALICE, username="alice", is_admin=False)
BOB_USER = SimpleNamespace(id=BOB, username="bob", is_admin=False)
ADMIN_USER = SimpleNamespace(id=ADMIN, username="admin", is_admin=True)

PUBLIC = "aaaa1111aaaa1111"
ALICE_DOC = "bbbb2222bbbb2222"
BOB_DOC = "cccc3333cccc3333"
UNKNOWN = "dddd4444dddd4444"
BAD_MARKER = "eeee5555eeee5555"
CONFLICTED = "ffff6666ffff6666"
MISSING = "1111222233334444"


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


FIXTURE_CHUNKS = (
    doc(PUBLIC, "公共手册.md", visibility="public")
    + doc(ALICE_DOC, "Alice笔记.md", visibility="private", owner=str(ALICE))
    + doc(BOB_DOC, "Bob笔记.md", visibility="private", owner=str(BOB))
    + doc(UNKNOWN, "历史文档.md")
    + doc(BAD_MARKER, "坏标记.md", visibility="shared")
    + doc(CONFLICTED, "矛盾.md", visibility="public", owner=str(ALICE))
)


class FakeStore:
    """只实现端点真正用到的读取/删除能力，并记录删除调用。"""

    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.deleted: list[str] = []
        self.delete_mode = "normal"       # normal / partial / already_absent

    def all_chunks(self):
        return list(self._chunks)

    def get_document_chunks(self, doc_id):
        return [c for c in self._chunks if c.doc_id == doc_id]

    def delete_document(self, doc_id):
        if self.delete_mode == "crash":
            raise RuntimeError(
                "写入失败：PermissionError(13, 'Permission denied', "
                r"'E:\\CodexConfig\\rag_store\\chunks.json.4242-deadbeef.tmp')"
            )
        if self.delete_mode == "partial":
            return DeleteResult(
                doc_id=doc_id, status="partial", chroma_deleted=2,
                chunks_removed=0, json_saved=False, bm25_invalidated=False,
                error=(
                    "元数据写入失败：PermissionError(13, 'Permission denied', "
                    r"'E:\\CodexConfig\\rag_store\\chunks.json.4242-deadbeef.tmp')"
                ),
            )
        if self.delete_mode == "already_absent":
            return DeleteResult(
                doc_id=doc_id, status="already_absent", chroma_deleted=0,
                chunks_removed=0, json_saved=False, bm25_invalidated=False,
            )
        removed = [c for c in self._chunks if c.doc_id == doc_id]
        self._chunks = [c for c in self._chunks if c.doc_id != doc_id]
        self.deleted.append(doc_id)
        return DeleteResult(
            doc_id=doc_id, status="deleted",
            chroma_deleted=len(removed), chunks_removed=len(removed),
            json_saved=True, bm25_invalidated=True,
        )


@pytest.fixture
def store(monkeypatch):
    fake = FakeStore(FIXTURE_CHUNKS)
    monkeypatch.setattr("agent.rag.get_store", lambda: fake)
    return fake


class Api:
    """把 TestClient 包一层，负责切换当前登录用户。"""

    def __init__(self, store):
        self.client = TestClient(app)
        self.store = store

    def as_user(self, user):
        app.dependency_overrides[get_current_user] = lambda: user
        return self

    def list_docs(self, **params):
        response = self.client.get("/api/knowledge", params=params)
        assert response.status_code == 200, response.text
        return response.json()

    def preview(self, doc_id):
        return self.client.get(f"/api/knowledge/{doc_id}/content")

    def delete(self, doc_id):
        return self.client.delete(f"/api/knowledge/{doc_id}")


@pytest.fixture
def api(store):
    instance = Api(store)
    yield instance
    app.dependency_overrides.clear()


class TestListPermissions:
    def test_owner_sees_own_private_and_public(self, api):
        documents = api.as_user(ALICE_USER).list_docs()

        assert {d["doc_id"] for d in documents["documents"]} == {PUBLIC, ALICE_DOC}
        assert next(d for d in documents["documents"] if d["doc_id"] == ALICE_DOC)["mine"]

    def test_other_user_does_not_see_private_document(self, api):
        documents = api.as_user(BOB_USER).list_docs()

        visible = {d["doc_id"] for d in documents["documents"]}
        assert ALICE_DOC not in visible
        assert BOB_DOC in visible

    def test_admin_does_not_see_others_private_document(self, api):
        documents = api.as_user(ADMIN_USER).list_docs()

        visible = {d["doc_id"] for d in documents["documents"]}
        assert visible == {PUBLIC}

    @pytest.mark.parametrize("user", [ALICE_USER, BOB_USER, ADMIN_USER])
    @pytest.mark.parametrize("doc_id", [UNKNOWN, BAD_MARKER, CONFLICTED])
    def test_unknown_documents_never_appear_in_lists(self, api, user, doc_id):
        documents = api.as_user(user).list_docs()
        assert doc_id not in {d["doc_id"] for d in documents["documents"]}

    def test_server_side_filtering_survives_search_and_paging(self, api):
        # 用文件名搜索 + mine=true 也不能把他人/不可见文档"搜"出来
        searched = api.as_user(BOB_USER).list_docs(q="Alice", page=1, page_size=50)
        mine = api.as_user(BOB_USER).list_docs(mine=True, page=1, page_size=50)

        assert searched["total"] == 0
        assert {d["doc_id"] for d in mine["documents"]} == {BOB_DOC}


class TestPreviewPermissions:
    def test_owner_can_preview_own_private(self, api):
        response = api.as_user(ALICE_USER).preview(ALICE_DOC)

        assert response.status_code == 200
        assert response.json()["doc_id"] == ALICE_DOC
        assert response.json()["chunks"]

    def test_other_user_cannot_preview_private(self, api):
        assert api.as_user(BOB_USER).preview(ALICE_DOC).status_code == 404

    def test_admin_cannot_preview_others_private(self, api):
        assert api.as_user(ADMIN_USER).preview(ALICE_DOC).status_code == 404

    @pytest.mark.parametrize("user", [ALICE_USER, BOB_USER, ADMIN_USER])
    def test_public_is_previewable_by_authenticated_users(self, api, user):
        assert api.as_user(user).preview(PUBLIC).status_code == 200

    @pytest.mark.parametrize("doc_id", [UNKNOWN, BAD_MARKER, CONFLICTED])
    def test_unknown_or_invalid_documents_are_404(self, api, doc_id):
        assert api.as_user(ALICE_USER).preview(doc_id).status_code == 404
        assert api.as_user(ADMIN_USER).preview(doc_id).status_code == 404

    def test_missing_document_is_404(self, api):
        assert api.as_user(ALICE_USER).preview(MISSING).status_code == 404

    def test_denied_preview_matches_missing_document_response(self, api):
        denied = api.as_user(BOB_USER).preview(ALICE_DOC)
        missing = api.as_user(BOB_USER).preview(MISSING)

        assert denied.status_code == missing.status_code == 404
        assert denied.json() == missing.json()


class TestDeletePermissions:
    def test_owner_can_delete_own_private(self, api, store):
        response = api.as_user(ALICE_USER).delete(ALICE_DOC)

        assert response.status_code == 200
        assert response.json()["status"] == "deleted"
        assert store.deleted == [ALICE_DOC]

    def test_other_user_cannot_delete_private(self, api, store):
        assert api.as_user(BOB_USER).delete(ALICE_DOC).status_code == 404
        assert store.deleted == []

    def test_normal_user_cannot_delete_public(self, api, store):
        assert api.as_user(ALICE_USER).delete(PUBLIC).status_code == 404
        assert store.deleted == []

    def test_admin_can_delete_public(self, api, store):
        response = api.as_user(ADMIN_USER).delete(PUBLIC)

        assert response.status_code == 200
        assert store.deleted == [PUBLIC]

    def test_admin_cannot_delete_others_private(self, api, store):
        assert api.as_user(ADMIN_USER).delete(ALICE_DOC).status_code == 404
        assert store.deleted == []

    @pytest.mark.parametrize("doc_id", [UNKNOWN, BAD_MARKER, CONFLICTED])
    def test_nobody_can_delete_unknown(self, api, store, doc_id):
        assert api.as_user(ALICE_USER).delete(doc_id).status_code == 404
        assert api.as_user(ADMIN_USER).delete(doc_id).status_code == 404
        assert store.deleted == []

    def test_missing_document_is_404(self, api, store):
        assert api.as_user(ALICE_USER).delete(MISSING).status_code == 404
        assert store.deleted == []

    def test_denied_delete_matches_missing_document_response(self, api):
        denied = api.as_user(ADMIN_USER).delete(ALICE_DOC)
        missing = api.as_user(ADMIN_USER).delete(MISSING)

        assert denied.status_code == missing.status_code == 404
        assert denied.json() == missing.json()


class _RecordingLogger:
    """只记录调用文本的假 logger，用来验证"完整异常确实进了服务端日志"。"""

    def __init__(self):
        self.messages: list[str] = []

    def _record(self, message, *args, **_kwargs):
        self.messages.append(str(message) % args if args else str(message))

    def info(self, message, *args, **kwargs):
        self._record(message, *args, **kwargs)

    def warning(self, message, *args, **kwargs):
        self._record(message, *args, **kwargs)

    def error(self, message, *args, **kwargs):
        self._record(message, *args, **kwargs)

    def exception(self, message, *args, **kwargs):
        self._record(message, *args, **kwargs)


def assert_no_internal_leak(text: str) -> None:
    """HTTP 响应里不得出现底层异常、路径、临时文件名等内部实现细节。"""
    lowered = text.lower()
    for marker in (
        "permissionerror", "oserror", "chunks.json", ".tmp", "codexconfig", "rag_store",
    ):
        assert marker not in lowered, f"响应泄露内部细节：{marker}"
    assert not re.search(r"[A-Za-z]:\\", text), "响应泄露 Windows 盘符路径"
    assert not re.search(r"/(?:usr|home|var|tmp|app|root)/", text), "响应泄露 POSIX 绝对路径"


class TestDeleteOutcomes:
    """阶段 8B：删除结果结构在 HTTP 层的语义（成功 / 幂等 / 部分失败 / 源文件告警）。"""

    def test_successful_delete_reports_full_result(self, api, store):
        response = api.as_user(ALICE_USER).delete(ALICE_DOC)

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "deleted"
        assert body["doc_id"] == ALICE_DOC
        assert body["chunks"] == 2
        assert body["already_deleted"] is False
        assert body["warnings"] == []

    def test_already_absent_is_idempotent_200(self, api, store):
        store.delete_mode = "already_absent"

        response = api.as_user(ALICE_USER).delete(ALICE_DOC)

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "deleted"
        assert body["already_deleted"] is True
        assert body["chunks"] == 0

    def test_partial_failure_is_not_reported_as_success(self, api, store, monkeypatch):
        records: list[dict] = []
        monkeypatch.setattr(
            "server.audit.record",
            lambda action, **kwargs: records.append({"action": action, **kwargs}),
        )
        logger = _RecordingLogger()
        monkeypatch.setattr("server.main.logger", logger)
        store.delete_mode = "partial"

        response = api.as_user(ALICE_USER).delete(ALICE_DOC)

        assert response.status_code == 500
        detail = response.json()["detail"]
        assert "删除未完成" in detail                 # 稳定、用户可理解的文案
        request_id = response.headers.get("x-request-id")
        assert request_id and request_id in detail     # 复用既有 request_id 机制
        assert_no_internal_leak(detail)                # 不回显底层细节
        assert store.deleted == []                 # 绝不能被当成删除成功
        assert records and records[-1]["outcome"] == "failure"
        # 完整原因仍然留在服务端：日志与审计都要能排查
        assert any("PermissionError" in m for m in logger.messages)
        assert any("PermissionError" in (r.get("detail") or "") for r in records)

    def test_chroma_failure_detail_does_not_leak_internals(
        self, api, store, monkeypatch
    ):
        logger = _RecordingLogger()
        monkeypatch.setattr("server.main.logger", logger)
        store.delete_mode = "crash"

        response = api.as_user(ALICE_USER).delete(ALICE_DOC)

        assert response.status_code == 500
        detail = response.json()["detail"]
        assert "删除失败" in detail
        assert_no_internal_leak(detail)
        assert any("PermissionError" in m for m in logger.messages)
        assert store.deleted == []

    def test_sequential_repeat_delete_returns_404(self, api, store):
        """顺序重复删除仍是 404（不依赖 already_deleted 把它变成 200）。"""
        assert api.as_user(ALICE_USER).delete(ALICE_DOC).status_code == 200

        second = api.as_user(ALICE_USER).delete(ALICE_DOC)

        assert second.status_code == 404
        assert second.json()["detail"] == "文档不存在"
        assert "already_deleted" not in second.json()

    def test_source_file_cleanup_failure_is_reported_in_warnings(
        self, api, store, monkeypatch
    ):
        store._chunks.append(
            Chunk(
                chunk_id="file-doc-0", doc_id="ffff8888ffff8888",
                source="带源文件的笔记.md", chunk_index=0, text="内容",
                metadata={
                    "visibility": "private", "owner": str(ALICE),
                    "file_id": "u6/abc.md",
                },
            )
        )

        def broken_resolve(user_id, file_id):
            raise RuntimeError("文件不存在或已过期")

        monkeypatch.setattr(
            "server.main.uploads.resolve_user_file", broken_resolve
        )

        response = api.as_user(ALICE_USER).delete("ffff8888ffff8888")

        assert response.status_code == 200
        body = response.json()
        assert body["warnings"] == ["源文件未删除：RuntimeError"]
        assert "u6/abc.md" not in json.dumps(body)   # 不泄露内部文件标识

    @pytest.mark.parametrize("mode", ["partial", "already_absent"])
    def test_permission_rules_are_unchanged_for_non_owner(self, api, store, mode):
        """无论底层返回什么，越权删除仍然是 404（权限判定在调用之前）。"""
        store.delete_mode = mode

        assert api.as_user(BOB_USER).delete(ALICE_DOC).status_code == 404
        assert api.as_user(ADMIN_USER).delete(ALICE_DOC).status_code == 404


class TestAuditWording:
    """审计日志的 detail 要反映新的权限语义。"""

    def _capture(self, monkeypatch):
        records = []
        monkeypatch.setattr(
            "server.audit.record",
            lambda action, **kwargs: records.append({"action": action, **kwargs}),
        )
        return records

    def test_public_deletion_is_recorded_as_public_document(self, api, monkeypatch):
        records = self._capture(monkeypatch)

        api.as_user(ADMIN_USER).delete(PUBLIC)

        assert records[-1]["detail"] == "公共文档"

    def test_owner_deletion_is_recorded_as_own_document(self, api, monkeypatch):
        records = self._capture(monkeypatch)

        api.as_user(ALICE_USER).delete(ALICE_DOC)

        assert records[-1]["detail"] == "本人文档"
