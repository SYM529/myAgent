"""无身份检索路径（MCP / 离线脚本 / 未接线路径）的安全默认测试。

工具与 graph 节点在**没有请求级身份上下文**时只能检索 public 文档。这条规则由
``Retriever.retrieve(scope=None) → public_scope()`` 保证，但此前没有任何测试锁定它：
任何人把调用改成"全量"或"传 all_documents_scope()"都不会被拦住。这里把不变量钉死。

隔离方式：
- 全程使用 fake store / fake reranker，**不打开真实 Chroma、不触碰真实 rag_store**；
- ``agent.rag.get_retriever`` 被 monkeypatch 成使用 fake store 的 Retriever
  （工具与节点都在函数内部 ``from agent.rag import get_retriever``，因此替换包属性即可）；
- graph 节点所在模块会 import ``agent.llm``，而后者在导入时调用 ``load_dotenv()``。
  与既有 ``tests/test_knowledge_api_permissions.py`` 同样处理：**先**写入假环境变量，
  python-dotenv 默认不覆盖已存在的值，因此进程里不会引入真实密钥。
- 断言前确认 ``get_context() is None``——所谓"无身份"必须是真的没有上下文。
"""

import ast
import os
from pathlib import Path

import pytest

os.environ.setdefault("LLM_API_KEY", "test-key-not-real")
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")

from agent.graph.nodes import knowledge_expert  # noqa: E402
from agent.rag import all_documents_scope, public_scope  # noqa: E402
from agent.rag.chunker import Chunk  # noqa: E402
from agent.rag.context import get_context, reset_context  # noqa: E402
from agent.rag.retriever import Retriever  # noqa: E402
from agent.rag.store import RetrievalHit  # noqa: E402
from agent.rag.tool import search_knowledge_base  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PUBLIC_DOC = "aaaa1111aaaa1111"
PRIVATE_DOC = "bbbb2222bbbb2222"
UNKNOWN_DOC = "dddd4444dddd4444"

PUBLIC_SOURCE = "公共手册.md"
PRIVATE_SOURCE = "私人笔记.md"
UNKNOWN_SOURCE = "历史文档.md"

PUBLIC_MARKER = "PUBLIC-CONTENT"
PRIVATE_MARKER = "PRIVATE-CONTENT"
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
            PRIVATE_DOC, PRIVATE_SOURCE, f"{PRIVATE_MARKER} 某用户的私有笔记",
            visibility="private", owner="6",
        ),
        make_chunk(UNKNOWN_DOC, UNKNOWN_SOURCE, f"{UNKNOWN_MARKER} 历史无标记文档"),
    ]


class FakeStore:
    """只实现检索路径需要的方法；命中与否可控，便于覆盖"命中/未命中"。"""

    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.scopes: list = []
        self.hits_enabled = True
        self.failure: Exception | None = None

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
        return [
            RetrievalHit(chunk=chunk, score=1.0)
            for chunk in self._chunks
            if chunk.doc_id in visible_doc_ids
        ][:top_k]


class FakeReranker:
    def rerank(self, query, chunks, top_n):
        return [(chunk, 0.9 - 0.01 * index) for index, chunk in enumerate(chunks)]


@pytest.fixture(autouse=True)
def _no_identity():
    """确保每个用例都是"真的没有身份"（引用已改为请求级，无需清全局）。"""
    reset_context()
    yield
    reset_context()


@pytest.fixture
def store():
    return FakeStore(fixture_chunks())


@pytest.fixture
def wired(monkeypatch, store):
    """把 agent.rag 的 store/retriever 都换成 fake，杜绝碰到真实 rag_store。"""
    monkeypatch.setattr("agent.rag.get_store", lambda: store)
    monkeypatch.setattr(
        "agent.rag.get_retriever",
        lambda: Retriever(store, reranker=FakeReranker(), top_k=5),
    )
    return store


class TestToolWithoutIdentity:
    """工具（MCP 与聊天共用同一条实现）在无身份时只能检索 public。"""

    def test_precondition_no_context(self):
        assert get_context() is None

    def test_scope_is_public_only(self, wired):
        search_knowledge_base.invoke({"query": "CONTENT 说明 笔记 文档"})

        assert wired.scopes[-1] == public_scope(store=wired)
        assert wired.scopes[-1] != all_documents_scope(store=wired)
        assert wired.scopes[-1] == frozenset({PUBLIC_DOC})

    def test_output_has_no_private_or_unknown_content(self, wired):
        output = search_knowledge_base.invoke({"query": "CONTENT 说明 笔记 文档"})

        assert PUBLIC_MARKER in output
        for leaked in (PRIVATE_MARKER, UNKNOWN_MARKER, PRIVATE_SOURCE, UNKNOWN_SOURCE):
            assert leaked not in output

    def test_output_citations_are_public_only(self, wired):
        output = search_knowledge_base.invoke({"query": "CONTENT 说明 笔记 文档"})

        assert PUBLIC_SOURCE in output
        assert PRIVATE_DOC not in output
        assert UNKNOWN_DOC not in output

    def test_no_hit_returns_message_without_any_content(self, wired):
        wired.hits_enabled = False

        output = search_knowledge_base.invoke({"query": "CONTENT 说明 笔记 文档"})

        assert "没有检索到" in output
        for leaked in (PUBLIC_MARKER, PRIVATE_MARKER, UNKNOWN_MARKER):
            assert leaked not in output

    def test_no_hit_still_uses_public_scope(self, wired):
        wired.hits_enabled = False

        search_knowledge_base.invoke({"query": "CONTENT"})

        assert wired.scopes[-1] == public_scope(store=wired)


class TestToolFailurePath:
    def test_error_message_is_readable_and_carries_no_secrets(self, wired):
        wired.failure = RuntimeError(
            f"boom SECRET-TOKEN-abc {PRIVATE_MARKER} {PRIVATE_SOURCE}"
        )

        output = search_knowledge_base.invoke({"query": "CONTENT"})

        assert "知识库检索失败" in output
        assert "RuntimeError" in output
        for leaked in ("SECRET-TOKEN-abc", PRIVATE_MARKER, PRIVATE_SOURCE):
            assert leaked not in output

    def test_failure_does_not_leak_other_documents(self, wired):
        wired.failure = ValueError(f"{UNKNOWN_MARKER} {UNKNOWN_SOURCE}")

        output = search_knowledge_base.invoke({"query": "CONTENT"})

        assert UNKNOWN_MARKER not in output
        assert UNKNOWN_SOURCE not in output
        assert PUBLIC_MARKER not in output


class TestGraphNodeWithoutIdentity:
    def test_context_and_citations_are_public_only(self, wired):
        result = knowledge_expert({"question": "CONTENT 说明 笔记 文档"})

        assert PUBLIC_MARKER in result["knowledge_context"]
        assert PRIVATE_MARKER not in result["knowledge_context"]
        assert UNKNOWN_MARKER not in result["knowledge_context"]
        assert {c["source"] for c in result["citations"]} == {PUBLIC_SOURCE}

    def test_scope_is_public_only(self, wired):
        knowledge_expert({"question": "CONTENT"})

        assert wired.scopes[-1] == public_scope(store=wired)
        assert wired.scopes[-1] != all_documents_scope(store=wired)

    def test_no_hit_returns_empty_context(self, wired):
        wired.hits_enabled = False

        assert knowledge_expert({"question": "CONTENT"}) == {
            "knowledge_context": "",
            "citations": [],
        }

    def test_failure_reports_only_the_error_type(self, wired):
        wired.failure = RuntimeError(f"boom SECRET-TOKEN-abc {PRIVATE_MARKER}")

        result = knowledge_expert({"question": "CONTENT"})

        assert result["errors"] == ["知识库检索失败：RuntimeError"]
        assert "SECRET-TOKEN-abc" not in str(result)
        assert PRIVATE_MARKER not in str(result)


def referenced_names(relative_path: str) -> set[str]:
    """用 AST 收集模块里出现的名字（忽略注释、docstring 与空白格式）。

    比正则更抗误报：注释与文档字符串在 AST 里根本不出现。
    """
    tree = ast.parse((PROJECT_ROOT / relative_path).read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.alias):
            names.add(node.name.split(".")[-1])
            if node.asname:
                names.add(node.asname)
    return names


class TestAllDocumentsScopeStaysOffline:
    """全量 scope 只允许出现在离线评测里，不能进任何在线路径。"""

    @pytest.mark.parametrize("relative", [
        "agent/rag/tool.py",
        "agent/graph/nodes.py",
        "agent/mcp_server.py",
        "server/main.py",
    ])
    def test_online_paths_do_not_reference_all_documents_scope(self, relative):
        assert "all_documents_scope" not in referenced_names(relative)

    def test_eval_script_is_the_explicit_opt_in(self):
        # 正向对照：全量 scope 必须仍被离线评测显式使用，否则这条测试失去意义
        assert "all_documents_scope" in referenced_names("evals/retrieval_eval.py")

    def test_not_exported_from_package(self):
        tree = ast.parse(
            (PROJECT_ROOT / "agent/rag/__init__.py").read_text(encoding="utf-8")
        )
        exported: set[str] = set()
        for node in tree.body:
            targets = getattr(node, "targets", [])
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "__all__"
                for target in targets
            ):
                exported = {
                    element.value
                    for element in node.value.elts
                    if isinstance(element, ast.Constant)
                }

        assert exported, "没解析到 __all__，测试本身需要更新"
        assert "all_documents_scope" not in exported
        assert "public_scope" in exported
        assert "visible_doc_ids_for" in exported
