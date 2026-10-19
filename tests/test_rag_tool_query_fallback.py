"""检索工具的短 query 兜底。

模型有时把问题缩成“UI 组件库”这类关键词，丢掉“前端”限定词后混合检索会漏召回。
请求上下文里有用户原问题时，工具应改用原问题检索。
"""

from agent.rag.context import RagRequestContext, reset_context, set_context
from agent.rag.retriever import Citation, RetrievalResult
import agent.rag.tool as tool_module


class FakeRetriever:
    def __init__(self):
        self.queries = []
        self.store = object()

    def retrieve(self, query, *, scope=None):
        self.queries.append(query)
        citation = Citation(
            index=1, source="项目技术说明.md", chunk_index=0,
            heading="整体架构", score=0.9, text="Vue 3 与 Element Plus",
        )
        return RetrievalResult([citation], "[1] Vue 3 与 Element Plus", True)


def run_tool(monkeypatch, query, user_message=""):
    retriever = FakeRetriever()
    monkeypatch.setattr("agent.rag.get_retriever", lambda: retriever)
    monkeypatch.setattr(
        "agent.rag.scope_for_request", lambda *, store: frozenset({"d"})
    )
    set_context(RagRequestContext(user_message=user_message))
    try:
        output = tool_module.search_knowledge_base.invoke({"query": query})
    finally:
        reset_context()
    return retriever.queries, output


def test_short_query_uses_user_message(monkeypatch):
    queries, output = run_tool(
        monkeypatch, "UI 组件库", "前端使用的是什么 UI 组件库？"
    )
    assert queries == ["前端使用的是什么 UI 组件库？"]
    assert "Element Plus" in output


def test_long_query_is_kept(monkeypatch):
    queries, _ = run_tool(
        monkeypatch, "前端使用的是什么 UI 组件库？", "前端使用的是什么 UI 组件库？"
    )
    assert queries == ["前端使用的是什么 UI 组件库？"]


def test_no_user_message_keeps_query(monkeypatch):
    queries, _ = run_tool(monkeypatch, "UI 组件库", "")
    assert queries == ["UI 组件库"]
