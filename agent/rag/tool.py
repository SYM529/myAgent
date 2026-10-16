"""把知识库检索包装成 Agent 工具。

工具描述会影响模型"什么时候调用它"，所以要写清楚适用与不适用场景，
否则模型要么过度调用、要么根本不用。

引用如何回到当前请求
--------------------
工具由 Agent 在**工作线程**里执行（langchain 的 ``run_in_executor`` 会把上下文
复制进线程），线程内 ``ContextVar.set()`` 不会回传父任务——所以引用不能靠
"重新绑定变量"传递。这里统一写进 ``RagRequestContext.citations``：父任务创建的
同一个 list 对象，工具只做原地 ``clear()`` / ``extend()``。

没有请求上下文时（MCP、脚本）什么都不写：既没有模块级全局，也不新建任何
共享引用容器，避免"换个形式的全局状态"在并发下串号。
"""

from __future__ import annotations

import logging

from langchain_core.tools import tool

from agent.rag.context import get_context

logger = logging.getLogger(__name__)

ANSWER_INSTRUCTION = (
    "请基于以上资料回答，并在相关句子末尾用 [编号] 标注引用来源。"
    "只使用资料中明确包含的信息；若资料不足以回答，直接说明知识库中没有，"
    "不要补充你自己的推测。"
)


def _write_citations(context, result) -> None:
    """把本次引用写进当前请求；**原地**修改，绝不重新绑定容器。"""
    if context is None:
        return          # 无请求上下文（MCP/脚本）：不产生任何共享状态
    context.citations.clear()
    context.citations.extend(
        {
            "index": c.index,
            "source": c.source,
            "heading": c.heading,
            "score": c.score,
            "text": c.text,
        }
        for c in result.citations
    )


def _clear_citations(context) -> None:
    """清空当前请求的引用（未命中或失败时不允许残留上一次的结果）。"""
    if context is not None:
        context.citations.clear()


@tool
def search_knowledge_base(query: str) -> str:
    """检索本地知识库，获取与问题相关的原始资料片段。

    适用于：用户问题涉及项目文档、产品说明、内部资料、上传过的文件内容等
    可能存在于知识库中的信息。

    不适用于：天气、新闻、股价等实时信息（应使用联网搜索）。

    输入应为能表达检索意图的关键词或完整问题。
    """
    context = get_context()
    try:
        from agent.rag import get_retriever, scope_for_request

        retriever = get_retriever()
        # 请求级检索范围：用**与检索同一个 store** 计算（避免默认参数另取一份 store），
        # 无身份/身份无效时由 scope_for_request 回退到 public。
        # 范围在**召回之前**传给 Retriever，越界片段不会进入精排与上下文。
        scope = scope_for_request(store=retriever.store)
        result = retriever.retrieve(query, scope=scope)
    except Exception as e:
        # 工具失败不能让整个 Agent 崩掉，返回可读错误让模型自行降级。
        # 同时清空本请求的引用：失败事件不能携带同一请求里上一次成功调用的引用
        # （只影响本请求的容器；无上下文时没有容器可清）。
        _clear_citations(context)
        logger.warning("知识库检索失败：%s", e)
        return f"知识库检索失败：{type(e).__name__}。可以改用联网搜索，或如实告知用户。"

    if not result.hit:
        _clear_citations(context)      # 未命中：不允许留上一次的残留引用
        return (
            "知识库中没有检索到相关资料。请不要编造内容："
            "可以如实告知用户知识库里没有，或改用联网搜索获取公开信息。"
        )

    _write_citations(context, result)

    citations = "\n".join(
        f"[{c.index}] {c.source}" + (f" · {c.heading}" if c.heading else "")
        for c in result.citations
    )
    return (
        f"检索到 {len(result.citations)} 条相关资料：\n\n"
        f"{result.context}\n\n"
        f"引用来源对照：\n{citations}\n\n"
        f"{ANSWER_INSTRUCTION}"
    )
