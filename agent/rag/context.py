"""请求级 RAG 上下文：当前用户身份 + 请求级引用容器。

为什么必须是请求级
------------------
检索工具由 Agent 在**工作线程**里执行：langchain 的 ``run_in_executor`` 用
``copy_context().run`` 提交，langgraph 的 executor 同样先 ``copy_context()``
再提交。由此得到两条硬约束：

- **读** ContextVar 可靠：工作线程能看到请求任务里设置的身份；
- **写** ContextVar 不可靠：线程内 ``set()`` 只改该线程的副本，不会回传父任务。

所以身份放进 ContextVar（只读方向），引用容器由**父任务创建**、工具只做
``clear()`` / ``extend()`` 原地修改——跨线程可见的是同一个 list 对象，
而不是变量绑定。模块级共享容器在并发下会把 A 的引用下发给 B，
本模块刻意不提供任何全局状态，也不提供"默认对象"。
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass, field


@dataclass
class RagRequestContext:
    """一次请求内共享的 RAG 上下文。

    默认值即"无身份"：``user_id=None``、``is_admin=False``——
    任何拿不到身份的调用都必须按 public-only 处理（fail closed）。
    """

    user_id: int | None = None
    is_admin: bool = False
    citations: list[dict] = field(default_factory=list)
    # 用户本轮原始问题。模型的工具 query 可能被缩成几个关键词，
    # 过短时工具会用它兜底，避免丢掉“前端/后端”这类限定词。
    user_message: str = ""


_current: ContextVar[RagRequestContext | None] = ContextVar(
    "rag_request_context", default=None
)


def get_context() -> RagRequestContext | None:
    """当前请求的上下文；未设置时返回 ``None``。

    ``None`` 表示"无身份调用"（例如 MCP 服务、离线脚本），调用方必须走
    public-only 的安全默认。这里**不会**返回一个带共享 citations 的默认对象——
    那等于把模块级共享状态换个形式带回来。
    """
    return _current.get()


def set_context(context: RagRequestContext) -> Token:
    """设置当前请求的上下文，返回可回滚的 Token。

    由服务层在请求任务内调用（与 ``uploads.current_user_id`` 同一位置）。
    """
    return _current.set(context)


def reset_context() -> None:
    """清除当前上下文（请求收尾与测试用）。"""
    _current.set(None)
