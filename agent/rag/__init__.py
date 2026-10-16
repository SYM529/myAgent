"""RAG：文档入库 + 混合检索 + 引用溯源。

模块级刻意不创建客户端——否则一旦没配 Embedding API Key，
整个后端会在导入阶段就崩掉。客户端按需懒加载，由调用方处理不可用情况。
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

from agent.rag.chunker import Chunk, chunk_documents
from agent.rag.config import DOCS_DIR, PERSIST_DIR
from agent.rag.context import get_context
from agent.rag.loader import Document, load_directory
from agent.rag.retriever import Citation, RetrievalResult, Retriever
from agent.rag.store import DeleteResult, HybridStore
from agent.rag.visibility import (
    VISIBILITY_PUBLIC,
    DocAccess,
    can_read,
    classify_document,
    normalize_owner,
)

logger = logging.getLogger(__name__)

_store: HybridStore | None = None
_retriever: Retriever | None = None


def get_store() -> HybridStore:
    """获取向量库实例（单例）。"""
    global _store
    if _store is None:
        _store = HybridStore()
    return _store


def get_retriever() -> Retriever:
    """获取检索器（单例）。"""
    global _retriever
    if _retriever is None:
        _retriever = Retriever(get_store())
    return _retriever


def reset() -> None:
    """重置单例，主要用于测试与切换配置后重建。"""
    global _store, _retriever
    _store = None
    _retriever = None


def summarize_documents(chunks) -> list[dict]:
    """把 chunk 按文档聚合，供"知识库里有哪些文档"的展示使用。

    刻意做成**纯函数**（只吃 chunk 列表、不碰 store）：
    真实的 store 依赖向量库与 Embedding 服务，
    没法在无 API Key 的环境里构造出来，也就没法单测。
    把聚合逻辑抽出来之后，这部分就能用普通测试覆盖。

    owner / file_id / ingested_at 都来自 chunk 的 metadata：
    它们是在上传入库时由服务层写入的，CLI 入库的共享文档没有这些字段。
    """
    grouped: dict[str, dict] = {}
    for chunk in chunks:
        meta = chunk.metadata or {}
        info = grouped.get(chunk.doc_id)
        if info is None:
            info = {
                "doc_id": chunk.doc_id,
                "source": chunk.source,
                "chunks": 0,
                "owner": meta.get("owner") or None,
                "file_id": meta.get("file_id") or None,
                "ingested_at": meta.get("ingested_at") or None,
                "_headings": set(),
            }
            grouped[chunk.doc_id] = info

        info["chunks"] += 1
        # 同一文档重新上传后 source 可能变过，以最后见到的为准
        info["source"] = chunk.source
        # 归属信息只要**任一个** chunk 带着就够：
        # 归属决定了谁能删这份文档，漏读会直接导致"自己的文档删不掉"，
        # 或者更糟——被判成共享文档而谁都删不了。
        for key in ("owner", "file_id", "ingested_at"):
            if info.get(key) is None and meta.get(key):
                info[key] = meta[key]
        heading = meta.get("heading")
        if heading:
            info["_headings"].add(heading)

    documents = []
    for info in grouped.values():
        info["headings"] = sorted(info.pop("_headings"))[:5]
        documents.append(info)

    documents.sort(key=lambda d: (d["source"] or "", d["doc_id"]))
    return documents


def list_documents() -> list[dict]:
    """列出知识库中的文档（聚合视图）。"""
    return summarize_documents(get_store().all_chunks())


def delete_document(doc_id: str) -> DeleteResult:
    """删除一个文档的全部 chunk，返回结构化结果（区分删除成功 / 本就不存在 / 部分失败）。"""
    return get_store().delete_document(doc_id)


def get_document_chunks(doc_id: str) -> list[dict]:
    """取某个文档的全部 chunk（按序），供前端点开预览原文。"""
    return [
        {
            "chunk_index": c.chunk_index,
            "heading": c.metadata.get("heading") or "",
            "text": c.text,
        }
        for c in get_store().get_document_chunks(doc_id)
    ]


def document_access(doc_id: str) -> DocAccess | None:
    """按 doc_id 聚合判定归属与可见性；文档不存在返回 None。

    读取与删除接口统一用它做判定，避免每个端点各写一套"谁能不能看"的规则。
    判定逻辑本身在 ``agent.rag.visibility``，这里只负责把 chunk 聚合成文档。
    """
    chunks = [c for c in get_store().all_chunks() if c.doc_id == doc_id]
    if not chunks:
        return None
    return classify_document([c.metadata or {} for c in chunks])


def list_documents_for(user_id: int, *, is_admin: bool = False) -> list[dict]:
    """列出当前用户可见的文档（服务端过滤，不依赖前端标记）。

    - public：所有已认证用户可见；
    - private：仅 owner 本人可见（管理员也不能看别人的）；
    - unknown / 元数据非法或互相矛盾：任何人（含管理员）都看不到。

    过滤先于搜索与分页：先把不可见的文档整篇剔除，再由接口层做关键词过滤与
    分页，否则"第几页有几条"会随可见集合变化而泄露信息。
    """
    store = get_store()
    chunks = store.all_chunks()
    allowed = _allowed_doc_ids(
        _group_by_document(chunks), user_id=user_id, is_admin=is_admin
    )
    return summarize_documents([c for c in chunks if c.doc_id in allowed])


# ---------- 检索范围（scope） ----------
#
# 判定规则只来自 agent.rag.visibility：先按 doc_id 聚合**整篇**文档的所有 chunk
# 元数据，再交给 classify_document / can_read。列表、检索、预览、删除共用同一套，
# 任何"只抽查一个 chunk"或"缺字段就当公共"的写法都会被测试拦住。


def _group_by_document(chunks: Iterable[Chunk]) -> dict[str, list[Chunk]]:
    """把 chunk 按 doc_id 聚合。可见性判定必须看整篇文档，不能只看某一块。"""
    grouped: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.doc_id, []).append(chunk)
    return grouped


def _allowed_doc_ids(
    grouped: dict[str, list[Chunk]], *, user_id: int, is_admin: bool = False
) -> set[str]:
    """当前用户可见的 doc_id：public + 本人 private；unknown / 他人 private 排除。"""
    return {
        doc_id
        for doc_id, group in grouped.items()
        if can_read(
            classify_document([c.metadata or {} for c in group]),
            user_id=user_id,
            is_admin=is_admin,
        )
    }


def visible_doc_ids_for(
    user_id: int, *, is_admin: bool = False, store: HybridStore | None = None
) -> frozenset[str]:
    """当前用户可检索的 doc_id 全集（public + 本人 private），供检索范围使用。

    与列表、预览、删除共用 ``visibility`` 的判定：unknown、元数据非法或互相矛盾、
    他人 private 一律不在集合里（fail closed）。
    """
    if store is None:
        store = get_store()
    return frozenset(
        _allowed_doc_ids(
            _group_by_document(store.all_chunks()),
            user_id=user_id,
            is_admin=is_admin,
        )
    )


def public_scope(*, store: HybridStore | None = None) -> frozenset[str]:
    """只含**明确声明为 public** 的文档（显式 ``visibility="public"`` 且没有 owner）。

    这是"无身份调用"（MCP、离线脚本、未接线路径）的安全默认范围。

    实现上直接用 ``DocAccess.is_public``：**不能**用
    ``can_read(access, user_id=None)`` 推导——那条路径对"没有可信身份"一律返回
    False，算出来永远是空集合，等于静默地什么都搜不到。
    """
    if store is None:
        store = get_store()
    return frozenset(
        doc_id
        for doc_id, group in _group_by_document(store.all_chunks()).items()
        if classify_document([c.metadata or {} for c in group]).is_public
    )


def all_documents_scope(*, store: HybridStore | None = None) -> frozenset[str]:
    """全量 doc_id，不做任何可见性判定。

    仅限**离线评测 / 运维脚本显式 opt-in**（当前唯一调用点是
    ``evals/retrieval_eval.py``）。它刻意不放进 ``__all__``，也不得成为
    任何在线检索路径的默认值——普通聊天必须走 ``visible_doc_ids_for`` 或
    ``public_scope``。
    """
    if store is None:
        store = get_store()
    return frozenset(c.doc_id for c in store.all_chunks())


def scope_for_request(*, store: HybridStore) -> frozenset[str]:
    """按**当前请求上下文**计算检索范围；认不出身份时一律回退 public。

    这是在线聊天（ReAct 工具与 graph 节点）唯一的"身份 → 检索范围"入口，
    两条路径共用，避免各写一套判定。

    规则（fail closed）：
    - 没有请求上下文 → 仅 public；
    - 有上下文但 ``user_id`` 缺失 / 为 0 / 负数 / 非数字字符串 / 布尔值 /
      其他无效类型 → 仅 public；
    - 身份有效 → ``visible_doc_ids_for(user_id, is_admin=...)``
      （public + 本人 private；管理员**不会**因此看到他人 private）；
    - 解析过程中出现任何意外异常 → 按无身份处理（仅 public），不扩大权限。

    全量范围 ``all_documents_scope()`` 绝不在这里使用。

    ``store`` 是**必填**的：默认值会让"忘记传 store"悄悄构造出一份真实
    HybridStore（进而打开真实 Chroma）。调用方应传"即将用来检索的那个 store"
    （``retriever.store``），保证 scope 与实际检索语料来自同一实例。
    """
    try:
        context = get_context()
        if context is None:
            return public_scope(store=store)

        user_id = normalize_owner(context.user_id)
        if user_id is None:
            return public_scope(store=store)

        return visible_doc_ids_for(
            int(user_id),
            is_admin=bool(context.is_admin),
            store=store,
        )
    except Exception as e:
        # 身份解析出任何意外问题时按"无身份"处理：宁可少给，不可多给
        logger.warning("解析请求身份失败，本次按 public-only 处理：%s", e)
        return public_scope(store=store)


def is_ready() -> bool:
    """知识库是否已就绪（有内容且可检索）。"""
    try:
        return get_store().count() > 0
    except Exception as e:
        logger.warning("知识库不可用：%s", e)
        return False


def ingest_directory(
    directory: Path = DOCS_DIR,
    rebuild: bool = False,
    *,
    visibility: str,
) -> dict:
    """把目录下的文档灌进知识库，返回统计信息。

    visibility 是**必填**的：不提供默认值。可见性一旦能靠缺省推断，
    迟早会把本该私有的资料静默写成公共——"缺 owner 就当公共"正是要根除的旧行为。
    本轮只接受 ``"public"``（CLI 导入即项目公共文档）；私有导入需要 owner，
    属于后续迁移工具的范围，不在本轮实现。

    公共文档必须没有归属：owner 与 public 并存属于自相矛盾的元数据，
    会被判定为 unknown，所以这里显式清掉 owner / file_id。

    rebuild=True 时先清空，避免旧文档的 chunk 与新版混在一起。
    """
    if visibility != VISIBILITY_PUBLIC:
        raise ValueError(
            f"本轮仅支持 visibility={VISIBILITY_PUBLIC!r}（收到 {visibility!r}）："
            "私有文档请走 Web 上传；CLI 的私有导入属于后续迁移工具"
        )

    store = get_store()
    if rebuild:
        store.clear()

    documents = load_directory(directory)
    if not documents:
        return {"documents": 0, "chunks": 0, "total": store.count()}

    chunks = chunk_documents(documents)
    for chunk in chunks:
        chunk.metadata.pop("owner", None)
        chunk.metadata.pop("file_id", None)
        chunk.metadata["visibility"] = VISIBILITY_PUBLIC

    written = store.add(chunks)
    return {
        "documents": len(documents),
        "chunks": written,
        "total": store.count(),
        "sources": sorted({c.source for c in chunks}),
        "visibility": VISIBILITY_PUBLIC,
    }


__all__ = [
    "Chunk",
    "Citation",
    "DeleteResult",
    "Document",
    "HybridStore",
    "RetrievalResult",
    "Retriever",
    "chunk_documents",
    "delete_document",
    "document_access",
    "get_retriever",
    "get_store",
    "ingest_directory",
    "is_ready",
    "list_documents",
    "list_documents_for",
    "public_scope",
    "scope_for_request",
    "summarize_documents",
    "visible_doc_ids_for",
    "load_directory",
    "PERSIST_DIR",
    "DOCS_DIR",
]
