"""上传文件的接收、存储与管控。

三条安全底线
------------
1. **不使用客户端给的文件名落盘。** 文件名是用户输入，包含 `../`、空字节、
   超长名、Windows 保留名都可能导致路径穿越或写入失败。这里统一生成
   `{uuid}{后缀}` 作为磁盘名，原始名只用于展示和类型判断。
2. **按用户隔离 + resolve 后校验。** 文件 ID 形如 `u1/xxx.png`，解析时先比对用户段，
   再把路径 `resolve()` 后确认仍在上传根目录内——防止符号链接或 `..` 逃出目录。
3. **限制类型与大小。** 只放行图片与可解析文档。图片是按 token 计费的
   （会被编码成 base64 发给模型），大小上限远比普通附件敏感。

为什么要这段代码而不是直接用 `UploadFile.filename`
------------------------------------------------
`filename` 完全由客户端控制。把它拼进路径就等于把"往哪个路径写文件"
这件事交给了请求方，这是典型的路径穿越漏洞。
"""

from __future__ import annotations

import contextvars
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

UPLOAD_ROOT = Path(os.getenv("UPLOAD_DIR", "./uploads"))

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
# 与 agent.rag.loader.SUPPORTED_SUFFIXES 保持一致，只是拆成"文档"这一类
DOC_SUFFIXES = {
    ".pdf", ".md", ".markdown", ".txt", ".log",
    ".csv", ".html", ".htm", ".json", ".yaml", ".yml",
    ".docx", ".xlsx", ".pptx",
}

# 图片会被编码成 base64 塞进模型请求，体积直接换算成 token 成本；
# 文档只取文本嵌入，放宽一些。
MAX_IMAGE_BYTES = int(os.getenv("UPLOAD_MAX_IMAGE_MB", "10")) * 1024 * 1024
MAX_DOC_BYTES = int(os.getenv("UPLOAD_MAX_DOC_MB", "20")) * 1024 * 1024

KIND_IMAGE = "image"
KIND_DOCUMENT = "document"


class UploadRejected(Exception):
    """上传被拒绝。携带可直接展示给用户的原因。"""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def suffix_of(filename: str) -> str:
    """取后缀并转小写。没有后缀返回空串。"""
    if not filename:
        return ""
    name = Path(filename).name
    if "." not in name:
        return ""
    return name[name.rfind("."):].lower()


def classify_file(filename: str) -> str:
    """判断文件类别，返回 'image' / 'document'，不支持时抛 UploadRejected。

    只用后缀判断是不够严谨的（改后缀就能绕过），这里做的是"快速拒绝"：
    真正的内容合法性由后续的解析器负责——图片能否被 Base64 正常编码、
    文档能否被 loader 读出文本。任一环节失败都会被接住并提示用户。
    """
    suffix = suffix_of(filename)
    if suffix in IMAGE_SUFFIXES:
        return KIND_IMAGE
    if suffix in DOC_SUFFIXES:
        return KIND_DOCUMENT
    raise UploadRejected(
        f"不支持的文件类型：{suffix or '未知'}，"
        f"支持图片（{'/'.join(sorted(IMAGE_SUFFIXES))}）"
        f"与文档（{'/'.join(sorted(DOC_SUFFIXES))}）"
    )


def max_bytes_for(kind: str) -> int:
    return MAX_IMAGE_BYTES if kind == KIND_IMAGE else MAX_DOC_BYTES


def check_size(kind: str, size: int) -> None:
    limit = max_bytes_for(kind)
    if size <= 0:
        raise UploadRejected("文件内容为空")
    if size > limit:
        raise UploadRejected(
            f"文件过大（{size / 1024 / 1024:.1f}MB），"
            f"上限 {limit / 1024 / 1024:.0f}MB"
        )


def user_dir(user_id: int, root: Path = UPLOAD_ROOT) -> Path:
    """用户上传目录。"""
    return root / f"u{user_id}"


def resolve_user_file(user_id: int, file_id: str, root: Path = UPLOAD_ROOT) -> Path:
    """把 file_id 解析为磁盘路径，并做权属与逃逸校验。

    file_id 形如 `u1/<uuid>.png`，第一段必须是当前用户的目录名，
    相当于在服务层再做一次越权检查——即使 ID 泄露，别人也解析不到你的文件。
    """
    if not file_id or "/" not in file_id:
        raise UploadRejected("file_id 非法")

    owner, name = file_id.split("/", 1)
    if owner != f"u{user_id}":
        raise UploadRejected("无权访问该文件")
    if not name or "/" in name or "\\" in name or name.startswith("."):
        raise UploadRejected("file_id 非法")

    path = (root / owner / name).resolve()
    root_resolved = root.resolve()
    if not str(path).startswith(str(root_resolved) + os.sep):
        raise UploadRejected("文件路径越界")
    if not path.is_file():
        raise UploadRejected("文件不存在或已过期")

    return path


# 当前请求所属的用户。
# 用 ContextVar 是因为并发请求分属不同 task，它会按 task 自动隔离，
# 不需要加锁，也不会像模块级全局变量那样在并发下串到别人的请求里。
current_user_id: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "current_user_id", default=None
)


def resolve_uploaded(file_id: str) -> Path:
    """按当前请求的用户身份解析 file_id，供 Agent 的图片工具调用。

    工具由模型调用，参数不可信，所以归属判断必须在服务端完成——
    file_id 里虽然带了 `u{user_id}`，真正的校验依据是当前请求的登录用户。
    """
    user_id = current_user_id.get()
    if user_id is None:
        raise UploadRejected("无法确认当前请求的用户身份")
    return resolve_user_file(user_id, file_id)


def save_upload(
    user_id: int, filename: str, data: bytes, root: Path = UPLOAD_ROOT
) -> dict:
    """保存上传内容，返回可直接回给前端的描述。

    磁盘名用 uuid 重新生成，与原始文件名彻底解耦。
    """
    kind = classify_file(filename)
    check_size(kind, len(data))

    directory = user_dir(user_id, root)
    directory.mkdir(parents=True, exist_ok=True)

    suffix = suffix_of(filename)
    disk_name = f"{uuid.uuid4().hex}{suffix}"
    path = directory / disk_name
    path.write_bytes(data)

    logger.info("接收上传：user=%s kind=%s size=%d -> %s", user_id, kind, len(data), path)

    return {
        "file_id": f"u{user_id}/{disk_name}",
        "filename": Path(filename).name,
        "kind": kind,
        "size": len(data),
        "path": str(path),
    }


def ingest_document_into_kb(
    path: Path,
    display_name: str | None = None,
    owner: int | None = None,
    file_id: str | None = None,
) -> dict:
    """把单个文档增量写入知识库。

    Web 上传的文档**一律是私人文档**：每个 chunk 都必须带一致的
    ``owner`` 与 ``visibility='private'``，缺一不可。没有 owner 的文档在判定规则里
    是"来源不明"，只有 CLI 显式声明的公共文档才允许无归属——所以这里 owner 缺失
    直接拒绝，而不是静默写成共享文档。

    doc_id 带用户命名空间（见 ``load_document`` 的 namespace 参数）：不同用户上传
    同一份内容各得一份记录，不会互相覆盖归属与源文件；同一用户重复上传仍幂等。

    返回写入结果。这里刻意**不**调用 rebuild：
    新增自己的文档，不应该把别人（或示例文档）的知识清掉。

    幂等性是免费的：`loader.load_document` 用内容哈希做 doc_id，
    `store.add` 走 upsert——同一份文档重复上传不会产生重复 chunk。
    """
    from agent.rag import get_store
    from agent.rag.chunker import chunk_documents
    from agent.rag.loader import load_document
    from agent.rag.visibility import VISIBILITY_PRIVATE

    if owner is None:
        raise ValueError("Web 上传必须带 owner；公共文档请走 CLI 导入")
    owner_text = str(owner)
    if not owner_text.isdigit() or int(owner_text) <= 0:
        raise ValueError(f"owner 非法：{owner_text!r}")

    document = load_document(path, root=path.parent, namespace=f"u{owner_text}")
    # 磁盘名是出于安全随机生成的 UUID，但它会一路传递到"引用来源"展示给用户。
    # 必须改回用户上传时的原始文件名，否则引用里全是 UUID，用户根本无从核验。
    document.source = display_name or path.name

    chunks = chunk_documents([document])

    # 归属信息写进每个 chunk 的 metadata：
    # 知识库是所有人共用的，只靠"来源文件名"无法判断这份文档是谁传的，
    # 而没有归属信息就无法安全地开放删除接口——任何人都能删掉别人的资料。
    ingested_at = datetime.now().isoformat(timespec="seconds")
    for chunk in chunks:
        chunk.metadata["ingested_at"] = ingested_at
        chunk.metadata["owner"] = owner_text
        chunk.metadata["visibility"] = VISIBILITY_PRIVATE
        if file_id:
            chunk.metadata["file_id"] = file_id
    written = get_store().add(chunks)
    return {
        "ingested": True,
        "chunks": written,
        "source": document.source,
        "total": get_store().count(),
    }
