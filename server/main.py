import asyncio
import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

# 必须先于所有读取环境变量的模块执行：
# server.auth 在导入期校验 JWT_SECRET，server.database 在导入期读取 DATABASE_URL。
# 顺序错了会在启动阶段直接崩，而不是等到某个请求才暴露。
load_dotenv()

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, field_validator
from langchain_core.messages import (
    AIMessage, HumanMessage, SystemMessage, ToolMessage,
)
from sqlalchemy.orm import Session
from agent.pricing import estimate_cost, sum_usage
from agent import resilience
from agent.rag import delete_document as rag_delete_document
from agent.rag.store import DELETE_STATUS_ALREADY_ABSENT
from agent.rag import document_access as rag_document_access
from agent.rag import get_document_chunks as rag_get_document_chunks
from agent.rag import list_documents as rag_list_documents
from agent.rag import list_documents_for as rag_list_documents_for
from agent.rag.context import (
    RagRequestContext,
    reset_context as reset_rag_context,
    set_context as set_rag_context,
)
from agent.rag.visibility import can_delete, can_read
from agent.service import MODEL_NAME, agent, capability_signature
from server import audit
from server import cache as cache_mod
from server import memory, metrics, reports as reports_mod, sessions as sessions_mod
from server import uploads
from server.database import get_db, User
from server.logging_setup import (
    RequestContextMiddleware, current_request_id, request_user_var, setup_logging,
)
from server import prom
from server.ratelimit import (
    SlidingWindowLimiter, build_guard, build_limiter,
)
from server.uploads import current_user_id, resolve_uploaded

from server.auth import (
    hash_password, verify_password, create_access_token, get_current_user
)
import redis

logger = logging.getLogger(__name__)

# 结构化日志必须在任何业务日志之前生效，否则早期日志没有 request_id
setup_logging()

app = FastAPI(title="Agent Service")

# 请求上下文（request_id / 访问日志）。放在最外层，
# 这样后面所有日志都能带上同一次请求的 ID
app.add_middleware(RequestContextMiddleware)

# 生产环境不要再用 "*"：带凭证时必须显式列出来源，否则浏览器会直接拒绝
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

r = redis.Redis.from_url(
    os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    decode_responses=True,
    socket_connect_timeout=2,
    socket_timeout=2,
)

SESSION_TTL = 1800
MAX_HISTORY = 20
# 单请求硬上限。带图请求 = 主模型决策 + 视觉模型看图 + 主模型作答，
# 三段串行天然比纯文本慢，60s 曾在服务商瞬时变慢时被整单掐掉，放宽到 90s
AGENT_TIMEOUT = 90

# 会话保留时长。默认从 30 分钟提高到 7 天：
# 侧边栏要展示历史对话，半小时就消失的话"保存历史"名存实亡。
SESSION_TTL_HOURS = float(os.getenv("SESSION_TTL_HOURS", "168"))
SESSION_TTL = int(SESSION_TTL_HOURS * 3600)

# 限流与并发控制。计数集中在 Redis，Redis 不可用时自动降级为进程内计数。
limiter = build_limiter(r)
guard = build_guard()

# 认证与上传的独立额度：与 Agent 请求分开计数，互不挤占。
# 登录按用户名计数：公网部署在 nginx 后面，request.client.host 是代理地址，
# 按 IP 会把所有用户算进同一个桶。上传按登录用户计数，防止反复触发
# Embedding 计费（一次上传背后是解析 + 向量化 + 入库）。
AUTH_RATE_LIMIT_PER_MIN = int(os.getenv("AUTH_RATE_LIMIT_PER_MIN", "10"))
UPLOAD_RATE_LIMIT_PER_MIN = int(os.getenv("UPLOAD_RATE_LIMIT_PER_MIN", "10"))
auth_limiter = SlidingWindowLimiter(r, limit=AUTH_RATE_LIMIT_PER_MIN)
upload_limiter = SlidingWindowLimiter(r, limit=UPLOAD_RATE_LIMIT_PER_MIN)


def _enforce_quota(counter, identity: str, message: str) -> None:
    """通用额度检查：超限统一返回 429 + Retry-After。"""
    result = counter.check(identity)
    if result.allowed:
        return
    retry = max(1, int(result.retry_after))
    prom.record_rate_limited()
    raise HTTPException(
        status_code=429,
        detail=f"{message}，请 {retry} 秒后重试",
        headers={"Retry-After": str(retry)},
    )



# 上游（模型服务商）熔断器：连续失败达到阈值就快速失败，冷却后再探活
provider_breaker = resilience.get_breaker(
    "llm_provider",
    failure_threshold=int(os.getenv("CIRCUIT_FAILURE_THRESHOLD", "5")),
    reset_timeout=float(os.getenv("CIRCUIT_RESET_SECONDS", "60")),
)
# 状态变化同步到 Prometheus，便于在 Grafana 上看到"什么时候熔断了"
provider_breaker.set_on_change(
    lambda name, state: prom.set_circuit_state(name, state)
)

# 把"file_id -> 文件路径"的解析能力交给 Agent 的图片工具。
# 归属校验在 uploads 内部完成：工具接收到的是模型给的参数，不可信，
# 真正的依据是上下文里当前请求所属的登录用户。
try:
    from agent.vision import set_file_resolver

    set_file_resolver(resolve_uploaded)
except Exception as e:  # 视觉能力未配置时不应影响其余功能
    logger.warning("图片理解工具未挂载：%s", e)

# 语义缓存：不可用时不挂载，与知识库、图片理解保持一致的降级策略
semantic_cache = None
cache_embed = None
if cache_mod.available():
    try:
        cache_embed = cache_mod.get_embedder()
        semantic_cache = cache_mod.SemanticCache()
    except Exception as e:
        logger.warning("语义缓存不可用：%s", e)
        semantic_cache = None
        cache_embed = None

# 结果会随外部状态变化的工具：缓存它们的返回等于给用户过期答案，
# 所以只要本轮用过这些工具，本轮答案就不进缓存。
VOLATILE_TOOLS = {"get_current_time", "web_search"}


def _context_fingerprint(messages) -> str:
    """本轮之前的历史指纹。

    命中判断必须带上它：多轮对话里"它多少钱？"在不同语境下指向不同的东西。
    历史为空时返回空串，代表"新会话"这一固定语境。
    """
    if not messages:
        return ""
    raw = "\n".join(f"{m.type}:{m.content or ''}" for m in messages)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _used_volatile_tool(messages) -> bool:
    """本轮是否用过结果随时间变化的工具。"""
    for msg in messages:
        if msg.type == "tool" and getattr(msg, "name", "") in VOLATILE_TOOLS:
            return True
    return False


def _now_iso() -> str:
    """事件时间戳（UTC ISO 格式）。前端据此做按天分组的时间分隔线。"""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sse(data: dict) -> str:
    """组装一个 SSE 事件。统一补上缺省字段与时间戳，前端不必再判断。"""
    payload = {"content": "", "name": "", "ts": _now_iso()}
    payload.update(data)
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
# 形如 u1/<uuid>.png：第一段是用户目录，第二段由服务端生成。
# 这里只做格式校验，真正的归属判断仍在 uploads 里按登录用户复核。
FILE_ID_PATTERN = re.compile(r"^u\d+/[A-Za-z0-9_-]+\.[A-Za-z0-9]+$")
# doc_id 由 loader 取内容哈希的前 16 位十六进制
DOC_ID_PATTERN = re.compile(r"^[a-f0-9]{16}$")


class RenameRequest(BaseModel):
    title: str

    @field_validator("title")
    @classmethod
    def check_title(cls, v: str) -> str:
        v = v.strip()
        if not 1 <= len(v) <= 60:
            raise ValueError("标题需为 1-60 个字符")
        return v


# ================== 数据模型 ==================

class RegisterRequest(BaseModel):
    username: str
    password: str

    @field_validator("username")
    @classmethod
    def check_username(cls, v: str) -> str:
        v = v.strip()
        if not (3 <= len(v) <= 32) or not USERNAME_PATTERN.match(v):
            raise ValueError("用户名需为 3-32 位字母、数字、下划线、点或短横线")
        return v


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    # 本次问答附带的已上传文件（图片走视觉模型，文档已直接入库）
    file_id: str | None = None

    @field_validator("session_id")
    @classmethod
    def check_session_id(cls, v: str) -> str:
        if not SESSION_ID_PATTERN.match(v):
            raise ValueError("session_id 只能由字母、数字、下划线、短横线组成（1-64 位）")
        return v

    @field_validator("file_id")
    @classmethod
    def check_file_id(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip()
        if not FILE_ID_PATTERN.match(v):
            raise ValueError("file_id 格式非法")
        return v

    @field_validator("message")
    @classmethod
    def check_message(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("消息不能为空")
        if len(v) > 2000:
            raise ValueError("消息过长（上限 2000 字）")
        return v


# ================== 会话存储（按 user_id 隔离） ==================

def session_key(user_id: int, session_id: str) -> str:
    return f"session:{user_id}:{session_id}"


def _msg_to_dict(msg) -> dict:
    """序列化消息。保留 tool_calls，否则多轮对话中模型会丢失工具调用上下文。"""
    data = {"role": msg.type, "content": msg.content or ""}
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        data["tool_calls"] = tool_calls
    if msg.type == "tool":
        data["tool_call_id"] = getattr(msg, "tool_call_id", "") or ""
        data["name"] = getattr(msg, "name", "") or ""
    return data


def _dict_to_msg(data: dict):
    role = data.get("role")
    content = data.get("content", "")
    if role in ("user", "human"):
        return HumanMessage(content=content)
    if role == "system":
        # 长对话压缩后，摘要以 system 消息存在。
        # 反序列化必须还原成 SystemMessage，否则会被当成 AI 回复，
        # 模型就会把"我总结的内容"误认为"我自己说过的话"。
        return SystemMessage(content=content)
    if role == "tool":
        return ToolMessage(
            content=content,
            tool_call_id=data.get("tool_call_id", "") or "",
            name=data.get("name", "") or "",
        )
    return AIMessage(content=content, tool_calls=data.get("tool_calls") or [])


def _trim_history(messages: list) -> list:
    """裁剪历史，并保证首尾完整：tool 消息不能脱离它对应的 AI 消息。"""
    trimmed = list(messages[-MAX_HISTORY:])
    while trimmed and trimmed[0].type == "tool":
        trimmed.pop(0)
    # 丢弃尾部缺少工具结果的 tool_calls，避免模型收到不完整的调用序列
    while trimmed and getattr(trimmed[-1], "tool_calls", None):
        trimmed.pop()
    return trimmed


def get_history(user_id: int, session_id: str) -> list:
    try:
        data = r.get(session_key(user_id, session_id))
    except redis.RedisError as e:
        logger.warning("Redis 不可用，本次请求不读取历史: %s", e)
        return []
    if not data:
        return []
    try:
        raw = json.loads(data)
    except json.JSONDecodeError:
        logger.warning("会话历史解析失败，已忽略")
        return []
    return [_dict_to_msg(m) for m in raw if isinstance(m, dict)]


def save_history(user_id: int, session_id: str, messages: list):
    trimmed = _trim_history(messages)
    try:
        r.setex(
            session_key(user_id, session_id),
            SESSION_TTL,
            json.dumps([_msg_to_dict(m) for m in trimmed], ensure_ascii=False),
        )
    except redis.RedisError as e:
        logger.warning("Redis 不可用，会话历史未持久化: %s", e)


# ================== 认证接口 ==================

@app.post("/api/register")
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    # 公网部署时建议 REGISTER_ENABLED=0：注册消耗的是你自己的 API 额度，
    # 先在本地/服务器上建好账号，再关闭注册口。
    if os.getenv("REGISTER_ENABLED", "1").lower() in ("0", "off", "false", "no"):
        raise HTTPException(status_code=403, detail="当前未开放注册，请联系管理员开通账号")

    if db.query(User).filter(User.username == req.username).first():
        raise HTTPException(status_code=400, detail="用户名已存在")
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="密码至少 6 位")

    user = User(username=req.username, hashed_password=hash_password(req.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    audit.record(
        audit.REGISTER, actor=req.username, target=req.username,
        request_id=current_request_id(),
    )
    return {"id": user.id, "username": user.username}


@app.post("/api/login")
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    # 按用户名限流（统一小写，避免换大小写绕过）。必须在校验密码之前：
    # 否则暴力破解的每次尝试都已经打到 bcrypt 上，限流失去意义。
    _enforce_quota(
        auth_limiter, f"login:{form_data.username.strip().lower()}",
        "登录尝试过于频繁",
    )
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        # 登录失败也要审计：连续失败是最直接的暴力破解信号
        audit.record(
            audit.LOGIN_FAILED, actor=form_data.username, outcome="failure",
            detail="用户名或密码错误", request_id=current_request_id(),
        )
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    audit.record(
        audit.LOGIN, actor=user.username, request_id=current_request_id()
    )
    token = create_access_token(user.id, getattr(user, "password_version", 0) or 0)
    return {
        "access_token": token,
        "token_type": "bearer",
        "username": user.username,
        "is_admin": bool(user.is_admin),
    }


@app.get("/api/me")
def me(current_user: User = Depends(get_current_user)):
    return {
        "id": current_user.id,
        "username": current_user.username,
        "is_admin": bool(current_user.is_admin),
        "created_at": (
            current_user.created_at.isoformat() if current_user.created_at else None
        ),
    }


class PasswordChange(BaseModel):
    old_password: str
    new_password: str

    @field_validator("new_password")
    @classmethod
    def check_length(cls, v: str) -> str:
        if len(v) < 6:
            raise ValueError("新密码至少 6 位")
        return v


@app.post("/api/password")
def change_password(
    req: PasswordChange,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """修改当前用户的密码。

    两步安全处理：
    1. 必须校验旧密码——否则任何人拿到一个未锁屏的浏览器就能改走账号。
    2. 密码版本号自增，让改密前签发的 token 立即失效，并当场签发新 token，
       这样用户改完密码不必重新登录。
    """
    if not verify_password(req.old_password, current_user.hashed_password):
        raise HTTPException(status_code=400, detail="原密码不正确")
    if req.old_password == req.new_password:
        raise HTTPException(status_code=400, detail="新密码不能与原密码相同")

    current_user.hashed_password = hash_password(req.new_password)
    current_user.password_version = int(current_user.password_version or 0) + 1
    db.commit()
    audit.record(
        audit.CHANGE_PASSWORD, actor=current_user.username,
        detail="旧 token 已全部失效", request_id=current_request_id(),
    )

    token = create_access_token(current_user.id, current_user.password_version)
    logger.info("用户 %s 修改了密码", current_user.username)
    return {"status": "changed", "access_token": token}


@app.delete("/api/sessions")
def clear_all_sessions(current_user: User = Depends(get_current_user)):
    """清空当前用户的全部会话（只影响本人）。

    会话本体与索引条目必须一起删：只删索引会留下孤儿历史数据，
    只删历史则侧边栏会出现点进去空白的"幽灵会话"。
    """
    try:
        ids = [s["session_id"] for s in sessions_mod.list_sessions(r, current_user.id)]
        pipe = r.pipeline()
        for session_id in ids:
            pipe.delete(session_key(current_user.id, session_id))
        pipe.delete(sessions_mod.index_key(current_user.id))
        pipe.execute()
    except redis.RedisError as e:
        logger.warning("Redis 不可用，会话未清除: %s", e)
        raise HTTPException(status_code=503, detail="会话服务暂不可用")
    audit.record(
        audit.CLEAR_ALL_SESSIONS, actor=current_user.username,
        target=f"{len(ids)} 个会话", request_id=current_request_id(),
    )
    return {"status": "cleared", "count": len(ids)}


def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """管理员守卫：权限判定必须在服务端做，前端藏按钮只是展示层配合。

    否则任何人直接 curl 报告接口照样能看，"管理员可见"就成了摆设。
    """
    if not bool(getattr(current_user, "is_admin", False)):
        raise HTTPException(status_code=403, detail="仅管理员可访问评测看板")
    return current_user


@app.get("/api/knowledge")
def list_knowledge(
    q: str = "",
    page: int = 1,
    page_size: int = 10,
    mine: bool = False,
    current_user: User = Depends(get_current_user),
):
    """知识库文档列表（按文档聚合，不是按 chunk 罗列）。

    支持按来源文件名模糊搜索 + 分页：文档是长期累积的，
    几十份之后一屏罗列的体验会迅速劣化。

    可见性在**服务端**过滤（``rag_list_documents_for``）：只返回 public 文档与
    本人上传的 private 文档；他人 private 与 unknown 文档对任何人（含管理员）
    都不出现。过滤先于搜索与分页，不能靠前端标记隐藏。

    额外返回 `mine` 字段：前端据此决定能否显示删除按钮，
    避免"点了删除才被告知无权"的体验。
    """
    # 分页参数防御：客户端传什么都不能把服务端打挂
    page = max(1, page)
    page_size = min(max(1, page_size), 50)
    q = (q or "").strip().lower()

    is_admin = bool(getattr(current_user, "is_admin", False))
    try:
        documents = rag_list_documents_for(current_user.id, is_admin=is_admin)
    except Exception as e:
        logger.warning("读取知识库列表失败：%s", e)
        raise HTTPException(status_code=503, detail=f"知识库暂不可用：{e}")

    for doc in documents:
        doc["mine"] = str(doc.get("owner")) == str(current_user.id)

    # mine=true：个人中心用，只看自己上传的文档
    if mine:
        documents = [d for d in documents if d["mine"]]

    if q:
        # 搜索按"来源文件名"匹配：用户对文件名的记忆远强于对 doc_id 哈希的记忆
        documents = [d for d in documents if q in (d.get("source") or "").lower()]

    total = len(documents)
    start = (page - 1) * page_size
    return {
        "documents": documents[start:start + page_size],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@app.get("/api/knowledge/{doc_id}/content")
def knowledge_content(doc_id: str, current_user: User = Depends(get_current_user)):
    """查看某个文档的 chunk 原文（按入库顺序）。

    只有 public 文档与本人上传的 private 文档能看。他人 private、unknown 以及
    元数据非法/矛盾的文档，对任何人（含管理员）都返回 404——与"文档不存在"
    同响应，不泄露它是否存在。
    """
    if not DOC_ID_PATTERN.match(doc_id):
        raise HTTPException(status_code=400, detail="doc_id 格式非法")

    is_admin = bool(getattr(current_user, "is_admin", False))
    try:
        access = rag_document_access(doc_id)
        if access is None or not can_read(
            access, user_id=current_user.id, is_admin=is_admin
        ):
            # 与"文档不存在"同一响应：越权者无法据此判断文档是否存在
            raise HTTPException(status_code=404, detail="文档不存在或已被删除")
        chunks = rag_get_document_chunks(doc_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"知识库暂不可用：{e}")

    if not chunks:
        raise HTTPException(status_code=404, detail="文档不存在或已被删除")

    return {"doc_id": doc_id, "chunks": chunks}


@app.delete("/api/knowledge/{doc_id}")
def delete_knowledge(
    doc_id: str,
    current_user: User = Depends(get_current_user),
):
    """删除一个文档的全部 chunk。

    删除规则与读取共用 ``agent.rag.visibility``：
    - public：按现有业务约定仅管理员可删；
    - private：仅 owner 本人可删（管理员删他人的同样拒绝）；
    - unknown / 元数据非法或矛盾：任何人（含管理员）都不能删。

    无权删除时返回 404，与"文档不存在"同响应——不泄露文档是否存在，
    也不让人靠状态码探测他人文档。

    结果语义（阶段 8B 起）：
    - 正常删除 → 200；
    - 检查时可见、执行时已被并发删除 → 200 + ``already_deleted: true``（幂等）；
    - 向量库已删但元数据写入失败 → **500**，明确说明未完成并给出原因，
      绝不谎报成功；
    - 源文件清理失败 → 200，但在 ``warnings`` 里如实列出，不隐瞒残留。
    """
    if not DOC_ID_PATTERN.match(doc_id):
        raise HTTPException(status_code=400, detail="doc_id 格式非法")

    try:
        documents = rag_list_documents()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"知识库暂不可用：{e}")

    target = next((d for d in documents if d["doc_id"] == doc_id), None)
    if target is None:
        raise HTTPException(status_code=404, detail="文档不存在")

    is_admin = bool(getattr(current_user, "is_admin", False))
    access = rag_document_access(doc_id)
    if access is None or not can_delete(
        access, user_id=current_user.id, is_admin=is_admin
    ):
        raise HTTPException(status_code=404, detail="文档不存在")

    if access.is_public:
        logger.info(
            "管理员 %s 删除公共文档 %s（%s）",
            current_user.username, doc_id, target.get("source"),
        )

    try:
        result = rag_delete_document(doc_id)
    except Exception as e:
        # 完整异常只进服务端日志：HTTP 响应不回显底层细节（路径、临时文件名、PID…）
        logger.warning("删除文档失败：%s", e)
        raise HTTPException(
            status_code=500,
            detail=(
                "删除失败：知识库存储暂时不可用，请稍后重试；"
                "若持续失败，请联系管理员并提供请求编号："
                f"{current_request_id()}"
            ),
        )

    if result.partial:
        # 向量库已删、元数据未更新：必须显式失败，并提供运维可识别的线索
        logger.error("删除未完成（元数据保存失败）：doc_id=%s 原因=%s", doc_id, result.error)
        audit.record(
            audit.DELETE_DOCUMENT, actor=current_user.username,
            target=target.get("source") or doc_id,
            outcome="failure",
            detail=f"部分失败：{result.error or '原因未知'}"[:200],
            request_id=current_request_id(),
        )
        raise HTTPException(
            status_code=500,
            detail=(
                "删除未完成：向量库已移除该文档，但元数据未保存成功。"
                "请先联系管理员核对，不要重复上传同一文件；请求编号："
                f"{current_request_id()}"
            ),
        )

    # 资料变了，缓存里基于旧资料生成的答案不再可信
    if semantic_cache is not None:
        semantic_cache.clear()

    # 顺带清理上传的源文件，避免磁盘只增不减
    warnings: list[str] = []
    file_id = target.get("file_id")
    if file_id:
        try:
            uploads.resolve_user_file(current_user.id, file_id).unlink(missing_ok=True)
        except Exception as e:
            logger.warning("删除源文件失败：%s", e)
            warnings.append(f"源文件未删除：{type(e).__name__}")

    audit.record(
        audit.DELETE_DOCUMENT, actor=current_user.username,
        target=f"{target.get('source') or doc_id}（{result.chunks_removed} 段）",
        detail="公共文档" if access.is_public else "本人文档",
        request_id=current_request_id(),
    )
    return {
        "status": "deleted",
        "doc_id": doc_id,
        "chunks": result.chunks_removed,
        "already_deleted": result.status == DELETE_STATUS_ALREADY_ABSENT,
        "warnings": warnings,
    }


@app.get("/api/reports")
def reports(current_user: User = Depends(require_admin)):
    """评测看板数据：趋势、最新指标、检索指标与缓存标定，一次取全。

    报告目录里混进损坏文件只跳过该文件，不影响整个看板——
    展示层不应该因为一个写了一半的 JSON 而 500。
    """
    try:
        return reports_mod.collect_reports()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


def enforce_upload_rate_limit(current_user: User = Depends(get_current_user)) -> User:
    """上传按用户限流：上传会触发解析与 Embedding 计费，需要单独设额度。"""
    _enforce_quota(upload_limiter, f"upload:{current_user.id}", "上传过于频繁")
    return current_user


@app.post("/api/upload")
async def upload(
    file: UploadFile = File(...),
    current_user: User = Depends(enforce_upload_rate_limit),
):
    """接收一个上传文件。

    - 图片：留存下来，供本次对话的 `analyze_image` 工具调用视觉模型。
    - 文档：直接增量写入知识库，之后由 Agent 用既有的检索工具带引用作答。

    两处刻意的实现细节：

    1. **先定上限、按上限读取。** `file.read(size)` 只读到指定字节数，
       不在读到内存之后才发现超限——SSE 服务对内存尖峰尤其敏感。
    2. **入库失败不影响上传。** 文件已经安全落盘，入库失败只是
       "暂时检索不到"，把原因告诉用户即可，不必让整个请求失败。
    """
    try:
        kind = uploads.classify_file(file.filename or "")
        # 多读 1 个字节用于发现超限：正好等于上限时仍然接受
        limit = uploads.max_bytes_for(kind) + 1
        data = await file.read(limit)
        uploads.check_size(kind, len(data))
        info = uploads.save_upload(current_user.id, file.filename or "", data)
    except uploads.UploadRejected as e:
        raise HTTPException(status_code=400, detail=e.reason)

    result = {
        "file_id": info["file_id"],
        "filename": info["filename"],
        "kind": info["kind"],
        "size": info["size"],
    }

    if info["kind"] == uploads.KIND_DOCUMENT:
        try:
            # 用原始文件名做引用来源：磁盘名是随机 UUID，展示出来无法核验
            # 解析 + Embedding + 向量库写入全是同步阻塞操作（含网络调用）。
            # 直接放在 async 端点里会卡住事件循环，其它用户的 SSE 也会停摆。
            result["knowledge"] = await asyncio.to_thread(
                uploads.ingest_document_into_kb,
                Path(info["path"]),
                display_name=info["filename"],
                owner=current_user.id,      # 用于后续的删除鉴权
                file_id=info["file_id"],    # 删除文档时一并清理源文件
            )
            # 知识库内容变了，此前基于旧资料生成的答案就不再可信，
            # 必须让缓存失效——否则缓存会把"过时但很像"的答案继续给用户。
            if semantic_cache is not None:
                semantic_cache.clear()
                logger.info("知识库已更新，语义缓存已清空")
        except Exception as e:
            logger.warning("文档入库失败：%s", e)
            result["knowledge"] = {"ingested": False, "error": str(e)}
            audit.record(
                audit.UPLOAD, actor=current_user.username,
                target=info["filename"], outcome="failure", detail=str(e),
                request_id=current_request_id(),
            )

    if info["kind"] == uploads.KIND_DOCUMENT:
        audit.record(
            audit.UPLOAD, actor=current_user.username, target=info["filename"],
            detail=f"{result['knowledge'].get('chunks', 0)} 段入库"
            if result.get("knowledge", {}).get("ingested") else "入库失败",
            request_id=current_request_id(),
        )
    return result


# ================== Agent 接口（需登录） ==================

def enforce_rate_limit(current_user: User = Depends(get_current_user)) -> User:
    """按用户限流，超额返回 429 并带上 Retry-After。

    做成依赖而不是写在路由体里，将来其它重接口要挂限流时不必复制一番。
    计数粒度是 user_id：单个用户刷接口不会连累其他人。
    """
    result = limiter.check(str(current_user.id))
    if not result.allowed:
        retry = max(1, int(result.retry_after))
        prom.record_rate_limited()
        logger.warning(
            "触发限流（user=%s，%ss 后可重试）", current_user.username, retry,
            extra={"event": "rate_limited", "retry_after": retry},
        )
        raise HTTPException(
            status_code=429,
            detail=f"请求过于频繁，请 {retry} 秒后重试",
            headers={"Retry-After": str(retry)},
        )
    return current_user


@app.get("/")
def root():
    try:
        r.ping()
        redis_ok = True
    except redis.RedisError:
        redis_ok = False
    return {"status": "ok", "redis": redis_ok}


def metrics_guard(current_user: User = Depends(get_current_user)) -> User:
    """指标端点守卫：默认只允许管理员。

    指标里含请求量、成本、错误率等运营数据，
    公网裸奔等于把内部数据公开；内网可信环境可设 METRICS_PUBLIC=1 关闭。
    """
    if not bool(getattr(current_user, "is_admin", False)):
        raise HTTPException(status_code=403, detail="仅管理员可查看指标")
    return current_user


def _no_auth() -> None:
    return None


# 依赖在导入期确定：公开模式直接换成一个空依赖，
# 避免在运行期反复判断环境变量
_metrics_dep = (
    _no_auth
    if os.getenv("METRICS_PUBLIC", "0").lower() in ("1", "true", "yes")
    else metrics_guard
)


@app.get("/api/audit")
def read_audit(
    limit: int = 50,
    all_users: bool = False,
    current_user: User = Depends(get_current_user),
):
    """审计记录查询。

    - 默认只返回**自己的**操作记录
    - `all_users=true` 需要管理员：用于排查"这个文档是谁删的"
    """
    if all_users and not bool(getattr(current_user, "is_admin", False)):
        raise HTTPException(status_code=403, detail="仅管理员可查看全部审计记录")

    actor = None if all_users else current_user.username
    return {"records": audit.recent(actor=actor, limit=limit)}


@app.get("/metrics")
def metrics_endpoint(_guard=Depends(_metrics_dep)):
    """Prometheus 抓取端点。"""
    body, content_type = prom.render()
    return Response(content=body, media_type=content_type)


@app.get("/api/usage")
def usage(current_user: User = Depends(get_current_user)):
    """当前用户的用量与成本（按天聚合，默认近 7 天）。

    有了它，"加了缓存省了多少""某个用户刷了多少"才是有据可查的数字，
    而不是靠账单倒推。
    """
    summary = metrics.read_usage(r, current_user.id, days=7)
    summary["cache"] = semantic_cache.stats() if semantic_cache is not None else None
    return summary


@app.post("/api/agent/stream")
async def agent_stream(
    req: ChatRequest,
    current_user: User = Depends(enforce_rate_limit)
):
    user_id = current_user.id
    is_admin = bool(getattr(current_user, "is_admin", False))

    history = get_history(user_id, req.session_id)
    # 历史指纹要在追加本轮消息**之前**算：它代表"发这句之前聊过什么"
    context_id = _context_fingerprint(history)
    has_image = bool(
        req.file_id and uploads.suffix_of(req.file_id) in uploads.IMAGE_SUFFIXES
    )

    # 图片要把 file_id 标注进消息，模型才知道"有图可看"以及该用哪个 file_id；
    # 文档在上传时已经直接入库，不必再占用上下文。
    if has_image:
        history.append(HumanMessage(
            content=f"{req.message}\n[用户附带了一张图片，file_id={req.file_id}]"
        ))
    else:
        history.append(HumanMessage(content=req.message))

    async def event_generator():
        # 让 Agent 的工具能确认"当前请求属于谁"。
        # ContextVar 按 task 隔离，并发请求之间不会互相串扰。
        current_user_id.set(user_id)

        # 所有清理都放在下面的 finally 里：正常结束、熔断早退、生成器被取消，
        # 并发名额都必须归还——漏一次就少一个名额，最终所有请求都会 503。
        prom.inc_inflight()
        timer = prom.Timer()
        collected = []          # 本轮产生的全部消息，用于写回会话历史
        last_ai = None          # 最后一条 AI 消息，用于提取最终答案
        errored = False         # 本轮是否出错（决定要不要把熔断器恢复为闭合）
        provider_called = False # 只有真正请求过上游，"成功"才配记账
        signature = capability_signature()

        try:
            # ---- 熔断检查 ----
            # 上游持续故障时快速失败：否则每个请求都要硬等 90s 超时，
            # 并发名额被占满，本来的"部分失败"会变成"整体不可用"。
            if not provider_breaker.allow():
                prom.record_error("circuit_open")
                logger.error(
                    "上游熔断中，快速失败（session=%s）", req.session_id,
                    extra={"event": "circuit_open_reject"},
                )
                yield _sse({
                    "type": "error",
                    "content": "上游模型暂时不可用（已熔断），请稍后再试。",
                })
                yield "data: [DONE]\n\n"
                return  # 本次不会走到检索，也不设置 RAG 上下文

            # 本请求的 RAG 上下文：身份 + 引用容器。
            # rag_ctx 是本请求的局部对象——工具在工作线程里原地写它持有的 citations，
            # SSE 读同一个对象；不使用任何模块级共享状态，避免并发下串号。
            rag_ctx = RagRequestContext(
                user_id=user_id, is_admin=is_admin, citations=[],
                user_message=req.message,
            )
            set_rag_context(rag_ctx)

            logger.info(
                "开始处理对话（session=%s，历史 %d 条，带图=%s）",
                req.session_id, len(history), has_image,
                extra={"event": "agent_start", "session_id": req.session_id},
            )

            # ---- 第一站：语义缓存 ----
            # 命中就直接下发答案，模型、检索、工具这一整条链路都不再执行。
            # 带图片时不查缓存：同一个问题配不同的图，答案本就该不同。
            if semantic_cache is not None and cache_embed is not None and not has_image:
                vector = None
                try:
                    # 向量化是同步网络调用，丢到线程里，避免阻塞事件循环
                    vector = await asyncio.to_thread(cache_embed, req.message)
                except Exception as e:
                    logger.warning("缓存向量化失败，本次不走缓存：%s", e)

                if vector:
                    entry = semantic_cache.lookup(
                        user_id, vector, signature, context_id,
                        numbers=cache_mod.number_signature(req.message),
                    )
                    if entry is not None:
                        yield _sse({"type": "human", "content": req.message})
                        yield _sse({
                            "type": "ai", "content": entry.answer, "cached": True,
                        })
                        # 命中不消耗模型 token，但请求仍然要计入用量统计
                        empty_usage = {"input_tokens": 0, "output_tokens": 0}
                        yield _sse({
                            "type": "usage", "tokens": empty_usage,
                            "cost_cny": 0.0, "cached": True,
                        })
                        metrics.record_usage(
                            r, user_id, empty_usage, 0.0, cached=True
                        )
                        prom.record_cache_hit()
                        # 命中也要写回历史，否则下一轮的上下文会少一问一答
                        save_history(
                            user_id, req.session_id,
                            history + [AIMessage(content=entry.answer)],
                        )
                        sessions_mod.record_session(
                            r, user_id, req.session_id, req.message, SESSION_TTL
                        )
                        yield "data: [DONE]\n\n"
                        return

            # 动态超时：带图请求 = 主模型 + 视觉模型 + 主模型，三段串行，
            # 固定预算会在服务商延迟波动时误杀正常请求（实测波动 2s~22s+）。
            budget = AGENT_TIMEOUT
            if req.file_id:
                budget += 60  # 追加一个视觉调用的时间预算（与 VISION_TIMEOUT 同量级）
            async with asyncio.timeout(budget):
                provider_called = True   # 从这里开始才真正依赖上游
                async for event in agent.astream(
                    {"messages": history},
                    stream_mode="values"
                ):
                    messages = event.get("messages") or []
                    if not messages:
                        # 图架构下 classify 等中间节点不产出消息，跳过
                        continue
                    last_msg = messages[-1]

                    # values 模式会把同一条消息重复推送，按内容去重避免前端重复渲染
                    if collected:
                        prev = collected[-1]
                        if (prev.type == last_msg.type
                                and (prev.content or "") == (last_msg.content or "")
                                and getattr(prev, "name", "") == getattr(last_msg, "name", "")):
                            continue

                    collected.append(last_msg)
                    if last_msg.type == "ai":
                        last_ai = last_msg

                    tool_name = ""
                    if last_msg.type == "tool":
                        tool_name = getattr(last_msg, "name", "") or ""
                    data = {
                        "type": last_msg.type,
                        "content": last_msg.content or "",
                        "name": tool_name,
                        "ts": _now_iso(),
                    }
                    # 知识库引用随事件下发，前端据此展示"答案出自哪里"。
                    # react 架构读本请求的上下文（工具已在工作线程里原地写入），
                    # graph 架构直接从 state 取。
                    if tool_name == "search_knowledge_base":
                        data["citations"] = list(rag_ctx.citations)
                    if not data.get("citations") and event.get("citations"):
                        data["citations"] = event["citations"]
                    yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

            # 只有不带 tool_calls 的 AI 消息才是最终答案（带 tool_calls 说明还要继续调用工具）
            final_answer = ""
            if last_ai is not None and not getattr(last_ai, "tool_calls", None):
                final_answer = last_ai.content or ""

            if final_answer:
                # ---- 用量与成本埋点 ----
                # usage_metadata 挂在 AI 消息上，是唯一可信的实际用量来源；
                # 自己按字符估 token 会估偏，账单对不上就失去了降本的意义。
                usage = sum_usage(collected)
                cost = estimate_cost(usage, MODEL_NAME)
                yield _sse({
                    "type": "usage", "tokens": usage,
                    "cost_cny": round(cost, 6), "cached": False,
                })
                metrics.record_usage(r, user_id, usage, cost, cached=False)
                # Prometheus 侧同步记一份：用户维度看 /api/usage，
                # 全局趋势与告警看 /metrics
                prom.record_tokens(
                    usage.get("input_tokens", 0), usage.get("output_tokens", 0)
                )
                prom.record_cost(cost)

                # ---- 写入语义缓存 ----
                # 只缓存"不会过期"的答案：用过实时工具或带图片的一律不缓存，
                # 否则就是把过期答案包装成缓存收益。
                if (
                    semantic_cache is not None and cache_embed is not None
                    and not has_image and not _used_volatile_tool(collected)
                ):
                    try:
                        vector = await asyncio.to_thread(cache_embed, req.message)
                        semantic_cache.store(
                            user_id, req.message, final_answer,
                            vector, signature, context_id,
                            numbers=cache_mod.number_signature(req.message),
                        )
                    except Exception as e:
                        logger.warning("写入语义缓存失败：%s", e)

                # 保存完整序列（含工具调用），多轮对话才能延续上下文
                old = get_history(user_id, req.session_id)
                # 历史过长时先压缩成摘要。这一步发生在流式内容已经发完之后，
                # 只影响 [DONE] 的到达时间，用户侧此刻已经看到完整回答。
                messages, _compressed = await memory.maybe_compress(old + collected)
                save_history(user_id, req.session_id, messages)
                # 登记进会话索引：新会话在这里诞生，老会话在这里续命
                sessions_mod.record_session(
                    r, user_id, req.session_id, req.message, SESSION_TTL
                )

            yield "data: [DONE]\n\n"

        except asyncio.TimeoutError:
            errored = True
            prom.record_error("timeout")
            provider_breaker.record_failure()
            logger.warning(
                "Agent 请求超时（>%ss，user=%s，session=%s）",
                budget, user_id, req.session_id,
                extra={"event": "agent_timeout", "budget": budget},
            )
            err = {"type": "error", "content": f"请求超时（>{budget}秒），请重试"}
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
        except Exception as e:
            errored = True
            prom.record_error("exception")
            # 只有"上游类"故障才计入熔断：
            # 用户参数错误、内容被拒这类确定性失败不该让整条链路熔断
            if resilience.is_transient(e):
                provider_breaker.record_failure()
            logger.exception(
                "Agent 请求异常（user=%s，session=%s）：%s",
                user_id, req.session_id, e,
                extra={"event": "agent_error"},
            )
            # 完整异常只进服务端日志：把上游报错原文回给客户端会泄露
            # 服务地址、内部标识等细节，用户只需要一个可追溯的请求编号。
            err = {"type": "error", "content": (
                "服务异常，请稍后重试；若持续失败，请提供请求编号："
                f"{current_request_id()}"
            )}
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
        finally:
            # 上下文清理：包一层 try 是为了万无一失——它绝不能影响下面的
            # 并发名额归还、inflight 计数与熔断状态维护
            try:
                reset_rag_context()
            except Exception as e:      # pragma: no cover - 防御性分支
                logger.warning("清理 RAG 请求上下文失败：%s", e)
            # 必须释放：生成器被取消（客户端断开）时也要归还名额，
            # 否则并发名额会被慢慢泄漏，服务逐渐拒掉所有请求。
            guard.release()
            prom.dec_inflight()
            # 只有本轮真的调用过上游并且没出错，才能记一次成功。
            # 缓存命中与熔断早退都没碰上游：半开状态下若被它们误判为恢复，
            # 熔断器会立刻闭合，真实故障马上再次打满。
            if not errored and provider_called:
                provider_breaker.record_success()
            # 流式接口的总耗时（含模型与工具时间），是延迟告警的主指标
            elapsed = time.perf_counter() - timer.start
            prom.observe_request("/api/agent/stream", 200, elapsed)
            logger.info(
                "对话处理结束（耗时 %.2fs，session=%s）", elapsed, req.session_id,
                extra={
                    "event": "agent_finish",
                    "duration_s": round(elapsed, 2),
                    "session_id": req.session_id,
                },
            )

    # 并发闸门：拿不到名额立刻失败，不让请求排队堆积（排队比拒绝更伤）
    if not await guard.acquire_nowait():
        raise HTTPException(status_code=503, detail="服务繁忙，请稍后重试")

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/sessions")
def list_user_sessions(current_user: User = Depends(get_current_user)):
    """当前用户的会话列表（侧边栏数据源），按最近活跃倒序。"""
    return {"sessions": sessions_mod.list_sessions(r, current_user.id)}


@app.get("/api/session/{session_id}")
def read_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
):
    """读取一个会话的历史，用于切换会话时回放。

    只返回人机对话内容；工具过程与压缩摘要属于内部实现，
    回放时不需要，也不该让用户看到自己没发过的"思考过程"。
    """
    if not SESSION_ID_PATTERN.match(session_id):
        raise HTTPException(status_code=400, detail="session_id 格式非法")

    messages = get_history(current_user.id, session_id)
    return {
        "session_id": session_id,
        "messages": sessions_mod.readable_messages(messages),
    }


@app.put("/api/session/{session_id}")
def rename_session(
    session_id: str,
    req: RenameRequest,
    current_user: User = Depends(get_current_user),
):
    """重命名会话（侧边栏）。只改索引里的展示名，不动消息数据。"""
    if not SESSION_ID_PATTERN.match(session_id):
        raise HTTPException(status_code=400, detail="session_id 格式非法")

    ok = sessions_mod.rename_session(r, current_user.id, session_id, req.title)
    if not ok:
        # 会话从未聊过或已被 TTL 回收，重命名无从谈起
        raise HTTPException(status_code=404, detail="会话不存在或已过期")
    return {"status": "renamed", "session_id": session_id, "title": req.title}


@app.delete("/api/session/{session_id}")
def clear_session(
    session_id: str,
    current_user: User = Depends(get_current_user)
):
    if not SESSION_ID_PATTERN.match(session_id):
        raise HTTPException(status_code=400, detail="session_id 格式非法")
    try:
        # 会话本体与索引条目要一起删，否则侧边栏会出现"幽灵会话"
        pipe = r.pipeline()
        pipe.delete(session_key(current_user.id, session_id))
        pipe.hdel(sessions_mod.index_key(current_user.id), session_id)
        pipe.execute()
    except redis.RedisError as e:
        logger.warning("Redis 不可用，会话未清除: %s", e)
        raise HTTPException(status_code=503, detail="会话服务暂不可用")
    audit.record(
        audit.CLEAR_SESSION, actor=current_user.username, target=session_id,
        request_id=current_request_id(),
    )
    return {"status": "cleared", "session_id": session_id}
