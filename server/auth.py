import os
from datetime import datetime, timedelta, timezone
import bcrypt
from jose import jwt, JWTError
from fastapi import Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from server.database import get_db, User

SECRET_KEY = os.getenv("JWT_SECRET")
# 绝不使用可预测的兜底密钥：缺配置时直接拒绝启动，避免 token 被伪造
if not SECRET_KEY or SECRET_KEY == "change-me-in-production":
    raise RuntimeError(
        "未配置 JWT_SECRET。请执行 "
        '`python -c "import secrets; print(secrets.token_urlsafe(32))"` '
        "生成后写入 .env"
    )

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7   # 7 天

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/login")


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def create_access_token(user_id: int, password_version: int = 0) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    # pv = 密码版本号：改密后旧 token 的 pv 与库中不一致，会被判失效
    payload = {"sub": str(user_id), "exp": expire, "pv": int(password_version or 0)}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def _load_user(db: Session, user_id: int) -> User | None:
    """从库里取用户。独立成同步函数，供异步依赖丢进线程池执行。

    get_current_user 必须是 async（ContextVar 要在请求任务里写入），
    但同步 ORM 查询不能直接在里面做——那会把事件循环卡住。
    """
    return db.query(User).filter(User.id == user_id).first()


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """注意：这里刻意声明为 async。

    同步依赖会被 FastAPI 放进线程池执行，在那里修改 ContextVar
    不会传回外层协程——访问日志就永远看不到 user 字段。
    异步依赖在同一个 task 的上下文中执行，日志上下文才能真正穿透。
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="登录已失效，请重新登录",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = int(payload.get("sub"))
    except (JWTError, TypeError, ValueError):
        raise credentials_exception

    # 查询丢进线程池：async 依赖里直接做同步 DB I/O 会阻塞整个事件循环
    user = await run_in_threadpool(_load_user, db, user_id)
    if not user:
        raise credentials_exception

    # 改过密码的账号，改密前签发的 token 一律作废
    if int(payload.get("pv", 0)) != int(getattr(user, "password_version", 0) or 0):
        raise credentials_exception

    # 写入日志上下文：之后的日志能定位到具体账号（排查"某个用户反馈慢"时很关键）
    try:
        from server.logging_setup import request_user_var

        request_user_var.set(user.username)
    except Exception:  # pragma: no cover
        pass
    return user