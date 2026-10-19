"""SSE 并发闸门与熔断交互的回归测试。

覆盖两个曾经的真实缺陷：

1. 熔断早退发生在 ``try/finally`` 之外，``guard.release()`` 永远不执行，
   并发名额只减不增——上游恢复后服务也无法自愈；
2. 缓存命中与熔断早退都被记为"上游成功"，半开状态下可能误闭合熔断器。

另外验证上传文档的入库（解析 + Embedding + 写向量库）跑在工作线程，
不阻塞其它请求的 SSE。
"""

import asyncio
import atexit
import os
import shutil
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

_TMP_DIR = Path(tempfile.mkdtemp(prefix="stream-guard-test-"))


def _cleanup_tmp() -> None:
    shutil.rmtree(_TMP_DIR, ignore_errors=True)


atexit.register(_cleanup_tmp)

# 与其它 API 测试一致：假配置，避免读真实 .env
os.environ["RAG_ENABLED"] = "off"
os.environ["VISION_ENABLED"] = "off"
os.environ["JWT_SECRET"] = "test-secret-not-real"
os.environ["LLM_API_KEY"] = "test-key-not-real"
os.environ["DATABASE_URL"] = f"sqlite:///{(_TMP_DIR / 'test.db').as_posix()}"
os.environ["REDIS_URL"] = "redis://127.0.0.1:1/0"

from fastapi import HTTPException  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402

import server.main as main_module  # noqa: E402
from server.ratelimit import ConcurrencyGuard, SlidingWindowLimiter  # noqa: E402

USER = SimpleNamespace(id=6, username="alice", is_admin=False)


class FakeBreaker:
    """记录成功/失败记账次数，用来断言"该记的记、不该记的不记"。"""

    def __init__(self, allow: bool = True):
        self._allow = allow
        self.allow_calls = 0
        self.success_calls = 0
        self.failure_calls = 0

    def allow(self) -> bool:
        self.allow_calls += 1
        return self._allow

    def record_success(self) -> None:
        self.success_calls += 1

    def record_failure(self) -> None:
        self.failure_calls += 1


class FakeCache:
    def lookup(self, *args, **kwargs):
        return SimpleNamespace(answer="缓存答案")

    def store(self, *args, **kwargs):
        raise AssertionError("命中后不应再写缓存")

    def clear(self):
        pass

    def stats(self):
        return {}


class FakeAgent:
    """吐一条最终 AI 消息，不调用任何工具。"""

    def astream(self, payload, stream_mode="values"):
        async def generator():
            yield {"messages": [AIMessage(content="正常回答")]}

        return generator()


class FakeUpload:
    def __init__(self, filename: str, data: bytes):
        self.filename = filename
        self._data = data
        self._pos = 0

    async def read(self, size: int = -1) -> bytes:
        chunk = self._data[self._pos:self._pos + size]
        self._pos += len(chunk)
        return chunk


def run_stream(req=None):
    """直接驱动端点并抽干 SSE 生成器（依赖由调用方 monkeypatch 替换）。"""
    req = req or main_module.ChatRequest(message="你好", session_id="s1")

    async def scenario():
        response = await main_module.agent_stream(req, USER)
        return [chunk async for chunk in response.body_iterator]

    return asyncio.run(scenario())


@pytest.fixture
def isolated_stream(monkeypatch):
    monkeypatch.setattr(main_module, "get_history", lambda *a, **k: [])
    monkeypatch.setattr(main_module, "save_history", lambda *a, **k: None)
    monkeypatch.setattr(main_module, "semantic_cache", None)
    monkeypatch.setattr(main_module, "cache_embed", None)
    return monkeypatch


def test_circuit_open_returns_concurrency_slot(isolated_stream):
    guard = ConcurrencyGuard(limit=1)
    breaker = FakeBreaker(allow=False)
    isolated_stream.setattr(main_module, "guard", guard)
    isolated_stream.setattr(main_module, "provider_breaker", breaker)

    chunks = run_stream()

    assert any("熔断" in chunk for chunk in chunks)
    # 关键断言：名额已归还（修复前这里会是 1，且下一个请求永远 503）
    assert guard.active == 0
    assert asyncio.run(guard.acquire_nowait()) is True
    # 熔断早退没有访问上游，不能记成一次成功
    assert breaker.success_calls == 0


def test_cache_hit_does_not_mark_provider_healthy(isolated_stream):
    guard = ConcurrencyGuard(limit=1)
    breaker = FakeBreaker(allow=True)
    isolated_stream.setattr(main_module, "guard", guard)
    isolated_stream.setattr(main_module, "provider_breaker", breaker)
    isolated_stream.setattr(main_module, "semantic_cache", FakeCache())
    isolated_stream.setattr(main_module, "cache_embed", lambda text: [1.0, 0.0])

    chunks = run_stream()

    assert any("缓存答案" in chunk for chunk in chunks)
    assert guard.active == 0
    assert breaker.success_calls == 0


def test_successful_run_marks_provider_healthy(isolated_stream):
    guard = ConcurrencyGuard(limit=1)
    breaker = FakeBreaker(allow=True)
    isolated_stream.setattr(main_module, "guard", guard)
    isolated_stream.setattr(main_module, "provider_breaker", breaker)
    isolated_stream.setattr(main_module, "agent", FakeAgent())

    chunks = run_stream()

    assert any("正常回答" in chunk for chunk in chunks)
    assert guard.active == 0
    assert breaker.success_calls == 1


def test_upload_ingest_runs_off_event_loop(isolated_stream):
    seen = {}

    def fake_ingest(*args, **kwargs):
        seen["thread"] = threading.get_ident()
        return {"ingested": True, "chunks": 1, "source": "notes.md"}

    isolated_stream.setattr(
        main_module.uploads, "save_upload",
        lambda uid, name, data: {
            "file_id": f"u{uid}/abc.md",
            "filename": name,
            "kind": "document",
            "size": len(data),
            "path": str(_TMP_DIR / "abc.md"),
        },
    )
    isolated_stream.setattr(main_module.uploads, "ingest_document_into_kb", fake_ingest)
    isolated_stream.setattr(main_module.audit, "record", lambda *a, **k: None)

    async def scenario():
        loop_thread = threading.get_ident()
        result = await main_module.upload(FakeUpload("notes.md", b"# hello"), USER)
        return loop_thread, result

    loop_thread, result = asyncio.run(scenario())

    assert result["knowledge"]["ingested"] is True
    # 入库在工作线程执行：事件循环线程不会被 Embedding/解析卡住
    assert seen["thread"] != loop_thread


def test_login_rate_limited_by_username(monkeypatch):
    """同一用户名的连续登录尝试超过额度后返回 429；大小写不构成新桶。"""
    limiter = SlidingWindowLimiter(None, limit=2, window=60, clock=lambda: 1000.0)
    monkeypatch.setattr(main_module, "auth_limiter", limiter)
    monkeypatch.setattr(main_module.audit, "record", lambda *a, **k: None)

    class FakeDB:
        def query(self, *args, **kwargs):
            class Query:
                def filter(self, *args, **kwargs):
                    return self

                def first(self):
                    return None

            return Query()

    db = FakeDB()
    for _ in range(2):
        with pytest.raises(HTTPException) as excinfo:
            main_module.login(SimpleNamespace(username="alice", password="wrong"), db)
        assert excinfo.value.status_code == 401

    # 换大小写必须命中同一个桶，否则限流可以被轻易绕过
    with pytest.raises(HTTPException) as excinfo:
        main_module.login(SimpleNamespace(username="ALICE", password="wrong"), db)
    assert excinfo.value.status_code == 429
    assert excinfo.value.headers["Retry-After"] == "60"


def test_upload_rate_limited_per_user(monkeypatch):
    """上传额度按用户隔离：一个用户超限不影响其他用户。"""
    limiter = SlidingWindowLimiter(None, limit=1, window=60, clock=lambda: 1000.0)
    monkeypatch.setattr(main_module, "upload_limiter", limiter)

    assert main_module.enforce_upload_rate_limit(USER) is USER

    with pytest.raises(HTTPException) as excinfo:
        main_module.enforce_upload_rate_limit(USER)
    assert excinfo.value.status_code == 429

    other = SimpleNamespace(id=7, username="bob", is_admin=False)
    assert main_module.enforce_upload_rate_limit(other) is other
