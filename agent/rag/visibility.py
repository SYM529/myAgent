"""知识库文档的可见性与授权判定（纯函数，无 IO）。

为什么单独成模块
----------------
列表、检索、预览、删除是四条独立代码路径。规则各写一遍，迟早出现
"列表收紧了、检索忘了收"的偏差——私有文档会顺着检索漏出去。
所以判定只在这里定义一次，其余模块调用同一套函数。

三种状态
--------
- public：**必须**明确 ``visibility="public"`` 且没有 owner，对应 CLI 入库的团队资料。
- private：有明确的 owner。历史文档只写了 owner、没有 visibility 的，同样按 private。
- unknown：字段缺失、值非法、或彼此矛盾。一律隔离：普通 API 完全不可见、
  不可读取、不可删除，管理员也一样。

判定取向是"读不出来就不放行"：任何解析失败都落到 unknown，绝不回退成 public。
默认共享是这类系统里最难收回的错误——数据已经流出去，改代码也追不回来。

边界约定
--------
- ``owner`` 只认"正整数或数字字符串"，与上传入库写入的 ``str(user_id)`` 一致；
  int 与带空白的字符串会被归一化成同一个规范形式。
- ``visibility`` 容忍大小写与首尾空白；空串按"没写"处理（等价于历史的无标记文档）。
- 写了但无法解析的值（如 ``owner="abc"``、``visibility="shared"``）会让整篇文档隔离，
  而不是被当作没写——被写坏的数据不配得到信任。
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

VISIBILITY_PUBLIC = "public"
VISIBILITY_PRIVATE = "private"
VISIBILITY_UNKNOWN = "unknown"

# 单个 chunk 的证据类型（内部使用，不对外暴露）：
#   public / private：明确的归属声明
#   _SILENT：这两个字段一个都没写（历史数据的常态）
#   _INVALID：写了但读不出来或值非法——整篇文档都要被隔离
_SILENT = "silent"
_INVALID = "invalid"

# 哨兵：字段取值本身抛异常，与"字段为 None"必须区分开
_UNREADABLE = object()


@dataclass(frozen=True)
class DocAccess:
    """一篇文档的判定结果。

    只有 private 才带 owner；public 与 unknown 的 owner 恒为 None。
    对外的判定函数只接受本类型，避免调用方随手传个 dict 造成规则分叉。
    """

    visibility: str
    owner: str | None = None

    @property
    def is_public(self) -> bool:
        return self.visibility == VISIBILITY_PUBLIC

    @property
    def is_private(self) -> bool:
        return self.visibility == VISIBILITY_PRIVATE

    @property
    def is_unknown(self) -> bool:
        return self.visibility == VISIBILITY_UNKNOWN


def _unknown() -> DocAccess:
    return DocAccess(VISIBILITY_UNKNOWN, None)


def _public() -> DocAccess:
    return DocAccess(VISIBILITY_PUBLIC, None)


def _private(owner: str) -> DocAccess:
    return DocAccess(VISIBILITY_PRIVATE, owner)


def _read(mapping: Mapping, key: str) -> Any:
    """取字段值；取值过程本身抛异常时返回哨兵，交由调用方按非法处理。"""
    try:
        return mapping.get(key)
    except Exception as e:  # 防御性：metadata 可能来自 JSON、第三方对象
        logger.warning("读取 metadata 字段 %s 失败：%s", key, e)
        return _UNREADABLE


def _has_value(raw: Any) -> bool:
    """判断字段是否"写了东西"：None 与空白字符串视为没写。"""
    if raw is None:
        return False
    if isinstance(raw, str):
        return bool(raw.strip())
    return True


def _normalize_owner(raw: Any) -> str | None:
    """把 owner 归一化成非空数字字符串；无法解析时返回 None。

    None 的含义是"拿不到可信归属"，调用方必须按 unknown 处理——
    绝不能把它当成"公共文档"。
    """
    if isinstance(raw, bool):        # bool 是 int 的子类，必须先挡掉
        return None
    if isinstance(raw, int):
        return str(raw) if raw > 0 else None
    if isinstance(raw, str):
        text = raw.strip()
        # isdigit() 会放行全角数字，int() 同样能解析，二者一致即可
        if text.isdigit() and int(text) > 0:
            return str(int(text))    # 归一化："007" 与 7 视为同一个人
        return None
    return None


def _normalize_visibility(raw: Any) -> str | None:
    """归一化 visibility：缺失/空白返回 None，非法值返回 _INVALID。"""
    if raw is None:
        return None
    if not isinstance(raw, str):
        return _INVALID
    text = raw.strip().lower()
    if not text:
        return None
    if text in (VISIBILITY_PUBLIC, VISIBILITY_PRIVATE):
        return text
    return _INVALID


def _evidence(meta: Any) -> tuple[str, str | None]:
    """单个 chunk 的证据：返回 (类型, owner)，类型见模块顶部说明。"""
    if not isinstance(meta, Mapping):
        return (_INVALID, None)

    raw_visibility = _read(meta, "visibility")
    raw_owner = _read(meta, "owner")
    if raw_visibility is _UNREADABLE or raw_owner is _UNREADABLE:
        return (_INVALID, None)

    owner = _normalize_owner(raw_owner)
    if _has_value(raw_owner) and owner is None:
        # 写了 owner 却解析不出合法用户 id：宁可隔离，也不猜归属
        return (_INVALID, None)

    visibility = _normalize_visibility(raw_visibility)
    if visibility is _INVALID:
        return (_INVALID, None)

    if visibility == VISIBILITY_PUBLIC:
        # 公共文档不允许带 owner：两者同时出现说明数据被写坏或改过
        return (_INVALID, None) if owner is not None else (VISIBILITY_PUBLIC, None)

    if visibility == VISIBILITY_PRIVATE:
        return (VISIBILITY_PRIVATE, owner) if owner else (_INVALID, None)

    # visibility 缺失：只有 owner 的属于历史 Web 上传，按 private 处理
    if owner:
        return (VISIBILITY_PRIVATE, owner)
    return (_SILENT, None)


def classify_metadata(meta: Any) -> DocAccess:
    """判定单个 chunk 的 metadata。

    只用于单块场景（例如测试与排查）；文档级判定请用 ``classify_document``——
    归属可能写在文档里的任意一个 chunk 上。
    """
    kind, owner = _evidence(meta)
    if kind == VISIBILITY_PRIVATE and owner:
        return _private(owner)
    if kind == VISIBILITY_PUBLIC:
        return _public()
    return _unknown()


def classify_document(chunk_metas: Iterable[Any]) -> DocAccess:
    """按文档聚合判定：把该文档所有 chunk 的 metadata 传进来。

    聚合规则与 ``agent.rag.summarize_documents`` 的"任一 chunk 带着归属就够"保持一致：
    只在第一个 chunk 上看 owner，会让"自己的文档被判成 unknown 而删不掉"。

    但聚合不等于放宽：
    - 任一 chunk 出现非法值 → 整篇 unknown；
    - 不同 chunk 给出互相矛盾的归属（两个 owner、或 public 与 private 并存）→ 整篇 unknown；
    - 没有任何 chunk 声明归属 → unknown（缺省不共享）。
    """
    kinds: set[str] = set()
    owners: set[str] = set()
    seen = False

    try:
        iterator = iter(chunk_metas)
    except TypeError:
        logger.warning("classify_document 收到不可迭代的入参，按 unknown 处理")
        return _unknown()

    try:
        for meta in iterator:
            seen = True
            kind, owner = _evidence(meta)
            if kind == _INVALID:
                return _unknown()
            kinds.add(kind)
            if kind == VISIBILITY_PRIVATE and owner:
                owners.add(owner)
    except Exception as e:  # 迭代过程出错也必须落到最保守的判定
        logger.warning("聚合文档归属时出错，按 unknown 处理：%s", e)
        return _unknown()

    if not seen:
        return _unknown()

    declared = kinds - {_SILENT}
    if not declared:
        return _unknown()                     # 全部 chunk 都没写归属
    if declared == {VISIBILITY_PUBLIC}:
        return _public() if not owners else _unknown()
    if declared == {VISIBILITY_PRIVATE}:
        if len(owners) != 1:
            return _unknown()                 # 没有 owner 或出现多个 owner
        return _private(next(iter(owners)))
    return _unknown()                         # public 与 private 混在一篇里


# ---------- 写路径要用的公开入口 ----------
#
# 判定规则只在本模块定义一次。写路径（store.add）与审计工具都通过这些入口调用，
# 避免出现"这个函数把空串当没写、那个函数当非法"这类互相矛盾的语义。


def _describe(value: Any, limit: int = 24) -> str:
    """把不可信的元数据值渲染成可安全放进错误信息的短文本。

    非字符串只打印类型名——元数据里可能是任意对象，不能把内容倒进日志或报错。
    """
    if value is None:
        return "None"
    if isinstance(value, str):
        text = " ".join(value.split())
        return repr(text[:limit]) if text else "''"
    return f"<{type(value).__name__}>"


def normalize_visibility(value: Any) -> str | None:
    """写路径的严格入口：只有明确的 public / private 才算声明。

    缺失、空白串、非字符串、无法识别的值一律返回 None——调用方对它们的态度相同：
    拒绝写入，而不是"当作没写"或"猜一个"。
    """
    normalized = _normalize_visibility(value)
    if normalized in (VISIBILITY_PUBLIC, VISIBILITY_PRIVATE):
        return normalized
    return None


def normalize_owner(value: Any) -> str | None:
    """写路径的 owner 归一化：正整数/数字字符串 → 规范形式，其余 → None。"""
    return _normalize_owner(value)


def owner_declared(value: Any) -> bool:
    """元数据里是否"声明了" owner。

    **空串与纯空白一律视为没声明**（与 ``_has_value`` 保持同一语义），
    其余非 None 值都算声明——包括写坏的值，因为"写了但读不出来"必须被拒绝，
    不能靠"当作没写"混过去。
    """
    return _has_value(value)


def inspect_document(doc_id: str, chunk_metas: Iterable[Any]) -> tuple[DocAccess, list[str]]:
    """逐 chunk 检查一篇文档的元数据是否完整一致，返回 (判定结果, 问题清单)。

    与 ``classify_document`` 的区别：
    - ``classify_document`` 回答"整篇能不能分类"（聚合时允许部分 chunk 沉默）；
    - ``inspect_document`` 回答"每个 chunk 是否都完整一致"——写路径必须用后者，
      否则"一个 chunk 声明、其余沉默"会带着不完整的元数据落库。

    问题清单为空时判定结果才可信；非空时调用方必须拒绝写入。
    本函数只报告问题，不抛异常，便于批量聚合后一次性报错。
    """
    issues: list[str] = []
    metas: list[Any] = []
    visibilities: set[str] = set()
    owners: set[str] = set()

    for index, meta in enumerate(chunk_metas):
        metas.append(meta)
        if not isinstance(meta, Mapping):
            issues.append(f"{doc_id} 第 {index} 块 metadata 不是映射：{type(meta).__name__}")
            continue

        raw_visibility = _read(meta, "visibility")
        raw_owner = _read(meta, "owner")
        if raw_visibility is _UNREADABLE or raw_owner is _UNREADABLE:
            issues.append(f"{doc_id} 第 {index} 块元数据读取失败")
            continue

        visibility = normalize_visibility(raw_visibility)
        if visibility is None:
            issues.append(
                f"{doc_id} 第 {index} 块 visibility 缺失或非法：{_describe(raw_visibility)}"
            )
            continue
        visibilities.add(visibility)

        owner = normalize_owner(raw_owner)
        if visibility == VISIBILITY_PUBLIC:
            if owner_declared(raw_owner):
                issues.append(
                    f"{doc_id} 第 {index} 块：public 文档不允许声明 owner"
                    f"（{_describe(raw_owner)}）"
                )
        elif owner is None:
            issues.append(
                f"{doc_id} 第 {index} 块：private 文档必须带有效 owner"
                f"（{_describe(raw_owner)}）"
            )
        else:
            owners.add(owner)

    if len(visibilities) > 1:
        issues.append(f"{doc_id} 混用多种可见性声明：{sorted(visibilities)}")
    if len(owners) > 1:
        issues.append(f"{doc_id} 出现多个不同 owner：{sorted(owners)}")

    access = classify_document(metas)
    if not issues and access.is_unknown:
        issues.append(f"{doc_id} 无法判定为 public 或 private（元数据互相矛盾）")
    return access, issues


def _normalize_access(access: Any) -> tuple[str, str | None]:
    """把判定结果再校验一遍：手工构造的畸形 DocAccess 一律按 unknown 处理。"""
    if not isinstance(access, DocAccess):
        return (VISIBILITY_UNKNOWN, None)

    visibility = access.visibility
    owner = _normalize_owner(access.owner)
    if visibility == VISIBILITY_PUBLIC:
        return (VISIBILITY_PUBLIC, None) if owner is None else (VISIBILITY_UNKNOWN, None)
    if visibility == VISIBILITY_PRIVATE:
        return (VISIBILITY_PRIVATE, owner) if owner else (VISIBILITY_UNKNOWN, None)
    return (VISIBILITY_UNKNOWN, None)


def _normalize_user_id(user_id: Any) -> str | None:
    """当前登录用户的 id，规则与 owner 相同。"""
    return _normalize_owner(user_id)


def _normalize_admin(is_admin: Any) -> bool:
    try:
        return bool(is_admin)
    except Exception:  # __bool__ 抛异常的对象不能被当成管理员
        return False


def can_read(access: Any, *, user_id: Any, is_admin: bool = False) -> bool:
    """能否预览这篇文档的原文。

    - public：所有已认证用户可读；
    - private：仅所属用户；
    - unknown：任何人都不行。

    管理员在这里**不额外放行**（他人私人文档对管理员同样不可见）。
    保留 ``is_admin`` 参数是为了与 ``can_delete`` 保持同一签名，调用方不必分支。
    """
    visibility, owner = _normalize_access(access)
    uid = _normalize_user_id(user_id)
    if uid is None:
        return False                          # 没有可信身份，一律不放行
    if visibility == VISIBILITY_PUBLIC:
        return True
    if visibility == VISIBILITY_PRIVATE:
        return uid == owner
    return False


def can_retrieve(access: Any, *, user_id: Any, is_admin: bool = False) -> bool:
    """能否被检索召回。

    与 ``can_read`` 目前是同一套规则：能被检索到就等于能被读到（引用里带着原文片段），
    所以检索侧必须在召回阶段就过滤，而不是召回后再丢弃。
    单独留一个函数名，是为了将来检索需要更严（例如只允许命中标题）时，
    不必去改调用方的语义。
    """
    return can_read(access, user_id=user_id, is_admin=is_admin)


def can_delete(access: Any, *, user_id: Any, is_admin: bool = False) -> bool:
    """能否删除这篇文档。

    - public：按现有约定，仅管理员可删；
    - private：仅所属用户（管理员删除他人的私人文档会被拒绝）；
    - unknown：任何人都不行，包括管理员——来源不明就先隔离，等人来认领。
    """
    visibility, owner = _normalize_access(access)
    uid = _normalize_user_id(user_id)
    if uid is None:
        return False
    if visibility == VISIBILITY_PRIVATE:
        return uid == owner
    if visibility == VISIBILITY_PUBLIC:
        return _normalize_admin(is_admin)
    return False
