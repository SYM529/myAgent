"""显式认领历史 unknown 文档为 public（一次性、可审计的迁移工具）。

背景与边界
----------
历史文档缺少 owner/visibility，会被判为 unknown，对所有用户不可见。
直接重灌相同内容会被写入保护拒绝（"既有记录是 unknown → 拒绝写入"），
所以迁移只能走**显式认领**：只接受命令行明确给出的 doc_id 白名单，逐条验证

- 文档存在于 chunks.json；
- JSON 侧判定为 unknown（或已 public，此时作为幂等/补写分支）；
- 每个 chunk 都没有 owner/visibility 字段残留（None 与空白字符串视为没写）；
- Chroma 侧的 chunk id 集合与 JSON 完全一致，且判定同样是 unknown；

然后**仅修改 metadata**：给目标文档的全部 chunk 写入 ``visibility="public"``
（不带 owner）。先复用 HybridStore 的原子写落盘 chunks.json，成功后切换内存，
最后按批同步 Chroma metadata。doc_id / chunk_id / 正文 / 向量一律不变。

失败语义
--------
- 任何校验不通过 → 抛 ClaimError，零写入；
- chunks.json 写入失败 → 抛 ClaimError，内存与 Chroma 均未动；
- Chroma 更新失败 → JSON 与内存已完成（两者一致），如实返回
  ``ok=False``、``chroma_incomplete=True``，并用 ``--repair-chroma-only``
  重试补齐（该模式只写 Chroma，不重复写 JSON）。

用法
----
    python -m agent.rag.claim --persist-dir ./rag_store \
        --doc-id <id> [--doc-id <id> ...] --visibility public \
        [--dry-run] [--yes] [--repair-chroma-only]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Sequence

from agent.rag.config import PERSIST_DIR
from agent.rag.store import HybridStore
from agent.rag.visibility import (
    VISIBILITY_PRIVATE,
    VISIBILITY_PUBLIC,
    VISIBILITY_UNKNOWN,
    classify_document,
)

CLAIM_BATCH_SIZE = 100
SUPPORTED_VISIBILITIES = (VISIBILITY_PUBLIC,)


class ClaimError(RuntimeError):
    """认领被拒绝（保证零写入）时抛出。"""


class _NoEmbedding:
    """认领只改 metadata，不读也不写向量；这里显式拒绝任何 embedding 调用。

    这样 CLI 在**没有 embedding 凭据**的环境（如 CI、纯运维机）也能运行——
    这正是它和上传/检索工具的本质区别。
    """

    def embed(self, texts):
        raise RuntimeError("claim 流程不应调用 embedding")

    def embed_one(self, text):
        raise RuntimeError("claim 流程不应调用 embedding")


def _group_by_doc(chunks: Iterable) -> dict[str, list]:
    grouped: dict[str, list] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.doc_id, []).append(chunk)
    return grouped


def _has_ownership_field(metadata: dict) -> bool:
    """owner/visibility 是否写过非空值（None 与空白字符串视为没写）。"""
    for key in ("owner", "visibility"):
        value = metadata.get(key) if isinstance(metadata, dict) else None
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return True
    return False


def _chroma_state(
    store: HybridStore, chunk_ids: Sequence[str]
) -> tuple[list[str], list[dict]]:
    """读取 Chroma 侧这些 chunk 的 id 与 metadata；结构异常一律拒绝。"""
    found = store._collection.get(ids=list(chunk_ids), include=["metadatas"])
    if not isinstance(found, dict) or "ids" not in found:
        raise ClaimError("Chroma 返回缺少 ids，无法确认既有记录，拒绝继续")
    raw_ids = found.get("ids")
    raw_metas = found.get("metadatas")
    if not isinstance(raw_ids, (list, tuple)):
        raise ClaimError(f"Chroma ids 结构异常（{type(raw_ids).__name__}），拒绝继续")
    if not isinstance(raw_metas, (list, tuple)) or len(raw_metas) != len(raw_ids):
        raise ClaimError("Chroma metadata 数量与 ids 不一致，拒绝继续")
    for item in raw_ids:
        if not isinstance(item, str) or not item.strip():
            raise ClaimError("Chroma 返回了非法 chunk id，拒绝继续")
    return [str(item) for item in raw_ids], [
        dict(meta or {}) for meta in raw_metas
    ]


def _entry(doc_id: str, chunks: list, before: str, after: str) -> dict:
    return {
        "doc_id": doc_id,
        "source": chunks[0].source,
        "chunks": len(chunks),
        "before": before,
        "after": after,
    }


def _update_chroma(store: HybridStore, updates: list[tuple[str, dict]]) -> None:
    for start in range(0, len(updates), CLAIM_BATCH_SIZE):
        batch = updates[start : start + CLAIM_BATCH_SIZE]
        store._collection.update(
            ids=[chunk_id for chunk_id, _ in batch],
            metadatas=[meta for _, meta in batch],
        )


def claim_documents(
    store: HybridStore,
    doc_ids: Sequence[str],
    *,
    visibility: str = VISIBILITY_PUBLIC,
    dry_run: bool = False,
    repair_chroma_only: bool = False,
) -> dict:
    """把白名单内的 unknown 文档认领为 public；返回结构化报告。"""
    if visibility not in SUPPORTED_VISIBILITIES:
        raise ClaimError(f"仅支持 --visibility public（收到 {visibility!r}）")

    ids: list[str] = []
    for doc_id in doc_ids:
        if not isinstance(doc_id, str) or not doc_id.strip():
            raise ClaimError(f"doc_id 非法：{doc_id!r}")
        if doc_id not in ids:
            ids.append(doc_id)
    if not ids:
        raise ClaimError("必须显式指定至少一个 --doc-id")

    grouped = _group_by_doc(store.all_chunks())
    entries: list[dict] = []
    targets: list[tuple[str, list, str]] = []
    for doc_id in ids:
        chunks = grouped.get(doc_id)
        if not chunks:
            raise ClaimError(f"{doc_id}：文档不存在")
        metas = [dict(chunk.metadata or {}) for chunk in chunks]
        access = classify_document(metas)

        if repair_chroma_only:
            if not access.is_public:
                raise ClaimError(
                    f"{doc_id}：JSON 侧当前不是合法 public（{access.visibility}），"
                    "无法执行 --repair-chroma-only"
                )
            entries.append(_entry(doc_id, chunks, access.visibility, VISIBILITY_PUBLIC))
            targets.append((doc_id, chunks, "repair"))
            continue

        if access.is_public:
            entries.append(_entry(doc_id, chunks, access.visibility, VISIBILITY_PUBLIC))
            targets.append((doc_id, chunks, "skip"))
            continue

        if access.visibility != VISIBILITY_UNKNOWN:
            raise ClaimError(
                f"{doc_id}：当前判定为 {access.visibility}，只允许认领 unknown 文档"
            )
        if any(_has_ownership_field(meta) for meta in metas):
            raise ClaimError(
                f"{doc_id}：存在 owner/visibility 字段残留（可能非法或冲突），"
                "拒绝自动认领；请先人工核查"
            )
        entries.append(_entry(doc_id, chunks, access.visibility, VISIBILITY_PUBLIC))
        targets.append((doc_id, chunks, "claim"))

    # Chroma 预检：两侧 id 集合必须一致；可认领文档在 Chroma 侧也必须是 unknown。
    chroma_updates: list[tuple[str, dict]] = []
    needs_repair: list[str] = []
    claimed: dict[str, list] = {}
    for doc_id, chunks, mode in targets:
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        found_ids, found_metas = _chroma_state(store, chunk_ids)
        if set(found_ids) != set(chunk_ids):
            raise ClaimError(
                f"{doc_id}：两侧 chunk 集合不一致（JSON {len(chunk_ids)} 块 / "
                f"Chroma {len(found_ids)} 块），拒绝继续；请先人工核对"
            )
        chroma_access = classify_document(found_metas)

        if mode == "claim":
            if chroma_access.visibility != VISIBILITY_UNKNOWN or any(
                _has_ownership_field(meta) for meta in found_metas
            ):
                raise ClaimError(
                    f"{doc_id}：Chroma 侧判定为 {chroma_access.visibility} 或存在归属字段，"
                    "拒绝继续；请先排查两侧一致性"
                )
            claimed[doc_id] = chunks
            chroma_updates.extend(
                (chunk_id, {**meta, "visibility": VISIBILITY_PUBLIC})
                for chunk_id, meta in zip(found_ids, found_metas)
            )
        elif mode == "skip":
            if chroma_access.is_public:
                continue  # 完全幂等：两侧都已合法 public
            needs_repair.append(doc_id)
        else:  # repair：只补 Chroma，但绝不在 Chroma 侧留下 owner 残留
            if any(meta.get("owner") not in (None, "") for meta in found_metas):
                raise ClaimError(
                    f"{doc_id}：Chroma 侧存在 owner 残留，拒绝补写；请先人工核查"
                )
            if chroma_access.is_public:
                continue
            chroma_updates.extend(
                (chunk_id, {**meta, "visibility": VISIBILITY_PUBLIC})
                for chunk_id, meta in zip(found_ids, found_metas)
            )

    report: dict = {
        "ok": True,
        "dry_run": bool(dry_run),
        "repair_chroma_only": bool(repair_chroma_only),
        "changed_json": False,
        "changed_chroma": False,
        "chroma_incomplete": False,
        "documents": entries,
        "error": None,
    }

    if needs_repair:
        report.update(
            ok=False,
            chroma_incomplete=True,
            error=(
                "以下文档 JSON 侧已 public 但 Chroma 未同步："
                + ", ".join(needs_repair)
                + "；请使用 --repair-chroma-only 补齐（该模式只写 Chroma）"
            ),
        )
        return report

    if dry_run:
        return report

    if repair_chroma_only:
        try:
            _update_chroma(store, chroma_updates)
        except Exception as error:  # noqa: BLE001 - 原样上报，交由运维判断
            report.update(
                ok=False,
                chroma_incomplete=True,
                error=f"Chroma 更新失败：{error}；可重试 --repair-chroma-only",
            )
            return report
        report["changed_chroma"] = bool(chroma_updates)
        return report

    if claimed:
        candidate = dict(store._chunks)
        for chunks in claimed.values():
            for chunk in chunks:
                merged = dict(chunk.metadata or {})
                merged["visibility"] = VISIBILITY_PUBLIC
                merged.pop("owner", None)
                candidate[chunk.chunk_id] = replace(chunk, metadata=merged)
        try:
            store._save_chunks(candidate)
        except Exception as error:  # noqa: BLE001 - 原文件由原子写保证不被破坏
            raise ClaimError(f"JSON 保存失败，未做任何修改：{error}") from error
        store._chunks = candidate
        store._invalidate_bm25()
        report["changed_json"] = True

    if chroma_updates:
        try:
            _update_chroma(store, chroma_updates)
        except Exception as error:  # noqa: BLE001 - 原样上报，交由运维判断
            report.update(
                ok=False,
                chroma_incomplete=True,
                error=(
                    f"Chroma 更新失败：{error}；JSON 已保存、内存已切换，"
                    "请使用 --repair-chroma-only 重试补齐（不会重复写 JSON）"
                ),
            )
            return report
        report["changed_chroma"] = True

    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m agent.rag.claim",
        description=(
            "把白名单中的历史 unknown 文档显式认领为 public"
            "（仅改 metadata，保留原 doc_id / chunk / 向量）。"
        ),
    )
    parser.add_argument("--persist-dir", type=Path, default=PERSIST_DIR)
    parser.add_argument(
        "--doc-id",
        action="append",
        default=[],
        metavar="DOC_ID",
        help="要认领的 doc_id（可重复；不接受通配符）",
    )
    parser.add_argument(
        "--visibility",
        default=VISIBILITY_PUBLIC,
        choices=list(SUPPORTED_VISIBILITIES),
    )
    parser.add_argument("--dry-run", action="store_true", help="只输出将修改的内容，不写任何存储")
    parser.add_argument("--yes", action="store_true", help="确认执行非 dry-run 的写操作")
    parser.add_argument(
        "--repair-chroma-only",
        action="store_true",
        help="仅补齐 Chroma metadata（用于上次 Chroma 更新失败后的重试）",
    )
    args = parser.parse_args(argv)

    if not args.dry_run and not args.yes:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": (
                        "非 dry-run 属于写操作，必须显式加 --yes 确认；"
                        "请先用 --dry-run 审阅将要修改的内容"
                    ),
                },
                ensure_ascii=False,
            )
        )
        return 2

    store = HybridStore(persist_dir=args.persist_dir, embedding_client=_NoEmbedding())
    try:
        report = claim_documents(
            store,
            args.doc_id,
            visibility=args.visibility,
            dry_run=args.dry_run,
            repair_chroma_only=args.repair_chroma_only,
        )
    except ClaimError as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
