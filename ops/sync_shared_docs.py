"""只同步 CLI 导入的公共文档，不碰用户上传。

doc_id 是内容哈希，文档修改后会生成新的 doc_id；直接增量 ingest 会把
新旧两版都留在库里，旧的仍可能被检索到。--rebuild 虽然干净，但会清空
用户上传的私有文档。

本脚本只删除“来源匹配当前 docs 目录、且无 owner/file_id”的旧版本
（private 文档除外），然后增量写入新版本。默认 dry-run，确认后加 --apply。

用法（容器内）：
    python /app/sync_shared_docs.py ./docs
    python /app/sync_shared_docs.py ./docs --apply
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for _stream in (getattr(sys, "stdout", None), getattr(sys, "stderr", None)):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from agent.rag import get_store, ingest_directory  # noqa: E402
from agent.rag.loader import load_directory  # noqa: E402

PUBLIC = "public"


def stale_public_doc_ids(chunks, current_ids: set[str], current_sources: set[str]) -> list[str]:
    grouped: dict[str, list] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.doc_id, []).append(chunk)

    stale: list[str] = []
    for doc_id, group in grouped.items():
        if doc_id in current_ids:
            continue
        if not {c.source for c in group} & current_sources:
            continue
        metas = [c.metadata or {} for c in group]
        # 早期 CLI 导入可能没有 visibility 元数据；owner/file_id 才是
        # “用户上传”的可靠标志，private 文档必须保留。
        if any(meta.get("owner") or meta.get("file_id") for meta in metas):
            continue
        if any(meta.get("visibility") == "private" for meta in metas):
            continue
        stale.append(doc_id)
    return sorted(stale)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="同步 CLI 公共文档，保留用户上传")
    parser.add_argument("directory", nargs="?", default="./docs")
    parser.add_argument("--apply", action="store_true", help="执行删除与入库；默认只预览")
    args = parser.parse_args(argv)

    docs_dir = Path(args.directory)
    if not docs_dir.is_dir():
        raise SystemExit(f"目录不存在：{docs_dir}")

    documents = load_directory(docs_dir)
    current_ids = {document.doc_id for document in documents}
    current_sources = {document.source for document in documents}

    store = get_store()
    before = store.count()
    stale = stale_public_doc_ids(store.all_chunks(), current_ids, current_sources)

    print(f"docs 文档：{len(documents)} 篇；库内 chunk：{before}；待清理旧版本：{len(stale)} 个")
    for doc_id in stale:
        sources = sorted({c.source for c in store.get_document_chunks(doc_id)})
        print(f"  - {doc_id}  {', '.join(sources)}")

    if not args.apply:
        print("dry-run：未改动知识库。确认后加 --apply。")
        return 0

    for doc_id in stale:
        result = store.delete_document(doc_id)
        detail = f"（{result.error}）" if result.error else ""
        print(f"删除 {doc_id}：{result.status}{detail}")

    stats = ingest_directory(docs_dir, rebuild=False, visibility=PUBLIC)
    print(
        f"入库完成：文档 {stats['documents']} 篇，写入 {stats['chunks']} chunks，"
        f"库内总数 {stats['total']}（清理前 {before}）"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
