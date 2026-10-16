"""知识库只读审计：统计、隔离清单与存储一致性检查。

为什么不用 HybridStore / PersistentClient
----------------------------------------
``chromadb.PersistentClient`` 打开时会创建目录、执行 schema 迁移、写回 sqlite。
一个只想"看看现状"的工具不该有这些副作用：审计不能改变被审计对象，
更不能在运维准备恢复数据之前动到库。所以这里：

- 只读 chunks.json（纯 JSON 解析，不写回）；
- 用 sqlite3 的 ``mode=ro`` 直连 ``chroma.sqlite3``，只取 chunk id 与归属类元数据键；
- 完全不 import chromadb、不构造任何客户端、不调用 add / upsert / delete / clear / 重建。

代价是依赖 Chroma 的内部表结构（``embeddings.embedding_id`` 与 ``embedding_metadata``）。
版本变化导致表结构对不上时，工具明确报"无法读取"，不猜、更不会修。

只输出元数据
------------
chunk 正文（chunks.json 的 ``text``、Chroma 的 ``chroma:document``）一律不读取、不打印；
上传文件 id（``file_id``）也不输出——它指向磁盘上具体某个文件。
清单里只有 doc_id、来源文件名、chunk 数、归属状态与判定结论。

统计口径
--------
所有覆盖率都按**文档**计（不是按 chunk），且刻意拆开两个容易混淆的口径：

- visibility 字段存在率：任一 chunk 写了非空 visibility 的文档占比，**含写坏的值**
  （例如 ``visibility="shared"``、块间 public/private 互相矛盾）；
- 合法标记率：拿到唯一且合法 ``public``/``private`` 的文档占比——只有这部分
  能被自动判定，剩下的是人工确认清单的来源。

两个数字不一样时，差额就是"写了但写坏了"的文档，这类必须先隔离再排查，
所以不能在报告里合成一个指标。

BM25 没有持久化文件（``store.py`` 里定义了 ``bm25.pkl`` 路径但从未写入，索引在进程内重建），
所以这里只报告文件是否存在，绝不反序列化——反序列化不受信任的 pickle 等于执行任意代码。

用法::

    python -m agent.rag.audit                  # 人类可读的清单
    python -m agent.rag.audit --json           # 机器可读（仍然只打到终端）
    python -m agent.rag.audit --persist-dir ./rag_store
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from agent.rag.config import PERSIST_DIR
from agent.rag.visibility import (
    VISIBILITY_PRIVATE,
    VISIBILITY_PUBLIC,
    VISIBILITY_UNKNOWN,
    classify_document,
)

JSON_STATUS_OK = "ok"
JSON_STATUS_MISSING = "missing"
JSON_STATUS_CORRUPT = "corrupt"

CHUNKS_FILENAME = "chunks.json"
CHROMA_DIRNAME = "chroma"
CHROMA_DB_FILENAME = "chroma.sqlite3"
BM25_FILENAME = "bm25.pkl"

# 只统计这两个键的存在情况，用于比对两侧的归属元数据覆盖
_OWNERSHIP_KEYS = ("owner", "visibility")

# 单次报告里最多列出的 id 数量，避免库很大时刷屏
_MAX_ID_SAMPLE = 10


@dataclass(frozen=True)
class DocumentEntry:
    """文档级清单的一行。字段全部来自元数据，不含任何正文。"""

    doc_id: str
    source: str
    chunks: int
    owner: str | None              # 归一化后的归属用户 id；None 表示读不出归属
    owner_present: bool            # 原始元数据里是否写过 owner 字段
    visibility_present: bool       # 原始元数据里是否写过非空 visibility 字段
    visibility_marker: str | None  # 唯一且合法的 public/private；None 表示没写或写坏了
    metadata_invalid: bool         # 出现了无法解析或互相矛盾的元数据
    status: str                    # public / private / unknown
    needs_review: bool             # unknown → 待人工确认

    @property
    def visibility_invalid(self) -> bool:
        """写了 visibility 却拿不到唯一合法值（非法值，或块间互相矛盾）。"""
        return self.visibility_present and self.visibility_marker is None

    @property
    def note(self) -> str:
        if self.status == VISIBILITY_PUBLIC:
            return "公共文档（CLI 入库，所有人可检索）"
        if self.status == VISIBILITY_PRIVATE:
            if self.visibility_marker:
                return "私人文档（仅归属用户可检索、预览、删除）"
            return "私人文档（历史数据：无 visibility 标记，按 owner 判定）"
        if self.metadata_invalid:
            return "待人工确认：元数据非法或互相矛盾，不推断为公共或私人"
        return "待人工确认：无 owner、无 visibility 标记，来源无法确认"


@dataclass(frozen=True)
class ChromaSnapshot:
    """Chroma 侧只读快照。chunk_ids 用于与 chunks.json 比对。"""

    available: bool
    chunks: int = 0
    chunk_ids: tuple[str, ...] = ()
    docs_with_owner: int = 0
    docs_with_visibility: int = 0
    error: str | None = None


@dataclass(frozen=True)
class ChunkLoad:
    """chunks.json 的读取结果。"""

    status: str
    items: list[dict]
    skipped: int = 0
    error: str | None = None


@dataclass(frozen=True)
class Consistency:
    """两侧存储的差异。只报告，不修复。"""

    chroma_available: bool
    chroma_error: str | None
    chroma_chunks: int
    only_in_chunks_json: list[str]
    only_in_chroma: list[str]
    chroma_docs_with_owner: int
    chroma_docs_with_visibility: int

    @property
    def consistent(self) -> bool:
        return (
            self.chroma_available
            and not self.only_in_chunks_json
            and not self.only_in_chroma
        )


@dataclass(frozen=True)
class AuditReport:
    """一次审计的完整结论。"""

    persist_dir: str
    chunks_status: str
    chunks_error: str | None
    chunks_skipped: int
    unattributed_chunks: int
    documents: list[DocumentEntry]
    total_documents: int
    total_chunks: int
    public_documents: int
    private_documents: int
    unknown_documents: int
    documents_with_owner: int
    # visibility 有两个不同口径，刻意分开统计，避免"字段存在"被误读成"标记合法"：
    #   字段存在率：任一 chunk 写了非空 visibility（值可能非法，甚至互相矛盾）
    #   合法标记率：该文档拿到了唯一且合法的 public/private 标记
    documents_with_visibility_field: int
    documents_with_valid_visibility: int
    duplicate_sources: list[tuple[str, list[str]]]
    bm25_file_present: bool
    consistency: Consistency

    @property
    def owner_coverage(self) -> float:
        if not self.total_documents:
            return 0.0
        return self.documents_with_owner / self.total_documents

    @property
    def visibility_field_coverage(self) -> float:
        """原始 visibility 字段的存在率（含非法值与矛盾值）。"""
        if not self.total_documents:
            return 0.0
        return self.documents_with_visibility_field / self.total_documents

    @property
    def valid_visibility_coverage(self) -> float:
        """唯一且合法标记的覆盖率——只有这部分文档能被自动判定。"""
        if not self.total_documents:
            return 0.0
        return self.documents_with_valid_visibility / self.total_documents

    @property
    def review_queue(self) -> list[DocumentEntry]:
        """待人工确认的文档（unknown），恢复方案的第一手依据。"""
        return [doc for doc in self.documents if doc.needs_review]


# ---------- 读取：chunks.json ----------


def load_chunks(path: Path) -> ChunkLoad:
    """读取 chunks.json。文件缺失、损坏都只报告，不创建也不修复。"""
    path = Path(path)
    if not path.is_file():
        return ChunkLoad(JSON_STATUS_MISSING, [], error=f"未找到 {path.name}")

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return ChunkLoad(
            JSON_STATUS_CORRUPT, [], error=f"JSON 解析失败：{e.msg}（第 {e.lineno} 行）"
        )
    except OSError as e:
        return ChunkLoad(JSON_STATUS_CORRUPT, [], error=f"读取失败：{e}")

    if not isinstance(raw, list):
        return ChunkLoad(JSON_STATUS_CORRUPT, [], error="顶层结构不是数组")

    items = [item for item in raw if isinstance(item, dict)]
    return ChunkLoad(JSON_STATUS_OK, items, skipped=len(raw) - len(items))


# ---------- 读取：Chroma（sqlite 只读） ----------


def read_chroma(persist_dir: Path) -> ChromaSnapshot:
    """用 sqlite 只读连接读取 chunk id 与归属键覆盖情况。

    不 import chromadb、不构造 PersistentClient——那会创建目录并跑 schema 迁移。
    """
    db_path = Path(persist_dir) / CHROMA_DIRNAME / CHROMA_DB_FILENAME
    if not db_path.is_file():
        return ChromaSnapshot(available=False, error=f"未找到 {CHROMA_DB_FILENAME}")

    try:
        # mode=ro：连接无法写入；timeout 短一些，避免库被占用时挂住审计
        connection = sqlite3.connect(
            f"file:{db_path.as_posix()}?mode=ro", uri=True, timeout=1.0
        )
    except sqlite3.Error as e:
        return ChromaSnapshot(available=False, error=f"打开失败：{e}")

    try:
        cursor = connection.cursor()
        chunk_ids = [
            row[0]
            for row in cursor.execute(
                "SELECT embedding_id FROM embeddings WHERE embedding_id IS NOT NULL"
            )
        ]
        # 只取归属类键名，不取任何正文（正文存在 chroma:document 里，不在本查询范围内）
        placeholders = ", ".join("?" for _ in _OWNERSHIP_KEYS)
        rows = cursor.execute(
            "SELECT e.embedding_id, m.key FROM embeddings e "
            "JOIN embedding_metadata m ON m.id = e.id "
            f"WHERE m.key IN ({placeholders})",
            _OWNERSHIP_KEYS,
        ).fetchall()
    except sqlite3.Error as e:
        return ChromaSnapshot(
            available=False,
            error=f"读取失败：{e}（表结构可能随 Chroma 版本变化）",
        )
    finally:
        connection.close()

    owners: set[str] = set()
    visibilities: set[str] = set()
    for chunk_id, key in rows:
        doc_id = _doc_id_of(str(chunk_id))
        if not doc_id:
            continue
        if key == "owner":
            owners.add(doc_id)
        elif key == "visibility":
            visibilities.add(doc_id)

    return ChromaSnapshot(
        available=True,
        chunks=len(chunk_ids),
        chunk_ids=tuple(str(cid) for cid in chunk_ids),
        docs_with_owner=len(owners),
        docs_with_visibility=len(visibilities),
    )


def _doc_id_of(chunk_id: str) -> str:
    """从 chunk_id 反推 doc_id（形如 ``{doc_id}-{chunk_index}``）。"""
    head, sep, tail = chunk_id.rpartition("-")
    if sep and tail.isdigit() and head:
        return head
    return ""


# ---------- 汇总 ----------


def _present(value: Any) -> bool:
    """字段是否"写了东西"：None 与空白字符串视为没写。"""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _marker(value: Any) -> str | None:
    """把原始 visibility 值渲染成可安全打印的短标记。

    非字符串只打印类型名——元数据里可能是任意对象，不能把它的内容倒进报告。
    """
    if value is None:
        return None
    if isinstance(value, str):
        text = " ".join(value.split())
        return text[:24] if text else None
    return f"<{type(value).__name__}>"


def _document_entry(doc_id: str, items: list[dict]) -> DocumentEntry:
    metas = []
    for item in items:
        meta = item.get("metadata")
        metas.append(meta if isinstance(meta, dict) else None)

    access = classify_document(metas)

    owner_present = any(_present(meta.get("owner")) for meta in metas if meta)
    visibility_present = any(_present(meta.get("visibility")) for meta in metas if meta)
    markers = {m for m in (_marker(meta.get("visibility")) for meta in metas if meta) if m}
    declared = sorted(m for m in markers if m in (VISIBILITY_PUBLIC, VISIBILITY_PRIVATE))
    visibility_marker = declared[0] if len(declared) == 1 and len(markers) == 1 else None

    # 判定为 unknown 时再细分原因：写了东西但读不出来，还是根本没写
    metadata_invalid = access.is_unknown and (owner_present or bool(markers))

    # source 与 summarize_documents 一致：以最后见到的为准
    source = ""
    for item in items:
        candidate = item.get("source")
        if isinstance(candidate, str) and candidate:
            source = candidate

    return DocumentEntry(
        doc_id=doc_id,
        source=source,
        chunks=len(items),
        owner=access.owner,
        owner_present=owner_present,
        visibility_present=visibility_present,
        visibility_marker=visibility_marker,
        metadata_invalid=metadata_invalid,
        status=access.visibility,
        needs_review=access.is_unknown,
    )


def build_report(persist_dir: Path | str = PERSIST_DIR) -> AuditReport:
    """执行一次完整的只读审计。不创建文件、不修改任何存储。"""
    persist = Path(persist_dir)

    load = load_chunks(persist / CHUNKS_FILENAME)
    chroma = read_chroma(persist)

    grouped: dict[str, list[dict]] = {}
    unattributed = 0
    for item in load.items:
        doc_id = item.get("doc_id")
        if not isinstance(doc_id, str) or not doc_id:
            unattributed += 1
            continue
        grouped.setdefault(doc_id, []).append(item)

    documents = [_document_entry(doc_id, items) for doc_id, items in grouped.items()]
    documents.sort(key=lambda d: (d.source or "", d.doc_id))

    by_source: dict[str, list[str]] = {}
    for doc in documents:
        if doc.source:
            by_source.setdefault(doc.source, []).append(doc.doc_id)
    duplicates = sorted(
        (source, sorted(ids)) for source, ids in by_source.items() if len(ids) > 1
    )

    json_ids = {
        item.get("chunk_id")
        for item in load.items
        if isinstance(item.get("chunk_id"), str)
    }
    chroma_ids = set(chroma.chunk_ids)

    consistency = Consistency(
        chroma_available=chroma.available,
        chroma_error=chroma.error,
        chroma_chunks=chroma.chunks,
        only_in_chunks_json=sorted(json_ids - chroma_ids) if chroma.available else [],
        only_in_chroma=sorted(chroma_ids - json_ids) if chroma.available else [],
        chroma_docs_with_owner=chroma.docs_with_owner,
        chroma_docs_with_visibility=chroma.docs_with_visibility,
    )

    return AuditReport(
        persist_dir=str(persist),
        chunks_status=load.status,
        chunks_error=load.error,
        chunks_skipped=load.skipped,
        unattributed_chunks=unattributed,
        documents=documents,
        total_documents=len(documents),
        total_chunks=len(load.items),
        public_documents=sum(1 for d in documents if d.status == VISIBILITY_PUBLIC),
        private_documents=sum(1 for d in documents if d.status == VISIBILITY_PRIVATE),
        unknown_documents=sum(1 for d in documents if d.status == VISIBILITY_UNKNOWN),
        documents_with_owner=sum(1 for d in documents if d.owner_present),
        documents_with_visibility_field=sum(
            1 for d in documents if d.visibility_present
        ),
        documents_with_valid_visibility=sum(
            1 for d in documents if d.visibility_marker is not None
        ),
        duplicate_sources=duplicates,
        bm25_file_present=(persist / BM25_FILENAME).is_file(),
        consistency=consistency,
    )


# ---------- 渲染 ----------


def _sanitize(value: Any, limit: int = 48) -> str:
    """把不可信字符串压成适合终端的一行。

    文件名和 doc_id 都来自上传数据，可能夹带换行或控制字符——
    直接打印会搅乱报告，甚至伪造出并不存在的行。
    """
    if not isinstance(value, str):
        return "(非字符串)"
    text = "".join(ch for ch in value if ch.isprintable())
    text = " ".join(text.split())
    if not text:
        return "(空)"
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _width(text: str) -> int:
    """终端显示宽度：中日韩全角字符占两列，否则表格会错位。"""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def _pad(text: str, width: int, align: str = "left") -> str:
    space = max(0, width - _width(text))
    if align == "right":
        return " " * space + text
    return text + " " * space


def render_report(report: AuditReport) -> str:
    """渲染成终端文本。只打元数据，不落盘。"""
    lines: list[str] = []
    lines.append("知识库只读审计")
    lines.append(f"持久化目录：{_sanitize(report.persist_dir, 80)}")

    if report.chunks_status == JSON_STATUS_OK:
        lines.append(f"chunks.json：可读，{report.total_chunks} 个 chunk")
    else:
        lines.append(f"chunks.json：{report.chunks_status}（{report.chunks_error}）")
    if report.chunks_skipped:
        lines.append(f"  忽略的非对象条目：{report.chunks_skipped}")
    if report.unattributed_chunks:
        lines.append(f"  无法归属的 chunk（doc_id 缺失或非法）：{report.unattributed_chunks}")

    consistency = report.consistency
    if consistency.chroma_available:
        lines.append(f"Chroma：可读，{consistency.chroma_chunks} 个 chunk")
    else:
        lines.append(f"Chroma：不可读（{consistency.chroma_error}）")
    if report.bm25_file_present:
        lines.append(f"BM25：存在 {BM25_FILENAME}（本工具不做反序列化）")
    else:
        lines.append("BM25：无持久化文件（索引在进程内重建，不参与比对）")

    lines.append("")
    lines.append(f"文档总数：{report.total_documents}　chunk 总数：{report.total_chunks}")
    lines.append(
        f"分类：public {report.public_documents}　"
        f"private {report.private_documents}　"
        f"unknown（待人工确认）{report.unknown_documents}"
    )
    lines.append(
        f"覆盖：owner {report.documents_with_owner}/{report.total_documents}"
        f"（{_pct(report.owner_coverage)}）　"
        f"visibility 字段 {report.documents_with_visibility_field}/{report.total_documents}"
        f"（{_pct(report.visibility_field_coverage)}）　"
        f"合法标记 {report.documents_with_valid_visibility}/{report.total_documents}"
        f"（{_pct(report.valid_visibility_coverage)}）"
    )
    if not report.duplicate_sources:
        lines.append("重复来源：无")
    else:
        lines.append(f"重复来源：{len(report.duplicate_sources)} 组（同名不同 doc_id）")
        for source, doc_ids in report.duplicate_sources:
            joined = ", ".join(_sanitize(i, 16) for i in doc_ids)
            lines.append(f"  - {_sanitize(source)} → {joined}")

    lines.append("")
    header = "".join([
        _pad("doc_id", 18), _pad("chunk", 6, "right"), "  ",
        _pad("owner", 10), _pad("visibility", 12), _pad("分类", 9),
        _pad("来源", 28), "说明",
    ])
    lines.append(header)
    lines.append("-" * _width(header))
    for doc in report.documents:
        owner = doc.owner if doc.owner else ("(非法)" if doc.owner_present else "-")
        visibility = doc.visibility_marker or (
            "(无效)" if doc.visibility_invalid else "-"
        )
        lines.append("".join([
            _pad(_sanitize(doc.doc_id, 18), 18),
            _pad(str(doc.chunks), 6, "right"), "  ",
            _pad(str(owner), 10),
            _pad(str(visibility), 12),
            _pad(doc.status, 9),
            _pad(_sanitize(doc.source, 28), 28),
            doc.note,
        ]))

    if report.review_queue:
        lines.append("")
        lines.append(
            f"待人工确认（unknown）：{len(report.review_queue)} 篇"
            "——先保持隔离，确认来源后再决定认领方式"
        )
        for doc in report.review_queue:
            lines.append(
                f"  - {_sanitize(doc.doc_id, 18)}  {_sanitize(doc.source)}  {doc.chunks} 段"
            )

    lines.append("")
    if not consistency.chroma_available:
        lines.append("一致性：无法比对（Chroma 不可读）")
    elif consistency.consistent:
        lines.append("一致性：一致（两侧 chunk id 集合完全相同）")
    else:
        lines.append(
            "一致性：不一致——仅 chunks.json "
            f"{len(consistency.only_in_chunks_json)} 个，仅 Chroma "
            f"{len(consistency.only_in_chroma)} 个（只报告差异，本工具不修复）"
        )
        for label, ids in (
            ("仅存在于 chunks.json", consistency.only_in_chunks_json),
            ("仅存在于 Chroma", consistency.only_in_chroma),
        ):
            for chunk_id in ids[:_MAX_ID_SAMPLE]:
                lines.append(f"  - {label}：{_sanitize(chunk_id, 24)}")
            if len(ids) > _MAX_ID_SAMPLE:
                lines.append(f"  - {label}：另有 {len(ids) - _MAX_ID_SAMPLE} 个未列出")
    lines.append(
        "归属元数据：Chroma 侧带 owner 的文档 "
        f"{consistency.chroma_docs_with_owner}，带 visibility 的文档 "
        f"{consistency.chroma_docs_with_visibility}"
    )
    return "\n".join(lines)


def render_json(report: AuditReport) -> str:
    """机器可读输出。同样只打到终端，不写文件。"""
    return json.dumps(asdict(report), ensure_ascii=False, indent=2)


# ---------- CLI ----------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m agent.rag.audit",
        description="知识库只读审计：统计、隔离清单与存储一致性检查（不写入任何文件）",
    )
    parser.add_argument(
        "--persist-dir",
        default=str(PERSIST_DIR),
        help="RAG 持久化目录（默认 RAG_PERSIST_DIR）",
    )
    parser.add_argument("--json", action="store_true", help="输出 JSON（仍然只打到终端）")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    # 与 `python -m agent.rag.ingest` 保持一致：管道/重定向下也输出 UTF-8，
    # 否则中文清单在默认 GBK 的终端里会变成乱码
    for stream in (getattr(sys, "stdout", None), getattr(sys, "stderr", None)):
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    args = _build_parser().parse_args(argv)
    persist = Path(args.persist_dir)

    if not persist.exists():
        print(f"数据目录不存在：{persist}", file=sys.stderr)
        return 1

    report = build_report(persist)
    print(render_json(report) if args.json else render_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
