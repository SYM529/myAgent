"""混合检索存储：向量检索 + BM25 关键词检索，用 RRF 融合。

为什么必须混合：纯向量检索对型号、缩写、代码标识符这类"字面匹配很关键"
的查询召回很差；纯关键词检索又抓不住语义相近但用词不同的情况。
两者互补，用 RRF（Reciprocal Rank Fusion）融合排名，不需要归一化分数。
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import re
import threading
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from agent.rag.chunker import Chunk
from agent.rag.config import PERSIST_DIR
from agent.rag.embeddings import EmbeddingClient
from agent.rag.visibility import DocAccess, classify_document, inspect_document

logger = logging.getLogger(__name__)

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")

try:  # 中文分词优先用 jieba，没有安装时降级为按字切分
    import jieba  # type: ignore

    def _tokenize(text: str) -> list[str]:
        lowered = text.lower()
        return [t for t in jieba.lcut(lowered) if t.strip()]
except ImportError:
    def _tokenize(text: str) -> list[str]:
        return _TOKEN_PATTERN.findall(text.lower())


@dataclass
class RetrievalHit:
    """一次召回结果，保留两路召回的排名便于分析。"""

    chunk: Chunk
    score: float             # RRF 融合分
    vector_rank: int | None = None
    bm25_rank: int | None = None


def _reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """RRF：各路召回只贡献 1/(k+rank)，避免不同量纲的分数直接相加。"""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return scores


@dataclass(frozen=True)
class ChromaDocumentState:
    """Chroma 侧某篇文档的既有状态：chunk id 集合 + 归属判定。

    id 集合必须单独带出来：只比较"归属判定"会漏掉"两侧 chunk 对不上"的情况
    （多一个孤儿 chunk、少一个 chunk 都可能不影响文档级判定）。
    """

    chunk_ids: frozenset[str]
    access: DocAccess


DELETE_STATUS_DELETED = "deleted"
DELETE_STATUS_ALREADY_ABSENT = "already_absent"
DELETE_STATUS_PARTIAL = "partial"


@dataclass(frozen=True)
class DeleteResult:
    """一次删除的结果：明确区分正常删除、本来就不存在、部分失败。

    - ``deleted``：向量库、内存、chunks.json 三处都已移除；
    - ``already_absent``：两边都没有这篇文档（幂等，什么都没改）；
    - ``partial``：向量库已删，但元数据（chunks.json）写入失败——
      内存、JSON、BM25 都保持原样，必须人工核对后再重试，绝不谎报成功。

    ``chroma_deleted`` 的含义是"**请求** Chroma 删除的 chunk 数"（取自内存索引里
    该文档的 chunk id 数）——Chroma 的 ``delete`` 不返回回执，所以它**不是**向量库
    确认删除的数量；``partial`` 时该字段仍会填这个请求数。
    ``chunks_removed`` 才是真正从内存与 chunks.json 移除的条数（partial 时为 0）。

    ``error`` 只带可运维识别的原因（OS 错误、临时文件名），不含文档正文或密钥。
    """

    doc_id: str
    status: str
    chroma_deleted: int
    chunks_removed: int
    json_saved: bool
    bm25_invalidated: bool
    error: str | None = None

    @property
    def ok(self) -> bool:
        """是否达到"这篇文档已不在知识库里"的终态（含幂等情形）。"""
        return self.status in (DELETE_STATUS_DELETED, DELETE_STATUS_ALREADY_ABSENT)

    @property
    def partial(self) -> bool:
        return self.status == DELETE_STATUS_PARTIAL


class HybridStore:
    """Chroma 负责向量检索，BM25 负责字面召回，对外暴露统一的 search。"""

    def __init__(
        self,
        persist_dir: Path = PERSIST_DIR,
        collection_name: str = "knowledge_base",
        embedding_client: EmbeddingClient | None = None,
    ):
        import chromadb
        from chromadb.config import Settings

        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embedding = embedding_client or EmbeddingClient()

        self._collection_name = collection_name
        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir / "chroma"),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        self._chunks_path = self.persist_dir / "chunks.json"
        self._bm25_path = self.persist_dir / "bm25.pkl"
        self._chunks: dict[str, Chunk] = self._load_chunks()
        self._bm25 = None
        # 写侧互斥：add / delete_document / clear 的"读当前状态 → 改 → 写盘 →
        # 成功后切换内存"必须整体串行，否则并发删除（或删除与上传并发）会基于
        # 同一份陈旧快照互相覆盖，出现"报成功但没删掉 / 新上传被覆盖"。
        # 用 RLock：_save_chunks 等内部方法可能被同一线程再次进入。
        # 只保护写路径；检索/列表等只读者不加锁（它们读到的是不可变快照对象）。
        self._lock = threading.RLock()

    # ---------- 持久化 ----------

    def _load_chunks(self) -> dict[str, Chunk]:
        if not self._chunks_path.exists():
            return {}
        try:
            raw = json.loads(self._chunks_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("chunks.json 损坏，按空库处理")
            return {}
        return {
            item["chunk_id"]: Chunk(
                chunk_id=item["chunk_id"],
                doc_id=item["doc_id"],
                source=item["source"],
                chunk_index=item["chunk_index"],
                text=item["text"],
                metadata=item.get("metadata", {}),
            )
            for item in raw
        }

    def _save_chunks(self, chunks: Mapping[str, Chunk] | None = None) -> None:
        """把 chunk 索引写成 chunks.json —— **原子替换**，不留半截文件。

        顺序：同目录唯一临时文件 → 完整写入 → ``flush`` + ``fsync`` →
        ``os.replace`` 替换正式文件。任何一步失败都会清理本次的临时文件、
        保留原文件，并抛出带上下文的错误。

        ``chunks`` 允许传入"即将生效的索引"：删除流程先用它把新内容落盘，
        写盘成功之后才切换内存状态——这样内存与 JSON 永远一致。

        保证边界（不要夸大）：
        - 保证的是"不会出现被截断或半写的 chunks.json"（进程崩溃 / 写入失败）；
        - ``os.replace`` 只在**同一文件系统内**具有原子性；
        - fsync 只覆盖该文件本身的落盘，不承诺目录项也已持久化，
          更不构成与 Chroma 之间的跨存储事务——那一层是删除一致性要另行处理的事。
        """
        source = self._chunks if chunks is None else chunks
        payload = [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "source": c.source,
                "chunk_index": c.chunk_index,
                "text": c.text,
                "metadata": c.metadata,
            }
            for c in source.values()
        ]
        try:
            text = json.dumps(payload, ensure_ascii=False, indent=2)
        except (TypeError, ValueError) as e:
            raise RuntimeError(f"chunks.json 序列化失败，原文件未改动：{e}") from e

        # 唯一名字：同目录并发写入、以及崩溃残留都不会互相覆盖
        temp_path = self._chunks_path.with_name(
            f"{self._chunks_path.name}.{os.getpid()}-{uuid.uuid4().hex}.tmp"
        )
        try:
            with open(temp_path, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())      # 数据先落盘，再替换
            os.replace(temp_path, self._chunks_path)
        except Exception as e:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError as cleanup_error:   # pragma: no cover - 清理失败只记录
                logger.warning("清理临时文件失败（%s）：%s", temp_path, cleanup_error)
            raise RuntimeError(
                "写入 chunks.json 失败，原文件保持不变"
                f"（临时文件 {temp_path.name}）：{e}"
            ) from e

    # ---------- BM25 ----------

    def _build_bm25(self):
        if not self._chunks:
            return None
        from rank_bm25 import BM25Okapi

        ordered = sorted(self._chunks.values(), key=lambda c: c.chunk_id)
        corpus = [_tokenize(c.text) for c in ordered]
        return BM25Okapi(corpus), [c.chunk_id for c in ordered]

    def _get_bm25(self):
        if self._bm25 is None:
            self._bm25 = self._build_bm25()
        return self._bm25

    def _invalidate_bm25(self) -> None:
        self._bm25 = None

    # ---------- 写入 ----------

    def _ensure_collection(self) -> None:
        """重新获取 collection 句柄。

        触发场景：运维在**另一个进程**执行 `ingest --rebuild` 清库后，
        本进程缓存的句柄立即失效，之后写入会报
        "Collection [...] does not exist"，而且会一直失败下去。

        重新拉一次句柄就能恢复。与其要求"重建后必须重启服务"，
        不如让写入自己站起来——热重建是常规操作，不该拖上重启。
        """
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _upsert(self, ids, documents, vectors, metadatas) -> None:
        """写入向量数据；集合句柄失效时自动恢复并重试一次。"""
        try:
            self._collection.upsert(
                ids=ids, documents=documents, embeddings=vectors, metadatas=metadatas
            )
        except Exception as e:
            # 只针对句柄失效做重试，其它错误原样抛出，避免掩盖真实问题
            if "does not exist" not in str(e):
                raise
            logger.warning("Chroma 集合句柄失效，重新获取后重试：%s", e)
            self._ensure_collection()
            self._collection.upsert(
                ids=ids, documents=documents, embeddings=vectors, metadatas=metadatas
            )

    def add(self, chunks: list[Chunk]) -> int:
        """写入 chunk，已存在的 chunk_id 会被覆盖。返回写入条数。

        两道闸门都在任何写入动作之前，任何一道不过就直接抛错、不落半点数据：

        1. **逐 chunk 完整性**（``_validate_batch``）：每个 chunk 都必须带明确
        合法的 visibility；public 文档不许声明 owner，private 文档必须每块都带
        同一个有效 owner；同一文档不许混用可见性。规则本身在
        ``agent.rag.visibility.inspect_document``，与检索、删除共用一套语义。
        2. **不许静默覆盖既有归属**（``_assert_no_conflicting_overwrite``）：
        写入前按 doc_id 比对 chunks.json 与 Chroma 的既有记录，
        两侧 chunk 集合必须完全相等，归属判定（visibility/owner）也必须一致；
        既有记录是 unknown、或新旧归属不同，一律拒绝。

        整批先校验、后落盘：批量导入里混进一篇坏文档时，不做"写一半"。

        整个"校验 → 比对既有记录 → 写向量库 → 写盘 → 切换内存"都在写锁内完成，
        与 ``delete_document`` 串行，避免并发下互相覆盖。
        """
        if not chunks:
            return 0

        with self._lock:
            return self._add_locked(chunks)

    def _add_locked(self, chunks: list[Chunk]) -> int:
        markers = self._validate_batch(chunks)
        self._assert_no_conflicting_overwrite(chunks, markers)

        vectors = self.embedding.embed([c.text for c in chunks])
        self._upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            vectors=vectors,
            metadatas=[
                {
                    "source": c.source,
                    "doc_id": c.doc_id,
                    "chunk_index": c.chunk_index,
                    **{k: v for k, v in c.metadata.items() if v},
                }
                for c in chunks
            ],
        )
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk

        self._save_chunks()
        self._invalidate_bm25()
        return len(chunks)

    @staticmethod
    def _validate_batch(chunks: list[Chunk]) -> dict[str, DocAccess]:
        """逐 chunk 校验整批数据，返回 {doc_id: 判定结果}；任何问题直接拒绝。"""
        per_document: dict[str, list[Chunk]] = {}
        for chunk in chunks:
            per_document.setdefault(chunk.doc_id, []).append(chunk)

        markers: dict[str, DocAccess] = {}
        problems: list[str] = []
        for doc_id, doc_chunks in sorted(per_document.items()):
            access, issues = inspect_document(
                doc_id, [c.metadata or {} for c in doc_chunks]
            )
            problems.extend(issues)
            if not issues:
                markers[doc_id] = access

        if problems:
            raise ValueError(
                "拒绝写入：元数据校验未通过\n  - " + "\n  - ".join(problems)
            )
        return markers

    def _read_chroma_document(self, doc_id: str) -> ChromaDocumentState | None:
        """读取 Chroma 中该 doc_id 的既有状态；**确认没有记录**时返回 None。

        三条硬性约定：

        1. 判断"有没有记录"只看 ``found["ids"]``，不看 metadata 是否为空——
        否则"有 id 但 metadata 丢了"会被误判成"没有记录"，从而放行覆盖。
        2. ids 非空却拿不到可用的 metadata（键缺失、为 None、条数不匹配、
        不是映射、空字典）时直接抛错拒绝：这类记录无法确认既有归属，
        不能过滤掉坏条目后继续分类，那等于把坏数据当成不存在。
        3. 查询异常同样抛错拒绝，不做任何兜底猜测。

        metadata 齐备但分类结果是 unknown 时**不在这里抛错**：判定结果照常返回，
        由调用方的 unknown 分支拒绝，这样错误信息能保留
        "历史文档需要专门迁移流程"的提示。

        只取 metadatas，不拉正文。
        """
        try:
            found = self._collection.get(
                where={"doc_id": doc_id}, include=["metadatas"]
            )
        except Exception as e:
            raise RuntimeError(
                f"无法读取 Chroma 既有记录（doc_id={doc_id}），"
                f"为避免覆盖归属已拒绝写入：{e}"
            ) from e

        if not isinstance(found, Mapping):
            raise RuntimeError(
                f"Chroma 返回结构异常（doc_id={doc_id}）："
                f"结果不是映射（{type(found).__name__}），无法确认既有归属，拒绝写入"
            )

        # 严格校验 ids：不做 str() 隐式转换——转换会把 None/123 之类的坏值
        # 伪装成看起来合法的 id，掩盖真正的结构问题
        raw_ids = found.get("ids")
        if not isinstance(raw_ids, (list, tuple)):
            raise RuntimeError(
                f"Chroma 返回结构异常（doc_id={doc_id}）：ids 缺失或不是列表"
                f"（ids={type(raw_ids).__name__}），无法确认既有归属，拒绝写入"
            )
        invalid_ids = [
            index
            for index, item in enumerate(raw_ids)
            if not isinstance(item, str) or not item
        ]
        if invalid_ids:
            raise RuntimeError(
                f"Chroma 返回结构异常（doc_id={doc_id}）：第 {invalid_ids} 个 id "
                "为空或不是字符串，无法确认既有归属，拒绝写入"
            )

        chunk_ids = list(raw_ids)
        if not chunk_ids:
            return None                       # 查询成功且确认没有既有记录

        metadatas = found.get("metadatas")
        if not isinstance(metadatas, (list, tuple)) or len(metadatas) != len(chunk_ids):
            raise RuntimeError(
                f"Chroma 既有记录（doc_id={doc_id}）有 {len(chunk_ids)} 个 chunk，"
                f"但 metadata 缺失或条数不匹配（metadatas={type(metadatas).__name__}），"
                "无法确认既有归属，拒绝写入"
            )

        broken = [
            index
            for index, meta in enumerate(metadatas)
            if not isinstance(meta, Mapping) or not meta
        ]
        if broken:
            raise RuntimeError(
                f"Chroma 既有记录（doc_id={doc_id}）的第 {broken} 个 chunk "
                "缺少可用 metadata（缺失、None 或空），无法确认既有归属，拒绝写入"
            )

        return ChromaDocumentState(
            chunk_ids=frozenset(chunk_ids),
            access=classify_document(metadatas),
        )

    def _assert_no_conflicting_overwrite(
        self, chunks: list[Chunk], markers: dict[str, DocAccess]
    ) -> None:
        """拒绝任何会改写既有归属的写入。

        - 只有一侧有记录（chunks.json 或 Chroma 缺另一侧）→ 拒绝，不选边；
        - 两侧 chunk id 集合必须完全相等 → 不等就拒绝（防止孤儿 chunk 或
          部分写入造成的隐性差异；只比归属判定会漏掉这类情况）；
        - 两侧判定出来的 (visibility, owner) 不一致 → 拒绝；
        - 既有记录是 unknown → 拒绝（不得把来源不明的历史文档改成 public）；
        - 新批次的 (visibility, owner) 与既有不同 → 拒绝（覆盖 private→public、
          换 owner、public→private 都在此列）；
        - 批次的 chunk 集合与既有不一致 → 拒绝（防止分块参数变化后，
          同 doc_id 下新旧 chunk 并存、元数据混杂）。

        file_id 刻意不参与比较：同一用户重复上传会产生新的 file_id，
        这是既有的幂等更新语义，不属于归属冲突。
        """
        per_document: dict[str, list[Chunk]] = {}
        for chunk in chunks:
            per_document.setdefault(chunk.doc_id, []).append(chunk)

        problems: list[str] = []
        for doc_id, doc_chunks in sorted(per_document.items()):
            existing = [c for c in self._chunks.values() if c.doc_id == doc_id]
            chroma_state = self._read_chroma_document(doc_id)

            if not existing and chroma_state is None:
                continue                      # 全新文档，没有可覆盖的归属
            if not existing:
                problems.append(
                    f"{doc_id}：Chroma 有记录但 chunks.json 没有（两侧不一致，拒绝写入）"
                )
                continue
            if chroma_state is None:
                problems.append(
                    f"{doc_id}：chunks.json 有记录但 Chroma 没有（两侧不一致，拒绝写入）"
                )
                continue

            json_ids = {c.chunk_id for c in existing}
            chroma_ids = set(chroma_state.chunk_ids)
            if json_ids != chroma_ids:
                problems.append(
                    f"{doc_id}：两侧 chunk 集合不一致——Chroma 有 {len(chroma_ids)} 块，"
                    f"chunks.json 有 {len(json_ids)} 块，拒绝写入以免留下孤儿 chunk"
                )
                continue

            json_access = classify_document([c.metadata or {} for c in existing])
            if (json_access.visibility, json_access.owner) != (
                chroma_state.access.visibility,
                chroma_state.access.owner,
            ):
                problems.append(
                    f"{doc_id}：两侧既有记录不一致——Chroma 是 "
                    f"{chroma_state.access.visibility}"
                    f"（owner={chroma_state.access.owner}），chunks.json 是 "
                    f"{json_access.visibility}"
                    f"（owner={json_access.owner}）（拒绝写入，请先排查数据）"
                )
                continue

            if json_access.is_unknown:
                problems.append(
                    f"{doc_id}：既有记录无法判定归属（unknown），拒绝写入；"
                    "历史文档需要专门的迁移流程，不能靠重新导入覆盖"
                )
                continue

            expected = markers[doc_id]
            if (json_access.visibility, json_access.owner) != (
                expected.visibility,
                expected.owner,
            ):
                problems.append(
                    f"{doc_id}：既有 {json_access.visibility}"
                    f"（owner={json_access.owner}）记录，本次是 "
                    f"{expected.visibility}（owner={expected.owner}），"
                    "不允许改写既有归属"
                )
                continue

            incoming_ids = {c.chunk_id for c in doc_chunks}
            if json_ids != incoming_ids:
                problems.append(
                    f"{doc_id}：分块结果与既有记录不一致（既有 {len(json_ids)} 块，"
                    f"本次 {len(incoming_ids)} 块），拒绝写入以免新旧 chunk 混杂"
                )

        if problems:
            raise ValueError(
                "拒绝写入：与既有文档冲突\n  - " + "\n  - ".join(problems)
            )

    def clear(self) -> None:
        """清空知识库（整库重建路径；同样走写锁，避免与并发删除/上传互相覆盖）。"""
        with self._lock:
            self._clear_locked()

    def _clear_locked(self) -> None:
        try:
            self._client.delete_collection(self._collection.name)
        except Exception as e:
            logger.warning("删除 collection 失败：%s", e)
        self._collection = self._client.get_or_create_collection(
            name=self._collection.name, metadata={"hnsw:space": "cosine"}
        )
        self._chunks = {}
        self._save_chunks()
        self._invalidate_bm25()

    def count(self) -> int:
        return len(self._chunks)

    def all_chunks(self) -> list[Chunk]:
        """返回全部 chunk，供"知识库里有哪些文档"的展示使用。"""
        return list(self._chunks.values())

    def get_document_chunks(self, doc_id: str) -> list[Chunk]:
        """返回某个文档的全部 chunk（按 chunk_index 排序），供前端点开预览。

        内容都在内存缓存里，不需要查向量库——预览是纯读操作，
        不应该有检索那种毫秒级以上的开销。
        """
        chunks = [c for c in self._chunks.values() if c.doc_id == doc_id]
        chunks.sort(key=lambda c: c.chunk_index)
        return chunks

    def delete_document(self, doc_id: str) -> DeleteResult:
        """删除某个文档的全部 chunk，返回结构化结果（见 ``DeleteResult``）。

        执行顺序（阶段 8B 起）：
        1. 收集该文档在**内存索引**里的 chunk id；
        2. ids 为空时查向量库：两边都没有 → ``already_absent``（幂等）；
           向量库仍有残留 → ``partial``（不自动猜着删，交人工核对）；
        3. 删向量库（句柄失效重试一次）；失败直接抛异常，内存/JSON 完全未动；
        4. 用**原子写**把"去掉该文档"的索引落到 chunks.json；
           失败 → ``partial``：向量库已删，但内存、JSON、BM25 一律保持原样；
        5. 只有写盘成功后才切换内存、失效 BM25。

        不变量：内存与 chunks.json 永远一致；BM25 只在成功切换后失效。
        向量库可能领先于元数据（``partial``），此时必须显式上报。

        整个"读内存 → 删向量库 → 写盘 → 切换内存"都在写锁内完成：
        否则两个并发删除各自基于陈旧快照写盘，后写者会把前一次删除覆盖掉
        （Chroma 已删、元数据把它们"复活"），也就是"报成功但没删掉"。
        """
        with self._lock:
            return self._delete_document_locked(doc_id)

    def _delete_document_locked(self, doc_id: str) -> DeleteResult:
        ids = [cid for cid, c in self._chunks.items() if c.doc_id == doc_id]
        if not ids:
            residual = self._read_chroma_document(doc_id)
            if residual is None:
                logger.info("文档 %s 不存在，删除请求按幂等处理", doc_id)
                return DeleteResult(
                    doc_id=doc_id, status=DELETE_STATUS_ALREADY_ABSENT,
                    chroma_deleted=0, chunks_removed=0,
                    json_saved=False, bm25_invalidated=False,
                )
            return DeleteResult(
                doc_id=doc_id, status=DELETE_STATUS_PARTIAL,
                chroma_deleted=0, chunks_removed=0,
                json_saved=False, bm25_invalidated=False,
                error="内存索引里没有该文档，但向量库仍有记录：两侧不一致，需人工核对",
            )

        try:
            self._collection.delete(ids=ids)
        except Exception as e:
            # 与写入同理：别的进程重建过库时句柄失效，重新获取后重试
            if "does not exist" not in str(e):
                raise
            logger.warning("Chroma 集合句柄失效，重新获取后重试删除：%s", e)
            self._ensure_collection()
            self._collection.delete(ids=ids)

        remaining = {cid: c for cid, c in self._chunks.items() if c.doc_id != doc_id}
        try:
            self._save_chunks(remaining)
        except Exception as e:
            # 向量库已删、元数据未更新：内存与 JSON 都保持旧内容（两者仍一致），
            # 但必须把"半删"如实报出去，绝不能报成功
            logger.error("删除 %s 时元数据写入失败：%s", doc_id, e)
            return DeleteResult(
                doc_id=doc_id, status=DELETE_STATUS_PARTIAL,
                chroma_deleted=len(ids), chunks_removed=0,
                json_saved=False, bm25_invalidated=False,
                error=f"元数据写入失败：{e}",
            )

        # 写盘成功后，才切换内存并让 BM25 失效
        self._chunks = remaining
        self._invalidate_bm25()
        logger.info("已删除文档 %s（%d 个 chunk）", doc_id, len(ids))
        return DeleteResult(
            doc_id=doc_id, status=DELETE_STATUS_DELETED,
            chroma_deleted=len(ids), chunks_removed=len(ids),
            json_saved=True, bm25_invalidated=True,
        )

    # ---------- 检索 ----------

    def search(
        self,
        query: str,
        top_k: int = 20,
        *,
        visible_doc_ids: frozenset[str],
    ) -> list[RetrievalHit]:
        """混合检索，返回按 RRF 融合分排序的结果。

        范围是**必填**的：没有 scope 就不存在"全库检索"这个选项——忘传会直接
        TypeError，而不是静默扩大范围（fail closed 优先于方便）。
        空集合、以及显式传入 ``None``，都会在 embedding / Chroma / BM25 / RRF
        之前直接返回空结果。

        不可见文档既不能进入 RRF 候选，也不能占用 top_k 配额：
        向量侧下推 ``where={"doc_id": {"$in": [...]}}``，BM25 侧先按可见
        chunk 过滤、再截断。
        """
        if not self._chunks:
            return []
        if not visible_doc_ids:
            # None 与空集合同义：没有任何可见文档 → 不查、不算、不排序
            return []

        allowed_chunk_ids = {
            chunk_id
            for chunk_id, chunk in self._chunks.items()
            if chunk.doc_id in visible_doc_ids
        }
        if not allowed_chunk_ids:
            return []

        vector_ranking: list[str] = []
        try:
            query_vector = self.embedding.embed_one(query)
            result = self._collection.query(
                query_embeddings=[query_vector],
                n_results=min(top_k, len(allowed_chunk_ids)),
                where={"doc_id": {"$in": sorted(visible_doc_ids)}},
            )
            # 双保险：向量库若返回范围外的 id（跨进程写入、脏数据），也不许进排名
            vector_ranking = [
                chunk_id
                for chunk_id in result.get("ids", [[]])[0]
                if chunk_id in allowed_chunk_ids
            ]
        except Exception as e:
            logger.warning("向量检索失败，仅使用 BM25：%s", e)

        bm25_ranking: list[str] = []
        bm25 = self._get_bm25()
        if bm25 is not None:
            index, ids = bm25
            scores = index.get_scores(_tokenize(query))
            ranked = sorted(zip(ids, scores), key=lambda x: x[1], reverse=True)
            # 注意 ranked 的元素是 (chunk_id, score) 元组，不是 chunk 对象。
            # 必须先按可见范围过滤、再截断：否则不可见文档会占掉 top_k 配额。
            bm25_ranking = [
                chunk_id for chunk_id, _ in ranked if chunk_id in allowed_chunk_ids
            ][:top_k]

        rankings = [r for r in (vector_ranking, bm25_ranking) if r]
        if not rankings:
            return []

        from agent.rag.config import RRF_K

        fused = _reciprocal_rank_fusion(rankings, k=RRF_K)
        vector_positions = {cid: i + 1 for i, cid in enumerate(vector_ranking)}
        bm25_positions = {cid: i + 1 for i, cid in enumerate(bm25_ranking)}

        ordered = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]
        hits = []
        for chunk_id, score in ordered:
            chunk = self._chunks.get(chunk_id)
            if chunk is None or chunk.doc_id not in visible_doc_ids:
                continue          # 兜底：已删除或范围外的 id 一律不进结果
            hits.append(RetrievalHit(
                chunk=chunk,
                score=score,
                vector_rank=vector_positions.get(chunk_id),
                bm25_rank=bm25_positions.get(chunk_id),
            ))
        return hits
