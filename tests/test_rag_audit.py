"""只读审计工具测试。

这组测试要证明两件事，缺一不可：

1. **结论正确**：统计口径、unknown 隔离清单、重复来源、Chroma 与 chunks.json
   的差异都要如实报告；异常数据只报告、不修复；
2. **只读**：审计跑完之后，被审计目录里的文件一个都不能多、一个都不能变
   （大小、修改时间、内容哈希全部比对）——这是"只读工具"这句承诺的验收方式。

所以下面每个涉及落盘的用例都用 tmp_path 里的**假库**，绝不碰真实的 rag_store。
"""

import ast
import hashlib
import inspect
import json
import sqlite3
from pathlib import Path

import pytest

from agent.rag import audit

# 正文哨兵：任何输出里出现它，就说明工具把文档内容打印出来了
CANARY_TEXT = "机密正文-CANARY-不可外泄"


def write_chunks(persist: Path, items: list) -> Path:
    persist.mkdir(parents=True, exist_ok=True)
    path = persist / audit.CHUNKS_FILENAME
    path.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    return path


def make_chunk(doc_id: str, index: int, source: str = "手册.md", **meta) -> dict:
    return {
        "chunk_id": f"{doc_id}-{index}",
        "doc_id": doc_id,
        "source": source,
        "chunk_index": index,
        "text": f"{CANARY_TEXT} 第 {index} 段",
        "metadata": dict(meta),
    }


def write_chroma(persist: Path, entries: list) -> Path:
    """在 tmp_path 里伪造一个最小 chroma.sqlite3（只含审计需要的两张表）。"""
    directory = persist / audit.CHROMA_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)
    db_path = directory / audit.CHROMA_DB_FILENAME

    connection = sqlite3.connect(db_path)
    connection.execute(
        "CREATE TABLE embeddings ("
        "id INTEGER PRIMARY KEY, segment_id TEXT, embedding_id TEXT, "
        "seq_id BLOB, created_at TIMESTAMP)"
    )
    connection.execute(
        "CREATE TABLE embedding_metadata ("
        "id INTEGER, key TEXT, string_value TEXT, int_value INTEGER, "
        "float_value REAL, bool_value INTEGER)"
    )
    for row_id, (chunk_id, meta) in enumerate(entries, start=1):
        connection.execute(
            "INSERT INTO embeddings (id, embedding_id) VALUES (?, ?)", (row_id, chunk_id)
        )
        for key, value in meta.items():
            connection.execute(
                "INSERT INTO embedding_metadata (id, key, string_value) VALUES (?, ?, ?)",
                (row_id, key, str(value)),
            )
        # 正文单独存一行：审计绝不能把它读出来
        connection.execute(
            "INSERT INTO embedding_metadata (id, key, string_value) VALUES (?, ?, ?)",
            (row_id, "chroma:document", CANARY_TEXT),
        )
    connection.commit()
    connection.close()
    return db_path


def snapshot(root: Path) -> dict:
    """目录快照：相对路径 → (大小, mtime_ns, sha256)。用于证明只读。"""
    state = {}
    for path in sorted(root.rglob("*")):
        key = str(path.relative_to(root))
        if path.is_file():
            state[key] = (
                path.stat().st_size,
                path.stat().st_mtime_ns,
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        else:
            state[key + "/"] = "dir"
    return state


class TestLoadChunks:
    def test_missing_file_is_reported_not_created(self, tmp_path):
        persist = tmp_path / "rag_store"

        load = audit.load_chunks(persist / audit.CHUNKS_FILENAME)

        assert load.status == audit.JSON_STATUS_MISSING
        assert load.items == []
        assert not persist.exists()          # 不能顺手把目录建出来

    def test_empty_store_is_ok(self, tmp_path):
        persist = tmp_path / "rag_store"
        write_chunks(persist, [])

        load = audit.load_chunks(persist / audit.CHUNKS_FILENAME)

        assert load.status == audit.JSON_STATUS_OK
        assert load.items == []

    def test_corrupt_json_is_reported(self, tmp_path):
        persist = tmp_path / "rag_store"
        persist.mkdir()
        path = persist / audit.CHUNKS_FILENAME
        path.write_text("{ 这不是 JSON", encoding="utf-8")

        load = audit.load_chunks(path)

        assert load.status == audit.JSON_STATUS_CORRUPT
        assert "JSON 解析失败" in load.error
        assert path.read_text(encoding="utf-8") == "{ 这不是 JSON"   # 未被改写

    def test_non_list_top_level_is_reported(self, tmp_path):
        persist = tmp_path / "rag_store"
        persist.mkdir()
        path = persist / audit.CHUNKS_FILENAME
        path.write_text('{"chunks": []}', encoding="utf-8")

        assert audit.load_chunks(path).status == audit.JSON_STATUS_CORRUPT

    def test_non_dict_items_are_skipped_and_counted(self, tmp_path):
        persist = tmp_path / "rag_store"
        write_chunks(persist, [make_chunk("aaaa1111aaaa1111", 0), "垃圾", 42])

        load = audit.load_chunks(persist / audit.CHUNKS_FILENAME)

        assert load.status == audit.JSON_STATUS_OK
        assert len(load.items) == 1
        assert load.skipped == 2


class TestDocumentClassification:
    def test_public_private_and_unknown_are_counted(self, tmp_path):
        persist = tmp_path / "rag_store"
        write_chunks(persist, [
            make_chunk("aaaa1111aaaa1111", 0, "公共.md", visibility="public"),
            make_chunk("aaaa1111aaaa1111", 1, "公共.md", visibility="public"),
            make_chunk("bbbb2222bbbb2222", 0, "私人.md",
                       visibility="private", owner="6"),
            make_chunk("cccc3333cccc3333", 0, "无标记.md"),
        ])

        report = audit.build_report(persist)

        assert report.total_documents == 3
        assert report.total_chunks == 4
        assert report.public_documents == 1
        assert report.private_documents == 1
        assert report.unknown_documents == 1
        assert report.documents_with_owner == 1
        assert report.documents_with_visibility_field == 2
        assert report.documents_with_valid_visibility == 2
        assert report.owner_coverage == pytest.approx(1 / 3)
        assert report.visibility_field_coverage == pytest.approx(2 / 3)
        assert report.valid_visibility_coverage == pytest.approx(2 / 3)

    def test_legacy_owner_only_document_is_private(self, tmp_path):
        persist = tmp_path / "rag_store"
        write_chunks(persist, [
            make_chunk("bbbb2222bbbb2222", 0, "老文档.md"),
            make_chunk("bbbb2222bbbb2222", 1, "老文档.md", owner="9"),
        ])

        entry = audit.build_report(persist).documents[0]

        assert entry.status == "private"
        assert entry.owner == "9"
        assert entry.visibility_present is False
        assert entry.visibility_marker is None
        assert entry.visibility_invalid is False
        assert entry.needs_review is False
        assert "历史数据" in entry.note

    def test_unknown_document_is_flagged_for_manual_review(self, tmp_path):
        persist = tmp_path / "rag_store"
        write_chunks(persist, [make_chunk("cccc3333cccc3333", 0, "无标记.md")])

        report = audit.build_report(persist)
        entry = report.documents[0]

        assert entry.status == "unknown"
        assert entry.needs_review is True
        assert "待人工确认" in entry.note
        assert report.review_queue == [entry]
        assert entry.status != "public"       # 不得被推断成公共文档
        assert entry.owner is None            # 也不得被推断成某个人的私人文档

    @pytest.mark.parametrize("meta", [
        {"owner": "abc"},                          # owner 无法解析
        {"owner": 0},
        {"visibility": "shared"},                  # 无法识别的标记
        {"visibility": 123},
        {"visibility": "public", "owner": "6"},    # 互相矛盾
    ])
    def test_invalid_metadata_is_unknown_and_marked_invalid(self, tmp_path, meta):
        persist = tmp_path / "rag_store"
        write_chunks(persist, [make_chunk("dddd4444dddd4444", 0, "坏数据.md", **meta)])

        entry = audit.build_report(persist).documents[0]

        assert entry.status == "unknown"
        assert entry.metadata_invalid is True
        assert entry.needs_review is True
        assert "元数据非法或互相矛盾" in entry.note

    def test_conflicting_owners_are_unknown(self, tmp_path):
        persist = tmp_path / "rag_store"
        write_chunks(persist, [
            make_chunk("eeee5555eeee5555", 0, "冲突.md", owner="6"),
            make_chunk("eeee5555eeee5555", 1, "冲突.md", owner="7"),
        ])

        entry = audit.build_report(persist).documents[0]

        assert entry.status == "unknown"
        assert entry.needs_review is True

    def test_non_mapping_metadata_is_tolerated(self, tmp_path):
        persist = tmp_path / "rag_store"
        item = make_chunk("ffff6666ffff6666", 0, "怪元数据.md")
        item["metadata"] = "not-a-dict"
        write_chunks(persist, [item])

        entry = audit.build_report(persist).documents[0]

        assert entry.status == "unknown"
        assert entry.chunks == 1

    def test_chunk_without_valid_doc_id_is_counted_separately(self, tmp_path):
        persist = tmp_path / "rag_store"
        broken = make_chunk("aaaa1111aaaa1111", 0)
        broken.pop("doc_id")
        write_chunks(persist, [make_chunk("aaaa1111aaaa1111", 1), broken])

        report = audit.build_report(persist)

        assert report.total_chunks == 2
        assert report.total_documents == 1
        assert report.unattributed_chunks == 1

    def test_duplicate_sources_are_reported(self, tmp_path):
        persist = tmp_path / "rag_store"
        write_chunks(persist, [
            make_chunk("aaaa1111aaaa1111", 0, "同名.md", visibility="private", owner="6"),
            make_chunk("bbbb2222bbbb2222", 0, "同名.md", visibility="private", owner="7"),
            make_chunk("cccc3333cccc3333", 0, "独一份.md"),
        ])

        report = audit.build_report(persist)

        assert report.duplicate_sources == [
            ("同名.md", ["aaaa1111aaaa1111", "bbbb2222bbbb2222"])
        ]

    def test_single_document_is_not_a_duplicate(self, tmp_path):
        persist = tmp_path / "rag_store"
        write_chunks(persist, [
            make_chunk("aaaa1111aaaa1111", 0, "手册.md"),
            make_chunk("aaaa1111aaaa1111", 1, "手册.md"),
        ])

        assert audit.build_report(persist).duplicate_sources == []

    def test_empty_store_reports_zeroes(self, tmp_path):
        persist = tmp_path / "rag_store"
        write_chunks(persist, [])

        report = audit.build_report(persist)

        assert report.total_documents == 0
        assert report.owner_coverage == 0.0
        assert report.visibility_field_coverage == 0.0
        assert report.valid_visibility_coverage == 0.0
        assert audit.render_report(report)


class TestVisibilityCoverageSemantics:
    """两个口径必须分开：字段存在率（含写坏的值）≠ 合法标记覆盖率。"""

    def test_illegal_marker_counts_as_field_but_not_valid(self, tmp_path):
        persist = tmp_path / "rag_store"
        write_chunks(persist, [
            make_chunk("aaaa1111aaaa1111", 0, "坏标记.md", visibility="shared"),
        ])

        report = audit.build_report(persist)
        entry = report.documents[0]

        assert entry.status == "unknown"             # 不能被当成公共或私人
        assert entry.needs_review is True
        assert entry.visibility_present is True      # 字段写了
        assert entry.visibility_marker is None       # 但没有合法标记
        assert entry.visibility_invalid is True
        assert report.documents_with_visibility_field == 1
        assert report.documents_with_valid_visibility == 0
        assert report.visibility_field_coverage == 1.0
        assert report.valid_visibility_coverage == 0.0

    def test_non_string_marker_counts_as_field_but_not_valid(self, tmp_path):
        persist = tmp_path / "rag_store"
        write_chunks(persist, [
            make_chunk("aaaa1111aaaa1111", 0, "坏标记.md", visibility=123)
        ])

        entry = audit.build_report(persist).documents[0]

        assert entry.status == "unknown"
        assert entry.visibility_present is True
        assert entry.visibility_invalid is True

    def test_conflicting_markers_count_as_field_but_not_valid(self, tmp_path):
        persist = tmp_path / "rag_store"
        write_chunks(persist, [
            make_chunk("aaaa1111aaaa1111", 0, "矛盾.md", visibility="public"),
            make_chunk("aaaa1111aaaa1111", 1, "矛盾.md", visibility="private", owner="6"),
        ])

        report = audit.build_report(persist)
        entry = report.documents[0]

        assert entry.visibility_present is True
        assert entry.visibility_invalid is True
        assert report.documents_with_visibility_field == 1
        assert report.documents_with_valid_visibility == 0

    def test_valid_marker_counts_in_both(self, tmp_path):
        persist = tmp_path / "rag_store"
        write_chunks(persist, [
            make_chunk("aaaa1111aaaa1111", 0, "公共.md", visibility="public"),
            make_chunk("aaaa1111aaaa1111", 1, "公共.md", visibility="public"),
        ])

        report = audit.build_report(persist)
        entry = report.documents[0]

        assert entry.visibility_present is True
        assert entry.visibility_marker == "public"
        assert entry.visibility_invalid is False
        assert report.documents_with_visibility_field == 1
        assert report.documents_with_valid_visibility == 1

    def test_missing_field_counts_in_neither(self, tmp_path):
        persist = tmp_path / "rag_store"
        write_chunks(persist, [make_chunk("aaaa1111aaaa1111", 0, "没写.md", owner="6")])

        report = audit.build_report(persist)
        entry = report.documents[0]

        assert entry.status == "private"          # 有 owner，按历史数据判定
        assert entry.visibility_present is False
        assert entry.visibility_invalid is False
        assert report.documents_with_visibility_field == 0
        assert report.documents_with_valid_visibility == 0

    def test_blank_marker_is_treated_as_missing(self, tmp_path):
        # 空串沿用第一阶段的约定：等同于"没写"，而不是"写了非法值"
        persist = tmp_path / "rag_store"
        write_chunks(persist, [
            make_chunk("aaaa1111aaaa1111", 0, "空串.md", visibility="  ")
        ])

        entry = audit.build_report(persist).documents[0]

        assert entry.visibility_present is False
        assert entry.visibility_invalid is False

    def test_mixed_store_keeps_the_two_rates_distinct(self, tmp_path):
        persist = tmp_path / "rag_store"
        write_chunks(persist, [
            make_chunk("aaaa1111aaaa1111", 0, "合法.md", visibility="public"),
            make_chunk("bbbb2222bbbb2222", 0, "非法.md", visibility="shared"),
            make_chunk("cccc3333cccc3333", 0, "没写.md"),
        ])

        report = audit.build_report(persist)

        assert report.documents_with_visibility_field == 2
        assert report.documents_with_valid_visibility == 1
        assert report.visibility_field_coverage == pytest.approx(2 / 3)
        assert report.valid_visibility_coverage == pytest.approx(1 / 3)

    def test_summary_prints_both_rates(self, tmp_path):
        persist = tmp_path / "rag_store"
        write_chunks(persist, [
            make_chunk("aaaa1111aaaa1111", 0, "非法.md", visibility="shared"),
        ])

        text = audit.render_report(audit.build_report(persist))

        assert "visibility 字段 1/1（100.0%）" in text
        assert "合法标记 0/1（0.0%）" in text
        assert "(无效)" in text          # 写了但无效的值要一眼看得出来


class TestChromaReading:
    def test_reads_ids_and_ownership_keys(self, tmp_path):
        persist = tmp_path / "rag_store"
        write_chroma(persist, [
            ("aaaa1111aaaa1111-0", {}),
            ("bbbb2222bbbb2222-0", {"owner": "6"}),
            ("bbbb2222bbbb2222-1", {"owner": "6"}),
            ("cccc3333cccc3333-0", {"visibility": "public"}),
        ])

        found = audit.read_chroma(persist)

        assert found.available is True
        assert found.chunks == 4
        assert set(found.chunk_ids) == {
            "aaaa1111aaaa1111-0", "bbbb2222bbbb2222-0",
            "bbbb2222bbbb2222-1", "cccc3333cccc3333-0",
        }
        assert found.docs_with_owner == 1        # 两行 owner 属于同一篇文档
        assert found.docs_with_visibility == 1

    def test_missing_database_is_reported(self, tmp_path):
        found = audit.read_chroma(tmp_path / "rag_store")

        assert found.available is False
        assert found.chunks == 0
        assert "未找到" in found.error

    def test_corrupt_database_is_reported(self, tmp_path):
        persist = tmp_path / "rag_store"
        (persist / audit.CHROMA_DIRNAME).mkdir(parents=True)
        db_path = persist / audit.CHROMA_DIRNAME / audit.CHROMA_DB_FILENAME
        db_path.write_bytes(b"definitely not a sqlite file")

        found = audit.read_chroma(persist)

        assert found.available is False
        assert found.error is not None
        assert db_path.read_bytes() == b"definitely not a sqlite file"   # 未被改写

    def test_unexpected_schema_is_reported_not_guessed(self, tmp_path):
        persist = tmp_path / "rag_store"
        (persist / audit.CHROMA_DIRNAME).mkdir(parents=True)
        db_path = persist / audit.CHROMA_DIRNAME / audit.CHROMA_DB_FILENAME
        connection = sqlite3.connect(db_path)
        connection.execute("CREATE TABLE something_else (id TEXT)")
        connection.commit()
        connection.close()

        found = audit.read_chroma(persist)

        assert found.available is False
        assert "表结构" in found.error


class TestConsistency:
    def test_consistent_store_reports_no_difference(self, tmp_path):
        persist = tmp_path / "rag_store"
        chunks = [make_chunk("aaaa1111aaaa1111", 0), make_chunk("aaaa1111aaaa1111", 1)]
        write_chunks(persist, chunks)
        write_chroma(persist, [(c["chunk_id"], {}) for c in chunks])

        report = audit.build_report(persist)

        assert report.consistency.consistent is True
        assert report.consistency.only_in_chunks_json == []
        assert report.consistency.only_in_chroma == []

    def test_differences_are_reported_and_not_fixed(self, tmp_path):
        persist = tmp_path / "rag_store"
        write_chunks(persist, [
            make_chunk("aaaa1111aaaa1111", 0),
            make_chunk("bbbb2222bbbb2222", 0),
        ])
        write_chroma(persist, [
            ("aaaa1111aaaa1111-0", {}),
            ("cccc3333cccc3333-0", {}),
        ])
        before = snapshot(persist)

        report = audit.build_report(persist)

        assert report.consistency.consistent is False
        assert report.consistency.only_in_chunks_json == ["bbbb2222bbbb2222-0"]
        assert report.consistency.only_in_chroma == ["cccc3333cccc3333-0"]
        assert snapshot(persist) == before          # 只报告，不修复
        assert "不修复" in audit.render_report(report)

    def test_chroma_unavailable_skips_comparison(self, tmp_path):
        persist = tmp_path / "rag_store"
        write_chunks(persist, [make_chunk("aaaa1111aaaa1111", 0)])

        report = audit.build_report(persist)

        assert report.consistency.chroma_available is False
        assert report.consistency.only_in_chunks_json == []
        assert "无法比对" in audit.render_report(report)


class TestRendering:
    def test_manifest_carries_required_fields(self, tmp_path):
        persist = tmp_path / "rag_store"
        write_chunks(persist, [
            make_chunk("bbbb2222bbbb2222", 0, "私人.md", visibility="private", owner="6"),
            make_chunk("bbbb2222bbbb2222", 1, "私人.md", visibility="private", owner="6"),
        ])

        text = audit.render_report(audit.build_report(persist))

        assert "doc_id" in text and "owner" in text and "visibility" in text
        assert "来源" in text
        assert "bbbb2222bbbb2222" in text
        assert "私人.md" in text                  # 要求 3：清单必须带 source
        assert "private" in text
        assert " 2" in text                      # chunk 数

    def test_unknown_documents_are_listed_for_review(self, tmp_path):
        persist = tmp_path / "rag_store"
        write_chunks(persist, [
            make_chunk("cccc3333cccc3333", 0, "待认领.md"),
            make_chunk("dddd4444dddd4444", 0, "公共.md", visibility="public"),
        ])

        text = audit.render_report(audit.build_report(persist))

        assert "待人工确认：无 owner" in text            # 表内的 unknown 说明
        assert "待人工确认（unknown）：1 篇" in text      # 待认领清单标题
        tail = text.split("待人工确认（unknown）")[-1]
        assert "cccc3333cccc3333" in tail
        assert "dddd4444dddd4444" not in tail

    @pytest.mark.parametrize("renderer", ["report", "json"])
    def test_document_text_never_leaks(self, tmp_path, renderer):
        persist = tmp_path / "rag_store"
        write_chunks(persist, [make_chunk("aaaa1111aaaa1111", 0)])
        write_chroma(persist, [("aaaa1111aaaa1111-0", {})])

        report = audit.build_report(persist)
        output = (
            audit.render_report(report) if renderer == "report" else audit.render_json(report)
        )

        assert CANARY_TEXT not in output
        assert "chroma:document" not in output

    def test_control_characters_in_source_are_sanitized(self, tmp_path):
        persist = tmp_path / "rag_store"
        nasty = "手册\n伪造行\u001b[31m.md"
        write_chunks(persist, [make_chunk("aaaa1111aaaa1111", 0, nasty)])

        text = audit.render_report(audit.build_report(persist))

        assert "\u001b" not in text
        assert "\n伪造行" not in text

    def test_json_output_is_valid_json(self, tmp_path):
        persist = tmp_path / "rag_store"
        write_chunks(persist, [make_chunk("aaaa1111aaaa1111", 0, "手册.md")])

        payload = json.loads(audit.render_json(audit.build_report(persist)))

        assert payload["total_documents"] == 1
        assert payload["documents"][0]["doc_id"] == "aaaa1111aaaa1111"
        assert payload["documents"][0]["needs_review"] is True


class TestReadOnlyGuarantee:
    """审计必须不改变被审计对象——这一组就是这条承诺的验收。"""

    def test_full_audit_leaves_store_untouched(self, tmp_path):
        persist = tmp_path / "rag_store"
        chunks = [
            make_chunk("aaaa1111aaaa1111", 0, "公共.md", visibility="public"),
            make_chunk("bbbb2222bbbb2222", 0, "私人.md", visibility="private", owner="6"),
            make_chunk("cccc3333cccc3333", 0, "无标记.md"),
        ]
        write_chunks(persist, chunks)
        write_chroma(persist, [(c["chunk_id"], {}) for c in chunks])
        # 顺带放一个假的 bm25.pkl：确认工具只做存在性检查、不做反序列化
        (persist / audit.BM25_FILENAME).write_bytes(b"\x80\x04not-a-real-pickle")
        before = snapshot(persist)

        report = audit.build_report(persist)
        audit.render_report(report)
        audit.render_json(report)
        assert audit.main(["--persist-dir", str(persist)]) == 0
        assert audit.main(["--persist-dir", str(persist), "--json"]) == 0

        assert report.bm25_file_present is True
        assert snapshot(persist) == before

    def test_missing_directory_stays_missing(self, tmp_path):
        persist = tmp_path / "rag_store"

        report = audit.build_report(persist)

        assert report.total_documents == 0
        assert report.chunks_status == audit.JSON_STATUS_MISSING
        assert not persist.exists()
        assert audit.main(["--persist-dir", str(persist)]) == 1
        assert not persist.exists()

    def test_read_only_uri_blocks_writes(self, tmp_path):
        """审计用的正是这个 URI 形态：同样的写法必须写不进去。"""
        db_path = write_chroma(tmp_path / "rag_store", [("aaaa1111aaaa1111-0", {})])
        connection = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)

        try:
            with pytest.raises(sqlite3.OperationalError):
                connection.execute(
                    "INSERT INTO embeddings (id, embedding_id) VALUES (99, 'x-0')"
                )
        finally:
            connection.close()

    def test_module_never_imports_chroma_client(self):
        tree = ast.parse(Path(audit.__file__).read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])

        assert "chromadb" not in imported
        assert "mode=ro" in inspect.getsource(audit.read_chroma)

    def test_no_files_created_by_cli(self, tmp_path):
        persist = tmp_path / "rag_store"
        write_chunks(persist, [make_chunk("aaaa1111aaaa1111", 0)])
        before = {p.name for p in persist.iterdir()}

        audit.main(["--persist-dir", str(persist)])

        assert {p.name for p in persist.iterdir()} == before


class TestCli:
    def test_console_script_runs_and_prints_utf8(self, tmp_path):
        """按真实调用方式跑一遍（python -m），确认退出码与 UTF-8 输出。"""
        import subprocess
        import sys

        persist = tmp_path / "rag_store"
        write_chunks(persist, [
            make_chunk("aaaa1111aaaa1111", 0, "中文文档.md", visibility="public")
        ])

        proc = subprocess.run(
            [sys.executable, "-m", "agent.rag.audit", "--persist-dir", str(persist)],
            capture_output=True, text=True, encoding="utf-8", timeout=120,
        )

        assert proc.returncode == 0
        assert "知识库只读审计" in proc.stdout
        assert "中文文档.md" in proc.stdout

    def test_table_output_goes_to_stdout(self, tmp_path, capsys):
        persist = tmp_path / "rag_store"
        write_chunks(persist, [
            make_chunk("aaaa1111aaaa1111", 0, "公共.md", visibility="public")
        ])

        exit_code = audit.main(["--persist-dir", str(persist)])
        captured = capsys.readouterr()

        assert exit_code == 0
        assert "知识库只读审计" in captured.out
        assert CANARY_TEXT not in captured.out

    def test_json_flag_prints_json(self, tmp_path, capsys):
        persist = tmp_path / "rag_store"
        write_chunks(persist, [
            make_chunk("aaaa1111aaaa1111", 0, "公共.md", visibility="public")
        ])

        exit_code = audit.main(["--persist-dir", str(persist), "--json"])
        captured = capsys.readouterr()

        assert exit_code == 0
        assert json.loads(captured.out)["total_documents"] == 1
