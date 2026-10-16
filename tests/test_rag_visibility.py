"""知识库可见性与授权判定测试。

这组测试守的是 RAG 多用户隔离的**规则源头**：列表、检索、预览、删除
四条路径最终都要调用这里的函数，规则一旦被改松，越权会从四条路径同时漏出去。
所以下面的用例要锁死三件事：

1. **判定优先级**：字段缺失、值非法、互相矛盾的元数据必须落到 unknown，
   绝不能回退成 public（默认共享是收不回来的错误）；
2. **身份边界**：无效用户 id 不放行；管理员对他人私人文档不放行；
   公共文档的删除仍然只有管理员能做；
3. **聚合语义**：归属写在任意一个 chunk 上都算数，但互相矛盾时必须整篇隔离。
"""

from collections.abc import Mapping

import pytest

from agent.rag.visibility import (
    VISIBILITY_PRIVATE,
    VISIBILITY_PUBLIC,
    VISIBILITY_UNKNOWN,
    DocAccess,
    can_delete,
    can_read,
    can_retrieve,
    classify_document,
    classify_metadata,
    inspect_document,
    normalize_owner,
    normalize_visibility,
    owner_declared,
)


class _ExplodingMapping(Mapping):
    """取值就炸的映射：模拟被写坏 / 被篡改的 metadata。"""

    def __getitem__(self, key):
        raise RuntimeError("boom")

    def __iter__(self):
        raise RuntimeError("boom")

    def __len__(self):
        return 0

    def get(self, key, default=None):
        raise RuntimeError("boom")


class _ExplodingBool:
    def __bool__(self):
        raise RuntimeError("boom")


class TestClassifyMetadata:
    def test_public_requires_explicit_flag(self):
        access = classify_metadata({"visibility": "public"})

        assert access.visibility == VISIBILITY_PUBLIC
        assert access.owner is None
        assert access.is_public

    def test_public_with_owner_is_unknown(self):
        # 规则 2：public 必须没有 owner；两者并存说明数据不可信
        assert classify_metadata(
            {"visibility": "public", "owner": "6"}
        ).visibility == VISIBILITY_UNKNOWN

    def test_explicit_private_keeps_owner(self):
        access = classify_metadata({"visibility": "private", "owner": "6"})

        assert access.visibility == VISIBILITY_PRIVATE
        assert access.owner == "6"

    def test_private_without_owner_is_unknown(self):
        # 规则 3：private 必须有 owner，否则无从判断归属
        assert classify_metadata(
            {"visibility": "private"}
        ).visibility == VISIBILITY_UNKNOWN

    def test_legacy_owner_only_is_private(self):
        # 规则 3：历史 Web 上传没有 visibility 标记，但 owner 明确
        access = classify_metadata({"owner": "9"})

        assert access.visibility == VISIBILITY_PRIVATE
        assert access.owner == "9"

    def test_legacy_int_owner_is_normalized(self):
        assert classify_metadata({"owner": 9}).owner == "9"
        assert classify_metadata({"owner": " 9 "}).owner == "9"
        assert classify_metadata({"owner": "009"}).owner == "9"

    def test_no_fields_is_unknown(self):
        # 规则 4：缺 visibility 也缺 owner，不得默认公开
        assert classify_metadata({}).visibility == VISIBILITY_UNKNOWN

    def test_blank_fields_are_treated_as_missing(self):
        assert classify_metadata(
            {"visibility": "", "owner": ""}
        ).visibility == VISIBILITY_UNKNOWN
        # 空串visibility 等于"没写"：只有 owner 时按历史数据走 private
        assert classify_metadata(
            {"visibility": "   ", "owner": "6"}
        ).visibility == VISIBILITY_PRIVATE

    def test_case_and_whitespace_are_tolerated(self):
        assert classify_metadata(
            {"visibility": " PUBLIC "}
        ).visibility == VISIBILITY_PUBLIC
        assert classify_metadata(
            {"visibility": "Private", "owner": 6}
        ).visibility == VISIBILITY_PRIVATE

    @pytest.mark.parametrize("value", ["shared", "team", "public-ish", "yes"])
    def test_unknown_visibility_value_quarantines(self, value):
        # 写了无法识别的标记 → 隔离，而不是当成"没写"
        assert classify_metadata(
            {"visibility": value, "owner": "6"}
        ).visibility == VISIBILITY_UNKNOWN

    @pytest.mark.parametrize("value", [1, True, False, 6.5, ["public"], {"v": "public"}])
    def test_non_string_visibility_quarantines(self, value):
        assert classify_metadata(
            {"visibility": value}
        ).visibility == VISIBILITY_UNKNOWN

    @pytest.mark.parametrize("value", [
        "", "   ", "abc", "6.0", "-3", "0", 0, -1, 3.5, True, False, [], {},
    ])
    def test_invalid_owner_quarantines(self, value):
        # 规则 9：非法用户 id 不放行，也不被忽略
        assert classify_metadata(
            {"owner": value}
        ).visibility == VISIBILITY_UNKNOWN
        assert classify_metadata(
            {"visibility": "private", "owner": value}
        ).visibility == VISIBILITY_UNKNOWN

    @pytest.mark.parametrize("meta", [
        None, "public", "owner=6", 42, 6.5, ["public"], ("owner", "6"), {1, 2},
    ])
    def test_non_mapping_metadata_is_unknown(self, meta):
        assert classify_metadata(meta).visibility == VISIBILITY_UNKNOWN

    def test_unreadable_mapping_is_unknown(self):
        # 规则 9：取字段就抛异常，也不能放行
        access = classify_metadata(_ExplodingMapping())

        assert access.visibility == VISIBILITY_UNKNOWN
        assert not can_read(access, user_id=6)


class TestClassifyDocument:
    def test_empty_document_is_unknown(self):
        assert classify_document([]).visibility == VISIBILITY_UNKNOWN

    def test_owner_on_any_chunk_counts(self):
        # 归属只写在后面的 chunk 上也不能丢，否则自己的文档会判成 unknown 删不掉
        access = classify_document([{}, {"owner": "9"}])

        assert access.visibility == VISIBILITY_PRIVATE
        assert access.owner == "9"

    def test_all_chunks_marked_private(self):
        chunks = [
            {"visibility": "private", "owner": "6"},
            {"visibility": "private", "owner": "6"},
        ]

        assert classify_document(chunks).owner == "6"

    def test_all_chunks_marked_public(self):
        chunks = [{"visibility": "public"}, {"visibility": "public"}]

        assert classify_document(chunks).visibility == VISIBILITY_PUBLIC

    def test_public_with_silent_chunk_stays_public(self):
        # 缺字段的 chunk 是"没说话"，不是"反对"：不能把标记好的公共文档降级
        assert classify_document(
            [{"visibility": "public"}, {}]
        ).visibility == VISIBILITY_PUBLIC

    def test_conflicting_owners_are_unknown(self):
        assert classify_document(
            [{"owner": "6"}, {"owner": "7"}]
        ).visibility == VISIBILITY_UNKNOWN

    def test_public_and_private_mixed_is_unknown(self):
        assert classify_document(
            [{"visibility": "public"}, {"visibility": "private", "owner": "6"}]
        ).visibility == VISIBILITY_UNKNOWN

    def test_public_with_owner_on_another_chunk_is_unknown(self):
        # 规则 2：owner 出现在别的 chunk 上也算 public 带 owner，同样隔离
        assert classify_document(
            [{"visibility": "public"}, {"owner": "6"}]
        ).visibility == VISIBILITY_UNKNOWN

    def test_one_broken_chunk_quarantines_whole_document(self):
        assert classify_document(
            [{"visibility": "private", "owner": "6"}, {"owner": "abc"}]
        ).visibility == VISIBILITY_UNKNOWN

    def test_non_mapping_chunk_quarantines_whole_document(self):
        assert classify_document(
            [{"visibility": "public"}, None]
        ).visibility == VISIBILITY_UNKNOWN

    def test_non_iterable_input_is_unknown(self):
        assert classify_document(42).visibility == VISIBILITY_UNKNOWN


class TestReadAuthorization:
    @pytest.mark.parametrize("access,user_id,is_admin,expected", [
        # public：所有已认证用户可读（规则 6）
        (DocAccess(VISIBILITY_PUBLIC), 1, False, True),
        (DocAccess(VISIBILITY_PUBLIC), 2, False, True),
        (DocAccess(VISIBILITY_PUBLIC), 9, True, True),
        # private：仅所属用户（规则 7）
        (DocAccess(VISIBILITY_PRIVATE, "6"), 6, False, True),
        (DocAccess(VISIBILITY_PRIVATE, "6"), 7, False, False),
        (DocAccess(VISIBILITY_PRIVATE, "6"), 6, True, True),
        (DocAccess(VISIBILITY_PRIVATE, "6"), 1, True, False),
        # unknown：任何人都不行，包括管理员（规则 8）
        (DocAccess(VISIBILITY_UNKNOWN), 6, False, False),
        (DocAccess(VISIBILITY_UNKNOWN), 6, True, False),
        (DocAccess(VISIBILITY_UNKNOWN), 1, True, False),
    ])
    def test_read_matrix(self, access, user_id, is_admin, expected):
        assert can_read(access, user_id=user_id, is_admin=is_admin) is expected

    @pytest.mark.parametrize("access,user_id,is_admin", [
        (DocAccess(VISIBILITY_PUBLIC), 6, False),
        (DocAccess(VISIBILITY_PRIVATE, "6"), 6, False),
        (DocAccess(VISIBILITY_PRIVATE, "6"), 1, True),
        (DocAccess(VISIBILITY_UNKNOWN), 6, True),
    ])
    def test_retrieve_matches_read(self, access, user_id, is_admin):
        # 检索与预览必须同权：引用里带着原文片段，能被召回就等于能被读到
        assert can_retrieve(
            access, user_id=user_id, is_admin=is_admin
        ) is can_read(access, user_id=user_id, is_admin=is_admin)

    @pytest.mark.parametrize("user_id", [None, "", "abc", 0, -1, 3.5, True, [], {}])
    def test_invalid_user_id_never_reads(self, user_id):
        # 规则 9：拿不到可信身份时，连公共文档也不放行
        for access in (
            DocAccess(VISIBILITY_PUBLIC),
            DocAccess(VISIBILITY_PRIVATE, "6"),
            DocAccess(VISIBILITY_UNKNOWN),
        ):
            assert can_read(access, user_id=user_id) is False

    @pytest.mark.parametrize("access", [
        DocAccess(VISIBILITY_PRIVATE),                 # private 却没 owner
        DocAccess(VISIBILITY_PUBLIC, "6"),             # public 却带 owner
        DocAccess("shared"),                           # 未知状态
        DocAccess(VISIBILITY_UNKNOWN, "6"),
        DocAccess(VISIBILITY_PRIVATE, "abc"),
        None,
        "public",
        {},
    ])
    def test_malformed_access_is_denied(self, access):
        # 手工构造的畸形判定结果不能被当成通行证
        assert can_read(access, user_id=6) is False
        assert can_read(access, user_id=1, is_admin=True) is False


class TestDeleteAuthorization:
    @pytest.mark.parametrize("access,user_id,is_admin,expected", [
        # public：沿用现有约定，仅管理员可删（规则 6）
        (DocAccess(VISIBILITY_PUBLIC), 1, False, False),
        (DocAccess(VISIBILITY_PUBLIC), 9, False, False),
        (DocAccess(VISIBILITY_PUBLIC), 1, True, True),
        # private：仅所属用户；管理员删他人的私人文档会被拒（规则 7）
        (DocAccess(VISIBILITY_PRIVATE, "6"), 6, False, True),
        (DocAccess(VISIBILITY_PRIVATE, "6"), 7, False, False),
        (DocAccess(VISIBILITY_PRIVATE, "6"), 1, True, False),
        (DocAccess(VISIBILITY_PRIVATE, "6"), 6, True, True),
        # unknown：来源不明先隔离，管理员也不能删（规则 8）
        (DocAccess(VISIBILITY_UNKNOWN), 6, False, False),
        (DocAccess(VISIBILITY_UNKNOWN), 1, True, False),
    ])
    def test_delete_matrix(self, access, user_id, is_admin, expected):
        assert can_delete(access, user_id=user_id, is_admin=is_admin) is expected

    @pytest.mark.parametrize("user_id", [None, "", "abc", 0, -1, 3.5, True, [], {}])
    def test_invalid_user_id_never_deletes(self, user_id):
        for access in (
            DocAccess(VISIBILITY_PUBLIC),
            DocAccess(VISIBILITY_PRIVATE, "6"),
            DocAccess(VISIBILITY_UNKNOWN),
        ):
            assert can_delete(access, user_id=user_id, is_admin=True) is False

    def test_broken_is_admin_flag_denies_public_delete(self):
        # is_admin 的取值本身抛异常时，不能因为异常而放行
        assert can_delete(
            DocAccess(VISIBILITY_PUBLIC), user_id=1, is_admin=_ExplodingBool()
        ) is False

    @pytest.mark.parametrize("access", [
        DocAccess(VISIBILITY_PRIVATE),
        DocAccess(VISIBILITY_PUBLIC, "6"),
        DocAccess("shared"),
        None,
        "public",
    ])
    def test_malformed_access_is_denied(self, access):
        assert can_delete(access, user_id=6) is False
        assert can_delete(access, user_id=6, is_admin=True) is False


class TestWritePathHelpers:
    """写路径用的公开入口：严格归一化 + 逐 chunk 检查。"""

    @pytest.mark.parametrize("raw,expected", [
        ("public", VISIBILITY_PUBLIC),
        (" PUBLIC ", VISIBILITY_PUBLIC),
        ("private", VISIBILITY_PRIVATE),
        ("Private", VISIBILITY_PRIVATE),
        (None, None),
        ("", None),
        ("   ", None),
        ("shared", None),
        (123, None),
        (True, None),
        ([], None),
    ])
    def test_normalize_visibility_is_strict(self, raw, expected):
        # 写路径只认明确的 public/private：缺失与非法都返回 None（由调用方拒绝）
        assert normalize_visibility(raw) == expected

    @pytest.mark.parametrize("raw,expected", [
        (6, "6"),
        ("6", "6"),
        (" 6 ", "6"),
        ("006", "6"),
        (None, None),
        ("", None),
        ("   ", None),
        (0, None),
        (-1, None),
        ("abc", None),
        (True, None),
    ])
    def test_normalize_owner(self, raw, expected):
        assert normalize_owner(raw) == expected

    @pytest.mark.parametrize("raw,expected", [
        ("6", True),
        (6, True),
        (0, True),        # 写了但非法：也算"声明了"，必须被拒绝而不是被忽略
        (True, True),
        ("abc", True),
        ("", False),      # 空串与空白 = 没写
        ("   ", False),
        (None, False),
    ])
    def test_owner_declared_treats_blank_as_missing(self, raw, expected):
        assert owner_declared(raw) is expected


class TestInspectDocument:
    """inspect_document 回答的是"每个 chunk 是否都完整一致"。"""

    DOC = "aaaa1111aaaa1111"

    def test_consistent_public_document_has_no_issue(self):
        access, issues = inspect_document(self.DOC, [
            {"visibility": "public"},
            {"visibility": "public"},
        ])

        assert issues == []
        assert access.is_public

    def test_consistent_private_document_has_no_issue(self):
        access, issues = inspect_document(self.DOC, [
            {"visibility": "private", "owner": "6"},
            {"visibility": "private", "owner": "6"},
        ])

        assert issues == []
        assert access.is_private and access.owner == "6"

    def test_silent_chunk_is_reported_even_if_classifiable(self):
        # 关键差异：classify_document 认为整篇可以判定为 public，
        # 但 inspect_document 指出第二个 chunk 根本没有声明
        metas = [{"visibility": "public"}, {}]

        assert classify_document(metas).is_public is True
        access, issues = inspect_document(self.DOC, metas)

        assert access.is_public is True
        assert len(issues) == 1 and "visibility 缺失或非法" in issues[0]

    def test_public_chunk_with_owner_is_reported(self):
        _, issues = inspect_document(self.DOC, [
            {"visibility": "public"},
            {"visibility": "public", "owner": "6"},
        ])

        assert any("不允许声明 owner" in issue for issue in issues)

    def test_private_chunk_without_owner_is_reported(self):
        _, issues = inspect_document(self.DOC, [
            {"visibility": "private", "owner": "6"},
            {"visibility": "private"},
        ])

        assert any("必须带有效 owner" in issue for issue in issues)

    def test_different_owners_are_reported(self):
        _, issues = inspect_document(self.DOC, [
            {"visibility": "private", "owner": "6"},
            {"visibility": "private", "owner": "7"},
        ])

        assert any("多个不同 owner" in issue for issue in issues)

    def test_mixed_visibility_is_reported(self):
        _, issues = inspect_document(self.DOC, [
            {"visibility": "public"},
            {"visibility": "private", "owner": "6"},
        ])

        assert any("混用多种可见性声明" in issue for issue in issues)

    @pytest.mark.parametrize("value", ["shared", 123, "", "  ", None])
    def test_unusable_visibility_is_reported(self, value):
        _, issues = inspect_document(self.DOC, [{"visibility": value}])

        assert any("visibility 缺失或非法" in issue for issue in issues)

    def test_non_mapping_metadata_is_reported(self):
        _, issues = inspect_document(self.DOC, [{"visibility": "public"}, "not-a-dict"])

        assert any("不是映射" in issue for issue in issues)

    @pytest.mark.parametrize("owner", ["", "   "])
    def test_blank_owner_is_undeclared_for_public(self, owner):
        _, issues = inspect_document(self.DOC, [{"visibility": "public", "owner": owner}])

        assert issues == []

    @pytest.mark.parametrize("owner", ["", "   "])
    def test_blank_owner_is_missing_for_private(self, owner):
        _, issues = inspect_document(self.DOC, [{"visibility": "private", "owner": owner}])

        assert any("必须带有效 owner" in issue for issue in issues)

    def test_illegal_owner_value_is_reported_not_ignored(self):
        # owner="abc" 不能被当成"没写"——写了坏值同样要拒绝
        _, issues = inspect_document(self.DOC, [{"visibility": "public", "owner": "abc"}])

        assert any("不允许声明 owner" in issue for issue in issues)


class TestClassifyToAuthorizationFlow:
    """端到端串一遍：先判定、再授权，避免两条规则各写一半。"""

    def test_web_upload_metadata_is_readable_only_by_owner(self):
        access = classify_document(
            [{"visibility": "private", "owner": "6", "file_id": "u6/a.txt"}] * 3
        )

        assert can_read(access, user_id=6) is True
        assert can_read(access, user_id=7) is False
        assert can_delete(access, user_id=6) is True
        assert can_delete(access, user_id=1, is_admin=True) is False

    def test_cli_metadata_is_public_but_admin_only_for_delete(self):
        access = classify_document([{"visibility": "public", "chars": 120}] * 2)

        assert can_read(access, user_id=7) is True
        assert can_delete(access, user_id=7) is False
        assert can_delete(access, user_id=1, is_admin=True) is True

    def test_unmarked_legacy_document_is_isolated(self):
        # 本地历史数据（无 owner、无 visibility）默认隔离，等人认领后再恢复
        access = classify_document([{"chars": 80}, {"heading": "概览"}])

        assert access.visibility == VISIBILITY_UNKNOWN
        assert can_read(access, user_id=6) is False
        assert can_read(access, user_id=1, is_admin=True) is False
        assert can_delete(access, user_id=1, is_admin=True) is False
