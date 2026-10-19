"""知识库入库命令行工具。

    python -m agent.rag.ingest --visibility public            # 增量入库 ./docs
    python -m agent.rag.ingest ./my-docs --visibility public  # 指定目录
    python -m agent.rag.ingest --rebuild --visibility public  # 先清空再入库
    python -m agent.rag.ingest --stats          # 只看状态
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from agent.rag import get_store, ingest_directory  # noqa: E402
from agent.rag.config import DOCS_DIR  # noqa: E402


def main():
    for stream in (getattr(sys, "stdout", None), getattr(sys, "stderr", None)):
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="把文档灌入知识库")
    parser.add_argument("directory", nargs="?", default=str(DOCS_DIR), help="文档目录")
    parser.add_argument("--rebuild", action="store_true", help="清空后重建")
    parser.add_argument("--stats", action="store_true", help="只查看当前状态")
    parser.add_argument(
        "--visibility",
        choices=("public",),
        default=None,
        help="导入后的可见性。CLI 导入即项目公共文档（所有登录用户可检索），"
             "因此必须显式指定 public，不做默认值",
    )
    args = parser.parse_args()

    if args.stats:
        store = get_store()
        print(f"知识库 chunk 数：{store.count()}")
        return

    if args.visibility is None:
        print(
            "错误：必须显式指定 --visibility public。\n"
            "CLI 导入会把文档标记为公共文档（所有登录用户可检索），"
            "所以不提供默认值，避免误把私有资料公开。",
            file=sys.stderr,
        )
        raise SystemExit(2)

    directory = Path(args.directory)
    print(
        f"开始入库：{directory}"
        f"（rebuild={args.rebuild}，visibility={args.visibility}）"
    )

    try:
        stats = ingest_directory(
            directory, rebuild=args.rebuild, visibility=args.visibility
        )
    except FileNotFoundError as e:
        # 目录路径写错是运维最常见的失误，给友好提示而不是甩 traceback
        print(f"错误：{e}")
        raise SystemExit(1)
    if stats["documents"] == 0:
        from agent.rag.loader import SUPPORTED_SUFFIXES

        print(
            f"目录里没有可入库的文档（支持 {'/'.join(sorted(SUPPORTED_SUFFIXES))}）：{directory}"
        )
        return

    print(f"\n文档数：{stats['documents']}　新增 chunk：{stats['chunks']}　"
          f"库内总数：{stats['total']}")
    print(f"可见性：{stats.get('visibility', '未知')}（项目公共文档）")
    print("来源：")
    for source in stats.get("sources", []):
        print(f"  - {source}")


if __name__ == "__main__":
    main()
