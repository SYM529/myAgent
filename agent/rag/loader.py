"""文档加载：把不同格式的文件解析成纯文本和元数据。

元数据（来源文件名、页码等）会一路带到最终引用，
这是"引用可核验"的基础——没有溯源就无法判断回答是否忠实于原文。
"""

from __future__ import annotations

import csv
import hashlib
import io
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

# 纯文本类：直接按文本读（含编码回退），无需额外依赖
TEXT_SUFFIXES = {
    ".md", ".markdown", ".txt", ".log",
    ".json", ".yaml", ".yml", ".ini", ".sql", ".py", ".js", ".ts",
}
# 结构化类：需要转成"可读文本"再入库，否则分块会把表格/标签切碎
STRUCTURED_SUFFIXES = {".csv", ".html", ".htm", ".docx", ".xlsx", ".pptx"}

SUPPORTED_SUFFIXES = TEXT_SUFFIXES | STRUCTURED_SUFFIXES | {".pdf"}

# 需要额外依赖才能解析的格式。缺失时给出可操作提示，
# 而不是抛一个让人摸不着头脑的 ImportError。
OPTIONAL_DEPS = {
    ".docx": "python-docx",
    ".xlsx": "openpyxl",
    ".pptx": "python-pptx",
}


@dataclass
class Document:
    """一份原始文档。"""

    doc_id: str
    source: str          # 相对路径，用于展示
    text: str
    metadata: dict = field(default_factory=dict)


def _read_text_file(path: Path) -> str:
    for encoding in ("utf-8", "gbk", "utf-8-sig"):
        try:
            return path.read_text(encoding=encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    # 兜底：忽略无法解码的字符，保证入库不中断
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for index, page in enumerate(reader.pages, 1):
        text = (page.extract_text() or "").strip()
        if text:
            # 保留页码标记，分块后可以定位到具体页
            pages.append(f"[第 {index} 页]\n{text}")
    return "\n\n".join(pages)


class _HTMLTextExtractor(HTMLParser):
    """抽取网页正文：丢弃 script/style，其余按块级标签换行。

    用标准库而不是 BeautifulSoup：这里只需要"拿到能读的文本"，
    为一个解析动作引入第三方依赖不划算。
    """

    _BLOCK_TAGS = {
        "p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6",
        "section", "article", "table", "ul", "ol", "pre",
    }
    _SKIP_TAGS = {"script", "style", "noscript"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
        elif tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self._SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        elif tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._skip_depth == 0 and data.strip():
            self.parts.append(data.strip() + " ")

    def text(self) -> str:
        # 连续空行压成一个，避免分块时出现大量空白 chunk
        return "\n".join(
            line.strip() for line in "".join(self.parts).splitlines() if line.strip()
        )


def _read_csv(path: Path) -> str:
    """CSV 转成 Markdown 表格——分块器对表格是原子处理的，
    转成表格格式才不会被从中间切成两半。"""
    text = _read_text_file(path)
    reader = csv.reader(io.StringIO(text))
    rows = [row for row in reader if any(cell.strip() for cell in row)]
    if not rows:
        return ""

    lines = []
    header = rows[0]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in rows[1:]:
        # 列数不齐时补齐，Markdown 表格缺列会导致渲染错乱
        padded = row + [""] * (len(header) - len(row))
        lines.append("| " + " | ".join(padded[: len(header)]) + " |")
    return "\n".join(lines)


def _read_html(path: Path) -> str:
    extractor = _HTMLTextExtractor()
    extractor.feed(_read_text_file(path))
    return extractor.text()


def _read_docx(path: Path) -> str:
    import docx  # python-docx

    document = docx.Document(str(path))
    lines = []
    for para in document.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style = (para.style.name or "").lower()
        # 标题样式还原成 Markdown 标题：分块器按标题切分，
        # 丢了标题层次就等于丢了检索时的结构信息
        if style.startswith("heading"):
            level = "".join(ch for ch in style if ch.isdigit()) or "2"
            lines.append(f"{'#' * min(int(level), 6)} {text}")
        else:
            lines.append(text)

    for table in document.tables:
        rows = []
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):
                rows.append("| " + " | ".join(cells) + " |")
        if rows:
            lines.append("")
            lines.append(rows[0])
            if len(rows) > 1:
                lines.append("| " + " | ".join(["---"] * (rows[0].count("|") - 1)) + " |")
                lines.extend(rows[1:])
    return "\n".join(lines)


def _read_xlsx(path: Path) -> str:
    from openpyxl import load_workbook

    workbook = load_workbook(filename=str(path), data_only=True, read_only=True)
    blocks = []
    for sheet in workbook.worksheets:
        rows = []
        for row in sheet.iter_rows(values_only=True):
            cells = ["" if cell is None else str(cell).strip() for cell in row]
            if any(cells):
                rows.append(cells)
        if not rows:
            continue
        width = max(len(r) for r in rows)
        blocks.append(f"## {sheet.title}")
        blocks.append("| " + " | ".join(rows[0] + [""] * (width - len(rows[0]))) + " |")
        blocks.append("| " + " | ".join(["---"] * width) + " |")
        for row in rows[1:]:
            blocks.append("| " + " | ".join(row + [""] * (width - len(row))) + " |")
    workbook.close()
    return "\n\n".join(blocks)


def _read_pptx(path: Path) -> str:
    from pptx import Presentation  # python-pptx

    presentation = Presentation(str(path))
    blocks = []
    for index, slide in enumerate(presentation.slides, 1):
        texts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                line = shape.text_frame.text.strip()
                if line:
                    texts.append(line)
        if texts:
            blocks.append(f"[第 {index} 页]\n" + "\n".join(texts))
    return "\n\n".join(blocks)


_READERS = {
    ".pdf": _read_pdf,
    ".csv": _read_csv,
    ".html": _read_html,
    ".htm": _read_html,
    ".docx": _read_docx,
    ".xlsx": _read_xlsx,
    ".pptx": _read_pptx,
}


def _read_any(path: Path, suffix: str) -> str:
    """按格式选择解析方式；缺依赖时给出可操作的安装提示。"""
    reader = _READERS.get(suffix)
    if reader is None:  # 纯文本类
        return _read_text_file(path)
    try:
        return reader(path)
    except ImportError as e:
        package = OPTIONAL_DEPS.get(suffix, "对应解析库")
        raise ValueError(
            f"解析 {suffix} 需要安装依赖：pip install {package}（{e}）"
        ) from None


def load_document(
    path: Path, root: Path | None = None, *, namespace: str | None = None
) -> Document:
    """加载单个文件。root 用于计算相对路径作为展示用的 source。

    namespace 把 doc_id 限制在命名空间内（Web 上传传 ``u{user_id}``）：
    不同用户上传同一份内容会得到不同的 doc_id，因而不会互相覆盖归属与源文件记录；
    同一用户重复上传同一份内容仍是同一个 doc_id，保持 upsert 幂等。
    不传 namespace 时哈希方式与历史完全一致——CLI 公共文档保持原 doc_id，
    重新导入 ./docs 不会凭空多出一份副本。
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"不支持的文件类型：{suffix}，支持 {sorted(SUPPORTED_SUFFIXES)}")

    text = _read_any(path, suffix).strip()
    if not text:
        raise ValueError(f"文件内容为空：{path}")

    # doc_id 用内容哈希：同一份文档重复入库不会产生重复 chunk
    if namespace:
        digest_source = f"{namespace}:{text}"
    else:
        digest_source = text
    doc_id = hashlib.sha1(digest_source.encode("utf-8")).hexdigest()[:16]
    source = str(path.relative_to(root)) if root and path.is_relative_to(root) else str(path)

    return Document(doc_id=doc_id, source=source, text=text,
                    metadata={"suffix": suffix, "chars": len(text)})


def load_directory(directory: Path) -> list[Document]:
    """递归加载目录下所有支持的文档，单文件失败不影响其余文件。"""
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"文档目录不存在：{directory}")

    documents = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        try:
            documents.append(load_document(path, root=directory))
        except Exception as e:
            print(f"[warn] 跳过 {path.name}：{type(e).__name__}: {e}")
    return documents
