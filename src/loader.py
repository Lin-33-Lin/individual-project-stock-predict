"""文档读取：把 TXT / Markdown / PDF / Word / CSV 统一转成纯文本。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown", ".pdf", ".docx", ".csv"}


def _read_txt(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    return "\n\n".join(p for p in pages if p)


def _read_docx(path: Path) -> str:
    from docx import Document

    doc = Document(str(path))
    lines = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    # 表格里的内容（如办事材料清单）也一起转成文本，用「 | 」分隔
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip().replace("\n", " ") for c in row.cells]
            if any(cells):
                lines.append(" | ".join(cells))
    return "\n\n".join(lines)


def _read_csv(path: Path) -> str:
    """FAQ 表格：每行转成一个 Markdown 小节，便于按条目检索。"""
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    q_col = next((c for c in df.columns if "问题" in c or c.lower() == "question"), None)
    a_col = next((c for c in df.columns if "答案" in c or c.lower() in ("answer", "reference")), None)
    if q_col is None or a_col is None:
        raise ValueError("CSV 需包含「问题」与「答案」两列")
    parts = []
    for _, row in df.iterrows():
        q, a = str(row[q_col]).strip(), str(row[a_col]).strip()
        if q and a and q.lower() != "nan":
            parts.append(f"## {q}\n\n{a}")
    return "\n\n".join(parts)


_READERS = {
    ".txt": _read_txt,
    ".md": _read_txt,
    ".markdown": _read_txt,
    ".pdf": _read_pdf,
    ".docx": _read_docx,
    ".csv": _read_csv,
}


def read_document(path: Path) -> str:
    suffix = path.suffix.lower()
    reader = _READERS.get(suffix)
    if reader is None:
        raise ValueError(f"暂不支持的文件类型：{suffix}")
    return reader(path).strip()


def load_documents(know_dir: Path) -> list[dict]:
    """读取知识库目录下所有受支持的文档。

    返回 [{'name': 文件名, 'text': 正文, 'chars': 字数}, ...]
    """
    docs: list[dict] = []
    for path in sorted(know_dir.iterdir()):
        if path.is_dir() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        try:
            text = read_document(path)
        except Exception as exc:  # 单个文件坏掉不应拖垮整个知识库
            print(f"[warn] 跳过 {path.name}: {exc}")
            continue
        if text:
            docs.append({"name": path.name, "text": text, "chars": len(text)})
    return docs
