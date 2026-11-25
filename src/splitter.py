"""语义切片：按 Markdown 标题切出 FAQ 条目，超长章节再做带重叠的二次切分。

设计取向：一颗 FAQ 条目 = 一个检索单元。
校园问答的答案往往是"一问一答"式的，按标题切可以保证检索到的片段
本身就是一个完整答案，避免关键条款被 200/500 字的固定窗口切碎。
"""

from __future__ import annotations

import re

_HEADING_RE = re.compile(r"^(#{1,6})\s*(.+?)\s*$", re.M)
_SENTENCE_END = "。！？；\n"


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_long(text: str, size: int, overlap: int) -> list[str]:
    """把超长段落按句子边界切开，相邻片段保留 overlap 字重叠。"""
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []

    pieces, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # 尽量在句末断开
            cut = max(text.rfind(ch, start + size // 2, end) for ch in _SENTENCE_END)
            if cut > start:
                end = cut + 1
        pieces.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [p for p in pieces if p]


def split_document(doc_name: str, text: str, size: int = 420, overlap: int = 80) -> list[dict]:
    """切分单篇文档，返回片段列表：[{'doc','section','text'}, ...]"""
    text = _clean(text)
    if not text:
        return []

    # 文档标题（一级标题）本身不作为检索片段
    matches = list(_HEADING_RE.finditer(text))
    sections: list[tuple[str, str]] = []

    if not matches:
        sections.append((doc_name.rsplit(".", 1)[0], text))
    else:
        # 第一个标题之前若有正文，挂到"概述"
        head = text[: matches[0].start()].strip()
        if head:
            sections.append(("概述", head))
        for i, m in enumerate(matches):
            title = m.group(2).strip()
            body_start = m.end()
            body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[body_start:body_end].strip()
            level = len(m.group(1))
            if level == 1 and (not body or len(body) < 8):
                continue  # 一级标题当作文档名
            if body:
                sections.append((title, body))

    chunks: list[dict] = []
    for section, body in sections:
        # 去掉只有说明性质、不含知识的占位段落仍保留，因其可能含规则
        for piece in _split_long(body, size, overlap):
            chunks.append({"doc": doc_name, "section": section, "text": piece})
    return chunks


def split_corpus(docs: list[dict], size: int = 420, overlap: int = 80) -> list[dict]:
    all_chunks: list[dict] = []
    for doc in docs:
        for chunk in split_document(doc["name"], doc["text"], size, overlap):
            all_chunks.append(chunk)
    for idx, chunk in enumerate(all_chunks):
        chunk["id"] = idx
    return all_chunks
