"""知识库：文档 → 切片 → 向量 → 索引，并把索引缓存到磁盘。

一份索引由三样东西组成，缺一不可、参数必须一致：
  vectors.npy    归一化向量矩阵
  meta.json      切片文本 + 文档清单 + 构建参数（用于判断缓存是否过期）
  embedder.joblib TF-IDF 词表等有状态信息
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import INDEX_DIR, KNOW_DIR, Settings, ensure_dirs
from .embedder import build_embedder
from .loader import read_document, load_documents
from .retriever import Retriever
from .splitter import split_corpus
from .textutil import canonical

META_NAME = "meta.json"
VEC_NAME = "vectors.npy"
EMB_NAME = "embedder.joblib"


def files_signature(know_dir: Path = KNOW_DIR) -> str:
    """用文件名+大小+修改时间判断知识库是否变化，变了就重建索引。"""
    items = []
    if know_dir.exists():
        for p in sorted(know_dir.iterdir()):
            if p.is_file():
                stat = p.stat()
                items.append(f"{p.name}:{stat.st_size}:{int(stat.st_mtime)}")
    return "|".join(items)


class KnowledgeBase:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.chunks: list[dict] = []
        self.vectors: np.ndarray | None = None
        self.docs: list[dict] = []
        self.updated_at: str = ""
        self.embedder = build_embedder(
            settings.embed_backend, settings.hf_model, settings.ollama_url, settings.ollama_model
        )
        self.retriever: Retriever | None = None
        self.freshly_built = False

    # ------------------------------------------------------------ 构建
    def params(self) -> dict:
        return {
            "signature": self.settings.index_signature(),
            "chunk_size": self.settings.chunk_size,
            "chunk_overlap": self.settings.chunk_overlap,
        }

    def load_or_build(self) -> "KnowledgeBase":
        ensure_dirs()
        if self._cache_valid():
            try:
                self._load()
                return self
            except Exception as exc:  # 缓存损坏则重建
                print(f"[warn] 索引缓存读取失败，重建：{exc}")
        self.build()
        return self

    def _cache_valid(self) -> bool:
        meta_path = INDEX_DIR / META_NAME
        if not meta_path.exists() or not (INDEX_DIR / VEC_NAME).exists():
            return False
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        return meta.get("params") == self.params() and meta.get("files_sig") == files_signature()

    def build(self) -> None:
        ensure_dirs()
        self.docs = load_documents(KNOW_DIR)
        self.chunks = split_corpus(
            self.docs, size=self.settings.chunk_size, overlap=self.settings.chunk_overlap
        )

        if self.chunks:
            # 建索引时把小节标题一并拼进去：学生的问法往往与标题高度接近
            # （"毕业需要修满多少学分" ↔ "毕业需要修多少学分？"），只索引正文会白白丢掉这条最强线索。
            # 同时做同义词归一（手续↔流程、寝室↔宿舍），让口语问法和文档用词对齐。
            texts = [canonical(f"{c['section']}\n{c['text']}") for c in self.chunks]
            self.vectors = self.embedder.fit_transform(texts)
        else:
            self.vectors = np.zeros((0, 1), dtype=np.float32)

        self.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
        self._save()
        self.retriever = Retriever(self.chunks, self.vectors, self.embedder, self.settings)
        self.freshly_built = True

    # ------------------------------------------------------------ 持久化
    def _save(self) -> None:
        np.save(INDEX_DIR / VEC_NAME, np.asarray(self.vectors, dtype=np.float32))
        self.embedder.save(INDEX_DIR / EMB_NAME)
        meta = {
            "params": self.params(),
            "files_sig": files_signature(),
            "updated_at": self.updated_at,
            "docs": [{"name": d["name"], "chars": d["chars"]} for d in self.docs],
            "chunks": self.chunks,
        }
        (INDEX_DIR / META_NAME).write_text(
            json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    def _load(self) -> None:
        meta = json.loads((INDEX_DIR / META_NAME).read_text(encoding="utf-8"))
        self.chunks = meta.get("chunks", [])
        self.docs = meta.get("docs", [])
        self.updated_at = meta.get("updated_at", "")
        self.vectors = np.load(INDEX_DIR / VEC_NAME)
        if (INDEX_DIR / EMB_NAME).exists():
            self.embedder.load(INDEX_DIR / EMB_NAME)
        self.retriever = Retriever(self.chunks, self.vectors, self.embedder, self.settings)

    # ------------------------------------------------------------ 管理
    def add_text(self, name: str, text: str, suffix: str = ".md") -> None:
        ensure_dirs()
        safe = Path(name).stem.replace("/", "_").replace("\\", "_")
        path = KNOW_DIR / f"{safe}{suffix}"
        path.write_text(text, encoding="utf-8")
        self.build()

    def add_file(self, filename: str, data: bytes) -> None:
        ensure_dirs()
        path = KNOW_DIR / Path(filename).name
        path.write_bytes(data)
        self.build()

    def delete_doc(self, name: str) -> None:
        path = KNOW_DIR / name
        if path.exists():
            path.unlink()
        self.build()

    # ------------------------------------------------------------ 状态
    def stats(self) -> dict:
        return {
            "docs": len(self.docs),
            "chunks": len(self.chunks),
            "chars": sum(d.get("chars", 0) for d in self.docs),
            "updated_at": self.updated_at,
            "embed_backend": self.embedder.name,
            "index_backend": self.retriever.index.backend if self.retriever and self.retriever.index else "空",
        }

    def doc_names(self) -> list[str]:
        return [d["name"] for d in self.docs]

    def search(
        self,
        question: str,
        top_k: int | None = None,
        threshold: float | None = None,
        cosine_weight: float | None = None,
    ):
        if self.retriever is None:
            from .retriever import RetrievalResult

            return RetrievalResult()
        return self.retriever.search(
            question, top_k=top_k, threshold=threshold, cosine_weight=cosine_weight
        )


def clear_index() -> None:
    for name in (META_NAME, VEC_NAME, EMB_NAME):
        path = INDEX_DIR / name
        if path.exists():
            path.unlink()
