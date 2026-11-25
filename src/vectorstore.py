"""向量索引：优先用 FAISS（IndexFlatIP），没装则退化为 NumPy 全量内积。

知识库规模在几百到几千片段量级，两种实现结果一致；
NumPy 兜底是为了让"没装 faiss 的机器也能一键跑通"。
"""

from __future__ import annotations

import numpy as np

try:  # faiss 为可选依赖
    import faiss  # type: ignore

    HAS_FAISS = True
except Exception:  # pragma: no cover
    faiss = None
    HAS_FAISS = False


class VectorIndex:
    def __init__(self, vectors: np.ndarray):
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self._index = None
        if HAS_FAISS and len(self.vectors):
            index = faiss.IndexFlatIP(self.vectors.shape[1])
            index.add(self.vectors)
            self._index = index

    @property
    def backend(self) -> str:
        return "faiss" if self._index is not None else "numpy"

    def __len__(self) -> int:
        return len(self.vectors)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> tuple[list[int], list[float]]:
        """返回 (片段下标, 余弦相似度)，按相似度降序。"""
        if not len(self.vectors):
            return [], []
        q = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        q /= (np.linalg.norm(q) or 1.0)
        k = int(min(top_k, len(self.vectors)))

        if self._index is not None:
            scores, ids = self._index.search(q, k)
            return ids[0].tolist(), scores[0].tolist()

        sims = (self.vectors @ q[0]).astype(np.float32)
        order = np.argsort(-sims)[:k]
        return order.tolist(), sims[order].tolist()
