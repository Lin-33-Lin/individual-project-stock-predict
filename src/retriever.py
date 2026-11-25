"""检索器：向量召回 + 词面重合校验 + 相关度门槛（拒答的第一道闸）。

为什么要在余弦相似度之外再加一层"词面重合"？
余弦只反映整体语义接近，而 FAQ 场景的典型误召回恰恰来自"句子很像但主题无关"
（如"我要申请什么材料"能同时匹配奖学金和证件补办）。加入问题主题词在片段中的
IDF 加权命中率后，主题不对的片段会被压低，拒答判定也更稳。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .config import Settings
from .embedder import BaseEmbedder
from .textutil import canonical, bigrams, query_terms
from .vectorstore import VectorIndex


@dataclass
class RetrievedChunk:
    doc: str
    section: str
    text: str
    cosine: float
    overlap: float
    score: float
    id: int = -1

    def as_citation(self) -> dict:
        return {
            "doc": self.doc,
            "section": self.section,
            "text": self.text,
            "score": round(self.score, 4),
            "cosine": round(self.cosine, 4),
            "overlap": round(self.overlap, 4),
        }


@dataclass
class RetrievalResult:
    hits: list[RetrievedChunk] = field(default_factory=list)
    best_score: float = 0.0
    relevant: bool = False   # False 表示应进入拒答分支

    @property
    def citations(self) -> list[dict]:
        return [h.as_citation() for h in self.hits]


class Retriever:
    def __init__(self, chunks: list[dict], vectors, embedder: BaseEmbedder, settings: Settings):
        self.chunks = chunks
        self.embedder = embedder
        self.settings = settings
        self.index = VectorIndex(vectors) if vectors is not None and len(vectors) else None
        self.idf = self._build_idf()

    def _build_idf(self) -> dict[str, float]:
        """按片段统计二元组的逆文档频率。

        重合度用 IDF 加权而不是简单计数：命中"奖学金"这种只在少数文档出现的词
        得分高，命中"么开""时候"这种到处都是的碎词几乎不得分。这样主题不对的
        问题（"股票怎么开户"里恰好有"怎么开"）不会被虚高的重合度抬过阈值。
        """
        n_docs = max(len(self.chunks), 1)
        df: dict[str, int] = {}
        for chunk in self.chunks:
            for gram in bigrams(chunk["section"] + chunk["text"]):
                df[gram] = df.get(gram, 0) + 1
        return {gram: math.log(1 + n_docs / (1 + count)) for gram, count in df.items()}

    def _weighted_overlap(self, terms: set[str], chunk: dict) -> float:
        if not terms:
            return 0.0
        chunk_terms = bigrams(chunk["section"] + chunk["text"])
        default_idf = self.idf.get("__default__", 3.0)
        total = sum(self.idf.get(g, default_idf) for g in terms)
        if total <= 0:
            return 0.0
        matched = sum(self.idf.get(g, default_idf) for g in terms & chunk_terms)
        return matched / total

    def search(
        self,
        question: str,
        top_k: int | None = None,
        threshold: float | None = None,
        cosine_weight: float | None = None,
    ) -> RetrievalResult:
        question = (question or "").strip()
        if not question or self.index is None or not len(self.index):
            return RetrievalResult()

        top_k = top_k or self.settings.top_k
        threshold = self.settings.score_threshold if threshold is None else threshold

        q_vec = self.embedder.transform([canonical(question)])
        ids, cosines = self.index.search(q_vec, top_k=max(top_k * 2, 10))

        terms = query_terms(question)
        w = self.settings.cosine_weight if cosine_weight is None else cosine_weight

        hits: list[RetrievedChunk] = []
        for idx, cos in zip(ids, cosines):
            chunk = self.chunks[idx]
            overlap = self._weighted_overlap(terms, chunk)
            score = w * float(cos) + (1.0 - w) * overlap
            hits.append(
                RetrievedChunk(
                    doc=chunk["doc"],
                    section=chunk["section"],
                    text=chunk["text"],
                    cosine=float(cos),
                    overlap=float(overlap),
                    score=float(score),
                    id=int(idx),
                )
            )

        hits.sort(key=lambda h: h.score, reverse=True)
        hits = hits[:top_k]

        best = hits[0].score if hits else 0.0
        return RetrievalResult(hits=hits, best_score=best, relevant=best >= threshold)
