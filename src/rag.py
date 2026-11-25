"""问答主链路：问题 → 检索 → 相关度闸门 → 生成 → 引用整理。

这就是开题报告 3.4 节的"核心问答流程"，四步一一对应；
LangChain 里的 RetrievalQA 做的也是同一件事，这里显式写出来，方便逐段解释与调试。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .config import Settings
from .kb import KnowledgeBase
from .llm import (
    BaseGenerator,
    LLMError,
    REFUSAL_TEXT,
    ExtractiveGenerator,
    looks_like_refusal,
    resolve_generator,
)


@dataclass
class Answer:
    question: str
    text: str
    citations: list[dict] = field(default_factory=list)
    refused: bool = False
    mode: str = "extractive"
    latency: float = 0.0
    best_score: float = 0.0
    warning: str = ""

    def to_log(self) -> dict:
        return {
            "question": self.question,
            "answer": self.text,
            "mode": self.mode,
            "refused": self.refused,
            "latency": round(self.latency, 3),
            "best_score": round(self.best_score, 3),
            "top_doc": self.citations[0]["doc"] if self.citations else "",
            "top_section": self.citations[0]["section"] if self.citations else "",
        }


def _hint_sections(hits) -> str:
    """拒答时给出知识库里相近的主题，帮用户换一种问法。"""
    sections = []
    for h in hits[:3]:
        if h.section not in sections:
            sections.append(h.section)
    if not sections:
        return ""
    return "\n\n知识库中可能相近的主题：" + "、".join(f"「{s}」" for s in sections)


def answer_question(
    question: str,
    kb: KnowledgeBase,
    settings: Settings,
    generator: BaseGenerator | None = None,
) -> Answer:
    t0 = time.perf_counter()
    question = (question or "").strip()

    if not question:
        return Answer(question="", text="请输入你的问题。", refused=True, mode="empty")

    retrieval = kb.search(
        question,
        top_k=settings.top_k,
        threshold=settings.score_threshold,
        cosine_weight=settings.cosine_weight,
    )

    # 闸门一：没有相关片段，直接拒答，不调用模型
    if not retrieval.relevant:
        return Answer(
            question=question,
            text=REFUSAL_TEXT + _hint_sections(retrieval.hits),
            citations=[],
            refused=True,
            mode="refusal",
            latency=time.perf_counter() - t0,
            best_score=retrieval.best_score,
        )

    gen = generator or resolve_generator(settings)
    mode, warning, text = gen.name, "", ""
    try:
        text = gen.generate(question, retrieval.hits)
    except LLMError as exc:
        # 闸门二：模型不可用 → 检索直出，绝不给用户一个报错页面
        mode = "extractive(fallback)"
        warning = f"{exc}；已自动降级为检索直出模式。"
        text = ExtractiveGenerator().generate(question, retrieval.hits)

    refused = looks_like_refusal(text)
    return Answer(
        question=question,
        text=text,
        citations=[] if refused else retrieval.citations,
        refused=refused,
        mode=mode,
        latency=time.perf_counter() - t0,
        best_score=retrieval.best_score,
        warning=warning,
    )
