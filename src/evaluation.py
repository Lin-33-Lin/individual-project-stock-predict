"""评估：用测试集批量跑问答，算出开题报告 4.2 节的指标。

自动判定说明（必须写清楚，不能假装是人工标注）：
- 准确性：把系统回答与参考答案做字符二元组覆盖率比较，超过阈值的判为"语义一致"，
  另外要求命中来源文档。这是近似判定，最终仍需按报告里的方案由两名评估者人工裁定。
- 幻觉：检查回答中出现的数字能否在引用片段里找到，找不到即标记为疑似幻觉。
  这是廉价可复算的代理指标，能抓"编造时间/金额"这类典型错误。
- 耗时：逐条真实计时，直接算 P50 / P90 / P95。
"""

from __future__ import annotations

import re

import pandas as pd

from .config import Settings
from .kb import KnowledgeBase
from .llm import BaseGenerator
from .rag import answer_question

SIM_THRESHOLD = 0.55      # 判定"与参考答案语义一致"的相似度阈值
_CITE_RE = re.compile(r"\[\d+\]")
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
_NOISE_RE = re.compile(r"[^\u4e00-\u9fa5a-zA-Z0-9]+")


def _normalize(text: str) -> str:
    return _NOISE_RE.sub("", str(text or ""))


def _grams(text: str) -> set[str]:
    text = _normalize(text)
    return {text[i : i + 2] for i in range(len(text) - 1)} or ({text} if text else set())


def similarity(answer: str, reference: str) -> float:
    """参考答案要点被回答覆盖的比例（0.7 权重）+ 回答本身的精炼度（0.3 权重）。

    纯编辑距离对长度差异太敏感：参考答案是浓缩版，系统回答是原文摘录，
    同一件事会被判成"不一致"。改用二元组覆盖率后，两段长短不一的文字
    只要覆盖了同样的要点就能拿到高分。
    """
    ga, gr = _grams(answer), _grams(reference)
    if not gr or not ga:
        return 0.0
    inter = len(ga & gr)
    return 0.7 * (inter / len(gr)) + 0.3 * (inter / len(ga))


def _numbers(text: str) -> set[str]:
    # 先去掉 [1][2] 这类引用编号，否则会被误当成"编造的数字"
    stripped = _CITE_RE.sub("", str(text or ""))
    return set(_NUM_RE.findall(_normalize(stripped)))


def hallucination_suspect(answer: str, citations: list[dict]) -> bool:
    """回答里出现了引用片段中不存在的数字 → 疑似幻觉（编时间/编金额）。"""
    if not citations:
        return False
    # 文档名（如 03_宿舍与生活.md）本身带数字，也要算进合法数字集合
    ctx = "".join(f"{c.get('doc', '')}{c.get('section', '')}{c.get('text', '')}" for c in citations)
    return bool(_numbers(answer) - _numbers(ctx))


def run_eval(
    kb: KnowledgeBase,
    settings: Settings,
    testset: pd.DataFrame,
    generator: BaseGenerator,
    progress=None,
) -> pd.DataFrame:
    rows = []
    total = len(testset)
    for i, item in enumerate(testset.itertuples(index=False), 1):
        question = str(getattr(item, "question", "")).strip()
        category = str(getattr(item, "category", "未分类") or "未分类")
        reference = str(getattr(item, "reference", "") or "")
        is_in_scope = str(getattr(item, "is_in_scope", "yes")).strip().lower() in ("yes", "1", "true", "是")
        source_doc = str(getattr(item, "source_doc", "") or "")

        ans = answer_question(question, kb, settings, generator=generator)
        sim = similarity(ans.text, reference) if (is_in_scope and reference) else 0.0
        doc_ok = True
        if is_in_scope and source_doc:
            doc_ok = any(source_doc in c["doc"] for c in ans.citations)

        correct = bool(is_in_scope and not ans.refused and sim >= SIM_THRESHOLD and doc_ok)
        rows.append(
            {
                "question": question,
                "category": category,
                "in_scope": is_in_scope,
                "reference": reference,
                "answer": ans.text,
                "mode": ans.mode,
                "refused": ans.refused,
                "latency": round(ans.latency, 3),
                "best_score": ans.best_score,
                "similarity": round(sim, 3),
                "doc_hit": doc_ok,
                "correct": correct,
                "refusal_correct": ans.refused if not is_in_scope else None,
                "hallucination": hallucination_suspect(ans.text, ans.citations),
                "top_doc": ans.citations[0]["doc"] if ans.citations else "",
            }
        )
        if progress:
            progress(i, total, question)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {}

    in_scope = df[df["in_scope"]]
    out_scope = df[~df["in_scope"]]
    answered = df[~df["refused"]]

    acc = float(in_scope["correct"].mean()) if len(in_scope) else 0.0
    refuse_acc = float(out_scope["refusal_correct"].mean()) if len(out_scope) else 0.0
    false_refusal = float(in_scope["refused"].mean()) if len(in_scope) else 0.0

    lat = df["latency"].dropna()
    hal = float(answered["hallucination"].mean()) if len(answered) else 0.0

    by_cat = (
        in_scope.groupby("category")
        .agg(total=("correct", "size"), correct=("correct", "sum"), latency=("latency", "median"))
        .reset_index()
    )
    by_cat["accuracy"] = by_cat["correct"] / by_cat["total"]

    return {
        "n_total": int(len(df)),
        "n_in_scope": int(len(in_scope)),
        "n_out_scope": int(len(out_scope)),
        "accuracy": acc,
        "refusal_accuracy": refuse_acc,
        "false_refusal": false_refusal,
        "hallucination": hal,
        "p50": float(lat.quantile(0.5)) if len(lat) else 0.0,
        "p90": float(lat.quantile(0.9)) if len(lat) else 0.0,
        "p95": float(lat.quantile(0.95)) if len(lat) else 0.0,
        "avg_latency": float(lat.mean()) if len(lat) else 0.0,
        "by_category": by_cat,
        "mode": df["mode"].iloc[0] if len(df) else "",
    }


TARGETS = {
    "问答准确率": 0.85,
    "拒答准确率": 0.80,
    "回答耗时P50": 10.0,   # 秒，越小越好
    "幻觉率": 0.10,        # 越小越好
    "用户满意度": 4.0,     # 5 分制
}


def radar_data(summary: dict, satisfaction: float | None = None) -> pd.DataFrame:
    """把五项指标折算成 0~1 的"达标度"，雷达图上目标线恒为 1。

    satisfaction 为 None 或 0 时（还没有用户反馈）不画满意度轴，避免用 0 冒充实测值。
    """

    def higher_better(value, target):
        return min(max(value / target, 0.0), 1.0) if target else 0.0

    def lower_better(value, target):
        if value <= 0:
            return 1.0
        return min(target / value, 1.0)

    items = [
        ("问答准确率", higher_better(summary.get("accuracy", 0), TARGETS["问答准确率"]), summary.get("accuracy", 0)),
        ("拒答准确率", higher_better(summary.get("refusal_accuracy", 0), TARGETS["拒答准确率"]), summary.get("refusal_accuracy", 0)),
        ("回答耗时", lower_better(summary.get("p50", 0), TARGETS["回答耗时P50"]), summary.get("p50", 0)),
        ("低幻觉", lower_better(summary.get("hallucination", 0), TARGETS["幻觉率"]), summary.get("hallucination", 0)),
    ]
    if satisfaction:
        items.append(
            ("用户满意度", higher_better(satisfaction, TARGETS["用户满意度"]), satisfaction)
        )
    return pd.DataFrame(items, columns=["指标", "达标度", "实测值"])
