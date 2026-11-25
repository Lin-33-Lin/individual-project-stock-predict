"""命令行自检：不进 Streamlit，直接把知识库、问答、评估跑一遍。

用法：
    python scripts/selftest.py            # 建索引 + 8 个样例问题 + 全量评估
    python scripts/selftest.py --query    # 只测问答
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.config import EVAL_DIR, Settings  # noqa: E402
from src.evaluation import run_eval, summarize  # noqa: E402
from src.kb import KnowledgeBase  # noqa: E402
from src.llm import ExtractiveGenerator  # noqa: E402
from src.logstore import load_logs, analyze_logs  # noqa: E402
from src.rag import answer_question  # noqa: E402

SAMPLES = [
    ("每学期选课什么时候开始？", True),
    ("国家奖学金需要什么条件？", True),
    ("学生证丢了怎么补办？", True),
    ("宿舍水电费怎么充值？", True),
    ("怎么申请转专业？", True),
    ("今天成都会下雨吗？", False),
    ("帮我推荐一部电影", False),
    ("股票怎么开户", False),
]


def main() -> None:
    settings = Settings()
    kb = KnowledgeBase(settings).load_or_build()
    stats = kb.stats()
    print(f"[知识库] 文档 {stats['docs']} 篇 / 片段 {stats['chunks']} 个 / "
          f"向量 {stats['embed_backend']} / 索引 {stats['index_backend']}")

    gen = ExtractiveGenerator()
    print("\n=== 问答抽样 ===")
    ok = 0
    for q, expect_answer in SAMPLES:
        ans = answer_question(q, kb, settings, generator=gen)
        hit = (not ans.refused) == expect_answer
        ok += int(hit)
        top = ans.citations[0]["section"] if ans.citations else "-"
        mark = "✓" if hit else "✗"
        print(f"{mark} [{ans.best_score:.3f}] {q}  →  {'拒答' if ans.refused else top}")
    print(f"拒答判定符合预期：{ok}/{len(SAMPLES)}")

    if "--query" in sys.argv:
        return

    testset_path = EVAL_DIR / "testset.csv"
    if not testset_path.exists():
        print("\n未找到 data/eval/testset.csv，跳过评估。")
        return

    print("\n=== 全量评估 ===")
    df = pd.read_csv(testset_path)
    result = run_eval(kb, settings, df, generator=gen)
    sm = summarize(result)
    print(f"准确率 {sm['accuracy']:.1%}（目标 85%）")
    print(f"拒答准确率 {sm['refusal_accuracy']:.1%}（目标 80%）")
    print(f"误拒率 {sm['false_refusal']:.1%}")
    print(f"耗时 P50 {sm['p50']:.2f}s / P90 {sm['p90']:.2f}s")
    print(f"疑似幻觉率 {sm['hallucination']:.1%}")
    print("\n分类别：")
    print(sm["by_category"].to_string(index=False))

    wrong = result[(result["in_scope"]) & (~result["correct"])]
    if len(wrong):
        print("\n未通过条目：")
        for r in wrong.itertuples():
            print(f" - {r.question}  sim={r.similarity:.2f} refused={r.refused} doc_hit={r.doc_hit}")

    logs = load_logs()
    if not logs.empty:
        la = analyze_logs(logs)
        print(f"\n在线日志 {la['total']} 条，拒答率 {la['refusal_rate']:.1%}，"
              f"P50 {la['p50']:.2f}s，反馈 {la['up']}👍/{la['down']}👎")


if __name__ == "__main__":
    main()
