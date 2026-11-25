"""评估看板用的图表：柱状、直方、箱线、环形、雷达。

对应开题报告 4.3 节的可视化计划；统一用 Plotly，配色走学校的深蓝色系。
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

BLUE = "#1F3864"
LIGHT = "#8FAADC"
RED = "#C00000"
GREEN = "#2E7D32"
GRAY = "#7F7F7F"

_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(size=13),
    margin=dict(l=20, r=20, t=50, b=20),
)


def category_accuracy(by_cat: pd.DataFrame) -> go.Figure:
    """柱状图：各类别准确率，定位薄弱类别。"""
    fig = px.bar(
        by_cat,
        x="category",
        y="accuracy",
        text=by_cat["accuracy"].map(lambda v: f"{v:.0%}") if len(by_cat) else None,
        color="accuracy",
        color_continuous_scale=[[0, "#E8EEF8"], [1, BLUE]],
        labels={"category": "问题类别", "accuracy": "准确率"},
        height=340,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(**_THEME, coloraxis_showscale=False, yaxis=dict(tickformat=".0%", range=[0, 1.12]))
    fig.add_hline(y=0.85, line_dash="dash", line_color=RED,
                  annotation_text="目标 85%", annotation_position="right")
    return fig


def latency_hist(df: pd.DataFrame) -> go.Figure:
    """直方图：观察耗时分布与长尾。"""
    fig = px.histogram(df, x="latency", nbins=min(20, max(6, len(df) // 2)),
                       labels={"latency": "回答耗时（秒）"}, height=340, color_discrete_sequence=[LIGHT])
    fig.update_layout(**_THEME, yaxis_title="问题数", bargap=0.05)
    fig.add_vline(x=10, line_dash="dash", line_color=RED, annotation_text="P50 目标 10s")
    return fig


def latency_box(df: pd.DataFrame) -> go.Figure:
    """箱线图：按类别看耗时中位数与离散程度。"""
    fig = px.box(df, x="category", y="latency", color="category", height=320,
                 labels={"category": "问题类别", "latency": "耗时（秒）"},
                 color_discrete_sequence=[BLUE, LIGHT, "#4472C4", "#A5A5A5", "#D9E2F3"])
    fig.update_layout(**_THEME, showlegend=False)
    return fig


def outcome_donut(df: pd.DataFrame) -> go.Figure:
    """环形图：答对 / 答错 / 正确拒答 / 误拒 的构成。"""
    def classify(row):
        if not row["in_scope"]:
            return "正确拒答" if row["refusal_correct"] else "该拒未拒"
        return "答对" if row["correct"] else ("误拒" if row["refused"] else "答错")

    counts = df.apply(classify, axis=1).value_counts()
    colors = {
        "答对": GREEN, "正确拒答": BLUE, "答错": "#ED7D31", "该拒未拒": RED, "误拒": GRAY,
    }
    fig = go.Figure(
        go.Pie(
            labels=counts.index,
            values=counts.values,
            hole=0.45,
            marker=dict(colors=[colors.get(k, GRAY) for k in counts.index]),
            textinfo="label+percent",
        )
    )
    fig.update_layout(**_THEME, height=340, showlegend=False,
                      title=dict(text="评估结果构成", x=0.02, xanchor="left", font=dict(size=13)))
    return fig


def target_radar(radar: pd.DataFrame) -> go.Figure:
    """雷达图：五项指标达标度，外圈为 1.0（目标线）。"""
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=radar["达标度"], theta=radar["指标"], fill="toself",
        name="实测达标度", line_color=BLUE, fillcolor="rgba(31,56,100,0.25)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=[1.0] * len(radar), theta=radar["指标"], name="目标线",
        line=dict(color=RED, dash="dash"), fill=None,
    ))
    fig.update_layout(**_THEME, height=380, polar=dict(radialaxis=dict(range=[0, 1.05], showticklabels=False)),
                      legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center"))
    return fig


def feedback_pie(up: int, down: int) -> go.Figure:
    """在线反馈构成。"""
    fig = go.Figure(go.Pie(
        labels=["有帮助 👍", "没帮助 👎"], values=[up, down], hole=0.5,
        marker=dict(colors=[GREEN, RED]), textinfo="label+value",
    ))
    fig.update_layout(**_THEME, height=300, showlegend=False)
    return fig
