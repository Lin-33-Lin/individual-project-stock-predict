"""问答日志与反馈：每次问答落一行，点赞点踩回填，供第 4 章评估使用。

落盘为 data/logs/qa_log.csv（UTF-8），可以直接下载进 Excel 做二次分析。
注意：Streamlit Community Cloud 的磁盘是临时的，重启/重新部署会清空，
所以界面上同时提供"下载日志"按钮，重要数据请及时导出。
"""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import LOG_DIR

LOG_PATH = LOG_DIR / "qa_log.csv"

COLUMNS = [
    "log_id", "ts", "question", "answer", "mode", "refused",
    "latency", "best_score", "top_doc", "top_section", "feedback",
]


def _ensure() -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not LOG_PATH.exists():
        pd.DataFrame(columns=COLUMNS).to_csv(LOG_PATH, index=False, encoding="utf-8")
    return LOG_PATH


def append_log(
    question: str,
    answer: str,
    mode: str,
    refused: bool,
    latency: float,
    best_score: float,
    top_doc: str = "",
    top_section: str = "",
) -> str:
    path = _ensure()
    log_id = uuid.uuid4().hex[:10]
    row = pd.DataFrame(
        [[
            log_id,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            question,
            answer,
            mode,
            bool(refused),
            round(float(latency), 3),
            round(float(best_score), 3),
            top_doc,
            top_section,
            0,  # 0=未评价，1=有帮助，-1=没帮助
        ]],
        columns=COLUMNS,
    )
    row.to_csv(path, mode="a", header=False, index=False, encoding="utf-8")
    return log_id


def update_feedback(log_id: str, value: int) -> None:
    path = _ensure()
    df = pd.read_csv(path)
    if "feedback" not in df.columns:
        df["feedback"] = 0
    df.loc[df["log_id"] == log_id, "feedback"] = int(value)
    df.to_csv(path, index=False, encoding="utf-8")


def load_logs() -> pd.DataFrame:
    path = _ensure()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=COLUMNS)
    if df.empty:
        return df
    df["feedback"] = pd.to_numeric(df.get("feedback"), errors="coerce").fillna(0).astype(int)
    df["refused"] = df["refused"].astype(bool)
    df["latency"] = pd.to_numeric(df["latency"], errors="coerce")
    return df


def analyze_logs(df: pd.DataFrame) -> dict:
    """在线日志的运营指标：拒答率、反馈率、耗时分位数。"""
    if df is None or df.empty:
        return {
            "total": 0, "refusal_rate": 0.0, "up": 0, "down": 0,
            "rated": 0, "p50": 0.0, "p90": 0.0, "sat": 0.0, "by_doc": pd.DataFrame(),
        }

    total = len(df)
    lat = df["latency"].dropna()
    up = int((df["feedback"] == 1).sum())
    down = int((df["feedback"] == -1).sum())
    rated = up + down
    by_doc = (
        df[df["refused"] == False]
        .groupby("top_doc")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    # 满意度代理：点赞记 5 分、点踩记 1 分，折算成 5 点量表均值；无反馈则为 0
    satisfaction = ((up * 5.0 + down * 1.0) / rated) if rated else 0.0
    return {
        "total": total,
        "refusal_rate": float(df["refused"].mean()),
        "up": up,
        "down": down,
        "rated": rated,
        "approval_rate": (up / rated) if rated else 0.0,
        "satisfaction": satisfaction,
        "p50": float(lat.quantile(0.5)) if len(lat) else 0.0,
        "p90": float(lat.quantile(0.9)) if len(lat) else 0.0,
        "by_doc": by_doc,
    }
