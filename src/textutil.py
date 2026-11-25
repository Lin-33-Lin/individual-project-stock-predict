"""文本归一化：去噪、同义词归一、二元组切分。

轻量查询扩展就放在这里：不引入词典模型，只把校园场景里最常见的几组同义说法
统一成一个写法（"手续/流程""寝室/宿舍""挂失/遗失"），让检索和向量化看到同一套词。
"""

from __future__ import annotations

import re

_NOISE_RE = re.compile(r"[^\u4e00-\u9fa5a-zA-Z0-9]+")

# 只表达语气、不表达主题的高频二元组，参与重合度计算会虚抬分数
STOP_BIGRAMS = {
    "什么", "怎么", "怎样", "如何", "可以", "需要", "是否", "哪些", "多少",
    "时候", "请问", "一下", "应该", "我要", "我的", "是不", "不是", "有没",
    "没有", "这个", "那个", "如果", "以及", "还有", "能否", "为什", "么办",
}

# 同义归一：左边是学生的口语说法，右边是文档里的规范说法
SYNONYMS = (
    ("手续", "流程"),
    ("寝室", "宿舍"),
    ("挂失", "遗失"),
    ("补卡", "补办"),
    ("无法", "不能"),
    ("不可以", "不能"),
    ("多长时间", "多久"),
    ("收费标准", "多少钱"),
)


def canonical(text: str) -> str:
    text = str(text or "")
    for a, b in SYNONYMS:
        text = text.replace(a, b)
    return text


def clean(text: str) -> str:
    return _NOISE_RE.sub("", canonical(text))


def bigrams(text: str) -> set[str]:
    """字符二元组集合；单字文本退化为单字集合。"""
    text = clean(text)
    if not text:
        return set()
    if len(text) < 2:
        return {text}
    return {text[i : i + 2] for i in range(len(text) - 1)}


def query_terms(question: str) -> set[str]:
    """去掉"怎么/什么"这类语气二元组后的问题主题词。"""
    return {g for g in bigrams(question) if g not in STOP_BIGRAMS}
