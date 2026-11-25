"""全局配置：路径、检索参数、模型与密钥。

所有可调参数集中在这里，界面侧边栏调用的也是同一份 Settings，
保证"配置一处、问答与评估共用"，评估时跑的就是线上同一套参数。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------- 路径
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
KNOW_DIR = DATA_DIR / "knowledge"      # 知识库源文档
INDEX_DIR = DATA_DIR / "index"         # 向量索引缓存
LOG_DIR = DATA_DIR / "logs"            # 问答日志
EVAL_DIR = DATA_DIR / "eval"           # 测试集与评估结果


def ensure_dirs() -> None:
    for d in (DATA_DIR, KNOW_DIR, INDEX_DIR, LOG_DIR, EVAL_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- 密钥
def get_secret(key: str, default=None):
    """优先读 st.secrets（云端部署用 secrets.toml），其次环境变量。"""
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        # 本地没有 secrets.toml 时 st.secrets 会抛异常，属正常情况
        pass
    return os.environ.get(key, default)


# ---------------------------------------------------------------- 参数
@dataclass
class Settings:
    # —— 检索
    top_k: int = 5                 # 召回片段数
    score_threshold: float = 0.20   # 相关度阈值，低于此值进入拒答分支（tfidf 后端建议 0.18–0.22）
    cosine_weight: float = 0.65    # 相关度 = 向量余弦 × w + 词面重合 × (1-w)

    # —— 切片
    chunk_size: int = 420          # 单片段目标长度（字）
    chunk_overlap: int = 80        # 超长章节切分时的重叠字数

    # —— 向量化后端：tfidf（默认，零依赖零下载）| hf（本地句向量）| ollama
    embed_backend: str = "tfidf"
    hf_model: str = "BAAI/bge-small-zh-v1.5"

    # —— 生成后端：auto（自动择优）| ollama | api | extractive（检索直出）
    gen_backend: str = "auto"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"
    api_base: str = "https://api.deepseek.com/v1"
    api_key: str = ""
    api_model: str = "deepseek-chat"
    temperature: float = 0.2
    max_tokens: int = 400       # 回答要求不超 200 字，留足余量即可，越小越快

    # 影响索引内容的参数，变化时需要重建索引
    def index_signature(self) -> str:
        return f"{self.embed_backend}|{self.chunk_size}|{self.chunk_overlap}|{self.hf_model}"


def load_settings() -> Settings:
    """读取线上部署密钥，覆盖默认值。"""
    s = Settings()
    s.embed_backend = get_secret("EMBED_BACKEND", s.embed_backend)
    s.gen_backend = get_secret("GEN_BACKEND", s.gen_backend)
    s.ollama_url = get_secret("OLLAMA_URL", s.ollama_url)
    s.ollama_model = get_secret("OLLAMA_MODEL", s.ollama_model)
    s.api_base = get_secret("API_BASE", s.api_base)
    s.api_key = get_secret("API_KEY", s.api_key) or ""
    s.api_model = get_secret("API_MODEL", s.api_model)
    return s
