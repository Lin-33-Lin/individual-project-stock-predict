"""向量化后端：一个接口，三种实现，随时可替换。

- tfidf  ：字符级 TF-IDF（默认）。零下载、零 GPU、可离线，校园 FAQ 这种
           "问法与原文用词接近"的场景足够用，云端免费实例也跑得动。
- hf     ：本地句向量模型（如 bge-small-zh），语义召回更好，需要装 torch。
- ollama ：调用本地 Ollama 的 /api/embeddings，与本地大模型共用一套运行时。

三者输出统一为 L2 归一化的矩阵，余弦相似度 = 内积，与 FAISS IndexFlatIP 对齐。
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class BaseEmbedder:
    name = "base"
    needs_fit = False

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def transform(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def save(self, path: Path) -> None:  # 无状态后端默认不落盘
        pass

    def load(self, path: Path) -> None:
        pass


class TfidfEmbedder(BaseEmbedder):
    """字符 2-3gram 的 TF-IDF。

    中文不需要分词器，字符 n-gram 天然覆盖"奖学金/助学金"这类近义搭配，
    且不存在词典外词问题，小语料上比词级 TF-IDF 更稳。
    """

    name = "tfidf"
    needs_fit = True

    def __init__(self):
        self._vec = None

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vec = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 3),
            min_df=1,
            sublinear_tf=True,
            norm="l2",
        )
        return _l2_normalize(self._vec.fit_transform(texts).toarray())

    def transform(self, texts: list[str]) -> np.ndarray:
        if self._vec is None:
            raise RuntimeError("TF-IDF 向量器尚未 fit，请先构建索引")
        return _l2_normalize(self._vec.transform(texts).toarray())

    def save(self, path: Path) -> None:
        joblib.dump(self._vec, path)

    def load(self, path: Path) -> None:
        self._vec = joblib.load(path)


class HuggingFaceEmbedder(BaseEmbedder):
    """本地句向量模型。首次使用会下载权重（约 100MB），不适合免费云实例。"""

    name = "hf"

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        self.model_name = model_name
        self._model = None

    def _ensure(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        model = self._ensure()
        return _l2_normalize(model.encode(texts, batch_size=32, show_progress_bar=False))

    def transform(self, texts: list[str]) -> np.ndarray:
        return self.fit_transform(texts)


class OllamaEmbedder(BaseEmbedder):
    """调用 Ollama 的 embedding 接口（如 bge-m3、nomic-embed-text）。"""

    name = "ollama"

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "bge-m3"):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def _encode(self, text: str) -> list[float]:
        import requests

        resp = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        vectors = [self._encode(t) for t in texts]
        return _l2_normalize(np.array(vectors, dtype=np.float32))

    def transform(self, texts: list[str]) -> np.ndarray:
        return self.fit_transform(texts)


def build_embedder(backend: str, hf_model: str, ollama_url: str, ollama_model: str) -> BaseEmbedder:
    backend = (backend or "tfidf").lower()
    if backend == "hf":
        return HuggingFaceEmbedder(hf_model)
    if backend == "ollama":
        return OllamaEmbedder(ollama_url, ollama_model)
    return TfidfEmbedder()
