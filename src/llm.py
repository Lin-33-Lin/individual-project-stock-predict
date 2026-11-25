"""生成后端：本地 Ollama / OpenAI 兼容 API / 检索直出（无模型兜底）。

同一份提示词喂给不同后端，回答口径保持一致；任何后端异常都退化为
"检索直出"，保证演示现场不会因为网络或模型没启动而白屏。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import requests

from .config import Settings
from .retriever import RetrievedChunk

REFUSAL_TEXT = (
    "这个问题超出了我的知识库范围，暂时无法回答。"
    "建议咨询学生事务中心、教务处等相关部门，以官网最新通知为准。"
)

SYSTEM_PROMPT = """你是“校园 FAQ 智能问答小助手”，服务对象是本校在读学生。必须遵守：
1. 只依据【参考资料】回答，不得使用参考资料以外的知识，不得推测、不得编造。
2. 每条关键信息后用方括号标注来源编号，例如 [1][2]，编号对应参考资料序号。
3. 若参考资料不足以回答，只输出这一句：{refusal}
4. 中文回答，先给结论再给要点，分点清晰，总字数不超过 200 字。
5. 涉及时间、金额、材料清单时，必须与参考资料原文一致，不得改写数字。""".format(
    refusal=REFUSAL_TEXT
)

_REFUSAL_MARKERS = ("超出我的知识库范围", "暂时无法回答", "暂不了解", "无法根据参考资料", "没有相关资料")


class LLMError(RuntimeError):
    pass


def _format_contexts(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, 1):
        blocks.append(f"[{i}] 《{c.doc}》· {c.section}\n{c.text}")
    return "\n\n".join(blocks)


def _build_messages(question: str, chunks: list[RetrievedChunk]) -> list[dict]:
    user = (
        f"【参考资料】\n{_format_contexts(chunks)}\n\n"
        f"【用户问题】\n{question}\n\n"
        "请依据上述参考资料作答，并标注来源编号。"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def looks_like_refusal(text: str) -> bool:
    return any(m in (text or "") for m in _REFUSAL_MARKERS)


# ------------------------------------------------------------------ 后端
class BaseGenerator:
    name = "base"

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> str:
        raise NotImplementedError


class ExtractiveGenerator(BaseGenerator):
    """检索直出：从最相关片段里挑句子拼成答案，天然有据、零幻觉。"""

    name = "extractive"

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> str:
        from .textutil import bigrams, query_terms

        if not chunks:
            return REFUSAL_TEXT
        terms = query_terms(question)
        best = chunks[0]

        sentences = [s.strip() for s in re.split(r"[。！？；\n]", best.text) if s.strip()]
        if not sentences:
            return REFUSAL_TEXT

        # 按与问题的词面重合度挑最相关的 2~3 句，保留原文顺序
        scored = []
        for idx, s in enumerate(sentences):
            hit = len(terms & bigrams(s))
            scored.append((hit, idx, s))
        scored.sort(key=lambda x: (-x[0], x[1]))
        keep_idx = sorted(i for _, i, _ in scored[:3])
        body = "。".join(sentences[i] for i in keep_idx) + "。"

        lines = [f"根据《{best.doc}》中「{best.section}」的说明：", "", f"{body} [1]"]

        extra = [c for c in chunks[1:] if c.score >= 0.6 * best.score][:2]
        if extra:
            lines.append("")
            lines.append("**相关条款**")
            for i, c in enumerate(extra, 2):
                snippet = c.text.split("。")[0][:80]
                lines.append(f"- 《{c.doc}》· {c.section}：{snippet}。 [{i}]")
        return "\n".join(lines)


class OllamaGenerator(BaseGenerator):
    name = "ollama"

    def __init__(self, settings: Settings, model: str | None = None):
        self.settings = settings
        self.model = model or settings.ollama_model

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> str:
        payload = {
            "model": self.model,
            "messages": _build_messages(question, chunks),
            "stream": False,
            "options": {
                "temperature": self.settings.temperature,
                "num_predict": self.settings.max_tokens,
            },
        }
        try:
            resp = requests.post(
                f"{self.settings.ollama_url.rstrip('/')}/api/chat", json=payload, timeout=180
            )
            resp.raise_for_status()
            content = resp.json()["message"]["content"].strip()
            # 推理类模型（deepseek-r1 等）会带回思考过程，展示前去掉
            content = re.sub(r"<think.*?</think>", "", content, flags=re.S).strip()
            return content
        except Exception as exc:
            raise LLMError(f"Ollama 调用失败（模型 {self.model}）：{exc}") from exc


class ApiGenerator(BaseGenerator):
    """任何 OpenAI 兼容接口（DeepSeek、Moonshot、通义、OpenAI…）。"""

    name = "api"

    def __init__(self, settings: Settings):
        self.settings = settings

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> str:
        payload = {
            "model": self.settings.api_model,
            "messages": _build_messages(question, chunks),
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
        }
        try:
            resp = requests.post(
                f"{self.settings.api_base.rstrip('/')}/chat/completions",
                json=payload,
                headers=headers,
                timeout=90,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            raise LLMError(f"API 调用失败：{exc}") from exc


# ------------------------------------------------------------------ 选择
def discover_ollama_models(url: str, timeout: float = 2.0) -> list[str]:
    """列出本地 Ollama 已安装的模型；没启动或连不上返回空列表。"""
    try:
        resp = requests.get(f"{url.rstrip('/')}/api/tags", timeout=timeout)
        if resp.status_code != 200:
            return []
        return [m.get("name", "") for m in resp.json().get("models", []) if m.get("name")]
    except Exception:
        return []


def ollama_available(url: str, model: str | None = None, timeout: float = 2.0) -> bool:
    models = discover_ollama_models(url, timeout)
    if not models:
        return False
    if not model:
        return True
    return model in models or any(m.split(":")[0] == model for m in models)


def resolve_generator(settings: Settings) -> BaseGenerator:
    backend = (settings.gen_backend or "auto").lower()
    if backend == "extractive":
        return ExtractiveGenerator()
    if backend == "api":
        return ApiGenerator(settings)
    if backend == "ollama":
        models = discover_ollama_models(settings.ollama_url)
        chosen = settings.ollama_model
        if models and chosen not in models:
            chosen = models[0]  # 配置的模型没装，就用已经装好的那一个
        return OllamaGenerator(settings, chosen)
    # auto：有 API Key 用 API；否则直接走检索直出（最快、最稳、零幻觉）。
    # 本地 Ollama 不放进 auto：消费级笔记本上 1.5B–3B 模型单次回答动辄 20–40 秒，
    # 且是否安装、装了哪个模型因机器而异，交给用户显式选择更可控。
    if settings.api_key:
        return ApiGenerator(settings)
    return ExtractiveGenerator()
