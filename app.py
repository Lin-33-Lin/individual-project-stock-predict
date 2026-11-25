"""校园 FAQ 智能问答小助手 · Streamlit 主程序

运行：streamlit run app.py
四个页签对应开题报告的四大块：问答（3.1/3.4）、知识库（F1/第5章）、评估看板（第4章）、说明。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src import charts
from src.config import EVAL_DIR, KNOW_DIR, Settings, ensure_dirs, load_settings
from src.evaluation import SIM_THRESHOLD, radar_data, run_eval, summarize
from src.kb import KnowledgeBase, files_signature
from src.llm import ollama_available, resolve_generator
from src.logstore import LOG_PATH, analyze_logs, append_log, load_logs, update_feedback
from src.rag import answer_question

st.set_page_config(
    page_title="校园 FAQ 智能问答小助手",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

ensure_dirs()

EXAMPLES = [
    "每学期选课什么时候开始？",
    "国家奖学金申请需要什么条件？",
    "学生证丢了怎么补办？",
    "宿舍水电费怎么充值？",
]

GEN_LABELS = {
    "自动（有 Key 用 API，本地有 Ollama 用 Ollama）": "auto",
    "检索直出（不调用大模型，零成本零幻觉）": "extractive",
    "本地 Ollama": "ollama",
    "云端 API（OpenAI 兼容）": "api",
}
GEN_REVERSE = {v: k for k, v in GEN_LABELS.items()}


# ---------------------------------------------------------------- 会话状态
if "settings" not in st.session_state:
    st.session_state.settings = load_settings()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "kb_version" not in st.session_state:
    st.session_state.kb_version = 0
if "eval_df" not in st.session_state:
    st.session_state.eval_df = None
if "eval_sum" not in st.session_state:
    st.session_state.eval_sum = None

S: Settings = st.session_state.settings


# ---------------------------------------------------------------- 资源加载
@st.cache_resource(show_spinner=False)
def _load_kb_cached(
    key_sig: str,
    embed_backend: str,
    chunk_size: int,
    chunk_overlap: int,
    hf_model: str,
    ollama_url: str,
    ollama_model: str,
) -> KnowledgeBase:
    settings = Settings(
        embed_backend=embed_backend,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        hf_model=hf_model,
        ollama_url=ollama_url,
        ollama_model=ollama_model,
    )
    kb = KnowledgeBase(settings)
    kb.load_or_build()
    return kb


def get_kb() -> KnowledgeBase:
    key_sig = f"{files_signature()}|{S.index_signature()}|{st.session_state.kb_version}"
    return _load_kb_cached(
        key_sig,
        S.embed_backend,
        S.chunk_size,
        S.chunk_overlap,
        S.hf_model,
        S.ollama_url,
        S.ollama_model,
    )


@st.cache_resource(show_spinner=False)
def _load_generator_cached(gen_backend: str, api_key: str, api_base: str, api_model: str,
                           ollama_url: str, ollama_model: str, temperature: float):
    settings = Settings(
        gen_backend=gen_backend,
        api_key=api_key,
        api_base=api_base,
        api_model=api_model,
        ollama_url=ollama_url,
        ollama_model=ollama_model,
        temperature=temperature,
    )
    return resolve_generator(settings)


def get_generator():
    return _load_generator_cached(
        S.gen_backend, S.api_key, S.api_base, S.api_model,
        S.ollama_url, S.ollama_model, S.temperature,
    )


def refresh_index(note: str = "") -> None:
    st.session_state.kb_version += 1
    _load_kb_cached.clear()
    if note:
        st.toast(note)


# ---------------------------------------------------------------- 侧边栏
with st.sidebar:
    st.title("🎓 校园 FAQ 助手")
    st.caption("基于 RAG · 回答带出处 · 未知即拒答")

    with st.expander("⚙️ 检索与生成参数", expanded=True):
        S.gen_backend = GEN_LABELS[
            st.selectbox(
                "生成后端", list(GEN_LABELS.keys()),
                index=list(GEN_LABELS.values()).index(S.gen_backend) if S.gen_backend in GEN_REVERSE else 0,
            )
        ]
        S.top_k = st.slider("召回片段数 top-k", 2, 10, int(S.top_k))
        S.score_threshold = st.slider(
            "相关度阈值（低于则拒答）", 0.05, 0.80, float(S.score_threshold), 0.01,
            help="分数 = 语义余弦×权重 + 问题关键词命中率×(1-权重)；调高更保守、拒答更多。",
        )
        S.cosine_weight = st.slider("语义权重（余弦占比）", 0.20, 0.95, float(S.cosine_weight), 0.05)

    gen = get_generator()
    if gen.name == "extractive":
        st.info("当前为**检索直出**模式：不调用大模型，答案直接摘自知识库原文。")
    elif gen.name == "ollama":
        st.success(f"本地 Ollama 已连接 · {getattr(gen, 'model', S.ollama_model)}")
    else:
        st.success(f"云端 API 模式 · {S.api_model}")
    st.caption("「自动」= 有 API Key 用 API，否则检索直出；本地大模型请在下方显式选择。")

    kb = get_kb()
    stats = kb.stats()
    st.divider()
    st.markdown("**知识库状态**")
    c1, c2 = st.columns(2)
    c1.metric("文档", stats["docs"])
    c2.metric("片段", stats["chunks"])
    st.caption(
        f"向量后端 {stats['embed_backend']} · 索引 {stats['index_backend']}  \n"
        f"更新于 {stats['updated_at'] or '—'}"
    )
    if st.button("♻️ 重建索引", width="stretch"):
        refresh_index("索引已重建")
        st.rerun()

    st.divider()
    with st.expander("🔌 接口设置（可选）"):
        S.api_base = st.text_input("API Base", S.api_base)
        S.api_model = st.text_input("模型名", S.api_model)
        S.api_key = st.text_input("API Key", S.api_key, type="password")
        S.ollama_url = st.text_input("Ollama 地址", S.ollama_url)
        S.ollama_model = st.text_input("Ollama 模型", S.ollama_model)
        if st.button("测试 Ollama 连接", width="stretch"):
            ok = ollama_available(S.ollama_url)
            st.write("✅ 可连接" if ok else "❌ 连接失败（本地未启动 Ollama 时属正常）")

    st.caption("⚠️ 示例知识库仅为演示数据，请替换为本校公开文件。")


# ---------------------------------------------------------------- 问答交互
def _feedback(msg: dict, value: int) -> None:
    try:
        update_feedback(msg["log_id"], value)
        msg["feedback"] = value
    except Exception as exc:
        st.toast(f"反馈写入失败：{exc}")


def render_assistant(msg: dict) -> None:
    st.markdown(msg["answer"])
    if msg.get("warning"):
        st.warning(msg["warning"])

    cites = msg.get("citations") or []
    if cites:
        with st.expander(f"📎 来源引用（{len(cites)} 条）"):
            for i, c in enumerate(cites, 1):
                st.markdown(f"**[{i}] 《{c['doc']}》 · {c['section']}**")
                st.progress(
                    max(0.0, min(float(c["score"]), 1.0)),
                    text=f"相关度 {c['score']:.2f}（余弦 {c['cosine']:.2f} / 词面 {c['overlap']:.2f}）",
                )
                text = c["text"]
                st.caption(text if len(text) <= 260 else text[:260] + "…")

    st.caption(
        f"模式 {msg['mode']} · 耗时 {msg['latency']:.2f}s · "
        f"最高相关度 {msg['best_score']:.2f}"
        + (" · 已拒答" if msg["refused"] else "")
    )

    fb = msg.get("feedback", 0)
    if msg.get("log_id") == "none":
        return
    if fb == 0:
        col1, col2, _ = st.columns([1, 1, 5])
        if col1.button("👍 有帮助", key=f"up_{msg['log_id']}"):
            _feedback(msg, 1)
            st.rerun()
        if col2.button("👎 没帮助", key=f"down_{msg['log_id']}"):
            _feedback(msg, -1)
            st.rerun()
    else:
        st.caption("已记录反馈：" + ("👍 有帮助" if fb == 1 else "👎 没帮助"))


def ask(question: str) -> None:
    question = (question or "").strip()
    if not question:
        return
    st.session_state.messages.append({"role": "user", "content": question})

    if not kb.chunks:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "answer": "知识库还是空的：请先到「📚 知识库管理」上传校园文档，再回来提问。",
                "citations": [], "mode": "empty", "latency": 0.0,
                "best_score": 0.0, "refused": True, "warning": "", "log_id": "none",
                "feedback": 0,
            }
        )
        return

    with st.spinner("检索知识库并生成回答…"):
        ans = answer_question(question, kb, S, generator=get_generator())

    top = ans.citations[0] if ans.citations else {}
    log_id = append_log(
        question=question,
        answer=ans.text,
        mode=ans.mode,
        refused=ans.refused,
        latency=ans.latency,
        best_score=ans.best_score,
        top_doc=top.get("doc", ""),
        top_section=top.get("section", ""),
    )
    st.session_state.messages.append(
        {
            "role": "assistant",
            "answer": ans.text,
            "citations": ans.citations,
            "mode": ans.mode,
            "latency": ans.latency,
            "best_score": ans.best_score,
            "refused": ans.refused,
            "warning": ans.warning,
            "log_id": log_id,
            "feedback": 0,
        }
    )


# ---------------------------------------------------------------- 主界面
tab_chat, tab_kb, tab_eval, tab_about = st.tabs(
    ["💬 智能问答", "📚 知识库管理", "📊 评估看板", "ℹ️ 项目说明"]
)

# ---- 页签 1：问答
with tab_chat:
    st.subheader("有问题，直接问")
    st.caption("回答只依据本校公开文档；知识库没覆盖的问题会明确拒答，而不是编一个答案。")

    if not st.session_state.messages:
        st.markdown("**可以试试这些问题：**")
        cols = st.columns(2)
        for i, example in enumerate(EXAMPLES):
            if cols[i % 2].button(example, key=f"ex_{i}", width="stretch"):
                ask(example)
                st.rerun()
        st.divider()

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant", avatar="🎓"):
                render_assistant(msg)

    typed = st.chat_input("请输入你的校园事务问题…")
    if typed:
        ask(typed)
        st.rerun()

# ---- 页签 2：知识库
with tab_kb:
    st.subheader("知识库管理")
    st.caption("把学校的公开文件放进来，系统会自动完成解析、切片与向量化。")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("文档数", stats["docs"])
    c2.metric("片段数", stats["chunks"])
    c3.metric("总字数", f"{stats['chars']:,}")
    c4.metric("索引类型", stats["index_backend"])

    st.divider()
    left, right = st.columns([1.1, 1])

    with left:
        st.markdown("**① 上传文档**")
        files = st.file_uploader(
            "支持 TXT / Markdown / PDF / Word / CSV（FAQ 表格需含「问题」「答案」两列）",
            type=["txt", "md", "markdown", "pdf", "docx", "csv"],
            accept_multiple_files=True,
        )
        if files and st.button("⬆️ 导入并重建索引", type="primary", width="stretch"):
            for f in files:
                (KNOW_DIR / Path(f.name).name).write_bytes(f.getbuffer())
            refresh_index(f"已导入 {len(files)} 个文档")
            st.rerun()

    with right:
        st.markdown("**② 直接粘贴一段文本**")
        doc_title = st.text_input("文档标题", placeholder="例：图书馆借阅规则")
        doc_text = st.text_area("正文（建议用「## 问题」作小节标题）", height=150,
                                placeholder="## 借书期限\n本科生借阅期限为 30 天，可续借 1 次。")
        if st.button("保存为新文档", width="stretch"):
            if doc_title.strip() and doc_text.strip():
                kb.add_text(doc_title.strip(), doc_text.strip())
                refresh_index("已新增文档")
                st.rerun()
            else:
                st.warning("标题和正文都不能为空。")

    st.divider()
    st.markdown("**③ 现有文档**")
    chunk_counts = pd.Series([c["doc"] for c in kb.chunks]).value_counts() if kb.chunks else pd.Series(dtype=int)
    if not kb.docs:
        st.info("知识库为空，先上传或粘贴一份文档吧。")
    for doc in kb.docs:
        row = st.columns([4, 1, 1])
        row[0].write(f"📄 {doc['name']}")
        row[1].caption(f"{int(chunk_counts.get(doc['name'], 0))} 个片段 · {doc['chars']} 字")
        if row[2].button("删除", key=f"del_{doc['name']}"):
            kb.delete_doc(doc["name"])
            refresh_index(f"已删除 {doc['name']}")
            st.rerun()

    st.divider()
    st.caption(
        "索引缓存位于 `data/index/`，源文档位于 `data/knowledge/`。"
        "云端实例的磁盘是临时的，重新部署会重置——请把知识库文件随仓库一起提交。"
    )

# ---- 页签 3：评估看板
with tab_eval:
    st.subheader("评估看板")
    st.caption(
        "用测试集批量跑问答，计算开题报告 4.2 节的指标。"
        "准确性为字符级自动近似判定，最终结论仍需按方案人工标注。"
    )

    eval_path = EVAL_DIR / "testset.csv"
    logs = load_logs()
    la = analyze_logs(logs)
    default_df = pd.DataFrame()
    if eval_path.exists():
        try:
            default_df = pd.read_csv(eval_path)
        except Exception:
            default_df = pd.DataFrame()

    up = st.file_uploader("上传测试集（列：question, category, is_in_scope, reference, source_doc）",
                          type=["csv"], key="eval_up")
    testset = default_df
    if up is not None:
        try:
            testset = pd.read_csv(up)
        except Exception as exc:
            st.error(f"测试集读取失败：{exc}")

    if not testset.empty:
        st.dataframe(testset.head(10), width="stretch", hide_index=True)
        st.caption(f"共 {len(testset)} 条 · 知识库内 {(testset.get('is_in_scope', pd.Series()) == 'yes').sum()} 条")
    else:
        st.warning("没有可用测试集，请上传 CSV。")

    run_btn = st.button("▶️ 运行评估", type="primary", disabled=testset.empty or not kb.chunks)
    if run_btn:
        prog = st.progress(0.0, text="准备中…")
        generator = get_generator()

        def _cb(i, total, q):
            prog.progress(i / total, text=f"[{i}/{total}] {q[:28]}")

        result = run_eval(kb, S, testset, generator=generator, progress=_cb)
        prog.progress(1.0, text="评估完成")
        st.session_state.eval_df = result
        st.session_state.eval_sum = summarize(result)
        (EVAL_DIR / "last_eval.csv").write_text(
            result.to_csv(index=False), encoding="utf-8"
        )

    if st.session_state.eval_df is not None:
        df_eval = st.session_state.eval_df
        sm = st.session_state.eval_sum
        st.divider()

        m = st.columns(5)
        m[0].metric("问答准确率", f"{sm['accuracy']:.1%}", "目标 ≥85%",
                    delta_color="normal" if sm["accuracy"] >= 0.85 else "inverse")
        m[1].metric("拒答准确率", f"{sm['refusal_accuracy']:.1%}", "目标 ≥80%",
                    delta_color="normal" if sm["refusal_accuracy"] >= 0.80 else "inverse")
        m[2].metric("P50 耗时", f"{sm['p50']:.2f}s", "目标 ≤10s",
                    delta_color="normal" if sm["p50"] <= 10 else "inverse")
        m[3].metric("疑似幻觉率", f"{sm['hallucination']:.1%}", "目标 ≤10%",
                    delta_color="normal" if sm["hallucination"] <= 0.10 else "inverse")
        m[4].metric("误拒率（知识库内）", f"{sm['false_refusal']:.1%}", "越低越好",
                    delta_color="inverse")

        radar_df = radar_data(sm, satisfaction=la["satisfaction"] if la["rated"] else None)
        st.caption(
            "雷达图为「达标度」：实测值/目标值（耗时与幻觉为越小越好，取目标/实测），满格即达标。"
            + ("满意度由 👍=5 分、👎=1 分折算。" if la["rated"] else "尚无用户反馈，满意度轴暂不显示。")
        )
        r1, r2 = st.columns([1, 1])
        with r1:
            st.plotly_chart(charts.target_radar(radar_df), width="stretch")
        with r2:
            st.plotly_chart(charts.outcome_donut(df_eval), width="stretch")

        st.markdown("**分问题类别准确率**")
        st.plotly_chart(charts.category_accuracy(sm["by_category"]), width="stretch")

        h1, h2 = st.columns(2)
        with h1:
            st.markdown("**耗时分布**")
            st.plotly_chart(charts.latency_hist(df_eval), width="stretch")
        with h2:
            st.markdown("**各类别耗时**")
            st.plotly_chart(charts.latency_box(df_eval), width="stretch")

        with st.expander("查看逐条结果"):
            st.dataframe(df_eval, width="stretch", hide_index=True)
        st.download_button(
            "⬇️ 下载评估明细 CSV",
            data=df_eval.to_csv(index=False).encode("utf-8-sig"),
            file_name="eval_result.csv",
            mime="text/csv",
        )
        st.caption(
            f"判定口径：回答与参考答案字符相似度 ≥ {SIM_THRESHOLD:.0%} 且来源文档命中，"
            f"才算答对（共 {sm['n_in_scope']} 条库内问题 / {sm['n_out_scope']} 条库外问题）。"
        )

    st.divider()
    st.markdown("**在线问答日志**")
    logs = load_logs()
    if logs.empty:
        st.info("还没有问答记录。到「💬 智能问答」提几个问题后再回来。")
    else:
        la = analyze_logs(logs)
        k = st.columns(4)
        k[0].metric("问答次数", la["total"])
        k[1].metric("拒答率", f"{la['refusal_rate']:.1%}")
        k[2].metric("耗时 P50 / P90", f"{la['p50']:.2f}s / {la['p90']:.2f}s")
        k[3].metric("有效反馈", f"{la['up']} 👍 / {la['down']} 👎")
        if la["rated"]:
            st.plotly_chart(charts.feedback_pie(la["up"], la["down"]), width="stretch")
        st.dataframe(
            logs.sort_values("ts", ascending=False).head(50),
            width="stretch",
            hide_index=True,
        )
        st.download_button(
            "⬇️ 下载问答日志 CSV",
            data=LOG_PATH.read_bytes(),
            file_name="qa_log.csv",
            mime="text/csv",
        )

# ---- 页签 4：说明
with tab_about:
    st.subheader("这个系统是怎么跑起来的")
    st.markdown(
        """
**四层架构（对应开题报告图 3-1）**

| 层 | 本仓库实现 | 说明 |
| --- | --- | --- |
| 展示层 | `app.py`（Streamlit） | 问答、知识库、评估看板三个页签 |
| 应用层 | `src/rag.py` | 检索 → 相似度闸门 → 生成 → 引用整理，即 3.4 节的问答链 |
| 模型与检索层 | `src/embedder.py` / `src/llm.py` / `src/vectorstore.py` | 向量化、FAISS 检索、大模型推理，均可替换 |
| 数据层 | `data/knowledge/` + `data/index/` | 源文档与向量索引缓存 |
"""
    )

    st.markdown("**功能需求落点（F1–F5）**")
    st.markdown(
        """
- **F1 知识库管理**：`📚 知识库管理` 页签，上传即解析切片入库，实时显示文档数/片段数/更新时间。
- **F2 自然语言问答**：`💬 智能问答` 页签，自然语言直接问。
- **F3 引用溯源**：每条回答下方可展开「来源引用」，给出文档名、小节标题、原文片段与相关度分数。
- **F4 拒答机制**：相关度低于阈值即拒答，并给出知识库中相近的主题；不调用模型，从机制上杜绝编造。
- **F5 反馈与日志**：👍/👎 回填 `data/logs/qa_log.csv`，评估看板汇总拒答率、耗时与反馈。
"""
    )

    st.markdown("**三种生成后端：为什么这样设计**")
    st.markdown(
        """
| 后端 | 适用场景 | 成本 | 云端可用 |
| --- | --- | --- | --- |
| 检索直出 `extractive` | 只求"有据可依"，不生成新句子 | 零 | ✅ |
| 本地 Ollama `ollama` | 完全离线、数据不出校 | 零 | ❌ 免费云实例跑不了 |
| 云端 API `api` | 云端演示、回答更自然 | 按量计费 | ✅ |

Streamlit Community Cloud 是一个免费的 1 vCPU / 1GB 容器，装不动 Ollama，也装不下 torch。
因此默认使用字符级 TF-IDF 向量 + 检索直出，**开箱即用、零密钥**；
在本机要做完整语义 RAG 时，把 `EMBED_BACKEND` 改成 `hf`、`GEN_BACKEND` 改成 `ollama` 即可，
`src/embedder.py`、`src/llm.py` 已经按接口留好了位置。
"""
    )

    if gen.name == "extractive":
        st.info("当前运行的就是「检索直出」：答案逐句摘自知识库原文，不会出现幻觉。")

    st.markdown("**关于示例数据**")
    st.warning(
        "`data/knowledge/` 下的 5 篇文档是**演示用虚构内容**，不代表任何高校的真实规定，"
        "仅用于跑通流程与演示评估。答辩或上线前请替换为本校教务处、学工、后勤等部门公开发布的文件，"
        "并保留来源与更新时间。"
    )

    st.markdown("**已知局限（写在报告里比藏起来更安全）**")
    st.markdown(
        """
1. 字符级 TF-IDF 只捕捉"用词重合"，同义改写的问法可能召回失败——这是选择零成本方案付出的代价。
2. 准确率是自动近似判定，不是人工标注，真实幻觉率需按 4.1 节方案由两名评估者裁定。
3. 云端磁盘临时化：日志与上传文档在重新部署后丢失，需要持久化时应接对象存储或数据库。
4. 系统只做信息查询，不办理业务，不查询个人数据（对应 6.3 节不做清单）。
"""
    )

st.divider()
st.caption("数据分析与可视化 · 个人项目 ｜ 林雨格 42512063 ｜ 示例知识库仅供演示")
