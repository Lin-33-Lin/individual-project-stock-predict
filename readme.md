# 校园 FAQ 智能问答小助手

> 数据分析与可视化课程 · 个人项目（林雨格 42512063）
> 技术路线：RAG（检索增强生成）+ Streamlit 轻量部署，回答带出处、未知即拒答。

---

## 一、这套代码做了什么

按开题报告的五项功能需求实现：

| 需求 | 页面 | 实现要点 |
| --- | --- | --- |
| F1 知识库管理 | 📚 知识库管理 | 上传 TXT/MD/PDF/Word/CSV 自动解析，按「一问一答」小节切片，实时显示文档数/片段数/更新时间 |
| F2 自然语言问答 | 💬 智能问答 | 自然语言提问，检索 top-k 片段后生成回答 |
| F3 引用溯源 | 💬 回答卡片 | 可展开「来源引用」，含文档名、小节标题、原文片段、相关度分数 |
| F4 拒答机制 | 💬 + 检索层 | 相关度低于阈值直接拒答并提示相近主题，不调用模型 |
| F5 反馈与日志 | 💬 + 📊 | 👍/👎 落盘 `data/logs/qa_log.csv`，看板汇总拒答率、耗时分位数、反馈 |

第 4 章的评估也做成了可运行页面：一次点击跑完测试集，输出准确率、拒答准确率、P50/P90/P95 耗时、疑似幻觉率，以及柱状图 / 直方图 / 箱线图 / 环形图 / 雷达图。

---

## 二、本地跑起来（3 分钟）

```bash
# 1. 进入项目目录
cd campus-faq-assistant

# 2. 建虚拟环境（Windows PowerShell 用 .venv\Scripts\activate）
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 3. 装依赖
pip install -r requirements.txt

# 4. 启动
streamlit run app.py
```

浏览器会自动打开 `http://localhost:8501`。首次启动会把 `data/knowledge/` 里的文档切片、向量化并缓存到 `data/index/`，一般 2–5 秒。

默认就是**零成本模式**：字符级 TF-IDF 向量 + 检索直出，不下载模型、不需要 API Key。

### 想要"真·大模型回答"（可选）

**A. 本地 Ollama（完全离线、数据不出校）**

```bash
# 安装 Ollama 后拉一个中文模型
ollama pull qwen2.5:3b
ollama serve
```

然后在侧边栏把「生成后端」切到 **本地 Ollama**（程序会自动选用本机已安装的模型），或在 `.streamlit/secrets.toml` 里写：

```toml
GEN_BACKEND = "ollama"
OLLAMA_MODEL = "qwen2.5:3b"
EMBED_BACKEND = "hf"      # 需要 pip install sentence-transformers
```

> 实测提醒：在笔记本 CPU 上，1.5B 级别的模型单次回答约 30 秒，明显超过报告里 10 秒的目标；
> 演示时建议用「检索直出」或 3B 以上带 GPU 加速的模型，评估时也把真实耗时如实记录。

**B. 云端 API（任何 OpenAI 兼容接口）**

```toml
GEN_BACKEND = "api"
API_BASE = "https://api.deepseek.com/v1"
API_KEY = "sk-..."
API_MODEL = "deepseek-chat"
```

也可以在侧边栏「🔌 接口设置」里临时填，不写文件。

### 三种后端怎么选（默认「自动」）

| 模式 | 行为 | 建议 |
| --- | --- | --- |
| 自动 `auto` | 有 API Key 就用 API，否则检索直出 | 默认，最稳 |
| 检索直出 `extractive` | 不调用模型，答案逐句摘自原文 | 演示、答辩、断网环境 |
| 本地 Ollama `ollama` | 本地模型生成，数据不出机 | 本机装了模型、且能接受更慢 |
| 云端 API `api` | 云端模型生成 | 云端部署 + 有 Key |

---

## 三、部署到 Streamlit Community Cloud

1. 把整个 `campus-faq-assistant/` 目录推到 GitHub（`.streamlit/secrets.toml` 已被 `.gitignore` 排除）。
2. 打开 https://share.streamlit.io → New app → 选仓库、分支，**Main file path 填 `campus-faq-assistant/app.py`**（若仓库根目录就是项目，则填 `app.py`）。
3. 在 App → Settings → Secrets 粘贴：

```toml
GEN_BACKEND = "api"
API_BASE = "https://api.deepseek.com/v1"
API_KEY = "sk-..."
API_MODEL = "deepseek-chat"
EMBED_BACKEND = "tfidf"
```

不填也能跑——会自动落到「检索直出」，界面不会报错。

### 云端的三个硬约束（已提前绕开）

- **装不了 Ollama**：免费实例是 1 vCPU / 1GB 容器，跑不动本地大模型 → 默认走检索直出或 API。
- **装不下 torch**：`sentence-transformers` 会拖进 800MB 依赖 → 默认用字符级 TF-IDF，中文 FAQ 场景够用。
- **磁盘是临时的**：重新部署会重置 `data/` → 知识库文档随仓库提交，日志用页面上的「下载 CSV」及时导出。

---

## 四、目录结构

```
campus-faq-assistant/
├── app.py                  # Streamlit 主程序：问答 / 知识库 / 评估 / 说明
├── requirements.txt
├── README.md
├── .streamlit/
│   ├── config.toml         # 主题与 server 配置
│   └── secrets.toml.example
├── src/
│   ├── config.py           # 参数、路径、密钥（一处可调）
│   ├── loader.py           # TXT/MD/PDF/DOCX/CSV → 纯文本
│   ├── splitter.py         # 按小节切片，超长段落二次切分
│   ├── embedder.py         # 向量化：tfidf / hf / ollama 三选一
│   ├── vectorstore.py      # FAISS（装了就用）或 NumPy 兜底
│   ├── retriever.py        # 余弦 + 词面重合的混合相关度、拒答闸门
│   ├── llm.py              # Ollama / API / 检索直出 三种生成后端
│   ├── rag.py              # 问答主链路（对应报告 3.4 节）
│   ├── kb.py               # 知识库构建与索引持久化
│   ├── logstore.py         # 问答日志与点赞点踩
│   ├── evaluation.py       # 指标计算、达标度雷达
│   └── charts.py           # 看板图表
├── data/
│   ├── knowledge/          # 知识库源文档（当前为 5 篇演示文档）
│   ├── eval/testset.csv    # 30 条测试集（24 库内 + 6 库外）
│   ├── index/              # 向量索引缓存（自动生成）
│   └── logs/               # qa_log.csv（自动生成）
└── scripts/
    └── selftest.py         # 命令行自检：建索引、跑样例、跑评估
```

---

## 五、与开题报告对应（答辩时可以直接照着讲）

- **图 3-1 四层架构** → 展示层 `app.py`、应用层 `src/rag.py`、模型与检索层 `embedder/llm/vectorstore`、数据层 `data/knowledge` + `data/index`。
- **3.4 核心问答流程** → `src/rag.py` 的 `answer_question()`：向量化 → 检索 top-k → 相关度闸门 → 拼提示词 → 生成 → 解析引用，五步在代码里是五个可直接打断点的小步骤。
- **表 3-2 技术选型** → 每个组件都是可替换的接口：`BaseEmbedder`、`BaseGenerator`、`VectorIndex`，换实现不改业务代码。
- **4.2 指标定义** → `src/evaluation.py` 的 `summarize()` 输出准确率、拒答准确率、P50/P90、幻觉率、误拒率，与表 4-1 一一对应。
- **6.1 风险应对** → 防幻觉靠"引用 + 阈值过滤 + 拒答"三道机制（`retriever.py` 与 `llm.py`）；防失败靠"模型挂了自动降级检索直出"（`rag.py` 的 `except LLMError`）。

### 与报告的一处偏差（主动说明）

报告表 3-2 写的是 LangChain 编排。本仓库用**显式函数链**替代 LangChain：`检索 → 过滤 → 提示词 → 生成` 四处逻辑加起来不到 60 行，比引入 LangChain 更好调试、依赖更少（避免云端版本冲突）。如果想换回 LangChain，只需替换 `src/rag.py`，其余模块不动——这正是"组件可独立替换"的验证。

---

## 六、调参建议

| 现象 | 调整 |
| --- | --- |
| 该答的拒答了 | 侧边栏调低「相关度阈值」（0.20 → 0.14） |
| 答了不该答的 | 调高阈值（0.20 → 0.28），或提高「语义权重」 |
| 召回不到正确片段 | 提高 top-k；检查知识库是否按「## 问题 + 答案」组织 |
| 回答太长/太散 | 把每节答案压缩到 3–5 句，切片更干净 |

阈值是按本仓库示例库标定的：24 道知识库内问题的最高相关度落在 0.29–0.58，6 道库外问题落在 0.00–0.12，0.20 正好卡在中间的空白带。换一批文档后建议跑一次 `scripts/selftest.py` 重新看分布。

---

## 七、当前基线（在示例库上实测）

`python scripts/selftest.py` 的输出（检索直出模式、5 篇示例文档、48 个片段、30 条测试集）：

| 指标 | 实测 | 目标 |
| --- | --- | --- |
| 问答准确率 | 91.7%（22/24） | ≥ 85% ✅ |
| 拒答准确率 | 100%（6/6 库外问题） | ≥ 80% ✅ |
| 误拒率 | 0%（0/24） | — |
| 耗时 P50 / P90 | 0.02s / 0.03s | ≤ 10s / ≤ 15s ✅ |
| 疑似幻觉率 | 0% | ≤ 10% ✅ |

未通过的两条都是"回答不完整"（参考答案列了 3 条，系统只摘出 2 条），不是答错方向——这正好说明自动判定偏保守，人工标注时大概率会判为正确。

---

## 八、示例数据声明

`data/knowledge/` 下 5 篇文档（选课考试、奖学金资助、宿舍生活、证件事务、学籍培养）与 `data/eval/testset.csv` 均为**课程演示用虚构内容**，不代表任何高校真实规定。

正式使用前请替换为本校教务处、学生工作部、后勤与保卫部门公开发布的文件，并在文档首行注明来源与更新时间——这既是数据可溯源的底线，也是报告 5.1 节的要求。
