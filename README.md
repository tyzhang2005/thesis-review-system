# 🎓 Thesis RAG Evaluator - Backend

基于 LLM + RAG 的本科毕业论文自动化评审系统后端

> [![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
> [![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
> [![LangChain](https://img.shields.io/badge/LangChain-0.1+-orange.svg)](https://www.langchain.com/)
> [![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5+-yellow.svg)](https://www.trychroma.com/)

---

## 📖 项目简介

基于 **RAG 技术** 的智能论文评审系统后端服务，实现 18 维智能评分与个性化修改建议生成。

### 核心特性

- **500+ 专家案例库**: 基于向量数据库的混合检索系统
- **7 步评审 Pipeline**: 处理 2 万字论文的长文本评估流程
- **云端 API 集成**: 推理和嵌入模型均使用云端 API，无需本地 GPU
- **本地化 PDF 解析**: 推荐使用本地 MinerU 进行 PDF 转 Markdown
- **结构化输出**: Pydantic 约束 JSON Schema，确保输出质量

---

## 🛠️ 技术栈

| 技术      | 版本   | 用途          |
| --------- | ------ | ------------- |
| FastAPI   | 0.100+ | Web 框架      |
| LangChain | 0.1+   | LLM 集成      |
| ChromaDB  | 0.5+   | 向量数据库    |
| Pydantic  | 2.0+   | 数据校验      |
| MinerU    | -      | 本地 PDF 解析 |
| asyncio   | -      | 异步处理      |

---

## 📁 项目结构

```
backend/
├── main.py                    # FastAPI 入口
├── routers/                   # API 路由
│   ├── evaluation.py         # 论文评估核心
│   ├── aigc_detector.py      # AIGC 检测
│   ├── auth.py               # 用户认证
│   ├── file_handlers.py      # 文件处理
│   ├── advice_retrieval.py   # 建议检索
│   └── vectorstore.py        # 向量数据库
├── services/                  # 业务服务
│   ├── llm_utils.py         # LLM 工具
│   ├── pdf_annotator.py     # PDF 标注
│   ├── aigc_report_generator.py # AIGC 报告生成
│   └── markdown_processor.py # Markdown 处理
├── models/                    # 数据模型
│   └── schemas.py           # Pydantic 模型
├── prompts/                   # 提示词模板
│   ├── step1_classify.md
│   ├── step2_chapter_classify.md
│   ├── step3_retrieve_advice.md
│   ├── step4_engineering/
│   ├── step4_method/
│   ├── step4_theory/
│   ├── step5_workload_*.md
│   ├── step6_summary_advice.md
│   └── step7_comprehensive_scoring.md
├── templates/                 # 评估模板
│   ├── template_engineering.py
│   ├── template_method.py
│   ├── template_theory.py
│   └── evaluate_template.py
├── config/                    # 配置管理
│   ├── config.py
│   └── settings.json
└── data/                     # 运行时数据
    ├── uploads/              # 上传文件
    ├── databases/            # 向量数据库
    ├── processed/            # 处理结果
    └── results/              # 最终输出
```

---

## 🚀 快速开始

### 环境要求

- Python 3.10+
- Conda (推荐)

### 安装 MinerU (本地 PDF 解析)

**推荐使用本地 MinerU 以获得最佳解析效果**：

```bash
# 使用 Docker 运行 MinerU (推荐)
docker pull opendatalab/mineru:latest
docker run -d -p 8888:8888 opendatalab/mineru:latest

# 或从源码安装
git clone https://github.com/opendatalab/MinerU.git
cd MinerU
pip install -e .
```

### 安装 Python 依赖

```bash
# 使用 conda 创建环境
conda env create -f environment.yml
conda activate ruiwen

# 或使用 pip
pip install -r requirements.txt
```

### 配置环境变量

```bash
# 复制配置模板
cp .env.example .env

# 编辑配置，填入 API Keys
nano .env
```

**主要配置项**:

```env
# 云端 API 配置
CLOUD_API_KEY=sk-your-deepseek-api-key    # DeepSeek API Key
CLOUD_API_BASE=https://api.deepseek.com   # API Base URL

# 模型配置
MODEL_NAME=deepseek-chat                   # 推理模型
EMBEDDING_MODEL=text-embedding-3-small     # 嵌入模型

# MinerU 配置 (本地部署)
MINERU_API_URL=http://localhost:8888       # MinerU 服务地址
MINERU_TOKEN=                              # 本地部署可留空
```

### 启动服务

```bash
python main.py
# 访问 http://localhost:8000
# API 文档: http://localhost:8000/docs
```

---

## 📊 核心功能

### 1. 论文解析 (本地 MinerU)

- **MinerU 本地部署**: PDF → Markdown 高质量转换
- **元数据提取**: 标题、作者、摘要等
- **层级结构识别**: 章节树自动构建
- **分块处理**: 长文档智能分段

### 2. RAG 评估流程

```
PDF 上传 → MinerU解析 → 分类 → 检索案例 → 云端LLM评估 → 输出结果
                ↓           ↓         ↓            ↓
            Markdown      类型    向量DB      结构化JSON
```

**7 步 Pipeline**:

1. 论文分类 (工程/方法/理论)
2. 章节分类
3. 建议检索
4. 分类评估
5. 工作量评估
6. 总结建议
7. 综合评分

### 3. AIGC 检测

- 基于块的文本分析
- 多维度 AI 特征提取
- 检测报告生成
- PDF 标注输出

### 4. 人工评审

- 评审数据录入
- 批注管理
- 评分与建议保存

---

## 🔧 API 端点

### 评估相关

| 端点                  | 方法 | 描述         |
| --------------------- | ---- | ------------ |
| `/api/upload`         | POST | 上传论文     |
| `/api/qa`             | POST | 启动评估     |
| `/api/task/{user_id}` | GET  | 查询任务状态 |
| `/api/download_pdf`   | POST | 下载评审表   |

### AIGC 检测

| 端点                               | 方法 | 描述      |
| ---------------------------------- | ---- | --------- |
| `/api/aigc_detect`                 | POST | AIGC 检测 |
| `/api/aigc_detect/download-report` | GET  | 下载报告  |

### 建议检索

| 端点                     | 方法 | 描述         |
| ------------------------ | ---- | ------------ |
| `/api/initialize_advice` | POST | 初始化案例库 |
| `/api/search_advice`     | POST | 检索建议     |

### 认证

| 端点                 | 方法 | 描述     |
| -------------------- | ---- | -------- |
| `/api/register_user` | POST | 用户注册 |
| `/api/login_user`    | POST | 用户登录 |
| `/api/logout`        | POST | 用户登出 |





---

## 📄 许可证

MIT License
