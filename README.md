# 🎓 Thesis RAG Evaluator

基于 LLM + RAG 的本科毕业论文自动化评审系统

> [![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
> [![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
> [![LangChain](https://img.shields.io/badge/LangChain-0.1+-orange.svg)](https://www.langchain.com/)
> [![React](https://img.shields.io/badge/React-18-blue.svg)](https://react.dev/)

---

## 📖 项目简介

针对人工评审效率低、通用模型评分区分度差及评语模板化问题，构建基于 **RAG 技术** 的自动化评审系统。

### 核心亮点

- **500+ 专家案例库**: 将 180 篇往届人工评审表构建为结构化案例库
- **18 维智能评分**: 融合规则引擎与 LLM 语义分析，深度适配院系标准
- **RAG 混合检索**: 向量搜索 + 元数据过滤，生成具体可操作的修改建议
- **长文本处理**: 7 步评审 pipeline，智能处理 2 万字毕业论文
- **本地化部署**: 基于 vLLM 部署量化模型，支持多用户并发

## ✨ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                         用户端                                │
├─────────────────────────────────────────────────────────────┤
│  学生端        教师端        教务端        AIGC检测            │
│  上传论文      人工评审      统计分析      AI生成检测         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                       FastAPI 后端                            │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │ 论文解析  │  │ RAG检索  │  │ LLM评估  │  │ AIGC检测 │     │
│  │ MinerU   │  │ChromaDB │  │ vLLM     │  │ 分析器   │     │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      500+ 专家案例库                           │
│  分数、评语、思维链 | 论文类型 | 章节概要 | 结构化字段         │
└─────────────────────────────────────────────────────────────┘
```

## 🛠️ 技术栈

### 后端
| 技术 | 用途 |
|------|------|
| FastAPI | Web 框架 |
| LangChain | LLM 集成 |
| ChromaDB | 向量数据库 |
| vLLM | 模型推理 |
| PyTorch | 深度学习 |
| Pydantic | 数据校验 |
| MinerU | PDF 解析 |

### 前端
| 技术 | 用途 |
|------|------|
| React 18 | UI 框架 |
| TypeScript | 类型安全 |
| Vite | 构建工具 |
| Ant Design | UI 组件 |

## 🚀 快速开始

### 前置要求

- Python 3.10+
- Node.js 14+
- Conda (推荐)

### 前端

```bash
cd frontend
npm install
npm run dev
# 访问 http://localhost:3000
```

### 后端

```bash
cd backend

# 创建 conda 环境
conda env create -f environment.yml
conda activate ruiwen

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 填入 API Keys

# 启动服务
python main.py
# 访问 http://localhost:8000
```

## 📁 项目结构

```
ruiwen/
├── frontend/           # React 前端
│   ├── src/
│   │   ├── pages/     # 页面组件
│   │   ├── utils/     # 工具函数
│   │   └── components/# UI 组件
│   ├── public/
│   │   └── data/      # 静态数据
│   └── vite.config.ts
│
└── backend/           # Python 后端
    ├── routers/       # API 路由
    │   ├── evaluation.py      # 论文评估
    │   ├── aigc_detector.py   # AIGC 检测
    │   └── auth.py            # 认证
    ├── services/      # 业务逻辑
    │   ├── llm_utils.py      # LLM 工具
    │   ├── pdf_annotator.py  # PDF 标注
    │   └── markdown_processor.py
    ├── prompts/       # 提示词模板
    ├── templates/     # 评估模板
    ├── models/        # 数据模型
    ├── config/        # 配置管理
    └── main.py        # 应用入口
```

## 📊 评估体系

### 18 维评分

**格式规范 (6项)**
1. 结构完整性
2. 摘要和关键词规范性
3. 目录规范性
4. 章节规范性
5. 参考文献格式规范性
6. 致谢规范性

**内容质量 (12项)**
7. 选题契合度
8. 选题工作量适宜度
9. 选题学术价值
10. 文献检索和分析能力
11. 知识综合应用和研究深度
12. 专业方法工具运用
13. 专业技能和实践能力
14. 技术应用和外语能力
15. 创新性
16. 论证严谨性和科学性
17. 论文结构和语言表达
18. 成果价值

### 评分标准
- 每项 0-3 分
- 总分 0-54 分
- 等级: 优秀(48+) / 良好(42+) / 中等(36+) / 及格(30+) / 不及格(<30)

## 📈 量化成果

| 指标 | 结果 |
|------|------|
| 解析准确率 | 94% (180篇论文测试) |
| 评分一致性 | ICC(3,1)=0.82 |
| 四分类准确率 | 75% |
| 案例库规模 | 500+ 条专家案例 |
| 处理速度 | ~2分钟/篇 |

## 🔧 配置说明

### 环境变量

```bash
# 后端 .env
CLOUD_API_KEY=sk-your-api-key          # DeepSeek API Key
MINERU_TOKEN=Bearer-your-token         # MinerU Token
MODEL_NAME=deepseek-v3.2               # 模型名称
```

### 模型部署

**本地部署 (推荐)**:
- 推理模型: DeepSeek-R1-Distill-Qwen-32B-Q4_K_L
- 嵌入模型: Qwen3-Embedding-8B-Q4_K_M

**云端 API**:
- DeepSeek-v3.2
- Qwen3-Embedding-8B

## 📱 在线演示

- **GitHub Pages**: https://tyzhang2005.github.io/thesis-review-system/

## 📄 许可证

MIT License

---

## 🙏 致谢

本项目基于以下开源项目构建：
- [FastAPI](https://fastapi.tiangolo.com/)
- [LangChain](https://www.langchain.com/)
- [ChromaDB](https://www.trychroma.com/)
- [vLLM](https://github.com/vllm-project/vllm)
- [MinerU](https://github.com/opendatalab/MinerU)
