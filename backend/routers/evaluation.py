import asyncio
import json
import logging
import os
import re
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

from chromadb import PersistentClient
from chromadb.config import Settings
from config.config import (
    DATABASE_VECTORSTORE_DIR,
    UPLOAD_FOLDER,
    USER_MD_DIR,
    USER_RESULT_DIR,
    VLLM_EMBEDDING_ENDPOINT,
)
from fastapi import APIRouter, HTTPException
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document
from models.models import Query

# 导入Pydantic模型
from models.schemas import (
    ChapterClassificationResponse,
    ChapterEvaluationResponse,
    ComprehensiveScore,
    PaperClassificationResponse,
    WorkloadEvaluationResponse,
)

# 导入建议检索模块（用于在评估完成后检索用户评审结果建议）
from routers.advice_retrieval import retrieve_advice
from routers.file_handlers import convert_pdf_to_markdown
from routers.vectorstore import initialize_advice_database
from services.latex_generator import generate_latex_report
from services.llm_utils import (
    VLLMEmbeddings,
    async_llm,
    async_llm_structured,
    async_llm_structured_v2,
)
from services.markdown_processor import ChineseMarkdownSplitter
from services.prompt_service import PromptService
from services.reference_validator import ReferenceValidator

# 导入模板解析工具
from templates.template_parse import (
    get_complete_field_descriptions,
    get_parser_by_paper_type,
    parse_engineering_chapter_structure,
    parse_method_chapter_structure,
    parse_theory_chapter_structure,
)

# 初始化PromptService
prompt_service = PromptService()
logging.basicConfig(level=logging.INFO)
os.environ["ANONYMIZED_TELEMETRY"] = "False"

router = APIRouter()

# ============    进度相关  ===================
progress_store: Dict[str, Dict[str, Any]] = {}

# 评估阶段定义
EVALUATION_STAGES = [
    ("加载文档", 5),
    ("开始评估", 10),
    ("论文分类", 15),
    ("章节分类", 20),
    ("检索建议", 25),
    ("章节评估", 30),
    ("工作量评估", 70),
    ("汇总建议", 90),
    ("生成报告", 100),
]


def update_progress(user_id: str, stage_name: str, message: str = ""):
    """更新进度信息"""
    if user_id in progress_store:
        progress_value = next(
            (p for name, p in EVALUATION_STAGES if name == stage_name), 0
        )
        progress_store[user_id].update(
            {
                "stage": stage_name,
                "progress": progress_value,
                "message": message,
                "timestamp": time.time(),
            }
        )
        logging.info(
            f"进度更新 [{user_id}]: {stage_name} - {progress_value}% - {message}"
        )


def get_stage_progress(stage_name: str) -> int:
    """获取阶段对应的进度百分比"""
    return next((p for name, p in EVALUATION_STAGES if name == stage_name), 0)


# ===============================


def preprocess_markdown(content: str) -> str:
    """预处理Markdown内容"""
    content = re.sub(r"(\n#+\s+[\u4e00-\u9fa5]+)\n\n#+", r"\1", content)
    content = re.sub(r"(\$\$[\s\S]*?\$\$)", r"<!--formula-->\1<!--/formula-->", content)
    return content


def build_deep_context(docs: List[Document]) -> str:
    """构建带深度元数据的上下文"""
    context_lines = []
    for i, doc in enumerate(docs):
        meta = doc.metadata
        hierarchy = []
        if meta.get("chapter"):
            hierarchy.append(meta["chapter"])
        if meta.get("section"):
            hierarchy.append(meta["section"])
        if meta.get("subsection"):
            hierarchy.append(meta["subsection"])
        if meta.get("subsubsection"):
            hierarchy.append(meta["subsubsection"])
        content_length = len(doc.page_content)
        context_lines.append(doc.page_content)
    return "\n".join(context_lines)


def chapter_info(chapter_name, docs):
    """返回包含分块数量和序号的章节信息"""
    return {
        "chapter_name": chapter_name,
        "total_chunks": len(docs),
        "chunks_used": [
            {"index": idx, "preview": doc.page_content[:50]}
            for idx, doc in enumerate(docs[:50])  # 返回前3个分块
        ],
        "meta": docs[0].metadata if docs else {},
    }


def findword(ch, text):
    match = re.search(ch, text)
    if match:
        return match.start()
    return None


def checkorder(text):
    placelist = []
    # chlist = [r"诚信承诺书", r"摘要", r"目录|目\s*录", r"参考文献", r"致谢|致\s*谢", r"附录|附\s*录"]
    # infolist = ["承诺书", "摘要", "目录", "参考文献", "致谢", "附录"]
    chlist = [
        r"诚信承诺书",
        r"摘要",
        r"#\s*(目录|目\s*录|I|目|录)[\s\S]*",
        r"(#(| 1| i\\*)\n|参考文献)",
        r"致谢|致\s*谢",
        r"附录|附\s*录",
    ]
    infolist = ["承诺书", "摘要", "目录", "参考文献", "致谢", "附录"]
    has_reference = False
    missing = 0
    disorder = 0
    missing_text = ""
    disorder_text = ""
    # print(text)
    for i in range(len(chlist)):
        match = findword(chlist[i], text)
        placelist.append(match)
        if match == None:
            missing += 1
            missing_text += infolist[i] + " "
    for i in range(len(placelist) - 1):
        if placelist[i] != None:
            if placelist[i + 1] != None:
                if placelist[i] > placelist[i + 1]:
                    disorder += 1
                    disorder_text += infolist[i] + "与" + infolist[i + 1] + " "

    if not has_reference:
        missing += 1

    score = 12 - missing - disorder
    score = score if score > 0 else 0
    missing_text = "无" if missing_text == "" else missing_text
    disorder_text = "无" if disorder_text == "" else disorder_text
    final_text = "缺少： " + missing_text + "\n" + disorder_text + "顺序错误"
    return int(score / 4), final_text


def checkabstract(document):
    abstract = document.metadata.get("abstract", "N/A")
    english_abstract = document.metadata.get("english_abstract", "N/A")
    keywords = document.metadata.get("keywords", "N/A")
    score = 3
    info = ""

    abstract_length = len(abstract)
    english_abstract_length = len(english_abstract.split())
    keywords_length = len(keywords.split("；"))

    if abstract_length < 300:
        info += "中文摘要过短，"
        score -= 1
    elif abstract_length > 600:
        info += "中文摘要过长，"
        score -= 1
    if english_abstract_length < 200:
        info += "英文摘要过短，"
        score -= 1
    elif english_abstract_length > 400:
        info += "英文摘要过长，"
        score -= 1
    if keywords_length < 3:
        info += "关键词过少，"
        score -= 1
    elif keywords_length > 5:
        info += "关键词过多，"
        score -= 1
    """
    abstractstr = "中文摘要：" + abstract + "\n"
    english_abstractstr = "英文摘要：" + english_abstract

    # 1. 加载文档
    context = abstractstr + english_abstractstr

    # 2. 初始化向量存储
    embeddings = OllamaEmbeddings(model="deepseek-r1:8b")
    vectorstore0 = Chroma.from_texts(context, embeddings)

    # 3. 设置处理链
    llm = OllamaLLM(model="deepseek-r1:8b")
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore0.as_retriever())

    # 4. 评估查询
    query_ch = "您是一位经验丰富、认真负责、公正的学术审稿人，请帮我审阅这篇论文的中英文摘要是否内容一致。中英文摘要内容如下:" + context + "评估中英文摘要的内容一致性，可能的分数及其描述为: \n    1 较差\n    2 一般\n    3 良好\n    4 优秀\n用中文回答四个选项其中一个，即'较差，一般，良好，优秀'其中一个。如果内容基本一致就回答优秀，不一致就根据不一致的程度回答一个答案。"
    responses = qa_chain.invoke(query_ch)
    result_text = responses['result'].split("</think>\n\n")[1] if "</think>\n\n" in responses['result'] else responses[
        'result']
    qscore = review_to_score(responses['result'])
    if qscore < 2:
        info += "中英文摘要内容不一致"
    """
    # print(responses)
    return int((score * 4 + 3 * 6) / 10), info  # + "请修改。"


def get_title(metadata):
    chapter = metadata.get("chapter", "Unknown")
    section = metadata.get("section", "Unknown")
    # subsection = metadata.get('subsection', 'Unknown')
    # if (subsection):
    # return subsection.replace("# ", "")
    if section:
        return section.replace("# ", "")
    if chapter:
        return chapter.replace("# ", "")
    return ""


def checkbody(documents):
    content = documents[0].metadata.get("contents", "N/A").replace(" ", "")
    tmpinfo = ""
    if content == "缺失目录":
        tmpinfo += "[目录]缺失目录"
    first_title = documents[0].metadata.get("chapter", "Unknown")
    begin_right = 1 if "引言" in first_title or "绪论" in first_title else 0
    body_length = 0
    right_title = 0
    has_reference = False
    has_thank = False
    previous_title = None

    for idx, doc in enumerate(documents):
        # print(get_title(doc.metadata).replace(" ", ""))
        current_title = get_title(doc.metadata)

        if "附录" in doc.metadata.get("chapter", "Unknown"):
            continue

        if current_title != "Unknown" and current_title != previous_title:
            if get_title(doc.metadata).replace(" ", "") in content:
                right_title += 1
            elif content != "缺失目录" and get_title(doc.metadata) != "Unknown":
                tmpinfo += "[目录]目录中未找到" + get_title(doc.metadata) + "，" + "\n"
            # print(tmpinfo)

        previous_title = current_title

        if "参考文献" in doc.metadata.get("chapter", "Unknown"):
            has_reference = True
            last_title = documents[idx - 1].metadata.get("chapter", "Unknown")
            last_index = idx
            end_right = (
                1
                if "结论" in last_title or "讨论" in last_title or "总结" in last_title
                else 0
            )
            break
        body_length += len(doc.page_content)

    # print(body_length > 15000)
    length_right = 1 if body_length > 15000 else 0

    if has_reference:
        validator = ReferenceValidator()

        reference_section = documents[last_index].page_content

        ref_validations, entnum = validator.validate_reference(reference_section)

        ref_error = len(ref_validations)

        formatted_strings = []
        for item in ref_validations:
            # 处理每个文献条目
            entry = item["entry"]
            errors = "、".join(item["errors"])  # 用顿号连接错误列表
            formatted_strings.append(
                f"[参考文献]引用的参考文献：{entry},存在格式错误：{errors}\n"
            )
        refe = "\n".join(formatted_strings)

        # print(ref_error)

        # print(ref_validations)

        if len(ref_validations) == 0:
            ref_score = 1
            tmpinfo += "\n[参考文献]识别到的参考文献不符合格式规范\n"
        else:
            ref_score = 2
    else:
        ref_score = 0

    thank_score = 0

    for idx, doc in enumerate(documents[last_index + 1 :]):
        # print(get_title(doc.metadata).replace(" ", ""))
        if get_title(doc.metadata).replace(" ", "") in content:
            right_title += 1
        elif content != "缺失目录" and get_title(doc.metadata) != "Unknown":
            tmpinfo += "[目录]目录中未找到" + get_title(doc.metadata) + "，" + "\n"
            # print(tmpinfo)
        if "致谢" in doc.metadata.get("chapter", "Unknown").replace(" ", ""):
            has_thank = True
            thank_score += 1
            if len(doc.page_content) > 10:
                thank_score += 2
            elif len(doc.page_content) > 0:
                tmpinfo += "[致谢]致谢过短" + "\n"
                thank_score += 1
            else:
                tmpinfo += "[致谢]致谢为空" + "\n"

        # print(f"Chunk {idx+1} [{doc.metadata.get('chapter', 'Unknown')} > {doc.metadata.get('section', 'Unknown')} > {doc.metadata.get('subsection', 'Unknown')}]\n")
        # print(f"Length: {len(doc.page_content)} chars\n")
        # print(f"Content: {doc.page_content[:200]}...\n")
        # print("-" * 80 + "\n")
    # print(right_title, "/", len(documents))
    # print(thank_score)

    if not has_reference:
        tmpinfo += "[参考文献]缺失参考文献" + "\n"
    if not has_thank:
        tmpinfo += "[致谢]缺失致谢" + "\n"

    return (
        [
            int((right_title / len(documents)) * 3),
            int(begin_right + end_right + length_right),
            ref_score,
            thank_score,
        ],
        tmpinfo,
        refe,
    )


async def async_multilevel_evaluation(
    user_documents,
    processed_md,
    chapter_info,
    chapter_groups,
    data,
    evaluation_dir,
    user_id: str = None,
):

    # ========== 多级评估流程 ==========

    # 存储评估结果
    evaluation = []

    start_time = time.time()

    # 更新进度
    def update_stage(stage_name: str, message: str = ""):
        if user_id:
            update_progress(user_id, stage_name, message)

    # ---------- 并发章节评估 ----------

    # 步骤1: 论文类别分类
    logging.info(f"第一阶段：论文类别分类")
    update_stage("论文分类", "正在分析论文类型...")

    # 使用PromptService获取分类模板
    classify_template = prompt_service.get_template("step1_classify")

    for chapter_name, docs in chapter_groups.items():
        first_doc = docs[0]
        title = first_doc.metadata.get("title", "未知标题")
        break

    prompt = prompt_service.format_template(
        "step1_classify",
        title=first_doc.metadata.get("title", "未知标题"),
        abstract=first_doc.metadata.get("abstract", "")[:500].replace("\n", " "),
        keywords=", ".join(first_doc.metadata.get("keywords", "").split("，")[:5]),
        structure=first_doc.metadata.get("structure"),
    )

    response = await async_llm(prompt, data.model)

    target_labels = ["理论研究", "方法创新", "工程实现"]
    stage = "方法创新"
    for label in target_labels:
        if label in response:
            stage = label

    logging.info(f" 文章标题：{title}:分类结果： {stage}")

    for chapter_name, docs in chapter_groups.items():
        for doc in docs:
            doc.metadata.update({"type": stage})

    # 步骤2: 根据论文类型进行统一章节分类
    logging.info(f"第二阶段：统一章节分类")
    update_stage("章节分类", "正在分析论文章节结构...")

    PAPER_TYPE_STAGE_LABELS = {
        "理论研究": [
            "引言/绪论",
            "引言/绪论（包含相关工作）",
            "相关工作",
            "背景知识",
            "数据来源与处理",
            "模型与证明",
            "实验分析",
            "性能评估",
            "实验分析与性能评估",
            "结论展望",
        ],
        "方法创新": [
            "引言/绪论",
            "引言/绪论（包含相关工作）",
            "相关工作",
            "背景知识",
            "数据来源与处理",
            "方法构建",
            "实验验证",
            "结果分析",
            "实验验证与结果分析",
            "结论展望",
        ],
        "工程实现": [
            "引言/绪论",
            "引言/绪论（包含相关工作）",
            "相关工作",
            "背景知识",
            "数据来源与处理",
            "系统设计",
            "系统实现",
            "系统评估",
            "系统实现与评估",
            "结论展望",
        ],
    }

    # 映射到PromptService中的模板名称
    PAPER_TYPE_CLASSIFY_TEMPLATES = {
        "理论研究": "step2_chapter_classify_theory",
        "方法创新": "step2_chapter_classify_method",
        "工程实现": "step2_chapter_classify_engineering",
    }

    def classification_json_schema() -> Dict[str, Any]:
        """章节分类JSON Schema"""
        return {
            "type": "object",
            "properties": {
                "chapters": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "chapter_name": {"type": "string"},
                            "stage": {"type": "string"},
                        },
                        "required": ["chapter_name", "stage"],
                    },
                }
            },
            "required": ["chapters"],
        }

    current_paper_type = stage  # 从步骤1获取的论文类型
    classify_template = PAPER_TYPE_CLASSIFY_TEMPLATES.get(current_paper_type)
    target_labels = PAPER_TYPE_STAGE_LABELS.get(current_paper_type, [])

    async def unified_chapter_classification(data, paper_type, chapter_groups):
        """统一的章节分类处理"""

        # 构建所有章节信息字符串
        chapters_info = []
        for chapter_name, docs in chapter_groups.items():
            if not docs or any(
                exclude in chapter_name
                for exclude in ["摘要", "致谢", "参考文献", "谢"]
            ):
                continue

            first_doc = docs[0]

        # 选择对应的分类模板名称
        template_name = PAPER_TYPE_CLASSIFY_TEMPLATES.get(paper_type)

        if not template_name:
            logging.error(f"未知的论文类型: {paper_type}")
            return {}

        # 获取第一篇文档的元数据
        first_doc = next(iter(chapter_groups.values()))[0] if chapter_groups else None
        if not first_doc:
            return {}

        formatted_chapter_name = ""
        for chapter_name, docs in chapter_groups.items():
            if not docs or any(
                exclude in chapter_name
                for exclude in ["摘要", "致谢", "参考文献", "谢"]
            ):
                continue
            formatted_chapter_name += chapter_name + "\n"
        logging.info(f"{formatted_chapter_name}")

        # 使用PromptService构建完整提示词
        prompt = prompt_service.format_template(
            template_name,
            title=first_doc.metadata.get("title", "未知标题"),
            abstract=first_doc.metadata.get("abstract", "")[:500],
            keywords=", ".join(first_doc.metadata.get("keywords", "").split("，")[:5]),
            structure=first_doc.metadata.get("structure", ""),
            chapter_name=formatted_chapter_name,
        )

        # 调用LLM并约束输出格式
        schema = classification_json_schema()
        response = await async_llm_structured(prompt, data.model, schema)

        return response

    # 在主要流程中调用
    current_paper_type = stage  # 从步骤1获取的论文类型

    # 执行统一分类
    classification_result = await unified_chapter_classification(
        data, current_paper_type, chapter_groups
    )

    cleaned_response = re.sub(
        r"^```json\s*|\s*```$", "", classification_result.strip(), flags=re.MULTILINE
    )

    # 解析JSON响应
    classification_result = json.loads(cleaned_response)

    logging.info(f"{classification_result}")

    # 获取分类结果中的章节列表
    classified_chapters = classification_result.get("chapters", [])

    # 获取chapter_groups的键列表
    chapter_names = list(chapter_groups.keys())

    # 使用索引顺序来匹配阶段信息
    for idx, (chapter_name, docs) in enumerate(chapter_groups.items()):
        if not docs or any(
            exclude in chapter_name for exclude in ["摘要", "致谢", "参考文献", "谢"]
        ):
            continue

        # 根据索引获取对应的阶段类型
        if idx < len(classified_chapters):
            stage_type = classified_chapters[idx].get("stage", "其他内容")
        else:
            # 如果索引超出范围，使用默认值
            stage_type = "其他内容"
            logging.warning(f"章节 '{chapter_name}' 在分类结果中未找到对应阶段")

        # 在目标标签中查找匹配的阶段
        stage = "其他内容"
        for label in target_labels:
            if label == stage_type:
                stage = label
                break
            elif label in stage_type:
                stage = label
                break

        logging.info(f" {chapter_name}:分类结果： {stage}")

        # 为本章节的所有文档设置阶段
        for doc in docs:
            doc.metadata.update({"stage": stage})

    # 统一的章节Schema获取函数
    def get_chapter_schema(paper_type: str, chapter_type: str) -> Dict[str, Any]:
        """根据论文类型和章节类型获取对应的JSON Schema"""

        # 导入各模板模块的Schema函数
        from templates.template_engineering import (
            engineering_background_json_schema,
            engineering_conclusion_json_schema,
            engineering_data_processing_json_schema,
            engineering_experiment_json_schema,
            engineering_experiment_result_json_schema,
            engineering_introduction_json_schema,
            engineering_introduction_related_work_json_schema,
            engineering_methodology_json_schema,
            engineering_related_work_json_schema,
            engineering_result_analysis_json_schema,
        )
        from templates.template_method import (
            method_background_json_schema,
            method_conclusion_json_schema,
            method_data_processing_json_schema,
            method_experiment_json_schema,
            method_experiment_result_json_schema,
            method_introduction_json_schema,
            method_introduction_related_work_json_schema,
            method_methodology_json_schema,
            method_related_work_json_schema,
            method_result_analysis_json_schema,
        )
        from templates.template_theory import (
            theory_background_json_schema,
            theory_conclusion_json_schema,
            theory_data_processing_json_schema,
            theory_experiment_json_schema,
            theory_experiment_result_json_schema,
            theory_introduction_json_schema,
            theory_introduction_related_work_json_schema,
            theory_methodology_json_schema,
            theory_related_work_json_schema,
            theory_result_analysis_json_schema,
        )

        # 定义三类论文的Schema映射
        PAPER_TYPE_SCHEMA_MAP = {
            "理论研究": {
                "introduction": theory_introduction_json_schema,
                "related_work": theory_related_work_json_schema,
                "introduction_related_work": theory_introduction_related_work_json_schema,
                "background": theory_background_json_schema,
                "data_processing": theory_data_processing_json_schema,
                "methodology": theory_methodology_json_schema,
                "experiment": theory_experiment_json_schema,
                "result_analysis": theory_result_analysis_json_schema,
                "experiment_result": theory_experiment_result_json_schema,
                "conclusion": theory_conclusion_json_schema,
            },
            "方法创新": {
                "introduction": method_introduction_json_schema,
                "related_work": method_related_work_json_schema,
                "introduction_related_work": method_introduction_related_work_json_schema,
                "background": method_background_json_schema,
                "data_processing": method_data_processing_json_schema,
                "methodology": method_methodology_json_schema,
                "experiment": method_experiment_json_schema,
                "result_analysis": method_result_analysis_json_schema,
                "experiment_result": method_experiment_result_json_schema,
                "conclusion": method_conclusion_json_schema,
            },
            "工程实现": {
                "introduction": engineering_introduction_json_schema,
                "related_work": engineering_related_work_json_schema,
                "introduction_related_work": engineering_introduction_related_work_json_schema,
                "background": engineering_background_json_schema,
                "data_processing": engineering_data_processing_json_schema,
                "methodology": engineering_methodology_json_schema,
                "experiment": engineering_experiment_json_schema,
                "result_analysis": engineering_result_analysis_json_schema,
                "implementation_evaluation": engineering_experiment_result_json_schema,
                "conclusion": engineering_conclusion_json_schema,
            },
        }

        schema_func = PAPER_TYPE_SCHEMA_MAP.get(paper_type, {}).get(chapter_type)
        if schema_func:
            return schema_func()
        else:
            logging.warning(
                f"未找到论文类型 '{paper_type}' 章节类型 '{chapter_type}' 的Schema函数"
            )
            return {}

    # 步骤3：并发执行信息库检索建议
    logging.info(f"第三阶段：检索建议")
    update_stage("检索建议", "正在检索相关数据库...")

    async def generate_formatted_suggestions(
        prompt: str, stage: str, model: str
    ) -> str:
        """检索增强的建议生成 - 使用嵌入模型（支持本地和云端）"""

        try:
            # 1. 获取查询嵌入 - 使用工厂函数
            from config.config import (
                USE_CLOUD_EMBEDDING,
                get_collection_name,
                get_vectorstore_dir,
            )
            from services.llm_utils import create_embeddings

            embeddings = create_embeddings()

            # 2. 创建Chroma客户端 - 使用动态的数据库目录
            database_dir = get_vectorstore_dir()
            client = PersistentClient(
                path=str(database_dir), settings=Settings(anonymized_telemetry=False)
            )

            # 3. 获取集合 - 使用动态的集合名称
            collection_name = get_collection_name()

            # 检查集合是否存在
            existing_collections = [col.name for col in client.list_collections()]
            if collection_name not in existing_collections:
                logging.error(f"集合不存在: {collection_name}")
                logging.error(f"现有集合: {existing_collections}")
                raise HTTPException(
                    status_code=404,
                    detail=f"建议数据库集合不存在，请先初始化数据库。当前集合: {collection_name}",
                )

            collection = client.get_collection(collection_name)
            # logging.info(f"使用集合: {collection_name}, 数据库目录: {database_dir}")

            query_embedding = await embeddings.embed_query(prompt)

            # 执行查询
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=10,
                where={
                    "$or": [
                        {"stage": stage},
                        {"stage": "语言表达与术语"},
                        {"stage": "内容与逻辑"},
                    ]
                },
            )

            # 5. 格式化结果
            suggestions = []
            if results and "documents" in results and results["documents"]:
                for doc in results["documents"][0]:
                    suggestions.append(doc)

            return "\n".join(suggestions)

        except Exception as e:
            logging.error(f"检索增强失败: {str(e)}")
            return ""

    # 创建检索任务列表
    retrieval_tasks = []
    for chapter_name, docs in chapter_groups.items():
        if (
            not docs
            or "摘要" in chapter_name
            or "致谢" in chapter_name
            or "参考文献" in chapter_name
            or "谢" in chapter_name
        ):
            continue

        # 使用PromptService构建检索提示
        retrieval_prompt = prompt_service.format_template(
            "step3_retrieve_advice",
            title=docs[0].metadata.get("title", "未知标题"),
            abstract=docs[0].metadata.get("abstract", ""),
            keywords=docs[0].metadata.get("keywords", ""),
            chapter_name=docs[0].metadata.get("chapter", chapter_name),
            chapter_structure=docs[0].metadata.get("chapter_structure", ""),
            context=build_deep_context(docs)[:7500],
        )
        current_stage = docs[0].metadata.get("stage", "general")

        # 添加检索任务
        retrieval_tasks.append(
            generate_formatted_suggestions(retrieval_prompt, current_stage, data.model)
        )

    # 并发执行所有检索任务
    advice_suggestions = await asyncio.gather(*retrieval_tasks)

    # 步骤4.为各个章节构建信息结构体并初步评估
    logging.info(f"第四阶段：章节评估")
    update_stage("章节评估", "正在评估论文各章节内容...（此阶段时间可能较长）")

    # 定义三类论文的章节模板映射表 - 使用PromptService模板名称
    PAPER_TYPE_TEMPLATE_MAPPING = {
        "理论研究": {
            "引言/绪论": "step4_theory/introduction",
            "引言/绪论（包含相关工作）": "step4_theory/introduction_combined",
            "相关工作": "step4_theory/related_work",
            "背景知识": "step4_theory/background",
            "数据来源与处理": "step4_theory/data_processing",
            "模型与证明": "step4_theory/methodology",
            "实验分析": "step4_theory/experiment",
            "性能评估": "step4_theory/result_analysis",
            "实验分析与性能评估": "step4_theory/experiment_combined",
            "结论展望": "step4_theory/conclusion",
        },
        "方法创新": {
            "引言/绪论": "step4_method/introduction",
            "引言/绪论（包含相关工作）": "step4_method/introduction_combined",
            "相关工作": "step4_method/related_work",
            "背景知识": "step4_method/background",
            "数据来源与处理": "step4_method/data_processing",
            "方法构建": "step4_method/methodology",
            "实验验证": "step4_method/experiment",
            "结果分析": "step4_method/result_analysis",
            "实验验证与结果分析": "step4_method/experiment_combined",
            "结论展望": "step4_method/conclusion",
        },
        "工程实现": {
            "引言/绪论": "step4_engineering/introduction",
            "引言/绪论（包含相关工作）": "step4_engineering/introduction_combined",
            "相关工作": "step4_engineering/related_work",
            "背景知识": "step4_engineering/background",
            "数据来源与处理": "step4_engineering/data_processing",
            "系统设计": "step4_engineering/system_design",
            "系统实现": "step4_engineering/implementation",
            "系统评估": "step4_engineering/evaluation",
            "系统实现与评估": "step4_engineering/implementation_combined",
            "结论展望": "step4_engineering/conclusion",
        },
    }

    async_tasks = []
    for chapter_name, docs in chapter_groups.items():
        if (
            not docs
            or "摘要" in chapter_name
            or "致谢" in chapter_name
            or "参考文献" in chapter_name
            or "附录" in chapter_name
        ):
            continue

        first_doc = docs[0]
        context = build_deep_context(docs)

        # 获取论文类型和章节阶段
        paper_type = first_doc.metadata.get("type", "方法创新")  # 默认方法创新
        current_stage = first_doc.metadata.get("stage")

        # 根据论文类型和章节阶段选择对应的模板名称
        template_mapping = PAPER_TYPE_TEMPLATE_MAPPING.get(paper_type, {})
        template_name = template_mapping.get(current_stage)

        if not template_name:
            logging.warning(
                f"未找到论文类型 '{paper_type}' 章节 '{chapter_name}' 类型 '{current_stage}' 的模板"
            )
            continue

        try:

            advice_suggestions = await generate_formatted_suggestions(
                context, current_stage, data.model
            )

            # 使用PromptService格式化对应章节的模板
            prompt = prompt_service.format_template(
                template_name,
                chapter_structure=first_doc.metadata.get("chapter_structure", ""),
                suggestions=advice_suggestions,
                title=first_doc.metadata.get("title", "未知标题"),
                abstract=first_doc.metadata.get("abstract", "")[:500].replace(
                    "\n", " "
                ),
                keywords=", ".join(
                    first_doc.metadata.get("keywords", "").split("，")[:5]
                ),
                chapter_name=first_doc.metadata.get("chapter", ""),
                context=build_deep_context(docs)[:12000],
            )

            # 获取对应的章节类型标识符
            chapter_type_mapping = {
                "引言/绪论": "introduction",
                "相关工作": "related_work",
                "引言/绪论（包含相关工作）": "introduction_related_work",
                "背景知识": "background",
                "数据来源与处理": "data_processing",
                "模型与证明": "methodology",
                "方法构建": "methodology",
                "系统设计": "methodology",
                "实验分析": "experiment",
                "实验验证": "experiment",
                "系统实现": "experiment",
                "性能评估": "result_analysis",
                "结果分析": "result_analysis",
                "系统评估": "result_analysis",
                "实验分析与性能评估": "experiment_result",
                "实验验证与结果分析": "experiment_result",
                "系统实现与评估": "implementation_evaluation",
                "结论展望": "conclusion",
            }

            chapter_type = chapter_type_mapping.get(current_stage, "general")

            async_tasks.append((chapter_name, chapter_type, docs, prompt))

        except KeyError as e:
            logging.error(f"模板格式化失败，缺失键: {e}")
            continue
        except Exception as e:
            logging.error(f"模板处理异常: {e}")
            continue

    # 统一并发执行
    async_results = await asyncio.gather(
        *[
            async_llm_structured(
                prompt, data.model, get_chapter_schema(paper_type, chapter_type)
            )
            for _, chapter_type, _, prompt in async_tasks
        ]
    )

    # 信息结构体解析
    from datetime import datetime

    # 构建保存路径
    file_name = os.path.splitext(os.path.basename(data.file_path))[0]
    os.makedirs(evaluation_dir, exist_ok=True)

    information_file_path = os.path.join(evaluation_dir, "chapter_information.txt")

    # 获取文档元数据
    if user_documents and len(user_documents) > 0:
        first_doc = user_documents[0]
        title = first_doc.metadata.get("title", "N/A")
        student_name = first_doc.metadata.get("student_name", "N/A")
        student_id = first_doc.metadata.get("student_id", "N/A")
    else:
        title = "N/A"
        student_name = "N/A"
        student_id = "N/A"

    # 获取论文类型
    paper_type = current_paper_type  # 从之前的分类结果获取

    with open(information_file_path, "w", encoding="utf-8") as f:
        # 写入文件头部信息
        f.write("=" * 80 + "\n")
        f.write("📚 论文章节信息结构分析报告\n")
        f.write("=" * 80 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"用户名: {data.username}\n")
        f.write(f"文件: {file_name}\n")
        f.write(f"标题: {title}\n")
        f.write(f"学生姓名: {student_name}\n")
        f.write(f"学号: {student_id}\n")
        f.write(f"论文类型: {paper_type}\n\n")

        chapter_evaluation = {}

        # 获取对应的解析函数
        parser_func = get_parser_by_paper_type(paper_type)

        # 按章节顺序解析每个章节的JSON结构
        for idx, ((chapter_name, chapter_type, docs, _), response) in enumerate(
            zip(async_tasks, async_results)
        ):
            f.write("\n" + "=" * 60 + "\n")
            f.write(f"📖 第 {idx + 1} 章: {chapter_name}\n")
            f.write("=" * 60 + "\n")

            # 使用对应的解析函数解析JSON结构体
            parsed_data = parser_func(response, chapter_name)

            if "error" in parsed_data:
                f.write(f"❌ 解析失败: {parsed_data['error']}\n")
                f.write(f"原始响应内容:\n{parsed_data.get('raw_response', '无内容')}\n")
                continue

            chapter_key = f"chapter_{idx + 1}"
            chapter_evaluation[chapter_key] = {
                "chapter_data": parsed_data,
            }

            # 写入章节基本信息
            f.write(f"\n📋 章节基本信息:\n")
            f.write(f"  ├─ 章节名称: {parsed_data.get('chapter_name', 'N/A')}\n")
            f.write(f"  ├─ 章节类型: {parsed_data.get('chapter_type', 'N/A')}\n")

            # 写入章节摘要
            f.write(f"\n📝 章节内容摘要:\n")
            f.write(f"  {parsed_data.get('chapter_summary', '无摘要内容')}\n")

            # 写入章节评价
            f.write(f"\n💡 章节综合评价:\n")
            f.write(f"  {parsed_data.get('chapter_remark', '无评价内容')}\n")

            # 写入章节结构分析
            section_structure = parsed_data.get("section_structure", [])
            if section_structure:
                f.write(f"\n🏗️ 章节结构分析:\n")
                for i, section in enumerate(section_structure, 1):
                    f.write(f"  {i}. {section.get('section_title', '无标题')}\n")
                    if section.get("section_purpose"):
                        f.write(f"     目的: {section['section_purpose']}\n")
                    if section.get("key_points"):
                        f.write(f"     关键点: {', '.join(section['key_points'])}\n")
                    if section.get("weaknesses"):
                        f.write(f"     不足: {', '.join(section['weaknesses'])}\n")
                    f.write(f"     {'─' * 40}\n")

            # 写入提取信息
            extracted_info = parsed_data.get("extracted_info", {})
            if extracted_info:
                f.write(f"\n📊 提取信息:\n")
                field_descriptions = get_complete_field_descriptions()
                for field_name, value in extracted_info.items():
                    if value and value.strip():  # 只写入有实际内容的字段
                        description = field_descriptions.get(field_name, field_name)
                        f.write(f"  ├─ {description}: {value}\n")
            else:
                f.write(f"\n📊 提取信息: 无提取信息内容\n")
                logging.info(f"章节 {chapter_name} 无提取信息内容")

            # 写入专项评估
            f.write(f"\n🔍 专项评估结果:\n")
            evaluation_items = parsed_data.get("evaluation_items", {})
            field_descriptions = get_complete_field_descriptions()

            for field_name, evaluation_text in evaluation_items.items():
                if evaluation_text:
                    description = field_descriptions.get(field_name, field_name)
                    f.write(f"  ├─ {description}: {evaluation_text}\n")

            # 如果没有评估项，显示提示
            if not evaluation_items:
                f.write(f"  └─ 无专项评估内容\n")
                logging.info(f"专项评估解析错误")

            # 写入对评分的影响
            impact = parsed_data.get("scoring_impact")
            if impact:
                f.write(f"\n⚠️ 对评分的影响:\n")
                f.write(f"  {impact}\n")

            advice_list = parsed_data.get("advice", [])
            if advice_list:
                f.write(f"\n💡 修改建议:\n")
                for i, advice_item in enumerate(advice_list, 1):
                    position = advice_item.get("position", "未知位置")
                    suggestion = advice_item.get("suggestion", "")
                    if suggestion:  # 只写入有实际建议内容的条目
                        f.write(f"  {i}. [{position}] {suggestion}\n")
            else:
                f.write(f"\n💡 修改建议: 无具体修改建议\n")
                logging.info(f"章节 {chapter_name} 无修改建议内容")

        # 写入总结信息
        f.write("\n" + "=" * 80 + "\n")
        f.write("📊 分析总结\n")
        f.write("=" * 80 + "\n")
        f.write(f"总章节数: {len(async_tasks)}\n")
        f.write(
            f"成功解析: {len([r for r in async_results if 'error' not in str(r)])}\n"
        )
        f.write(f"论文类型: {paper_type}\n")
        f.write(f"生成完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    logging.info(f"第四阶段补充：专项评估项标签分析")

    # 导入分析函数
    from templates.template_parse import (
        analyze_evaluation_items,
        save_evaluation_analysis,
    )

    # 分析专项评估项标签
    evaluation_analysis = analyze_evaluation_items(chapter_evaluation)

    # 保存分析结果
    if "error" not in evaluation_analysis:
        save_evaluation_analysis(evaluation_dir, evaluation_analysis)
        logging.info(
            f"专项评估项标签分析完成，共{evaluation_analysis['total_evaluation_items']}个评估项"
        )
    else:
        logging.error(f"专项评估项标签分析失败: {evaluation_analysis['error']}")

    logging.info(f"章节信息结构已写入文件: {information_file_path}")

    # 步骤5.文章模块完整性和工作量评估
    logging.info(f"第五阶段：文章模块完整性和工作量评估")
    update_stage("工作量评估", "正在进行论文结构评估...")

    def get_workload_evaluation_template_name(paper_type: str) -> str:
        """根据论文类型返回对应的工作量评估模板名称"""
        template_names = {
            "理论研究": "step5_workload_theory",
            "方法创新": "step5_workload_method",
            "工程实现": "step5_workload_engineering",
        }
        return template_names.get(paper_type, "step5_workload_method")

    def extract_chapter_summaries_from_evaluation(chapter_evaluation: Dict) -> str:
        """从章节评估结果中提取章节摘要和结构信息"""
        summaries = []

        # 按章节顺序处理
        chapter_keys = sorted(
            [k for k in chapter_evaluation.keys() if k.startswith("chapter_")]
        )

        for chapter_key in chapter_keys:
            chapter_data = chapter_evaluation[chapter_key].get("chapter_data", {})

            # 提取章节基本信息
            chapter_name = chapter_data.get("chapter_name", "未知章节")
            chapter_summary = chapter_data.get("chapter_summary", "无摘要内容")

            # 构建章节摘要
            chapter_text = f"【{chapter_name}】\n"
            chapter_text += f"章节摘要：{chapter_summary}\n"

            # 提取章节结构
            section_structure = chapter_data.get("section_structure", [])
            if section_structure:
                chapter_text += "章节结构：\n"
                for i, section in enumerate(section_structure, 1):
                    section_title = section.get("section_title", "无标题")
                    section_purpose = section.get("section_purpose", "")
                    key_points = section.get("key_points", [])

                    chapter_text += f"  {i}. {section_title}\n"
                    if section_purpose:
                        chapter_text += f"     目的：{section_purpose}\n"
                    if key_points:
                        chapter_text += f"     关键点：{', '.join(key_points)}\n"

            summaries.append(chapter_text)

        return "\n\n".join(summaries)

    # 获取第一个文档的元数据
    first_doc = user_documents[0]
    main_meta = first_doc.metadata

    # 提取章节摘要信息
    chapter_summaries = extract_chapter_summaries_from_evaluation(chapter_evaluation)

    # 获取对应论文类型的模板名称
    template_name = get_workload_evaluation_template_name(paper_type)

    # 使用PromptService构建完整的提示词
    full_prompt = prompt_service.format_template(
        template_name,
        title=main_meta.get("title", "未知标题"),
        abstract=main_meta.get("abstract", "")[:500].replace("\n", " "),
        keywords=", ".join(main_meta.get("keywords", "").split("，")[:5]),
        structure=main_meta.get("structure", "无结构信息"),
        word_count_info=main_meta.get("word_count_info", "无字数统计"),
        chapter_summaries=chapter_summaries[:5000],
    )

    # 调用模型进行评估
    try:
        # 使用Pydantic模型生成JSON Schema
        from models.schemas import WorkloadEvaluationResponse

        schema = WorkloadEvaluationResponse.model_json_schema()
        workload_response = await async_llm_structured(full_prompt, data.model, schema)

        def parse_workload_evaluation(response_text: str) -> Dict[str, Any]:
            """解析工作量评估的结构体"""
            try:
                # 清理响应：移除 Markdown 代码块标记
                cleaned_response = re.sub(
                    r"^```json\s*|\s*```$",
                    "",
                    response_text.strip(),
                    flags=re.MULTILINE,
                )

                # 解析JSON响应
                result_data = json.loads(cleaned_response)

                # 构建解析结果
                parsed_workload = {
                    "structure_evaluation": result_data.get("structure_evaluation", {}),
                    "module_completeness_evaluation": result_data.get(
                        "module_completeness_evaluation", {}
                    ),
                    "workload_evaluation": result_data.get("workload_evaluation", {}),
                }

                return parsed_workload

            except json.JSONDecodeError as e:
                logging.error(f"工作量评估JSON解析失败: {e}")
                return {
                    "error": f"JSON解析失败: {str(e)}",
                    "raw_response": response_text,
                }
            except Exception as e:
                logging.error(f"工作量评估解析异常: {e}")
                return {"error": f"解析异常: {str(e)}", "raw_response": response_text}

        parsed_workload = parse_workload_evaluation(workload_response)

        if "error" in parsed_workload:
            raise Exception(f"工作量评估解析失败: {parsed_workload['error']}")

        # 保存工作量评估结果
        workload_file_path = os.path.join(evaluation_dir, "workload_evaluation.json")
        with open(workload_file_path, "w", encoding="utf-8") as f:
            json.dump(parsed_workload, f, ensure_ascii=False, indent=2)

        # 同时保存可读版本
        readable_file_path = os.path.join(
            evaluation_dir, "workload_evaluation_readable.txt"
        )
        with open(readable_file_path, "w", encoding="utf-8") as f:
            f.write("📊 文章模块完整性和工作量评估报告\n")
            f.write("=" * 80 + "\n")
            f.write(f"论文类型: {paper_type}\n")
            f.write(f"总字数: {main_meta.get('total_word_count', '未知')}\n")
            f.write(f"评估时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            # 结构评估
            f.write("一、论文结构评估\n")
            f.write("-" * 40 + "\n")
            structure_eval = parsed_workload.get("structure_evaluation", {})
            for key, item in structure_eval.items():
                score = item.get("score", 0)
                analysis = item.get("analysis", "")
                f.write(f"{key}: {score}分 - {analysis}\n")

            f.write("总评：")
            f.write("-" * 40 + "\n")
            summary_eval = parsed_workload.get("summary", {})
            f.write(f"{summary_eval.get('analysis', '无分析内容')}\n")

            # 工作量评估
            f.write("二、工作量评估\n")
            f.write("-" * 40 + "\n")
            workload_eval = parsed_workload.get("workload_evaluation", {})
            f.write(f"{workload_eval.get('analysis', '无分析内容')}\n")

        logging.info(f"工作量评估结果已写入: {workload_file_path}")

    except Exception as e:
        logging.error(f"工作量评估失败: {str(e)}")
        # 初始化默认的parsed_workload，避免后续代码引用未定义变量
        parsed_workload = {
            "structure_evaluation": {},
            "module_completeness_evaluation": {},
            "workload_evaluation": {"analysis": "工作量评估失败"},
        }

    # 步骤6.汇总修改建议
    logging.info(f"第六阶段：汇总修改建议")
    update_stage("汇总建议", "正在进行论文综合评估...")

    def extract_advice_from_chapters(chapter_evaluation: Dict) -> List[Dict]:
        """从章节评估结果中提取所有修改建议"""
        all_advice = []

        # 按章节顺序处理
        chapter_keys = sorted(
            [k for k in chapter_evaluation.keys() if k.startswith("chapter_")]
        )

        for chapter_key in chapter_keys:
            chapter_data = chapter_evaluation[chapter_key].get("chapter_data", {})
            chapter_name = chapter_data.get("chapter_name", "未知章节")
            advice_list = chapter_data.get("advice", [])

            for advice_item in advice_list:
                if advice_item.get("suggestion"):
                    all_advice.append(
                        {
                            "chapter": chapter_name,
                            "position": advice_item.get("position", ""),
                            "suggestion": advice_item.get("suggestion", ""),
                            "chapter_key": chapter_key,
                        }
                    )

        return all_advice

    async def run_step6_and_step7_parallel(
        chapter_evaluation: Dict,
        first_doc: Document,
        paper_type: str,
        data: Query,
        evaluation_dir: str,
        parsed_workload: Dict,
    ) -> tuple:
        """
        并行执行步骤6（汇总修改建议）和步骤7（综合评分）
        返回：(summary_advice_response, scorelist, score_summary)
        """

        async def step6_summary_advice() -> str:
            """步骤6：汇总修改建议"""
            try:
                # 提取所有章节的建议
                all_advice = extract_advice_from_chapters(chapter_evaluation)
                logging.info(
                    f"从 {len(chapter_evaluation)} 个章节中提取到 {len(all_advice)} 条修改建议"
                )

                if not all_advice:
                    return "未发现需要修改的问题。"

                # 构建章节建议字符串
                chapter_advices_str = ""
                for advice in all_advice:
                    chapter_advices_str += f"【{advice['chapter']}】{advice['position']}: {advice['suggestion']}\n"

                # 使用PromptService获取步骤6模板
                summary_prompt = prompt_service.format_template(
                    "step6_summary_advice",
                    title=first_doc.metadata.get("title", "未知标题"),
                    abstract=first_doc.metadata.get("abstract", "")[:500].replace(
                        "\n", " "
                    ),
                    keywords=", ".join(
                        first_doc.metadata.get("keywords", "").split("，")[:5]
                    ),
                    chapter_advices=chapter_advices_str[:8000],
                )

                # 调用模型生成汇总建议
                summary_advice_response = await async_llm(summary_prompt, data.model)

                # 清理响应内容
                pattern = r"<think>.*?</think>"
                summary_advice_response = re.sub(
                    pattern, "", summary_advice_response, flags=re.DOTALL
                )
                summary_advice_response = summary_advice_response.strip()

                # 保存汇总建议到文件
                summary_advice_file_path = os.path.join(
                    evaluation_dir, "summary_advice.txt"
                )
                with open(summary_advice_file_path, "w", encoding="utf-8") as f:
                    f.write("📝 论文修改建议汇总\n")
                    f.write("=" * 60 + "\n")
                    f.write(
                        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    )
                    f.write(
                        f"论文标题: {first_doc.metadata.get('title', '未知标题')}\n"
                    )
                    f.write(f"论文类型: {paper_type}\n")
                    f.write(f"总建议数: {len(all_advice)}\n")
                    f.write("=" * 60 + "\n\n")

                    f.write("💡 重要修改建议（AI汇总）:\n")
                    f.write("-" * 40 + "\n")
                    f.write(summary_advice_response)
                    f.write("\n\n")

                    f.write("📋 详细章节建议:\n")
                    f.write("-" * 40 + "\n")
                    for i, advice in enumerate(all_advice, 1):
                        f.write(f"{i}. 【{advice['chapter']}】{advice['position']}\n")
                        f.write(f"   建议: {advice['suggestion']}\n")
                        if i < len(all_advice):
                            f.write("   " + "-" * 30 + "\n")

                logging.info(f"汇总建议已写入文件: {summary_advice_file_path}")

                return summary_advice_response

            except Exception as e:
                logging.error(f"汇总修改建议失败: {str(e)}")
                return f"汇总建议生成失败: {str(e)}"

        async def step7_comprehensive_scoring() -> tuple:
            """步骤7：综合评分"""
            try:
                # 读取章节信息文件
                def read_file(file_path: str) -> str:
                    try:
                        with open(file_path, "r", encoding="utf-8") as file:
                            return file.read()
                    except Exception as e:
                        logging.error(f"读取文件失败: {str(e)}")
                        return ""

                evaluation_file_path = os.path.join(
                    evaluation_dir, "chapter_information.txt"
                )
                chapter_information = read_file(evaluation_file_path)

                # 使用PromptService获取步骤7模板
                main_meta = first_doc.metadata
                summary_prompt = prompt_service.format_template(
                    "step7_comprehensive_scoring",
                    workload_analysis=parsed_workload.get("workload_evaluation", {}),
                    title=main_meta.get("title", "未知标题"),
                    abstract=main_meta.get("abstract", "")[:500].replace("\n", " "),
                    keywords=", ".join(main_meta.get("keywords", "").split("，")[:5]),
                    chapter_evaluations=chapter_information,
                )

                def create_score_json_schema() -> Dict[str, Any]:
                    """创建分数评估的JSON Schema"""
                    return {
                        "type": "object",
                        "properties": {
                            "scores": {
                                "type": "object",
                                "properties": {
                                    "1": {
                                        "type": "number",
                                        "minimum": 0,
                                        "maximum": 100,
                                    },
                                    "2": {
                                        "type": "number",
                                        "minimum": 0,
                                        "maximum": 100,
                                    },
                                    "3": {
                                        "type": "number",
                                        "minimum": 0,
                                        "maximum": 100,
                                    },
                                    "4": {
                                        "type": "number",
                                        "minimum": 0,
                                        "maximum": 100,
                                    },
                                    "5": {
                                        "type": "number",
                                        "minimum": 0,
                                        "maximum": 100,
                                    },
                                    "6": {
                                        "type": "number",
                                        "minimum": 0,
                                        "maximum": 100,
                                    },
                                    "7": {
                                        "type": "number",
                                        "minimum": 0,
                                        "maximum": 100,
                                    },
                                    "8": {
                                        "type": "number",
                                        "minimum": 0,
                                        "maximum": 100,
                                    },
                                    "9": {
                                        "type": "number",
                                        "minimum": 0,
                                        "maximum": 100,
                                    },
                                    "10": {
                                        "type": "number",
                                        "minimum": 0,
                                        "maximum": 100,
                                    },
                                    "11": {
                                        "type": "number",
                                        "minimum": 0,
                                        "maximum": 100,
                                    },
                                    "12": {
                                        "type": "number",
                                        "minimum": 0,
                                        "maximum": 100,
                                    },
                                },
                                "required": [
                                    "1",
                                    "2",
                                    "3",
                                    "4",
                                    "5",
                                    "6",
                                    "7",
                                    "8",
                                    "9",
                                    "10",
                                    "11",
                                    "12",
                                ],
                            }
                        },
                        "required": ["scores"],
                    }

                # 调用模型生成综合评估
                response = await async_llm_structured(
                    summary_prompt, data.model, create_score_json_schema()
                )

                logging.info(f"第七阶段：综合评分")
                # 解析响应
                pattern = r"<think>.*?</think>"
                response = re.sub(pattern, "", response, flags=re.DOTALL)
                cleaned_response = re.sub(
                    r"^```json\s*|\s*```$", "", response.strip(), flags=re.MULTILINE
                )

                result_data = json.loads(cleaned_response)
                scores = result_data.get("scores", {})

                score_list = [scores.get(str(i), 0) for i in range(1, 13)]
                print("分数列表:", score_list)

                # 转换为等级分数 (0-3分制)
                endscores = []
                for i in range(12):
                    scorei = score_list[i]
                    if scorei > 85:  # 100-90为优秀
                        endscores.append(3)
                    elif scorei > 75:  # 89-75为良好
                        endscores.append(2)
                    elif scorei > 60:  # 74-60为一般
                        endscores.append(1)
                    else:  # 59以下为较差
                        endscores.append(0)

                print("等级分数:", endscores)

                # 从步骤5工作量统计中获取的结构评估项
                score_from_step5_list = []
                structure_eval = parsed_workload.get("structure_evaluation", {})
                for key, item in structure_eval.items():
                    score = item.get("score", 0)
                    score_from_step5_list.append(score)
                print("论文结构分数列表:", score_from_step5_list)

                # 转换为等级分数 (0-3分制)
                startscores = []
                for i in range(5):
                    scorei = score_from_step5_list[i]
                    if scorei >= 85:  # 100-90为优秀
                        startscores.append(3)
                    elif scorei >= 75:  # 89-75为良好
                        startscores.append(2)
                    elif scorei >= 60:  # 74-60为一般
                        startscores.append(1)
                    else:  # 59以下为较差
                        startscores.append(0)

                print("等级分数:", startscores)

                raw_scorelist = [0] * 19
                raw_scorelist[2:6] = checkbody(user_documents)[0]
                raw_scorelist[:4] = score_from_step5_list[:4]
                raw_scorelist[4] = raw_scorelist[4] * 15 + 50
                raw_scorelist[5] = score_from_step5_list[4]
                raw_scorelist[6:18] = score_list

                # 组装完整的scorelist (前6个分数由其他函数给出)
                scorelist = [0] * 19

                scorelist[0] = checkorder(processed_md)[0]
                scorelist[1] = checkabstract(user_documents[0])[0]
                scorelist[2:6] = checkbody(user_documents)[0]

                scorelist[6:18] = endscores  # 第7-18个分数

                print("基于人工逻辑的结构分数列表:", scorelist)

                scorelist[:4] = startscores[:4]
                scorelist[5] = startscores[4]

                print("基于模型语意的结构分数列表:", scorelist)

                def evaluate_paper_score(scorelist):

                    # 提取前18项评分
                    scores = scorelist[:18]

                    # 统计各等级数量
                    excellent_count = scores.count(3)  # 优秀
                    good_count = scores.count(2)  # 良好
                    average_count = scores.count(1)  # 一般
                    poor_count = scores.count(0)  # 较差

                    total_items = len(scores)

                    print(
                        f"统计结果: 优秀{excellent_count}项, 良好{good_count}项, 一般{average_count}项, 较差{poor_count}项"
                    )

                    # 确定等级
                    grade = ""
                    base_score = 0
                    max_score = 0

                    # 优秀等级条件: 1/2以上为优秀(>9项)，且一般的项<=2
                    if excellent_count > 9 and average_count + poor_count <= 3:
                        grade = "优秀"
                        base_score = 85
                        max_score = 98
                        # 优秀等级加分标准：优秀项加1分，良好项加0.5分

                        excellent_count -= 9

                        excellent_bonus = 1.5
                        good_bonus = 0.5
                        average_cost = -1
                        poor_cost = -1.5

                    # 良好等级条件: 12项及以上为优秀/良好，且一般的项<=4
                    elif excellent_count + good_count >= 12:
                        grade = "良好"
                        base_score = 70
                        max_score = 85
                        # 良好等级加分标准：优秀项加1.2分，良好项加0.8分
                        excellent_bonus = 1.2
                        good_bonus = 0.8
                        average_cost = -1
                        poor_cost = -1.5

                    elif excellent_count + good_count >= 8:
                        grade = "一般"
                        base_score = 60
                        max_score = 70
                        # 良好等级加分标准：优秀项加1.2分，良好项加0.8分
                        excellent_bonus = 1.2
                        good_bonus = 0.8
                        average_cost = 0
                        poor_cost = 0

                    # 其他情况为一般等级
                    else:
                        grade = "较差"
                        base_score = 50
                        max_score = 70
                        excellent_bonus = 1.5
                        good_bonus = 1
                        average_cost = 0
                        poor_cost = 0

                    # 计算加分
                    bonus = excellent_count * excellent_bonus + good_count * good_bonus
                    cost = average_count * average_cost + poor_count * poor_cost
                    # 计算最终分数
                    final_score = base_score + bonus + cost

                    # 不超过该等级的最大分数
                    # final_score = min(final_score, max_score)

                    # 四舍五入到整数
                    final_score = round(final_score)

                    print(
                        f"等级: {grade}, 基础分: {base_score}, 加分: {bonus:.1f}, 最终分数: {final_score}"
                    )
                    print(
                        f"加分标准: 优秀项+{excellent_bonus}分/项, 良好项+{good_bonus}分/项"
                    )

                    # 更新总分
                    scorelist[18] = final_score

                    return scorelist

                scorelist = evaluate_paper_score(scorelist)

                # 将综合评估结果写入文件
                result_file_path = os.path.join(evaluation_dir, "result.txt")
                with open(result_file_path, "w", encoding="utf-8") as f:
                    f.write("📝 综合评估报告\n")
                    f.write(f"Title: {first_doc.metadata.get('title', 'N/A')}\n")
                    f.write(f"Chapter Evaluations: {chapter_information[:1000]}...\n")
                    f.write(f"Summary Response: {response}\n")

                logging.info(f"综合评估结果已写入文件: {result_file_path}")

                raw_scorelist[18] = scorelist[18]
                return raw_scorelist, scorelist, "综合评分完成"

            except Exception as e:
                logging.error(f"综合评分失败: {str(e)}")
                return [], f"综合评分失败: {str(e)}"

        step6_task = step6_summary_advice()
        step7_task = step7_comprehensive_scoring()

        results = await asyncio.gather(step6_task, step7_task, return_exceptions=True)

        # 处理结果
        summary_advice_response = results[0]
        if isinstance(summary_advice_response, Exception):
            logging.error(f"步骤6执行失败: {summary_advice_response}")
            summary_advice_response = f"步骤6执行失败: {str(summary_advice_response)}"

        score_result = results[1]
        if isinstance(score_result, Exception):
            logging.error(f"步骤7执行失败: {score_result}")
            raw_scorelist, scorelist, score_summary = (
                [],
                f"步骤7执行失败: {str(score_result)}",
            )
        else:
            raw_scorelist, scorelist, score_summary = score_result

        return summary_advice_response, raw_scorelist, scorelist, score_summary

    summary_advice_response, raw_scorelist, scorelist, score_summary = (
        await run_step6_and_step7_parallel(
            chapter_evaluation=chapter_evaluation,
            first_doc=first_doc,
            paper_type=paper_type,
            data=data,
            evaluation_dir=evaluation_dir,
            parsed_workload=parsed_workload,
        )
    )
    logging.info(f"分数列表：{scorelist}")

    # checkbody(user_documents)[1] 表示 tmp_info, checkbody(user_documents)[2] 表示 refe 参考文献
    structure_eval = parsed_workload.get("structure_evaluation", {})
    advice_structure = ""
    for key, item in structure_eval.items():
        analysis = item.get("analysis", "")
        if analysis != "":
            advice_structure += analysis + "\n"
    advice_body = checkbody(user_documents)[2]
    advice_content = summary_advice_response

    logging.info(f"人工逻辑结构建议：{checkbody(user_documents)[1]}\n")
    logging.info(f"模型分析结构建议：{advice_structure}\n")

    # 在 async_multilevel_evaluation 函数的最后，return 之前添加以下代码

    # 构建评估结果文件
    result_file_path = os.path.join(evaluation_dir, "evaluation_result.txt")

    # 获取文档元数据
    first_doc = user_documents[0]
    metadata = {
        "student_name": first_doc.metadata.get("student_name", "N/A"),
        "student_id": first_doc.metadata.get("student_id", "N/A"),
        "paper_title": first_doc.metadata.get("title", "N/A"),
        "username": data.username,
        "file_name": os.path.basename(data.file_path),
        "process_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # 人工的结构评审项：checkbody(user_documents)[1]
    # 模型的结构评审项：advice_structure
    # 人工的参考文献评审项： advice_body

    # 构建评估数据
    evaluation_data = {
        "raw_scorelist": raw_scorelist,
        "score_list": scorelist,
        "advice_content": checkbody(user_documents)[1]
        + "\n"
        + advice_body
        + "\n"
        + advice_content,
        # "advice_content": advice_structure + '\n' + advice_body + '\n' + advice_content,
        "evaluation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # 构建专项评估项分析数据
    evaluation_analysis_data = {
        "total_evaluation_items": evaluation_analysis.get("total_evaluation_items", 0),
        "label_counts": evaluation_analysis.get("label_counts", {}),
        "label_percentages": evaluation_analysis.get("label_percentages", {}),
        "unlabeled_items": evaluation_analysis.get("unlabeled_items", 0),
        "unlabeled_percentage": evaluation_analysis.get("unlabeled_percentage", 0.0),
    }

    # 构建完整的结果字典
    result_dict = {
        "metadata": metadata,
        "evaluation_analysis": evaluation_analysis_data,
        "evaluation_data": evaluation_data,
    }

    # 写入文件
    with open(result_file_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)

    logging.info(f"评估结果已写入文件: {result_file_path}")

    return {
        "score": scorelist,
        "advice": advice_structure + "\n" + advice_body + "\n" + advice_content,
    }


@router.post("/qa")
async def run_evaluation(data: Query):
    try:
        # 生成唯一的用户任务ID
        user_id = f"{data.username}_{int(time.time())}"

        # 初始化进度信息
        progress_store[user_id] = {
            "stage": "准备开始",
            "progress": 0,
            "message": "正在初始化评估任务...",
            "timestamp": time.time(),
            "status": "running",
        }

        logging.info(f"创建评估任务: {user_id}")

        # 立即返回用户ID，让前端开始轮询
        # 实际评估在后台异步执行
        asyncio.create_task(execute_evaluation_task(data, user_id))

        return {
            "user_id": user_id,
            "status": "started",
            "message": "评估任务已开始，请使用 user_id 查询进度",
        }

    except Exception as e:
        logging.exception("创建评估任务失败")
        raise HTTPException(status_code=500, detail=f"创建评估任务失败: {str(e)}")


async def execute_evaluation_task(data: Query, user_id: str):
    """实际执行评估的后台任务"""
    try:
        # 更新进度
        update_progress(user_id, "开始评估", "正在加载和预处理文档...")

        # ========== 第一阶段：加载用户上传的文档 ==========

        logging.info(f"处理用户文件：{data.file_path}")

        userpath = os.path.join(UPLOAD_FOLDER, data.username)
        filename = os.path.basename(data.file_path)
        pdfpath = os.path.join(userpath, filename)
        logging.info("验证文件路径: %s", pdfpath)
        if not os.path.exists(pdfpath):
            raise HTTPException(status_code=407, detail="文件不存在")

        # 生成目标MD路径
        base_name = os.path.basename(data.file_path)
        md_filename = os.path.splitext(base_name)[0] + ".md"
        md_path = os.path.join(USER_MD_DIR, md_filename)

        # 文件存在性检查
        if not os.path.exists(md_path):
            logging.info(f"开始转换PDF: {data.file_path}")
            update_progress(user_id, "加载文档", "正在加载论文文件...")
            md_path = convert_pdf_to_markdown(data.file_path, data.username)
        else:
            logging.info(f"使用已存在的Markdown文件: {md_path}")

        # 加载并处理Markdown内容
        logging.info("加载并处理Markdown文档...")
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()

        processed_md = preprocess_markdown(md_content)

        # 执行结构化分块
        splitter = ChineseMarkdownSplitter()
        user_documents = splitter.split_text(processed_md)

        # 添加文档元数据
        for doc in user_documents:
            doc.metadata.update(
                {
                    "source": md_path,
                    "doc_type": "research_paper",
                    "processing_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )

        if not user_documents:
            raise HTTPException(status_code=400, detail="PDF文档为空或内容无效")

        # 将分块结果写入到文件中（测试用）
        file_name = os.path.splitext(os.path.basename(data.file_path))[0]

        # evaluation_dir = "/DATA/zhangtianyue_231300023/user_07test"

        evaluation_dir = os.path.join(USER_RESULT_DIR, file_name)

        os.makedirs(evaluation_dir, exist_ok=True)  # 确保目录存在

        output_chunks_file = os.path.join(evaluation_dir, "chunks_output.txt")

        # output_chunks_file = os.path.join(evaluation_dir, f"{file_name}.txt")

        with open(output_chunks_file, "w", encoding="utf-8") as f:
            f.write(f"Title: {doc.metadata.get('title', 'N/A')}\n")
            f.write(f"Student Name: {doc.metadata.get('student_name', 'N/A')}\n")
            f.write(f"Student ID: {doc.metadata.get('student_id', 'N/A')}\n")
            f.write(f"Abstract: {doc.metadata.get('abstract', 'N/A')}\n\n")
            f.write(f"Keywords: {doc.metadata.get('keywords', 'N/A')}\n\n")
            f.write(
                f"English Abstract: {doc.metadata.get('english_abstract', 'N/A')}\n\n"
            )
            f.write(
                f"English Keywords: {doc.metadata.get('english_keywords', 'N/A')}\n\n"
            )
            f.write("-" * 80 + "\n")
            f.write(f"Table of Contents: {doc.metadata.get('contents', 'N/A')}\n\n")

            f.write("-" * 80 + "\n")
            for idx, doc in enumerate(user_documents):
                f.write(
                    f"Chunk {idx + 1} [{doc.metadata.get('chapter', 'Unknown')} > {doc.metadata.get('section', 'Unknown')} > {doc.metadata.get('subsection', 'Unknown')}]\n"
                )
                f.write(f"Length: {len(doc.page_content)} chars\n")
                f.write(f"Content: {doc.page_content[:50]}...\n")
                f.write("-" * 80 + "\n")

        logging.info(f"分块结果已写入到文件: chunks_output.txt")

        chapter_structure_file = os.path.join(evaluation_dir, "chapter_structure.txt")
        with open(chapter_structure_file, "w", encoding="utf-8") as f:
            f.write(f"Title: {doc.metadata.get('title', 'N/A')}\n")
            f.write(f"Student Name: {doc.metadata.get('student_name', 'N/A')}\n")
            f.write(f"Student ID: {doc.metadata.get('student_id', 'N/A')}\n")
            f.write(f"Abstract: {doc.metadata.get('abstract', 'N/A')}\n\n")
            f.write(f"Keywords: {doc.metadata.get('keywords', 'N/A')}\n\n")
            f.write(
                f"Formula Words: {doc.metadata.get('total_formula_word_count', 'N/A')}\n\n"
            )
            f.write(
                f"Formula Ratio: {doc.metadata.get('formula_word_ratio', 'N/A')}\n\n"
            )
            f.write(f"Words:\n {doc.metadata.get('word_count_info', 'N/A')}\n\n")
            f.write(f"Structure:\n {doc.metadata.get('structure', 'N/A')}\n\n")

        logging.info(f"章节结构已写入到文件: chapter_structure.txt")

        start_evaluation = time.time()
        logging.info("开始多级评估流程...")

        # 元数据清洗函数
        from langchain_community.vectorstores.utils import filter_complex_metadata

        # 元数据清洗管道
        def process_metadata(docs: List[Document]) -> List[Document]:
            """双重清洗保障"""
            # 第一阶段：使用官方工具过滤复杂类型
            filtered_docs = filter_complex_metadata(docs)

            # 第二阶段：手动处理None值
            cleaned_docs = []
            for doc in filtered_docs:
                safe_meta = {
                    k: v if isinstance(v, (str, int, float, bool)) else str(v)
                    for k, v in doc.metadata.items()
                }
                cleaned_docs.append(
                    Document(page_content=doc.page_content, metadata=safe_meta)
                )
            return cleaned_docs

        # 按章节分组文档
        chapter_groups = {}
        for doc in process_metadata(user_documents):
            chapter = doc.metadata.get("chapter", "未分类")
            if chapter not in chapter_groups:
                chapter_groups[chapter] = []
            chapter_groups[chapter].append(doc)

        def chapter_info(chapter_name, docs):
            """返回包含分块数量和序号的章节信息"""
            return {
                "chapter_name": chapter_name,
                "total_chunks": len(docs),
                "chunks_used": [
                    {"index": idx, "preview": doc.page_content[:50]}
                    for idx, doc in enumerate(docs[:500])  # 返回前3个分块
                ],
                "meta": docs[0].metadata if docs else {},
            }

        update_progress(user_id, "开始评估", "开始多级评估流程...")
        multilevel_task = async_multilevel_evaluation(
            user_documents,
            processed_md,
            chapter_info,
            chapter_groups,
            data,
            evaluation_dir,
            user_id,
        )
        results = await asyncio.gather(multilevel_task, return_exceptions=True)
        # 检查执行结果
        if isinstance(results[0], Exception):
            logging.error(f"多级评估执行错误: {results[0]}")
            raise results[0]

        multilevel_result = results[0]

        logging.info(f"多级评估总耗时: {time.time() - start_evaluation:.2f}s")

        # 合并结果
        output = {
            "summary": multilevel_result.get("advice", ""),
            "section_scores": multilevel_result.get("score", []),
        }

        update_progress(user_id, "生成报告", "正在生成评估报告...")

        import subprocess

        output_file = os.path.join(evaluation_dir, "review_table.tex")

        logging.info(f"{output}")

        generate_latex_report(doc, output, output_file)
        xelatex_command = [
            "xelatex",
            "-interaction=nonstopmode",
            "-output-directory",
            evaluation_dir,
            output_file,
        ]

        try:
            subprocess.run(xelatex_command, check=True)
            print(f"Successfully compiled with XeLaTeX")
        except subprocess.CalledProcessError as e:
            print(f"XeLaTeX compilation error:", e)

        # 评估完成
        update_progress(user_id, "生成报告", "评估完成")

        # 存储最终结果
        progress_store[user_id]["status"] = "completed"
        progress_store[user_id]["result"] = {
            "summary": multilevel_result.get("advice", ""),
            "section_scores": multilevel_result.get("score", []),
        }

        logging.info(f"评估任务完成: {user_id}")

        # 在文档处理完成后，保存元数据到文件和进度存储
        if user_documents and len(user_documents) > 0:
            first_doc = user_documents[0]
            student_name = first_doc.metadata.get("student_name", "未知")
            student_id = first_doc.metadata.get("student_id", "未知")
            paper_title = first_doc.metadata.get("title", "未知")

            # 保存到进度存储
            progress_store[user_id].update(
                {
                    "student_name": student_name,
                    "student_id": student_id,
                    "paper_title": paper_title,
                    "evaluation_dir": evaluation_dir,
                }
            )

            # 保存元数据到文件
            metadata_file = os.path.join(evaluation_dir, "metadata.json")
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "student_name": student_name,
                        "student_id": student_id,
                        "paper_title": paper_title,
                        "username": data.username,
                        "file_name": os.path.basename(data.file_path),
                        "process_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            def save_evaluation_result(
                evaluation_dir: str, result_data: Dict, metadata: Dict
            ):
                """保存评估结果到文件"""
                result_file = os.path.join(evaluation_dir, "evaluation_result.json")

                evaluation_result = {
                    "metadata": metadata,
                    "evaluation_data": {
                        "score_list": result_data.get("score", []),
                        "advice_content": result_data.get("advice", ""),
                        "evaluation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                }

                with open(result_file, "w", encoding="utf-8") as f:
                    json.dump(evaluation_result, f, ensure_ascii=False, indent=2)

                return result_file

            save_evaluation_result(
                evaluation_dir,
                multilevel_result,
                {
                    "student_name": progress_store[user_id].get("student_name", "未知"),
                    "student_id": progress_store[user_id].get("student_id", "未知"),
                    "paper_title": progress_store[user_id].get("paper_title", "未知"),
                },
            )

            # ========== 同步调用建议检索功能 ==========
            logging.info("开始检索用户评审结果建议...")
            update_progress(user_id, "检索建议", "正在检索相关建议...")

            try:
                # 调用建议检索功能
                advice_result = await retrieve_advice(
                    {
                        "file_path": md_path,
                        "username": data.username,
                        "model": data.model,
                    }
                )

                # 将建议检索结果添加到进度存储中
                if advice_result.get("status") == "success":
                    progress_store[user_id]["advice_retrieval"] = {
                        "status": "success",
                        "output_dir": advice_result.get("output_dir", ""),
                        "txt_file": advice_result.get("txt_file", ""),
                        "json_file": advice_result.get("json_file", ""),
                        "total_chapters": advice_result.get("total_chapters", 0),
                        "total_advice": advice_result.get("total_advice", 0),
                    }
                    logging.info(
                        f"建议检索完成: {advice_result.get('total_advice', 0)} 条建议"
                    )
                else:
                    progress_store[user_id]["advice_retrieval"] = {
                        "status": "failed",
                        "error": "建议检索失败",
                    }
                    logging.warning("建议检索失败，但评估任务已完成")

            except Exception as advice_error:
                logging.error(f"建议检索异常: {str(advice_error)}")
                progress_store[user_id]["advice_retrieval"] = {
                    "status": "error",
                    "error": str(advice_error),
                }
                # 建议检索失败不影响评估任务的完成状态

    except Exception as e:
        logging.error(f"评估任务执行失败: {e}")
        update_progress(user_id, "错误", f"评估失败: {str(e)}")
        progress_store[user_id]["status"] = "error"

    finally:
        # 5分钟后清理进度信息（可选）
        async def cleanup():
            await asyncio.sleep(600)  # 5分钟
            if user_id in progress_store:
                del progress_store[user_id]
                logging.info(f"清理进度信息: {user_id}")

        asyncio.create_task(cleanup())


@router.get("/task/{user_id}")
async def get_task_result(user_id: str):
    """获取评估任务结果"""
    if user_id in progress_store:
        task_info = progress_store[user_id].copy()

        # 如果任务已完成，返回结果
        if task_info.get("status") == "completed" and "result" in task_info:
            return {
                "status": "completed",
                "result": task_info["result"],
                "progress_info": {
                    "stage": task_info.get("stage", ""),
                    "progress": task_info.get("progress", 0),
                    "message": task_info.get("message", ""),
                },
            }
        else:
            return {
                "status": task_info.get("status", "running"),
                "progress_info": {
                    "stage": task_info.get("stage", ""),
                    "progress": task_info.get("progress", 0),
                    "message": task_info.get("message", ""),
                },
            }
    else:
        return {"status": "not_found", "message": "任务不存在或已过期"}


@router.get("/task/{user_id}")
async def get_task_result(user_id: str):
    """获取评估任务结果"""
    if user_id in progress_store:
        task_info = progress_store[user_id].copy()

        # 如果任务已完成，返回结果
        if task_info.get("status") == "completed" and "result" in task_info:
            return {
                "status": "completed",
                "result": task_info["result"],
                "progress_info": {
                    "stage": task_info.get("stage", ""),
                    "progress": task_info.get("progress", 0),
                    "message": task_info.get("message", ""),
                },
            }
        else:
            return {
                "status": task_info.get("status", "running"),
                "progress_info": {
                    "stage": task_info.get("stage", ""),
                    "progress": task_info.get("progress", 0),
                    "message": task_info.get("message", ""),
                },
            }
    else:
        return {"status": "not_found", "message": "任务不存在或已过期"}


@router.get("/test_result/{user_id}")
async def get_test_result(user_id: str):
    """获取评估结果数据（从文件读取）"""
    if user_id in progress_store:
        task_info = progress_store[user_id]

        # 如果任务完成，尝试从文件读取
        if task_info.get("status") == "completed":
            evaluation_dir = task_info.get("evaluation_dir", "")

            if not evaluation_dir or not os.path.exists(evaluation_dir):
                return {"status": "error", "message": "评估目录不存在"}

            try:
                # 从文件读取评估结果
                result_file = os.path.join(evaluation_dir, "evaluation_result.json")
                if not os.path.exists(result_file):
                    return {"status": "error", "message": "评估结果文件不存在"}

                with open(result_file, "r", encoding="utf-8") as f:
                    evaluation_data = json.load(f)

                metadata = evaluation_data.get("metadata", {})
                eval_data = evaluation_data.get("evaluation_data", {})

                return {
                    "status": "success",
                    "data": {
                        "student_name": metadata.get("student_name", "未知"),
                        "student_id": metadata.get("student_id", "未知"),
                        "paper_title": metadata.get("paper_title", "未知"),
                        "score_list": eval_data.get("score_list", []),
                        "advice_content": eval_data.get("advice_content", ""),
                    },
                }

            except Exception as e:
                logging.error(f"读取评估结果文件失败: {str(e)}")
                return {"status": "error", "message": f"读取评估结果失败: {str(e)}"}
        else:
            return {
                "status": "processing",
                "message": "评估尚未完成",
                "progress": task_info.get("progress", 0),
            }
    else:
        return {"status": "not_found", "message": "任务不存在或已过期"}
