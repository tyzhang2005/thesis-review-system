import asyncio
import json
import logging
import os
import re
import time
import traceback
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from config.config import UPLOAD_FOLDER, USER_MD_DIR, USER_RESULT_DIR
from fastapi import APIRouter, HTTPException
from langchain_core.documents import Document
from models.models import Query
from models.schemas import HumanResultAnalysisResponse
from routers.file_handlers import convert_pdf_to_markdown
from services.llm_utils import async_llm, async_llm_structured
from services.markdown_processor import ChineseMarkdownSplitter
from services.prompt_service import PromptService

HUMAN_REVIEW_DIR = "/data/zhangtianyue/advice/2020_human"
USER_RESULT_DIR = "/data/zhangtianyue/advice/user_result"
CHAPTER_SUMMARY_DIR = "/data/zhangtianyue/advice/user_structure"
SOURCE = "NJUAI-2020"

logging.basicConfig(level=logging.INFO)
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# 初始化PromptService
prompt_service = PromptService()

router = APIRouter()


def preprocess_markdown(content: str) -> str:
    """预处理Markdown内容"""
    content = re.sub(r"(\n#+\s+[\u4e00-\u9fa5]+)\n\n#+", r"\1", content)
    content = re.sub(r"(\$\$[\s\S]*?\$\$)", r"<!--formula-->\1<!--/formula-->", content)
    return content


def find_human_review_file(student_id: str, student_name: str) -> Optional[str]:
    """
    根据学号和姓名查找人工评审表文件

    Args:
        student_id: 学号
        student_name: 姓名

    Returns:
        Optional[str]: 找到的文件路径，如果未找到返回None
    """
    if not os.path.exists(HUMAN_REVIEW_DIR):
        logging.warning(f"人工评审表文件夹不存在: {HUMAN_REVIEW_DIR}")
        return None

    # 清理学号和姓名中的空格
    clean_id = student_id.strip()
    clean_name = student_name.strip()

    # 按照学号_姓名.json的格式查找
    filename = f"{clean_id}_{clean_name}.json"
    file_path = os.path.join(HUMAN_REVIEW_DIR, filename)

    if os.path.exists(file_path):
        logging.info(f"找到人工评审表: {file_path}")
        return file_path

    logging.warning(f"未找到人工评审表: {filename}")
    return None


def load_human_review_data(
    student_id: str, student_name: str
) -> Optional[Dict[str, Any]]:
    """
    加载人工评审数据

    Args:
        student_id: 学号
        student_name: 姓名

    Returns:
        Optional[Dict]: 人工评审数据，如果找不到或解析失败返回None
    """
    # 查找人工评审表文件
    review_file = find_human_review_file(student_id, student_name)

    if not review_file:
        return None

    try:
        with open(review_file, "r", encoding="utf-8") as f:
            review_data = json.load(f)

        logging.info(f"成功加载人工评审数据")
        return review_data

    except Exception as e:
        logging.error(f"读取人工评审表失败: {e}")
        return None


def build_structured_context_for_chapter(
    chapter_name: str, docs: List[Document]
) -> str:
    """
    为章节构建结构化的上下文，包含块号信息

    Args:
        chapter_name: 章节名称
        docs: 该章节的所有文档块

    Returns:
        str: 结构化的上下文字符串
    """
    if not docs:
        return ""

    # 构建章节标题
    context_lines = [f"# {chapter_name}"]
    context_lines.append("=" * 60)

    # 添加每个块的详细信息
    for idx, doc in enumerate(docs):
        # 获取层级信息
        hierarchy_parts = []
        if doc.metadata.get("chapter"):
            hierarchy_parts.append(doc.metadata["chapter"])
        if doc.metadata.get("section"):
            hierarchy_parts.append(doc.metadata["section"])
        if doc.metadata.get("subsection"):
            hierarchy_parts.append(doc.metadata["subsection"])
        if doc.metadata.get("subsubsection"):
            hierarchy_parts.append(doc.metadata["subsubsection"])

        hierarchy_str = " > ".join(hierarchy_parts) if hierarchy_parts else "Unknown"

        # 构建块标题
        block_title = f"## 块 {idx + 1}/{len(docs)} [{hierarchy_str}]"
        context_lines.append(block_title)
        context_lines.append(f"长度: {len(doc.page_content)} 字符")
        context_lines.append("内容:")

        context_lines.append(doc.page_content)
        context_lines.append("-" * 40)

    return "\n".join(context_lines)


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
        context_lines.append(doc.page_content)
    return "\n".join(context_lines)


def get_chapter_structure_info(docs: List[Document]) -> Dict[str, Any]:
    """
    获取章节的结构化信息

    Args:
        docs: 章节的所有文档块

    Returns:
        Dict: 包含章节结构化信息的字典
    """
    if not docs:
        return {}

    first_doc = docs[0]

    # 收集所有层级信息
    all_hierarchies = []
    for doc in docs:
        hierarchy_parts = []
        if doc.metadata.get("chapter"):
            hierarchy_parts.append(doc.metadata["chapter"])
        if doc.metadata.get("section"):
            hierarchy_parts.append(doc.metadata["section"])
        if doc.metadata.get("subsection"):
            hierarchy_parts.append(doc.metadata["subsection"])
        if doc.metadata.get("subsubsection"):
            hierarchy_parts.append(doc.metadata["subsubsection"])

        if hierarchy_parts:
            all_hierarchies.append(" > ".join(hierarchy_parts))

    return {
        "chapter_name": first_doc.metadata.get("chapter", "Unknown"),
        "total_chunks": len(docs),
        "unique_hierarchies": list(set(all_hierarchies)),
        "content_length": sum(len(doc.page_content) for doc in docs),
    }


def extract_block_numbers(raw_text_str: str) -> List[int]:
    """
    从raw_text字符串中提取块号列表

    Args:
        raw_text_str: 包含块号列表的字符串，如"[3,4]"或"块号：[3,4]"

    Returns:
        List[int]: 块号列表
    """
    try:
        # 提取方括号内的内容
        match = re.search(r"\[([\d,\s]+)\]", raw_text_str)
        if match:
            numbers_str = match.group(1)
            # 分割并转换为整数
            numbers = [
                int(num.strip())
                for num in numbers_str.split(",")
                if num.strip().isdigit()
            ]
            return numbers
        return []
    except Exception as e:
        logging.error(f"提取块号失败: {e}")
        return []


def get_block_content_by_numbers(block_numbers: List[int], docs: List[Document]) -> str:
    """
    根据块号列表获取对应的块内容，并添加块号和上下文路径信息

    Args:
        block_numbers: 块号列表
        docs: 该章节的所有文档块

    Returns:
        str: 合并后的块内容，包含块号和层级信息
    """
    content_parts = []
    for block_num in block_numbers:
        # 块号是从1开始的，但列表索引是从0开始的
        idx = block_num - 1
        if 0 <= idx < len(docs):
            doc = docs[idx]

            # 获取层级信息
            hierarchy_parts = []
            if doc.metadata.get("chapter"):
                hierarchy_parts.append(doc.metadata["chapter"])
            if doc.metadata.get("section"):
                hierarchy_parts.append(doc.metadata["section"])
            if doc.metadata.get("subsection"):
                hierarchy_parts.append(doc.metadata["subsection"])
            if doc.metadata.get("subsubsection"):
                hierarchy_parts.append(doc.metadata["subsubsection"])

            hierarchy_str = (
                " > ".join(hierarchy_parts) if hierarchy_parts else "Unknown"
            )

            # 构建带块号信息的标题
            block_title = f"## 块 {block_num}/{len(docs)} [{hierarchy_str}]"

            # 将标题和内容合并
            block_content = f"{block_title}\n{doc.page_content}"
            content_parts.append(block_content)
        else:
            logging.warning(f"块号 {block_num} 超出范围，该章节共有 {len(docs)} 个块")

    return "\n\n" + ("\n\n" + "-" * 60 + "\n\n").join(content_parts) + "\n\n"


def load_chapter_summary_json(student_id: str, student_name: str) -> Optional[Dict]:
    """
    加载章节概要JSON文件

    Args:
        student_id: 学号
        student_name: 姓名

    Returns:
        Optional[Dict]: 章节概要数据，如果未找到返回None
    """
    # 构造文件名
    filename = f"{student_id}_{student_name}.json"
    file_path = os.path.join(CHAPTER_SUMMARY_DIR, filename)

    if not os.path.exists(file_path):
        logging.warning(f"未找到章节概要文件: {filename}")
        return None

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            chapter_data = json.load(f)

        logging.info(f"成功加载章节概要文件: {filename}")
        return chapter_data

    except Exception as e:
        logging.error(f"读取章节概要文件失败: {e}")
        return None


def format_chapter_summary_basic(chapter_data: Dict) -> str:
    """
    格式化章节概要（基础版本）- 只包含章节基本信息和章节概括

    Args:
        chapter_data: 章节数据
        exclude_chapter_name: 要排除的章节名称

    Returns:
        str: 格式化后的章节概要字符串
    """
    if not chapter_data or "chapters" not in chapter_data:
        return "无章节概要信息"

    formatted_summaries = []

    for chapter in chapter_data["chapters"]:
        chapter_name = chapter.get("chapter_name", "")

        # 获取章节基本信息
        basic_info = chapter.get("chapter_basic_info", {})
        chapter_summary = chapter.get("chapter_summary", "")

        # 格式化
        chapter_str = f"【{chapter_name}】\n"

        if chapter_summary:
            # 截断过长的摘要
            if len(chapter_summary) > 500:
                chapter_str += f"章节概括: {chapter_summary[:500]}...\n"
            else:
                chapter_str += f"章节概括: {chapter_summary}\n"

        formatted_summaries.append(chapter_str)

    if not formatted_summaries:
        return "无其他章节概要信息"

    return "\n".join(formatted_summaries)


def format_chapter_summary_detailed(chapter_data: Dict) -> str:
    """
    格式化章节概要（详细版本）- 包含章节基本信息、章节概括和小节信息

    Args:
        chapter_data: 章节数据

    Returns:
        str: 格式化后的章节概要字符串
    """
    if not chapter_data or "chapters" not in chapter_data:
        return "无章节概要信息"

    formatted_summaries = []

    for chapter in chapter_data["chapters"]:
        chapter_name = chapter.get("chapter_name", "")

        # 获取章节基本信息
        basic_info = chapter.get("chapter_basic_info", {})
        chapter_summary = chapter.get("chapter_summary", "")
        section_structure = chapter.get("section_structure", [])

        # 格式化
        chapter_str = f"【{chapter_name}】\n"

        if chapter_summary:
            # 截断过长的摘要
            if len(chapter_summary) > 300:
                chapter_str += f"章节概括: {chapter_summary[:300]}...\n"
            else:
                chapter_str += f"章节概括: {chapter_summary}\n"

        # 添加小节信息
        if section_structure:
            chapter_str += "章节结构:\n"
            for section in section_structure:
                section_title = section.get("section_title", "")
                purpose = section.get("purpose", "")
                key_points = section.get("key_points", [])

                if section_title:
                    chapter_str += f"  {section_title}\n"

                if purpose:
                    chapter_str += f"    目的: {purpose}\n"

                if key_points:
                    chapter_str += f"    关键点:\n"
                    for i, point in enumerate(key_points[:3]):  # 只取前3个关键点
                        if i < 3:
                            chapter_str += f"    - {point}\n"
                        else:
                            chapter_str += f"    - ...\n"
                            break

                chapter_str += "\n"

        formatted_summaries.append(chapter_str)

    if not formatted_summaries:
        return "无其他章节概要信息"

    return "\n".join(formatted_summaries)


def format_current_chapter_detailed(
    chapter_data: Dict, current_chapter_name: str
) -> str:
    """
    格式化当前章节的详细概要 - 只包含指定章节的详细信息（包括小节信息）

    Args:
        chapter_data: 章节数据
        current_chapter_name: 当前章节名称

    Returns:
        str: 格式化后的当前章节详细概要字符串
    """
    if not chapter_data or "chapters" not in chapter_data:
        return "无章节概要信息"

    # 查找当前章节
    for chapter in chapter_data["chapters"]:
        chapter_name = chapter.get("chapter_name", "")

        if chapter_name == current_chapter_name:
            # 获取章节基本信息
            basic_info = chapter.get("chapter_basic_info", {})
            chapter_summary = chapter.get("chapter_summary", "")
            section_structure = chapter.get("section_structure", [])

            # 格式化
            chapter_str = f"【{chapter_name}】\n"

            if chapter_summary:
                # 对于当前章节，不截断摘要
                chapter_str += f"章节概括: {chapter_summary}\n"

            # 添加小节信息
            if section_structure:
                chapter_str += "章节结构:\n"
                for section in section_structure:
                    section_title = section.get("section_title", "")
                    purpose = section.get("purpose", "")
                    key_points = section.get("key_points", [])

                    if section_title:
                        chapter_str += f"  {section_title}\n"

                    if purpose:
                        chapter_str += f"    目的: {purpose}\n"

                    if key_points:
                        chapter_str += f"    关键点:\n"
                        for i, point in enumerate(key_points):
                            chapter_str += f"    - {point}\n"

                    chapter_str += "\n"

            return chapter_str

    # 如果没找到当前章节，返回空字符串
    return f"未找到章节【{current_chapter_name}】的概要信息"


def rewrite_advice_item(
    advice_item: Dict,
    chapter_docs: List[Document],
    global_metadata: Dict,
    basic_chapter_summary: str,
    detailed_current_chapter_summary: str,
    paper_type: str = "",
    chapter_type: str = "",
) -> Dict:
    """
    重写单个建议项，添加新字段并填充原始文本

    Args:
        advice_item: 原始建议项
        chapter_docs: 该章节的所有文档块
        global_metadata: 全局元数据
        basic_chapter_summary: 所有章节的基础概要
        detailed_current_chapter_summary: 当前章节的详细概要

    Returns:
        OrderedDict: 按顺序排列的重写后建议项
    """
    # 提取块号并获取原文内容
    raw_text_str = advice_item.get("raw_text", "")
    block_numbers = extract_block_numbers(raw_text_str)

    if block_numbers:
        # 获取对应的块内容
        block_content = get_block_content_by_numbers(block_numbers, chapter_docs)
        raw_text_value = block_content
    else:
        # 如果没有块号，保留原内容
        logging.warning(f"未找到块号信息，保留原raw_text: {raw_text_str[:50]}...")
        raw_text_value = raw_text_str

    # 按照指定顺序创建OrderedDict
    rewritten_item = OrderedDict(
        [
            ("source", SOURCE),  # 第一个字段
            ("title", global_metadata.get("paper_title", "")),  # 第二个字段
            ("student_id", global_metadata.get("student_id", "")),  # 第三个字段
            ("student_name", global_metadata.get("student_name", "")),  # 可以添加更多
            ("structure", global_metadata.get("structure", "")),  # 第四个字段
            # 新增字段
            ("paper_type", paper_type),
            ("chapter_type", chapter_type),
            ("basic_chapter_summary", basic_chapter_summary),
            ("detailed_current_chapter_summary", detailed_current_chapter_summary),
            ("position", advice_item.get("position", "")),
            ("raw_text", raw_text_value),
            ("type", advice_item.get("type", "")),
            ("context", advice_item.get("context", "")),
            ("suggestion", advice_item.get("suggestion", "")),
            ("chain_of_thought", advice_item.get("chain_of_thought", "")),
            ("scoring_impact", advice_item.get("scoring_impact", "")),
        ]
    )

    return rewritten_item


# ==================== 论文类型和章节分类 ====================


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


async def classify_paper_type(metadata: Dict, chapter_groups: Dict, model: str) -> str:
    """
    使用 LLM 对论文进行类型分类

    Returns:
        论文类型: "理论研究", "方法创新", "工程实现"
    """
    classify_template = """
你是一个专业的本科毕业论文评估助手，接下来你将看到一篇计算机学科的本科毕业论文的标题、摘要、关键词和章节结构，
请基于以上信息，将论文归为理论研究、方法创新、系统实现中的一类，
- 若聚焦于理论证明、复杂性分析，则为「理论研究型」
- 若聚焦于提出新算法、新模型以提升性能，则为「方法创新型」 
- 若聚焦于构建一个完整的系统、平台或应用解决方案，则为「工程实现型」
你只能从以下选项中选择一个最匹配的类别名称：

理论研究 | 方法创新 | 工程实现

【论文元数据】
标题：{title}
摘要：{abstract}
关键词：{keywords}

【章节结构】
{structure}
"""

    # 获取第一个章节的文档来获取元数据
    first_doc = next(iter(chapter_groups.values()))[0] if chapter_groups else None

    classify_prompt = classify_template.format(
        title=metadata.get("paper_title", "未知标题"),
        abstract=(
            metadata.get("abstract", "")[:500].replace("\n", " ")
            if metadata.get("abstract")
            else ""
        ),
        keywords=(
            ", ".join(metadata.get("keywords", "").split("，")[:5])
            if metadata.get("keywords")
            else ""
        ),
        structure=(
            metadata.get("structure", "")[:1500]
            if metadata.get("structure")
            else "无结构信息"
        ),
    )

    response = await async_llm(classify_prompt, model)

    target_labels = ["理论研究", "方法创新", "工程实现"]
    paper_type = "方法创新"  # 默认值
    for label in target_labels:
        if label in response:
            paper_type = label
            break

    logging.info(f"论文类型分类结果: {paper_type}")
    return paper_type


async def classify_chapter_types(
    metadata: Dict, chapter_groups: Dict, paper_type: str, model: str
) -> Dict[str, str]:
    """
    使用 LLM 对章节进行类型分类

    Returns:
        Dict[str, str]: 章节名 -> 章节类型 的映射
    """
    PAPER_TYPE_CLASSIFY_TEMPLATES = {
        "理论研究": """
你是一个专业的本科毕业论文评估助手，请基于论文章节结构，
以章节为单元，进行章节阶段分类，为每个章节分配对应的章节阶段标签。
本文属于计算机学科的理论研究型论文，主要关注理论证明、模型分析和复杂性分析。

【任务要求】

对于每个章节，你需要从以下选项中选择最匹配的阶段名称，注意json结构体的"stage"字段只能从以下标签中选择其中一个：
引言/绪论|引言/绪论（包含相关工作）|相关工作|背景知识|数据来源与处理方式|模型与证明|实验分析|性能评估|实验分析与性能评估|结论展望

请特别注意区分：
1. 我们有"引言/绪论（包含相关工作）"和"实验分析与性能评估"两个合并章节的标签，旨在为不同结构的论文进行精确匹配，当一个章节同时包含如下所述的两部分内容时，需要选择该标签
2. 如果绪论章节包含相关工作，请选择"引言/绪论（包含相关工作）"，如果绪论和相关工作单独成章，请对对应的章节分别选择"引言/绪论"和"相关工作"，注意默认的"引言/绪论"标签不包含相关工作
3. 如果实验章节同时包含实验设计和结果分析，应选择"实验分析与性能评估"，如果章节仅包含实验设计，选择"实验分析"；仅包含结果分析，选择"性能评估"
4. 如果全文共计4-5章，需要优先考虑存在"引言/绪论（包含相关工作）"或"实验分析与性能评估"这两个合并章节的情况；如果全文总章数在6章及以上，则无需优先考虑

【论文元数据】
标题：{title}
摘要：{abstract}
关键词：{keywords}
论文结构：{structure}
章节标题：{chapter_name}

【输出要求】
请严格按照以下JSON格式输出，不要包含任何其他内容：
{{
    "chapters": [
        {{"chapter_name": "章节1名称", "stage": "章节阶段"}},
        {{"chapter_name": "章节2名称", "stage": "章节阶段"}}
    ]
}}
""",
        "方法创新": """
你是一个专业的本科毕业论文评估助手，请基于论文章节结构，
以章节为单元，进行章节阶段分类，为每个章节分配对应的章节阶段标签。
本文属于计算机学科的方法创新型论文，主要关注提出新算法、新模型以提升性能。

【任务要求】

对于每个章节，你需要从以下选项中选择最匹配的阶段名称，注意json结构体的"stage"字段只能从以下标签中选择其中一个：
引言/绪论|引言/绪论（包含相关工作）|相关工作|背景知识|数据来源与处理方式|方法构建|实验验证|结果分析|实验验证与结果分析|结论展望

请特别注意区分：
1. 我们有"引言/绪论（包含相关工作）"和"实验验证与结果分析"两个合并章节的标签，旨在为不同结构的论文进行精确匹配，当一个章节同时包含如下所述的两部分内容时，需要选择该标签
2. 如果绪论章节包含相关工作，请选择"引言/绪论（包含相关工作）"，如果绪论和相关工作单独成章，请对对应的章节分别选择"引言/绪论"和"相关工作"，注意默认的"引言/绪论"标签不包含相关工作
3. 如果实验章节同时包含实验验证和结果分析，应选择"实验验证与结果分析"，如果章节仅包含实验验证，选择"实验验证"；仅包含结果分析，选择"结果分析"
4. 如果全文共计4-5章，需要优先考虑存在"引言/绪论（包含相关工作）"或"实验验证与结果分析"这两个合并章节的情况；如果全文总章数在6章及以上，则无需优先考虑

【论文元数据】
标题：{title}
摘要：{abstract}
关键词：{keywords}
论文结构：{structure}
章节标题：{chapter_name}

【输出要求】
请严格按照以下JSON格式输出，不要包含任何其他内容：
{{
    "chapters": [
        {{"chapter_name": "章节1名称", "stage": "章节阶段"}},
        {{"chapter_name": "章节2名称", "stage": "章节阶段"}}
    ]
}}
""",
        "工程实现": """
你是一个专业的本科毕业论文评估助手，请基于论文章节结构，
以章节为单元，进行章节阶段分类，为每个章节分配对应的章节阶段标签。
本文属于计算机学科的工程实现型论文，主要关注系统设计、开发和评估。

【任务要求】

对于每个章节，你需要从以下选项中选择最匹配的阶段名称，注意json结构体的"stage"字段只能从以下标签中选择其中一个：
引言/绪论|引言/绪论（包含相关工作）|相关工作|背景知识|数据来源与处理方式|系统设计|系统实现|系统评估|系统实现与评估|结论展望

请特别注意区分：
1. 我们有"引言/绪论（包含相关工作）"和"系统实现与评估"两个合并章节的标签，旨在为不同结构的论文进行精确匹配，当一个章节同时包含如下所述的两部分内容时，需要选择该标签
2. 如果绪论章节包含相关工作，请选择"引言/绪论（包含相关工作）"，如果绪论和相关工作单独成章，请对对应的章节分别选择"引言/绪论"和"相关工作"，注意默认的"引言/绪论"标签不包含相关工作
3. 如果系统章节同时包含实现和评估，应选择"系统实现与评估"，如果章节仅包含系统实现，选择"系统实现"；仅包含系统评估，选择"系统评估"
4. 如果全文共计4-5章，需要优先考虑存在"引言/绪论（包含相关工作）"或"系统实现与评估"这两个合并章节的情况；如果全文总章数在6章及以上，则无需优先考虑

【论文元数据】
标题：{title}
摘要：{abstract}
关键词：{keywords}
论文结构：{structure}
章节标题：{chapter_name}

【输出要求】
请严格按照以下JSON格式输出，不要包含任何其他内容：
{{
    "chapters": [
        {{"chapter_name": "章节1名称", "stage": "章节阶段"}},
        {{"chapter_name": "章节2名称", "stage": "章节阶段"}}
    ]
}}
""",
    }

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

    classify_template = PAPER_TYPE_CLASSIFY_TEMPLATES.get(paper_type)
    target_labels = PAPER_TYPE_STAGE_LABELS.get(paper_type, [])

    if not classify_template:
        logging.error(f"未知的论文类型: {paper_type}")
        return {}

    # 获取第一篇文档的元数据
    first_doc = next(iter(chapter_groups.values()))[0] if chapter_groups else None
    if not first_doc:
        return {}

    # 构建所有章节信息字符串
    formatted_chapter_name = ""
    for chapter_name, docs in chapter_groups.items():
        if not docs or any(
            exclude in chapter_name for exclude in ["摘要", "致谢", "参考文献", "谢"]
        ):
            continue
        formatted_chapter_name += chapter_name + "\n"

    # 构建完整提示词
    prompt = classify_template.format(
        title=metadata.get("paper_title", "未知标题"),
        abstract=metadata.get("abstract", "")[:500],
        keywords=", ".join(metadata.get("keywords", "").split("，")[:5]),
        structure=metadata.get("structure", ""),
        chapter_name=formatted_chapter_name,
    )

    # 调用LLM并约束输出格式
    schema = classification_json_schema()
    response = await async_llm_structured(prompt, model, schema)

    # 清理响应
    cleaned_response = re.sub(
        r"^```json\s*|\s*```$", "", response.strip(), flags=re.MULTILINE
    )
    classification_result = json.loads(cleaned_response)

    # 获取分类结果中的章节列表
    classified_chapters = classification_result.get("chapters", [])

    # 构建章节名 -> 章节类型 的映射
    chapter_stage_map = {}
    chapter_names = [
        name
        for name in chapter_groups.keys()
        if not any(exclude in name for exclude in ["摘要", "致谢", "参考文献", "谢"])
    ]

    for idx, chapter_name in enumerate(chapter_names):
        if idx < len(classified_chapters):
            stage_type = classified_chapters[idx].get("stage", "其他")
        else:
            stage_type = "其他"
            logging.warning(f"章节 '{chapter_name}' 在分类结果中未找到对应阶段")

        # 在目标标签中查找匹配的阶段
        stage = "其他"
        for label in target_labels:
            if label == stage_type or label in stage_type:
                stage = label
                break

        chapter_stage_map[chapter_name] = stage
        logging.info(f"章节 '{chapter_name}' 类型: {stage}")

    # 处理特殊章节
    for chapter_name in chapter_groups.keys():
        if any(
            exclude in chapter_name for exclude in ["摘要", "致谢", "参考文献", "谢"]
        ):
            if "附录" in chapter_name or "appendix" in chapter_name.lower():
                chapter_stage_map[chapter_name] = "附录"
            elif "摘要" in chapter_name or "abstract" in chapter_name.lower():
                chapter_stage_map[chapter_name] = "摘要"
            elif "致谢" in chapter_name or "谢" in chapter_name:
                chapter_stage_map[chapter_name] = "致谢"
            elif "参考" in chapter_name or "文献" in chapter_name:
                chapter_stage_map[chapter_name] = "参考文献"

    return chapter_stage_map


@router.post("/human_analysis")
async def run_human_analysis(data: Query):
    """
    基于人工评审结果生成详细分析报告
    输入与evaluation.py完全相同的Query对象

    流程:
    1. 解析PDF获取论文内容
    2. 从论文元数据中获取学号和姓名
    3. 根据学号和姓名在HUMAN_REVIEW_DIR中查找人工评审表
    4. 如果找不到或解析失败，直接返回错误
    5. 加载人工评审数据
    6. 对论文进行分块和章节处理
    7. 并发生成详细分析报告
    """
    try:
        start_time = time.time()
        logging.info(f"开始人工评审分析: {data.file_path}, 用户: {data.username}")

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

        if not user_documents:
            raise HTTPException(status_code=400, detail="PDF文档为空或内容无效")

        # ========== 第二阶段：从论文元数据中获取学号和姓名 ==========

        # 获取第一个文档的元数据
        first_doc = user_documents[0]
        metadata = {
            "student_name": first_doc.metadata.get("student_name", ""),
            "student_id": first_doc.metadata.get("student_id", ""),
            "paper_title": first_doc.metadata.get("title", "N/A"),
            "abstract": first_doc.metadata.get("abstract", "N/A"),
            "keywords": first_doc.metadata.get("keywords", "N/A"),
            "english_abstract": first_doc.metadata.get("english_abstract", "N/A"),
            "english_keywords": first_doc.metadata.get("english_keywords", "N/A"),
            "contents": first_doc.metadata.get("contents", "N/A"),
            "structure": first_doc.metadata.get("structure", "N/A"),
            "word_count_info": first_doc.metadata.get("word_count_info", "N/A"),
        }

        student_name = metadata["student_name"]
        student_id = metadata["student_id"]

        if not student_name or not student_id:
            raise HTTPException(
                status_code=400, detail="论文元数据中缺少学号或姓名信息"
            )

        logging.info(f"从论文中获取到作者信息: 学号={student_id}, 姓名={student_name}")

        # ========== 第三阶段：加载人工评审数据 ==========

        logging.info(f"开始加载人工评审数据...")
        human_review_data = load_human_review_data(student_id, student_name)

        if not human_review_data:
            raise HTTPException(
                status_code=404,
                detail=f"未找到学号{student_id}姓名{student_name}的人工评审表",
            )

        # 提取评审信息
        score_list = human_review_data.get("evaluation_data", {}).get("score_list", [])
        human_advice = human_review_data.get("evaluation_data", {}).get("advice", "")

        if not score_list or not human_advice:
            raise HTTPException(status_code=400, detail="人工评审表格式不正确")

        def parse_and_rewrite_score_list(score_list: List[int]) -> str:
            """
            解析和重写人工评审分数列表

            Args:
                score_list: 19项分数列表，前18项为单项得分(3,2,1,0)，第19项为百分制总分

            Returns:
                str: 重写后的分数字符串
            """
            if not score_list or len(score_list) != 19:
                return "分数列表格式不正确"

            # 单项名称列表
            structure_items = [
                "结构完整性评估",
                "摘要和关键词规范性",
                "目录规范性",
                "章节规范性",
                "参考文献格式规范性",
                "致谢规范性",
            ]

            content_items = [
                "选题契合度",
                "选题工作量适宜度",
                "选题学术价值",
                "文献检索和分析能力",
                "知识综合应用和研究深度",
                "专业方法工具运用",
                "专业技能和实践能力",
                "技术应用和外语能力",
                "创新性",
                "论证严谨性和科学性",
                "论文结构和语言表达",
                "成果价值",
            ]

            # 等级映射
            grade_map = {3: "优秀", 2: "良好", 1: "一般", 0: "较差"}

            # 构建重写后的字符串
            result_lines = ["【单项得分】"]

            # 处理前6项结构项
            for i in range(6):
                if i < len(score_list):
                    score = score_list[i]
                    grade = grade_map.get(score, "未知")
                    result_lines.append(f"{i+1}.{structure_items[i]} {grade}")

            # 处理第7-18项内容项
            for i in range(6, 18):
                if i < len(score_list):
                    score = score_list[i]
                    grade = grade_map.get(score, "未知")
                    content_idx = i - 6
                    if content_idx < len(content_items):
                        result_lines.append(
                            f"{i+1}.{content_items[content_idx]} {grade}"
                        )

            # 处理第19项总分
            if len(score_list) >= 19:
                total_score = score_list[18]
                # 根据总分确定等级
                if total_score >= 90:
                    total_grade = "优秀"
                elif total_score >= 75:
                    total_grade = "良好"
                elif total_score >= 60:
                    total_grade = "合格"
                else:
                    total_grade = "不合格"

                result_lines.append("")
                result_lines.append("【论文总分】")
                result_lines.append(f"{total_score} （{total_grade}）")

            return "\n".join(result_lines)

        # 解析和重写分数列表
        rewritten_scorelist = parse_and_rewrite_score_list(score_list)
        logging.info(f"重写后的分数列表:\n{rewritten_scorelist}")

        logging.info(f"人工评审分数: {score_list}")
        logging.info(f"人工评审建议: {human_advice[:200]}...")

        # ========== 第四阶段：按章节分组文档 ==========

        # 使用evaluation.py中的方法按章节分组
        chapter_groups = {}
        for doc in user_documents:
            chapter = doc.metadata.get("chapter", "未分类")
            if chapter not in chapter_groups:
                chapter_groups[chapter] = []
            chapter_groups[chapter].append(doc)

        # 过滤掉不需要分析的章节
        filtered_chapter_groups = {}
        for chapter_name, docs in chapter_groups.items():
            if not docs or any(
                exclude in chapter_name
                for exclude in ["摘要", "致谢", "参考文献", "谢", "目录"]
            ):
                continue
            filtered_chapter_groups[chapter_name] = docs

        # ========== 第五阶段：创建结果目录 ==========

        file_name = os.path.splitext(os.path.basename(data.file_path))[0]

        evaluation_dir = os.path.join(USER_RESULT_DIR, f"{file_name}_human_analysis")
        os.makedirs(evaluation_dir, exist_ok=True)

        # 保存原始人工评审数据
        review_backup_file = os.path.join(evaluation_dir, "original_human_review.json")
        with open(review_backup_file, "w", encoding="utf-8") as f:
            json.dump(human_review_data, f, ensure_ascii=False, indent=2)

        # 保存分块信息到文件（用于调试）
        output_chunks_file = os.path.join(evaluation_dir, "chunks_output.txt")
        with open(output_chunks_file, "w", encoding="utf-8") as f:
            f.write(f"Title: {metadata['paper_title']}\n")
            f.write(f"Student Name: {metadata['student_name']}\n")
            f.write(f"Student ID: {metadata['student_id']}\n")
            f.write(f"Abstract: {metadata['abstract']}\n\n")
            f.write(f"Keywords: {metadata['keywords']}\n\n")
            f.write(f"English Abstract: {metadata['english_abstract']}\n\n")
            f.write(f"English Keywords: {metadata['english_keywords']}\n\n")
            f.write("-" * 80 + "\n")
            f.write(f"Table of Contents: {metadata['contents']}\n\n")

            f.write("-" * 80 + "\n")
            for idx, doc in enumerate(user_documents):
                f.write(
                    f"Chunk {idx + 1} [{doc.metadata.get('chapter', 'Unknown')} > {doc.metadata.get('section', 'Unknown')} > {doc.metadata.get('subsection', 'Unknown')}]\n"
                )
                f.write(f"Length: {len(doc.page_content)} chars\n")
                f.write(f"Content: {doc.page_content[:50]}...\n")
                f.write("-" * 80 + "\n")

        logging.info(f"分块结果已写入到文件: chunks_output.txt")

        # ========== 加载章节概要数据 ==========

        logging.info(f"开始加载章节概要数据...")
        chapter_summary_data = load_chapter_summary_json(student_id, student_name)

        if not chapter_summary_data:
            logging.warning("未找到章节概要文件，将使用空章节概要")

        # ========== 第五阶段：论文类型和章节类型分类 ==========

        logging.info("开始论文类型和章节类型分类...")

        # 5.1 论文类型分类
        paper_type = await classify_paper_type(
            metadata, filtered_chapter_groups, data.model
        )

        # 5.2 章节类型分类
        chapter_stage_map = await classify_chapter_types(
            metadata, filtered_chapter_groups, paper_type, data.model
        )

        logging.info(
            f"论文类型: {paper_type}, 章节分类完成，共 {len(chapter_stage_map)} 个章节"
        )

        # ========== 第六阶段：构建章节分析任务 ==========

        # 准备并发任务
        analysis_tasks = []
        chapter_info_list = []  # 用于保存章节信息（章节名和提示词）

        for chapter_name, docs in filtered_chapter_groups.items():

            # 构建结构化的章节上下文
            structured_context = build_structured_context_for_chapter(
                chapter_name, docs
            )

            if chapter_summary_data:
                # 这里有两种版本的实现，基础版本只包含各章节的概括，详细版本的包括小节
                """
                # 使用详细版本
                chapters_summary = format_chapter_summary_detailed(
                    chapter_summary_data
                )
                """
                # """
                # 使用基础版本
                chapters_summary = format_chapter_summary_basic(chapter_summary_data)
                # """

                current_chapter_detailed_summary = format_current_chapter_detailed(
                    chapter_summary_data, chapter_name
                )

            else:
                chapters_summary = "无其他章节概要信息"
                current_chapter_detailed_summary = "无当前章节详细信息"

            # 构建提示词
            prompt = prompt_service.format_template(
                "human_analysis",
                title=metadata["paper_title"],
                chapter_title=chapter_name,
                abstract=metadata["abstract"],
                keywords=metadata["keywords"],
                structure=(
                    metadata["structure"][:1500]
                    if metadata["structure"]
                    else "无结构信息"
                ),
                chapter_summary=chapters_summary,
                context=structured_context,
                human_scorelist=rewritten_scorelist,
                human_advice=human_advice[:3000],
            )

            chapter_info_list.append(
                {
                    "chapter_name": chapter_name,
                    "prompt": prompt,
                    "basic_chapter_summary": (
                        format_chapter_summary_basic(chapter_summary_data)
                        if chapter_summary_data
                        else "无章节概要信息"
                    ),
                    "detailed_current_chapter_summary": current_chapter_detailed_summary,
                }
            )

            # 获取schema
            schema = HumanResultAnalysisResponse.model_json_schema()

            # 添加到任务列表
            analysis_tasks.append(async_llm_structured(prompt, data.model, schema))

        chapter_responses_dir = os.path.join(evaluation_dir, "chapter_responses")
        os.makedirs(chapter_responses_dir, exist_ok=True)
        logging.info(f"创建章节响应保存目录: {chapter_responses_dir}")

        # ========== 第七阶段：并发执行分析 ==========

        logging.info(f"开始并发分析 {len(analysis_tasks)} 个章节...")
        analysis_results = await asyncio.gather(*analysis_tasks, return_exceptions=True)

        # ========== 保存章节提示词和响应到文件 ==========
        chapter_responses_dir = os.path.join(evaluation_dir, "chapter_responses")
        os.makedirs(chapter_responses_dir, exist_ok=True)
        logging.info(f"创建章节响应保存目录: {chapter_responses_dir}")

        for i, (chapter_info, raw_response) in enumerate(
            zip(chapter_info_list, analysis_results)
        ):
            response_filename = os.path.join(
                chapter_responses_dir, f"ch{i+1}_response.txt"
            )

            with open(response_filename, "w", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write(f"章节: {chapter_info['chapter_name']}\n")
                f.write("=" * 80 + "\n\n")

                # 写入提示词部分
                f.write("【提示词】\n")
                f.write("=" * 80 + "\n")
                f.write(chapter_info["prompt"])
                f.write("\n\n")

                f.write("=" * 80 + "\n\n")

                f.write("=" * 80 + "\n")

                if isinstance(raw_response, Exception):
                    f.write("分析失败\n")
                    f.write("=" * 80 + "\n")
                    f.write(f"错误类型: {type(raw_response).__name__}\n")
                    f.write(f"错误信息: {str(raw_response)}\n")
                    f.write("=" * 80 + "\n")
                    f.write("堆栈跟踪:\n")
                    f.write(traceback.format_exc())
                else:
                    f.write("原始响应:\n")
                    f.write("=" * 80 + "\n")
                    f.write(str(raw_response))

                logging.info(
                    f"已保存章节 {chapter_info['chapter_name']} 的提示词和响应到: chapter_responses/ch{i+1}_response.txt"
                )
        # ========== 第八阶段：合并和重写结果 ==========

        # 初始化格式和内容建议列表
        format_advice_list = []
        content_advice_list = []
        score_summary = {}

        successful_analyses = 0
        failed_chapters = []

        if chapter_summary_data:
            basic_chapter_summary_all = format_chapter_summary_basic(
                chapter_summary_data
            )
        else:
            basic_chapter_summary_all = "无章节概要信息"

        for i, result in enumerate(analysis_results):
            chapter_name = list(filtered_chapter_groups.keys())[i]

            if isinstance(result, Exception):
                logging.error(f"章节 '{chapter_name}' 分析失败: {result}")
                failed_chapters.append(chapter_name)
                continue

            try:
                # 清理响应：移除 Markdown 代码块标记
                cleaned_response = re.sub(
                    r"^```json\s*|\s*```$", "", str(result).strip(), flags=re.MULTILINE
                )

                # 解析JSON响应
                chapter_result = json.loads(cleaned_response)

                # 获取该章节的文档块
                chapter_docs = filtered_chapter_groups[chapter_name]

                # 获取该章节的概要信息
                chapter_info = chapter_info_list[i]
                basic_chapter_summary = chapter_info["basic_chapter_summary"]
                detailed_current_chapter_summary = chapter_info[
                    "detailed_current_chapter_summary"
                ]

                # 处理格式建议
                if "format" in chapter_result and "advice" in chapter_result["format"]:
                    for advice_item in chapter_result["format"]["advice"]:
                        chapter_stage = chapter_stage_map.get(chapter_name, "其他")
                        rewritten_item = rewrite_advice_item(
                            advice_item,
                            chapter_docs,
                            metadata,
                            basic_chapter_summary,
                            detailed_current_chapter_summary,
                            paper_type,
                            chapter_stage,
                        )
                        rewritten_item["chapter"] = chapter_name
                        format_advice_list.append(rewritten_item)

                # 处理内容建议
                if (
                    "content" in chapter_result
                    and "advice" in chapter_result["content"]
                ):
                    for advice_item in chapter_result["content"]["advice"]:
                        chapter_stage = chapter_stage_map.get(chapter_name, "其他")
                        rewritten_item = rewrite_advice_item(
                            advice_item,
                            chapter_docs,
                            metadata,
                            basic_chapter_summary,
                            detailed_current_chapter_summary,
                            paper_type,
                            chapter_stage,
                        )
                        rewritten_item["chapter"] = chapter_name
                        content_advice_list.append(rewritten_item)

                # 使用第一个成功章节的分数总结
                if "score_summary" in chapter_result and not score_summary:
                    score_summary = chapter_result["score_summary"]

                successful_analyses += 1

            except Exception as e:
                logging.error(f"章节 '{chapter_name}' 处理失败: {e}")
                traceback.print_exc()
                failed_chapters.append(chapter_name)

        # 如果没有成功的分析，直接返回错误
        if successful_analyses == 0:
            raise HTTPException(status_code=500, detail="所有章节分析都失败了")

        # 如果没有分数总结，创建一个默认的
        if not score_summary:
            score_summary = {
                "problem": "基于人工评审分数和建议生成的分析报告",
                "chain_of_thought": "根据人工评审分数和建议，结合论文原文内容进行详细分析。",
                "score": score_list[-1] if score_list else 0,
            }

        # ========== 第九阶段：保存结果到四个文件 ==========

        # 保存格式建议到JSON文件
        format_advice_json = {
            "source": "NJUAI-2020",
            "total_count": len(format_advice_list),
            "advice": format_advice_list,
        }

        format_advice_json_file = os.path.join(evaluation_dir, "format_advice.json")
        with open(format_advice_json_file, "w", encoding="utf-8") as f:
            json.dump(format_advice_json, f, ensure_ascii=False, indent=2)

        # 保存内容建议到JSON文件
        content_advice_json = {
            "source": "NJUAI-2020",
            "total_count": len(content_advice_list),
            "advice": content_advice_list,
        }

        content_advice_json_file = os.path.join(evaluation_dir, "content_advice.json")
        with open(content_advice_json_file, "w", encoding="utf-8") as f:
            json.dump(content_advice_json, f, ensure_ascii=False, indent=2)

        # 保存格式建议到文本文件（方便阅读）
        format_advice_txt_file = os.path.join(evaluation_dir, "format_advice.txt")
        with open(format_advice_txt_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("格式建议列表\n")
            f.write("=" * 80 + "\n")
            f.write(f"总建议数: {len(format_advice_list)}\n")
            f.write(f"论文标题: {metadata['paper_title']}\n")
            f.write(f"学号: {metadata['student_id']}\n")
            f.write(f"姓名: {metadata['student_name']}\n")
            f.write("=" * 80 + "\n\n")

            for i, advice in enumerate(format_advice_list, 1):
                f.write(f"建议 {i}:\n")
                f.write(f"  来源: {advice.get('source', 'NJUAI-2020')}\n")
                f.write(f"  标题: {advice.get('title', '')}\n")
                f.write(f"  学号: {advice.get('student_id', '')}\n")
                f.write(f"  姓名: {advice.get('student_name', '')}\n")
                f.write(f"  章节: {advice.get('chapter', '')}\n")

                # 新增字段
                if "structure" in advice and advice["structure"]:
                    f.write(f"  论文结构: {advice['structure'][:300]}...\n")

                if (
                    "basic_chapter_summary" in advice
                    and advice["basic_chapter_summary"]
                ):
                    # 截断过长的基础概要
                    if len(advice["basic_chapter_summary"]) > 500:
                        f.write(
                            f"  所有章节基础概要: {advice['basic_chapter_summary'][:500]}...\n"
                        )
                    else:
                        f.write(
                            f"  所有章节基础概要: {advice['basic_chapter_summary']}\n"
                        )

                if (
                    "detailed_current_chapter_summary" in advice
                    and advice["detailed_current_chapter_summary"]
                ):
                    # 截断过长的详细概要
                    if len(advice["detailed_current_chapter_summary"]) > 500:
                        f.write(
                            f"  当前章节详细概要: {advice['detailed_current_chapter_summary'][:500]}...\n"
                        )
                    else:
                        f.write(
                            f"  当前章节详细概要: {advice['detailed_current_chapter_summary']}\n"
                        )

                # 原有字段
                f.write(f"  位置: {advice.get('position', '')}\n")
                f.write(f"  类型: {advice.get('type', '')}\n")

                # 上下文字段（可能较长，需要截断）
                context = advice.get("context", "")
                if len(context) > 300:
                    f.write(f"  上下文: {context[:300]}...\n")
                else:
                    f.write(f"  上下文: {context}\n")

                # 原文字段（可能较长，需要截断）
                raw_text = advice.get("raw_text", "")
                if len(raw_text) > 500:
                    f.write(f"  原文: {raw_text[:500]}...\n")
                else:
                    f.write(f"  原文: {raw_text}\n")

                f.write(f"  建议: {advice.get('suggestion', '')}\n")

                # 思维链字段（可能较长，需要截断）
                chain_of_thought = advice.get("chain_of_thought", "")
                if len(chain_of_thought) > 300:
                    f.write(f"  思维链: {chain_of_thought[:300]}...\n")
                else:
                    f.write(f"  思维链: {chain_of_thought}\n")

                f.write(f"  评分影响: {advice.get('scoring_impact', '')}\n")
                f.write("-" * 60 + "\n\n")

        # 保存内容建议到文本文件（方便阅读）
        content_advice_txt_file = os.path.join(evaluation_dir, "content_advice.txt")
        with open(content_advice_txt_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("内容建议列表\n")
            f.write("=" * 80 + "\n")
            f.write(f"总建议数: {len(content_advice_list)}\n")
            f.write(f"论文标题: {metadata['paper_title']}\n")
            f.write(f"学号: {metadata['student_id']}\n")
            f.write(f"姓名: {metadata['student_name']}\n")
            f.write("=" * 80 + "\n\n")

            for i, advice in enumerate(content_advice_list, 1):
                f.write(f"建议 {i}:\n")
                f.write(f"  来源: {advice.get('source', 'NJUAI-2020')}\n")
                f.write(f"  标题: {advice.get('title', '')}\n")
                f.write(f"  学号: {advice.get('student_id', '')}\n")
                f.write(f"  姓名: {advice.get('student_name', '')}\n")
                f.write(f"  章节: {advice.get('chapter', '')}\n")

                # 新增字段
                if "structure" in advice and advice["structure"]:
                    f.write(f"  论文结构: {advice['structure'][:300]}...\n")

                if (
                    "basic_chapter_summary" in advice
                    and advice["basic_chapter_summary"]
                ):
                    # 截断过长的基础概要
                    if len(advice["basic_chapter_summary"]) > 500:
                        f.write(
                            f"  所有章节基础概要: {advice['basic_chapter_summary'][:500]}...\n"
                        )
                    else:
                        f.write(
                            f"  所有章节基础概要: {advice['basic_chapter_summary']}\n"
                        )

                if (
                    "detailed_current_chapter_summary" in advice
                    and advice["detailed_current_chapter_summary"]
                ):
                    # 截断过长的详细概要
                    if len(advice["detailed_current_chapter_summary"]) > 500:
                        f.write(
                            f"  当前章节详细概要: {advice['detailed_current_chapter_summary'][:500]}...\n"
                        )
                    else:
                        f.write(
                            f"  当前章节详细概要: {advice['detailed_current_chapter_summary']}\n"
                        )

                # 原有字段
                f.write(f"  位置: {advice.get('position', '')}\n")
                f.write(f"  类型: {advice.get('type', '')}\n")

                # 上下文字段（可能较长，需要截断）
                context = advice.get("context", "")
                if len(context) > 300:
                    f.write(f"  上下文: {context[:300]}...\n")
                else:
                    f.write(f"  上下文: {context}\n")

                # 原文字段（可能较长，需要截断）
                raw_text = advice.get("raw_text", "")
                if len(raw_text) > 500:
                    f.write(f"  原文: {raw_text[:500]}...\n")
                else:
                    f.write(f"  原文: {raw_text}\n")

                f.write(f"  建议: {advice.get('suggestion', '')}\n")

                # 思维链字段（可能较长，需要截断）
                chain_of_thought = advice.get("chain_of_thought", "")
                if len(chain_of_thought) > 300:
                    f.write(f"  思维链: {chain_of_thought[:300]}...\n")
                else:
                    f.write(f"  思维链: {chain_of_thought}\n")

                f.write(f"  评分影响: {advice.get('scoring_impact', '')}\n")
                f.write("-" * 60 + "\n\n")

        # ========== 第十阶段：构建返回结果 ==========

        total_time = time.time() - start_time

        result = {
            "status": "success",
            "analysis_time": round(total_time, 2),
            "successful_chapters": successful_analyses,
            "total_chapters": len(analysis_tasks),
            "failed_chapters": failed_chapters,
            "result_dir": evaluation_dir,
            "metadata": {
                "student_name": metadata["student_name"],
                "student_id": metadata["student_id"],
                "paper_title": metadata["paper_title"],
            },
            "human_review_source": f"{student_id}_{student_name}.json",
            "analysis_result": {
                "format_advice_count": len(format_advice_list),
                "content_advice_count": len(content_advice_list),
                "summary_score": score_summary.get("score", 0),
                "generated_files": [
                    "format_advice.json",
                    "content_advice.json",
                    "format_advice.txt",
                    "content_advice.txt",
                ],
            },
        }

        logging.info(f"人工评审分析完成，耗时: {total_time:.2f}秒")
        logging.info(f"分析结果保存在: {evaluation_dir}")
        logging.info(f"生成文件: format_advice.json ({len(format_advice_list)}条建议)")
        logging.info(
            f"生成文件: content_advice.json ({len(content_advice_list)}条建议)"
        )

        return result

    except Exception as e:
        logging.error(f"人工评审分析失败: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"人工评审分析失败: {str(e)}")
