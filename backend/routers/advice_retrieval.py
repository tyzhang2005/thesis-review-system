"""
建议检索模块
基于 database_user_result_cloud 向量数据库，为论文检索相关修改建议
与 evaluation.py 平行，但只检索建议，不进行评分
"""

import asyncio
import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List

from chromadb import PersistentClient
from chromadb.config import Settings

# 复用现有模块
from config.config import (
    USE_CLOUD_EMBEDDING,
    get_user_result_collection_name,
    get_user_result_vectorstore_dir,
)
from fastapi import APIRouter, HTTPException
from langchain_core.documents import Document
from services.llm_utils import async_llm, create_embeddings
from services.markdown_processor import ChineseMarkdownSplitter

router = APIRouter()

from config.config import ADVICE_RETRIEVAL_DIR

# ==================== 数据库检索函数 ====================


async def query_user_result_suggestions(
    prompt: str, advice_type: str = "all", n_results: int = 10
) -> Dict[str, List[Dict]]:
    """
    从用户评审结果向量数据库检索建议

    Args:
        prompt: 查询文本（论文元数据+章节内容）
        advice_type: 建议类型 ("content", "format", "all")
        n_results: 返回结果数量

    Returns:
        {
            "content": [...],  # 内容建议列表
            "format": [...]    # 格式建议列表
        }
    """
    try:
        database_dir = get_user_result_vectorstore_dir()
        client = PersistentClient(
            path=str(database_dir), settings=Settings(anonymized_telemetry=False)
        )

        # 生成查询嵌入
        embeddings = create_embeddings()
        query_embedding = await embeddings.embed_query(prompt)

        results = {}

        # 确定要查询的集合
        collections_to_query = []
        if advice_type in ["content", "all"]:
            content_collection = get_user_result_collection_name("content")
            try:
                if client.get_collection(content_collection):
                    collections_to_query.append(("content", content_collection))
            except Exception:
                pass

        if advice_type in ["format", "all"]:
            format_collection = get_user_result_collection_name("format")
            try:
                if client.get_collection(format_collection):
                    collections_to_query.append(("format", format_collection))
            except Exception:
                pass

        if not collections_to_query:
            logging.warning("未找到可查询的集合")
            return {"content": [], "format": []}

        # 查询每个集合
        for col_type, col_name in collections_to_query:
            collection = client.get_collection(col_name)
            query_result = collection.query(
                query_embeddings=[query_embedding], n_results=n_results
            )

            # 格式化结果
            formatted_results = []
            if query_result and query_result.get("ids") and query_result["ids"][0]:
                for i, doc_id in enumerate(query_result["ids"][0]):
                    result_item = {
                        "id": doc_id,
                        "document": (
                            query_result["documents"][0][i]
                            if query_result.get("documents")
                            else ""
                        ),
                        "metadata": (
                            query_result["metadatas"][0][i]
                            if query_result.get("metadatas")
                            else {}
                        ),
                        "distance": (
                            query_result["distances"][0][i]
                            if query_result.get("distances")
                            else 0.0
                        ),
                    }
                    formatted_results.append(result_item)

            results[col_type] = formatted_results

        return results

    except Exception as e:
        logging.error(f"检索用户评审结果失败: {str(e)}")
        return {"content": [], "format": []}


# ==================== 章节建议生成函数 ====================


async def generate_chapter_advice(
    chapter_name: str, chapter_content: str, paper_metadata: Dict, model: str
) -> Dict:
    """
    为单个章节生成修改建议

    Args:
        chapter_name: 章节名称
        chapter_content: 章节内容
        paper_metadata: 论文元数据
        model: LLM模型名称

    Returns:
        {
            "chapter_name": "章节名称",
            "chapter_type": "章节类型",
            "content_advice": [...],  # 内容建议
            "format_advice": [...],   # 格式建议
        }
    """
    # 构建检索提示词
    retrieval_prompt = f"""
请在专家评审案例数据库中，基于以下论文元数据和章节内容，检索出与该章节最相关的修改建议。
【论文元数据】
标题：{paper_metadata.get('title', '未知标题')}
摘要：{paper_metadata.get('abstract', '')[:500]}
关键词：{paper_metadata.get('keywords', '')}

【章节标题】{chapter_name}

【章节内容】
{chapter_content[:7500]}
"""

    # 检索建议
    query_results = await query_user_result_suggestions(retrieval_prompt, "all", 5)

    return {
        "chapter_name": chapter_name,
        "chapter_type": paper_metadata.get("chapter_type", "general"),
        "content_advice": query_results.get("content", []),
        "format_advice": query_results.get("format", []),
    }


# ==================== 汇总建议函数 ====================


async def summarize_all_advice(
    all_chapter_advice: List[Dict], paper_metadata: Dict, model: str
) -> str:
    """
    汇总所有章节的修改建议，选出最重要的5-10条

    Args:
        all_chapter_advice: 所有章节的建议列表
        paper_metadata: 论文元数据
        model: LLM模型名称

    Returns:
        汇总建议文本
    """
    try:
        # 提取所有建议
        all_advice = []

        for chapter_advice in all_chapter_advice:
            chapter_name = chapter_advice.get("chapter_name", "未知章节")

            # 处理内容建议
            for advice_item in chapter_advice.get("content_advice", []):
                metadata = advice_item.get("metadata", {})
                all_advice.append(
                    {
                        "chapter": chapter_name,
                        "position": metadata.get("position", ""),
                        "suggestion": metadata.get("suggestion", ""),
                        "advice_type": "content",
                    }
                )

            # 处理格式建议
            for advice_item in chapter_advice.get("format_advice", []):
                metadata = advice_item.get("metadata", {})
                all_advice.append(
                    {
                        "chapter": chapter_name,
                        "position": metadata.get("position", ""),
                        "suggestion": metadata.get("suggestion", ""),
                        "advice_type": "format",
                    }
                )

        if not all_advice:
            return "未发现需要修改的问题。"

        logging.info(
            f"从 {len(all_chapter_advice)} 个章节中提取到 {len(all_advice)} 条修改建议"
        )

        # 构建章节建议字符串
        chapter_advices_str = ""
        for advice in all_advice:
            chapter_advices_str += (
                f"【{advice['chapter']}】{advice['position']}: {advice['suggestion']}\n"
            )

        # 构建提示词（复用 evaluation.py 的模板）
        summary_advice_template = """
你是一个专业的本科毕业论文评估助手，请基于以下论文元数据内容和各章节的修改建议，总结给出修改建议：

要求：从各个章节中选择你认为重要的修改建议，不要只从一个章节中选择。句式与各章节修改建议一致，返回的修改建议不超过5条。

输出时请只输出修改建议，不要输出其他内容。

【论文元数据】
标题：{title}
摘要：{abstract}
关键词：{keywords}

【输出格式】
输出示例：
    [X.X节] 具体建议 \\n
    [X.X节] 具体建议 \\n
    [第X章] 具体建议 \\n

【各章节修改建议】
{chapter_advices}

请严格按照上述格式输出修改建议：
"""

        summary_prompt = summary_advice_template.format(
            title=paper_metadata.get("title", "未知标题"),
            abstract=paper_metadata.get("abstract", "")[:500].replace("\n", " "),
            keywords=", ".join(paper_metadata.get("keywords", "").split("，")[:5]),
            chapter_advices=chapter_advices_str[:8000],
        )

        # 调用模型生成汇总建议
        summary_advice_response = await async_llm(summary_prompt, model)

        # 清理响应内容
        pattern = r"```.*?```"
        summary_advice_response = re.sub(
            pattern, "", summary_advice_response, flags=re.DOTALL
        )
        summary_advice_response = summary_advice_response.strip()

        return summary_advice_response

    except Exception as e:
        logging.error(f"汇总修改建议失败: {str(e)}")
        return f"汇总建议生成失败: {str(e)}"


# ==================== 文档处理辅助函数 ====================


def build_chapter_context(docs: List[Document]) -> str:
    """从文档列表构建章节内容"""
    context_parts = []
    for doc in docs:
        if doc.page_content:
            context_parts.append(doc.page_content)
    return "\n\n".join(context_parts)


# ==================== 主入口函数 ====================


class Query:
    """请求参数模型"""

    def __init__(self, file_path: str, username: str, model: str):
        self.file_path = file_path
        self.username = username
        self.model = model


@router.post("/retrieve-advice/")
async def retrieve_advice(request: Dict[str, str]):
    """
    主入口：检索用户评审结果建议

    Request body:
        file_path: 论文文件路径（markdown格式）
        username: 用户名
        model: 使用的LLM模型（可选）

    Returns:
        {
            "status": "success",
            "output_dir": "advice_retrieval/{file_name}/",
            "txt_file": "advice_retrieval.txt",
            "json_file": "advice_retrieval.json"
        }
    """
    try:
        # 解析请求参数
        file_path = request.get("file_path", "")
        username = request.get("username", "")
        model = request.get("model", "deepseek-v3.2-exp")

        if not file_path:
            raise HTTPException(status_code=400, detail="file_path 参数不能为空")

        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"文件不存在: {file_path}")

        logging.info(f"开始处理建议检索请求 - 文件: {file_path}, 用户: {username}")

        # 1. 加载并处理文档
        logging.info("步骤1: 加载文档...")
        with open(file_path, "r", encoding="utf-8") as f:
            markdown_content = f.read()

        def preprocess_markdown(content: str) -> str:
            """预处理Markdown内容"""
            content = re.sub(r"(\n#+\s+[\u4e00-\u9fa5]+)\n\n#+", r"\1", content)
            content = re.sub(
                r"(\$\$[\s\S]*?\$\$)", r"<!--formula-->\1<!--/formula-->", content
            )
            return content

        # 预处理 Markdown 内容
        processed_md = preprocess_markdown(markdown_content)

        # 执行结构化分块
        splitter = ChineseMarkdownSplitter()
        documents = splitter.split_text(processed_md)

        if not documents:
            raise HTTPException(status_code=400, detail="文档解析失败，未找到内容")

        # 获取元数据
        first_doc = documents[0]
        paper_metadata = {
            "title": first_doc.metadata.get("title", "未知标题"),
            "abstract": first_doc.metadata.get("abstract", ""),
            "keywords": first_doc.metadata.get("keywords", ""),
            "student_name": first_doc.metadata.get("student_name", ""),
            "student_id": first_doc.metadata.get("student_id", ""),
            "paper_type": first_doc.metadata.get("type", "方法创新"),
        }

        # 2. 按章节分组
        logging.info("步骤2: 按章节分组...")
        chapter_groups = {}
        for doc in documents:
            chapter = doc.metadata.get("chapter", "")
            if chapter and chapter not in ["摘要", "致谢", "参考文献", "谢", "附录"]:
                if chapter not in chapter_groups:
                    chapter_groups[chapter] = []
                chapter_groups[chapter].append(doc)

        logging.info(f"共识别 {len(chapter_groups)} 个章节")

        # 3. 并发检索每个章节的建议
        logging.info("步骤3: 检索各章节建议...")
        retrieval_tasks = []

        for chapter_name, docs in chapter_groups.items():
            chapter_content = build_chapter_context(docs)
            chapter_metadata = paper_metadata.copy()
            chapter_metadata["chapter_type"] = docs[0].metadata.get("stage", "general")

            retrieval_tasks.append(
                generate_chapter_advice(
                    chapter_name, chapter_content, chapter_metadata, model
                )
            )

        # 并发执行
        all_chapter_advice = await asyncio.gather(*retrieval_tasks)

        # 4. 汇总建议
        logging.info("步骤4: 汇总修改建议...")
        summary_advice = await summarize_all_advice(
            all_chapter_advice, paper_metadata, model
        )

        # 5. 准备输出目录
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        output_dir = os.path.join(ADVICE_RETRIEVAL_DIR, file_name)
        os.makedirs(output_dir, exist_ok=True)

        # 6. 写入 TXT 文件
        txt_file_path = os.path.join(output_dir, "advice_retrieval.txt")
        with open(txt_file_path, "w", encoding="utf-8") as f:
            f.write(
                "╔════════════════════════════════════════════════════════════════╗\n"
            )
            f.write(
                "║           基于用户评审结果的论文修改建议报告                    ║\n"
            )
            f.write(
                "╚════════════════════════════════════════════════════════════════╝\n\n"
            )

            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"论文标题: {paper_metadata['title']}\n")
            f.write(f"学生姓名: {paper_metadata['student_name']}\n")
            f.write(f"学号: {paper_metadata['student_id']}\n")
            f.write(f"论文类型: {paper_metadata['paper_type']}\n\n")

            f.write(
                "────────────────────────────────────────────────────────────────\n"
            )
            f.write("💡 重要修改建议（AI汇总）\n")
            f.write(
                "────────────────────────────────────────────────────────────────\n"
            )
            f.write(summary_advice)
            f.write("\n\n")

            f.write(
                "────────────────────────────────────────────────────────────────\n"
            )
            f.write("📋 详细章节建议\n")
            f.write(
                "────────────────────────────────────────────────────────────────\n\n"
            )

            for idx, chapter_advice in enumerate(all_chapter_advice, 1):
                chapter_name = chapter_advice["chapter_name"]
                content_advice = chapter_advice.get("content_advice", [])
                format_advice = chapter_advice.get("format_advice", [])

                f.write(f"【{chapter_name}】\n")
                f.write("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

                if content_advice:
                    f.write("📌 内容建议:\n")
                    for i, advice_item in enumerate(
                        content_advice[:5], 1
                    ):  # 最多显示5条
                        metadata = advice_item.get("metadata", {})
                        position = metadata.get("position", "")
                        suggestion = metadata.get("suggestion", "")
                        if suggestion:
                            f.write(f"  {i}. [{position}] {suggestion}\n")
                    if not content_advice or not any(
                        a.get("metadata", {}).get("suggestion") for a in content_advice
                    ):
                        f.write("  无内容建议\n")
                    f.write("\n")

                if format_advice:
                    f.write("📌 格式建议:\n")
                    for i, advice_item in enumerate(
                        format_advice[:5], 1
                    ):  # 最多显示5条
                        metadata = advice_item.get("metadata", {})
                        position = metadata.get("position", "")
                        suggestion = metadata.get("suggestion", "")
                        if suggestion:
                            f.write(f"  {i}. [{position}] {suggestion}\n")
                    if not format_advice or not any(
                        a.get("metadata", {}).get("suggestion") for a in format_advice
                    ):
                        f.write("  无格式建议\n")
                    f.write("\n")

                f.write("\n")

        # 7. 写入 JSON 文件
        json_file_path = os.path.join(output_dir, "advice_retrieval.json")

        # 构建 JSON 数据结构
        json_data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "title": paper_metadata["title"],
                "student_name": paper_metadata["student_name"],
                "student_id": paper_metadata["student_id"],
                "paper_type": paper_metadata["paper_type"],
            },
            "summary_advice": summary_advice.split("\n") if summary_advice else [],
            "chapters": [],
        }

        for idx, chapter_advice in enumerate(all_chapter_advice, 1):
            chapter_data = {
                "chapter_index": idx,
                "chapter_name": chapter_advice["chapter_name"],
                "chapter_type": chapter_advice.get("chapter_type", "general"),
                "content_advice": [],
                "format_advice": [],
            }

            # 处理内容建议
            for advice_item in chapter_advice.get("content_advice", []):
                metadata = advice_item.get("metadata", {})
                chapter_data["content_advice"].append(
                    {
                        "source": metadata.get("source", ""),
                        "student_id": metadata.get("student_id", ""),
                        "student_name": metadata.get("student_name", ""),
                        "title": metadata.get("title", ""),
                        "chapter": metadata.get("chapter", ""),
                        "position": metadata.get("position", ""),
                        "context": metadata.get("context", ""),
                        "suggestion": metadata.get("suggestion", ""),
                        "chain_of_thought": metadata.get("chain_of_thought", ""),
                        "distance": advice_item.get("distance", 0.0),
                    }
                )

            # 处理格式建议
            for advice_item in chapter_advice.get("format_advice", []):
                metadata = advice_item.get("metadata", {})
                chapter_data["format_advice"].append(
                    {
                        "source": metadata.get("source", ""),
                        "student_id": metadata.get("student_id", ""),
                        "student_name": metadata.get("student_name", ""),
                        "title": metadata.get("title", ""),
                        "chapter": metadata.get("chapter", ""),
                        "position": metadata.get("position", ""),
                        "context": metadata.get("context", ""),
                        "suggestion": metadata.get("suggestion", ""),
                        "chain_of_thought": metadata.get("chain_of_thought", ""),
                        "distance": advice_item.get("distance", 0.0),
                    }
                )

            json_data["chapters"].append(chapter_data)

        with open(json_file_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

        logging.info(f"建议检索完成 - TXT: {txt_file_path}, JSON: {json_file_path}")

        return {
            "status": "success",
            "message": "建议检索完成",
            "output_dir": f"data/processed/advice_retrieval/{file_name}/",
            "txt_file": "advice_retrieval.txt",
            "json_file": "advice_retrieval.json",
            "total_chapters": len(all_chapter_advice),
            "total_advice": sum(
                len(c.get("content_advice", [])) + len(c.get("format_advice", []))
                for c in all_chapter_advice
            ),
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"建议检索失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"建议检索失败: {str(e)}")
