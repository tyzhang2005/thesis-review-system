"""
用户评审结果向量数据库模块
用于存储和检索从 E:/advice/user_result 目录读取的结构化评审建议
支持 content_advice.json 和 format_advice.json 两种建议类型
"""

import json
import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from chromadb import PersistentClient
from chromadb.config import Settings
from fastapi import APIRouter, HTTPException

# 配置常量 - 与config.py保持一致
USER_RESULT_ADVICE_PATH = "/DATA/zhangtianyue_231300023/advice/user_result"
DATABASE_USER_RESULT_DIR_LOCAL = "./data/database/database_user_result_local"
DATABASE_USER_RESULT_DIR_CLOUD = "./data/database/database_user_result_cloud"
USE_CLOUD_EMBEDDING = True  # 与config.py保持一致
EMBEDDING_DIMENSION = 2048

router = APIRouter()

# 动态导入配置（避免循环导入）
try:
    from config.config import EMBEDDING_DIMENSION as config_embedding_dim
    from config.config import USE_CLOUD_EMBEDDING as config_use_cloud
    from config.config import (
        get_user_result_collection_name,
        get_user_result_vectorstore_dir,
    )

    USE_CLOUD_EMBEDDING = config_use_cloud
    EMBEDDING_DIMENSION = config_embedding_dim
except ImportError:
    pass


def get_vectorstore_dir():
    """根据嵌入模型类型获取向量数据库目录"""
    if USE_CLOUD_EMBEDDING:
        return DATABASE_USER_RESULT_DIR_CLOUD
    else:
        return DATABASE_USER_RESULT_DIR_LOCAL


def get_collection_name(advice_type: str):
    """
    根据建议类型和嵌入模型获取集合名称
    advice_type: 'content' 或 'format'
    """
    if USE_CLOUD_EMBEDDING:
        return f"user_result_{advice_type}_collection_cloud"
    else:
        return f"user_result_{advice_type}_collection_local"


def scan_user_result_folders(base_path: str) -> List[Path]:
    """
    扫描用户评审结果目录，返回所有包含JSON文件的子文件夹路径

    Args:
        base_path: 用户评审结果根目录路径

    Returns:
        包含 content_advice.json 和 format_advice.json 的文件夹路径列表
    """
    base = Path(base_path)
    if not base.exists():
        raise HTTPException(
            status_code=404, detail=f"用户评审结果目录不存在: {base_path}"
        )

    folders = []
    for folder in base.iterdir():
        if folder.is_dir():
            content_json = folder / "content_advice.json"
            format_json = folder / "format_advice.json"
            if content_json.exists() and format_json.exists():
                folders.append(folder)

    logging.info(f"扫描到 {len(folders)} 个包含评审结果的文件夹")
    return folders


def load_advice_from_json(file_path: Path) -> List[Dict]:
    """
    从JSON文件加载评审建议数据

    Args:
        file_path: JSON文件路径

    Returns:
        建议列表
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("advice", [])
    except json.JSONDecodeError as e:
        logging.warning(f"JSON解析错误 {file_path}: {e}")
        return []
    except Exception as e:
        logging.warning(f"读取文件错误 {file_path}: {e}")
        return []


def prepare_documents_for_embedding(
    advice_list: List[Dict], advice_type: str
) -> Tuple[List[str], List[Dict], List[str]]:
    """
    准备用于嵌入的文档、元数据和ID

    Document字段顺序（宏观到微观，背景到核心）：
    1. title - 论文整体背景
    2. chapter - 所属章节
    3. position - 具体位置
    4. context - 问题上下文
    5. suggestion - 修改建议
    6. chain_of_thought - 分析过程
    7. raw_text - 原文片段

    Metadata（过滤标签）: source, student_id, student_name, type

    Args:
        advice_list: 建议列表
        advice_type: 建议类型 ('content' 或 'format')

    Returns:
        (文档列表, 元数据列表, ID列表)
    """
    documents = []
    metadatas = []
    ids = []

    for idx, advice in enumerate(advice_list):
        # 构建 document: 按照宏观到微观的顺序
        parts = []

        # 1. 论文标题 - 整体背景
        if advice.get("title"):
            parts.append(f"论文标题: {advice['title']}")

        # 2. 所属章节 - 宏观位置
        if advice.get("chapter"):
            parts.append(f"所属章节: {advice['chapter']}")

        # 3. 具体位置 - 精确定位
        if advice.get("position"):
            parts.append(f"问题位置: {advice['position']}")

        # 4. 问题上下文 - 当前在讲什么
        if advice.get("context"):
            parts.append(f"上下文: {advice['context']}")

        # 5. 修改建议 - 核心内容
        if advice.get("suggestion"):
            parts.append(f"修改建议: {advice['suggestion']}")

        # 6. 分析过程 - 深入理解
        if advice.get("chain_of_thought"):
            parts.append(f"分析过程: {advice['chain_of_thought']}")

        # 7. 原文片段 - 辅助材料
        if advice.get("raw_text"):
            raw_text = advice["raw_text"]
            # 限制raw_text长度避免过长
            if len(raw_text) > 3000:
                raw_text = raw_text[:3000] + "..."
            parts.append(f"原文片段: {raw_text}")

        document = "\n\n".join(parts)
        if not document:
            continue

        # 构建 metadata: 用于过滤的标签（不参与向量计算）
        metadata = {
            "source": advice.get("source", ""),
            "student_id": advice.get("student_id", ""),
            "student_name": advice.get("student_name", ""),
            "title": advice.get("title", ""),
            "chapter": advice.get("chapter", ""),
            "position": advice.get("position", ""),
            "type": advice.get("type", ""),
            # 从 JSON 读取的新字段
            "paper_type": advice.get("paper_type", ""),
            "chapter_type": advice.get("chapter_type", ""),
            # 展示字段
            "context": advice.get("context", ""),
            "suggestion": advice.get("suggestion", ""),
            "chain_of_thought": advice.get("chain_of_thought", ""),
            "advice_type": advice_type,
        }

        # 生成唯一ID
        doc_id = f"{advice_type}_{advice.get('student_id', 'unknown')}_{idx}"

        documents.append(document)
        metadatas.append(metadata)
        ids.append(doc_id)

    return documents, metadatas, ids


async def create_embeddings(documents: List[str]) -> List[List[float]]:
    """
    创建文档的嵌入向量

    Args:
        documents: 文档列表

    Returns:
        嵌入向量列表
    """
    # 动态导入create_embeddings避免循环导入
    from services.llm_utils import create_embeddings

    embeddings = create_embeddings()
    return await embeddings.embed_documents(documents)


def clean_embeddings(embeddings_list: List[List[float]]) -> List[List[float]]:
    """清理嵌入向量中的NaN值"""
    cleaned_embeddings = []
    nan_count = 0

    for i, embedding in enumerate(embeddings_list):
        if any(math.isnan(x) for x in embedding):
            nan_count += 1
            logging.warning(f"文档 {i} 的嵌入向量包含NaN值，使用零向量替代")
            cleaned_embedding = [0.0] * len(embedding)
        else:
            cleaned_embedding = embedding
        cleaned_embeddings.append(cleaned_embedding)

    if nan_count > 0:
        logging.warning(f"共发现 {nan_count} 个包含NaN值的嵌入向量")

    return cleaned_embeddings


@router.post("/initialize-user-result-database/")
async def initialize_user_result_database():
    """
    初始化用户评审结果向量数据库

    从 E:/advice/user_result 目录读取所有学生的评审建议JSON文件，
    分别为 content_advice 和 format_advice 创建独立的向量集合
    """
    try:
        # 确保目录存在
        database_dir = get_vectorstore_dir()
        os.makedirs(database_dir, exist_ok=True)

        logging.info(
            f"初始化用户评审结果数据库 - 目录: {database_dir}, 使用云端嵌入: {USE_CLOUD_EMBEDDING}"
        )

        # 扫描用户评审结果文件夹
        folders = scan_user_result_folders(USER_RESULT_ADVICE_PATH)

        if not folders:
            raise HTTPException(
                status_code=404,
                detail=f"未找到包含评审结果的文件夹: {USER_RESULT_ADVICE_PATH}",
            )

        # 初始化ChromaDB客户端
        client = PersistentClient(
            path=str(database_dir), settings=Settings(anonymized_telemetry=False)
        )

        # 收集所有建议
        all_content_advice = []
        all_format_advice = []

        for folder in folders:
            content_json = folder / "content_advice.json"
            format_json = folder / "format_advice.json"

            content_advice = load_advice_from_json(content_json)
            format_advice = load_advice_from_json(format_json)

            all_content_advice.extend(content_advice)
            all_format_advice.extend(format_advice)

            logging.info(
                f"从 {folder.name} 加载了 {len(content_advice)} 条内容建议, {len(format_advice)} 条格式建议"
            )

        total_content = len(all_content_advice)
        total_format = len(all_format_advice)

        if total_content == 0 and total_format == 0:
            raise HTTPException(status_code=400, detail="未找到任何有效的评审建议数据")

        logging.info(f"总共加载 {total_content} 条内容建议, {total_format} 条格式建议")

        # 处理内容建议
        content_stats = {}
        if total_content > 0:
            content_collection_name = get_collection_name("content")

            # 检查集合是否已存在
            existing_collections = [col.name for col in client.list_collections()]
            if content_collection_name in existing_collections:
                logging.info(f"内容建议集合已存在，删除重建: {content_collection_name}")
                client.delete_collection(content_collection_name)

            # 准备文档
            content_docs, content_metas, content_ids = prepare_documents_for_embedding(
                all_content_advice, "content"
            )

            if content_docs:
                # 生成嵌入
                content_embeddings = await create_embeddings(content_docs)
                content_embeddings = clean_embeddings(content_embeddings)

                # 创建集合
                content_collection = client.create_collection(
                    name=content_collection_name,
                    metadata={
                        "hnsw:space": "cosine",
                        "description": "用户评审结果-内容建议",
                    },
                )

                # 添加数据
                content_collection.add(
                    documents=content_docs,
                    metadatas=content_metas,
                    ids=content_ids,
                    embeddings=content_embeddings,
                )

                content_stats = {
                    "collection_name": content_collection_name,
                    "count": len(content_docs),
                    "embedding_dimension": (
                        len(content_embeddings[0]) if content_embeddings else 0
                    ),
                }
                logging.info(f"内容建议集合创建成功: {content_stats}")

        # 处理格式建议
        format_stats = {}
        if total_format > 0:
            format_collection_name = get_collection_name("format")

            # 检查集合是否已存在
            existing_collections = [col.name for col in client.list_collections()]
            if format_collection_name in existing_collections:
                logging.info(f"格式建议集合已存在，删除重建: {format_collection_name}")
                client.delete_collection(format_collection_name)

            # 准备文档
            format_docs, format_metas, format_ids = prepare_documents_for_embedding(
                all_format_advice, "format"
            )

            if format_docs:
                # 生成嵌入
                format_embeddings = await create_embeddings(format_docs)
                format_embeddings = clean_embeddings(format_embeddings)

                # 创建集合
                format_collection = client.create_collection(
                    name=format_collection_name,
                    metadata={
                        "hnsw:space": "cosine",
                        "description": "用户评审结果-格式建议",
                    },
                )

                # 添加数据
                format_collection.add(
                    documents=format_docs,
                    metadatas=format_metas,
                    ids=format_ids,
                    embeddings=format_embeddings,
                )

                format_stats = {
                    "collection_name": format_collection_name,
                    "count": len(format_docs),
                    "embedding_dimension": (
                        len(format_embeddings[0]) if format_embeddings else 0
                    ),
                }
                logging.info(f"格式建议集合创建成功: {format_stats}")

        return {
            "message": "用户评审结果向量数据库初始化成功",
            "database_dir": database_dir,
            "use_cloud": USE_CLOUD_EMBEDDING,
            "content_stats": content_stats,
            "format_stats": format_stats,
            "total_students": len(folders),
            "total_advice": total_content + total_format,
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"初始化用户评审结果数据库失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"初始化失败: {str(e)}")


@router.get("/user-result-database-status/")
async def get_user_result_database_status():
    """获取用户评审结果数据库状态"""
    try:
        database_dir = get_vectorstore_dir()
        status = {
            "use_cloud_embedding": USE_CLOUD_EMBEDDING,
            "database_dir": database_dir,
            "advice_source_path": USER_RESULT_ADVICE_PATH,
            "source_exists": os.path.exists(USER_RESULT_ADVICE_PATH),
        }

        # 检查文件夹数量
        if status["source_exists"]:
            try:
                folders = scan_user_result_folders(USER_RESULT_ADVICE_PATH)
                status["student_folders_count"] = len(folders)
            except Exception:
                status["student_folders_count"] = 0

        # 检查集合状态
        try:
            client = PersistentClient(
                path=str(database_dir), settings=Settings(anonymized_telemetry=False)
            )
            collections = client.list_collections()
            status["collections"] = [col.name for col in collections]

            # 检查特定集合
            content_collection = get_collection_name("content")
            format_collection = get_collection_name("format")

            status["content_collection_exists"] = (
                content_collection in status["collections"]
            )
            status["format_collection_exists"] = (
                format_collection in status["collections"]
            )

            # 获取集合中的文档数量
            if status["content_collection_exists"]:
                content_col = client.get_collection(content_collection)
                status["content_collection_count"] = content_col.count()

            if status["format_collection_exists"]:
                format_col = client.get_collection(format_collection)
                status["format_collection_count"] = format_col.count()

        except Exception as e:
            status["collection_error"] = str(e)

        return status

    except Exception as e:
        logging.error(f"获取用户评审结果数据库状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取状态失败: {str(e)}")


@router.post("/query-user-result-advice/")
async def query_user_result_advice(request: Dict[str, str]):
    """
    查询用户评审结果向量数据库

    Request body:
        query: 查询文本
        advice_type: 建议类型 ("content", "format", 或 "all")
        n_results: 返回结果数量（默认5）
        student_id: 可选，按学生ID过滤
    """
    try:
        query_text = request.get("query", "")
        advice_type = request.get("advice_type", "all")
        n_results = int(request.get("n_results", 5))
        student_id_filter = request.get("student_id", None)

        if not query_text:
            raise HTTPException(status_code=400, detail="查询文本不能为空")

        database_dir = get_vectorstore_dir()
        client = PersistentClient(
            path=str(database_dir), settings=Settings(anonymized_telemetry=False)
        )

        # 生成查询嵌入
        from services.llm_utils import create_embeddings

        embeddings = create_embeddings()
        query_embedding = await embeddings.embed_query(query_text)

        results = {}

        # 确定要查询的集合
        collections_to_query = []
        if advice_type in ["content", "all"]:
            content_collection = get_collection_name("content")
            try:
                if client.get_collection(content_collection):
                    collections_to_query.append(("content", content_collection))
            except Exception:
                pass

        if advice_type in ["format", "all"]:
            format_collection = get_collection_name("format")
            try:
                if client.get_collection(format_collection):
                    collections_to_query.append(("format", format_collection))
            except Exception:
                pass

        if not collections_to_query:
            raise HTTPException(
                status_code=404, detail="未找到可查询的集合，请先初始化数据库"
            )

        # 构建过滤条件
        where_filter = None
        if student_id_filter:
            where_filter = {"student_id": student_id_filter}

        # 查询每个集合
        for col_type, col_name in collections_to_query:
            collection = client.get_collection(col_name)
            query_result = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_filter,
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

        return {"query": query_text, "advice_type": advice_type, "results": results}

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"查询用户评审结果失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")
