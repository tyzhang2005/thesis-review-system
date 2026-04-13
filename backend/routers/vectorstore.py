import json
import logging
import math
import os
from typing import Dict, List

from chromadb import PersistentClient
from chromadb.config import Settings
from config.config import (
    DATABASE_VECTORSTORE_DIR_CLOUD,
    DATABASE_VECTORSTORE_DIR_LOCAL,
    EMBEDDING_DIMENSION,
    USE_CLOUD_EMBEDDING,
)
from fastapi import APIRouter, FastAPI, HTTPException
from services.llm_utils import create_embeddings

router = APIRouter()


def get_vectorstore_dir():
    """根据嵌入模型类型获取向量数据库目录"""
    if USE_CLOUD_EMBEDDING:
        return DATABASE_VECTORSTORE_DIR_CLOUD
    else:
        return DATABASE_VECTORSTORE_DIR_LOCAL


def get_collection_name():
    """根据嵌入模型类型获取集合名称"""
    if USE_CLOUD_EMBEDDING:
        return "advice_collection_cloud_4b"  # 云端集合
    else:
        return "advice_collection_local_4b"  # 本地集合


from config.config import get_collection_name, get_vectorstore_dir


@router.post("/initialize-database/")
async def initialize_advice_database(request: Dict[str, str]):
    try:
        model_name = request.get("model")

        # 使用动态的数据库目录和集合名称
        database_dir = get_vectorstore_dir()
        collection_name = get_collection_name()

        # 确保目录存在
        os.makedirs(database_dir, exist_ok=True)

        logging.info(
            f"初始化数据库 - 目录: {database_dir}, 集合: {collection_name}, 使用云端嵌入: {USE_CLOUD_EMBEDDING}"
        )

        # 检查数据库是否已存在
        client = PersistentClient(
            path=str(database_dir), settings=Settings(anonymized_telemetry=False)
        )

        # 检查集合是否已存在
        existing_collections = [col.name for col in client.list_collections()]
        if collection_name in existing_collections:
            logging.info("建议数据库已存在，跳过初始化")
            return {
                "message": "建议数据库已存在，无需重新初始化",
                "database_dir": database_dir,
                "collection_name": collection_name,
                "use_cloud": USE_CLOUD_EMBEDDING,
            }

        ADVICE_DATA_PATH = "./advice_data.txt"
        # 读取建议数据文件
        if not os.path.exists(ADVICE_DATA_PATH):
            raise HTTPException(
                status_code=404, detail=f"建议数据文件不存在: {ADVICE_DATA_PATH}"
            )

        documents = []
        metadatas = []
        ids = []

        with open(ADVICE_DATA_PATH, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    stage = data.get("stage", "")
                    suggestion = data.get("suggestion", "")

                    if stage and suggestion:
                        documents.append(suggestion)
                        metadatas.append({"stage": stage})
                        ids.append(str(i))
                except json.JSONDecodeError:
                    logging.warning(f"跳过无效的JSON行: {line}")
                    continue

        if not documents:
            raise HTTPException(status_code=400, detail="建议数据文件为空或格式错误")

        logging.info(f"成功加载 {len(documents)} 条建议")

        # 使用工厂函数创建嵌入模型实例
        embeddings = create_embeddings()

        # 创建集合并添加文档
        collection = client.create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

        # 生成嵌入向量
        embeddings_list = await embeddings.embed_documents(documents)

        def clean_embeddings(embeddings_list):
            """清理嵌入向量中的NaN值"""
            cleaned_embeddings = []
            nan_count = 0

            for i, embedding in enumerate(embeddings_list):
                # 检查是否有NaN值
                if any(math.isnan(x) for x in embedding):
                    nan_count += 1
                    logging.warning(f"文档 {i} 的嵌入向量包含NaN值，使用零向量替代")
                    # 用零向量替代
                    cleaned_embedding = [0.0] * len(embedding)
                else:
                    cleaned_embedding = embedding
                cleaned_embeddings.append(cleaned_embedding)

            if nan_count > 0:
                logging.warning(f"共发现 {nan_count} 个包含NaN值的嵌入向量")

            return cleaned_embeddings

        # 清理嵌入向量
        embeddings_list = clean_embeddings(embeddings_list)

        # 验证嵌入维度
        if embeddings_list and len(embeddings_list) > 0:
            actual_dimension = len(embeddings_list[0])
            expected_dimension = EMBEDDING_DIMENSION
            if actual_dimension != expected_dimension:
                logging.warning(
                    f"嵌入维度不匹配: 期望 {expected_dimension}, 实际 {actual_dimension}"
                )

        # 添加到集合
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings_list,
        )

        logging.info(f"成功初始化建议数据库，共 {len(documents)} 条建议")
        logging.info(f"数据库位置: {database_dir}")
        logging.info(f"集合名称: {collection_name}")
        logging.info(f"使用云端嵌入: {USE_CLOUD_EMBEDDING}")

        return {
            "message": "建议数据库初始化成功",
            "advice_count": len(documents),
            "database_dir": database_dir,
            "collection_name": collection_name,
            "use_cloud": USE_CLOUD_EMBEDDING,
            "embedding_dimension": len(embeddings_list[0]) if embeddings_list else 0,
        }

    except Exception as e:
        logging.error(f"初始化建议数据库失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"初始化失败: {str(e)}")


def load_suggestions_from_file(file_path: str) -> List[Dict]:
    """
    从.txt文件加载修改建议
    假设文件格式为每行一个JSON对象
    """
    suggestions = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:  # 跳过空行
                    try:
                        suggestion_data = json.loads(line)
                        suggestions.append(suggestion_data)
                    except json.JSONDecodeError as e:
                        logging.warning(f"解析JSON行时出错: {line}, 错误: {e}")
        logging.info(f"从文件 {file_path} 成功加载 {len(suggestions)} 条建议")
        return suggestions
    except FileNotFoundError:
        logging.error(f"建议文件不存在: {file_path}")
        raise HTTPException(status_code=404, detail=f"建议文件不存在: {file_path}")
    except Exception as e:
        logging.error(f"读取建议文件时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"读取建议文件失败: {str(e)}")


@router.get("/database-status/")
async def get_database_status():
    """获取数据库状态信息"""
    try:
        local_dir = DATABASE_VECTORSTORE_DIR_LOCAL
        cloud_dir = DATABASE_VECTORSTORE_DIR_CLOUD
        current_dir = get_vectorstore_dir()

        status = {
            "use_cloud_embedding": USE_CLOUD_EMBEDDING,
            "current_database_dir": current_dir,
            "local_database_exists": os.path.exists(local_dir),
            "cloud_database_exists": os.path.exists(cloud_dir),
            "embedding_dimension": EMBEDDING_DIMENSION,
        }

        # 检查集合是否存在
        try:
            client = PersistentClient(
                path=str(current_dir), settings=Settings(anonymized_telemetry=False)
            )
            collections = client.list_collections()
            status["collections"] = [col.name for col in collections]
            status["current_collection_exists"] = (
                get_collection_name() in status["collections"]
            )
        except Exception as e:
            status["collection_error"] = str(e)

        return status
    except Exception as e:
        logging.error(f"获取数据库状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取状态失败: {str(e)}")
