import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional, Type, TypeVar

import aiohttp
from config.config import (
    CLOUD_API_ENDPOINT,
    CLOUD_MODEL_HARDCODED,
    MAX_CONCURRENT,
    OLLAMA_HOST,
    USE_CLOUD_API,
    USE_VLLM,
    VLLM_EMBEDDING_ENDPOINT,
    VLLM_HOST,
    config,
)
from fastapi import HTTPException
from pydantic import BaseModel
from templates.evaluate_template import (
    create_analysis_json_schema,
    create_conclusion_json_schema,
    create_discussion_json_schema,
    create_introduction_json_schema,
    create_literature_review_json_schema,
    create_methodology_json_schema,
    get_chapter_schema,
)

# 泛型类型变量
T = TypeVar("T", bound=BaseModel)

llm_semaphore = asyncio.Semaphore(MAX_CONCURRENT)


async def get_llm_endpoint() -> str:
    """获取LLM端点URL"""
    if USE_CLOUD_API:
        return CLOUD_API_ENDPOINT
    else:
        return VLLM_HOST


async def get_llm_headers() -> Dict[str, str]:
    """获取LLM请求头"""
    headers = {"Content-Type": "application/json"}
    api_key = config.CLOUD_API_KEY
    if USE_CLOUD_API and api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def get_llm_model(model: str) -> str:
    """获取LLM模型名称（云端模式下使用硬编码名称）"""
    if USE_CLOUD_API:
        return CLOUD_MODEL_HARDCODED
    else:
        return model


async def create_llm_payload(prompt: str, model: str, **kwargs) -> Dict[str, Any]:
    """创建LLM请求负载"""
    # 确定实际使用的模型名称
    actual_model = get_llm_model(model)

    base_payload = {
        "model": actual_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": kwargs.get("temperature", 0.2),
        "max_tokens": kwargs.get("max_tokens", 2048),
        "top_p": kwargs.get("top_p", 0.95),
        "stream": False,
    }

    # 添加流式选项（云端API需要）
    if USE_CLOUD_API:
        pass

    # 添加响应格式（如果提供）
    if "response_format" in kwargs:
        base_payload["response_format"] = kwargs["response_format"]

    # 添加logit_bias（如果提供）- 仅限本地模式
    if not USE_CLOUD_API and "logit_bias" in kwargs:
        base_payload["logit_bias"] = kwargs["logit_bias"]

    return base_payload


# vllm结构化输出
async def async_llm_structured(
    prompt: str, model: str, json_schema: Dict[str, Any]
) -> str:
    """使用vLLM的引导解码器或云端API确保模型输出指定JSON格式"""

    async with llm_semaphore:
        try:
            endpoint = await get_llm_endpoint()
            headers = await get_llm_headers()

            # 构建请求payload，添加引导解码参数
            payload = await create_llm_payload(
                prompt=prompt,
                model=model,
                temperature=0.2,
                max_tokens=2048,
                top_p=0.95,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "json_schema", "schema": json_schema},
                },
            )

            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=600)
            ) as session:
                async with session.post(
                    f"{endpoint}/v1/chat/completions", json=payload, headers=headers
                ) as resp:
                    if resp.status != 200:
                        response_text = await resp.text()
                        logging.warning(
                            f"LLM结构化请求失败: {resp.status}, 响应: {response_text}"
                        )
                        # 降级到普通生成
                        return await async_llm(prompt, model)

                    data = await resp.json()
                    text = data["choices"][0]["message"]["content"].strip()

                    # 清理响应文本
                    if "</think>" in text:
                        text = text.split("</think>", 1)[-1].strip()

                    text = re.sub(r"^```json\s*", "", text).strip()
                    text = re.sub(r"\s*```$", "", text).strip()

                    return text

        except Exception as e:
            logging.error(f"结构化生成失败: {str(e)}")
            # 降级到普通生成
            return await async_llm(prompt, model)


async def async_llm_structured_v2(
    prompt: str,
    model: str,
    response_model: Type[T],
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> T:
    """
    使用Pydantic模型进行结构化输出（推荐使用）

    Args:
        prompt: 提示词
        model: 模型名称
        response_model: Pydantic模型类
        temperature: 温度参数
        max_tokens: 最大token数

    Returns:
        Pydantic模型实例

    Raises:
        ValueError: 当输出无法解析为指定模型时
    """
    try:
        # 从Pydantic模型生成JSON Schema
        json_schema = json.loads(response_model.model_json_schema())
    except Exception as e:
        raise ValueError(f"无法从Pydantic模型生成JSON Schema: {e}")

    # 调用现有的结构化输出函数
    json_str = await async_llm_structured(prompt, model, json_schema)

    # 解析JSON字符串为Pydantic模型
    try:
        result = response_model.model_validate_json(json_str)
        return result
    except Exception as e:
        # 尝试清理JSON后再解析
        try:
            cleaned_json = re.sub(r",\s*([}\]])", r"\1", json_str)  # 移除尾随逗号
            result = response_model.model_validate_json(cleaned_json)
            return result
        except Exception as e2:
            raise ValueError(
                f"无法将LLM输出解析为 {response_model.__name__}: {e2}\n"
                f"原始输出: {json_str[:500]}..."
            )


async def async_llm_structured_v2_with_distribution(
    prompt: str,
    model: str,
    response_model: Type[T],
    temperature: float = 0.4,
    max_tokens: int = 2048,
) -> T:
    """
    使用Pydantic模型进行结构化输出

    注意：logit bias功能已移除，此函数现在与async_llm_structured_v2功能相同

    Args:
        prompt: 提示词
        model: 模型名称
        response_model: Pydantic模型类
        temperature: 温度参数
        max_tokens: 最大token数

    Returns:
        Pydantic模型实例
    """
    # 直接调用标准结构化输出
    return await async_llm_structured_v2(
        prompt, model, response_model, temperature, max_tokens
    )


# 修改章节评估的建议生成部分
def create_advice_json_schema() -> Dict[str, Any]:
    """创建建议生成的JSON Schema"""
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "chapter": {"type": "string"},
                "original_text": {"type": "string"},
                "advice": {"type": "string"},
            },
            "required": ["chapter", "original_text", "advice"],
        },
    }


# 修改分数评估的JSON Schema
def create_score_json_schema() -> Dict[str, Any]:
    """创建分数评估的JSON Schema"""
    return {
        "type": "object",
        "properties": {
            "scores": {
                "type": "object",
                "properties": {
                    "1": {"type": "number", "minimum": 0, "maximum": 100},
                    "2": {"type": "number", "minimum": 0, "maximum": 100},
                    "3": {"type": "number", "minimum": 0, "maximum": 100},
                    "4": {"type": "number", "minimum": 0, "maximum": 100},
                    "5": {"type": "number", "minimum": 0, "maximum": 100},
                    "6": {"type": "number", "minimum": 0, "maximum": 100},
                    "7": {"type": "number", "minimum": 0, "maximum": 100},
                    "8": {"type": "number", "minimum": 0, "maximum": 100},
                    "9": {"type": "number", "minimum": 0, "maximum": 100},
                    "10": {"type": "number", "minimum": 0, "maximum": 100},
                    "11": {"type": "number", "minimum": 0, "maximum": 100},
                    "12": {"type": "number", "minimum": 0, "maximum": 100},
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


async def async_llm(prompt: str, model: str) -> str:
    """根据配置选择使用 vLLM 或 Ollama 或 云端API"""
    if USE_CLOUD_API:
        return await cloud_api_chat(prompt, model)
    elif USE_VLLM:
        return await vllm_chat(prompt, model)
    else:
        return await ollama_chat(prompt, model)


async def vllm_chat(prompt: str, model: str) -> str:
    """使用vLLM进行聊天"""
    async with llm_semaphore:
        endpoint = await get_llm_endpoint()
        headers = await get_llm_headers()

        payload = await create_llm_payload(
            prompt=prompt, model=model, temperature=0.2, max_tokens=2048, top_p=0.95
        )

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=600)
        ) as session:
            async with session.post(
                f"{endpoint}/v1/chat/completions", json=payload, headers=headers
            ) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"vLLM 请求失败: {resp.status}")
                data = await resp.json()
                text = data["choices"][0]["message"]["content"].strip()

                if "</think>" in text:
                    text = text.split("</think>", 1)[-1].strip()

                return text


async def cloud_api_chat(prompt: str, model: str) -> str:
    """使用云端API进行聊天"""
    async with llm_semaphore:
        endpoint = await get_llm_endpoint()
        headers = await get_llm_headers()

        payload = await create_llm_payload(
            prompt=prompt, model=model, temperature=0.2, max_tokens=2048, top_p=0.95
        )

        """
        logging.info(f"发送请求到云端API: {endpoint}/v1/chat/completions")
        logging.info(f"请求头: { {k: v for k, v in headers.items() if k != 'Authorization'} }")
        logging.info(f"请求负载模型: {payload.get('model')}")
        logging.info(f"请求负载消息长度: {len(payload.get('messages', []))}")
        """

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=600)
        ) as session:
            async with session.post(
                f"{endpoint}/v1/chat/completions", json=payload, headers=headers
            ) as resp:
                response_text = await resp.text()

                if resp.status != 200:
                    # 详细记录错误信息
                    logging.error(f"云端API请求失败:")
                    logging.error(f"状态码: {resp.status}")
                    logging.error(f"响应头: {dict(resp.headers)}")
                    logging.error(f"响应内容: {response_text}")
                    logging.error(f"请求URL: {endpoint}/v1/chat/completions")
                    logging.error(f"请求方法: POST")

                    raise RuntimeError(
                        f"云端API请求失败: {resp.status}, 响应: {response_text}"
                    )

                data = await resp.json()
                text = data["choices"][0]["message"]["content"].strip()

                # 清理响应文本
                if "</think>" in text:
                    text = text.split("</think>", 1)[-1].strip()

                return text


async def ollama_chat(prompt: str, model: str) -> str:
    """使用Ollama进行聊天"""
    async with llm_semaphore:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=600)
        ) as session:
            async with session.post(
                f"{OLLAMA_HOST}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
            ) as resp:
                data = await resp.json()
                text = data.get("response", "")
                return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def create_embeddings(use_cloud: bool = None, custom_endpoint: str = None):
    """
    创建嵌入模型实例的工厂函数

    Args:
        use_cloud: 是否使用云端嵌入模型，None表示使用配置默认值
        custom_endpoint: 自定义端点，如果提供则覆盖默认端点
    """
    if custom_endpoint:
        return VLLMEmbeddings(custom_endpoint)

    from config.config import USE_CLOUD_EMBEDDING

    if not USE_CLOUD_EMBEDDING:
        return VLLMEmbeddings(VLLM_EMBEDDING_ENDPOINT)
    else:
        from config.config import CLOUD_EMBEDDING_ENDPOINT

        return VLLMEmbeddings(CLOUD_EMBEDDING_ENDPOINT)


class VLLMEmbeddings:
    def __init__(self, vllm_endpoint: str = None):
        from config.config import (
            CLOUD_EMBEDDING_ENDPOINT,
            USE_CLOUD_EMBEDDING,
            VLLM_EMBEDDING_ENDPOINT,
        )

        # 如果提供了端点，使用提供的端点；否则根据配置选择
        if vllm_endpoint:
            self.endpoint = vllm_endpoint
            # 根据端点判断是否是云端
            self.use_cloud = "dashscope.aliyuncs.com" in vllm_endpoint
        else:
            self.use_cloud = USE_CLOUD_EMBEDDING
            self.endpoint = (
                CLOUD_EMBEDDING_ENDPOINT
                if USE_CLOUD_EMBEDDING
                else VLLM_EMBEDDING_ENDPOINT
            )

        self.timeout = aiohttp.ClientTimeout(total=60)
        self.batch_size = 10  # 云端API批量大小限制

    async def get_embedding_headers(self) -> Dict[str, str]:
        """获取嵌入模型请求头"""
        headers = {"Content-Type": "application/json"}
        if self.use_cloud:

            # 优先使用嵌入专用API密钥，如果没有则使用通用API密钥
            api_key = config.CLOUD_API_KEY
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            else:
                logging.warning("未找到云端嵌入API密钥")
        return headers

    async def create_embedding_payload(self, texts: List[str]) -> Dict[str, Any]:
        """创建嵌入模型请求负载"""
        from config.config import CLOUD_EMBEDDING_MODEL, EMBEDDING_DIMENSION

        if self.use_cloud:
            # 云端嵌入API格式 - 必须包含model参数
            payload = {
                "model": CLOUD_EMBEDDING_MODEL,
                "input": texts,
                "encoding_format": "float",
            }
            # 只有在云端模式下才添加维度参数
            if EMBEDDING_DIMENSION:
                payload["dimensions"] = EMBEDDING_DIMENSION
            return payload
        else:
            # 本地vLLM嵌入API格式
            return {"input": texts}

    async def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """处理单个批次的嵌入请求"""
        headers = await self.get_embedding_headers()
        payload = await self.create_embedding_payload(texts)

        # logging.info(f"处理嵌入批次 - 文本数量: {len(texts)}")

        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.post(
                self.endpoint, json=payload, headers=headers
            ) as response:
                if response.status != 200:
                    response_text = await response.text()
                    logging.error(
                        f"嵌入批次请求失败: {response.status}, 响应: {response_text}"
                    )
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"嵌入请求失败: {response.status} - {response_text}",
                    )

                result = await response.json()

                # 处理不同的响应格式
                if self.use_cloud:
                    # 云端API响应格式：{"data": [{"embedding": [...]}, ...]}
                    embeddings = [item["embedding"] for item in result["data"]]
                    return embeddings
                else:
                    # 本地vLLM响应格式：{"data": [{"embedding": [...]}, ...]} 或直接是嵌入列表
                    if "data" in result:
                        return [item["embedding"] for item in result["data"]]
                    else:
                        return result

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """为文档列表生成嵌入向量（异步）- 支持批量处理"""
        try:
            # logging.info(f"开始嵌入生成 - 总文本数量: {len(texts)}")
            # logging.info(f"使用云端: {self.use_cloud}")

            if self.use_cloud:
                # 云端API需要分批处理
                batch_size = self.batch_size
                all_embeddings = []

                # 将文本分成批次
                batches = [
                    texts[i : i + batch_size] for i in range(0, len(texts), batch_size)
                ]
                # logging.info(f"将 {len(texts)} 个文本分成 {len(batches)} 个批次，每批最多 {batch_size} 个")

                # 处理每个批次
                for i, batch in enumerate(batches):
                    # logging.info(f"处理批次 {i+1}/{len(batches)} - 文本数量: {len(batch)}")
                    try:
                        batch_embeddings = await self._embed_batch(batch)
                        all_embeddings.extend(batch_embeddings)
                        # logging.info(f"批次 {i+1} 完成，获得 {len(batch_embeddings)} 个嵌入向量")
                    except Exception as e:
                        # logging.error(f"批次 {i+1} 处理失败: {str(e)}")
                        raise

                # 验证总数
                if len(all_embeddings) != len(texts):
                    logging.warning(
                        f"嵌入向量数量不匹配: 期望 {len(texts)}，实际 {len(all_embeddings)}"
                    )

                # 记录实际维度
                if all_embeddings and len(all_embeddings) > 0:
                    actual_dim = len(all_embeddings[0])
                    # logging.info(f"云端嵌入生成完成，总嵌入向量: {len(all_embeddings)}，维度: {actual_dim}")

                return all_embeddings
            else:
                # 本地API可以一次性处理
                logging.info(f"本地嵌入请求 - 文本数量: {len(texts)}")
                embeddings = await self._embed_batch(texts)
                if embeddings and len(embeddings) > 0:
                    actual_dim = len(embeddings[0])
                    logging.info(f"本地嵌入生成完成，维度: {actual_dim}")
                return embeddings

        except aiohttp.ClientError as e:
            logging.error(f"网络请求失败: {str(e)}")
            raise HTTPException(status_code=500, detail=f"网络请求失败: {str(e)}")
        except Exception as e:
            logging.error(f"嵌入生成失败: {str(e)}")
            raise HTTPException(status_code=500, detail=f"嵌入生成失败: {str(e)}")

    async def embed_query(self, text: str) -> List[float]:
        """为单个查询生成嵌入向量（异步）"""
        return (await self.embed_documents([text]))[0]
