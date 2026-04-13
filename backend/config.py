import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量（用于敏感信息）
load_dotenv()

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def load_settings():
    """加载settings.json配置文件"""
    settings_path = BASE_DIR / "config" / "settings.json"
    try:
        with open(settings_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.warning(f"Settings file not found at {settings_path}, using defaults")
        return {}
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in settings file: {e}")
        return {}


# 加载配置
_settings = load_settings()


def get_setting(path: str, default=None):
    """
    从settings中获取嵌套配置值
    例如: get_setting("api.cloud.model", "default_model")
    """
    keys = path.split('.')
    value = _settings
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value


# ============================================================================
# 敏感信息 - 从环境变量读取
# ============================================================================

def get_env(key: str, default: str = "") -> str:
    """获取环境变量，如果不存在则返回默认值"""
    value = os.getenv(key, default)
    if not value and not default:
        logging.warning(f"Environment variable {key} not set")
    return value


# API密钥 - 从环境变量读取
CLOUD_API_KEY = get_env("CLOUD_API_KEY")
MINERU_TOKEN = get_env("MINERU_TOKEN")


# ============================================================================
# 基础路径配置
# ============================================================================

UPLOAD_FOLDER = DATA_DIR / "uploads"
PAPERS_FOLDER_PATH = DATA_DIR / "databases" / "test_database"
DATABASE_VECTORSTORE_DIR = DATA_DIR / "databases" / "chroma_db"
ADVICE_DATA_PATH = DATA_DIR / "advice_data.txt"
USER_MD_DIR = DATA_DIR / "processed" / "markdown"
USER_RESULT_DIR = DATA_DIR / "results" / "evaluations"
USER_AIGC_RESULT_DIR = DATA_DIR / "processed" / "aigc_detect"
MINERU_OUTPUT_DIR = DATA_DIR / "processed" / "mineru"
ADVICE_RETRIEVAL_DIR = DATA_DIR / "results" / "advice_retrieval"
ANNOTATED_PDF_DIR = DATA_DIR / "annotated_pdfs"

# 创建目录
for path in [UPLOAD_FOLDER, USER_MD_DIR, DATABASE_VECTORSTORE_DIR, PAPERS_FOLDER_PATH,
             MINERU_OUTPUT_DIR, ADVICE_RETRIEVAL_DIR, ANNOTATED_PDF_DIR]:
    path.mkdir(parents=True, exist_ok=True)


# ============================================================================
# API配置 - 从settings.json读取
# ============================================================================

USE_CLOUD_API = get_setting("api.use_cloud", True)
USE_VLLM = get_setting("api.use_vllm", True)
MAX_CONCURRENT = get_setting("api.max_concurrent", 32)

# 云端API配置
CLOUD_API_ENDPOINT = get_setting("api.cloud.endpoint", "https://dashscope.aliyuncs.com/compatible-mode")
CLOUD_MODEL = get_setting("api.cloud.model", "deepseek-v3.2-exp")

# 云端嵌入模型配置
USE_CLOUD_EMBEDDING = get_setting("models.use_cloud_embedding", True)
CLOUD_EMBEDDING_ENDPOINT = get_setting("api.cloud.embedding.endpoint",
                                       "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings")
CLOUD_EMBEDDING_MODEL = get_setting("api.cloud.embedding.model", "text-embedding-v4")
EMBEDDING_DIMENSION = get_setting("api.cloud.embedding.dimension", 2048)

# vLLM配置
VLLM_HOST = get_setting("api.vllm.host", "http://localhost:8003")
VLLM_EMBEDDING_ENDPOINT = get_setting("api.vllm.embedding_endpoint", "http://localhost:8004/v1/embeddings")

# Ollama配置
OLLAMA_HOST = get_setting("api.ollama.host", "http://localhost:11434")


# ============================================================================
# 模型配置
# ============================================================================

AIGC_MODEL_PATH = get_setting("models.aigc_detector_path", "./models/AIGC_text_detector")


# ============================================================================
# PDF转换配置
# ============================================================================

PDF_CONVERT_METHOD = get_setting("pdf.convert_method", "local")
MINERU_API_BASE = get_setting("pdf.mineru.api_base", "https://mineru.net/api/v4")

# 检查MinerU Token
if not MINERU_TOKEN:
    logging.warning("MINERU_TOKEN not set in environment variables. PDF conversion may not work.")


# ============================================================================
# 文件上传配置
# ============================================================================

MAX_FILE_SIZE = get_setting("upload.max_file_size_mb", 50) * 1024 * 1024
SUB_LIMIT = get_setting("upload.sub_limit", 3)


# ============================================================================
# 向量数据库配置
# ============================================================================

DATABASE_VECTORSTORE_DIR_LOCAL = DATA_DIR / get_setting("vectorstore.local_dir", "databases/vectorstore_local")
DATABASE_VECTORSTORE_DIR_CLOUD = DATA_DIR / get_setting("vectorstore.cloud_dir", "databases/vectorstore_cloud")
DATABASE_USER_RESULT_DIR_LOCAL = DATA_DIR / get_setting("vectorstore.user_result_local_dir", "databases/user_result_local")
DATABASE_USER_RESULT_DIR_CLOUD = DATA_DIR / get_setting("vectorstore.user_result_cloud_dir", "databases/user_result_cloud")

# 路径配置
USER_RESULT_ADVICE_PATH = get_setting("paths.user_result_advice", "./data/user_result")
# 如果是相对路径，转换为绝对路径
if not os.path.isabs(USER_RESULT_ADVICE_PATH):
    USER_RESULT_ADVICE_PATH = str(DATA_DIR / USER_RESULT_ADVICE_PATH)


def get_vectorstore_dir():
    """根据嵌入模型类型获取向量数据库目录"""
    if USE_CLOUD_EMBEDDING:
        return DATABASE_VECTORSTORE_DIR_CLOUD
    else:
        return DATABASE_VECTORSTORE_DIR_LOCAL


def get_collection_name():
    """根据嵌入模型类型获取集合名称"""
    prefix = get_setting(f"vectorstore.collection_prefix.{'cloud' if USE_CLOUD_EMBEDDING else 'local'}", "local_4b")
    return f"advice_collection_{prefix}"


def get_user_result_vectorstore_dir():
    """根据嵌入模型类型获取用户评审结果向量数据库目录"""
    if USE_CLOUD_EMBEDDING:
        return DATABASE_USER_RESULT_DIR_CLOUD
    else:
        return DATABASE_USER_RESULT_DIR_LOCAL


def get_user_result_collection_name(advice_type: str):
    """
    根据建议类型和嵌入模型获取集合名称
    advice_type: 'content' 或 'format'
    """
    prefix = get_setting(f"vectorstore.collection_prefix.{'cloud' if USE_CLOUD_EMBEDDING else 'local'}", "local_4b")
    return f"user_result_{advice_type}_collection_{prefix}"


# ============================================================================
# CORS配置
# ============================================================================

CORS_ORIGINS = get_setting("cors.origins", ["http://localhost:3000", "http://localhost:5173"])


# ============================================================================
# 兼容旧代码的导出
# ============================================================================

# 数据库文件
STORAGE_FILE = DATA_DIR / 'registered_users.json'

# 旧版本兼容
CLOUD_MODEL_HARDCODED = CLOUD_MODEL

# 使用config_manager管理云端api-key（保持向后兼容）
from config.config_manager import config_manager


class Config:
    """配置管理类 - 保持向后兼容"""
    def __init__(self):
        self._config_data = None
        self._api_key = None

    def load_config(self):
        """加载配置"""
        self._config_data = config_manager.load_config()
        self._api_key = self._config_data.get("CLOUD_API_KEY")

    def get(self, key, default=None):
        """获取配置值"""
        if self._config_data is None:
            raise RuntimeError("Config not loaded. Call load_config() first.")
        return self._config_data.get(key, default)

    @property
    def CLOUD_API_KEY(self):
        """获取云端API密钥"""
        # 优先从加密配置文件获取
        if self._api_key is None and self._config_data is None:
            try:
                self.load_config()
            except:
                pass
        # 如果加密配置没有，从环境变量获取
        if self._api_key is None:
            self._api_key = CLOUD_API_KEY
            if not self._api_key:
                raise RuntimeError(
                    "CLOUD_API_KEY not found. Please set it in .env file "
                    "or run create_config.py to create encrypted config."
                )
        return self._api_key


config = Config()
