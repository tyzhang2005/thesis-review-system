import logging
import os
import sys
from contextlib import asynccontextmanager

import uvicorn
from config.config import ANNOTATED_PDF_DIR, CORS_ORIGINS, STORAGE_FILE, config
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from routers.advice_retrieval import router as advice_retrieval_router
from routers.aigc_detector import router as aigc_router
from routers.auth import check_sub, load_users, login_user, logout, register_user
from routers.auth import router as auth_router
from routers.auth import sub_check
from routers.evaluation import router as eval_router
from routers.evaluation import run_evaluation
from routers.file_handlers import download_pdf
from routers.file_handlers import router as file_router
from routers.file_handlers import upload_file
from routers.human_analysis import router as database_build_router
from routers.user_result_vectorstore import router as user_result_vector_router
from routers.vectorstore import router as vector_router

# 配置日志和禁用遥测
logging.basicConfig(level=logging.INFO)
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# CORS_ORIGINS 现在从 config/settings.json 读取
# 可以通过环境变量 CORS_ORIGINS 覆盖
if os.getenv("CORS_ORIGINS"):
    CORS_ORIGINS = os.getenv("CORS_ORIGINS").split(",")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时执行
    setup_logging()
    config.load_config()
    print("Configuration loaded successfully")

    logging.info("应用启动，加载用户数据...")
    # 初始化用户数据
    if os.path.exists(STORAGE_FILE):
        load_users()

    yield

    # 关闭时执行（如果需要）
    logging.info("应用关闭")


app = FastAPI(lifespan=lifespan)

# CORS配置 - 从环境变量读取允许的来源
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)

# 包含路由
app.include_router(auth_router, prefix="/api")
app.include_router(file_router, prefix="/api")
app.include_router(eval_router, prefix="/api")
app.include_router(aigc_router, prefix="/api")
app.include_router(database_build_router, prefix="/api")
# 挂载路由到根路径
app.include_router(vector_router)
app.include_router(user_result_vector_router)
app.include_router(advice_retrieval_router)

# 挂载静态文件服务
app.mount(
    "/annotated_pdfs",
    StaticFiles(directory=str(ANNOTATED_PDF_DIR)),
    name="annotated_pdfs",
)


def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 清除现有的处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # 创建格式化器
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)

    # 添加处理器
    logger.addHandler(console_handler)


# 依赖注入，确保配置已加载
def get_config():
    if config._config_data is None:
        raise HTTPException(status_code=500, detail="Configuration not loaded")
    return config


@app.get("/")
async def health_check():
    return {"status": "running", "message": "Dissertation Evaluation System"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
