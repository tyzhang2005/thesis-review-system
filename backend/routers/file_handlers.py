import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import aiohttp
import uvicorn
from aiohttp import ClientSession
from chromadb import PersistentClient

# from langchain_ollama import OllamaEmbeddings, OllamaLLM
from chromadb.config import Settings
from config.config import (
    MAX_FILE_SIZE,
    MINERU_API_BASE,
    MINERU_OUTPUT_DIR,
    MINERU_TOKEN,
    PDF_CONVERT_METHOD,
    UPLOAD_FOLDER,
    USER_MD_DIR,
    USER_RESULT_DIR,
)
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from langchain_chroma import Chroma
from langchain_community.document_loaders import PDFPlumberLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from pydantic import BaseModel

router = APIRouter()


# PDF转Markdown
def convert_pdf_to_markdown(
    file_path: str, username: str, output_dir=USER_MD_DIR
) -> str:
    """PDF转换函数"""
    if PDF_CONVERT_METHOD == "local":
        return _convert_via_local(file_path, output_dir, username)
    elif PDF_CONVERT_METHOD == "cloud":
        return _convert_via_cloud(file_path, output_dir, username)
    else:
        raise ValueError(f"无效的PDF转换模式: {PDF_CONVERT_METHOD}")


# 本地转换实现
def _convert_via_local(file_path: str, output_dir: str, username: str) -> str:
    """使用 MinerU 命令行工具进行本地转换，保持原有文件路径结构"""

    try:

        # 确保文件路径是绝对路径
        uploads_path = os.path.join(
            UPLOAD_FOLDER, username, os.path.basename(file_path)
        )
        file_path = uploads_path
        logging.info(f"使用uploads路径: {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        # 原有magic_pdf转换逻辑的文件名处理
        base_name = os.path.basename(file_path)
        md_filename = os.path.splitext(base_name)[0] + ".md"
        md_path = os.path.join(output_dir, md_filename)

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 创建临时目录用于 MinerU 输出
        temp_output_dir = tempfile.mkdtemp()

        try:
            # 构建 MinerU 命令
            cmd = [
                "mineru",
                "-p",
                file_path,
                "-o",
                temp_output_dir,
                "-b",
                "pipeline",
                "-d",
                "cuda",
                "--source",
                "local",
                "--vram",
                "8",
            ]

            logging.info(f"执行 MinerU 命令: {' '.join(cmd)}")

            # 执行转换
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300  # 5分钟超时
            )

            logging.info(f"MinerU 返回码: {result.returncode}")
            if result.stdout:
                logging.info(f"MinerU 输出: {result.stdout}")
            if result.stderr:
                logging.warning(f"MinerU 错误输出: {result.stderr}")

            if result.returncode != 0:
                error_msg = f"MinerU 转换失败: {result.stderr}"
                logging.error(error_msg)
                raise Exception(error_msg)

            # 查找生成的 Markdown 文件
            input_filename = Path(file_path).stem
            md_file = find_markdown_file(temp_output_dir, input_filename)

            if not md_file:
                # 尝试在输出目录中查找任何 .md 文件
                md_files = list(Path(temp_output_dir).rglob("*.md"))
                if md_files:
                    md_file = str(md_files[0])
                    logging.info(f"找到Markdown文件: {md_file}")
                else:
                    # 列出临时目录中的所有文件来调试
                    all_files = list(Path(temp_output_dir).rglob("*"))
                    logging.warning(f"临时目录中的文件: {[str(f) for f in all_files]}")
                    raise FileNotFoundError("未找到生成的Markdown文件")

            # 保存所有 MinerU 输出文件到 mineru_output 目录
            mineru_output_path = MINERU_OUTPUT_DIR / input_filename
            mineru_output_path.mkdir(parents=True, exist_ok=True)

            # 复制所有文件从临时目录到输出目录
            temp_path = Path(temp_output_dir)
            copied_files = []
            for item in temp_path.rglob("*"):
                if item.is_file():
                    # 计算相对路径
                    rel_path = item.relative_to(temp_path)
                    dest_path = mineru_output_path / rel_path
                    # 确保目标目录存在
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, dest_path)
                    copied_files.append(str(rel_path))
                    logging.info(f"已复制文件: {rel_path}")

            logging.info(f"共复制 {len(copied_files)} 个文件到 {mineru_output_path}")

            # 同时将 md 文件复制到原有位置（向后兼容）
            shutil.copy2(md_file, md_path)
            logging.info(f"Markdown文件已复制至: {md_path}")

            # 验证写入结果
            if not os.path.exists(md_path):
                raise ValueError("生成的Markdown文件不存在")

            logging.info(f"转换成功，Markdown文件大小: {os.path.getsize(md_path)} 字节")
            return md_path

        except subprocess.TimeoutExpired:
            error_msg = "MinerU 转换超时"
            logging.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            logging.error(f"转换过程出错: {str(e)}")
            raise Exception(f"PDF转换失败: {str(e)}")
        finally:
            # 清理临时目录
            try:
                shutil.rmtree(temp_output_dir, ignore_errors=True)
                logging.info(f"已清理临时目录: {temp_output_dir}")
            except Exception as e:
                logging.warning(f"清理临时目录失败: {e}")

    except Exception as e:
        logging.error(f"MinerU 转换失败: {str(e)}")
        raise Exception(f"本地转换失败: {str(e)}")


def find_markdown_file(output_dir: str, filename: str) -> str:
    """在输出目录中查找Markdown文件"""
    output_path = Path(output_dir)

    # 首先在直接子目录中查找
    for pattern in [f"**/{filename}.md", f"**/*{filename}*.md"]:
        for md_file in output_path.glob(pattern):
            if md_file.is_file():
                return str(md_file)

    # 查找任何.md文件
    for md_file in output_path.rglob("*.md"):
        if md_file.is_file():
            return str(md_file)

    return None


# 云服务转换实现
def _convert_via_cloud(file_path: str, output_dir: str, username: str) -> str:
    """使用Mineru云服务API转换"""
    import logging
    import os
    import shutil
    import tempfile
    import time
    import zipfile
    from pathlib import Path

    import requests

    try:
        # 原有Mineru API转换逻辑
        base_name = os.path.basename(file_path)
        md_filename = os.path.splitext(base_name)[0] + ".md"
        md_path = os.path.join(output_dir, md_filename)

        lang = None

        # 辅助函数定义
        def create_upload_task(file_name):
            """申请文件上传链接并创建解析任务"""
            url = f"{MINERU_API_BASE}/file-urls/batch"
            headers = {
                "Authorization": MINERU_TOKEN,
                "Content-Type": "application/json",
            }
            payload = {
                "enable_formula": True,
                "enable_table": True,
                "language": lang or "auto",  # 使用传入的lang参数
                "model_version": "v2",
                "files": [{"name": file_name, "is_ocr": True}],
            }

            response = requests.post(url, headers=headers, json=payload)
            if response.status_code != 200 or response.json().get("code") != 0:
                raise Exception(f"任务创建失败: {response.text}")

            data = response.json().get("data", {})
            return data["batch_id"], data["file_urls"][0]

        def upload_file_to_mineru(upload_url, file_path):
            """上传本地文件到 MinerU"""
            with open(file_path, "rb") as f:
                headers = {"Content-Length": str(os.path.getsize(file_path))}
                response = requests.put(upload_url, data=f, headers=headers)
                if response.status_code != 200:
                    raise Exception(f"文件上传失败: {response.status_code}")

        def poll_task_result(batch_id):
            """轮询任务结果直到完成"""
            url = f"{MINERU_API_BASE}/extract-results/batch/{batch_id}"
            headers = {"Authorization": MINERU_TOKEN}

            max_retries = 100
            retry_interval = 5

            for _ in range(max_retries):
                response = requests.get(url, headers=headers)
                if response.status_code != 200:
                    raise Exception(f"查询失败: {response.status_code}")

                data = response.json().get("data", {})
                task = data.get("extract_result", [{}])[0]

                if task.get("state") == "done":
                    return task.get("full_zip_url")
                elif task.get("state") == "failed":
                    raise Exception(f"解析失败: {task.get('err_msg', '未知错误')}")

                logging.info(
                    f"任务处理中: {task.get('state')}, 进度: {task.get('extract_progress', {}).get('extracted_pages', 0)}页"
                )
                time.sleep(retry_interval)

            raise Exception("任务处理超时")

        def download_and_extract_md(zip_url, output_md_path):
            """下载并提取 Markdown 文件到指定路径"""
            # 下载 ZIP 文件
            response = requests.get(zip_url)
            zip_path = os.path.join(tempfile.gettempdir(), "mineru_temp.zip")
            with open(zip_path, "wb") as f:
                f.write(response.content)

            # 解压并查找 Markdown 文件
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                for file_info in zip_ref.infolist():
                    if file_info.filename.endswith(".md"):
                        # 提取到指定路径
                        with zip_ref.open(file_info) as source, open(
                            output_md_path, "wb"
                        ) as target:
                            shutil.copyfileobj(source, target)
                        return output_md_path

            raise Exception("未找到 Markdown 文件")

        try:
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.basename(file_path)
            md_filename = os.path.splitext(base_name)[0] + ".md"
            md_path = os.path.join(output_dir, md_filename)

            upload_username = os.path.join(UPLOAD_FOLDER, username)
            file_upload_path = os.path.join(upload_username, file_path)

            # 检查文件路径是否存在
            if not os.path.exists(file_upload_path):
                logging.info(f"文件不存在: {file_path}")
                raise FileNotFoundError(f"文件不存在: {file_path}")

            # 使用Mineru API进行转换
            logging.info("步骤 1: 创建上传任务")
            batch_id, upload_url = create_upload_task(base_name)

            logging.info("步骤 2: 上传文件")
            upload_file_to_mineru(upload_url, file_upload_path)

            logging.info("步骤 3: 等待解析完成")
            result_url = poll_task_result(batch_id)

            logging.info("步骤 4: 下载结果")
            download_and_extract_md(result_url, md_path)

            logging.info(f"解析成功! Markdown 文件已保存至: {md_path}")

        except Exception as e:
            logging.error(f"处理失败: {str(e)}")
            raise Exception(f"PDF转换失败: {str(e)}")

    except Exception as e:
        logging.error(f"云服务转换失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"云服务转换失败: {str(e)}")

    logging.info(f"使用云服务转换: {file_path}")
    return md_path


# 文件上传
@router.post("/upload")
async def upload_file(
    request: Request,
    pdfFile: UploadFile = File(..., description="要上传的PDF文件"),
    username: str = Form(..., description="用户名"),
):
    """接收并保存上传的文件。"""
    try:
        if int(request.headers.get("Content-Length")) > MAX_FILE_SIZE:  # 10MB 限制
            raise HTTPException(
                status_code=413,
                detail=f"文件过大，最大支持 {MAX_FILE_SIZE / 1024 / 1024:.1f}MB",
            )

        # 检查文件名是否为空
        if not pdfFile.filename.strip():
            raise HTTPException(status_code=400, detail="文件名不能为空")

        # 检查文件是否为PDF
        if not pdfFile.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="仅支持 PDF 文件")

        # 创建用户专属文件夹
        user_folder = UPLOAD_FOLDER / username
        user_folder.mkdir(exist_ok=True)

        # 保存文件
        file_path = user_folder / pdfFile.filename

        # 保存文件内容
        with file_path.open("wb") as buffer:
            content = await pdfFile.read()
            buffer.write(content)

        logging.info(f"用户 {username} 上传文件: {pdfFile.filename}")
        logging.info(f"文件大小: {pdfFile.size} 字节")
        logging.info(f"保存路径: {file_path}")

        return JSONResponse(
            {"success": True, "message": "文件上传成功", "file_path": str(file_path)}
        )
    except Exception as e:
        logging.error(f"文件上传失败: {e}")
        return JSONResponse({"error": "文件上传失败"}), 500


@router.post("/download_pdf")
async def download_pdf(request: Request):
    """下载指定用户的 PDF 文件。"""
    try:
        if int(request.headers.get("Content-Length")) > 1024:
            return JSONResponse(content={"error": "请求体过大"}, status_code=413)

        data = await request.json()
        logging.info(f"收到来自 {request.client.host} 的下载请求: {data}")

        if not isinstance(data, dict):
            return JSONResponse(content={"error": "无效的 JSON 格式"}, status_code=400)

        username = data.get("username")
        if not username or not isinstance(username, str):
            logging.error("缺少或无效的用户名")
            return JSONResponse(content={"error": "无效的用户名参数"}, status_code=401)

        # 检查用户文件夹是否存在
        user_folder = os.path.join(USER_RESULT_DIR)
        logging.info(f"检查用户文件夹: {user_folder}")
        if not os.path.exists(user_folder):
            logging.warning(f"用户文件夹不存在: {user_folder}")
            return JSONResponse(content={"error": "用户文件夹不存在"}, status_code=402)

        # 检查文件夹中是否有 PDF 文件夹
        pdfname = os.path.splitext(os.path.basename(data.get("pdfname")))[0]
        pdf_folders = os.path.join(user_folder, pdfname)
        logging.info(f"检查 PDF 文件夹: {pdf_folders}")
        if not os.path.exists(pdf_folders):
            logging.warning(f"用户 {username} 没有分析好的 PDF 文件夹")
            return JSONResponse(
                content={"error": "用户没有分析好的 PDF 文件夹"}, status_code=403
            )

        # 检查是否生成了评审表
        review_table_path = os.path.join(pdf_folders, "review_table.pdf")
        logging.info(f"检查评审表: {review_table_path}")
        if not os.path.exists(review_table_path):
            logging.warning(f"用户 {username} 没有生成评审表")
            return JSONResponse(
                content={"error": "用户没有生成评审表"}, status_code=405
            )
        else:
            logging.info(f"用户 {username} 生成评审表成功")
            return FileResponse(
                review_table_path,
                media_type="application/pdf",
                filename="review_table.pdf",
            )

    except json.JSONDecodeError:
        logging.error("无效的 JSON 负载")
        return JSONResponse(content={"error": "无效的 JSON 格式"}, status_code=400)
    except Exception as e:
        logging.error(f"意外错误: {e}", exc_info=True)
        return JSONResponse(content={"error": "服务器内部错误"}, status_code=500)
