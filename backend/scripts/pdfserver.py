import logging
import os
import shutil
import tempfile
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

# 配置常量
UPLOAD_FOLDER = Path("magicpdf_uploads")
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
USER_RESULT_DIR = Path("magicpdf_test")
USER_MD_DIR = Path("magicpdf_result")
PDF_CONVERT_METHOD = "local"  # "local" 或 "cloud"

# 创建必要的目录
UPLOAD_FOLDER.mkdir(exist_ok=True)
USER_RESULT_DIR.mkdir(exist_ok=True)
USER_MD_DIR.mkdir(exist_ok=True)

# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 创建 FastAPI 应用
app = FastAPI(title="PDF to Markdown Converter", version="1.0.0")

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/convert-pdf-to-markdown")
async def convert_pdf_to_markdown_endpoint(
    pdfFile: UploadFile = File(..., description="要转换的PDF文件"),
    username: str = Form(..., description="用户名"),
):
    """将上传的PDF文件转换为Markdown格式并返回"""
    try:
        # 验证文件类型
        if not pdfFile.filename or not pdfFile.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="仅支持 PDF 文件")

        # 创建用户专属文件夹
        user_folder = UPLOAD_FOLDER / username
        user_folder.mkdir(exist_ok=True)

        # 保存上传的PDF文件
        file_path = user_folder / pdfFile.filename
        with file_path.open("wb") as buffer:
            content = await pdfFile.read()
            buffer.write(content)

        logger.info(f"用户 {username} 上传PDF文件进行转换: {pdfFile.filename}")

        # 创建Markdown输出目录
        user_md_dir = USER_MD_DIR / username
        user_md_dir.mkdir(exist_ok=True)

        # 调用转换函数
        md_path = convert_pdf_to_markdown(str(file_path), username, str(user_md_dir))

        # 读取生成的Markdown文件内容
        with open(md_path, "r", encoding="utf-8") as f:
            markdown_content = f.read()

        # 返回Markdown内容
        return JSONResponse(
            {
                "success": True,
                "message": "PDF转换成功",
                "markdown_content": markdown_content,
                "file_name": os.path.basename(md_path),
                "file_path": md_path,
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PDF转换失败: {e}")
        return JSONResponse(
            {"success": False, "error": f"PDF转换失败: {str(e)}"}, status_code=500
        )


@app.post("/download-markdown")
async def download_markdown_endpoint(request: Request):
    """下载转换后的Markdown文件"""
    try:
        data = await request.json()
        username = data.get("username")
        file_path = data.get("file_path")

        if not username or not file_path:
            return JSONResponse({"error": "缺少用户名或文件路径"}, status_code=400)

        # 验证文件路径安全性
        if not file_path.startswith(str(USER_MD_DIR)):
            return JSONResponse({"error": "无效的文件路径"}, status_code=400)

        # 检查文件是否存在
        if not os.path.exists(file_path):
            return JSONResponse({"error": "Markdown文件不存在"}, status_code=404)

        # 返回文件
        return FileResponse(
            file_path, media_type="text/markdown", filename=os.path.basename(file_path)
        )

    except Exception as e:
        logger.error(f"下载Markdown文件失败: {e}")
        return JSONResponse({"error": "下载失败"}, status_code=500)


@app.post("/upload")
async def upload_file(
    pdfFile: UploadFile = File(..., description="要上传的PDF文件"),
    username: str = Form(..., description="用户名"),
):
    """接收并保存上传的文件"""
    try:
        # 检查文件是否为PDF
        if not pdfFile.filename or not pdfFile.filename.endswith(".pdf"):
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

        logger.info(f"用户 {username} 上传文件: {pdfFile.filename}")
        logger.info(f"文件大小: {len(content)} 字节")
        logger.info(f"保存路径: {file_path}")

        return JSONResponse(
            {"success": True, "message": "文件上传成功", "file_path": str(file_path)}
        )
    except Exception as e:
        logger.error(f"文件上传失败: {e}")
        return JSONResponse({"error": "文件上传失败"}, status_code=500)


def convert_pdf_to_markdown(
    file_path: str, username: str, output_dir: str = None
) -> str:
    """PDF转换函数"""
    if output_dir is None:
        output_dir = str(USER_MD_DIR / username)

    if PDF_CONVERT_METHOD == "local":
        return _convert_via_local(file_path, output_dir, username)
    elif PDF_CONVERT_METHOD == "cloud":
        return _convert_via_cloud(file_path, output_dir, username)
    else:
        raise ValueError(f"无效的PDF转换模式: {PDF_CONVERT_METHOD}")


def _convert_via_local(file_path: str, output_dir: str, username: str) -> str:
    """使用magic_pdf库进行本地转换"""
    try:
        import fitz  # PyMuPDF

        try:
            from magic_pdf.data.batch_build_dataset import batch_build_dataset
            from magic_pdf.data.data_reader_writer import FileBasedDataReader
            from magic_pdf.tools.common import batch_do_parse, do_parse
            from magic_pdf.utils.office_to_pdf import convert_file_to_pdf
        except ImportError as e:
            logger.error(f"magic_pdf 导入失败: {e}")
            raise HTTPException(status_code=500, detail=f"magic_pdf 库未安装: {e}")

        # 原有magic_pdf转换逻辑
        base_name = os.path.basename(file_path)
        md_filename = os.path.splitext(base_name)[0] + ".md"
        md_path = os.path.join(output_dir, md_filename)

        pdf_suffixes = [".pdf"]
        ms_office_suffixes = [".ppt", ".pptx", ".doc", ".docx"]
        image_suffixes = [".png", ".jpeg", ".jpg"]

        os.makedirs(output_dir, exist_ok=True)
        temp_dir = tempfile.mkdtemp()

        method = "auto"
        lang = None
        debug_able = False
        start_page_id = 0
        end_page_id = None

        def read_fn(path: Path):
            if path.suffix in ms_office_suffixes:
                convert_file_to_pdf(str(path), temp_dir)
                fn = os.path.join(temp_dir, f"{path.stem}.pdf")
            elif path.suffix in image_suffixes:
                with open(str(path), "rb") as f:
                    bits = f.read()
                pdf_bytes = fitz.open(stream=bits).convert_to_pdf()
                fn = os.path.join(temp_dir, f"{path.stem}.pdf")
                with open(fn, "wb") as f:
                    f.write(pdf_bytes)
            elif path.suffix in pdf_suffixes:
                fn = str(path)
            else:
                raise Exception(f"Unknown file suffix: {path.suffix}")

            disk_rw = FileBasedDataReader(os.path.dirname(fn))
            return disk_rw.read(os.path.basename(fn))

        def parse_doc(doc_path: Path, dataset=None):
            try:
                file_name = str(Path(doc_path).stem)
                if dataset is None:
                    pdf_data_or_dataset = read_fn(doc_path)
                else:
                    pdf_data_or_dataset = dataset
                do_parse(
                    output_dir,
                    file_name,
                    pdf_data_or_dataset,
                    [],
                    method,
                    debug_able,
                    start_page_id=start_page_id,
                    end_page_id=end_page_id,
                    lang=lang,
                )

            except Exception as e:
                logger.exception(e)

        try:
            base_name = os.path.basename(file_path)
            md_filename = os.path.splitext(base_name)[0] + ".md"
            md_path = os.path.join(output_dir, md_filename)

            # 检查文件路径是否存在
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"文件不存在: {file_path}")

            # 转换 PDF 为 Markdown
            if os.path.isdir(file_path):
                doc_paths = []
                for doc_path in Path(file_path).glob("*"):
                    if (
                        doc_path.suffix
                        in pdf_suffixes + image_suffixes + ms_office_suffixes
                    ):
                        if doc_path.suffix in ms_office_suffixes:
                            convert_file_to_pdf(str(doc_path), temp_dir)
                            doc_path = Path(
                                os.path.join(temp_dir, f"{doc_path.stem}.pdf")
                            )
                        elif doc_path.suffix in image_suffixes:
                            with open(str(doc_path), "rb") as f:
                                bits = f.read()
                                pdf_bytes = fitz.open(stream=bits).convert_to_pdf()
                            fn = os.path.join(temp_dir, f"{doc_path.stem}.pdf")
                            with open(fn, "wb") as f:
                                f.write(pdf_bytes)
                            doc_path = Path(fn)
                        doc_paths.append(doc_path)
                datasets = batch_build_dataset(doc_paths, 4, None)
                batch_do_parse(
                    output_dir,
                    [str(doc_path.stem) for doc_path in doc_paths],
                    datasets,
                    method,
                    debug_able,
                    lang=lang,
                )
            else:
                parse_doc(Path(file_path))

            generated_subdir = os.path.join(
                output_dir, os.path.splitext(base_name)[0], "auto"
            )
            generated_md = os.path.join(generated_subdir, md_filename)

            if os.path.exists(generated_md):
                # 移动目标文件到预期路径
                shutil.move(generated_md, md_path)
                logger.info(f"文件已移动至: {md_path}")

                # 递归删除生成目录及其父目录
                try:
                    # 先删除 auto 目录（允许非空）
                    shutil.rmtree(generated_subdir, ignore_errors=True)
                    # 再删除文件名目录（确保为空）
                    parent_dir = os.path.dirname(generated_subdir)
                    if len(os.listdir(parent_dir)) == 0:
                        os.rmdir(parent_dir)
                    else:
                        logger.warning(f"目录 {parent_dir} 非空，保留目录结构")
                except Exception as e:
                    logger.error(f"清理目录时出错: {str(e)}，但不影响主流程")

            else:
                raise ValueError("生成的Markdown文件路径不符合预期结构")

            # 验证写入结果
            if not os.path.exists(md_path):
                raise ValueError("生成的Markdown文件不存在")

            return md_path

        except Exception as e:
            logger.error(f"转换过程出错: {str(e)}")
            raise HTTPException(status_code=500, detail=f"PDF转换失败: {str(e)}")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    except Exception as e:
        logger.error(f"本地转换失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"本地转换失败: {str(e)}")


def _convert_via_cloud(file_path: str, output_dir: str, username: str) -> str:
    """使用云服务转换实现（保留接口但标记为未实现）"""
    logger.error("云服务转换模式当前未实现")
    raise HTTPException(status_code=501, detail="云服务转换模式当前未实现")


@app.get("/")
async def root():
    """根端点"""
    return {"message": "PDF to Markdown Converter API", "status": "running"}


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(
        "pdfserver:app", host="0.0.0.0", port=8005, reload=True, log_level="info"
    )
