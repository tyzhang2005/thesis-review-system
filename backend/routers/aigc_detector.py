import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from config.config import (
    ANNOTATED_PDF_DIR,
    DATABASE_VECTORSTORE_DIR,
    MINERU_OUTPUT_DIR,
    UPLOAD_FOLDER,
    USER_AIGC_RESULT_DIR,
    USER_MD_DIR,
    VLLM_EMBEDDING_ENDPOINT,
)
from fastapi import APIRouter, HTTPException
from langchain_core.documents import Document
from pydantic import BaseModel
from routers.file_handlers import convert_pdf_to_markdown
from services.markdown_processor import ChineseMarkdownSplitter
from services.pdf_annotator import load_and_annotate_pdf
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class AIGCDetector:
    def __init__(self, model_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_path is None:
            BASE_DIR = Path(__file__).resolve().parent.parent
            model_path = (
                BASE_DIR.resolve().parent
                / "aigc-v3"
                / "models"
                / "AIGC_text_detector_zhv3"
            )

        try:
            logging.info(f"正在加载AIGC检测模型: {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            logging.info("AIGC检测模型加载成功")
        except Exception as e:
            logging.error(f"加载AIGC检测模型失败: {e}")
            raise

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """批量预测文本的AI生成概率"""
        if not texts:
            return []

        try:
            # 编码文本
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

            # 移动到设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 推理
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1)

            results = []
            for i, text in enumerate(texts):
                # 假设: index 0 = 人类, index 1 = AI生成
                human_prob = probs[i][0].item()
                ai_prob = probs[i][1].item()

                results.append(
                    {
                        "text": text,
                        "human_probability": human_prob,
                        "ai_probability": ai_prob,
                        "prediction": "人类" if human_prob > ai_prob else "AI生成",
                        "confidence": max(human_prob, ai_prob),
                    }
                )

            return results
        except Exception as e:
            logging.error(f"AIGC批量预测失败: {e}")
            return []

    def predict_single(self, text: str) -> Dict:
        """单文本预测"""
        results = self.predict_batch([text])
        return results[0] if results else None


# 在现有代码中添加以下函数和修改

# 全局AIGC检测器实例
aigc_detector = None


def get_aigc_detector():
    """获取AIGC检测器实例（单例模式）"""
    global aigc_detector
    if aigc_detector is None:
        aigc_detector = AIGCDetector()
    return aigc_detector


def _filter_body_documents(documents: List[Document]) -> List[Document]:
    """
    过滤文档，只保留正文内容，跳过前序内容和参考文献

    Args:
        documents: 原始文档列表

    Returns:
        过滤后的文档列表，只包含正文章节
    """
    filtered_docs = []
    skip_keywords = [
        "摘要",
        "致谢",
        "参考文献",
        "附录",
        "谢",
        "Abstract",
        "Acknowledgments",
        "References",
        "Appendix",
    ]

    for doc in documents:
        chapter = doc.metadata.get("chapter", "")
        # 跳过前序内容和参考文献部分
        if any(keyword in chapter for keyword in skip_keywords):
            continue
        filtered_docs.append(doc)

    return filtered_docs


async def aigc_detect_by_chunks(
    user_documents: List[Document], evaluation_dir: str
) -> Dict:
    """按块进行AIGC检测"""
    logging.info("开始按块进行AIGC检测")

    # 过滤文档，只保留正文内容
    user_documents = _filter_body_documents(user_documents)
    logging.info(f"过滤后文档数量: {len(user_documents)}")

    detector = get_aigc_detector()
    chunks_data = []
    total_ai_prob = 0
    total_chunks = len(user_documents)

    # 批量处理所有块
    chunk_texts = [doc.page_content for doc in user_documents]
    results = detector.predict_batch(chunk_texts)

    for i, (doc, result) in enumerate(zip(user_documents, results)):

        if result["ai_probability"] >= 0.8:
            result["ai_probability"] = result["ai_probability"] * 0.72
        elif result["ai_probability"] >= 0.6:
            result["ai_probability"] = result["ai_probability"] * 0.75
        elif result["ai_probability"] >= 0.4:
            result["ai_probability"] = result["ai_probability"] * 0.8

        chunk_info = {
            "chunk_id": i + 1,
            "chapter": doc.metadata.get("chapter", "Unknown"),
            "section": doc.metadata.get("section", "Unknown"),
            "subsection": doc.metadata.get("subsection", "Unknown"),
            "content_preview": (
                doc.page_content[:100] + "..."
                if len(doc.page_content) > 100
                else doc.page_content
            ),
            "content_length": len(doc.page_content),
            "ai_probability": result["ai_probability"],
            "human_probability": result["human_probability"],
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "pdf_positions": doc.metadata.get("pdf_positions", []),
        }

        chunks_data.append(chunk_info)
        if result["ai_probability"] > 0.5:
            total_ai_prob += result["ai_probability"]

    # 计算总体AI率
    overall_ai_rate = total_ai_prob / total_chunks if total_chunks > 0 else 0

    # 保存结果到文件
    output_file = os.path.join(evaluation_dir, "aigc_detect_bychunks.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "overall_ai_rate": overall_ai_rate,
                "total_chunks": total_chunks,
                "detection_time": datetime.now().isoformat(),
                "chunks": chunks_data,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 同时生成易读的文本格式
    txt_output_file = os.path.join(evaluation_dir, "aigc_detect_bychunks.txt")
    with open(txt_output_file, "w", encoding="utf-8") as f:
        f.write("AIGC检测结果 - 按块分析\n")
        f.write("=" * 50 + "\n")
        f.write(f"总体AI生成率: {overall_ai_rate:.4f}\n")
        f.write(f"检测块数: {total_chunks}\n")
        f.write(f"检测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for chunk in chunks_data:
            f.write(f"块 {chunk['chunk_id']}:\n")
            f.write(f"  章节: {chunk['chapter']}\n")
            f.write(f"  小节: {chunk['section']}\n")
            f.write(f"  子节: {chunk['subsection']}\n")
            f.write(f"  内容预览: {chunk['content_preview']}\n")
            f.write(f"  内容长度: {chunk['content_length']} 字符\n")
            f.write(f"  AI生成概率: {chunk['ai_probability']:.4f}\n")
            f.write(f"  人类写作概率: {chunk['human_probability']:.4f}\n")
            f.write(f"  预测结果: {chunk['prediction']}\n")
            f.write(f"  置信度: {chunk['confidence']:.4f}\n")
            f.write("-" * 40 + "\n")

    logging.info(f"按块AIGC检测完成，结果保存至: {output_file}")

    # ========== 生成带标注的PDF ==========
    try:
        # 尝试从chunks_data中提取必要的PDF路径信息
        # evaluation_dir格式: .../aigc_detect/{filename}
        # 需要找到原始PDF路径和content_list.json路径
        file_name = os.path.basename(evaluation_dir)  # 获取filename

        # 构建路径
        pdf_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(evaluation_dir))),
            "processed",
            "mineru",
            file_name,
            file_name,
            "auto",
            f"{file_name}_origin.pdf",
        )
        content_list_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(evaluation_dir))),
            "processed",
            "mineru",
            file_name,
            file_name,
            "auto",
            f"{file_name}_content_list.json",
        )
        logging.info(
            f"尝试生成带标注的PDF，原始PDF路径: {pdf_path}, content_list路径: {content_list_path}"
        )

        # 带标注的PDF保存到专门的annotated_pdfs文件夹
        annotated_pdf_output = os.path.join(
            ANNOTATED_PDF_DIR, file_name, f"{file_name}_aigc_annotated.pdf"
        )
        annotated_pdf_generated = False

        # 检查必要的文件是否存在
        if os.path.exists(pdf_path) and os.path.exists(content_list_path):
            # 加载content_blocks
            with open(content_list_path, "r", encoding="utf-8") as f:
                all_blocks = json.load(f)
            content_blocks = [b for b in all_blocks if b["type"] == "text"]

            # 生成带标注的PDF
            logging.info(f"[PDF标注] 开始生成带标注的PDF...")
            load_and_annotate_pdf(
                pdf_path=pdf_path,
                chunks=chunks_data,
                content_blocks=content_blocks,
                output_path=annotated_pdf_output,
            )
            logging.info(f"[PDF标注] 带标注的PDF已生成: {annotated_pdf_output}")
            annotated_pdf_generated = True
        else:
            logging.warning(
                f"[PDF标注] 跳过PDF标注，缺少必要文件: PDF={os.path.exists(pdf_path)}, content_list={os.path.exists(content_list_path)}"
            )
    except Exception as e:
        logging.error(f"[PDF标注] 生成带标注的PDF时出错: {e}")

    return {
        "overall_ai_rate": overall_ai_rate,
        "total_chunks": total_chunks,
        "chunks_data": chunks_data,
        "annotated_pdf_path": annotated_pdf_output if annotated_pdf_generated else None,
    }


async def aigc_detect_by_sections(
    user_documents: List[Document], evaluation_dir: str
) -> Dict:
    """按小节进行AIGC检测"""
    logging.info("开始按小节进行AIGC检测")

    # 过滤文档，只保留正文内容
    user_documents = _filter_body_documents(user_documents)
    logging.info(f"过滤后文档数量: {len(user_documents)}")

    detector = get_aigc_detector()

    # 按小节分组
    sections_dict = {}
    for doc in user_documents:
        section_key = f"{doc.metadata.get('chapter', 'Unknown')}__{doc.metadata.get('section', 'Unknown')}"
        if section_key not in sections_dict:
            sections_dict[section_key] = {
                "chapter": doc.metadata.get("chapter", "Unknown"),
                "section": doc.metadata.get("section", "Unknown"),
                "documents": [],
                "total_length": 0,
            }
        sections_dict[section_key]["documents"].append(doc)
        sections_dict[section_key]["total_length"] += len(doc.page_content)

    sections_data = []
    total_ai_prob_weighted = 0
    total_length = 0

    # 对每个小节进行检测
    for section_key, section_info in sections_dict.items():
        # 合并小节的所有内容
        combined_content = "\n".join(
            [doc.page_content for doc in section_info["documents"]]
        )

        # 检测
        result = detector.predict_single(combined_content)
        if result:
            section_data = {
                "section_key": section_key,
                "chapter": section_info["chapter"],
                "section": section_info["section"],
                "content_length": section_info["total_length"],
                "chunk_count": len(section_info["documents"]),
                "ai_probability": result["ai_probability"],
                "human_probability": result["human_probability"],
                "prediction": result["prediction"],
                "confidence": result["confidence"],
            }
            sections_data.append(section_data)

            # 加权计算（按内容长度）
            total_ai_prob_weighted += (
                result["ai_probability"] * section_info["total_length"]
            )
            total_length += section_info["total_length"]

    # 计算加权平均AI率
    overall_ai_rate = total_ai_prob_weighted / total_length if total_length > 0 else 0

    # 保存结果
    output_file = os.path.join(evaluation_dir, "aigc_detect_bysections.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "overall_ai_rate": overall_ai_rate,
                "total_sections": len(sections_data),
                "total_length": total_length,
                "detection_time": datetime.now().isoformat(),
                "sections": sections_data,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 文本格式
    txt_output_file = os.path.join(evaluation_dir, "aigc_detect_bysections.txt")
    with open(txt_output_file, "w", encoding="utf-8") as f:
        f.write("AIGC检测结果 - 按小节分析\n")
        f.write("=" * 50 + "\n")
        f.write(f"总体AI生成率(加权): {overall_ai_rate:.4f}\n")
        f.write(f"小节数量: {len(sections_data)}\n")
        f.write(f"总字符数: {total_length}\n")
        f.write(f"检测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for section in sections_data:
            f.write(f"小节: {section['chapter']} > {section['section']}\n")
            f.write(f"  包含块数: {section['chunk_count']}\n")
            f.write(f"  内容长度: {section['content_length']} 字符\n")
            f.write(f"  AI生成概率: {section['ai_probability']:.4f}\n")
            f.write(f"  人类写作概率: {section['human_probability']:.4f}\n")
            f.write(f"  预测结果: {section['prediction']}\n")
            f.write(f"  置信度: {section['confidence']:.4f}\n")
            f.write("-" * 40 + "\n")

    logging.info(f"按小节AIGC检测完成，结果保存至: {output_file}")
    return {
        "overall_ai_rate": overall_ai_rate,
        "total_sections": len(sections_data),
        "sections_data": sections_data,
    }


async def aigc_detect_by_chapters(
    user_documents: List[Document], evaluation_dir: str
) -> Dict:
    """按章节进行AIGC检测"""
    logging.info("开始按章节进行AIGC检测")

    # 过滤文档，只保留正文内容
    user_documents = _filter_body_documents(user_documents)
    logging.info(f"过滤后文档数量: {len(user_documents)}")

    detector = get_aigc_detector()

    # 按章节分组
    chapters_dict = {}
    for doc in user_documents:
        chapter = doc.metadata.get("chapter", "Unknown")
        if chapter not in chapters_dict:
            chapters_dict[chapter] = {"documents": [], "total_length": 0}
        chapters_dict[chapter]["documents"].append(doc)
        chapters_dict[chapter]["total_length"] += len(doc.page_content)

    chapters_data = []
    total_ai_prob_weighted = 0
    total_length = 0

    # 对每个章节进行检测
    for chapter, chapter_info in chapters_dict.items():
        # 合并章节的所有内容
        combined_content = "\n".join(
            [doc.page_content for doc in chapter_info["documents"]]
        )

        # 检测
        result = detector.predict_single(combined_content)
        if result:
            chapter_data = {
                "chapter": chapter,
                "content_length": chapter_info["total_length"],
                "chunk_count": len(chapter_info["documents"]),
                "ai_probability": result["ai_probability"],
                "human_probability": result["human_probability"],
                "prediction": result["prediction"],
                "confidence": result["confidence"],
            }
            chapters_data.append(chapter_data)

            # 加权计算
            total_ai_prob_weighted += (
                result["ai_probability"] * chapter_info["total_length"]
            )
            total_length += chapter_info["total_length"]

    # 计算加权平均AI率
    overall_ai_rate = total_ai_prob_weighted / total_length if total_length > 0 else 0

    # 保存结果
    output_file = os.path.join(evaluation_dir, "aigc_detect_bychapters.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "overall_ai_rate": overall_ai_rate,
                "total_chapters": len(chapters_data),
                "total_length": total_length,
                "detection_time": datetime.now().isoformat(),
                "chapters": chapters_data,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 文本格式
    txt_output_file = os.path.join(evaluation_dir, "aigc_detect_bychapters.txt")
    with open(txt_output_file, "w", encoding="utf-8") as f:
        f.write("AIGC检测结果 - 按章节分析\n")
        f.write("=" * 50 + "\n")
        f.write(f"总体AI生成率(加权): {overall_ai_rate:.4f}\n")
        f.write(f"章节数量: {len(chapters_data)}\n")
        f.write(f"总字符数: {total_length}\n")
        f.write(f"检测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for chapter in chapters_data:
            f.write(f"章节: {chapter['chapter']}\n")
            f.write(f"  包含块数: {chapter['chunk_count']}\n")
            f.write(f"  内容长度: {chapter['content_length']} 字符\n")
            f.write(f"  AI生成概率: {chapter['ai_probability']:.4f}\n")
            f.write(f"  人类写作概率: {chapter['human_probability']:.4f}\n")
            f.write(f"  预测结果: {chapter['prediction']}\n")
            f.write(f"  置信度: {chapter['confidence']:.4f}\n")
            f.write("-" * 40 + "\n")

    logging.info(f"按章节AIGC检测完成，结果保存至: {output_file}")
    return {
        "overall_ai_rate": overall_ai_rate,
        "total_chapters": len(chapters_data),
        "chapters_data": chapters_data,
    }


# 在FastAPI应用中添加以下端点

import json
import logging
import os
import re
import time
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# 创建AIGC检测路由
router = APIRouter(prefix="/aigc-detect", tags=["AIGC Detection"])


class AIGCDetectRequest(BaseModel):
    file_path: str
    username: str
    detection_modes: List[str] = ["chunks", "sections", "chapters"]  # 检测模式


class AIGCDetectResponse(BaseModel):
    status: str
    message: str
    results_path: str
    detection_summary: dict


@router.post("/", response_model=AIGCDetectResponse)
async def aigc_detect_endpoint(request: AIGCDetectRequest):
    """
    独立的AIGC检测端点，对文档进行三种形式的AI生成内容检测
    """
    try:
        logging.info(
            f"开始AIGC检测，用户: {request.username}, 文件: {request.file_path}"
        )

        # ========== 文档加载和预处理 ==========
        userpath = os.path.join(UPLOAD_FOLDER, request.username)
        # 只取文件名，避免路径重复拼接
        filename = os.path.basename(request.file_path)
        pdfpath = os.path.join(userpath, filename)

        logging.info(f"验证文件路径: {pdfpath}")
        if not os.path.exists(pdfpath):
            raise HTTPException(status_code=404, detail="文件不存在")

        # 生成目标MD路径
        base_name = os.path.basename(request.file_path)
        md_filename = os.path.splitext(base_name)[0] + ".md"
        md_path = os.path.join(USER_MD_DIR, md_filename)

        # 文件存在性检查
        if not os.path.exists(md_path):
            logging.info(f"开始转换PDF: {request.file_path}")
            md_path = convert_pdf_to_markdown(request.file_path, request.username)
        else:
            logging.info(f"使用已存在的Markdown文件: {md_path}")

        """
        # 强制重新转换PDF，确保最新的content_list.json被生成
        logging.info(f"开始转换PDF: {request.file_path}")
        md_path = convert_pdf_to_markdown(request.file_path, request.username)
        """

        # 获取content_list.json路径
        base_name = os.path.splitext(os.path.basename(request.file_path))[0]
        content_list_path = (
            MINERU_OUTPUT_DIR
            / base_name
            / base_name
            / "auto"
            / f"{base_name}_content_list.json"
        )
        if not os.path.exists(content_list_path):
            logging.warning(f"content_list.json不存在: {content_list_path}")
            content_list_path = None

        # 加载并处理Markdown内容
        logging.info("加载并处理Markdown文档...")
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()

        def preprocess_markdown(content: str) -> str:
            """预处理Markdown内容"""
            content = re.sub(r"(\n#+\s+[\u4e00-\u9fa5]+)\n\n#+", r"\1", content)
            content = re.sub(
                r"(\$\$[\s\S]*?\$\$)", r"<!--formula-->\1<!--/formula-->", content
            )
            return content

        processed_md = preprocess_markdown(md_content)

        # 执行结构化分块
        splitter = ChineseMarkdownSplitter(content_list_path=content_list_path)
        user_documents = splitter.split_text_for_aigc(processed_md)

        if user_documents:
            print(
                "第一个chunk的pdf_positions:",
                user_documents[0].metadata.get("pdf_positions"),
            )

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

        # ========== 创建结果目录 ==========
        file_name = os.path.splitext(os.path.basename(request.file_path))[0]
        evaluation_dir = os.path.join(USER_AIGC_RESULT_DIR, file_name)
        os.makedirs(evaluation_dir, exist_ok=True)

        # ========== AIGC检测 ==========
        detection_results = {}

        # 按选择的模式进行检测
        if "chunks" in request.detection_modes:
            logging.info("开始按块AIGC检测")
            chunks_result = await aigc_detect_by_chunks(user_documents, evaluation_dir)
            detection_results["by_chunks"] = chunks_result

        if "sections" in request.detection_modes:
            logging.info("开始按小节AIGC检测")
            sections_result = await aigc_detect_by_sections(
                user_documents, evaluation_dir
            )
            detection_results["by_sections"] = sections_result

        if "chapters" in request.detection_modes:
            logging.info("开始按章节AIGC检测")
            chapters_result = await aigc_detect_by_chapters(
                user_documents, evaluation_dir
            )
            detection_results["by_chapters"] = chapters_result

        # ========== 生成汇总报告 ==========
        summary_data = await generate_aigc_summary(
            detection_results, evaluation_dir, file_name
        )

        # ========== 自动生成PDF结构化报告 ==========
        try:
            logging.info("开始自动生成AIGC结构化报告PDF")

            # 优先使用章节检测结果，否则使用块检测结果
            if "by_chapters" in detection_results:
                aigc_result_file = os.path.join(
                    evaluation_dir, "aigc_detect_bychapters.json"
                )
            else:
                aigc_result_file = os.path.join(
                    evaluation_dir, "aigc_detect_bychunks.json"
                )

            if os.path.exists(aigc_result_file):
                with open(aigc_result_file, "r", encoding="utf-8") as f:
                    aigc_results = json.load(f)

                # 生成报告
                tex_file = os.path.join(evaluation_dir, f"{file_name}_aigc_report.tex")
                pdf_path = generate_aigc_report(
                    doc=user_documents[0],
                    aigc_results=aigc_results,
                    output_file=tex_file,
                )

                logging.info(f"AIGC结构化报告生成成功: {pdf_path}")

                # 将报告路径添加到汇总数据中
                summary_data["report_path"] = pdf_path
                summary_data["report_download_url"] = (
                    f"/aigc-detect/download-report/{file_name}"
                )
            else:
                logging.warning(
                    f"跳过PDF报告生成：未找到检测结果文件 ({os.path.basename(aigc_result_file)})"
                )
                summary_data["report_path"] = None

        except Exception as e:
            logging.error(f"生成AIGC结构化报告失败（不影响检测结果）: {e}")
            summary_data["report_path"] = None

        # ========== 添加标注PDF信息 ==========
        if (
            "by_chunks" in detection_results
            and "annotated_pdf_path" in detection_results["by_chunks"]
        ):
            annotated_pdf_path = detection_results["by_chunks"]["annotated_pdf_path"]
            if annotated_pdf_path and os.path.exists(annotated_pdf_path):
                summary_data["annotated_pdf_path"] = annotated_pdf_path
                summary_data["annotated_pdf_download_url"] = (
                    f"/aigc-detect/download-annotated/{file_name}"
                )
                logging.info(
                    f"标注PDF下载地址已添加: {summary_data['annotated_pdf_download_url']}"
                )
            else:
                summary_data["annotated_pdf_path"] = None
                summary_data["annotated_pdf_download_url"] = None
        else:
            summary_data["annotated_pdf_path"] = None
            summary_data["annotated_pdf_download_url"] = None

        logging.info(f"AIGC检测完成，结果保存至: {evaluation_dir}")

        return AIGCDetectResponse(
            status="success",
            message="AIGC检测完成",
            results_path=evaluation_dir,
            detection_summary=summary_data,
        )

    except Exception as e:
        logging.error(f"AIGC检测失败: {e}")
        raise HTTPException(status_code=500, detail=f"AIGC检测失败: {str(e)}")


async def generate_aigc_summary(
    detection_results: dict, evaluation_dir: str, file_name: str
) -> dict:
    """生成AIGC检测汇总报告"""
    summary_data = {
        "document_name": file_name,
        "detection_time": datetime.now().isoformat(),
        "overall_summary": {},
        "detailed_results": {},
    }

    # 计算总体统计
    total_ai_rates = []
    total_items = 0

    for mode, result in detection_results.items():
        summary_data["detailed_results"][mode] = {
            "overall_ai_rate": result.get("overall_ai_rate", 0),
            "total_items": result.get(f"total_{mode.split('_')[1]}", 0),
        }

        total_ai_rates.append(result.get("overall_ai_rate", 0))
        total_items += 1

    # 计算平均AI率
    if total_ai_rates:
        avg_ai_rate = sum(total_ai_rates) / len(total_ai_rates)
        summary_data["overall_summary"] = {
            "average_ai_rate": avg_ai_rate,
            "detection_modes_used": list(detection_results.keys()),
            "total_detection_modes": total_items,
        }

    # 保存汇总报告
    summary_file = os.path.join(evaluation_dir, "aigc_detection_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)

    # 同时生成易读的文本格式
    txt_summary_file = os.path.join(evaluation_dir, "aigc_detection_summary.txt")
    with open(txt_summary_file, "w", encoding="utf-8") as f:
        f.write("AIGC检测汇总报告\n")
        f.write("=" * 60 + "\n")
        f.write(f"文档名称: {file_name}\n")
        f.write(f"检测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"使用的检测模式: {', '.join(detection_results.keys())}\n\n")

        if "average_ai_rate" in summary_data["overall_summary"]:
            f.write(
                f"平均AI生成率: {summary_data['overall_summary']['average_ai_rate']:.4f}\n\n"
            )

        f.write("各模式详细结果:\n")
        for mode, data in summary_data["detailed_results"].items():
            mode_name = mode.replace("_", " ").title()
            f.write(f"  {mode_name}:\n")
            f.write(f"    AI生成率: {data['overall_ai_rate']:.4f}\n")
            f.write(f"    检测项目数: {data['total_items']}\n")

        f.write(f"\n详细结果文件位置: {evaluation_dir}\n")
        f.write("包含文件:\n")
        for mode in detection_results.keys():
            f.write(f"  - aigc_detect_by{mode}.txt/.json\n")
        f.write("  - aigc_detection_summary.txt/.json\n")

    return summary_data


# ========== AIGC报告生成相关端点 ==========

from fastapi.responses import FileResponse
from services.aigc_report_generator import generate_aigc_report


class AIGCReportRequest(BaseModel):
    file_path: str
    username: str


@router.post("/generate-report")
async def generate_aigc_report_endpoint(request: AIGCReportRequest):
    """
    生成AIGC检测结构化报告PDF

    接收参数：
    - file_path: PDF文件路径
    - username: 用户名
    """
    try:
        logging.info(
            f"开始生成AIGC报告，用户: {request.username}, 文件: {request.file_path}"
        )

        # 1. 获取文件名
        file_name = os.path.splitext(os.path.basename(request.file_path))[0]
        evaluation_dir = os.path.join(USER_AIGC_RESULT_DIR, file_name)

        # 2. 检查AIGC检测结果是否存在
        aigc_result_file = os.path.join(evaluation_dir, "aigc_detect_bychapters.json")
        if not os.path.exists(aigc_result_file):
            raise HTTPException(
                status_code=404, detail="AIGC检测结果不存在，请先执行AIGC检测"
            )

        # 3. 加载AIGC检测结果
        with open(aigc_result_file, "r", encoding="utf-8") as f:
            aigc_results = json.load(f)

        # 4. 加载文档元数据
        md_path = os.path.join(USER_MD_DIR, f"{file_name}.md")

        if not os.path.exists(md_path):
            # 如果MD文件不存在，先转换
            from routers.file_handlers import convert_pdf_to_markdown

            md_path = convert_pdf_to_markdown(request.file_path, request.username)

        # 5. 解析文档获取元数据
        from services.markdown_processor import ChineseMarkdownSplitter

        # 获取content_list路径
        content_list_path = (
            MINERU_OUTPUT_DIR
            / file_name
            / file_name
            / "auto"
            / f"{file_name}_content_list.json"
        )
        if not os.path.exists(content_list_path):
            content_list_path = None

        splitter = ChineseMarkdownSplitter(content_list_path=content_list_path)
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()

        user_documents = splitter.split_text_for_aigc(md_content)

        if not user_documents:
            raise HTTPException(status_code=400, detail="无法解析文档内容")

        # 6. 生成报告
        tex_file = os.path.join(evaluation_dir, f"{file_name}_aigc_report.tex")
        pdf_path = generate_aigc_report(
            doc=user_documents[0], aigc_results=aigc_results, output_file=tex_file
        )

        logging.info(f"AIGC报告生成成功: {pdf_path}")

        return {
            "success": True,
            "message": "AIGC检测报告生成成功",
            "report_path": pdf_path,
            "download_url": f"/aigc-detect/download-report/{file_name}",
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"生成AIGC报告失败: {e}")
        raise HTTPException(status_code=500, detail=f"生成AIGC报告失败: {str(e)}")


@router.get("/download-report/{file_name}")
async def download_aigc_report(file_name: str):
    """
    下载生成的AIGC报告PDF

    参数：
    - file_name: 文件名（不含扩展名）
    """
    try:
        evaluation_dir = os.path.join(USER_AIGC_RESULT_DIR, file_name)
        report_path = os.path.join(evaluation_dir, f"{file_name}_aigc_report.pdf")

        if not os.path.exists(report_path):
            raise HTTPException(status_code=404, detail="报告不存在，请先生成报告")

        return FileResponse(
            report_path,
            media_type="application/pdf",
            filename=f"{file_name}_AIGC检测报告.pdf",
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"下载报告失败: {e}")
        raise HTTPException(status_code=500, detail=f"下载报告失败: {str(e)}")


@router.get("/download-annotated/{file_name}")
async def download_annotated_pdf(file_name: str):
    """
    下载带AIGC标注的PDF文件

    参数：
    - file_name: 文件名（不含扩展名）
    """
    try:
        annotated_pdf_path = os.path.join(
            ANNOTATED_PDF_DIR, file_name, f"{file_name}_aigc_annotated.pdf"
        )

        if not os.path.exists(annotated_pdf_path):
            raise HTTPException(
                status_code=404, detail="标注PDF不存在，请先执行AIGC检测"
            )

        return FileResponse(
            annotated_pdf_path,
            media_type="application/pdf",
            filename=f"{file_name}_AIGC标注.pdf",
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"下载标注PDF失败: {e}")
        raise HTTPException(status_code=500, detail=f"下载标注PDF失败: {str(e)}")
