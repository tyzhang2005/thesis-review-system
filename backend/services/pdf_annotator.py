"""
PDF 标注工具模块
完全按照 MinerU 的实现方式，在 PDF 层面添加 AIGC 检测结果的色块标注

参考: mineru/utils/draw_bbox.py
"""

import os
from io import BytesIO
from typing import Any, Dict, List

from pypdf import PageObject, PdfReader, PdfWriter
from reportlab.lib.colors import Color
from reportlab.pdfgen import canvas


def get_color_by_ai_probability(ai_probability: float) -> List[int]:
    """
    根据 AI 概率返回 RGB 颜色

    Args:
        ai_probability: AI 生成的概率 (0-1)

    Returns:
        RGB 颜色列表，范围 0-255
    """
    if ai_probability < 0.4:
        return [76, 175, 80]  # 绿色 - 40%以下
    elif ai_probability < 0.5:
        return [255, 193, 7]  # 黄色 - 40%-50%
    elif ai_probability < 0.6:
        return [200, 150, 255]  # 淡紫色 - 50%-60%
    elif ai_probability < 0.7:
        return [255, 152, 0]  # 橙色 - 60%-70%
    else:
        return [220, 38, 127]  # 红色 - 70%以上


def cal_canvas_rect(page, bbox):
    """
    完全复用 MinerU 的坐标转换函数

    Calculate the rectangle coordinates on the canvas based on the original PDF page and bounding box.

    注意：bbox 是归一化到 1000x1000 的坐标（来自 content_list.json），
    需要先反归一化到实际 PDF 尺寸

    Args:
        page: A PyPDF2 Page object representing a single page in the PDF.
        bbox: [x0, y0, x1, y1] representing the bounding box coordinates (归一化到 1000x1000).

    Returns:
        rect: [x0, y0, width, height] representing the rectangle coordinates on the canvas.
    """
    page_width, page_height = float(page.cropbox[2]), float(page.cropbox[3])

    # 反归一化：将 1000x1000 的坐标转换到实际 PDF 尺寸
    x0, y0, x1, y1 = bbox
    x0 = x0 * page_width / 1000
    y0 = y0 * page_height / 1000
    x1 = x1 * page_width / 1000
    y1 = y1 * page_height / 1000

    actual_width = page_width
    actual_height = page_height

    rotation_obj = page.get("/Rotate", 0)
    try:
        rotation = int(rotation_obj) % 360
    except (ValueError, TypeError):
        rotation = 0

    if rotation in [90, 270]:
        actual_width, actual_height = actual_height, actual_width

    rect_w = abs(x1 - x0)
    rect_h = abs(y1 - y0)

    if rotation == 270:
        rect_w, rect_h = rect_h, rect_w
        x0 = actual_height - y1
        y0 = actual_width - x1
    elif rotation == 180:
        x0 = page_width - x1
    elif rotation == 90:
        rect_w, rect_h = rect_h, rect_w
        x0, y0 = y0, x0
    else:  # rotation == 0
        y0 = page_height - y1

    rect = [x0, y0, rect_w, rect_h]
    return rect


def draw_aigc_annotations(page_idx, page, c, page_annotations):
    """
    在单页 Canvas 上绘制 AIGC 标注

    完全按照 MinerU 的 draw_bbox_without_number 方式实现，并添加块信息标签

    Args:
        page_idx: 页面索引
        page: PDF 页面对象
        c: reportlab Canvas 对象
        page_annotations: 该页的标注数据

    Returns:
        更新后的 Canvas 对象
    """
    if page_idx not in page_annotations:
        return c

    from reportlab.lib.colors import white

    for annotation in page_annotations[page_idx]:
        bbox = annotation["bbox"]
        color_rgb = annotation["color"]
        block_idx = annotation.get("block_idx", "")
        text_len = annotation.get("text_len", 0)
        ai_prob = annotation.get("ai_probability", 0)

        # 使用 MinerU 的坐标转换
        rect = cal_canvas_rect(page, bbox)

        # 使用 MinerU 相同的透明度: 0.3
        new_rgb = [float(color) / 255 for color in color_rgb]
        c.setFillColorRGB(new_rgb[0], new_rgb[1], new_rgb[2], 0.3)
        c.rect(rect[0], rect[1], rect[2], rect[3], stroke=0, fill=1)

        # 添加块信息标签
        # 标签文本格式: #{block_idx} Len:{text_len} AI:{ai_rate:.1%}
        label_text = f"#{block_idx} Len:{text_len} AI:{ai_prob:.1%}"

        # 设置字体
        c.setFont("Helvetica", 8)

        # 计算文本宽度
        text_width = c.stringWidth(label_text, "Helvetica", 8)

        # 标签框尺寸（添加一些padding）
        label_padding = 2
        label_width = text_width + 2 * label_padding
        label_height = 12

        # 计算标签位置：在色块的右上角
        # rect[0], rect[1] 是左下角坐标
        # rect[2], rect[3] 是宽高
        label_x = rect[0] + rect[2] - label_width  # 右对齐
        label_y = rect[1] + rect[3] - label_height  # 顶部

        # 确保标签不超出色块边界
        if label_x < rect[0]:
            label_x = rect[0]
        if label_y < rect[1]:
            label_y = rect[1] + 2  # 稍微向上偏移

        # 绘制白色背景框
        c.setFillColor(white, alpha=0.9)
        c.rect(label_x, label_y, label_width, label_height, stroke=0, fill=1)

        # 绘制文本
        c.setFillColorRGB(0, 0, 0, alpha=1)  # 黑色文本
        c.drawString(label_x + label_padding, label_y + 2, label_text)

    return c


def build_bbox_map(
    content_blocks: List[Dict[str, Any]],
) -> Dict[int, Dict[int, List[float]]]:
    """
    建立 block_idx 到 bbox 的映射

    Args:
        content_blocks: content_list.json 的 blocks 列表

    Returns:
        {page_idx: {block_idx: bbox}} 的嵌套字典
    """
    block_bbox_map = {}

    for block_idx, block in enumerate(content_blocks):
        page_idx = block["page_idx"]

        if page_idx not in block_bbox_map:
            block_bbox_map[page_idx] = {}

        # 使用全局索引（在过滤后的 content_blocks 中的位置）作为 block_idx
        block_bbox_map[page_idx][block_idx] = block["bbox"]

    return block_bbox_map


def organize_annotations_by_page(
    chunks: List[Dict[str, Any]],
    block_bbox_map: Dict[int, Dict[int, List[float]]],
    content_blocks: List[Dict[str, Any]],
) -> Dict[int, List[Dict[str, Any]]]:
    """
    按页面组织标注数据，包含块元数据（块号、长度、AI率）

    Args:
        chunks: AIGC 检测结果 chunks 列表
        block_bbox_map: block_idx 到 bbox 的映射
        content_blocks: content_list.json 的 blocks 列表（用于获取文本长度）

    Returns:
        {page_idx: [annotations]} 的字典
    """
    page_annotations = {}

    # 建立 block_idx 到文本长度的映射
    block_text_len = {}
    for idx, block in enumerate(content_blocks):
        block_text_len[idx] = len(block.get("text", ""))

    # 使用 chunks 索引作为块号
    for chunk_idx, chunk in enumerate(chunks):
        # 获取该 chunk 的颜色和 AI 概率
        ai_probability = chunk["ai_probability"]
        color = get_color_by_ai_probability(ai_probability)
        # 修复：chunks 数据结构使用 content_length 字段，不是 text
        text_len = chunk.get("content_length", 0)

        # 遍历该 chunk 在 PDF 中的所有位置
        for pos in chunk.get("pdf_positions", []):
            page_idx = pos["page_idx"]
            block_idx = pos["block_idx"]

            # 初始化该页的标注列表
            if page_idx not in page_annotations:
                page_annotations[page_idx] = []

            # 查找对应的 bbox
            if page_idx in block_bbox_map and block_idx in block_bbox_map[page_idx]:
                bbox = block_bbox_map[page_idx][block_idx]

                page_annotations[page_idx].append(
                    {
                        "bbox": bbox,
                        "color": color,
                        "block_idx": chunk_idx,  # 使用 chunk 索引作为块号
                        "text_len": text_len,  # 使用 chunk 文本长度
                        "ai_probability": ai_probability,
                    }
                )

    return page_annotations


def annotate_pdf_with_aigc_results(
    pdf_bytes: bytes,
    chunks: List[Dict[str, Any]],
    content_blocks: List[Dict[str, Any]],
    output_path: str,
) -> None:
    """
    完全按照 MinerU 的方式生成带标注的 PDF

    参考: mineru/utils/draw_bbox.py 中的 draw_layout_bbox 函数

    Args:
        pdf_bytes: 原始 PDF 字节
        chunks: AIGC 检测结果 chunks 列表
        content_blocks: content_list.json 的 blocks (type='text' 过滤后)
        output_path: 输出 PDF 路径
    """
    # 1. 建立 block_idx 到 bbox 的映射
    block_bbox_map = build_bbox_map(content_blocks)

    # 2. 按页面组织标注数据（带颜色和块元数据）
    page_annotations = organize_annotations_by_page(
        chunks, block_bbox_map, content_blocks
    )

    # 打印标注统计信息
    total_annotations = sum(len(anns) for anns in page_annotations.values())
    print(
        f"[PDF标注] 共 {len(page_annotations)} 页有标注，总计 {total_annotations} 个色块"
    )

    # 3. 按照 MinerU 的方式处理 PDF
    pdf_bytes_io = BytesIO(pdf_bytes)
    pdf_docs = PdfReader(pdf_bytes_io)
    output_pdf = PdfWriter()

    # 4. 逐页处理（完全按照 MinerU 的方式）
    for i, page in enumerate(pdf_docs.pages):
        # 获取原始页面尺寸
        page_width = float(page.cropbox[2])
        page_height = float(page.cropbox[3])
        custom_page_size = (page_width, page_height)

        # 创建覆盖层 Canvas（与 MinerU 完全相同）
        packet = BytesIO()
        c = canvas.Canvas(packet, pagesize=custom_page_size)

        # 绘制标注
        c = draw_aigc_annotations(i, page, c, page_annotations)

        c.save()
        packet.seek(0)
        overlay_pdf = PdfReader(packet)

        # 合并到原页面（与 MinerU 完全相同）
        if len(overlay_pdf.pages) > 0:
            new_page = PageObject(pdf=None)
            new_page.update(page)
            page = new_page
            page.merge_page(overlay_pdf.pages[0])
            print(
                f"[PDF标注] 第 {i + 1} 页: 添加了 {len(page_annotations.get(i, []))} 个色块"
            )

        output_pdf.add_page(page)

    # 5. 保存结果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        output_pdf.write(f)

    print(f"[PDF标注] 标注 PDF 已保存到: {output_path}")


def load_and_annotate_pdf(
    pdf_path: str,
    chunks: List[Dict[str, Any]],
    content_blocks: List[Dict[str, Any]],
    output_path: str,
) -> None:
    """
    从文件加载 PDF 并添加标注

    Args:
        pdf_path: 原始 PDF 文件路径
        chunks: AIGC 检测结果 chunks 列表
        content_blocks: content_list.json 的 blocks
        output_path: 输出 PDF 路径
    """
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    annotate_pdf_with_aigc_results(pdf_bytes, chunks, content_blocks, output_path)
