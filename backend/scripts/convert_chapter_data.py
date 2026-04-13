#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 chapter_structure.txt 和 chapter_information.txt 转换为 JSON 格式
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

# 配置
USER_RESULT_DIR = Path("/DATA/zhangtianyue_231300023/eva/backend/user_result")


def parse_chapter_structure(file_path: Path) -> Dict[str, Any]:
    """
    解析 chapter_structure.txt 文件

    返回格式:
    {
        "paper_info": {...},
        "word_count_breakdown": {...},
        "structure_tree": [...]
    }
    """
    if not file_path.exists():
        return None

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    result = {"paper_info": {}, "word_count_breakdown": {}, "structure_tree": []}

    # 解析基本信息
    # Title: xxx
    title_match = re.search(r"Title:\s*(.+)", content)
    if title_match:
        result["paper_info"]["title"] = title_match.group(1).strip()

    # Student Name: xxx
    name_match = re.search(r"Student Name:\s*(.+)", content)
    if name_match:
        result["paper_info"]["student_name"] = name_match.group(1).strip()

    # Student ID: xxx
    id_match = re.search(r"Student ID:\s*(.+)", content)
    if id_match:
        result["paper_info"]["student_id"] = id_match.group(1).strip()

    # Abstract: xxx (多行)
    abstract_match = re.search(
        r"Abstract:\s*(.+?)(?=\nKeywords:|$)", content, re.DOTALL
    )
    if abstract_match:
        result["paper_info"]["abstract"] = abstract_match.group(1).strip()

    # Keywords: xxx
    keywords_match = re.search(r"Keywords:\s*(.+)", content)
    if keywords_match:
        keywords_str = keywords_match.group(1).strip()
        # 支持多种分隔符：分号、顿号、逗号
        keywords = re.split(r"[;、,，]", keywords_str)
        result["paper_info"]["keywords"] = [k.strip() for k in keywords if k.strip()]

    # Formula Words: xxx
    formula_words_match = re.search(r"Formula Words:\s*(\d+)", content)
    if formula_words_match:
        result["paper_info"]["formula_words"] = int(formula_words_match.group(1))

    # Formula Ratio: xxx%
    formula_ratio_match = re.search(r"Formula Ratio:\s*([\d.]+)%", content)
    if formula_ratio_match:
        result["paper_info"]["formula_ratio"] = float(formula_ratio_match.group(1))

    # 解析字数统计
    # 总字数
    total_words_match = re.search(r"论文总字数:\s*(\d+)", content)
    if total_words_match:
        result["paper_info"]["total_words"] = int(total_words_match.group(1))

    # 各部分字数统计
    section_match = re.search(
        r"各部分字数统计:\s*(.+?)(?=\nStructure:|$)", content, re.DOTALL
    )
    if section_match:
        section_text = section_match.group(1)
        # 按行匹配，支持包含英文、数字、空格等任意字符的章节名
        lines = section_text.strip().split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # 匹配 "XXX: NNN字" 格式，章节名可以是任意字符
            match = re.match(r"^(.+?):\s*(\d+)\s*字$", line)
            if match:
                name = match.group(1).strip()
                words = int(match.group(2))
                result["word_count_breakdown"][name] = words

    # 解析章节结构树
    structure_match = re.search(r"Structure:\s*(.+)", content, re.DOTALL)
    if structure_match:
        structure_text = structure_match.group(1).strip()
        result["structure_tree"] = parse_structure_tree(structure_text)

    return result


def parse_structure_tree(text: str) -> List[Dict[str, Any]]:
    """
    解析章节层级结构

    输入格式:
    第一章 绪论
        1.1 研究背景
            1.1.1 子小节
        1.2 相关工作

    返回格式:
    [
        {
            "key": "1",
            "title": "第一章 绪论",
            "level": 1,
            "children": [...]
        }
    ]
    """
    lines = text.strip().split("\n")
    root_nodes = []
    stack = []  # (node, level)

    for line in lines:
        if not line.strip():
            continue

        # 计算缩进级别
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        level = indent // 4 + 1  # 假设每级缩进4个空格

        # 提取章节号和标题
        # 匹配格式：第一章 xxx, 1.1 xxx, 1.1.1 xxx
        # 修复后的正则表达式
        match = re.match(r"(第[\u4e00-\u9fa5\d]+章|[\d\.]+)\s*(.*)", stripped)
        if match:
            key_num = match.group(1).strip()
            title_content = match.group(2).strip() if match.group(2) else ""

            # 如果没有单独的标题内容，使用完整的stripped作为标题
            if not title_content:
                title = stripped.strip()
            else:
                # 对于"第一章 绪论"这样的格式，保留完整的"第一章 绪论"
                title = stripped.strip()

            node = {"key": key_num, "title": title, "level": level, "children": []}

            # 找到合适的父节点
            while stack and stack[-1][1] >= level:
                stack.pop()

            if stack:
                stack[-1][0]["children"].append(node)
            else:
                root_nodes.append(node)

            stack.append((node, level))

    return root_nodes


def parse_chapter_information(file_path: Path) -> Dict[str, Any]:
    """
    解析 chapter_information.txt 文件

    返回格式:
    {
        "chapters": [
            {
                "chapter_number": 1,
                "chapter_name": "第一章 绪论",
                "chapter_type": "introduction",
                "summary": "...",
                "evaluation": {...},
                "structure_analysis": [...],
                "extracted_info": {...},
                "evaluation_items": [...],
                "score_impact": "...",
                "suggestions": [...]
            }
        ]
    }
    """
    if not file_path.exists():
        return None

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 按章节分割（使用 ====== 分隔符）
    chapter_blocks = re.split(r"=+\s*📖", content)

    result = {"chapters": []}

    for block in chapter_blocks:
        if not block.strip():
            continue

        chapter = parse_single_chapter(block)
        if chapter:
            result["chapters"].append(chapter)

    return result


def parse_single_chapter(block: str) -> Dict[str, Any]:
    """
    解析单个章节块
    """
    chapter = {
        "chapter_number": 0,
        "chapter_name": "",
        "chapter_type": "",
        "summary": "",
        "evaluation": {"strengths": "", "weaknesses": ""},
        "structure_analysis": [],
        "extracted_info": {"research_question": [], "scope": [], "innovations": []},
        "evaluation_items": [],
        "score_impact": "",
        "suggestions": [],
    }

    # 提取章节号和名称
    # 格式：第 1 章: 第一章 绪论
    title_match = re.search(r"第\s*(\d+)\s*章:\s*(.+)", block)
    if title_match:
        chapter["chapter_number"] = int(title_match.group(1))
        chapter["chapter_name"] = title_match.group(2).strip()

    # 提取章节类型
    type_match = re.search(r"章节类型:\s*(\w+)", block)
    if type_match:
        chapter["chapter_type"] = type_match.group(1).strip()

    # 提取章节内容摘要
    summary_match = re.search(
        r"📝\s*章节内容摘要:\s*(.+?)(?=💡|🏗️|📊|🔍)", block, re.DOTALL
    )
    if summary_match:
        chapter["summary"] = summary_match.group(1).strip()

    # 提取章节综合评价
    eval_match = re.search(r"💡\s*章节综合评价:\s*(.+?)(?=🏗️|📊|🔍)", block, re.DOTALL)
    if eval_match:
        eval_text = eval_match.group(1).strip()
        # 尝试分离优点和不足
        strengths_match = re.search(
            r"优点[:：]\s*(.+?)(?=不足|$)", eval_text, re.DOTALL
        )
        weaknesses_match = re.search(r"不足[:：]\s*(.+)", eval_text, re.DOTALL)

        if strengths_match:
            chapter["evaluation"]["strengths"] = strengths_match.group(1).strip()
        if weaknesses_match:
            chapter["evaluation"]["weaknesses"] = weaknesses_match.group(1).strip()

    # 提取结构分析
    structure_match = re.search(
        r"🏗️\s*章节结构分析:\s*(.+?)(?=📊|🔍|⚠️|💡)", block, re.DOTALL
    )
    if structure_match:
        structure_text = structure_match.group(1).strip()
        chapter["structure_analysis"] = parse_structure_analysis(structure_text)

    # 提取信息
    info_match = re.search(r"📊\s*提取信息:\s*(.+?)(?=🔍|⚠️|💡)", block, re.DOTALL)
    if info_match:
        info_text = info_match.group(1).strip()
        chapter["extracted_info"] = parse_extracted_info(info_text)

    # 提取专项评估结果
    items_match = re.search(r"🔍\s*专项评估结果:\s*(.+?)(?=⚠️|💡)", block, re.DOTALL)
    if items_match:
        items_text = items_match.group(1).strip()
        chapter["evaluation_items"] = parse_evaluation_items(items_text)

    # 提取对评分的影响
    impact_match = re.search(r"⚠️\s*对评分的影响:\s*(.+?)(?=💡|$)", block, re.DOTALL)
    if impact_match:
        chapter["score_impact"] = impact_match.group(1).strip()

    # 提取修改建议
    suggestions_match = re.search(r"💡\s*修改建议:\s*(.+?)(?=$|📖)", block, re.DOTALL)
    if suggestions_match:
        suggestions_text = suggestions_match.group(1).strip()
        chapter["suggestions"] = parse_suggestions(suggestions_text)

    return chapter


def parse_structure_analysis(text: str) -> List[Dict[str, Any]]:
    """
    解析结构分析部分

    格式:
    1. 1.1 研究背景
       目的: xxx
       关键点: xxx
       不足: xxx
    """
    sections = []

    # 按数字序号分割
    items = re.split(r"^\d+\.\s*", text, flags=re.MULTILINE)

    for item in items:
        if not item.strip():
            continue

        section = {"section": "", "purpose": "", "key_points": [], "deficiencies": ""}

        # 提取小节标题（第一行）
        lines = item.strip().split("\n")
        if lines:
            section["section"] = lines[0].strip()

        # 提取目的
        purpose_match = re.search(
            r"目的[:：]\s*(.+?)(?=关键点|不足|$)", item, re.DOTALL
        )
        if purpose_match:
            section["purpose"] = purpose_match.group(1).strip()

        # 提取关键点
        points_match = re.search(r"关键点[:：]\s*(.+?)(?=不足|$)", item, re.DOTALL)
        if points_match:
            points_text = points_match.group(1).strip()
            # 分割关键点（可能是列表格式）
            # 修复后的正则表达式：支持连字符、中点、分号等常见分隔符
            points = re.split(r"[-·；;]\s*", points_text)
            section["key_points"] = [p.strip() for p in points if p.strip()]

        # 提取不足
        deficiencies_match = re.search(r"不足[:：]\s*(.+)", item, re.DOTALL)
        if deficiencies_match:
            section["deficiencies"] = deficiencies_match.group(1).strip()

        if section["section"]:
            sections.append(section)

    return sections


def parse_extracted_info(text: str) -> Dict[str, List[str]]:
    """
    解析提取信息部分
    """
    info = {"research_question": [], "scope": [], "innovations": []}

    # 核心研究问题
    question_match = re.search(
        r"核心研究问题.*?[:：]\s*(.+?)(?=研究范围|创新点|$)", text, re.DOTALL
    )
    if question_match:
        question_text = question_match.group(1).strip()
        info["research_question"] = re.split(r"[\n;；]", question_text)

    # 研究范围
    scope_match = re.search(
        r"研究范围.*?[:：]\s*(.+?)(?=创新点|核心研究|$)", text, re.DOTALL
    )
    if scope_match:
        scope_text = scope_match.group(1).strip()
        info["scope"] = re.split(r"[\n;；]", scope_text)

    # 创新点
    innovations_match = re.search(
        r"创新点.*?[:：]\s*(.+?)(?=核心研究|研究范围|$)", text, re.DOTALL
    )
    if innovations_match:
        innovations_text = innovations_match.group(1).strip()
        info["innovations"] = re.split(r"[\n;；]", innovations_text)

    return info


def parse_evaluation_items(text: str) -> List[Dict[str, str]]:
    """
    解析专项评估项

    格式:
    ├─ 理论背景阐述质量: [评估内容][标签]
    ├─ xxx: xxx[无问题]
    """
    items = []

    # 匹配每一项
    lines = text.split("\n")
    for line in lines:
        # 提取评估项和内容
        match = re.search(r"[├│─\s]*(.+?)[:：]\s*(.+)\[([^\]]+)\]", line)
        if match:
            item_name = match.group(1).strip()
            item_name = re.sub(r"^[├│─\s]+", "", item_name)
            assessment = match.group(2).strip()
            label = match.group(3).strip()

            items.append({"item": item_name, "assessment": assessment, "label": label})

    return items


def parse_suggestions(text: str) -> List[Dict[str, str]]:
    """
    解析修改建议

    格式:
    1. [1.1 研究背景] 具体建议内容
    2. [1.2 xxx] 建议
    """
    suggestions = []

    # 按数字序号分割
    items = re.split(r"^\d+\.\s*", text, flags=re.MULTILINE)

    for item in items:
        if not item.strip():
            continue

        # 提取位置和建议
        match = re.match(r"\[([^\]]+)\]\s*(.+)", item.strip())
        if match:
            location = match.group(1).strip()
            advice = match.group(2).strip()

            suggestions.append({"location": location, "advice": advice})

    return suggestions


def convert_single_student(student_folder: Path) -> bool:
    """
    转换单个学生的数据
    """
    print(f"处理学生: {student_folder.name}")

    structure_file = student_folder / "chapter_structure.txt"
    information_file = student_folder / "chapter_information.txt"

    # 检查文件是否存在
    if not structure_file.exists():
        print(f"  跳过: 未找到 chapter_structure.txt")
        return False

    if not information_file.exists():
        print(f"  跳过: 未找到 chapter_information.txt")
        return False

    # 解析文件
    structure_data = parse_chapter_structure(structure_file)
    information_data = parse_chapter_information(information_file)

    # 保存 JSON
    if structure_data:
        output_file = student_folder / "chapter_structure.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(structure_data, f, ensure_ascii=False, indent=2)
        print(f"  生成: chapter_structure.json")

    if information_data:
        output_file = student_folder / "chapter_information.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(information_data, f, ensure_ascii=False, indent=2)
        print(f"  生成: chapter_information.json")

    return True


def convert_all_students():
    """
    转换所有学生的数据
    """
    print("=" * 60)
    print("开始转换学生数据")
    print("=" * 60)

    if not USER_RESULT_DIR.exists():
        print(f"错误: 目录不存在 {USER_RESULT_DIR}")
        return

    # 统计
    total = 0
    success = 0
    failed = 0

    # 遍历所有学生文件夹
    for student_folder in USER_RESULT_DIR.iterdir():
        if not student_folder.is_dir():
            continue

        total += 1
        if convert_single_student(student_folder):
            success += 1
        else:
            failed += 1

    print("=" * 60)
    print(f"转换完成!")
    print(f"总计: {total}")
    print(f"成功: {success}")
    print(f"失败: {failed}")
    print("=" * 60)


if __name__ == "__main__":
    convert_all_students()
