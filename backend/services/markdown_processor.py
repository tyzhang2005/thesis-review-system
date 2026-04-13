import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config.config import MINERU_OUTPUT_DIR
from langchain_core.documents import Document


class ChineseMarkdownSplitter:
    def __init__(self, content_list_path: Optional[str] = None):
        # 优化后的标题匹配模式（按优先级排序）
        self.header_patterns = [
            # 匹配“第X章”，
            (r"^#\s第[一二三四五六七八九十]+章[\s\S]+", "chapter"),
            # 匹配“X.X.X 小节”格式
            (r"^[\s#]*[\d]+[\.|\u3002]+[\d]+[\.|\u3002]+[\d]+[\s\S]+", "subsection"),
            # 匹配“X.X 节”格式
            (r"^[\s#]*[\d]+[\.|\u3002]+[\d]+[\s\S]+", "section"),
            (r"^#\s*([\d]|[一二三四五六七八九十]、)+[\s\S]+", "weak_chapter"),
            # 匹配带括号的子标题（如“（1）”或“(1)”）
            (r"^[\s#]*[（|（][\d]+[）|）][\s\S]*", "subsubsection"),
            # 匹配特殊章节（如“附录”、“参考文献”等）
            (r"^[\s#]*(附录|参考文献|致\s*谢|致谢|声明)[\s\S]*$", "special_chapter"),
        ]
        # 层级关系映射表
        self.level_map = {
            "chapter": 1,
            "section": 2,
            "subsection": 3,
            "subsubsection": 4,
            "special_chapter": 1,
        }
        self.min_chunk_size = 0  # 最小分块字符数
        self.max_chunk_size = 200000  # 最大分块字符数

        self.merge_threshold = 500  # 合并阈值
        self.split_threshold = 1200  # 分割阈值

        # AIGC检测专用阈值
        self.aigc_merge_threshold = 100  # AIGC检测合并阈值
        self.aigc_split_threshold = 500  # AIGC检测分割阈值

        # 新增：调试输出路径
        self.enable_debug_output = False
        self.debug_output_path = "math_formula_debug.txt"
        self.debug_info = []

        # AIGC检测使用 与原文pdf位置映射相关
        self.content_list_path = content_list_path
        self.content_list = None
        self.content_blocks = None  # 存储type='text'的块
        self.matched_blocks = set()  # 跟踪已匹配的blocks，确保不重复分配

        if content_list_path:
            self._load_content_list(content_list_path)

    def _parse_structure(self, text: str) -> List[Dict]:
        lines = text.split("\n")
        structure = []
        current_hierarchy = {
            "title": None,
            "student_name": None,
            "student_id": None,
            "abstract": None,
            "keywords": None,
            "english_abstract": None,
            "english_keywords": None,
            "contents": "",
            "chapter": None,
            "section": None,
            "subsection": None,
            "subsubsection": None,
            "content": [],
        }

        # 标志位，用于控制跳过目录和英文部分
        skip_content = False
        abstract_started = False  # 用于标记摘要是否已经开始
        english_abstract_started = False
        english_keywords_ended = False
        contents_started = False
        contents_ended = False
        chapter_cn = False
        fist_title_index = -1
        content_index = -1

        if not re.search(r"#\s*(目录|目\s*录|I|目|录)[\s\S]*", text):
            for i in range(len(lines) - 1, 0, -1):
                line_ch = (
                    lines[i]
                    .replace(" ", "")
                    .replace(".", "")
                    .replace("：", "")
                    .replace(":", "")
                    .replace("、", "")
                )
                # print(line_ch)
                if fist_title_index != -1 and re.search(
                    r"^(1|第一章|一)[\u4e00-\u9fa5a-zA-Z]+", line_ch
                ):
                    # print(line_ch)
                    content_index = i
                    for j in range(content_index, fist_title_index):
                        current_hierarchy["contents"] += lines[j] + "\n"
                    contents_ended = True
                    break
                if (
                    re.match(r"^#(1|第一章|一)[\u4e00-\u9fa5a-zA-Z]+", line_ch)
                    and fist_title_index == -1
                ):
                    # print(lines[i])
                    fist_title_index = i
                    contents_started = True
                    # skip_content = True

                if re.search(
                    r"(?:Keyword|英文关键词|Key\s*words|KEYWORDS)[：:]?[\s\S]*",
                    lines[i],
                ):
                    break

            if not contents_ended:
                contents_ended = True
                current_hierarchy["contents"] += "缺失目录"

        # print(text)
        for index, line in enumerate(lines):
            line = line.strip()
            header_found = False

            # 检测学号
            if not current_hierarchy["student_id"] and line:

                # 匹配包含“年级”和“学号”的情况，允许数字之间有空格
                id_match = re.search(r"学\s*号[：:]?\s*(\d{9})", line)
                if id_match:
                    # print(id_match)
                    current_hierarchy["student_id"] = (
                        id_match.group(1).split()[-1].split("：")[-1]
                    )
                    # continue

            # 检测标题（假设标题在文档开头，无特定格式）
            if (
                not current_hierarchy["title"]
                and line
                and not line.startswith("#")
                and not current_hierarchy["abstract"]
            ):
                # if (lines[0] != "# 南京大学"):
                # current_hierarchy["title"] = re.sub(r"# ", "", lines[0]).strip()
                # 检测中文标题格式
                # print(line)
                ti_match = re.search(r"题\s*目：?[\s\S]+", line)
                if ti_match:
                    # print(re.sub(r"^题目：", r"", line))
                    current_hierarchy["title"] = (
                        ti_match.group(0)
                        .split("院系")[0]
                        .split("年级")[0]
                        .split("题目")[-1]
                        .split("题 目")[-1]
                        .split("年 级")[0]
                    )
                # continue

            # 检测学生姓名
            if not current_hierarchy["student_name"] and line:

                name_match = re.search(
                    r"(?:学生姓名|本科生姓名)\s*[:：]?\s*([\u4e00-\u9fa5]{2,4})", line
                )
                if not name_match:
                    # 尝试其他可能的格式
                    name_match = re.match(r"学生姓名\s*([\u4e00-\u9fa5]{2,4})", line)
                if name_match:
                    current_hierarchy["student_name"] = (
                        name_match.group(1).strip().split("指")[0]
                    )
                    continue

            # 检测摘要开始
            if not abstract_started and re.search(r"[\s#]*摘要：?[\s\S]*", line):
                abstract_started = True
                current_hierarchy["abstract"] = line
                continue

            if (
                abstract_started
                and not current_hierarchy["title"]
                and lines[0] != "# 南京大学"
            ):
                current_hierarchy["title"] = re.sub(r"# ", "", lines[0]).strip()

            # 如果摘要已经开始，并且尚未遇到关键词，将内容添加到摘要
            if abstract_started and not current_hierarchy["keywords"]:
                kw_match = re.search(r"(?:关键词|关键字)：[\s\S]*", line)
                # print(kw_match)
                if kw_match:
                    # print(re.search(r"(?:关键词|关键字)：[\s\S]*", line))
                    current_hierarchy["keywords"] = kw_match.group(0)
                    current_hierarchy["abstract"] += "\n" + line.split("关")[0]
                    current_hierarchy["abstract"] = (
                        current_hierarchy["abstract"]
                        .split("摘要：")[-1]
                        .split("# ：")[-1]
                    )
                    if len(current_hierarchy["keywords"].split("：")[-1]) == 0:
                        for i in range(index + 1, len(lines)):
                            if len(lines[i]) != 0:
                                current_hierarchy["keywords"] += lines[i]
                                break

                else:
                    current_hierarchy["abstract"] += "\n" + line
                continue

            # 检测关键词部分
            # if not current_hierarchy["keywords"] and re.match(r"^关键词：[\s\S]*", line):
            # current_hierarchy["keywords"] = re.sub(r"^关键词：", "", line).strip()
            # continue

            # 检测摘要开始
            if not english_abstract_started and re.search(
                r"^[\s\S]*(?:英文摘要|Abstract|ABSTRACT)[\s\S]*", line
            ):
                # print(line)
                english_abstract_started = True
                current_hierarchy["english_abstract"] = line.split("ABSTRACT:")[-1]
                continue

            # 如果摘要已经开始，并且尚未遇到关键词，将内容添加到摘要
            if english_abstract_started and not english_keywords_ended:
                ekw_match = re.search(
                    r"(?:Keyword|英文关键词|Key\s*words|KEYWORDS)[：:]?[\s\S]*", line
                )
                # print(ekw_match)

                if ekw_match:  # 修复：支持半角/全角冒号

                    # print(ekw_match)
                    current_hierarchy["english_keywords"] = (
                        ekw_match.group(0)
                        .split("KEYWORDS")[-1]
                        .split("Key words")[-1]
                        .split("英文关键词")[-1]
                        .split(":")[-1]
                    )
                    current_hierarchy["english_abstract"] += (
                        "\n"
                        + line.split("KEYWORDS")[0]
                        .split("Key words")[0]
                        .split("英文关键词")[0]
                        .split("Keyword")[0]
                    )
                    current_hierarchy["english_abstract"] = (
                        current_hierarchy["english_abstract"]
                        .split("ABSTRACT")[-1]
                        .split("Abstract")[-1]
                        .split("英文摘要")[-1]
                        .split("：")[-1]
                        .split(":")[-1]
                        .split("#")[0]
                    )

                    english_keywords_ended = True  # 标记关键词已结束
                    for i in range(index + 1, len(lines)):
                        if len(lines[i]) == 0:
                            continue
                        if lines[i][0] != "#":
                            current_hierarchy["english_keywords"] += lines[i]

                        if lines[i][0] == "#":
                            break
                else:
                    current_hierarchy["english_abstract"] += "\n" + line
                continue

            # 检测目录开始
            if not contents_started and re.match(
                r"^#\s*(目录|目\s*录|I|目|录)[\s\S]*", line
            ):
                # print(file)
                # print(line)
                contents_started = True
                current_hierarchy["contents"] += line + "\n"
                contents = line + "\n"

                for i in range(index + 1, len(lines)):
                    contents_ch = (
                        contents.replace("#", "")
                        .replace(" ", "")
                        .replace("\n", "")
                        .replace(".", "")
                        .replace("：", "")
                        .replace(":", "")
                        .replace("、", "")
                    )
                    line_ch = (
                        lines[i]
                        .replace("#", "")
                        .replace(" ", "")
                        .replace(".", "")
                        .replace("：", "")
                        .replace(":", "")
                        .replace("、", "")
                    )
                    # print([contents_ch], [line_ch])
                    if (
                        line_ch in contents_ch
                        and re.match(r"^(1|第一章)[\u4e00-\u9fa5a-zA-Z]+", line_ch)
                        and "#" in lines[i]
                    ):
                        # print(file)

                        # print(lines[i])
                        contents_ended = True
                        current_hierarchy["contents"] = contents
                        # skip_content = True
                        fist_title_index = i
                        break
                    contents += lines[i] + "\n"
                if not contents_ended:
                    for i in range(len(lines) - 1, index, -1):
                        line_ch = (
                            lines[i]
                            .replace(" ", "")
                            .replace(".", "")
                            .replace("：", "")
                            .replace(":", "")
                            .replace("、", "")
                        )
                        # print(line_ch)
                        if re.match(r"^#(1|第一章)[\u4e00-\u9fa5a-zA-Z]+", line_ch):
                            # print(lines[i])
                            fist_title_index = i
                            break
                continue

            """
            # 如果目录已经开始，并且尚未遇到致谢，将内容添加到目录
            if contents_started and not contents_ended:
                contents_ch = current_hierarchy["contents"].replace("#", "").replace(" ", "").replace("\n", "").replace(".", "").replace("：", "").replace(":", "")
                line_ch = line.replace("#", "").replace(" ", "").replace(".", "").replace("：", "").replace(":", "")
                #print([contents_ch], [line_ch])
                if line_ch in contents_ch and re.match(r"^(1|第一章)[\u4e00-\u9fa5a-zA-Z]+", line_ch):
                    #print(file)
                    print([contents_ch], [line_ch])
                    print(line)
                    contents_ended = True
                    skip_content = True
                current_hierarchy["contents"] += line + "\n"
                continue
            """

            if contents_started and not skip_content:
                if index == fist_title_index:
                    contents_ch = (
                        current_hierarchy["contents"]
                        .replace("#", "")
                        .replace(" ", "")
                        .replace("\n", "")
                        .replace(".", "")
                        .replace("：", "")
                        .replace(":", "")
                    )
                    skip_content = True
                    working = True
                    # print([contents_ch])
                    # print(line)
                else:
                    if not contents_ended:
                        current_hierarchy["contents"] += line + "\n"
                    continue

            # if skip_content and working:
            #    if re.match(r"^[\s#]*(第[一二三四五六七八九十]+章|[\d]+[\.])[\s\S]+", line):
            #        working = False
            #        current_hierarchy["chapter"] = line.replace("# ", "")
            #        current_hierarchy["content"] = []
            #    continue

            # 检测标题层级（优化后的逻辑）

            if line == "#" or line == "# 1" or line == "# i\\*":
                if current_hierarchy["content"]:
                    structure.append(current_hierarchy.copy())
                    current_hierarchy["content"] = []
                current_hierarchy["chapter"] = "参考文献"
                current_hierarchy["section"] = None
                current_hierarchy["subsection"] = None
                current_hierarchy["subsubsection"] = None
                continue

            for pattern, level_type in self.header_patterns:

                if re.search(pattern, line):
                    # 处理章节跳转时保存当前内容
                    # print(line, level_type)
                    if not chapter_cn and level_type == "chapter":
                        chapter_cn = True
                    line_ch = (
                        line.replace("#", "")
                        .replace(" ", "")
                        .replace(".", "")
                        .replace("：", "")
                        .replace(":", "")
                        .replace("、", "")
                    )

                    if level_type == "special_chapter":
                        level_type = "chapter"

                    if level_type == "weak_chapter":
                        if not chapter_cn:
                            level_type = "chapter"
                        else:
                            current_hierarchy["content"].append(line)
                            break

                    if current_hierarchy[level_type]:
                        # print(current_hierarchy[level_type].replace("#", "").replace(" ", "").replace(".", "").replace("：", "").replace(":","").replace("、", "") , line_ch)
                        if (
                            current_hierarchy[level_type]
                            .replace("#", "")
                            .replace(" ", "")
                            .replace(".", "")
                            .replace("：", "")
                            .replace(":", "")
                            .replace("、", "")
                            in line_ch
                        ):
                            current_hierarchy["content"].append(line)
                            break

                    if current_hierarchy["content"] and current_hierarchy["chapter"]:
                        # print(current_hierarchy["content"])
                        structure.append(current_hierarchy.copy())
                        current_hierarchy["content"] = []

                    # 更新层级结构
                    if level_type == "chapter":
                        current_hierarchy["chapter"] = line.replace("# ", "")
                        current_hierarchy["section"] = None
                        current_hierarchy["subsection"] = None
                        current_hierarchy["subsubsection"] = None
                    elif level_type == "section":
                        current_hierarchy["section"] = line.replace("# ", "")
                        current_hierarchy["subsection"] = None
                        current_hierarchy["subsubsection"] = None
                    elif level_type == "subsection":
                        current_hierarchy["subsection"] = line.replace("# ", "")
                        current_hierarchy["subsubsection"] = None
                    elif level_type == "subsubsection":
                        current_hierarchy["subsubsection"] = line.replace("# ", "")
                    elif level_type == "special_chapter":
                        current_hierarchy["chapter"] = line.replace("# ", "")
                        current_hierarchy["section"] = None
                        current_hierarchy["subsection"] = None
                        current_hierarchy["subsubsection"] = None
                    header_found = True
                    break

            if not header_found and line:
                # print(line)
                current_hierarchy["content"].append(line)

        # 添加最后一个块
        if current_hierarchy["content"]:
            structure.append(current_hierarchy)

        if len(structure) == 0 and current_hierarchy["chapter"]:
            structure.append(current_hierarchy)

        if not contents_started:
            # print(file)
            print("失败")
        return structure

    def _split_into_paragraphs(self, content: str) -> List[str]:
        """将内容按自然段分割"""
        # 使用换行符分割段落
        paragraphs = re.split(r"\n", content)
        # 过滤空段落
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        return paragraphs

    def _merge_paragraphs(self, paragraphs: List[str]) -> List[str]:
        """合并小段落"""
        merged = []
        current_chunk = ""

        for para in paragraphs:
            # 如果当前块为空，直接添加
            if not current_chunk:
                current_chunk = para
                continue

            # 如果当前段落很小，尝试合并
            if len(current_chunk) < self.merge_threshold:
                # 合并后不超过分割阈值则合并
                if len(current_chunk) + len(para) < self.split_threshold:
                    current_chunk += "\n\n" + para
                else:
                    # 超过分割阈值，先保存当前块，然后开始新块
                    merged.append(current_chunk)
                    current_chunk = para
            else:
                # 当前段落已经足够大，先保存
                merged.append(current_chunk)
                current_chunk = para

        # 处理最后一个块
        if merged and len(current_chunk) < self.merge_threshold:
            # 尝试与上一个块合并
            last_chunk = merged[-1]
            if len(last_chunk) + len(current_chunk) <= self.split_threshold:
                merged[-1] = last_chunk + "\n\n" + current_chunk
            else:
                merged.append(current_chunk)
        else:
            merged.append(current_chunk)

        return merged

    def _extract_chapter_structure(self, structure_items: List[Dict]) -> Dict:
        """
        从结构项中提取章节结构

        Args:
            structure_items: 结构项列表（字典）

        Returns:
            dict: 章节结构字典
        """
        chapter_structure = {}
        last_hierarchy = {}  # 记录上一个块的层级信息，用于去重

        for item in structure_items:
            # 获取当前块的层级信息
            chapter = item.get("chapter")
            section = item.get("section")
            subsection = item.get("subsection")
            subsubsection = item.get("subsubsection")

            # 如果所有层级都为空，跳过
            if not any([chapter, section, subsection, subsubsection]):
                continue

            # 检查是否与上一个块层级完全相同
            current_hierarchy = {
                "chapter": chapter,
                "section": section,
                "subsection": subsection,
                "subsubsection": subsubsection,
            }

            if current_hierarchy == last_hierarchy:
                continue  # 跳过重复的层级

            last_hierarchy = current_hierarchy.copy()

            # 处理章节
            if chapter:
                # 清理章节标题中的特殊字符
                clean_chapter = chapter.replace("#", "").strip()
                if clean_chapter not in chapter_structure:
                    chapter_structure[clean_chapter] = {}

                # 处理小节
                if section:
                    clean_section = section.replace("#", "").strip()
                    if clean_section not in chapter_structure[clean_chapter]:
                        chapter_structure[clean_chapter][clean_section] = []

                    # 处理子小节
                    if subsection:
                        clean_subsection = subsection.replace("#", "").strip()
                        subsection_entry = clean_subsection

                        # 处理子子小节
                        if subsubsection:
                            clean_subsubsection = subsubsection.replace("#", "").strip()
                            subsection_entry += f"\n          {clean_subsubsection}"

                        # 避免重复添加
                        if (
                            subsection_entry
                            not in chapter_structure[clean_chapter][clean_section]
                        ):
                            chapter_structure[clean_chapter][clean_section].append(
                                subsection_entry
                            )

        return chapter_structure

    def _format_chapter_structure(self, chapter_structure: Dict) -> str:
        """
        格式化章节结构为字符串

        Args:
            chapter_structure: 章节结构字典

        Returns:
            str: 格式化后的章节结构
        """
        if not chapter_structure:
            return "未检测到清晰的章节结构"

        formatted_structure = ""

        for chapter, sections in chapter_structure.items():
            formatted_structure += f"{chapter}\n"

            for section, subsections in sections.items():
                formatted_structure += f"    {section}\n"

                for subsection in subsections:
                    # 检查是否是包含子子小节的格式
                    if "\n" in subsection:
                        lines = subsection.split("\n")
                        formatted_structure += f"        {lines[0]}\n"
                        for line in lines[1:]:
                            formatted_structure += f"        {line}\n"
                    else:
                        formatted_structure += f"        {subsection}\n"

            formatted_structure += "\n"

        return formatted_structure.rstrip()  # 移除末尾多余的换行

    def _format_single_chapter(
        self,
        chapter_structure: Dict,
        chapter_title: str,
        chapter_word_counts: Dict = None,
    ) -> str:
        """
        格式化单个章节的结构为字符串

        Args:
            chapter_structure: 章节结构字典
            chapter_title: 要格式化的章节标题
            chapter_word_counts: 章节字数统计字典

        Returns:
            str: 格式化后的单个章节结构
        """
        if not chapter_structure or chapter_title not in chapter_structure:
            return f"未找到章节: {chapter_title}"

        # 如果有字数统计信息，添加到标题后
        if chapter_word_counts and chapter_title in chapter_word_counts:
            word_count = chapter_word_counts[chapter_title]
            formatted_chapter = (
                f"{chapter_title} (本章字数统计：{word_count}字)\n \n{chapter_title}\n "
            )
        else:
            formatted_chapter = f"{chapter_title}\n"

        sections = chapter_structure[chapter_title]

        for section, subsections in sections.items():
            formatted_chapter += f"    {section}\n"

            for subsection in subsections:
                # 检查是否是包含子子小节的格式
                if "\n" in subsection:
                    lines = subsection.split("\n")
                    formatted_chapter += f"        {lines[0]}\n"
                    for line in lines[1:]:
                        formatted_chapter += f"        {line}\n"
                else:
                    formatted_chapter += f"        {subsection}\n"

        return formatted_chapter

    def _get_all_chapter_titles(self, chapter_structure: Dict) -> List[str]:
        """
        获取所有章节标题列表

        Args:
            chapter_structure: 章节结构字典

        Returns:
            list: 章节标题列表
        """
        return list(chapter_structure.keys()) if chapter_structure else []

    def _split_math_and_text(self, text: str) -> List[Tuple[str, str]]:
        """将文本分割为公式部分和普通文本部分，支持单美元和双美元公式"""
        parts = []
        i = 0
        n = len(text)

        while i < n:
            # 查找双美元公式开始标记
            if (
                i < n - 1
                and text[i : i + 2] == "$$"
                and (i == 0 or text[i - 1] != "\\")
            ):
                # 找到双美元公式结束位置
                j = i + 2
                while j < n - 1:
                    if text[j : j + 2] == "$$" and (j == 0 or text[j - 1] != "\\"):
                        break
                    j += 1

                if j < n - 1:  # 找到了匹配的$$
                    # 添加公式前的文本部分
                    if i > 0:
                        parts.append(("text", text[:i]))

                    # 添加双美元公式部分（不包括$$符号）
                    math_content = text[i + 2 : j]
                    parts.append(("display_math", math_content))

                    # 更新文本和索引
                    text = text[j + 2 :]
                    n = len(text)
                    i = 0
                else:
                    # 没有找到匹配的$$，将剩余部分作为文本
                    parts.append(("text", text))
                    break
            # 查找单美元公式开始标记
            elif text[i] == "$" and (i == 0 or text[i - 1] != "\\"):
                # 找到单美元公式结束位置
                j = i + 1
                while j < n:
                    if text[j] == "$" and (j == 0 or text[j - 1] != "\\"):
                        break
                    j += 1

                if j < n:  # 找到了匹配的$
                    # 添加公式前的文本部分
                    if i > 0:
                        parts.append(("text", text[:i]))

                    # 添加单美元公式部分（不包括$符号）
                    math_content = text[i + 1 : j]
                    parts.append(("inline_math", math_content))

                    # 更新文本和索引
                    text = text[j + 1 :]
                    n = len(text)
                    i = 0
                else:
                    # 没有找到匹配的$，将剩余部分作为文本
                    parts.append(("text", text))
                    break
            else:
                i += 1

        # 处理剩余的文本
        if text:
            parts.append(("text", text))

        return parts

    def _count_math_content(self, math_text: str, math_type: str) -> int:
        """统计公式内容字数，区分单美元和双美元环境"""
        if math_type == "inline_math":
            return self._count_inline_math(math_text)
        else:  # display_math
            return self._count_display_math(math_text)

    def _count_inline_math(self, math_text: str) -> int:
        """统计单美元环境中的公式字数"""
        count = 0
        i = 0
        n = len(math_text)

        while i < n:
            char = math_text[i]

            # 跳过空白字符
            if char.isspace():
                i += 1
                continue

            # 处理命令：以\开头的命令
            if char == "\\":
                # 找到命令结束位置
                j = i + 1
                while j < n and math_text[j].isalpha():
                    j += 1

                command = math_text[i:j]

                # 检查命令后是否有括号内容
                if j < n and math_text[j] == "{":
                    # 找到匹配的}
                    k = j + 1
                    brace_depth = 1
                    while k < n and brace_depth > 0:
                        if math_text[k] == "{":
                            brace_depth += 1
                        elif math_text[k] == "}":
                            brace_depth -= 1
                        k += 1

                    if brace_depth == 0:
                        # 括号内容整体算1个字
                        count += 1
                        i = k
                    else:
                        # 没有匹配的}，命令本身算1个字
                        count += 1
                        i = j
                else:
                    # 命令本身算1个字
                    count += 1
                    i = j
                continue

            # 处理带括号的内容（非命令）
            if char == "{":
                # 找到匹配的}
                j = i + 1
                brace_depth = 1
                while j < n and brace_depth > 0:
                    if math_text[j] == "{":
                        brace_depth += 1
                    elif math_text[j] == "}":
                        brace_depth -= 1
                    j += 1

                if brace_depth == 0:
                    # 括号内容整体算1个字
                    count += 1
                    i = j
                else:
                    # 没有匹配的}，跳过
                    i += 1
                continue

            # 处理上标符号 ^
            if char == "^":
                # ^本身不算字
                i += 1
                # 检查^后是否有括号内容
                if i < n and math_text[i] == "{":
                    # 找到匹配的}
                    j = i + 1
                    brace_depth = 1
                    while j < n and brace_depth > 0:
                        if math_text[j] == "{":
                            brace_depth += 1
                        elif math_text[j] == "}":
                            brace_depth -= 1
                        j += 1

                    if brace_depth == 0:
                        # 括号内容整体算1个字
                        count += 1
                        i = j
                    else:
                        i += 1
                else:
                    # ^后是单个字符，算1个字
                    if i < n and not math_text[i].isspace():
                        count += 1
                        i += 1
                continue

            # 处理下标符号 _
            if char == "_":
                # _本身不算字
                i += 1
                # 检查_后是否有括号内容
                if i < n and math_text[i] == "{":
                    # 找到匹配的}
                    j = i + 1
                    brace_depth = 1
                    while j < n and brace_depth > 0:
                        if math_text[j] == "{":
                            brace_depth += 1
                        elif math_text[j] == "}":
                            brace_depth -= 1
                        j += 1

                    if brace_depth == 0:
                        # 括号内容整体算1个字
                        count += 1
                        i = j
                    else:
                        i += 1
                else:
                    # _后是单个字符，算1个字
                    if i < n and not math_text[i].isspace():
                        count += 1
                        i += 1
                continue

            # 处理区间括号
            if char == "[" or char == "]":
                count += 1
                i += 1
                continue

            # 处理冒号
            if char == ":":
                count += 1
                i += 1
                continue

            # 处理逗号 - 不算字
            if char == ",":
                i += 1
                continue

            # 处理等号 - 不算字
            if char == "=":
                i += 1
                continue

            # 处理加减号 - 不算字
            if char == "+" or char == "-":
                i += 1
                continue

            # 处理其他字符（字母、数字等）
            if not char.isspace():
                count += 1

            i += 1

        return count

    def _count_display_math(self, math_text: str) -> int:
        """统计双美元环境中的公式字数"""
        count = 0
        i = 0
        n = len(math_text)

        # 移除空白字符
        math_text = re.sub(r"\s+", "", math_text)
        n = len(math_text)

        while i < n:
            char = math_text[i]

            # 处理命令：以\开头的命令
            if char == "\\":
                # 找到命令结束位置
                j = i + 1
                while j < n and math_text[j].isalpha():
                    j += 1

                command = math_text[i:j]

                # 特殊处理\tag命令
                if command == "\\tag":
                    # 找到\tag后的括号内容
                    if j < n and math_text[j] == "{":
                        k = j + 1
                        brace_depth = 1
                        while k < n and brace_depth > 0:
                            if math_text[k] == "{":
                                brace_depth += 1
                            elif math_text[k] == "}":
                                brace_depth -= 1
                            k += 1

                        if brace_depth == 0:
                            # 提取括号内的内容
                            tag_content = math_text[j + 1 : k - 1]
                            # 统计数字和字母，忽略标点符号
                            tag_count = len(re.findall(r"[a-zA-Z0-9]", tag_content))
                            count += tag_count
                            i = k
                        else:
                            i = j
                    else:
                        i = j
                    continue

                # 检查命令后是否有括号内容
                if j < n and math_text[j] == "{":
                    # 找到匹配的}
                    k = j + 1
                    brace_depth = 1
                    while k < n and brace_depth > 0:
                        if math_text[k] == "{":
                            brace_depth += 1
                        elif math_text[k] == "}":
                            brace_depth -= 1
                        k += 1

                    if brace_depth == 0:
                        # 命令和括号内容整体算1个字
                        count += 1
                        i = k
                    else:
                        # 没有匹配的}，命令本身算1个字
                        count += 1
                        i = j
                else:
                    # 命令本身算1个字
                    count += 1
                    i = j
                continue

            # 处理带括号的内容（非命令）
            if char == "{":
                # 找到匹配的}
                j = i + 1
                brace_depth = 1
                while j < n and brace_depth > 0:
                    if math_text[j] == "{":
                        brace_depth += 1
                    elif math_text[j] == "}":
                        brace_depth -= 1
                    j += 1

                if brace_depth == 0:
                    # 括号内容整体算1个字
                    count += 1
                    i = j
                else:
                    # 没有匹配的}，跳过
                    i += 1
                continue

            # 处理上标符号 ^
            if char == "^":
                # ^本身不算字
                i += 1
                # 检查^后是否有括号内容
                if i < n and math_text[i] == "{":
                    # 找到匹配的}
                    j = i + 1
                    brace_depth = 1
                    while j < n and brace_depth > 0:
                        if math_text[j] == "{":
                            brace_depth += 1
                        elif math_text[j] == "}":
                            brace_depth -= 1
                        j += 1

                    if brace_depth == 0:
                        # 上标整体算1个字
                        count += 1
                        i = j
                    else:
                        i += 1
                else:
                    # ^后是单个字符，上标算1个字
                    if i < n and not math_text[i].isspace():
                        count += 1
                        i += 1
                continue

            # 处理下标符号 _
            if char == "_":
                # _本身不算字
                i += 1
                # 检查_后是否有括号内容
                if i < n and math_text[i] == "{":
                    # 找到匹配的}
                    j = i + 1
                    brace_depth = 1
                    while j < n and brace_depth > 0:
                        if math_text[j] == "{":
                            brace_depth += 1
                        elif math_text[j] == "}":
                            brace_depth -= 1
                        j += 1

                    if brace_depth == 0:
                        # 下标整体算1个字
                        count += 1
                        i = j
                    else:
                        i += 1
                else:
                    # _后是单个字符，下标算1个字
                    if i < n and not math_text[i].isspace():
                        count += 1
                        i += 1
                continue

            # 处理字母和数字 - 每个算1个字
            if char.isalnum():
                count += 1

            # 其他字符（标点符号等）在双美元环境中不算字
            i += 1

        return count

    def _count_characters(self, text: str) -> Tuple[int, int]:
        """按照Word标准和公式规则统计字数，返回(总字数, 公式字数)"""
        if not text:
            return 0, 0

        # 预处理：分割文本为公式部分和普通文本部分
        parts = self._split_math_and_text(text)
        total_count = 0
        formula_count = 0

        for part_type, content in parts:
            if part_type in ["inline_math", "display_math"]:
                math_count = self._count_math_content(content, part_type)
                total_count += math_count
                formula_count += math_count
                # 记录调试信息
                self.debug_info.append(
                    {"type": part_type, "content": content, "count": math_count}
                )
            else:
                text_count = self._count_text_content(content)
                total_count += text_count
                # 记录调试信息
                if content.strip():  # 只记录非空文本
                    self.debug_info.append(
                        {
                            "type": "text",
                            "content": (
                                content[:100] + "..." if len(content) > 100 else content
                            ),
                            "count": text_count,
                        }
                    )

        return total_count, formula_count

    def _count_text_content(self, text: str) -> int:
        """统计普通文本内容字数，排除图片链接"""
        total_count = 0

        # 预处理：移除图片链接
        text = self._remove_image_links(text)

        # 预处理：移除HTML标签
        text = re.sub(r"<[^>]+>", "", text)

        # 预处理：将参考文献标记为特殊令牌
        text = re.sub(r"\[(\d+)\]", r"[REF\1]", text)

        # 分割文本为令牌
        tokens = re.findall(
            r"\[REF\d+\]|[\u4e00-\u9fff\u3000-\u303f]+|[a-zA-Z]+|[0-9]+|.", text
        )

        for token in tokens:
            # 跳过空白字符
            if token.isspace():
                continue

            # 参考文献整体算1个字
            if token.startswith("[REF"):
                total_count += 1
            # 中文字符（包括中文标点）算1个字
            elif re.match(r"^[\u4e00-\u9fff\u3000-\u303f]+$", token):
                total_count += len(token)
            # 完整的英文单词算1个字
            elif re.match(r"^[a-zA-Z]+$", token):
                total_count += 1
            # 数字序列算1个字
            elif re.match(r"^[0-9]+$", token):
                total_count += 1
            # 英文标点符号算0.5个字
            elif re.match(r"^[^\u4e00-\u9fff\u3000-\u303f\s]$", token):
                total_count += 0.5
            # 其他字符（如混合字符）按中文字符计算
            else:
                # 计算其中的中文字符数量
                chinese_chars = re.findall(r"[\u4e00-\u9fff\u3000-\u303f]", token)
                total_count += len(chinese_chars)

                # 计算英文单词数量
                english_words = re.findall(r"[a-zA-Z]+", token)
                total_count += len(english_words)

                # 计算数字序列数量
                number_sequences = re.findall(r"[0-9]+", token)
                total_count += len(number_sequences)

                # 计算英文标点数量
                english_punctuation = re.findall(
                    r"[^\u4e00-\u9fff\u3000-\u303f\sa-zA-Z0-9]", token
                )
                total_count += len(english_punctuation) * 0.5

        # 向上取整
        return int(total_count + 0.5)

    def _remove_image_links(self, text: str) -> str:
        """移除Markdown图片链接，防止图片文件名被计入字数"""
        # 匹配Markdown图片语法: ![](path/to/image.jpg)
        # 支持常见的图片扩展名
        image_extensions = [
            r"\.jpg",
            r"\.jpeg",
            r"\.png",
            r"\.gif",
            r"\.bmp",
            r"\.svg",
            r"\.webp",
        ]
        extension_pattern = "|".join(image_extensions)

        # 匹配图片链接并移除
        pattern = r"!\[\]\([^)]+" + extension_pattern + r"[^)]*\)"
        text = re.sub(pattern, "", text)

        return text

    def _write_debug_info(self, chapter_title: str = "Unknown Chapter"):
        """将调试信息写入txt文件（覆盖模式）"""
        if not self.enable_debug_output:
            return
        if not self.debug_info:
            return

        try:
            with open(self.debug_output_path, "a", encoding="utf-8") as f:
                f.write(f"{'='*80}\n")
                f.write(f"章节: {chapter_title}\n")
                f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{'='*80}\n\n")

                total_math_count = 0
                math_formulas = []

                for i, item in enumerate(self.debug_info):
                    if item["type"] in ["inline_math", "display_math"]:
                        total_math_count += item["count"]
                        math_formulas.append(item)

                        # 根据公式类型显示不同的包围符号
                        surround = "$" if item["type"] == "inline_math" else "$$"
                        f.write(f"{item['type']} 公式 #{len(math_formulas)}:\n")
                        f.write(f"  内容: {surround}{item['content']}{surround}\n")
                        f.write(f"  字数: {item['count']}\n")
                        f.write(
                            f"  分析: {self._analyze_math_formula(item['content'], item['type'])}\n\n"
                        )
                    elif item["type"] == "text" and item["count"] > 0:
                        f.write(f"文本段 #{i+1}:\n")
                        f.write(f"  内容: {item['content']}\n")
                        f.write(f"  字数: {item['count']}\n\n")

                f.write(f"{'-'*80}\n")
                f.write(f"本章节数学公式总数: {len(math_formulas)}\n")
                f.write(f"本章节数学公式总字数: {total_math_count}\n")
                f.write(
                    f"本章节总字数: {sum(item['count'] for item in self.debug_info)}\n"
                )

        except Exception as e:
            print(f"写入调试信息失败: {e}")

    def _analyze_math_formula(self, formula: str, formula_type: str) -> str:
        """分析公式并返回详细的字数统计信息"""
        analysis = []

        if formula_type == "inline_math":
            # 单美元环境的分析
            # 统计命令数量
            commands = re.findall(r"\\([a-zA-Z]+)", formula)
            if commands:
                analysis.append(f"命令: {len(commands)}个 ({', '.join(set(commands))})")

            # 统计带括号的内容
            brace_groups = re.findall(r"\{[^{}]*\}", formula)
            if brace_groups:
                analysis.append(f"括号组: {len(brace_groups)}个")

            # 统计单个字符
            single_chars = re.findall(r"[a-zA-Z]", formula)
            if single_chars:
                analysis.append(f"单个字母: {len(single_chars)}个")

            # 统计区间括号
            brackets = formula.count("[") + formula.count("]")
            if brackets:
                analysis.append(f"区间括号: {brackets}个")

            # 统计冒号
            colons = formula.count(":")
            if colons:
                analysis.append(f"冒号: {colons}个")

            # 统计上标和下标
            superscripts = len(re.findall(r"\^\{[^{}]+\}", formula)) + len(
                re.findall(r"\^[a-zA-Z0-9]", formula)
            )
            subscripts = len(re.findall(r"_\{[^{}]+\}", formula)) + len(
                re.findall(r"_[a-zA-Z0-9]", formula)
            )
            if superscripts:
                analysis.append(f"上标: {superscripts}个")
            if subscripts:
                analysis.append(f"下标: {subscripts}个")

        else:  # display_math
            # 双美元环境的分析
            # 统计命令数量
            commands = re.findall(r"\\([a-zA-Z]+)", formula)
            if commands:
                analysis.append(f"命令: {len(commands)}个 ({', '.join(set(commands))})")

            # 统计变量（字母序列）
            variables = re.findall(r"[a-zA-Z]+", formula)
            if variables:
                analysis.append(f"变量: {len(variables)}个")

            # 统计数字
            numbers = re.findall(r"\b\d+\b", formula)
            if numbers:
                analysis.append(f"数字: {len(numbers)}个")

            # 特殊处理\tag命令
            tag_matches = re.findall(r"\\tag\{[^}]+\}", formula)
            if tag_matches:
                analysis.append(f"标签: {len(tag_matches)}个")

            # 统计上下标
            superscripts = len(re.findall(r"\^\{[^{}]+\}", formula)) + len(
                re.findall(r"\^[a-zA-Z0-9]", formula)
            )
            subscripts = len(re.findall(r"_\{[^{}]+\}", formula)) + len(
                re.findall(r"_[a-zA-Z0-9]", formula)
            )
            if superscripts:
                analysis.append(f"上标: {superscripts}个")
            if subscripts:
                analysis.append(f"下标: {subscripts}个")

        return "; ".join(analysis) if analysis else "无特殊元素"

    def _calculate_word_counts(self, structure_items: List[Dict]) -> Dict:
        """
        计算各个章节的字数和论文总字数，包括摘要、关键词、目录

        Args:
            structure_items: 结构项列表

        Returns:
            dict: 包含字数字典和总字数的字典
        """
        chapter_word_counts = {}
        total_word_count = 0
        total_formula_word_count = 0  # 新增：总公式字数

        # 统计章节内容字数
        for item in structure_items:
            # 重置调试信息
            self.debug_info = []

            # 获取当前块的内容
            content = "\n".join(item["content"])

            # 计算当前块的字数（按照Word标准和公式规则）
            word_count, formula_count = self._count_characters(content)

            # 写入调试信息（仅在启用时）
            chapter_title = item.get("chapter", "Unknown Chapter")
            self._write_debug_info(chapter_title)

            # 获取章节信息
            chapter = item.get("chapter")

            # 如果没有章节信息，跳过
            if not chapter:
                continue

            # 清理章节标题
            clean_chapter = chapter.replace("#", "").strip()

            # 累加章节字数和公式字数
            if clean_chapter in chapter_word_counts:
                chapter_word_counts[clean_chapter] += word_count
            else:
                chapter_word_counts[clean_chapter] = word_count

            # 累加总字数和总公式字数
            total_word_count += word_count
            total_formula_word_count += formula_count

        # 统计摘要、关键词、目录的字数（从第一个块中提取）
        if structure_items:
            first_item = structure_items[0]

            # 重置调试信息
            self.debug_info = []

            # 统计摘要字数
            if first_item.get("abstract"):
                abstract_count, abstract_formula_count = self._count_characters(
                    first_item["abstract"]
                )
                total_word_count += abstract_count
                total_formula_word_count += abstract_formula_count
            else:
                abstract_count = 0
                abstract_formula_count = 0

            # 统计关键词字数
            if first_item.get("keywords"):
                keywords_count, keywords_formula_count = self._count_characters(
                    first_item["keywords"]
                )
                total_word_count += keywords_count
                total_formula_word_count += keywords_formula_count
            else:
                keywords_count = 0
                keywords_formula_count = 0

            # 统计英文摘要字数
            if first_item.get("english_abstract"):
                english_abstract_count, english_abstract_formula_count = (
                    self._count_characters(first_item["english_abstract"])
                )
                total_word_count += english_abstract_count
                total_formula_word_count += english_abstract_formula_count
            else:
                english_abstract_count = 0
                english_abstract_formula_count = 0

            # 统计英文关键词字数
            if first_item.get("english_keywords"):
                english_keywords_count, english_keywords_formula_count = (
                    self._count_characters(first_item["english_keywords"])
                )
                total_word_count += english_keywords_count
                total_formula_word_count += english_keywords_formula_count
            else:
                english_keywords_count = 0
                english_keywords_formula_count = 0

            # 统计目录字数
            if first_item.get("contents"):
                contents_count, contents_formula_count = self._count_characters(
                    first_item["contents"]
                )
                total_word_count += contents_count
                total_formula_word_count += contents_formula_count
            else:
                contents_count = 0
                contents_formula_count = 0
        else:
            abstract_count = keywords_count = english_abstract_count = (
                english_keywords_count
            ) = contents_count = 0
            abstract_formula_count = keywords_formula_count = (
                english_abstract_formula_count
            ) = english_keywords_formula_count = contents_formula_count = 0

        # 计算公式字数占比
        formula_word_ratio = 0.0
        if total_word_count > 0:
            formula_word_ratio = round(
                total_formula_word_count / total_word_count * 100, 2
            )

        return {
            "chapter_word_counts": chapter_word_counts,
            "total_word_count": total_word_count,
            "total_formula_word_count": total_formula_word_count,  # 新增：总公式字数
            "formula_word_ratio": formula_word_ratio,  # 新增：公式字数占比
            "non_chapter_counts": {
                "abstract": abstract_count,
                "keywords": keywords_count,
                "english_abstract": english_abstract_count,
                "english_keywords": english_keywords_count,
                "contents": contents_count,
            },
        }

    def _format_word_count_info(self, word_count_data: Dict) -> str:
        """
        格式化字数统计信息为字符串，包括摘要、关键词、目录

        Args:
            word_count_data: 字数统计数据

        Returns:
            str: 格式化后的字数统计信息
        """
        chapter_word_counts = word_count_data.get("chapter_word_counts", {})
        total_word_count = word_count_data.get("total_word_count", 0)
        non_chapter_counts = word_count_data.get("non_chapter_counts", {})

        if not chapter_word_counts and not non_chapter_counts:
            return f"总字数: {total_word_count}"

        formatted_info = f"论文总字数: {total_word_count}\n\n"

        # 添加非章节部分字数统计
        if non_chapter_counts:
            formatted_info += "各部分字数统计:\n"
            if non_chapter_counts.get("abstract", 0) > 0:
                formatted_info += f"  摘要: {non_chapter_counts['abstract']}字\n"
            if non_chapter_counts.get("keywords", 0) > 0:
                formatted_info += f"  关键词: {non_chapter_counts['keywords']}字\n"
            if non_chapter_counts.get("english_abstract", 0) > 0:
                formatted_info += (
                    f"  英文摘要: {non_chapter_counts['english_abstract']}字\n"
                )
            if non_chapter_counts.get("english_keywords", 0) > 0:
                formatted_info += (
                    f"  英文关键词: {non_chapter_counts['english_keywords']}字\n"
                )
            if non_chapter_counts.get("contents", 0) > 0:
                formatted_info += f"  目录: {non_chapter_counts['contents']}字\n"
            formatted_info += "\n"

        # 添加各章节字数统计
        if chapter_word_counts:
            # formatted_info += "各章节字数统计:\n"
            for chapter, count in chapter_word_counts.items():
                formatted_info += f"  {chapter}: {count}字\n"
        else:
            formatted_info += "未检测到章节字数统计\n"

        return formatted_info

    def split_text(self, text: str) -> List[Document]:
        user_documents = self._parse_structure(text)
        chunks = []

        # 提取章节结构信息
        chapter_structure = self._extract_chapter_structure(user_documents)
        complete_structure_str = self._format_chapter_structure(chapter_structure)
        all_chapter_titles = self._get_all_chapter_titles(chapter_structure)

        # 计算字数统计
        word_count_data = self._calculate_word_counts(user_documents)
        word_count_info = self._format_word_count_info(word_count_data)

        # 生成单个章节结构列表

        chapter_structure_list = []

        chapter_structure_mapping = {}
        for chapter_title in all_chapter_titles:
            single_chapter_str = self._format_single_chapter(
                chapter_structure, chapter_title, word_count_data["chapter_word_counts"]
            )
            chapter_structure_list.append(single_chapter_str)
            chapter_structure_mapping[chapter_title] = single_chapter_str

        for item in user_documents:
            # 构建当前块的元数据
            if item["title"]:
                item["title"] = item["title"].replace(" ", "")
            if item["student_name"]:
                item["student_name"] = item["student_name"].replace(" ", "")
            if item["student_id"]:
                item["student_id"] = item["student_id"].replace(" ", "")
            if item["abstract"]:
                item["abstract"] = item["abstract"].replace("#", "").replace("\n", "")
            if item["english_abstract"]:
                item["english_abstract"] = item["english_abstract"].replace("\n", "")

            current_chapter = item["chapter"]
            current_chapter_structure = chapter_structure_mapping.get(
                current_chapter, ""
            )

            # 将当前块的内容合并为一个字符串
            content = "\n".join(item["content"])

            if item["subsection"]:
                paragraphs = self._split_into_paragraphs(content)

                current_paragraphs = paragraphs
                max_depth = 5
                for _ in range(max_depth):
                    merged_paragraphs = self._merge_paragraphs(current_paragraphs)
                    if len(merged_paragraphs) == len(current_paragraphs):
                        if all(
                            len(p) >= self.merge_threshold for p in merged_paragraphs
                        ):
                            break
                    current_paragraphs = merged_paragraphs
                content_chunks = current_paragraphs

                # 为每个 chunk 单独构建位置映射，确保不重叠
                for chunk in content_chunks:
                    # 为当前 chunk 单独调用 _build_position_mapping
                    pdf_positions = (
                        self._build_position_mapping(chunk)
                        if self.content_blocks
                        else []
                    )

                    metadata = {
                        "title": item["title"],
                        "student_name": item["student_name"],
                        "student_id": item["student_id"],
                        "abstract": item["abstract"],
                        "keywords": item["keywords"],
                        "english_abstract": item["english_abstract"],
                        "english_keywords": item["english_keywords"],
                        "contents": item["contents"],
                        "chapter": item["chapter"],
                        "section": item["section"],
                        "subsection": item["subsection"],
                        "subsubsection": item["subsubsection"],
                        "structure": complete_structure_str,
                        "chapter_structure": current_chapter_structure,
                        "word_count_info": word_count_info,
                        "total_word_count": word_count_data["total_word_count"],
                        "chapter_word_counts": word_count_data["chapter_word_counts"],
                        "total_formula_word_count": word_count_data[
                            "total_formula_word_count"
                        ],
                        "formula_word_ratio": word_count_data["formula_word_ratio"],
                        "pdf_positions": pdf_positions,
                    }
                    chunks.append(Document(page_content=chunk, metadata=metadata))
            elif item["section"]:
                paragraphs = self._split_into_paragraphs(content)

                current_paragraphs = paragraphs

                max_depth = 5
                for _ in range(max_depth):
                    merged_paragraphs = self._merge_paragraphs(current_paragraphs)
                    if len(merged_paragraphs) == len(current_paragraphs):
                        if all(
                            len(p) >= self.merge_threshold for p in merged_paragraphs
                        ):
                            break
                    current_paragraphs = merged_paragraphs
                content_chunks = current_paragraphs

                # 为每个 chunk 单独构建位置映射，确保不重叠
                for chunk in content_chunks:
                    # 为当前 chunk 单独调用 _build_position_mapping
                    pdf_positions = (
                        self._build_position_mapping(chunk)
                        if self.content_blocks
                        else []
                    )

                    metadata = {
                        "title": item["title"],
                        "student_name": item["student_name"],
                        "student_id": item["student_id"],
                        "abstract": item["abstract"],
                        "keywords": item["keywords"],
                        "english_abstract": item["english_abstract"],
                        "english_keywords": item["english_keywords"],
                        "contents": item["contents"],
                        "chapter": item["chapter"],
                        "section": item["section"],
                        "subsection": item["subsection"],
                        "subsubsection": item["subsubsection"],
                        "structure": complete_structure_str,
                        "chapter_structure": current_chapter_structure,
                        "word_count_info": word_count_info,
                        "total_word_count": word_count_data["total_word_count"],
                        "chapter_word_counts": word_count_data["chapter_word_counts"],
                        "total_formula_word_count": word_count_data[
                            "total_formula_word_count"
                        ],
                        "formula_word_ratio": word_count_data["formula_word_ratio"],
                        "pdf_positions": pdf_positions,
                    }
                    chunks.append(Document(page_content=chunk, metadata=metadata))
            else:
                # 对于不分块的情况，单独构建位置映射
                pdf_positions = (
                    self._build_position_mapping(content) if self.content_blocks else []
                )

                metadata = {
                    "title": item["title"],
                    "student_name": item["student_name"],
                    "student_id": item["student_id"],
                    "abstract": item["abstract"],
                    "keywords": item["keywords"],
                    "english_abstract": item["english_abstract"],
                    "english_keywords": item["english_keywords"],
                    "contents": item["contents"],
                    "chapter": item["chapter"],
                    "section": item["section"],
                    "subsection": item["subsection"],
                    "subsubsection": item["subsubsection"],
                    "structure": complete_structure_str,
                    "chapter_structure": current_chapter_structure,
                    "word_count_info": word_count_info,
                    "total_word_count": word_count_data["total_word_count"],
                    "chapter_word_counts": word_count_data["chapter_word_counts"],
                    "total_formula_word_count": word_count_data[
                        "total_formula_word_count"
                    ],
                    "formula_word_ratio": word_count_data["formula_word_ratio"],
                    "pdf_positions": pdf_positions,
                }
                chunks.append(Document(page_content=content, metadata=metadata))
        return chunks

    def split_text_for_aigc(self, text: str) -> List[Document]:
        """
        专门用于AIGC检测的文本分块方法

        使用AIGC专用的合并和分割阈值，生成适合AIGC检测的文本块

        Args:
            text: 要分块的markdown文本

        Returns:
            List[Document]: 分块后的文档列表，每个文档包含pdf_positions元数据
        """
        user_documents = self._parse_structure(text)
        chunks = []

        # 提取章节结构信息
        chapter_structure = self._extract_chapter_structure(user_documents)
        complete_structure_str = self._format_chapter_structure(chapter_structure)
        all_chapter_titles = self._get_all_chapter_titles(chapter_structure)

        # 计算字数统计
        word_count_data = self._calculate_word_counts(user_documents)
        word_count_info = self._format_word_count_info(word_count_data)

        # 生成章节结构映射
        chapter_structure_mapping = {}
        for chapter_title in all_chapter_titles:
            single_chapter_str = self._format_single_chapter(
                chapter_structure, chapter_title, word_count_data["chapter_word_counts"]
            )
            chapter_structure_mapping[chapter_title] = single_chapter_str

        for item in user_documents:
            # 清理元数据
            if item["title"]:
                item["title"] = item["title"].replace(" ", "")
            if item["student_name"]:
                item["student_name"] = item["student_name"].replace(" ", "")
            if item["student_id"]:
                item["student_id"] = item["student_id"].replace(" ", "")
            if item["abstract"]:
                item["abstract"] = item["abstract"].replace("#", "").replace("\n", "")
            if item["english_abstract"]:
                item["english_abstract"] = item["english_abstract"].replace("\n", "")

            current_chapter = item["chapter"]
            current_chapter_structure = chapter_structure_mapping.get(
                current_chapter, ""
            )

            # 将当前块的内容合并为一个字符串
            content = "\n".join(item["content"])

            # 使用AIGC专用阈值进行分块
            if item["subsection"] or item["section"]:
                paragraphs = self._split_into_paragraphs(content)

                current_paragraphs = paragraphs
                max_depth = 5
                for _ in range(max_depth):
                    # 使用AIGC专用阈值
                    merged_paragraphs = self._merge_paragraphs_for_aigc(
                        current_paragraphs
                    )
                    if len(merged_paragraphs) == len(current_paragraphs):
                        if all(
                            len(p) >= self.aigc_merge_threshold
                            for p in merged_paragraphs
                        ):
                            break
                    current_paragraphs = merged_paragraphs
                content_chunks = current_paragraphs

                # 为每个 chunk 单独构建位置映射，确保不重叠
                for chunk in content_chunks:
                    # 为当前 chunk 单独调用 _build_position_mapping
                    pdf_positions = (
                        self._build_position_mapping(chunk)
                        if self.content_blocks
                        else []
                    )

                    metadata = {
                        "title": item["title"],
                        "student_name": item["student_name"],
                        "student_id": item["student_id"],
                        "abstract": item["abstract"],
                        "keywords": item["keywords"],
                        "english_abstract": item["english_abstract"],
                        "english_keywords": item["english_keywords"],
                        "contents": item["contents"],
                        "chapter": item["chapter"],
                        "section": item["section"],
                        "subsection": item["subsection"],
                        "subsubsection": item["subsubsection"],
                        "structure": complete_structure_str,
                        "chapter_structure": current_chapter_structure,
                        "word_count_info": word_count_info,
                        "total_word_count": word_count_data["total_word_count"],
                        "chapter_word_counts": word_count_data["chapter_word_counts"],
                        "total_formula_word_count": word_count_data[
                            "total_formula_word_count"
                        ],
                        "formula_word_ratio": word_count_data["formula_word_ratio"],
                        "pdf_positions": pdf_positions,
                    }
                    chunks.append(Document(page_content=chunk, metadata=metadata))
            else:
                # 对于不分块的情况，单独构建位置映射
                pdf_positions = (
                    self._build_position_mapping(content) if self.content_blocks else []
                )

                metadata = {
                    "title": item["title"],
                    "student_name": item["student_name"],
                    "student_id": item["student_id"],
                    "abstract": item["abstract"],
                    "keywords": item["keywords"],
                    "english_abstract": item["english_abstract"],
                    "english_keywords": item["english_keywords"],
                    "contents": item["contents"],
                    "chapter": item["chapter"],
                    "section": item["section"],
                    "subsection": item["subsection"],
                    "subsubsection": item["subsubsection"],
                    "structure": complete_structure_str,
                    "chapter_structure": current_chapter_structure,
                    "word_count_info": word_count_info,
                    "total_word_count": word_count_data["total_word_count"],
                    "chapter_word_counts": word_count_data["chapter_word_counts"],
                    "total_formula_word_count": word_count_data[
                        "total_formula_word_count"
                    ],
                    "formula_word_ratio": word_count_data["formula_word_ratio"],
                    "pdf_positions": pdf_positions,
                }
                chunks.append(Document(page_content=content, metadata=metadata))

        return chunks

    def _merge_paragraphs_for_aigc(self, paragraphs: List[str]) -> List[str]:
        """
        使用AIGC专用阈值合并小段落

        Args:
            paragraphs: 段落列表

        Returns:
            合并后的段落列表
        """
        merged = []
        current_chunk = ""

        for para in paragraphs:
            if not current_chunk:
                current_chunk = para
                continue

            # 使用AIGC专用合并阈值
            if len(current_chunk) < self.aigc_merge_threshold:
                # 使用AIGC专用分割阈值判断
                if len(current_chunk) + len(para) < self.aigc_split_threshold:
                    current_chunk += "\n\n" + para
                else:
                    merged.append(current_chunk)
                    current_chunk = para
            else:
                merged.append(current_chunk)
                current_chunk = para

        # 处理最后一个块
        if merged and len(current_chunk) < self.aigc_merge_threshold:
            last_chunk = merged[-1]
            if len(last_chunk) + len(current_chunk) <= self.aigc_split_threshold:
                merged[-1] = last_chunk + "\n\n" + current_chunk
            else:
                merged.append(current_chunk)
        else:
            merged.append(current_chunk)

        return merged

    def _remove_toc(self, text: str) -> str:
        """识别并移除目录到第一章之间的内容"""
        toc_pattern = r"(?s)(目录)(.*?)(?=\n\n#+\s*第[一二三四五六七八九十]+章)"
        text = re.sub(toc_pattern, "", text)
        return text

    def _load_content_list(self, content_list_path: str) -> None:
        """加载MinerU解析的content_list.json文件"""
        try:
            with open(content_list_path, "r", encoding="utf-8") as f:
                self.content_list = json.load(f)

            # 提取所有type='text'的块，按顺序存储
            self.content_blocks = [
                block for block in self.content_list if block.get("type") == "text"
            ]
            # 重置已匹配的blocks集合
            self.matched_blocks = set()
            logging.info(
                f"成功加载content_list.json，共{len(self.content_blocks)}个文本块"
            )
        except Exception as e:
            logging.warning(f"加载content_list.json失败: {e}")
            self.content_list = None
            self.content_blocks = []

    def _build_position_mapping(self, markdown_text: str) -> List[Dict]:
        """
        基于 MinerU 的顺序特性，将 Markdown 行与 content_blocks 对齐

        改进策略：
        1. 使用全局 matched_blocks 跟踪，确保每个 block 只被匹配一次
        2. 按 chunk 顺序递增分配 blocks
        3. 每次只匹配未被其他 chunk 匹配过的 blocks

        返回格式: [{"page_idx": 0, "block_idx": 5}, ...]
        """
        if not self.content_blocks:
            return []

        positions = []
        lines = markdown_text.split("\n")

        for line in lines:
            # 只跳过完全空行的行
            if not line.strip():
                continue

            line_normalized = self._normalize_text(line)

            # 如果清理后的行太短，跳过
            if len(line_normalized) < 3:
                continue

            # 在所有未匹配的 blocks 中全局搜索
            matched = False

            for test_idx in range(len(self.content_blocks)):
                # 跳过已经被其他 chunk 匹配过的 blocks
                if test_idx in self.matched_blocks:
                    continue

                block = self.content_blocks[test_idx]
                block_text = block.get("text", "").strip()

                if not block_text:
                    continue

                # 处理多行 block
                if "\n" in block_text:
                    block_lines = block_text.split("\n")
                    for block_line in block_lines:
                        block_line_normalized = self._normalize_text(block_line)
                        # 双向匹配
                        if len(block_line_normalized) >= 3 and (
                            block_line_normalized in line_normalized
                            or line_normalized in block_line_normalized
                        ):
                            positions.append(
                                {"page_idx": block["page_idx"], "block_idx": test_idx}
                            )
                            self.matched_blocks.add(test_idx)  # 标记为已匹配
                            matched = True
                            break
                    if matched:
                        break
                else:
                    # 单行 block
                    block_normalized = self._normalize_text(block_text)
                    # 双向匹配
                    if len(block_normalized) >= 3 and (
                        block_normalized in line_normalized
                        or line_normalized in block_normalized
                    ):
                        positions.append(
                            {"page_idx": block["page_idx"], "block_idx": test_idx}
                        )
                        self.matched_blocks.add(test_idx)  # 标记为已匹配
                        matched = True
                        break

        # 添加调试日志
        logging.info(
            f"位置映射: Markdown {len(lines)} 行, 匹配 {len(positions)} 个 positions, 累计已匹配 {len(self.matched_blocks)} 个 blocks"
        )
        if positions:
            page_indices = sorted(set(p["page_idx"] for p in positions))
            block_indices = [p["block_idx"] for p in positions]
            logging.info(f"本次匹配 blocks: {block_indices}, 覆盖页面: {page_indices}")

        return positions

    def _normalize_text(self, text: str) -> str:
        """标准化文本：去除空格，用于模糊匹配"""
        # 去除所有空格
        text = re.sub(r"\s+", "", text)
        return text.lower()
