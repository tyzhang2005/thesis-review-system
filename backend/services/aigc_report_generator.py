"""
AIGC检测结构化报告生成器 - HTML/CSS + WeasyPrint 版本

使用 HTML 模板和 CSS 样式，通过 WeasyPrint 生成包含环图、进度条等可视化元素的 PDF 报告
"""

import logging
import math
import os
import tempfile
from datetime import datetime
from typing import Dict, List, Optional

from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)


class AIGCReportGenerator:
    """AIGC检测报告生成器 - HTML/CSS + WeasyPrint 版本"""

    # 颜色主题定义（RGB格式，与LaTeX版本一致）
    COLORS = {
        "safegreen": "rgb(76,175,80)",  # <40%
        "warningyellow": "rgb(255,193,7)",  # 40-50%
        "lilac": "rgb(200,150,255)",  # 50-60%
        "warningorange": "rgb(255,152,0)",  # 60-70%
        "riskred": "rgb(220,38,127)",  # >=70%
        "deepblue": "rgb(0,51,102)",  # 标题/表头
        "lightblue": "rgb(102,178,255)",  # 环图人工部分
        "graybg": "rgb(229,229,229)",  # 进度条背景
    }

    def __init__(self, output_dir: Optional[str] = None):
        """
        初始化报告生成器

        Args:
            output_dir: 输出目录，默认为临时目录
        """
        self.output_dir = output_dir or tempfile.gettempdir()

    @staticmethod
    def get_rate_color(rate: float) -> str:
        """
        根据AIGC占比返回颜色名称（与PDF标注一致）

        Args:
            rate: AIGC占比 (0-1)

        Returns:
            str: 颜色名称
        """
        if rate < 0.4:
            return "safegreen"
        elif rate < 0.5:
            return "warningyellow"
        elif rate < 0.6:
            return "lilac"
        elif rate < 0.7:
            return "warningorange"
        else:
            return "riskred"

    @staticmethod
    def get_risk_level(rate: float) -> str:
        """
        根据AIGC占比返回风险等级

        Args:
            rate: AIGC占比 (0-1)

        Returns:
            str: 风险等级
        """
        if rate < 0.4:
            return "低"
        elif rate < 0.7:
            return "中"
        else:
            return "高"

    @staticmethod
    def escape_html(text: str) -> str:
        """
        转义HTML特殊字符

        Args:
            text: 原始文本

        Returns:
            str: 转义后的文本
        """
        if not text:
            return ""
        replacements = {
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&#x27;",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    # 兼容旧方法名
    escape_latex = escape_html

    def _generate_donut_chart(self, aigc_rate: float) -> str:
        """
        生成SVG环图HTML代码

        Args:
            aigc_rate: AIGC占比 (0-1)

        Returns:
            str: SVG HTML代码
        """
        aigc_percent = aigc_rate * 100
        color_name = self.get_rate_color(aigc_rate)
        color_rgb = self.COLORS[color_name]

        # 计算扇形路径
        # SVG坐标系: 中心(80,80), 半径70
        # 从12点方向开始(0,-70相对中心), 顺时针绘制
        cx, cy, r = 80, 80, 70

        if aigc_percent >= 100:
            # 完整圆
            aigc_path = f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{color_rgb}"/>'
        elif aigc_percent <= 0:
            # 只有背景色
            aigc_path = ""
        else:
            # 计算结束点坐标 (从12点方向顺时针)
            angle_rad = math.radians(aigc_percent * 3.6 - 90)  # -90使起点在12点方向
            end_x = cx + r * math.cos(angle_rad)
            end_y = cy + r * math.sin(angle_rad)

            # 判断是否是大弧（超过180度）
            large_arc = 1 if aigc_percent > 50 else 0

            # SVG path: M=移动到中心, L=画线到起点, A=画弧到终点, Z=闭合
            aigc_path = f"""
            <path d="M {cx} {cy} L {cx} {cy - r} A {r} {r} 0 {large_arc} 1 {end_x} {end_y} Z"
                  fill="{color_rgb}" stroke="none"/>"""

        return f"""
        <div class="donut-chart-container">
            <svg width="160" height="160" viewBox="0 0 160 160" xmlns="http://www.w3.org/2000/svg">
                <!-- 背景圆（人工写作部分） -->
                <circle cx="{cx}" cy="{cy}" r="{r}" fill="{self.COLORS['lightblue']}" stroke="none"/>
                <!-- AIGC扇形 -->
                {aigc_path}
                <!-- 内圆遮罩（环状效果） -->
                <circle cx="{cx}" cy="{cy}" r="45" fill="white" stroke="none"/>
            </svg>
            <!-- 中心文字 -->
            <div class="donut-center">
                <div class="donut-percent">{aigc_percent:.1f}%</div>
                <div class="donut-label">AIGC率</div>
            </div>
        </div>"""

    def _generate_progress_bar(self, rate: float) -> str:
        """
        生成HTML/CSS进度条

        Args:
            rate: AIGC占比 (0-1)

        Returns:
            str: HTML进度条代码
        """
        color_name = self.get_rate_color(rate)
        color_rgb = self.COLORS[color_name]
        percent = rate * 100

        return f"""
        <div class="progress-bar-container">
            <div class="progress-bar">
                <div class="progress-fill" style="width: {percent:.1f}%; background-color: {color_rgb};"></div>
            </div>
            <div class="progress-text">{percent:.1f}%</div>
        </div>"""

    def _get_css_styles(self) -> str:
        """
        获取完整的CSS样式

        Returns:
            str: CSS样式字符串
        """
        return """
        <style>
            @page {
                size: A4;
                margin: 2cm 2cm 2.5cm 2cm;
            }

            @page:first {
                margin-top: 2cm;
            }

            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                font-family: "Microsoft YaHei", "PingFang SC", "Source Han Sans CN", "SimSun", sans-serif;
                font-size: 12pt;
                line-height: 1.4;
                color: #333;
            }

            /* 标题栏 */
            .header-line {
                background-color: rgb(0,51,102);
                height: 3pt;
                margin-bottom: 0.5cm;
            }

            .report-title {
                text-align: center;
                color: rgb(0,51,102);
                font-size: 28pt;
                font-weight: bold;
                margin-bottom: 0.3cm;
            }

            .subtitle {
                text-align: center;
                color: gray;
                font-size: 16pt;
                margin-bottom: 0.8cm;
            }

            /* 表格通用样式 */
            .info-table {
                width: 100%;
                max-width: 14cm;
                margin: 0 auto;
                border-collapse: collapse;
                font-size: 11pt;
            }

            .info-table th, .info-table td {
                border: 1px solid #333;
                padding: 6px 10px;
                text-align: left;
                vertical-align: middle;
            }

            .info-table th {
                background-color: rgb(0,51,102);
                color: white;
                font-weight: bold;
                width: 22%;
            }

            .info-table td:nth-child(2) {
                width: 38%;
            }

            .info-table td:nth-child(3) {
                width: 40%;
            }

            /* 环图样式 */
            .donut-chart-container {
                position: relative;
                width: 160px;
                height: 160px;
                margin: 0 auto;
            }

            .donut-center {
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                text-align: center;
            }

            .donut-percent {
                font-size: 22pt;
                font-weight: bold;
                color: #333;
            }

            .donut-label {
                font-size: 10pt;
                color: #666;
                margin-top: 2px;
            }

            /* 风险评估表 */
            .risk-table {
                width: 100%;
                max-width: 14cm;
                margin: 0.3cm auto 0;
                border-collapse: collapse;
                font-size: 11pt;
            }

            .risk-table th, .risk-table td {
                border: 1px solid #333;
                padding: 10px;
                text-align: left;
            }

            .risk-table th {
                background-color: rgb(0,51,102);
                color: white;
                font-weight: bold;
            }

            /* 章节分析页 */
            .section-title {
                text-align: center;
                color: rgb(0,51,102);
                font-size: 18pt;
                font-weight: bold;
                margin-bottom: 0.5cm;
            }

            .chapter-table {
                width: 100%;
                margin: 0 auto;
                border-collapse: collapse;
                font-size: 10pt;
            }

            .chapter-table th, .chapter-table td {
                border: 1px solid #333;
                padding: 8px;
                text-align: center;
                vertical-align: middle;
            }

            .chapter-table th {
                background-color: rgb(0,51,102);
                color: white;
                font-weight: bold;
            }

            .chapter-table td:first-child {
                text-align: left;
                padding-left: 10px;
            }

            /* 进度条样式 */
            .progress-bar-container {
                display: flex;
                align-items: center;
                gap: 8px;
                min-width: 200px;
            }

            .progress-bar {
                width: 180px;
                height: 10px;
                background-color: rgb(229,229,229);
                border-radius: 5px;
                overflow: hidden;
            }

            .progress-fill {
                height: 100%;
                border-radius: 5px;
            }

            .progress-text {
                font-size: 9pt;
                color: #666;
                min-width: 45px;
                text-align: right;
            }

            /* 风险等级颜色 */
            .risk-low { color: rgb(76,175,80); font-weight: bold; }
            .risk-medium { color: rgb(255,152,0); font-weight: bold; }
            .risk-high { color: rgb(220,38,127); font-weight: bold; }

            /* 图例 */
            .legend {
                display: flex;
                justify-content: center;
                flex-wrap: wrap;
                gap: 15px 25px;
                margin-top: 1cm;
                font-size: 9pt;
            }

            .legend-item {
                display: flex;
                align-items: center;
                gap: 5px;
            }

            .legend-box {
                width: 14px;
                height: 14px;
            }

            /* 页脚 */
            .footer {
                position: fixed;
                bottom: 1.5cm;
                left: 0;
                right: 0;
                text-align: center;
                font-size: 9pt;
                color: gray;
            }

            /* 分页控制 */
            .page-break {
                page-break-after: always;
            }

            .no-page-break {
                page-break-inside: avoid;
            }
        </style>
        """

    def _aggregate_chunks_by_chapters(
        self, chunks: List[Dict], chapter_word_counts: Dict
    ) -> List[Dict]:
        """
        将chunk级别的AIGC检测结果按章节聚合

        Args:
            chunks: chunk检测结果列表，每个chunk包含chapter, section, ai_probability, text等
            chapter_word_counts: 章节字数统计字典

        Returns:
            章节级别的AIGC数据列表，格式与chapters检测结果一致
        """
        from collections import defaultdict

        # 按章节分组chunks
        chapter_chunks = defaultdict(list)
        chapter_text_lengths = defaultdict(int)

        for chunk in chunks:
            chapter = chunk.get("chapter", "未知章节")
            chapter_chunks[chapter].append(chunk)
            # chunks 数据结构使用 content_length 而非 text 字段
            chapter_text_lengths[chapter] += chunk.get("content_length", 0)

        # 聚合每章的AIGC率
        aggregated_chapters = []

        for chapter_name, chapter_chunk_list in chapter_chunks.items():
            total_length = chapter_text_lengths[chapter_name]

            # 计算加权平均AIGC率（低于40%的chunk按0计入）
            weighted_ai_sum = 0.0
            for chunk in chapter_chunk_list:
                ai_prob = chunk.get("ai_probability", 0)
                # chunks 数据结构使用 content_length 而非 text 字段
                chunk_len = chunk.get("content_length", 0)
                # 低于40%的按0计入（不加到weighted_ai_sum）
                if ai_prob >= 0.4:
                    weighted_ai_sum += ai_prob * chunk_len

            avg_ai_rate = weighted_ai_sum / total_length if total_length > 0 else 0

            # 从chapter_word_counts获取更准确的字数（如果有的话）
            actual_word_count = chapter_word_counts.get(chapter_name, total_length)

            aggregated_chapters.append(
                {
                    "chapter": chapter_name,
                    "ai_probability": avg_ai_rate,
                    "content_length": actual_word_count,
                }
            )

        return aggregated_chapters

    def _calculate_statistics(self, doc: Document, aigc_results: Dict) -> Dict:
        """
        计算报告统计数据

        Args:
            doc: 文档对象
            aigc_results: AIGC检测结果

        Returns:
            Dict: 统计数据
        """
        total_words = doc.metadata.get("total_word_count", 0)
        overall_ai_rate = aigc_results.get("overall_ai_rate", 0)

        aigc_words = int(total_words * overall_ai_rate)
        human_words = total_words - aigc_words

        return {
            "total_words": total_words,
            "aigc_words": aigc_words,
            "human_words": human_words,
            "overall_ai_rate": overall_ai_rate,
        }

    def _get_risk_description(self, rate: float) -> str:
        """获取风险描述"""
        if rate < 0.4:
            return "论文整体AIGC含量较低，以人工创作为主。"
        elif rate < 0.7:
            return "论文中存在部分AIGC生成内容，建议对高风险章节进行人工复核。"
        else:
            return "论文AIGC含量较高，建议进行全面人工复核和修改。"

    def _write_header_section(
        self, file_handle, doc: Document, aigc_results: Dict, stats: Dict
    ) -> None:
        """
        写入标题和基本信息表（含环图）

        Args:
            file_handle: 文件句柄
            doc: 文档对象
            aigc_results: AIGC检测结果
            stats: 统计数据
        """
        title = self.escape_html(doc.metadata.get("title", "N/A"))
        student_name = self.escape_html(doc.metadata.get("student_name", "N/A"))
        student_id = self.escape_html(doc.metadata.get("student_id", "N/A"))
        aigc_rate = aigc_results.get("overall_ai_rate", 0)
        donut_chart = self._generate_donut_chart(aigc_rate)

        # 计算检测日期
        detect_date = datetime.now().strftime("%Y年%m月%d日")
        aigc_percent = aigc_rate * 100

        header = f"""
        <!-- ========== 封面页 ========== -->
        <div class="header-line"></div>

        <h1 class="report-title">AIGC检测结构化报告</h1>
        <p class="subtitle">睿文智评AI预审评估系统</p>

        <!-- 项目信息表 -->
        <table class="info-table">
            <tr>
                <th style="width: 22%;">项目</th>
                <th style="width: 38%;">信息</th>
                <th style="width: 40%;">AIGC率</th>
            </tr>
            <tr>
                <td>论文标题</td>
                <td>{title}</td>
                <td rowspan="4" style="background-color: white; text-align: center; vertical-align: middle;">
                    {donut_chart}
                </td>
            </tr>
            <tr>
                <td>学生姓名</td>
                <td>{student_name}</td>
            </tr>
            <tr>
                <td>学号</td>
                <td>{student_id}</td>
            </tr>
            <tr>
                <td>检测日期</td>
                <td>{detect_date}</td>
            </tr>
        </table>

        <!-- 统计数据表 -->
        <table class="info-table" style="margin-top: 0.3cm; table-layout: fixed;">
            <colgroup>
                <col style="width: 25%;">
                <col style="width: 25%;">
                <col style="width: 25%;">
                <col style="width: 25%;">
            </colgroup>
            <tr>
                <th colspan="4">统计项目</th>
            </tr>
            <tr>
                <td style="background-color: white; text-align: center;">总字数</td>
                <td style="background-color: white; text-align: center;">疑似字数</td>
                <td style="background-color: white; text-align: center;">人工写作字数</td>
                <td style="background-color: white; text-align: center;">总体AIGC率</td>
            </tr>
            <tr>
                <td style="text-align: center;">{stats['total_words']:,}</td>
                <td style="text-align: center;">{stats['aigc_words']:,}</td>
                <td style="text-align: center;">{stats['human_words']:,}</td>
                <td style="text-align: center;">{aigc_percent:.1f}%</td>
            </tr>
        </table>
        """
        file_handle.write(header)

    def _write_risk_assessment(self, file_handle, aigc_results: Dict) -> None:
        """
        写入风险评估部分

        Args:
            file_handle: 文件句柄
            aigc_results: AIGC检测结果
        """
        aigc_rate = aigc_results.get("overall_ai_rate", 0)
        risk_level = self.get_risk_level(aigc_rate)
        risk_description = self.escape_html(self._get_risk_description(aigc_rate))

        risk_section = f"""
        <!-- 风险评估 -->
        <table class="risk-table">
            <tr>
                <th colspan="2">风险评估</th>
            </tr>
            <tr>
                <td style="width: 120px;"><strong>风险等级：</strong></td>
                <td>{risk_level}</td>
            </tr>
            <tr>
                <td><strong>评估说明：</strong></td>
                <td>{risk_description}</td>
            </tr>
        </table>
        """
        file_handle.write(risk_section)

    def _write_chapter_analysis(
        self, file_handle, chapters_data: List[Dict], chapter_word_counts: Dict
    ) -> None:
        """
        写入章节详细分析HTML

        Args:
            file_handle: 文件句柄
            chapters_data: 章节AIGC数据
            chapter_word_counts: 章节字数统计
        """
        file_handle.write("""
        <!-- ========== 章节详细分析 ========== -->
        <table class="chapter-table" style="margin-top: 0.5cm;">
        """)

        # 蓝色表头行
        file_handle.write("""
            <tr>
                <th colspan="5" style="text-align: left; padding-left: 10px;">章节详细分析</th>
            </tr>
        """)

        # 列标题行（白色背景）
        file_handle.write("""
            <tr style="background-color: white;">
                <th style="width: 28%; background-color: white; color: #333; text-align: left; padding-left: 10px;">章节名称</th>
                <th style="width: 15%; background-color: white; color: #333;">总字数</th>
                <th style="width: 15%; background-color: white; color: #333;">疑似字数</th>
                <th style="width: 27%; background-color: white; color: #333;">AIGC占比</th>
                <th style="width: 15%; background-color: white; color: #333;">风险等级</th>
            </tr>
        """)

        # 写入每章数据
        for chapter in chapters_data:
            chapter_name = self.escape_html(chapter.get("chapter", "N/A"))
            total_words = chapter.get("content_length", 0)
            # 从字数统计获取实际字数
            actual_words = chapter_word_counts.get(
                chapter.get("chapter", ""), total_words
            )
            ai_rate = chapter.get("ai_probability", 0)
            aigc_words = int(actual_words * ai_rate)
            progress_bar = self._generate_progress_bar(ai_rate)
            risk_level = self.get_risk_level(ai_rate)

            # 风险等级CSS类
            if ai_rate < 0.4:
                risk_class = "risk-low"
            elif ai_rate < 0.7:
                risk_class = "risk-medium"
            else:
                risk_class = "risk-high"

            row = f"""
                <tr style="background-color: white;">
                    <td style="text-align: left; padding-left: 10px;">{chapter_name}</td>
                    <td>{actual_words:,}</td>
                    <td>{aigc_words:,}</td>
                    <td>{progress_bar}</td>
                    <td class="{risk_class}">{risk_level}</td>
                </tr>
            """
            file_handle.write(row)

        # 图例行
        file_handle.write("""
            <tr style="background-color: white;">
                <td colspan="5" style="padding: 10px 0;">
                    <div style="display: flex; justify-content: center; flex-wrap: wrap; gap: 15px 25px; font-size: 9pt;">
                        <div style="display: flex; align-items: center; gap: 5px;">
                            <div style="width: 14px; height: 14px; background-color: rgb(76,175,80);"></div>
                            <span>低风险 (&lt;40%)</span>
                        </div>
                        <div style="display: flex; align-items: center; gap: 5px;">
                            <div style="width: 14px; height: 14px; background-color: rgb(255,193,7);"></div>
                            <span>较低风险 (40-50%)</span>
                        </div>
                        <div style="display: flex; align-items: center; gap: 5px;">
                            <div style="width: 14px; height: 14px; background-color: rgb(200,150,255);"></div>
                            <span>中等风险 (50-60%)</span>
                        </div>
                        <div style="display: flex; align-items: center; gap: 5px;">
                            <div style="width: 14px; height: 14px; background-color: rgb(255,152,0);"></div>
                            <span>较高风险 (60-70%)</span>
                        </div>
                        <div style="display: flex; align-items: center; gap: 5px;">
                            <div style="width: 14px; height: 14px; background-color: rgb(220,38,127);"></div>
                            <span>高风险 (&ge;70%)</span>
                        </div>
                    </div>
                </td>
            </tr>
        </table>
        """)

    def generate_report(
        self, doc: Document, aigc_results: Dict, output_file: str
    ) -> str:
        """
        生成AIGC检测结构化报告PDF

        Args:
            doc: 包含元数据的LangChain Document对象
            aigc_results: AIGC检测结果字典
            output_file: 输出HTML文件路径

        Returns:
            str: 生成的PDF文件路径

        Raises:
            Exception: WeasyPrint编译失败
        """
        logging.info(f"开始生成AIGC检测报告: {output_file}")

        # 计算统计数据
        stats = self._calculate_statistics(doc, aigc_results)

        # 获取章节数据 - 支持chunks和chapters两种格式
        chapters_data = aigc_results.get("chapters", [])
        chapter_word_counts = doc.metadata.get("chapter_word_counts", {})

        # 如果没有chapters数据但有chunks数据，则聚合chunks
        if not chapters_data and "chunks" in aigc_results:
            logging.info("检测到chunks格式数据，正在按章节聚合...")
            chapters_data = self._aggregate_chunks_by_chapters(
                aigc_results["chunks"], chapter_word_counts
            )
            logging.info(f"聚合完成，共 {len(chapters_data)} 个章节")

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # 写入HTML文件
        with open(output_file, "w", encoding="utf-8") as f:
            # 写入HTML头部
            f.write("<!DOCTYPE html>\n")
            f.write('<html lang="zh-CN">\n')
            f.write("<head>\n")
            f.write('<meta charset="UTF-8">\n')
            f.write("<title>AIGC检测报告</title>\n")
            f.write(self._get_css_styles())
            f.write("</head>\n")
            f.write("<body>\n")

            # 写入标题和基本信息表
            self._write_header_section(f, doc, aigc_results, stats)

            # 写入章节详细分析
            self._write_chapter_analysis(f, chapters_data, chapter_word_counts)

            # 写入风险评估
            self._write_risk_assessment(f, aigc_results)

            # 关闭HTML
            f.write("</body>\n")
            f.write("</html>\n")

        logging.info(f"HTML文件生成完成: {output_file}")

        # 编译为PDF
        pdf_path = self._compile_html_to_pdf(output_file)

        logging.info(f"AIGC检测报告生成完成: {pdf_path}")
        return pdf_path

    def _compile_html_to_pdf(self, html_file: str) -> str:
        """
        使用WeasyPrint编译HTML文件为PDF

        Args:
            html_file: HTML文件路径

        Returns:
            str: 生成的PDF文件路径

        Raises:
            Exception: WeasyPrint编译失败
        """
        try:
            from weasyprint import CSS, HTML
        except ImportError:
            raise ImportError(
                "WeasyPrint未安装，请运行: pip install weasyprint\n"
                "Windows额外需要: pip install weasyprint[cairo]"
            )

        output_dir = os.path.dirname(html_file)
        base_name = os.path.splitext(os.path.basename(html_file))[0]
        pdf_path = os.path.join(output_dir, f"{base_name}.pdf")

        try:
            # 读取 HTML 文件
            with open(html_file, "r", encoding="utf-8") as f:
                html_content = f.read()

            # 使用 WeasyPrint 生成 PDF
            HTML(string=html_content, base_url=str(output_dir)).write_pdf(pdf_path)

            # 清理临时 HTML 文件
            if os.path.exists(html_file):
                os.remove(html_file)

            if not os.path.exists(pdf_path):
                raise FileNotFoundError("PDF文件未生成")

            return pdf_path

        except Exception as e:
            logging.error(f"WeasyPrint编译失败: {e}")
            # 保留HTML文件用于调试
            logging.info(f"HTML文件已保留用于调试: {html_file}")
            raise


# 便捷函数
def generate_aigc_report(doc: Document, aigc_results: Dict, output_file: str) -> str:
    """
    生成AIGC检测结构化报告PDF

    Args:
        doc: 包含元数据的LangChain Document对象
        aigc_results: AIGC检测结果字典
        output_file: 输出HTML文件路径

    Returns:
        str: 生成的PDF文件路径
    """
    generator = AIGCReportGenerator()
    return generator.generate_report(doc, aigc_results, output_file)
