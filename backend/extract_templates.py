#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
提取提示词模板到Markdown文件
"""
import re
import os
from pathlib import Path

def extract_template_to_md(template_file: str, output_dir: Path):
    """从Python文件提取模板并保存为Markdown"""

    # 读取源文件
    with open(template_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 提取所有模板
    pattern = r'(\w+_template)\s*=\s*"""([\s\S]*?)"""'
    matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)

    print(f"从 {template_file} 提取到 {len(matches)} 个模板")

    for template_name, template_body in matches:
        # 确定输出文件名
        if 'theory_' in template_name:
            # 理论型模板
            if 'introduction' in template_name:
                if 'related_work' in template_name:
                    md_file = output_dir / "step4_theory" / "introduction_combined.md"
                else:
                    md_file = output_dir / "step4_theory" / "introduction.md"
            elif 'related_work' in template_name:
                md_file = output_dir / "step4_theory" / "related_work.md"
            elif 'background' in template_name:
                md_file = output_dir / "step4_theory" / "background.md"
            elif 'data_processing' in template_name:
                md_file = output_dir / "step4_theory" / "data_processing.md"
            elif 'methodology' in template_name:
                md_file = output_dir / "step4_theory" / "methodology.md"
            elif 'experiment' in template_name:
                if 'result' in template_name:
                    md_file = output_dir / "step4_theory" / "experiment_combined.md"
                else:
                    md_file = output_dir / "step4_theory" / "experiment.md"
            elif 'result_analysis' in template_name:
                md_file = output_dir / "step4_theory" / "result_analysis.md"
            elif 'conclusion' in template_name:
                md_file = output_dir / "step4_theory" / "conclusion.md"
            else:
                continue
        elif 'method_' in template_name:
            # 方法创新型模板 - 类似处理
            if 'introduction' in template_name:
                if 'related_work' in template_name:
                    md_file = output_dir / "step4_method" / "introduction_combined.md"
                else:
                    md_file = output_dir / "step4_method" / "introduction.md"
            elif 'related_work' in template_name:
                md_file = output_dir / "step4_method" / "related_work.md"
            elif 'background' in template_name:
                md_file = output_dir / "step4_method" / "background.md"
            elif 'data_processing' in template_name:
                md_file = output_dir / "step4_method" / "data_processing.md"
            elif 'methodology' in template_name:
                md_file = output_dir / "step4_method" / "methodology.md"
            elif 'experiment' in template_name:
                if 'result' in template_name:
                    md_file = output_dir / "step4_method" / "experiment_combined.md"
                else:
                    md_file = output_dir / "step4_method" / "experiment.md"
            elif 'result_analysis' in template_name:
                md_file = output_dir / "step4_method" / "result_analysis.md"
            elif 'conclusion' in template_name:
                md_file = output_dir / "step4_method" / "conclusion.md"
            else:
                continue
        elif 'engineering_' in template_name:
            # 工程实现型模板 - 类似处理
            if 'introduction' in template_name:
                if 'related_work' in template_name:
                    md_file = output_dir / "step4_engineering" / "introduction_combined.md"
                else:
                    md_file = output_dir / "step4_engineering" / "introduction.md"
            elif 'related_work' in template_name:
                md_file = output_dir / "step4_engineering" / "related_work.md"
            elif 'background' in template_name:
                md_file = output_dir / "step4_engineering" / "background.md"
            elif 'data_processing' in template_name:
                md_file = output_dir / "step4_engineering" / "data_processing.md"
            elif 'methodology' in template_name:
                md_file = output_dir / "step4_engineering" / "system_design.md"
            elif 'experiment' in template_name:
                if 'result' in template_name:
                    md_file = output_dir / "step4_engineering" / "implementation_combined.md"
                else:
                    md_file = output_dir / "step4_engineering" / "implementation.md"
            elif 'result_analysis' in template_name:
                md_file = output_dir / "step4_engineering" / "evaluation.md"
            elif 'conclusion' in template_name:
                md_file = output_dir / "step4_engineering" / "conclusion.md"
            else:
                continue
        else:
            continue

        # 保存为Markdown
        md_file.parent.mkdir(parents=True, exist_ok=True)
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(template_body.strip())

        print(f"  保存: {md_file.name} ({len(template_body)} 字符)")

def extract_evaluation_templates(base_dir: Path):
    """提取evaluation.py中的硬编码提示词"""

    eval_file = base_dir / "routers" / "evaluation.py"
    if not eval_file.exists():
        print(f"文件不存在: {eval_file}")
        return

    with open(eval_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 提取步骤6的汇总建议模板
    pattern = r'summary_advice_template\s*=\s*"""([\s\S]*?)"""'
    matches = re.findall(pattern, content, re.DOTALL)

    if matches:
        with open(base_dir / "prompts" / "step6_summary_advice.md", 'w', encoding='utf-8') as f:
            f.write(matches[0].strip())
        print(f"提取 step6_summary_advice ({len(matches[0])} 字符)")

    # 提取步骤7的综合评分模板
    pattern = r'summary_prompt_template\s*=\s*"""([\s\S]*?)"""'
    matches = re.findall(pattern, content, re.DOTALL)

    if matches:
        with open(base_dir / "prompts" / "step7_comprehensive_scoring.md", 'w', encoding='utf-8') as f:
            f.write(matches[0].strip())
        print(f"提取 step7_comprehensive_scoring ({len(matches[0])} 字符)")

def extract_workload_templates(base_dir: Path):
    """提取template_summary.py中的工作量评估提示词"""

    summary_file = base_dir / "templates" / "template_summary.py"
    if not summary_file.exists():
        print(f"文件不存在: {summary_file}")
        return

    with open(summary_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 提取三个工作量评估模板
    pattern = r'def get_(theory|method|engineering)_workload_template\(\):.*?return """([\s\S]*?)"""'
    matches = re.findall(pattern, content, re.DOTALL)

    for paper_type, template_body in matches:
        output_file = base_dir / "prompts" / f"step5_workload_{paper_type}.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(template_body.strip())
        print(f"提取 step5_workload_{paper_type} ({len(template_body)} 字符)")

if __name__ == "__main__":
    base_dir = Path("C:/Users/24478/Desktop/backend")
    prompts_dir = base_dir / "prompts"

    print("=" * 60)
    print("提取提示词模板到Markdown文件")
    print("=" * 60)

    print("\n### 提取章节评估模板 ###")
    for template_file in ['templates/template_theory.py', 'templates/template_method.py', 'templates/template_engineering.py']:
        template_path = base_dir / template_file
        if template_path.exists():
            extract_template_to_md(str(template_path), prompts_dir)

    print("\n### 提取evaluation.py中的提示词 ###")
    extract_evaluation_templates(base_dir)

    print("\n### 提取template_summary.py中的工作量评估提示词 ###")
    extract_workload_templates(base_dir)

    print("\n完成!")
