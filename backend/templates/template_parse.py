import json
import logging
import os
import re
import time
from typing import Any, Dict, List


def parse_theory_chapter_structure(
    response_text: str, chapter_name: str
) -> Dict[str, Any]:
    """解析理论研究型论文的章节结构体"""
    try:
        # 清理响应：移除 Markdown 代码块标记
        cleaned_response = re.sub(
            r"^```json\s*|\s*```$", "", response_text.strip(), flags=re.MULTILINE
        )

        # 解析JSON响应
        result_data = json.loads(cleaned_response)
        chapter_data = result_data.get("chapter", {})

        # 构建解析结果
        parsed_structure = {
            "chapter_name": chapter_data.get("chapter_name", chapter_name),
            "chapter_type": chapter_data.get("chapter_type", "unknown"),
            "chapter_summary": chapter_data.get("chapter_summary", ""),
            "chapter_remark": chapter_data.get("chapter_remark", ""),
            "section_structure": [],
            "extracted_info": {},
            "evaluation_items": {},
            "scoring_impact": chapter_data.get("scoring_impact", ""),
            "advice": [],
        }

        # 解析章节结构
        section_structure = chapter_data.get("section_structure", [])
        for section in section_structure:
            parsed_section = {
                "section_title": section.get("section_title", ""),
                "section_purpose": section.get("section_purpose", ""),
                "key_points": section.get("key_points", []),
                "weaknesses": section.get("weaknesses", []),
            }
            parsed_structure["section_structure"].append(parsed_section)

        # 解析提取信息
        extracted_info = chapter_data.get("extracted_info", {})
        if extracted_info:
            for field_name, field_value in extracted_info.items():
                if isinstance(field_value, str) and field_value.strip():
                    parsed_structure["extracted_info"][field_name] = field_value
                else:
                    logging.debug(
                        f"跳过非字符串或空提取信息字段: {field_name} = {field_value}"
                    )
        else:
            logging.info(f"章节 {chapter_name} 的 extracted_info 为空或不存在")

        # 解析评估项目
        evaluation_items = chapter_data.get("evaluation_items", {})
        if evaluation_items:
            for field_name, field_value in evaluation_items.items():
                if isinstance(field_value, str) and field_value.strip():
                    # 过滤掉默认的占位符文本
                    if "用一句话基于证据评价" not in field_value:
                        parsed_structure["evaluation_items"][field_name] = field_value
                    else:
                        logging.debug(f"跳过占位符字段: {field_name}")
                else:
                    logging.debug(f"跳过非字符串或空字段: {field_name} = {field_value}")
        else:
            logging.warning(f"章节 {chapter_name} 的 evaluation_items 为空或不存在")

        # 解析建议信息
        advice_data = result_data.get("advice", [])
        if advice_data:
            for advice_item in advice_data:
                parsed_advice = {
                    "position": advice_item.get("position", ""),
                    "suggestion": advice_item.get("suggestion", ""),
                }
                parsed_structure["advice"].append(parsed_advice)
        else:
            logging.info(f"章节 {chapter_name} 的 advice 为空或不存在")

        # 记录解析结果统计
        eval_count = len(parsed_structure["evaluation_items"])
        advice_count = len(parsed_structure["advice"])
        extracted_count = len(parsed_structure["extracted_info"])
        logging.info(
            f"理论研究型章节 {chapter_name} 解析完成: {extracted_count} 个提取信息, {eval_count} 个评估项, {advice_count} 条建议"
        )

        return parsed_structure

    except json.JSONDecodeError as e:
        logging.error(f"理论研究型章节JSON解析失败: {e}")
        return {"error": f"JSON解析失败: {str(e)}", "raw_response": response_text}
    except Exception as e:
        logging.error(f"解析理论研究型章节时发生错误: {e}")
        return {"error": f"解析失败: {str(e)}", "raw_response": response_text}


def parse_method_chapter_structure(
    response_text: str, chapter_name: str
) -> Dict[str, Any]:
    """解析方法创新型论文的章节结构体"""
    try:
        # 清理响应：移除 Markdown 代码块标记
        cleaned_response = re.sub(
            r"^```json\s*|\s*```$", "", response_text.strip(), flags=re.MULTILINE
        )

        # 解析JSON响应
        result_data = json.loads(cleaned_response)
        chapter_data = result_data.get("chapter", {})

        # 构建解析结果
        parsed_structure = {
            "chapter_name": chapter_data.get("chapter_name", chapter_name),
            "chapter_type": chapter_data.get("chapter_type", "unknown"),
            "chapter_summary": chapter_data.get("chapter_summary", ""),
            "chapter_remark": chapter_data.get("chapter_remark", ""),
            "section_structure": [],
            "extracted_info": {},
            "evaluation_items": {},
            "scoring_impact": chapter_data.get("scoring_impact", ""),
            "advice": [],
        }

        # 解析章节结构
        section_structure = chapter_data.get("section_structure", [])
        for section in section_structure:
            parsed_section = {
                "section_title": section.get("section_title", ""),
                "section_purpose": section.get("section_purpose", ""),
                "key_points": section.get("key_points", []),
                "weaknesses": section.get("weaknesses", []),
            }
            parsed_structure["section_structure"].append(parsed_section)

        # 解析提取信息
        extracted_info = chapter_data.get("extracted_info", {})
        if extracted_info:
            for field_name, field_value in extracted_info.items():
                if isinstance(field_value, str) and field_value.strip():
                    parsed_structure["extracted_info"][field_name] = field_value
                else:
                    logging.debug(
                        f"跳过非字符串或空提取信息字段: {field_name} = {field_value}"
                    )
        else:
            logging.info(f"章节 {chapter_name} 的 extracted_info 为空或不存在")

        # 解析评估项目
        evaluation_items = chapter_data.get("evaluation_items", {})
        if evaluation_items:
            for field_name, field_value in evaluation_items.items():
                if isinstance(field_value, str) and field_value.strip():
                    # 过滤掉默认的占位符文本
                    if "用一句话基于证据评价" not in field_value:
                        parsed_structure["evaluation_items"][field_name] = field_value
                    else:
                        logging.debug(f"跳过占位符字段: {field_name}")
                else:
                    logging.debug(f"跳过非字符串或空字段: {field_name} = {field_value}")
        else:
            logging.warning(f"章节 {chapter_name} 的 evaluation_items 为空或不存在")

        # 解析建议信息
        advice_data = result_data.get("advice", [])
        if advice_data:
            for advice_item in advice_data:
                parsed_advice = {
                    "position": advice_item.get("position", ""),
                    "suggestion": advice_item.get("suggestion", ""),
                }
                parsed_structure["advice"].append(parsed_advice)
        else:
            logging.info(f"章节 {chapter_name} 的 advice 为空或不存在")

        # 记录解析结果统计
        eval_count = len(parsed_structure["evaluation_items"])
        advice_count = len(parsed_structure["advice"])
        extracted_count = len(parsed_structure["extracted_info"])
        logging.info(
            f"方法创新型章节 {chapter_name} 解析完成: {extracted_count} 个提取信息, {eval_count} 个评估项, {advice_count} 条建议"
        )

        return parsed_structure

    except json.JSONDecodeError as e:
        logging.error(f"方法创新型章节JSON解析失败: {e}")
        return {"error": f"JSON解析失败: {str(e)}", "raw_response": response_text}
    except Exception as e:
        logging.error(f"解析方法创新型章节时发生错误: {e}")
        return {"error": f"解析失败: {str(e)}", "raw_response": response_text}


def parse_engineering_chapter_structure(
    response_text: str, chapter_name: str
) -> Dict[str, Any]:
    """解析工程实现型论文的章节结构体"""
    try:
        # 清理响应：移除 Markdown 代码块标记
        cleaned_response = re.sub(
            r"^```json\s*|\s*```$", "", response_text.strip(), flags=re.MULTILINE
        )

        # 解析JSON响应
        result_data = json.loads(cleaned_response)
        chapter_data = result_data.get("chapter", {})

        # 构建解析结果
        parsed_structure = {
            "chapter_name": chapter_data.get("chapter_name", chapter_name),
            "chapter_type": chapter_data.get("chapter_type", "unknown"),
            "chapter_summary": chapter_data.get("chapter_summary", ""),
            "chapter_remark": chapter_data.get("chapter_remark", ""),
            "section_structure": [],
            "extracted_info": {},
            "evaluation_items": {},
            "scoring_impact": chapter_data.get("scoring_impact", ""),
            "advice": [],
        }

        # 解析章节结构
        section_structure = chapter_data.get("section_structure", [])
        for section in section_structure:
            parsed_section = {
                "section_title": section.get("section_title", ""),
                "section_purpose": section.get("section_purpose", ""),
                "key_points": section.get("key_points", []),
                "weaknesses": section.get("weaknesses", []),
            }
            parsed_structure["section_structure"].append(parsed_section)

        # 解析提取信息
        extracted_info = chapter_data.get("extracted_info", {})
        if extracted_info:
            for field_name, field_value in extracted_info.items():
                if isinstance(field_value, str) and field_value.strip():
                    parsed_structure["extracted_info"][field_name] = field_value
                else:
                    logging.debug(
                        f"跳过非字符串或空提取信息字段: {field_name} = {field_value}"
                    )
        else:
            logging.info(f"章节 {chapter_name} 的 extracted_info 为空或不存在")

        # 解析评估项目
        evaluation_items = chapter_data.get("evaluation_items", {})
        if evaluation_items:
            for field_name, field_value in evaluation_items.items():
                if isinstance(field_value, str) and field_value.strip():
                    # 过滤掉默认的占位符文本
                    if "用一句话基于证据评价" not in field_value:
                        parsed_structure["evaluation_items"][field_name] = field_value
                    else:
                        logging.debug(f"跳过占位符字段: {field_name}")
                else:
                    logging.debug(f"跳过非字符串或空字段: {field_name} = {field_value}")
        else:
            logging.warning(f"章节 {chapter_name} 的 evaluation_items 为空或不存在")

        # 解析建议信息
        advice_data = result_data.get("advice", [])
        if advice_data:
            for advice_item in advice_data:
                parsed_advice = {
                    "position": advice_item.get("position", ""),
                    "suggestion": advice_item.get("suggestion", ""),
                }
                parsed_structure["advice"].append(parsed_advice)
        else:
            logging.info(f"章节 {chapter_name} 的 advice 为空或不存在")

        # 记录解析结果统计
        eval_count = len(parsed_structure["evaluation_items"])
        advice_count = len(parsed_structure["advice"])
        extracted_count = len(parsed_structure["extracted_info"])
        logging.info(
            f"工程实现型章节 {chapter_name} 解析完成: {extracted_count} 个提取信息, {eval_count} 个评估项, {advice_count} 条建议"
        )

        return parsed_structure

    except json.JSONDecodeError as e:
        logging.error(f"工程实现型章节JSON解析失败: {e}")
        return {"error": f"JSON解析失败: {str(e)}", "raw_response": response_text}
    except Exception as e:
        logging.error(f"解析工程实现型章节时发生错误: {e}")
        return {"error": f"解析失败: {str(e)}", "raw_response": response_text}


def analyze_evaluation_items(chapter_evaluation: Dict) -> Dict[str, Any]:
    """分析专项评估项中的标签数量"""
    try:
        # 定义标签类型
        label_types = ["[无问题]", "[轻微]", "[中度]", "[严重]"]
        label_counts = {label: 0 for label in label_types}
        total_evaluation_items = 0

        # 按章节顺序处理
        chapter_keys = sorted(
            [k for k in chapter_evaluation.keys() if k.startswith("chapter_")]
        )

        # 统计每个章节的评估项
        for chapter_key in chapter_keys:
            chapter_data = chapter_evaluation[chapter_key].get("chapter_data", {})
            evaluation_items = chapter_data.get("evaluation_items", {})

            for field_name, evaluation_text in evaluation_items.items():
                if isinstance(evaluation_text, str) and evaluation_text.strip():
                    total_evaluation_items += 1

                    # 检查评估文本中是否包含标签
                    for label in label_types:
                        if label in evaluation_text:
                            label_counts[label] += 1
                            break  # 每个评估项只统计一个标签

        # 计算百分比
        label_percentages = {}
        for label, count in label_counts.items():
            if total_evaluation_items > 0:
                percentage = (count / total_evaluation_items) * 100
                label_percentages[label] = round(percentage, 2)
            else:
                label_percentages[label] = 0.0

        analysis_result = {
            "total_evaluation_items": total_evaluation_items,
            "label_counts": label_counts,
            "label_percentages": label_percentages,
            "analysis_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        logging.info(
            f"专项评估项分析完成: 共{total_evaluation_items}个评估项, 标签分布: {label_counts}"
        )

        return analysis_result

    except Exception as e:
        logging.error(f"专项评估项分析失败: {str(e)}")
        return {
            "error": f"分析失败: {str(e)}",
            "total_evaluation_items": 0,
            "label_counts": {},
            "label_percentages": {},
        }


def save_evaluation_analysis(evaluation_dir: str, analysis_result: Dict):
    """保存专项评估项分析结果到文件"""
    try:
        analysis_file_path = os.path.join(evaluation_dir, "evaitem_analysis.txt")

        with open(analysis_file_path, "w", encoding="utf-8") as f:
            f.write("📊 专项评估项分析报告\n")
            f.write("=" * 60 + "\n")
            f.write(f"生成时间: {analysis_result.get('analysis_time', '未知')}\n")
            f.write(
                f"总评估项数量: {analysis_result.get('total_evaluation_items', 0)}\n"
            )
            f.write("=" * 60 + "\n\n")

            # 写入标签统计
            f.write("🏷️ 标签分布统计\n")
            f.write("-" * 40 + "\n")

            label_counts = analysis_result.get("label_counts", {})
            label_percentages = analysis_result.get("label_percentages", {})

            if label_counts:
                for label in ["[无问题]", "[轻微]", "[中度]", "[严重]"]:
                    count = label_counts.get(label, 0)
                    percentage = label_percentages.get(label, 0.0)
                    f.write(f"{label}: {count} 项 ({percentage}%)\n")

                # 计算问题项比例（排除[无问题]）
                total_problem_items = sum(
                    count
                    for label, count in label_counts.items()
                    if label != "[无问题]"
                )
                total_items = analysis_result.get("total_evaluation_items", 0)

                if total_items > 0:
                    problem_percentage = (total_problem_items / total_items) * 100
                    f.write(f"\n📈 问题项总体统计:\n")
                    f.write(f"问题评估项总数: {total_problem_items} 项\n")
                    f.write(f"问题项占比: {round(problem_percentage, 2)}%\n")

                    # 严重程度分析
                    if total_problem_items > 0:
                        f.write(f"\n⚠️ 问题严重程度分析:\n")
                        for label in ["[轻微]", "[中度]", "[严重]"]:
                            count = label_counts.get(label, 0)
                            if total_problem_items > 0:
                                severity_percentage = (
                                    count / total_problem_items
                                ) * 100
                                f.write(
                                    f"{label}: {count} 项 ({round(severity_percentage, 2)}%)\n"
                                )
            else:
                f.write("无标签统计数据\n")

            f.write("\n" + "=" * 60 + "\n")
            f.write("📝 分析说明:\n")
            f.write("- [无问题]: 该项内容符合要求，无需修改\n")
            f.write("- [轻微]: 存在小问题，建议优化但不影响整体质量\n")
            f.write("- [中度]: 存在明显问题，需要修改以提升质量\n")
            f.write("- [严重]: 存在重大问题，必须修改否则影响论文通过\n")

        logging.info(f"专项评估项分析结果已写入: {analysis_file_path}")
        return analysis_file_path

    except Exception as e:
        logging.error(f"保存专项评估项分析失败: {str(e)}")
        return None


def get_parser_by_paper_type(paper_type: str):
    """根据论文类型获取对应的解析函数"""
    parser_map = {
        "理论研究": parse_theory_chapter_structure,
        "方法创新": parse_method_chapter_structure,
        "工程实现": parse_engineering_chapter_structure,
    }
    return parser_map.get(paper_type, parse_theory_chapter_structure)


def get_complete_field_descriptions():
    """返回完整的字段描述映射"""
    return {
        # 理论研究型字段
        "research_background": "理论背景阐述质量",
        "problem_statement": "理论问题表述的严格性",
        "research_significance": "理论意义的价值",
        "thesis_statement": "核心理论论点的清晰度",
        "motivation_clarity": "理论动机的严谨性",
        "scope_delimitation": "理论研究范围的界定",
        "theoretical_taxonomy": "理论流派的分类和梳理逻辑",
        "method_comparison": "不同理论方法的对比分析",
        "research_gap_analysis": "研究空白分析的逻辑严密性",
        "critical_thinking": "对现有理论的批判性思考",
        "logical_coherence": "章节内部逻辑连贯性",
        "model_rigor": "理论模型的严谨性",
        "proof_completeness": "证明过程的完整性",
        "theoretical_innovation": "理论创新的价值",
        "logical_coherence": "逻辑推导的严密性",
        "technical_soundness": "技术方法的正确性",
        "assumption_validation": "模型假设的合理性",
        "verification_design": "理论验证方案的设计质量",
        "parameter_configuration": "实验参数设置的合理性",
        "analysis_tool_selection": "分析工具选择的恰当性",
        "experimental_rigor": "实验过程的严谨性",
        "data_processing": "数据处理方法的科学性",
        "performance_analysis": "理论性能分析的深度",
        "boundary_validation": "边界条件验证的完整性",
        "theoretical_comparison": "与现有理论的对比分析",
        "contribution_clarity": "理论贡献的明确性",
        "limitation_awareness": "理论局限性的认识",
        "theoretical_contribution": "理论贡献总结的明确性",
        "research_questions": "研究问题回答的完整性",
        "limitation_analysis": "理论局限性的客观性",
        "future_directions": "未来研究方向的合理性",
        "conclusion_coherence": "结论的逻辑一致性",
        # 方法创新型字段
        "technical_evolution": "技术发展脉络的梳理",
        "performance_comparison": "性能对比分析的深度",
        "limitation_analysis": "现有方法局限性分析的准确性",
        "innovation_context": "创新背景的论证逻辑",
        "analysis_objectivity": "技术分析的客观性",
        "design_rationale": "方法设计的合理性",
        "technical_feasibility": "技术实现的可行性",
        "innovation_significance": "创新点的价值",
        "implementation_details": "实现细节的完整性",
        "methodology_rigor": "方法论的严谨性",
        "practical_applicability": "实际应用潜力",
        "experimental_setup": "实验环境配置的完整性",
        "dataset_preparation": "数据集准备的质量",
        "evaluation_metrics": "评估指标选择的恰当性",
        "baseline_selection": "基线方法选择的代表性",
        "validation_protocol": "验证流程的科学性",
        "performance_comparison": "性能对比分析的客观性",
        "ablation_analysis": "消融实验分析的深度",
        "visualization_quality": "可视化展示的效果",
        "advantage_validation": "方法优势验证的充分性",
        "application_potential": "应用前景探讨的合理性",
        "innovation_summary": "方法创新总结的清晰度",
        "performance_advantage": "性能优势归纳的客观性",
        "application_value": "应用价值分析的合理性",
        "technical_limitations": "技术局限说明的完整性",
        "improvement_directions": "改进方向的前瞻性",
        # 工程实现型字段
        "system_comparison": "相关系统对比的全面性",
        "technology_analysis": "技术方案分析的深度",
        "engineering_insights": "工程实践洞察的价值",
        "requirement_mapping": "需求与技术方案的映射关系",
        "practical_relevance": "分析与工程实践的相关性",
        "architecture_rationale": "架构设计的合理性",
        "module_cohesion": "模块设计的耦合度",
        "technology_selection": "技术选型的恰当性",
        "scalability_consideration": "可扩展性设计",
        "design_completeness": "设计方案的完整性",
        "practical_constraints": "实际约束的考虑",
        "development_environment": "开发环境配置的完整性",
        "core_implementation": "核心功能实现的质量",
        "technical_details": "关键技术细节的充分性",
        "integration_process": "系统集成过程的规范性",
        "deployment_solution": "部署方案的可行性",
        "function_testing": "功能测试的全面性",
        "performance_benchmark": "性能基准数据的完整性",
        "user_experience": "用户体验反馈的质量",
        "case_study": "案例研究的效果",
        "system_stability": "系统稳定性评估的充分性",
        "core_implementation": "核心功能实现的质量",
        "system_integration": "系统集成过程的规范性",
        "function_testing": "功能测试的全面性",
        "performance_evaluation": "性能评估的充分性",
        "user_validation": "用户验证的有效性",
        "system_achievement": "系统建设成果的总结",
        "functional_summary": "功能性能总结的客观性",
        "user_value": "用户价值体现的充分性",
        "engineering_experience": "工程经验提炼的价值",
        "evolution_planning": "系统演进规划的可行性",
        # 提取信息字段描述
        "research_focus": "核心研究问题和具体研究目标",
        "scope_definition": "研究范围的明确界定和边界条件",
        "innovation_claims": "明确声明的创新点和理论贡献",
        "literature_scope": "文献覆盖范围和关键文献列表",
        "theoretical_frameworks": "主要理论流派和核心理论框架",
        "research_gaps": "识别的研究空白和理论不足",
        "theoretical_focus": "核心理论问题和研究目标",
        "scope_boundaries": "理论研究范围和边界条件",
        "theoretical_innovations": "理论创新点和学术贡献",
        "literature_framework": "理论流派分类和关键文献",
        "model_framework": "理论模型的核心框架和关键定义",
        "proof_structure": "主要定理证明思路和关键推导步骤",
        "theoretical_insights": "理论分析的重要结论和洞察",
        "verification_framework": "理论验证的整体框架和关键实验设计",
        "experimental_parameters": "重要实验参数设置和配置细节",
        "analysis_methodology": "核心分析方法和验证工具",
        "performance_results": "关键性能指标和分析结果",
        "comparative_analysis": "与现有理论的对比数据和结论",
        "core_contributions": "核心理论贡献和创新点",
        "research_limitations": "研究工作的局限性和不足之处",
        "future_directions": "未来研究方向和潜在发展",
        "experimental_design": "理论验证实验设计和关键参数设置",
        "performance_metrics": "核心性能指标和分析结果",
        "comparative_evidence": "与现有理论的对比证据",
    }
