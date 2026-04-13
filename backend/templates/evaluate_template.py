from typing import Any, Dict

introduction_template = """
    你是一个严格的本科毕业论文评估专家，需要对论文质量进行精准区分。
    接下来你将看到一篇本科毕业论文的绪论部分，你需要阅读该章节内容，
    ，并用200-300字概括本章节内容，再根据评估标准对全文打分，要求评分具有明显的区分度

    【章节概括评价】
    绪论（Introduction）章节
    按照给定的json结构体要求，阅读章节内容并提取对应信息，总结概括内容，并对指定项目进行简要评价。

    
    【输出要求】
    输出必须是一个纯粹的JSON对象，格式严格遵循：
    {{
        "chapter":
        {{                  
            "chapter_name":  "第一章 绪论",     
            "chapter_type": "introduction",
            "chapter_summary": "用200-300字概括本章主要内容，包括研究背景、问题和意义。",
            "chapter_remark":  "基于证据客观评价本章的优点和不足（例如：背景阐述是否清晰、问题界定是否明确、动机是否充分）。",
            "section_structure": ["按照文章实际结构，提取小节标题并分析概括小节内容，若本章未分节可留空“
                {{
                "section_title": "提取实际小节标题",
                "section_purpose": "用一句话说明该小节的核心目的",
                "key_points": ["用短语列出2-3个关键论点"],
                "weaknesses": ["可选：仅当存在具体缺点时填写，如“背景阐述缺乏数据支持”"]
                }}
            ],
            "research_background_evaluation": "用一句话基于证据评价研究背景阐述的清晰度和充分性，包括优点和不足（例如：是否引用最新文献、数据支持是否充分）。",
            "problem_statement_evaluation": "用一句话基于证据评价研究问题表述的明确性，包括优点和不足（例如：是否聚焦核心问题、是否存在模糊性）。",
            "research_significance_evaluation": "用一句话基于证据评价研究意义的阐述质量，包括优点和不足（例如：是否区分学术与实践价值、是否夸大）。",
            "thesis_statement": "用一句话概括论文的核心论点，并评价其清晰度和逻辑性。",
            "motivation_clarity_evaluation": "用一句话基于证据评价研究动机的阐述质量，包括优点和不足（例如：是否基于现实需求、是否缺乏实证支持）。",
            "scope_delimitation_evaluation": "用一句话基于证据评价研究范围界定的明确性，包括优点和不足（例如：边界是否清晰、是否遗漏关键方面）。",
            "impact_on_scoring": "可选：仅当本章存在明确问题时，用一句话说明对后续打分的影响（例如：问题陈述模糊可能降低‘选题契合度’得分）。"                          
        }}

    }}

    【论文元数据】
    标题：{title}
    摘要：{abstract}
    关键词：{keywords}

    【章节名称】
    {chapter_name}

    【章节内容】
    {context}
    
    """

literature_review_template = """
    你是一个严格的本科毕业论文评估专家，需要对论文质量进行精准区分。
    接下来你将看到一篇本科毕业论文的文献综述部分，你需要阅读该章节内容，
    ，并用200-300字概括本章节内容，再根据评估标准对全文打分，要求评分具有明显的区分度

    【章节概括评价】
    文献综述（Literature Review）章节
    按照给定的json结构体要求，阅读章节内容并提取对应信息，总结概括内容，并对指定项目进行简要评价。

    【输出要求】
    输出必须是一个纯粹的JSON对象，格式严格遵循：
    {{
        "chapter":
        {{  
            "chapter_name":  "第二章 文献综述（请按实际章节名称填写）",     
            "chapter_type": "literature_review",
            "chapter_summary": "用200-300字概括文献综述的整体框架和主要发现。",
            "chapter_remark":  "基于证据客观评价本章的优点和不足（例如：文献覆盖是否全面、批判性分析是否深入、组织是否逻辑）。",
            "section_structure": [ "按照文章实际结构，提取小节标题并分析概括小节内容，若本章未分节可留空“
                {{
                "section_title": "提取实际小节标题",
                "section_summary": "简要概括小节内容",
                "weaknesses": ["可选：仅当存在具体缺点时填写，如“文献选择偏颇”"]
                }}
            ],
            "literature_information":[
                "research_domain": "识别研究领域名称",
                "key_scholars_theories": "列出文献综述部分支出的代表性学者和理论",
                "research_gaps": "用一句话指出该领域的研究空白",
            ],
            "review_scope_evaluation": "用一句话基于证据评价文献覆盖的广度和深度，包括优点和不足（例如：是否涵盖关键文献、是否缺乏最新研究）。",
            "critical_analysis_evaluation": "用一句话基于证据评价文献批判性分析的质量，包括优点和不足（例如：是否仅描述而未评价、是否对比冲突观点）。",
            "thematic_organization_evaluation": "用一句话基于证据评价文献组织的逻辑性，包括优点和不足（例如：结构是否清晰、主题是否重叠）。",
            "relevance_to_topic_evaluation": "用一句话基于证据评价文献与主题的相关性，包括优点和不足（例如：是否聚焦核心问题、是否存在冗余引用）。",
            "workload_sufficiency_evaluation": "用一句话基于证据评价文献工作量的适宜度，包括优点和不足（例如：文献数量和质量是否达标、是否涵盖多语种文献）。",
            "citation_quality_evaluation": "用一句话基于证据评价引用文献的质量，包括优点和不足（例如：是否权威和时效、是否引用关键研究）。",
            "synthesis_ability_evaluation": "用一句话基于证据评价文献综合归纳能力，包括优点和不足（例如：是否提炼关键见解、是否建立理论联系）。",
            "impact_on_scoring": "可选：仅当本章存在明确问题时，用一句话说明对后续打分的影响（例如：文献覆盖不全可能降低‘文献检索和分析能力’得分）。"
        }}
              
    }}

    【论文元数据】
    标题：{title}
    摘要：{abstract}
    关键词：{keywords}

    【章节名称】
    {chapter_name}

    【章节内容】
    {context}
    
    """

methdology_template = """
    你是一个严格的本科毕业论文评估专家，需要对论文质量进行精准区分。
    接下来你将看到一篇本科毕业论文的方法论部分，你需要阅读该章节内容，
    ，并用200-300字概括本章节内容，再根据评估标准对全文打分，要求评分具有明显的区分度

    【章节概括评价】
    方法论（Methodology）章节
    按照给定的json结构体要求，阅读章节内容并提取对应信息，总结概括内容，并对指定项目进行简要评价。

    【输出要求】
    输出必须是一个纯粹的JSON对象，格式严格遵循：
    {{
        "chapter":
        {{  
            "chapter_name":  "第三章 研究方法（请按实际章节名称填写）",     
            "chapter_type": "methodology",
            "chapter_summary": "用200-300字概括研究方法和理论框架的核心内容。"
            "chapter_remark":  "基于证据客观评价本章的优点和不足（例如：方法是否匹配问题、技术细节是否充分、框架是否严谨）。",
            "section_structure": [ "按照文章实际结构，提取小节标题并分析概括小节内容，若本章未分节可留空“
                {{
                "section_title": "提取实际小节标题",
                "section_summary": "简要概括小节内容",
                "weaknesses": ["可选：仅当存在具体缺点时填写，如“方法描述过于笼统”"]
                }}                
            ],
            "information":["按照文章实际内容，总结本章节中方法论涉及内容的相关信息“
                {{
                "methodological_approach": "识别具体研究方法",
                "rationale": "用一句话说明方法选择的理由",
                }}
            ],
            "methodology_appropriateness_evaluation": "用一句话基于证据评价方法与研究问题的匹配度，包括优点和不足（例如：是否适用、是否论证选择理由）。",
            "technical_details_evaluation": "用一句话基于证据评价方法描述的具体程度，包括优点和不足（例如：参数和步骤是否清晰、是否提供示例）。",
            "framework_rigor_evaluation": "用一句话基于证据评价理论框架的严谨性，包括优点和不足（例如：逻辑是否严密、是否处理潜在偏差）。",
            "innovation_in_methods_evaluation": "用一句话基于证据评价方法的创新性，包括优点和不足（例如：是否有新意、是否对比现有方法）。",
            "feasibility_assessment_evaluation": "用一句话基于证据评价方案的可行性，包括优点和不足（例如：是否考虑资源和技术约束）。",
            "conceptual_clarity_evaluation": "用一句话基于证据评价核心概念的清晰度，包括优点和不足（例如：术语是否明确、是否解释得当）。",
            "validity_consideration_evaluation": "用一句话基于证据评价效度保障措施，包括优点和不足（例如：是否处理偏差、验证是否充分）。",
            "impact_on_scoring": "可选：仅当本章存在明确问题时，用一句话说明对后续打分的影响（例如：方法描述不具体可能降低‘专业方法工具运用’得分）。"
        }}
    }}

    【论文元数据】
    标题：{title}
    摘要：{abstract}
    关键词：{keywords}

    【章节名称】
    {chapter_name}

    【章节内容】
    {context}
    
    """

analysis_template = """
    你是一个严格的本科毕业论文评估专家，需要对论文质量进行精准区分。
    接下来你将看到一篇本科毕业论文的实验设计/数据分析部分，你需要阅读该章节内容，
    ，并用200-300字概括本章节内容，再根据评估标准对全文打分，要求评分具有明显的区分度

    【章节概括评价】
    实验设计/数据分析（Analysis）章节
    按照给定的json结构体要求，阅读章节内容并提取对应信息，总结概括内容，并对指定项目进行简要评价。

    【输出要求】
    输出必须是一个纯粹的JSON对象，格式严格遵循：
    {{
        "chapter":
        {{  
            "chapter_name":  "第四章 实验设计（请按实际章节名称填写）",     
            "chapter_type": "analysis",
            "chapter_summary": "用200-300字概括实验设计或数据分析的整体方案。",
            "chapter_remark": 基于证据客观评价本章的优点和不足（例如：数据质量是否可靠、分析过程是否严谨、设计是否科学）。",
            "section_structure": [ "按照文章实际结构，提取小节标题并分析概括小节内容，若本章未分节可留空“
                {{
                "section_title": "提取实际小节标题",
                "section_summary": "简要概括小节内容",
                "weaknesses": ["可选：仅当存在具体缺点时填写，如“数据样本太小”"]
                }}     
            ],
            "information": ["按照文章实际内容，总结本章节中实验设计和数据分析涉及内容的相关信息“
                {{
                "data_source": "说明数据来源或实验对象",
                "analysis_method": "说明具体分析技术",
                "key_findings": "用一句话概括主要发现",
                }}
            ],
            "experimental_design_evaluation": "用一句话基于证据评价实验设计的科学性，包括优点和不足（例如：是否有对照组、变量控制是否合理）。",
            "data_adequacy_evaluation": "用一句话基于证据评价数据规模和质量，包括优点和不足（例如：样本是否充足、来源是否可靠）。",
            "analysis_rigor_evaluation": "用一句话基于证据评价分析过程的严谨性，包括优点和不足（例如：是否处理异常值、方法是否适用）。",
            "result_reliability_evaluation": "用一句话基于证据评价结果的可信度，包括优点和不足（例如：误差是否讨论、重复性是否验证）。",
            "workload_evidence_evaluation": "用一句话基于证据评价工作量的具体体现，包括优点和不足（例如：实验次数或数据量是否充分）。",
            "procedure_detail_evaluation": "用一句话基于证据评价步骤描述的详细程度，包括优点和不足（例如：关键步骤是否清晰、是否提供工具信息）。",
            "statistical_appropriateness_evaluation": "用一句话基于证据评价统计方法的选择，包括优点和不足（例如：方法是否适用、假设是否验证）。",
            "impact_on_scoring": "可选：仅当本章存在明确问题时，用一句话说明对后续打分的影响（例如：数据不足可能降低‘知识综合应用和研究深度’得分）。"
        }}
    }}

    【论文元数据】
    标题：{title}
    摘要：{abstract}
    关键词：{keywords}

    【章节名称】
    {chapter_name}

    【章节内容】
    {context}
    
    """

discussion_template = """
    你是一个严格的本科毕业论文评估专家，需要对论文质量进行精准区分。
    接下来你将看到一篇本科毕业论文的实验结果与讨论部分，你需要阅读该章节内容，
    ，并用200-300字概括本章节内容，再根据评估标准对全文打分，要求评分具有明显的区分度

    【章节概括评价】
    实验结果与讨论（Discussion）章节
    按照给定的json结构体要求，阅读章节内容并提取对应信息，总结概括内容，并对指定项目进行简要评价。

    

    【输出要求】
    输出必须是一个纯粹的JSON对象，格式严格遵循：
    {{
        "chapter":
        {{  
            "chapter_name":  "第五章 实验结果（请按实际章节名称填写）",     
            "chapter_type": "discussion",
            "chapter_summary": "用200-300字概括主要发现和讨论要点。",
            "chapter_remark": "基于证据客观评价本章的优点和不足（例如：结果解读是否深入、论证是否逻辑、是否关联文献）。",
            "section_structure": [ "按照文章实际结构，提取小节标题并分析概括小节内容，若本章未分节可留空“
                {{
                "section_title": "提取实际小节标题",
                "section_summary": "简要概括小节内容",
                "weaknesses": ["可选：仅当存在具体缺点时填写，如“解读缺乏深度”"]
                }}     
            ],
            "information": [
                {{
                "finding": "用一句话陈述具体发现",
                "interpretation": "用一句话说明发现的含义",
                "relation_to_literature": "用一句话关联已有研究",
                }}
            ],
            "result_presentation_evaluation": "用一句话基于证据评价结果呈现的清晰度，包括优点和不足（例如：图表是否规范、关键发现是否突出）。",
            "interpretation_depth_evaluation": "用一句话基于证据评价结果解读的深度，包括优点和不足（例如：是否挖掘根本原因、是否考虑替代解释）。",
            "theoretical_implications_evaluation": "用一句话基于证据评价理论意义的探讨，包括优点和不足（例如：是否链接现有理论、贡献是否明确）。",
            "limitations_acknowledgment_evaluation": "用一句话基于证据评价局限性的认识，包括优点和不足（例如：是否全面承认局限、是否说明影响）。",
            "argument_coherence_evaluation": "用一句话基于证据评价论证的逻辑性，包括优点和不足（例如：是否基于数据推理、是否存在矛盾）。",
            "finding_significance_evaluation": "用一句话基于证据评价发现的重要性，包括优点和不足（例如：是否区分显著性与偶然性、实际价值是否明确）。",
            "critical_thinking_evaluation": "用一句话基于证据评价批判性思维体现，包括优点和不足（例如：是否质疑假设、是否讨论反例）。",
            "impact_on_scoring": "可选：仅当本章存在明确问题时，用一句话说明对后续打分的影响（例如：论证不严谨可能降低‘论证严谨性和科学性’得分）。"
        }}
    }}  

    【论文元数据】
    标题：{title}
    摘要：{abstract}
    关键词：{keywords}

    【章节名称】
    {chapter_name}

    【章节内容】
    {context}

"""

conclusion_template = """
    你是一个严格的本科毕业论文评估专家，需要对论文质量进行精准区分。
    接下来你将看到一篇本科毕业论文的总结和展望部分，你需要阅读该章节内容，
    ，并用200-300字概括本章节内容，再根据评估标准对全文打分，要求评分具有明显的区分度

    【章节概括评价】
    总结和展望（Conclusion）章节
    按照给定的json结构体要求，阅读章节内容并提取对应信息，总结概括内容，并对指定项目进行简要评价。

    【输出要求】
    输出必须是一个纯粹的JSON对象，格式严格遵循：
    {{
        "chapter":
        {{  
            "chapter_name":  "第X章 总结与展望（请按实际章节名称填写）",     
            "chapter_type": "conclusion",
            "chapter_summary": "用200-300字概括结论部分的核心内容。基于证据客观评价本章的优点和不足（例如：是否回答研究问题、建议是否可行、贡献总结是否明确）。",
            "section_structure": [ "按照文章实际结构，提取小节标题并分析概括小节内容，若本章未分节可留空“
                {{
                "section_title": "提取实际小节标题",
                "section_summary": "简要概括小节内容",
                "weaknesses": ["可选：仅当存在具体缺点时填写，如“结论泛泛而谈”"]
                }}
            ],
            "research_questions_answered_evaluation": "用一句话基于证据评价研究问题回答的完整性，包括优点和不足（例如：是否覆盖所有问题、回答是否直接）。",
            "contribution_summary_evaluation": "用一句话基于证据评价研究贡献总结的明确性，包括优点和不足（例如：是否区分学术与实践价值、是否对比已有研究）。",
            "practical_implications_evaluation": "用一句话基于证据评价实践意义的阐述，包括优点和不足（例如：是否提供应用场景、是否脱离实际）。",
            "future_directions_evaluation": "用一句话基于证据评价未来研究建议的价值，包括优点和不足（例如：是否基于本文局限、是否具体可行）。",
            "conclusion_strength_evaluation": "用一句话基于证据评价结论的说服力，包括优点和不足（例如：是否基于证据、逻辑是否严密）。",
            "answer_completeness_evaluation": "用一句话基于证据评价问题回答的全面性，包括优点和不足（例如：是否整合发现、是否回应目标）。",
            "recommendation_feasibility_evaluation": "用一句话基于证据评价建议的可行性，包括优点和不足（例如：是否考虑资源约束、是否可操作）。",
            "impact_on_scoring": "可选：仅当本章存在明确问题时，用一句话说明对后续打分的影响（例如：结论薄弱可能降低‘成果价值’得分）。"
        }}
    }}   

    【论文元数据】
    标题：{title}
    摘要：{abstract}
    关键词：{keywords}

    【章节名称】
    {chapter_name}

    【章节内容】
    {context}

"""
# 在 llm_utils.py 中添加以下JSON Schema定义


def create_introduction_json_schema() -> Dict[str, Any]:
    """创建绪论章节的JSON Schema"""
    return {
        "type": "object",
        "properties": {
            "chapter": {
                "type": "object",
                "properties": {
                    "chapter_name": {"type": "string"},
                    "chapter_type": {"type": "string", "const": "introduction"},
                    "chapter_summary": {"type": "string"},
                    "chapter_remark": {"type": "string"},
                    "section_structure": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "section_title": {"type": "string"},
                                "section_purpose": {"type": "string"},
                                "key_points": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "weaknesses": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": [
                                "section_title",
                                "section_purpose",
                                "key_points",
                            ],
                        },
                    },
                    "research_background_evaluation": {"type": "string"},
                    "problem_statement_evaluation": {"type": "string"},
                    "research_significance_evaluation": {"type": "string"},
                    "thesis_statement": {"type": "string"},
                    "motivation_clarity_evaluation": {"type": "string"},
                    "scope_delimitation_evaluation": {"type": "string"},
                    "impact_on_scoring": {"type": "string"},
                },
                "required": [
                    "chapter_name",
                    "chapter_type",
                    "chapter_summary",
                    "chapter_remark",
                    "section_structure",
                    "research_background_evaluation",
                    "problem_statement_evaluation",
                    "research_significance_evaluation",
                    "thesis_statement",
                    "motivation_clarity_evaluation",
                    "scope_delimitation_evaluation",
                ],
            }
        },
        "required": ["chapter"],
    }


def create_literature_review_json_schema() -> Dict[str, Any]:
    """创建文献综述章节的JSON Schema"""
    return {
        "type": "object",
        "properties": {
            "chapter": {
                "type": "object",
                "properties": {
                    "chapter_name": {"type": "string"},
                    "chapter_type": {"type": "string", "const": "literature_review"},
                    "chapter_summary": {"type": "string"},
                    "chapter_remark": {"type": "string"},
                    "section_structure": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "section_title": {"type": "string"},
                                "section_summary": {"type": "string"},
                                "weaknesses": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": ["section_title", "section_summary"],
                        },
                    },
                    "literature_information": {
                        "type": "object",
                        "properties": {
                            "research_domain": {"type": "string"},
                            "key_scholars_theories": {"type": "string"},
                            "research_gaps": {"type": "string"},
                        },
                        "required": [
                            "research_domain",
                            "key_scholars_theories",
                            "research_gaps",
                        ],
                    },
                    "review_scope_evaluation": {"type": "string"},
                    "critical_analysis_evaluation": {"type": "string"},
                    "thematic_organization_evaluation": {"type": "string"},
                    "relevance_to_topic_evaluation": {"type": "string"},
                    "workload_sufficiency_evaluation": {"type": "string"},
                    "citation_quality_evaluation": {"type": "string"},
                    "synthesis_ability_evaluation": {"type": "string"},
                    "impact_on_scoring": {"type": "string"},
                },
                "required": [
                    "chapter_name",
                    "chapter_type",
                    "chapter_summary",
                    "chapter_remark",
                    "section_structure",
                    "literature_information",
                    "review_scope_evaluation",
                    "critical_analysis_evaluation",
                    "thematic_organization_evaluation",
                    "relevance_to_topic_evaluation",
                    "workload_sufficiency_evaluation",
                    "citation_quality_evaluation",
                    "synthesis_ability_evaluation",
                ],
            }
        },
        "required": ["chapter"],
    }


def create_methodology_json_schema() -> Dict[str, Any]:
    """创建方法论章节的JSON Schema"""
    return {
        "type": "object",
        "properties": {
            "chapter": {
                "type": "object",
                "properties": {
                    "chapter_name": {"type": "string"},
                    "chapter_type": {"type": "string", "const": "methodology"},
                    "chapter_summary": {"type": "string"},
                    "chapter_remark": {"type": "string"},
                    "section_structure": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "section_title": {"type": "string"},
                                "section_summary": {"type": "string"},
                                "weaknesses": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": ["section_title", "section_summary"],
                        },
                    },
                    "information": {
                        "type": "object",
                        "properties": {
                            "methodological_approach": {"type": "string"},
                            "rationale": {"type": "string"},
                        },
                        "required": ["methodological_approach", "rationale"],
                    },
                    "methodology_appropriateness_evaluation": {"type": "string"},
                    "technical_details_evaluation": {"type": "string"},
                    "framework_rigor_evaluation": {"type": "string"},
                    "innovation_in_methods_evaluation": {"type": "string"},
                    "feasibility_assessment_evaluation": {"type": "string"},
                    "conceptual_clarity_evaluation": {"type": "string"},
                    "validity_consideration_evaluation": {"type": "string"},
                    "impact_on_scoring": {"type": "string"},
                },
                "required": [
                    "chapter_name",
                    "chapter_type",
                    "chapter_summary",
                    "chapter_remark",
                    "section_structure",
                    "information",
                    "methodology_appropriateness_evaluation",
                    "technical_details_evaluation",
                    "framework_rigor_evaluation",
                    "innovation_in_methods_evaluation",
                    "feasibility_assessment_evaluation",
                    "conceptual_clarity_evaluation",
                    "validity_consideration_evaluation",
                ],
            },
        },
        "required": ["chapter"],
    }


def create_analysis_json_schema() -> Dict[str, Any]:
    """创建分析章节的JSON Schema"""
    return {
        "type": "object",
        "properties": {
            "chapter": {
                "type": "object",
                "properties": {
                    "chapter_name": {"type": "string"},
                    "chapter_type": {"type": "string", "const": "analysis"},
                    "chapter_summary": {"type": "string"},
                    "chapter_remark": {"type": "string"},
                    "section_structure": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "section_title": {"type": "string"},
                                "section_summary": {"type": "string"},
                                "weaknesses": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": ["section_title", "section_summary"],
                        },
                    },
                    "information": {
                        "type": "object",
                        "properties": {
                            "data_source": {"type": "string"},
                            "analysis_method": {"type": "string"},
                            "key_findings": {"type": "string"},
                        },
                        "required": ["data_source", "analysis_method", "key_findings"],
                    },
                    "experimental_design_evaluation": {"type": "string"},
                    "data_adequacy_evaluation": {"type": "string"},
                    "analysis_rigor_evaluation": {"type": "string"},
                    "result_reliability_evaluation": {"type": "string"},
                    "workload_evidence_evaluation": {"type": "string"},
                    "procedure_detail_evaluation": {"type": "string"},
                    "statistical_appropriateness_evaluation": {"type": "string"},
                    "impact_on_scoring": {"type": "string"},
                },
                "required": [
                    "chapter_name",
                    "chapter_type",
                    "chapter_summary",
                    "chapter_remark",
                    "section_structure",
                    "information",
                    "experimental_design_evaluation",
                    "data_adequacy_evaluation",
                    "analysis_rigor_evaluation",
                    "result_reliability_evaluation",
                    "workload_evidence_evaluation",
                    "procedure_detail_evaluation",
                    "statistical_appropriateness_evaluation",
                ],
            },
        },
        "required": ["chapter"],
    }


def create_discussion_json_schema() -> Dict[str, Any]:
    """创建讨论章节的JSON Schema"""
    return {
        "type": "object",
        "properties": {
            "chapter": {
                "type": "object",
                "properties": {
                    "chapter_name": {"type": "string"},
                    "chapter_type": {"type": "string", "const": "discussion"},
                    "chapter_summary": {"type": "string"},
                    "chapter_remark": {"type": "string"},
                    "section_structure": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "section_title": {"type": "string"},
                                "section_summary": {"type": "string"},
                                "weaknesses": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": ["section_title", "section_summary"],
                        },
                    },
                    "information": {
                        "type": "object",
                        "properties": {
                            "finding": {"type": "string"},
                            "interpretation": {"type": "string"},
                            "relation_to_literature": {"type": "string"},
                        },
                        "required": [
                            "finding",
                            "interpretation",
                            "relation_to_literature",
                        ],
                    },
                    "result_presentation_evaluation": {"type": "string"},
                    "interpretation_depth_evaluation": {"type": "string"},
                    "theoretical_implications_evaluation": {"type": "string"},
                    "limitations_acknowledgment_evaluation": {"type": "string"},
                    "argument_coherence_evaluation": {"type": "string"},
                    "finding_significance_evaluation": {"type": "string"},
                    "critical_thinking_evaluation": {"type": "string"},
                    "impact_on_scoring": {"type": "string"},
                },
                "required": [
                    "chapter_name",
                    "chapter_type",
                    "chapter_summary",
                    "chapter_remark",
                    "section_structure",
                    "information",
                    "result_presentation_evaluation",
                    "interpretation_depth_evaluation",
                    "theoretical_implications_evaluation",
                    "limitations_acknowledgment_evaluation",
                    "argument_coherence_evaluation",
                    "finding_significance_evaluation",
                    "critical_thinking_evaluation",
                ],
            },
        },
        "required": ["chapter"],
    }


def create_conclusion_json_schema() -> Dict[str, Any]:
    """创建结论章节的JSON Schema"""
    return {
        "type": "object",
        "properties": {
            "chapter": {
                "type": "object",
                "properties": {
                    "chapter_name": {"type": "string"},
                    "chapter_type": {"type": "string", "const": "conclusion"},
                    "chapter_summary": {"type": "string"},
                    "chapter_remark": {"type": "string"},
                    "section_structure": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "section_title": {"type": "string"},
                                "section_summary": {"type": "string"},
                                "weaknesses": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": ["section_title", "section_summary"],
                        },
                    },
                    "research_questions_answered_evaluation": {"type": "string"},
                    "contribution_summary_evaluation": {"type": "string"},
                    "practical_implications_evaluation": {"type": "string"},
                    "future_directions_evaluation": {"type": "string"},
                    "conclusion_strength_evaluation": {"type": "string"},
                    "answer_completeness_evaluation": {"type": "string"},
                    "recommendation_feasibility_evaluation": {"type": "string"},
                    "impact_on_scoring": {"type": "string"},
                },
                "required": [
                    "chapter_name",
                    "chapter_type",
                    "chapter_summary",
                    "chapter_remark",
                    "section_structure",
                    "research_questions_answered_evaluation",
                    "contribution_summary_evaluation",
                    "practical_implications_evaluation",
                    "future_directions_evaluation",
                    "conclusion_strength_evaluation",
                    "answer_completeness_evaluation",
                    "recommendation_feasibility_evaluation",
                ],
            },
        },
        "required": ["chapter"],
    }


def get_chapter_schema(chapter_type: str) -> Dict[str, Any]:
    """根据章节类型获取对应的JSON Schema"""
    # 创建章节类型到JSON Schema的映射
    CHAPTER_SCHEMA_MAP = {
        "introduction": create_introduction_json_schema,
        "literature_review": create_literature_review_json_schema,
        "methodology": create_methodology_json_schema,
        "analysis": create_analysis_json_schema,
        "discussion": create_discussion_json_schema,
        "conclusion": create_conclusion_json_schema,
    }
    if chapter_type in CHAPTER_SCHEMA_MAP:
        return CHAPTER_SCHEMA_MAP[chapter_type]()
