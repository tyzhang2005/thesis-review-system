#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pydantic模型定义：用于LLM结构化输出的类型验证
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal


# ==================== 步骤1：论文分类 ====================

class PaperClassificationResponse(BaseModel):
    """论文分类响应"""
    paper_type: Literal["理论研究", "方法创新", "工程实现"] = Field(
        description="论文类型：理论研究、方法创新或工程实现"
    )
    confidence: float = Field(default=1.0, description="分类置信度", ge=0, le=1)
    reasoning: Optional[str] = Field(default="", description="分类理由")


# ==================== 步骤2：章节分类 ====================

class ChapterInfo(BaseModel):
    """单个章节信息"""
    chapter_name: str = Field(description="章节名称")
    stage: str = Field(description="章节阶段标签")


class ChapterClassificationResponse(BaseModel):
    """章节分类响应"""
    chapters: List[ChapterInfo] = Field(description="章节分类结果列表")


# ==================== 步骤4：章节评估 ====================

class Issue(BaseModel):
    """评估问题"""
    position: str = Field(description="问题所在位置")
    content: str = Field(description="问题内容")
    suggestion: str = Field(description="修改建议")


class ChapterEvaluationResponse(BaseModel):
    """章节评估响应"""
    score: int = Field(description="评分", ge=0, le=100)
    issues: List[Issue] = Field(default_factory=list, description="问题列表")
    summary: str = Field(description="章节评估总结")


# ==================== 步骤5：工作量评估 ====================

class EvaluationItem(BaseModel):
    """评估项"""
    score: int = Field(description="评分", ge=0, le=100)
    analysis: str = Field(default="", description="评语分析")


class StructureEvaluation(BaseModel):
    """论文格式评估"""
    completeness: EvaluationItem = Field(description="结构完整性评估")
    abstract_and_keywords: EvaluationItem = Field(description="摘要和关键词规范性")
    catalog_standardization: EvaluationItem = Field(description="目录规范性")
    chapter_standardization: EvaluationItem = Field(description="章节规范性")
    acknowledgement_standardization: EvaluationItem = Field(description="致谢规范性")


class WorkloadEvaluationResponse(BaseModel):
    """工作量评估响应"""
    structure_evaluation: StructureEvaluation = Field(description="论文格式评估")
    summary: Dict[str, str] = Field(description="格式评语汇总", default={"analysis": ""})
    workload_evaluation: Dict[str, str] = Field(description="工作量评估", default={"analysis": ""})


# ==================== 步骤7：综合评分 ====================

class ComprehensiveScore(BaseModel):
    """综合评分响应"""
    total_score: int = Field(description="总分", ge=0, le=100)
    structure_score: int = Field(description="结构分", ge=0, le=100)
    content_score: int = Field(description="内容分", ge=0, le=100)
    writing_score: int = Field(description="写作分", ge=0, le=100)
    overall_evaluation: str = Field(description="综合评价")
    suggestions: List[str] = Field(default_factory=list, description="改进建议")


# ==================== 通用类型 ====================

class LLMResponse(BaseModel):
    """通用LLM文本响应（用于非结构化输出）"""
    content: str = Field(description="LLM返回的文本内容")


# ==================== 人工评审专家案例构建 ====================

class AdviceItem(BaseModel):
    """单条评审建议"""
    position: str = Field(description="原文中问题内容所在的实际章节和小节标题")
    raw_text: str = Field(description="填写该小节的块号列表，如[3,4]")
    type: str = Field(description="简要概括问题内容，对问题归类")
    context: str = Field(description="详细总结问题所在内容的章节内容")
    suggestion: str = Field(description="修改建议")
    chain_of_thought: str = Field(description="推理过程")
    scoring_impact: str = Field(description="评分影响[无不良影响/轻微/一般/中度/严重]")


class FormatAdvice(BaseModel):
    """格式问题建议"""
    advice: List[AdviceItem] = Field(default_factory=list, description="格式相关问题列表")


class ContentAdvice(BaseModel):
    """内容问题建议"""
    advice: List[AdviceItem] = Field(default_factory=list, description="内容相关问题列表")


class ScoreSummary(BaseModel):
    """评分总结"""
    problem: str = Field(description="根据人工评审分数，总结问题对总分的影响")
    chain_of_thought: str = Field(description="描述定档和给分的过程")
    score: int = Field(description="百分制总分", ge=0, le=100)


class HumanResultAnalysisResponse(BaseModel):
    """人工评审分析响应"""
    format: FormatAdvice = Field(description="格式问题")
    content: ContentAdvice = Field(description="内容问题")
    score_summary: ScoreSummary = Field(description="评分总结")


# 导出所有模型的字典映射，方便根据步骤名称获取
SCHEMA_MAP = {
    "step1_classify": PaperClassificationResponse,
    "step2_chapter_classify": ChapterClassificationResponse,
    "step4_chapter_eval": ChapterEvaluationResponse,
    "step5_workload": WorkloadEvaluationResponse,
    "step7_comprehensive": ComprehensiveScore,
    "human_analysis": HumanResultAnalysisResponse,
}
