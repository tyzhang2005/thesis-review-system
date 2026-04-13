#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pydantic模型定义包
"""

from .schemas import (
    SCHEMA_MAP,
    ChapterClassificationResponse,
    ChapterEvaluationResponse,
    ChapterInfo,
    ComprehensiveScore,
    EvaluationItem,
    Issue,
    LLMResponse,
    PaperClassificationResponse,
    StructureEvaluation,
    WorkloadEvaluationResponse,
)

__all__ = [
    "PaperClassificationResponse",
    "ChapterInfo",
    "ChapterClassificationResponse",
    "Issue",
    "ChapterEvaluationResponse",
    "EvaluationItem",
    "StructureEvaluation",
    "WorkloadEvaluationResponse",
    "ComprehensiveScore",
    "LLMResponse",
    "SCHEMA_MAP",
]
