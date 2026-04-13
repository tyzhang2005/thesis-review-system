#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
提示词服务：负责从Markdown文件加载和管理提示词模板
"""
import os
from pathlib import Path
from typing import Dict, Any, Optional
from jinja2 import Template


class PromptService:
    """提示词模板服务"""

    def __init__(self, prompts_dir: str = None):
        """
        初始化提示词服务

        Args:
            prompts_dir: 提示词目录路径，默认为backend/prompts
        """
        if prompts_dir is None:
            # 默认路径：相对于此文件所在位置
            current_dir = Path(__file__).parent
            prompts_dir = current_dir.parent / "prompts"

        self.prompts_dir = Path(prompts_dir)
        self._templates: Dict[str, str] = {}
        self._load_all_templates()

    def _load_all_templates(self):
        """预加载所有提示词模板到内存"""
        if not self.prompts_dir.exists():
            raise FileNotFoundError(f"提示词目录不存在: {self.prompts_dir}")

        # 遍历所有.md文件
        for md_file in self.prompts_dir.rglob("*.md"):
            # 计算相对于prompts_dir的路径作为模板名称
            rel_path = md_file.relative_to(self.prompts_dir)
            template_name = str(rel_path.with_suffix('')).replace('\\', '/')

            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()

            self._templates[template_name] = content

    def get_template(self, template_name: str) -> str:
        """
        获取原始模板内容

        Args:
            template_name: 模板名称，如 "step1_classify" 或 "step4_theory/introduction"

        Returns:
            模板内容字符串

        Raises:
            KeyError: 模板不存在
        """
        if template_name not in self._templates:
            available = list(self._templates.keys())
            raise KeyError(
                f"模板 '{template_name}' 不存在。\n"
                f"可用模板: {available[:10]}..." if len(available) > 10
                else f"可用模板: {available}"
            )
        return self._templates[template_name]

    def format_template(self, template_name: str, **kwargs) -> str:
        """
        格式化模板（使用Python的.format语法）

        Args:
            template_name: 模板名称
            **kwargs: 格式化参数

        Returns:
            格式化后的提示词字符串
        """
        template = self.get_template(template_name)
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"模板 '{template_name}' 缺少参数: {e}")

    def render_template(self, template_name: str, **kwargs) -> str:
        """
        渲染模板（使用Jinja2语法）

        Args:
            template_name: 模板名称
            **kwargs: 渲染参数

        Returns:
            渲染后的提示词字符串
        """
        template = self.get_template(template_name)
        try:
            jinja_template = Template(template)
            return jinja_template.render(**kwargs)
        except Exception as e:
            raise ValueError(f"模板 '{template_name}' 渲染失败: {e}")

    def list_templates(self, prefix: str = "") -> list[str]:
        """
        列出所有模板名称

        Args:
            prefix: 可选的前缀过滤

        Returns:
            模板名称列表
        """
        if prefix:
            return [name for name in self._templates.keys() if name.startswith(prefix)]
        return list(self._templates.keys())

    def reload(self):
        """重新加载所有模板（用于开发时的热更新）"""
        self._templates.clear()
        self._load_all_templates()


# ==================== 便捷函数 ====================

# 全局单例
_prompt_service: Optional[PromptService] = None


def get_prompt_service() -> PromptService:
    """获取全局PromptService单例"""
    global _prompt_service
    if _prompt_service is None:
        _prompt_service = PromptService()
    return _prompt_service


def get_template(template_name: str) -> str:
    """快捷方式：获取原始模板"""
    return get_prompt_service().get_template(template_name)


def format_template(template_name: str, **kwargs) -> str:
    """快捷方式：格式化模板"""
    return get_prompt_service().format_template(template_name, **kwargs)
