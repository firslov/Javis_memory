"""实体提取服务

从对话文本中提取结构化信息：实体、关键点、决策、行动项
"""
import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from config.logging import get_logger


logger = get_logger(__name__)


@dataclass
class ExtractedEntities:
    """提取的结构化信息"""
    entities: List[Dict[str, str]]  # 实体列表
    key_points: List[str]  # 关键点
    decisions: List[str]  # 决策
    action_items: List[str]  # 行动项


class EntityExtractor:
    """实体提取器 - 使用LLM进行结构化提取"""

    def __init__(self, llm_forwarder=None):
        """初始化提取器

        Args:
            llm_forwarder: LLM转发器（用于调用LLM）
        """
        self.llm_forwarder = llm_forwarder

    async def extract(
        self,
        text: str,
        use_llm: bool = False,
    ) -> ExtractedEntities:
        """从文本中提取结构化信息

        Args:
            text: 输入文本
            use_llm: 是否使用LLM提取（更准确但较慢）

        Returns:
            ExtractedEntities 提取结果
        """
        if use_llm and self.llm_forwarder:
            return await self._extract_with_llm(text)
        else:
            return self._extract_with_rules(text)

    def _extract_with_rules(self, text: str) -> ExtractedEntities:
        """使用规则提取（快速但简单）"""
        entities = []
        key_points = []

        # 提取实体（简单模式）
        patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "url": r'https?://\S+',
            "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            "date": r'\b\d{4}-\d{2}-\d{2}\b',
        }

        for entity_type, pattern in patterns.items():
            matches = re.findall(pattern, text)
            for match in matches:
                entities.append({"type": entity_type, "value": match})

        # 提取关键句子（包含特定关键词的行）
        lines = text.split('\n')
        keyword_patterns = [
            r'(重要|关键|必须|需要|记住|remember|important|key)',
            r'(决定|确定|decided|determined)',
            r'(TODO|待办|行动|action)',
        ]

        for line in lines:
            line = line.strip()
            if len(line) < 10:  # 跳过太短的行
                continue
            if any(re.search(pattern, line, re.IGNORECASE) for pattern in keyword_patterns):
                key_points.append(line)

        return ExtractedEntities(
            entities=entities,
            key_points=key_points[:10],  # 限制数量
            decisions=[],
            action_items=[],
        )

    async def _extract_with_llm(self, text: str) -> ExtractedEntities:
        """使用LLM提取（准确但较慢）"""
        if not self.llm_forwarder:
            return self._extract_with_rules(text)

        prompt = f"""请从以下文本中提取结构化信息，以JSON格式返回：

文本：
{text[:2000]}

请提取：
1. entities: 实体列表（人名、项目名、概念等），格式为 {{"type": "...", "value": "..."}}
2. key_points: 关键信息点列表
3. decisions: 明确的决策或决定
4. action_items: 需要采取的行动项

返回格式：
{{
    "entities": [{{"type": "person", "value": "张三"}}, ...],
    "key_points": ["...", ...],
    "decisions": ["...", ...],
    "action_items": ["...", ...]
}}

只返回JSON，不要其他内容。"""

        try:
            # 调用LLM - 获取可用模型
            models = self.llm_forwarder.get_available_models()
            if not models:
                return self._extract_with_rules(text)

            # 选择第一个可用模型
            model = list(models.keys())[0]

            # 构造请求
            from openai import AsyncOpenAI

            # 获取服务器配置
            server = list(self.llm_forwarder.servers.values())[0]
            client = AsyncOpenAI(
                base_url=server.base_url,
                api_key=server.api_key,
            )

            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )

            content = response.choices[0].message.content

            # 解析JSON
            # 移除可能的markdown代码块标记
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]

            extracted = json.loads(content.strip())

            return ExtractedEntities(
                entities=extracted.get("entities", []),
                key_points=extracted.get("key_points", []),
                decisions=extracted.get("decisions", []),
                action_items=extracted.get("action_items", []),
            )

        except Exception as e:
            logger.warning(f"[ENTITY] LLM extraction failed: {e}, falling back to rules")
            return self._extract_with_rules(text)


# 全局实例
_extractor: Optional[EntityExtractor] = None


def get_entity_extractor(llm_forwarder=None) -> EntityExtractor:
    """获取实体提取器实例"""
    global _extractor
    if _extractor is None or (_extractor.llm_forwarder is None and llm_forwarder is not None):
        _extractor = EntityExtractor(llm_forwarder)
    return _extractor
