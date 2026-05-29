"""
意图路由节点
根据用户输入判断应该调用哪个Agent
"""

from typing import Dict, Any, Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_qwq import ChatQwen
from pydantic import BaseModel, Field
from loguru import logger

from app.config import config


class Intent(BaseModel):
    """意图识别输出格式"""
    intent: Literal["knowledge", "ops", "doc_process", "mixed"] = Field(
        description="""用户意图类型：
        - 'knowledge': 需要知识库检索/问答
        - 'ops': 需要执行运维操作
        - 'doc_process': 需要处理文档
        - 'mixed': 混合意图，需要多种能力配合"""
    )
    confidence: float = Field(
        description="意图识别置信度，范围 0-1",
        default=0.0
    )
    reason: str = Field(
        description="意图判断的理由",
        default=""
    )


# 路由提示词
router_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一个意图识别专家，负责分析用户输入并判断应该调用哪个Agent。

你需要识别以下四种意图类型：
1. **knowledge** - 知识库检索/问答
   - 用户询问知识、文档内容、历史问题等
   - 示例："查找XXX的文档"、"XXX是如何配置的"、"介绍一下XXX"

2. **ops** - 运维操作
   - 用户需要执行具体的运维操作
   - 示例："重启服务"、"查看日志"、"部署应用"、"扩容"

3. **doc_process** - 文档处理
   - 用户需要处理文档，如生成报告、转换格式等
   - 示例："生成周报"、"导出PDF"、"汇总文档"

4. **mixed** - 混合意图
   - 需要多种能力配合完成
   - 示例："先查文档，再执行操作"

判断标准：
- 如果用户输入同时涉及多种意图，识别为 'mixed'
- 如果不确定用户意图，选择最可能的一种
- 优先识别为 'knowledge'（知识库问答是最常见的场景）

请输出JSON格式的意图结果。"""
        ),
        ("placeholder", "{messages}"),
    ]
)


class Router:
    """意图路由类"""

    def __init__(self):
        """初始化路由"""
        self.llm = ChatQwen(
            model=config.rag_model,
            api_key=config.dashscope_api_key,
            temperature=0
        )
        self.chain = router_prompt | self.llm.with_structured_output(Intent)
        logger.info("Router 初始化完成")

    async def route(self, user_input: str) -> Intent:
        """
        识别用户意图

        Args:
            user_input: 用户输入

        Returns:
            Intent: 意图识别结果
        """
        logger.info(f"开始识别意图: {user_input}")

        try:
            result = await self.chain.ainvoke({
                "messages": [("user", user_input)]
            })

            if isinstance(result, Intent):
                logger.info(f"意图识别结果: {result.intent} (置信度: {result.confidence})")
                return result
            else:
                # 默认返回 knowledge
                logger.warning(f"意图识别返回格式异常，使用默认值: knowledge")
                return Intent(intent="knowledge", confidence=0.0, reason="默认 fallback")

        except Exception as e:
            logger.error(f"意图识别失败: {e}")
            # 降级返回 knowledge
            return Intent(intent="knowledge", confidence=0.0, reason=f"识别失败: {str(e)}")


# 全局单例
router = Router()