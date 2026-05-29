"""Knowledge Agent - 基于 LangGraph 的外部知识搜索代理

使用 ReAct 模式，通过 exa-web-search 搜索外部知识，
使用 firecrawl 爬取网页内容。
"""

from typing import Annotated, Any, AsyncGenerator, Dict, Sequence

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages
from loguru import logger
from typing_extensions import TypedDict
from langchain_qwq import ChatQwen

from app.config import config
from app.agent.knowledge.tools import get_knowledge_tools


class KnowledgeAgentState(TypedDict):
    """Knowledge Agent 状态"""
    messages: Annotated[Sequence[BaseMessage], add_messages]


def trim_messages_middleware(state: KnowledgeAgentState) -> dict[str, Any] | None:
    """
    修剪消息历史，只保留最近的几条消息以适应上下文窗口

    策略：
    - 保留第一条系统消息（System Message）
    - 保留最近的 6 条消息（3 轮对话）
    - 当消息少于等于 7 条时，不做修剪

    Args:
        state: Agent 状态

    Returns:
        包含修剪后消息的字典，如果无需修剪则返回 None
    """
    messages = state["messages"]

    if len(messages) <= 7:
        return None

    first_msg = messages[0]
    recent_messages = messages[-6:] if len(messages) % 2 == 0 else messages[-7:]

    new_messages = [first_msg] + list(recent_messages)

    logger.debug(f"修剪消息历史: {len(messages)} -> {len(new_messages)} 条")

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }


class KnowledgeAgent:
    """Knowledge Agent - 用于搜索外部知识和爬取网页内容"""

    def __init__(self, streaming: bool = True):
        """初始化 Knowledge Agent

        Args:
            streaming: 是否启用流式输出，默认为 True
        """
        self.model_name = config.rag_model
        self.streaming = streaming
        self.system_prompt = self._build_system_prompt()

        self.model = ChatQwen(
            model=self.model_name,
            api_key=config.dashscope_api_key,
            temperature=0.7,
            streaming=streaming,
        )

        # 基础工具（暂无专用基础工具，MCP 工具在异步初始化时获取）
        self.tools = []

        # MCP 工具（延迟初始化）
        self.mcp_tools: list = []

        # 内存检查点
        self.checkpointer = MemorySaver()

        # Agent 初始化状态
        self.agent = None
        self._agent_initialized = False

        logger.info(f"Knowledge Agent 初始化完成 (ChatQwen), model={self.model_name}, streaming={streaming}")

    async def _initialize_agent(self):
        """异步初始化 Agent（包括 MCP 工具）"""
        if self._agent_initialized:
            return

        # 获取 MCP 工具
        self.mcp_tools = await get_knowledge_tools()
        logger.info(f"成功加载 {len(self.mcp_tools)} 个 MCP 工具")

        # 合并所有工具
        all_tools = self.tools + self.mcp_tools

        self.agent = create_react_agent(
            self.model,
            tools=all_tools,
            checkpointer=self.checkpointer,
        )

        self._agent_initialized = True

        if all_tools:
            tool_names = [tool.name if hasattr(tool, "name") else str(tool) for tool in all_tools]
            logger.info(f"Knowledge Agent 可用工具列表: {', '.join(tool_names)}")

    def _build_system_prompt(self) -> str:
        """构建系统提示词"""
        from textwrap import dedent

        return dedent("""
            你是一个专业的外部知识搜索助手，能够使用搜索和爬取工具来获取实时信息。

            工作能力：
            1. 使用 exa-web-search 搜索互联网上的最新信息、新闻、技术文档等
            2. 使用 firecrawl 爬取指定网页的完整内容
            3. 综合多个来源的信息，整理出准确、全面的回答

            工作原则：
            1. 理解用户需求，选择合适的搜索策略
            2. 搜索时尽量使用精准的关键词，提高搜索质量
            3. 爬取网页时注意提取关键信息，过滤无关内容
            4. 基于多个来源综合回答，避免单一来源的偏差
            5. 如有不确定的信息，明确标注来源

            回答要求：
            - 保持客观、专业的语气
            - 回答中包含信息来源，便于用户验证
            - 如果搜索或爬取失败，诚实地告知用户并尝试其他方法

            请根据用户的问题，灵活使用 exa-web-search 和 firecrawl 工具，提供有价值的回答。
        """).strip()

    async def query(
        self,
        question: str,
        session_id: str,
    ) -> str:
        """
        非流式处理用户问题

        Args:
            question: 用户问题
            session_id: 会话ID（作为 thread_id）

        Returns:
            str: 完整答案
        """
        try:
            await self._initialize_agent()

            logger.info(f"[会话 {session_id}] Knowledge Agent 收到查询（非流式）: {question}")

            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=question)
            ]

            agent_input = {"messages": messages}

            config_dict = {
                "configurable": {
                    "thread_id": session_id
                }
            }

            result = await self.agent.ainvoke(
                input=agent_input,
                config=config_dict,
            )

            messages_result = result.get("messages", [])
            if messages_result:
                last_message = messages_result[-1]
                answer = last_message.content if hasattr(last_message, 'content') else str(last_message)

                if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                    tool_names = [tc.get("name", "unknown") for tc in last_message.tool_calls]
                    logger.info(f"[会话 {session_id}] Knowledge Agent 调用了工具: {tool_names}")

                logger.info(f"[会话 {session_id}] Knowledge Agent 查询完成（非流式）")
                return answer

            logger.warning(f"[会话 {session_id}] Knowledge Agent 返回结果为空")
            return ""

        except Exception as e:
            logger.error(f"[会话 {session_id}] Knowledge Agent 查询失败（非流式）: {e}")
            raise

    async def query_stream(
        self,
        question: str,
        session_id: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        流式处理用户问题

        Args:
            question: 用户问题
            session_id: 会话ID（作为 thread_id）

        Yields:
            Dict[str, Any]: 包含流式数据的字典
        """
        try:
            await self._initialize_agent()

            logger.info(f"[会话 {session_id}] Knowledge Agent 收到查询（流式）: {question}")

            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=question)
            ]

            agent_input = {"messages": messages}

            config_dict = {
                "configurable": {
                    "thread_id": session_id
                }
            }

            async for token, metadata in self.agent.astream(
                input=agent_input,
                config=config_dict,
                stream_mode="messages",
            ):
                node_name = metadata.get('langgraph_node', 'unknown') if isinstance(metadata, dict) else 'unknown'
                message_type = type(token).__name__

                if message_type in ("AIMessage", "AIMessageChunk"):
                    content_blocks = getattr(token, 'content_blocks', None)

                    if content_blocks and isinstance(content_blocks, list):
                        for block in content_blocks:
                            if isinstance(block, dict) and block.get('type') == 'text':
                                text_content = block.get('text', '')
                                if text_content:
                                    yield {
                                        "type": "content",
                                        "data": text_content,
                                        "node": node_name
                                    }

            logger.info(f"[会话 {session_id}] Knowledge Agent 查询完成（流式）")
            yield {"type": "complete"}

        except Exception as e:
            logger.error(f"[会话 {session_id}] Knowledge Agent 查询失败（流式）: {e}")
            yield {
                "type": "error",
                "data": str(e)
            }
            raise

    def get_session_history(self, session_id: str) -> list:
        """获取会话历史"""
        try:
            config = {"configurable": {"thread_id": session_id}}

            checkpoint_tuple = self.checkpointer.get(config)

            if not checkpoint_tuple:
                logger.info(f"获取会话历史: {session_id}, 消息数量: 0")
                return []

            if hasattr(checkpoint_tuple, 'checkpoint'):
                checkpoint_data = checkpoint_tuple.checkpoint
            else:
                checkpoint_data = checkpoint_tuple[0] if checkpoint_tuple else {}

            messages = checkpoint_data.get("channel_values", {}).get("messages", [])

            history = []
            for msg in messages:
                if isinstance(msg, SystemMessage):
                    continue

                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                content = msg.content if hasattr(msg, 'content') else str(msg)

                from datetime import datetime
                history.append({
                    "role": role,
                    "content": content,
                    "timestamp": datetime.now().isoformat()
                })

            logger.info(f"获取会话历史: {session_id}, 消息数量: {len(history)}")
            return history

        except Exception as e:
            logger.error(f"获取会话历史失败: {session_id}, 错误: {e}")
            return []

    def clear_session(self, session_id: str) -> bool:
        """清空会话历史"""
        try:
            self.checkpointer.delete_thread(session_id)
            logger.info(f"已清除会话历史: {session_id}")
            return True

        except Exception as e:
            logger.error(f"清空会话历史失败: {session_id}, 错误: {e}")
            return False

    async def cleanup(self):
        """清理资源"""
        try:
            logger.info("清理 Knowledge Agent 资源...")
            logger.info("Knowledge Agent 资源已清理")
        except Exception as e:
            logger.error(f"清理资源失败: {e}")


# 全局单例 - 启用流式输出
knowledge_agent = KnowledgeAgent(streaming=True)