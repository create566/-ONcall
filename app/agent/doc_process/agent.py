"""文档处理 Agent - 基于 LangGraph ReAct 模式的智能代理

提供文档读取、写入、列表和向量化功能
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
from langgraph.graph.message import add_messages
from loguru import logger
from typing_extensions import TypedDict
from langchain_qwq import ChatQwen

from app.config import config
from app.agent.doc_process.tools import (
    read_document,
    write_document,
    list_documents,
    vectorize_document,
    search_vectorized_docs,
)
from app.agent.mcp_client import get_mcp_client_with_retry


class DocProcessState(TypedDict):
    """DocProcess Agent 状态"""
    messages: Annotated[Sequence[BaseMessage], add_messages]


def trim_messages_middleware(state: DocProcessState) -> dict[str, Any] | None:
    """
    修剪消息历史，只保留最近的几条消息以适应上下文窗口

    策略：
    - 保留第一条系统消息（System Message）
    - 保留最近的 6 条消息（3 轮对话）
    - 当消息少于等于 7 条时，不做修剪
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
            RemoveMessage(id="remove_all"),
            *new_messages
        ]
    }


class DocProcessAgent:
    """文档处理 Agent - 使用 LangGraph + ChatQwen + ReAct 模式"""

    def __init__(self, streaming: bool = True):
        """初始化 DocProcess Agent

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

        # 定义文档处理工具
        self.tools = [
            read_document,
            write_document,
            list_documents,
            vectorize_document,
            search_vectorized_docs,
        ]

        # MCP 客户端工具
        self.mcp_tools: list = []

        # 内存检查点
        self.checkpointer = MemorySaver()

        # Agent 初始化标志
        self.agent = None
        self._agent_initialized = False

        logger.info(f"DocProcess Agent 初始化完成 (ChatQwen), model={self.model_name}, streaming={streaming}")

    async def _initialize_agent(self):
        """异步初始化 Agent（包括 MCP 工具）"""
        if self._agent_initialized:
            return

        # 使用全局 MCP 客户端管理器
        mcp_client = await get_mcp_client_with_retry()
        mcp_tools = await mcp_client.get_tools()
        logger.info(f"成功加载 {len(mcp_tools)} 个 MCP 工具")

        self.mcp_tools = mcp_tools

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
            logger.info(f"DocProcess Agent 可用工具列表: {', '.join(tool_names)}")

    def _build_system_prompt(self) -> str:
        """
        构建系统提示词

        Returns:
            str: 系统提示词
        """
        from textwrap import dedent

        return dedent("""
            你是一个专业的文档处理助手，可以帮助用户完成以下任务：

            1. 读取文档：读取 D:\\桌面\\智能仓库 目录下的 txt, md, docx, pdf 文件
            2. 写入文档：创建或修改 txt, md 文件
            3. 列出文档：查看目录下有哪些文档文件
            4. 向量化文档：将文档内容分割并存储到 Milvus 向量数据库
            5. 搜索文档：在已向量化的文档中搜索相关内容

            工作原则:
            1. 理解用户需求，选择合适的工具来完成任务
            2. 读取文件时，返回文件的核心内容（不包含原始路径信息）
            3. 向量化文档时，自动进行文档分割和向量化存储
            4. 如果操作失败，清晰地说明错误原因

            文件根目录: D:\\桌面\\智能仓库
            支持的文件格式: txt, md, docx, pdf

            请根据用户的需求，灵活使用可用工具完成任务。
        """).strip()

    async def query(
        self,
        question: str,
        session_id: str,
    ) -> str:
        """
        非流式处理用户问题（一次性返回完整答案）

        Args:
            question: 用户问题
            session_id: 会话ID（作为 thread_id）

        Returns:
            str: 完整答案
        """
        try:
            await self._initialize_agent()

            logger.info(f"[会话 {session_id}] DocProcess Agent 收到查询（非流式）: {question}")

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
                    logger.info(f"[会话 {session_id}] Agent 调用了工具: {tool_names}")

                logger.info(f"[会话 {session_id}] DocProcess Agent 查询完成（非流式）")
                return answer

            logger.warning(f"[会话 {session_id}] Agent 返回结果为空")
            return ""

        except Exception as e:
            logger.error(f"[会话 {session_id}] DocProcess Agent 查询失败（非流式）: {e}")
            raise

    async def query_stream(
        self,
        question: str,
        session_id: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        流式处理用户问题（逐步返回答案片段）

        Args:
            question: 用户问题
            session_id: 会话ID（作为 thread_id）

        Yields:
            Dict[str, Any]: 包含流式数据的字典
        """
        try:
            await self._initialize_agent()

            logger.info(f"[会话 {session_id}] DocProcess Agent 收到查询（流式）: {question}")

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

            logger.info(f"[会话 {session_id}] DocProcess Agent 查询完成（流式）")
            yield {"type": "complete"}

        except Exception as e:
            logger.error(f"[会话 {session_id}] DocProcess Agent 查询失败（流式）: {e}")
            yield {
                "type": "error",
                "data": str(e)
            }
            raise

    def get_session_history(self, session_id: str) -> list:
        """
        获取会话历史

        Args:
            session_id: 会话ID

        Returns:
            list: 消息历史列表
        """
        try:
            checkpointer_config = {"configurable": {"thread_id": session_id}}
            checkpoint_tuple = self.checkpointer.get(checkpointer_config)

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
                    "timestamp": getattr(msg, 'timestamp', datetime.now().isoformat())
                })

            logger.info(f"获取会话历史: {session_id}, 消息数量: {len(history)}")
            return history

        except Exception as e:
            logger.error(f"获取会话历史失败: {session_id}, 错误: {e}")
            return []

    def clear_session(self, session_id: str) -> bool:
        """
        清空会话历史

        Args:
            session_id: 会话ID

        Returns:
            bool: 是否成功
        """
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
            logger.info("清理 DocProcess Agent 资源...")
            logger.info("DocProcess Agent 资源已清理")
        except Exception as e:
            logger.error(f"清理资源失败: {e}")


# 全局单例 - 启用流式输出
doc_process_agent = DocProcessAgent(streaming=True)