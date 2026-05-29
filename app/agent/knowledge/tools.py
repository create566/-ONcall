"""Knowledge Agent 专用工具

通过 MCP 客户端获取外部工具：
- exa-web-search: 网络搜索
- firecrawl: 网页爬取
"""

from typing import List
from langchain_core.tools import BaseTool
from loguru import logger

from app.agent.mcp_client import get_mcp_client_with_retry


async def get_knowledge_tools() -> List[BaseTool]:
    """
    获取 Knowledge Agent 专用的 MCP 工具

    包括：
    - exa-web-search: 网络搜索工具
    - firecrawl: 网页爬取工具

    Returns:
        List[BaseTool]: MCP 工具列表
    """
    try:
        mcp_client = await get_mcp_client_with_retry()
        tools = await mcp_client.get_tools()

        # 过滤出 Knowledge Agent 需要的工具
        knowledge_tool_names = {"exa-web-search", "firecrawl"}
        knowledge_tools = [
            tool for tool in tools
            if hasattr(tool, "name") and tool.name in knowledge_tool_names
        ]

        found_names = [t.name for t in knowledge_tools]
        logger.info(f"加载 Knowledge Agent 工具: {found_names}")

        return knowledge_tools

    except Exception as e:
        logger.error(f"获取 Knowledge Agent 工具失败: {e}")
        return []