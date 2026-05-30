"""配置管理模块

使用 Pydantic Settings 实现类型安全的配置管理
"""

from typing import Dict, Any
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # 应用配置
    app_name: str = "SuperBizAgent"
    app_version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 9900

    # DashScope 配置
    dashscope_api_key: str = "sk-4c69e976439c463690cfaa583b11ba33"  # 默认空字符串，实际使用需从环境变量加载
    dashscope_model: str = "qwen-max"
    dashscope_embedding_model: str = "text-embedding-v1"  # v4 支持多种维度（默认 1024）

    # Milvus 配置
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_timeout: int = 10000  # 毫秒

    # RAG 配置
    rag_top_k: int = 3
    rag_model: str = "qwen-max"  # 使用快速响应模型，不带扩展思考

    # 文档分块配置
    chunk_max_size: int = 800
    chunk_overlap: int = 100

    # MCP 服务配置
    mcp_cls_transport: str = "streamable-http"
    mcp_cls_url: str = "http://localhost:8003/mcp"
    mcp_monitor_transport: str = "streamable-http"
    mcp_monitor_url: str = "http://localhost:8004/mcp"

    # 新增 MCP 配置
    mcp_supabase_transport: str = "streamable-http"
    mcp_supabase_url: str = "https://uhkijolcmoxjkoznhpvl.supabase.co"
    mcp_filesystem_path: str = r"D:\桌面\智能仓库"

    # Exa Web Search 配置
    mcp_exa_api_key: str = "d50aee63-4d8c-450c-b86a-8c8c447dff5e"

    # Firecrawl 配置
    mcp_firecrawl_api_key: str = "fc-9ddd198f89da44b4a9681f33ad719c0c"

    # 飞书配置
    feishu_webhook_url: str = "https://open.feishu.cn/open-apis/bot/v2/hook/dccf310b-0184-4c2a-9944-6a36febc8a27"
    feishu_enabled: bool = True
    feishu_app_id: str = "cli_aa93d9202afa5bd9"
    feishu_app_secret: str = "4muhsqpI7Rno2YOdofE3tbx0rtMiHOhO"

    @property
    def mcp_servers(self) -> Dict[str, Dict[str, Any]]:
        """获取完整的 MCP 服务器配置"""
        servers = {
            "cls": {
                "transport": self.mcp_cls_transport,
                "url": self.mcp_cls_url,
            },
            "monitor": {
                "transport": self.mcp_monitor_transport,
                "url": self.mcp_monitor_url,
            },
            "supabase": {
                "transport": self.mcp_supabase_transport,
                "url": self.mcp_supabase_url,
            },
            "exa-web-search": {
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "exa-mcp-server"],
            },
            "firecrawl": {
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "firecrawl-mcp"],
                "env": {
                    "FIRECRAWL_API_KEY": config.mcp_firecrawl_api_key,
                },
            },
            "filesystem": {
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", self.mcp_filesystem_path],
            },
        }
        return servers


# 全局配置实例
config = Settings()
