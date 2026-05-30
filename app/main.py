"""FastAPI 应用入口

主应用程序，配置路由、中间件、静态文件等
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import os

from app.config import config
from loguru import logger
from app.api import chat, health, file, aiops, orchestration, doc_process
from app.core.milvus_client import milvus_manager
from app.tools.feishu_ws_bot import start_feishu_ws_client
from app.services.rag_agent_service import rag_agent_service


async def handle_feishu_message(open_id: str, text: str) -> str:
    """处理飞书消息，调用 RAG Agent"""
    try:
        logger.info(f"处理飞书消息: open_id={open_id}, text={text}")
        answer = await rag_agent_service.query(text, session_id=open_id)
        return answer if answer else "处理中，请稍候..."
    except Exception as e:
        logger.error(f"处理飞书消息失败: {e}")
        return "抱歉，处理消息时出现错误，请稍后再试。"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    logger.info("=" * 60)
    logger.info(f"🚀 {config.app_name} v{config.app_version} 启动中...")
    logger.info(f"📝 环境: {'开发' if config.debug else '生产'}")
    logger.info(f"🌐 监听地址: http://{config.host}:{config.port}")
    logger.info(f"📚 API 文档: http://{config.host}:{config.port}/docs")

    # 连接 Milvus
    logger.info("🔌 正在连接 Milvus...")
    milvus_manager.connect()
    logger.info("✅ Milvus 连接成功")

    # 预热 MCP 客户端
    logger.info("🔄 正在预热 MCP 客户端...")
    try:
        from app.agent.mcp_client import get_mcp_client_with_retry
        mcp_client = await get_mcp_client_with_retry()
        tools = await mcp_client.get_tools()
        logger.info(f"✅ MCP 客户端预热完成，加载了 {len(tools)} 个工具")
    except Exception as e:
        logger.warning(f"⚠️ MCP 客户端预热失败: {e}，将在首次使用时重试")

    # 启动飞书机器人
    if config.feishu_enabled:
        logger.info("🔔 正在启动飞书机器人...")
        try:
            start_feishu_ws_client(handle_feishu_message)
            logger.info("✅ 飞书机器人启动成功")
        except Exception as e:
            logger.warning(f"⚠️ 飞书机器人启动失败: {e}，将继续启动其他服务")

    logger.info("=" * 60)

    yield

    # 关闭时执行
    logger.info("🔌 正在关闭 Milvus 连接...")
    milvus_manager.close()
    logger.info(f"👋 {config.app_name} 关闭")


# 创建 FastAPI 应用
app = FastAPI(
    title=config.app_name,
    version=config.app_version,
    description="基于 LangChain 的智能oncall运维系统",
    lifespan=lifespan
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(health.router, tags=["健康检查"])
app.include_router(chat.router, prefix="/api", tags=["对话"])
app.include_router(file.router, prefix="/api", tags=["文件管理"])
app.include_router(aiops.router, prefix="/api", tags=["AIOps智能运维"])
app.include_router(orchestration.router, prefix="/api", tags=["编排服务"])
app.include_router(doc_process.router, prefix="/api", tags=["文档处理"])

# 挂载静态文件
static_dir = "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def root():
    """返回首页"""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "message": f"Welcome to {config.app_name} API",
        "version": config.app_version,
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
        log_level="info"
    )
