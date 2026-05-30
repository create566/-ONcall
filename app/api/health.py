"""健康检查接口"""

from typing import Any
from fastapi import APIRouter
from app.config import config
from app.core.milvus_client import milvus_manager
from app.models.response import UnifiedResponse
from loguru import logger

router = APIRouter()


@router.get("/health")
async def health_check():
    """健康检查接口
    检查服务状态和数据库连接状态

    Returns:
        统一格式的健康检查结果
    """
    health_data: dict[str, Any] = {
        "service": config.app_name,
        "version": config.app_version,
        "status": "healthy"
    }

    try:
        milvus_healthy = milvus_manager.health_check()
        health_data["milvus"] = {
            "status": "connected" if milvus_healthy else "disconnected",
            "message": "Milvus 连接正常" if milvus_healthy else "Milvus 连接异常"
        }
    except Exception as e:
        logger.warning(f"Milvus 健康检查失败: {e}")
        health_data["milvus"] = {
            "status": "error",
            "message": f"Milvus 检查失败: {str(e)}"
        }

    if health_data["milvus"]["status"] != "connected":
        return UnifiedResponse.error(
            error_message="服务不可用: Milvus 连接异常",
            code=503
        )

    return UnifiedResponse.success(result=health_data)
