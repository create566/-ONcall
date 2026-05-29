"""
编排层 HTTP 接口

提供基于 Supervisor 的多 Agent 编排接口，支持普通调用和流式调用（SSE）
"""

import json
from typing import Optional
from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel, Field
from loguru import logger

from app.agent.orchestration.supervisor import supervisor_service

router = APIRouter()


class OrchestrateRequest(BaseModel):
    """编排请求"""

    user_input: str = Field(..., description="用户输入", alias="userInput")
    session_id: Optional[str] = Field(
        default="default",
        description="会话ID，用于追踪编排历史"
    )

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "userInput": "帮我查询最近的告警情况",
                "session_id": "session-123"
            }
        }


class OrchestrateResponse(BaseModel):
    """编排响应（非流式）"""

    code: int = 200
    message: str = "success"
    data: dict

    class Config:
        json_schema_extra = {
            "example": {
                "code": 200,
                "message": "success",
                "data": {
                    "success": True,
                    "intent": "knowledge",
                    "response": "这是查询结果..."
                }
            }
        }


@router.post("/orchestrate")
async def orchestrate(request: OrchestrateRequest):
    """多 Agent 编排接口（普通调用）

    请求体:
    {
        "userInput": "用户输入",
        "session_id": "可选会话ID"
    }

    响应:
    {
        "code": 200,
        "message": "success",
        "data": {
            "success": true,
            "intent": "knowledge|ops|doc_process|mixed",
            "response": "最终响应内容",
            "errorMessage": null
        }
    }

    Args:
        request: 编排请求

    Returns:
        统一格式的编排响应
    """
    session_id = request.session_id or "default"
    logger.info(f"[会话 {session_id}] 收到编排请求: {request.user_input[:50]}...")

    try:
        # 收集所有事件直到完成
        final_response = None
        intent = None
        error_message = None

        async for event in supervisor_service.execute(
            user_input=request.user_input,
            session_id=session_id
        ):
            event_type = event.get("type", "unknown")

            if event_type == "intent_detected":
                intent = event.get("intent")

            elif event_type == "complete":
                final_response = event.get("response", "")
                break

            elif event_type == "error":
                error_message = event.get("message", str(event))
                break

            elif event_type == "timeout":
                error_message = event.get("message", "处理超时")
                break

        if error_message:
            return {
                "code": 500,
                "message": "error",
                "data": {
                    "success": False,
                    "intent": intent,
                    "response": None,
                    "errorMessage": error_message
                }
            }

        logger.info(f"[会话 {session_id}] 编排完成，intent={intent}")

        return {
            "code": 200,
            "message": "success",
            "data": {
                "success": True,
                "intent": intent or "unknown",
                "response": final_response or "",
                "errorMessage": None
            }
        }

    except Exception as e:
        logger.error(f"[会话 {session_id}] 编排接口错误: {e}", exc_info=True)
        return {
            "code": 500,
            "message": "error",
            "data": {
                "success": False,
                "intent": None,
                "response": None,
                "errorMessage": str(e)
            }
        }


@router.post("/orchestrate_stream")
async def orchestrate_stream(request: OrchestrateRequest):
    """多 Agent 编排接口（流式 SSE）

    请求体:
    {
        "userInput": "用户输入",
        "session_id": "可选会话ID"
    }

    SSE 事件类型：

    1. `intent_detected` - 意图识别完成
       ```json
       {
         "type": "intent_detected",
         "stage": "router",
         "intent": "knowledge",
         "message": "意图识别完成: knowledge"
       }
       ```

    2. `agent_complete` - Agent 执行完成
       ```json
       {
         "type": "agent_complete",
         "stage": "knowledge_agent|ops_agent|doc_process_agent",
         "message": "xxx 执行完成"
       }
       ```

    3. `aggregated` - 结果聚合完成
       ```json
       {
         "type": "aggregated",
         "stage": "aggregator",
         "message": "结果聚合完成"
       }
       ```

    4. `content` - 内容块（中间响应）
       ```json
       {
         "type": "content",
         "data": "内容块"
       }
       ```

    5. `complete` - 处理完成
       ```json
       {
         "type": "complete",
         "stage": "complete",
         "message": "处理完成",
         "response": "最终响应内容"
       }
       ```

    6. `error` - 错误信息
       ```json
       {
         "type": "error",
         "stage": "error",
         "message": "处理出错: ..."
       }
       ```

    7. `timeout` - 超时信息
       ```json
       {
         "type": "timeout",
         "stage": "timeout",
         "message": "处理超时，请稍后重试"
       }
       ```

    Args:
        request: 编排请求

    Returns:
        SSE 事件流
    """
    session_id = request.session_id or "default"
    logger.info(f"[会话 {session_id}] 收到流式编排请求: {request.user_input[:50]}...")

    async def event_generator():
        try:
            async for event in supervisor_service.execute(
                user_input=request.user_input,
                session_id=session_id
            ):
                event_type = event.get("type", "unknown")

                # 意图检测事件
                if event_type == "intent_detected":
                    yield {
                        "event": "message",
                        "data": json.dumps({
                            "type": "intent_detected",
                            "stage": event.get("stage", "router"),
                            "intent": event.get("intent"),
                            "message": event.get("message")
                        }, ensure_ascii=False)
                    }

                # Agent 完成事件
                elif event_type == "agent_complete":
                    yield {
                        "event": "message",
                        "data": json.dumps({
                            "type": "agent_complete",
                            "stage": event.get("stage"),
                            "message": event.get("message")
                        }, ensure_ascii=False)
                    }

                # 聚合完成事件
                elif event_type == "aggregated":
                    yield {
                        "event": "message",
                        "data": json.dumps({
                            "type": "aggregated",
                            "stage": event.get("stage", "aggregator"),
                            "message": event.get("message")
                        }, ensure_ascii=False)
                    }

                # 内容块事件
                elif event_type == "content":
                    yield {
                        "event": "message",
                        "data": json.dumps({
                            "type": "content",
                            "data": event.get("data", "")
                        }, ensure_ascii=False)
                    }

                # 完成事件
                elif event_type == "complete":
                    yield {
                        "event": "message",
                        "data": json.dumps({
                            "type": "complete",
                            "stage": event.get("stage", "complete"),
                            "message": event.get("message"),
                            "response": event.get("response", "")
                        }, ensure_ascii=False)
                    }

                # 错误事件
                elif event_type == "error":
                    yield {
                        "event": "message",
                        "data": json.dumps({
                            "type": "error",
                            "stage": event.get("stage", "error"),
                            "message": event.get("message")
                        }, ensure_ascii=False)
                    }

                # 超时事件
                elif event_type == "timeout":
                    yield {
                        "event": "message",
                        "data": json.dumps({
                            "type": "timeout",
                            "stage": event.get("stage", "timeout"),
                            "message": event.get("message")
                        }, ensure_ascii=False)
                    }

                # 完成或错误事件，结束流
                if event_type in ["complete", "error", "timeout"]:
                    break

            logger.info(f"[会话 {session_id}] 流式编排完成")

        except Exception as e:
            logger.error(f"[会话 {session_id}] 流式编排异常: {e}", exc_info=True)
            yield {
                "event": "message",
                "data": json.dumps({
                    "type": "error",
                    "stage": "exception",
                    "message": f"编排异常: {str(e)}"
                }, ensure_ascii=False)
            }

    return EventSourceResponse(event_generator())


@router.get("/orchestrate/circuit_breaker")
async def get_circuit_breaker_status():
    """获取各 Agent 熔断器状态

    Returns:
        各 Agent 熔断器状态信息
    """
    try:
        status = supervisor_service.get_circuit_breaker_status()
        return {
            "code": 200,
            "message": "success",
            "data": status
        }
    except Exception as e:
        logger.error(f"获取熔断器状态错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))