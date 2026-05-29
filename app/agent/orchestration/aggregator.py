"""
结果聚合节点
将多个 Agent 的执行结果聚合成最终响应
"""

from typing import Dict, Any, Optional
from loguru import logger

from app.agent.orchestration.state import OrchestrationState


# 聚合器提示词
AGGREGATOR_PROMPT = """你是一个专业的智能助手，负责将多个 Agent 的执行结果聚合成完整的回答。

你需要聚合以下来源的信息：
{context}

聚合原则：
1. 综合各 Agent 的结果，提供完整、一致的回答
2. 如果多个结果有冲突，优先选择最新的结果，并说明冲突情况
3. 如果某个 Agent 执行失败或返回空，跳过该来源
4. 保持回答的逻辑性和连贯性
5. 如果确实无法从提供的信息中得出结论，明确告知用户

请基于以上信息，生成一个完整、有价值的回答。"""


async def aggregator(state: OrchestrationState) -> OrchestrationState:
    """
    聚合节点：将多个 Agent 的结果聚合成最终响应

    Args:
        state: 编排层状态

    Returns:
        OrchestrationState: 更新后的状态，包含最终响应
    """
    logger.info("执行聚合节点...")

    # 收集各 Agent 的结果
    knowledge_result = state.get("knowledge_result", "")
    ops_result = state.get("ops_result", "")
    doc_process_result = state.get("doc_process_result", "")

    # 构建上下文
    context_parts = []
    context_parts.append(f"【知识库检索结果】:\n{knowledge_result if knowledge_result else '(无)'}")

    if ops_result:
        context_parts.append(f"【运维操作结果】:\n{ops_result}")

    if doc_process_result:
        context_parts.append(f"【文档处理结果】:\n{doc_process_result}")

    context = "\n\n".join(context_parts)

    # 检查是否有任何结果
    if not any([knowledge_result, ops_result, doc_process_result]):
        logger.warning("所有 Agent 结果为空，使用降级响应")
        state["final_response"] = "抱歉，暂时无法处理您的请求，请稍后重试。"
        return state

    # 构建聚合提示词
    prompt = AGGREGATOR_PROMPT.format(context=context)

    # 调用 LLM 进行聚合
    try:
        from langchain_qwq import ChatQwen
        from app.config import config

        llm = ChatQwen(
            model=config.dashscope_model,
            api_key=config.dashscope_api_key,
            temperature=0.7,
        )

        from langchain_core.messages import HumanMessage
        messages = [
            ("user", prompt)
        ]

        response = await llm.ainvoke(messages)
        final_response = response.content if hasattr(response, 'content') else str(response)

        logger.info("聚合完成")
        state["final_response"] = final_response

    except Exception as e:
        logger.error(f"聚合失败: {e}")
        # 降级策略：直接拼接各结果
        fallback_response = f"""根据搜索和处理结果：

{knowledge_result or ''}

{doc_process_result or ''}

注：部分服务暂时不可用，以上为可用的结果。"""
        state["final_response"] = fallback_response

    return state


async def aggregator_simple(state: OrchestrationState) -> OrchestrationState:
    """
    简单聚合器：直接拼接各 Agent 结果，不调用 LLM

    Args:
        state: 编排层状态

    Returns:
        OrchestrationState: 更新后的状态
    """
    logger.info("执行简单聚合...")

    knowledge_result = state.get("knowledge_result", "")
    ops_result = state.get("ops_result", "")
    doc_process_result = state.get("doc_process_result", "")

    # 直接拼接结果
    parts = []

    if knowledge_result:
        parts.append(f"【知识库】\n{knowledge_result}")

    if ops_result:
        parts.append(f"【运维】\n{ops_result}")

    if doc_process_result:
        parts.append(f"【文档】\n{doc_process_result}")

    if parts:
        state["final_response"] = "\n\n---\n\n".join(parts)
    else:
        state["final_response"] = "抱歉，暂时无法处理您的请求，请稍后重试。"

    return state