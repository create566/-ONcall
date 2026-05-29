"""
编排层状态定义
定义全局统一的状态结构
"""

from typing import TypedDict, Optional, Any


class OrchestrationState(TypedDict, total=False):
    """
    编排层统一状态定义

    包含用户输入、意图识别结果、各Agent执行结果和最终响应
    """

    # 用户输入（任务描述）
    user_input: str

    # 意图识别结果 (knowledge/ops/doc_process/mixed)
    intent: Optional[str]

    # 知识库检索结果
    knowledge_result: Optional[Any]

    # 运维操作执行结果
    ops_result: Optional[Any]

    # 文档处理结果
    doc_process_result: Optional[Any]

    # 最终响应/报告
    final_response: Optional[str]