"""
Supervisor 主图 - 编排层核心
使用 StateGraph 创建编排图，负责任务分发和结果聚合
"""

import asyncio
from typing import AsyncGenerator, Dict, Any, Literal

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from loguru import logger

from app.agent.orchestration.state import OrchestrationState
from app.agent.orchestration.router import router, Intent
from app.agent.orchestration.circuit_breaker import CircuitBreaker
from app.agent.orchestration.aggregator import aggregator, aggregator_simple
from app.agent.knowledge.agent import knowledge_agent
from app.agent.doc_process.agent import doc_process_agent


# 节点名称常量
NODE_ROUTER = "router"
NODE_KNOWLEDGE_AGENT = "knowledge_agent"
NODE_OPS_AGENT = "ops_agent"
NODE_DOC_PROCESS_AGENT = "doc_process_agent"
NODE_AGGREGATOR = "aggregator"
NODE_RESPONSE = "response"

# 超时常量
SUPERVISOR_TIMEOUT = 30  # 30秒超时


class SupervisorService:
    """Supervisor 主图服务 - 编排层核心"""

    def __init__(self):
        """初始化 Supervisor 服务"""
        self.checkpointer = MemorySaver()

        # 初始化各 Agent 的熔断器
        self.knowledge_cb = CircuitBreaker(
            name="knowledge_agent",
            max_failures=5,
            recovery_timeout=60.0
        )
        self.ops_cb = CircuitBreaker(
            name="ops_agent",
            max_failures=3,
            recovery_timeout=60.0
        )
        self.doc_process_cb = CircuitBreaker(
            name="doc_process_agent",
            max_failures=5,
            recovery_timeout=60.0
        )

        self.graph = self._build_graph()
        logger.info("Supervisor 主图初始化完成")

    def _build_graph(self):
        """构建编排工作流图"""
        logger.info("构建 Supervisor 工作流图...")

        # 创建状态图
        workflow = StateGraph(OrchestrationState)

        # 添加节点
        workflow.add_node(NODE_ROUTER, self._router_node)
        workflow.add_node(NODE_KNOWLEDGE_AGENT, self._knowledge_agent_node)
        workflow.add_node(NODE_OPS_AGENT, self._ops_agent_node)
        workflow.add_node(NODE_DOC_PROCESS_AGENT, self._doc_process_agent_node)
        workflow.add_node(NODE_AGGREGATOR, self._aggregator_node)
        workflow.add_node(NODE_RESPONSE, self._response_node)

        # 设置入口点
        workflow.set_entry_point(NODE_ROUTER)

        # 路由节点的条件边：根据 intent 分发到对应 Agent
        def route_based_on_intent(state: OrchestrationState) -> Literal[
            NODE_KNOWLEDGE_AGENT,
            NODE_OPS_AGENT,
            NODE_DOC_PROCESS_AGENT,
            NODE_AGGREGATOR,  # mixed 场景直接到聚合器
        ]:
            """根据 intent 分发到对应 Agent"""
            intent = state.get("intent", "knowledge")
            logger.info(f"路由决策: intent={intent}")

            if intent == "knowledge":
                return NODE_KNOWLEDGE_AGENT
            elif intent == "ops":
                return NODE_OPS_AGENT
            elif intent == "doc_process":
                return NODE_DOC_PROCESS_AGENT
            elif intent == "mixed":
                # mixed 场景：先调用 knowledge，再调用 doc_process
                # 这里简化为直接到 aggregator，让 aggregator 收集 knowledge 和 doc_process 的结果
                return NODE_AGGREGATOR
            else:
                return NODE_AGGREGATOR

        workflow.add_conditional_edges(
            NODE_ROUTER,
            route_based_on_intent,
            {
                NODE_KNOWLEDGE_AGENT: NODE_KNOWLEDGE_AGENT,
                NODE_OPS_AGENT: NODE_OPS_AGENT,
                NODE_DOC_PROCESS_AGENT: NODE_DOC_PROCESS_AGENT,
                NODE_AGGREGATOR: NODE_AGGREGATOR,
            }
        )

        # Agent 节点都连接到聚合器
        workflow.add_edge(NODE_KNOWLEDGE_AGENT, NODE_AGGREGATOR)
        workflow.add_edge(NODE_OPS_AGENT, NODE_AGGREGATOR)
        workflow.add_edge(NODE_DOC_PROCESS_AGENT, NODE_AGGREGATOR)

        # 聚合器连接到响应节点
        workflow.add_edge(NODE_AGGREGATOR, NODE_RESPONSE)

        # 响应节点结束
        workflow.add_edge(NODE_RESPONSE, END)

        # 编译工作流
        compiled_graph = workflow.compile(checkpointer=self.checkpointer)

        logger.info("Supervisor 工作流图构建完成")
        return compiled_graph

    async def _router_node(self, state: OrchestrationState) -> OrchestrationState:
        """
        路由节点：识别用户意图

        Args:
            state: 编排层状态

        Returns:
            OrchestrationState: 更新后的状态，包含 intent
        """
        logger.info("执行路由节点...")

        user_input = state.get("user_input", "")
        if not user_input:
            logger.warning("用户输入为空")
            state["intent"] = "knowledge"
            return state

        try:
            # 调用 Router 识别意图
            intent_result: Intent = await router.route(user_input)
            state["intent"] = intent_result.intent
            logger.info(f"意图识别完成: {intent_result.intent} (置信度: {intent_result.confidence})")

        except Exception as e:
            logger.error(f"意图识别失败: {e}")
            state["intent"] = "knowledge"  # 降级为 knowledge

        return state

    async def _knowledge_agent_node(self, state: OrchestrationState) -> OrchestrationState:
        """
        Knowledge Agent 节点

        Args:
            state: 编排层状态

        Returns:
            OrchestrationState: 更新后的状态，包含 knowledge_result
        """
        logger.info("执行 Knowledge Agent 节点...")

        user_input = state.get("user_input", "")

        try:
            # 使用熔断器包装调用
            result = await self.knowledge_cb.call(
                knowledge_agent.query,
                question=user_input,
                session_id="supervisor",  # 可以用 user_input 的 hash 作为 session_id
            )

            state["knowledge_result"] = result
            logger.info("Knowledge Agent 执行完成")

        except Exception as e:
            logger.error(f"Knowledge Agent 执行失败: {e}")
            state["knowledge_result"] = f"知识库服务暂时不可用: {str(e)}"

        return state

    async def _ops_agent_node(self, state: OrchestrationState) -> OrchestrationState:
        """
        Ops Agent 节点（暂未实现，返回友好消息）

        Args:
            state: 编排层状态

        Returns:
            OrchestrationState: 更新后的状态，包含 ops_result
        """
        logger.info("执行 Ops Agent 节点...")

        # Ops Agent 暂不存在，使用降级消息
        logger.warning("Ops Agent 尚未实现，使用降级响应")

        fallback_message = (
            "运维操作功能正在开发中，暂时无法执行运维任务。\n"
            "您可以：\n"
            "1. 使用知识库查询运维文档\n"
            "2. 联系运维团队获取帮助"
        )

        state["ops_result"] = fallback_message

        return state

    async def _doc_process_agent_node(self, state: OrchestrationState) -> OrchestrationState:
        """
        DocProcess Agent 节点

        Args:
            state: 编排层状态

        Returns:
            OrchestrationState: 更新后的状态，包含 doc_process_result
        """
        logger.info("执行 DocProcess Agent 节点...")

        user_input = state.get("user_input", "")

        try:
            # 使用熔断器包装调用
            result = await self.doc_process_cb.call(
                doc_process_agent.query,
                question=user_input,
                session_id="supervisor",
            )

            state["doc_process_result"] = result
            logger.info("DocProcess Agent 执行完成")

        except Exception as e:
            logger.error(f"DocProcess Agent 执行失败: {e}")
            state["doc_process_result"] = f"文档处理服务暂时不可用: {str(e)}"

        return state

    async def _aggregator_node(self, state: OrchestrationState) -> OrchestrationState:
        """
        聚合器节点

        Args:
            state: 编排层状态

        Returns:
            OrchestrationState: 更新后的状态，包含 final_response
        """
        logger.info("执行聚合器节点...")

        try:
            # 调用聚合器
            state = await aggregator(state)
            logger.info("聚合完成")

        except Exception as e:
            logger.error(f"聚合失败: {e}")
            # 使用简单聚合器作为降级
            state = await aggregator_simple(state)

        return state

    async def _response_node(self, state: OrchestrationState) -> OrchestrationState:
        """
        最终响应节点（不做额外处理，仅记录日志）

        Args:
            state: 编排层状态

        Returns:
            OrchestrationState: 保持状态不变
        """
        logger.info("执行最终响应节点")

        final_response = state.get("final_response", "")
        if final_response:
            logger.info(f"最终响应长度: {len(final_response)} 字符")
        else:
            logger.warning("最终响应为空")

        return state

    async def execute(
        self,
        user_input: str,
        session_id: str = "default"
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        执行 Supervisor 编排流程

        Args:
            user_input: 用户输入
            session_id: 会话ID

        Yields:
            Dict[str, Any]: 流式事件
        """
        logger.info(f"[会话 {session_id}] 开始执行 Supervisor 流程: {user_input[:100]}...")

        try:
            # 初始化状态
            initial_state: OrchestrationState = {
                "user_input": user_input,
                "intent": None,
                "knowledge_result": None,
                "ops_result": None,
                "doc_process_result": None,
                "final_response": None
            }

            # 配置
            config_dict = {
                "configurable": {
                    "thread_id": session_id
                }
            }

            # 使用 asyncio.timeout 实现 30 秒超时
            try:
                async with asyncio.timeout(SUPERVISOR_TIMEOUT):
                    async for event in self.graph.astream(
                        input=initial_state,
                        config=config_dict,
                        stream_mode="values"
                    ):
                        # 解析事件
                        for node_name, node_output in event.items():
                            logger.info(f"节点 '{node_name}' 输出")

                            # 根据节点类型生成事件
                            if node_name == NODE_ROUTER:
                                intent = node_output.get("intent", "unknown") if node_output else "unknown"
                                yield {
                                    "type": "intent_detected",
                                    "stage": "router",
                                    "intent": intent,
                                    "message": f"意图识别完成: {intent}"
                                }

                            elif node_name in [NODE_KNOWLEDGE_AGENT, NODE_OPS_AGENT, NODE_DOC_PROCESS_AGENT]:
                                yield {
                                    "type": "agent_complete",
                                    "stage": node_name,
                                    "message": f"{node_name} 执行完成"
                                }

                            elif node_name == NODE_AGGREGATOR:
                                yield {
                                    "type": "aggregated",
                                    "stage": "aggregator",
                                    "message": "结果聚合完成"
                                }

                            elif node_name == NODE_RESPONSE:
                                final_response = node_output.get("final_response", "") if node_output else ""
                                yield {
                                    "type": "complete",
                                    "stage": "complete",
                                    "message": "处理完成",
                                    "response": final_response
                                }

            except asyncio.TimeoutError:
                logger.error(f"[会话 {session_id}] Supervisor 执行超时 ({SUPERVISOR_TIMEOUT}秒)")
                yield {
                    "type": "timeout",
                    "stage": "timeout",
                    "message": f"处理超时，请稍后重试（超时时间: {SUPERVISOR_TIMEOUT}秒）"
                }
                return

            logger.info(f"[会话 {session_id}] Supervisor 执行完成")

        except Exception as e:
            logger.error(f"[会话 {session_id}] Supervisor 执行失败: {e}", exc_info=True)
            yield {
                "type": "error",
                "stage": "error",
                "message": f"处理出错: {str(e)}"
            }

    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """获取各 Agent 熔断器状态"""
        return {
            "knowledge_agent": self.knowledge_cb.get_status(),
            "ops_agent": self.ops_cb.get_status(),
            "doc_process_agent": self.doc_process_cb.get_status(),
        }


# 全局单例
supervisor_service = SupervisorService()