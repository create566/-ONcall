"""
熔断器实现
保护系统免受级联故障影响
"""

import time
from typing import Callable, Any, Optional, Dict
from enum import Enum
from loguru import logger


class CircuitState(Enum):
    """熔断器状态"""
    CLOSED = "closed"      # 关闭状态，正常调用
    OPEN = "open"          # 打开状态，拒绝调用
    HALF_OPEN = "half_open"  # 半开状态，尝试恢复


class CircuitBreaker:
    """
    熔断器类

    保护系统免受级联故障影响，当调用失败次数超过阈值时触发熔断，
    熔断期间拒绝所有请求，一段时间后尝试部分请求看是否恢复
    """

    def __init__(
        self,
        name: str = "default",
        max_failures: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 3
    ):
        """
        初始化熔断器

        Args:
            name: 熔断器名称
            max_failures: 最大失败次数，达到此次数后触发熔断
            recovery_timeout: 恢复超时时间（秒），熔断后等待此时间后进入半开状态
            half_open_max_calls: 半开状态下允许的最大调用次数
        """
        self.name = name
        self.max_failures = max_failures
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._state = CircuitState.CLOSED
        self._half_open_calls = 0

        logger.info(
            f"CircuitBreaker '{name}' 初始化完成: "
            f"max_failures={max_failures}, recovery_timeout={recovery_timeout}s"
        )

    @property
    def state(self) -> CircuitState:
        """获取当前熔断器状态"""
        if self._state == CircuitState.OPEN:
            # 检查是否应该转换到半开状态
            if self._last_failure_time:
                elapsed = time.time() - self._last_failure_time
                if elapsed >= self.recovery_timeout:
                    logger.info(f"CircuitBreaker '{self.name}' 从 OPEN 转换到 HALF_OPEN")
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
        return self._state

    def record_failure(self) -> None:
        """记录一次失败调用"""
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._state == CircuitState.HALF_OPEN:
            # 半开状态下失败，重新打开熔断器
            logger.warning(
                f"CircuitBreaker '{self.name}' 在 HALF_OPEN 状态下失败，"
                f"重新打开熔断器 (失败次数: {self._failure_count}/{self.max_failures})"
            )
            self._state = CircuitState.OPEN

        elif self._failure_count >= self.max_failures:
            # 达到最大失败次数，触发熔断
            logger.warning(
                f"CircuitBreaker '{self.name}' 触发熔断！"
                f"失败次数: {self._failure_count}/{self.max_failures}"
            )
            self._state = CircuitState.OPEN

    def record_success(self) -> None:
        """记录一次成功调用"""
        if self._state == CircuitState.HALF_OPEN:
            self._half_open_calls += 1
            if self._half_open_calls >= self.half_open_max_calls:
                # 半开状态下成功调用达到阈值，关闭熔断器
                logger.info(
                    f"CircuitBreaker '{self.name}' 恢复成功，从 HALF_OPEN 转换到 CLOSED"
                )
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._half_open_calls = 0
        elif self._state == CircuitState.CLOSED:
            # 成功调用，重置失败计数（缓慢恢复）
            self._failure_count = max(0, self._failure_count - 1)

    def allow_request(self) -> bool:
        """检查是否允许请求通过"""
        current_state = self.state

        if current_state == CircuitState.CLOSED:
            return True

        if current_state == CircuitState.OPEN:
            return False

        if current_state == CircuitState.HALF_OPEN:
            return self._half_open_calls < self.half_open_max_calls

        return False

    def get_degraded_message(self) -> str:
        """获取熔断时的降级消息"""
        return (
            "服务暂时不可用，由于连续调用失败，触发了熔断保护机制。"
            "请稍后重试，或联系管理员检查服务状态。"
        )

    async def call(
        self,
        func: Callable,
        *args,
        fallback_value: Any = None,
        **kwargs
    ) -> Any:
        """
        通过熔断器执行调用

        Args:
            func: 要调用的函数
            *args: 位置参数
            fallback_value: 熔断时的降级返回值
            **kwargs: 关键字参数

        Returns:
            函数执行结果，或熔断时的降级值
        """
        if not self.allow_request():
            logger.warning(
                f"CircuitBreaker '{self.name}' 拒绝请求，当前状态: {self.state.value}"
            )
            return fallback_value if fallback_value is not None else self.get_degraded_message()

        try:
            # 执行调用
            if callable(func):
                result = await func(*args, **kwargs)
            else:
                # 已经是协程
                result = await func

            # 记录成功
            self.record_success()
            return result

        except Exception as e:
            # 记录失败
            self.record_failure()
            logger.error(f"CircuitBreaker '{self.name}' 调用失败: {e}")
            return fallback_value if fallback_value is not None else self.get_degraded_message()

    def reset(self) -> None:
        """重置熔断器"""
        logger.info(f"CircuitBreaker '{self.name}' 重置")
        self._failure_count = 0
        self._last_failure_time = None
        self._state = CircuitState.CLOSED
        self._half_open_calls = 0

    def get_status(self) -> Dict[str, Any]:
        """获取熔断器状态信息"""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "max_failures": self.max_failures,
            "recovery_timeout": self.recovery_timeout,
            "last_failure_time": self._last_failure_time,
        }