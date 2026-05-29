"""飞书机器人客户端

支持消息接收和自动回复
"""

import asyncio
import hashlib
import time
import base64
import hmac
from typing import Optional, Callable, Awaitable
from loguru import logger

import requests

from app.config import config


class FeishuClient:
    """飞书客户端"""

    def __init__(self):
        self.app_id = config.feishu_app_id
        self.app_secret = config.feishu_app_secret
        self.api_base = "https://open.feishu.cn/open-apis"
        self._access_token: Optional[str] = None
        self._token_expires_at: float = 0

    async def get_access_token(self) -> str:
        """获取 access_token"""
        if self._access_token and time.time() < self._token_expires_at - 60:
            return self._access_token

        url = f"{self.api_base}/auth/v3/tenant_access_token/internal"
        payload = {
            "app_id": self.app_id,
            "app_secret": self.app_secret
        }

        response = requests.post(url, json=payload, timeout=10)
        result = response.json()

        if result.get("code") == 0:
            self._access_token = result["tenant_access_token"]
            # 提前 5 分钟过期
            self._token_expires_at = time.time() + result.get("expire", 7200) - 300
            logger.info("飞书 access_token 获取成功")
            return self._access_token
        else:
            raise Exception(f"获取 access_token 失败: {result}")

    async def send_message(self, receive_id: str, msg_type: str, content: dict) -> bool:
        """发送消息"""
        token = await self.get_access_token()
        url = f"{self.api_base}/im/v1/messages?receive_id_type=open_id"

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        payload = {
            "receive_id": receive_id,
            "msg_type": msg_type,
            "content": content
        }

        response = requests.post(url, headers=headers, json=payload, timeout=10)
        result = response.json()

        if result.get("code") == 0:
            logger.info(f"飞书消息发送成功: {receive_id}")
            return True
        else:
            logger.error(f"飞书消息发送失败: {result}")
            return False

    async def reply_message(self, message_id: str, msg_type: str, content: dict) -> bool:
        """回复消息"""
        token = await self.get_access_token()
        url = f"{self.api_base}/im/v1/messages/{message_id}/reply"

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        payload = {
            "msg_type": msg_type,
            "content": content
        }

        response = requests.post(url, headers=headers, json=payload, timeout=10)
        result = response.json()

        if result.get("code") == 0:
            logger.info(f"飞书消息回复成功: {message_id}")
            return True
        else:
            logger.error(f"飞书消息回复失败: {result}")
            return False

    async def send_text_to_user(self, open_id: str, text: str) -> bool:
        """发送文本消息给用户"""
        return await self.send_message(
            receive_id=open_id,
            msg_type="text",
            content={"text": text}
        )

    async def send_text_to_group(self, group_id: str, text: str) -> bool:
        """发送文本消息到群"""
        return await self.send_message(
            receive_id=group_id,
            msg_type="text",
            content={"text": text}
        )


class FeishuBot:
    """飞书机器人

    用于接收消息并自动回复
    """

    def __init__(self, message_handler: Optional[Callable[[str, str], Awaitable[str]]] = None):
        """
        初始化飞书机器人

        Args:
            message_handler: 消息处理函数，接受 (open_id, text)，返回回复文本
        """
        self.client = FeishuClient()
        self.message_handler = message_handler
        logger.info("FeishuBot 初始化完成")

    def verify_webhook(self, challenge: str, verification_token: str) -> dict:
        """验证 Webhook

        用于接收事件订阅的验证请求
        """
        return {
            "challenge": challenge
        }

    async def handle_event(self, event_data: dict) -> Optional[dict]:
        """处理事件"""
        event_type = event_data.get("event", {}).get("type")

        if event_type == "im.message.receive_v1":
            # 收到消息事件
            message = event_data.get("event", {}).get("message", {})
            sender = event_data.get("event", {}).get("sender", {})

            message_type = message.get("message_type")
            content = message.get("content", "{}")

            # 解析消息内容
            import json
            try:
                content_obj = json.loads(content)
            except:
                content_obj = {}

            text = content_obj.get("text", "").strip()
            open_id = sender.get("open_id", "")

            logger.info(f"收到飞书消息: open_id={open_id}, text={text[:50]}...")

            # 处理消息并获取回复
            if self.message_handler:
                reply_text = await self.message_handler(open_id, text)
                if reply_text:
                    # 回复消息
                    message_id = message.get("message_id")
                    if message_id:
                        await self.client.reply_message(
                            message_id=message_id,
                            msg_type="text",
                            content={"text": reply_text}
                        )
            return {"code": 0}

        return None

    async def register_webhook_events(self) -> bool:
        """注册 Webhook 事件

        需要在飞书开放平台配置 Webhook 地址
        """
        token = await self.client.get_access_token()
        url = f"{self.client.api_base}/event/v1/outbound_bot/websocket"

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        response = requests.get(url, headers=headers, timeout=10)
        result = response.json()

        if result.get("code") == 0:
            logger.info("飞书 Webhook 事件注册成功")
            return True
        else:
            logger.error(f"飞书 Webhook 事件注册失败: {result}")
            return False


# 全局机器人实例
_feishu_bot: Optional[FeishuBot] = None


def init_feishu_bot(message_handler: Optional[Callable[[str, str], Awaitable[str]]] = None):
    """初始化全局飞书机器人"""
    global _feishu_bot
    _feishu_bot = FeishuBot(message_handler)
    return _feishu_bot


def get_feishu_bot() -> Optional[FeishuBot]:
    """获取全局飞书机器人"""
    return _feishu_bot