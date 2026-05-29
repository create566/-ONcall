"""飞书机器人客户端（使用官方 SDK）

支持长连接接收消息和自动回复
"""

import asyncio
from typing import Optional, Callable, Awaitable
from loguru import logger

from app.config import config


class FeishuBotListener:
    """飞书机器人消息监听器

    使用官方 lark-oapi SDK 的 WebSocket 模式接收消息
    """

    def __init__(self, message_handler: Optional[Callable[[str, str], Awaitable[str]]] = None):
        """
        初始化飞书机器人

        Args:
            message_handler: 消息处理函数，接受 (open_id, text)，返回回复文本
        """
        self.message_handler = message_handler
        self.app_id = config.feishu_app_id
        self.app_secret = config.feishu_app_secret
        self.client = None
        self.event_cache = []
        logger.info("FeishuBotListener 初始化完成")

    def start_websocket(self):
        """启动 WebSocket 长连接

        这是一个同步方法，实际在后台运行
        """
        try:
            import lark_oapi as lark

            # 创建事件处理器
            class EventHandler(lark.EventDispatcherHandler):
                async def on_im_message_receive_v1(self, data: lark.im.message.receive_v1.Event) -> None:
                    """处理接收消息事件"""
                    try:
                        message = data.event.message
                        sender = data.event.sender

                        message_type = message.message_type
                        content_str = message.content

                        # 解析消息内容
                        import json
                        try:
                            content_obj = json.loads(content_str)
                        except:
                            content_obj = {}

                        text = content_obj.get("text", "").strip()
                        open_id = sender.open_id if sender else ""

                        if not text or not open_id:
                            return

                        logger.info(f"收到飞书消息: open_id={open_id}, text={text[:50]}...")

                        # 处理消息并获取回复
                        if self.message_handler:
                            reply_text = await self.message_handler(open_id, text)
                            if reply_text:
                                # 回复消息
                                message_id = message.message_id
                                if message_id:
                                    await self._reply_message(message_id, reply_text)
                    except Exception as e:
                        logger.error(f"处理飞书消息异常: {e}", exc_info=True)

            # 创建客户端
            self.client = lark.WsClient(self.app_id, self.app_secret, EventHandler.builder())

            # 启动 WebSocket 连接
            self.client.start()

        except ImportError:
            logger.error("请先安装 lark-oapi: pip install lark-oapi>=1.3.0")
            raise
        except Exception as e:
            logger.error(f"启动飞书 WebSocket 失败: {e}", exc_info=True)
            raise

    async def _reply_message(self, message_id: str, text: str) -> bool:
        """回复消息"""
        try:
            import lark_oapi as lark

            client = lark.Client(self.app_id, self.app_secret)

            response = client.im.message.reply(
                lark.im.message.ReplyMessageRequest(
                    message_id=message_id,
                    body=lark.im.message.ReplyMessageRequestBody(
                        msg_type="text",
                        content=lark.utils.JSONUtil.to_json({"text": text})
                    )
                )
            )

            if response.code == 0:
                logger.info(f"飞书消息回复成功: {message_id}")
                return True
            else:
                logger.error(f"飞书消息回复失败: {response}")
                return False

        except Exception as e:
            logger.error(f"回复飞书消息异常: {e}", exc_info=True)
            return False


# 全局机器人实例
_feishu_bot_listener: Optional[FeishuBotListener] = None


def init_feishu_bot_ws(message_handler: Optional[Callable[[str, str], Awaitable[str]]] = None):
    """初始化全局飞书机器人（WebSocket 模式）"""
    global _feishu_bot_listener
    _feishu_bot_listener = FeishuBotListener(message_handler)
    return _feishu_bot_listener


def get_feishu_bot_ws() -> Optional[FeishuBotListener]:
    """获取全局飞书机器人"""
    return _feishu_bot_listener


def start_feishu_ws_client(message_handler: Optional[Callable[[str, str], Awaitable[str]]] = None):
    """启动飞书 WebSocket 客户端（便捷方法）"""
    bot = init_feishu_bot_ws(message_handler)
    bot.start_websocket()
    return bot