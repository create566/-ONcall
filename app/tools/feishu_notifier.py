"""飞书通知工具

支持报警推送到飞书群
"""

import requests
from typing import Optional
from loguru import logger

from app.config import config


class FeishuNotifier:
    """飞书通知器"""

    def __init__(self, webhook_url: str = None):
        """
        初始化飞书通知器

        Args:
            webhook_url: 飞书群机器人的 Webhook 地址，默认使用配置
        """
        self.webhook_url = webhook_url or config.feishu_webhook_url
        self.enabled = config.feishu_enabled
        if self.enabled:
            logger.info(f"FeishuNotifier 初始化完成，Webhook: {self.webhook_url[:50]}...")

    def send_text(self, text: str) -> bool:
        """
        发送文本消息到飞书群

        Args:
            text: 消息内容

        Returns:
            bool: 是否发送成功
        """
        if not self.enabled:
            logger.debug("飞书通知未启用，跳过发送")
            return False

        try:
            payload = {
                "msg_type": "text",
                "content": {"text": text}
            }
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            result = response.json()

            if result.get("code") == 0:
                logger.info(f"飞书消息发送成功: {text[:50]}...")
                return True
            else:
                logger.error(f"飞书消息发送失败: {result}")
                return False

        except Exception as e:
            logger.error(f"飞书消息发送异常: {e}")
            return False

    def send_markdown(self, title: str, content: str) -> bool:
        """
        发送 Markdown 消息到飞书群

        Args:
            title: 标题
            content: 内容（支持 Markdown 格式）

        Returns:
            bool: 是否发送成功
        """
        try:
            payload = {
                "msg_type": "interactive",
                "card": {
                    "header": {
                        "title": {"tag": "plain_text", "content": title},
                        "template": "red"  # 红色警报，可选 red/orange/yellow/green/blue/purple/gray
                    },
                    "elements": [
                        {
                            "tag": "markdown",
                            "content": content
                        }
                    ]
                }
            }
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            result = response.json()

            if result.get("code") == 0:
                logger.info(f"飞书 Markdown 消息发送成功: {title}")
                return True
            else:
                logger.error(f"飞书 Markdown 消息发送失败: {result}")
                return False

        except Exception as e:
            logger.error(f"飞书 Markdown 消息发送异常: {e}")
            return False

    def send_aiops_alert(self, session_id: str, report: str) -> bool:
        """
        发送 AIOps 报警到飞书群

        Args:
            session_id: 会话 ID
            report: 诊断报告内容

        Returns:
            bool: 是否发送成功
        """
        content = f"""**AIOps 故障诊断报告**
**会话 ID**: `{session_id}`

**诊断结果**:
{report}

---
*由智能 OnCall 系统自动发送*"""

        return self.send_markdown("🔴 AIOps 报警", content)


# 全局通知器（需要外部设置）
_feishu_notifier: Optional[FeishuNotifier] = None


def init_feishu_notifier(webhook_url: str):
    """初始化全局飞书通知器"""
    global _feishu_notifier
    _feishu_notifier = FeishuNotifier(webhook_url)
    logger.info("飞书通知器初始化完成")


def get_feishu_notifier() -> Optional[FeishuNotifier]:
    """获取全局飞书通知器"""
    return _feishu_notifier


def send_alert(session_id: str, report: str) -> bool:
    """发送 AIOps 报警（快捷方法）"""
    if _feishu_notifier is None:
        logger.warning("飞书通知器未初始化，跳过报警")
        return False
    return _feishu_notifier.send_aiops_alert(session_id, report)