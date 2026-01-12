# -*- coding: utf-8 -*-
"""
vLLM 多轮对话示例
作者: zhouzhiyong
说明: 演示如何管理和维护多轮对话的上下文
"""

import requests
import json


class ChatSession:
    """对话会话类：管理对话历史"""

    def __init__(self, model_path="/home/zzy/weitiao/models/Qwen2.5-7B-Instruct",
                 base_url="http://36.155.142.146:8000", temperature=0.7):
        """
        初始化对话会话

        Args:
            model_path: 模型路径
            base_url: vLLM 服务地址
            temperature: 温度参数
        """
        self.model_path = model_path
        self.base_url = base_url
        self.temperature = temperature
        self.messages = []  # 对话历史（关键！）

    def chat(self, user_input, stream=False):
        """
        发送用户输入并获取回复

        Args:
            user_input: 用户输入
            stream: 是否使用流式输出

        Returns:
            assistant_reply: 助手回复
        """
        # 1. 添加用户消息到历史
        self.messages.append({
            "role": "user",
            "content": user_input
        })

        # 2. 构造请求数据（包含完整历史）
        url = f"{self.base_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        data = {
            "model": self.model_path,
            "messages": self.messages,  # 发送完整历史！
            "temperature": self.temperature,
            "stream": stream
        }

        try:
            if stream:
                # 流式输出
                response = requests.post(url, headers=headers, json=data, stream=True)
                response.raise_for_status()

                print("💬 Assistant: ", end="", flush=True)
                full_reply = ""

                for line in response.iter_lines():
                    if not line:
                        continue
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data_str = line[6:]
                        if data_str == '[DONE]':
                            break
                        try:
                            chunk = json.loads(data_str)
                            content = chunk['choices'][0]['delta'].get('content', '')
                            if content:
                                print(content, end="", flush=True)
                                full_reply += content
                        except json.JSONDecodeError:
                            continue

                print()  # 换行
                assistant_reply = full_reply

            else:
                # 非流式输出
                response = requests.post(url, headers=headers, json=data)
                response.raise_for_status()
                result = response.json()

                assistant_reply = result['choices'][0]['message']['content']
                print(f"💬 Assistant: {assistant_reply}")

            # 3. 添加助手回复到历史（重要！）
            self.messages.append({
                "role": "assistant",
                "content": assistant_reply
            })

            return assistant_reply

        except requests.exceptions.RequestException as e:
            print(f"❌ 请求错误: {e}")
            return None

    def clear_history(self):
        """清空对话历史"""
        self.messages = []
        print("🗑️  对话历史已清空")

    def get_history_length(self):
        """获取对话历史长度"""
        return len(self.messages)

    def show_history(self):
        """显示对话历史"""
        print("\n📜 对话历史:")
        print("=" * 60)
        for i, msg in enumerate(self.messages):
            role = msg["role"].upper()
            content = msg["content"]
            # 截断长内容
            if len(content) > 100:
                content = content[:100] + "..."
            print(f"{i+1}. [{role}] {content}")
        print("=" * 60)


def main():
    """主函数：演示多轮对话"""
    print("=" * 60)
    print("vLLM 多轮对话演示")
    print("=" * 60)
    print()

    # 创建对话会话
    session = ChatSession()

    # 多轮对话示例
    conversations = [
        "你好，我叫小明",
        "记住我的名字了吗？",
        "我叫什么名字？",  # 测试记忆
        "我今天要学习 Python",
        "我刚才说我要学什么？",  # 测试短期记忆
    ]

    for i, user_input in enumerate(conversations, 1):
        print(f"\n【第 {i} 轮对话】")
        print(f"👤 User: {user_input}")

        session.chat(user_input, stream=True)

        # 显示当前历史长度
        print(f"📊 当前历史记录数: {session.get_history_length()} 条")

    print("\n" + "=" * 60)
    print("对话结束，显示完整历史")
    print("=" * 60)
    session.show_history()

    print("\n" + "=" * 60)
    print("清空历史，开始新对话")
    print("=" * 60)
    session.clear_history()
    session.chat("你好，还记得我叫什么名字吗？", stream=True)

    print("\n" + "=" * 60)
    print("演示结束")
    print("=" * 60)


if __name__ == "__main__":
    main()
