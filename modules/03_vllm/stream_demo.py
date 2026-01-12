# -*- coding: utf-8 -*-
"""
vLLM 流式输出示例
作者: zhouzhiyong
说明: 演示如何使用 vLLM 的流式 API 进行实时对话
"""

import requests
import json


def stream_chat(prompt, model_path="/home/zzy/weitiao/models/Qwen2.5-7B-Instruct",
                base_url="http://36.155.142.146:8000", temperature=0.7):
    """
    流式对话函数

    Args:
        prompt: 用户输入的问题
        model_path: 模型路径
        base_url: vLLM 服务地址（局域网 IP）
        temperature: 温度参数（0.0-1.0）

    Returns:
        完整回复文本
    """
    url = f"{base_url}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model_path,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "stream": True  # 关键参数：启用流式输出
    }

    print(f"🤖 Prompt: {prompt}")
    print(f"📡 Connecting to {base_url}...")
    print("💬 Assistant: ", end="", flush=True)

    full_response = ""

    try:
        # 发送 POST 请求，启用流式传输
        response = requests.post(url, headers=headers, json=data, stream=True)
        response.raise_for_status()

        # 逐行读取 SSE (Server-Sent Events) 格式
        for line in response.iter_lines():
            if not line:
                continue

            line = line.decode('utf-8')

            # SSE 格式：data: {...}
            if line.startswith('data: '):
                data_str = line[6:]  # 去掉 'data: ' 前缀

                # 结束标志
                if data_str == '[DONE]':
                    break

                try:
                    # 解析 JSON 数据
                    chunk = json.loads(data_str)
                    delta = chunk['choices'][0].get('delta', {})
                    content = delta.get('content', '')

                    if content:
                        print(content, end="", flush=True)  # 实时输出
                        full_response += content

                except json.JSONDecodeError:
                    continue

        print()  # 换行
        return full_response

    except requests.exceptions.RequestException as e:
        print(f"\n❌ 请求错误: {e}")
        return None


def main():
    """主函数：演示不同的对话场景"""
    print("=" * 60)
    print("vLLM 流式输出演示")
    print("=" * 60)
    print()

    # 测试场景
    test_prompts = [
        "用三个词描述深度学习",
        "什么是 Transformer 模型？",
        "写一首关于春天的短诗"
    ]

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n【测试 {i}/{len(test_prompts)}】")
        print("-" * 60)

        response = stream_chat(prompt)

        if response:
            print(f"\n✅ 完成！生成了 {len(response)} 个字符")

        print()

    print("=" * 60)
    print("演示结束")
    print("=" * 60)


if __name__ == "__main__":
    main()
