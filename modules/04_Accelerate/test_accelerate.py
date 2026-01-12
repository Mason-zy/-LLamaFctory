# -*- coding: utf-8 -*-
"""
Accelerate 测试脚本
作者: zhouzhiyong
说明: 演示 Accelerate 的核心 API 使用（单卡/双卡通用）
"""

import torch
from accelerate import Accelerator

print("=" * 60)
print("Accelerate 测试脚本")
print("=" * 60)

# 初始化 Accelerator
accelerator = Accelerator()

# 创建简单模型
model = torch.nn.Linear(10, 10)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
dataloader = torch.utils.data.DataLoader(
    torch.randn(100, 10), batch_size=10
)

# 核心魔法：prepare()
# 自动将 model、optimizer、dataloader 分配到正确的设备
model, optimizer, dataloader = accelerator.prepare(
    model, optimizer, dataloader
)

# 训练循环
for epoch in range(2):
    for batch in dataloader:
        outputs = model(batch)
        loss = outputs.sum()

        # 替换 loss.backward()
        # 自动处理多卡梯度同步
        accelerator.backward(loss)

        optimizer.step()
        optimizer.zero_grad()

    # 只在主进程打印（避免重复输出）
    if accelerator.is_main_process:
        print(f"Epoch {epoch} completed")

# 输出配置信息
print(f"\n{'=' * 60}")
print(f"使用设备: {accelerator.device}")
print(f"进程索引: {accelerator.process_index}")
print(f"进程总数: {accelerator.num_processes}")
print(f"混合精度: {accelerator.mixed_precision}")
print(f"梯度累积步数: {accelerator.gradient_accumulation_steps}")
print("=" * 60)
