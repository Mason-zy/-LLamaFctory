# Author: zhouzhiyong
# Day 6: Accelerate 实战训练 - 大模型加速比验证
# 创建日期: 2026-01-13

import time
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader, TensorDataset

print("=" * 60)
print("Accelerate 大模型加速比验证")
print("=" * 60)

# 记录开始时间
start_time = time.time()

# 初始化 Accelerator
accelerator = Accelerator()

# 创建更大的数据集
torch.manual_seed(42)
X = torch.randn(50000, 1024)  # 50000 样本，1024 维特征
y = torch.randint(0, 100, (50000,))  # 100 类分类
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4)

# 创建更大的模型（6 层 MLP，约 50M 参数）
model = torch.nn.Sequential(
    torch.nn.Linear(1024, 4096),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.1),
    torch.nn.Linear(4096, 4096),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.1),
    torch.nn.Linear(4096, 2048),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.1),
    torch.nn.Linear(2048, 1024),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.1),
    torch.nn.Linear(1024, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 100)
)

# 计算参数量
total_params = sum(p.numel() for p in model.parameters())

# 优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# 学习率调度器
num_epochs = 5
num_training_steps = num_epochs * len(dataloader)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_training_steps)

# Accelerate 核心：prepare()
model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, dataloader, lr_scheduler
)

# 只在主进程打印配置信息
if accelerator.is_main_process:
    print(f"\n配置信息:")
    print(f"  - 模型参数量: {total_params/1e6:.1f}M")
    print(f"  - 设备: {accelerator.device}")
    print(f"  - 进程数: {accelerator.num_processes}")
    print(f"  - 混合精度: {accelerator.mixed_precision}")
    print(f"  - 数据集大小: {len(dataset)}")
    print(f"  - Batch Size: 256 × {accelerator.num_processes} = {256 * accelerator.num_processes}")
    print(f"  - 训练轮数: {num_epochs}")
    print()

# 训练循环
model.train()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
        # 前向传播
        outputs = model(X_batch)
        loss = F.cross_entropy(outputs, y_batch)

        # 反向传播
        accelerator.backward(loss)

        # 优化器步骤
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        # 统计
        epoch_loss += loss.detach()
        _, predicted = outputs.max(1)
        total += y_batch.size(0)
        correct += predicted.eq(y_batch).sum().detach()

    # 只在主进程打印
    if accelerator.is_main_process:
        avg_loss = epoch_loss.item() / len(dataloader)
        accuracy = 100. * correct.item() / total
        print(f"Epoch {epoch+1:2d}/{num_epochs} | Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}%")

# 训练结束
end_time = time.time()
training_time = end_time - start_time

# 汇总结果
if accelerator.is_main_process:
    print()
    print("=" * 60)
    print("训练完成!")
    print("=" * 60)
    print(f"  - 模型参数量: {total_params/1e6:.1f}M")
    print(f"  - 总训练时间: {training_time:.2f} 秒")
    print(f"  - 平均每轮时间: {training_time/num_epochs:.2f} 秒")
    print(f"  - 进程数: {accelerator.num_processes}")
    print(f"  - 混合精度: {accelerator.mixed_precision}")
    print("=" * 60)
