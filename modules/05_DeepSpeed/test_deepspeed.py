# Author: zhouzhiyong
# Day 7: DeepSpeed ZeRO 显存对比测试
# 创建日期: 2026-01-13

import time
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader, TensorDataset

print("=" * 60)
print("DeepSpeed ZeRO 显存对比测试")
print("=" * 60)

start_time = time.time()

# 初始化 Accelerator (自动读取 DeepSpeed 配置)
accelerator = Accelerator()

# 创建更大的数据集（测试显存占用）
torch.manual_seed(42)
X = torch.randn(100000, 2048)  # 100K 样本，2048 维
y = torch.randint(0, 100, (100000,))
dataset = TensorDataset(X, y)

# 注意：batch_size 会被 ds_config.json 覆盖
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

# 创建大模型（100M 参数，约 400MB FP32）
model = torch.nn.Sequential(
    torch.nn.Linear(2048, 4096),
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
    torch.nn.Linear(1024, 100)
)

total_params = sum(p.numel() for p in model.parameters())

# 优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# 学习率调度器
num_epochs = 3
num_training_steps = num_epochs * len(dataloader)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_training_steps)

# Accelerate prepare（自动应用 DeepSpeed）
model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, dataloader, lr_scheduler
)

# 打印配置
if accelerator.is_main_process:
    print(f"\n配置信息:")
    print(f"  - 模型参数量: {total_params/1e6:.1f}M")
    print(f"  - FP32 显存占用: {total_params * 4 / 1024 / 1024:.1f} MB")
    print(f"  - 设备: {accelerator.device}")
    print(f"  - 进程数: {accelerator.num_processes}")
    print(f"  - 混合精度: {accelerator.mixed_precision}")
    print(f"  - 数据集大小: {len(dataset)}")
    print()

# 训练循环
model.train()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    num_batches = 0

    for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
        outputs = model(X_batch)
        loss = F.cross_entropy(outputs, y_batch)

        # DeepSpeed 自动处理梯度累积
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        epoch_loss += loss.detach()
        num_batches += 1

    if accelerator.is_main_process:
        avg_loss = epoch_loss.item() / num_batches
        print(f"Epoch {epoch+1:2d}/{num_epochs} | Loss: {avg_loss:.4f}")

# 训练结束
end_time = time.time()
training_time = end_time - start_time

if accelerator.is_main_process:
    print()
    print("=" * 60)
    print("训练完成!")
    print("=" * 60)
    print(f"  - 模型参数量: {total_params/1e6:.1f}M")
    print(f"  - 总训练时间: {training_time:.2f} 秒")
    print(f"  - 平均每轮时间: {training_time/num_epochs:.2f} 秒")
    print(f"  - DeepSpeed Stage: 见配置文件")
    print("=" * 60)
    print("\n提示: 使用 nvidia-smi 查看显存占用")
