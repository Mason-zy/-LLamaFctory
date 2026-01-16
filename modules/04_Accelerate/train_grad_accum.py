# Author: zhouzhiyong
# Day 6: Accelerate 实战训练 - 梯度累积实验
# 创建日期: 2026-01-13

import time
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader, TensorDataset

# 梯度累积步数（可调整：1, 2, 4, 8）
GRADIENT_ACCUMULATION_STEPS = 4

print("=" * 60)
print("Accelerate 梯度累积实验")
print("=" * 60)

start_time = time.time()

# 初始化 Accelerator（指定梯度累积）
accelerator = Accelerator(gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS)

# 数据集
torch.manual_seed(42)
X = torch.randn(50000, 1024)
y = torch.randint(0, 100, (50000,))
dataset = TensorDataset(X, y)
batch_size = 64  # 小 batch，通过累积模拟大 batch
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# 模型（32M 参数）
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

total_params = sum(p.numel() for p in model.parameters())
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

num_epochs = 5
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

# 计算有效 batch size
effective_batch = batch_size * accelerator.num_processes * GRADIENT_ACCUMULATION_STEPS

if accelerator.is_main_process:
    print(f"\n配置信息:")
    print(f"  - 模型参数量: {total_params/1e6:.1f}M")
    print(f"  - 设备: {accelerator.device}")
    print(f"  - 进程数: {accelerator.num_processes}")
    print(f"  - 混合精度: {accelerator.mixed_precision}")
    print(f"  - 微 Batch Size: {batch_size}")
    print(f"  - 梯度累积步数: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"  - 有效 Batch Size: {batch_size} × {accelerator.num_processes} × {GRADIENT_ACCUMULATION_STEPS} = {effective_batch}")
    print()

# 训练循环
model.train()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    num_batches = 0

    for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
        # 关键：使用 accumulate 上下文管理器
        with accelerator.accumulate(model):
            outputs = model(X_batch)
            loss = F.cross_entropy(outputs, y_batch)
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss += loss.detach()
        num_batches += 1

    if accelerator.is_main_process:
        avg_loss = epoch_loss.item() / num_batches
        print(f"Epoch {epoch+1:2d}/{num_epochs} | Loss: {avg_loss:.4f}")

end_time = time.time()
training_time = end_time - start_time

if accelerator.is_main_process:
    print()
    print("=" * 60)
    print("训练完成!")
    print("=" * 60)
    print(f"  - 梯度累积步数: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"  - 有效 Batch Size: {effective_batch}")
    print(f"  - 总训练时间: {training_time:.2f} 秒")
    print(f"  - 混合精度: {accelerator.mixed_precision}")
    print("=" * 60)
