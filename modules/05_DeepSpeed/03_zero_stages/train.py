# ZeRO 1/2/3 显存对比测试
# 使用同一个模型，对比不同 ZeRO 阶段的显存占用
# 模型参数: ~300M

import os
import sys
import time
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# 1. 初始化
# ============================================================
accelerator = Accelerator()

# 获取当前 ZeRO 阶段（从配置文件名推断）
config_file = os.environ.get("ACCELERATE_CONFIG_FILE", "unknown")
if "zero1" in config_file:
    zero_stage = "ZeRO-1"
elif "zero2" in config_file:
    zero_stage = "ZeRO-2"
elif "zero3" in config_file:
    zero_stage = "ZeRO-3"
else:
    zero_stage = "Unknown"

if accelerator.is_main_process:
    print("=" * 60)
    print(f"ZeRO 显存对比测试 - {zero_stage}")
    print("=" * 60)

start_time = time.time()

# ============================================================
# 2. 准备数据
# ============================================================
torch.manual_seed(42)
X = torch.randn(20000, 2048)
y = torch.randint(0, 100, (20000,))
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# ============================================================
# 3. 定义模型（~300M 参数）
# ============================================================
model = torch.nn.Sequential(
    torch.nn.Linear(2048, 4096),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.1),
    torch.nn.Linear(4096, 8192),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.1),
    torch.nn.Linear(8192, 8192),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.1),
    torch.nn.Linear(8192, 8192),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.1),
    torch.nn.Linear(8192, 8192),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.1),
    torch.nn.Linear(8192, 4096),
    torch.nn.ReLU(),
    torch.nn.Linear(4096, 100)
)

total_params = sum(p.numel() for p in model.parameters())

# ============================================================
# 4. 优化器 & Prepare
# ============================================================
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

# ============================================================
# 5. 打印配置
# ============================================================
if accelerator.is_main_process:
    print(f"\n配置信息:")
    print(f"  - ZeRO 阶段: {zero_stage}")
    print(f"  - 模型参数量: {total_params/1e6:.1f}M")
    print(f"  - FP32 理论显存: {total_params * 4 / 1024**2:.1f} MB")
    print(f"  - 进程数: {accelerator.num_processes}")
    print(f"  - 混合精度: {accelerator.mixed_precision}")
    print()
    print("开始训练，请用 nvidia-smi 观察显存...")
    print()

# ============================================================
# 6. 训练循环
# ============================================================
num_epochs = 3
model.train()

for epoch in range(num_epochs):
    total_loss = 0.0
    for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
        outputs = model(X_batch)
        loss = F.cross_entropy(outputs, y_batch)

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.detach()

    if accelerator.is_main_process:
        avg_loss = total_loss.item() / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")

# ============================================================
# 7. 完成
# ============================================================
end_time = time.time()
training_time = end_time - start_time

if accelerator.is_main_process:
    print()
    print("=" * 60)
    print(f"{zero_stage} 测试完成!")
    print(f"  - 训练时间: {training_time:.2f} 秒")
    print("=" * 60)
    print()
    print("记录当前显存占用，然后运行下一个配置进行对比")
