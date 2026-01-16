# Accelerate + DeepSpeed 用法
# 代码更简洁，Accelerator() 自动检测 DeepSpeed 环境
# 模型参数: ~300M，适合 4090 测试

import time
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader, TensorDataset

print("=" * 60)
print("Accelerate + DeepSpeed 训练")
print("=" * 60)

start_time = time.time()

# ============================================================
# 1. 初始化 Accelerator（自动检测 DeepSpeed）
# ============================================================
accelerator = Accelerator()

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
# 4. 优化器
# ============================================================
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# ============================================================
# 5. Accelerate prepare（核心！）
# ============================================================
# accelerator.prepare() 会自动：
#   - 检测是否有 DeepSpeed 环境
#   - 如果有，调用 deepspeed.initialize()
#   - 如果没有，使用普通 DDP
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

# ============================================================
# 6. 打印配置
# ============================================================
if accelerator.is_main_process:
    print(f"\n配置信息:")
    print(f"  - 模型参数量: {total_params/1e6:.1f}M")
    print(f"  - 设备: {accelerator.device}")
    print(f"  - 进程数: {accelerator.num_processes}")
    print(f"  - 混合精度: {accelerator.mixed_precision}")
    print(f"  - 分布式类型: {accelerator.distributed_type}")
    print()

# ============================================================
# 7. 训练循环
# ============================================================
num_epochs = 3
model.train()

for epoch in range(num_epochs):
    total_loss = 0.0
    for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
        outputs = model(X_batch)
        loss = F.cross_entropy(outputs, y_batch)

        # 统一 API（不管有没有 DeepSpeed）
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.detach()

    if accelerator.is_main_process:
        avg_loss = total_loss.item() / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")

# ============================================================
# 8. 完成
# ============================================================
end_time = time.time()

if accelerator.is_main_process:
    print()
    print("=" * 60)
    print("训练完成!")
    print(f"  - 总时间: {end_time - start_time:.2f} 秒")
    print("=" * 60)
