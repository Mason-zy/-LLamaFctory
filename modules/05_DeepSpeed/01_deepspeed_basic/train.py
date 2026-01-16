# DeepSpeed 原生用法（不依赖 Accelerate）
# 这是最基础的 DeepSpeed 使用方式
# 模型参数: ~300M，适合 4090 测试

import os
import torch
import torch.nn.functional as F
import deepspeed
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# 1. 分布式初始化（DeepSpeed 要求）
# ============================================================
deepspeed.init_distributed()
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)

print(f"[Rank {local_rank}] 初始化完成")

# ============================================================
# 2. 准备数据
# ============================================================
torch.manual_seed(42)
X = torch.randn(20000, 2048)  # 20K 样本，2048 维
y = torch.randint(0, 100, (20000,))
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# ============================================================
# 3. 定义模型（~300M 参数）
# ============================================================
model = torch.nn.Sequential(
    torch.nn.Linear(2048, 4096),   # 8M
    torch.nn.ReLU(),
    torch.nn.Dropout(0.1),
    torch.nn.Linear(4096, 8192),   # 33M
    torch.nn.ReLU(),
    torch.nn.Dropout(0.1),
    torch.nn.Linear(8192, 8192),   # 67M
    torch.nn.ReLU(),
    torch.nn.Dropout(0.1),
    torch.nn.Linear(8192, 8192),   # 67M
    torch.nn.ReLU(),
    torch.nn.Dropout(0.1),
    torch.nn.Linear(8192, 8192),   # 67M
    torch.nn.ReLU(),
    torch.nn.Dropout(0.1),
    torch.nn.Linear(8192, 4096),   # 33M
    torch.nn.ReLU(),
    torch.nn.Linear(4096, 100)     # 0.4M
)                                  # 总计: ~275M

total_params = sum(p.numel() for p in model.parameters())
if local_rank == 0:
    print(f"模型参数量: {total_params/1e6:.1f}M")
    print(f"FP32 显存占用: {total_params * 4 / 1024**2:.1f} MB")

# ============================================================
# 4. DeepSpeed 初始化（核心！）
# ============================================================
# deepspeed.initialize() 会：
#   - 包装模型（应用 ZeRO 优化）
#   - 创建优化器（根据 ds_config.json）
#   - 处理混合精度
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config="ds_config.json"  # 配置文件路径
)

# ============================================================
# 5. 训练循环
# ============================================================
num_epochs = 3
model_engine.train()

for epoch in range(num_epochs):
    total_loss = 0.0
    for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
        # 数据移到 GPU
        X_batch = X_batch.to(model_engine.device)
        y_batch = y_batch.to(model_engine.device)

        # 前向传播
        outputs = model_engine(X_batch)
        loss = F.cross_entropy(outputs, y_batch)

        # 反向传播（DeepSpeed 特有 API）
        model_engine.backward(loss)

        # 更新参数（DeepSpeed 特有 API）
        model_engine.step()

        total_loss += loss.item()

    if local_rank == 0:
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")

if local_rank == 0:
    print("\n训练完成!")
    print("提示: 使用 watch -n 1 nvidia-smi 查看显存占用")
