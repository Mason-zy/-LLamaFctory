# 01 - DeepSpeed 原生用法

> 最简单的 DeepSpeed 使用方式，不依赖 Accelerate

## 文件说明

| 文件 | 作用 |
|------|------|
| `train.py` | 训练代码（使用 DeepSpeed 原生 API） |
| `ds_config.json` | DeepSpeed 配置（ZeRO stage、混合精度等） |
| `run.sh` | 启动脚本 |

## 运行方法

```bash
cd 01_deepspeed_basic

# 双卡运行
deepspeed --include localhost:6,7 train.py

# 单卡运行
deepspeed --include localhost:6 train.py
```

## 代码核心

```python
import deepspeed

# 1. 初始化分布式
deepspeed.init_distributed()

# 2. 初始化 DeepSpeed（自动应用 ZeRO 优化）
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config="ds_config.json"
)

# 3. 训练循环（使用 DeepSpeed 特有 API）
for batch in dataloader:
    loss = model_engine(batch)
    model_engine.backward(loss)  # 不是 loss.backward()
    model_engine.step()          # 不是 optimizer.step()
```

## 配置文件说明

```json
{
  "train_batch_size": 128,              // 全局 batch size
  "train_micro_batch_size_per_gpu": 64, // 每卡 batch size
  "zero_optimization": {"stage": 1},    // ZeRO 阶段
  "bf16": {"enabled": true}             // 混合精度
}
```
