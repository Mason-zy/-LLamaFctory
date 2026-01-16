# 02 - Accelerate + DeepSpeed

> 使用 Accelerate 封装，代码更简洁

## 文件说明

| 文件 | 作用 |
|------|------|
| `train.py` | 训练代码（使用 Accelerator API） |
| `ds_config.json` | DeepSpeed 配置（ZeRO、混合精度） |
| `accelerate_config.yaml` | Accelerate 配置（GPU、指向 ds_config） |
| `run.sh` | 启动脚本 |

## 配置文件关系

```
accelerate_config.yaml
    ├── gpu_ids: "6,7"                      ← GPU 选择
    ├── distributed_type: DEEPSPEED         ← 使用 DeepSpeed
    └── deepspeed_config_file: ds_config.json  ← 指向 DeepSpeed 配置
                    │
                    ▼
            ds_config.json
                ├── stage: 2
                ├── bf16: true
                └── batch_size
```

## 运行方法

```bash
cd 02_accelerate_deepspeed

# 方法 1: accelerate launch（推荐）
accelerate launch --config_file accelerate_config.yaml train.py

# 方法 2: deepspeed 直接启动（效果一样）
deepspeed --include localhost:6,7 train.py --deepspeed_config ds_config.json
```

## 代码核心

```python
from accelerate import Accelerator

# 1. 初始化（自动检测 DeepSpeed）
accelerator = Accelerator()

# 2. prepare（自动应用 ZeRO）
model, optimizer, dataloader = accelerator.prepare(...)

# 3. 训练（统一 API）
for batch in dataloader:
    loss = model(batch)
    accelerator.backward(loss)  # 统一 API
    optimizer.step()            # 正常用法
```

## 与 01 的区别

| 对比 | 01 原生 DeepSpeed | 02 Accelerate |
|------|-------------------|---------------|
| 初始化 | `deepspeed.init_distributed()` | `Accelerator()` |
| 反向传播 | `model_engine.backward(loss)` | `accelerator.backward(loss)` |
| 参数更新 | `model_engine.step()` | `optimizer.step()` |
| 代码量 | 多 | 少 |
| 灵活性 | 高 | 中 |
