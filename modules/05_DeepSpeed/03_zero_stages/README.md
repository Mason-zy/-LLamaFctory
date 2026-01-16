# 03 - ZeRO 1/2/3 显存对比

> 对比 ZeRO 三个阶段的显存占用

## 文件说明

| 文件 | 作用 |
|------|------|
| `train.py` | 训练代码（自动识别当前 ZeRO 阶段） |
| `ds_zero1.json` | ZeRO-1 配置（只切分优化器） |
| `ds_zero2.json` | ZeRO-2 配置（切分优化器+梯度） |
| `ds_zero3.json` | ZeRO-3 配置（切分全部） |
| `accelerate_zero*.yaml` | 对应的 Accelerate 配置 |
| `compare.sh` | 依次运行三个配置的对比脚本 |

## 运行方法

### 方法 1: 使用对比脚本

```bash
cd 03_zero_stages

# 终端 1: 监控显存
watch -n 1 nvidia-smi

# 终端 2: 运行对比
bash compare.sh
```

### 方法 2: 单独测试

```bash
# ZeRO-1
accelerate launch --config_file accelerate_zero1.yaml train.py

# ZeRO-2
accelerate launch --config_file accelerate_zero2.yaml train.py

# ZeRO-3
accelerate launch --config_file accelerate_zero3.yaml train.py
```

## ZeRO 三阶段对比

| 阶段 | 切分内容 | 显存节省 | 通信开销 | 适用场景 |
|------|----------|----------|----------|----------|
| ZeRO-1 | 优化器状态 | ~4× | 无增加 | 默认首选 |
| ZeRO-2 | +梯度 | ~8× | 略有增加 | 大多数微调 |
| ZeRO-3 | +模型参数 | 线性扩展 | 明显增加 | 超大模型 |

## 预期结果（300M 模型，双卡）

| 配置 | 预期显存/卡 | 说明 |
|------|-------------|------|
| ZeRO-1 | ~6-8 GB | 优化器状态切分 |
| ZeRO-2 | ~4-6 GB | 优化器+梯度切分 |
| ZeRO-3 | ~3-4 GB | 全部切分 |

实际数值请运行后记录到下表:

| 配置 | 实测显存/卡 | 训练时间 |
|------|-------------|----------|
| ZeRO-1 | ___ GB | ___ 秒 |
| ZeRO-2 | ___ GB | ___ 秒 |
| ZeRO-3 | ___ GB | ___ 秒 |
