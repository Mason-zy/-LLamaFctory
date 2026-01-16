# DeepSpeed 学习模块

> 大模型显存优化的终极利器，ZeRO 算法让小显存跑大模型

## 学习路径

```
01_deepspeed_basic/        ← 第一步：最简单的 DeepSpeed 用法
       ↓
02_accelerate_deepspeed/   ← 第二步：Accelerate + DeepSpeed
       ↓
03_zero_stages/            ← 第三步：ZeRO 1/2/3 显存对比
```

## 目录说明

| 目录 | 学什么 | 关键点 |
|------|--------|--------|
| `01_deepspeed_basic` | DeepSpeed 原生用法 | `deepspeed.initialize()` |
| `02_accelerate_deepspeed` | Accelerate 封装 | `Accelerator()` + 配置文件关系 |
| `03_zero_stages` | ZeRO 三阶段对比 | 显存占用实测 |

## 核心概念速查

### ZeRO 三阶段

| 阶段 | 切分内容 | 显存节省 | 适用场景 |
|------|----------|----------|----------|
| ZeRO-1 | 优化器状态 | ~4× | 默认首选 |
| ZeRO-2 | +梯度 | ~8× | 大多数微调 |
| ZeRO-3 | +模型参数 | 线性扩展 | 超大模型 |

### 两种启动方式

```bash
# 方式 1: DeepSpeed 直接启动
deepspeed --include localhost:6,7 train.py --deepspeed_config ds_config.json

# 方式 2: Accelerate 启动
accelerate launch --config_file accelerate_config.yaml train.py
```

### 配置文件关系

```
【DeepSpeed 直接启动】
只需要 ds_config.json，GPU 在命令行指定

【Accelerate 启动】
accelerate_config.yaml
    ├── gpu_ids: "6,7"              ← GPU 选择
    ├── distributed_type: DEEPSPEED
    └── deepspeed_config_file: ds_config.json  ← 指向 DeepSpeed 配置
```

## 快速开始

```bash
# 1. 进入目录
cd modules/05_DeepSpeed

# 2. 从最简单的开始
cd 01_deepspeed_basic
deepspeed --include localhost:6,7 train.py

# 3. 监控显存
watch -n 1 nvidia-smi
```

## 参考资源

- [DeepSpeed GitHub](https://github.com/microsoft/DeepSpeed)
- [DeepSpeed 官方文档](https://www.deepspeed.ai/)
- [ZeRO 论文](https://arxiv.org/abs/1910.02054)
