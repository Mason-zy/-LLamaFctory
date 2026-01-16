# DeepSpeed 学习计划

> 创建日期: 2026-01-13
> 更新日期: 2026-01-16

## 学习目标

- [x] 理解 ZeRO 三阶段原理
- [x] 掌握 DeepSpeed 原生用法
- [x] 掌握 Accelerate + DeepSpeed 配合
- [ ] 对比 ZeRO-1/2/3 的显存占用（待实测）

---

## 模块结构

```
05_DeepSpeed/
├── readme.md                    # 总览导航
├── PLAN.md                      # 本文件
│
├── 01_deepspeed_basic/          # DeepSpeed 原生用法
│   ├── train.py
│   ├── ds_config.json
│   └── run.sh
│
├── 02_accelerate_deepspeed/     # Accelerate + DeepSpeed
│   ├── train.py
│   ├── ds_config.json
│   ├── accelerate_config.yaml
│   └── run.sh
│
└── 03_zero_stages/              # ZeRO 1/2/3 对比
    ├── train.py
    ├── ds_zero1.json / ds_zero2.json / ds_zero3.json
    ├── accelerate_zero1.yaml / accelerate_zero2.yaml / accelerate_zero3.yaml
    └── compare.sh
```

---

## 执行日志

### 2026-01-16 重构

**问题**: 原有结构混乱，配置文件太多，两种启动方式混在一起

**解决**: 重构为三个子目录，从简单到复杂:
1. `01_deepspeed_basic` - 最简单的 DeepSpeed 原生用法
2. `02_accelerate_deepspeed` - Accelerate 封装
3. `03_zero_stages` - ZeRO 对比实验

**状态**: ✅ 重构完成

---

### 待执行：ZeRO 显存对比实验

```bash
cd 03_zero_stages

# 终端 1: 监控显存
watch -n 1 nvidia-smi

# 终端 2: 运行对比
bash compare.sh
```

**预期结果（300M 模型，双卡）**:

| 配置 | 预期显存/卡 | 实测显存/卡 | 训练时间 |
|------|-------------|-------------|----------|
| ZeRO-1 | ~6-8 GB | ___ GB | ___ 秒 |
| ZeRO-2 | ~4-6 GB | ___ GB | ___ 秒 |
| ZeRO-3 | ~3-4 GB | ___ GB | ___ 秒 |

**状态**: ⏳ 待执行

---

## 核心知识点

### 两种启动方式

| 启动方式 | 命令 | 需要的配置 |
|----------|------|-----------|
| DeepSpeed 直接 | `deepspeed --include localhost:6,7 train.py` | `ds_config.json` |
| Accelerate | `accelerate launch --config_file xxx.yaml train.py` | `accelerate.yaml` + `ds_config.json` |

### ZeRO 三阶段

| 阶段 | 切分内容 | 显存节省 | 通信开销 |
|------|----------|----------|----------|
| ZeRO-1 | 优化器状态 | ~4× | 无增加 |
| ZeRO-2 | +梯度 | ~8× | 略有增加 |
| ZeRO-3 | +模型参数 | 线性扩展 | 明显增加 |

---

## 参考资源

- [DeepSpeed GitHub](https://github.com/microsoft/DeepSpeed)
- [DeepSpeed 官方文档](https://www.deepspeed.ai/)
