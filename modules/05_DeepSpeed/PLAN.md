# Author: zhouzhiyong
# Day 7-8: DeepSpeed 显存优化学习计划
# 创建日期: 2026-01-13

## 学习目标

- [ ] 理解 ZeRO 三阶段原理（优化器状态/梯度/参数切分）
- [ ] 掌握 ds_config.json 配置文件
- [ ] 使用 Accelerate + DeepSpeed 训练
- [ ] 对比 ZeRO-1/2/3 的显存占用

---

## Day 7 执行日志（2026-01-13）

### 步骤 1：环境准备

```bash
cd /home/zzy/weitiao/modules/05_DeepSpeed

# 激活环境
conda activate videofen

# 验证 PyTorch 和 CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# 设置使用 GPU 6,7（空闲的两张 4090）
export CUDA_VISIBLE_DEVICES=6,7

# 验证设置
echo $CUDA_VISIBLE_DEVICES
```
**状态**: ⏳ 待执行
**预期**: PyTorch >= 2.0, CUDA >= 12.0, GPU 6,7 可用

---

### 步骤 2：安装 DeepSpeed

```bash
# 安装 DeepSpeed
pip install deepspeed

# 验证安装
python -c "import deepspeed; print(f'DeepSpeed version: {deepspeed.__version__}')"
```
**状态**: ✅ 已完成
**实际结果**: ✅ 成功
```
DeepSpeed version: 0.18.3
```

---

### 步骤 3：阅读 ZeRO 原理

**ZeRO 三阶段对比**：

| 阶段 | 切分内容 | 显存节省 | 通信开销 | 适用场景 |
|------|----------|----------|----------|----------|
| **ZeRO-1** | 优化器状态 | 4× | 几乎无增加 | 默认首选 |
| **ZeRO-2** | + 梯度 | 8× | 略有增加 | 大多数微调 |
| **ZeRO-3** | + 模型参数 | 线性扩展 | 明显增加 | 超大模型 |

**核心思想**：
```
传统 DDP：
GPU 0: [完整模型] + [完整优化器] + [完整梯度]
GPU 1: [完整模型] + [完整优化器] + [完整梯度]
→ 冗余 3 倍！

ZeRO-2：
GPU 0: [完整模型] + [优化器1/2] + [梯度1/2]
GPU 1: [完整模型] + [优化器2/2] + [梯度2/2]
→ 省显存 8×！
```

**状态**: ⏳ 待阅读
**参考**: `modules/05_DeepSpeed/readme.md`

---

### 步骤 4：创建 ds_config.json 配置文件

**ZeRO-2 配置模板**：

```json
{
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {"device": "cpu"},
    "overlap_comm": true
  },
  "bf16": {
    "enabled": true
  },
  "gradient_accumulation_steps": 4,
  "train_batch_size": 256
}
```

**状态**: ⏳ 待创建

---

### 步骤 5：配置 Accelerate + DeepSpeed

```bash
# 使用 Accelerate 配置向导
accelerate config

# 关键选项：
# - distributed_type: DEEPSPEED
# - zero_stage: 2
# - offload_optimizer: true
```

**状态**: ⏳ 待执行

---

### 步骤 6：创建测试脚本

**目标**：对比 ZeRO-1/2/3 的显存占用

```python
# test_deepspeed.py
# 使用大模型测试 ZeRO 效果
```

**状态**: ⏳ 待创建

---

### 步骤 7：运行 ZeRO-2 训练测试

```bash
# 使用 Accelerate + DeepSpeed 启动
accelerate launch --config_file accelerate_config.yaml test_deepspeed.py
```

**状态**: ⏳ 待执行

---

### 步骤 8：对比显存占用

| 配置 | 显存占用/卡 | 训练时间 | 加速比 |
|------|------------|----------|--------|
| 无优化 | ? GB | ? 秒 | 1.0× |
| ZeRO-1 | ? GB | ? 秒 | ?× |
| ZeRO-2 | ? GB | ? 秒 | ?× |
| ZeRO-3 | ? GB | ? 秒 | ?× |

**状态**: ⏳ 待记录

---

## Day 8 任务预告

- [ ] ZeRO-Offload 实验（CPU 卸载）
- [ ] 与 Accelerate 集成训练
- [ ] 完整的微调流程

---

## 📊 Day 7 验收进度

- [ ] DeepSpeed 安装成功
- [ ] 理解 ZeRO 三阶段原理
- [ ] 创建 ds_config.json
- [ ] Accelerate + DeepSpeed 配置成功
- [ ] 运行 ZeRO-2 训练

---

## 参考资源

- [DeepSpeed GitHub](https://github.com/microsoft/DeepSpeed)
- [ZeRO 论文](https://arxiv.org/abs/1910.02054)
- [DeepSpeed 官方文档](https://www.deepspeed.ai/)
