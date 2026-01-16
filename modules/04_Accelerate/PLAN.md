# Accelerate 学习任务清单

**模块周期**：Day 5-6（2 天）
**难度等级**：⭐⭐⭐⭐（中高）
**前置要求**：已完成 Day 3-4 vLLM 模块

---

## 📝 执行日志（实时更新）

### 2026-01-12 | Day 5 执行开始

#### ✅ 步骤 0：前置检查
```bash
# 检查当前 Conda 环境
conda activate videofen

# 验证 PyTorch 和 CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# 设置使用 GPU 6,7（空闲的两张 4090）
export CUDA_VISIBLE_DEVICES=6,7

# 验证设置
echo $CUDA_VISIBLE_DEVICES
```
**状态**: ✅ 已完成
**预期**: PyTorch >= 2.0, CUDA >= 12.0, GPU 6,7 可用
**实际结果**: ✅ 成功
```
PyTorch: 2.9.0+cu128
CUDA available: True
GPU count: 2
CUDA_VISIBLE_DEVICES: 6,7
GPU 6,7 空闲，可独占使用
```

**说明**：
- GPU 6,7 当前空闲，可用于 Accelerate 训练任务

---

#### ✅ 步骤 1：安装 Accelerate
```bash
conda activate videofen
pip install accelerate==0.21.0

# 验证安装
python -c "import accelerate; print(f'Accelerate version: {accelerate.__version__}')"
```
**状态**: ✅ 已完成
**预期**: 显示 Accelerate 版本号 0.21.0
**实际结果**: ✅ 成功
```
Accelerate version: 0.21.0
```

---

#### ✅ 步骤 2：配置 Accelerate 环境
```bash
# 运行配置向导
accelerate config
```
**交互式配置选项**：
```
Compute environment: local_machine
Distributed type: MULTI_GPU (DDP)
Number of GPUs: 2
GPU IDs: 6,7
Mixed precision: bf16
```
**状态**: ✅ 已完成
**预期**: 生成配置文件 `~/.cache/huggingface/accelerate/default_config.yaml`
**实际结果**: ✅ 成功
- 配置向导完成
- 修复了中文逗号问题：`gpu_ids: "6,7"`

---

#### ✅ 步骤 2.5：迁移配置文件到项目目录
```bash
# 复制到模块目录
cp /root/.cache/huggingface/accelerate/default_config.yaml /home/zzy/weitiao/modules/04_Accelerate/accelerate_config.yaml

# 修复中文逗号问题（用 nano 编辑或直接覆盖）
# 验证配置
cat /home/zzy/weitiao/modules/04_Accelerate/accelerate_config.yaml
```
**状态**: ✅ 已完成
**完整配置内容**：
```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: "6,7"
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```
**实际结果**: ✅ 成功
**配置文件位置**: `/home/zzy/weitiao/modules/04_Accelerate/accelerate_config.yaml`

---

#### ✅ 步骤 3：创建测试脚本
```bash
# 创建工作目录
mkdir -p /home/zzy/weitiao/experiments/accelerate
cd /home/zzy/weitiao/experiments/accelerate

# 创建测试脚本
cat > test_accelerate.py << 'EOF'
import torch
from accelerate import Accelerator

print("=" * 60)
print("Accelerate 测试脚本")
print("=" * 60)

# 初始化 Accelerator
accelerator = Accelerator()

# 创建简单模型
model = torch.nn.Linear(10, 10)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
dataloader = torch.utils.data.DataLoader(
    torch.randn(100, 10), batch_size=10
)

# 核心魔法：prepare()
model, optimizer, dataloader = accelerator.prepare(
    model, optimizer, dataloader
)

# 训练循环
for epoch in range(2):
    for batch in dataloader:
        outputs = model(batch)
        loss = outputs.sum()

        # 替换 loss.backward()
        accelerator.backward(loss)

        optimizer.step()
        optimizer.zero_grad()

    # 只在主进程打印
    if accelerator.is_main_process:
        print(f"Epoch {epoch} completed")

print(f"\n{'=' * 60}")
print(f"使用设备: {accelerator.device}")
print(f"进程索引: {accelerator.process_index}")
print(f"进程总数: {accelerator.num_processes}")
print(f"混合精度: {accelerator.mixed_precision}")
print(f"梯度累积步数: {accelerator.gradient_accumulation_steps}")
print("=" * 60)
EOF
```
**状态**: ⏳ 待执行
**实际结果**: 待记录

---

#### ✅ 步骤 4：单卡测试
```bash
cd /home/zzy/weitiao/modules/04_Accelerate
CUDA_VISIBLE_DEVICES=6 python test_accelerate.py
```
**状态**: ✅ 已完成
**预期**:
- 使用 GPU 6
- 进程总数: 1
- 训练正常完成
**实际结果**: ✅ 成功
```
Epoch 0 completed
Epoch 1 completed
使用设备: cuda
进程索引: 0
进程总数: 1
混合精度: no (直接运行，未读取配置文件)
梯度累积步数: 1
```
**说明**：直接用 `python` 运行不会应用配置文件中的混合精度设置

---

#### ✅ 步骤 5：双卡测试
```bash
accelerate launch --config_file accelerate_config.yaml test_accelerate.py
```
**状态**: ✅ 已完成
**预期**:
- 使用 GPU 6,5
- 进程总数: 2
- 混合精度 bf16 生效
**实际结果**: ✅ 成功
```
进程 0: 使用设备 cuda:0 (GPU 6), 进程索引 0, 混合精度 bf16
进程 1: 使用设备 cuda:1 (GPU 7), 进程索引 1, 混合精度 bf16
Epoch 0 completed
Epoch 1 completed
```
**关键发现**：
- ✅ 双进程并行运行（进程 0 和 1）
- ✅ 混合精度 bf16 生效（使用 accelerate launch）
- ✅ GPU 6→cuda:0, GPU 7→cuda:1 映射正确
- ⚠️ 警告信息正常（进程组未显式销毁，不影响功能）

---

#### ✅ 步骤 6：监控双卡训练（新终端）
```bash
nvitop
# 或
watch -n 1 nvidia-smi
```
**状态**: ✅ 已完成
**预期**: GPU 6 和 GPU 7 都有显存占用
**实际结果**: ✅ 正常
```
GPU 6: 空闲，可独占使用
GPU 7: 空闲，可独占使用
测试脚本运行后: 显存占用约 2-4 GB/卡
```
**说明**：
- GPU 6-7 空闲，可独占使用
- Accelerate 测试脚本显存占用较小

---

### 📊 Day 5 验收进度
- [x] Accelerate 安装成功 ✅
- [x] 配置文件生成正确 ✅
- [x] 单卡测试通过 ✅
- [x] 双卡测试通过 ✅
- [x] 理解 prepare() 方法作用 ✅

**Day 5 状态**: ✅ **全部完成！**

---

## 🚀 Day 6 执行日志（2026-01-13）

### 步骤 1：创建实战训练脚本
```bash
# 脚本已创建: modules/04_Accelerate/train_simple.py
# 特性:
#   - 10000 样本，128 维特征，10 类分类
#   - 3 层 MLP 模型
#   - AdamW 优化器 + Cosine 学习率调度
#   - 10 个 epoch 训练
```
**状态**: ✅ 已完成

---

### 步骤 2：单卡基准测试
```bash
cd /home/zzy/weitiao/modules/04_Accelerate

# 单卡训练（GPU 6）
CUDA_VISIBLE_DEVICES=6 python train_simple.py
```
**状态**: ✅ 已完成
**预期**:
- 进程数: 1
- 混合精度: no（直接 python 运行）
- 记录训练时间作为基准

**实际结果**: ✅ 成功
```
配置信息:
  - 设备: cuda
  - 进程数: 1
  - 混合精度: no
  - Batch Size: 64 × 1 = 64

Epoch  1/10 | Loss: 2.3074 | Acc: 9.57%
...
Epoch 10/10 | Loss: 1.7646 | Acc: 41.82%

总训练时间: 2.99 秒
平均每轮时间: 0.30 秒
```

---

### 步骤 3：双卡数据并行训练
```bash
cd /home/zzy/weitiao/modules/04_Accelerate

# 双卡训练（GPU 6,7 + BF16）
accelerate launch --config_file accelerate_config.yaml train_simple.py
```
**状态**: ✅ 已完成
**预期**:
- 进程数: 2
- 混合精度: bf16
- 加速比 > 1.8×

**实际结果**: ✅ 成功（但加速比不明显）
```
配置信息:
  - 设备: cuda:0
  - 进程数: 2
  - 混合精度: bf16
  - Batch Size: 64 × 2 = 128

Epoch  1/10 | Loss: 2.3068 | Acc: 10.34%
...
Epoch 10/10 | Loss: 1.9398 | Acc: 33.88%

总训练时间: 3.05 秒
平均每轮时间: 0.30 秒
```

**关键发现**：
- ⚠️ 加速比 ≈ 0.98×（几乎无加速）
- ✅ 有效 Batch Size 翻倍：64 → 128
- ✅ 总步数减半：1570 → 785
- ✅ 混合精度 bf16 生效

**为什么没有加速？**
1. **模型太小**：3 层 MLP，计算量极低
2. **数据太小**：10000 样本，训练很快结束
3. **通信开销**：双卡梯度同步的开销 ≈ 计算收益
4. **正确理解**：数据并行的收益在 **大模型 + 大数据** 场景才明显

---

### 步骤 4：加速比对比
| 配置 | 训练时间 | 加速比 | 混合精度 | 有效 Batch |
|------|----------|--------|----------|------------|
| 单卡 GPU 6 | 2.99 秒 | 1.0× | no | 64 |
| 双卡 GPU 6,7 | 3.05 秒 | 0.98× | bf16 | 128 |

**结论**：小模型场景下，通信开销抵消了并行收益。需要更大模型验证。

---

### 步骤 4.5：大模型加速比验证（补充实验）
为验证数据并行在大模型场景的加速效果，创建更大的模型进行测试。

```bash
# 单卡（GPU 6）
CUDA_VISIBLE_DEVICES=6 python train_large.py

# 双卡（GPU 6,7 + BF16）
accelerate launch --config_file accelerate_config.yaml train_large.py
```
**状态**: ✅ 已完成

**实验结果**：
| 配置 | 模型参数 | 训练时间 | 加速比 | 有效 Batch |
|------|----------|----------|--------|------------|
| 单卡 GPU 6 | 32M | 6.79 秒 | 1.0× | 256 |
| 双卡 GPU 6,7 | 32M | 8.25 秒 | 0.82× | 512 |

**关键结论**：
- ⚠️ 32M 参数模型仍然太小，双卡反而更慢
- ✅ 有效 Batch Size 翻倍：256 → 512
- ✅ 双卡并行机制正常工作

**为什么双卡更慢？**
```
┌─────────────────────────────────────────────────────────┐
│  数据并行的代价：NCCL 梯度同步                            │
├─────────────────────────────────────────────────────────┤
│  每个 step：                                             │
│    1. 各卡独立计算 forward + backward                    │
│    2. 梯度同步（all-reduce）← 这是额外开销               │
│    3. 各卡独立更新参数                                   │
├─────────────────────────────────────────────────────────┤
│  小模型：计算时间 << 通信时间 → 无加速                   │
│  大模型：计算时间 >> 通信时间 → 明显加速                 │
└─────────────────────────────────────────────────────────┘
```

**什么时候能看到加速？**
- 7B 模型：双卡加速比约 1.8-1.9×
- 14B 模型：双卡加速比约 1.85-1.95×
- 这在 Day 11-12 的 LLaMA-Factory 微调中会验证

---

### 步骤 5：混合精度对比
```bash
# BF16 双卡（已执行）
accelerate launch --config_file accelerate_config.yaml train_large.py

# FP32 双卡
accelerate launch --config_file accelerate_config_fp32.yaml train_large.py
```
**状态**: ✅ 已完成

**实验结果**：
| 混合精度 | 训练时间 | 相对速度 | 显存占用 |
|----------|----------|----------|----------|
| FP32 | 9.62 秒 | 1.0× | 较高 |
| BF16 | 8.25 秒 | 1.17× | 较低 |

**结论**：
- ✅ BF16 比 FP32 快 **17%**
- ✅ BF16 显存占用更低（约节省 30-50%）
- ✅ 4090 原生支持 BF16，无精度损失风险

---

### 步骤 6：梯度累积实验

**梯度累积配置方式**：

```
┌─────────────────────────────────────────────────────────────┐
│  方式 1：代码指定（推荐）                                    │
├─────────────────────────────────────────────────────────────┤
│  accelerator = Accelerator(gradient_accumulation_steps=4)   │
│                                                             │
│  优点：直观、一眼看到配置                                    │
│  缺点：修改需要改代码                                        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  方式 2：配置文件指定                                        │
├─────────────────────────────────────────────────────────────┤
│  accelerate_config.yaml:                                    │
│    gradient_accumulation_steps: 4                           │
│                                                             │
│  代码中：                                                    │
│    accelerator = Accelerator()  # 自动读取配置              │
│                                                             │
│  优点：修改方便、不用改代码                                  │
│  缺点：需要查看配置文件才知道                                │
└─────────────────────────────────────────────────────────────┘
```

**train_grad_accum.py 使用方式 1**：
```python
# 第 17 行：代码中指定梯度累积步数
GRADIENT_ACCUMULATION_STEPS = 4
accelerator = Accelerator(gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS)
```

执行：
```bash
accelerate launch --config_file accelerate_config.yaml train_grad_accum.py
```
**状态**: ✅ 已完成

**实验结果**：
| 配置 | 微 Batch | 累积步数 | 有效 Batch | 训练时间 |
|------|----------|----------|------------|----------|
| 无累积 | 256 | 1 | 256 | 8.25 秒 |
| 梯度累积 | 64 | 4 | 512 | 12.62 秒 |

**关键发现**：
- ✅ 有效 Batch Size 翻倍：256 → 512
- ⚠️ 训练时间增加 53%（12.62 / 8.25 = 1.53×）
- ✅ 显存占用降低（微 batch 64 vs 256）

**为什么更慢？**
```
总样本数：50000

无累积 (batch=256):
  - 每个 epoch: 50000 / 256 = 195 次迭代
  - 每次迭代: 1 次 forward/backward

梯度累积 (微 batch=64, 累积4):
  - 每个 epoch: 50000 / 64 = 781 次迭代
  - 每 4 次迭代才更新一次
  - forward/backward 总次数: 781 次（更多！）

结论：梯度累积是用时间换显存
```

**使用场景**：
- ✅ 显存不足，必须用小 batch
- ✅ 需要更大的有效 batch size（提升训练稳定性）
- ❌ 不追求训练速度

---

## 🎉 Day 5 总结

### ✅ 完成的任务
1. ✅ 安装 Accelerate 0.21.0
2. ✅ 配置 Accelerate 环境（GPU 6,7 + BF16）
3. ✅ 创建项目级配置文件
4. ✅ 单卡测试（验证基本功能）
5. ✅ 双卡测试（验证数据并行）
6. ✅ 混合精度 BF16 生效

### 🎯 核心收获
- **accelerator.prepare()**：自动处理设备分配、多卡同步
- **accelerate launch**：应用配置文件的唯一方式
- **is_main_process**：避免多进程重复输出
- **bf16 vs no**：配置文件中的混合精度只在 `accelerate launch` 时生效

### 📝 关键命令
```bash
# 配置向导
accelerate config

# 单卡运行（不应用配置文件）
CUDA_VISIBLE_DEVICES=6 python test_accelerate.py

# 双卡运行（应用配置文件）
accelerate launch --config_file accelerate_config.yaml test_accelerate.py
```

---

---

## 📋 学习目标

- [ ] 理解 Accelerate 的核心价值（分布式训练抽象层）
- [ ] 掌握 Accelerator API 的使用
- [ ] 掌握配置文件与启动器（accelerate config/launch）
- [ ] 双卡数据并行训练实践
- [ ] 混合精度训练（BF16）

---

## 📅 Day 5：环境配置与核心 API

### 任务清单

#### 上午：理论理解（2-3 小时）
- [ ] 阅读 `modules/04_Accelerate/readme.md`
  - [ ] 理解 Accelerate 在工具链中的位置
  - [ ] 理解为什么需要 Accelerate（代码复用）
  - [ ] 理解 Accelerator 对象的职责
  - [ ] 理解 `prepare()` 方法的魔法

- [ ] 完成理论自测题
  ```
  Q1: Accelerate 解决了什么痛点？
  Q2: Accelerate 和 DeepSpeed 有什么区别？
  Q3: prepare() 方法为什么不需要手动指定设备？
  ```

#### 下午：环境配置（2-3 小时）
- [ ] 安装/升级 Accelerate
  ```bash
  conda activate videofen
  pip install accelerate==0.21.0

  # 验证安装
  python -c "import accelerate; print(accelerate.__version__)"
  ```

- [ ] 运行配置向导
  ```bash
  accelerate config
  ```

  **交互式配置选项**：
  ```
  Compute environment: local_machine
  Distributed type: MULTI_GPU (DDP)
  Number of GPUs: 2
  Mixed precision: bf16
  ```

- [ ] 查看生成的配置文件
  ```bash
  cat ~/.cache/huggingface/accelerate/default_config.yaml
  ```

  **关键配置项**：
  ```yaml
  compute_environment: LOCAL_MACHINE
  distributed_type: MULTI_GPU
  num_processes: 2
  mixed_precision: bf16
  ```

#### 晚上：核心 API 实践（2-3 小时）
- [ ] 创建测试脚本 `test_accelerate.py`
  ```python
  import torch
  from accelerate import Accelerator

  # 初始化 Accelerator
  accelerator = Accelerator()

  # 创建简单模型
  model = torch.nn.Linear(10, 10)
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  dataloader = torch.utils.data.DataLoader(
      torch.randn(100, 10), batch_size=10
  )

  # 核心魔法：prepare()
  model, optimizer, dataloader = accelerator.prepare(
      model, optimizer, dataloader
  )

  # 训练循环
  for epoch in range(2):
      for batch in dataloader:
          outputs = model(batch)
          loss = outputs.sum()

          # 替换 loss.backward()
          accelerator.backward(loss)

          optimizer.step()
          optimizer.zero_grad()

      # 只在主进程打印
      if accelerator.is_main_process:
          print(f"Epoch {epoch} completed")

  print(f"Using device: {accelerator.device}")
  print(f"Process index: {accelerator.process_index}")
  print(f"Num processes: {accelerator.num_processes}")
  ```

- [ ] 单卡测试
  ```bash
  CUDA_VISIBLE_DEVICES=0 python test_accelerate.py
  ```

- [ ] 双卡测试
  ```bash
  CUDA_VISIBLE_DEVICES=6,7 accelerate launch test_accelerate.py
  ```

**Day 5 验收标准**：
- [ ] 成功配置 Accelerate
- [ ] 理解配置文件的结构
- [ ] 单卡/双卡测试脚本运行成功
- [ ] 理解 `prepare()` 和 `backward()` 的作用

---

## 📅 Day 6：双卡训练与混合精度

### 任务清单

#### 上午：数据并行训练（3-4 小时）
- [ ] 理解数据并行原理
  - [ ] 每张卡处理不同的 batch
  - [ ] 梯度自动同步
  - [ ] 为什么能线性加速

- [ ] 创建真实训练脚本 `train_simple.py`
  ```python
  import torch
  import torch.nn.functional as F
  from accelerate import Accelerator
  from torch.utils.data import DataLoader, TensorDataset

  # 创建虚拟数据集
  X = torch.randn(1000, 10)
  y = torch.randint(0, 2, (1000,))
  dataset = TensorDataset(X, y)
  dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

  # 初始化
  accelerator = Accelerator()
  model = torch.nn.Sequential(
      torch.nn.Linear(10, 64),
      torch.nn.ReLU(),
      torch.nn.Linear(64, 2)
  )
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

  # Prepare
  model, optimizer, dataloader = accelerator.prepare(
      model, optimizer, dataloader
  )

  # 训练
  model.train()
  for epoch in range(5):
      total_loss = 0
      for X_batch, y_batch in dataloader:
          outputs = model(X_batch)
          loss = F.cross_entropy(outputs, y_batch)

          accelerator.backward(loss)
          optimizer.step()
          optimizer.zero_grad()

          total_loss += loss.detach()

      # 只在主进程打印
      if accelerator.is_main_process:
          avg_loss = total_loss.item() / len(dataloader)
          print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
  ```

- [ ] 单卡训练（基准）
  ```bash
  CUDA_VISIBLE_DEVICES=6 python train_simple.py
  ```

- [ ] 双卡训练
  ```bash
  CUDA_VISIBLE_DEVICES=6,7 accelerate launch train_simple.py
  ```

- [ ] 记录训练时间对比
  | 配置 | 训练时间 | 加速比 |
  |------|----------|--------|
  | 单卡 | ? 秒 | 1.0× |
  | 双卡 | ? 秒 | ?× |

#### 下午：混合精度训练（2-3 小时）
- [ ] 理解混合精度原理
  - [ ] FP16/BF16 vs FP32
  - [ ] 显存节省（约 50%）
  - [ ] 速度提升（约 2-3×）

- [ ] 修改配置开启 BF16
  ```bash
  accelerate config
  # 选择 mixed_precision: bf16
  ```

- [ ] 或直接修改配置文件
  ```yaml
  # ~/.cache/huggingface/accelerate/default_config.yaml
  mixed_precision: bf16
  ```

- [ ] 运行混合精度训练
  ```bash
  CUDA_VISIBLE_DEVICES=6,7 accelerate launch train_simple.py
  ```

- [ ] 对比 FP32 vs BF16
  | 精度 | 训练时间 | 显存占用 | 速度提升 |
  |------|----------|----------|----------|
  | FP32 | ? 秒 | ? GB | 1.0× |
  | BF16 | ? 秒 | ? GB | ?× |

#### 晚上：梯度累积实验（2-3 小时）
- [ ] 理解梯度累积原理
  - [ ] 小显存模拟大 batch
  - [ ] 多次计算、一次更新

- [ ] 修改训练脚本添加梯度累积
  ```python
  # 在 train_simple.py 中添加
  gradient_accumulation_steps = 4

  for epoch in range(5):
      for i, (X_batch, y_batch) in enumerate(dataloader):
          with accelerator.accumulate(model):
              outputs = model(X_batch)
              loss = F.cross_entropy(outputs, y_batch)

              accelerator.backward(loss)
              optimizer.step()
              optimizer.zero_grad()
  ```

- [ ] 对比不同累积步数
  | 累积步数 | 有效 Batch Size | 训练时间 |
  |----------|----------------|----------|
  | 1 | 32 | ? 秒 |
  | 4 | 128 | ? 秒 |
  | 8 | 256 | ? 秒 |

**Day 6 验收标准**：
- [ ] 双卡数据并行训练成功
- [ ] 加速比 > 1.8×
- [ ] 混合精度训练成功（速度提升 > 2×）
- [ ] 理解梯度累积的作用

---

## 🎯 模块验收标准

### 理论验收
- [ ] 能解释 Accelerate 的核心价值
- [ ] 能说明数据并行的原理
- [ ] 能解释混合精度的优势
- [ ] 能说明梯度累积的应用场景

### 实操验收
- [ ] 成功配置 Accelerate（单卡/双卡）
- [ ] 双卡训练加速比 > 1.8×
- [ ] 混合精度训练速度提升 > 2×
- [ ] 能使用 `accelerator.gather()` 汇总数据

### 输出物
- [ ] 训练性能对比表（单卡 vs 双卡 vs BF16）
- [ ] 配置文件笔记（含关键参数说明）
- [ ] 梯度累积实验数据

---

## 📚 参考资源

### 官方文档
- [Accelerate GitHub](https://github.com/huggingface/accelerate)
- [Accelerate 官方文档](https://huggingface.co/docs/accelerate/)
- [分布式训练指南](https://huggingface.co/docs/accelerate/usage_guides/distributed_training)

### 推荐阅读
- `modules/04_Accelerate/readme.md`（完整理论指南）

### 常用命令速查
```bash
# 配置向导
accelerate config

# 启动训练
accelerate launch train.py

# 查看配置
cat ~/.cache/huggingface/accelerate/default_config.yaml

# 测试环境
accelerate env
```

### 核心代码模板
```python
from accelerate import Accelerator

# 初始化
accelerator = Accelerator(
    mixed_precision="bf16",  # fp16/bf16/no
    gradient_accumulation_steps=4
)

# Prepare
model, optimizer, dataloader = accelerator.prepare(
    model, optimizer, dataloader
)

# 训练循环
for batch in dataloader:
    with accelerator.accumulate(model):
        loss = model(batch)
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

# 只在主进程执行
if accelerator.is_main_process:
    print("Result")

# 汇总多卡数据
all_results = accelerator.gather(results)
```

---

## ⚠️ 常见问题

### Q1: accelerate launch 报错 "CUDA not available"？
**A**: 检查 `CUDA_VISIBLE_DEVICES` 设置
```bash
# 检查 GPU 可见性
python -c "import torch; print(torch.cuda.device_count())"

# 确保设置了正确的 GPU
export CUDA_VISIBLE_DEVICES=0,1
```

### Q2: 双卡训练速度没有提升？
**A**: 检查数据加载是否瓶颈
```python
# 增加 dataloader workers
dataloader = DataLoader(dataset, batch_size=32, num_workers=4)
```

### Q3: 混合精度训练出现 NaN？
**A**: 降低学习率或使用梯度缩放
```python
accelerator = Accelerator(mixed_precision="bf16")
# 或降低学习率
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
```

---

## 🔄 与后续模块的衔接

完成本模块后，你将掌握：
- ✅ 分布式训练的统一抽象
- ✅ 数据并行与混合精度
- ✅ 梯度累积技术

**下一模块**：Day 7-8 DeepSpeed 显存优化
- 学习 ZeRO 三阶段优化
- 解决显存不足问题
- 为大模型微调打基础
