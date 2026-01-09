# Accelerate 学习任务清单

**模块周期**：Day 5-6（2 天）
**难度等级**：⭐⭐⭐⭐（中高）
**前置要求**：已完成 Day 3-4 vLLM 模块

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
  CUDA_VISIBLE_DEVICES=0,1 accelerate launch test_accelerate.py
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
  CUDA_VISIBLE_DEVICES=0 python train_simple.py
  ```

- [ ] 双卡训练
  ```bash
  CUDA_VISIBLE_DEVICES=0,1 accelerate launch train_simple.py
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
  CUDA_VISIBLE_DEVICES=0,1 accelerate launch train_simple.py
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
