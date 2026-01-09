# Accelerate vs DeepSpeed: 多卡分布式技术详解 (基于 LLaMA-Factory)

> **适用场景**: 本仓库以 **LLaMA-Factory** 为主,目标是在多卡环境下完成 **推理和训练** 任务。
>
> **核心结论** (先看结论再看细节):
> - **多卡推理** 有两种完全不同的目标:
>   - **显存分摊** (单次对话/单次生成用多张卡): 使用 `llamafactory-cli chat/webui`,让框架自动切分模型
>   - **吞吐并行** (批量处理很多输入): 使用 `accelerate launch`/`deepspeed` 启动多进程做数据并行
> - **训练**: DeepSpeed 的 ZeRO 技术在显存优化上更强大,而 Accelerate 更简单易用
> - **推荐**: 推理优先用 Accelerate,训练优先用 DeepSpeed

---

## 📚 目录

1. [通俗理解:建筑施工队比喻](#1-通俗理解建筑施工队比喻)
2. [核心概念详解](#2-核心概念详解)
3. [两种多卡推理模式](#3-两种多卡推理模式)
4. [实战命令指南](#4-实战命令指南)
5. [如何选择:Accelerate 还是 DeepSpeed](#5-如何选择accelerate-还是-deepspeed)
6. [GPU 映射说明](#6-gpu-映射说明)
7. [性能对比与监控](#7-性能对比与监控)
8. [常见问题排错](#8-常见问题排错)

---

## 1. 通俗理解:建筑施工队比喻

我们可以用一个**建筑施工队**的比喻来理解这三者的关系:

### 🏗️ 三大角色

#### **LLaMA-Factory: 包工头 (The Commander)**
- **定位**: 上层框架 / 一站式工具箱
- **作用**: 整合资源,发号施令
  - 处理数据加载
  - 管理 Prompt 模板
  - 封装训练/推理流程
- **你需要做的**: 告诉它"我要微调 Qwen2.5"、"用这个数据集"、"学习率 1e-5"
- **关系**: 负责发号施令,底层脏活累活交给 Accelerate 和 DeepSpeed

#### **⚖️ Accelerate (Hugging Face): 现场调度员 (The Manager)**
- **定位**: 硬件抽象层 / 启动器
- **作用**: 充当"翻译官",让同一套代码运行在不同硬件上
- **核心功能**:
  - `accelerate launch`: 自动分配 GPU 进程 (4张卡 = 4个进程)
  - 处理混合精度 (FP16/BF16)
  - 配置分布式环境 (rank/world_size 等)
- **与 LLaMA-Factory 的关系**: 必须配合使用,LLaMA-Factory 底层构建在 Accelerate 之上

#### **🚀 DeepSpeed (Microsoft): 重型机械 / 压缩大师 (The Turbo Engine)**
- **定位**: 显存优化与加速引擎
- **作用**: 不是用来"启动"任务,而是用来**省显存**和**提速**
- **核心绝招 - ZeRO 技术**:
  - **普通训练**: 把模型参数复制到每张卡 (非常占显存)
  - **DeepSpeed (ZeRO Stage 2/3)**: 把模型**切碎**分散存储,训练时再拼起来
  - **Offload**: 显存不够时,把数据暂时扔到内存 (CPU RAM) 里
- **适用场景**: 训练大模型时显存不够,或想提升训练速度

### 📊 层级关系图

```
+-------------------------------------------------------+
|                用户 (User / CLI / WebUI)               |
+-------------------------------------------------------+
|           LLaMA-Factory (业务逻辑层)                   |  ← 你在这里操作
+-------------------------------------------------------+
|             Accelerate (硬件调度层)                    |  ← 负责多卡通信
+-------------------------------------------------------+
|             DeepSpeed (可选的优化后端)                  |  ← 负责省显存、切分模型
+-------------------------------------------------------+
|               PyTorch (基础计算框架)                   |
+-------------------------------------------------------+
|                CUDA / GPU (物理硬件)                   |
+-------------------------------------------------------+
```

**两种模式**:
- **常规模式**: LLaMA-Factory → Accelerate → PyTorch
- **高性能模式**: LLaMA-Factory → Accelerate → DeepSpeed → PyTorch

---

## 2. 核心概念详解

### 2.1 Accelerate 是什么?

**定义**: Hugging Face 的分布式启动器和封装层

**核心能力**:
- ✅ 启动多进程 (每张 GPU 一个进程)
- ✅ 配置 PyTorch Distributed 环境
- ✅ 提供 CLI 工具 (`accelerate launch`, `accelerate config`)
- ✅ 支持混合精度训练
- ✅ 跨平台兼容 (GPU/TPU/CPU)

**一句话理解**: "我想用 4 张 GPU 跑同一段 Python 程序",Accelerate 帮你把程序**复制成 4 个进程**并让它们互相通信。

### 2.2 DeepSpeed 是什么?

**定义**: 微软的分布式训练/推理优化框架

**核心能力**:
- ✅ **ZeRO 优化** (显存杀手):
  - ZeRO-1: 优化器状态切分
  - ZeRO-2: 梯度切分
  - ZeRO-3: 参数切分 (最省显存)
- ✅ **CPU/NVMe Offload**: 把数据挪到 CPU 或磁盘
- ✅ **训练吞吐优化**: 提升训练速度
- ✅ 也能当启动器 (`deepspeed --num_gpus 4`)

**一句话理解**: 不仅能"开 4 个进程",还试图在训练/推理中**更省显存/更高吞吐**,尤其适合大模型训练。

### 2.3 共同点: 都能当 Launcher

无论 Accelerate 还是 DeepSpeed,最基础的功能都是:
- 让同一个 Python 入口在多张 GPU 上以多进程方式运行

**对比**:
```bash
# Accelerate 启动 4 个进程
accelerate launch --num_processes 4 python script.py

# DeepSpeed 启动 4 个进程
deepspeed --num_gpus 4 script.py
```

**在 LLaMA-Factory 中的使用**:
- `-m llamafactory.launcher` (等价于 `python -m llamafactory.launcher`)
- 这是 pip 安装版的主要入口
- ⚠️ 注意: `llamafactory.train` 不能直接用 `python -m` 启动

---

## 3. 两种多卡推理模式

### 🎯 场景 A: 数据并行 (Data Parallelism) → 批量处理

**原理**:
- 有 4 张卡,就把模型**复制 4 份**
- 有 100 个问题要问,每张卡处理 25 个
- 每张卡独立工作,最后汇总结果

**工具**: `accelerate launch` 或 `deepspeed`

**缺点**:
- ❌ **不适合聊天 (Chat)**: 聊天是串行的 (你一句,它回一句)
- ❌ 启动多个进程会导致端口冲突
- ❌ 单次对话无法利用多卡并行

**适用场景**:
- ✅ 批量评测 (`--stage sft --do_predict`)
- ✅ 批量生成
- ✅ 数据集推理

---

### 🎯 场景 B: 模型并行 (Model Parallelism) → 单次对话

**原理**:
- 模型太大 (如 72B),单张卡放不下
- 把模型的第 1-10 层放在 GPU0,11-20 层放在 GPU1...
- 数据像流水线一样流过所有显卡

**工具**: HuggingFace 的 `device_map="auto"`

**实现方式**:
- ❌ **不需要** `accelerate launch`
- ✅ 只需指定 `CUDA_VISIBLE_DEVICES=0,1,2,3`
- ✅ 框架内部自动检测多卡并进行切分

**适用场景**:
- ✅ 交互式聊天 (`llamafactory-cli chat`)
- ✅ WebUI 演示
- ✅ 单次大模型推理

---

### 💡 选择决策树

```
我要做多卡推理
    │
    ├─ 我要聊天/交互演示
    │   └─→ 使用场景 B (模型并行)
    │        └─→ 直接运行 llamafactory-cli chat
    │
    └─ 我要批量处理数据集
        └─→ 使用场景 A (数据并行)
             └─→ 使用 accelerate launch
```

---

## 4. 实战命令指南

### 📌 重要说明

- `accelerate launch`/`deepspeed` 只是"多进程启动器",不会自动变成"对话式"
- 以下命令对应 **批量预测/批量推理**,必须提供 `--dataset/--dataset_dir/--output_dir`
- 交互式对话请使用 `llamafactory-cli chat` (见场景 B)

---

### 4.1 场景 A: 批量推理 (数据并行)

#### 🚀 方案 1: Accelerate (推荐)

**优点**: 参数少,行为直观,排错容易,贴近 HF 生态

```bash
# 设置环境变量
CUDA_VISIBLE_DEVICES=6,7 \              # 指定物理 GPU (这里用 6 号和 7 号卡)
HF_HUB_OFFLINE=1 \                       # 离线模式 (不联网下载)
TRANSFORMERS_OFFLINE=1 \
TRANSFORMERS_NO_FLASH_ATTENTION=1 \      # 禁用 flash attention (兼容性)

# 启动命令
accelerate launch \
  --num_processes 2 \                    # 启动 2 个进程 (对应 2 张卡)
  --main_process_port 29501 \            # 主进程端口 (避免冲突)
  -m llamafactory.launcher \             # LLaMA-Factory 入口

  # ===== 任务配置 =====
  --stage sft \                          # 训练阶段: sft (监督微调)
  --model_name_or_path /path/to/Qwen2.5-7B-Instruct \  # 模型路径
  --template qwen \                      # 模型模板
  --infer_backend huggingface \          # 推理后端

  # ===== 数据配置 =====
  --dataset YOUR_DATASET_NAME \          # 数据集名称
  --dataset_dir /ABS/PATH/TO/datasets \  # 数据集目录
  --do_predict \                         # 开启预测模式
  --predict_with_generate \              # 使用生成式预测

  # ===== 输出配置 =====
  --output_dir outputs/predict_qwen2.5_7b \  # 输出目录

  # ===== 性能配置 =====
  --gpus 0,1 \                           # 可见 GPU 索引 (0=物理卡6, 1=物理卡7)
  --cutoff_len 2048 \                    # 最大序列长度
  --per_device_eval_batch_size 1         # 每设备批次大小
```

#### 🔧 方案 2: DeepSpeed

**优点**: 如果后续要训练,可以提前熟悉 DeepSpeed 的用法

```bash
# 环境变量同上
CUDA_VISIBLE_DEVICES=6,7 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
TRANSFORMERS_NO_FLASH_ATTENTION=1 \

# 启动命令 (区别在这里)
deepspeed \
  --num_gpus 2 \                         # 启动 2 个 GPU 进程
  -m llamafactory.launcher \             # LLaMA-Factory 入口

  # 后面的参数与 Accelerate 版本完全相同
  --stage sft \
  --model_name_or_path /path/to/Qwen2.5-7B-Instruct \
  --template qwen \
  --infer_backend huggingface \
  --dataset YOUR_DATASET_NAME \
  --dataset_dir /ABS/PATH/TO/datasets \
  --do_predict \
  --predict_with_generate \
  --output_dir outputs/predict_qwen2.5_7b \
  --gpus 0,1 \
  --cutoff_len 2048 \
  --per_device_eval_batch_size 1
```

**两个版本的差异**:
- ✅ 命令格式略有不同
- ✅ 日志风格不同
- ✅ HF 推理后端下,性能差异通常不大 (因为没用上 DS 引擎的优化能力)

---

### 4.2 场景 B: 交互式聊天 (模型并行)

**⚠️ 重要: 千万别用 accelerate launch!**

```bash
# 方式 1: 命令行聊天
CUDA_VISIBLE_DEVICES=0,1,2,3 \           # 指定可见 GPU
llamafactory-cli chat \
  --model_name_or_path /path/to/Qwen2.5-72B-Instruct \  # 大模型
  --template qwen \
  --infer_backend huggingface

# 方式 2: WebUI
CUDA_VISIBLE_DEVICES=0,1,2,3 \
llamafactory-cli webui \
  --model_name_or_path /path/to/Qwen2.5-72B-Instruct \
  --template qwen
```

**工作原理**:
- 框架使用 `device_map="auto"` 自动检测多卡
- 如果模型是 7B (单卡能放下),可能只用 GPU0
- 如果模型是 72B (需要多卡),会自动切分到所有可见卡

---

### 4.3 场景 C: 训练 (结合 DeepSpeed 优化)

**命令结构**:

```bash
accelerate launch \
  --config_file examples/accelerate/ds_zero2_config.yaml \  # DeepSpeed 配置
  -m llamafactory.launcher \
  --stage sft \
  --model_name_or_path /path/to/model \
  --dataset YOUR_DATA \
  --output_dir outputs/sft_model \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-5 \
  --num_train_epochs 3
```

**解释**:
- Accelerate 启动多卡进程
- 根据 yaml 配置加载 DeepSpeed 引擎
- DeepSpeed 的 ZeRO 技术优化显存使用

**典型 DeepSpeed 配置文件** (`ds_zero2_config.yaml`):

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
num_machines: 1
num_processes: 4
gpu_ids: "0,1,2,3"
main_process_port: 29500

deepspeed_config:
  gradient_accumulation_steps: 4
  zero_stage: 2                 # ZeRO Stage 2
  gradient_clipping: 1.0
  offload_optimizer_device: cpu  # 优化器状态放 CPU (可选)
  offload_param_device: cpu      # 参数放 CPU (可选)
```

---

## 5. 如何选择:Accelerate 还是 DeepSpeed

### 🎯 推理场景

| 场景 | 推荐方案 | 理由 |
|------|---------|------|
| **交互式聊天** | 直接启动 (`llamafactory-cli chat`) | 利用 `device_map="auto"` 自动切分模型 |
| **批量推理** | `accelerate launch` | 简单、稳定、易排错 |
| **大模型推理 (70B+)** | 直接启动 (多卡) | 模型并行,无需多进程 |

**结论**: 推理场景优先用 **Accelerate** 或直接启动

---

### 🎯 训练场景

| 模型规模 | 显存需求 | 推荐方案 |
|---------|---------|---------|
| **7B 及以下** | 单卡够用 | Accelerate (简单) |
| **13B-30B** | 需要优化 | Accelerate + DeepSpeed ZeRO-2 |
| **70B+** | 显存紧张 | Accelerate + DeepSpeed ZeRO-3 + Offload |

**结论**: 训练场景优先用 **DeepSpeed** (ZeRO 技术)

---

### 📊 对比总结

| 特性 | Accelerate | DeepSpeed |
|------|-----------|-----------|
| **上手难度** | ⭐⭐ 简单 | ⭐⭐⭐⭐ 复杂 |
| **配置复杂度** | 低 (基本无需配置) | 高 (需要 yaml 配置) |
| **显存优化** | 一般 | ⭐⭐⭐⭐⭐ 极强 (ZeRO) |
| **训练速度** | 好 | 更好 (批量大时) |
| **推理场景** | ⭐⭐⭐⭐ 推荐 | ⭐⭐ 不常用 |
| **训练场景** | ⭐⭐⭐ 够用 | ⭐⭐⭐⭐⭐ 推荐 |
| **排错难度** | 容易 | 较难 |

---

## 6. GPU 映射说明

### 🎯 为什么同时出现 `CUDA_VISIBLE_DEVICES` 和 `--gpus`?

**场景**: 你有 8 张卡 (0-7),但只想用空闲的 6 号和 7 号卡

```bash
CUDA_VISIBLE_DEVICES=6,7 accelerate launch --num_processes 2 ... --gpus 0,1
```

**映射关系**:

```
物理 GPU:
  GPU 0  GPU 1  GPU 2  GPU 3  GPU 4  GPU 5  GPU 6  GPU 7
                                          ↓       ↓
可见 GPU (CUDA_VISIBLE_DEVICES=6,7):
  可见 GPU 0  可见 GPU 1
    ↓           ↓
映射到 (--gpus 0,1):
  进程 0      进程 1
  物理卡 6    物理卡 7
```

**记忆方法**:
- `CUDA_VISIBLE_DEVICES`: 过滤物理卡,只让进程"看见"指定的卡
- `--gpus`: 在"可见卡"中的索引 (永远从 0 开始)

**示例**:

```bash
# 示例 1: 使用物理卡 2,5,7
CUDA_VISIBLE_DEVICES=2,5,7 ... --gpus 0,1,2
# 可见卡 0 → 物理卡 2
# 可见卡 1 → 物理卡 5
# 可见卡 2 → 物理卡 7

# 示例 2: 使用物理卡 0,1 (推荐,代码更统一)
CUDA_VISIBLE_DEVICES=0,1 ... --gpus 0,1
# 可见卡 0 → 物理卡 0
# 可见卡 1 → 物理卡 1
```

**优势**:
- 不管物理卡号是多少,都能用 `--gpus 0,1,...` 这种固定写法
- 代码更通用,不需要针对不同机器修改

---

## 7. 性能对比与监控

### 📊 关键指标

做"单卡 vs 多卡"对比时,建议记录:

| 指标 | 说明 | 如何监控 |
|------|------|---------|
| **显存峰值** | 每张卡的最大显存占用 | `nvidia-smi` 或 `watch -n 1 nvidia-smi` |
| **吞吐速度** | 每秒生成的 token 数 | 工具输出中的 `tokens/s` |
| **总耗时** | 完成任务的总时间 | 命令执行时间 |
| **首 token 延迟** | 开始生成到第一个 token 的时间 | 手动计时或工具输出 |

---

### 🔍 监控命令

#### 实时监控 GPU
```bash
# 方式 1: 每秒刷新
watch -n 1 nvidia-smi

# 方式 2: 更详细的监控
nvidia-smi dmon -s u -d 1

# 方式 3: 记录到文件
nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free --format=csv -l 1 > gpu_log.csv
```

#### 分析日志
```bash
# 查看吞吐
grep "tokens/s" output.log

# 查看显存峰值
grep "max memory" output.log
```

---

### 📈 性能预期

**经验判断** (不承诺具体数值):

| 模型规模 | 单卡显存 | 多卡收益 |
|---------|---------|---------|
| **7B** | 16-24GB | ⭐⭐ 收益有限 (通信开销) |
| **13B** | 24-32GB | ⭐⭐⭐ 明显提升 |
| **70B+** | 需要 4x A100 | ⭐⭐⭐⭐ 必须用多卡 |

**多卡的主要收益**:
- ✅ **容纳更大模型** (显存分摊)
- ✅ **批量处理时提升吞吐** (数据并行)
- ⚠️ 单次推理时,收益可能不明显 (甚至可能变慢,因为通信开销)

---

## 8. 常见问题排错

### 🔥 问题 1: 端口被占用

**错误信息**:
```
Address already in use
OSError: [Errno 98] Address already in use
```

**解决方案**:
```bash
# 方案 1: 修改端口
accelerate launch --main_process_port 29502 ...  # 改成其他端口

# 方案 2: 杀掉占用端口的进程
lsof -i :29501          # 查看占用端口的进程
kill -9 <PID>           # 杀掉进程

# Windows:
netstat -ano | findstr :29501
taskkill /PID <PID> /F
```

---

### 🔥 问题 2: GPU 用错了

**症状**: 想用卡 6,7,实际用了卡 0,1

**排查步骤**:
```bash
# 1. 检查环境变量
echo $CUDA_VISIBLE_DEVICES  # Linux
echo %CUDA_VISIBLE_DEVICES% # Windows

# 2. 检查可见卡
python -c "import torch; print(torch.cuda.device_count())"

# 3. 检查进程使用的卡
nvidia-smi

# 4. 确认命令参数
# 确保 --gpus 用的是可见卡索引,不是物理卡号
```

**正确写法**:
```bash
# ✅ 正确
CUDA_VISIBLE_DEVICES=6,7 ... --gpus 0,1

# ❌ 错误
CUDA_VISIBLE_DEVICES=6,7 ... --gpus 6,7  # 错误!应该用 0,1
```

---

### 🔥 问题 3: 离线模式误联网

**症状**: 离线环境仍然尝试下载模型

**解决方案**:
```bash
# 确保环境变量都在同一行命令中
CUDA_VISIBLE_DEVICES=0,1 \
HF_HUB_OFFLINE=1 \              # 必须
TRANSFORMERS_OFFLINE=1 \        # 必须
accelerate launch ...

# 或者设置环境变量
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

---

### 🔥 问题 4: Flash Attention 报错

**错误信息**:
```
ImportError: Flash Attention only support GPU
RuntimeError: Flash attention only support GPU
```

**解决方案**:
```bash
# 禁用 flash attention
TRANSFORMERS_NO_FLASH_ATTENTION=1 accelerate launch ...

# 或者在代码中设置
use_flash_attn=False
```

---

### 🔥 问题 5: 多卡推理没效果 (速度没提升)

**可能原因**:

1. **用了错误的多卡模式**
   - 单次推理用了数据并行 (应该用模型并行)
   - 检查: 是否用了 `accelerate launch` 而不是直接启动

2. **通信开销大于计算收益**
   - 小模型 (7B) 多卡可能反而更慢
   - 检查: 模型规模是否适合多卡

3. **实际只用了一张卡**
   - 检查: `nvidia-smi` 确认每张卡的显存占用
   - 如果只有一张卡显存占用高,说明没有真正多卡并行

**验证方法**:
```bash
# 启动前记录
nvidia-smi

# 启动后立即查看
nvidia-smi

# 应该看到多张卡的显存占用都增加了
```

---

### 🔥 问题 6: 多进程死锁

**症状**: 程序卡住不动,没有输出

**可能原因**:
- 进程间通信失败
- 主进程端口被防火墙拦截
- NCCL 环境配置问题

**解决方案**:
```bash
# 设置 NCCL 调试信息
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1  # 禁用 InfiniBand (如果不需要)

# 使用 TCP 代替 IB
export NCCL_SOCKET_IFNAME=eth0

# 减少进程数测试
accelerate launch --num_processes 1 ...  # 先测试单进程
```

---

## 📝 总结与最佳实践

### ✅ 推荐工作流程

1. **明确任务类型**
   - 聊天? → 直接启动
   - 批量推理? → `accelerate launch`
   - 训练? → `accelerate launch` + DeepSpeed

2. **检查 GPU 资源**
   ```bash
   nvidia-smi  # 查看空闲卡
   ```

3. **选择正确的命令模板**
   - 参考"实战命令指南"章节

4. **先小规模测试**
   - 小数据集
   - 少进程
   - 验证成功后再扩大规模

5. **监控性能**
   - 显存占用
   - 吞吐速度
   - 调整参数优化

---

### 🎯 快速决策表

| 你要做什么 | 用什么 | 命令模板 |
|-----------|--------|---------|
| 聊天/演示 | 直接启动 | `llamafactory-cli chat` |
| 批量推理 | Accelerate | `accelerate launch -m llamafactory.launcher --do_predict` |
| 小模型训练 | Accelerate | `accelerate launch -m llamafactory.launcher --stage sft` |
| 大模型训练 | DeepSpeed | `accelerate launch --config_file ds_zero2.yaml -m llamafactory.launcher` |

---

### 📚 推荐阅读

- [Accelerate 官方文档](https://huggingface.co/docs/accelerate/index)
- [DeepSpeed 官方文档](https://www.deepspeed.ai/)
- [LLaMA-Factory 文档](https://github.com/hiyouga/LLaMA-Factory)

---

**文档版本**: v2.0 (优化版)
**最后更新**: 2025-01-09
**维护者**: [Your Name]
