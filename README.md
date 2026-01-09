# 大模型推理与微调学习项目

## 环境配置

| 项目 | 配置 |
|------|------|
| GPU | 2× RTX 4090 (24GB×2 = 48GB 总显存) |
| 推荐模型规模 | 7B 全精度 / 14B 量化 / 32B 双卡张量并行 |
| Conda 环境 | videofen |
| PyTorch | 2.3.1 + CUDA 12.1 |

---

## 学习路径

```
vLLM (推理) → Accelerate (分布式抽象) → DeepSpeed (显存优化) → LLaMA-Factory (集成框架)
```

---

## 一、vLLM - 高性能推理引擎

### 为什么用 vLLM？

传统 HuggingFace 推理的问题：
- KV Cache 按最大长度预分配，浪费显存
- 请求串行处理，GPU 利用率低

vLLM 的解决方案：
| 技术 | 原理 | 效果 |
|------|------|------|
| **PagedAttention** | KV Cache 按需分页分配（类似操作系统虚拟内存） | 显存利用率 ↑ 2-4× |
| **Continuous Batching** | 动态插入/移除请求，不等整批完成 | 吞吐量 ↑ 2-24× |
| **Tensor Parallelism** | 模型切分到多卡，每卡只算一部分 | 支持更大模型 |

### 双卡部署实操

```bash
# 1. 安装
pip install vllm

# 2. 启动 OpenAI 兼容 API（双卡张量并行）
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-14B-Instruct \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --port 8000

# 3. 测试
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "Qwen/Qwen2.5-14B-Instruct", "messages": [{"role": "user", "content": "你好"}]}'
```

### GPU 监控工具

| 工具 | 用途 | 命令 |
|------|------|------|
| **nvidia-smi** | 基础监控（显存、利用率） | `watch -n 1 nvidia-smi` |
| **nvitop** | 交互式 TUI，类似 htop | `pip install nvitop && nvitop` |
| **gpustat** | 简洁单行输出 | `pip install gpustat && gpustat -i 1` |
| **Prometheus + Grafana** | 生产级监控 | 见 `modules/03_vllm/` |

**关键指标：**
- `GPU-Util`：计算利用率，推理时应 > 80%
- `Memory-Usage`：显存占用，vLLM 应接近 `gpu-memory-utilization` 设定值
- `Power`：功耗，4090 峰值 450W，持续高功耗说明负载饱和

详细内容见：`modules/03_vllm/`

---

## 二、Accelerate - 分布式训练抽象层

### 为什么用 Accelerate？

**问题**：PyTorch 原生分布式代码繁琐
```python
# 原生写法：到处都是分布式判断
if torch.distributed.get_rank() == 0:
    save_model()
model = DDP(model, device_ids=[local_rank])
```

**Accelerate 的价值**：一套代码，多种硬件
```python
# Accelerate 写法：干净统一
from accelerate import Accelerator
accelerator = Accelerator()
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
# 后续代码完全一样，自动适配单卡/多卡/TPU
```

### 核心概念

| 概念 | 说明 |
|------|------|
| `Accelerator` | 核心类，自动检测硬件环境 |
| `prepare()` | 包装模型/优化器/数据，注入分布式逻辑 |
| `backward()` | 替代 `loss.backward()`，自动处理梯度同步 |
| `accelerate config` | 交互式配置分布式策略 |
| `accelerate launch` | 统一启动器，替代 `torchrun` |

### 使用场景（不限于大模型）

1. **单机多卡数据并行**：最常见，每卡跑不同 batch
2. **混合精度训练**：FP16/BF16 自动管理
3. **梯度累积**：小显存模拟大 batch
4. **DeepSpeed/FSDP 集成**：一行配置切换后端

```bash
# 配置向导
accelerate config

# 启动训练（自动读取配置）
accelerate launch train.py
```

详细内容见：`modules/04_Accelerate/`

---

## 三、DeepSpeed - 显存优化利器

### 为什么用 DeepSpeed？

**核心矛盾**：模型太大，显存不够

**传统数据并行 (DDP)**：每张卡都有完整的模型副本 + 优化器状态 + 梯度
- 7B 模型，FP32 训练：28GB 参数 + 56GB 优化器状态 + 28GB 梯度 = 112GB/卡

**DeepSpeed ZeRO 的思路**：把冗余的东西切分到不同卡

### ZeRO 三阶段

| 阶段 | 切分内容 | 显存节省 | 通信开销 | 适用场景 |
|------|----------|----------|----------|----------|
| **ZeRO-1** | 优化器状态 | 4× | 几乎无增加 | 默认首选 |
| **ZeRO-2** | + 梯度 | 8× | 略有增加 | 大多数微调 |
| **ZeRO-3** | + 模型参数 | 线性扩展 | 明显增加 | 超大模型 |

**ZeRO-Offload**：显存还不够？把数据卸载到 CPU 内存甚至 NVMe

### 关键配置 (ds_config.json)

```json
{
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {"device": "cpu"},
    "overlap_comm": true
  },
  "bf16": {"enabled": true},
  "gradient_accumulation_steps": 4
}
```

### 与 Accelerate 配合

```bash
# 方式一：通过 Accelerate 配置 DeepSpeed
accelerate config  # 选择 DeepSpeed，指定 stage

# 方式二：直接用 DeepSpeed 启动器
deepspeed --num_gpus=2 train.py --deepspeed ds_config.json
```

详细内容见：`modules/05_DeepSpeed/`

---

## 四、LLaMA-Factory - 统一微调框架

### 它集成了什么？

```
┌─────────────────────────────────────────────────────┐
│                  LLaMA-Factory                       │
├─────────────────────────────────────────────────────┤
│  推理后端    │ HuggingFace / vLLM / SGLang          │
│  训练后端    │ Accelerate + DeepSpeed / FSDP        │
│  微调方法    │ Full / LoRA / QLoRA / DoRA           │
│  数据格式    │ 统一 JSON 格式，支持多模态            │
│  界面       │ CLI / Web UI / API                    │
└─────────────────────────────────────────────────────┘
```

### 原理：如何调用这些工具？

1. **入口统一**：`llamafactory-cli` 命令行 / `src/train.py` 脚本
2. **配置驱动**：YAML 配置文件指定模型、数据、训练策略
3. **后端选择**：
   - 指定 `--deepspeed ds_config.json` → 启用 DeepSpeed
   - 指定 `--infer_backend vllm` → 推理用 vLLM
   - 默认自动使用 Accelerate 管理分布式

### 常用命令

```bash
# 推理（单卡）
CUDA_VISIBLE_DEVICES=0 llamafactory-cli chat \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --template qwen

# 微调（双卡 + DeepSpeed ZeRO-2）
CUDA_VISIBLE_DEVICES=0,1 deepspeed --num_gpus 2 src/train.py \
    --deepspeed examples/deepspeed/ds_z2_bf16.json \
    --stage sft \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --dataset alpaca_zh \
    --template qwen \
    --finetuning_type lora

# Web UI
llamafactory-cli webui
```

---

## 模块目录说明

```
modules/
├── 01_single_gpu_smoke/    # 单卡冒烟测试（环境验证）
├── 02_qwen_image_edit/     # 图像编辑模型部署
├── 03_vllm/                # vLLM 详细教程
├── 04_Accelerate/          # Accelerate 详细教程
└── 05_DeepSpeed/           # DeepSpeed 详细教程
```

---

## 指导文件说明

| 文件 | 用途 |
|------|------|
| `README.md` | 项目总览与学习路径（本文件） |
| `.CLAUDE.md` | AI 助手上下文（环境快照、约束） |
| `modules/*/readme.md` | 各模块详细教程 |
| `modules/*/PLAN.md` | 具体任务清单 |
