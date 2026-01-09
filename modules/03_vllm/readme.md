# vLLM 完全指南：从原理到实战

> 大模型初学者必读：全方位理解 vLLM 的定位、原理、实战应用

---

## 目录

1. [vLLM 是什么](#1-vllm-是什么)
2. [核心原理详解](#2-核心原理详解)
3. [并行策略解析](#3-并行策略解析)
4. [工具生态对比](#4-工具生态对比)
5. [快速上手](#5-快速上手)
6. [实战学习路线](#6-实战学习路线)
7. [常见问题 FAQ](#7-常见问题-faq)

---

## 1. vLLM 是什么

### 1.1 定位

**vLLM 是一个专注于大模型推理的高性能引擎，不是训练工具。**

| 维度 | 说明 |
|------|------|
| **核心功能** | 快速、高效地部署大语言模型 |
| **主要场景** | 在线 API 服务、高并发推理 |
| **开发团队** | UC Berkeley 等 |
| **开源协议** | Apache 2.0 |

### 1.2 在工具链中的位置

```
┌─────────────────────────────────────────────────────────────┐
│                      大模型生命周期                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  预训练    →   微调   →   量化   →   部署推理               │
│  (DeepSpeed)  (LlamaFactory)   (AWQ)     (vLLM)            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**一句话**: vLLM 是模型微调后，用来部署上线的最后一公里工具。

---

## 2. 核心原理详解

### 2.1 PagedAttention - 显存管理革命

#### 问题：传统 KV Cache 的显存浪费

```
传统方式：预分配连续显存块
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Seq1: [████████████████░░░░░░░░░░░░░░░░░░] 预留 2048，实际用 800
Seq2: [████████████████████████░░░░░░░░░░] 预留 2048，实际用 1400
Seq3: [████████████░░░░░░░░░░░░░░░░░░░░░] 预留 2048，实际用 600
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
问题：
  • 必须按最大序列长度预分配
  • 显存碎片化严重
  • 平均浪费 60-80% 显存空间
```

#### 解决：PagedAttention 分页机制

```
PagedAttention：分页式管理
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
物理显存块 (Block，每个 16 个 Token)
┌────┬────┬────┬────┬────┬────┬────┬────┐
│ 0  │ 1  │ 2  │ 3  │ 4  │ 5  │ 6  │ 7  │
└────┴────┴────┴────┴────┴────┴────┴────┘

逻辑映射：
Seq1 → [Block 0][Block 2][Block 5]  ← 不连续也能用
Seq2 → [Block 1][Block 3]
Seq3 → [Block 4][Block 6][Block 7]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
优势：
  ✓ 显存利用率接近 100%
  ✓ 支持动态扩容（缺页就分配）
  ✓ 内存共享（相同 Prompt 可共享 Cache）
```

#### 技术类比

| PagedAttention | 操作系统 |
|----------------|----------|
| KV Cache Block | 内存页（Page） |
| Block Manager | 内存管理器（MMU） |
| 显存池 | 物理内存 |

---

### 2.2 Continuous Batching - 吞吐量优化

#### 问题：静态批处理的低效

```
传统 Static Batching：
时间 →
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Seq1: [██████████████████████████████████] 100 tokens
Seq2: [████████████]----空转等待------------ 40 tokens
Seq3: [██████████████████]----等待--------- 70 tokens
      ↑              ↑
   Batch 开始    所有请求必须一起结束
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
问题：短请求完成后，GPU 空转等待长请求
```

#### 解决：Continuous Batching（连续批处理）

```
Continuous Batching：
时间 →
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Seq1: [██████████████████████████████████] 完成
Seq2: [████████████][新请求1 ████████]     动态插入
Seq3: [██████████████████][新请求2 █████]  动态插入
      ↑             ↑              ↑
   Batch 开始   Seq2完成插入新请求  Seq3完成插入新请求
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
优势：
  ✓ GPU 始终满载运行
  ✓ 吞吐量提升 3-10 倍
  ✓ 适合在线服务场景
```

#### 餐厅类比

| Static Batching | Continuous Batching |
|-----------------|---------------------|
| 等所有人点完菜才做菜 | 来一个菜做一个 |
| 吃得快的人要等吃得慢的 | 吃完就走，翻台率高 |

---

### 2.3 核心指标体系

| 指标 | 全称 | 说明 | 优化目标 |
|------|------|------|----------|
| **TTFT** | Time to First Token | 首字延迟 | 越低越好（用户体验） |
| **TPOT** | Time Per Output Token | 每个生成 Token 的延迟 | 越低越好 |
| **Throughput** | Requests/Second | 每秒处理请求数 | 越高越好 |
| **Token/s** | Tokens Per Second | 每秒生成 Token 数 | 越高越好 |

```
TTFT = 从发请求 → 收到第一个 Token 的时间
TPOT = 每个后续 Token 的平均间隔时间
总延迟 = TTFT + (生成Token数 - 1) × TPOT
```

---

## 3. 并行策略解析

### 3.1 vLLM 支持的并行方式

| 并行类型 | vLLM 支持 | 适用场景 | 命令参数 |
|---------|----------|----------|----------|
| **张量并行 (TP)** | ✅ | 单个请求太大会显存不足 | `--tensor-parallel-size` |
| **流水线并行 (PP)** | ✅ | 模型层数太多 | `--pipeline-parallel-size` |
| **数据并行 (DP)** | ❌ | 推理场景意义不大 | - |

### 3.2 张量并行详解

#### 什么是张量并行？

将单个请求的计算分散到多张 GPU 上。

```
示例：4 卡运行 72B 模型

GPU 0:  [第 1-18 层] + 注意力头 1/4
GPU 1:  [第 19-36 层] + 注意力头 2/4
GPU 2:  [第 37-54 层] + 注意力头 3/4
GPU 3:  [第 55-72 层] + 注意力头 4/4

推理时：
  输入 → GPU0 并行计算 → GPU1 并行计算 → GPU2 → GPU3 → 输出
```

#### 使用示例

```bash
# 单卡运行 7B 模型
vllm serve Qwen/Qwen2-7B-Instruct

# 2 卡运行 14B 模型（张量并行）
vllm serve Qwen/Qwen2-14B-Instruct --tensor-parallel-size 2

# 4 卡运行 72B 模型
vllm serve Qwen/Qwen2-72B-Instruct --tensor-parallel-size 4
```

### 3.3 为什么推理不用数据并行？

| 场景 | 数据并行 | 张量并行 |
|------|---------|---------|
| **训练** | ✅ 主流 | 大模型用 |
| **原因** | 大 Batch，能线性加速 | 降低单卡显存压力 |
| **推理** | ❌ 意义小 | ✅ 主流 |
| **原因** | 推理追求低延迟，单请求尽快返回 | 单请求太大，必须分到多卡 |

**简单理解**：
- 数据并行 = 多个复制一起跑（适合大批量）
- 张量并行 = 多个人一起抬大桌子（适合单个太重）

---

## 4. 工具生态对比

### 4.1 训练 vs 推理工具全景图

```
┌─────────────────────────────────────────────────────────────┐
│                    大模型工具链全景                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  训练/微调阶段                    推理/部署阶段              │
│  ┌─────────────────────┐         ┌─────────────────────┐   │
│  │ LlamaFactory        │         │ vLLM                │   │
│  │ ├─ WebUI            │         │ ├─ 高吞吐           │   │
│  │ ├─ 100+ 模型        │         │ ├─ PagedAttention   │   │
│  │ └─ 一站式微调       │         │ └─ API 兼容         │   │
│  └─────────────────────┘         └─────────────────────┘   │
│           ↓                              ↓                  │
│  ┌─────────────────────┐         ┌─────────────────────┐   │
│  │ Accelerate          │         │ TGI                 │   │
│  │ ├─ 统一抽象         │         │ ├─ HuggingFace 官方 │   │
│  │ └─ 简化分布式       │         │ └─ 生产级稳定       │   │
│  └─────────────────────┘         └─────────────────────┘   │
│           ↓                              ↓                  │
│  ┌─────────────────────┐         ┌─────────────────────┐   │
│  │ DeepSpeed           │         │ Ollama              │   │
│  │ ├─ ZeRO 优化        │         │ ├─ 本地一键运行     │   │
│  │ └─ 极限省显存       │         │ └─ 简单易用         │   │
│  └─────────────────────┘         └─────────────────────┘   │
│                                                             │
│  通用工具                                                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ huggingface-cli: 模型下载/上传                      │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 推理工具对比表

| 工具 | 定位 | 优势 | 劣势 | 适用场景 |
|------|------|------|------|----------|
| **vLLM** | 高性能推理 | 吞吐量极高、PagedAttention | 相对新，生态略少 | **生产环境高并发** |
| **TGI** | HuggingFace 推理 | 稳定、功能全面 | 配置较复杂 | 企业级部署 |
| **Ollama** | 本地简易部署 | 极简、一键运行 | 性能一般 | **个人学习/测试** |
| **llama.cpp** | CPU 推理 | 支持 CPU、量化强 | 不适合大并发 | 边缘设备/Mac |

### 4.3 选择建议

```
你的需求是什么？

┌─ 个人学习/测试
│  └─→ Ollama（最简单）
│
├─ 生产环境高并发 API
│  └─→ vLLM（首选）
│
├─ 企业级稳定部署
│  └─→ TGI
│
└─ 无 GPU 场景/边缘设备
   └─→ llama.cpp
```

---

## 5. 快速上手

### 5.1 安装

```bash
# 方式 1：pip 安装（推荐）
pip install vllm

# 方式 2：从源码安装（最新特性）
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .

# 验证安装
python -c "import vllm; print(vllm.__version__)"
```

### 5.2 基础使用

#### 方式 1：命令行启动（OpenAI API 兼容）

```bash
# 启动服务
vllm serve Qwen/Qwen2-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 2048

# 调用（与 OpenAI API 完全一致）
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2-7B-Instruct",
    "messages": [{"role": "user", "content": "你好"}],
    "temperature": 0.7
  }'
```

#### 方式 2：Python API

```python
from vllm import LLM, SamplingParams

# 初始化模型
llm = LLM(
    model="Qwen/Qwen2-7B-Instruct",
    gpu_memory_utilization=0.9,
    max_model_len=2048,
    tensor_parallel_size=1  # 多卡设为 2/4/8
)

# 生成参数
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512
)

# 批量推理
prompts = ["你好，介绍一下自己", "什么是人工智能？"]
outputs = llm.generate(prompts, sampling_params)

# 打印结果
for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated: {output.outputs[0].text}\n")
```

#### 方式 3：多卡部署

```bash
# 2 卡张量并行
vllm serve Qwen/Qwen2-14B-Instruct \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.9

# 4 卡张量并行（72B 模型）
vllm serve Qwen/Qwen2-72B-Instruct \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.95
```

### 5.3 核心参数说明

| 参数 | 说明 | 默认值 | 调优建议 |
|------|------|--------|----------|
| `--gpu-memory-utilization` | GPU 显存利用率 | 0.9 | 显存充足可设 0.95 |
| `--max-model-len` | 最大上下文长度 | 模型配置值 | 需求越长占用越多显存 |
| `--tensor-parallel-size` | 张量并行 GPU 数 | 1 | 单卡不够时增加 |
| `--block-size` | KV Cache 块大小 | 16 | 通常不需调整 |
| `--max-num-seqs` | 最大并发序列数 | 256 | 根据实际并发调整 |

### 5.4 性能测试

```bash
# 使用 vLLM 自带基准测试脚本
python benchmarks/benchmark_serving.py \
  --model Qwen/Qwen2-7B-Instruct \
  --dataset-name sharegpt \
  --dataset-path ./data/sharegpt.json \
  --request-rate 10  # 每秒 10 个请求

# 输出关键指标
# - TTFT (Time to First Token)
# - TPOT (Time Per Output Token)
# - Throughput (requests/second)
```

---

## 6. 实战学习路线

> 从 0 到 1 搭建生产级推理系统的 4 周闭环计划

### 第一阶段：生产级平替（从 Ollama 到 vLLM）

**目标：** 掌握工业级推理框架的配置与压测，建立性能基准。

#### 第 1-3 天：核心参数调优

**任务：** 丢掉 Ollama，用 vLLM 部署你最常用的模型（如 Qwen2.5-7B）。

```bash
# 实操：对比不同参数配置
# 配置 1：保守配置
vllm serve Qwen/Qwen2-7B-Instruct \
  --gpu-memory-utilization 0.7 \
  --max-model-len 1024

# 配置 2：激进配置
vllm serve Qwen/Qwen2-7B-Instruct \
  --gpu-memory-utilization 0.95 \
  --max-model-len 4096
```

**关键点：** 理解 vLLM 的 **PagedAttention** 如何解决显存碎片化问题。

**产出：** 一份参数对比表（显存占用 vs 上下文长度）

---

#### 第 4-7 天：基准测试（Benchmarking）

**任务：** 使用 vLLM 自带的 `benchmark_serving.py` 进行压力测试。

```bash
# 准备测试数据集
wget https://raw.githubusercontent.com/vllm-project/vllm/main/benchmarks/sharegpt.json

# 运行压测
python benchmarks/benchmark_serving.py \
  --model Qwen/Qwen2-7B-Instruct \
  --dataset-name sharegpt \
  --dataset-path ./sharegpt.json \
  --num-prompts 1000 \
  --request-rate 5
```

**测试矩阵：**

| 并发数 | Request Rate | TTFT (ms) | TPOT (ms) | Throughput (req/s) |
|--------|--------------|-----------|-----------|-------------------|
| 1      | 1            | ?         | ?         | ?                 |
| 10     | 5            | ?         | ?         | ?                 |
| 50     | 10           | ?         | ?         | ?                 |
| 100    | 20           | ?         | ?         | ?                 |

**产出：** 一份性能报告，搞清楚你的显存能支撑多少并发上限。

---

### 第二阶段：可观测性落地（监控栈搭建）

**目标：** 告别 `nvidia-smi`，建立标准化的监控大屏。

#### 第 8-10 天：监控链路打通

**任务：** 启动 Prometheus 抓取 vLLM 的 `/metrics` 接口。

```yaml
# docker-compose.yml
version: '3.8'
services:
  vllm:
    image: vllm/vllm-openai:latest
    command: >
      --model Qwen/Qwen2-7B-Instruct
      --gpu-memory-utilization 0.9
    ports:
      - "8000:8000"
    environment:
      - VLLM_USAGE_SOURCE=production

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
```

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'vllm'
    static_configs:
      - targets: ['vllm:8000']
    metrics_path: /metrics
```

**实操：** 部署 `dcgm-exporter`（获取 GPU 细粒度指标）

```bash
# 部署 DCGM Exporter
docker run -d \
  --name=dcgm-exporter \
  --gpus all \
  -p 9400:9400 \
  nvidia/dcgm-exporter:latest
```

---

#### 第 11-14 天：Grafana 大盘配置

**任务：** 导入或手搭一个仪表板。

**核心图表：**

1. **Request Stats**
   - 当前运行中的请求数
   - 排队中的请求数
   - 请求成功率

2. **KV Cache Usage**
   - 显存池剩余百分比
   - 判断是否需要扩容

3. **Throughput**
   - 每秒 Token 生成数
   - 每秒处理请求数

4. **GPU Metrics**
   - GPU 利用率
   - 显存带宽利用率
   - 温度监控

**产出：** 一个能实时观察压力变化的可视化面板。

---

### 第三阶段：极限性能优化（量化与高级调度）

**目标：** 在不增加硬件的前提下，提升 2-3 倍吞吐。

#### 第 15-18 天：量化实战

**任务：** 使用 **AutoAWQ** 或 **AutoGPTQ** 对模型进行 INT4/FP8 量化。

```bash
# 安装量化工具
pip install autoawq

# 量化模型
python -m awq.fuse \
  --model_path Qwen/Qwen2-7B-Instruct \
  --w_bit 4 \
  --q_group_size 128

# 使用量化后的模型
vllm serve Qwen/Qwen2-7B-Instruct-AWQ-INT4 \
  --quantization awq \
  --gpu-memory-utilization 0.9
```

**对比：** 测量量化前后的指标变化

| 指标 | FP16 | INT4 | 提升 |
|------|------|------|------|
| 显存占用 | 14 GB | 4 GB | 71% ↓ |
| TTFT | 120 ms | 80 ms | 33% ↑ |
| Token/s | 45 | 65 | 44% ↑ |

---

#### 第 19-21 天：进阶调度技术

**任务：** 尝试 **Speculative Decoding**（投机采样）。

```bash
# 用小模型作为草稿模型，加速大模型
vllm serve Qwen/Qwen2-7B-Instruct \
  --speculative-model Qwen/Qwen-0.5B \
  --num-speculative-tokens 5
```

**关键点：** 观察在不同 Prompt 复杂度下，加速比的变化。

**原理：**
```
小模型（草稿） → 快速生成候选 Token
     ↓
大模型（验证） → 并行验证多个 Token
     ↓
接受/拒绝 → 拒绝则回退，接受则加速
```

**产出：** 量化/投机解码的加速对比报告。

---

### 第四阶段：深度诊断与多模态扩展（技术护城河）

**目标：** 能讲清瓶颈在哪里，并具备处理图像/视频模型的能力。

#### 第 22-25 天：瓶颈分析（Profiling）

**任务：** 使用 **NVIDIA Nsight Systems** 抓取推理过程中的 Kernel 执行。

```bash
# 安装 Nsight Systems
# 下载：https://developer.nvidia.com/nsight-systems

# Profile vLLM 推理
nsys profile \
  --output=report.qdrep \
  --force-overwrite=true \
  python your_inference_script.py

# 分析报告
# 识别瓶颈：
# - Compute-bound（计算受限）→ 升级 GPU
# - Memory-bound（带宽受限）→ 启用量化
```

**实操：** 识别当前是 **Compute-bound** 还是 **Memory-bound**。

---

#### 第 26-28 天：多模态/Diffusion 部署

**任务 A：** 用 vLLM 部署 VLM 模型（LLaVA）

```bash
vllm serve llava-hf/llava-1.5-7b-hf \
  --max-model-len 4096 \
  --limit-mm-per-prompt image=1
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="empty")

response = client.chat.completions.create(
    model="llava-hf/llava-1.5-7b-hf",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "描述这张图片"},
            {"type": "image_url", "image_url": {"url": "image.jpg"}}
        ]
    }]
)
```

**任务 B：** 用 TensorRT 加速 Stable Diffusion

```bash
# 使用 TensorRT 加速 SD
trtexec --onnx=model.onnx --saveEngine=model.trt
```

**产出：** 跑通一个"文生图"或"图生文"的完整 Pipeline。

---

### 学习路线总结表

| 阶段 | 核心工具 | 解决的问题 | 关键产出 |
|------|----------|-----------|----------|
| **1. 推理层** | vLLM | 摆脱 Demo 级工具，进入生产级调度 | 性能基准报告 (Latency vs. Throughput) |
| **2. 监控层** | Prometheus + Grafana | 变"盲目运行"为"透明监控" | 实时性能大盘 |
| **3. 优化层** | AWQ / Speculative Decoding | 压榨硬件性能，降低部署成本 | 量化/加速对比数据 |
| **4. 专家层** | Nsight / vLLM-Omni | 掌握底层诊断，扩展到多模态领域 | 瓶颈分析图谱 / 多模态 Demo |

---

## 7. 常见问题 FAQ

### Q1: vLLM 和 Ollama 怎么选？

**A:**

```
选择决策树：

你的场景是什么？
├─ 个人学习/本地测试
│  ├─ 只想快速跑起来
│  │  └─→ Ollama（一条命令搞定）
│  └─ 想学习生产级部署
│     └─→ vLLM（学习曲线陡但能力强）
│
└─ 生产环境部署
   ├─ 高并发 API 服务
   │  └─→ vLLM（吞吐量高 3-10 倍）
   └─ 简单内部工具
      └─→ Ollama（部署简单）
```

---

### Q2: 多卡部署时，TP 和 PP 怎么选？

**A:**

| 场景 | 推荐方案 | 原因 |
|------|---------|------|
| 4 卡以内 | 张量并行 (TP) | 通信开销小，实现简单 |
| 4-8 卡 | TP + PP 混合 | 平衡通信和负载 |
| 8 卡以上 | TP + PP | 必须组合使用 |

```bash
# 8 卡推荐配置
vllm serve Qwen/Qwen2-72B-Instruct \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 2
```

---

### Q3: vLLM 支持哪些模型？

**A:** 支持主流 HuggingFace 格式模型：

- ✅ Llama 系列（Llama2, Llama3）
- ✅ Qwen 系列（Qwen, Qwen2）
- ✅ ChatGLM 系列
- ✅ Baichuan 系列
- ✅ Mistral 系列
- ✅ 多模态模型（LLaVA）

**注意：** 需确保模型兼容 HuggingFace Transformers 库。

---

### Q4: 显存不足怎么办？

**A:** 三种方案按优先级：

1. **启用量化**（首选）
   ```bash
   vllm serve model --quantization awq  # 或 gptq, bitsandbytes
   ```

2. **减小上下文长度**
   ```bash
   vllm serve model --max-model-len 1024  # 从 4096 降到 1024
   ```

3. **多卡张量并行**
   ```bash
   vllm serve model --tensor-parallel-size 2
   ```

---

### Q5: 如何监控 vLLM 的性能？

**A:** 三种方式：

```bash
# 方式 1：内置 Metrics
curl http://localhost:8000/metrics

# 方式 2：Prometheus + Grafana
# （见第二阶段监控栈搭建）

# 方式 3：日志分析
vllm serve model --verbose
```

---

### Q6: vLLM 的性能瓶颈在哪里？

**A:** 分三种情况：

| 瓶颈类型 | 表现 | 解决方案 |
|---------|------|---------|
| **Compute-bound** | GPU 利用率高但吞吐低 | 升级 GPU / 减小模型 |
| **Memory-bound** | 显存带宽跑满 | 启用量化 |
| **IO-bound** | CPU/GPU 等待数据 | 优化数据加载 |

**诊断方法：** 使用 Nsight Systems Profile（见第四阶段）。

---

### Q7: vLLM 能用于训练吗？

**A:** ❌ 不能。

vLLM 是专门的推理引擎，不支持训练。

**训练工具链：**
```
预训练 → DeepSpeed
微调 → LlamaFactory
推理 → vLLM
```

---

### Q8: 如何调试 vLLM 性能问题？

**A:** 调试流程：

```
1. 查看日志
   vllm serve model --verbose

2. 检查 Metrics
   curl http://localhost:8000/metrics | grep vllm

3. Profile 分析
   nsys profile python script.py

4. 对照基准
   # 运行官方 benchmark 对比
```

---

## 附录：参考资源

### 官方文档
- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [vLLM 官方文档](https://docs.vllm.ai/)
- [PagedAttention 论文](https://arxiv.org/abs/2309.06180)

### 推荐阅读
- [vLLM: Easy, Fast, and Cheap LLM Serving](https://vllm.ai/)
- [Continuous Batching 技术解析](https://luyuhuang.github.io/2023/08/23/continuous-batching.html)

### 社区资源
- [vLLM Discord](https://discord.gg/vllm)
- [HuggingFace Forums](https://discuss.huggingface.co/)

---

**总结一句话：vLLM 是让大模型推理"又快又省显存"的终极利器，从原理到实战，从部署到优化，掌握它就掌握了生产级大模型服务的关键能力。**
