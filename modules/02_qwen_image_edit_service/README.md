# Qwen-Image-Edit-2511 全栈部署与推理指南

**文档版本**: 2.0.0
**发布日期**: 2025-12-26
**适用环境**: Linux (CentOS/Ubuntu), CUDA 12+, PyTorch 2.3+

---

## 目录

1. [技术概述](#1-技术概述)
2. [模型原理与核心技术](#2-模型原理与核心技术)
3. [推理工具对比](#3-推理工具对比)
4. [部署架构设计](#4-部署架构设计)
5. [环境构建与依赖管理](#5-环境构建与依赖管理)
6. [模型资产本地化](#6-模型资产本地化)
7. [核心代码实现](#7-核心代码实现)
8. [Web 服务封装](#8-web-服务封装)
9. [资源调度与优化](#9-资源调度与优化)
10. [生产级部署方案](#10-生产级部署方案)
11. [故障排查手册](#11-故障排查手册)
12. [快速开始指南](#12-快速开始指南)
13. [附录：完整代码](#13-附录完整代码)

---

效果图：
![Qwen-Image-Edit-2511 效果示例](https://i-blog.csdnimg.cn/direct/d52bced4163948a887cd85eb542ba19a.png)

---

## 1. 技术概述

本指南旨在阐述如何在私有化基础设施中,基于 `Qwen/Qwen-Image-Edit-2511` 模型构建高可用、可扩展的图像编辑推理服务。该方案采用业界标准的 `diffusers` 推理框架,结合中国大陆网络环境下的工程化适配,解决了模型资产管理、异构计算资源调度(GPU/CPU 降级)、显存优化及服务化封装等核心问题。

**核心价值**:
- **开箱即用**: 提供标准化的环境构建与模型加载流程
- **资源弹性**: 支持在 GPU 显存受限或被占用场景下,自动/手动降级至 CPU 推理,确保服务可用性
- **工程规范**: 遵循生产级目录结构、日志规范与配置管理

---

## 2. 模型原理与核心技术

### 2.1 什么是 Qwen-Image-Edit-2511

`Qwen/Qwen-Image-Edit-2511` 是一个**图像编辑(Image Edit)** 模型:
- 给它一张输入图 + 一句编辑指令(prompt),它输出一张修改后的新图
- 重点是"在原图基础上按指令改",而不是从零生成
- 可以理解为: **"可控的图像改写器"**

### 2.2 核心技术一: Diffusion 原理

#### 2.2.1 Diffusion 一句话理解

Diffusion(扩散模型)在推理时通常不是"一步出图",而是**迭代式采样**:
- 从噪声/中间状态开始
- 循环 N 次逐步变清晰
- 最后得到结果图

这个 N 就是你在 UI 里看到的 `Steps / num_inference_steps`。

#### 2.2.2 Steps 是什么、该怎么选

`Steps`=采样迭代次数,影响两件事:
- 质量与稳定性(一般 steps 多更稳,但收益递减)
- 耗时(近似随 steps 线性增长)

在 CPU 冒烟场景:
- 建议 steps 先用 10～15(先确保"能出图")

### 2.3 核心技术二: diffusers Pipeline

#### 2.3.1 为什么用 diffusers

diffusers 是 Hugging Face 的 diffusion 工程框架。
它的价值不是"提出新算法",而是把工程复杂度封装成可复用 pipeline:
- 下载/加载模型组件
- 管理 dtype/device(CPU/GPU、bf16/fp16 等)
- 管理采样循环与 scheduler
- 最终输出图像

本仓库使用官方路线: `diffusers.QwenImageEditPlusPipeline`。

#### 2.3.2 pipeline 组件架构

`QwenImageEditPlusPipeline` 编排了以下关键组件:

| 组件 | 职责 |
|------|------|
| **Text Encoder (Qwen2-VL)** | 把 prompt 变成向量,让模型"理解你要改什么" |
| **VAE** | 图像的潜变量编解码,将高维像素数据压缩为低维表示 |
| **UNet / DiT** | 核心去噪网络,每个 step 都要跑它(最耗时) |
| **Scheduler** | 控制采样步数与去噪轨迹,平衡质量与速度 |

你看到的日志如:
- `Loading pipeline components...`
- `Loading checkpoint shards...`
就是在加载这些组件(权重可能被分成多个 shard)。

### 2.4 核心技术三: Gradio

Gradio 的定位: **最快把一个 Python 推理函数变成网页交互**。

对新手最友好的一点是:
- 你不需要写前端、也不用先做 FastAPI
- 直接把 `edit_image(image, prompt, ...) -> output_image` 绑到按钮即可

并发相关概念:
- `queue`: 把请求排队,避免同时把机器打爆
- `default_concurrency_limit`: 限制同一时间处理多少个请求(本仓库默认 1,保护服务器)

---

## 3. 推理工具对比

| 工具 | 核心优势 | 适用场景 | Qwen-Image-Edit 支持 |
|------|----------|----------|---------------------|
| **vLLM** | ⚡ 高吞吐量、OpenAI 兼容 | LLM 文本生成 | ❌ 不直接支持(需适配) |
| **Diffusers** | ✅ 官方支持、多模态 | 图像生成/编辑 | ✅ 完美支持 |
| **FastAPI** | ✅ 生产级 API 服务 | 企业集成 | ✅ 需自行封装 |
| **Gradio** | ✅ 快速 Web UI | 演示/测试 | ✅ 完美支持 |
| **A1111/ComfyUI** | ✅ 可视化工作流 | 个人使用 | ⚠️ 需自定义节点 |

**推荐方案**:
- **推理层**: Diffusers(官方支持)
- **服务层**: FastAPI(生产 API) + Gradio(Web UI)

### 3.1 是否使用 vLLM

**结论: 第一阶段不依赖 vLLM。**
- vLLM 的主战场是 LLM(KV cache/连续批处理等)
- `Qwen-Image-Edit-2511` 属于 diffusion 管线,最直接的落地方式是 diffusers
- 官方 repo 提到的 **vLLM-Omni / SGLang-Diffusion / LightX2V** 属于"可选加速路线",建议在 diffusers 方案跑通并形成稳定基线后,再评估替换后端

---

## 4. 部署架构设计

### 4.1 单机部署架构

```
┌─────────────────────────────────────────────────────────┐
│  客户端 (浏览器/业务系统)                                │
└──────────────────────────┬──────────────────────────────┘
                          │ HTTP/REST
                          ↓
┌─────────────────────────────────────────────────────────┐
│  API 网关 / Gradio UI                                 │
└──────────────────────────┬──────────────────────────────┘
                          │
                          ↓
┌─────────────────────────────────────────────────────────┐
│  推理服务 (Python + Diffusers)                         │
│  ├── 模型加载与缓存                                    │
│  ├── 资源管理(显存/CPU)                              │
│  └── 推理执行                                         │
└─────────────────────────────────────────────────────────┘
```

### 4.2 多机/容器化架构

```
┌─────────────────────────────────────────────────────────┐
│  负载均衡器 (Nginx/Kong)                               │
└──────────────────────────┬──────────────────────────────┘
                          │
                          ↓
┌─────────────────────────────────────────────────────────┐
│  推理服务集群 (Kubernetes)                             │
│  ├── 服务 1: GPU 节点 (diffusers)                      │
│  ├── 服务 2: CPU 降级节点 (diffusers)                  │
│  └── 服务 3: 监控/日志 (Prometheus/ELK)                │
└─────────────────────────────────────────────────────────┘
```

### 4.3 多 GPU 并发策略

**推荐: 一进程一张卡**

- 启动 N 个 worker 进程(N=4),每个 worker 绑定一张 GPU(逻辑 0..3 映射物理 4..7)
- 每个 worker 进程内: 常驻加载一次 pipeline,循环处理请求
- 外层 API 网关:
  - 维护任务队列
  - 以轮询/最短队列策略将任务分发给空闲 worker
  - 提供 `/health` 查看 worker 存活、GPU、队列长度、最近错误

**优点**:
- 稳定: 避免单进程跨卡切分带来的不确定性
- 可观测: 每张卡的异常隔离,单 worker 崩溃可自动拉起
- 扩展简单: 增加/减少 worker 数量即可

---

## 5. 环境构建与依赖管理

### 5.1 硬件与系统要求

| 组件 | 最低配置 | 推荐配置 | 说明 |
|------|----------|----------|------|
| **GPU** | 24GB VRAM (RTX 3090) | 48GB+ VRAM (A800/A100) | 支持 BF16/FP16 |
| **CPU** | 8 vCPU | 32 vCPU+ | CPU 降级模式 |
| **RAM** | 32GB | 64GB+ | 模型加载与 Offload |
| **Disk** | 50GB SSD | 100GB+ SSD | 模型权重 + 缓存 |

### 5.2 软件依赖栈

#### 5.2.1 创建 Conda 环境

```bash
# 创建 Conda 环境
conda create -n qwen_edit python=3.10 -y
conda activate qwen_edit
```

#### 5.2.2 安装核心依赖

```bash
# 升级 pip
python -m pip install -U pip

# 安装 PyTorch (CUDA 12.1 示例)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 安装核心依赖
pip install -U \
  fastapi \
  "uvicorn[standard]" \
  pillow \
  requests \
  huggingface_hub \
  accelerate \
  transformers \
  protobuf \
  sentencepiece

# 官方建议: 安装 diffusers 最新版
pip install -U git+https://github.com/huggingface/diffusers
```

#### 5.2.3 验证安装

```bash
python - <<'PY'
from diffusers import QwenImageEditPlusPipeline
print('OK: QwenImageEditPlusPipeline importable')
PY
```

### 5.3 网络适配策略

#### 5.3.1 配置环境变量

```bash
# 中国大陆镜像(推荐)
export HF_ENDPOINT=https://hf-mirror.com

# 固定缓存目录(强烈建议)
export HF_HOME=/home/zzy/weitiao/cache/hf
export HUGGINGFACE_HUB_CACHE=/home/zzy/weitiao/cache/hf/hub
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE"

# 模型目录
export QWEN_EDIT_2511_DIR=/home/zzy/weitiao/models/Qwen-Image-Edit-2511
mkdir -p "$QWEN_EDIT_2511_DIR"

# 离线模式(模型下载完成后)
export HF_HUB_OFFLINE=1
```

---

## 6. 模型资产本地化

### 6.1 模型格式说明

| 格式 | 开发者 | 优势 | 适用场景 | Qwen-Image-Edit 支持 |
|------|--------|------|----------|---------------------|
| **Safetensors** | Hugging Face | ✅ 安全、快速、跨框架 | Diffusers 推理 | ✅ 官方推荐 |
| **GGUF** | llama.cpp | ✅ 量化、单文件 | Ollama/轻量级 | ❌ 需转换 |
| **PyTorch .bin** | PyTorch | ✅ 传统格式 | 旧版系统 | ❌ 不推荐 |
| **ONNX** | Microsoft | ✅ 跨平台 | 部署优化 | ❌ 需转换 |

**关键点**: Qwen-Image-Edit-2511 使用 Safetensors 格式,这是 Hugging Face 生态的标准格式。

### 6.2 模型下载方法

#### 6.2.1 方法一: `huggingface_hub.snapshot_download`(推荐)

```bash
python - <<'PY'
import os
from huggingface_hub import snapshot_download

repo_id = 'Qwen/Qwen-Image-Edit-2511'
local_dir = os.environ.get('QWEN_EDIT_2511_DIR') or './models/Qwen-Image-Edit-2511'

print('Downloading to:', os.path.abspath(local_dir))
snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
)
print('Download done.')
PY
```

#### 6.2.2 方法二: `huggingface-cli download`

```bash
huggingface-cli download \
  Qwen/Qwen-Image-Edit-2511 \
  --local-dir "$QWEN_EDIT_2511_DIR" \
  --local-dir-use-symlinks False \
  --resume-download
```

### 6.3 下载完成验收

```bash
# 检查关键文件
ls -la "$QWEN_EDIT_2511_DIR" | head
test -f "$QWEN_EDIT_2511_DIR/model_index.json" && echo "OK: model_index.json"
```

---

## 7. 核心代码实现

### 7.1 模型加载与优化

```python
import torch
from diffusers import QwenImageEditPlusPipeline

def load_pipeline(model_dir, use_cpu_offload=False):
    # 精度选择
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # 加载 Pipeline
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        variant="bf16" if dtype == torch.bfloat16 else None
    )

    # 显存优化
    if use_cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    # VAE 分块解码
    pipe.enable_vae_tiling()

    return pipe
```

### 7.2 推理执行

```python
from PIL import Image

def run_inference(pipe, image_path, prompt):
    input_image = Image.open(image_path).convert("RGB")
    generator = torch.Generator(device=pipe.device).manual_seed(42)

    output = pipe(
        prompt=prompt,
        image=input_image,
        num_inference_steps=30,
        guidance_scale=1.0,
        true_cfg_scale=4.0,
        generator=generator
    )

    return output.images[0]
```

---

## 8. Web 服务封装

### 8.1 FastAPI 服务

```python
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

app = FastAPI()

class EditRequest(BaseModel):
    prompt: str
    seed: int = 42
    steps: int = 30
    true_cfg_scale: float = 4.0
    guidance_scale: float = 1.0

@app.post("/edit")
async def edit_image(request: EditRequest, file: UploadFile = File(...)):
    # 模型加载与推理逻辑
    pass
```

### 8.2 API 设计

#### 8.2.1 `GET /health`

返回:
- 服务版本、模型名
- worker 列表: `gpu_id`、`ready`、`queue_len`、`last_error`、`uptime_s`

#### 8.2.2 `POST /edit`

请求参数(JSON):
- `prompt`: 编辑指令(必填)
- `images`: 图片输入
  - `image_base64[]`: base64 编码 PNG/JPEG
  - `image_url[]`: URL(用于测试)
- 可选推理参数:
  - `seed`(默认 0 或随机)
  - `num_inference_steps`(默认 40)
  - `true_cfg_scale`(默认 4.0)
  - `guidance_scale`(默认 1.0)

响应:
- `image_base64`(输出 PNG base64)
- 或 `output_path`(保存到本地路径)

---

## 9. 资源调度与优化

### 9.1 显存管理策略

#### 9.1.1 多 GPU 自动分片

```python
# 多 GPU 自动分片
if gpu_count >= 2:
    max_memory = {}
    for i in range(gpu_count):
        total_gib = int(torch.cuda.get_device_properties(i).total_memory / (1024**3))
        max_gib = max(4, total_gib - 6)  # 预留 6GB 显存
        max_memory[i] = f"{max_gib}GiB"
```

#### 9.1.2 灵活指定 GPU 设备

默认情况下,脚本会自动检测所有可见 GPU。如果你希望指定特定显卡:

```bash
# 仅使用前两张卡
export CUDA_VISIBLE_DEVICES=0,1

# 启动服务
python gradio_app.py
```

**注意**:
- 启动前请务必运行 `nvidia-smi` 确认目标显卡显存空闲(建议剩余 > 20GB)
- 如果显卡被其他进程(如 vLLM)占用,强行启动会导致 OOM 或服务崩溃

### 9.2 CPU 降级优化

```python
def _maybe_limit_resources():
    # 限制 CPU 线程数
    torch.set_num_threads(max(1, (os.cpu_count() or 1) // 2))

    # 降低进程优先级
    try:
        os.nice(5)
    except Exception:
        pass
```

### 9.3 GPU 资源检查

#### 9.3.1 检查 GPU 是否可用

**重要**: GPU 是否可用,不是看"有 GPU",而是看"空闲显存"

diffusion 模型推理需要大量显存;如果 GPU 上只剩 1～2GiB,基本不用尝试。

检查命令:

```bash
nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free --format=csv,noheader,nounits
```

#### 9.3.2 常见问题: vLLM 占满 GPU

你可能会看到进程名类似: `VLLM::Worker_TP0/TP1/...`
这通常是后台容器在跑 LLM 服务,它会长时间占用 GPU 显存。

如果你**不允许停止**这些服务,那么就不要再纠结 diffusion 的加载参数:
- GPU 方案不可行
- 只能先走 **CPU 冒烟** 或等待 GPU 空闲

查看占用进程:

```bash
nvidia-smi -i 6,7 --query-compute-apps=pid,process_name,used_memory --format=csv
```

定位源头:

```bash
docker ps --no-trunc | grep -i vllm || true
```

---

## 10. 生产级部署方案

### 10.1 Docker 容器化

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
ENV HF_ENDPOINT=https://hf-mirror.com
ENV QWEN_EDIT_2511_DIR=/app/models
CMD ["python", "gradio_app.py"]
```

### 10.2 Kubernetes 部署

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: qwen-edit
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: qwen-edit
        image: your-registry/qwen-edit:latest
        resources:
          limits:
            nvidia.com/gpu: 1
        env:
        - name: QWEN_EDIT_FORCE_CPU
          value: "0"
```

---

## 11. 故障排查手册

| 错误现象 | 可能原因 | 解决方案 |
|----------|----------|----------|
| **CUDA out of memory** | 显存不足 | 1. 开启 Model Offload<br>2. 降低分辨率<br>3. 增加 Headroom |
| **网络错误** | HF 访问问题 | 1. 检查 HF_ENDPOINT<br>2. 确认模型已下载<br>3. 设置 HF_HUB_OFFLINE=1 |
| **推理卡住** | CPU 负载高 | 1. 限制线程数<br>2. 降低采样步数<br>3. 检查进程优先级 |
| **图片全黑** | VAE 问题 | 1. 开启 VAE Tiling<br>2. 切换 FP32 测试<br>3. 检查输入格式 |
| **ModuleNotFoundError: diffusers** | 环境错误 | 1. 确认在正确的 conda 环境<br>2. 重新安装 diffusers |

### 11.1 排查步骤(按优先级)

#### 11.1.1 先看环境:是不是在正确的环境?

```bash
which python
python -c "from diffusers import QwenImageEditPlusPipeline"
```

#### 11.1.2 再看模型目录:是否完整落盘?

```bash
test -f "$QWEN_EDIT_2511_DIR/model_index.json"
```

#### 11.1.3 再看 GPU:是否真空闲?

```bash
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits
```

若 `free` 很小,GPU 方案直接放弃,先 CPU 冒烟。

#### 11.1.4 CPU 冒烟太慢?

- `Steps` 降到 10～15
- `max_side` 降到 512 或更小

---

## 12. 快速开始指南

### 12.1 前置检查(一次性)

在开始前先确认环境:

```bash
# 1. 进入工作目录
cd /home/zzy/weitiao

# 2. 激活环境
conda activate videofen
python -V

# 3. 确认 GPU 可见
nvidia-smi

# 4. 验证 PyTorch CUDA
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('cuda version:', torch.version.cuda)
print('gpu count:', torch.cuda.device_count())
PY

# 5. 预留磁盘空间(至少数十 GB)
df -h /home/zzy/weitiao
```

### 12.2 配置环境变量

```bash
# 中国大陆镜像
export HF_ENDPOINT=https://hf-mirror.com

# 固定缓存目录
export HF_HOME=/home/zzy/weitiao/cache/hf
export HUGGINGFACE_HUB_CACHE=/home/zzy/weitiao/cache/hf/hub
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE"

# 模型目录
export QWEN_EDIT_2511_DIR=/home/zzy/weitiao/models/Qwen-Image-Edit-2511
mkdir -p "$QWEN_EDIT_2511_DIR"
```

### 12.3 下载模型

参考第 6 节"模型资产本地化"的下载方法。

下载完成后,建议开启离线模式:

```bash
export HF_HUB_OFFLINE=1
```

### 12.4 单卡冒烟测试(GPU)

目标:不做服务化,先用最小脚本把"加载 + 编辑 + 保存图片"跑通。

```bash
export CUDA_VISIBLE_DEVICES=6

python - <<'PY'
import os
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

model_dir = os.environ['QWEN_EDIT_2511_DIR']
input_path = '/path/to/input.png'
output_path = '/path/to/output.png'
prompt = '把图片里的主体颜色改成紫色,并保持整体风格一致。'

os.makedirs(os.path.dirname(output_path), exist_ok=True)

pipe = QwenImageEditPlusPipeline.from_pretrained(model_dir, torch_dtype=torch.bfloat16)
pipe.to('cuda')
pipe.set_progress_bar_config(disable=None)

image = Image.open(input_path).convert('RGB')
inputs = {
    'image': [image],
    'prompt': prompt,
    'generator': torch.manual_seed(0),
    'true_cfg_scale': 4.0,
    'negative_prompt': ' ',
    'num_inference_steps': 40,
    'guidance_scale': 1.0,
    'num_images_per_prompt': 1,
}

with torch.inference_mode():
    out = pipe(**inputs)
    out.images[0].save(output_path)

print('Saved:', output_path)
PY
```

验收:
- 输出图片生成且可打开
- 过程中无 CUDA error / OOM

如果 OOM:
- 优先减小输入分辨率
- 或减少 `num_inference_steps`

### 12.5 Gradio WebUI 启动

#### 12.5.1 GPU 模式

```bash
export CUDA_VISIBLE_DEVICES=6
export QWEN_EDIT_2511_DIR=/home/zzy/weitiao/models/Qwen-Image-Edit-2511

python modules/02_qwen_image_edit_service/gradio_app.py
```

访问:
- 本机: `http://127.0.0.1:7860`
- 远程: `http://<服务器IP>:7860`

#### 12.5.2 CPU 冒烟模式

当 GPU 被占满时,可以先用 CPU 跑通端到端链路:

```bash
cd /home/zzy/weitiao/modules/02_qwen_image_edit_service
conda activate videofen

export QWEN_EDIT_2511_DIR=/home/zzy/weitiao/models/Qwen-Image-Edit-2511
export QWEN_EDIT_FORCE_CPU=1

export QWEN_EDIT_MAX_CPU_THREADS=32
export QWEN_EDIT_NICE=10
export QWEN_EDIT_MAX_CONCURRENCY=1

python gradio_app.py
```

CPU 冒烟调参建议:
- `Steps`: 10～15(先别用 40)
- `最大边长 max_side`: 512(或更小)

### 12.6 参数说明

在本仓库 UI 里,你重点理解:

| 参数 | 作用 | 建议值 |
|------|------|--------|
| `prompt` | 决定改什么 | 具体描述编辑需求 |
| `max_side` | 决定像素规模 | CPU: 512, GPU: 768～1024 |
| `Steps` | 采样步数 | CPU: 10～15, GPU: 30～40 |
| `seed` | 复现控制 | 0 固定, -1 随机 |
| `true_cfg_scale` | 编辑强度 | 3～5 起步 |
| `guidance_scale` | 可能被模型忽略 | 1.0(默认) |

### 12.7 为什么会"卡在 0%"

技术点: 进度条 `0/40` 代表还在跑第 0→1 步,CPU 上第一步可能很久。

判断方式: 开另一个终端看 `top`,只要 `python gradio_app.py` 还在持续吃 CPU,通常就是在计算,不是死锁。

---

## 13. 附录完整代码

### 13.1 环境配置脚本

```bash
#!/bin/bash
# setup_env.sh

# 创建 Conda 环境
conda create -n qwen_edit python=3.10 -y
conda activate qwen_edit

# 安装依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/huggingface/diffusers
pip install accelerate transformers protobuf sentencepiece fastapi uvicorn pillow

# 配置环境变量
echo "export HF_ENDPOINT=https://hf-mirror.com" >> ~/.bashrc
echo "export HF_HOME=/home/user/cache" >> ~/.bashrc
echo "export QWEN_EDIT_2511_DIR=/home/user/models/Qwen-Image-Edit-2511" >> ~/.bashrc
echo "export HF_HUB_OFFLINE=1" >> ~/.bashrc

source ~/.bashrc
```

### 13.2 模型下载脚本

```python
# download_model.py
from huggingface_hub import snapshot_download
import os

def download_qwen_edit():
    model_dir = os.environ.get("QWEN_EDIT_2511_DIR")
    if not model_dir:
        raise RuntimeError("QWEN_EDIT_2511_DIR not set")

    snapshot_download(
        repo_id="Qwen/Qwen-Image-Edit-2511",
        local_dir=model_dir,
        resume_download=True,
        local_dir_use_symlinks=False
    )

if __name__ == "__main__":
    download_qwen_edit()
```

### 13.3 完整 Gradio 应用

```python
import os
from typing import Optional
import gradio as gr
import torch
from PIL import Image

def _get_model_dir() -> str:
    model_dir = os.environ.get("QWEN_EDIT_2511_DIR")
    if not model_dir:
        raise RuntimeError(
            "Missing env var QWEN_EDIT_2511_DIR. "
            "Set it to your local model directory."
        )
    return model_dir

def _force_cpu() -> bool:
    return os.environ.get("QWEN_EDIT_FORCE_CPU", "0") == "1"

def _maybe_limit_resources() -> None:
    """Best-effort resource limits to avoid overloading the host."""
    default_threads = max(1, (os.cpu_count() or 1) // 2)
    max_threads = int(os.environ.get("QWEN_EDIT_MAX_CPU_THREADS", str(default_threads)))
    max_threads = max(1, max_threads)

    try:
        nice_delta = int(os.environ.get("QWEN_EDIT_NICE", "5"))
        if nice_delta != 0:
            os.nice(nice_delta)
    except Exception:
        pass

    try:
        torch.set_num_threads(max_threads)
        torch.set_num_interop_threads(min(4, max_threads))
    except Exception:
        pass

_PIPE = None

def _format_gib(num_bytes: int) -> str:
    return f"{num_bytes / (1024 ** 3):.2f} GiB"

def _assert_vram_headroom() -> None:
    """Fail fast if visible GPUs are already heavily occupied."""
    if not torch.cuda.is_available():
        return

    min_free_gib = float(os.environ.get("QWEN_EDIT_MIN_FREE_GIB", "6"))
    min_free_bytes = int(min_free_gib * (1024 ** 3))

    bad = []
    for i in range(torch.cuda.device_count()):
        free_b, total_b = torch.cuda.mem_get_info(i)
        if free_b < min_free_bytes:
            bad.append((i, free_b, total_b))

    if bad:
        details = ", ".join(
            [f"cuda:{i} free={_format_gib(free_b)}/{_format_gib(total_b)}" for i, free_b, total_b in bad]
        )
        raise RuntimeError(
            f"Not enough free VRAM. Need >= {min_free_gib:.0f} GiB free per GPU, got: {details}."
        )

def _get_pipe():
    global _PIPE
    if _PIPE is not None:
        return _PIPE

    from diffusers import QwenImageEditPlusPipeline

    model_dir = _get_model_dir()

    if _force_cpu():
        pipe = QwenImageEditPlusPipeline.from_pretrained(model_dir, torch_dtype=torch.float32)
        pipe.to("cpu")
        pipe.set_progress_bar_config(disable=None)
        for method_name, args in (
            ("enable_attention_slicing", ("max",)),
            ("enable_vae_slicing", ()),
            ("enable_vae_tiling", ()),
        ):
            fn = getattr(pipe, method_name, None)
            if callable(fn):
                try:
                    fn(*args)
                except Exception:
                    pass
        _PIPE = pipe
        return _PIPE

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available (set QWEN_EDIT_FORCE_CPU=1 to run on CPU)")

    _assert_vram_headroom()

    gpu_count = torch.cuda.device_count()
    if gpu_count >= 2:
        headroom_gib = int(os.environ.get("QWEN_EDIT_GPU_HEADROOM_GIB", "6"))
        gpu0_extra_headroom_gib = int(os.environ.get("QWEN_EDIT_GPU0_EXTRA_HEADROOM_GIB", "4"))
        max_memory = {}
        for i in range(gpu_count):
            total_gib = int(torch.cuda.get_device_properties(i).total_memory / (1024**3))
            effective_headroom = headroom_gib + (gpu0_extra_headroom_gib if i == 0 else 0)
            max_gib = max(4, total_gib - effective_headroom)
            max_memory[i] = f"{max_gib}GiB"
        max_memory["cpu"] = os.environ.get("QWEN_EDIT_CPU_MAX_MEMORY", "120GiB")

        offload_folder = os.environ.get("QWEN_EDIT_OFFLOAD_FOLDER", "/tmp/offload")
        os.makedirs(offload_folder, exist_ok=True)

        torch_dtype = torch.bfloat16 if os.environ.get("QWEN_EDIT_DTYPE", "bf16") == "bf16" else torch.float16

        pipe = QwenImageEditPlusPipeline.from_pretrained(
            model_dir,
            torch_dtype=torch_dtype,
            device_map="balanced",
            max_memory=max_memory,
            low_cpu_mem_usage=True,
            offload_state_dict=True,
            offload_folder=offload_folder,
        )
    else:
        torch_dtype = torch.bfloat16 if os.environ.get("QWEN_EDIT_DTYPE", "bf16") == "bf16" else torch.float16
        pipe = QwenImageEditPlusPipeline.from_pretrained(model_dir, torch_dtype=torch_dtype)
        pipe.to("cuda")
    pipe.set_progress_bar_config(disable=None)

    for method_name, args in (
        ("enable_attention_slicing", ("max",)),
        ("enable_vae_slicing", ()),
        ("enable_vae_tiling", ()),
    ):
        fn = getattr(pipe, method_name, None)
        if callable(fn):
            try:
                fn(*args)
            except Exception:
                pass

    _PIPE = pipe
    return _PIPE

def _maybe_resize(image: Image.Image, max_side: int) -> Image.Image:
    if max_side <= 0:
        return image

    w, h = image.size
    if max(w, h) <= max_side:
        return image

    scale = max_side / float(max(w, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return image.resize((new_w, new_h), Image.LANCZOS)

@torch.inference_mode()
def edit_image(
    image: Optional[Image.Image],
    prompt: str,
    seed: int,
    num_inference_steps: int,
    true_cfg_scale: float,
    guidance_scale: float,
    max_side: int,
):
    if image is None:
        raise gr.Error("请先上传一张图片")
    if not prompt or not prompt.strip():
        raise gr.Error("请先输入编辑需求(prompt)")

    if (not _force_cpu()) and (not torch.cuda.is_available()):
        raise gr.Error("CUDA 不可用")

    pipe = _get_pipe()

    image = image.convert("RGB")
    image = _maybe_resize(image, max_side=max_side)

    generator_device = "cpu" if _force_cpu() else "cuda:0"
    if seed < 0:
        generator = torch.Generator(device=generator_device).seed()
    else:
        generator = torch.Generator(device=generator_device).manual_seed(int(seed))

    inputs = {
        "image": [image],
        "prompt": prompt.strip(),
        "generator": generator,
        "true_cfg_scale": float(true_cfg_scale),
        "negative_prompt": " ",
        "num_inference_steps": int(num_inference_steps),
        "guidance_scale": float(guidance_scale),
        "num_images_per_prompt": 1,
    }

    out = pipe(**inputs)
    return out.images[0]

def main():
    title = "Qwen-Image-Edit-2511 本地交互式 Demo"

    _maybe_limit_resources()

    if os.environ.get("QWEN_EDIT_EAGER_LOAD", "1") == "1":
        _get_pipe()

    with gr.Blocks(title=title) as demo:
        gr.Markdown("# Qwen-Image-Edit-2511(本地交互式)")

        with gr.Row():
            with gr.Column(scale=1):
                image_in = gr.Image(type="pil", label="输入图片")
                prompt = gr.Textbox(lines=3, label="编辑需求(Prompt)")

                with gr.Row():
                    seed = gr.Number(value=0, precision=0, label="Seed(-1 随机)")
                    steps = gr.Slider(minimum=10, maximum=80, step=1, value=40, label="Steps")

                with gr.Row():
                    true_cfg = gr.Slider(minimum=1.0, maximum=8.0, step=0.1, value=4.0, label="true_cfg_scale")
                    guidance = gr.Slider(minimum=0.5, maximum=3.0, step=0.1, value=1.0, label="guidance_scale")

                max_side = gr.Slider(
                    minimum=0,
                    maximum=2048,
                    step=64,
                    value=768,
                    label="最大边长(>0 时自动缩放,避免 OOM)",
                )

                run = gr.Button("生成", variant="primary")

            with gr.Column(scale=1):
                image_out = gr.Image(type="pil", label="输出结果")

        run.click(
            fn=edit_image,
            inputs=[image_in, prompt, seed, steps, true_cfg, guidance, max_side],
            outputs=[image_out],
        )

    demo.queue(max_size=20, default_concurrency_limit=int(os.environ.get("QWEN_EDIT_MAX_CONCURRENCY", "1")))
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))

if __name__ == "__main__":
    main()
```

---

## 文档使用说明

1. **环境准备**: 运行 `setup_env.sh` 脚本
2. **模型下载**: 运行 `download_model.py`
3. **启动服务**: 运行 `gradio_app.py`
4. **访问界面**: 浏览器访问 `http://your-server:7860`

**配置调整**:
- 修改环境变量调整资源限制
- 调整 `max_side` 控制图像分辨率
- 调整 `true_cfg_scale` 控制编辑强度

---

## 进阶主题

### 第二阶段: 可选加速路线

当 diffusers 基线稳定后,可评估:
- **vLLM-Omni**: 面向 Qwen-Image 系列的高性能推理路径(需要额外工程验证)
- **SGLang-Diffusion**: diffusion 推理的另一套执行与并发
- **LightX2V / 蒸馏(Lightning)**: 降低 NFEs 提升速度(会改变质量/行为,需验收)

建议策略: 先对齐输出质量与一致性,再做加速替换。

---

**祝部署顺利!** 如有问题,请参考第 11 节故障排查手册。
