# Qwen-Image-Edit-2511 部署与封装调用方案（可照抄执行 Runbook）

日期：2025-12-26

> 说明：本文既是 Runbook（可照抄执行），也按“博客教程”写法补齐了关键概念解释。

## 0. 快速开始（博客版：从下载到冒烟出图）

如果你只想最快跑通“能出第一张编辑结果图”，按下面顺序做即可。

### 0.1 进入环境（最常见坑：跑在 base 导致缺 diffusers）

```bash
cd /home/zzy/weitiao

# 一定要进入 videofen（不要在 base 里跑）
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate videofen
﻿# Qwen-Image-Edit-2511：技术原理 + 部署实践（可发博客版）

日期：2025-12-26

本文目标是“技术分享写法”：
- 先把**关键技术与概念**讲清楚（新手能听懂）。
- 再按“下载→部署→运行→冒烟→排障”的顺序实操，并在每一步同步解释背后的技术点。

---

## 1. 这项技术解决什么问题（先讲清楚模型的作用）

`Qwen/Qwen-Image-Edit-2511` 是一个**图像编辑（Image Edit）**模型：
- 给它一张输入图 + 一句编辑指令（prompt），它输出一张修改后的新图。
- 重点是“在原图基础上按指令改”，而不是从零生成。

你可以把它理解为：**“可控的图像改写器”**。

---

## 2. 核心技术一：Diffusion（为什么会有 steps、为什么会慢）

### 2.1 Diffusion 一句话理解

Diffusion（扩散模型）在推理时通常不是“一步出图”，而是**迭代式采样**：
- 从噪声/中间状态开始
- 循环 N 次逐步变清晰
- 最后得到结果图

这个 N 就是你在 UI 里看到的 `Steps / num_inference_steps`。

### 2.2 Steps 是什么、该怎么选

`Steps`=采样迭代次数，影响两件事：
- 质量与稳定性（一般 steps 多更稳，但收益递减）
- 耗时（近似随 steps 线性增长）

在 CPU 冒烟场景：
- 建议 steps 先用 10～15（先确保“能出图”）

---

## 3. 核心技术二：diffusers Pipeline（它把哪些组件串起来）

### 3.1 为什么用 diffusers

diffusers 是 Hugging Face 的 diffusion 工程框架。
它的价值不是“提出新算法”，而是把工程复杂度封装成可复用 pipeline：
- 下载/加载模型组件
- 管理 dtype/device（CPU/GPU、bf16/fp16 等）
- 管理采样循环与 scheduler
- 最终输出图像

本仓库使用官方路线：`diffusers.QwenImageEditPlusPipeline`。

### 3.2 pipeline 里你需要认识的组件（只记职责）

从工程角度，管线通常包含：
- 文本编码器：把 prompt 变成向量（让模型“理解你要改什么”）
- 图像编码/预处理：把输入图变成内部表示
- 去噪主干网络：每个 step 都要跑它（最耗时）
- VAE 解码：把潜变量还原成像素图
- scheduler：控制每一步“怎么走”

你看到的：
- `Loading pipeline components...`
- `Loading checkpoint shards...`
就是在加载这些组件（权重可能被分成多个 shard）。

---

## 4. 核心技术三：Gradio（为什么适合做“最小交互前端”）

Gradio 的定位：**最快把一个 Python 推理函数变成网页交互**。

对新手最友好的一点是：
- 你不需要写前端、也不用先做 FastAPI
- 直接把 `edit_image(image, prompt, ...) -> output_image` 绑到按钮即可

并发相关概念：
- `queue`：把请求排队，避免同时把机器打爆
- `default_concurrency_limit`：限制同一时间处理多少个请求（本仓库默认 1，保护服务器）

---

## 5. 核心技术四：资源与显存（为什么“GPU 被占用”会导致看似玄学的 OOM）

### 5.1 GPU 是否可用，不是看“有 GPU”，而是看“空闲显存”

diffusion 模型推理需要大量显存；如果 GPU 上只剩 1～2GiB，基本不用尝试。

检查命令：

```bash
nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free --format=csv,noheader,nounits
```

### 5.2 常见真实根因：后台 vLLM（LLM 服务）占满 GPU

你可能会看到进程名类似：`VLLM::Worker_TP0/TP1/...`。
这通常是后台容器在跑 LLM 服务（例如 `python -m vllm.entrypoints.openai.api_server ...`），它会长时间占用 GPU 显存。

如果你**不允许停止**这些服务，那么就不要再纠结 diffusion 的加载参数：
- GPU 方案不可行
- 只能先走 **CPU 冒烟** 或等待 GPU 空闲

---

## 6. 部署实践（按步骤做，并同步解释每一步的技术点）

本节按“从零到能出图”的顺序写。

### 6.1 进入正确的 Python 环境（最常见坑：跑在 base）

技术点：同一台机器可能有多个 conda 环境；你在 base 里跑会缺 `diffusers`，表现为 `ModuleNotFoundError: diffusers`。

```bash
cd /home/zzy/weitiao
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate videofen

which python
python -c "import sys; print(sys.prefix)"
```

验收：`sys.prefix` 应为 `/opt/anaconda3/envs/videofen`（或同类路径）。

### 6.2 大陆下载的技术点：HF 镜像 + 固定缓存

技术点：
- `HF_ENDPOINT` 控制 huggingface_hub 下载走镜像站
- 固定 `HF_HOME/HUGGINGFACE_HUB_CACHE` 便于离线运行与复现

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/home/zzy/weitiao/cache/hf
export HUGGINGFACE_HUB_CACHE=/home/zzy/weitiao/cache/hf/hub
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE"

export QWEN_EDIT_2511_DIR=/home/zzy/weitiao/models/Qwen-Image-Edit-2511
mkdir -p "$QWEN_EDIT_2511_DIR"
```

### 6.3 模型下载（把权重完整落盘）

技术点：`snapshot_download` 会把模型仓库快照完整下载到指定目录，后续可以 `from_pretrained(local_dir)` 离线加载。

```bash
python - <<'PY'
import os
from huggingface_hub import snapshot_download

repo_id = 'Qwen/Qwen-Image-Edit-2511'
local_dir = os.environ['QWEN_EDIT_2511_DIR']

print('Downloading to:', local_dir)
snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
)
print('Download done.')
PY
```

验收：

```bash
test -f "$QWEN_EDIT_2511_DIR/model_index.json" && echo "OK: model_index.json"
```

### 6.4 依赖验收（先验证 pipeline 能 import）

技术点：对新手来说，“import 能否成功”是第一道硬门槛。

```bash
python -c "import diffusers; print('diffusers', diffusers.__version__); from diffusers import QwenImageEditPlusPipeline; print('pipeline ok')"
```

### 6.5 先判断走 GPU 还是 CPU（关键分岔点）

技术点：diffusion 推理强依赖显存；GPU 被占满会导致加载阶段 OOM。

```bash
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits
```

结论规则（简单但实用）：
- 如果你准备使用的 GPU 上 `free` 只有几百 MiB～2GiB：不要尝试 GPU 推理。

### 6.6 CPU 冒烟：Gradio 交互跑通闭环（保护服务器不被拖慢）

技术点：
- CPU 模式非常慢，但能验证“下载/加载/推理/UI/参数”全链路
- 通过限制 PyTorch 线程 + 降低 nice + 限制 Gradio 并发，避免把整机打满

```bash
cd /home/zzy/weitiao/modules/02_qwen_image_edit_service

export QWEN_EDIT_2511_DIR=/home/zzy/weitiao/models/Qwen-Image-Edit-2511
export QWEN_EDIT_FORCE_CPU=1

export QWEN_EDIT_MAX_CPU_THREADS=32
export QWEN_EDIT_NICE=10
export QWEN_EDIT_MAX_CONCURRENCY=1

python gradio_app.py
```

访问：
- 本机：`http://127.0.0.1:7860`
- 远程：`http://<服务器IP>:7860`

CPU 冒烟参数建议（先快后好）：
- `Steps`：10～15
- `max_side`：512（或更小）

### 6.7 为什么会“卡在 0%”：diffusion 等待现象的正确解释

技术点：进度条 `0/40` 代表还在跑第 0→1 步，CPU 上第一步可能很久。
判断方式：开另一个终端看 `top`，只要 `python gradio_app.py` 还在持续吃 CPU，通常就是在计算，不是死锁。

---

## 7. 参数与效果（新手最常问的 6 个旋钮）

在本仓库 UI 里，你重点理解：
- `prompt`：决定改什么
- `max_side`：决定像素规模（直接影响速度/资源）
- `Steps`：采样步数（质量↑ vs 时间↑）
- `seed`：复现控制
- `true_cfg_scale`：编辑强度/听 prompt 程度（建议 3～5 起步）
- `guidance_scale`：可能被模型忽略（日志会提示 ignored），不是主旋钮

---

## 8. 排障（按优先级，不走弯路）

1) 先看环境：是不是在 `videofen`？
- `which python`
- `python -c "from diffusers import QwenImageEditPlusPipeline"`

2) 再看模型目录：是否完整落盘？
- `test -f "$QWEN_EDIT_2511_DIR/model_index.json"`

3) 再看 GPU：是否真空闲？
- `nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits`
- 若 `free` 很小，GPU 方案直接放弃，先 CPU 冒烟

4) CPU 冒烟太慢：
- `Steps` 降到 10～15
- `max_side` 降到 512 或更小

---

## 9. 进阶：服务化与并发（写博客可作为“下一篇”）

当你能稳定出图后，下一步才建议做服务化封装：
- FastAPI + worker（每 GPU 一个进程）
- `/health` + 队列与限流

第一阶段不依赖 vLLM（vLLM 是 LLM 推理优化主线，diffusion 先以 diffusers 稳定为主）。
- 本机：`http://127.0.0.1:7860`
- 远程：`http://<服务器IP>:7860`

CPU 冒烟建议：先把 `Steps` 设为 10～15、最大边长 `max_side` 设为 512（更快出结果）。

## 1. 目标与边界

**目标**
- 在本机部署最新版图像编辑模型 `Qwen/Qwen-Image-Edit-2511`。
- 将“图像编辑能力”封装为可调用服务，支持批量/并发请求。
- 优先追求：可复现、稳定、可观测（health/日志）、易排障。

**硬性资源约束（本仓库环境约定）**
- 仅使用后四张卡：`CUDA_VISIBLE_DEVICES=4,5,6,7`
- CPU 使用约一半核心，避免占满整机
- 先冒烟再放大：先单卡跑通，再扩到 4 卡并发

**不在本方案第一阶段范围**
- 训练/微调（LoRA/全参）
- 复杂的多租户鉴权、计费、灰度等平台化能力

## 2. 技术选型

### 2.1 推理栈（推荐）
- **Diffusers**：使用官方示例同款管线 `diffusers.QwenImageEditPlusPipeline`
- **Torch**：本机已有 PyTorch + CUDA
- **PIL**：图片读写与编码

选择理由：官方 Hugging Face 模型卡与 `QwenLM/Qwen-Image` 仓库示例均以 diffusers 为主线，最稳、最少兼容性风险。

### 2.2 是否使用 vLLM

**结论：第一阶段不依赖 vLLM。**
- vLLM 的主战场是 LLM（KV cache/连续批处理等）。
- `Qwen-Image-Edit-2511` 属于 diffusion 管线，最直接的落地方式是 diffusers。
- 官方 repo 提到的 **vLLM-Omni / SGLang-Diffusion / LightX2V** 属于“可选加速路线”，建议在 diffusers 方案跑通并形成稳定基线后，再评估替换后端。

## 3. 服务形态与并发策略

### 3.1 对外接口形态

两种封装形态可选：
1) **HTTP 服务（推荐）**：FastAPI + Uvicorn，适合跨语言/跨机器调用
2) **本地 Python SDK**：直接 import 调用，适合单机脚本/批处理

本方案第一阶段以 **HTTP 服务** 为主（最通用），并提供最小 Python client 示例。

### 3.2 多 GPU 并发策略（推荐：一进程一张卡）

- 启动 N 个 worker 进程（N=4），每个 worker 绑定一张 GPU（逻辑 0..3 映射物理 4..7）。
- 每个 worker 进程内：常驻加载一次 pipeline，循环处理请求。
- 外层 API 网关：
  - 维护任务队列
  - 以轮询/最短队列策略将任务分发给空闲 worker
  - 提供 `/health` 查看 worker 存活、GPU、队列长度、最近错误

**优点**
- 稳定：避免单进程跨卡切分带来的不确定性
- 可观测：每张卡的异常隔离，单 worker 崩溃可自动拉起
- 扩展简单：增加/减少 worker 数量即可

## 4. API 设计（最小可用）

### 4.1 `GET /health`
返回：
- 服务版本、模型名
- worker 列表：`gpu_id`、`ready`、`queue_len`、`last_error`、`uptime_s`

### 4.2 `POST /edit`
请求参数（建议 JSON）：
- `prompt`：编辑指令（必填）
- `images`：图片输入（建议支持以下两种之一或同时支持）
  - `image_base64[]`：base64 编码 PNG/JPEG
  - `image_url[]`：URL（用于测试；生产建议走对象存储或 base64）
- 可选推理参数（先做少量）：
  - `seed`（默认 0 或随机）
  - `num_inference_steps`（默认 40，按官方示例）
  - `true_cfg_scale`（默认 4.0，按官方示例）
  - `guidance_scale`（默认 1.0，按官方示例）

响应：
- `image_base64`（输出 PNG base64）
- 或 `output_path`（保存到本地路径，适合内网批处理）

## 5. 部署与运行步骤（先冒烟再放大）

本节目标：你在**中国大陆网络**环境下，也能从“空目录”一步步做到：
1) 把模型权重完整下载到本地缓存/目录
2) 单卡（GPU4）冒烟跑通一次图像编辑
3) 扩展为四卡并发（GPU4/5/6/7）服务形态

### 5.0 前置检查（一次性）

在开始前先确认环境：

1) 进入本仓库工作目录：

```bash
cd /home/zzy/weitiao
```

2) 激活你现有环境（按本仓库约定是 `videofen`）：

```bash
conda activate videofen
python -V
```

3) 确认 GPU 可见与驱动正常：

```bash
nvidia-smi
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('cuda version:', torch.version.cuda)
print('gpu count:', torch.cuda.device_count())
PY
```

4) 预留磁盘空间（模型与缓存会很大；建议至少预留数十 GB）：

```bash
df -h /home/zzy/weitiao
```

### 5.1 统一缓存目录（强烈建议）

把 Hugging Face 缓存固定到仓库下，方便离线/迁移/清理：

```bash
export HF_HOME=/home/zzy/weitiao/cache/hf
export HUGGINGFACE_HUB_CACHE=/home/zzy/weitiao/cache/hf/hub
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE"
```

如果你希望“下载到一个确定目录（不依赖 cache 命中）”，也可以额外准备一个模型目录：

```bash
export QWEN_EDIT_2511_DIR=/home/zzy/weitiao/models/Qwen-Image-Edit-2511
mkdir -p "$QWEN_EDIT_2511_DIR"
```

### 5.2 中国大陆下载（镜像/离线）

#### 5.2.1 方案 A：使用 Hugging Face 镜像 Endpoint（推荐优先尝试）

很多情况下只需要指定镜像域名即可：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

说明：
- 这个变量会让 `huggingface_hub` / diffusers 下载时改用镜像站点。
- 如果你有企业网络代理，也可以再配合 `HTTPS_PROXY`/`HTTP_PROXY`（如有需要再加）。

#### 5.2.2 方案 B：先把权重离线下载到本地，再离线运行

当网络不稳定/无法直连时，推荐把模型完整落盘，然后开启离线模式运行。

### 5.3 安装/升级依赖（与官方示例对齐）

`Qwen-Image-Edit-2511` 官方模型卡建议安装最新版 diffusers（git 方式）。在 `videofen` 环境中执行：

```bash
python -m pip install -U pip

# 核心依赖
python -m pip install -U \
  fastapi \
  "uvicorn[standard]" \
  pillow \
  requests \
  huggingface_hub

# 官方建议：安装 diffusers 最新版
python -m pip install -U git+https://github.com/huggingface/diffusers
```

验收：

```bash
python - <<'PY'
from diffusers import QwenImageEditPlusPipeline
print('OK: QwenImageEditPlusPipeline importable')
PY
```

### 5.4 模型下载（把 2511 权重完整下载到本地）

#### 5.4.1 下载方式 1：`huggingface_hub.snapshot_download`（可控、可复用）

优点：能指定 `local_dir`，下载后目录结构清晰，后续可直接 `from_pretrained(local_dir)`。

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

下载完成后，做一个最小验收（确认关键文件存在）：

```bash
ls -la "$QWEN_EDIT_2511_DIR" | head
test -f "$QWEN_EDIT_2511_DIR/model_index.json" && echo "OK: model_index.json"
```

#### 5.4.2 下载方式 2：`huggingface-cli download`（命令行）

如果你更习惯 CLI：

```bash
huggingface-cli download \
  Qwen/Qwen-Image-Edit-2511 \
  --local-dir "$QWEN_EDIT_2511_DIR" \
  --local-dir-use-symlinks False \
  --resume-download
```

### 5.5 单卡冒烟（GPU4）

目标：不做服务化，先用最小脚本把“加载 + 编辑 + 保存图片”跑通。

1) 准备一张输入图片（放到任意路径，例如：`/home/zzy/weitiao/assets/input.png`）。

2) 执行冒烟脚本（注意：只用 GPU4）：

```bash
export CUDA_VISIBLE_DEVICES=4

python - <<'PY'
import os
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

model_dir = os.environ['QWEN_EDIT_2511_DIR']
input_path = os.environ.get('QWEN_EDIT_INPUT', '/home/zzy/weitiao/modules/01_single_gpu_smoke/image.png')
output_path = os.environ.get('QWEN_EDIT_OUTPUT', '/home/zzy/weitiao/logs/qwen_image_edit_2511_smoke.png')

prompt = os.environ.get('QWEN_EDIT_PROMPT', '把图片里的主体颜色改成紫色，并保持整体风格一致。')

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

验收：
- `logs/qwen_image_edit_2511_smoke.png` 生成且可打开
- 过程中无 CUDA error / OOM

如果 OOM：
- 优先减小输入分辨率（把输入图片缩小）
- 或减少 `num_inference_steps`

### 5.6 切换离线运行（可选，但大陆环境建议配套）

当你确认模型已完整下载到本地后，可以开启离线模式，避免运行时再发起联网请求：

```bash
export HF_HUB_OFFLINE=1
```

说明：
- 开启离线后，如果本地缺文件会直接报错，更容易定位“到底缺了什么”。

### 5.7 四卡服务化（GPU4/5/6/7）

本方案推荐的并发形态是“一进程一张卡”。实现落地后，启动方式会是：

```bash
export CUDA_VISIBLE_DEVICES=4,5,6,7
export HF_HOME=/home/zzy/weitiao/cache/hf
export HUGGINGFACE_HUB_CACHE=/home/zzy/weitiao/cache/hf/hub
export QWEN_EDIT_2511_DIR=/home/zzy/weitiao/models/Qwen-Image-Edit-2511
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

# 预期：启动 1 个 master + 4 个 worker（每个 worker 绑定 cuda:0..3）
```

验收标准（服务化阶段）：
- `/health` 显示 4 个 worker ready
- 连续并发请求（例如 10 个）不死锁、不崩溃
- 单 worker 异常不会拖垮整个服务（可重启恢复）

> 注：上面 5.5/5.7 已覆盖“单卡冒烟 / 四卡服务化”两步，这里不再重复。

## 6. 依赖与版本策略

- diffusers：优先使用官方建议的最新版（git 安装）以匹配 Qwen-Image 系列最新 pipeline。
- torch：使用环境内已安装版本。

如果遇到“pipeline 类不存在/参数不匹配”，优先升级 diffusers 版本。

## 7. 风险点与排障预案

- **diffusers 版本不匹配**：表现为找不到 `QwenImageEditPlusPipeline` 或参数缺失。
  - 处理：安装/升级到最新 diffusers（git）。
- **多进程 CUDA 初始化问题**：表现为 worker 启动即报 CUDA error。
  - 处理：采用 `spawn` 启动方式；确保每个 worker 进程只在自己进程内首次触发 CUDA。
- **显存不足（OOM）**：
  - 处理：降低分辨率、降低 batch（本方案默认 batch=1）、减少 steps。
- **输入图片格式问题**：
  - 处理：统一转 `RGB` 或 `RGBA`，强制尺寸上限，限制单次图片数量。

## 8. 第二阶段（可选加速路线）

当 diffusers 基线稳定后，可评估：
- vLLM-Omni：面向 Qwen-Image 系列的高性能推理路径（需要额外工程验证）
- SGLang-Diffusion：diffusion 推理的另一套执行与并发
- LightX2V / 蒸馏（Lightning）：降低 NFEs 提升速度（会改变质量/行为，需验收）

建议策略：先对齐输出质量与一致性，再做加速替换。

## 9. 最简单交互式“小前端”（Gradio）

如果你暂时只想要“一个小前端，交互式输入需求并返回结果”，推荐直接用 Gradio 本地起 WebUI（无需先做 FastAPI/多 worker）。

### 9.1 依赖安装（中国大陆建议先设镜像）

```bash
cd /home/zzy/weitiao
conda activate videofen

# 大陆镜像（如可用）
export HF_ENDPOINT=https://hf-mirror.com

# 固定缓存目录（可选但建议）
export HF_HOME=/home/zzy/weitiao/cache/hf
export HUGGINGFACE_HUB_CACHE=/home/zzy/weitiao/cache/hf/hub
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE"

# 安装 gradio 相关依赖
python -m pip install -U -r modules/02_qwen_image_edit_service/requirements_gradio.txt

# 安装 diffusers 最新版（官方建议）
python -m pip install -U git+https://github.com/huggingface/diffusers
```

### 9.2 模型下载（一次即可）

沿用本文件第 5.4 节，把 `Qwen/Qwen-Image-Edit-2511` 下载到本地目录，并设置：

```bash
export QWEN_EDIT_2511_DIR=/home/zzy/weitiao/models/Qwen-Image-Edit-2511
```

下载完成后（建议）开启离线运行：

```bash
export HF_HUB_OFFLINE=1
```

### 9.3 启动 WebUI（单卡 GPU4）

```bash
export CUDA_VISIBLE_DEVICES=4
export QWEN_EDIT_2511_DIR=/home/zzy/weitiao/models/Qwen-Image-Edit-2511

python modules/02_qwen_image_edit_service/gradio_app.py
```

打开浏览器访问：
- `http://<你的机器IP>:7860`

验收：
- 页面可打开
- 上传图片 + 输入 prompt 点击“生成”能返回结果图

### 9.4 如果单卡 OOM：改为四卡分片（GPU4/5/6/7）

当你遇到类似报错：`torch.OutOfMemoryError`，说明单卡 24GB 不够或显存碎片严重。此时按下面方式启动：

```bash
export CUDA_VISIBLE_DEVICES=4,5,6,7
export QWEN_EDIT_2511_DIR=/home/zzy/weitiao/models/Qwen-Image-Edit-2511

# 降低显存碎片导致的大块分配失败
export PYTORCH_ALLOC_CONF=expandable_segments:True

# 给每张卡预留 headroom（单位 GiB），默认 6GiB（更稳，避免加载阶段峰值 OOM）
export QWEN_EDIT_GPU_HEADROOM_GIB=6

# 允许把部分权重/状态字典落到 CPU/磁盘（更稳，但会慢一些）
export QWEN_EDIT_OFFLOAD_FOLDER=/home/zzy/weitiao/cache/offload/qwen_image_edit_2511

python modules/02_qwen_image_edit_service/gradio_app.py
```

仍然 OOM 的处理顺序（从简单到激进）：
1) 把页面里的“最大边长”从 1024 降到 768 或 512
2) 把 steps 从 40 降到 30/20
3) 确认 GPU4-7 没有其它进程占显存：`nvidia-smi`（必要时先停掉占用进程）

#### 9.4.1 常见坑：vLLM/其它任务占满 GPU

如果 `nvidia-smi -i 4,5,6,7` 里看到类似 `VLLM::Worker_TP0/TP1/...` 或其它任务占用 20GB+，那么：
- 你**无法**使用这些 GPU 跑 Qwen-Image-Edit（无论单卡还是分片）。
- 这不是参数没调好，而是**显存没有空闲**。

本机历史排查结论：这些 `VLLM::Worker_*` 往往来自后台 Docker 容器（`python -m vllm.entrypoints.openai.api_server ...`）。
你可以用下面命令定位源头（不一定有权限停，但至少能解释“为什么总是回来”）：

```bash
docker ps --no-trunc | grep -i vllm || true
```

查看占用进程：

```bash
nvidia-smi -i 4,5,6,7 --query-compute-apps=pid,process_name,used_memory --format=csv
```

如果你有权限停止占用（先尝试温和退出，再必要时强杀）：

```bash
kill -15 <PID>
sleep 2
kill -9 <PID>
```

如果你**不允许**停止任何占用者，那么只能走“CPU 冒烟”或等待 GPU 空闲。

### 9.5 GPU 不空闲时：CPU 冒烟（推荐写博客时强调这一点）

当 GPU 被他人任务或容器服务占满时，你仍然可以先用 CPU 跑通“端到端链路”（下载→加载→编辑→出图）。
这一步的意义是：验证环境、权重、pipeline、UI、参数都对；等 GPU 空闲后再切回 GPU 提速。

启动命令（已内置资源限制，避免把服务器压满）：

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

CPU 冒烟调参建议（为了尽快出第一张）：
- `Steps`：10～15（先别用 40）
- `最大边长 max_side`：512（或更小）

## 10. 关键技术解读（写博客用）

### 10.1 diffusers / pipeline 在做什么

`diffusers.QwenImageEditPlusPipeline` 是一个“把多组件串起来”的推理管线：
- 读入输入图片与文本指令
- 将它们编码成模型可处理的中间表示
- 通过扩散采样（迭代 steps 次）逐步生成编辑后的结果

### 10.2 diffusion 为何会“等待很久”（尤其是 CPU）

你点击“生成”后看到的进度条一般形如：`0%| | 0/40`。

这不是卡死的充分证据，常见原因：
- **CPU 很慢**：每一步都要做大量矩阵计算，40 步在 CPU 上可能非常久。
- **第一步更慢**：可能包含额外初始化、缓存、内存分配等。

判断是否仍在计算的方式：
- 看 `top`：如果 `python gradio_app.py` 持续有 CPU 占用，通常仍在跑。

### 10.3 Steps（num_inference_steps）是什么

`Steps` 就是扩散采样的迭代步数：从噪声/中间状态逐步迭代到最终图片。
- 步数越多：通常质量更好/更稳定，但收益递减
- 步数越少：更快，但可能编辑不明显或细节不稳
- CPU 场景下：耗时基本近似随 steps 线性增长

### 10.4 true_cfg_scale 与 guidance_scale

本仓库 UI 同时暴露了 `true_cfg_scale` 与 `guidance_scale`：
- `true_cfg_scale`：更像“听 prompt 的程度/编辑强度”的核心旋钮（建议 3～5 起步）
- `guidance_scale`：在该模型上可能会被忽略（你会看到日志提示 ignored），因此不要指望它能明显改变效果

### 10.5 为什么 Gradio 会显示 Running on 0.0.0.0

`server_name="0.0.0.0"` 表示监听所有网卡地址，方便远程访问。
浏览器访问时不要用 `0.0.0.0`，改用 `127.0.0.1` 或服务器实际 IP。

### 10.6 资源限制：为什么要限制 CPU 线程与并发

CPU 冒烟的目标是“跑通链路”，不是抢占整机。
脚本支持：
- `QWEN_EDIT_MAX_CPU_THREADS`：限制 PyTorch CPU 线程数（默认约半核）
- `QWEN_EDIT_NICE`：降低进程优先级，避免影响其它服务
- `QWEN_EDIT_MAX_CONCURRENCY`：限制 Gradio 队列并发（默认 1）

