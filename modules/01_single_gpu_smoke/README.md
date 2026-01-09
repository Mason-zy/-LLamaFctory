# 模块 01：单卡推理冒烟测试

> **目标**：在单张 GPU 上完成最小可行推理，验证模型权重、模板、tokenizer 与多模态链路，形成可靠基线。

---

## 目录

1. [学习目标](#学习目标)
2. [前置准备](#前置准备)
3. [环境配置](#环境配置)
4. [模型下载](#模型下载)
5. [实战步骤](#实战步骤)
6. [Web UI 使用](#web-ui-使用)
7. [常见问题](#常见问题)
8. [附录：多卡推理](#附录多卡推理)

---

## 学习目标

完成本模块后，你将掌握：

- ✅ 使用 LlamaFactory CLI 进行模型推理
- ✅ 理解模型模板（Template）的作用
- ✅ 掌握本地模型下载与离线部署
- ✅ 了解显存占用与性能基线测试
- ✅ 多模态（文本+图像）模型的基本使用

---

## 前置准备

### 硬件要求

| 资源 | 最低配置 | 推荐配置 |
|------|---------|---------|
| GPU | 1 张（显存 ≥ 16GB） | NVIDIA 4090 / A100 |
| 内存 | 32GB | 64GB |
| 磁盘 | 50GB 可用空间 | 100GB+ SSD |

### 预期显存占用

| 模型 | 显存占用 | 说明 |
|------|---------|------|
| Qwen2.5-7B-Instruct | ~14 GB | 文本模型 |
| Qwen2-VL-7B-Instruct | ~18 GB | 多模态模型 |

---

## 环境配置

### Step 0：准备目录与镜像配置

> **说明**：此步骤在仓库根目录执行一次即可

```bash
# 切换到工作目录
cd /home/zzy/weitiao

# 创建必要的目录结构
mkdir -p logs models .cache/huggingface

# 配置 PyPI 镜像源（加速 Python 包下载）
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 配置 HuggingFace 镜像与缓存目录
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/home/zzy/weitiao/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/home/zzy/weitiao/.cache/huggingface/hub
```

**提示**：将上述 `export` 命令写入 `~/.bashrc` 可永久生效

```bash
# 添加到 ~/.bashrc
cat >> ~/.bashrc << 'EOF'
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/home/zzy/weitiao/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/home/zzy/weitiao/.cache/huggingface/hub
EOF

# 重新加载配置
source ~/.bashrc
```

---

### Step 1：安装 LlamaFactory

```bash
# 激活 Conda 环境
conda activate videofen

# 升级 pip
pip install -U pip

# 移除可能冲突的 flash-attn（可选）
pip uninstall -y flash-attn flash_attn flash-attn-xformers || true

# 安装 LlamaFactory 及依赖
pip install -U --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  llamafactory==0.9.3 \
  transformers==4.52.4 \
  accelerate==1.7.0 \
  datasets==3.6.0 \
  gradio==5.31.0 \
  gradio_client==1.10.1 \
  tokenizers==0.21.1

# 设置环境变量（禁用 flash attention）
export TRANSFORMERS_NO_FLASH_ATTENTION=1
```

#### 验证安装

```bash
# 检查 LlamaFactory CLI
llamafactory-cli --help

# 验证 Python 导入
python - <<'PY'
import llamafactory
from llamafactory.cli import main
print('✓ LlamaFactory 安装成功')
PY

# 验证核心依赖
python - <<'PY'
import torch, transformers, deepspeed
print('✓ torch', torch.__version__)
print('✓ transformers', transformers.__version__)
print('✓ deepspeed', deepspeed.__version__)
PY
```

---

## 模型下载

### Step 2：下载模型到本地

> **说明**：推荐提前下载模型到本地，避免推理时临时下载

#### 确认 huggingface_hub 版本

```bash
pip install -U "huggingface_hub==0.36.0"
```

#### 下载文本模型（Qwen2.5-7B-Instruct）

```bash
hf download Qwen/Qwen2.5-7B-Instruct \
  --local-dir /home/zzy/weitiao/models/Qwen2.5-7B-Instruct \
  --max-workers 4
```

#### 下载多模态模型（Qwen2-VL-7B-Instruct）

```bash
hf download Qwen/Qwen2-VL-7B-Instruct \
  --local-dir /home/zzy/weitiao/models/Qwen2-VL-7B-Instruct \
  --max-workers 4
```

#### 备用下载方式

如果 `hf download` 失败，可以使用 huggingface-cli：

```bash
huggingface-cli download Qwen/Qwen2-VL-7B-Instruct \
  --local-dir /home/zzy/weitiao/models/Qwen2-VL-7B-Instruct \
  --local-dir-use-symlinks False
```

#### 验证模型文件

确保下载的目录包含以下关键文件：

```
Qwen2.5-7B-Instruct/
├── config.json
├── tokenizer.json
├── tokenizer_config.json
├── model-00001-of-00008.safetensors
├── model-00002-of-00008.safetensors
└── ...
```

---

## 实战步骤

### Step 3：文本模型推理冒烟测试

**目标**：验证 Qwen2.5-7B-Instruct 能正常对话

```bash
CUDA_VISIBLE_DEVICES=6 TRANSFORMERS_NO_FLASH_ATTENTION=1 \
llamafactory-cli chat \
  --model_name_or_path /home/zzy/weitiao/models/Qwen2.5-7B-Instruct \
  --template qwen
```

**预期结果**：

```
>>
User: 你好
Assistant: 你好！有什么我可以帮助你的吗？
```

**成功标志**：
- ✅ 出现交互提示符 `>> User:`
- ✅ 输入中/英文后模型能正常回复
- ✅ 显存占用约 14GB（使用 `nvidia-smi` 查看）

---

### Step 4：多模态模型推理冒烟测试

**目标**：验证 Qwen2-VL 能理解图像内容

#### 准备测试图片

```bash
# 下载一张示例图片
wget https://example.com/sample.jpg -O /home/zzy/weitiao/sample.jpg
```

#### 启动多模态对话

```bash
CUDA_VISIBLE_DEVICES=6 TRANSFORMERS_NO_FLASH_ATTENTION=1 \
llamafactory-cli chat \
  --model_name_or_path /home/zzy/weitiao/models/Qwen2-VL-7B-Instruct \
  --template qwen2_vl
```

**测试方法**：

1. 根据提示上传图片：`/home/zzy/weitiao/sample.jpg`
2. 输入问题：`描述这张图片的内容`

**预期结果**：

```
>>
Image: /home/zzy/weitiao/sample.jpg
User: 描述这张图片的内容
Assistant: 这是一张...
```

**成功标志**：
- ✅ 模型能正确加载图片
- ✅ 能生成合理的图像描述
- ✅ 显存占用约 18GB

---

### Step 5：记录测试结果

创建日志文件记录测试结果：

```bash
cat >> /home/zzy/weitiao/logs/smoke_test.log << 'EOF'
====================================
[2025-12-24] 单卡推理冒烟测试
====================================

文本模型（Qwen2.5-7B-Instruct）:
  命令: CUDA_VISIBLE_DEVICES=6 llamafactory-cli chat \
        --model_name_or_path /home/zzy/weitiao/models/Qwen2.5-7B-Instruct \
        --template qwen
  显存峰值: ~14GB
  延迟: ~X s/turn
  状态: ✓ 通过
  备注: 正常吐字

多模态模型（Qwen2-VL-7B-Instruct）:
  命令: CUDA_VISIBLE_DEVICES=6 llamafactory-cli chat \
        --model_name_or_path /home/zzy/weitiao/models/Qwen2-VL-7B-Instruct \
        --template qwen2_vl
  显存峰值: ~18GB
  延迟: ~X s/turn
  状态: ✓ 通过
  备注: 图像理解正常

环境信息:
  GPU: NVIDIA 4090 (GPU 6)
  PyTorch: 2.3.1
  CUDA: 12.1
  LlamaFactory: 0.9.3
EOF
```

---

## Web UI 使用

### Step 6：启动 Web UI（可选）

如果你更习惯图形界面，可以使用 LlamaFactory 的 Web UI。

#### 安装 Web 依赖

```bash
pip install -U gradio fastapi uvicorn
```

#### 启动 Web UI

**文本模型**：

```bash
CUDA_VISIBLE_DEVICES=6 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
TRANSFORMERS_NO_FLASH_ATTENTION=1 \
llamafactory-cli webui \
  --model_name_or_path /home/zzy/weitiao/models/Qwen2.5-7B-Instruct \
  --cache_dir /home/zzy/weitiao/.cache/huggingface \
  --template qwen \
  --server_name 0.0.0.0 \
  --port 7860
```

**多模态模型**：

```bash
CUDA_VISIBLE_DEVICES=6 \
TRANSFORMERS_NO_FLASH_ATTENTION=1 \
llamafactory-cli webui \
  --model_name_or_path /home/zzy/weitiao/models/Qwen2-VL-7B-Instruct \
  --template qwen2_vl \
  --server_name 0.0.0.0 \
  --port 7860
```

#### 访问界面

- **本机访问**：http://127.0.0.1:7860
- **远程访问**：http://<服务器IP>:7860

#### UI 配置建议

| 配置项 | 推荐值 | 说明 |
|--------|--------|------|
| 模型路径 | `/home/zzy/weitiao/models/Qwen2.5-7B-Instruct` | 必须使用本地路径 |
| 推理引擎 | huggingface | 标准推理 |
| 微调方法 | none | 不使用微调 |
| 量化 | none | 不使用量化 |
| RoPE | auto | 自动配置 |

---

## 常见问题

### 问题 1：显存不足（OOM）

**现象**：
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**解决方案**：

| 方案 | 操作 | 效果 |
|------|------|------|
| 1. 切换到空闲 GPU | `CUDA_VISIBLE_DEVICES=X` | 避免多任务争抢 |
| 2. 使用更小的模型 | 换用 3B 或 1B 参数模型 | 降低显存需求 |
| 3. 减小序列长度 | 在配置中设置 `max_length` | 减少显存占用 |

---

### 问题 2：模型下载失败

**现象**：
```
ConnectionError: Can't reach huggingface.co
```

**解决方案**：

```bash
# 方案 1：使用镜像源
export HF_ENDPOINT=https://hf-mirror.com

# 方案 2：配置代理（如有）
export HTTPS_PROXY=http://127.0.0.1:7890

# 方案 3：降低并发
hf download ... --max-workers 1

# 方案 4：多次重试（支持断点续传）
hf download ... --local-dir /path/to/model
```

---

### 问题 3：模板错误

**现象**：
```
ValueError: Template 'qwen' not found
```

**解决方案**：

| 模型 | 正确模板 |
|------|---------|
| Qwen2.5-7B-Instruct | `--template qwen` |
| Qwen2-VL-7B-Instruct | `--template qwen2_vl` |
| Llama3-8B-Instruct | `--template llama3` |

---

### 问题 4：Token 加载失败

**现象**：
```
OSError: Can't load tokenizer for 'Qwen/Qwen2.5-7B-Instruct'
```

**解决方案**：

```bash
# 1. 重新下载 tokenizer 文件
hf download Qwen/Qwen2.5-7B-Instruct \
  --local-dir /home/zzy/weitiao/models/Qwen2.5-7B-Instruct \
  --max-workers 4

# 2. 清理损坏的缓存
rm -rf /home/zzy/weitiao/.cache/huggingface/hub/models--Qwen--*

# 3. 临时关闭离线模式补齐文件
unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE
```

---

### 问题 5：Web UI 加载错误

**现象**：UI 自动切换到在线模型并报错

**解决方案**：

```bash
# 1. 强制使用离线模式
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 llamafactory-cli webui ...

# 2. 清理残留缓存
rm -rf /home/zzy/weitiao/.cache/huggingface/hub/models--Qwen--Qwen-7B

# 3. 在 UI 中手动输入完整本地路径
# 不要从下拉菜单选择
```

---

## 附录：多卡推理

> **前置条件**：完成上述单卡测试，并有 2 张以上空闲 GPU

### 使用多张 GPU 推理

```bash
CUDA_VISIBLE_DEVICES=6,7 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
TRANSFORMERS_NO_FLASH_ATTENTION=1 \
llamafactory-cli chat \
  --model_name_or_path /home/zzy/weitiao/models/Qwen2.5-7B-Instruct \
  --template qwen \
  --infer_backend huggingface
```

### 验证多卡生效

**方法**：在推理过程中另开终端查看 GPU 状态

```bash
watch -n 1 nvidia-smi
```

**预期结果**：GPU 6 和 GPU 7 都有显存占用

### 技术说明

| 概念 | 说明 | 使用场景 |
|------|------|----------|
| **设备映射** | 单进程多卡，自动分片 | 交互式推理 |
| **数据并行** | 多进程多卡，复制模型 | 训练/批量推理 |
| **张量并行** | 单请求跨卡 | 大模型推理 |

**注意**：`llamafactory-cli chat` 默认使用设备映射模式，适合交互式推理。

---

## 完成检查清单

完成本模块后，请确认以下项目：

- [ ] 文本模型（Qwen2.5-7B）能正常对话
- [ ] 多模态模型（Qwen2-VL）能理解图像
- [ ] 记录了显存峰值和延迟数据
- [ ] 理解了 `--template` 参数的作用
- [ ] 掌握了本地模型下载与离线部署
- [ ] 能独立排查常见错误

---

## 下一步

完成本模块后，可以进入：

- **模块 02**：多卡微调实战
- **模块 03**：vLLM 高性能推理
- **模块 04**：量化与性能优化

---

## 参考资源

- [LlamaFactory 官方文档](https://github.com/hiyouga/LLaMA-Factory)
- [Qwen2.5 模型卡](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [Qwen2-VL 模型卡](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)
