# vLLM 学习任务清单

**模块周期**：Day 3-4（2 天）
**难度等级**：⭐⭐⭐（中等）
**前置要求**：已完成 Day 1 单卡推理冒烟

---

## 📝 执行日志（实时更新）

### 2026-01-09 | Day 3 执行开始

#### ✅ 步骤 1：检查 GPU 资源
```bash
nvidia-smi
```
**状态**: ⏳ 待执行
**预期**: 确认至少有一张空闲 GPU（显存 > 16GB）
**实际结果**: 待记录

---

#### ✅ 步骤 2：安装 vLLM
```bash
conda activate videofen
pip install vllm
python -c "import vllm; print(vllm.__version__)"
```
**状态**: ⏳ 待执行
**预期**: 显示 vllm 版本号
**实际结果**: 待记录

---

#### ✅ 步骤 3：安装监控工具
```bash
pip install nvitop
```
**状态**: ⏳ 待执行
**实际结果**: 待记录

---

#### ✅ 步骤 4：单卡部署 7B 模型
```bash
export HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 2048
```
**状态**: ⏳ 待执行
**预期**: 看到 "Uvicorn running on http://0.0.0.0:8000"
**实际结果**: 待记录

---

#### ✅ 步骤 5：GPU 监控（新终端）
```bash
nvitop
```
**状态**: ⏳ 待执行
**预期**: GPU 0 显存占用约 7-8GB，GPU 利用率 > 80%
**实际结果**: 待记录

---

#### ✅ 步骤 6：API 测试（第三个终端）
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "你好，请介绍一下你自己"}],
    "temperature": 0.7
  }'
```
**状态**: ⏳ 待执行
**预期**: 返回 JSON 格式的模型回复
**实际结果**: 待记录

---

#### ✅ 步骤 7：查看性能指标
```bash
curl http://localhost:8000/metrics
```
**状态**: ⏳ 待执行
**预期**: 显示 vLLM 性能指标
**实际结果**: 待记录

---

### 📊 Day 3 验收进度
- [ ] vLLM 服务成功启动
- [ ] API 请求返回正常响应
- [ ] nvitop 显示显存利用率 > 90%
- [ ] GPU 利用率 > 80%

---

## 📋 学习目标

- [ ] 理解 vLLM 的核心价值（高性能推理引擎）
- [ ] 掌握 PagedAttention 原理（显存管理革命）
- [ ] 掌握 Continuous Batching 原理（吞吐量优化）
- [ ] 双卡张量并行部署 14B 模型
- [ ] OpenAI 兼容 API 测试与性能对比

---

## 📅 Day 3：基础理论与单卡部署

### 任务清单

#### 上午：理论学习（2-3 小时）
- [ ] 阅读 `modules/03_vllm/readme.md` 第 1-3 章
  - [ ] 理解 vLLM 在工具链中的位置（推理 vs 训练）
  - [ ] 理解 PagedAttention 机制（分页式显存管理）
  - [ ] 理解 Continuous Batching（连续批处理）
  - [ ] 理解核心指标（TTFT、TPOT、Throughput）

- [ ] 完成理论自测题
  ```
  Q1: vLLM 为什么能比 HuggingFace 推理快 3-10 倍？
  Q2: PagedAttention 和操作系统的虚拟内存有什么类比关系？
  Q3: Continuous Batching 解决了什么问题？
  ```

#### 下午：环境准备与安装（1-2 小时）
- [ ] 检查 GPU 资源
  ```bash
  nvidia-smi
  # 确认至少有一张空闲 GPU（建议显存 > 16GB）
  ```

- [ ] 安装 vLLM
  ```bash
  conda activate videofen
  pip install vllm

  # 验证安装
  python -c "import vllm; print(vllm.__version__)"
  ```

- [ ] （可选）安装监控工具
  ```bash
  pip install nvitop gpustat
  ```

#### 晚上：单卡部署冒烟（2-3 小时）
- [ ] 下载模型（如果本地没有）
  ```bash
  # 使用 HF 镜像下载
  export HF_ENDPOINT=https://hf-mirror.com
  huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
    --local-dir /path/to/models/Qwen2.5-7B-Instruct \
    --local-dir-use-symlinks False
  ```

- [ ] 单卡部署 7B 模型
  ```bash
  CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 2048
  ```

- [ ] OpenAI API 测试
  ```bash
  curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "Qwen/Qwen2.5-7B-Instruct",
      "messages": [{"role": "user", "content": "你好，请介绍一下你自己"}],
      "temperature": 0.7
    }'
  ```

- [ ] GPU 监控验证
  ```bash
  # 打开新终端运行
  nvitop
  # 观察：
  # - 显存利用率是否 > 90%
  # - GPU 利用率是否 > 80%
  ```

**Day 3 验收标准**：
- [ ] vLLM 服务成功启动
- [ ] API 请求返回正常响应
- [ ] 显存利用率 > 90%
- [ ] 能用 nvitop 监控 GPU 状态

---

## 📅 Day 4：双卡部署与性能测试

### 任务清单

#### 上午：双卡张量并行（3-4 小时）
- [ ] 理解张量并行原理
  - [ ] 阅读理论：张量并行 vs 数据并行
  - [ ] 理解为什么要用张量并行（单卡显存不足）

- [ ] 下载 14B 模型（如果本地没有）
  ```bash
  export HF_ENDPOINT=https://hf-mirror.com
  huggingface-cli download Qwen/Qwen2.5-14B-Instruct \
    --local-dir /path/to/models/Qwen2.5-14B-Instruct \
    --local-dir-use-symlinks False
  ```

- [ ] 双卡部署 14B 模型
  ```bash
  CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen2.5-14B-Instruct \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 2048 \
    --port 8000
  ```

- [ ] 验证双卡负载均衡
  ```bash
  nvitop
  # 观察两张卡的显存占用是否均衡（约 10-12GB/卡）
  ```

- [ ] API 功能测试
  ```bash
  curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "Qwen/Qwen2.5-14B-Instruct",
      "messages": [{"role": "user", "content": "写一首关于春天的诗"}],
      "max_tokens": 512
    }'
  ```

#### 下午：性能对比测试（3-4 小时）
- [ ] 性能基准测试脚本编写
  ```python
  # benchmark_vllm.py
  import time
  import requests
  import json

  def benchmark_vllm(prompt, num_runs=10):
    url = "http://localhost:8000/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
      "model": "Qwen/Qwen2.5-14B-Instruct",
      "messages": [{"role": "user", "content": prompt}],
      "max_tokens": 256
    }

    latencies = []
    for _ in range(num_runs):
      start = time.time()
      response = requests.post(url, headers=headers, json=data)
      end = time.time()
      latencies.append((end - start) * 1000)  # ms

    return {
      "avg_latency_ms": sum(latencies) / len(latencies),
      "min_latency_ms": min(latencies),
      "max_latency_ms": max(latencies)
    }

  if __name__ == "__main__":
    result = benchmark_vllm("解释一下什么是深度学习")
    print(json.dumps(result, indent=2))
  ```

- [ ] 运行基准测试
  ```bash
  python benchmark_vllm.py
  ```

- [ ] 对比 vLLM vs HuggingFace
  | 指标 | vLLM | HuggingFace | 提升倍数 |
  |------|------|-------------|----------|
  | 显存利用率 | ? | ? | ? |
  | 平均延迟 | ? | ? | ? |
  | 吞吐量 | ? | ? | ? |

#### 晚上：监控与日志（1-2 小时）
- [ ] 查看 vLLM 内置 Metrics
  ```bash
  curl http://localhost:8000/metrics
  ```

- [ ] 关键指标解读
  - `vllm:num_requests_running`: 运行中的请求数
  - `vllm:num_requests_waiting`: 排队中的请求数
  - `vllm:gpu_cache_usage_perc`: KV Cache 显存使用率
  - `vllm:time_to_first_token_ms`: TTFT
  - `vllm:time_per_output_token_ms`: TPOT

- [ ] （可选）Prometheus + Grafana 监控
  - [ ] 部署 Prometheus
  - [ ] 配置 Grafana Dashboard
  - [ ] 实时监控 vLLM 性能指标

**Day 4 验收标准**：
- [ ] 双卡 14B 模型成功部署
- [ ] 两张卡显存占用均衡（误差 < 10%）
- [ ] 完成性能基准测试
- [ ] 能解读关键性能指标
- [ ] （可选）搭建监控 Dashboard

---

## 🎯 模块验收标准

### 理论验收
- [ ] 能用自己的话解释 PagedAttention 原理
- [ ] 能用自己的话解释 Continuous Batching 优势
- [ ] 能说明张量并行与数据并行的区别

### 实操验收
- [ ] 单卡 7B 模型部署成功（API 可用）
- [ ] 双卡 14B 模型部署成功（负载均衡）
- [ ] 完成性能对比测试（vLLM vs HF）
- [ ] 能使用 nvitop/gpustat 监控 GPU

### 输出物
- [ ] 性能对比报告（表格形式）
- [ ] 部署命令笔记（含参数说明）
- [ ] （可选）监控 Dashboard 截图

---

## 📚 参考资源

### 官方文档
- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [vLLM 官方文档](https://docs.vllm.ai/)
- [PagedAttention 论文](https://arxiv.org/abs/2309.06180)

### 推荐阅读
- `modules/03_vllm/readme.md`（完整理论指南）
- [Continuous Batching 技术解析](https://luyuhuang.github.io/2023/08/23/continuous-batching.html)

### 常用命令速查
```bash
# 启动 vLLM 服务
vllm serve <model_path> [options]

# 核心参数
--tensor-parallel-size <n>      # 张量并行 GPU 数
--gpu-memory-utilization <0.9>  # GPU 显存利用率
--max-model-len <2048>          # 最大上下文长度
--host 0.0.0.0                   # 监听地址
--port 8000                      # 监听端口

# 查看指标
curl http://localhost:8000/metrics
```

---

## ⚠️ 常见问题

### Q1: vLLM 启动报显存不足？
**A**: 降低 `gpu-memory-utilization` 或 `max-model-len`
```bash
vllm serve model --gpu-memory-utilization 0.7 --max-model-len 1024
```

### Q2: 双卡部署时显存不均衡？
**A**: 检查 `CUDA_VISIBLE_DEVICES` 设置，确保两张卡都可见
```bash
# 查看可见 GPU
python -c "import torch; print(torch.cuda.device_count())"
```

### Q3: API 请求超时？
**A**: 增加 `max-model-len` 或降低请求并发数

---

## 🔄 与后续模块的衔接

完成本模块后，你将掌握：
- ✅ 生产级推理引擎的使用
- ✅ 多卡张量并行部署
- ✅ 性能监控与调优

**下一模块**：Day 5-6 Accelerate 分布式训练
- 学习如何统一管理单卡/多卡训练
- 为 DeepSpeed 显存优化打基础
