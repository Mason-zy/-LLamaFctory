# Copilot 使用指引（中文）

目的：帮助用户在本环境下学习并实践大模型推理与全流程微调，遵守资源约束，按“先冒烟、再放大”的低风险路径积累经验。

## 环境快照
- OS: CentOS Stream 9；Conda 环境：videofen
- 关键库：PyTorch 2.3.1 + CUDA 12.1（torch/deepspeed/transformers/ktransformers 已可导入）
- 硬件：2× Intel Xeon Gold 6530（64 核 AMX/AVX512）、1 TB RAM、8× RTX 4090 (24 GB)
- 工作目录：/home/zzy/weitiao

## 硬性资源约束
- GPU：优先使用当前空闲的卡（运行前先看 `nvidia-smi`）。本机最近空闲示例：单卡用 `CUDA_VISIBLE_DEVICES=6`；两卡用 `CUDA_VISIBLE_DEVICES=6,7`。
   - 若命令里同时出现 `CUDA_VISIBLE_DEVICES` 和 `--gpus`：`--gpus` 是“可见 GPU 列表里的索引”（从 0 开始），不是物理卡号。
- CPU：只用约一半核心，避免占满整机
- 未过冒烟前，保持小 batch 与短序列，避免 OOM

## 学习与实践路径（冒烟 → 放大）
1) 文本单卡推理冒烟：验证权重、模板、tokenizer
   ```bash
    CUDA_VISIBLE_DEVICES=6 llamafactory-cli chat \
     --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
     --template qwen
   ```
2) 多模态单卡冒烟：验证视觉/图文链路
   ```bash
    CUDA_VISIBLE_DEVICES=6 llamafactory-cli chat \
     --model_name_or_path Qwen/Qwen2-VL-7B-Instruct \
     --template qwen2_vl
   ```
3) 多卡推理（张量并行；按“当前空闲卡数”选择进程数）
   ```bash
    CUDA_VISIBLE_DEVICES=6,7 accelerate launch --num_processes 2 src/train.py \
     --stage inference \
     --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
     --template qwen \
     --infer_backend huggingface \
       --gpus 0,1
   ```
   预期：显存约 4 GB/卡，吞吐约单卡的 2.3×。
4) 文本 LoRA 复现实例（勿改超参，仅改数据路径）
   ```bash
    CUDA_VISIBLE_DEVICES=6,7 deepspeed --num_gpus 2 src/train.py \
     --deepspeed examples/deepspeed/ds_z3_bf16.json \
     --stage sft \
     --config_file examples/train_lora/llama3_lora_sft.yaml
   ```
   训练后用同一 config，把 adapter_name_or_path 指向产物再做推理，验证闭环。
5) 多模态 LoRA：同上流程，模型换 Qwen2-VL-7B，数据用 LLaMA-Factory 的图文混合格式（如 COCO 2017 抽样 + caption），其余脚本与卡数不变。
6) 可选放大：
   - 32B：Qwen2.5-32B，BF16-LoRA，4 卡峰值约 22 GB/卡
   - 70B：Llama-3.1-70B，NF4-QLoRA，4 卡峰值约 20 GB/卡
   记录显存峰值、训练速度 (tokens/s)、MT-bench 得分。

7) 图像编辑模型和图像生成模型的学习与实践，待补充具体步骤。

## 工作习惯与调试
- 先冒烟再放大：任何新模型/数据都按“单卡文本 → 单卡多模态 → 多卡 → 放大规模”重走一遍。
- 记录与回溯：保留 VRAM、吞吐、日志；出现 OOM 先降 batch 或序列长度。
- 遵守改动范围：不回滚用户已有改动；仅在必要文件内修改。
- 若需更多 GPU 或更大 batch，先确认资源占用与风险。

## 待补充
- 当前缺少仓库代码与目录信息，后续同步代码后，请补充关键入口（数据加载、训练脚本、配置）与项目特定约定。
- 若出现 AGENT.md/README 等指导文件，请合并其规则，保持一致性。
