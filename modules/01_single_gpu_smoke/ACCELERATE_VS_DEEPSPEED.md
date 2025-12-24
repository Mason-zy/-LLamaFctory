# accelerate vs deepspeed：两种“多卡启动/分布式”技术的详细讲解（结合 LLaMA-Factory）

适用范围：本仓库当前路径下的实践以 **LLaMA-Factory** 为主；你当前目标是 **多卡推理跑通**（不做微调）。本文重点解释：
- `accelerate` 与 `deepspeed` 各自是什么、解决什么问题
- 在“多卡推理（HF 推理栈）”里两者的真实差异
- 什么时候选哪个、如何排错、如何对比性能

> 结论先放前面：
> - 你说的“多卡推理”其实有两种目标，命令会完全不同：
>   - **显存分摊（单次对话/单次生成用多张卡）**：通常用 `llamafactory-cli chat/webui` 单进程推理，让 HF 的 `device_map="auto"`（或框架内部逻辑）把模型切到多卡。
>   - **吞吐并行（批量跑很多条输入）**：用 `accelerate launch`/`deepspeed` 起多进程做数据并行，每张卡一个进程。
> - `deepspeed` 在训练（特别是 ZeRO）场景更强；如果只当 launcher，用处和 accelerate 类似，但复杂度通常更高。

---

## 1. 两者分别是什么？

### 1.1 accelerate 是什么
- **accelerate = Hugging Face 的分布式启动器/封装层**。
- 核心能力：
  - 启动多进程（每张 GPU 一个进程）
  - 配置/注入 PyTorch Distributed 需要的环境（rank/world_size/master_addr/port 等）
  - 提供统一的 CLI（`accelerate launch ...`）与可选配置（`accelerate config`）

把它理解成：
- “我想用 4 张 GPU 跑同一段 Python 程序”，accelerate 帮你把程序 **复制成 4 个进程** 并让它们能互相通信。

### 1.2 deepspeed 是什么
- **DeepSpeed = 微软的分布式训练/推理优化框架**。
- 它也能做启动（`deepspeed --num_gpus 4 ...`），但更重要的是它的“引擎能力”：
  - ZeRO（把参数/梯度/优化器状态切分到多卡）
  - CPU/NVMe offload（把一部分状态挪到 CPU 或磁盘）
  - 训练吞吐优化、显存优化
  - 某些推理路径的优化（取决于你是否真的走 DS 推理引擎）

把它理解成：
- 不仅能“开 4 个进程”，还试图在训练/推理中 **更省显存/更高吞吐**，尤其在训练阶段最常用。

---

## 2. 共同点：它们都能当 launcher

无论 accelerate 还是 deepspeed，最基础都在做同一件事：
- 让同一个 Python 入口在多张 GPU 上以多进程方式运行

典型对应关系：
- `accelerate launch --num_processes 4 ...` ≈ “启动 4 个进程”
- `deepspeed --num_gpus 4 ...` ≈ “启动 4 个进程”

在你当前环境（pip 安装的 LLaMA-Factory 0.9.3）里，作为“可执行入口”的典型用法是：
- `-m llamafactory.launcher`（等价 `python -m llamafactory.launcher`）

特别提醒（你这次踩坑的根因）：
- `llamafactory.train` 在 pip 安装版里是一个 package，但没有 `__main__.py`，因此 **不能** `python -m llamafactory.train`，也就 **不能** `accelerate launch -m llamafactory.train`。
- pip 安装版会额外提供命令行入口 `llamafactory-cli`（以及同义的 `lmf`），很多“官方示例”的推理/聊天就是走这个入口。

也就是说：
- accelerate/deepspeed **只负责启动与分布式环境**
- 具体“推理怎么切、模型怎么加载、模板怎么用”仍由 **LLaMA-Factory + transformers** 决定

---

## 3. 关键差异：你是否真的“用上” DeepSpeed 的引擎能力

### 3.1 你现在的多卡推理命令在做什么
如果你用 `accelerate launch -m llamafactory.launcher` 这一类入口，它更像是“训练/评测/预测的一体化脚本”。

因此当你说“批量推理”时，通常指的是：
- 指定数据集（`--dataset ... --dataset_dir ...`）
- 打开预测开关（`--do_predict`，并常配 `--predict_with_generate`）
- 输出到目录（`--output_dir ...`）

这个组合通常意味着：
- 推理执行主要走 **transformers / HF 推理栈**
- DeepSpeed（如果你用它启动）在很多情况下更像是“只负责拉起多进程”

所以在这种场景下：
- `deepspeed` 相对 `accelerate` 的优势 **可能不会明显体现**（因为你没有进入 DS 引擎主导的推理路径）

### 3.2 什么时候 DeepSpeed 会明显更值得
常见情况是 **训练**：
- LoRA/全参训练，尤其模型更大、batch 更大时
- 需要 ZeRO-2/ZeRO-3 来显著降低显存压力

对推理来说，如果你希望利用 DS 的推理优化能力，通常需要满足：
- 选择/启用 DS 推理引擎相关路径（这取决于上层框架是否支持、如何配置）
- 而不是仅仅把 `deepspeed` 当作 launcher

---

## 4. 在你当前“多卡推理”任务里怎么选

### 4.1 推荐默认选 accelerate（更贴近 HF 生态）
原因：
- 参数更少，行为更直观
- 排错时更容易定位（端口、进程数、GPU 映射等）
- 和 `transformers`/`accelerate` 生态一致（很多示例都基于 accelerate）

### 4.2 什么时候用 deepspeed 也合理
- 你已经确定后续要做训练，并且想提前熟悉 `deepspeed` 的启动方式
- 你的团队/现有脚本体系就是 DS 为主
- 或者你明确要跑 DS 引擎相关能力（需要额外配置与验证）

---

## 5. GPU 映射：为什么命令里同时出现 CUDA_VISIBLE_DEVICES 和 --gpus

你现在的资源约束是“只用后排 4 卡”，所以我们用：
- `CUDA_VISIBLE_DEVICES=4,5,6,7`：让进程**只能看见**物理 GPU 4/5/6/7

随后 LLaMA-Factory 参数：
- `--gpus 0,1,2,3`：指的是“在可见 GPU 列表中的索引”

因此对应关系是：
- 可见 GPU 0 → 物理 GPU 4
- 可见 GPU 1 → 物理 GPU 5
- 可见 GPU 2 → 物理 GPU 6
- 可见 GPU 3 → 物理 GPU 7

这套写法的好处：
- 不管物理卡号是多少，你都能用 `--gpus 0,1,2,3` 这种固定写法

---

## 6. 两套启动命令模板（你可以二选一）

> 重要说明：
> - `accelerate launch`/`deepspeed` 本身只是“多进程启动器”，不会自动把它变成“对话式”。
> - 下面两条命令模板对应的是 **批量预测/批量推理（对数据集做 do_predict）**，所以必须提供 `--dataset/--dataset_dir/--output_dir`。
> - 如果你的目标是 **交互式单次对话**，请用 `llamafactory-cli chat`（见第 3 部分的解释）。

### 6.1 accelerate 版（推荐）
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRANSFORMERS_NO_FLASH_ATTENTION=1 \
accelerate launch --num_processes 4 --main_process_port 29501 -m llamafactory.launcher \
  --stage sft \
  --model_name_or_path /home/zzy/weitiao/models/Qwen2.5-7B-Instruct \
  --template qwen \
  --infer_backend huggingface \
  --dataset YOUR_DATASET_NAME \
  --dataset_dir /ABS/PATH/TO/datasets \
  --do_predict \
  --predict_with_generate \
  --output_dir outputs/predict_qwen2.5_7b \
  --gpus 0,1,2,3 \
  --cutoff_len 2048 \
  --per_device_eval_batch_size 1
```

### 6.2 deepspeed 版（更像“同样的事换个 launcher”）
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRANSFORMERS_NO_FLASH_ATTENTION=1 \
deepspeed --num_gpus 4 -m llamafactory.launcher \
  --stage sft \
  --model_name_or_path /home/zzy/weitiao/models/Qwen2.5-7B-Instruct \
  --template qwen \
  --infer_backend huggingface \
  --dataset YOUR_DATASET_NAME \
  --dataset_dir /ABS/PATH/TO/datasets \
  --do_predict \
  --predict_with_generate \
  --output_dir outputs/predict_qwen2.5_7b \
  --gpus 0,1,2,3 \
  --cutoff_len 2048 \
  --per_device_eval_batch_size 1
```

注意：以上两条命令在“HF 推理后端”下的差异往往主要体现为：
- 启动方式不同、日志风格不同
- 某些环境变量/端口冲突时的报错不同

---

## 7. 性能/显存对比：应该看什么指标

你今天要做“单卡 vs 多卡推理”的对比，建议记录：
- **显存峰值**：每张卡的 `max memory`（用 `nvidia-smi` 观察）
- **吞吐**：同一段 prompt 生成同样长度时的 tokens/s（如果工具不给 tokens/s，就记录 N 次生成的总耗时）
- **首 token 延迟**（可选）：开始生成到输出第一个 token 的时间

经验判断（不承诺具体数值）：
- 7B 级别模型，多卡的收益不一定线性（受限于通信、分割方式、KV cache 等）。
- 多卡更常见的收益是：
  - **容纳更大模型**（显存分摊）
  - 或者在特定并行策略下提升吞吐

---

## 8. 常见坑与排错（两者通用）

- 端口被占用：
  - accelerate：改 `--main_process_port 29501` 为别的端口
  - deepspeed：也可能需要设置/更换 master port（具体取决于 DS 版本与环境变量）
- GPU 用错：确认 `CUDA_VISIBLE_DEVICES=4,5,6,7` 是否生效；确认 `--gpus 0,1,2,3` 没写成物理卡号。
- 离线误联网：确认 `HF_HUB_OFFLINE=1` 与 `TRANSFORMERS_OFFLINE=1` 都在同一条命令前缀里。
- flash-attn 相关报错：确认 `TRANSFORMERS_NO_FLASH_ATTENTION=1`。

---

## 9. 你当前阶段的“最小建议”

- 今天目标是“多卡推理跑通并记录”，建议先只跑：
  - accelerate 版（更稳、更贴 HF）
- 如果你想对比 launcher 的差异，再补跑：
  - deepspeed 版（对比日志、是否更容易遇到端口/环境问题）

如果你把两次运行的关键输出（启动日志前 30 行 + nvidia-smi 截图/数值）贴出来，我可以帮你判断：
- 是否真的进入了多卡并行
- 哪个启动方式在你这台机子上更稳定
- 是否存在“看起来多卡但实际上单卡”的误配




这份教学文档将为您彻底理清 **Accelerate**、**DeepSpeed** 和 **LLaMA-Factory** 的关系，并解释多卡推理的底层逻辑。

我们可以用一个**“建筑施工队”**的通俗比喻来理解这三者：

---

# 教学文档：LLM 训练与推理的三驾马车

## 1. 角色大揭秘：他们是干什么的？

### 🛠️ **LLaMA-Factory：包工头 (The Commander)**
*   **定位**：**上层框架 / 一站式工具箱**。
*   **作用**：它不直接管理显卡底层，而是负责整合资源。它把数据处理、模型加载、Prompt 模板、训练流程都封装好了。
*   **你在这个层级做什么**：你只需要告诉它“我要微调 Qwen2.5”，“用这个数据集”，“学习率设为 1e-5”。你不需要写一行 PyTorch 代码。
*   **关系**：它负责发号施令，底层脏活累活交给 Accelerate 和 DeepSpeed。

### ⚖️ **Accelerate (Hugging Face)：现场调度员 (The Manager)**
*   **定位**：**硬件抽象层 / 启动器**。
*   **作用**：它是一个“翻译官”。原生的 PyTorch 代码在单卡、多卡、TPU 上的写法是不一样的。Accelerate 的作用就是让同一套代码能运行在不同的硬件环境上。它负责**启动分布式进程**。
*   **核心功能**：
    *   `accelerate launch`：帮你自动分配 GPU 进程（比如你有 4 张卡，它就启动 4 个进程）。
    *   处理混合精度（FP16/BF16）。
*   **能否配合 LLaMA-Factory？** **必须配合**。LLaMA-Factory 的底层就是构建在 Accelerate 之上的。

### 🚀 **DeepSpeed (Microsoft)：重型机械 / 压缩大师 (The Turbo Engine)**
*   **定位**：**显存优化与加速引擎**。
*   **作用**：它不是用来“启动”任务的，而是用来**省显存**和**提速**的。
*   **核心绝招 (ZeRO 技术)**：
    *   普通的训练会把模型参数复制到每张卡上（非常占显存）。
    *   DeepSpeed (ZeRO Stage 2/3) 会把模型**切碎**，这张卡存一部分，那张卡存一部分，训练时再拼起来。这让小显存也能跑大模型。
    *   **Offload**：显存不够时，把数据暂时扔到内存（CPU RAM）里去。

---

## 2. 三者关系图谱

它们是一个**层级堆叠**的关系，而不是并列关系：

```text
+-------------------------------------------------------+
|                用户 (User / CLI / WebUI)               |
+-------------------------------------------------------+
|           LLaMA-Factory (业务逻辑层)                   |  <-- 你在这里操作
+-------------------------------------------------------+
|             Accelerate (硬件调度层)                    |  <-- 负责管理多卡通信
+-------------------------------------------------------+
|             DeepSpeed (可选的优化后端)                  |  <-- 负责省显存、切分模型
+-------------------------------------------------------+
|               PyTorch (基础计算框架)                   |
+-------------------------------------------------------+
|                CUDA / GPU (物理硬件)                   |
+-------------------------------------------------------+
```

*   **常规模式**：LLaMA-Factory 调用 Accelerate，Accelerate 调用 PyTorch。
*   **高性能模式**：LLaMA-Factory 调用 Accelerate，Accelerate 配置 DeepSpeed 策略，DeepSpeed 优化 PyTorch。

---

## 3. 多卡推理是如何实现的？

您在之前的报错中遇到了困惑，是因为混淆了**训练中的多卡**和**推理中的多卡**。它们的实现逻辑完全不同。

### 场景 A：数据并行 (Data Parallelism) -> 适合批量处理
*   **原理**：你有 4 张卡，就把模型**复制 4 份**。如果有 100 个问题要问，每张卡处理 25 个。
*   **工具**：`accelerate launch`。
*   **缺点**：**不适合聊天 (Chat)**。因为聊天是串行的（你说一句，它回一句），启动 4 个进程会导致 4 个模型同时等你说话，或者端口冲突。
*   **适用**：`--stage sft --do_predict`（批量跑测试集）。

### 场景 B：模型并行 (Pipeline Parallelism) -> 适合单次对话
*   **原理**：模型太大（比如 72B），单张卡放不下。把模型的第 1-10 层放在 GPU0，11-20 层放在 GPU1... 数据流像流水线一样流过所有显卡。
*   **工具**：HuggingFace 的 `device_map="auto"`。
*   **实现方式**：
    在 LLaMA-Factory 中，**不需要** `accelerate launch`。只需要指定 `CUDA_VISIBLE_DEVICES=0,1,2,3`，代码内部会自动检测多卡并进行切分。

**💡 结论：**
*   **如果是聊天/WebUI**：不要用 `accelerate launch`，直接用 `python` 或 `llamafactory-cli`，框架会自动把大模型切分到多张卡上（如果显存够大，也可以只用一张）。
*   **如果是微调训练**：必须用 `accelerate launch`，因为训练需要大规模并行计算。

---

## 4. 实战操作指南 (Cheat Sheet)

### 情况 1：我要微调训练 (Training)
此时需要三个工具火力全开。

**命令结构：**
```bash
# 调用 accelerate 启动器
accelerate launch \
    --config_file examples/accelerate/ds_zero2_config.yaml \  # 告诉 accelerate 启用 DeepSpeed
    -m llamafactory.launcher \                                # 启动 LLaMA-Factory
    --stage sft \
    ...
```
*   **解释**：Accelerate 负责拉起多卡进程，并根据 yaml 配置文件加载 DeepSpeed 引擎来优化显存。

### 情况 2：我要批量评测数据 (Batch Evaluation)
你想快点跑完测试集，可以用数据并行。

**命令结构：**
```bash
accelerate launch \
    -m llamafactory.launcher \
    --stage sft \
    --do_predict \
    ...
```

### 情况 3：我要聊天/演示 (Inference / Chat)
**千万别用 accelerate launch！**

**命令结构：**
```bash
# 直接指定可见的卡
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli chat \
    --model_name_or_path ... \
    --infer_backend huggingface
```
*   **解释**：这里 LLaMA-Factory 会使用 `device_map="auto"`。假设模型是 7B，显存 24G，它可能只会用 GPU0。假设模型是 72B，它会自动把模型切碎铺在 0,1,2,3 号卡上。

---

## 5. 总结

1.  **Accelerate** 是**管家**，负责把你的代码分发到多张显卡上运行。
2.  **DeepSpeed** 是**压缩机**，当你显存不够训练大模型，或者想练得更快时，通过 Accelerate 的配置文件开启它。
3.  **LLaMA-Factory** 是**老板**，你只跟老板对话。
4.  **多卡推理**：
    *   **聊天 (Chat)**：直接运行 `llamafactory-cli`，让它自动分配 (`device_map="auto"`)。
    *   **刷分 (Benchmark)**：可以用 `accelerate launch` 并行刷。

**修正您之前的错误命令：**
您之前尝试用 `accelerate launch` 去启动 `chat`（聊天模式），这是**杀鸡用牛刀且用错了刀法**。对于 Chat，直接单进程启动，让内部库自己去管理显存即可。