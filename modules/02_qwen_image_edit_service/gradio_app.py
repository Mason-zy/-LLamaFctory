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
            "Set it to your local model directory, e.g. /home/zzy/weitiao/models/Qwen-Image-Edit-2511"
        )
    return model_dir


def _force_cpu() -> bool:
    return os.environ.get("QWEN_EDIT_FORCE_CPU", "0") == "1"


def _maybe_limit_resources() -> None:
    """Best-effort resource limits to avoid overloading the host.

    - Caps torch CPU threads (intra/inter-op)
    - Optionally lowers process priority (nice)
    """

    # Default: use about half of the machine cores.
    default_threads = max(1, (os.cpu_count() or 1) // 2)
    max_threads = int(os.environ.get("QWEN_EDIT_MAX_CPU_THREADS", str(default_threads)))
    max_threads = max(1, max_threads)

    # Lower priority so background services stay responsive.
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
    """Fail fast if visible GPUs are already heavily occupied.

    This prevents confusing OOMs during `from_pretrained()` that are actually caused by other jobs
    (e.g. vLLM workers) using most of the VRAM.
    """

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
            "Not enough free VRAM on visible GPUs. "
            f"Need >= {min_free_gib:.0f} GiB free per GPU, but got: {details}. "
            "Please stop other GPU jobs first (check with `nvidia-smi -i 4,5,6,7`)."
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

    # If multiple GPUs are visible, shard the pipeline across GPUs to reduce per-GPU VRAM.
    # This is the simplest way to make big diffusion models fit when a single 24GB card is not enough.
    gpu_count = torch.cuda.device_count()
    if gpu_count >= 2:
        # Leave some headroom on each GPU for activations/temporary buffers to reduce OOM risk.
        # NOTE: When CUDA_VISIBLE_DEVICES=4,5,6,7, the visible GPU indices are 0..3.
        headroom_gib = int(os.environ.get("QWEN_EDIT_GPU_HEADROOM_GIB", "6"))
        gpu0_extra_headroom_gib = int(os.environ.get("QWEN_EDIT_GPU0_EXTRA_HEADROOM_GIB", "4"))
        max_memory = {}
        for i in range(gpu_count):
            total_gib = int(torch.cuda.get_device_properties(i).total_memory / (1024**3))
            effective_headroom = headroom_gib + (gpu0_extra_headroom_gib if i == 0 else 0)
            max_gib = max(4, total_gib - effective_headroom)
            max_memory[i] = f"{max_gib}GiB"
        # Allow offload if needed.
        max_memory["cpu"] = os.environ.get("QWEN_EDIT_CPU_MAX_MEMORY", "120GiB")

        offload_folder = os.environ.get(
            "QWEN_EDIT_OFFLOAD_FOLDER", "/home/zzy/weitiao/cache/offload/qwen_image_edit_2511"
        )
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

    # Reduce peak memory during VAE + attention.
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
        raise gr.Error("请先输入编辑需求（prompt）")

    if (not _force_cpu()) and (not torch.cuda.is_available()):
        raise gr.Error("CUDA 不可用：请检查 NVIDIA 驱动与 torch CUDA 环境（或设置 QWEN_EDIT_FORCE_CPU=1 用 CPU 冒烟）")

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

    # Eager-load the pipeline at startup so OOMs show immediately (instead of after clicking).
    if os.environ.get("QWEN_EDIT_EAGER_LOAD", "1") == "1":
        _get_pipe()

    with gr.Blocks(title=title) as demo:
        gr.Markdown(
            """
# Qwen-Image-Edit-2511（本地交互式）

- 上传图片 → 输入编辑需求 → 点击生成 → 返回结果图
- 建议先确保已完成模型下载，并设置：`QWEN_EDIT_2511_DIR` 指向本地模型目录
""".strip()
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_in = gr.Image(type="pil", label="输入图片")
                prompt = gr.Textbox(lines=3, label="编辑需求（Prompt）")

                with gr.Row():
                    seed = gr.Number(value=0, precision=0, label="Seed（-1 随机）")
                    steps = gr.Slider(minimum=10, maximum=80, step=1, value=40, label="Steps")

                with gr.Row():
                    true_cfg = gr.Slider(minimum=1.0, maximum=8.0, step=0.1, value=4.0, label="true_cfg_scale")
                    guidance = gr.Slider(minimum=0.5, maximum=3.0, step=0.1, value=1.0, label="guidance_scale")

                max_side = gr.Slider(
                    minimum=0,
                    maximum=2048,
                    step=64,
                    value=768,
                    label="最大边长（>0 时自动缩放，避免 OOM）",
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
