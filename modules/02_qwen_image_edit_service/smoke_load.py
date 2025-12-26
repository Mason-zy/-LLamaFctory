import os
import time

import torch


def _fmt_gib(num_bytes: int) -> str:
    return f"{num_bytes / (1024**3):.2f} GiB"


def _print_cuda_mem(prefix: str) -> None:
    if not torch.cuda.is_available():
        print(f"{prefix} CUDA not available")
        return

    parts = []
    for i in range(torch.cuda.device_count()):
        free_b, total_b = torch.cuda.mem_get_info(i)
        parts.append(f"cuda:{i} free={_fmt_gib(free_b)}/{_fmt_gib(total_b)}")
    print(prefix + " " + " | ".join(parts), flush=True)


def main() -> None:
    print("== Qwen-Image-Edit-2511 smoke load ==")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"QWEN_EDIT_2511_DIR={os.environ.get('QWEN_EDIT_2511_DIR')}")
    print(f"QWEN_EDIT_GPU_HEADROOM_GIB={os.environ.get('QWEN_EDIT_GPU_HEADROOM_GIB')}")
    print(f"QWEN_EDIT_GPU0_EXTRA_HEADROOM_GIB={os.environ.get('QWEN_EDIT_GPU0_EXTRA_HEADROOM_GIB')}")
    print(f"QWEN_EDIT_CPU_MAX_MEMORY={os.environ.get('QWEN_EDIT_CPU_MAX_MEMORY')}")
    print(f"QWEN_EDIT_OFFLOAD_FOLDER={os.environ.get('QWEN_EDIT_OFFLOAD_FOLDER')}")
    print(f"QWEN_EDIT_DTYPE={os.environ.get('QWEN_EDIT_DTYPE')}")

    _print_cuda_mem("Before load:")

    t0 = time.time()
    from gradio_app import _get_pipe

    pipe = _get_pipe()
    dt = time.time() - t0

    _print_cuda_mem("After load :")
    print(f"Loaded OK in {dt:.1f}s. Pipeline={type(pipe).__name__}")


if __name__ == "__main__":
    main()
