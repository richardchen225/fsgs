import os
from typing import Any

import torch


def cuda_memory_debug_enabled() -> bool:
    return os.environ.get("FSGS_CUDA_MEMORY_DEBUG", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def reset_cuda_peak_memory() -> None:
    if cuda_memory_debug_enabled() and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def log_cuda_memory(label: str, **details: Any) -> None:
    if not cuda_memory_debug_enabled() or not torch.cuda.is_available():
        return

    device = torch.cuda.current_device()
    gib = float(1024**3)
    stats = torch.cuda.memory_stats(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    rank = (
        torch.distributed.get_rank()
        if torch.distributed.is_available() and torch.distributed.is_initialized()
        else 0
    )
    detail_text = " ".join(f"{key}={value}" for key, value in details.items())
    print(
        f"[CUDA MEMORY][rank={rank}] {label} "
        f"allocated={torch.cuda.memory_allocated(device) / gib:.3f}GiB "
        f"reserved={torch.cuda.memory_reserved(device) / gib:.3f}GiB "
        f"peak={torch.cuda.max_memory_allocated(device) / gib:.3f}GiB "
        f"inactive_split={stats.get('inactive_split_bytes.all.current', 0) / gib:.3f}GiB "
        f"driver_free={free_bytes / gib:.3f}/{total_bytes / gib:.3f}GiB "
        f"alloc_retries={stats.get('num_alloc_retries', 0)} "
        f"ooms={stats.get('num_ooms', 0)}"
        + (f" {detail_text}" if detail_text else ""),
        flush=True,
    )
