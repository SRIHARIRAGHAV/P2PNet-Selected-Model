from contextlib import nullcontext

import torch


def select_device(config):
    requested = str(config["gpu"]["device"]).lower()
    if requested == "cpu":
        return torch.device("cpu")

    cuda_available = torch.cuda.is_available()
    if requested == "cuda" and not cuda_available:
        raise RuntimeError(
            "CUDA training was requested with gpu.device=cuda, "
            "but CUDA is unavailable in this PyTorch environment."
        )
    if requested == "auto" and not cuda_available:
        return torch.device("cpu")
    if requested not in {"auto", "cuda"}:
        raise RuntimeError("gpu.device must be 'auto', 'cpu', or 'cuda'.")

    index = int(config["gpu"]["gpu_device_index"])
    if index < 0 or index >= torch.cuda.device_count():
        raise RuntimeError(
            f"Requested GPU index {index}, but only "
            f"{torch.cuda.device_count()} CUDA device(s) are available."
        )
    torch.cuda.set_device(index)
    device = torch.device("cuda", index)
    properties = torch.cuda.get_device_properties(device)
    free_memory, total_memory = torch.cuda.mem_get_info(device)
    print(f"CUDA device: {properties.name}")
    print(
        "GPU memory: {:.2f} GiB total, {:.2f} GiB free".format(
            total_memory / (1024 ** 3),
            free_memory / (1024 ** 3),
        )
    )
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA build: {torch.version.cuda}")
    return device


def amp_settings(config, device):
    enabled = device.type == "cuda" and bool(config["gpu"]["mixed_precision"])
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[str(config["gpu"]["amp_dtype"]).lower()]
    if enabled and dtype is torch.bfloat16:
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "bfloat16 mixed precision was requested, but the selected "
                "CUDA device does not support it."
            )
    return enabled, dtype


def autocast_context(enabled, dtype, device):
    if enabled:
        return torch.autocast(device_type=device.type, dtype=dtype)
    return nullcontext()


def cuda_memory_stats(device):
    if device.type != "cuda":
        return {}
    return {
        "allocated_mb": torch.cuda.memory_allocated(device) / (1024 ** 2),
        "reserved_mb": torch.cuda.memory_reserved(device) / (1024 ** 2),
        "peak_allocated_mb": (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        ),
    }


def format_cuda_memory(stats):
    if not stats:
        return "CUDA memory stats unavailable (CPU device)"
    return (
        "allocated={allocated_mb:.0f} MiB, reserved={reserved_mb:.0f} MiB, "
        "peak_allocated={peak_allocated_mb:.0f} MiB"
    ).format(**stats)
