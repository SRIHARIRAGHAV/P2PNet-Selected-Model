import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import load_config


def main():
    config = load_config()
    report = {
        "torch_installed": True,
        "torch_version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "gpu_names": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
        "status": "ok" if torch.cuda.is_available() else "cuda_unavailable",
    }
    if torch.cuda.is_available():
        index = config["gpu"]["gpu_device_index"]
        free, total = torch.cuda.mem_get_info(index)
        report["selected_gpu"] = report["gpu_names"][index]
        report["memory_free_bytes"] = free
        report["memory_total_bytes"] = total
    root = Path(__file__).resolve().parents[1]
    output = root / "logs" / "gpu_verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
