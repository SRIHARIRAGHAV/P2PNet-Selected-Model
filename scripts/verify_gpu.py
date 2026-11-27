import json
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import load_config
from utils.device import select_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=str(
            Path(__file__).resolve().parents[1] / "logs" / "gpu_verification.json"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config()
    device = select_device(config)
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
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        report["selected_gpu"] = torch.cuda.get_device_name(device)
        report["memory_free_bytes"] = free
        report["memory_total_bytes"] = total
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.overwrite else "x"
    with output.open(mode, encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
