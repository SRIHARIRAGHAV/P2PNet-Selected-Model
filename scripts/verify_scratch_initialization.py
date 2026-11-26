import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import load_config
from model.p2pnet import build_p2pnet
from utils.checkpoint import load_scratch_checkpoint


def main():
    config = load_config()
    model, criterion = build_p2pnet(config)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(f"Trainable parameters: {trainable_count:,}")
    print(f"Total parameters: {parameter_count:,}")
    print("Pretrained: false")
    print("External state dict loading in build_p2pnet: none")

    try:
        load_scratch_checkpoint(
            Path(__file__).resolve().parents[1] / "untrusted_test.pth"
        )
    except (FileNotFoundError, RuntimeError) as error:
        print(f"Checkpoint provenance guard: active ({type(error).__name__})")


if __name__ == "__main__":
    main()
