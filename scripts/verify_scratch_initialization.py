import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import load_config
from models import P2PNet


def main():
    config = load_config()
    model = P2PNet(
        row=int(config["model"]["row"]),
        line=int(config["model"]["line"]),
        num_classes=int(config["model"]["num_classes"]),
        pretrained=False,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(f"Trainable parameters: {trainable_count:,}")
    print(f"Total parameters: {parameter_count:,}")
    print("Pretrained: false")
    print("External state dict loading in build_p2pnet: none")

if __name__ == "__main__":
    main()
