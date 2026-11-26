import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import load_config
from model.p2pnet import build_p2pnet


def test_scratch_model_forward():
    config = load_config()
    model, criterion = build_p2pnet(config)
    assert config["model"]["pretrained"] is False
    images = torch.randn(1, 3, 128, 128)
    outputs = model(images)
    assert outputs["pred_logits"].shape[:2] == (1, 1024)
    assert outputs["pred_points"].shape[:2] == (1, 1024)
    targets = [{"point": torch.tensor([[64.0, 64.0]]), "labels": torch.ones(1, dtype=torch.long)}]
    losses = criterion(outputs, targets)
    assert all(torch.isfinite(loss) for loss in losses.values())
