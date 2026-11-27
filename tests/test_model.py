import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import load_config
from models import HungarianMatcherCrowd, P2PNet, SetCriterionCrowd


def test_scratch_model_forward():
    config = load_config()
    model = P2PNet(
        row=int(config["model"]["row"]),
        line=int(config["model"]["line"]),
        num_classes=int(config["model"]["num_classes"]),
        pretrained=False,
    )
    criterion = SetCriterionCrowd(
        num_classes=int(config["model"]["num_classes"]),
        matcher=HungarianMatcherCrowd(
            cost_class=float(config["model"]["set_cost_class"]),
            cost_point=float(config["model"]["set_cost_point"]),
        ),
        eos_coef=float(config["model"]["eos_coef"]),
        point_loss_coef=float(config["model"]["point_loss_coef"]),
    )
    assert config["model"]["pretrained"] is False
    images = torch.randn(1, 3, 128, 128)
    outputs = model(images)
    assert outputs["pred_logits"].shape[:2] == (1, 1024)
    assert outputs["pred_points"].shape[:2] == (1, 1024)
    targets = [{"point": torch.tensor([[64.0, 64.0]]), "labels": torch.ones(1, dtype=torch.long)}]
    losses = criterion(outputs, targets)
    assert all(torch.isfinite(loss) for loss in losses.values())
    assert all(loss.dtype == torch.float32 for loss in losses.values())


def test_hungarian_matcher_builds_costs_on_cpu_and_keeps_all_targets():
    matcher = HungarianMatcherCrowd(cost_class=0.0, cost_point=1.0)
    outputs = {
        "pred_logits": torch.zeros(1, 3, 2),
        "pred_points": torch.tensor([[[100.0, 100.0], [1.0, 1.0], [8.0, 8.0]]]),
    }
    targets = [{"point": torch.tensor([[8.0, 8.0], [1.0, 1.0]])}]
    original_cdist = torch.cdist

    def cpu_cdist(source, target, p=2):
        assert source.device.type == "cpu"
        assert target.device.type == "cpu"
        return original_cdist(source, target, p=p)

    from unittest.mock import patch

    with patch("torch.cdist", side_effect=cpu_cdist):
        source_indices, target_indices = matcher(outputs, targets)[0]

    assert source_indices.tolist() == [1, 2]
    assert target_indices.tolist() == [1, 0]
