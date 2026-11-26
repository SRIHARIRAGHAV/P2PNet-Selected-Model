import sys
from pathlib import Path
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import load_config
from model.p2pnet import build_p2pnet
from utils.checkpoint import load_scratch_checkpoint, save_checkpoint
from train import select_device, validate_config


def test_gradient_and_untrusted_checkpoint_guard(tmp_path):
    config = load_config()
    model, criterion = build_p2pnet(config)
    expected_device = "cuda" if torch.cuda.is_available() else "cpu"
    assert select_device(config).type == expected_device
    images = torch.randn(1, 3, 128, 128)
    targets = [{"point": torch.tensor([[64.0, 64.0]]), "labels": torch.ones(1, dtype=torch.long)}]
    outputs = model(images)
    losses = criterion(outputs, targets)
    loss = losses["loss_ce"] + losses["loss_point"]
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())

    fake = tmp_path / "fake.pth"
    torch.save({"model": model.state_dict()}, fake)
    try:
        load_scratch_checkpoint(fake)
        raise AssertionError("untrusted checkpoint was accepted")
    except RuntimeError:
        pass


def test_training_rejects_pretrained_and_checkpoint_resume():
    config = load_config()
    config["model"]["pretrained"] = True
    with pytest.raises(RuntimeError, match="Pretrained"):
        validate_config(config)

    config = load_config()
    config["training"]["resume"] = True
    with pytest.raises(RuntimeError, match="resume"):
        validate_config(config)

    config = load_config()
    config["training"]["resume_checkpoint"] = "existing.pth"
    with pytest.raises(RuntimeError, match="Checkpoint loading"):
        validate_config(config)


def test_requested_cuda_fails_when_unavailable(monkeypatch):
    config = load_config()
    config["gpu"]["device"] = "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA training was requested"):
        select_device(config)


def test_checkpoint_round_trip_restores_training_state_on_cpu(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    loss = model(torch.ones(1, 2)).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()

    checkpoint_path = tmp_path / "scratch.pth"
    config = {
        "gpu": {"device": "cpu"},
        "model": {"architecture": "P2PNet"},
        "dataset": {"name": "JHU-CROWD++"},
        "training": {"mode": "scratch"},
        "project": {"random_seed": 42},
    }
    validation_metrics = {"MAE": 1.5, "RMSE": 2.0}
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        scheduler,
        None,
        epoch=3,
        best_metric=1.5,
        config=config,
        validation_metrics=validation_metrics,
    )
    checkpoint = load_scratch_checkpoint(checkpoint_path)

    assert checkpoint["epoch"] == 3
    assert checkpoint["config"]["gpu"]["device"] == "cpu"
    assert checkpoint["scheduler"] == scheduler.state_dict()
    assert checkpoint["optimizer"]["state"]
    assert all(tensor.device.type == "cpu" for tensor in checkpoint["model"].values())
    assert checkpoint["architecture"] == "P2PNet"
    assert checkpoint["dataset"] == "JHU-CROWD++"
    assert checkpoint["training_mode"] == "scratch"
    assert checkpoint["random_seed"] == 42
    assert checkpoint["validation_metrics"] == validation_metrics
