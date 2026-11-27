import sys
from copy import deepcopy
from pathlib import Path
import random

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import load_config
from utils.checkpoint import load_scratch_checkpoint, save_checkpoint
from train import (
    _validate_resume_config,
    build_model_and_criterion,
    select_device,
    train_one_epoch,
    validate_config,
)
from utils.reproducibility import (
    capture_rng_state,
    restore_rng_state,
    set_seed,
)


def test_gradient_and_untrusted_checkpoint_guard(tmp_path):
    config = load_config()
    expected_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = select_device(config)
    assert device.type == expected_device
    model, criterion = build_model_and_criterion(config, device)
    images = torch.randn(1, 3, 128, 128, device=device)
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


def test_training_rejects_pretrained_and_checks_resume_compatibility():
    config = load_config()
    config["model"]["pretrained"] = True
    with pytest.raises(RuntimeError, match="Pretrained"):
        validate_config(config)

    config = load_config()
    validate_config(config)
    _validate_resume_config(deepcopy(config), config)
    incompatible = deepcopy(config)
    incompatible["model"]["row"] += 1
    with pytest.raises(ValueError, match="model.row"):
        _validate_resume_config(incompatible, config)


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
        rng_state=capture_rng_state(),
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
    assert checkpoint["global_step"] == 0
    assert checkpoint["rng_state"]["torch_cpu"].dtype == torch.uint8


def test_rng_checkpoint_restores_python_numpy_and_torch_generators():
    set_seed(19)
    state = capture_rng_state()
    expected = (
        random.random(),
        np.random.random(),
        torch.rand(1),
    )
    set_seed(73)
    restore_rng_state(state)
    assert random.random() == expected[0]
    assert np.random.random() == expected[1]
    assert torch.equal(torch.rand(1), expected[2])


def test_gradient_accumulation_steps_only_on_window_boundaries():
    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, images):
            values = self.weight.expand(images.shape[0], 1, 2)
            return {"pred_logits": values, "pred_points": values[..., :1].expand(-1, -1, 2)}

    class ToyCriterion(torch.nn.Module):
        def forward(self, outputs, targets):
            return {
                "loss_ce": outputs["pred_logits"].mean().square(),
                "loss_point": outputs["pred_points"].mean().square(),
            }

    model = ToyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    config = {
        "training": {
            "gradient_accumulation_steps": 2,
            "gradient_clip_norm": 0.0,
        },
        "gpu": {
            "mixed_precision": False,
            "amp_dtype": "float16",
            "pinned_memory": False,
            "non_blocking": False,
        },
    }
    batch = (torch.ones(1, 3, 32, 32), [{"point": torch.empty(0, 2)}])
    losses = train_one_epoch(
        model,
        ToyCriterion(),
        [batch, batch, batch],
        optimizer,
        None,
        torch.device("cpu"),
        config,
    )
    assert losses["optimizer_steps"] == 2
