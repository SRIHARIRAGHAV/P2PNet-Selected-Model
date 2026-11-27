import os
from pathlib import Path
import tempfile

import torch


PROVENANCE = "P2PNet-Scratch-Selected_Model"


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch,
                    best_metric, config, validation_metrics=None,
                    overwrite=False, global_step=0, rng_state=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite checkpoint: {path}")
    metadata = {
        "architecture": config.get("model", {}).get("architecture", "P2PNet"),
        "dataset": config.get("dataset", {}).get("name", "JHU-CROWD++"),
        "training_mode": config.get("training", {}).get("mode", "scratch"),
        "random_seed": config.get("project", {}).get("random_seed"),
        "epoch": epoch,
        "global_step": global_step,
        "rng_state": rng_state,
        "validation_metrics": dict(validation_metrics or {}),
    }
    payload = {
        "provenance": PROVENANCE,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        **metadata,
        "metadata": metadata,
        "epoch": epoch,
        "best_metric": best_metric,
        "config": config,
        "pretrained": False,
    }
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite checkpoint: {path}")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def load_scratch_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("provenance") != PROVENANCE:
        raise RuntimeError(
            "Rejected checkpoint: it is not a provenance-verified "
            "P2PNet-Scratch-Selected_Model checkpoint."
        )
    if payload.get("pretrained") is not False:
        raise RuntimeError("Rejected checkpoint: pretrained metadata is set.")
    required = {"model", "optimizer", "scheduler", "epoch", "config"}
    missing = required.difference(payload)
    if missing:
        raise RuntimeError(
            f"Rejected checkpoint: missing required keys: {sorted(missing)}"
        )
    return payload
