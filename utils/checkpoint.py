from pathlib import Path
import torch


PROVENANCE = "P2PNet-Scratch-Selected_Model"


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch,
                    best_metric, config, validation_metrics=None,
                    overwrite=False):
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
    torch.save(payload, path)


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
