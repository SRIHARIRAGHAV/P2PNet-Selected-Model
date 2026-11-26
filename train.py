import argparse
from contextlib import nullcontext
from copy import deepcopy
import math
from pathlib import Path
import tempfile
import time

import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from config import load_config
from dataset.dataset_loader import JHUCrowdDataset, collate_train, collate_validation
from models import HungarianMatcherCrowd, P2PNet, SetCriterionCrowd
from utils.checkpoint import save_checkpoint
from utils.logging_utils import TrainingLogger, process_ram_usage_mb
from utils.metrics import crowd_metrics
from utils.reproducibility import make_generator, seed_worker, set_seed


ROOT = Path(__file__).resolve().parent


def select_device(config):
    requested = str(config["gpu"]["device"]).lower()
    mixed_precision = bool(config["gpu"]["mixed_precision"])
    if requested == "cpu":
        if mixed_precision:
            raise RuntimeError("Mixed precision requires a CUDA device.")
        return torch.device("cpu")

    cuda_available = torch.cuda.is_available()
    if requested == "cuda" and not cuda_available:
        raise RuntimeError(
            "CUDA training was requested with gpu.device=cuda, "
            "but CUDA is unavailable in this PyTorch environment."
        )
    if requested == "auto" and not cuda_available:
        if mixed_precision:
            raise RuntimeError(
                "Mixed precision was requested, but CUDA is unavailable."
            )
        return torch.device("cpu")
    if requested not in {"auto", "cuda"}:
        raise RuntimeError("gpu.device must be 'auto', 'cpu', or 'cuda'.")

    index = int(config["gpu"]["gpu_device_index"])
    if index < 0 or index >= torch.cuda.device_count():
        raise RuntimeError(
            f"Requested GPU index {index}, but only "
            f"{torch.cuda.device_count()} CUDA device(s) are available."
        )
    torch.cuda.set_device(index)
    device = torch.device("cuda", index)
    print(f"CUDA device: {torch.cuda.get_device_name(index)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA build: {torch.version.cuda}")
    return device


def configure_cpu_threads(config, device):
    if device.type == "cpu":
        torch.set_num_threads(int(config["training"]["cpu_num_threads"]))
        torch.set_num_interop_threads(
            int(config["training"]["cpu_interop_threads"])
        )


def validate_config(config):
    if config["model"].get("pretrained") is not False:
        raise RuntimeError("Pretrained model initialization is forbidden.")
    if config["training"].get("mode") != "scratch":
        raise RuntimeError("training.mode must be 'scratch'.")
    if config["training"].get("resume", False):
        raise RuntimeError("Checkpoint resume is forbidden for scratch training.")
    if config["training"].get("resume_checkpoint", ""):
        raise RuntimeError("Checkpoint loading is forbidden for training.")
    if int(config["training"]["epochs"]) < 1:
        raise ValueError("training.epochs must be positive.")
    if int(config["training"]["validation_interval"]) < 1:
        raise ValueError("training.validation_interval must be positive.")
    if int(config["checkpoints"]["save_interval"]) < 1:
        raise ValueError("checkpoints.save_interval must be positive.")
    if float(config["training"]["gradient_clip_norm"]) < 0:
        raise ValueError("training.gradient_clip_norm cannot be negative.")
    if float(config["training"]["learning_rate"]) <= 0:
        raise ValueError("training.learning_rate must be positive.")
    if float(config["training"]["backbone_learning_rate"]) <= 0:
        raise ValueError("training.backbone_learning_rate must be positive.")
    if float(config["training"]["weight_decay"]) < 0:
        raise ValueError("training.weight_decay cannot be negative.")
    if str(config["training"]["optimizer"]).lower() not in {"adam", "adamw"}:
        raise ValueError("training.optimizer must be Adam or AdamW.")
    if str(config["training"]["lr_scheduler"]).lower() != "steplr":
        raise ValueError("training.lr_scheduler must be StepLR.")
    if str(config["checkpoints"]["best_model_metric"]).upper() not in {"MAE", "RMSE"}:
        raise ValueError("checkpoints.best_model_metric must be MAE or RMSE.")
    if int(config["dataset"]["num_workers"]) < 0:
        raise ValueError("dataset.num_workers cannot be negative.")
    if str(config["gpu"]["device"]).lower() not in {"auto", "cpu", "cuda"}:
        raise ValueError("gpu.device must be 'auto', 'cpu', or 'cuda'.")
    if str(config["gpu"]["amp_dtype"]).lower() not in {"float16", "bfloat16"}:
        raise ValueError("gpu.amp_dtype must be 'float16' or 'bfloat16'.")


def build_model_and_criterion(config, device):
    model_config = config["model"]
    if model_config.get("pretrained") is not False:
        raise RuntimeError("P2PNet pretrained initialization is forbidden.")
    model = P2PNet(
        row=int(model_config["row"]),
        line=int(model_config["line"]),
        num_classes=int(model_config["num_classes"]),
        pretrained=False,
    )
    matcher = HungarianMatcherCrowd(
        cost_class=float(model_config["set_cost_class"]),
        cost_point=float(model_config["set_cost_point"]),
    )
    criterion = SetCriterionCrowd(
        num_classes=int(model_config["num_classes"]),
        matcher=matcher,
        eos_coef=float(model_config["eos_coef"]),
        point_loss_coef=float(model_config["point_loss_coef"]),
    )
    model.to(device)
    criterion.to(device)
    if any(parameter.device != device for parameter in model.parameters()):
        raise RuntimeError(f"Model parameters were not initialized on {device}.")
    return model, criterion


def create_dataloaders(config, device, smoke_test=False):
    dataset_config = config
    if smoke_test:
        dataset_config = deepcopy(config)
        dataset_config["dataset"]["augmentation"]["patches_per_image"] = 1

    train_dataset = JHUCrowdDataset(dataset_config, "train")
    validation_dataset = JHUCrowdDataset(
        dataset_config,
        "validation",
        smoke_test=smoke_test,
    )
    if not len(train_dataset) or not len(validation_dataset):
        raise RuntimeError("Training or validation dataset is empty.")

    training = config["training"]
    if smoke_test:
        train_sample_limit = (
            int(training["smoke_test_train_batches"])
            * int(training["batch_size"])
        )
        validation_sample_limit = int(training["smoke_test_validation_batches"])
        train_dataset = Subset(
            train_dataset,
            range(min(train_sample_limit, len(train_dataset))),
        )
        validation_dataset = Subset(
            validation_dataset,
            range(min(validation_sample_limit, len(validation_dataset))),
        )

    gpu = config["gpu"]
    use_pinned_memory = (
        device.type == "cuda" and bool(gpu["pinned_memory"])
    )
    workers = int(config["dataset"]["num_workers"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        num_workers=workers,
        pin_memory=use_pinned_memory,
        worker_init_fn=seed_worker,
        generator=make_generator(int(config["project"]["random_seed"])),
        collate_fn=collate_train,
        drop_last=False,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=use_pinned_memory,
        worker_init_fn=seed_worker,
        generator=make_generator(int(config["project"]["random_seed"]) + 1),
        collate_fn=collate_validation,
        drop_last=False,
    )
    return train_loader, validation_loader


def build_optimizer(config, model):
    training = config["training"]
    backbone_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith("backbone.")
    ]
    other_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("backbone.")
    ]
    if not backbone_parameters or not other_parameters:
        raise RuntimeError("Expected trainable backbone and prediction-head parameters.")

    parameter_groups = [
        {
            "params": other_parameters,
            "lr": float(training["learning_rate"]),
        },
        {
            "params": backbone_parameters,
            "lr": float(training["backbone_learning_rate"]),
        },
    ]
    name = str(training["optimizer"]).lower()
    optimizers = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }
    if name not in optimizers:
        raise ValueError(f"Unsupported optimizer in YAML: {training['optimizer']}")
    return optimizers[name](
        parameter_groups,
        weight_decay=float(training["weight_decay"]),
    )


def build_scheduler(config, optimizer):
    training = config["training"]
    name = str(training["lr_scheduler"]).lower()
    if name != "steplr":
        raise ValueError(
            f"Unsupported learning-rate scheduler in YAML: "
            f"{training['lr_scheduler']}"
        )
    return torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(training["lr_drop"]),
        gamma=float(training["lr_gamma"]),
    )


def _amp_settings(config, device):
    enabled = device.type == "cuda" and bool(config["gpu"]["mixed_precision"])
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[str(config["gpu"]["amp_dtype"]).lower()]
    if enabled and dtype is torch.bfloat16:
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "bfloat16 mixed precision was requested, but the selected "
                "CUDA device does not support it."
            )
    return enabled, dtype


def _autocast_context(enabled, dtype):
    if enabled:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def train_one_epoch(
    model,
    criterion,
    loader,
    optimizer,
    scaler,
    device,
    config,
    max_batches=None,
):
    model.train()
    criterion.train()
    amp_enabled, amp_dtype = _amp_settings(config, device)
    loss_totals = {"total": 0.0, "classification": 0.0, "localization": 0.0}
    batch_count = 0

    for batch_index, (images, targets) in enumerate(loader):
        non_blocking = (
            device.type == "cuda" and bool(config["gpu"]["non_blocking"])
        )
        images = images.to(device, non_blocking=non_blocking)
        targets = [
            {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in target.items()
            }
            for target in targets
        ]
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(amp_enabled, amp_dtype):
            outputs = model(images)
            losses = criterion(outputs, targets)
            total_loss = sum(losses.values())

        loss_values = {
            "total": float(total_loss.detach().item()),
            "classification": float(losses["loss_ce"].detach().item()),
            "localization": float(losses["loss_point"].detach().item()),
        }
        if not all(math.isfinite(value) for value in loss_values.values()):
            raise RuntimeError(f"Non-finite training loss: {loss_values}")

        if scaler is not None and scaler.is_enabled():
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            if config["training"]["gradient_clip_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(config["training"]["gradient_clip_norm"]),
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if config["training"]["gradient_clip_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(config["training"]["gradient_clip_norm"]),
                )
            optimizer.step()

        for key, value in loss_values.items():
            loss_totals[key] += value
        batch_count += 1
        if max_batches is not None and batch_count >= max_batches:
            break

    if batch_count == 0:
        raise RuntimeError("Training loader produced no batches.")
    return {key: value / batch_count for key, value in loss_totals.items()}


@torch.inference_mode()
def validate(model, criterion, loader, device, config, max_batches=None):
    model.eval()
    criterion.eval()
    amp_enabled, amp_dtype = _amp_settings(config, device)
    actual_counts = []
    predicted_counts = []
    loss_totals = {"total": 0.0, "classification": 0.0, "localization": 0.0}
    batch_count = 0

    for images, targets in loader:
        images = images.to(
            device,
            non_blocking=(
                device.type == "cuda" and bool(config["gpu"]["non_blocking"])
            ),
        )
        device_targets = [
            {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in target.items()
            }
            for target in targets
        ]
        with _autocast_context(amp_enabled, amp_dtype):
            outputs = model(images)
            losses = criterion(outputs, device_targets)
        values = {
            "total": sum(losses.values()),
            "classification": losses["loss_ce"],
            "localization": losses["loss_point"],
        }
        for key, value in values.items():
            loss_totals[key] += float(value.item())

        scores = outputs["pred_logits"].softmax(-1)[..., 1]
        threshold = float(
            config["evaluation"]["prediction_confidence_threshold"]
        )
        for batch_index, target in enumerate(targets):
            predicted_counts.append(int((scores[batch_index] > threshold).sum().item()))
            actual_counts.append(len(target["point"]))
        batch_count += 1
        if max_batches is not None and batch_count >= max_batches:
            break

    if batch_count == 0:
        raise RuntimeError("Validation loader produced no batches.")
    metrics = crowd_metrics(actual_counts, predicted_counts)
    return {
        "loss_total": loss_totals["total"] / batch_count,
        "loss_classification": loss_totals["classification"] / batch_count,
        "loss_localization": loss_totals["localization"] / batch_count,
        **metrics,
    }


def _resolve_output_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _assert_fresh_outputs(
    checkpoint_dir,
    log_dir,
    tensorboard_dir,
    best_metric_name,
):
    protected = [
        checkpoint_dir / "latest.pth",
        checkpoint_dir / "best_{}.pth".format(best_metric_name.lower()),
        log_dir / "history.csv",
    ]
    if tensorboard_dir.exists():
        protected.extend(tensorboard_dir.glob("events.out.tfevents.*"))
    existing = [path for path in protected if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing training outputs: "
            + ", ".join(str(path) for path in existing)
        )


def run_training(config, device, output_root=None, smoke_test=False):
    set_seed(int(config["project"]["random_seed"]))
    configure_cpu_threads(config, device)
    model, criterion = build_model_and_criterion(config, device)
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer)

    amp_enabled, amp_dtype = _amp_settings(config, device)
    scaler = None
    if device.type == "cuda":
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=amp_enabled and amp_dtype is torch.float16,
        )

    train_loader, validation_loader = create_dataloaders(
        config,
        device,
        smoke_test=smoke_test,
    )
    if output_root is None:
        checkpoint_dir = _resolve_output_path(
            config["checkpoints"]["checkpoint_dir"]
        )
        log_dir = _resolve_output_path(config["training"]["log_dir"])
        tensorboard_dir = _resolve_output_path(
            config["training"]["tensorboard_log_dir"]
        )
        best_metric_name = str(
            config["checkpoints"]["best_model_metric"]
        ).upper()
        _assert_fresh_outputs(
            checkpoint_dir,
            log_dir,
            tensorboard_dir,
            best_metric_name,
        )
    else:
        output_root = Path(output_root)
        checkpoint_dir = output_root / "checkpoints"
        log_dir = output_root / "logs"
        tensorboard_dir = log_dir / "tensorboard"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    logger = TrainingLogger(log_dir)
    threshold = float(
        config["evaluation"]["prediction_confidence_threshold"]
    )
    epochs = 1 if smoke_test else int(config["training"]["epochs"])
    save_interval = int(config["checkpoints"]["save_interval"])
    validation_interval = int(config["training"]["validation_interval"])
    best_metric_name = str(
        config["checkpoints"]["best_model_metric"]
    ).upper()
    best_checkpoint_path = checkpoint_dir / "best_{}.pth".format(
        best_metric_name.lower()
    )
    best_metric_value = float("inf")
    latest_validation_metrics = {}
    start_time = time.perf_counter()
    smoke_batches = (
        int(config["training"]["smoke_test_train_batches"])
        if smoke_test else None
    )
    smoke_validation_batches = (
        int(config["training"]["smoke_test_validation_batches"])
        if smoke_test else None
    )

    try:
        for epoch in range(epochs):
            epoch_started = time.perf_counter()
            train_losses = train_one_epoch(
                model,
                criterion,
                train_loader,
                optimizer,
                scaler,
                device,
                config,
                max_batches=smoke_batches,
            )
            scheduler.step()

            should_validate = (
                (epoch + 1) % validation_interval == 0
                or epoch == epochs - 1
            )
            if should_validate:
                latest_validation_metrics = validate(
                    model,
                    criterion,
                    validation_loader,
                    device,
                    config,
                    max_batches=smoke_validation_batches,
                )
                current_best_metric = latest_validation_metrics[best_metric_name]
                if current_best_metric < best_metric_value:
                    best_metric_value = current_best_metric
                    save_checkpoint(
                        best_checkpoint_path,
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        epoch + 1,
                        best_metric_value,
                        config,
                        validation_metrics=latest_validation_metrics,
                    )

            validation_metrics = latest_validation_metrics
            row = {
                "epoch": epoch + 1,
                "training_loss": train_losses["total"],
                "classification_loss": train_losses["classification"],
                "localization_loss": train_losses["localization"],
                "validation_loss": validation_metrics.get("loss_total", ""),
                "validation_mae": validation_metrics.get("MAE", ""),
                "validation_rmse": validation_metrics.get("RMSE", ""),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "epoch_duration_seconds": round(
                    time.perf_counter() - epoch_started,
                    2,
                ),
                "ram_usage_mb": (
                    process_ram_usage_mb()
                    if config["gpu"]["memory_monitoring"] else ""
                ) or "",
            }
            logger.log(row)

            for key, value in train_losses.items():
                writer.add_scalar("train/loss_{}".format(key), value, epoch + 1)
            for key in ("loss_total", "loss_classification", "loss_localization"):
                if key in validation_metrics:
                    writer.add_scalar(
                        "validation/{}".format(key),
                        validation_metrics[key],
                        epoch + 1,
                    )
            for key in ("MAE", "RMSE", "MAPE"):
                if key in validation_metrics:
                    writer.add_scalar(
                        "validation/{}".format(key),
                        validation_metrics[key],
                        epoch + 1,
                    )
            writer.add_scalar(
                "train/learning_rate",
                optimizer.param_groups[0]["lr"],
                epoch + 1,
            )
            writer.flush()

            if (epoch + 1) % save_interval == 0 or epoch == epochs - 1:
                save_checkpoint(
                    checkpoint_dir / "latest.pth",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch + 1,
                    best_metric_value,
                    config,
                    validation_metrics=validation_metrics,
                    overwrite=True,
                )

            print(
                "Epoch {}/{}: loss={:.6f} cls={:.6f} point={:.6f} "
                "val_MAE={} val_RMSE={} duration={:.2f}s".format(
                    epoch + 1,
                    epochs,
                    train_losses["total"],
                    train_losses["classification"],
                    train_losses["localization"],
                    validation_metrics.get("MAE", ""),
                    validation_metrics.get("RMSE", ""),
                    row["epoch_duration_seconds"],
                )
            )
    finally:
        writer.close()

    if not (checkpoint_dir / "latest.pth").is_file():
        raise RuntimeError("Training completed without saving latest.pth.")
    if not best_checkpoint_path.is_file():
        raise RuntimeError(
            "Training completed without saving {}.".format(best_checkpoint_path.name)
        )
    if not (log_dir / "history.csv").is_file():
        raise RuntimeError("Training completed without writing history.csv.")
    if not list(tensorboard_dir.glob("events.out.tfevents.*")):
        raise RuntimeError("Training completed without TensorBoard event files.")
    return {
        "latest_checkpoint": checkpoint_dir / "latest.pth",
        "best_checkpoint": best_checkpoint_path,
        "history": log_dir / "history.csv",
        "tensorboard_dir": tensorboard_dir,
        "best_metric_name": best_metric_name,
        "best_metric_value": best_metric_value,
        "elapsed_seconds": time.perf_counter() - start_time,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "p2pnet_scratch.yaml"),
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="requested",
        default=None,
        help="Rejected: checkpoint resume is disabled for scratch training.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Rejected: training never loads a checkpoint.",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Rejected: pretrained initialization is disabled.",
    )
    args = parser.parse_args()
    if args.resume is not None or args.checkpoint is not None:
        raise RuntimeError("Checkpoint loading/resume is forbidden for training.")
    if args.pretrained:
        raise RuntimeError("Pretrained model initialization is forbidden.")

    config = load_config(args.config)
    validate_config(config)
    device = select_device(config)
    if device.type == "cpu":
        print("Training device: CPU")
    if args.smoke_test:
        with tempfile.TemporaryDirectory(prefix="p2pnet-smoke-") as temp_dir:
            result = run_training(
                config,
                device,
                output_root=temp_dir,
                smoke_test=True,
            )
            print(
                "Smoke test passed: latest/best checkpoints, CSV log, and "
                "TensorBoard events were created in a temporary directory."
            )
            print(
                "Best validation {}: {:.6f}; elapsed: {:.2f}s".format(
                    result["best_metric_name"],
                    result["best_metric_value"],
                    result["elapsed_seconds"],
                )
            )
    else:
        result = run_training(config, device)
        print(
            "Training complete. Best {}: {:.6f}".format(
                result["best_metric_name"],
                result["best_metric_value"],
            )
        )


if __name__ == "__main__":
    main()
