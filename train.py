import argparse
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
from models import build_p2pnet
from utils.checkpoint import load_scratch_checkpoint, save_checkpoint
from utils.device import (
    amp_settings,
    autocast_context,
    cuda_memory_stats,
    format_cuda_memory,
    select_device,
)
from utils.logging_utils import TrainingLogger, process_ram_usage_mb
from utils.metrics import crowd_metrics
from utils.reproducibility import (
    capture_rng_state,
    make_generator,
    restore_rng_state,
    seed_worker,
    set_seed,
)


ROOT = Path(__file__).resolve().parent


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
    if int(config["training"]["epochs"]) < 1:
        raise ValueError("training.epochs must be positive.")
    if int(config["training"]["validation_interval"]) < 1:
        raise ValueError("training.validation_interval must be positive.")
    if int(config["training"]["gradient_accumulation_steps"]) < 1:
        raise ValueError(
            "training.gradient_accumulation_steps must be positive."
        )
    if int(config["checkpoints"]["save_interval"]) < 1:
        raise ValueError("checkpoints.save_interval must be positive.")
    if int(config["training"]["smoke_test_train_batches"]) < 1:
        raise ValueError("training.smoke_test_train_batches must be positive.")
    if int(config["training"]["smoke_test_validation_batches"]) < 1:
        raise ValueError(
            "training.smoke_test_validation_batches must be positive."
        )
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
    if int(config["dataset"]["prefetch_factor"]) < 1:
        raise ValueError("dataset.prefetch_factor must be positive.")
    if int(config["evaluation"]["batch_size"]) != 1:
        raise ValueError("evaluation.batch_size must be 1.")
    if str(config["gpu"]["device"]).lower() not in {"auto", "cpu", "cuda"}:
        raise ValueError("gpu.device must be 'auto', 'cpu', or 'cuda'.")
    if str(config["gpu"]["amp_dtype"]).lower() not in {"float16", "bfloat16"}:
        raise ValueError("gpu.amp_dtype must be 'float16' or 'bfloat16'.")
    if not 0.0 <= float(
        config["evaluation"]["prediction_confidence_threshold"]
    ) <= 1.0:
        raise ValueError(
            "evaluation.prediction_confidence_threshold must be in [0, 1]."
        )
    height, width = map(int, config["model"]["input_resolution"])
    if height < 32 or width < 32 or height % 8 or width % 8:
        raise ValueError(
            "model.input_resolution dimensions must be at least 32 and "
            "divisible by 8."
        )
    if int(config["dataset"]["augmentation"]["patch_size"]) != height:
        raise ValueError(
            "dataset.augmentation.patch_size must match the configured "
            "square model.input_resolution."
        )
    if height != width:
        raise ValueError("model.input_resolution must be square for patch training.")


def build_model_and_criterion(config, device):
    model, criterion = build_p2pnet(config)
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
    for dataset in (train_dataset, validation_dataset):
        if dataset.empty_annotation_count:
            print(
                f"Dataset {dataset.split}: "
                f"{dataset.empty_annotation_count} empty annotation file(s); "
                "these images remain in the dataset."
            )
        if dataset.clamped_annotation_count:
            print(
                f"Dataset {dataset.split}: retained and clamped "
                f"{dataset.clamped_annotation_count} out-of-bounds point(s)."
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
    loader_options = (
        {"prefetch_factor": int(config["dataset"]["prefetch_factor"])}
        if workers > 0 else {}
    )
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
        **loader_options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(config["evaluation"]["batch_size"]),
        shuffle=False,
        num_workers=workers,
        pin_memory=use_pinned_memory,
        worker_init_fn=seed_worker,
        generator=make_generator(int(config["project"]["random_seed"]) + 1),
        collate_fn=collate_validation,
        drop_last=False,
        **loader_options,
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
    amp_enabled, amp_dtype = amp_settings(config, device)
    loss_totals = {"total": 0.0, "classification": 0.0, "localization": 0.0}
    batch_count = 0
    optimizer_steps = 0
    accumulation_steps = int(config["training"]["gradient_accumulation_steps"])
    batch_limit = len(loader)
    if max_batches is not None:
        batch_limit = min(batch_limit, max_batches)
    if batch_limit == 0:
        raise RuntimeError("Training loader produced no batches.")
    optimizer.zero_grad(set_to_none=True)

    for batch_index, (images, targets) in enumerate(loader):
        if batch_index >= batch_limit:
            break
        non_blocking = (
            device.type == "cuda"
            and bool(config["gpu"]["pinned_memory"])
            and bool(config["gpu"]["non_blocking"])
        )
        try:
            images = images.to(device, non_blocking=non_blocking)
            with autocast_context(amp_enabled, amp_dtype, device):
                outputs = model(images)
                if (
                    not torch.isfinite(outputs["pred_logits"]).all()
                    or not torch.isfinite(outputs["pred_points"]).all()
                ):
                    raise RuntimeError(
                        f"Model produced NaN or Inf values in batch "
                        f"{batch_index + 1}."
                    )
                losses = criterion(outputs, targets)
                total_loss = sum(losses.values())
        except torch.cuda.OutOfMemoryError as error:
            stats = cuda_memory_stats(device)
            raise RuntimeError(
                f"CUDA out of memory during training batch {batch_index + 1} "
                f"with input shape {tuple(images.shape)}; "
                f"{format_cuda_memory(stats)}. Reduce the configured "
                "training batch size or crop/patch resolution."
            ) from error

        loss_values = {
            "total": float(total_loss.detach().item()),
            "classification": float(losses["loss_ce"].detach().item()),
            "localization": float(losses["loss_point"].detach().item()),
        }
        if not all(math.isfinite(value) for value in loss_values.values()):
            raise RuntimeError(f"Non-finite training loss: {loss_values}")

        window_start = (batch_index // accumulation_steps) * accumulation_steps
        window_size = min(accumulation_steps, batch_limit - window_start)
        scaled_loss = total_loss / window_size
        try:
            if scaler is not None and scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            should_step = (
                (batch_index + 1) % accumulation_steps == 0
                or batch_index + 1 == batch_limit
            )
            if should_step:
                if scaler is not None and scaler.is_enabled():
                    scaler.unscale_(optimizer)
                if config["training"]["gradient_clip_norm"] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        float(config["training"]["gradient_clip_norm"]),
                    )
                if scaler is not None and scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
        except torch.cuda.OutOfMemoryError as error:
            stats = cuda_memory_stats(device)
            raise RuntimeError(
                f"CUDA out of memory during training backward/optimizer "
                f"for batch {batch_index + 1} with input shape "
                f"{tuple(images.shape)}; {format_cuda_memory(stats)}."
            ) from error

        for key, value in loss_values.items():
            loss_totals[key] += value
        batch_count += 1
        del outputs, losses, total_loss, scaled_loss, images

    if batch_count == 0:
        raise RuntimeError("Training loader produced no batches.")
    return {
        **{key: value / batch_count for key, value in loss_totals.items()},
        "optimizer_steps": optimizer_steps,
    }


@torch.inference_mode()
def validate(model, criterion, loader, device, config, max_batches=None):
    model.eval()
    criterion.eval()
    amp_enabled, amp_dtype = amp_settings(config, device)
    actual_counts = []
    predicted_counts = []
    loss_totals = {"total": 0.0, "classification": 0.0, "localization": 0.0}
    batch_count = 0

    for batch_index, (images, targets) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        non_blocking = (
            device.type == "cuda"
            and bool(config["gpu"]["pinned_memory"])
            and bool(config["gpu"]["non_blocking"])
        )
        try:
            images = images.to(device, non_blocking=non_blocking)
            with autocast_context(amp_enabled, amp_dtype, device):
                outputs = model(images)
                if (
                    not torch.isfinite(outputs["pred_logits"]).all()
                    or not torch.isfinite(outputs["pred_points"]).all()
                ):
                    raise RuntimeError(
                        f"Model produced NaN or Inf values in validation "
                        f"batch {batch_index + 1}."
                    )
                losses = criterion(outputs, targets)
        except torch.cuda.OutOfMemoryError as error:
            stats = cuda_memory_stats(device)
            raise RuntimeError(
                f"CUDA out of memory during validation batch {batch_index + 1} "
                f"with input shape {tuple(images.shape)}; "
                f"{format_cuda_memory(stats)}. Reduce "
                "dataset.validation_max_side."
            ) from error
        values = {
            "total": sum(losses.values()),
            "classification": losses["loss_ce"],
            "localization": losses["loss_point"],
        }
        if not all(
            math.isfinite(float(value.detach().item()))
            for value in values.values()
        ):
            raise RuntimeError(
                f"Non-finite validation loss in batch {batch_index + 1}."
            )
        for key, value in values.items():
            loss_totals[key] += float(value.item())

        scores = outputs["pred_logits"].float().softmax(-1)[..., 1]
        threshold = float(
            config["evaluation"]["prediction_confidence_threshold"]
        )
        for batch_index, target in enumerate(targets):
            predicted_counts.append(int((scores[batch_index] > threshold).sum().item()))
            actual_counts.append(len(target["point"]))
        batch_count += 1
        del outputs, losses, values, images, scores

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
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path


def _assert_fresh_outputs(
    checkpoint_dir,
    log_dir,
    tensorboard_dir,
):
    protected = list(checkpoint_dir.glob("*.pth"))
    protected.append(log_dir / "history.csv")
    if tensorboard_dir.exists():
        protected.extend(tensorboard_dir.glob("events.out.tfevents.*"))
    existing = [path for path in protected if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing training outputs: "
            + ", ".join(str(path) for path in existing)
        )


def _validate_resume_config(saved_config, config):
    if not isinstance(saved_config, dict):
        raise ValueError("Checkpoint does not contain a valid training config.")
    comparable = {
        "project": ("random_seed",),
        "dataset": (
            "name",
            "dataset_root",
            "train_manifest",
            "validation_manifest",
            "train_image_dir",
            "train_annotation_dir",
            "validation_image_dir",
            "validation_annotation_dir",
            "validation_max_side",
            "augmentation",
        ),
        "model": tuple(config["model"].keys()),
        "training": (
            "mode",
            "batch_size",
            "optimizer",
            "learning_rate",
            "backbone_learning_rate",
            "weight_decay",
            "lr_scheduler",
            "lr_drop",
            "lr_gamma",
            "gradient_accumulation_steps",
            "gradient_clip_norm",
        ),
        "gpu": ("mixed_precision", "amp_dtype"),
    }
    for section, keys in comparable.items():
        for key in keys:
            if saved_config.get(section, {}).get(key) != config.get(
                section, {}
            ).get(key):
                raise ValueError(
                    f"Cannot resume: checkpoint config mismatch at "
                    f"{section}.{key}."
                )


def run_training(
    config,
    device,
    output_root=None,
    smoke_test=False,
    resume_checkpoint=None,
):
    if smoke_test and resume_checkpoint is not None:
        raise ValueError("A smoke test cannot resume a training checkpoint.")
    if output_root is None:
        checkpoint_dir = _resolve_output_path(
            config["checkpoints"]["checkpoint_dir"]
        )
        log_dir = _resolve_output_path(config["training"]["log_dir"])
        tensorboard_dir = _resolve_output_path(
            config["training"]["tensorboard_log_dir"]
        )
    else:
        output_root = Path(output_root).expanduser().resolve()
        checkpoint_dir = output_root / "checkpoints"
        log_dir = output_root / "logs"
        tensorboard_dir = log_dir / "tensorboard"

    best_metric_name = str(
        config["checkpoints"]["best_model_metric"]
    ).upper()
    best_checkpoint_path = checkpoint_dir / "best_{}.pth".format(
        best_metric_name.lower()
    )
    if resume_checkpoint is None:
        _assert_fresh_outputs(
            checkpoint_dir,
            log_dir,
            tensorboard_dir,
        )
    elif not Path(resume_checkpoint).is_file():
        raise FileNotFoundError(
            f"Resume checkpoint not found: {resume_checkpoint}"
        )
    elif not best_checkpoint_path.is_file():
        raise FileNotFoundError(
            "Resume requires the experiment's existing best checkpoint "
            f"{best_checkpoint_path}; use the original output directory."
        )

    set_seed(int(config["project"]["random_seed"]))
    configure_cpu_threads(config, device)
    model, criterion = build_model_and_criterion(config, device)
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer)

    amp_enabled, amp_dtype = amp_settings(config, device)
    scaler = None
    if device.type == "cuda":
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=amp_enabled and amp_dtype is torch.float16,
        )

    start_epoch = 0
    global_step = 0
    best_metric_value = float("inf")
    latest_validation_metrics = {}
    if resume_checkpoint is not None:
        checkpoint = load_scratch_checkpoint(resume_checkpoint)
        _validate_resume_config(checkpoint.get("config"), config)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler_state = checkpoint.get("scheduler")
        if scheduler_state is None:
            raise ValueError("Cannot resume: checkpoint has no scheduler state.")
        scheduler.load_state_dict(scheduler_state)
        scaler_state = checkpoint.get("scaler")
        if scaler is not None and scaler.is_enabled():
            if scaler_state is None:
                raise ValueError(
                    "Cannot resume AMP training: checkpoint has no scaler state."
                )
            scaler.load_state_dict(scaler_state)
        start_epoch = int(checkpoint["epoch"])
        global_step = int(checkpoint.get("global_step", 0))
        best_metric_value = float(checkpoint.get("best_metric", float("inf")))
        latest_validation_metrics = dict(
            checkpoint.get("validation_metrics", {})
        )
        if start_epoch >= int(config["training"]["epochs"]):
            raise ValueError(
                f"Checkpoint is already at epoch {start_epoch}; "
                "increase training.epochs before resuming."
            )

    train_loader, validation_loader = create_dataloaders(
        config,
        device,
        smoke_test=smoke_test,
    )
    if resume_checkpoint is not None:
        rng_state = checkpoint.get("rng_state")
        if rng_state is None:
            print(
                "Checkpoint has no RNG state; continuing from the configured "
                "seed, not the exact interrupted random sequence."
            )
        else:
            restore_rng_state(rng_state, train_loader, validation_loader)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    logger = TrainingLogger(log_dir, resume=resume_checkpoint is not None)
    epochs = 1 if smoke_test else int(config["training"]["epochs"])
    save_interval = int(config["checkpoints"]["save_interval"])
    validation_interval = int(config["training"]["validation_interval"])
    best_metric_name = str(
        config["checkpoints"]["best_model_metric"]
    ).upper()
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
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for epoch in range(start_epoch, epochs):
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
            global_step += int(train_losses.pop("optimizer_steps"))
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
                        overwrite=True,
                        global_step=global_step,
                        rng_state=capture_rng_state(
                            train_loader,
                            validation_loader,
                        ),
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
            memory_stats = (
                cuda_memory_stats(device)
                if config["gpu"]["memory_monitoring"] else {}
            )
            for key, value in memory_stats.items():
                writer.add_scalar(
                    "gpu_memory/{}".format(key),
                    value,
                    epoch + 1,
                )
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
                    global_step=global_step,
                    rng_state=capture_rng_state(
                        train_loader,
                        validation_loader,
                    ),
                )

            if memory_stats:
                print("GPU memory: " + format_cuda_memory(memory_stats))
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
        "global_step": global_step,
        "elapsed_seconds": time.perf_counter() - start_time,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "p2pnet_scratch.yaml"),
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume", help="Resume from a scratch training checkpoint.")
    parser.add_argument("--output-dir", help="Use a fresh experiment output directory.")
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Rejected: pretrained initialization is disabled.",
    )
    args = parser.parse_args()
    if args.pretrained:
        raise RuntimeError("Pretrained model initialization is forbidden.")
    if args.smoke_test and args.resume:
        raise RuntimeError("--smoke-test and --resume cannot be used together.")

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
        result = run_training(
            config,
            device,
            output_root=args.output_dir,
            resume_checkpoint=args.resume,
        )
        print(
            "Training complete. Best {}: {:.6f}".format(
                result["best_metric_name"],
                result["best_metric_value"],
            )
        )


if __name__ == "__main__":
    main()
