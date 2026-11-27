import math
import os
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent


def load_config(path=None):
    config_path = Path(path) if path is not None else (
        PROJECT_ROOT / "configs" / "p2pnet_scratch.yaml"
    )
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {config_path}")
    required_sections = (
        "project",
        "dataset",
        "model",
        "training",
        "gpu",
        "evaluation",
        "checkpoints",
    )
    for section in required_sections:
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Configuration section '{section}' must be a mapping")

    dataset = config.get("dataset")
    if not isinstance(dataset, dict) or not dataset.get("dataset_root"):
        raise ValueError("dataset.dataset_root must be configured")
    dataset_root = Path(
        os.environ.get("JHU_CROWD_ROOT", dataset["dataset_root"])
    ).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = PROJECT_ROOT / dataset_root
    dataset["dataset_root"] = str(dataset_root.resolve())

    if config["model"].get("pretrained") is not False:
        raise RuntimeError("model.pretrained must be false")
    if config["training"].get("mode") != "scratch":
        raise RuntimeError("training.mode must be scratch")
    if "resume" in config["training"] or "resume_checkpoint" in config["training"]:
        raise RuntimeError(
            "Use the explicit --resume command-line option, not training.resume."
        )
    device = str(config["gpu"].get("device", "")).lower()
    if device not in {"auto", "cpu", "cuda"}:
        raise RuntimeError("gpu.device must be 'auto', 'cpu', or 'cuda'")
    if int(config["gpu"].get("gpu_device_index", -1)) < 0:
        raise RuntimeError("gpu.gpu_device_index cannot be negative")
    if str(config["gpu"].get("amp_dtype", "")).lower() not in {
        "float16",
        "bfloat16",
    }:
        raise RuntimeError("gpu.amp_dtype must be float16 or bfloat16")
    if int(config["training"].get("cpu_num_threads", 0)) < 1:
        raise RuntimeError("training.cpu_num_threads must be a positive integer")
    if int(config["training"].get("cpu_interop_threads", 0)) < 1:
        raise RuntimeError("training.cpu_interop_threads must be a positive integer")
    if int(config["dataset"].get("num_workers", -1)) < 0:
        raise RuntimeError("dataset.num_workers cannot be negative")
    if int(config["dataset"].get("prefetch_factor", 0)) < 1:
        raise RuntimeError("dataset.prefetch_factor must be positive")
    if int(config["training"].get("batch_size", 0)) < 1:
        raise RuntimeError("training.batch_size must be a positive integer")
    if int(config["training"].get("gradient_accumulation_steps", 0)) < 1:
        raise RuntimeError(
            "training.gradient_accumulation_steps must be a positive integer"
        )
    if int(config["training"].get("epochs", 0)) < 1:
        raise RuntimeError("training.epochs must be positive")
    if int(config["training"].get("validation_interval", 0)) < 1:
        raise RuntimeError("training.validation_interval must be positive")
    if int(config["training"].get("smoke_test_train_batches", 0)) < 1:
        raise RuntimeError("training.smoke_test_train_batches must be positive")
    if int(config["training"].get("smoke_test_validation_batches", 0)) < 1:
        raise RuntimeError("training.smoke_test_validation_batches must be positive")
    if int(config["checkpoints"].get("save_interval", 0)) < 1:
        raise RuntimeError("checkpoints.save_interval must be positive")
    patch_size = int(config["dataset"]["augmentation"].get("patch_size", 0))
    if patch_size < 32 or patch_size % 8:
        raise RuntimeError(
            "dataset.augmentation.patch_size must be at least 32 and divisible by 8"
        )
    max_side = int(config["dataset"].get("validation_max_side", 0))
    if max_side < 32:
        raise RuntimeError("dataset.validation_max_side must be at least 32")
    if int(config["evaluation"].get("batch_size", 0)) != 1:
        raise RuntimeError(
            "evaluation.batch_size must be 1 to keep validation sequential."
        )
    if int(config["dataset"]["augmentation"].get("patches_per_image", 0)) < 1:
        raise RuntimeError("dataset.augmentation.patches_per_image must be positive")
    for key in ("random_scale_min", "random_scale_max"):
        value = float(config["dataset"]["augmentation"].get(key, float("nan")))
        if not math.isfinite(value) or value <= 0:
            raise RuntimeError(
                f"dataset.augmentation.{key} must be finite and positive"
            )
    if (
        float(config["dataset"]["augmentation"]["random_scale_max"])
        < float(config["dataset"]["augmentation"]["random_scale_min"])
    ):
        raise RuntimeError("Random augmentation scale bounds are reversed")
    resolution = config["model"].get("input_resolution")
    if (
        not isinstance(resolution, (list, tuple))
        or len(resolution) != 2
        or any(int(dimension) < 32 or int(dimension) % 8 for dimension in resolution)
    ):
        raise RuntimeError(
            "model.input_resolution must contain two dimensions >=32 and divisible by 8"
        )
    if int(resolution[0]) != int(resolution[1]) or int(resolution[0]) != patch_size:
        raise RuntimeError(
            "model.input_resolution must be square and match patch_size"
        )
    for key in ("row", "line", "num_classes"):
        if int(config["model"].get(key, 0)) < 1:
            raise RuntimeError(f"model.{key} must be positive")
    for key in ("set_cost_class", "set_cost_point", "point_loss_coef"):
        value = float(config["model"].get(key, float("nan")))
        if not math.isfinite(value) or value < 0:
            raise RuntimeError(f"model.{key} must be finite and non-negative")
    if (
        float(config["model"]["set_cost_class"]) == 0
        and float(config["model"]["set_cost_point"]) == 0
    ):
        raise RuntimeError("At least one Hungarian matching cost must be positive")
    if (
        not math.isfinite(float(config["model"].get("eos_coef", float("nan"))))
        or float(config["model"]["eos_coef"]) <= 0
    ):
        raise RuntimeError("model.eos_coef must be finite and positive")
    for key in (
        "learning_rate",
        "backbone_learning_rate",
        "lr_gamma",
    ):
        value = float(config["training"].get(key, float("nan")))
        if not math.isfinite(value) or value <= 0:
            raise RuntimeError(f"training.{key} must be finite and positive")
    for key in ("weight_decay", "gradient_clip_norm"):
        value = float(config["training"].get(key, float("nan")))
        if not math.isfinite(value) or value < 0:
            raise RuntimeError(f"training.{key} must be finite and non-negative")
    if int(config["training"].get("lr_drop", 0)) < 1:
        raise RuntimeError("training.lr_drop must be positive")
    threshold = float(
        config["evaluation"].get(
            "prediction_confidence_threshold",
            float("nan"),
        )
    )
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise RuntimeError(
            "evaluation.prediction_confidence_threshold must be in [0, 1]"
        )
    if str(config["checkpoints"].get("best_model_metric", "")).upper() not in {
        "MAE",
        "RMSE",
    }:
        raise RuntimeError("checkpoints.best_model_metric must be MAE or RMSE")
    if str(config["training"].get("optimizer", "")).lower() not in {
        "adam",
        "adamw",
    }:
        raise RuntimeError("training.optimizer must be Adam or AdamW")
    if str(config["training"].get("lr_scheduler", "")).lower() != "steplr":
        raise RuntimeError("training.lr_scheduler must be StepLR")
    if str(config["evaluation"].get("zero_gt_mape", "")) != "omit":
        raise RuntimeError("evaluation.zero_gt_mape must be 'omit'")
    return config
