import argparse
import csv
import json
import time
from pathlib import Path

import torch

from config import load_config
from dataset.dataset_loader import JHUCrowdDataset, collate_validation
from models import P2PNet
from utils.checkpoint import load_scratch_checkpoint
from utils.device import (
    amp_settings,
    autocast_context,
    cuda_memory_stats,
    select_device,
)
from utils.metrics import crowd_metrics
from utils.visualization import save_actual_vs_predicted, save_point_visualization


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/p2pnet_scratch.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    config = load_config(args.config)
    if int(config["evaluation"]["batch_size"]) != 1:
        raise ValueError("Evaluation requires evaluation.batch_size=1.")
    device = select_device(config)
    amp_enabled, amp_dtype = amp_settings(config, device)
    model_config = config["model"]
    model = P2PNet(
        row=int(model_config["row"]),
        line=int(model_config["line"]),
        num_classes=int(model_config["num_classes"]),
        pretrained=False,
    )
    checkpoint = load_scratch_checkpoint(args.checkpoint)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    model.eval()

    dataset = JHUCrowdDataset(config, args.split)
    if not len(dataset):
        raise RuntimeError(f"The {args.split} dataset split is empty.")
    workers = int(config["dataset"]["num_workers"])
    loader_options = (
        {"prefetch_factor": int(config["dataset"]["prefetch_factor"])}
        if workers > 0 else {}
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=(
            device.type == "cuda" and bool(config["gpu"]["pinned_memory"])
        ),
        collate_fn=collate_validation,
        **loader_options,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    threshold = config["evaluation"]["prediction_confidence_threshold"]
    root = Path(__file__).resolve().parent
    evaluation_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir else root / "outputs" / "evaluation"
    )
    if evaluation_dir.exists() and any(evaluation_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite evaluation results in {evaluation_dir}. "
            "Choose a new --output-dir."
        )
    prediction_dir = evaluation_dir / "predictions"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    actual = []
    predicted = []
    latencies = []
    csv_path = evaluation_dir / "per_image_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "image", "actual_count", "predicted_count", "absolute_error",
            "latency_ms",
        ])
        writer.writeheader()
        with torch.inference_mode():
            for images, targets in loader:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
                images = images.to(
                    device,
                    non_blocking=(
                        device.type == "cuda"
                        and bool(config["gpu"]["pinned_memory"])
                        and bool(config["gpu"]["non_blocking"])
                    ),
                )
                try:
                    with autocast_context(amp_enabled, amp_dtype, device):
                        outputs = model(images)
                    if (
                        not torch.isfinite(outputs["pred_logits"]).all()
                        or not torch.isfinite(outputs["pred_points"]).all()
                    ):
                        raise RuntimeError(
                            "Model produced NaN or Inf during evaluation."
                        )
                except torch.cuda.OutOfMemoryError as error:
                    stats = cuda_memory_stats(device)
                    raise RuntimeError(
                        f"CUDA out of memory during evaluation for input "
                        f"shape {tuple(images.shape)}; allocated="
                        f"{stats.get('allocated_mb', 0):.0f} MiB, reserved="
                        f"{stats.get('reserved_mb', 0):.0f} MiB, peak="
                        f"{stats.get('peak_allocated_mb', 0):.0f} MiB."
                    ) from error
                scores = outputs["pred_logits"].float().softmax(-1)[..., 1]
                valid = scores[0] > threshold
                points = (
                    outputs["pred_points"][0, valid]
                    .float()
                    .detach()
                    .cpu()
                    .numpy()
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                latency_ms = (time.perf_counter() - start) * 1000.0

                target = targets[0]
                points[:, 0] *= float(target["scale_x"])
                points[:, 1] *= float(target["scale_y"])
                actual_count = len(target["point"])
                predicted_count = len(points)
                image_path = target["image_path"]
                actual.append(actual_count)
                predicted.append(predicted_count)
                latencies.append(latency_ms)
                writer.writerow({
                    "image": Path(image_path).name,
                    "actual_count": actual_count,
                    "predicted_count": predicted_count,
                    "absolute_error": abs(predicted_count - actual_count),
                    "latency_ms": latency_ms,
                })
                save_point_visualization(
                    image_path,
                    points,
                    prediction_dir / f"{Path(image_path).stem}_predicted.jpg",
                )
                del outputs, scores, images, valid

    metrics = crowd_metrics(actual, predicted)
    mean_latency = sum(latencies) / max(1, len(latencies))
    memory_stats = cuda_memory_stats(device)
    summary = {
        **metrics,
        "average_inference_latency_ms": mean_latency,
        "FPS": 1000.0 / mean_latency if mean_latency > 0 else 0.0,
        "device": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda" else "CPU"
        ),
        "batch_size": 1,
        "input_resolution": (
            "Aspect-preserving resize, max side {} pixels".format(
                config["dataset"]["validation_max_side"]
            )
        ),
        "precision_mode": (
            "AMP-{}".format(config["gpu"]["amp_dtype"])
            if amp_enabled else "FP32"
        ),
        "evaluation_image_count": len(actual),
        "evaluation_split": args.split,
        "timing_methodology": (
            "CUDA synchronize before and after model inference"
            if device.type == "cuda" else "CPU wall-clock model inference"
        ),
        "gpu_memory": memory_stats,
    }
    with open(evaluation_dir / "performance_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    save_actual_vs_predicted(
        actual,
        predicted,
        evaluation_dir / "actual_vs_predicted.png",
    )
    print(json.dumps(summary, indent=2))
    print(f"Per-image CSV: {csv_path}")


if __name__ == "__main__":
    main()
