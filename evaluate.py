import argparse
import csv
import json
import time
from pathlib import Path

import torch

from config import load_config
from model.p2pnet import build_p2pnet
from dataset.dataset_loader import JHUCrowdDataset, collate_validation
from utils.checkpoint import load_scratch_checkpoint
from utils.metrics import crowd_metrics
from utils.visualization import save_actual_vs_predicted, save_point_visualization


def require_gpu(config):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. GPU evaluation cannot start.")
    index = int(config["gpu"]["gpu_device_index"])
    torch.cuda.set_device(index)
    return torch.device(f"cuda:{index}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/p2pnet_scratch.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--benchmark-bs1", action="store_true")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(
        f"cuda:{config['gpu']['gpu_device_index']}"
        if torch.cuda.is_available() else "cpu"
    )
    model, _ = build_p2pnet(config)
    checkpoint = load_scratch_checkpoint(args.checkpoint)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    dataset = JHUCrowdDataset(config, args.split)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config["dataset"]["num_workers"],
        pin_memory=config["gpu"]["pinned_memory"],
        collate_fn=collate_validation,
    )
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
    prediction_dir = root / "outputs" / "predictions"
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
                torch.cuda.synchronize()
                start = time.perf_counter()
                images = images.to(device, non_blocking=config["gpu"]["non_blocking"])
                outputs = model(images)
                scores = outputs["pred_logits"].softmax(-1)[0, :, 1]
                valid = scores > threshold
                points = outputs["pred_points"][0, valid].detach().cpu().numpy()
                torch.cuda.synchronize()
                latency_ms = (time.perf_counter() - start) * 1000.0

                target = targets[0]
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
                    threshold,
                )

    metrics = crowd_metrics(actual, predicted)
    mean_latency = sum(latencies) / max(1, len(latencies))
    summary = {
        **metrics,
        "average_inference_latency_ms": mean_latency,
        "FPS": 1000.0 / mean_latency if mean_latency > 0 else 0.0,
        "gpu_name": torch.cuda.get_device_name(device),
        "batch_size": 1,
        "input_resolution": "Native model input per source image",
        "precision_mode": "AMP" if config["gpu"]["mixed_precision"] else "FP32",
        "evaluation_image_count": len(actual),
        "evaluation_split": args.split,
        "timing_methodology": "CUDA synchronize before and after model inference",
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
