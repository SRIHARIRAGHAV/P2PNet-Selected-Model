import json
import math
import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import load_config


def verify_split(root, image_dir, gt_dir, split):
    images = sorted((root / image_dir).glob("*.jpg"))
    annotations = sorted((root / gt_dir).glob("*.txt"))
    image_ids = {path.stem for path in images}
    gt_ids = {path.stem for path in annotations}
    point_count = 0
    coordinate_errors = 0
    malformed_points = 0
    clamped_points = 0
    sample = None
    for image_path in images:
        with Image.open(image_path) as image:
            width, height = image.size
        gt_path = root / gt_dir / f"{image_path.stem}.txt"
        if gt_path.exists():
            correction_stats = {}
            with open(gt_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    fields = line.strip().split()
                    if len(fields) >= 2:
                        try:
                            x, y = float(fields[0]), float(fields[1])
                        except (TypeError, ValueError):
                            malformed_points += 1
                            continue
                        if not (math.isfinite(x) and math.isfinite(y)):
                            malformed_points += 1
                            continue
                        point_count += 1
                        if not (0 <= x < width and 0 <= y < height):
                            coordinate_errors += 1
                            clamped_points += 1
                            sample = sample or (
                                str(gt_path),
                                x,
                                y,
                                width,
                                height,
                            )
    return {
        "images": len(images),
        "annotations": len(annotations),
        "missing_annotations": len(image_ids - gt_ids),
        "missing_images": len(gt_ids - image_ids),
        "points": point_count,
        "out_of_bounds_points": coordinate_errors,
        "clamped_points": clamped_points,
        "malformed_points": malformed_points,
        "first_out_of_bounds_example": sample,
    }


def main():
    config = load_config()
    root = Path(config["dataset"]["dataset_root"])
    report = {
        "dataset_root": str(root),
        "train": verify_split(
            root,
            config["dataset"]["train_image_dir"],
            config["dataset"]["train_annotation_dir"],
            "train",
        ),
        "validation": verify_split(
            root,
            config["dataset"]["validation_image_dir"],
            config["dataset"]["validation_annotation_dir"],
            "validation",
        ),
        "test": verify_split(
            root,
            config["dataset"]["test_image_dir"],
            config["dataset"]["test_annotation_dir"],
            "test",
        ),
    }
    output = Path(__file__).resolve().parents[1] / "logs" / "dataset_verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
