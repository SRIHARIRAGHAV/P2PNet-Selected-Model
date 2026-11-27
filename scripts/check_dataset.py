import argparse
import csv
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import load_config
from dataset.annotations import load_jhu_annotations


SPLITS = {
    "train": ("train_manifest", "train_image_dir", "train_annotation_dir"),
    "validation": (
        "validation_manifest",
        "validation_image_dir",
        "validation_annotation_dir",
    ),
    "test": ("test_manifest", "test_image_dir", "test_annotation_dir"),
}


def _manifest_stems(manifest_path):
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")
    stems = []
    duplicates = []
    seen = set()
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row_number, row in enumerate(csv.reader(handle), start=1):
            if not row or not row[0].strip():
                continue
            stem = Path(row[0].strip()).stem
            if stem in seen:
                duplicates.append({"row": row_number, "image_id": stem})
                continue
            stems.append(stem)
            seen.add(stem)
    return stems, duplicates


def validate_split(dataset_config, split):
    root = Path(dataset_config["dataset_root"])
    manifest_key, image_key, annotation_key = SPLITS[split]
    manifest_path = root / dataset_config[manifest_key]
    image_dir = root / dataset_config[image_key]
    annotation_dir = root / dataset_config[annotation_key]

    stems, duplicate_manifest_entries = _manifest_stems(manifest_path)
    images_by_stem = {
        path.stem: path for path in image_dir.glob("*.jpg") if path.is_file()
    }
    annotations_by_stem = {
        path.stem: path for path in annotation_dir.glob("*.txt") if path.is_file()
    }

    missing_images = []
    missing_annotations = []
    unreadable_images = []
    point_counts = []
    invalid_annotation_counts = {
        "malformed_or_non_finite": 0,
        "out_of_bounds_clamped": 0,
        "total": 0,
    }

    for stem in stems:
        image_path = images_by_stem.get(stem)
        annotation_path = annotations_by_stem.get(stem)
        if image_path is None:
            missing_images.append(stem)
        if annotation_path is None:
            missing_annotations.append(stem)
        if image_path is None or annotation_path is None:
            continue

        try:
            with Image.open(image_path) as image:
                width, height = image.size
                image.verify()
        except (OSError, ValueError) as error:
            unreadable_images.append({"image_id": stem, "error": str(error)})
            continue

        annotation_stats = {}
        points = load_jhu_annotations(
            annotation_path,
            image_width=width,
            image_height=height,
            correction_stats=annotation_stats,
            strict=False,
        )
        point_counts.append(int(len(points)))
        malformed = (
            annotation_stats["malformed_annotations"]
            + annotation_stats["non_finite_annotations"]
        )
        out_of_bounds = annotation_stats["out_of_bounds_annotations"]
        invalid_annotation_counts["malformed_or_non_finite"] += malformed
        invalid_annotation_counts["out_of_bounds_clamped"] += out_of_bounds
        invalid_annotation_counts["total"] += annotation_stats["invalid_annotations"]

    manifest_ids = set(stems)
    image_ids = set(images_by_stem)
    annotation_ids = set(annotations_by_stem)
    missing_pair_ids = sorted(set(missing_images) | set(missing_annotations))
    average = (
        sum(point_counts) / len(point_counts)
        if point_counts else None
    )
    return {
        "manifest_entries": len(stems),
        "number_of_images": len(image_ids),
        "number_of_paired_images": len(point_counts),
        "total_annotations": sum(point_counts),
        "minimum_people_per_image": min(point_counts) if point_counts else None,
        "maximum_people_per_image": max(point_counts) if point_counts else None,
        "average_people_per_image": average,
        "statistics_images": len(point_counts),
        "missing_image_annotation_pairs": {
            "count": len(missing_pair_ids),
            "missing_images": missing_images,
            "missing_annotations": missing_annotations,
        },
        "invalid_annotations": invalid_annotation_counts,
        "unlisted_image_files": sorted(image_ids - manifest_ids),
        "unlisted_annotation_files": sorted(annotation_ids - manifest_ids),
        "duplicate_manifest_entries": duplicate_manifest_entries,
        "unreadable_images": unreadable_images,
    }


def build_report(config):
    dataset_config = config["dataset"]
    return {
        "dataset": dataset_config["name"],
        "dataset_root": str(Path(dataset_config["dataset_root"])),
        "annotation_policy": (
            "Malformed and non-finite rows are reported. Finite out-of-bounds "
            "head points are clamped to image bounds and retained; they are "
            "counted separately as invalid annotations."
        ),
        "splits": {
            split: validate_split(dataset_config, split)
            for split in SPLITS
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Validate JHU-CROWD++ files and report point statistics."
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "p2pnet_scratch.yaml"),
    )
    parser.add_argument(
        "--output",
        help="Optional JSON report path; existing files require --overwrite.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing the selected report file.",
    )
    args = parser.parse_args()

    report = build_report(load_config(args.config))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if args.overwrite else "x"
        with output_path.open(mode, encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print("Report saved to: {}".format(output_path))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
