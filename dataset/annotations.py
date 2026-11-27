from pathlib import Path
import numpy as np
from PIL import Image


def load_jhu_annotations(annotation_path, image_width=None, image_height=None,
                         correction_stats=None, strict=True):
    points = []
    annotation_path = Path(annotation_path)
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
    if correction_stats is not None:
        correction_stats.setdefault("invalid_annotations", 0)
        correction_stats.setdefault("malformed_annotations", 0)
        correction_stats.setdefault("non_finite_annotations", 0)
        correction_stats.setdefault("out_of_bounds_annotations", 0)
        correction_stats.setdefault("clamped_points", 0)
    if image_width is not None and image_width < 1:
        raise ValueError("image_width must be positive")
    if image_height is not None and image_height < 1:
        raise ValueError("image_height must be positive")

    with annotation_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.strip().split()
            if not fields:
                continue
            if len(fields) < 2:
                if strict:
                    raise ValueError(
                        f"Malformed annotation at {annotation_path}:{line_number}"
                    )
                if correction_stats is not None:
                    correction_stats["invalid_annotations"] += 1
                    correction_stats["malformed_annotations"] += 1
                continue
            try:
                x, y = float(fields[0]), float(fields[1])
            except (TypeError, ValueError):
                if strict:
                    raise ValueError(
                        f"Malformed annotation at {annotation_path}:{line_number}"
                    )
                if correction_stats is not None:
                    correction_stats["invalid_annotations"] += 1
                    correction_stats["malformed_annotations"] += 1
                continue
            if not np.isfinite(x) or not np.isfinite(y):
                if strict:
                    raise ValueError(
                        f"Non-finite annotation at {annotation_path}:{line_number}"
                    )
                if correction_stats is not None:
                    correction_stats["invalid_annotations"] += 1
                    correction_stats["non_finite_annotations"] += 1
                continue
            clamped = False
            if image_width is not None:
                corrected_x = float(np.clip(x, 0.0, max(0.0, image_width - 1.0)))
                clamped = clamped or corrected_x != x
                x = corrected_x
            if image_height is not None:
                corrected_y = float(np.clip(y, 0.0, max(0.0, image_height - 1.0)))
                clamped = clamped or corrected_y != y
                y = corrected_y
            if clamped and correction_stats is not None:
                correction_stats["clamped_points"] += 1
                correction_stats["out_of_bounds_annotations"] += 1
                correction_stats["invalid_annotations"] += 1
            points.append([x, y])
    return np.asarray(points, dtype=np.float32).reshape(-1, 2)


def read_image_size(image_path):
    with Image.open(image_path) as image:
        return image.size
