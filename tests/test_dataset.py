import sys
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import load_config
from dataset.dataset_loader import JHUCrowdDataset
from dataset.annotations import load_jhu_annotations
from dataset.transforms import extract_random_patches, resize_for_model


def test_jhu_dataset_splits_exist():
    config = load_config()
    for split in ("train", "validation", "test"):
        dataset = JHUCrowdDataset(config, split)
        assert len(dataset) > 0


def test_out_of_bounds_points_are_clamped(tmp_path):
    gt_path = tmp_path / "sample.txt"
    gt_path.write_text("2381 3280 9 11 1 0\n-5 20 9 11 1 0\n")
    stats = {}
    points = load_jhu_annotations(
        gt_path,
        image_width=4928,
        image_height=3280,
        correction_stats=stats,
    )
    assert len(points) == 2
    assert np.allclose(points[0], [2381.0, 3279.0])
    assert np.allclose(points[1], [0.0, 20.0])
    assert stats["clamped_points"] == 2


def test_horizontal_flip_preserves_valid_pixel_coordinates(monkeypatch):
    monkeypatch.setattr("dataset.transforms.random.uniform", lambda low, high: 1.0)
    monkeypatch.setattr("dataset.transforms.random.randint", lambda low, high: 0)
    monkeypatch.setattr("dataset.transforms.random.random", lambda: 0.0)
    image = Image.new("RGB", (128, 128))
    patches, targets = extract_random_patches(
        image,
        np.asarray([[0.0, 0.0]], dtype=np.float32),
        patch_size=128,
        patch_count=1,
        random_scale_min=1.0,
        random_scale_max=1.0,
        random_flip=True,
    )
    assert patches.shape == (1, 3, 128, 128)
    assert torch.allclose(targets[0]["point"], torch.tensor([[127.0, 0.0]]))


def test_evaluation_resize_preserves_aspect_ratio():
    image = Image.new("RGB", (200, 100))
    resized, scale_x, scale_y = resize_for_model(image, max_side=50)
    assert resized.size == (50, 25)
    assert scale_x == scale_y == 0.25


def test_malformed_annotations_fail_with_file_and_line(tmp_path):
    gt_path = tmp_path / "bad.txt"
    gt_path.write_text("10 20\ninvalid row\n")
    with pytest.raises(ValueError, match=r"bad.txt:2"):
        load_jhu_annotations(gt_path)
