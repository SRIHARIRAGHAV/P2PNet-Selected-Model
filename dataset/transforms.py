import random
import math
import numpy as np
import torch
from PIL import Image


IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]
_IMAGE_MEAN_TENSOR = torch.tensor(IMAGE_MEAN)[:, None, None]
_IMAGE_STD_TENSOR = torch.tensor(IMAGE_STD)[:, None, None]


def training_transform():
    return transform_image


def resize_for_model(image, max_side):
    if max_side < 1:
        raise ValueError("max_side must be positive")
    width, height = image.size
    if width < 1 or height < 1:
        raise ValueError("image dimensions must be positive")
    scale = min(1.0, max_side / max(width, height))
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = (
        image
        if (resized_width, resized_height) == (width, height)
        else image.resize(
            (resized_width, resized_height),
            Image.Resampling.BILINEAR,
        )
    )
    return resized, resized_width / width, resized_height / height


def transform_image(image):
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Model images must be RGB")
    tensor = torch.from_numpy(array.transpose(2, 0, 1).copy()).div_(255.0)
    return (tensor - _IMAGE_MEAN_TENSOR) / _IMAGE_STD_TENSOR


def extract_random_patches(image, points, patch_size=128, patch_count=4,
                           random_scale_min=0.7, random_scale_max=1.3,
                           random_flip=True):
    if patch_size < 1 or patch_count < 1:
        raise ValueError("patch_size and patch_count must be positive")
    if (
        not math.isfinite(random_scale_min)
        or not math.isfinite(random_scale_max)
        or random_scale_min <= 0
        or random_scale_max < random_scale_min
    ):
        raise ValueError("Random scale bounds must be finite, positive, and ordered")

    points = np.asarray(points, dtype=np.float32)
    if points.size == 0:
        points = np.zeros((0, 2), dtype=np.float32)
    elif points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    else:
        points = points.copy()

    scale = random.uniform(random_scale_min, random_scale_max)
    width, height = image.size
    if width < 1 or height < 1:
        raise ValueError("image dimensions must be positive")
    scale = max(scale, patch_size / width, patch_size / height)
    new_width = max(patch_size, round(width * scale))
    new_height = max(patch_size, round(height * scale))
    resized = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
    scaled_points = points.copy()
    scaled_points[:, 0] *= new_width / width
    scaled_points[:, 1] *= new_height / height

    patches = []
    targets = []
    to_tensor = training_transform()
    for _ in range(patch_count):
        left = random.randint(0, max(0, new_width - patch_size))
        top = random.randint(0, max(0, new_height - patch_size))
        patch = resized.crop((left, top, left + patch_size, top + patch_size))
        patch_points = scaled_points.copy()
        patch_points[:, 0] -= left
        patch_points[:, 1] -= top
        inside = (
            (patch_points[:, 0] >= 0)
            & (patch_points[:, 0] < patch_size)
            & (patch_points[:, 1] >= 0)
            & (patch_points[:, 1] < patch_size)
        )
        patch_points = patch_points[inside]

        if random_flip and random.random() < 0.5:
            patch = patch.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if len(patch_points):
                patch_points[:, 0] = patch_size - 1 - patch_points[:, 0]

        patches.append(to_tensor(patch))
        targets.append({
            "point": torch.from_numpy(patch_points.astype(np.float32)),
            "labels": torch.ones(len(patch_points), dtype=torch.long),
        })
    return torch.stack(patches), targets
