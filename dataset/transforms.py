import random
import math
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms


IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]


def training_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
    ])


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
    new_width = max(patch_size, int(width * scale))
    new_height = max(patch_size, int(height * scale))
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
