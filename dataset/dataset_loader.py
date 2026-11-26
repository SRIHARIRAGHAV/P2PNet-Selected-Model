import csv
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .annotations import load_jhu_annotations
from .transforms import training_transform, extract_random_patches


_SPLIT_CONFIG = {
    "train": ("train_manifest", "train_image_dir", "train_annotation_dir"),
    "validation": (
        "validation_manifest",
        "validation_image_dir",
        "validation_annotation_dir",
    ),
    "test": ("test_manifest", "test_image_dir", "test_annotation_dir"),
}


class JHUCrowdDataset(Dataset):
    def __init__(self, config, split="train", smoke_test=False):
        dataset = config["dataset"]
        root = Path(dataset["dataset_root"])
        if split not in _SPLIT_CONFIG:
            raise ValueError(f"Unknown JHU-CROWD++ split: {split}")
        manifest_key, image_key, annotation_key = _SPLIT_CONFIG[split]
        manifest = root / dataset[manifest_key]
        image_dir = root / dataset[image_key]
        gt_dir = root / dataset[annotation_key]

        self.image_dir = image_dir
        self.gt_dir = gt_dir
        self.split = split
        self.smoke_test = smoke_test
        self.config = config
        if not manifest.is_file():
            raise FileNotFoundError(f"Dataset manifest not found: {manifest}")
        samples = []
        seen_stems = set()
        with manifest.open("r", newline="", encoding="utf-8-sig") as handle:
            for row in csv.reader(handle):
                if not row or not row[0].strip():
                    continue
                stem = Path(row[0].strip()).stem
                if stem in seen_stems:
                    raise ValueError(f"Duplicate image entry in manifest: {stem}")
                seen_stems.add(stem)
                image_path = image_dir / f"{stem}.jpg"
                if not image_path.is_file():
                    raise FileNotFoundError(
                        f"Manifest entry has no matching image: {image_path}"
                    )
                annotation_path = gt_dir / f"{stem}.txt"
                if not annotation_path.is_file():
                    raise FileNotFoundError(
                        f"Manifest entry has no matching annotation: {annotation_path}"
                    )
                samples.append((image_path, annotation_path))
        self.samples = samples
        self.image_paths = [image_path for image_path, _ in samples]
        self.transform = training_transform()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, annotation_path = self.samples[index]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        points = load_jhu_annotations(
            annotation_path,
            image.width,
            image.height,
        )

        if self.split == "train" or self.smoke_test:
            augmentation = self.config["dataset"]["augmentation"]
            patches, targets = extract_random_patches(
                image,
                points,
                patch_size=augmentation["patch_size"],
                patch_count=augmentation["patches_per_image"],
                random_scale_min=augmentation["random_scale_min"],
                random_scale_max=augmentation["random_scale_max"],
                random_flip=augmentation["random_flip"],
            )
            for target in targets:
                target["image_id"] = torch.tensor([index], dtype=torch.long)
            if self.split == "train":
                return patches, targets
            return patches[0], {
                **targets[0],
                "image_path": str(image_path),
            }

        tensor = self.transform(image)
        target = {
            "point": torch.from_numpy(points.astype(np.float32)),
            "labels": torch.ones(len(points), dtype=torch.long),
            "image_id": torch.tensor([index], dtype=torch.long),
            "image_path": str(image_path),
        }
        return tensor, target


def collate_train(batch):
    tensors = []
    targets = []
    for patches, patch_targets in batch:
        if patches.shape[0] != len(patch_targets):
            raise ValueError("Each training patch must have one target dictionary")
        tensors.append(patches)
        targets.extend(patch_targets)
    if not tensors:
        raise ValueError("Cannot collate an empty training batch")
    return torch.cat(tensors, dim=0), targets


def collate_validation(batch, stride=8):
    if not batch:
        raise ValueError("Cannot collate an empty evaluation batch")
    if stride < 1:
        raise ValueError("stride must be positive")
    tensors = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    max_height = (max(t.shape[1] for t in tensors) + stride - 1) // stride * stride
    max_width = (max(t.shape[2] for t in tensors) + stride - 1) // stride * stride
    padded = torch.zeros(
        len(tensors),
        3,
        max_height,
        max_width,
        dtype=tensors[0].dtype,
    )
    for index, tensor in enumerate(tensors):
        padded[index, :, :tensor.shape[1], :tensor.shape[2]] = tensor
    return padded, targets
