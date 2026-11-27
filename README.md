# P2PNet Scratch Crowd Counting

This project trains a point-based crowd-counting P2PNet from random
initialization. It does not use pretrained or external model weights. Training,
validation, evaluation, and inference all use the same implementation in
`models/`; the `model/` package remains as a compatibility import path.

## Architecture and objective

- Randomly initialized VGG16-BN backbone and feature pyramid.
- Stride-8 classification and point-regression heads with four anchor points
  per feature location.
- Background/person classification and the original class-cost plus
  Euclidean-point-cost Hungarian assignment.
- Cross-entropy classification and matched-point MSE localization losses.
- No ground-truth point subsampling. Training crops keep every annotated point
  that lies inside each crop; validation resizes all annotations consistently.

## Environment and dataset

Use Python 3.10 and install a PyTorch build compatible with the target CUDA
driver first, followed by the remaining dependencies. PyTorch is intentionally
not pinned in `requirements.txt`, so installing project dependencies will not
replace a CUDA-specific PyTorch build.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

By default, the dataset root is the sibling path
`../P2PNet/dataset/JHU-CROWD++`, relative to this project. Override it on Linux
when the dataset is elsewhere:

```bash
export JHU_CROWD_ROOT=/absolute/path/to/JHU-CROWD++
```

Each split must provide an `image_labels.txt` manifest and matching `.jpg`
images and `.txt` annotations in the configured directories. Startup scans
image/annotation pairs, verifies image readability and parses every annotation
file. Empty annotation files are allowed and reported; malformed or non-finite
annotation rows fail with their file and line. Finite out-of-bounds points are
clamped to valid image coordinates and retained.

## Memory and resolution settings

The default training batch size and validation batch size are both 1.
`training.gradient_accumulation_steps` controls effective optimizer batch size
without holding multiple forward graphs. CUDA FP16 autocast and gradient
scaling are enabled by default; loss/matching calculations use FP32. Hungarian
matching and its complete cost matrix are computed on CPU, preserving all
prediction/ground-truth assignments while avoiding a large CUDA `cdist`
allocation.

Training crops default to 128×128. Validation/test images are resized
proportionally, without upscaling, to a configurable maximum side of 1024
pixels; annotation coordinates are scaled by the same per-axis resize factors.
If validation still exceeds memory, lower `dataset.validation_max_side`.
Worker count, prefetch factor, pinning, AMP dtype, and memory diagnostics are
configurable in `configs/p2pnet_scratch.yaml`. CPU is used automatically when
CUDA is unavailable; explicitly setting `gpu.device: cuda` instead fails
clearly if CUDA cannot be selected.

## Validation and smoke test

Run the dataset report, scratch initialization check, and unit tests:

```bash
python scripts/check_dataset.py --output logs/dataset-validation-report.json
python scripts/verify_scratch.py
python -m pytest tests
```

Run a short end-to-end train/validation/checkpoint smoke test. It uses a
temporary output directory and does not overwrite existing results:

```bash
python -u train.py --config configs/p2pnet_scratch.yaml --smoke-test
```

## Training, outputs, and resume

Start a fresh run in the configured checkpoint/log directories:

```bash
python -u train.py --config configs/p2pnet_scratch.yaml
```

Existing checkpoints, history, and TensorBoard event files are never silently
overwritten by a fresh run. Use a new experiment directory when the default
outputs already exist:

```bash
python -u train.py --config configs/p2pnet_scratch.yaml --output-dir runs/experiment-02
```

Resume only from a provenance-verified checkpoint written by this project.
Model, dataset, optimization, and precision settings are checked for
compatibility before restoring model, optimizer, scheduler, AMP scaler, epoch,
and global-step state:

```bash
python -u train.py --config configs/p2pnet_scratch.yaml --resume checkpoints/latest.pth
```

Resuming is distinct from a fresh scratch run; no pretrained checkpoint is
accepted. Checkpoints are written atomically, and best/latest checkpoints are
only replaced intentionally as training progresses.

## Evaluation and inference

Evaluate the validation split into a new output directory:

```bash
python evaluate.py --config configs/p2pnet_scratch.yaml --checkpoint checkpoints/best_mae.pth --split validation --output-dir outputs/evaluation-run-01
```

Image and video inference also use the configured maximum side and preserve
the source coordinate system when drawing predictions. Output files are not
overwritten.

## GPU memory reporting and limitations

The trainer reports CUDA allocated, reserved, and peak allocated memory in the
console and TensorBoard, and records host peak RSS in the CSV history. CUDA
out-of-memory failures include the phase, input shape, and allocator figures;
failed batches are not skipped. Actual VRAM use depends on image aspect ratio,
crowd-point count, driver, and other processes. Run the smoke test on the
target RTX 5060 before full training; this repository's current Windows
development environment does not provide that GPU, so RTX 5060 behavior must
be verified on the Linux training host.
