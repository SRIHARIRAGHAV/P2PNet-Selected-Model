# P2PNet-Scratch

This directory contains an independent P2PNet implementation that trains from random initialization on JHU-CROWD++. The trainer uses CUDA when available with `gpu.device: auto`, or CPU when CUDA is unavailable. It does not import or load weights from the existing P2PNet, CSRNet, APGCC, or pretrained torchvision models.

## Architecture

The implementation follows the original P2PNet design:

- A randomly initialized VGG16-BN feature backbone.
- A 256-channel FPN/decoder.
- Point and confidence branches at the stride-8 P3 feature level.
- 2-by-2 anchor points, giving four candidates per feature location.
- Two classes internally: background and person.
- Hungarian matching using class cost plus Euclidean point cost.
- Cross-entropy classification loss and MSE point loss.
- Predicted points as anchor positions plus 100 times the learned regression offsets.

The implementation is separate from all existing model folders. No existing P2PNet file is imported or modified.

## Strict scratch policy

`model.pretrained=false` and `training.mode=scratch` are validated at startup. Training always constructs a new randomly initialized model. The training command rejects pretrained and checkpoint arguments and has no checkpoint-loading or resume path. Checkpoints are saved with model, optimizer, scheduler, scaler, configuration, and validation metadata.

## Dataset

The dataset remains in place:

`C:\SA_CC\Crowd_Counting_Module\P2PNet\dataset\JHU-CROWD++`

The loader uses:

- Train: `train\images` and `train\gt`
- Validation: `val\images` and `val\gt`
- Test: `test\images` and `test\gt`

JHU annotation files contain `x y width height ...`; only the first two coordinates are used as head points. Existing out-of-bounds coordinates are retained and clamped to the valid image range. This prevents valid annotated people from being discarded due to a one-pixel boundary overflow. Malformed or non-finite coordinates are ignored. Source files are never modified.

## Installation

Install PyTorch with a build compatible with the target CUDA runtime if using an NVIDIA GPU. From PowerShell:

```powershell
cd C:\SA_CC\Crowd_Counting_Module\Selected_Model
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Configuration

All pipeline settings are in `configs\p2pnet_scratch.yaml`. Defaults include `gpu.device: auto`, `training.batch_size: 1`, `dataset.num_workers: 0`, `gpu.mixed_precision: false`, and `gpu.pinned_memory: false`. CPU intra-op/inter-op threads default to 2/1. Images are decoded as samples are requested; only manifest paths are held in memory.

Set `gpu.device: cuda` to require an NVIDIA GPU; training fails clearly if CUDA or the selected GPU index is unavailable. Mixed precision and pinned memory can be enabled in the YAML when appropriate. The P2PNet architecture is unchanged by device selection.

## Verification

Run these before training:

```powershell
python scripts\check_dataset.py
python scripts\verify_scratch.py
python -m pytest tests
```

The dataset report includes per-split image/annotation counts, crowd-count
ranges and averages, missing pairs, and invalid annotation coordinates. It is
written to `logs\dataset_validation.json`.

`gpu.device: auto` selects CUDA when available and otherwise uses CPU. Set it
to `cuda` to require an NVIDIA GPU; training exits with an explicit error if
CUDA is unavailable. Training never loads pretrained weights or resumes a
checkpoint.

## Short training smoke test

The smoke test trains on a small number of JHU-CROWD++ crops, validates on one
coordinate-transformed validation crop, and checks checkpoint, CSV, and
TensorBoard output creation. Its temporary outputs are removed when it exits;
it does not launch full training:

```powershell
python train.py --smoke-test
```


## Full training

After explicit approval and successful smoke testing:

```powershell
python train.py --config configs\p2pnet_scratch.yaml
```

Checkpoint loading and resume are intentionally disabled for training.

## Evaluation

Evaluate a scratch checkpoint against the untouched test split:

```powershell
python evaluate.py --checkpoint checkpoints\best_mae.pth --config configs\p2pnet_scratch.yaml
```

Outputs:

- `outputs\evaluation\per_image_results.csv`
- `outputs\evaluation\performance_summary.json`
- `outputs\evaluation\actual_vs_predicted.png`
- `outputs\predictions\*_predicted.jpg`

MAPE excludes zero-ground-truth images consistently.

## Image inference

```powershell
python inference.py --checkpoint checkpoints\best_mae.pth --image "C:\path\to\image.jpg" --output outputs\predictions\result.jpg
```

## Video inference

```powershell
python predict_video.py --checkpoint checkpoints\best_mae.pth --video "C:\path\to\video.mp4" --output outputs\predictions\video_output.mp4 --frame-sampling 1
```

The frame-sampling argument processes every Nth frame. RTSP support can be added later by replacing the `cv2.VideoCapture` source.
