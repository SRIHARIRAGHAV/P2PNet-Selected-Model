import argparse
import math
from pathlib import Path

import cv2
import torch
from PIL import Image

from config import load_config
from dataset.transforms import resize_for_model, transform_image
from models import P2PNet
from utils.checkpoint import load_scratch_checkpoint
from utils.device import amp_settings, autocast_context, select_device


def frame_tensor(frame, device, max_side):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized, scale_x, scale_y = resize_for_model(
        Image.fromarray(rgb),
        max_side,
    )
    tensor = transform_image(resized).unsqueeze(0).to(device)
    return tensor, scale_x, scale_y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/p2pnet_scratch.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", default="outputs/predictions/video_output.mp4")
    parser.add_argument("--frame-sampling", type=int, default=1)
    args = parser.parse_args()

    if args.frame_sampling < 1:
        raise ValueError("--frame-sampling must be at least 1")
    config = load_config(args.config)
    device = select_device(config)
    amp_enabled, amp_dtype = amp_settings(config, device)
    model_config = config["model"]
    model = P2PNet(
        row=int(model_config["row"]),
        line=int(model_config["line"]),
        num_classes=int(model_config["num_classes"]),
        pretrained=False,
    )
    payload = load_scratch_checkpoint(args.checkpoint)
    model.load_state_dict(payload["model"], strict=True)
    model.to(device)
    model.eval()
    threshold = config["evaluation"]["prediction_confidence_threshold"]

    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        fps = 25.0
    if width < 1 or height < 1:
        capture.release()
        raise RuntimeError(f"Video has invalid frame dimensions: {args.video}")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite video output: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create video output: {output}")

    frame_number = 0
    try:
        with torch.inference_mode():
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_number % args.frame_sampling == 0:
                    tensor, scale_x, scale_y = frame_tensor(
                        frame,
                        device,
                        int(config["dataset"]["validation_max_side"]),
                    )
                    with autocast_context(amp_enabled, amp_dtype, device):
                        outputs = model(tensor)
                    if (
                        not torch.isfinite(outputs["pred_logits"]).all()
                        or not torch.isfinite(outputs["pred_points"]).all()
                    ):
                        raise RuntimeError(
                            f"Model produced NaN or Inf in video frame "
                            f"{frame_number}."
                        )
                    scores = outputs["pred_logits"].float().softmax(-1)[0, :, 1]
                    valid = scores > threshold
                    points = (
                        outputs["pred_points"][0, valid]
                        .float()
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    points[:, 0] *= scale_x
                    points[:, 1] *= scale_y
                    for point in points:
                        cv2.circle(
                            frame,
                            (int(round(point[0])), int(round(point[1]))),
                            3,
                            (0, 0, 255),
                            -1,
                        )
                    cv2.putText(
                        frame,
                        f"Crowd count: {len(points)}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        2,
                    )
                    del outputs, scores, tensor, valid
                writer.write(frame)
                frame_number += 1
    finally:
        capture.release()
        writer.release()
    print(f"Saved annotated video: {output}")


if __name__ == "__main__":
    main()
