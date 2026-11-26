import argparse
from pathlib import Path

import cv2
import torch

from config import load_config
from model.p2pnet import build_p2pnet
from utils.checkpoint import load_scratch_checkpoint


def frame_tensor(frame, device):
    tensor = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
    tensor = tensor.float().div(255)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    return ((tensor - mean) / std).unsqueeze(0).to(device)


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
    device = torch.device(
        f"cuda:{config['gpu']['gpu_device_index']}"
        if torch.cuda.is_available() else "cpu"
    )
    model, _ = build_p2pnet(config)
    payload = load_scratch_checkpoint(args.checkpoint)
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    threshold = config["evaluation"]["prediction_confidence_threshold"]

    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_number = 0
    with torch.inference_mode():
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_number % args.frame_sampling == 0:
                outputs = model(frame_tensor(frame, device))
                scores = outputs["pred_logits"].softmax(-1)[0, :, 1]
                valid = scores > threshold
                points = outputs["pred_points"][0, valid].detach().cpu().numpy()
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
            writer.write(frame)
            frame_number += 1

    capture.release()
    writer.release()
    print(f"Saved annotated video: {output}")


if __name__ == "__main__":
    main()
