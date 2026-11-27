import argparse
from pathlib import Path

import cv2
import torch
from PIL import Image

from config import load_config
from dataset.transforms import resize_for_model, transform_image
from models import P2PNet
from utils.checkpoint import load_scratch_checkpoint
from utils.device import amp_settings, autocast_context, select_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/p2pnet_scratch.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="outputs/predictions/inference.jpg")
    args = parser.parse_args()

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

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(args.image)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized, scale_x, scale_y = resize_for_model(
        Image.fromarray(rgb),
        int(config["dataset"]["validation_max_side"]),
    )
    tensor = transform_image(resized)
    tensor = tensor.unsqueeze(0).to(device)

    with torch.inference_mode():
        with autocast_context(amp_enabled, amp_dtype, device):
            outputs = model(tensor)
        if (
            not torch.isfinite(outputs["pred_logits"]).all()
            or not torch.isfinite(outputs["pred_points"]).all()
        ):
            raise RuntimeError("Model produced NaN or Inf during inference.")
        scores = outputs["pred_logits"].float().softmax(-1)[0, :, 1]
        threshold = config["evaluation"]["prediction_confidence_threshold"]
        valid = scores > threshold
        points = outputs["pred_points"][0, valid].float().detach().cpu().numpy()
        points[:, 0] *= scale_x
        points[:, 1] *= scale_y
        confidence = scores[valid].detach().cpu().numpy()

    for point, score in zip(points, confidence):
        x = int(round(point[0]))
        y = int(round(point[1]))
        cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
    cv2.putText(
        image,
        f"Crowd count: {len(points)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
    )
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite inference output: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), image):
        raise OSError(f"Could not write inference output: {output}")
    print(f"Predicted crowd count: {len(points)}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
