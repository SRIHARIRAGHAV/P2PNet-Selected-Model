import argparse
from pathlib import Path

import cv2
import torch

from config import load_config
from model.p2pnet import build_p2pnet
from utils.checkpoint import load_scratch_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/p2pnet_scratch.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="outputs/predictions/inference.jpg")
    args = parser.parse_args()

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

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(args.image)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    tensor = (tensor - mean) / std
    tensor = tensor.unsqueeze(0).to(device)

    with torch.inference_mode():
        outputs = model(tensor)
        scores = outputs["pred_logits"].softmax(-1)[0, :, 1]
        threshold = config["evaluation"]["prediction_confidence_threshold"]
        valid = scores > threshold
        points = outputs["pred_points"][0, valid].detach().cpu().numpy()
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
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), image)
    print(f"Predicted crowd count: {len(points)}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
