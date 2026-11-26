import numpy as np
import torch
import torch.nn as nn

from .backbone import RandomInitializedVGG16BN
from .criterion import SetCriterionCrowd
from .matcher import HungarianMatcherCrowd


def generate_anchor_points(stride, row, line):
    row_step = stride / row
    line_step = stride / line
    shift_x = (np.arange(1, line + 1) - 0.5) * line_step - stride / 2
    shift_y = (np.arange(1, row + 1) - 0.5) * row_step - stride / 2
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    return np.vstack((shift_x.ravel(), shift_y.ravel())).transpose()


def generate_shifted_anchors(height, width, stride, row, line, device):
    anchor_points = generate_anchor_points(stride, row, line)
    grid_y = (np.arange(0, height) + 0.5) * stride
    grid_x = (np.arange(0, width) + 0.5) * stride
    shift_x, shift_y = np.meshgrid(grid_x, grid_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel())).transpose()
    anchors = shifts[:, None, :] + anchor_points[None, :, :]
    anchors = anchors.reshape(-1, 2).astype(np.float32)
    return torch.from_numpy(anchors).to(device)


class RegressionModel(nn.Module):
    def __init__(self, num_anchor_points=4, feature_size=256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(256, feature_size, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_size, feature_size, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_size, num_anchor_points * 2, 3, padding=1),
        )

    def forward(self, x):
        out = self.layers(x).permute(0, 2, 3, 1)
        return out.contiguous().view(x.shape[0], -1, 2)


class ClassificationModel(nn.Module):
    def __init__(self, num_anchor_points=4, num_classes=2, feature_size=256):
        super().__init__()
        self.num_anchor_points = num_anchor_points
        self.num_classes = num_classes
        self.layers = nn.Sequential(
            nn.Conv2d(256, feature_size, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_size, feature_size, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_size, num_anchor_points * num_classes, 3, padding=1),
        )

    def forward(self, x):
        out = self.layers(x).permute(0, 2, 3, 1)
        out = out.view(
            x.shape[0],
            out.shape[1],
            out.shape[2],
            self.num_anchor_points,
            self.num_classes,
        )
        return out.contiguous().view(x.shape[0], -1, self.num_classes)


class Decoder(nn.Module):
    def __init__(self, feature_size=256):
        super().__init__()
        self.p5_1 = nn.Conv2d(512, feature_size, 1)
        self.p5_2 = nn.Conv2d(feature_size, feature_size, 3, padding=1)
        self.p4_1 = nn.Conv2d(512, feature_size, 1)
        self.p4_2 = nn.Conv2d(feature_size, feature_size, 3, padding=1)
        self.p3_1 = nn.Conv2d(256, feature_size, 1)
        self.p3_2 = nn.Conv2d(feature_size, feature_size, 3, padding=1)

    def forward(self, c3, c4, c5):
        p5 = self.p5_2(self.p5_1(c5))
        p4 = self.p4_1(c4) + torch.nn.functional.interpolate(
            p5, size=c4.shape[-2:], mode="nearest"
        )
        p4 = self.p4_2(p4)
        p3 = self.p3_1(c3) + torch.nn.functional.interpolate(
            p4, size=c3.shape[-2:], mode="nearest"
        )
        p3 = self.p3_2(p3)
        return p3, p4, p5


class P2PNet(nn.Module):
    def __init__(self, row=2, line=2, num_classes=1, pretrained=False):
        super().__init__()
        if pretrained:
            raise RuntimeError("Pretrained P2PNet initialization is forbidden")
        self.backbone = RandomInitializedVGG16BN()
        self.fpn = Decoder()
        num_anchor_points = row * line
        self.regression = RegressionModel(num_anchor_points)
        self.classification = ClassificationModel(
            num_anchor_points=num_anchor_points,
            num_classes=num_classes + 1,
        )
        self.row = row
        self.line = line
        self.num_classes = num_classes

    def forward(self, samples):
        features = self.backbone(samples)
        p3, p4, p5 = self.fpn(features[2], features[3], features[4])
        regression = self.regression(p4) * 100
        classification = self.classification(p4)
        _, _, height, width = p4.shape
        anchors = generate_shifted_anchors(
            height,
            width,
            stride=8,
            row=self.row,
            line=self.line,
            device=samples.device,
        )
        anchors = anchors.unsqueeze(0).repeat(samples.shape[0], 1, 1)
        return {
            "pred_logits": classification,
            "pred_points": regression + anchors,
        }


def build_p2pnet(config):
    model_config = config["model"]
    if model_config.get("pretrained", False):
        raise RuntimeError("Config explicitly enables pretrained weights")
    if config["training"].get("mode") != "scratch":
        raise RuntimeError("Only scratch training is permitted")
    model = P2PNet(
        row=model_config["row"],
        line=model_config["line"],
        num_classes=model_config["num_classes"],
        pretrained=False,
    )
    matcher = HungarianMatcherCrowd(
        cost_class=model_config["set_cost_class"],
        cost_point=model_config["set_cost_point"],
    )
    criterion = SetCriterionCrowd(
        num_classes=model_config["num_classes"],
        matcher=matcher,
        eos_coef=model_config["eos_coef"],
        point_loss_coef=model_config["point_loss_coef"],
    )
    return model, criterion
