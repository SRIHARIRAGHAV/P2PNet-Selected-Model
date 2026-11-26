import math

import torch
from torch import nn
from torch.nn import functional as F

from .backbone import VGG16BNBackbone


def _initialize_randomly(module):
    for layer in module.modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(
                layer.weight,
                mode="fan_out",
                nonlinearity="relu",
            )
            if layer.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                bound = 1.0 / math.sqrt(fan_in)
                nn.init.uniform_(layer.bias, -bound, bound)
        elif isinstance(layer, nn.BatchNorm2d):
            nn.init.normal_(layer.weight, mean=1.0, std=0.02)
            nn.init.normal_(layer.bias, mean=0.0, std=0.02)


class FeaturePyramidNetwork(nn.Module):
    def __init__(self, feature_size=256):
        super().__init__()
        self.p5_lateral = nn.Conv2d(512, feature_size, kernel_size=1)
        self.p5_output = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.p4_lateral = nn.Conv2d(512, feature_size, kernel_size=1)
        self.p4_output = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.p3_lateral = nn.Conv2d(256, feature_size, kernel_size=1)
        self.p3_output = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)

    def forward(self, c3, c4, c5):
        p5 = self.p5_output(self.p5_lateral(c5))
        p4 = self.p4_lateral(c4) + F.interpolate(
            p5,
            size=c4.shape[-2:],
            mode="nearest",
        )
        p4 = self.p4_output(p4)
        p3 = self.p3_lateral(c3) + F.interpolate(
            p4,
            size=c3.shape[-2:],
            mode="nearest",
        )
        p3 = self.p3_output(p3)
        return p3


class PredictionHead(nn.Module):
    def __init__(self, output_channels, feature_size=256):
        super().__init__()
        layers = []
        for _ in range(3):
            layers.extend([
                nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ])
        layers.append(
            nn.Conv2d(feature_size, output_channels, kernel_size=3, padding=1)
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, features):
        return self.layers(features)


def generate_shifted_anchors(height, width, stride, row, line, device, dtype):
    anchor_x = (
        (torch.arange(line, device=device, dtype=dtype) + 0.5) * stride / line
        - stride / 2
    )
    anchor_y = (
        (torch.arange(row, device=device, dtype=dtype) + 0.5) * stride / row
        - stride / 2
    )
    offset_y, offset_x = torch.meshgrid(anchor_y, anchor_x, indexing="ij")
    offsets = torch.stack((offset_x.reshape(-1), offset_y.reshape(-1)), dim=1)

    shift_x = (torch.arange(width, device=device, dtype=dtype) + 0.5) * stride
    shift_y = (torch.arange(height, device=device, dtype=dtype) + 0.5) * stride
    grid_y, grid_x = torch.meshgrid(shift_y, shift_x, indexing="ij")
    shifts = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=1)
    return (shifts[:, None, :] + offsets[None, :, :]).reshape(-1, 2)


class P2PNet(nn.Module):
    """Point-to-point crowd detector with a VGG16-BN/FPN feature extractor."""

    def __init__(self, row=2, line=2, num_classes=1, pretrained=False):
        super().__init__()
        if pretrained:
            raise RuntimeError("Pretrained P2PNet initialization is forbidden")
        if row < 1 or line < 1 or num_classes < 1:
            raise ValueError("row, line, and num_classes must be positive")

        self.backbone = VGG16BNBackbone(pretrained=False)
        self.fpn = FeaturePyramidNetwork()
        self.row = row
        self.line = line
        self.num_classes = num_classes
        self.num_anchor_points = row * line
        self.classification = PredictionHead(
            self.num_anchor_points * (num_classes + 1)
        )
        self.regression = PredictionHead(self.num_anchor_points * 2)
        _initialize_randomly(self)

    def forward(self, images):
        c3, c4, c5 = self.backbone(images)
        features = self.fpn(c3, c4, c5)
        batch_size, _, height, width = features.shape

        logits = self.classification(features)
        logits = logits.permute(0, 2, 3, 1).contiguous()
        logits = logits.view(
            batch_size,
            height * width * self.num_anchor_points,
            self.num_classes + 1,
        )

        offsets = self.regression(features)
        offsets = offsets.permute(0, 2, 3, 1).contiguous()
        offsets = offsets.view(batch_size, -1, 2) * 100.0
        anchors = generate_shifted_anchors(
            height,
            width,
            stride=8,
            row=self.row,
            line=self.line,
            device=features.device,
            dtype=features.dtype,
        )
        points = offsets + anchors.unsqueeze(0)
        return {"pred_logits": logits, "pred_points": points}
