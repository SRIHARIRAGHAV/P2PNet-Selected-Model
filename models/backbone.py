from torch import nn


def _vgg_block(in_channels, out_channels, convolutions):
    layers = []
    for _ in range(convolutions):
        layers.extend([
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ])
        in_channels = out_channels
    return nn.Sequential(*layers)


class VGG16BNBackbone(nn.Module):
    """VGG16-BN feature extractor returning stride-8, -16, and -32 maps."""

    def __init__(self, pretrained=False):
        super().__init__()
        if pretrained:
            raise RuntimeError("Pretrained backbone weights are forbidden")

        self.stage1 = _vgg_block(3, 64, 2)
        self.stage2 = _vgg_block(64, 128, 2)
        self.stage3 = _vgg_block(128, 256, 3)
        self.stage4 = _vgg_block(256, 512, 3)
        self.stage5 = _vgg_block(512, 512, 3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, images):
        features = self.pool(self.stage1(images))
        features = self.pool(self.stage2(features))
        c3 = self.pool(self.stage3(features))
        c4 = self.pool(self.stage4(c3))
        c5 = self.pool(self.stage5(c4))
        return c3, c4, c5
