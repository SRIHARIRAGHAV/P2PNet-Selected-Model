import torch
from torch import nn


def make_vgg16_bn():
    cfg = [
        64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
        512, 512, 512, "M", 512, 512, 512, "M",
    ]
    layers = []
    in_channels = 3
    for value in cfg:
        if value == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers.append(nn.Conv2d(in_channels, value, 3, padding=1))
            layers.append(nn.BatchNorm2d(value))
            layers.append(nn.ReLU(inplace=True))
            in_channels = value
    return nn.Sequential(*layers)


class RandomInitializedVGG16BN(nn.Module):
    def __init__(self):
        super().__init__()
        features = list(make_vgg16_bn().children())
        self.body1 = nn.Sequential(*features[:13])
        self.body2 = nn.Sequential(*features[13:23])
        self.body3 = nn.Sequential(*features[23:33])
        self.body4 = nn.Sequential(*features[33:43])

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x1 = self.body1(x)
        x2 = self.body2(x1)
        x3 = self.body3(x2)
        x4 = self.body4(x3)
        return [x, x1, x2, x3, x4]
