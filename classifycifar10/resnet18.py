import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Type


class BasicBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.act = torch.nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act(out)
        return out


class ResNet(nn.Module):

    def __init__(
        self,
        layers: list[int],
        num_classes: int = 1000,
        in_channels: int = 3,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Layer 1: 64 -> 64, no spatial downsampling
        self.layer1 = nn.Sequential(*[BasicBlock(64, 64) for _ in range(layers[0])])

        # Layer 2: 64 -> 128, first block halves spatial dims
        self.layer2 = nn.Sequential(
            BasicBlock(
                64,
                128,
                stride=2,
                downsample=nn.Sequential(
                    nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
                    nn.BatchNorm2d(128),
                ),
            ),
            *[BasicBlock(128, 128) for _ in range(1, layers[1])]
        )

        # Layer 3: 128 -> 256, first block halves spatial dims
        self.layer3 = nn.Sequential(
            BasicBlock(
                128,
                256,
                stride=2,
                downsample=nn.Sequential(
                    nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False),
                    nn.BatchNorm2d(256),
                ),
            ),
            *[BasicBlock(256, 256) for _ in range(1, layers[2])]
        )

        # Layer 4: 256 -> 512, first block halves spatial dims
        self.layer4 = nn.Sequential(
            BasicBlock(
                256,
                512,
                stride=2,
                downsample=nn.Sequential(
                    nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False),
                    nn.BatchNorm2d(512),
                ),
            ),
            *[BasicBlock(512, 512) for _ in range(1, layers[3])]
        )

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = torch.nn.Dropout(0.1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem
        x = F.relu(self.bn1(self.conv1(x)))
        # x = self.maxpool(x)

        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Classification head
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
