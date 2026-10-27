"""M1 模型: 3D CNN(主) 与 2.5D CNN(对照)。

创新点在任务/信号(人眼盲区), backbone 用成熟简洁结构, 避免与"信号可学习性"混杂。
CNN3D: 4 个 (conv3x3x2 + BN + ReLu + maxpool2) 块, GAP, FC。输入 (B,1,48,48,48)。
CNN2D: 同构 2D 版, 输入 (B,3,48,48)(中央3切片作通道)。
"""
import torch.nn as nn


def _block3(i, o):
    return nn.Sequential(
        nn.Conv3d(i, o, 3, padding=1), nn.BatchNorm3d(o), nn.ReLU(inplace=True),
        nn.Conv3d(o, o, 3, padding=1), nn.BatchNorm3d(o), nn.ReLU(inplace=True),
        nn.MaxPool3d(2),
    )


def _block2(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class CNN3D(nn.Module):
    def __init__(self, in_ch=1, num_classes=2, base=32):
        super().__init__()
        self.feat = nn.Sequential(
            _block3(in_ch, base), _block3(base, base * 2),
            _block3(base * 2, base * 4), _block3(base * 4, base * 8),
        )
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(base * 8, num_classes)

    def forward(self, x):
        f = self.gap(self.feat(x)).flatten(1)
        return self.fc(self.drop(f))


class CNN2D(nn.Module):
    def __init__(self, in_ch=3, num_classes=2, base=32):
        super().__init__()
        self.feat = nn.Sequential(
            _block2(in_ch, base), _block2(base, base * 2),
            _block2(base * 2, base * 4), _block2(base * 4, base * 8),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(base * 8, num_classes)

    def forward(self, x):
        f = self.gap(self.feat(x)).flatten(1)
        return self.fc(self.drop(f))


def build_model(name, num_classes=2):
    if name == "3d":
        return CNN3D(num_classes=num_classes)
    if name == "2d":
        return CNN2D(num_classes=num_classes)
    raise ValueError(name)
