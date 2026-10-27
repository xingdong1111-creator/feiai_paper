"""3D 体积编码器: 整例 CT (B,1,D,H,W) -> 特征图 (B,C,D/16,H/16,W/16)。
strided 3D-CNN, 总下采样 16 倍; 从零训练(后续可换 Merlin/LCTfound 预训练权重)。
"""
import torch
import torch.nn as nn


def block(cin, cout, stride):
    return nn.Sequential(
        nn.Conv3d(cin, cout, 3, stride=stride, padding=1, bias=False), nn.BatchNorm3d(cout), nn.GELU(),
        nn.Conv3d(cout, cout, 3, stride=1, padding=1, bias=False), nn.BatchNorm3d(cout), nn.GELU(),
    )


class Encoder3D(nn.Module):
    def __init__(self, in_ch=1, base=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_ch, base, 3, stride=1, padding=1, bias=False), nn.BatchNorm3d(base), nn.GELU())
        self.b1 = block(base, base, 2)        # /2
        self.b2 = block(base, base * 2, 2)    # /4
        self.b3 = block(base * 2, base * 4, 2)  # /8
        self.b4 = block(base * 4, base * 8, 2)  # /16
        self.out_ch = base * 8

    def forward(self, x):
        x = self.stem(x)
        x = self.b1(x); x = self.b2(x); x = self.b3(x); x = self.b4(x)
        return x   # (B, base*8, D/16, H/16, W/16)
