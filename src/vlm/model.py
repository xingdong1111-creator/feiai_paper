"""one-in-one-out 骨架: 整例体积进, 每结节坐标条件采样 + 分类头, 一次前向输出全结节 logits。
坐标条件 query = 把结节归一化坐标映射到 3D 特征图上做 grid_sample 取局部特征,
拼上坐标嵌入与全局上下文, 经 MLP 出 blind/consensus logits。无 LLM(阶段1)。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F_
from .encoder3d import Encoder3D


class CoordQueryHead(nn.Module):
    def __init__(self, feat_ch, coord_dim=3, coord_emb=64, hidden=256, num_classes=2):
        super().__init__()
        self.coord_mlp = nn.Sequential(
            nn.Linear(coord_dim, coord_emb), nn.GELU(),
            nn.Linear(coord_emb, coord_emb), nn.GELU())
        # 局部特征(feat_ch) + 全局上下文(feat_ch) + 坐标嵌入(coord_emb)
        self.head = nn.Sequential(
            nn.Linear(feat_ch * 2 + coord_emb, hidden), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden, num_classes))

    def forward(self, feat, coords):
        """feat: (B,C,D,H,W); coords: (B,N,3) 归一化 (x,y,z)∈[0,1]; 返回 (B,N,num_classes)。"""
        B, C, D, H, W = feat.shape
        N = coords.shape[1]
        # grid_sample 3D: grid (B,N,1,1,3), 末维 (x,y,z)∈[-1,1]
        grid = coords * 2.0 - 1.0
        grid = grid.view(B, N, 1, 1, 3)
        sampled = F_.grid_sample(feat, grid, mode="bilinear", align_corners=False)  # (B,C,N,1,1)
        sampled = sampled.view(B, C, N).transpose(1, 2)   # (B,N,C)
        glob = feat.mean(dim=(2, 3, 4)).unsqueeze(1).expand(B, N, C)   # (B,N,C) 全局上下文
        cemb = self.coord_mlp(coords)   # (B,N,coord_emb)
        h = torch.cat([sampled, glob, cemb], dim=-1)
        return self.head(h)   # (B,N,num_classes)


class OneInOutNet(nn.Module):
    def __init__(self, base=32, coord_emb=64, num_classes=2):
        super().__init__()
        self.encoder = Encoder3D(in_ch=1, base=base)
        self.head = CoordQueryHead(self.encoder.out_ch, coord_emb=coord_emb, num_classes=num_classes)

    def forward(self, volume, coords):
        feat = self.encoder(volume)        # (B,C,D/16,H/16,W/16)
        logits = self.head(feat, coords)   # (B,N,num_classes)
        return logits


def build_oneinout(base=32, num_classes=2):
    return OneInOutNet(base=base, num_classes=num_classes)
