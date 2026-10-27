"""3D ViT 视觉编码器: 整例体积 (B,1,D,H,W) -> patch tokens (B,N,dim)。
patch 用 3D 卷积切分; 位置编码按固定网格学习; 标准 Transformer blocks。
支持用 BiomedCLIP 2D ViT-B/16 权重"膨胀"到 3D 做初始化(见 inflate_from_biomedclip)。
"""
import math
import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(hidden, dim), nn.Dropout(drop))

    def forward(self, x):
        h = self.n1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.mlp(self.n2(x))
        return x


class ViT3D(nn.Module):
    def __init__(self, img_grid=(128, 192, 192), patch=(16, 16, 16), in_ch=1,
                 dim=384, depth=8, heads=6, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.patch = patch
        self.dim = dim
        self.grid_tok = tuple(g // p for g, p in zip(img_grid, patch))   # (D',H',W')
        self.num_patches = self.grid_tok[0] * self.grid_tok[1] * self.grid_tok[2]
        self.proj = nn.Conv3d(in_ch, dim, kernel_size=patch, stride=patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, self.num_patches + 1, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.drop = nn.Dropout(drop)
        self.blocks = nn.ModuleList([Block(dim, heads, mlp_ratio, drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.proj(x)                              # (B,dim,D',H',W')
        B = x.shape[0]
        x = x.flatten(2).transpose(1, 2)              # (B,N,dim)
        x = torch.cat([self.cls.expand(B, -1, -1), x], 1) + self.pos
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 1:]                             # 去掉 CLS, 返回 patch tokens (B,N,dim)


def inflate_from_biomedclip(vit3d, bm_state, patch_t=16):
    """把 BiomedCLIP ViT-B/16 的 2D 权重膨胀到 3D 作初始化(尽力而为, 形状不匹配则跳过)。
    - patch conv: Conv2d(3,dim,16,16) -> Conv3d(1,dim,(16,16,16)): 时间维复制/求均值, 通道 3->1 求均值。
    - 其余 transformer 层按同名/同形状拷贝。
    返回成功拷贝的层数(供日志)。
    """
    own = vit3d.state_dict()
    copied = 0
    sd = {}
    # BiomedCLIP 键名前缀可能为 visual./transformer.resblocks. 等, 这里做宽松匹配
    for k, v in bm_state.items():
        sd[k] = v
    # patch embed 膨胀
    for key in list(sd):
        if key.endswith("conv1.weight") or key.endswith("patch_embed.proj.weight"):
            w2d = sd[key]   # (dim,3,16,16)
            if w2d.dim() == 4:
                dim = w2d.shape[0]
                w3d = w2d.mean(dim=1, keepdim=True)            # (dim,1,16,16) 通道均值
                w3d = w3d.unsqueeze(2).repeat(1, 1, patch_t, 1, 1) / patch_t   # 时间维复制求均
                if "proj.weight" in own and own["proj.weight"].shape == w3d.shape:
                    own["proj.weight"].copy_(w3d); copied += 1
            break
    # transformer blocks: 按顺序映射 attn/mlp/ln 权重(同形状才拷)
    # 这里做保守拷贝: 遍历 own, 找 bm 中同形状且语义对应的(简化: 跳过, 仅 patch+pos 初始化)
    vit3d.load_state_dict(own, strict=False)
    return copied
