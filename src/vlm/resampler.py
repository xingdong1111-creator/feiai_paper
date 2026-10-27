"""Perceiver 重采样器 + projector: 把 ViT 的变长 patch tokens 压成固定 K 个视觉 token,
再投影到 LLM(Qwen2.5) 的词嵌入维度。这是"整例 -> 少量 token -> LLM"的关键压缩环节。
"""
import torch
import torch.nn as nn


class ResamplerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.ln_q = nn.LayerNorm(dim)
        self.ln_kv = nn.LayerNorm(dim)
        self.cross = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, q, kv):
        hq = self.ln_q(q); hkv = self.ln_kv(kv)
        a, _ = self.cross(hq, hkv, hkv, need_weights=False)
        q = q + a
        q = q + self.mlp(self.ln2(q))
        return q


class PerceiverResampler(nn.Module):
    """K 个可学习 query 对 ViT tokens 做 cross-attention, 输出固定 (B,K,dim)。"""
    def __init__(self, dim, num_queries=64, depth=4, heads=8, mlp_ratio=4.0):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)
        self.blocks = nn.ModuleList([ResamplerBlock(dim, heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):                  # x: (B,N,dim)
        B = x.shape[0]
        q = self.query.expand(B, -1, -1)
        for blk in self.blocks:
            q = blk(q, x)
        return self.norm(q)                # (B,K,dim)


class Projector(nn.Module):
    """视觉 token dim -> LLM 词嵌入 dim (Qwen2.5-1.5B=1536, 3B=2048, 7B=3584)。"""
    def __init__(self, dim, llm_dim, hidden=None):
        super().__init__()
        hidden = hidden or llm_dim
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, llm_dim))

    def forward(self, x):
        return self.net(x)
