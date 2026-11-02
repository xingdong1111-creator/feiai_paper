"""Path B (grounding): ViT3D + 空间池化(保留3D排布) + projector + Qwen2.5(手写LoRA)。
与 stage2 的区别: 视觉 token 保留空间结构(供 LLM 定位), 坐标是【输出】不是输入,
LLM 生成 JSON 文字 [{coord:[x,y,z], r, na, sub}, ...]。一次输入(整例CT) -> 一次生成(全部结节)。
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .vit3d import ViT3D
from .resampler import Projector


# ----------------------------- 手写 LoRA -----------------------------
class LoRALinear(nn.Module):
    """包裹一个冻结的 nn.Linear, 旁路低秩适配器 A(r,in)@B(out,r)。base 冻结, 只训 A/B。"""

    def __init__(self, base_linear, r=16, alpha=32, dropout=0.0):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad_(False)
        in_f = base_linear.in_features
        out_f = base_linear.out_features
        self.lora_A = nn.Parameter(torch.empty(r, in_f))    # fp32, 训练
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        wd = self.base.weight.dtype          # base 可能是 fp16(冻结半精度)而 x 是 fp32
        base_out = self.base(x.to(wd))       # 显式对齐, 防 CPU回退/检查点重算时 autocast 不生效
        lora = (self.dropout(x).float() @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
        return base_out + lora.to(base_out.dtype)


def inject_lora(llm, r=16, alpha=32, targets=("q_proj", "k_proj", "v_proj", "o_proj")):
    """把 LLM 注意力的 q/k/v/o 投影替换成 LoRALinear。返回替换数。"""
    replaced = 0
    for _, module in llm.named_modules():
        for name, child in list(module.named_children()):
            if name in targets and isinstance(child, nn.Linear):
                setattr(module, name, LoRALinear(child, r, alpha))
                replaced += 1
    return replaced


# --------------------------- 空间池化(保留3D排布) ---------------------------
class SpatialPool(nn.Module):
    """ViT patch tokens (B,N,dim)[按(D',H',W')光栅序] -> AdaptiveAvgPool3d 到 out_grid,
    加 3D 位置编码, 保留空间排布供 LLM 定位。输出 (B, M, dim), M=pd*ph*pw。"""

    def __init__(self, dim, src_grid, out_grid=(4, 6, 6)):
        super().__init__()
        self.src_grid = tuple(src_grid)
        self.out_grid = tuple(out_grid)
        self.dim = dim
        self.pool = nn.AdaptiveAvgPool3d(self.out_grid)
        self.norm = nn.LayerNorm(dim)
        M = self.out_grid[0] * self.out_grid[1] * self.out_grid[2]
        self.pos = nn.Parameter(torch.zeros(1, M, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x):                       # (B,N,dim)
        B, N, C = x.shape
        D, H, W = self.src_grid
        x = x.transpose(1, 2).reshape(B, C, D, H, W)
        x = self.pool(x)                        # (B,C,pd,ph,pw)
        x = x.flatten(2).transpose(1, 2)        # (B,M,C) 光栅序
        return self.norm(x + self.pos)


# ------------------------------ GroundingVLM ------------------------------
class GroundingVLM(nn.Module):
    def __init__(self, llm_path, img_grid=(128, 192, 192), patch=(16, 16, 16),
                 dim=384, depth=8, heads=6, vis_grid=(4, 6, 6),
                 lora_r=16, lora_alpha=32, llm_dtype=torch.float32):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.llm = AutoModelForCausalLM.from_pretrained(llm_path, dtype=llm_dtype)
        for p in self.llm.parameters():         # 冻结 base, 仅 LoRA 可训
            p.requires_grad_(False)
        self.llm.eval()
        llm_dim = self.llm.get_input_embeddings().weight.shape[1]

        self.vit = ViT3D(img_grid=img_grid, patch=patch, dim=dim, depth=depth, heads=heads)
        self.pool = SpatialPool(dim, self.vit.grid_tok, vis_grid)
        self.projector = Projector(dim=dim, llm_dim=llm_dim)
        self.num_vis = vis_grid[0] * vis_grid[1] * vis_grid[2]
        self.llm_dim = llm_dim
        self.n_lora = inject_lora(self.llm, lora_r, lora_alpha)

    def encode_image(self, volume):
        tok = self.vit(volume)                  # (B,N,dim)
        sp = self.pool(tok)                     # (B,M,dim) 保留空间
        return self.projector(sp)               # (B,M,llm_dim)

    def _merge(self, volume, input_ids, attention_mask):
        vis = self.encode_image(volume)                                   # (B,M,D)
        B, M, D = vis.shape
        text = self.llm.get_input_embeddings()(input_ids).to(vis.dtype)   # (B,L,D)
        inputs_embeds = torch.cat([vis, text], dim=1)
        vmask = torch.ones(B, M, dtype=attention_mask.dtype, device=attention_mask.device)
        am = torch.cat([vmask, attention_mask], dim=1)
        return inputs_embeds, am, M

    def forward(self, volume, input_ids, attention_mask, labels=None):
        inputs_embeds, am, M = self._merge(volume, input_ids, attention_mask)
        if labels is not None:
            B = labels.size(0)
            vlab = torch.full((B, M), -100, dtype=labels.dtype, device=labels.device)
            labels = torch.cat([vlab, labels], dim=1)     # 视觉段不计 loss
        return self.llm(inputs_embeds=inputs_embeds, attention_mask=am, labels=labels)

    @torch.no_grad()
    def generate(self, volume, input_ids, attention_mask, max_new_tokens=256, **kw):
        inputs_embeds, am, M = self._merge(volume, input_ids, attention_mask)
        return self.llm.generate(inputs_embeds=inputs_embeds, attention_mask=am,
                                 max_new_tokens=max_new_tokens,
                                 pad_token_id=self.tokenizer.pad_token_id, **kw)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]   # vit+pool+projector+LoRA


def build_groundingvlm(llm_path, img_grid=(128, 192, 192), **kw):
    return GroundingVLM(llm_path, img_grid=img_grid, **kw)
