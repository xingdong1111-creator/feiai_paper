"""阶段2: 3D ViT + Perceiver 重采样 + projector + 冻结 Qwen2.5(LLaVA 式接线)。
一次输入(整例体积) -> K 个视觉 token -> 前置拼到文本嵌入 -> Qwen2.5 一次生成 per-nodule JSON。
只训视觉侧(ViT/重采样/projector), LLM 冻结(后续可加 LoRA)。
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .vit3d import ViT3D
from .resampler import PerceiverResampler, Projector


class QwenVLM(nn.Module):
    def __init__(self, llm_path, img_grid=(128, 192, 192), patch=(16, 16, 16),
                 dim=384, depth=8, heads=6, num_queries=64, resampler_depth=4,
                 freeze_llm=True, llm_dtype=torch.float32):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.llm = AutoModelForCausalLM.from_pretrained(llm_path, dtype=llm_dtype)
        if freeze_llm:
            for p in self.llm.parameters():
                p.requires_grad_(False)
            self.llm.eval()
        llm_dim = self.llm.get_input_embeddings().weight.shape[1]   # 自动取 hidden(1.5B=1536)
        self.vit = ViT3D(img_grid=img_grid, patch=patch, dim=dim, depth=depth, heads=heads)
        self.resampler = PerceiverResampler(dim=dim, num_queries=num_queries,
                                            depth=resampler_depth, heads=heads)
        self.projector = Projector(dim=dim, llm_dim=llm_dim)
        self.llm_dim = llm_dim
        self.num_queries = num_queries
        # last-hidden-state 读出头：每个结节坐标位置的隐状态 -> 类别(blind/consensus) + 隐蔽度回归
        self.head = nn.Sequential(nn.Linear(llm_dim, 512), nn.GELU(), nn.Linear(512, 2))
        self.subtlety_head = nn.Sequential(nn.Linear(llm_dim, 256), nn.GELU(), nn.Linear(256, 1))

    def encode_image(self, volume):
        tok = self.vit(volume)              # (B,N,dim)
        q = self.resampler(tok)             # (B,K,dim)
        return self.projector(q)            # (B,K,llm_dim)

    def forward(self, volume, input_ids, attention_mask, labels=None):
        vis = self.encode_image(volume)                       # (B,K,D)
        B, K, D = vis.shape
        emb = self.llm.get_input_embeddings()
        text = emb(input_ids).to(vis.dtype)                   # (B,L,D)
        inputs_embeds = torch.cat([vis, text], dim=1)         # 视觉前置
        vmask = torch.ones(B, K, dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask = torch.cat([vmask, attention_mask], dim=1)
        if labels is not None:
            vlab = torch.full((B, K), -100, dtype=labels.dtype, device=labels.device)
            labels = torch.cat([vlab, labels], dim=1)         # 视觉段不计 loss
        return self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)

    def forward_head(self, volume, input_ids, attention_mask, readout_idx):
        """last-hidden-state 读出模式（你的方案）：不生成文字，而是在每个结节坐标
        token 位置取隐状态 -> 小头预测。readout_idx: (B,N) 每个结节在文本序列中的 token 位置。
        返回 logits(B,N,2) 与 subtlety(B,N)。"""
        vis = self.encode_image(volume)                                  # (B,K,D)
        text = self.llm.get_input_embeddings()(input_ids).to(vis.dtype)  # (B,L,D)
        inputs_embeds = torch.cat([vis, text], dim=1)                    # (B,K+L,D)
        K = vis.size(1)
        am = torch.cat([torch.ones(vis.shape[:2], dtype=attention_mask.dtype, device=vis.device),
                        attention_mask], dim=1)
        h = self.llm.model(inputs_embeds=inputs_embeds, attention_mask=am).last_hidden_state  # (B,K+L,D)
        pos = (K + readout_idx).clamp(max=h.size(1) - 1)                 # (B,N) 全序列位置
        bidx = torch.arange(h.size(0), device=h.device).unsqueeze(1).expand(-1, pos.size(1))
        h_n = h[bidx, pos]                                               # (B,N,D)
        logits = self.head(h_n)                                          # (B,N,2)
        subtlety = self.subtlety_head(h_n).squeeze(-1)                   # (B,N)
        return logits, subtlety

    def trainable_parameters(self):
        mods = [self.vit, self.resampler, self.projector, self.head, self.subtlety_head]
        return [p for m in mods for p in m.parameters() if p.requires_grad]


def build_qwenvlm(llm_path, img_grid=(128, 192, 192), **kw):
    return QwenVLM(llm_path, img_grid=img_grid, **kw)
