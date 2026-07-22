"""
LoRA adapter: wraps a frozen nn.Linear with a small trainable low-rank
branch. Used to fine-tune DA3's attention layers (qkv, proj) without
touching the pretrained weights.
"""

import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False

        in_features = base_linear.in_features
        out_features = base_linear.out_features
        self.scale = alpha / rank

        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        base_out = self.base(x)
        lora_out = x @ self.lora_A.T @ self.lora_B.T
        return base_out + self.scale * lora_out


def inject_lora(net, rank: int = 16, alpha: float = 32.0):
    """
    Freezes the whole net, then replaces qkv/proj in every attention
    block with a LoRALinear wrapper. Returns the list of adapted paths
    so we can confirm coverage.
    """
    for p in net.parameters():
        p.requires_grad = False

    adapted = []
    for i, block in enumerate(net.backbone.pretrained.blocks):
        for name in ("qkv", "proj"):
            linear = getattr(block.attn, name)
            setattr(block.attn, name, LoRALinear(linear, rank=rank, alpha=alpha))
            adapted.append(f"blocks[{i}].attn.{name}")

    return adapted
