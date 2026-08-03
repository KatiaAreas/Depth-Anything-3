"""
LoRA adapter: wraps a frozen nn.Linear with a small trainable low-rank
branch. Used to fine-tune DA3's attention layers (qkv, proj) without
touching the pretrained weights.

Two modes:
  - Uniform LoRA (original): every block gets the same rank.
  - Vector-LoRA: each block gets its own rank, following the principle
    from DARES (Zeinoddin et al., 2024, arXiv:2408.17433): earlier layers
    learn more general features and get more parameters, later layers get
    fewer. DARES's own per-layer rank values weren't accessible to me
    (their repo blocks automated access to the relevant file), so this is
    my own linearly decreasing schedule following their stated principle,
    not a reproduction of their exact numbers. The schedule is built to
    average out to the same rank as the uniform baseline, so both use
    roughly the same total number of trainable parameters, keeping the
    comparison fair.
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
        self.rank = rank
        self.scale = alpha / rank

        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        base_out = self.base(x)
        lora_out = x @ self.lora_A.T @ self.lora_B.T
        return base_out + self.scale * lora_out


def make_vector_lora_ranks(num_blocks: int, rank_max: int, rank_min: int) -> list[int]:
    """
    Linearly decreasing rank per block: block 0 (earliest) gets rank_max,
    the last block gets rank_min, evenly spaced in between.

    Average rank = (rank_max + rank_min) / 2. Pick rank_max/rank_min so
    this average matches whatever uniform rank you're comparing against,
    e.g. rank_max=24, rank_min=8 averages to 16, the same as our uniform
    baseline.
    """
    if num_blocks <= 1:
        return [rank_max]
    step = (rank_max - rank_min) / (num_blocks - 1)
    return [round(rank_max - i * step) for i in range(num_blocks)]


def inject_lora(
    net,
    rank: int = 16,
    alpha: float = 32.0,
    ranks: list[int] | None = None,
    alpha_to_rank_ratio: float = 2.0,
) -> list[str]:
    """
    Freezes the whole net, then replaces qkv/proj in every attention
    block with a LoRALinear wrapper.

    rank, alpha: used for every block (uniform LoRA), ignored if `ranks`
                 is given.
    ranks: optional list of per-block ranks (Vector-LoRA), one entry per
           attention block, in order. alpha for each block is scaled to
           keep alpha/rank constant (alpha_to_rank_ratio) across blocks,
           matching the ratio used in the uniform case (32/16 = 2.0).

    Returns the list of adapted module paths (with the rank used at each,
    for a quick sanity check on the actual schedule applied).
    """
    for p in net.parameters():
        p.requires_grad = False

    adapted = []
    blocks = net.backbone.pretrained.blocks

    if ranks is not None:
        assert len(ranks) == len(blocks), (
            f"ranks list has {len(ranks)} entries but there are {len(blocks)} blocks"
        )

    for i, block in enumerate(blocks):
        block_rank = ranks[i] if ranks is not None else rank
        block_alpha = block_rank * alpha_to_rank_ratio if ranks is not None else alpha
        attn = block.attn
        for name in ("qkv", "proj"):
            if not hasattr(attn, name):
                continue
            linear = getattr(attn, name)
            if not isinstance(linear, nn.Linear):
                continue
            setattr(attn, name, LoRALinear(linear, rank=block_rank, alpha=block_alpha))
            adapted.append(f"blocks[{i}].attn.{name} (r={block_rank})")

    return adapted


def unfreeze_head(net) -> int:
    count = 0
    for p in net.head.parameters():
        p.requires_grad = True
        count += p.numel()
    return count


def trainable_parameter_summary(net) -> dict:
    total = sum(p.numel() for p in net.parameters())
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "trainable_pct": 100.0 * trainable / total,
    }
