"""
LoRA adapter for DA3's DINOv2 backbone.

Two distinct modes, kept separate rather than sharing one mechanism,
since they really are different methods:

  - Uniform LoRA (inject_lora / LoRALinear): every block gets the same
    rank, applied to the entire fused qkv output (q, k, and v alike)
    plus the output projection (proj). Includes the standard alpha/rank
    scaling from the original LoRA paper (Hu et al., 2021).

  - Vector-LoRA (inject_vector_lora / QKVLoRA): a faithful port of
    DARES's actual method (Zeinoddin et al., 2024, arXiv:2408.17433),
    confirmed directly against their source (networks/dares.py,
    LoRAInitializer / _LoRA_qkv classes):
      - only q and v get adapted, k and the output projection are
        left untouched entirely
      - no alpha/rank scaling; the delta is added raw (W += deltaW)
      - rank varies per block: their schedule is
        r = [14,14,12,12,10,10,8,8,8,8,8,8], one value per block, over
        their 12-block backbone. DA3's backbone has 24 blocks, so each
        of their 12 values is repeated across 2 consecutive blocks here,
        preserving the exact taper shape and endpoints (14 down to 8).
    Their code assumes separate .query/.key/.value nn.Linear sublayers
    (true in their HF transformers backbone); DA3's qkv is a single
    fused Linear(dim, 3*dim), so QKVLoRA below adds the q and v deltas
    to the corresponding thirds of that fused output instead.

Because Vector-LoRA here skips k and proj entirely, its total trainable
parameter count is NOT the same as the uniform baseline (about 41.7% of
it, given the same backbone width) -- this is a real property of
faithfully reproducing DARES's method, not a bug to budget-match away.
"""

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Uniform LoRA (original implementation, unchanged)
# ---------------------------------------------------------------------------

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


def inject_lora(net, rank: int = 16, alpha: float = 32.0) -> list[str]:
    """Uniform LoRA: same rank everywhere, applied to qkv (all of q,k,v) and proj."""
    for p in net.parameters():
        p.requires_grad = False

    adapted = []
    blocks = net.backbone.pretrained.blocks
    for i, block in enumerate(blocks):
        attn = block.attn
        for name in ("qkv", "proj"):
            if not hasattr(attn, name):
                continue
            linear = getattr(attn, name)
            if not isinstance(linear, nn.Linear):
                continue
            setattr(attn, name, LoRALinear(linear, rank=rank, alpha=alpha))
            adapted.append(f"blocks[{i}].attn.{name} (r={rank})")

    return adapted


# ---------------------------------------------------------------------------
# Vector-LoRA (faithful DARES port, for a fused qkv layer)
# ---------------------------------------------------------------------------

# DARES's actual schedule (networks/dares.py, LoRAInitializer default),
# one value per block, over their 12-block backbone.
DARES_RANK_SCHEDULE_12_BLOCKS = [14, 14, 12, 12, 10, 10, 8, 8, 8, 8, 8, 8]


def make_dares_rank_schedule(num_blocks: int) -> list[int]:
    """
    Stretches DARES's 12-value schedule to `num_blocks` by repeating each
    value across an equal number of consecutive blocks (e.g. each value
    twice, for a 24-block backbone), preserving the exact taper shape
    and endpoints rather than re-deriving a new curve.
    """
    base = DARES_RANK_SCHEDULE_12_BLOCKS
    if num_blocks % len(base) != 0:
        raise ValueError(
            f"num_blocks ({num_blocks}) must be a multiple of {len(base)} "
            f"to stretch DARES's schedule evenly; got a non-multiple."
        )
    repeat = num_blocks // len(base)
    return [base[i // repeat] for i in range(num_blocks)]


class QKVLoRA(nn.Module):
    """
    Wraps a fused qkv Linear(dim, 3*dim) and adds independent LoRA deltas
    to the q and v thirds of the output only, leaving k untouched --
    the fused-layer equivalent of DARES's lora=['q','v'] choice (their
    code targets separate .query/.key/.value sublayers, which DA3's
    qkv does not have).

    No alpha/rank scaling, matching DARES's _LoRA_qkv.forward exactly
    (W += deltaW, no scaling term).
    """

    def __init__(self, qkv_linear: nn.Linear, rank: int):
        super().__init__()
        self.qkv = qkv_linear
        for p in self.qkv.parameters():
            p.requires_grad = False

        dim = qkv_linear.in_features  # qkv: Linear(dim, 3*dim)
        self.dim = dim

        self.lora_a_q = nn.Linear(dim, rank, bias=False)
        self.lora_b_q = nn.Linear(rank, dim, bias=False)
        self.lora_a_v = nn.Linear(dim, rank, bias=False)
        self.lora_b_v = nn.Linear(rank, dim, bias=False)

        nn.init.kaiming_uniform_(self.lora_a_q.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_a_v.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b_q.weight)
        nn.init.zeros_(self.lora_b_v.weight)

    def forward(self, x):
        qkv_out = self.qkv(x)
        q, k, v = qkv_out.split(self.dim, dim=-1)
        q = q + self.lora_b_q(self.lora_a_q(x))
        v = v + self.lora_b_v(self.lora_a_v(x))
        return torch.cat([q, k, v], dim=-1)


def inject_vector_lora(net, ranks: list[int] | None = None) -> list[str]:
    """
    Vector-LoRA: q/v-only, no scaling, per-block rank schedule. If
    `ranks` is omitted, uses DARES's real schedule stretched to however
    many blocks the backbone has (must be a multiple of 12).
    """
    for p in net.parameters():
        p.requires_grad = False

    blocks = net.backbone.pretrained.blocks
    if ranks is None:
        ranks = make_dares_rank_schedule(len(blocks))
    assert len(ranks) == len(blocks), (
        f"ranks list has {len(ranks)} entries but there are {len(blocks)} blocks"
    )

    adapted = []
    for i, block in enumerate(blocks):
        attn = block.attn
        if not hasattr(attn, "qkv") or not isinstance(attn.qkv, nn.Linear):
            continue
        attn.qkv = QKVLoRA(attn.qkv, rank=ranks[i])
        adapted.append(f"blocks[{i}].attn.qkv (q+v only, r={ranks[i]})")

    return adapted


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

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
