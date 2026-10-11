"""Minimal LoRA (low-rank adaptation) for the linear layers of a transformer encoder, plus the knowledge-distillation loss.

LoRALinear(base, r, alpha): y = base(x) + (alpha / r) * B(A(x)), base frozen, A (r x in) Kaiming-initialised, B (out x r) zero-initialised,
so the adapted model equals the pretrained one at step 0. Implemented here (instead of the peft package) so it works on any smp encoder
whose attention / MLP projections are nn.Linear layers.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

TARGET_NAMES = ("q", "kv", "proj", "fc1", "fc2")      # MiT / SegFormer attention (q, kv, proj) and MLP (fc1, fc2) projections


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base, self.scale = base, alpha / r
        for p in self.base.parameters():
            p.requires_grad = False
        self.lora_a = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * F.linear(F.linear(x, self.lora_a), self.lora_b)


def apply_lora(encoder: nn.Module, r: int, alpha: float | None = None, targets=TARGET_NAMES) -> int:
    """Freeze the whole encoder, wrap its target Linear layers with LoRA. Returns the number of wrapped layers (0 means the names did not match)."""
    for p in encoder.parameters():
        p.requires_grad = False
    n = 0
    for parent in list(encoder.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear) and name in targets:
                setattr(parent, name, LoRALinear(child, r, alpha if alpha is not None else 2 * r))
                n += 1
    if n == 0:
        raise RuntimeError("apply_lora wrapped no layers: encoder linear layers are not named " + str(targets))
    return n


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, base_loss: torch.Tensor, alpha: float, temp: float) -> torch.Tensor:
    """(1 - alpha) * base_loss + alpha * T^2 * KL(teacher_T || student_T), averaged over pixels."""
    if alpha <= 0:
        return base_loss
    s = F.log_softmax(student_logits / temp, dim=1)
    t = F.softmax(teacher_logits.detach() / temp, dim=1)
    kl = F.kl_div(s, t, reduction="none").sum(dim=1).mean() * temp * temp
    return (1 - alpha) * base_loss + alpha * kl


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    return total, sum(p.numel() for p in model.parameters() if p.requires_grad)
