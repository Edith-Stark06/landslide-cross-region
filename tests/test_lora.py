import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    import torch
    import torch.nn as nn
    from landslide.lora import LoRALinear, apply_lora, count_params, kd_loss
except OSError as e:
    pytest.skip(f"torch unavailable here: {e}", allow_module_level=True)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(8, 8); self.kv = nn.Linear(8, 16); self.proj = nn.Linear(8, 8); self.norm = nn.LayerNorm(8)


class Enc(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block(), Block()]); self.fc1 = nn.Linear(8, 32); self.fc2 = nn.Linear(32, 8)

    def forward(self, x):
        for b in self.blocks:
            x = b.proj(b.q(x) + b.kv(x)[..., :8])
        return self.fc2(self.fc1(x))


def test_lora_is_identity_at_init_and_wraps_all_targets():
    torch.manual_seed(0)
    enc = Enc()
    x = torch.randn(3, 8)
    before = enc(x)
    n = apply_lora(enc, r=2)
    assert n == 2 * 3 + 2                                    # q, kv, proj in two blocks + fc1 + fc2
    assert torch.allclose(enc(x), before, atol=1e-6)         # LoRA B is zero-initialised


def test_only_lora_params_are_trainable_and_far_fewer():
    enc = Enc()
    total_before, _ = count_params(enc)
    apply_lora(enc, r=2)
    total, trainable = count_params(enc)
    assert 0 < trainable < 0.5 * total
    assert all(("lora_" in n) for n, p in enc.named_parameters() if p.requires_grad)
    enc(torch.randn(2, 8)).sum().backward()
    assert all(p.grad is None for n, p in enc.named_parameters() if "lora_" not in n)


def test_lora_raises_when_names_do_not_match():
    with pytest.raises(RuntimeError):
        apply_lora(nn.Sequential(nn.Linear(4, 4)), r=2)


def test_kd_loss_zero_kl_when_student_equals_teacher_and_alpha_zero_is_base():
    logits = torch.randn(2, 2, 4, 4)
    base = torch.tensor(1.0)
    assert torch.allclose(kd_loss(logits, logits, base, alpha=0.5, temp=2.0), 0.5 * base, atol=1e-6)
    assert kd_loss(logits, torch.randn_like(logits), base, alpha=0.0, temp=2.0) == base
    assert kd_loss(logits, torch.randn_like(logits), base, alpha=1.0, temp=2.0) > 0
