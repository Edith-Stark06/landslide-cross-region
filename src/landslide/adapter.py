"""Band adapter: lets one pretrained 3-channel segmenter consume the channel bank (RGB, NIR, red edge, SWIR, slope, DEM, SAR slots).

BandAdapter: 1x1 convolution bank -> 3 channels, initialised as the identity on R,G,B and zero elsewhere, so training starts exactly at
the RGB-only model (no extra capacity can hurt at step 0). Missing channels are zeroed with the availability mask (zero = dataset mean after
normalisation). During training, the non-RGB channels are randomly dropped (per sample with prob p_drop_all, and per channel with prob
p_drop_each) so the network cannot depend on a channel that the target sensor does not have.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from . import channels


class BandAdapter(nn.Module):
    def __init__(self, n_bank: int = len(channels.BANK), p_drop_all: float = 0.3, p_drop_each: float = 0.2):
        super().__init__()
        self.n_bank, self.p_all, self.p_each = n_bank, p_drop_all, p_drop_each
        self.proj = nn.Conv2d(n_bank, 3, kernel_size=1, bias=True)
        with torch.no_grad():
            self.proj.weight.zero_(); self.proj.bias.zero_()
            for i, c in enumerate(channels.RGB):
                self.proj.weight[i, channels.BANK.index(c), 0, 0] = 1.0

    def forward(self, x: torch.Tensor, avail: torch.Tensor) -> torch.Tensor:
        m = avail.to(x.dtype)
        if self.training:
            keep = (torch.rand_like(m) > self.p_each).to(x.dtype)
            keep[:, :3] = 1.0                                               # R,G,B are never dropped
            drop_all = (torch.rand(m.shape[0], 1, device=m.device) < self.p_all).to(x.dtype)
            keep[:, 3:] = keep[:, 3:] * (1.0 - drop_all)                    # sometimes behave as a pure RGB sensor
            m = m * keep
        return self.proj(x * m[:, :, None, None])


class AdaptedSegmenter(nn.Module):
    needs_avail = True

    def __init__(self, net: nn.Module, n_bank: int = len(channels.BANK)):
        super().__init__()
        self.adapter, self.net = BandAdapter(n_bank), net

    def forward(self, x: torch.Tensor, avail: torch.Tensor | None = None) -> torch.Tensor:
        if avail is None:
            avail = torch.ones(x.shape[0], x.shape[1], device=x.device)
        return self.net(self.adapter(x, avail))
