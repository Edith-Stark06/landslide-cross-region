import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    import torch
    from landslide import channels
    from landslide.adapter import AdaptedSegmenter, BandAdapter
except OSError as e:                                    # Windows Application Control sometimes blocks torch DLLs on this machine
    pytest.skip(f"torch unavailable here: {e}", allow_module_level=True)


def test_identity_init_equals_rgb_passthrough():
    a = BandAdapter().eval()
    x = torch.randn(2, len(channels.BANK), 8, 8)
    out = a(x, torch.ones(2, len(channels.BANK)))
    assert torch.allclose(out, x[:, :3], atol=1e-6)               # starts exactly as the RGB-only model


def test_missing_channels_cannot_influence_output():
    a = BandAdapter().eval()
    with torch.no_grad():
        a.proj.weight.normal_()
    x = torch.randn(1, len(channels.BANK), 4, 4)
    avail = torch.zeros(1, len(channels.BANK)); avail[:, :3] = 1  # RGB only
    x2 = x.clone(); x2[:, 3:] += 100.0                             # change the unavailable channels arbitrarily
    assert torch.allclose(a(x, avail), a(x2, avail))


def test_train_dropout_never_drops_rgb_and_sometimes_drops_others():
    torch.manual_seed(0)
    a = BandAdapter(p_drop_all=0.5, p_drop_each=0.5).train()
    with torch.no_grad():
        a.proj.weight.zero_(); a.proj.weight[:, :] = 1.0             # output = sum of kept channels
        a.proj.bias.zero_()
    x = torch.ones(512, len(channels.BANK), 2, 2)
    out = a(x, torch.ones(512, len(channels.BANK)))[:, 0, 0, 0]
    assert out.min() >= 3.0                                         # R,G,B always kept
    assert out.max() > out.min()                                    # other channels are dropped in some samples


def test_wrapper_shapes_with_stub_net():
    net = torch.nn.Conv2d(3, 2, 1)
    m = AdaptedSegmenter(net).eval()
    assert m(torch.randn(2, len(channels.BANK), 16, 16), torch.ones(2, len(channels.BANK))).shape == (2, 2, 16, 16)
    assert m(torch.randn(2, len(channels.BANK), 16, 16)).shape == (2, 2, 16, 16)
