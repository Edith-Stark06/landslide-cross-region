"""CPU inference latency of every model family used in the paper (random weights; speed does not depend on them).

Batch size 1, fp32, torch.inference_mode, 5 warm-up runs then N timed runs; reports median and 10th/90th percentile in ms, parameters,
fp32 weight size and GFLOPs (multiply-adds, counted with torch.utils.flop_counter). Threads: all physical cores and 1.
Run: python -I scripts/benchmark_latency.py   ->  results/tables/latency_cpu.csv, latency_cpu.md
Measured on whatever machine runs it; the machine description is written next to the table.
"""
import platform
import statistics
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.flop_counter import FlopCounterMode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from landslide import channels  # noqa: E402
from landslide.run import build_model  # noqa: E402

MODELS = [("U-Net R34", "unet", 3), ("DeepLabV3+ R34", "deeplabv3plus", 3), ("SegFormer-B0", "segformer_b0", 3),
          ("SegFormer-B0 + band adapter", "adapter_segformer_b0", len(channels.BANK)), ("SegFormer-B3 (teacher)", "segformer_b3", 3)]


def bench(model, x, avail, n):
    with torch.inference_mode():
        for _ in range(5):
            model(x, avail) if getattr(model, "needs_avail", False) else model(x)
        t = []
        for _ in range(n):
            t0 = time.perf_counter()
            model(x, avail) if getattr(model, "needs_avail", False) else model(x)
            t.append((time.perf_counter() - t0) * 1000)
    q = statistics.quantiles(t, n=10)
    return statistics.median(t), q[0], q[-1]


rows = []
physical = max(1, (torch.get_num_threads() // 2) or 1)
for label, kind, in_ch in MODELS:
    m = build_model(dict(model=kind, encoder="resnet34", pretrained=False), in_ch).eval()
    params = sum(p.numel() for p in m.parameters())
    for px in (128, 256):
        x = torch.randn(1, in_ch, px, px)
        avail = torch.ones(1, in_ch)
        with FlopCounterMode(display=False) as fc, torch.inference_mode():
            m(x, avail) if getattr(m, "needs_avail", False) else m(x)
        for threads in sorted({1, torch.get_num_threads()}):
            torch.set_num_threads(threads)
            n = 30 if "B3" not in label else 15
            med, p10, p90 = bench(m, x, avail, n)
            rows.append(dict(model=label, input_px=px, threads=threads, median_ms=round(med, 1), p10_ms=round(p10, 1), p90_ms=round(p90, 1),
                             params_M=round(params / 1e6, 2), fp32_MB=round(params * 4 / 1e6, 1), GFLOPs=round(fc.get_total_flops() / 2e9, 2), runs=n))
            print(rows[-1], flush=True)
        torch.set_num_threads(torch.get_num_threads())
df = pd.DataFrame(rows)
out = ROOT / "results" / "tables"
out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / "latency_cpu.csv", index=False)
machine = f"{platform.processor()} | {platform.system()} {platform.release()} | torch {torch.__version__} | logical threads {torch.get_num_threads()}"
(out / "latency_cpu.md").write_text(f"Machine: {machine}\n\n" + df.to_markdown(index=False) + "\n", encoding="utf-8")
print(machine)
