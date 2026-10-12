"""Measure blank space on every page of a compiled PDF (two-column layout).

For each page it reports, per column (left/right half of the text area), the blank space at the bottom of the column in inches and
the largest blank vertical run inside the column body in inches. The last page is allowed to end early. Requires pdftoppm (MiKTeX).
Usage: python -I scripts/measure_page_gaps.py main.pdf [dpi]
"""
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")
pdf = Path(sys.argv[1])
dpi = int(sys.argv[2]) if len(sys.argv) > 2 else 40
bin_dir = Path.home() / "AppData/Local/Programs/MiKTeX/miktex/bin/x64"
with tempfile.TemporaryDirectory() as td:
    subprocess.run([str(bin_dir / "pdftoppm.exe"), "-r", str(dpi), "-gray", "-png", str(pdf), str(Path(td) / "p")], check=True)
    pages = sorted(Path(td).glob("p-*.png"))
    n = len(pages)
    for k, f in enumerate(pages, 1):
        im = Image.open(f).convert("L")
        w, h = im.size
        px = im.load()
        top, bottom = int(0.95 * dpi), h - int(0.55 * dpi)        # text area below the running head, above the bottom margin
        margin_l, margin_r = int(0.62 * dpi), w - int(0.62 * dpi)
        mid = w // 2
        out = []
        for name, x0, x1 in (("L", margin_l, mid - 2), ("R", mid + 2, margin_r)):
            rows = []
            for y in range(top, bottom):
                rows.append(any(px[x, y] < 200 for x in range(x0, x1, 2)))
            last = max((i for i, r in enumerate(rows) if r), default=-1)
            bottom_gap = (len(rows) - 1 - last) / dpi
            run = best = 0
            for r in rows[: last + 1]:
                run = 0 if r else run + 1
                best = max(best, run)
            out.append(f"{name}: bottom {bottom_gap:4.1f} in, inner run {best / dpi:3.1f} in")
        flag = ""
        if k < n and any(float(s.split("bottom")[1].split("in")[0]) > 0.6 for s in out):
            flag = "  <-- gap"
        print(f"page {k:2d}  " + " | ".join(out) + flag)
