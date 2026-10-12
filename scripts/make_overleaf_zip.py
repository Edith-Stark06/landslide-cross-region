"""Package the manuscript as a self-contained Overleaf project (zip).

Layout inside the zip:
  main.tex                 the manuscript (paths rewritten to figures/ and generated/, inputenc+fontenc added for pdfLaTeX)
  generated/*.tex          numbers.tex (all result macros), tab_*.tex (tables), refs.tex (bibliography)
  figures/*.png            the six figures used by the manuscript
  README.txt               how to compile on Overleaf; what is generated vs hand-written
Only the files the manuscript actually references are included. Run after make_paper_assets.py and the figure scripts.
Run: python -I scripts/make_overleaf_zip.py -> paper/landslide_manuscript_overleaf.zip
"""
import re
import shutil
import sys
import zipfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "paper"
OUT = P / "landslide_manuscript_overleaf.zip"
tex = (P / "manuscript.tex").read_text(encoding="utf-8")

tex = tex.replace("../figures/", "figures/")
tex = tex.replace("% Compile (offline, from this directory): tectonic -X compile --only-cached manuscript.tex",
                  "% Overleaf: set the main document to main.tex and the compiler to pdfLaTeX (default).")
tex = tex.replace("\\documentclass[11pt]{article}\n", "\\documentclass[11pt]{article}\n\\usepackage[utf8]{inputenc}\n\\usepackage[T1]{fontenc}\n", 1)

figs = sorted(set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{(figures/[^}]*)\}", tex)))
inputs = sorted(set(re.findall(r"\\input\{(generated/[^}]*)\}", tex)))
missing = [f for f in figs if not (ROOT / f).exists()] + [i for i in inputs if not (P / i).exists()]
assert not missing, f"missing files: {missing}"

readme = f"""Overleaf project: manuscript on event-isolated evaluation of cross-region landslide segmentation
=================================================================================================

HOW TO COMPILE
1. Overleaf -> New Project -> Upload Project -> choose this zip.
2. Menu -> Main document: main.tex ; Compiler: pdfLaTeX (default) ; TeX Live: any recent version.
3. Recompile. The preamble uses only standard packages: article, inputenc, fontenc, amsmath, amssymb, graphicx, booktabs, multirow,
   array, xcolor, url, cite. The bibliography is a hand-built thebibliography (generated/refs.tex), so no BibTeX run is needed.
4. This package was NOT compiled on the machine that produced it (no LaTeX installation there). It passed static checks (balanced
   environments, table column counts, undefined macros, citations and labels). If Overleaf shows an error, copy the first error
   message from the log and send it back.

WHAT IS GENERATED AND WHAT IS NOT
* generated/numbers.tex defines every result number used in the text as a macro (for example \\SegIouAllP). The numbers come from stored
  run files, not from typing. Do not edit them by hand; to change a result, regenerate (see the repository README).
* generated/tab_*.tex are the tables, generated from the same run files. generated/refs.tex is the reference list.
* Red bracketed text in the PDF marks placeholders for the authors (author names, funding, competing interests, CRediT contributions,
  ethics, acknowledgements, and confirmation of the AI-use statement).
* Figures in figures/ are the final PNGs produced by the repository scripts.

BEFORE SUBMISSION (see paper/REVIEWER_NOTES.md in the repository)
* Convert to the target journal's template (keep the \\input{{generated/...}} structure or inline the tables).
* Verify every reference and every number quoted from other papers (paper/references.json and paper/literature_values.json in the
  repository hold their verification status and provenance).
* Repository: https://github.com/Edith-Stark06/landslide-cross-region
"""

tmp = ROOT / ".cache" / "overleaf_build"
if tmp.exists():
    shutil.rmtree(tmp)
(tmp / "generated").mkdir(parents=True)
(tmp / "figures").mkdir()
(tmp / "main.tex").write_text(tex, encoding="utf-8", newline="\n")
for i in inputs + ["generated/refs.tex"]:
    if (P / i).exists():
        shutil.copy(P / i, tmp / i)
for f in figs:
    shutil.copy(ROOT / f, tmp / f)
(tmp / "README.txt").write_text(readme, encoding="utf-8", newline="\n")
if OUT.exists():
    OUT.unlink()
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for fp in sorted(tmp.rglob("*")):
        if fp.is_file():
            z.write(fp, fp.relative_to(tmp).as_posix())
with zipfile.ZipFile(OUT) as z:
    names = z.namelist()
    print(f"{OUT.name}: {len(names)} files, {OUT.stat().st_size / 1e6:.2f} MB")
    for n in names:
        print("  ", n, f"{z.getinfo(n).file_size / 1e3:.0f} kB")
