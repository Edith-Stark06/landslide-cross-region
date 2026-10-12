"""Package the IEEE-format manuscript as an Overleaf project: paper/landslide_manuscript_ieee_overleaf.zip

  main.tex                  IEEEtran [journal] version (standard class; compiled and checked locally with pdfLaTeX)
  main_ieeeaccess.tex       same body in the official IEEE Access class (needs ieeeaccess.cls from IEEE's template; not compiled here)
  generated/*.tex           numbers.tex (result macros), tab_*.tex (tables), refs_ieee.tex (IEEE-style reference list)
  figures/*.png             the six figures
  README.txt
Run after make_ieee_version.py: python -I scripts/make_overleaf_zip_ieee.py
"""
import re
import shutil
import sys
import zipfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "paper"
OUT = P / "landslide_manuscript_ieee_overleaf.zip"


def prep(name):
    t = (P / name).read_text(encoding="utf-8-sig")
    return t.replace("../figures/", "figures/")


mains = {"main.tex": prep("manuscript_ieeetran.tex"), "main_ieeeaccess.tex": prep("manuscript_ieeeaccess.tex")}
figs = sorted({f for t in mains.values() for f in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{(figures/[^}]*)\}", t)})
inputs = sorted({i for t in mains.values() for i in re.findall(r"\\input\{(generated/[^}]*)\}", t)})
missing = [f for f in figs if not (ROOT / f).exists()] + [i for i in inputs if not (P / i).exists()]
assert not missing, f"missing files: {missing}"

readme = """Overleaf project: manuscript re-typeset in IEEE two-column format
================================================================================

WHAT THIS IS
The text, numbers, tables and figures are identical to the single-column manuscript; only the layout was changed to follow the
IEEE Access template (two columns, IEEE headings, Fig./Table captions, IEEE-style reference list, appendices before the
declarations, references last). See paper/REVIEWER_NOTES.md and the repository README for what is generated and what is not.

TWO ENTRY FILES
1. main.tex  (default)  - class IEEEtran [journal]. This class is part of every TeX Live and Overleaf installation, so the project
   compiles as it is: Menu -> Compiler: pdfLaTeX, Main document: main.tex. It was compiled locally with pdfLaTeX (MiKTeX 25.12).
   It reproduces the IEEE two-column body layout but not the IEEE Access title-page branding.
2. main_ieeeaccess.tex  - class ieeeaccess, the class used by the IEEE Access template. ieeeaccess.cls is distributed by IEEE with the
   template and is NOT included here. To use it: in Overleaf create a project from the IEEE Access template (Templates gallery),
   upload generated/, figures/ and main_ieeeaccess.tex into it, and set main_ieeeaccess.tex as the main document (or copy the
   template's ieeeaccess.cls, spotcolor.sty and logo images next to it). This file could not be compiled where it was produced.

BEFORE SUBMISSION (the template and IEEE Access policy ask for the following; the text was deliberately not changed)
* Abstract: the template asks for 150-250 words, one paragraph, no abbreviations, footnotes or references; check the abstract.
* Index terms: the template asks for alphabetical order.
* Red bracketed text marks placeholders for the authors (author names, affiliations, corresponding author, funding, competing
  interests, CRediT contributions, ethics, acknowledgements, AI-use statement). The template puts funding information in the
  first-page footnote and the acknowledgment in its own section; move the text from Declarations if you wish.
* Author biographies and photographs (end of the template) are not included.
* Verify every reference and every number quoted from other papers (paper/references.json and paper/literature_values.json).
"""

tmp = ROOT / ".cache" / "ieee_build"
if tmp.exists():
    shutil.rmtree(tmp)
(tmp / "generated").mkdir(parents=True)
(tmp / "figures").mkdir()
for n, t in mains.items():
    (tmp / n).write_text(t, encoding="utf-8", newline="\n")
for i in inputs:
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
