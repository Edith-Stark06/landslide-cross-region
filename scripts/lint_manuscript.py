"""Static integrity checks for paper/manuscript.tex (no LaTeX needed).

1. every generated-style macro (\\CamelCase...) used in the text is defined in generated/numbers.tex or in the manuscript preamble;
2. every \\cite key exists in paper/references.json, and every reference is cited;
3. every \\ref has a \\label, every \\input / \\includegraphics target exists;
4. braces balance;
5. lists every literal number left in the prose (outside math, macros, commands) so each can be audited by hand.
Run: python -I scripts/lint_manuscript.py
"""
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "paper"
tex = (P / "manuscript.tex").read_text(encoding="utf-8")
body = re.sub(r"(?<!\\)%.*", "", tex)
defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", (P / "generated" / "numbers.tex").read_text(encoding="utf-8")))
defined |= set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", body))
ok = True

used = set(re.findall(r"\\([A-Z][A-Za-z]+)", body))
builtin_upper = {"Sigma"}
undefined = sorted(m for m in used if m not in defined and m not in builtin_upper)
print("1. undefined generated macros:", undefined or "none"); ok &= not undefined
unused = sorted(m for m in defined if m not in used and not m.startswith(("todo", "iou")))
print(f"   ({len(defined)} macros defined, {len(used & defined)} used; the rest are available for the text)")

refs = {r["key"] for r in json.loads((P / "references.json").read_text(encoding="utf-8"))}
cited = set()
for grp in re.findall(r"\\cite\{([^}]*)\}", body):
    cited |= {k.strip() for k in grp.split(",")}
gen_cited = set()
for f in (P / "generated").glob("tab_*.tex"):
    for grp in re.findall(r"\\cite\{([^}]*)\}", f.read_text(encoding="utf-8")):
        gen_cited |= {k.strip() for k in grp.split(",")}
cited |= gen_cited
print("2. cited but missing from references.json:", sorted(cited - refs) or "none"); ok &= not (cited - refs)
print("   in references.json but never cited:", sorted(refs - cited) or "none"); ok &= not (refs - cited)

labels = set(re.findall(r"\\label\{([^}]*)\}", body))
rf = set(re.findall(r"\\ref\{([^}]*)\}", body))
print("3. \\ref without \\label:", sorted(rf - labels) or "none"); ok &= not (rf - labels)
print("   labels never referenced:", sorted(labels - rf) or "none")
missing = []
for f in re.findall(r"\\input\{([^}]*)\}", body):
    if not (P / f).exists():
        missing.append(f)
for f in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", body):
    if not (P / f).exists():
        missing.append(f)
print("   missing input/figure files:", missing or "none"); ok &= not missing
for f in (P / "generated").glob("tab_*.tex"):                      # tables must not contain unresolved placeholders
    t = f.read_text(encoding="utf-8")
    if "nan" in t.lower().replace("financial", "") and re.search(r"\bnan\b", t):
        print("   NaN found in", f.name); ok = False

bal = sum(1 if c == "{" else -1 if c == "}" else 0 for c in re.sub(r"\\[{}]", "", body))
print("4. brace balance:", bal); ok &= bal == 0

# 5. literal numbers in prose
doc = body.split("\\begin{document}")[1].split("\\end{document}")[0]
doc = re.sub(r"\\begin\{abstract\}", "", doc)
doc = re.sub(r"\$[^$]*\$", " ", doc)                                              # math
doc = re.sub(r"\\(cite|ref|label|input|url|texttt|includegraphics|todo|section|subsection|paragraph|begin|end)(\[[^\]]*\])?\{[^}]*\}", " ", doc)
doc = re.sub(r"\\[A-Z][A-Za-z]+\{\}", " ", doc)                                   # generated macros
doc = re.sub(r"\\[A-Z][A-Za-z]+", " ", doc)
print("5. literal numbers left in the prose (audit each against its source):")
seen = 0
for m in re.finditer(r"(?<![A-Za-z\\])(\d[\d{},.]*\d|\d)(?![A-Za-z]*\})", doc):
    s = max(0, m.start() - 45); e = min(len(doc), m.end() + 35)
    print("   ", repr(doc[s:e].replace("\n", " ")), "->", m.group(1)); seen += 1
print(f"   {seen} literals")
print("\nRESULT:", "all structural checks passed" if ok else "PROBLEMS FOUND (see above)")
