"""Check that the IEEE-format manuscripts differ from paper/manuscript.tex in layout only: same captions, table inputs, figures,
section titles, abstract and keywords, and every long prose line identical apart from the declared substitutions
(Figure~ -> Fig.~, \\texttt{path} -> \\url{path}). Run from anywhere: python -I scripts/check_ieee_equivalence.py"""
import os
import re
import sys
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[1])
sys.stdout.reconfigure(encoding="utf-8")
a = open("paper/manuscript.tex", encoding="utf-8-sig").read()
b = open("paper/manuscript_ieeetran.tex", encoding="utf-8-sig").read()
c = open("paper/manuscript_ieeeaccess.tex", encoding="utf-8-sig").read()


def cap(t):
    return re.findall(r"\\caption\{.*\}\\label\{[^}]*\}", t)


print("captions identical (order-insensitive):", sorted(cap(a)) == sorted(cap(b)) == sorted(cap(c)), len(cap(a)), len(cap(b)))


def inp(t):
    return sorted(re.findall(r"\\input\{generated/(tab_\w+)\.tex\}", t))


print("table inputs identical:", inp(a) == inp(b) == inp(c), len(inp(a)))


def fig(t):
    return sorted(re.findall(r"\\includegraphics\[[^\]]*\]\{[^}]*?(fig_\w+\.png)\}", t))


print("figures identical:", fig(a) == fig(b) == fig(c), len(fig(a)))


def sec(t):
    return sorted(re.findall(r"\\(?:sub)*section\*?\{([^}]*)\}", t))


print("section titles identical:", sec(a) == sec(b) == sec(c), len(sec(a)))


def ab(t):
    return re.search(r"\\begin\{abstract\}\n(.*?)\n\\end\{abstract\}", t, re.S).group(1)


print("abstract identical:", ab(a) == ab(b) == ab(c))
skip = ("\\caption", "\\resizebox", "\\includegraphics", "\\begin", "\\end", "\\input", "\\noindent\\textbf{Keywords")
body_a = [l for l in a.split("\\begin{document}")[1].splitlines() if len(l) > 60 and not l.startswith(skip)]


def norm(s):
    s = re.sub(r"\\texttt\{([^{}]*)\}", lambda m: "\\url{" + m.group(1).replace("\\_", "_") + "}", s)
    return s.replace("Figures~", "Figs.~").replace("Figure~", "Fig.~")


lines_b = set(b.splitlines())
miss = [l for l in body_a if norm(l) not in lines_b and l not in lines_b]
print("long source lines not found verbatim (after the declared substitutions):", len(miss), "of", len(body_a))
for l in miss[:5]:
    print("  ", l[:120])
kw_a = re.search(r"Keywords:\}\s*(.*)", a).group(1)
kw_b = re.search(r"\\begin\{IEEEkeywords\}\n(.*)\n", b).group(1)
print("keywords: same set:", sorted(x.strip() for x in kw_a.split(";")) == sorted(x.strip() for x in kw_b.split(",")),
      "| same order:", [x.strip() for x in kw_a.split(";")] == [x.strip() for x in kw_b.split(",")])
words = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-\.%/\\{}]*", re.sub(r"\\(emph|textbf)\{", "", ab(a)))
print("abstract words (approx, each macro = 1 word):", len(words))
print("keywords alphabetical already:", [x.strip() for x in kw_a.split(";")] == sorted(x.strip() for x in kw_a.split(";")))
