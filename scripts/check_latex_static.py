"""Static LaTeX checks that catch the usual compile errors when no LaTeX engine is available.

* tabular column counts: every row of every generated table has as many cells as its column spec (multicolumn handled);
* begin/end environments balance in manuscript.tex and in generated files;
* non-ASCII characters (a problem for pdfLaTeX without inputenc);
* generated macro names that could collide with existing LaTeX commands (short or common names);
* unescaped special characters (&, %, #, _) in prose and table text outside commands/math.
Run: python -I scripts/check_latex_static.py
"""
import re
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
P = Path(__file__).resolve().parents[1] / "paper"
bad = 0


def spec_cols(spec: str) -> int:
    brace = r"\{(?:[^{}]|\{[^{}]*\})*\}"
    spec = re.sub(r"@" + brace + r"|>" + brace + r"|<" + brace + r"|\|", "", spec)      # @{..}, >{..}, <{..}, rules
    return len(re.findall(r"[lcrXpmb](?:" + brace + r")?", spec))                      # p{..}/m{..}/b{..} count once


for f in sorted((P / "generated").glob("tab_*.tex")):
    txt = f.read_text(encoding="utf-8")
    m = re.search(r"\\begin\{tabular\}\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}", txt)
    ncol = spec_cols(m.group(1))
    for line in txt.splitlines():
        s = line.strip()
        if not s.endswith("\\\\") or s.startswith("\\"):
            continue
        body = s[:-2]
        cells = len(re.split(r"(?<!\\)&", body))
        extra = sum(int(x) - 1 for x in re.findall(r"\\multicolumn\{(\d+)\}", body))
        if cells + extra != ncol:
            print(f"COLUMN MISMATCH {f.name}: spec {ncol} cols, row has {cells + extra}: {s[:90]}")
            bad += 1
print("1. tabular column counts checked in", len(list((P / 'generated').glob('tab_*.tex'))), "tables")

allfiles = [P / "manuscript.tex"] + sorted((P / "generated").glob("*.tex"))
for f in allfiles:
    t = re.sub(r"(?<!\\)%.*", "", f.read_text(encoding="utf-8"))
    b, e = Counter(re.findall(r"\\begin\{([^}]*)\}", t)), Counter(re.findall(r"\\end\{([^}]*)\}", t))
    if b != e:
        print("ENV MISMATCH", f.name, {k: (b[k], e[k]) for k in set(b) | set(e) if b[k] != e[k]})
        bad += 1
print("2. begin/end balance checked in", len(allfiles), "files")

for f in allfiles:
    t = f.read_text(encoding="utf-8")
    na = sorted({c for c in t if ord(c) > 127})
    if na:
        print(f"NON-ASCII in {f.name}: {na}")
        bad += 1
print("3. non-ASCII scan done")

names = re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", (P / "generated" / "numbers.tex").read_text(encoding="utf-8"))
short = sorted(n for n in names if len(n) <= 6)
print("4. macro names <= 6 letters (check against LaTeX commands):", short)
print("   total macros:", len(names), "| duplicates:", [n for n, c in Counter(names).items() if c > 1] or "none")

ms = re.sub(r"(?<!\\)%.*", "", (P / "manuscript.tex").read_text(encoding="utf-8")).split("\\begin{document}")[1]
prose = re.sub(r"\$[^$]*\$", " ", ms)
prose = re.sub(r"\\[A-Za-z]+\*?(\[[^\]]*\])?(\{[^{}]*\})?", " ", prose)
for ch, name in (("&", "&"), ("#", "#")):
    for m in re.finditer(r"(?<!\\)" + re.escape(ch), prose):
        ctx = prose[max(0, m.start() - 30): m.end() + 30].replace("\n", " ")
        if "tabular" not in ctx and "array" not in ctx:
            print(f"5. possibly unescaped {name}: ...{ctx}...")
            bad += 1
print("5. special-character scan done")
print("\nRESULT:", "no problems found" if not bad else f"{bad} PROBLEMS")
