"""Render paper/manuscript.tex to a readable Markdown preview (paper/manuscript_preview.md) with every generated macro replaced by its value.

This is a convenience for proof-reading without a LaTeX installation; the .tex file is the authoritative manuscript. Section/table/figure numbers
and citation numbers are reconstructed by order of appearance and may differ slightly from a real LaTeX build.
Run: python -I scripts/make_preview.py
"""
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "paper"
tex = (P / "manuscript.tex").read_text(encoding="utf-8")
tex = re.sub(r"(?<!\\)%.*", "", tex)
mac = dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}", (P / "generated" / "numbers.tex").read_text(encoding="utf-8")))
refs = json.loads((P / "references.json").read_text(encoding="utf-8"))
doc = tex.split("\\begin{document}")[1].split("\\end{document}")[0]


def clean(s: str) -> str:
    s = s.replace("$\\pm$", "±").replace("$-$", "−").replace("{,}", ",").replace("$^\\dagger$", "†").replace("$^\\ddagger$", "‡")
    s = s.replace("$^\\circ$", "°").replace("$<$", "<").replace("$\\leq$", "≤").replace("$\\times$", "×")
    s = s.replace("\\alpha", "α").replace("\\times", "×").replace("\\mathrm{", "{").replace("\\|", "‖").replace("\\,", " ")
    s = s.replace("\\%", "%").replace("~", " ").replace("\\&", "&").replace("\\_", "_").replace("\\#", "#")
    s = s.replace("--", "–").replace("``", "“").replace("''", "”")
    s = re.sub(r"\\emph\{([^{}]*)\}", r"*\1*", s)
    s = re.sub(r"\\textbf\{([^{}]*)\}", r"**\1**", s)
    s = re.sub(r"\\texttt\{([^{}]*)\}", r"`\1`", s)
    s = re.sub(r"\\url\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\todo\{([^{}]*)\}", r"**[TO DO: \1]**", s)
    s = re.sub(r"\$([^$]*)\$", r"\1", s)
    return s


for _ in range(3):                                           # macros may contain other text; two passes are enough
    doc = re.sub(r"\\iou\{\}", "IoU", doc)
    doc = re.sub(r"\\([A-Z][A-Za-z]+)(\{\})?", lambda m: mac.get(m.group(1), m.group(0)) if m.group(1) in mac else m.group(0), doc)

cite_no = {}
for grp in re.findall(r"\\cite\{([^}]*)\}", doc):
    for k in (x.strip() for x in grp.split(",")):
        cite_no.setdefault(k, len(cite_no) + 1)
for r in refs:
    cite_no.setdefault(r["key"], len(cite_no) + 1)
sec = [0, 0]
tab = fig = 0
labels = {}
out_lines = []
appendix = False
for m in re.finditer(r"\\appendix", doc):
    appendix = True


def table_md(path: Path) -> str:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith(("\\begin", "\\end", "\\toprule", "\\midrule", "\\bottomrule")):
            continue
        line = re.sub(r"\\multicolumn\{\d+\}\{[^}]*\}\{([^}]*)\}", r"\1", line).rstrip("\\ ").strip()
        line = re.sub(r"\\cite\{([^}]*)\}", lambda m: "[" + ",".join(str(cite_no.get(k.strip(), "?")) for k in m.group(1).split(",")) + "]", line)
        cells = [clean(c.strip()) for c in line.split("&")]
        rows.append(cells)
    if not rows:
        return ""
    w = max(len(r) for r in rows)
    rows = [r + [""] * (w - len(r)) for r in rows]
    md = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * w]
    md += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(md)


# first pass: numbering of labels
s_i = [0, 0]
t_i = f_i = 0
app = False
for tok in re.finditer(r"\\(appendix)|\\(section|subsection)\*?\{([^}]*)\}(?:\\label\{([^}]*)\})?|\\begin\{(table|figure)\}|\\label\{([^}]*)\}", doc):
    if tok.group(1):
        app = True; s_i = [0, 0]
    elif tok.group(2) == "section":
        s_i[0] += 1; s_i[1] = 0
        if tok.group(4):
            labels[tok.group(4)] = (chr(64 + s_i[0]) if app else str(s_i[0]))
    elif tok.group(2) == "subsection":
        s_i[1] += 1
        if tok.group(4):
            labels[tok.group(4)] = f"{s_i[0]}.{s_i[1]}"
    elif tok.group(5) == "table":
        t_i += 1; last = ("Table", t_i)
    elif tok.group(5) == "figure":
        f_i += 1; last = ("Figure", f_i)
    elif tok.group(6):
        if tok.group(6) in labels:
            continue
        if tok.group(6).startswith(("tab:",)):
            labels[tok.group(6)] = f"{t_i}"
        elif tok.group(6).startswith("fig:"):
            labels[tok.group(6)] = f"{f_i}"
        else:
            labels.setdefault(tok.group(6), "?")

body = doc
body = re.sub(r"\\maketitle", "", body)
body = re.sub(r"\\begin\{abstract\}", "\n## Abstract\n", body)
body = re.sub(r"\\end\{abstract\}", "\n", body)
app = False
sc = [0, 0]


def sec_sub(m):
    global sc
    kind, star, title = m.group(1), m.group(2), m.group(3)
    if star:
        return f"\n## {clean(title)}\n"
    if kind == "section":
        sc[0] += 1; sc[1] = 0
        return f"\n## {sc[0] if not app else chr(64 + sc[0])}. {clean(title)}\n"
    sc[1] += 1
    return f"\n### {sc[0] if not app else chr(64 + sc[0])}.{sc[1]} {clean(title)}\n"


parts = body.split("\\appendix")
fixed = []
for i, part in enumerate(parts):
    app = i == 1
    sc = [0, 0]
    part = re.sub(r"\\(section|subsection)(\*?)\{([^}]*)\}", sec_sub, part)
    fixed.append(part)
body = "\n\n## Appendices\n".join(fixed)
body = re.sub(r"\\label\{[^}]*\}", "", body)
body = re.sub(r"\\paragraph\{([^}]*)\}", lambda m: f"\n**{clean(m.group(1))}** ", body)
body = re.sub(r"\\cite\{([^}]*)\}", lambda m: "[" + ",".join(str(cite_no.get(k.strip(), "?")) for k in m.group(1).split(",")) + "]", body)
body = re.sub(r"\\(?:ref)\{([^}]*)\}", lambda m: labels.get(m.group(1), "?"), body)
body = re.sub(r"\\begin\{itemize\}(\\setlength\\itemsep\{0pt\})?", "", body)
body = re.sub(r"\\end\{itemize\}", "", body)
body = re.sub(r"\\item\s*", "\n- ", body)
body = re.sub(r"\\noindent\s*", "", body)
body = re.sub(r"\\section\*\{([^}]*)\}", r"\n## \1\n", body)


def float_sub(m):
    env, inner = m.group(1), m.group(2)
    cap = re.search(r"\\caption\{((?:[^{}]|\{[^{}]*\})*)\}", inner)
    cap = clean(cap.group(1)) if cap else ""
    inp = re.search(r"\\input\{generated/([^}]*)\}", inner)
    img = re.search(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", inner)
    if env == "table":
        return "\n" + (table_md(P / "generated" / inp.group(1)) if inp else "") + f"\n\n*{cap}*\n"
    return f"\n![]({img.group(1) if img else ''})\n\n*{cap}*\n"


body = re.sub(r"\\begin\{(table|figure)\}(?:\[[^\]]*\])?(.*?)\\end\{\1\}", float_sub, body, flags=re.S)
body = re.sub(r"\\input\{generated/refs\.tex\}",
              "\n## References\n\n" + "\n".join(f"[{cite_no[r['key']]}] " + clean(r["text"]) for r in sorted(refs, key=lambda r: cite_no[r["key"]])) + "\n", body)
body = re.sub(r"\\(?:centering|small|scriptsize)", "", body)
body = clean(body)
body = re.sub(r"\\[A-Za-z]+\*?(\[[^\]]*\])?", "", body)       # leftover simple commands
body = body.replace("{}", "").replace("{", "").replace("}", "")
head = re.search(r"\\title\{(.*)\}\s*\\author", tex, flags=re.S)
title = clean(re.sub(r"\\textbf\{(.*)\}", r"\1", head.group(1)).strip()) if head else "Manuscript"
out = f"# {title}\n\n*Preview rendered from `manuscript.tex` by `scripts/make_preview.py`; the .tex is authoritative. Citation and cross-reference numbers are reconstructed.*\n\n" + re.sub(r"\n{3,}", "\n\n", body)
(P / "manuscript_preview.md").write_text(out, encoding="utf-8")
print("preview words:", len(out.split()), "| unresolved '?' refs:", out.count("[?]") + out.count(" ? "))
