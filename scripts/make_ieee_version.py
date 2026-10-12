"""Re-typeset paper/manuscript.tex in IEEE two-column layout (format only; no wording, numbers or results are changed).

Writes
  paper/manuscript_ieeetran.tex    IEEEtran [journal] class (standard in TeX Live / MiKTeX / Overleaf; compiled and checked locally)
  paper/manuscript_ieeeaccess.tex  the same body in the official IEEE Access class (ieeeaccess.cls, supplied by IEEE; NOT available here)
  paper/generated/refs_ieee.tex    the reference list re-ordered into IEEE style (from paper/references.json, same facts)

What changes (all demanded by the IEEE Access template, section K, M and the reference rules):
  * class, packages, title/author block, abstract and index terms in the class's own commands; keywords separated by commas
  * figures: wide ones become figure* (span both columns), the Pareto plot stays one column; "Figure~" -> "Fig.~"
  * tables: caption above the table; wide tables become table* and are fitted to the width (scaled down only if wider than the page)
  * appendices are placed before the unnumbered Data availability/Declarations (the template: appendixes precede the acknowledgment),
    the reference list is last
  * long \\texttt{...} file names/identifiers in running text become \\url{...} so that they can break across lines
  * references: initials first, title in quotes, year after the venue (author-year order -> IEEE order)
Everything else (every sentence, every number macro, every table body, every figure) is copied verbatim. A line-level audit of
the prose is printed so the substitutions can be reviewed.
Run: python -I scripts/make_ieee_version.py
"""
import difflib
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "paper"
src = (P / "manuscript.tex").read_text(encoding="utf-8-sig")

# Tables that fit one column at \footnotesize (natural width <= 1.15 x column width, measured with IEEEtran [journal]:
# column 252 pt, page 516 pt; composition 282 pt, perseed 267 pt, events 249 pt). All other tables span both columns.
SINGLE_COLUMN_TABLES = {"tab_composition", "tab_perseed", "tab_events"}

# ---------------------------------------------------------------------------------------------------------------- split
pre, rest = src.split("\\begin{document}", 1)
body = rest.split("\\end{document}", 1)[0]
title = re.search(r"\\title\{\\textbf\{(.*)\}\}", pre).group(1)
author = re.search(r"\\author\{(.*)\}", pre).group(1)
date = re.search(r"\\date\{(.*)\}", pre).group(1)
macros = "\n".join(l for l in pre.splitlines() if l.startswith("\\newcommand"))
numbers_input = "\\input{generated/numbers.tex}"

abstract = re.search(r"\\begin\{abstract\}\n(.*?)\n\\end\{abstract\}", body, re.S).group(1)
kw = re.search(r"\\noindent\\textbf\{Keywords:\}\s*(.*)", body).group(1).strip()
keywords = ", ".join(k.strip() for k in kw.split(";"))           # IEEE: separated by commas

main_start = body.index("\\section{Introduction}")
decl_start = body.index("\\section*{Data and code availability}")
refs_at = body.index("\\input{generated/refs.tex}")
app_at = body.index("\\appendix")
main_part = body[main_start:decl_start]
decl_part = body[decl_start:refs_at]
app_part = body[app_at + len("\\appendix"):]


# ----------------------------------------------------------------------------------------------------- transformations
def fig_block(m):
    width, path, tail = m.group("w"), m.group("p"), m.group("tail")
    single = width.startswith("0.72")                                   # the Pareto plot is the only one-column figure
    env = "figure" if single else "figure*"
    w = "\\columnwidth" if single else "\\textwidth"
    return f"\\begin{{{env}}}[!t]\\centering\n\\includegraphics[width={w}]{{{path}}}\n{tail}\\end{{{env}}}"


def tab_block(m):
    name = re.search(r"generated/(tab_\w+)\.tex", m.group("content")).group(1)
    caption = m.group("caption")
    env = "table" if name in SINGLE_COLUMN_TABLES else "table*"
    w = "\\columnwidth" if env == "table" else "\\textwidth"
    return (f"\\begin{{{env}}}[!t]\\centering\n{caption}\n\\footnotesize\n"
            f"\\fitbox{{{w}}}{{\\input{{generated/{name}.tex}}}}\n\\end{{{env}}}")


FIG = re.compile(r"\\begin\{figure\}\[t\]\\centering\n\\includegraphics\[width=(?P<w>[^\]]*)\]\{(?P<p>[^}]*)\}\n(?P<tail>\\caption.*\n)\\end\{figure\}")
TAB = re.compile(r"\\begin\{table\}\[(?:t|H)\]\\centering\\\w+\n(?P<content>(?:\\resizebox.*|\\input.*)\n)(?P<caption>\\caption.*)\n\\end\{table\}")


def to_url(m):
    t = m.group(1).replace("\\_", "_")
    return "\\url{" + t + "}"


def transform(text):
    text = FIG.sub(fig_block, text)
    text = TAB.sub(tab_block, text)
    text = re.sub(r"\bFigures~", "Figs.~", text)
    text = re.sub(r"\bFigure~", "Fig.~", text)
    out = []
    for line in text.split("\n"):
        if "\\caption" not in line:                                     # \url is fragile in captions; leave those as they are
            line = re.sub(r"\\texttt\{([^{}]*)\}", to_url, line)
        out.append(line)
    return "\n".join(out)


main_t, decl_t, app_t = transform(main_part), transform(decl_part), transform(app_part)

# audit: prose lines (outside floats) that differ from the source
orig_lines = [l for l in (main_part + decl_part + app_part).splitlines()]
new_lines = [l for l in (main_t + decl_t + app_t).splitlines()]
print("== line-level audit of body text (source -> IEEE); floats are rebuilt, see below ==")
changed = 0
for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, orig_lines, new_lines, autojunk=False).get_opcodes():
    if tag == "equal":
        continue
    o, n = orig_lines[i1:i2], new_lines[j1:j2]
    if any(("\\begin{table" in x or "\\begin{figure" in x or "\\end{table" in x or "\\end{figure" in x or "\\resizebox" in x
            or x.startswith("\\input{generated/tab_") or "\\includegraphics" in x or "\\centering" in x or x.startswith("\\caption")
            or x.startswith("\\footnotesize") or x.startswith("\\fitbox")) for x in o + n):
        continue
    changed += 1
    a, b = " ".join(o), " ".join(n)
    for t2, a1, a2, b1, b2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if t2 != "equal":
            print(f"  '{a[max(0, a1 - 18):a2 + 6]}'  ->  '{b[max(0, b1 - 18):b2 + 6]}'")
print(f"({changed} prose hunks changed)")
leftover = re.findall(r"[^.\n]*\bFigure\b[^\n]{0,30}", main_t + decl_t + app_t)
print("remaining 'Figure' words:", leftover or "none")

# ------------------------------------------------------------------------------------------------------------ references
def initials_ok(tok):
    return re.fullmatch(r"(?:[A-Z]\.(?:-[A-Z]\.)?)(?: [A-Z]\.(?:-[A-Z]\.)?)*", tok) is not None


def split_depth0(s, sep):
    parts, depth, cur, i = [], 0, "", 0
    while i < len(s):
        c = s[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        if depth == 0 and s.startswith(sep, i):
            parts.append(cur)
            cur, i = "", i + len(sep)
            continue
        cur += c
        i += 1
    parts.append(cur)
    return parts


def fmt_authors(auth):
    toks = [t for t in auth.split(", ")]
    names, i, etal = [], 0, False
    while i < len(toks):
        if toks[i] == "et al.":
            etal = True
            i += 1
        else:
            assert initials_ok(toks[i + 1]), (auth, toks[i + 1])
            names.append(f"{toks[i + 1]} {toks[i]}")
            i += 2
    if etal:
        return ", ".join(names) + " et al."
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f", and {names[-1]}"


def fmt_ref(text):
    m = re.match(r"^(?P<auth>.+?) \((?P<year>\d{4})\)\. (?P<title>.+?)\. (?P<venue>(?:In )?\\emph\{.*|arXiv:.*)$", text)
    if not m:                                                                       # software entry without a year
        m = re.match(r"^(?P<s>[^,]+), (?P<i>(?:[A-Z]\.\s?)+) (?P<rest>.*)$", text)
        return f"{m.group('i').strip()} {m.group('s')}, {m.group('rest')}"
    auth, year, ttl, venue = m.group("auth"), m.group("year"), m.group("title"), m.group("venue")
    ids = []
    id_re = re.compile(r"Data: doi:\S+|doi:\S+|arXiv:\S+(?: \([^)]*\))?")
    for f in id_re.finditer(venue):                                                  # keep the order of the source
        ids.append(f.group(0).rstrip("."))
    venue = id_re.sub("", venue)
    ids = [re.sub(r"^(Data: )?doi:", lambda k: (k.group(1) or "") + "doi: ", i) for i in ids]
    venue = venue.strip().rstrip(".").rstrip(",").strip()
    parts = [p.strip() for p in split_depth0(venue, ", ") if p.strip()] if venue else []
    arx = [i for i in ids if i.startswith("arXiv:")]
    other = [i for i in ids if not i.startswith("arXiv:")]
    out = []
    if parts:
        name = parts[0]
        is_proc = name.startswith("In ")
        out.append(("in " + name[3:]) if is_proc else name)
        extra, seen_vol = [], False
        for p in parts[1:]:
            mm = re.fullmatch(r"(\d+)\((\d+)\)", p)
            if mm:
                extra += [f"vol. {mm.group(1)}", f"no. {mm.group(2)}"]
                seen_vol = True
            elif re.fullmatch(r"\d+", p) and not seen_vol:
                extra.append(f"vol. {p}")
                seen_vol = True
            elif "--" in p:
                extra.append(f"pp. {p}")
            elif re.fullmatch(r"\d+", p) or re.fullmatch(r"[A-Za-z]+\d+", p):
                extra.append(f"Art. no. {p}")                                       # a bare number after the volume is an article number
            else:
                extra.append(p)                                                      # e.g. "PMLR 70"
        if is_proc:                                                                  # conference: year first, then pages
            out += [e for e in extra if not e.startswith("pp.")] + [year] + [e for e in extra if e.startswith("pp.")]
        else:
            out += extra + [year]
        out += arx + other
    else:
        out += arx + [year] + other
    return f"{fmt_authors(auth)}, ``{ttl},'' " + ", ".join(out) + "."


refs =json.loads((P / "references.json").read_text(encoding="utf-8"))
bib = ["\\begin{thebibliography}{99}"]
tok = lambda s: re.findall(r"[a-z0-9]+", re.sub(r"\\[A-Za-z]+|[{}\\]", " ", s.replace("\\\"o", "o")).lower())
problems = 0
for r in refs:
    new = fmt_ref(r["text"])
    bib.append(f"\\bibitem{{{r['key']}}} {new}")
    missing = sorted(set(tok(r["text"])) - set(tok(new)))
    if missing:
        problems += 1
        print("REF: tokens lost for", r["key"], missing)
bib.append("\\end{thebibliography}")
(P / "generated" / "refs_ieee.tex").write_text("\n".join(bib) + "\n", encoding="utf-8", newline="\n")
print(f"references: {len(refs)} reformatted, entries that lost a token: {problems}")

# -------------------------------------------------------------------------------------------------------------- assemble
common_pkgs = r"""\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{booktabs,multirow,array,xcolor,url,placeins}
\urlstyle{tt}
\emergencystretch=2em
\renewcommand{\arraystretch}{1.08}
\newsavebox{\fitboxbin}
% typeset a table at its natural width, scaled down only if it is wider than #1
\newcommand{\fitbox}[2]{\sbox{\fitboxbin}{#2}\ifdim\wd\fitboxbin>#1\resizebox{#1}{!}{\usebox{\fitboxbin}}\else\usebox{\fitboxbin}\fi}
"""
tail = f"""
{main_t.rstrip()}

\\appendices
{app_t.strip()}

\\FloatBarrier
{decl_t.strip()}

\\input{{generated/refs_ieee.tex}}

\\end{{document}}
"""
short_title = title

ieeetran = f"""% IEEE two-column version of paper/manuscript.tex (format only; generated by scripts/make_ieee_version.py - do not edit by hand).
% Class: IEEEtran [journal]. For the official IEEE Access class use manuscript_ieeeaccess.tex.
\\documentclass[journal]{{IEEEtran}}
{common_pkgs}{numbers_input}
{macros}

\\begin{{document}}
\\title{{{title}}}
\\author{{{author}\\thanks{{{date}.}}}}
\\markboth{{Author \\MakeLowercase{{\\textit{{et al.}}}}: {short_title}}}{{Author \\MakeLowercase{{\\textit{{et al.}}}}: {short_title}}}
\\maketitle
\\begin{{abstract}}
{abstract}
\\end{{abstract}}

\\begin{{IEEEkeywords}}
{keywords}
\\end{{IEEEkeywords}}

\\IEEEpeerreviewmaketitle
{tail}"""

access = f"""% IEEE Access version of paper/manuscript.tex (format only; generated by scripts/make_ieee_version.py - do not edit by hand).
% Requires the official class files from the IEEE Access template (ieeeaccess.cls, spotcolor.sty, logo images); see README.txt.
\\documentclass{{ieeeaccess}}
{common_pkgs.replace("\\usepackage[T1]{fontenc}\n", "").replace("\\usepackage[utf8]{inputenc}\n", "")}{numbers_input}
{macros}

\\begin{{document}}
\\history{{{date}}}
\\doi{{\\todo{{assigned by IEEE}}}}

\\title{{{title}}}
\\author{{{author}}}
\\markboth{{Author \\headeretal: {short_title}}}{{Author \\headeretal: {short_title}}}
\\corresp{{\\todo{{Corresponding author: to be completed by the authors}}}}

\\begin{{abstract}}
{abstract}
\\end{{abstract}}

\\begin{{keywords}}
{keywords}
\\end{{keywords}}

\\titlepgskip=-15pt
\\maketitle
{tail}"""

(P / "manuscript_ieeetran.tex").write_text(ieeetran, encoding="utf-8", newline="\n")
(P / "manuscript_ieeeaccess.tex").write_text(access, encoding="utf-8", newline="\n")
print("wrote manuscript_ieeetran.tex, manuscript_ieeeaccess.tex, generated/refs_ieee.tex")
