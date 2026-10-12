"""Verify paper/landslide_manuscript_overleaf.zip by extracting it fresh and checking that it is self-contained.
Run: python -I scripts/verify_overleaf_zip.py
"""
import re
import sys
import tempfile
import zipfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
zp = ROOT / "paper" / "landslide_manuscript_overleaf.zip"
with tempfile.TemporaryDirectory() as td:
    d = Path(td)
    with zipfile.ZipFile(zp) as z:
        assert z.testzip() is None, "corrupt zip"
        z.extractall(d)
    t = (d / "main.tex").read_text(encoding="utf-8")
    body = re.sub(r"(?<!\\)%.*", "", t)
    refs = re.findall(r"\\input\{([^}]*)\}", body) + re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", body)
    print("files in zip:", sum(1 for p in d.rglob("*") if p.is_file()))
    print("files referenced by main.tex:", len(refs), "| missing:", [p for p in refs if not (d / p).exists()] or "none")
    print("parent-directory paths (../):", "../" in body)
    print("documentclass count:", body.count("\\documentclass"), "| begin{document}:", body.count("\\begin{document}"), "| end{document}:", body.count("\\end{document}"))
    print("main.tex ends with \\end{document}:", body.rstrip().endswith("\\end{document}"))
    print("inputenc + fontenc present:", "{inputenc}" in body and "{fontenc}" in body)
    print("author placeholders (\\todo{...}):", len(re.findall(r"\\todo\{", body)))
    print("cite keys used:", len({k.strip() for g in re.findall(r"\\cite\{([^}]*)\}", body) for k in g.split(",")}),
          "| bibitems in refs.tex:", len(re.findall(r"\\bibitem\{", (d / "generated" / "refs.tex").read_text(encoding="utf-8"))))
    print("numbers.tex identical to repo copy:", (d / "generated" / "numbers.tex").read_bytes() == (ROOT / "paper" / "generated" / "numbers.tex").read_bytes())
    big = [(p.name, round(p.stat().st_size / 1e6, 2)) for p in d.rglob("*") if p.is_file() and p.stat().st_size > 1e6]
    print("files > 1 MB:", big)
