# Build LaTeX and PDF (tracked changes + hypersetup)

Builds all configured MyST documents to LaTeX and PDF in their frontmatter output folders. For the tracked-changes file, green `<span>` additions are converted to `\textcolor{green!50!black}{...}` and any MyST escaping in the generated .tex is fixed so the PDF shows dark green. All generated .tex files get the same `\hypersetup` (blue links/citations, black anchors). No "-for-tex" suffix: the script builds from a temp copy with the **same filename** as the source so output names match (e.g. `manuscript-tracked.tex`).

**Default files:** `09-manuscript-tracked.md`, `01-manuscript.md`, `02-appendix.ipynb`, `rebuttal_letter.md`

```bash
python3 scripts/build-tracked-pdf.py
# or specific files:
python3 scripts/build-tracked-pdf.py 09-manuscript-tracked.md rebuttal_letter.md
```

**What the script does (per file):**

1. Converts `<span style="color: green;">...</span>` to `\textcolor{green!50!black}{...}` (in a temp copy with the same name as the source).
2. Runs `myst build ... --tex --pdf` so output goes to the path in frontmatter (e.g. `latex/fep-ann-manuscript-tracked`).
3. Post-processes the generated .tex: unescapes `\textcolor` (handles `{\textbackslash}...` and `\\textcolor`), and replaces `\hypersetup{...}` with link/cite colors (blue, anchor black).
4. Rebuilds the PDF with `pdflatex` in that folder.

**Outputs**

All outputs stay in each file's export folder (e.g. `latex/fep-ann-manuscript-tracked/`). No files are copied to the repo root.

**Requirements**

- `myst` CLI and `pdflatex` on your PATH.
