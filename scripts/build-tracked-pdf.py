#!/usr/bin/env python3
"""
Build LaTeX and PDF for MyST documents. For the tracked-changes file, green
<span> additions are converted to \\textcolor and unescaped in the generated .tex.
Respects each file's frontmatter (output folder and filenames). Applies custom
\\hypersetup for link/cite colors. No "-for-tex" suffix: builds from a staging copy
inside the repo (so MyST uses project root for output and fig/ paths).

Usage:
  python3 scripts/build-tracked-pdf.py [file1.md file2.ipynb ...]

Default: 09-manuscript-tracked.md, 01-manuscript.md, 02-appendix.ipynb, rebuttal_letter.md
"""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
# Staging dir inside repo so MyST writes to repo's latex/ and resolves fig/ from repo root
STAGING_DIR = REPO_ROOT / "_myst_build"
DEFAULT_SOURCES = [
    REPO_ROOT / "09-manuscript-tracked.md",
    REPO_ROOT / "01-manuscript.md",
    REPO_ROOT / "02-appendix.ipynb",
    REPO_ROOT / "rebuttal_letter.md",
]

OPEN_PATTERN = re.compile(r'<span\s+style="color:\s*green;">')
CLOSE_TAG = "</span>"
TRACKED_COLOR = "green!50!black"

# MyST can escape as {\textbackslash}textcolor\{green!50!black\}\{ or as \\textcolor...
ESCAPED_OPEN = re.compile(
    r'\{\\textbackslash\}textcolor\\\{' + re.escape(TRACKED_COLOR) + r'\\\}\\\{'
)
# Double backslash makes LaTeX print \textcolor as text
DOUBLE_BS_OPEN = re.compile(r'\\\\textcolor\\\{' + re.escape(TRACKED_COLOR) + r'\\\}\{')

HYPERSETUP_BLOCK = r"""\hypersetup{colorlinks = true,
linkcolor = blue,
urlcolor  = blue,
citecolor = blue,
anchorcolor = black}"""


def convert_green_spans_to_latex(text: str) -> str:
    out = []
    rest = text
    while True:
        m = OPEN_PATTERN.search(rest)
        if not m:
            out.append(rest)
            break
        out.append(rest[: m.start()])
        out.append(f"\\textcolor{{{TRACKED_COLOR}}}{{")
        after_open = rest[m.end() :]
        close_idx = after_open.find(CLOSE_TAG)
        if close_idx == -1:
            out.append(after_open)
            break
        out.append(after_open[:close_idx])
        out.append("}")
        rest = after_open[close_idx + len(CLOSE_TAG) :]
    return "".join(out)


def unescape_textcolor_in_tex(tex: str) -> str:
    # 1) {\textbackslash}textcolor\{green!50!black\}\{ ... \} -> \textcolor{green!50!black}{ ... }
    result = []
    rest = tex
    while True:
        m = ESCAPED_OPEN.search(rest)
        if not m:
            result.append(rest)
            break
        result.append(rest[: m.start()])
        result.append(f"\\textcolor{{{TRACKED_COLOR}}}{{")
        pos = m.end()
        depth = 1
        content_start = pos
        while pos < len(rest) and depth > 0:
            if rest[pos : pos + 2] == r"\{":
                pos += 2
                depth += 1
            elif rest[pos : pos + 2] == r"\}":
                pos += 2
                depth -= 1
                if depth == 0:
                    result.append(rest[content_start : pos - 2])
                    result.append("}")
                    break
            else:
                pos += 1
        if depth != 0:
            result.append(rest[content_start:])
            break
        rest = rest[pos:]
    tex = "".join(result)

    # 2) \\textcolor{...}{ -> \textcolor{...}{ (if MyST outputs double backslash so it prints as text)
    tex = DOUBLE_BS_OPEN.sub(r"\\textcolor{" + TRACKED_COLOR + "}{", tex)
    return tex


def apply_hypersetup(tex: str) -> str:
    """Replace existing \\hypersetup{...} with our block (handles multiline)."""
    match = re.search(r"\\hypersetup\s*\{", tex)
    if not match:
        return tex
    start = match.end() - 1  # position of {
    depth = 1
    pos = start + 1
    while pos < len(tex) and depth > 0:
        if tex[pos] == "{":
            depth += 1
        elif tex[pos] == "}":
            depth -= 1
        pos += 1
    if depth != 0:
        return tex
    return tex[: match.start()] + HYPERSETUP_BLOCK.strip() + tex[pos:]


def add_caption_package(tex: str) -> str:
    """Ensure \\usepackage{caption} is in the preamble (for \\ContinuedFloat)."""
    if "\\usepackage{caption}" in tex or "\\usepackage{caption}\n" in tex:
        return tex
    return tex.replace("\\begin{document}", "\\usepackage{caption}\n\\begin{document}", 1)


def add_table_small_font(tex: str) -> str:
    """Make font in table environments smaller: inject \\scriptsize after \\begin{table}."""
    # Inject right after \begin{table} so it reliably applies (e.g. tables inside quote)
    tex = re.sub(r"\\begin\{table\}(?!\\scriptsize)", r"\\begin{table}\\scriptsize", tex)
    return tex


# Two-figure block for fig-digits so the long caption is not cropped (\\ContinuedFloat keeps same figure number).
FIG_DIGITS_TWO_FIGURES = r"""\begin{figure}[!htbp]
\centering
\includegraphics[width=1\linewidth]{fig/results.png}
\caption[]{\textbf{Adaptive self-organization and generalization in a free-energy minimizing attractor network.} \newline
Simulation results from training the network on a single, handwritten example for each of the 10 digits (0-9), with variations in training precision and evidence strength to explore different learning regimes (\href{https://pni-lab.github.io/fep-attractor-network/simulation-digits}{Simulation 2}).
\textbf{A}: Performance landscapes as a function of inference temperature (inverse precision) and training evidence strength (bias magnitude). Retrieval performance (reconstructing noisy variants of the 10 training patterns, top left), one-shot generalization (reconstructing a noisy variants of unseen handwritten digits, top right), attractor orthogonality (mean squared angular difference from 90° indicating higher orthogonality for lower values, bottom left), and the number of attractors (when initialized with the 10 training patterns, bottom right) are shown. Optimal regions (contoured) highlight parameter settings that yield good generalization and highly orthogonal attractors. Contours in the top left and top right highlight the most efficient parameter settings for retrieval and generalization, respectively. Both contours are overlaid on the two bottom plots.
\textbf{B}: Conceptual illustration of training regimes. With low temperature (high precision) high model complexity is allowed (``accuracy pumping'') and attractors will tend to exactly match the training data. On the contrary, high temperatures (low precision) result in a single fixed point attractor and reduced recognition performance. However, such networks will be able to generalize to new data, suggesting the existence of ``soft attractors'' (e.g. saddle-like structures) that are not local minima on the free energy landscape, yet affect the steady-state posterior distribution in a non-negligible way (especially with longer mixing-times).}
\label{fig-digits}
\end{figure}

\begin{figure}
  \ContinuedFloat
  \caption{(continued) A balanced regime can be found with intermediate precision during training, where both recognition and generalization performance are high. This is exactly the regime that promotes attractor orthogonalization, crucial for efficient representation and generalization. The complexity restrictions on these models cause them to re-use the same attractors to represent different patterns (see e.g. the single attractor belonging to the digits 5 and 7 in the example on panel D), which eventually leads to approximate orthogonality. Panels C-E provide examples of network behavior on a handwritten digit task across different regimes, including (i) training data (same in all cases); (ii) fixed-point attractors (obtained with deterministic update); (iii) attractor-orthogonality (polar histogram of the pairwise angles between attractors); (iv) retrieval and 1-shot generalization performance ($R^2$ between the noisy input pattern and the network output after 100 time steps, for 100 randomly sampled patterns) and (v) illustrative example cases from the recognition and 1-shot generalization tests (noisy input, network output and true pattern).
\textbf{C}: High complexity: Attractors are sharp and similar to training data; good recognition, limited generalization.
\textbf{D}: Balanced complexity (orthogonalization): Attractors are distinct and quasi-orthogonal, enabling strong recognition and generalization from noisy inputs. The balanced regime clearly demonstrates the network's ability to form an orthogonal basis, facilitating effective generalization as predicted by the free-energy minimization framework.
\textbf{E}: Low complexity: There is only a single fixed-point attractor. Recognition performance is lower, but generalization remains considerable.}
\end{figure}
"""


def replace_fig_digits_with_continued_float(tex: str) -> str:
    """Replace the single fig-digits figure with the two-figure \\ContinuedFloat block so the caption is not cropped."""
    label = "\\label{fig-digits}"
    idx = tex.find(label)
    if idx == -1:
        return tex
    # Already replaced (two figures with \ContinuedFloat)
    if "\\ContinuedFloat" in tex[: idx + len(label) + 500]:
        return tex
    # Find the start of this figure: last \begin{figure} before the label
    fig_begin = tex.rfind("\\begin{figure}", 0, idx)
    if fig_begin == -1:
        return tex
    # Find the matching \end{figure}
    fig_end = tex.find("\\end{figure}", idx)
    if fig_end == -1:
        return tex
    fig_end += len("\\end{figure}")
    return tex[:fig_begin] + FIG_DIGITS_TWO_FIGURES + tex[fig_end:]


# Notebook filename (no path) -> (Simulation number, URL slug for pni-lab.github.io/fep-attractor-network/)
SIMULATION_NOTEBOOKS = {
    "03-simulation-demo.ipynb": (1, "simulation-demo"),
    "04-simulation-digits.ipynb": (2, "simulation-digits"),
    "05-simulation-digits-continuous-sequence.ipynb": (3, "simulation-digits-continuous-sequence"),
    "06-simulation-digits-catastrophic-forgetting.ipynb": (4, "simulation-digits-catastrophic-forgetting"),
    "07-simulation-scaling-jax.ipynb": (5, "simulation-scaling-jax"),
    "08-simulation-faces-jax.ipynb": (6, "simulation-faces-jax"),
}
SIMULATION_BASE_URL = "https://pni-lab.github.io/fep-attractor-network"
APPENDIX_BASE_URL = "https://pni-lab.github.io/fep-attractor-network/appendix"


def fix_appendix_refs(tex: str) -> str:
    """Turn plain 'Appendix~N' text into links to the published appendix (e.g. #appendix-3)."""
    for n in range(1, 10):
        plain = f"Appendix~{n}"
        url = f"{APPENDIX_BASE_URL}/#appendix-{n}"
        linked = f"\\href{{{url}}}{{{plain}}}"
        tex = tex.replace(plain, linked)
    return tex


def fix_notebook_hrefs(tex: str) -> str:
    """Replace \\href{*.ipynb}{} with \\href{BASE_URL/slug}{Simulation N} so links show and work in PDF."""
    for filename, (num, slug) in SIMULATION_NOTEBOOKS.items():
        url = f"{SIMULATION_BASE_URL}/{slug}"
        tex = tex.replace(f"\\href{{{filename}}}{{}}", f"\\href{{{url}}}{{Simulation {num}}}")
    return tex


def add_line_numbers_for_tracked(tex: str) -> str:
    """Add lineno package and \\linenumbers so the tracked manuscript PDF has line numbers."""
    if "\\usepackage{lineno}" in tex:
        return tex
    tex = tex.replace("\\begin{document}", "\\usepackage{lineno}\n\\begin{document}", 1)
    # Start line numbering from the first line of the body
    tex = tex.replace("\\begin{document}\n", "\\begin{document}\n\\linenumbers\n", 1)
    return tex


def _strip_balanced_brace_block(tex: str, start: int, open_brace: int) -> tuple[str, int]:
    """Return (content inside braces, position after closing brace). open_brace is the index of {."""
    depth = 1
    pos = open_brace + 1
    while pos < len(tex) and depth > 0:
        if tex[pos] == "{":
            depth += 1
        elif tex[pos] == "}":
            depth -= 1
        pos += 1
    return tex[open_brace + 1 : pos - 1], pos


def rebuttal_remove_section_numbers_keywords_author(tex: str) -> str:
    """Remove section numbering, \\keywords, and \\author list from rebuttal letter .tex."""
    # 1) Disable section numbering
    if "\\setcounter{secnumdepth}{-1}" not in tex:
        tex = tex.replace("\\begin{document}", "\\setcounter{secnumdepth}{-1}\n\\begin{document}", 1)
    # 2) Remove \\author{...} (replace with empty author so \\maketitle still runs but shows no authors)
    author_start = tex.find("\\author{")
    if author_start != -1:
        _, end = _strip_balanced_brace_block(tex, author_start, author_start + 7)  # 7 = len("\\author{") - 1, { at +7
        tex = tex[: author_start] + "\\author{}" + tex[end:]
    # 3) Remove \\keywords{...}
    kw_start = tex.find("\\keywords{")
    if kw_start != -1:
        _, end = _strip_balanced_brace_block(tex, kw_start, kw_start + 9)
        tex = tex[:kw_start] + tex[end:]
    # 4) Remove pdfauthor and pdfkeywords from \hypersetup (so PDF metadata has no author/keywords)
    tex = re.sub(r"pdfauthor=\{\\@author\},\s*\n?", "", tex)
    tex = re.sub(r"pdfkeywords=\{[^}]*\},\s*\n?", "", tex)
    # 5) Convert standalone \newline (from <br> in source) to paragraph break + vertical skip
    tex = re.sub(r"^\s*\\newline\s*$", r"\\par\\vspace{0.8em}", tex, flags=re.MULTILINE)
    return tex


def get_frontmatter_text(source: Path) -> str:
    if source.suffix.lower() == ".ipynb":
        nb = json.loads(source.read_text(encoding="utf-8"))
        for cell in nb.get("cells", []):
            if cell.get("cell_type") == "markdown":
                src = "".join(cell.get("source", []))
                if "---" in src and "exports:" in src:
                    end = src.index("---", 3) if "---" in src[3:] else len(src)
                    return src[4:end]
        return ""
    text = source.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return ""
    end = text.index("---", 3) if "---" in text[3:] else len(text)
    return text[4:end]


def get_latex_output_dir(source: Path) -> Optional[Path]:
    head = get_frontmatter_text(source)
    m = re.search(r"output:\s*(\S+)", head)
    if not m:
        return None
    out = m.group(1).strip().strip("'\"").rstrip("/")
    if not out.startswith("latex"):
        return None
    return REPO_ROOT / out


def find_generated_tex(latex_dir: Path) -> Optional[Path]:
    """Find main .tex in output dir (contains \\begin{document}). Prefer one with textcolor/hypersetup."""
    if not latex_dir.exists():
        return None
    candidates = []
    for p in latex_dir.rglob("*.tex"):
        try:
            c = p.read_text(encoding="utf-8")
            if "\\begin{document}" not in c:
                continue
            has_fix = "textcolor" in c or "textbackslash" in c or "hypersetup" in c
            candidates.append((p, has_fix, p.stat().st_mtime))
        except OSError:
            continue
    if not candidates:
        return None
    # Prefer file that has our content (tracked or hypersetup), else newest main .tex
    candidates.sort(key=lambda x: (not x[1], -x[2]))
    return candidates[0][0]


def build_one(source: Path) -> None:
    source = source.resolve()
    if not source.exists():
        print(f"Skip (missing): {source.relative_to(REPO_ROOT)}", file=sys.stderr)
        return

    content = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".ipynb":
        nb = json.loads(content)
        for cell in nb.get("cells", []):
            if cell.get("cell_type") == "markdown":
                src = "".join(cell.get("source", []))
                if OPEN_PATTERN.search(src):
                    new_src = convert_green_spans_to_latex(src)
                    # Keep notebook format: list of lines (with trailing \n)
                    lines = new_src.splitlines(keepends=True)
                    cell["source"] = lines if lines else [""]
        to_build = json.dumps(nb, ensure_ascii=False, indent=1)
    else:
        to_build = convert_green_spans_to_latex(content)

    latex_dir = get_latex_output_dir(source)
    if latex_dir is None:
        print(f"Skip (no latex output in frontmatter): {source.relative_to(REPO_ROOT)}", file=sys.stderr)
        return

    # Build from staging file inside repo so MyST uses project root (output -> repo/latex/, fig/ resolves)
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    staging_path = STAGING_DIR / source.name
    try:
        staging_path.write_text(to_build, encoding="utf-8")
        subprocess.run(
            ["myst", "build", str(staging_path), "--tex", "--pdf"],
            cwd=REPO_ROOT,
            check=True,
        )
    finally:
        staging_path.unlink(missing_ok=True)

    # MyST writes under staging dir; copy to repo latex/ so we find .tex and can run pdflatex
    staging_out = STAGING_DIR / latex_dir.relative_to(REPO_ROOT)
    if staging_out.exists():
        latex_dir.mkdir(parents=True, exist_ok=True)
        for item in staging_out.iterdir():
            dest = latex_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)

    tex_path = find_generated_tex(latex_dir)
    if tex_path is None:
        print(f"Generated .tex not found under {latex_dir.relative_to(REPO_ROOT)}", file=sys.stderr)
        return

    # So that \includegraphics{fig/...} resolve when running pdflatex from .tex's directory
    fig_link = tex_path.parent / "fig"
    if not fig_link.exists() and (REPO_ROOT / "fig").exists():
        try:
            fig_link.symlink_to(REPO_ROOT / "fig")
        except OSError:
            pass

    tex_content = tex_path.read_text(encoding="utf-8")
    fixed = unescape_textcolor_in_tex(tex_content)
    fixed = apply_hypersetup(fixed)
    fixed = add_caption_package(fixed)
    fixed = add_table_small_font(fixed)
    fixed = replace_fig_digits_with_continued_float(fixed)
    fixed = fix_notebook_hrefs(fixed)
    fixed = fix_appendix_refs(fixed)
    if source.name == "09-manuscript-tracked.md":
        fixed = add_line_numbers_for_tracked(fixed)
    if source.name == "rebuttal_letter.md":
        fixed = rebuttal_remove_section_numbers_keywords_author(fixed)
    tex_path.write_text(fixed, encoding="utf-8")

    tex_dir = tex_path.parent
    stem = tex_path.stem
    # First pdflatex: generate .aux for refs and citations
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", tex_path.name],
        cwd=tex_dir,
        check=False,
        capture_output=True,
    )
    # Run bibtex if this document has a bibliography (so citations resolve)
    aux = tex_dir / f"{stem}.aux"
    if aux.exists() and "\\bibdata{" in aux.read_text(encoding="utf-8", errors="replace"):
        subprocess.run(
            ["bibtex", stem],
            cwd=tex_dir,
            check=False,
            capture_output=True,
        )
    # Two more pdflatex runs: incorporate .bbl and resolve all refs/citations
    for _ in range(2):
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", tex_path.name],
            cwd=tex_dir,
            check=False,
            capture_output=True,
        )
    pdf_in_texdir = tex_dir / (stem + ".pdf")
    # Overwrite top-level PDF so the canonical path has the full build (with fig), not MyST's broken one
    if pdf_in_texdir.exists():
        pdf_at_root = latex_dir / (tex_path.stem + ".pdf")
        shutil.copy2(pdf_in_texdir, pdf_at_root)
    rel_tex = tex_path.relative_to(REPO_ROOT)
    print(f"  {source.name} -> {rel_tex}")
    if pdf_in_texdir.exists():
        print(f"       PDF: {latex_dir.relative_to(REPO_ROOT)}/{tex_path.stem}.pdf")


def main() -> None:
    if len(sys.argv) > 1:
        sources = [Path(p) if Path(p).is_absolute() else REPO_ROOT / p for p in sys.argv[1:]]
    else:
        sources = DEFAULT_SOURCES

    print("Building LaTeX + PDF (tracked color fix + hypersetup)...")
    for s in sources:
        build_one(s)
    print("Done.")


if __name__ == "__main__":
    main()
