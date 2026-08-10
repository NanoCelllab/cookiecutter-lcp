"""Export a marimo notebook to a paginated, print-friendly PDF report.

`marimo export pdf` goes through nbconvert's WebPDF exporter, which has two
limitations that matter for a report a student hands in:

- It never adds page numbers. Playwright's ``page.pdf()`` only gets headers
  and footers when the caller passes ``display_header_footer``/
  ``footer_template``, and nbconvert's ``WebPDFExporter.run_playwright()``
  never does — there's no CLI flag for it either.
- Long source lines are silently clipped, not wrapped. nbconvert's ``lab``
  template renders code-cell inputs inside ``.jp-InputArea-editor``, styled
  with ``overflow: hidden`` and no ``white-space`` rule — that's fine in the
  live JupyterLab UI (CodeMirror provides its own horizontal scrollbar
  there), but in a static export anything past the box's fixed width just
  disappears.

This module renders the same nbconvert HTML but drives Playwright directly,
so both are fixable: one CSS override for wrapping, and header/footer
templates for page numbers.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import nbformat
from nbconvert import HTMLExporter
from playwright.sync_api import sync_playwright

_PRINT_CSS = """
<style>
  .jp-InputArea, .jp-InputArea-editor {
    overflow: visible !important;
  }
  .jp-InputArea-editor pre, .highlight pre {
    white-space: pre-wrap !important;
    overflow-wrap: anywhere !important;
    word-break: break-word !important;
  }
</style>
"""

_FOOTER_TEMPLATE = """
<div style="width:100%; font-size:9px; color:#666; text-align:center; font-family:sans-serif;">
  <span class="pageNumber"></span> / <span class="totalPages"></span>
</div>
"""

_MARGIN = {"top": "18mm", "bottom": "16mm", "left": "12mm", "right": "12mm"}


def _header_template(title: str) -> str:
    return (
        '<div style="width:100%; font-size:9px; color:#666; text-align:center; '
        f'font-family:sans-serif; padding-top:4px;">{title}</div>'
    )


def export_notebook_pdf(
    notebook_path: Path,
    output_path: Path,
    *,
    include_code: bool = True,
    title: str | None = None,
) -> None:
    """Run `notebook_path` headlessly and render it to a paginated PDF."""
    notebook_path = Path(notebook_path).resolve()
    output_path = Path(output_path).resolve()
    resolved_title = title or notebook_path.stem

    with tempfile.TemporaryDirectory() as tmp:
        ipynb_path = Path(tmp) / f"{notebook_path.stem}.ipynb"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "marimo",
                "export",
                "ipynb",
                str(notebook_path),
                "-o",
                str(ipynb_path),
                "-f",
            ],
            check=True,
        )

        notebook_node = nbformat.read(ipynb_path, as_version=4)
        exporter = HTMLExporter(template_name="lab", exclude_input=not include_code)
        html, _ = exporter.from_notebook_node(notebook_node)
        html = html.replace("<title>Notebook</title>", f"<title>{resolved_title}</title>")
        html = html.replace("</head>", f"{_PRINT_CSS}</head>")

        html_path = Path(tmp) / f"{notebook_path.stem}.html"
        html_path.write_text(html, encoding="utf-8")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page()
            page.emulate_media(media="print")
            page.goto(f"file://{html_path}", wait_until="networkidle")
            page.pdf(
                path=str(output_path),
                print_background=True,
                display_header_footer=True,
                header_template=_header_template(resolved_title),
                footer_template=_FOOTER_TEMPLATE,
                margin=_MARGIN,
            )
            browser.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path, help="Path to a marimo notebook (.py)")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PDF path (default: alongside the notebook)",
    )
    parser.add_argument(
        "--no-code",
        action="store_true",
        help="Omit code cell inputs from the report",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Report title shown in the page header (default: notebook filename)",
    )
    args = parser.parse_args()

    output = args.output or args.notebook.with_suffix(".pdf")
    export_notebook_pdf(args.notebook, output, include_code=not args.no_code, title=args.title)
    print(f"✓ Wrote {output}")


if __name__ == "__main__":
    main()
