"""Export a marimo notebook to a paginated, print-friendly PDF report.

`marimo export pdf` goes through nbconvert's WebPDF exporter, which has three
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
- Any cell output produced by a marimo layout helper (``mo.vstack``,
  ``mo.hstack``, ...) around a figure — even a single one — is serialized
  as a ``<marimo-mime-renderer data-data="...">`` custom element: a base64
  image wrapped in a JSON blob that only marimo's own JS frontend knows how
  to decode into an ``<img>``. A bare ``fig`` as a cell's last expression
  *does* export as a plain ``image/png`` output (nbconvert renders that
  fine) — it's specifically the "multiple/wrapped figures in one output"
  path that's affected. nbconvert has no idea what to do with the custom
  element, and Playwright never runs marimo's JS, so every such figure is
  silently blank in a plain nbconvert/Playwright render.

This module renders the same nbconvert HTML but drives Playwright directly,
so all three are fixable: one CSS override for wrapping, header/footer
templates for page numbers, and a decode pass that turns marimo's
mimebundle elements back into plain ``<img>`` tags before handing the HTML
to Playwright.
"""

from __future__ import annotations

import argparse
import html as _html
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import nbformat
from nbconvert import HTMLExporter
from playwright.sync_api import sync_playwright

_MARIMO_MIME_RENDERER_RE = re.compile(
    r"<marimo-mime-renderer\b[^>]*>\s*</marimo-mime-renderer>", re.DOTALL,
)
_ATTR_RE_TEMPLATE = r"""{name}=(['"])(.*?)\1"""


def _extract_attr(tag: str, name: str) -> str | None:
    match = re.search(_ATTR_RE_TEMPLATE.format(name=name), tag, re.DOTALL)
    return match.group(2) if match else None


def _decode_marimo_mime_renderers(html: str) -> str:
    """Replace marimo's ``<marimo-mime-renderer>`` output wrapper with plain HTML.

    marimo emits this custom element (instead of a plain ``image/png``
    output) whenever a figure is shown through a layout helper like
    ``mo.vstack``/``mo.hstack`` rather than as a cell's bare last
    expression. Its ``data-mime``/``data-data`` attributes (order is not
    guaranteed) are HTML-escaped JSON strings holding a mimetype -> data-URI
    mapping (plus a ``__metadata__`` entry with image dimensions); decoding
    that here is the only way such a figure shows up at all in a static
    render.
    """

    def _replace(match: re.Match[str]) -> str:
        tag = match.group(0)
        mime_attr, data_attr = _extract_attr(tag, "data-mime"), _extract_attr(tag, "data-data")
        if mime_attr is None or data_attr is None:
            return tag
        try:
            # Both attributes are JSON-encoded twice over: once for the
            # attribute's own string value, once more for the value that
            # string holds (a mimetype, or a mimetype -> data-URI dict).
            mime = json.loads(_html.unescape(mime_attr))
            bundle = json.loads(json.loads(_html.unescape(data_attr)))
        except (ValueError, TypeError):
            return match.group(0)
        if mime != "application/vnd.marimo+mimebundle":
            return match.group(0)

        metadata = bundle.get("__metadata__", {})
        for mimetype, content in bundle.items():
            if mimetype == "__metadata__" or not isinstance(content, str):
                continue
            if mimetype.startswith("image/"):
                dims = metadata.get(mimetype, {})
                size = f' width="{dims["width"]}" height="{dims["height"]}"' if "width" in dims else ""
                return f'<img src="{_html.escape(content)}" style="max-width:100%;"{size}>'
            if mimetype == "text/html":
                return content
            if mimetype == "text/plain":
                return f"<pre>{_html.escape(content)}</pre>"
        return match.group(0)  # nothing renderable in the bundle; leave as-is

    return _MARIMO_MIME_RENDERER_RE.sub(_replace, html)


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
                "--include-outputs",  # default is --no-include-outputs: source only, no prints/figures
                "-f",
            ],
            check=True,
        )

        notebook_node = nbformat.read(ipynb_path, as_version=4)
        exporter = HTMLExporter(template_name="lab", exclude_input=not include_code)
        html, _ = exporter.from_notebook_node(notebook_node)
        html = html.replace("<title>Notebook</title>", f"<title>{resolved_title}</title>")
        html = html.replace("</head>", f"{_PRINT_CSS}</head>")
        html = _decode_marimo_mime_renderers(html)

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
