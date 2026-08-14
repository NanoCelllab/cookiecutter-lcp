"""Export helpers for marimo analysis reports.

The preferred interface is :class:`SessionReportSaver`, an AnyWidget that
asks the browser for the experiment's ``reports/`` directory and saves both
HTML and PDF snapshots of the *current marimo session*.  It deliberately
uses marimo's live-session export endpoints, so no notebook cells are run
again and stochastic or expensive results cannot silently change.

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

import anywidget
import nbformat
import traitlets
from nbconvert import HTMLExporter
from playwright.sync_api import sync_playwright


class SessionReportSaver(anywidget.AnyWidget):
    """Browser control that saves the current marimo session as HTML + PDF.

    Directory handles are retained by the browser in IndexedDB.  Chromium
    browsers can therefore write directly to the chosen ``reports/`` folder;
    other browsers fall back to ordinary downloads.
    """

    basename = traitlets.Unicode("analysis_report").tag(sync=True)
    suggested_directory = traitlets.Unicode("reports").tag(sync=True)

    _esm = r"""
function render({ model, el }) {
  const root = document.createElement("section");
  root.className = "lcp-session-report-saver";
  root.innerHTML = `
    <style>
      .lcp-session-report-saver {
        border: 1px solid color-mix(in srgb, var(--blue-7, #2563eb) 28%, transparent);
        border-radius: 12px; padding: 16px; background: var(--slate-1, #f8fafc);
        color: var(--slate-12, #172033); font: 14px/1.5 system-ui, sans-serif;
      }
      .lcp-session-report-saver button {
        border: 0; border-radius: 8px; padding: 9px 14px; cursor: pointer;
        color: white; background: var(--blue-9, #2563eb); font-weight: 650;
      }
      .lcp-session-report-saver button:disabled { opacity: .6; cursor: wait; }
      .lcp-session-report-saver code { overflow-wrap: anywhere; }
      .lcp-session-report-saver .hint { margin: 0 0 12px; color: var(--slate-11, #475569); }
      .lcp-session-report-saver .status { margin-top: 12px; white-space: pre-wrap; }
      .lcp-session-report-saver .ok { color: #08783e; }
      .lcp-session-report-saver .error { color: #b42318; }
      .lcp-session-report-saver .download-links { display: flex; gap: 8px; margin-top: 10px; }
      .lcp-session-report-saver .download-links a {
        border: 1px solid currentColor; border-radius: 7px; padding: 6px 10px;
        color: var(--blue-9, #2563eb); text-decoration: none; font-weight: 650;
      }
      @media print { .lcp-session-report-saver { display: none !important; } }
    </style>
    <p class="hint">Suggested folder: <code></code><br>
      Select this folder the first time. The browser will remember it for future exports.</p>
    <button type="button">Save record (HTML + PDF)</button>
    <div class="status" role="status" aria-live="polite">No cells will be run again.</div>`;

  const code = root.querySelector("code");
  const button = root.querySelector("button");
  const status = root.querySelector(".status");
  code.textContent = model.get("suggested_directory");
  el.replaceChildren(root);

  const setStatus = (message, kind = "") => {
    status.className = `status ${kind}`;
    status.textContent = message;
  };

  const openDatabase = () => new Promise((resolve, reject) => {
    const request = indexedDB.open("lcp-analysis-report-folders", 1);
    request.onupgradeneeded = () => request.result.createObjectStore("handles");
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });

  async function storedHandle(operation, value) {
    const db = await openDatabase();
    return await new Promise((resolve, reject) => {
      const transaction = db.transaction("handles", operation === "get" ? "readonly" : "readwrite");
      const store = transaction.objectStore("handles");
      const request = operation === "get"
        ? store.get("reports-directory")
        : store.put(value, "reports-directory");
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    }).finally(() => db.close());
  }

  async function chooseDirectory() {
    let handle = null;
    try { handle = await storedHandle("get"); } catch (_) { /* picker still works */ }
    if (handle) {
      let permission = await handle.queryPermission({ mode: "readwrite" });
      if (permission === "prompt") permission = await handle.requestPermission({ mode: "readwrite" });
      if (permission === "granted") return handle;
    }
    handle = await window.showDirectoryPicker({
      id: "lcp-analysis-reports", mode: "readwrite", startIn: "documents"
    });
    try { await storedHandle("put", handle); } catch (_) { /* stable picker id also remembers it */ }
    return handle;
  }

  async function runtimeClient() {
    const urls = performance.getEntriesByType("resource").map((entry) => entry.name);
    for (const url of urls.filter((value) => /\/assets\/config-[^/]+\.js(?:\?|$)/.test(value))) {
      try {
        const module = await import(url);
        if (typeof module.r !== "function") continue;
        const client = module.r();
        if (client && typeof client.headers === "function" && typeof client.formatHttpURL === "function") {
          return client;
        }
      } catch (_) { /* try the next config module */ }
    }
    throw new Error("The current marimo session could not be accessed. Reload the page and try again.");
  }

  async function exportBlob(client, path, body) {
    const response = await fetch(client.formatHttpURL({ path }).toString(), {
      method: "POST",
      headers: { ...client.headers(), "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!response.ok) {
      const detail = await response.text();
      throw new Error(`Export failed at ${path} (${response.status}): ${detail.slice(0, 300)}`);
    }
    return await response.blob();
  }

  async function improveHtml(blob) {
    const css = `<style id="lcp-report-readability">
      :root { color-scheme: light; }
      body { line-height: 1.55; }
      main, marimo-island { max-width: 1180px; margin-inline: auto; }
      img, svg, canvas { max-width: 100%; height: auto; }
      pre, code { white-space: pre-wrap; overflow-wrap: anywhere; }
      table { display: block; max-width: 100%; overflow-x: auto; border-collapse: collapse; }
      th, td { padding: .35rem .55rem; vertical-align: top; }
      @media print {
        details { display: block !important; }
        pre, figure, table { break-inside: avoid; }
        a { color: inherit; text-decoration: none; }
      }
    </style>`;
    const source = await blob.text();
    const enhanced = source.includes("</head>")
      ? source.replace("</head>", `${css}</head>`)
      : `${css}${source}`;
    return new Blob([enhanced], { type: "text/html;charset=utf-8" });
  }

  const timestamp = () => {
    const now = new Date();
    const pad = (n) => String(n).padStart(2, "0");
    return `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}_${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
  };

  async function writeFile(directory, name, blob) {
    const file = await directory.getFileHandle(name, { create: true });
    const writable = await file.createWritable();
    await writable.write(blob);
    await writable.close();
  }

  const fallbackUrls = [];
  function offerDownloads(files, isSafari) {
    setStatus(
      isSafari
        ? "Safari cannot write directly to the reports/ folder or start both downloads from one click. Use each button below to save the HTML and PDF separately. No cells were run again."
        : "This browser cannot write directly to a folder. Use each button below to save the HTML and PDF separately. No cells were run again.",
      "ok",
    );
    const links = document.createElement("div");
    links.className = "download-links";
    for (const [label, name, blob] of files) {
      const url = URL.createObjectURL(blob);
      fallbackUrls.push(url);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = name;
      anchor.textContent = label;
      links.append(anchor);
    }
    status.append(links);
  }

  button.addEventListener("click", async () => {
    button.disabled = true;
    try {
      fallbackUrls.splice(0).forEach((url) => URL.revokeObjectURL(url));
      let directory = null;
      if ("showDirectoryPicker" in window) {
        setStatus("Choose the experiment's reports/ folder…");
        directory = await chooseDirectory();
      }
      setStatus("Capturing the current session as HTML and PDF…");
      const client = await runtimeClient();
      const [rawHtml, pdf] = await Promise.all([
        exportBlob(client, "/api/export/html", {
          download: false, files: [], includeCode: true, assetUrl: null
        }),
        exportBlob(client, "/api/export/pdf", {
          webpdf: true, preset: "document", includeInputs: false,
          rasterizeOutputs: false, rasterScale: 4.0, rasterServer: "static"
        }),
      ]);
      const html = await improveHtml(rawHtml);
      const suffix = timestamp();
      const stem = model.get("basename").replace(/[^A-Za-z0-9._-]+/g, "_");
      const htmlName = `${stem}_${suffix}.html`;
      const pdfName = `${stem}_${suffix}.pdf`;
      if (directory) {
        await writeFile(directory, htmlName, html);
        await writeFile(directory, pdfName, pdf);
        setStatus(`✓ Record saved in the selected folder:\n  ${htmlName}\n  ${pdfName}\nNo cells were run again.`, "ok");
      } else {
        const isSafari = /^((?!chrome|android).)*safari/i.test(navigator.userAgent);
        offerDownloads(
          [
            ["Download HTML", htmlName, html],
            ["Download PDF", pdfName, pdf],
          ],
          isSafari,
        );
      }
    } catch (error) {
      if (error && error.name === "AbortError") {
        setStatus("Saving was cancelled; no files were created.");
      } else {
        setStatus(`The record could not be saved: ${error?.message ?? error}`, "error");
      }
    } finally {
      button.disabled = false;
    }
  });
}
export default { render };
"""

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
