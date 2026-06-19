"""Render presentation.md to a PDF (markdown -> styled HTML -> headless Chrome).

Portrait A4, images scaled to fit, light page-break hints between sections.

    python make_pdf.py
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
MD = HERE / "presentation.md"
OUT = HERE / "presentation.pdf"

CSS = """
@page { size: A4 portrait; margin: 14mm 14mm 16mm; }
* { box-sizing: border-box; }
body { font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
       font-size: 11pt; line-height: 1.45; color: #1a1a1a; margin: 0; }
h1 { font-size: 22pt; margin: 0 0 4px; }
h2 { font-size: 15pt; margin: 22px 0 8px; padding-top: 6px;
     border-top: 2px solid #2d2d2d; page-break-after: avoid; }
h3 { font-size: 12pt; margin: 16px 0 6px; page-break-after: avoid; }
p { margin: 6px 0; }
ul { margin: 6px 0 10px; padding-left: 20px; }
li { margin: 3px 0; }
em { color: #444; }
code { font-family: "DejaVu Sans Mono", Menlo, Consolas, monospace; font-size: 9.5pt;
       background: #f2f2f2; padding: 0 3px; border-radius: 2px; }
pre { background: #f6f6f6; border: 1px solid #e2e2e2; border-radius: 4px;
      padding: 10px 12px; overflow-x: auto; page-break-inside: avoid; }
pre code { background: none; padding: 0; font-size: 8.5pt; line-height: 1.35; }
table { border-collapse: collapse; width: 100%; font-size: 9.5pt; margin: 10px 0;
        page-break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 4px 8px; text-align: left; }
th { background: #2d2d2d; color: #fff; }
tr:nth-child(even) td { background: #f6f6f6; }
img { display: block; max-width: 88%; max-height: 150mm; height: auto; margin: 10px auto 4px;
      border: 1px solid #ddd; border-radius: 4px; page-break-inside: avoid; }
blockquote { font-size: 10pt; color: #444; border-left: 3px solid #ccc;
             margin: 10px 0; padding: 4px 12px; background: #fafafa; }
hr { display: none; }
"""


def find_chrome() -> str:
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        path = shutil.which(name)
        if path:
            return path
    raise SystemExit("No Chrome/Chromium found to print the PDF.")


def main() -> None:
    if not MD.exists():
        raise SystemExit(f"{MD} not found.")
    html_body = markdown.markdown(
        MD.read_text(), extensions=["tables", "fenced_code", "sane_lists"]
    )
    html = (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<style>{CSS}</style></head><body>{html_body}</body></html>")

    # Write the temp HTML inside this folder so relative image paths resolve.
    with tempfile.NamedTemporaryFile("w", suffix=".html", dir=HERE, delete=False) as f:
        f.write(html)
        html_path = Path(f.name)
    try:
        subprocess.run(
            [find_chrome(), "--headless=new", "--no-pdf-header-footer",
             "--no-sandbox", f"--print-to-pdf={OUT}", html_path.as_uri()],
            check=True, capture_output=True, text=True,
        )
    finally:
        html_path.unlink(missing_ok=True)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
