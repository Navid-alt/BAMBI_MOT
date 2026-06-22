"""Render docs/report.md -> docs/report.pdf via headless Chrome.

Markdown -> styled HTML -> PDF. The HTML is written next to report.md (in docs/)
so relative image paths (images/*.png) resolve during printing.

    PYTHONPATH=.. uv --directory ../detection_models run python docs/make_pdf.py
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent

CSS = """
@page { size: A4 portrait; margin: 14mm 12mm; }
* { box-sizing: border-box; }
body { font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
       font-size: 10.5pt; color: #1a1a1a; margin: 0; line-height: 1.45; }
h1 { font-size: 19pt; margin: 0 0 4px; }
h2 { font-size: 14pt; margin: 18px 0 6px; border-bottom: 2px solid #2563eb;
     padding-bottom: 2px; }
h3 { font-size: 11.5pt; margin: 12px 0 4px; color: #333; }
p, li { margin: 3px 0; }
img { max-width: 100%; height: auto; display: block; margin: 8px auto;
      border: 1px solid #ddd; }
code { font-family: "DejaVu Sans Mono", Menlo, Consolas, monospace; font-size: 9pt;
       background: #f2f2f2; padding: 0 3px; border-radius: 2px; }
pre { background: #f6f8fa; padding: 8px 10px; border-radius: 4px; font-size: 8.5pt;
      overflow-x: auto; }
pre code { background: none; }
table { border-collapse: collapse; width: 100%; font-size: 9pt; margin: 8px 0; }
th, td { border: 1px solid #bbb; padding: 3px 7px; text-align: left; }
th { background: #2d2d2d; color: #fff; }
tr:nth-child(even) td { background: #f6f6f6; }
hr { border: none; border-top: 1px solid #ccc; margin: 14px 0; }
"""


def find_chrome() -> str:
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        if shutil.which(name):
            return shutil.which(name)
    raise SystemExit("No Chrome/Chromium found to print the PDF.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--md", type=Path, default=HERE / "report.md")
    ap.add_argument("--out", type=Path, default=HERE / "report.pdf")
    args = ap.parse_args()

    body = markdown.markdown(args.md.read_text(),
                             extensions=["tables", "fenced_code"])
    html = (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<style>{CSS}</style></head><body>{body}</body></html>")
    html_path = HERE / "report.html"
    html_path.write_text(html)
    try:
        subprocess.run(
            [find_chrome(), "--headless=new", "--no-pdf-header-footer", "--no-sandbox",
             f"--print-to-pdf={args.out}", html_path.as_uri()],
            check=True, capture_output=True, text=True,
        )
    finally:
        html_path.unlink(missing_ok=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
