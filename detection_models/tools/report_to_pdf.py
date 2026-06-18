"""Render reports/speed_report.md to a PDF with minimal page margins.

Markdown -> styled HTML -> PDF via headless Google Chrome. Chrome honours the
``@page { margin: ... }`` CSS, so margins are set as small as is practical and
the page is landscape A4 to fit the wide results table.

    uv run python tools/report_to_pdf.py
    uv run python tools/report_to_pdf.py --md reports/speed_report.md --margin 5mm
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]

CSS = """
@page {{ size: A4 landscape; margin: {margin}; }}
* {{ box-sizing: border-box; }}
body {{ font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        font-size: 10pt; color: #1a1a1a; margin: 0; }}
h1 {{ font-size: 15pt; margin: 0 0 6px; }}
ul {{ margin: 4px 0 10px; padding-left: 18px; }}
li {{ margin: 1px 0; }}
code {{ font-family: "DejaVu Sans Mono", Menlo, Consolas, monospace; font-size: 9pt;
        background: #f2f2f2; padding: 0 2px; border-radius: 2px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 8.5pt; }}
th, td {{ border: 1px solid #bbb; padding: 3px 6px; text-align: left;
          white-space: nowrap; }}
th {{ background: #2d2d2d; color: #fff; }}
tr:nth-child(even) td {{ background: #f6f6f6; }}
blockquote {{ font-size: 8.5pt; color: #444; border-left: 3px solid #ccc;
              margin: 8px 0; padding: 2px 10px; }}
"""


def find_chrome() -> str:
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        path = shutil.which(name)
        if path:
            return path
    raise SystemExit("No Chrome/Chromium found to print the PDF.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--md", type=Path, default=ROOT / "reports" / "speed_report.md")
    ap.add_argument("--out", type=Path, default=None, help="output PDF (default: <md>.pdf)")
    ap.add_argument("--margin", default="5mm", help="page margin (CSS unit), e.g. 5mm or 0")
    args = ap.parse_args()

    if not args.md.exists():
        raise SystemExit(f"{args.md} not found — run the benchmark first.")
    out = args.out or args.md.with_suffix(".pdf")

    html_body = markdown.markdown(args.md.read_text(), extensions=["tables"])
    html = (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<style>{CSS.format(margin=args.margin)}</style></head>"
            f"<body>{html_body}</body></html>")

    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as f:
        f.write(html)
        html_path = Path(f.name)
    try:
        subprocess.run(
            [find_chrome(), "--headless=new", "--no-pdf-header-footer",
             "--no-sandbox", f"--print-to-pdf={out}", html_path.as_uri()],
            check=True, capture_output=True, text=True,
        )
    finally:
        html_path.unlink(missing_ok=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
