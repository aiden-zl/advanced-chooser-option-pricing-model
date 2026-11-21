"""Render HTML report sources to PDF with headless Google Chrome.

Usage:  python -m src.build_pdf deliverables/week1/data_specification.html [more.html ...]
Each PDF is written next to its HTML source.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    shutil.which("google-chrome") or "",
    shutil.which("chromium") or "",
]


def chrome() -> str:
    for c in CHROME_CANDIDATES:
        if c and Path(c).exists():
            return c
    raise SystemExit("Google Chrome / Chromium not found; needed to print PDFs.")


def build(html: Path) -> Path:
    html = html.resolve()
    pdf = html.with_suffix(".pdf")
    subprocess.run(
        [chrome(), "--headless", "--disable-gpu", "--no-pdf-header-footer",
         "--allow-file-access-from-files", f"--print-to-pdf={pdf}", html.as_uri()],
        check=True, capture_output=True, timeout=120,
    )
    print(f"wrote {pdf}")
    return pdf


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        build(Path(arg))
