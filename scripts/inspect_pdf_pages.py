"""Inspect PDF page extraction quality (learning / debugging helper).

Examples:
    uv run python scripts/inspect_pdf_pages.py "ge dishwasher.pdf" --pages 9,21,22
    uv run python scripts/inspect_pdf_pages.py data/sample_manual.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.pdf_service import extract_pdf_pages


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect extracted PDF page text.")
    parser.add_argument("pdf", type=Path, help="Path to a PDF file")
    parser.add_argument(
        "--pages",
        default="",
        help="Comma-separated PDF page numbers (default: all summary + first page sample)",
    )
    parser.add_argument("--chars", type=int, default=500, help="Chars to print per page")
    args = parser.parse_args()

    content = args.pdf.read_bytes()
    pages = extract_pdf_pages(content)
    by_number = {int(p["page_number"]): p for p in pages}

    print(f"file: {args.pdf}")
    print(f"pages_with_text: {len(pages)}")
    mismatches = [
        p
        for p in pages
        if p.get("printed_page") is not None
        and int(p["printed_page"]) != int(p["page_number"])
    ]
    print(f"printed_page mismatches: {len(mismatches)}")
    for p in mismatches[:10]:
        print(
            f"  pdf={p['page_number']} printed={p['printed_page']} "
            f"mode={p['extraction_mode']}"
        )

    if args.pages.strip():
        wanted = [int(x.strip()) for x in args.pages.split(",") if x.strip()]
    else:
        wanted = [int(pages[0]["page_number"])]

    for number in wanted:
        page = by_number.get(number)
        if page is None:
            print(f"\n--- page {number}: MISSING (no extractable text) ---")
            continue
        text = str(page["text"])
        print(
            f"\n--- page {number} | printed={page.get('printed_page')} "
            f"| mode={page.get('extraction_mode')} | chars={len(text)} ---"
        )
        print(text[: args.chars])
        if len(text) > args.chars:
            print("...")


if __name__ == "__main__":
    main()
