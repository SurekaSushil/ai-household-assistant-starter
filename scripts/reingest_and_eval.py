"""Re-upload a PDF, point evaluation cases at the new document_id, run retrieval eval.

Example:
    uv run python scripts/reingest_and_eval.py "ge dishwasher.pdf"
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "evaluation" / "rag_cases.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    pdf = args.pdf if args.pdf.is_absolute() else ROOT / args.pdf
    with httpx.Client(base_url=args.base_url, timeout=900.0) as client:
        client.get("/health").raise_for_status()
        response = client.post(
            "/documents/upload",
            files={"file": (pdf.name, pdf.read_bytes(), "application/pdf")},
        )
        response.raise_for_status()
        payload = response.json()

    document_id = payload["document_id"]
    print(
        f"uploaded document_id={document_id} "
        f"pages={payload.get('pages_with_text')} chunks={payload.get('chunks_stored')}"
    )

    text = CASES_PATH.read_text(encoding="utf-8")
    updated = re.sub(
        r'"document_id":\s*"[0-9a-fA-F-]{36}"',
        f'"document_id": "{document_id}"',
        text,
    )
    CASES_PATH.write_text(updated, encoding="utf-8")
    print(f"updated {CASES_PATH}")

    if args.skip_eval:
        return

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_retrieval.py"),
            "--base-url",
            args.base_url,
        ],
        cwd=ROOT,
        check=False,
    )
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
