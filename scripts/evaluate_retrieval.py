"""Evaluate retrieval quality against evaluation/rag_cases.json.

Requires FastAPI to be running, e.g.:
    uv run uvicorn app.main:app --host 127.0.0.1 --port 8001

Then:
    uv run python scripts/evaluate_retrieval.py
    uv run python scripts/evaluate_retrieval.py --base-url http://127.0.0.1:8001
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "evaluation" / "rag_cases.json"


def load_cases() -> list[dict]:
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))


def search(
    client: httpx.Client,
    *,
    question: str,
    document_id: str | None,
    top_k: int,
) -> list[dict]:
    payload = {
        "question": question,
        "document_id": document_id,
        "top_k": top_k,
    }
    response = client.post("/search", json=payload)
    response.raise_for_status()
    return response.json()


def ask(
    client: httpx.Client,
    *,
    question: str,
    document_id: str | None,
) -> dict:
    payload = {
        "question": question,
        "document_id": document_id,
        "temperature": 0.2,
    }
    response = client.post("/ask", json=payload)
    response.raise_for_status()
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval cases.")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8001",
        help="FastAPI base URL (default: http://127.0.0.1:8001)",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--check-unsupported-ask",
        action="store_true",
        help="Also call /ask for unsupported cases and check for insufficiency wording.",
    )
    args = parser.parse_args()

    cases = load_cases()
    supported = [c for c in cases if c.get("kind") != "unsupported"]
    unsupported = [c for c in cases if c.get("kind") == "unsupported"]

    top1_hits = 0
    top5_hits = 0
    rows: list[str] = []

    with httpx.Client(base_url=args.base_url, timeout=180.0) as client:
        try:
            client.get("/health").raise_for_status()
        except httpx.ConnectError:
            raise SystemExit(
                f"Cannot connect to {args.base_url}.\n"
                "Start FastAPI first in another terminal, then re-run this script:\n"
                "  uv run uvicorn app.main:app --host 127.0.0.1 --port 8001\n"
                "  uv run python scripts/evaluate_retrieval.py"
            ) from None

        for case in supported:
            hits = search(
                client,
                question=case["question"],
                document_id=case["document_id"],
                top_k=args.top_k,
            )
            pages = [int(hit["page_number"]) for hit in hits]
            expected = set(case["expected_pages"])
            top1 = bool(pages) and pages[0] in expected
            top5 = any(page in expected for page in pages[:5])
            top1_hits += int(top1)
            top5_hits += int(top5)
            status = "OK" if top5 else "MISS"
            rows.append(
                f"[{status}] {case['name']}: pages={pages[:5]} "
                f"expected={sorted(expected)} top1={top1} top5={top5}"
            )

        unsupported_ok = 0
        if args.check_unsupported_ask:
            for case in unsupported:
                result = ask(
                    client,
                    question=case["question"],
                    document_id=case["document_id"],
                )
                answer = result.get("answer", "").lower()
                ok = (
                    "insufficient" in answer
                    or "not support" in answer
                    or "do not support" in answer
                    or "doesn't support" in answer
                    or "cannot determine" in answer
                )
                unsupported_ok += int(ok)
                status = "OK" if ok else "WEAK"
                rows.append(
                    f"[{status}] {case['name']} (unsupported ask): "
                    f"{result.get('answer', '')[:160]!r}"
                )

    print("Retrieval evaluation")
    print(f"base_url: {args.base_url}")
    print(f"cases file: {CASES_PATH}")
    print()
    for row in rows:
        print(row)
    print()

    if supported:
        top1_acc = top1_hits / len(supported)
        top5_acc = top5_hits / len(supported)
        print(f"Supported cases: {len(supported)}")
        print(f"Top-1 accuracy: {top1_acc:.0%} ({top1_hits}/{len(supported)})")
        print(f"Top-5 accuracy: {top5_acc:.0%} ({top5_hits}/{len(supported)})")
        if top5_acc >= 0.8:
            print("Week 2 target: Top-5 >= 80%  PASS")
        else:
            print("Week 2 target: Top-5 >= 80%  NOT YET")

    print(f"Unsupported cases defined: {len(unsupported)}")
    if args.check_unsupported_ask and unsupported:
        print(
            "Unsupported /ask insufficiency: "
            f"{unsupported_ok}/{len(unsupported)}"
        )


if __name__ == "__main__":
    main()
