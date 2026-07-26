"""Confirm POST /chat/stream delivers partial text before completion.

Requires the FastAPI server to already be running, e.g.:
    uv run fastapi dev app/main.py

Then:
    uv run python scripts/test_stream.py
"""

from __future__ import annotations

import time

import httpx

BASE_URL = "http://127.0.0.1:8000"

payload = {
    "message": "List five dishwasher maintenance tips, one short sentence each.",
    "system_prompt": "You are concise.",
    "temperature": 0.2,
}


def main() -> None:
    print("Streaming from POST /chat/stream ...\n")
    started = time.perf_counter()
    first_chunk_at: float | None = None
    chunks = 0
    total_chars = 0

    with httpx.Client(timeout=180.0) as client:
        with client.stream(
            "POST",
            f"{BASE_URL}/chat/stream",
            json=payload,
        ) as response:
            response.raise_for_status()
            for chunk in response.iter_text():
                if not chunk:
                    continue
                now = time.perf_counter()
                if first_chunk_at is None:
                    first_chunk_at = now
                    print(
                        f"[first chunk after {first_chunk_at - started:.2f}s]\n"
                    )
                chunks += 1
                total_chars += len(chunk)
                print(chunk, end="", flush=True)

    finished = time.perf_counter()
    print("\n")
    print("---")
    if first_chunk_at is None:
        print("No chunks received.")
        return

    print(f"chunks: {chunks}")
    print(f"chars: {total_chars}")
    print(f"time to first chunk: {first_chunk_at - started:.2f}s")
    print(f"time to complete: {finished - started:.2f}s")
    if finished - first_chunk_at > 0.05 and chunks > 1:
        print("OK: partial text arrived before the stream finished.")
    else:
        print("WARNING: could not clearly confirm incremental streaming.")


if __name__ == "__main__":
    main()
