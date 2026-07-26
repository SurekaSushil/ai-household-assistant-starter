"""Call POST /chat at two temperatures and print the answers.

Requires the FastAPI server to already be running, e.g.:
    uv run fastapi dev app/main.py

Then in another terminal:
    uv run python scripts/compare_temperature.py
"""

from __future__ import annotations

import httpx

BASE_URL = "http://127.0.0.1:8000"
ITERATIONS = 3
TEMPERATURES = (0.2, 0.8)

MESSAGE = (
    "Name three possible causes of a dishwasher not draining, "
    "in one short paragraph."
)
SYSTEM_PROMPT = (
    "You are a careful household equipment assistant. "
    "Answer clearly, state uncertainty, and do not invent part numbers."
)


def chat(temperature: float) -> dict:
    payload = {
        "message": MESSAGE,
        "system_prompt": SYSTEM_PROMPT,
        "temperature": temperature,
    }
    with httpx.Client(timeout=180.0) as client:
        response = client.post(f"{BASE_URL}/chat", json=payload)
        response.raise_for_status()
        return response.json()


def main() -> None:
    print(f"Prompt: {MESSAGE}\n")

    for temperature in TEMPERATURES:
        print("=" * 60)
        print(f"temperature = {temperature}")
        print("=" * 60)

        for i in range(1, ITERATIONS + 1):
            data = chat(temperature)
            print(f"\n--- run {i} ---")
            print(data["answer"])
            print(
                f"(model={data['model']}, "
                f"output_tokens={data.get('output_tokens')})"
            )
        print()


if __name__ == "__main__":
    main()
