"""Call POST /chat with two system prompts and print the answers.

Requires the FastAPI server to already be running, e.g.:
    uv run fastapi dev app/main.py

Then in another terminal:
    uv run python scripts/compare_system_prompt.py
"""

from __future__ import annotations

import httpx

BASE_URL = "http://127.0.0.1:8000"
ITERATIONS = 3
TEMPERATURE = 0.2

MESSAGE = (
    "My Bosch dishwasher is not draining. "
    "Which replacement part should I buy?"
)

SYSTEM_PROMPTS = {
    "A": "You are concise. Answer in no more than three bullets.",
    "B": (
        "You are a cautious appliance technician. "
        "State what information is missing and do not guess part numbers."
    ),
}


def chat(system_prompt: str) -> dict:
    payload = {
        "message": MESSAGE,
        "system_prompt": system_prompt,
        "temperature": TEMPERATURE,
    }
    with httpx.Client(timeout=180.0) as client:
        response = client.post(f"{BASE_URL}/chat", json=payload)
        response.raise_for_status()
        return response.json()


def main() -> None:
    print(f"Prompt: {MESSAGE}")
    print(f"temperature: {TEMPERATURE}\n")

    for label, system_prompt in SYSTEM_PROMPTS.items():
        print("=" * 60)
        print(f"system prompt {label}")
        print(f"{system_prompt}")
        print("=" * 60)

        for i in range(1, ITERATIONS + 1):
            data = chat(system_prompt)
            print(f"\n--- run {i} ---")
            print(data["answer"])
            print(
                f"(model={data['model']}, "
                f"output_tokens={data.get('output_tokens')})"
            )
        print()


if __name__ == "__main__":
    main()
