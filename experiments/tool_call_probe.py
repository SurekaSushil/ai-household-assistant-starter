"""One-file probe: ask Ollama for a calculator tool call. No FastAPI, no execution."""

import asyncio
import json

import httpx

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate basic arithmetic.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    }
]


async def main() -> None:
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "qwen3:4b-instruct",
                "messages": [
                    {"role": "user", "content": "What is (37 * 19) + 8?"}
                ],
                "tools": TOOLS,
                "stream": False,
            },
        )
        response.raise_for_status()
        print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
