# AI Household Assistant Starter

This is the first milestone in an AI application-engineering learning path:

**Browser / Swagger UI -> FastAPI -> Ollama HTTP API -> local open model**

## Prerequisites

- Windows 10 or 11
- Git
- uv
- Ollama
- Cursor or VS Code

## First run

Open PowerShell in this folder.

```powershell
uv sync
Copy-Item .env.example .env
ollama run gemma3:1b
```

Keep Ollama running. Open a second PowerShell terminal in this folder:

```powershell
uv run fastapi dev app/main.py
```

Open:

- http://127.0.0.1:8000/health
- http://127.0.0.1:8000/docs

In `/docs`, expand `POST /chat`, choose **Try it out**, and use:

```json
{
  "message": "What information do you need before identifying a dishwasher replacement part?",
  "temperature": 0.2
}
```

## Useful commands

```powershell
uv run python -m compileall app
uv run fastapi dev app/main.py
ollama list
git status
```

## Important

Do not add Docker, Qdrant, LangGraph, RunPod, or vLLM yet. First make sure you can explain every step in the request flow.
