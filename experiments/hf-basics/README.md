# Session 6 — Hugging Face basics

Separate from the FastAPI + Ollama app. Goal: see tokenization and generation
directly with Transformers + CPU PyTorch.

## Setup (once)

From this folder:

```powershell
cd C:\Projects\ai_household_assistant_starter\experiments\hf-basics
uv sync
```

This installs:

- **PyTorch CPU** from the official CPU wheel index (`https://download.pytorch.org/whl/cpu`)
- **Transformers** from PyPI

Kept separate from the FastAPI app’s `.venv` on purpose.

## Run

```powershell
uv run python tokenize_and_generate.py
```

You should see:

1. Token IDs and decoded token pieces for one sentence
2. A short CPU text-generation sample from `sshleifer/tiny-gpt2`

First run downloads the model weights (small).
