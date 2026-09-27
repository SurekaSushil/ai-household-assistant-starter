# Learning Log

Use this file after every session.

## Date

2026-07-26 — retrieval tuning

## What I changed

1. PDF extraction: strip control chars, unglue words (`DISHWASHERGetting` → `DISHWASHER Getting`), pick layout only when it adds real alphanumeric content, keep PDF page index as citation truth, store optional `printed_page`.
2. Chunk size trials at 800 / 1600 (no Top-1 gain) → kept 1200.
3. Overlap trials at 100 / 300 → **100 won** (88% Top-1).
4. `top_k=8` measured; no Top-1/Top-5 change → kept 5.
5. Skipped new embedding/rerank because Top-1 is strong.

Scores: baseline 69%/100% → final **88%/100%**. Log: `evaluation/tuning_log.md`.

## What I can now explain without looking it up

- Re-upload is required after extraction or chunk changes because Qdrant stores frozen embeddings of old chunk text.
- Change one knob at a time or you cannot tell what helped.
- Top-5 can be perfect while Top-1 is weak — ranking inside the top 5 matters for `/ask` quality.
- Higher `top_k` does not fix a wrong #1 hit; it only gives the LLM more context.

## What failed

- Chunk size alone did not move the aggregate Top-1 score.
- Larger overlap (300) hurt vs smaller overlap (100).

## Error message and root cause

- Upload 502 from Ollama once (`dial tcp ... actively refused`) — embedding runner not ready; retry succeeded.

## One experiment for the next session

- Manually audit ≥10 `/ask` citations on the final index, and run `--check-unsupported-ask`.
