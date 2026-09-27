# Retrieval tuning log

One variable changed per experiment. Re-ingest when stored text or embeddings change.

## Baseline (before any tuning)

| Field | Value |
|-------|--------|
| Date | 2026-07-26 |
| Change | none (freeze) |
| `chunk_size_chars` | 1200 |
| `chunk_overlap_chars` | 200 |
| `rag_top_k` | 5 |
| `ollama_embedding_model` | embeddinggemma |
| PDF extraction | plain `extract_text()`, strip only |
| Document | GE dishwasher manual |
| Top-1 | **69%** (11/16) |
| Top-5 | **100%** (16/16) |

Top-1 misses (right page in top 5, not first):

- `start-and-close-door`
- `water-temperature`
- `cancel-cycle`
- `leak-detected`
- `limited-warranty-proof`

Primary goal: do not regress Top-5. Secondary: raise Top-1.

## Experiment 1 — PDF extraction + page metadata

| Field | Value |
|-------|--------|
| Change | Improved `pdf_service`: normalize glued words / control chars; choose layout vs plain by meaningful length; parse `printed_page` near doc codes; store `printed_page` in payload. **Held constant:** chunk 1200/200, top_k 5, embeddinggemma |
| Document ID | `f127701c-9de6-4fd4-afb3-73799f471a13` |
| Top-1 | **75%** (12/16) — was 69% |
| Top-5 | **100%** (16/16) — unchanged |
| Delta | +1 Top-1 hit (`limited-warranty-proof`) |

Remaining Top-1 misses: `start-and-close-door`, `water-temperature`, `cancel-cycle`, `leak-detected`.

## Experiment 2a — chunk_size 800 (overlap held at 200)

| Field | Value |
|-------|--------|
| Change | `CHUNK_SIZE_CHARS=800` only |
| Document ID | `1c8f2151-0129-468a-9649-3f03ce3835d0` |
| Chunks stored | 373 (was ~242) |
| Top-1 | **75%** (12/16) — unchanged vs Exp 1 |
| Top-5 | **100%** (16/16) |
| Notes | `cancel-cycle` gained Top-1; `water-standing-in-tub` lost Top-1. No net gain. |

## Experiment 2b — chunk_size 1600 (overlap held at 200)

| Field | Value |
|-------|--------|
| Change | `CHUNK_SIZE_CHARS=1600` only |
| Document ID | `ab6ad0c0-23ab-4212-9625-116b76b5bb89` |
| Chunks stored | 184 |
| Top-1 | **75%** (12/16) — unchanged |
| Top-5 | **100%** (16/16) |
| Decision | Keep **1200** (Exp 1 winner config) — no size beat it on Top-1. |

## Experiment 2c — overlap 100 (size held at 1200)

| Field | Value |
|-------|--------|
| Change | `CHUNK_OVERLAP_CHARS=100` only |
| Document ID | `5db3544c-c38c-4e0b-acf6-4a7098390430` |
| Chunks stored | 224 |
| Top-1 | **88%** (14/16) — was 75% |
| Top-5 | **100%** (16/16) |
| Notes | Fixed Top-1 for `start-and-close-door`, `leak-detected`. Remaining misses: `water-temperature`, `cancel-cycle`. |

## Experiment 2d — overlap 300 (size held at 1200)

| Field | Value |
|-------|--------|
| Change | `CHUNK_OVERLAP_CHARS=300` only |
| Document ID | `bd8d44c3-f630-44a0-b424-232afbfec2c7` |
| Chunks stored | 251 |
| Top-1 | **81%** (13/16) — worse than overlap 100 |
| Top-5 | **100%** (16/16) |
| Decision | Keep **overlap 100** (best so far). |

## Experiment 3 — top_k 8 (index unchanged: 1200/100)

| Field | Value |
|-------|--------|
| Change | Eval `--top-k 8` only (no re-embed). App `RAG_TOP_K` still 5. |
| Document ID | `f159f107-9f83-408f-8848-138af5cc5584` |
| Top-1 | **88%** (14/16) — unchanged |
| Top-5 | **100%** (16/16) — unchanged (metric still inspects first 5 hits) |
| Decision | Keep **`rag_top_k=5`**. Raising k does not fix rank-1 misses. |

## Experiment 4 — embedding / rerank

| Field | Value |
|-------|--------|
| Change | **Skipped** |
| Reason | Top-1 is **88%** after Steps 1–3 (above the ~70% “still weak” threshold). Plan says only compare another embedding or add reranking if Top-1 stays weak. |
| Next time | If you add harder cases and Top-1 drops, try one alternate embed model XOR a reranker — not both at once. |

## Final chosen config

| Knob | Value |
|------|--------|
| PDF extraction | normalized plain/layout select + `printed_page` metadata |
| `chunk_size_chars` | **1200** |
| `chunk_overlap_chars` | **100** |
| `rag_top_k` | **5** |
| `ollama_embedding_model` | embeddinggemma |
| Document ID | `f159f107-9f83-408f-8848-138af5cc5584` |
| Top-1 | **88%** (was 69%) |
| Top-5 | **100%** (held) |
