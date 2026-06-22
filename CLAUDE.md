# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Single-script pipeline that builds a SQLite vocab database (`vocab.db`) for Japanese flashcards from a ranked frequency corpus (`corpus.txt`), enriched per-word via an LLM call (reading, example sentences, kanji mnemonic).

## Commands

```bash
# setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# run the build (resumable — re-running skips rows already done)
python build_db.py                              # openrouter backend, default model, 1 worker
python build_db.py --backend ollama --host http://gb10-001:11434
python build_db.py --limit 20 --workers 4        # quick test run
python build_db.py --think                      # enable model reasoning (~10-15x slower)
```

`OPENROUTER_API_KEY` must be set in the environment when `--backend openrouter` (the default) is used.

No test suite, linter, or build step beyond the script itself.

## Architecture

- `corpus.txt` — Anki-export-style TSV (`#separator:tab` / `#html:false` header, then `rank\tword\tpos\tgloss\tfreqstats` lines). `freqstats` is itself `|`-delimited: `count | percent | tag`. Only the first 5000 ranked rows are used (`parse_corpus`).
- `build_db.py` — the entire pipeline:
  - `init_db` creates two tables: `words` (one row per corpus rank, tracks `status`/`attempts`/`error` for resumability) and `examples` (FK `word_rank -> words.rank`).
  - Per word, `compute_row` builds a JSON-output prompt (`build_prompt`) and calls either OpenRouter (`call_openrouter`) or a local Ollama instance (`call_ollama`) via `call_llm`. This is pure network/CPU work, safe to run concurrently in a `ThreadPoolExecutor`.
  - The LLM-provided reading is cross-checked against SudachiPy's dictionary reading (`dict_reading`); on mismatch, `reading_mismatch` is flagged but both readings are kept rather than one being discarded.
  - `persist_result` does all SQLite writes and runs only on the main thread — sqlite3 connections here are not thread-safe, so DB access is deliberately kept off the worker threads.
  - Main loop polls `words` for `status IN ('pending', 'error')` rows below `--max-attempts`, dispatches a round through the thread pool, persists results, and repeats (`retry round N`) until nothing pending/retryable remains. Rows that exhaust `--max-attempts` end as `status = 'failed'`.
- `notify.sh` — posts a Telegram message; used as an external completion/error notifier for long-running builds (invoked manually or piped from build output, not called by `build_db.py` itself). Contains a live bot token — treat as a secret, don't echo its contents back or commit changes that expose it further.
- `vocab.db` — generated artifact, not hand-edited. Full pipeline db (every enrichment script's `status`/`attempts`/`error` tracking columns, QC audit trails, etc.) — local working file only, gitignored, not deployed.
- `build_public_db.py` — strips `vocab.db` down to only the columns `app.js` actually queries (see its `SELECT`s) and writes `vocab_public.db`, the file actually fetched by the browser (sql.js loads the whole db client-side, so anything not in this slim copy still bloats every visitor's download). Run this after any pipeline change, before deploying. `vocab_public.db` is the only db file committed (via git-lfs, like `audio/`/`images/`).
