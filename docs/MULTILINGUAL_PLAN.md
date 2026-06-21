# Multilingual deck plan

Goal: reuse the JP pipeline (corpus → LLM enrich → SQLite → TTS → images → GH Pages)
to build decks for other languages. Build order, ranked by popularity × simplicity
(latin/simple script, mature TTS, no reading-disambiguation problem):

1. Spanish
2. French
3. German
4. Italian
5. Portuguese
6. Korean
7. Russian
8. Indonesian
9. Turkish
10. Vietnamese

Skip-tier (complexity ~= or worse than Japanese): Mandarin, Arabic, Thai.

## Phase 0 — generalize the pipeline (do once, before lang #1)

Current scripts are JP-hardcoded: `build_db.py`, `generate_audio.py`,
`generate_breakdown.py`, `generate_image_prompts.py`, `generate_images.py`.
Each duplicates corpus parsing + resumable-row-loop + threading. Refactor before
scaling to 10 languages, don't copy-paste per language.

- [ ] Extract shared lib (`pipeline/common.py` or similar):
  - corpus parsing (rank/word/pos/gloss/freqstats TSV reader)
  - SQLite schema + resumable main-loop (status pending/done/error/failed, retry rounds, main-thread-only writes)
  - LLM client wrapper (openrouter/ollama backends, JSON-output prompting, threaded dispatch)
- [ ] Per-language config object: corpus source path, LLM prompt template (fields differ:
  JP needs reading+kanji mnemonic, FR/DE need gender/article, ES/IT mostly just example
  sentences), reading-check tool (SudachiPy for JP only — skip for latin-script langs,
  no ambiguity), TTS backend + voice/model, output paths.
- [ ] Restructure outputs per lang: `langs/<code>/{corpus.txt, vocab.db, audio/, images/}`
  instead of flat root files (root stays as-is for `ja`, or gets moved under `langs/ja/`).
- [ ] Site (`index.html`/`app.js`/`style.css`): add language switcher, point at
  per-lang `vocab.db` + asset folders.
- [ ] TTS backend survey per language family (latin-script vs Korean vs Russian) —
  confirm Piper/Coqui voice availability before committing to phase-1 lang.

## Phase N — per language (repeat for each, in rank order above)

1. Source frequency corpus (top 5000), convert to existing TSV format.
2. Run LLM enrich pass (reuse shared lib, lang-specific prompt config).
3. Run TTS pass (lang-specific backend/voice; Piper/Coqui for latin langs — confirm
   Korean/Russian options in phase 0 survey).
4. Run image-prompt + image-gen pass (reuse as-is, just feeds translated word/gloss).
5. Wire into site (language switcher entry), publish to GH Pages.
6. Update `MEMORY.md` / project memory with completion status (counts, any blockers).

## Notes / open questions

- Corpus sourcing per language not yet decided — likely existing frequency
  lists (Wiktionary frequency lists, or repurpose other Anki frequency decks).
- TTS backend choice per language deferred to phase 0 survey, not decided yet.
- No reading-disambiguation step needed for any of the 10 (all latin or Hangul/Cyrillic
  alphabetic — no logographic ambiguity like kanji).
