#!/usr/bin/env python3
"""Cross-check example sentence readings against SudachiPy's dictionary parse.

The LLM-generated `reading` for an example sentence is sometimes wrong in a
way the per-word reading check (words.reading_mismatch) can't catch — e.g. it
picks an isolated-dictionary-entry reading instead of the one a sentence's
grammatical context calls for ("後で" read as "のち" instead of "あとで").
SudachiPy parses the whole sentence morphologically (the same class of tool
TTS frontends like misaki/MeCab use). It catches real context errors (e.g.
"後で" mis-read as "のち" instead of "あとで"), but it also disagrees on common
heteronyms purely because the dictionary cites a formal/literary reading
(私→わたくし, 日本→にっぽん, 明日→あす) where the LLM picked the casual one
that actually fits these example sentences — not every mismatch means the LLM
is wrong. So this only flags; it does not overwrite `jp_reading`. A human
reviews flagged rows (surfaced in the UI as a warning badge) and decides
which reading is right per-sentence.

For every row in `examples`, this:
  - computes the dictionary reading of `jp` via SudachiPy
  - backs up the original LLM reading into `reading_llm` (first run only)
  - stores the dictionary reading in `reading_dict`
  - flags `reading_mismatch` when they disagree (ignoring whitespace), unless
    the only difference is one of the known heteronyms in HETERONYMS below
    (the dictionary only ever returns one of several valid readings for these,
    so a difference there isn't a sign of an LLM error)

`jp_reading` (the text the UI displays) is left untouched.

Deterministic and idempotent (no LLM, no network), so safe to re-run after
edits to examples.jp.
"""
import itertools
import sqlite3

from build_db import dict_reading, get_tokenizer, normalize_reading

# Words where SudachiPy's dictionary always returns one citation reading even
# though the other is just as valid (and usually what these example sentences
# actually use). Add to this as more recurring false positives turn up.
HETERONYMS = {
    "私": ["わたし", "わたくし"],
    "明日": ["あした", "あす", "みょうにち"],
    "日本": ["にほん", "にっぽん"],
    "今日": ["きょう", "こんにち"],
    "何": ["なに", "なん"],
}


def ensure_columns(conn):
    cols = {row[1] for row in conn.execute("PRAGMA table_info(examples)")}
    if "reading_llm" not in cols:
        conn.execute("ALTER TABLE examples ADD COLUMN reading_llm TEXT")
    if "reading_dict" not in cols:
        conn.execute("ALTER TABLE examples ADD COLUMN reading_dict TEXT")
    if "reading_mismatch" not in cols:
        conn.execute("ALTER TABLE examples ADD COLUMN reading_mismatch INTEGER NOT NULL DEFAULT 0")
    conn.commit()


def strip_spaces(s):
    return (s or "").replace(" ", "").replace("　", "")


def heteronym_match(jp, llm_reading_norm):
    """True if llm_reading_norm matches jp under some combination of accepted
    heteronym readings (see HETERONYMS), i.e. the dict/LLM disagreement is
    just a formal-vs-casual reading choice, not an actual error."""
    tokens = list(get_tokenizer().tokenize(jp))
    options = [HETERONYMS.get(t.surface(), [normalize_reading(t.reading_form())]) for t in tokens]
    n_combo = 1
    for o in options:
        n_combo *= len(o)
    if n_combo == 1 or n_combo > 32:
        return False  # no heteronym tokens here, or too many combos to bother
    return any(strip_spaces("".join(c)) == llm_reading_norm for c in itertools.product(*options))


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Cross-check example readings against SudachiPy, flag mismatches.")
    ap.add_argument("--db", default="vocab.db")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    ensure_columns(conn)

    rows = conn.execute("SELECT id, jp, jp_reading, reading_llm FROM examples ORDER BY id").fetchall()
    n_mismatch = 0
    for row in rows:
        llm_reading = row["reading_llm"] or row["jp_reading"] or ""
        dict_val = dict_reading(row["jp"])
        llm_norm = strip_spaces(normalize_reading(llm_reading))
        differs = dict_val and llm_norm != strip_spaces(dict_val)
        mismatch = 1 if differs and not heteronym_match(row["jp"], llm_norm) else 0
        if mismatch:
            n_mismatch += 1
        conn.execute(
            "UPDATE examples SET reading_llm = ?, reading_dict = ?, reading_mismatch = ? WHERE id = ?",
            (llm_reading, dict_val, mismatch, row["id"]),
        )
    conn.commit()
    print(f"checked {len(rows)} examples, {n_mismatch} mismatch(es) flagged")


if __name__ == "__main__":
    main()
