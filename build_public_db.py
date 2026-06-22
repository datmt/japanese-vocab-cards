#!/usr/bin/env python3
"""Build a slim, browser-facing vocab_public.db from the full pipeline db.

app.js fetches the whole db file into the browser via sql.js (no server-side
querying), so every byte in the deployed db is downloaded by every visitor.
The full db (vocab.db) carries ~25 extra pipeline/QC tracking columns
(status/attempts/error for every enrichment stage, gloss review audit trail,
STT/translation check scratch fields, etc.) that app.js never reads — see
the SELECT statements in app.js for the exact column set actually used.

This script copies only those columns into a fresh vocab_public.db. Run
after any pipeline/content change, before deploying.
"""
import argparse
import os
import sqlite3


def build(src_path, dst_path):
    if os.path.exists(dst_path):
        os.remove(dst_path)
    src = sqlite3.connect(src_path)
    dst = sqlite3.connect(dst_path)

    dst.executescript(
        """
        CREATE TABLE words (
            rank INTEGER PRIMARY KEY,
            headword TEXT NOT NULL,
            pos TEXT,
            gloss TEXT,
            has_kanji INTEGER NOT NULL,
            reading_llm TEXT,
            reading_dict TEXT,
            mnemonic TEXT,
            image_path TEXT
        );
        CREATE TABLE examples (
            id INTEGER PRIMARY KEY,
            word_rank INTEGER NOT NULL REFERENCES words(rank),
            jp TEXT NOT NULL,
            jp_reading TEXT,
            en TEXT NOT NULL,
            audio_path TEXT,
            grammar_note TEXT
        );
        CREATE TABLE example_breakdown (
            id INTEGER PRIMARY KEY,
            example_id INTEGER NOT NULL REFERENCES examples(id),
            seq INTEGER NOT NULL,
            surface TEXT NOT NULL,
            reading TEXT,
            pos TEXT,
            meaning TEXT,
            note TEXT
        );
        CREATE INDEX idx_examples_word_rank ON examples(word_rank);
        CREATE INDEX idx_breakdown_example_id ON example_breakdown(example_id);
        """
    )

    words = src.execute(
        "SELECT rank, headword, pos, gloss, has_kanji, reading_llm, reading_dict, mnemonic, image_path FROM words"
    ).fetchall()
    dst.executemany("INSERT INTO words VALUES (?,?,?,?,?,?,?,?,?)", words)

    examples = src.execute(
        "SELECT id, word_rank, jp, jp_reading, en, audio_path, grammar_note FROM examples"
    ).fetchall()
    dst.executemany("INSERT INTO examples VALUES (?,?,?,?,?,?,?)", examples)

    breakdown = src.execute(
        "SELECT id, example_id, seq, surface, reading, pos, meaning, note FROM example_breakdown"
    ).fetchall()
    dst.executemany("INSERT INTO example_breakdown VALUES (?,?,?,?,?,?,?,?)", breakdown)

    dst.commit()
    dst.execute("VACUUM")
    dst.close()
    src.close()


def main():
    ap = argparse.ArgumentParser(description="Build the slim browser-facing vocab_public.db from the full pipeline db.")
    ap.add_argument("--src", default="vocab.db")
    ap.add_argument("--dst", default="vocab_public.db")
    args = ap.parse_args()

    build(args.src, args.dst)
    src_size = os.path.getsize(args.src)
    dst_size = os.path.getsize(args.dst)
    print(f"{args.src}: {src_size/1e6:.2f}MB -> {args.dst}: {dst_size/1e6:.2f}MB")


if __name__ == "__main__":
    main()
