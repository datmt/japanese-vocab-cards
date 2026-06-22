#!/usr/bin/env python3
"""One-off LLM quality check for a specific list of (example_id) rows.

Used to judge a small set of examples flagged by check_word_readings.py as
'no_match' with zero kanji overlap between headword and jp — i.e. rows where
the example plausibly doesn't actually demonstrate the headword at all.
Asks an LLM (deliberately a different model than whatever generated the
example, to avoid rubber-stamping) whether the sentence correctly uses the
headword given its gloss/pos. Rows judged invalid are deleted (examples row
+ its example_breakdown rows + audio file), not regenerated — a second LLM
attempt already failed for these in the prior bulk pass.

Input: a JSON file (list of objects with id/jp/en/rank/headword/pos/gloss),
e.g. produced ad hoc from a SQL query. Output: prints a verdict per row and
writes results to --out for review before deleting anything.
"""
import argparse
import json
import os
import sqlite3

from build_db import call_llm, extract_json


def build_validation_prompt(row):
    return f"""You are a Japanese teacher reviewing a flashcard example sentence.

Vocabulary word: {row['headword']}
Part of speech: {row['pos']}
English gloss: {row['gloss']}

Example sentence: {row['jp']}
English translation: {row['en']}

Does this example sentence correctly and naturally demonstrate the vocabulary
word above (the exact word, or a standard inflected form of it — not just a
related word, synonym, or a different word sharing a kanji)? Respond with
ONLY a JSON object (no markdown, no commentary):

{{"valid": true or false, "reason": "<one short sentence>"}}
"""


def main():
    ap = argparse.ArgumentParser(description="LLM-validate a list of examples against their headword.")
    ap.add_argument("--in", dest="infile", required=True, help="JSON file: list of example row dicts")
    ap.add_argument("--out", default="/tmp/validate_results.json")
    ap.add_argument("--backend", default="openrouter")
    ap.add_argument("--model", default="google/gemini-2.5-flash")
    args = ap.parse_args()

    with open(args.infile) as f:
        rows = json.load(f)

    results = []
    for row in rows:
        prompt = build_validation_prompt(row)
        try:
            data = call_llm(args.backend, None, args.model, prompt, timeout=120)
            valid = bool(data.get("valid"))
            reason = data.get("reason", "")
        except Exception as e:
            valid, reason = None, f"ERROR: {e}"
        results.append({**row, "valid": valid, "reason": reason})
        print(f"id={row['id']} rank={row['rank']} headword={row['headword']!r} -> "
              f"valid={valid} ({reason}) | jp={row['jp']!r}")

    with open(args.out, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    n_invalid = sum(1 for r in results if r["valid"] is False)
    n_error = sum(1 for r in results if r["valid"] is None)
    print(f"\nsummary: {len(results)} checked, invalid={n_invalid}, errors={n_error}")


if __name__ == "__main__":
    main()
