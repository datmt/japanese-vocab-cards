#!/usr/bin/env python3
"""Flag examples where the headword itself was likely mispronounced, without
diluting the signal across a whole sentence.

check_audio_readings.py compares STT output against jp_reading as one long
fuzzy-match ratio. On long sentences a real mispronunciation of just the
headword (e.g. 伯父/おじ rendered as はくじ) barely moves the whole-string
ratio and slips under threshold — see "Known unfixed flaw: whole-sentence
fuzzy ratio dilutes short word-level errors" in docs/audio_reading_qc.md.

This script localizes the check instead: for each example, find the
example_breakdown row that corresponds to the example's headword (matched by
surface or by SudachiPy dictionary_form(), since the headword may appear
inflected — e.g. 引っ越して -> 引っ越す), take that row's per-sentence,
context-resolved reading as ground truth, and check whether it survives as a
substring of the stt_reading already collected by check_audio_readings.py.
No new STT calls.

Resumable via examples.word_check_status. Pure local computation (SudachiPy
+ string ops), no network calls, so it's run single-threaded/synchronously.
"""
import argparse
import re
import sqlite3

from build_db import get_tokenizer, normalize_reading
from check_audio_readings import PUNCT_RE, expand_chōonpu

# Corpus headwords for suru-verbs are stored with the conjugation hint still
# attached (e.g. "仕事（する）"), which never appears verbatim in a sentence
# or breakdown surface. Strip it before matching.
PAREN_RE = re.compile(r"[（(][^）)]*[）)]")


def strip_headword(headword):
    return PAREN_RE.sub("", headword).strip()


def ensure_columns(conn):
    cols = {row[1] for row in conn.execute("PRAGMA table_info(examples)")}
    if "word_target_reading" not in cols:
        conn.execute("ALTER TABLE examples ADD COLUMN word_target_reading TEXT")
    if "word_stt_match" not in cols:
        conn.execute("ALTER TABLE examples ADD COLUMN word_stt_match INTEGER")
    if "word_check_status" not in cols:
        conn.execute("ALTER TABLE examples ADD COLUMN word_check_status TEXT NOT NULL DEFAULT 'pending'")
    if "word_check_note" not in cols:
        conn.execute("ALTER TABLE examples ADD COLUMN word_check_note TEXT")
    conn.commit()


def find_headword_breakdown(headword, breakdown_rows):
    """Return the first example_breakdown row whose surface resolves to
    headword (directly, or via SudachiPy dictionary_form), or None if the
    headword doesn't appear in the breakdown.

    Inflected fragments (て-form stems, ます-stems, etc.) tokenize wrong in
    isolation — e.g. tokenizing "あり" alone gives the noun "あり" (ant), not
    the verb stem of "ある", because Sudachi has no following auxiliary to
    disambiguate. Re-tokenizing with one neighbor surface on each side as
    local context (still using begin()/end() offsets to pick out only the
    morpheme(s) overlapping the row's own span) fixes this without needing
    to align the whole sentence.
    """
    headword = strip_headword(headword)
    for row in breakdown_rows:
        if row["surface"] == headword:
            return row
        for m in get_tokenizer().tokenize(row["surface"]):
            if m.dictionary_form() == headword:
                return row
    for i, row in enumerate(breakdown_rows):
        surface = row["surface"]
        prev = breakdown_rows[i - 1]["surface"] if i > 0 else ""
        nxt = breakdown_rows[i + 1]["surface"] if i + 1 < len(breakdown_rows) else ""
        window = prev + surface + nxt
        start, end = len(prev), len(prev) + len(surface)
        for m in get_tokenizer().tokenize(window):
            if m.begin() < end and m.end() > start and m.dictionary_form() == headword:
                return row
    return None


def norm_for_compare(s):
    return expand_chōonpu(PUNCT_RE.sub("", normalize_reading(s or "")))


def main():
    ap = argparse.ArgumentParser(description="Flag headword-level TTS mispronunciations using existing STT data.")
    ap.add_argument("--db", default="vocab.db")
    ap.add_argument("--limit", type=int, default=None, help="only process N rows (testing)")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    ensure_columns(conn)

    pending = conn.execute(
        "SELECT e.*, w.headword FROM examples e JOIN words w ON w.rank = e.word_rank "
        "WHERE e.audio_status = 'done' AND e.stt_check_status = 'done' "
        "AND e.breakdown_status = 'done' AND e.word_check_status = 'pending' "
        "ORDER BY e.id"
    ).fetchall()
    if args.limit:
        pending = pending[: args.limit]
    total = len(pending)
    print(f"{total} example(s) to word-check")

    n_done = n_no_match = n_mismatch = n_conflict = 0
    for i, row in enumerate(pending, 1):
        breakdown_rows = conn.execute(
            "SELECT * FROM example_breakdown WHERE example_id = ? ORDER BY seq", (row["id"],)
        ).fetchall()
        match = find_headword_breakdown(row["headword"], breakdown_rows)
        if match is None or not match["reading"]:
            conn.execute(
                "UPDATE examples SET word_check_status = 'no_match', word_check_note = ? WHERE id = ?",
                (f"headword {row['headword']!r} not found in breakdown", row["id"]),
            )
            n_no_match += 1
        else:
            target = match["reading"]
            # example_breakdown.reading is itself LLM-generated and not always
            # right (e.g. headword 実 broken down with reading 'み' while the
            # curated jp_reading for that exact sentence says 'じつ') —
            # trusting it blindly produces false TTS-mismatch flags that are
            # really breakdown-data errors. Only treat it as ground truth if
            # it's consistent with jp_reading, the value actually used to
            # judge the audio everywhere else.
            if norm_for_compare(target) not in norm_for_compare(row["jp_reading"]):
                conn.execute(
                    "UPDATE examples SET word_target_reading = ?, word_check_status = 'breakdown_conflict', "
                    "word_check_note = ? WHERE id = ?",
                    (target, f"breakdown reading {target!r} not found in jp_reading {row['jp_reading']!r}", row["id"]),
                )
                n_conflict += 1
            else:
                stt_match = norm_for_compare(target) in norm_for_compare(row["stt_reading"])
                conn.execute(
                    "UPDATE examples SET word_target_reading = ?, word_stt_match = ?, "
                    "word_check_status = 'done', word_check_note = NULL WHERE id = ?",
                    (target, int(stt_match), row["id"]),
                )
                n_done += 1
                if not stt_match:
                    n_mismatch += 1
                    print(f"id={row['id']} word_rank={row['word_rank']} headword={row['headword']} "
                          f"target={target!r} stt_reading={row['stt_reading']!r} -> MISMATCH")
        if i % 500 == 0:
            conn.commit()
            print(f"[{i}/{total}]")

    conn.commit()
    print(f"summary: done={n_done} no_match={n_no_match} breakdown_conflict={n_conflict} word_mismatch={n_mismatch}")


if __name__ == "__main__":
    main()
