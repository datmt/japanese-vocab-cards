#!/usr/bin/env python3
"""Resolve words.reading_mismatch via LLM arbitration between candidates.

Same idea as resolve_reading_mismatches.py but for the word-level mismatch
(words.reading_llm vs words.reading_dict) computed during the original
build_db.py run. Without a sentence for context, a lot of these are genuine
heteronym ambiguity the dict can't resolve (私→わたくし vs わたし), or the
dict reading is outright garbage because the corpus headword carries
parenthetical annotation ("ございます (<ござる）", "（お）母さん") that
SudachiPy chokes on. The LLM, given the headword + part of speech + English
gloss, is usually able to tell.

For every flagged row in `words`, this:
  - sends headword + pos + gloss + both candidate readings to an LLM
  - overwrites `reading_llm` (the field the UI displays first) with its
    decision
  - clears `reading_mismatch`
  - leaves `reading_dict` untouched as an audit trail

A result is rejected (left flagged, retried) if it still contains kanji, or
if it drops a kana character that's already literally present in the
headword outside of any parenthetical annotation — annotations like "（お）"
are stripped before that check since they're notes, not part of the reading.

Resumable via `words.reading_arb_status`, same pattern as the examples-table
arbitration script.
"""
import argparse
import re
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from build_db import KANJI_RE, call_llm, normalize_reading
from resolve_reading_mismatches import kana_subsequence_ok

MAX_ATTEMPTS_DEFAULT = 3
BRACKET_RE = re.compile(r"[(（][^)）]*[)）]")


def core_word(headword):
    """Strip parenthetical annotations ("（お）母さん" -> "母さん") so the
    kana-survives check isn't fooled by notes that aren't part of the word."""
    stripped = BRACKET_RE.sub("", headword).strip()
    return stripped or headword


def ensure_columns(conn):
    cols = {row[1] for row in conn.execute("PRAGMA table_info(words)")}
    if "reading_arb_status" not in cols:
        conn.execute("ALTER TABLE words ADD COLUMN reading_arb_status TEXT NOT NULL DEFAULT 'pending'")
    if "reading_arb_attempts" not in cols:
        conn.execute("ALTER TABLE words ADD COLUMN reading_arb_attempts INTEGER NOT NULL DEFAULT 0")
    if "reading_arb_error" not in cols:
        conn.execute("ALTER TABLE words ADD COLUMN reading_arb_error TEXT")
    conn.commit()


def build_prompt(headword, pos, gloss, reading_llm, reading_dict):
    return f"""You are a Japanese reading-disambiguation assistant. Two candidate readings for the dictionary entry below disagree. Decide which is correct given the part of speech and gloss, or write a corrected reading if both have a problem. Respond with ONLY a JSON object (no markdown, no commentary).

Word: {headword}
Part of speech: {pos}
English gloss: {gloss}
Candidate A (free-form reading): {reading_llm}
Candidate B (dictionary/morphological-parser reading): {reading_dict}

Note: Candidate B comes from an automatic parser. It is sometimes nonsense (e.g. if the entry contains punctuation/annotations like parentheses) or picks an overly formal/literary reading that doesn't fit the gloss — don't trust it blindly, but it does sometimes catch a real error in Candidate A.

Rules:
- Output hiragana ONLY — every kanji must be converted, none left as-is.
- Keep any parenthetical part of the word (e.g. "（お）") reflected in the reading if it's meant to be read, drop it if it's just a usage note.

Return JSON with this exact shape:
{{"reading": "<the single correct reading, entirely in hiragana>"}}
"""


def compute_row(backend, host, model, row, timeout=300, think=False):
    """Network/CPU work only, no DB access — safe to run from worker threads."""
    rank = row["rank"]
    prompt = build_prompt(row["headword"], row["pos"], row["gloss"], row["reading_llm"], row["reading_dict"])
    try:
        data = call_llm(backend, host, model, prompt, timeout=timeout, think=think)
        reading = normalize_reading(data.get("reading", ""))
        if not reading:
            raise ValueError(f"incomplete LLM response: {data!r}")
        if KANJI_RE.search(reading):
            raise ValueError(f"reading still contains kanji: {reading!r}")
        if not kana_subsequence_ok(core_word(row["headword"]), reading):
            raise ValueError(f"reading dropped a kana character from the headword: {reading!r}")
        return {"rank": rank, "ok": True, "reading": reading}
    except Exception as e:
        return {"rank": rank, "ok": False, "error": str(e)[:500]}


def persist_result(conn, row, result, max_attempts):
    """DB writes only — call from the main thread to keep sqlite single-writer."""
    rank = result["rank"]
    if result["ok"]:
        conn.execute(
            "UPDATE words SET reading_llm = ?, reading_mismatch = 0, "
            "reading_arb_status = 'done', reading_arb_error = NULL WHERE rank = ?",
            (result["reading"], rank),
        )
    else:
        attempts = row["reading_arb_attempts"] + 1
        status = "failed" if attempts >= max_attempts else "error"
        conn.execute(
            "UPDATE words SET reading_arb_status = ?, reading_arb_attempts = ?, reading_arb_error = ? WHERE rank = ?",
            (status, attempts, result["error"], rank),
        )
    conn.commit()


def main():
    ap = argparse.ArgumentParser(description="Resolve words.reading_mismatch via LLM arbitration.")
    ap.add_argument("--db", default="vocab.db")
    ap.add_argument("--backend", choices=["openrouter", "ollama"], default="openrouter")
    ap.add_argument("--host", default="http://gb10-001:11434", help="ollama host (ignored for openrouter backend)")
    ap.add_argument("--model", default=None, help="defaults to qwen/qwen3.6-35b-a3b (openrouter) or qwen3.6:35b-a3b (ollama)")
    ap.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS_DEFAULT)
    ap.add_argument("--limit", type=int, default=None, help="only process N rows (testing)")
    ap.add_argument("--workers", type=int, default=1, help="parallel LLM requests in flight")
    ap.add_argument("--timeout", type=int, default=300, help="per-request LLM timeout in seconds")
    ap.add_argument("--retry-backoff", type=float, default=3.0, help="seconds to wait before retrying errored rows")
    ap.add_argument("--think", action="store_true", help="enable model reasoning/thinking (slower, ~10-15x; default off)")
    args = ap.parse_args()

    if args.model is None:
        args.model = "qwen/qwen3.6-35b-a3b" if args.backend == "openrouter" else "qwen3.6:35b-a3b"

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    ensure_columns(conn)

    pending = conn.execute(
        "SELECT * FROM words WHERE reading_mismatch = 1 AND reading_arb_status IN ('pending', 'error') ORDER BY rank"
    ).fetchall()
    if args.limit:
        pending = pending[: args.limit]
    target_ranks = [r["rank"] for r in pending]
    total = len(target_ranks)
    print(f"{total} mismatched words to arbitrate (backend={args.backend}, model={args.model}, workers={args.workers}, think={args.think})")

    t0 = time.time()
    n_done = 0
    round_num = 0
    while True:
        round_num += 1
        if not target_ranks:
            break
        placeholders = ",".join("?" * len(target_ranks))
        batch = conn.execute(
            f"SELECT * FROM words WHERE rank IN ({placeholders}) AND reading_mismatch = 1 "
            f"AND reading_arb_status IN ('pending', 'error') AND reading_arb_attempts < ? ORDER BY rank",
            (*target_ranks, args.max_attempts),
        ).fetchall()
        if not batch:
            break
        if round_num > 1:
            print(f"-- retry round {round_num}: {len(batch)} row(s) --", file=sys.stderr)
            time.sleep(args.retry_backoff)

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(compute_row, args.backend, args.host, args.model, row, args.timeout, args.think): row
                for row in batch
            }
            for fut in as_completed(futures):
                row = futures[fut]
                result = fut.result()
                persist_result(conn, row, result, args.max_attempts)
                if result["ok"]:
                    n_done += 1
                    tag = f"resolved -> {result['reading']!r}"
                else:
                    attempts = row["reading_arb_attempts"] + 1
                    will_retry = attempts < args.max_attempts
                    tag = f"{'ERROR, will retry' if will_retry else 'FAILED'} ({result['error']})"
                    if not will_retry:
                        n_done += 1
                elapsed = time.time() - t0
                rate = n_done / elapsed if elapsed > 0 else 0
                eta = (total - n_done) / rate if rate > 0 else float("inf")
                print(
                    f"[{n_done}/{total}] rank={row['rank']} headword={row['headword']!r} -> {tag}  (eta {eta/60:.1f}m)",
                    file=sys.stderr,
                )

    summary = {
        r["reading_arb_status"]: r["c"]
        for r in conn.execute(
            "SELECT reading_arb_status, count(*) c FROM words WHERE reading_arb_status != 'pending' OR reading_mismatch = 1 GROUP BY reading_arb_status"
        ).fetchall()
    }
    remaining = conn.execute("SELECT count(*) FROM words WHERE reading_mismatch = 1").fetchone()[0]
    print(f"summary: {summary} still flagged={remaining}")


if __name__ == "__main__":
    main()
