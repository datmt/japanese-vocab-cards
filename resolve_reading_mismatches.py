#!/usr/bin/env python3
"""Resolve examples.reading_mismatch via LLM arbitration between candidates.

check_example_readings.py flags sentences where the LLM-generated reading and
SudachiPy's dictionary parse disagree, but it can't say which one is right —
sometimes the dict is right (LLM dropped a particle), sometimes the LLM is
right (dict only knows a formal/literary reading that doesn't fit a casual
sentence). This is a narrower task than original generation: given the
sentence, its translation, and both candidate readings, just pick/correct —
much less room to go off the rails than free-form reading generation.

For every flagged row in `examples`, this:
  - sends the sentence + both candidate readings to an LLM
  - overwrites `jp_reading` (the one the UI displays) with its decision
  - clears `reading_mismatch` so the UI warning badge goes away
  - leaves `reading_llm` / `reading_dict` untouched as an audit trail of what
    the two candidates were

Resumable: progress tracked per-row in `examples.reading_arb_status`, same
pattern as generate_breakdown.py's `breakdown_status` column.
"""
import argparse
import re
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from build_db import KANJI_RE, call_llm, normalize_reading

MAX_ATTEMPTS_DEFAULT = 3
KANA_RE = re.compile(r"[ぁ-ヿ]")


def kana_subsequence_ok(jp, reading):
    """Every kana character already in `jp` (e.g. particles) must survive into
    `reading`, in order — catches the LLM silently dropping a particle rather
    than mistranscribing a kanji."""
    skeleton = normalize_reading("".join(ch for ch in jp if KANA_RE.match(ch)))
    i = 0
    for ch in skeleton:
        idx = reading.find(ch, i)
        if idx == -1:
            return False
        i = idx + 1
    return True


def ensure_columns(conn):
    cols = {row[1] for row in conn.execute("PRAGMA table_info(examples)")}
    if "reading_arb_status" not in cols:
        conn.execute("ALTER TABLE examples ADD COLUMN reading_arb_status TEXT NOT NULL DEFAULT 'pending'")
    if "reading_arb_attempts" not in cols:
        conn.execute("ALTER TABLE examples ADD COLUMN reading_arb_attempts INTEGER NOT NULL DEFAULT 0")
    if "reading_arb_error" not in cols:
        conn.execute("ALTER TABLE examples ADD COLUMN reading_arb_error TEXT")
    conn.commit()


def build_prompt(jp, en, reading_llm, reading_dict):
    return f"""You are a Japanese reading-disambiguation assistant. Two candidate kana readings for the sentence below disagree. Decide which is correct for THIS sentence, or write a corrected reading if both have a problem (e.g. a dropped particle). Respond with ONLY a JSON object (no markdown, no commentary).

Sentence: {jp}
English translation: {en}
Candidate A (free-form reading): {reading_llm}
Candidate B (dictionary/morphological-parser reading): {reading_dict}

Rules:
- Output hiragana ONLY — every kanji must be converted, none left as-is.
- Do not drop any particles (は/の/を/に/へ/で/が/etc.) that are present in the sentence.
- Cover the full sentence, in order, including punctuation.

Return JSON with this exact shape:
{{"reading": "<the single correct reading for the full sentence, entirely in hiragana, covering every character and particle>"}}
"""


def compute_row(backend, host, model, row, timeout=300, think=False):
    """Network/CPU work only, no DB access — safe to run from worker threads."""
    row_id = row["id"]
    prompt = build_prompt(row["jp"], row["en"], row["reading_llm"], row["reading_dict"])
    try:
        data = call_llm(backend, host, model, prompt, timeout=timeout, think=think)
        reading = normalize_reading(data.get("reading", ""))
        if not reading:
            raise ValueError(f"incomplete LLM response: {data!r}")
        if KANJI_RE.search(reading):
            raise ValueError(f"reading still contains kanji: {reading!r}")
        if not kana_subsequence_ok(row["jp"], reading):
            raise ValueError(f"reading dropped a kana character from the sentence: {reading!r}")
        return {"id": row_id, "ok": True, "reading": reading}
    except Exception as e:
        return {"id": row_id, "ok": False, "error": str(e)[:500]}


def persist_result(conn, row, result, max_attempts):
    """DB writes only — call from the main thread to keep sqlite single-writer."""
    row_id = result["id"]
    if result["ok"]:
        conn.execute(
            "UPDATE examples SET jp_reading = ?, reading_mismatch = 0, "
            "reading_arb_status = 'done', reading_arb_error = NULL WHERE id = ?",
            (result["reading"], row_id),
        )
    else:
        attempts = row["reading_arb_attempts"] + 1
        status = "failed" if attempts >= max_attempts else "error"
        conn.execute(
            "UPDATE examples SET reading_arb_status = ?, reading_arb_attempts = ?, reading_arb_error = ? WHERE id = ?",
            (status, attempts, result["error"], row_id),
        )
    conn.commit()


def main():
    ap = argparse.ArgumentParser(description="Resolve examples.reading_mismatch via LLM arbitration.")
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
        "SELECT * FROM examples WHERE reading_mismatch = 1 AND reading_arb_status IN ('pending', 'error') ORDER BY id"
    ).fetchall()
    if args.limit:
        pending = pending[: args.limit]
    target_ids = [r["id"] for r in pending]
    total = len(target_ids)
    print(f"{total} mismatched examples to arbitrate (backend={args.backend}, model={args.model}, workers={args.workers}, think={args.think})")

    t0 = time.time()
    n_done = 0
    round_num = 0
    while True:
        round_num += 1
        if not target_ids:
            break
        placeholders = ",".join("?" * len(target_ids))
        batch = conn.execute(
            f"SELECT * FROM examples WHERE id IN ({placeholders}) AND reading_mismatch = 1 "
            f"AND reading_arb_status IN ('pending', 'error') AND reading_arb_attempts < ? ORDER BY id",
            (*target_ids, args.max_attempts),
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
                    f"[{n_done}/{total}] id={row['id']} jp={row['jp']!r} -> {tag}  (eta {eta/60:.1f}m)",
                    file=sys.stderr,
                )

    final = conn.execute(
        "SELECT reading_arb_status, count(*) c FROM examples WHERE reading_arb_status != 'pending' OR reading_mismatch = 1 GROUP BY reading_arb_status"
    ).fetchall()
    summary = {r["reading_arb_status"]: r["c"] for r in final}
    remaining = conn.execute("SELECT count(*) FROM examples WHERE reading_mismatch = 1").fetchone()[0]
    print(f"summary: {summary} still flagged={remaining}")


if __name__ == "__main__":
    main()
