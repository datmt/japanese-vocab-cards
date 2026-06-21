#!/usr/bin/env python3
"""Generate word-by-word vocab/grammar breakdowns for each example sentence via an LLM.

For every row in `examples`, asks the LLM to split the sentence into tokens
(surface form, reading, part of speech, English meaning, grammar note) and to
write a short overall grammar note for the sentence. Deliberately LLM-driven
rather than dictionary/tokenizer-based, since a dictionary lookup misses
sentence context (why a verb is conjugated that way, a particle's role here).

Resumable: progress tracked per-row in `examples.breakdown_status`, same
pattern as generate_audio.py's `audio_status` column.
"""
import argparse
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from build_db import call_llm

MAX_ATTEMPTS_DEFAULT = 3


def ensure_columns(conn):
    cols = {row[1] for row in conn.execute("PRAGMA table_info(examples)")}
    if "grammar_note" not in cols:
        conn.execute("ALTER TABLE examples ADD COLUMN grammar_note TEXT")
    if "breakdown_status" not in cols:
        conn.execute("ALTER TABLE examples ADD COLUMN breakdown_status TEXT NOT NULL DEFAULT 'pending'")
    if "breakdown_attempts" not in cols:
        conn.execute("ALTER TABLE examples ADD COLUMN breakdown_attempts INTEGER NOT NULL DEFAULT 0")
    if "breakdown_error" not in cols:
        conn.execute("ALTER TABLE examples ADD COLUMN breakdown_error TEXT")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS example_breakdown (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            example_id INTEGER NOT NULL REFERENCES examples(id),
            seq INTEGER NOT NULL,
            surface TEXT NOT NULL,
            reading TEXT,
            pos TEXT,
            meaning TEXT,
            note TEXT
        );
        """
    )
    conn.commit()


def build_breakdown_prompt(jp, reading, en):
    return f"""You are a Japanese language teacher. For the example sentence below, break it down word-by-word and explain the grammar. Respond with ONLY a JSON object (no markdown, no commentary).

Sentence: {jp}
Reading: {reading}
Translation: {en}

Return JSON with this exact shape:
{{
  "grammar_note": "<1-3 sentence explanation of notable grammar points in this sentence>",
  "tokens": [
    {{"surface": "<exact substring from the sentence>", "reading": "<reading written in hiragana ONLY, never romaji>", "pos": "<noun/verb/particle/adjective/adverb/etc>", "meaning": "<English meaning>", "note": "<conjugation/grammar role if notable, else empty string>"}}
    // one entry per word/particle, covering the full sentence in order
  ]
}}

"reading" must always be kana (e.g. the particle は is written は, never "wa", even though it's pronounced "wa")."""


def compute_row(backend, host, model, row, timeout=300, think=False):
    """Network/CPU work only, no DB access — safe to run from worker threads."""
    row_id = row["id"]
    prompt = build_breakdown_prompt(row["jp"], row["jp_reading"], row["en"])
    try:
        data = call_llm(backend, host, model, prompt, timeout=timeout, think=think)
        grammar_note = (data.get("grammar_note") or "").strip()
        tokens = data.get("tokens", [])
        if not grammar_note or not tokens:
            raise ValueError(f"incomplete LLM response: {data!r}")
        return {"id": row_id, "ok": True, "grammar_note": grammar_note, "tokens": tokens}
    except Exception as e:
        return {"id": row_id, "ok": False, "error": str(e)[:500]}


def persist_result(conn, row, result, max_attempts):
    """DB writes only — call from the main thread to keep sqlite single-writer."""
    row_id = result["id"]
    if result["ok"]:
        conn.execute("DELETE FROM example_breakdown WHERE example_id = ?", (row_id,))
        conn.executemany(
            "INSERT INTO example_breakdown (example_id, seq, surface, reading, pos, meaning, note) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (row_id, i, t.get("surface", ""), t.get("reading", ""), t.get("pos", ""),
                 t.get("meaning", ""), t.get("note", ""))
                for i, t in enumerate(result["tokens"])
            ],
        )
        conn.execute(
            "UPDATE examples SET grammar_note = ?, breakdown_status = 'done', breakdown_error = NULL WHERE id = ?",
            (result["grammar_note"], row_id),
        )
    else:
        attempts = row["breakdown_attempts"] + 1
        status = "failed" if attempts >= max_attempts else "error"
        conn.execute(
            "UPDATE examples SET breakdown_status = ?, breakdown_attempts = ?, breakdown_error = ? WHERE id = ?",
            (status, attempts, result["error"], row_id),
        )
    conn.commit()


def main():
    ap = argparse.ArgumentParser(description="Generate vocab/grammar breakdowns for example sentences via an LLM.")
    ap.add_argument("--db", default="vocab.db")
    ap.add_argument("--backend", choices=["openrouter", "ollama"], default="ollama")
    ap.add_argument("--host", default="http://gb10-001:11434", help="ollama host (ignored for openrouter backend)")
    ap.add_argument("--model", default=None, help="defaults to qwen/qwen3.6-35b-a3b (openrouter) or qwen3.6:35b-a3b (ollama)")
    ap.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS_DEFAULT)
    ap.add_argument("--limit", type=int, default=None, help="only process N rows (testing)")
    ap.add_argument("--min-id", type=int, default=None, help="only process examples.id >= this (for splitting work across two concurrent runs)")
    ap.add_argument("--max-id", type=int, default=None, help="only process examples.id <= this (for splitting work across two concurrent runs)")
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

    clauses = ["breakdown_status IN ('pending', 'error')"]
    params = []
    if args.min_id is not None:
        clauses.append("id >= ?")
        params.append(args.min_id)
    if args.max_id is not None:
        clauses.append("id <= ?")
        params.append(args.max_id)
    pending = conn.execute(
        f"SELECT * FROM examples WHERE {' AND '.join(clauses)} ORDER BY id", params
    ).fetchall()
    if args.limit:
        pending = pending[: args.limit]
    target_ids = [r["id"] for r in pending]
    total = len(target_ids)
    print(f"{total} examples to process (backend={args.backend}, model={args.model}, workers={args.workers}, think={args.think})")

    t0 = time.time()
    n_done = 0
    round_num = 0
    while True:
        round_num += 1
        if not target_ids:
            break
        placeholders = ",".join("?" * len(target_ids))
        batch = conn.execute(
            f"SELECT * FROM examples WHERE id IN ({placeholders}) AND breakdown_status IN ('pending', 'error') "
            f"AND breakdown_attempts < ? ORDER BY id",
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
                    tag = "ok"
                else:
                    attempts = row["breakdown_attempts"] + 1
                    will_retry = attempts < args.max_attempts
                    tag = f"{'ERROR, will retry' if will_retry else 'FAILED'} ({result['error']})"
                    if not will_retry:
                        n_done += 1
                elapsed = time.time() - t0
                rate = n_done / elapsed if elapsed > 0 else 0
                eta = (total - n_done) / rate if rate > 0 else float("inf")
                print(
                    f"[{n_done}/{total}] id={row['id']} word_rank={row['word_rank']} -> {tag}  (eta {eta/60:.1f}m)",
                    file=sys.stderr,
                )

    final = conn.execute(
        "SELECT breakdown_status, count(*) c FROM examples GROUP BY breakdown_status"
    ).fetchall()
    summary = {r["breakdown_status"]: r["c"] for r in final}
    print(f"summary: {summary}")


if __name__ == "__main__":
    main()
