#!/usr/bin/env python3
"""Resolve examples.translation_mismatch by regenerating `en` via LLM.

check_translations.py flags (jp, en) pairs where a judge model found the
translation inaccurate, and records *why* in translation_issue. This sends
the sentence, the bad translation, and that specific issue to an LLM and asks
for a corrected translation — a narrower task than original generation: given
a concrete defect description, fix just that, rather than translating from
scratch.

For every flagged row, this overwrites `en` with the LLM's correction and
clears `translation_mismatch`. `translation_issue` is left in place as an
audit trail of what was wrong.

Resumable: progress tracked per-row in `examples.translation_resolve_status`,
same pattern as resolve_reading_mismatches.py's `reading_arb_status`.
"""
import argparse
import re
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from build_db import call_llm

MAX_ATTEMPTS_DEFAULT = 3
JAPANESE_RE = re.compile(r"[぀-ヿ一-鿿]")


def ensure_columns(conn):
    cols = {row[1] for row in conn.execute("PRAGMA table_info(examples)")}
    if "translation_resolve_status" not in cols:
        conn.execute("ALTER TABLE examples ADD COLUMN translation_resolve_status TEXT NOT NULL DEFAULT 'pending'")
    if "translation_resolve_attempts" not in cols:
        conn.execute("ALTER TABLE examples ADD COLUMN translation_resolve_attempts INTEGER NOT NULL DEFAULT 0")
    if "translation_resolve_error" not in cols:
        conn.execute("ALTER TABLE examples ADD COLUMN translation_resolve_error TEXT")
    conn.commit()


def build_prompt(jp, en, issue):
    return f"""You are a Japanese-to-English translation correction assistant. A QC judge flagged the English translation below as inaccurate, with a specific reason. Produce a corrected English translation that fixes exactly that problem while staying natural and faithful to the Japanese.

Japanese: {jp}
Current (flagged) English: {en}
Problem found: {issue}

Rules:
- Write natural, fluent English — never use bracket placeholders like "[someone]" or "[X]".
- If the Japanese leaves the subject unspecified or ambiguous, pick the most natural English rendering (e.g. "I", "you", or "they" as appropriate for the sentence) rather than a placeholder.

Respond with ONLY a JSON object (no markdown, no commentary):
{{"en": "<the corrected English translation>"}}
"""


def compute_row(backend, host, model, row, timeout=120, think=False):
    """Network/CPU work only, no DB access — safe to run from worker threads."""
    row_id = row["id"]
    try:
        prompt = build_prompt(row["jp"], row["en"], row["translation_issue"])
        data = call_llm(backend, host, model, prompt, timeout=timeout, think=think)
        en = (data.get("en") or "").strip()
        if not en:
            raise ValueError(f"incomplete LLM response: {data!r}")
        if JAPANESE_RE.search(en):
            raise ValueError(f"corrected 'en' still contains Japanese script: {en!r}")
        return {"id": row_id, "ok": True, "en": en}
    except Exception as e:
        return {"id": row_id, "ok": False, "error": str(e)[:500]}


def persist_result(conn, row, result, max_attempts):
    """DB writes only — call from the main thread to keep sqlite single-writer."""
    row_id = result["id"]
    if result["ok"]:
        conn.execute(
            "UPDATE examples SET en = ?, translation_mismatch = 0, "
            "translation_resolve_status = 'done', translation_resolve_error = NULL WHERE id = ?",
            (result["en"], row_id),
        )
    else:
        attempts = row["translation_resolve_attempts"] + 1
        status = "failed" if attempts >= max_attempts else "error"
        conn.execute(
            "UPDATE examples SET translation_resolve_status = ?, translation_resolve_attempts = ?, "
            "translation_resolve_error = ? WHERE id = ?",
            (status, attempts, result["error"], row_id),
        )
    conn.commit()


def main():
    ap = argparse.ArgumentParser(description="Resolve examples.translation_mismatch by regenerating `en`.")
    ap.add_argument("--db", default="vocab.db")
    ap.add_argument("--backend", choices=["openrouter", "ollama"], default="openrouter")
    ap.add_argument("--host", default="http://gb10-001:11434", help="ollama host (ignored for openrouter backend)")
    ap.add_argument("--model", default=None, help="defaults to qwen/qwen3.6-35b-a3b (openrouter) or qwen3.6:35b-a3b (ollama)")
    ap.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS_DEFAULT)
    ap.add_argument("--limit", type=int, default=None, help="only process N rows (testing)")
    ap.add_argument("--workers", type=int, default=1, help="parallel LLM requests in flight")
    ap.add_argument("--timeout", type=int, default=120, help="per-request LLM timeout in seconds")
    ap.add_argument("--retry-backoff", type=float, default=3.0, help="seconds to wait before retrying errored rows")
    ap.add_argument("--think", action="store_true", help="enable model reasoning/thinking (slower; default off)")
    args = ap.parse_args()

    if args.model is None:
        args.model = "qwen/qwen3.6-35b-a3b" if args.backend == "openrouter" else "qwen3.6:35b-a3b"

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    ensure_columns(conn)

    pending = conn.execute(
        "SELECT * FROM examples WHERE translation_mismatch = 1 AND translation_resolve_status IN ('pending', 'error') ORDER BY id"
    ).fetchall()
    if args.limit:
        pending = pending[: args.limit]
    target_ids = [r["id"] for r in pending]
    total = len(target_ids)
    print(f"{total} flagged translations to resolve (backend={args.backend}, model={args.model}, workers={args.workers})")

    t0 = time.time()
    n_done = 0
    round_num = 0
    while True:
        round_num += 1
        if not target_ids:
            break
        placeholders = ",".join("?" * len(target_ids))
        batch = conn.execute(
            f"SELECT * FROM examples WHERE id IN ({placeholders}) AND translation_mismatch = 1 "
            f"AND translation_resolve_status IN ('pending', 'error') AND translation_resolve_attempts < ? ORDER BY id",
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
                    tag = f"fixed -> {result['en']!r}"
                else:
                    attempts = row["translation_resolve_attempts"] + 1
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
        "SELECT translation_resolve_status, count(*) c FROM examples WHERE translation_resolve_status != 'pending' GROUP BY translation_resolve_status"
    ).fetchall()
    summary = {r["translation_resolve_status"]: r["c"] for r in final}
    remaining = conn.execute("SELECT count(*) FROM examples WHERE translation_mismatch = 1").fetchone()[0]
    print(f"summary: {summary} still flagged={remaining}")


if __name__ == "__main__":
    main()
