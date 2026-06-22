#!/usr/bin/env python3
"""Second-opinion review of check_gloss_quality.py's flagged glosses.

check_gloss_quality.py uses a local qwen3.6:27b judge to flag corpus.txt
glosses it thinks are wrong (words.gloss_mismatch = 1). That judge is known
to be noisy on this task — it sometimes rambles, contradicts itself, or
flags pos-vs-gloss style mismatches that aren't really errors (see
docs/gloss check thread). This sends each flagged row, plus the original
judge's complaint, to a stronger model (Gemini 3.1 Pro on OpenRouter) and
asks it to confirm whether the gloss is actually wrong and — if so — propose
a corrected gloss in the same terse corpus.txt style. Rows confirmed real are
then applied directly to words.gloss (original value preserved in
words.gloss_original).

Resumable via words.gloss_review_status, same pattern as check_translations.py.
"""
import argparse
import os
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from build_db import OPENROUTER_URL, extract_json

DEFAULT_MODEL = "google/gemini-3.1-pro-preview"
MAX_ATTEMPTS_DEFAULT = 3


def ensure_columns(conn):
    cols = {row[1] for row in conn.execute("PRAGMA table_info(words)")}
    if "gloss_review_real" not in cols:
        conn.execute("ALTER TABLE words ADD COLUMN gloss_review_real INTEGER")
    if "gloss_corrected" not in cols:
        conn.execute("ALTER TABLE words ADD COLUMN gloss_corrected TEXT")
    if "gloss_review_note" not in cols:
        conn.execute("ALTER TABLE words ADD COLUMN gloss_review_note TEXT")
    if "gloss_review_status" not in cols:
        conn.execute("ALTER TABLE words ADD COLUMN gloss_review_status TEXT NOT NULL DEFAULT 'pending'")
    if "gloss_review_attempts" not in cols:
        conn.execute("ALTER TABLE words ADD COLUMN gloss_review_attempts INTEGER NOT NULL DEFAULT 0")
    if "gloss_review_error" not in cols:
        conn.execute("ALTER TABLE words ADD COLUMN gloss_review_error TEXT")
    if "gloss_original" not in cols:
        conn.execute("ALTER TABLE words ADD COLUMN gloss_original TEXT")
    conn.commit()


def build_prompt(word, pos, gloss, first_judge_issue):
    return f"""You are a senior Japanese-English dictionary editor doing a second-opinion review. A first-pass automated judge flagged this corpus entry's English gloss as potentially wrong. Decide whether the gloss is ACTUALLY wrong, and if so propose a fix.

Word: {word}
Part of speech: {pos}
Current gloss: {gloss}
First judge's complaint: {first_judge_issue}

Notes on this corpus's gloss style (match it if you propose a correction):
- Glosses are terse: single words, or comma/semicolon-separated short senses (e.g. "rapidly, fast, soon").
- Particles/auxiliaries are often glossed with a plain ALL-CAPS grammatical-function label (e.g. "PAST", "TOPIC", "ADVERSATIVE") instead of an English translation — this is intentional convention, not an error, unless the label itself names the wrong function.
- A gloss listing multiple senses is fine even if not every sense is equally common; only treat it as wrong if it's misleading or actually incorrect as a whole, not just less detailed than an ideal dictionary entry would be.
- The first judge is known to sometimes be wrong, ramble, or nitpick non-issues (e.g. complaining a particle gloss isn't an ALL-CAPS label when a short translation is also acceptable). Use your own judgment, don't defer to it.

Respond with ONLY a JSON object (no markdown, no commentary):
{{"real_problem": true or false, "corrected_gloss": "<terse replacement gloss in this corpus's style, or null if real_problem is false>", "note": "<one short sentence explaining your verdict>"}}
"""


def call_judge(model, prompt, timeout=60):
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set in environment")
    resp = requests.post(
        OPENROUTER_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "reasoning": {"effort": "low"},
            "max_tokens": 1200,
            "temperature": 0,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"openrouter error: {data['error']}")
    content = data["choices"][0]["message"]["content"]
    return extract_json(content)


def compute_row(model, row, timeout=60):
    """Network work only, no DB access — safe to run from worker threads."""
    try:
        prompt = build_prompt(row["word"], row["pos"], row["gloss"], row["gloss_issue"])
        result = call_judge(model, prompt, timeout=timeout)
        real = result.get("real_problem")
        if not isinstance(real, bool):
            raise ValueError(f"missing/invalid 'real_problem' field: {result!r}")
        corrected = result.get("corrected_gloss") if real else None
        note = result.get("note")
        return {"rank": row["rank"], "ok": True, "real": real, "corrected": corrected, "note": note}
    except Exception as e:
        return {"rank": row["rank"], "ok": False, "error": str(e)[:500]}


def persist_result(conn, row, result, max_attempts, apply):
    """DB writes only — call from the main thread to keep sqlite single-writer."""
    rank = result["rank"]
    if result["ok"]:
        conn.execute(
            "UPDATE words SET gloss_review_real = ?, gloss_corrected = ?, gloss_review_note = ?, "
            "gloss_review_status = 'done', gloss_review_error = NULL WHERE rank = ?",
            (int(result["real"]), result["corrected"], result["note"], rank),
        )
        if apply and result["real"] and result["corrected"]:
            conn.execute(
                "UPDATE words SET gloss_original = COALESCE(gloss_original, gloss), gloss = ? WHERE rank = ?",
                (result["corrected"], rank),
            )
    else:
        attempts = row["gloss_review_attempts"] + 1
        status = "failed" if attempts >= max_attempts else "error"
        conn.execute(
            "UPDATE words SET gloss_review_status = ?, gloss_review_attempts = ?, "
            "gloss_review_error = ? WHERE rank = ?",
            (status, attempts, result["error"], rank),
        )
    conn.commit()


def main():
    ap = argparse.ArgumentParser(description="Second-opinion review of flagged glosses using a stronger OpenRouter model.")
    ap.add_argument("--db", default="vocab.db")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="OpenRouter judge model")
    ap.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS_DEFAULT)
    ap.add_argument("--limit", type=int, default=None, help="only process N rows (testing)")
    ap.add_argument("--workers", type=int, default=4, help="parallel judge requests in flight")
    ap.add_argument("--timeout", type=int, default=60, help="per-request timeout in seconds")
    ap.add_argument("--retry-backoff", type=float, default=3.0, help="seconds to wait before retrying errored rows")
    ap.add_argument("--no-apply", action="store_true", help="record verdicts but don't write corrected glosses into words.gloss")
    args = ap.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        sys.exit("OPENROUTER_API_KEY not set in environment")

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    ensure_columns(conn)

    pending = conn.execute(
        "SELECT * FROM words WHERE gloss_mismatch = 1 AND gloss_review_status IN ('pending', 'error') ORDER BY rank"
    ).fetchall()
    if args.limit:
        pending = pending[: args.limit]
    target_ranks = [r["rank"] for r in pending]
    total = len(target_ranks)
    apply = not args.no_apply
    print(f"{total} flagged words to review (model={args.model}, workers={args.workers}, apply={apply})")

    t0 = time.time()
    n_done = 0
    n_confirmed = 0
    round_num = 0
    while True:
        round_num += 1
        if not target_ranks:
            break
        placeholders = ",".join("?" * len(target_ranks))
        batch = conn.execute(
            f"SELECT * FROM words WHERE rank IN ({placeholders}) AND gloss_review_status IN ('pending', 'error') "
            f"AND gloss_review_attempts < ? ORDER BY rank",
            (*target_ranks, args.max_attempts),
        ).fetchall()
        if not batch:
            break
        if round_num > 1:
            print(f"-- retry round {round_num}: {len(batch)} row(s) --", file=sys.stderr)
            time.sleep(args.retry_backoff)

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(compute_row, args.model, row, args.timeout): row
                for row in batch
            }
            for fut in as_completed(futures):
                row = futures[fut]
                result = fut.result()
                persist_result(conn, row, result, args.max_attempts, apply)
                if result["ok"]:
                    n_done += 1
                    if result["real"]:
                        n_confirmed += 1
                        tag = f"CONFIRMED -> {result['corrected']!r} ({result['note']})"
                    else:
                        tag = f"false positive ({result['note']})"
                else:
                    attempts = row["gloss_review_attempts"] + 1
                    will_retry = attempts < args.max_attempts
                    tag = f"{'ERROR, will retry' if will_retry else 'FAILED'} ({result['error']})"
                    if not will_retry:
                        n_done += 1
                elapsed = time.time() - t0
                rate = n_done / elapsed if elapsed > 0 else 0
                eta = (total - n_done) / rate if rate > 0 else float("inf")
                print(
                    f"[{n_done}/{total}] rank={row['rank']} word={row['word']!r} -> {tag}  (eta {eta/60:.1f}m)",
                    file=sys.stderr,
                )

    final = conn.execute(
        "SELECT gloss_review_status, count(*) c FROM words WHERE gloss_mismatch = 1 GROUP BY gloss_review_status"
    ).fetchall()
    summary = {r["gloss_review_status"]: r["c"] for r in final}
    print(f"summary: {summary} confirmed real={n_confirmed}")


if __name__ == "__main__":
    main()
