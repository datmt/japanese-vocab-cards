#!/usr/bin/env python3
"""Flag corpus entries whose English gloss is inaccurate or misleading.

corpus.txt's word/pos/gloss columns come from the source frequency list, not
from this project's own LLM pipeline. This sends each (word, pos, gloss)
triple to a local Ollama judge and flags glosses the judge thinks are wrong
or confusingly incomplete for that word.

Uses qwen3.6:27b (dense) rather than build_db.py's default qwen3.6:35b-a3b
(MoE) since both are "ollama qwen" but a different family/architecture avoids
one model's blind spots being invisible to the check.

Resumable via words.gloss_check_status, same pattern as check_translations.py.
"""
import argparse
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from build_db import call_ollama

DEFAULT_HOST = "http://gb10-001:11434"
DEFAULT_MODEL = "qwen3.6:27b"
MAX_ATTEMPTS_DEFAULT = 3


def ensure_columns(conn):
    cols = {row[1] for row in conn.execute("PRAGMA table_info(words)")}
    if "gloss_mismatch" not in cols:
        conn.execute("ALTER TABLE words ADD COLUMN gloss_mismatch INTEGER")
    if "gloss_issue" not in cols:
        conn.execute("ALTER TABLE words ADD COLUMN gloss_issue TEXT")
    if "gloss_check_status" not in cols:
        conn.execute("ALTER TABLE words ADD COLUMN gloss_check_status TEXT NOT NULL DEFAULT 'pending'")
    if "gloss_check_attempts" not in cols:
        conn.execute("ALTER TABLE words ADD COLUMN gloss_check_attempts INTEGER NOT NULL DEFAULT 0")
    if "gloss_check_error" not in cols:
        conn.execute("ALTER TABLE words ADD COLUMN gloss_check_error TEXT")
    conn.commit()


def build_prompt(word, pos, gloss):
    return f"""You are a Japanese-English dictionary QC judge. Check whether the English gloss below is an accurate, usable meaning for the Japanese word, given its part of speech.

For particles/auxiliaries (pos starting with "p." or "aux." or "cp."), the gloss is often a terse ALL-CAPS grammatical-function label (e.g. "PAST", "TOPIC", "ADVERSATIVE", "CONTINUATION") rather than a plain English translation — this is standard shorthand in Japanese textbooks and is correct by convention, not an error. Only flag a particle/auxiliary gloss if the label names a clearly wrong grammatical function for that word's known senses (not just "incomplete" — these words are multi-functional and one labeled sense is fine), or if you are confident the label is simply wrong, not merely less detailed than you'd write yourself. When unsure about a particle/auxiliary's grammar terminology, prefer accurate: true.

For content words (nouns, verbs, adjectives, adverbs, interjections), flag the gloss if it states the wrong meaning, a different word's meaning, or is too strong/weak in connotation to be useful (e.g. "Wow!" given for a word that actually means "well, then"). Glosses often list multiple comma-separated senses (e.g. "rapidly, fast, soon") — judge the gloss as a whole, not sense-by-sense: only flag it if NONE of the listed senses are valid meanings of the word. If at least one listed sense is correct, treat the gloss as accurate even if other listed senses are weaker, redundant, or debatable.

Word: {word}
Part of speech: {pos}
English gloss: {gloss}

Respond with ONLY a JSON object (no markdown, no commentary):
{{"accurate": true or false, "issue": "<one short sentence describing the problem, or null if accurate>"}}
"""


def compute_row(host, model, row, timeout=120):
    """Network work only, no DB access — safe to run from worker threads."""
    try:
        prompt = build_prompt(row["word"], row["pos"], row["gloss"])
        result = call_ollama(host, model, prompt, timeout=timeout)
        accurate = result.get("accurate")
        if not isinstance(accurate, bool):
            raise ValueError(f"missing/invalid 'accurate' field: {result!r}")
        issue = result.get("issue") if not accurate else None
        return {"rank": row["rank"], "ok": True, "mismatch": not accurate, "issue": issue}
    except Exception as e:
        return {"rank": row["rank"], "ok": False, "error": str(e)[:500]}


def persist_result(conn, row, result, max_attempts):
    """DB writes only — call from the main thread to keep sqlite single-writer."""
    rank = result["rank"]
    if result["ok"]:
        conn.execute(
            "UPDATE words SET gloss_mismatch = ?, gloss_issue = ?, "
            "gloss_check_status = 'done', gloss_check_error = NULL WHERE rank = ?",
            (int(result["mismatch"]), result["issue"], rank),
        )
    else:
        attempts = row["gloss_check_attempts"] + 1
        status = "failed" if attempts >= max_attempts else "error"
        conn.execute(
            "UPDATE words SET gloss_check_status = ?, gloss_check_attempts = ?, "
            "gloss_check_error = ? WHERE rank = ?",
            (status, attempts, result["error"], rank),
        )
    conn.commit()


def main():
    ap = argparse.ArgumentParser(description="Flag corpus glosses where the judge model disagrees with the gloss.")
    ap.add_argument("--db", default="vocab.db")
    ap.add_argument("--host", default=DEFAULT_HOST, help="ollama host")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="ollama judge model")
    ap.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS_DEFAULT)
    ap.add_argument("--limit", type=int, default=None, help="only process N rows (testing)")
    ap.add_argument("--workers", type=int, default=1, help="parallel judge requests in flight")
    ap.add_argument("--timeout", type=int, default=120, help="per-request timeout in seconds")
    ap.add_argument("--retry-backoff", type=float, default=3.0, help="seconds to wait before retrying errored rows")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    ensure_columns(conn)

    pending = conn.execute(
        "SELECT * FROM words WHERE gloss_check_status IN ('pending', 'error') ORDER BY rank"
    ).fetchall()
    if args.limit:
        pending = pending[: args.limit]
    target_ranks = [r["rank"] for r in pending]
    total = len(target_ranks)
    print(f"{total} words to judge (model={args.model}, workers={args.workers})")

    t0 = time.time()
    n_done = 0
    round_num = 0
    while True:
        round_num += 1
        if not target_ranks:
            break
        placeholders = ",".join("?" * len(target_ranks))
        batch = conn.execute(
            f"SELECT * FROM words WHERE rank IN ({placeholders}) AND gloss_check_status IN ('pending', 'error') "
            f"AND gloss_check_attempts < ? ORDER BY rank",
            (*target_ranks, args.max_attempts),
        ).fetchall()
        if not batch:
            break
        if round_num > 1:
            print(f"-- retry round {round_num}: {len(batch)} row(s) --", file=sys.stderr)
            time.sleep(args.retry_backoff)

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(compute_row, args.host, args.model, row, args.timeout): row
                for row in batch
            }
            for fut in as_completed(futures):
                row = futures[fut]
                result = fut.result()
                persist_result(conn, row, result, args.max_attempts)
                if result["ok"]:
                    n_done += 1
                    tag = "MISMATCH" if result["mismatch"] else "ok"
                    if result["mismatch"]:
                        tag += f" ({result['issue']})"
                else:
                    attempts = row["gloss_check_attempts"] + 1
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
        "SELECT gloss_check_status, count(*) c FROM words GROUP BY gloss_check_status"
    ).fetchall()
    summary = {r["gloss_check_status"]: r["c"] for r in final}
    flagged = conn.execute("SELECT count(*) FROM words WHERE gloss_mismatch = 1").fetchone()[0]
    print(f"summary: {summary} flagged mismatches={flagged}")


if __name__ == "__main__":
    main()
