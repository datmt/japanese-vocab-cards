#!/usr/bin/env python3
"""Generate a flat illustration image for each word's image_prompt via the
gb10 flux txt2img server, and save a compressed WEBP to images/.

Resumable: progress tracked per-row in `words.image_status`, same
pattern as generate_audio.py / build_db.py's `status` column.
"""
import argparse
import io
import os
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image

MAX_ATTEMPTS_DEFAULT = 3


def ensure_columns(conn):
    cols = {row[1] for row in conn.execute("PRAGMA table_info(words)")}
    if "image_path" not in cols:
        conn.execute("ALTER TABLE words ADD COLUMN image_path TEXT")
    if "image_status" not in cols:
        conn.execute("ALTER TABLE words ADD COLUMN image_status TEXT NOT NULL DEFAULT 'pending'")
    if "image_attempts" not in cols:
        conn.execute("ALTER TABLE words ADD COLUMN image_attempts INTEGER NOT NULL DEFAULT 0")
    if "image_error" not in cols:
        conn.execute("ALTER TABLE words ADD COLUMN image_error TEXT")
    conn.commit()


def call_flux(host, prompt, aspect_ratio="1:1", steps=6, timeout=120):
    resp = requests.post(
        f"{host}/generate/raw",
        json={"prompt": prompt, "aspect_ratio": aspect_ratio, "steps": steps},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.content


def compute_row(host, steps, image_dir, row, timeout=120):
    """Network work + unique-file write only, no sqlite access — safe from worker threads."""
    rank = row["rank"]
    try:
        prompt = row["image_prompt"].strip()
        if not prompt:
            raise ValueError("empty image prompt")
        png_bytes = call_flux(host, prompt, steps=steps, timeout=timeout)
        im = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        path = os.path.join(image_dir, f"{rank:04d}.webp")
        im.save(path, "WEBP", quality=80)
        return {"rank": rank, "ok": True, "image_path": path}
    except Exception as e:
        return {"rank": rank, "ok": False, "error": str(e)[:500]}


def persist_result(conn, row, result, max_attempts):
    """sqlite writes only — call from the main thread to keep sqlite single-writer."""
    rank = result["rank"]
    if result["ok"]:
        conn.execute(
            "UPDATE words SET image_path = ?, image_status = 'done', image_error = NULL WHERE rank = ?",
            (result["image_path"], rank),
        )
    else:
        attempts = row["image_attempts"] + 1
        status = "failed" if attempts >= max_attempts else "error"
        conn.execute(
            "UPDATE words SET image_status = ?, image_attempts = ?, image_error = ? WHERE rank = ?",
            (status, attempts, result["error"], rank),
        )
    conn.commit()


def main():
    ap = argparse.ArgumentParser(description="Generate flux txt2img images from word image prompts.")
    ap.add_argument("--db", default="vocab.db")
    ap.add_argument("--host", default="http://gb10-001:8003", help="flux server host")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--image-dir", default="images")
    ap.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS_DEFAULT)
    ap.add_argument("--limit", type=int, default=None, help="only process N rows (testing)")
    ap.add_argument("--workers", type=int, default=1, help="parallel flux requests in flight")
    ap.add_argument("--timeout", type=int, default=120, help="per-request timeout in seconds")
    ap.add_argument("--retry-backoff", type=float, default=3.0, help="seconds to wait before retrying errored rows")
    args = ap.parse_args()

    os.makedirs(args.image_dir, exist_ok=True)

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    ensure_columns(conn)

    pending = conn.execute(
        "SELECT * FROM words WHERE image_prompt IS NOT NULL AND image_status IN ('pending', 'error') ORDER BY rank"
    ).fetchall()
    if args.limit:
        pending = pending[: args.limit]
    target_ranks = [r["rank"] for r in pending]
    total = len(target_ranks)
    print(f"{total} words to process (host={args.host}, steps={args.steps}, workers={args.workers})")

    t0 = time.time()
    n_done = 0
    round_num = 0
    while True:
        round_num += 1
        if not target_ranks:
            break
        placeholders = ",".join("?" * len(target_ranks))
        batch = conn.execute(
            f"SELECT * FROM words WHERE rank IN ({placeholders}) AND image_status IN ('pending', 'error') "
            f"AND image_attempts < ? ORDER BY rank",
            (*target_ranks, args.max_attempts),
        ).fetchall()
        if not batch:
            break
        if round_num > 1:
            print(f"-- retry round {round_num}: {len(batch)} row(s) --", file=sys.stderr)
            time.sleep(args.retry_backoff)

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(compute_row, args.host, args.steps, args.image_dir, row, args.timeout): row
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
                    attempts = row["image_attempts"] + 1
                    will_retry = attempts < args.max_attempts
                    tag = f"{'ERROR, will retry' if will_retry else 'FAILED'} ({result['error']})"
                    if not will_retry:
                        n_done += 1
                elapsed = time.time() - t0
                rate = n_done / elapsed if elapsed > 0 else 0
                eta = (total - n_done) / rate if rate > 0 else float("inf")
                print(
                    f"[{n_done}/{total}] rank={row['rank']} {row['word']!r} -> {tag}  (eta {eta/60:.1f}m)",
                    file=sys.stderr,
                )

    final = conn.execute(
        "SELECT image_status, count(*) c FROM words WHERE image_prompt IS NOT NULL GROUP BY image_status"
    ).fetchall()
    summary = {r["image_status"]: r["c"] for r in final}
    print(f"summary: {summary}")


if __name__ == "__main__":
    main()
