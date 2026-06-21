#!/usr/bin/env python3
"""Generate TTS audio (Japanese only) for each example sentence via Qwen3-TTS.

Calls the Qwen3-TTS server's /speak endpoint for every row in `examples`,
saving each clip to audio/{word_rank:04d}_{id}.mp3 and recording the path
in examples.audio_path.

Resumable: progress tracked per-row in `examples.audio_status`, same
pattern as build_db.py's `status` column.
"""
import argparse
import os
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

MAX_ATTEMPTS_DEFAULT = 3


def ensure_columns(conn):
    cols = {row[1] for row in conn.execute("PRAGMA table_info(examples)")}
    if "audio_path" not in cols:
        conn.execute("ALTER TABLE examples ADD COLUMN audio_path TEXT")
    if "audio_status" not in cols:
        conn.execute("ALTER TABLE examples ADD COLUMN audio_status TEXT NOT NULL DEFAULT 'pending'")
    if "audio_attempts" not in cols:
        conn.execute("ALTER TABLE examples ADD COLUMN audio_attempts INTEGER NOT NULL DEFAULT 0")
    if "audio_error" not in cols:
        conn.execute("ALTER TABLE examples ADD COLUMN audio_error TEXT")
    conn.commit()


def call_qwen_tts(host, voice, text, speed=1.0, timeout=120):
    resp = requests.post(
        f"{host}/speak",
        json={
            "text": text,
            "language": "Japanese",
            "voice_id": voice,
            "speed": speed,
            "format": "mp3",
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.content


def compute_row(host, voice, speed, audio_dir, row, timeout=120):
    """Network work + unique-file write only, no sqlite access — safe from worker threads."""
    row_id = row["id"]
    try:
        text = row["jp"].strip()
        if not text:
            raise ValueError("empty jp text")
        audio = call_qwen_tts(host, voice, text, speed=speed, timeout=timeout)
        if not audio:
            raise ValueError("empty audio response")
        path = os.path.join(audio_dir, f"{row['word_rank']:04d}_{row_id}.mp3")
        with open(path, "wb") as f:
            f.write(audio)
        return {"id": row_id, "ok": True, "audio_path": path}
    except Exception as e:
        return {"id": row_id, "ok": False, "error": str(e)[:500]}


def persist_result(conn, row, result, max_attempts):
    """sqlite writes only — call from the main thread to keep sqlite single-writer."""
    row_id = result["id"]
    if result["ok"]:
        conn.execute(
            "UPDATE examples SET audio_path = ?, audio_status = 'done', audio_error = NULL WHERE id = ?",
            (result["audio_path"], row_id),
        )
    else:
        attempts = row["audio_attempts"] + 1
        status = "failed" if attempts >= max_attempts else "error"
        conn.execute(
            "UPDATE examples SET audio_status = ?, audio_attempts = ?, audio_error = ? WHERE id = ?",
            (status, attempts, result["error"], row_id),
        )
    conn.commit()


def main():
    ap = argparse.ArgumentParser(description="Generate TTS audio for example sentences via Qwen3-TTS.")
    ap.add_argument("--db", default="vocab.db")
    ap.add_argument("--host", default="http://gb10-001:8886", help="Qwen3-TTS server host")
    ap.add_argument("--voice", default="ja_female_sora")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--audio-dir", default="audio")
    ap.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS_DEFAULT)
    ap.add_argument("--limit", type=int, default=None, help="only process N rows (testing)")
    ap.add_argument("--workers", type=int, default=1, help="parallel TTS requests in flight")
    ap.add_argument("--timeout", type=int, default=120, help="per-request timeout in seconds")
    ap.add_argument("--retry-backoff", type=float, default=3.0, help="seconds to wait before retrying errored rows")
    args = ap.parse_args()

    os.makedirs(args.audio_dir, exist_ok=True)

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    ensure_columns(conn)

    pending = conn.execute(
        "SELECT * FROM examples WHERE audio_status IN ('pending', 'error') ORDER BY id"
    ).fetchall()
    if args.limit:
        pending = pending[: args.limit]
    target_ids = [r["id"] for r in pending]
    total = len(target_ids)
    print(f"{total} examples to process (host={args.host}, voice={args.voice}, workers={args.workers})")

    t0 = time.time()
    n_done = 0
    round_num = 0
    while True:
        round_num += 1
        if not target_ids:
            break
        placeholders = ",".join("?" * len(target_ids))
        batch = conn.execute(
            f"SELECT * FROM examples WHERE id IN ({placeholders}) AND audio_status IN ('pending', 'error') "
            f"AND audio_attempts < ? ORDER BY id",
            (*target_ids, args.max_attempts),
        ).fetchall()
        if not batch:
            break
        if round_num > 1:
            print(f"-- retry round {round_num}: {len(batch)} row(s) --", file=sys.stderr)
            time.sleep(args.retry_backoff)

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(compute_row, args.host, args.voice, args.speed, args.audio_dir, row, args.timeout): row
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
                    attempts = row["audio_attempts"] + 1
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
        "SELECT audio_status, count(*) c FROM examples GROUP BY audio_status"
    ).fetchall()
    summary = {r["audio_status"]: r["c"] for r in final}
    print(f"summary: {summary}")


if __name__ == "__main__":
    main()
