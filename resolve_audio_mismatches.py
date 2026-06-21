#!/usr/bin/env python3
"""Resolve examples.stt_mismatch by resynthesizing audio for flagged rows.

check_audio_readings.py flags audio where Whisper STT disagrees with
jp_reading. Two root causes account for most flags:

  - TTS decoding runaway: the engine loops on a stock filler phrase
    ("kanji wa tsukaimasen", etc.) instead of the sentence. A plain retry
    with the same `jp` text usually fixes it (transient failure).
  - Arabic-digit misreading: `jp` legitimately contains arabic numerals
    (e.g. "2,000円" — normal Japanese writing), but the TTS engine reads
    them digit-by-digit instead of as a number, while `jp_reading` already
    has the correct spoken-form kana for the whole sentence. Synthesizing
    from `jp_reading` instead of `jp` sidesteps the engine's lack of
    number normalization.

This only touches rows whose id is listed in --ids-file (one id per line)
so it can be scoped to a specific batch of flagged mismatches rather than
re-synthesizing anything else that happens to be flagged.

After running, re-run check_audio_readings.py to re-verify the new clips
(rows reset to stt_check_status='pending' here are exactly the ones it
will re-check).
"""
import argparse
import os
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from generate_audio import call_qwen_tts


def load_ids(path):
    with open(path) as f:
        return [int(line.strip()) for line in f if line.strip()]


def compute_row(host, voice, speed, audio_dir, row, timeout=120):
    """Network work + unique-file write only, no sqlite access — safe from worker threads."""
    row_id = row["id"]
    try:
        has_digit = any(c.isdigit() for c in row["jp"])
        text = (row["jp_reading"] if has_digit else row["jp"]).strip()
        if not text:
            raise ValueError("empty synthesis text")
        audio = call_qwen_tts(host, voice, text, speed=speed, timeout=timeout)
        if not audio:
            raise ValueError("empty audio response")
        path = os.path.join(audio_dir, f"{row['word_rank']:04d}_{row_id}.mp3")
        with open(path, "wb") as f:
            f.write(audio)
        return {"id": row_id, "ok": True, "audio_path": path, "source": "jp_reading" if has_digit else "jp"}
    except Exception as e:
        return {"id": row_id, "ok": False, "error": str(e)[:500]}


def persist_result(conn, row, result, max_attempts):
    """sqlite writes only — call from the main thread to keep sqlite single-writer."""
    row_id = result["id"]
    if result["ok"]:
        conn.execute(
            """
            UPDATE examples SET
                audio_path = ?, audio_status = 'done', audio_error = NULL,
                stt_check_status = 'pending', stt_check_attempts = 0, stt_check_error = NULL,
                stt_transcript = NULL, stt_reading = NULL, stt_similarity = NULL, stt_mismatch = 0
            WHERE id = ?
            """,
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
    ap = argparse.ArgumentParser(description="Resynthesize audio for flagged stt_mismatch rows.")
    ap.add_argument("--db", default="vocab.db")
    ap.add_argument("--ids-file", required=True, help="file with one examples.id per line")
    ap.add_argument("--host", default="http://gb10-001:8886", help="Qwen3-TTS server host")
    ap.add_argument("--voice", default="ja_female_sora")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--audio-dir", default="audio")
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--workers", type=int, default=1, help="parallel TTS requests in flight")
    ap.add_argument("--timeout", type=int, default=120, help="per-request timeout in seconds")
    ap.add_argument("--retry-backoff", type=float, default=3.0, help="seconds to wait before retrying errored rows")
    args = ap.parse_args()

    target_ids = load_ids(args.ids_file)
    if not target_ids:
        sys.exit("no ids in --ids-file")

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    total = len(target_ids)
    print(f"{total} flagged examples to resynthesize (host={args.host}, voice={args.voice}, workers={args.workers})")

    t0 = time.time()
    n_done = 0
    round_num = 0
    pending_ids = list(target_ids)
    while True:
        round_num += 1
        if not pending_ids:
            break
        placeholders = ",".join("?" * len(pending_ids))
        batch = conn.execute(
            f"SELECT * FROM examples WHERE id IN ({placeholders}) "
            f"AND audio_status != 'failed' AND audio_attempts < ? ORDER BY id",
            (*pending_ids, args.max_attempts),
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
                    tag = f"ok (from {result['source']})"
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

        done_now = {
            r["id"] for r in conn.execute(
                f"SELECT id FROM examples WHERE id IN ({placeholders}) AND audio_status = 'done'",
                pending_ids,
            ).fetchall()
        }
        failed_now = {
            r["id"] for r in conn.execute(
                f"SELECT id FROM examples WHERE id IN ({placeholders}) AND audio_status = 'failed'",
                pending_ids,
            ).fetchall()
        }
        pending_ids = [i for i in pending_ids if i not in done_now and i not in failed_now]

    final = conn.execute(
        f"SELECT audio_status, count(*) c FROM examples WHERE id IN ({','.join('?' * len(target_ids))}) GROUP BY audio_status",
        target_ids,
    ).fetchall()
    summary = {r["audio_status"]: r["c"] for r in final}
    print(f"summary: {summary}")


if __name__ == "__main__":
    main()
