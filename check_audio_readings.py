#!/usr/bin/env python3
"""Flag example audio clips where Qwen3-TTS likely mispronounced the sentence.

generate_audio.py feeds the raw kanji sentence (`examples.jp`) to the TTS
server's own grapheme-to-phoneme step — that internal G2P can disagree with
our curated `jp_reading` (the dict/LLM-arbitrated reading the UI displays),
in which case the rendered clip says something subtly different from what we
claim it says. This sends each clip to an STT server, derives the spoken
reading from the transcript via SudachiPy, and flags rows where it diverges
from `jp_reading` by more than a fuzzy-match tolerance.

ASR has its own noise (recognition slips unrelated to TTS pronunciation), so
a flag here is a candidate for review, not a verdict — see
resolve_reading_mismatches.py for the precedent of a second pass (LLM or
manual) deciding what to do with flagged rows; this script only flags.

Resumable via `examples.stt_check_status`, same pattern as the other
per-row enrichment scripts.
"""
import argparse
import difflib
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from build_db import dict_reading, normalize_reading

MAX_ATTEMPTS_DEFAULT = 3
DEFAULT_SIMILARITY_THRESHOLD = 0.85

# Seeds Whisper's decoder context toward an all-hiragana style. Without this,
# Whisper defaults to standard kanji orthography, and re-deriving a reading
# from that text via SudachiPy is brittle: punctuation Whisper drops (esp.
# commas) changes Sudachi's sentence segmentation, which can flip its
# homograph reading choice (e.g. 本 -> もと instead of ほん) independent of
# what was actually said — see docs/audio_reading_qc.md for the investigation.
# With the bias, Whisper either outputs kana directly (revealing the true
# spoken reading) or at least preserves punctuation well enough that
# dict_reading() segments correctly.
KANA_BIAS_PROMPT = "ええ、あの、これはぜんぶひらがなでかいたぶんしょうです。かんじはつかいません。"


def ensure_columns(conn):
    cols = {row[1] for row in conn.execute("PRAGMA table_info(examples)")}
    if "stt_transcript" not in cols:
        conn.execute("ALTER TABLE examples ADD COLUMN stt_transcript TEXT")
    if "stt_reading" not in cols:
        conn.execute("ALTER TABLE examples ADD COLUMN stt_reading TEXT")
    if "stt_similarity" not in cols:
        conn.execute("ALTER TABLE examples ADD COLUMN stt_similarity REAL")
    if "stt_mismatch" not in cols:
        conn.execute("ALTER TABLE examples ADD COLUMN stt_mismatch INTEGER NOT NULL DEFAULT 0")
    if "stt_check_status" not in cols:
        conn.execute("ALTER TABLE examples ADD COLUMN stt_check_status TEXT NOT NULL DEFAULT 'pending'")
    if "stt_check_attempts" not in cols:
        conn.execute("ALTER TABLE examples ADD COLUMN stt_check_attempts INTEGER NOT NULL DEFAULT 0")
    if "stt_check_error" not in cols:
        conn.execute("ALTER TABLE examples ADD COLUMN stt_check_error TEXT")
    conn.commit()


def call_whisper(host, audio_path, timeout=120):
    """POST the clip to the gb10 whisper server, return its raw text transcript."""
    with open(audio_path, "rb") as f:
        resp = requests.post(
            f"{host}/transcribe",
            files={"file": (audio_path, f, "audio/wav")},
            data={
                "language": "ja", "task": "transcribe",
                "initial_prompt": KANA_BIAS_PROMPT, "response_format": "verbose_json",
            },
            timeout=timeout,
        )
    resp.raise_for_status()
    return resp.json()["text"]


def compute_row(host, row, threshold, timeout=120):
    """Network/CPU work only, no DB access — safe to run from worker threads."""
    row_id = row["id"]
    try:
        transcript = call_whisper(host, row["audio_path"], timeout=timeout)
        if not transcript or not transcript.strip():
            raise ValueError("empty transcript")
        stt_reading = dict_reading(transcript)
        target_reading = normalize_reading(row["jp_reading"] or "")
        sim = difflib.SequenceMatcher(None, stt_reading, target_reading).ratio()
        return {
            "id": row_id,
            "ok": True,
            "transcript": transcript,
            "stt_reading": stt_reading,
            "similarity": sim,
            "mismatch": sim < threshold,
        }
    except Exception as e:
        return {"id": row_id, "ok": False, "error": str(e)[:500]}


def persist_result(conn, row, result, max_attempts):
    """DB writes only — call from the main thread to keep sqlite single-writer."""
    row_id = result["id"]
    if result["ok"]:
        conn.execute(
            "UPDATE examples SET stt_transcript = ?, stt_reading = ?, stt_similarity = ?, "
            "stt_mismatch = ?, stt_check_status = 'done', stt_check_error = NULL WHERE id = ?",
            (result["transcript"], result["stt_reading"], result["similarity"], int(result["mismatch"]), row_id),
        )
    else:
        attempts = row["stt_check_attempts"] + 1
        status = "failed" if attempts >= max_attempts else "error"
        conn.execute(
            "UPDATE examples SET stt_check_status = ?, stt_check_attempts = ?, stt_check_error = ? WHERE id = ?",
            (status, attempts, result["error"], row_id),
        )
    conn.commit()


def main():
    ap = argparse.ArgumentParser(description="Flag example audio where STT disagrees with jp_reading.")
    ap.add_argument("--db", default="vocab.db")
    ap.add_argument("--host", default="http://gb10-001:8887", help="whisper STT server host")
    ap.add_argument(
        "--threshold", type=float, default=DEFAULT_SIMILARITY_THRESHOLD,
        help="fuzzy-match ratio (0-1) below which a row is flagged as mismatch",
    )
    ap.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS_DEFAULT)
    ap.add_argument("--limit", type=int, default=None, help="only process N rows (testing)")
    ap.add_argument("--workers", type=int, default=1, help="parallel STT requests in flight")
    ap.add_argument("--timeout", type=int, default=120, help="per-request timeout in seconds")
    ap.add_argument("--retry-backoff", type=float, default=3.0, help="seconds to wait before retrying errored rows")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    ensure_columns(conn)

    pending = conn.execute(
        "SELECT * FROM examples WHERE audio_status = 'done' AND audio_path IS NOT NULL "
        "AND stt_check_status IN ('pending', 'error') ORDER BY id"
    ).fetchall()
    if args.limit:
        pending = pending[: args.limit]
    target_ids = [r["id"] for r in pending]
    total = len(target_ids)
    print(f"{total} clips to STT-check (host={args.host}, threshold={args.threshold}, workers={args.workers})")

    t0 = time.time()
    n_done = 0
    round_num = 0
    while True:
        round_num += 1
        if not target_ids:
            break
        placeholders = ",".join("?" * len(target_ids))
        batch = conn.execute(
            f"SELECT * FROM examples WHERE id IN ({placeholders}) AND stt_check_status IN ('pending', 'error') "
            f"AND stt_check_attempts < ? ORDER BY id",
            (*target_ids, args.max_attempts),
        ).fetchall()
        if not batch:
            break
        if round_num > 1:
            print(f"-- retry round {round_num}: {len(batch)} row(s) --", file=sys.stderr)
            time.sleep(args.retry_backoff)

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(compute_row, args.host, row, args.threshold, args.timeout): row
                for row in batch
            }
            for fut in as_completed(futures):
                row = futures[fut]
                result = fut.result()
                persist_result(conn, row, result, args.max_attempts)
                if result["ok"]:
                    n_done += 1
                    tag = f"sim={result['similarity']:.2f} {'MISMATCH' if result['mismatch'] else 'ok'}"
                else:
                    attempts = row["stt_check_attempts"] + 1
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
        "SELECT stt_check_status, count(*) c FROM examples GROUP BY stt_check_status"
    ).fetchall()
    summary = {r["stt_check_status"]: r["c"] for r in final}
    flagged = conn.execute("SELECT count(*) FROM examples WHERE stt_mismatch = 1").fetchone()[0]
    print(f"summary: {summary} flagged mismatches={flagged}")


if __name__ == "__main__":
    main()
