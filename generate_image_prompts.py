#!/usr/bin/env python3
"""Turn each word's kanji mnemonic into a clean txt2img scene prompt.

Mnemonics are written as instructions to a learner ("Imagine...",
"Visualize yourself...") and often ask the reader to picture the kanji
itself. Diffusion models render that literally (producing garbled text)
and follow the meta-phrasing instead of just depicting the scene. This
script calls an LLM to rewrite each mnemonic into a plain scene
description with no meta phrasing and no instruction to draw the kanji
glyph (the glyph gets composited on top of the generated image later,
as text, not by the diffusion model).

Resumable: progress tracked per-row in `words.image_prompt_status`,
same pattern as build_db.py's `status` column.
"""
import argparse
import os
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from build_db import OPENROUTER_URL

MAX_ATTEMPTS_DEFAULT = 3

PROMPT_TEMPLATE = """You write prompts for a text-to-image diffusion model. Convert the kanji mnemonic below into a short, vivid txt2img scene prompt.

Rules:
- Output ONLY the image prompt, no commentary, no markdown, no quotes.
- Describe the scene directly (e.g. "a heart resting on a rice field at sunset"), never instructions to a reader ("imagine", "visualize", "picture yourself").
- Never ask for any kanji, kana, or text to be drawn/rendered in the image — the model can't render CJK text reliably. Describe only the visual metaphor/components (e.g. "a heart shape" not "the kanji 心").
- Keep it concrete and illustratable: objects, composition, setting, mood, art style (e.g. "flat minimalist illustration", "soft pastel colors").
- One sentence, under 40 words.

Word: {word}
Mnemonic: {mnemonic}

Image prompt:"""


def build_prompt(word, mnemonic):
    return PROMPT_TEMPLATE.format(word=word, mnemonic=mnemonic)


def call_ollama_text(host, model, prompt, timeout=300, think=False):
    resp = requests.post(
        f"{host}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": think,
            "options": {"temperature": 0.5},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def call_openrouter_text(model, prompt, timeout=300, think=False):
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set in environment")
    resp = requests.post(
        OPENROUTER_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "reasoning": {"enabled": think},
            "temperature": 0.5,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"openrouter error: {data['error']}")
    return data["choices"][0]["message"]["content"]


def call_llm_text(backend, host, model, prompt, timeout=300, think=False):
    if backend == "openrouter":
        return call_openrouter_text(model, prompt, timeout=timeout, think=think)
    return call_ollama_text(host, model, prompt, timeout=timeout, think=think)


def ensure_columns(conn):
    cols = {row[1] for row in conn.execute("PRAGMA table_info(words)")}
    if "image_prompt" not in cols:
        conn.execute("ALTER TABLE words ADD COLUMN image_prompt TEXT")
    if "image_prompt_status" not in cols:
        conn.execute("ALTER TABLE words ADD COLUMN image_prompt_status TEXT NOT NULL DEFAULT 'pending'")
    if "image_prompt_attempts" not in cols:
        conn.execute("ALTER TABLE words ADD COLUMN image_prompt_attempts INTEGER NOT NULL DEFAULT 0")
    if "image_prompt_error" not in cols:
        conn.execute("ALTER TABLE words ADD COLUMN image_prompt_error TEXT")
    conn.commit()


def compute_row(backend, host, model, row, timeout=300, think=False):
    """Network work only, no DB access — safe to run from worker threads."""
    rank = row["rank"]
    prompt = build_prompt(row["word"], row["mnemonic"])
    try:
        content = call_llm_text(backend, host, model, prompt, timeout=timeout, think=think)
        text = content.strip().strip('"')
        if not text:
            raise ValueError("empty image prompt")
        return {"rank": rank, "ok": True, "image_prompt": text}
    except Exception as e:
        return {"rank": rank, "ok": False, "error": str(e)[:500]}


def persist_result(conn, row, result, max_attempts):
    """DB writes only — call from the main thread to keep sqlite single-writer."""
    rank = result["rank"]
    if result["ok"]:
        conn.execute(
            "UPDATE words SET image_prompt = ?, image_prompt_status = 'done', image_prompt_error = NULL WHERE rank = ?",
            (result["image_prompt"], rank),
        )
    else:
        attempts = row["image_prompt_attempts"] + 1
        status = "failed" if attempts >= max_attempts else "error"
        conn.execute(
            "UPDATE words SET image_prompt_status = ?, image_prompt_attempts = ?, image_prompt_error = ? WHERE rank = ?",
            (status, attempts, result["error"], rank),
        )
    conn.commit()


def main():
    ap = argparse.ArgumentParser(description="Generate txt2img scene prompts from kanji mnemonics.")
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
    if args.backend == "openrouter" and not os.environ.get("OPENROUTER_API_KEY"):
        sys.exit("OPENROUTER_API_KEY not set in environment (required for --backend openrouter)")

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    ensure_columns(conn)

    pending = conn.execute(
        "SELECT * FROM words WHERE has_kanji = 1 AND mnemonic IS NOT NULL "
        "AND image_prompt_status IN ('pending', 'error') ORDER BY rank"
    ).fetchall()
    if args.limit:
        pending = pending[: args.limit]
    target_ranks = [r["rank"] for r in pending]
    total = len(target_ranks)
    print(f"{total} words to process (backend={args.backend}, model={args.model}, workers={args.workers})")

    t0 = time.time()
    n_done = 0
    round_num = 0
    while True:
        round_num += 1
        if not target_ranks:
            break
        placeholders = ",".join("?" * len(target_ranks))
        batch = conn.execute(
            f"SELECT * FROM words WHERE rank IN ({placeholders}) AND image_prompt_status IN ('pending', 'error') "
            f"AND image_prompt_attempts < ? ORDER BY rank",
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
                    tag = "ok"
                else:
                    attempts = row["image_prompt_attempts"] + 1
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
        "SELECT image_prompt_status, count(*) c FROM words WHERE has_kanji = 1 AND mnemonic IS NOT NULL GROUP BY image_prompt_status"
    ).fetchall()
    summary = {r["image_prompt_status"]: r["c"] for r in final}
    print(f"summary: {summary}")


if __name__ == "__main__":
    main()
