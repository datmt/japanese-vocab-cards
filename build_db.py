#!/usr/bin/env python3
"""Build a SQLite vocab database from corpus.txt, enriched via Ollama LLM.

For each of the ~5000 ranked entries in corpus.txt, calls a local Ollama
model to generate: a reading (kana), N example sentences (jp + reading + en),
and a kanji mnemonic (only for words containing kanji). The LLM-generated
reading is cross-checked against SudachiPy's dictionary reading; mismatches
are flagged but both readings are kept.

Resumable: progress is tracked per-row in the `words.status` column, so
re-running the script skips already-completed rows and retries
pending/errored ones.
"""
import argparse
import json
import re
import sqlite3
import sys
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from sudachipy import dictionary

KANJI_RE = re.compile(r"[一-鿿]")
RANK_LINE_RE = re.compile(r"^(\d+)\t(.+?)\t(.+?)\t(.+?)\t(.+)$")
WORD_SPLIT_RE = re.compile(r"[，,/]")

MAX_ATTEMPTS_DEFAULT = 3


def kata_to_hira(s: str) -> str:
    return "".join(
        chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in s
    )


def normalize_reading(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    return kata_to_hira(s).strip()


def parse_corpus(path):
    """Parse corpus.txt, return the first 5000 ranked rows (rank\\tword\\tpos\\tgloss\\tfreqstats)."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            m = RANK_LINE_RE.match(line)
            if not m:
                continue
            rank, word, pos, gloss, freq = m.groups()
            parts = [p.strip() for p in freq.split("|")]
            freq_count = int(parts[0]) if parts and parts[0].isdigit() else None
            freq_percent = None
            if len(parts) > 1:
                try:
                    freq_percent = float(parts[1])
                except ValueError:
                    pass
            freq_tag = parts[2] if len(parts) > 2 else None
            headword = WORD_SPLIT_RE.split(word)[0].strip()
            rows.append(
                {
                    "rank": int(rank),
                    "word": word,
                    "headword": headword,
                    "pos": pos.strip(),
                    "gloss": gloss.strip(),
                    "freq_count": freq_count,
                    "freq_percent": freq_percent,
                    "freq_tag": freq_tag,
                    "has_kanji": 1 if KANJI_RE.search(headword) else 0,
                }
            )
    return rows


def init_db(conn, rows):
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS words (
            rank INTEGER PRIMARY KEY,
            word TEXT NOT NULL,
            headword TEXT NOT NULL,
            pos TEXT,
            gloss TEXT,
            freq_count INTEGER,
            freq_percent REAL,
            freq_tag TEXT,
            has_kanji INTEGER NOT NULL,
            reading_llm TEXT,
            reading_dict TEXT,
            reading_mismatch INTEGER,
            mnemonic TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            attempts INTEGER NOT NULL DEFAULT 0,
            error TEXT
        );
        CREATE TABLE IF NOT EXISTS examples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            word_rank INTEGER NOT NULL REFERENCES words(rank),
            jp TEXT NOT NULL,
            jp_reading TEXT,
            en TEXT NOT NULL
        );
        """
    )
    conn.executemany(
        """
        INSERT OR IGNORE INTO words
            (rank, word, headword, pos, gloss, freq_count, freq_percent, freq_tag, has_kanji)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                r["rank"], r["word"], r["headword"], r["pos"], r["gloss"],
                r["freq_count"], r["freq_percent"], r["freq_tag"], r["has_kanji"],
            )
            for r in rows
        ],
    )
    conn.commit()


def build_prompt(word, pos, gloss, has_kanji, n_examples):
    mnemonic_instr = (
        '"mnemonic": "<a short, vivid mnemonic in English to remember the kanji shape/meaning for this word>",'
        if has_kanji
        else ""
    )
    return f"""You are a Japanese dictionary assistant. For the vocabulary entry below, respond with ONLY a JSON object (no markdown, no commentary).

Word: {word}
Part of speech: {pos}
English gloss: {gloss}

Return JSON with this exact shape:
{{
  "reading": "<the word's reading written in hiragana>",
  {mnemonic_instr}
  "examples": [
    {{"jp": "<natural Japanese example sentence using the word>", "reading": "<full sentence reading in hiragana>", "en": "<English translation>"}}
    // exactly {n_examples} items total
  ]
}}
"""


def call_ollama(host, model, prompt, timeout=120):
    resp = requests.post(
        f"{host}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.3},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    content = resp.json()["message"]["content"]
    return json.loads(content)


_thread_local = threading.local()


def get_tokenizer():
    tok = getattr(_thread_local, "tokenizer", None)
    if tok is None:
        tok = dictionary.Dictionary().create()
        _thread_local.tokenizer = tok
    return tok


def dict_reading(headword):
    try:
        tokens = get_tokenizer().tokenize(headword)
        if not tokens:
            return ""
        return normalize_reading("".join(t.reading_form() for t in tokens))
    except Exception:
        return ""


def compute_row(host, model, n_examples, row):
    """Network/CPU work only, no DB access — safe to run from worker threads."""
    rank = row["rank"]
    prompt = build_prompt(row["word"], row["pos"], row["gloss"], row["has_kanji"], n_examples)
    try:
        data = call_ollama(host, model, prompt)
        reading_llm = normalize_reading(data.get("reading", ""))
        examples = data.get("examples", [])[:n_examples]
        if not reading_llm or not examples:
            raise ValueError(f"incomplete LLM response: {data!r}")
        mnemonic = data.get("mnemonic") if row["has_kanji"] else None

        reading_dict_val = dict_reading(row["headword"])
        mismatch = (
            1
            if reading_dict_val and reading_llm and reading_dict_val != reading_llm
            else 0
        )
        return {
            "rank": rank,
            "ok": True,
            "reading_llm": reading_llm,
            "reading_dict": reading_dict_val,
            "mismatch": mismatch,
            "mnemonic": mnemonic,
            "examples": examples,
        }
    except Exception as e:
        return {"rank": rank, "ok": False, "error": str(e)[:500]}


def persist_result(conn, row, result, max_attempts):
    """DB writes only — call from the main thread to keep sqlite single-writer."""
    rank = result["rank"]
    if result["ok"]:
        conn.execute("DELETE FROM examples WHERE word_rank = ?", (rank,))
        conn.executemany(
            "INSERT INTO examples (word_rank, jp, jp_reading, en) VALUES (?, ?, ?, ?)",
            [
                (rank, e.get("jp", ""), e.get("reading", ""), e.get("en", ""))
                for e in result["examples"]
            ],
        )
        conn.execute(
            """
            UPDATE words SET
                reading_llm = ?, reading_dict = ?, reading_mismatch = ?,
                mnemonic = ?, status = 'done', error = NULL
            WHERE rank = ?
            """,
            (result["reading_llm"], result["reading_dict"], result["mismatch"], result["mnemonic"], rank),
        )
    else:
        attempts = row["attempts"] + 1
        status = "failed" if attempts >= max_attempts else "error"
        conn.execute(
            "UPDATE words SET status = ?, attempts = ?, error = ? WHERE rank = ?",
            (status, attempts, result["error"], rank),
        )
    conn.commit()


def main():
    ap = argparse.ArgumentParser(description="Build Japanese vocab SQLite DB via Ollama LLM enrichment.")
    ap.add_argument("--corpus", default="corpus.txt")
    ap.add_argument("--db", default="vocab.db")
    ap.add_argument("--host", default="http://gb10-001:11434")
    ap.add_argument("--model", default="qwen3.6:35b-a3b")
    ap.add_argument("--examples", type=int, default=2, help="number of example sentences per word")
    ap.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS_DEFAULT)
    ap.add_argument("--limit", type=int, default=None, help="only process N rows (testing)")
    ap.add_argument("--workers", type=int, default=1, help="parallel ollama requests in flight")
    args = ap.parse_args()

    rows = parse_corpus(args.corpus)
    print(f"parsed {len(rows)} ranked entries from {args.corpus}")

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    init_db(conn, rows)

    pending = conn.execute(
        "SELECT * FROM words WHERE status IN ('pending', 'error') ORDER BY rank"
    ).fetchall()
    if args.limit:
        pending = pending[: args.limit]

    total = len(pending)
    print(f"{total} words to process (model={args.model}, host={args.host}, workers={args.workers})")

    done = mismatches = failed = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(compute_row, args.host, args.model, args.examples, row): row
            for row in pending
        }
        for i, fut in enumerate(as_completed(futures), 1):
            row = futures[fut]
            result = fut.result()
            persist_result(conn, row, result, args.max_attempts)
            if result["ok"]:
                done += 1
                if result["mismatch"]:
                    mismatches += 1
                tag = "MISMATCH" if result["mismatch"] else "ok"
            else:
                failed += 1
                tag = f"FAILED ({result['error']})"
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            eta = (total - i) / rate if rate > 0 else float("inf")
            print(f"[{i}/{total}] rank={row['rank']} {row['word']!r} -> {tag}  (eta {eta/60:.1f}m)", file=sys.stderr)

    print(f"done={done} mismatches={mismatches} failed={failed}")


if __name__ == "__main__":
    main()
