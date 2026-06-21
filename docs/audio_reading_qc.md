# Audio reading QC (TTS mispronunciation detection)

## Problem

`generate_audio.py` feeds the raw kanji sentence (`examples.jp`) to Qwen3-TTS.
The TTS server does its own grapheme-to-phoneme (G2P) internally — it never
sees our curated `jp_reading` (the reading the UI displays, already
arbitrated by `check_example_readings.py` / `resolve_reading_mismatches.py`).
For homographs (本 = ほん "book" or もと "origin", 私 = わたし or わたくし,
何 = なに or なん, 日本 = にほん or にっぽん, ...) the TTS engine's G2P can
pick a different — often equally "correct" — reading than the one we show
the learner. The result: the clip says something subtly different from what
the flashcard claims it says.

`check_audio_readings.py` exists to flag these clips by sending the rendered
audio to an STT (Whisper) server and comparing what it hears against
`jp_reading`.

## First attempt: STT text → SudachiPy reading, and why it failed

The first version transcribed the clip with Whisper (plain transcription,
no special prompting), ran the transcript through `dict_reading()`
(SudachiPy morphological reading, same helper `build_db.py` and
`check_example_readings.py` already use), and compared that reading against
`jp_reading` with a fuzzy ratio (`difflib.SequenceMatcher`).

A 300-row test batch showed a clean-looking two-cluster distribution (most
rows 0.9–1.0, a tail at 0.7–0.85) — but manually inspecting the low scorers
turned up a real bug:

```
>>> from build_db import dict_reading
>>> dict_reading("彼は今、本を読んでいる。")   # with comma, as in jp_reading's source
'かれはいま、ほんをよんでいる。'
>>> dict_reading("彼は今本を読んでいる")       # comma dropped, as Whisper transcribed it
'かれはいまもとをよんでいる'
```

Identical sentence, identical audio — the only difference is Whisper didn't
reproduce the comma. SudachiPy's sentence segmentation uses that comma as a
boundary cue, and without it the tokenizer re-groups the sentence differently
and picks the **other** citation reading for 本. The "mismatch" had nothing
to do with what the TTS engine actually pronounced; it was an artifact of
re-running text through a reading-inference step that is itself sensitive to
incidental punctuation loss in STT output. Two rows originally reported as
"genuine TTS misreads" (本→もと, 方→ほう) turned out to be exactly this
artifact, confirmed by reproducing it directly against `dict_reading()`.

This also raised a scoping question: even where SudachiPy *did* resolve a
reading reliably, a homograph clip and the dictionary's resolution are both
opinions about the text — neither one tells you what the TTS engine's G2P
actually produced acoustically. Re-deriving a reading from STT's output text
inherits all of SudachiPy's own homograph guesswork, on top of whatever
errors STT introduced. Comparing kanji surface text directly (Whisper output
vs `jp`, punctuation stripped) was considered as an alternative, but it has
the opposite problem: kanji spelling doesn't change between homograph
readings, so a real TTS misread (もと spoken where ほん was intended) would
be invisible to a pure text diff. Neither "reading vs reading" nor "kanji
text vs kanji text" actually observes the one thing we care about: which
reading the audio contains.

## Fix: bias Whisper toward kana output

Whisper's `initial_prompt` parameter seeds its decoder context and can bias
output style. Passing an all-hiragana sentence that explicitly states "this
is written entirely in hiragana, no kanji" pushes Whisper to render its
transcript in kana more often, and — even when it doesn't go fully kana —
measurably improves punctuation fidelity (the comma-loss bug above stops
happening in practice).

```python
KANA_BIAS_PROMPT = "ええ、あの、これはぜんぶひらがなでかいたぶんしょうです。かんじはつかいません。"
```

Verified against the gb10 Whisper server (`http://gb10-001:8887/transcribe`)
on the previously-flagged rows:

| id | `jp_reading` (target) | unbiased STT → `dict_reading()` | biased STT output |
|----|------------------------|----------------------------------|--------------------|
| 25  | かれはいま、ほんをよんでいる。 | もと (false flag — segmentation artifact) | かれはいま、ほんをよんでいる。 — exact match |
| 182 | このかた、たなかともうします。 | ほう (false flag) | kept kanji, but recovered the comma — `dict_reading()` now resolves it correctly too |
| 79  | なにをたべますか。 | なん (false flag) | なにをたべますか。 — exact match |
| 84  | わたしはにほんにいきます。 | にっぽん (false flag) | わたしはにほんにいきます。 — exact match |
| 87  | それはなにですか。 | なん (ambiguous before) | それはなんですか。 — genuinely different from target, real flag |
| 1   | (already exact) | matched | matched — no regression |

Four of six previously-flagged rows turned out to be false positives caused
by the segmentation bug; the bias prompt resolves them. id=87 is the
interesting case: the bias prompt's kana output shows the TTS engine really
did say なん where the flashcard displays なに — that's a genuine,
trustworthy flag, not a guess about what SudachiPy thinks the text should
say.

The `dict_reading()` step on the STT output is still kept in
`check_audio_readings.py` — when Whisper doesn't fully convert to kana
(proper nouns especially tend to stay in kanji, e.g. 田中 in id=182), we
still need to derive a reading from whatever mixed kanji/kana text comes
back, and with punctuation now reliably preserved, that derivation is no
longer artifact-prone the way it was without the bias prompt.

## Caveats / known limitations

- STT has its own noise independent of this fix: vowel-length notation
  choices (あー vs ああ — same sound, different transliteration) and
  occasional hallucinated words off spoken punctuation (a "?" sometimes
  transcribed as 記号) will still produce a handful of low-similarity false
  positives. This is why the script flags *candidates for review*, not a
  final verdict.
- The fuzzy-match `--threshold` (default 0.85) was tuned by eyeballing the
  similarity distribution on a few hundred rows: a clear gap exists between
  the clean cluster (0.9–1.0) and the flagged cluster (0.70–0.85). Re-check
  this distribution if the corpus or TTS voice changes meaningfully.
- This script only flags; it does not correct. See
  `resolve_reading_mismatches.py` / `resolve_word_reading_mismatches.py` for
  the precedent of a second pass (LLM arbitration) deciding what to do with
  flagged rows — for audio, the correction path (re-render with a forced
  reading, manual review, or a stronger/paid STT/TTS pass on the small
  flagged set) is still an open decision.
