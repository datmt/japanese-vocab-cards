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

## Second fix: strip punctuation before comparing

Even with the bias prompt, a remaining class of false positive showed up:
Whisper's punctuation-*glyph* choice doesn't always match the corpus text's,
even when the actual words/reading are identical. id=80:

```
jp_reading:  なんですか。   (corpus uses a full stop)
stt_reading: なんですか?   (Whisper guessed a question mark — reasonable, it is a question)
```

Same reading, same pronunciation — but on an 8-character string, one glyph
difference dropped the ratio to 0.83, into flagged territory. TTS doesn't
speak punctuation, so it shouldn't count against a reading match. Fix:
strip `。、！？!?,.` and whitespace from both sides before computing the
similarity ratio (the unstripped values are still stored in `stt_reading`/
`jp_reading` for the audit trail — only the comparison strips them).

## Third fix: normalize chōonpu notation before comparing

Same family of bug, different glyph: Whisper sometimes writes a long vowel
as a doubled vowel kana instead of the chōonpu mark ー — same sound, two
spellings. id=243:

```
jp_reading:  おー、すごい！
stt_reading: おお、すごい。
```

After punctuation-stripping, `おーすごい` vs `おおすごい` still differ by one
char (sim 0.8, flagged). Fix: expand `ー` to the vowel of the preceding kana
on both sides before comparing (`expand_chōonpu()`), so おー and おお both
normalize to おお. Verified id=243 now scores 1.0.

## Known unfixed flaw: kana-bias can break jukujikun readings

id=107 (何時ですか / いつですか — 何時 is jukujikun, idiomatic, not built
from each character's own on/kun-yomi):

```
unbiased STT: 何時ですか?    (kept kanji intact)
biased STT:   なにじですか。  (sounded out 何=なに, 時=じ — wrong, audio is fine)
```

Listened to the clip directly — TTS said いつ correctly. The bias prompt's
push toward "spell everything phonetically" makes Whisper guess a
regularized character-by-character reading for this jukujikun instead of
transcribing what it actually heard. Note the unbiased pass got this one
right (`dict_reading("何時ですか?")` correctly resolves to いつですか via
SudachiPy's jukujikun dictionary entry, since the kanji form survived) —
opposite of the punctuation case, where biased was the fix and unbiased was
broken. Neither pass alone is reliably correct across both failure modes.

## Known unfixed flaw: whole-sentence fuzzy ratio dilutes short word-level errors

word_rank=1108 (伯父, correct reading おじ): TTS pronounced it はくじ — not
おじ, not even the formal alt はくふ, just wrong. The biased STT transcript
correctly heard はくじ (STT is not at fault here). But the example sentences
are long (20+ chars), so 3 wrong characters barely move the whole-string
fuzzy ratio: id=2211 scored 0.93, id=2212 scored 0.94 — both well above the
0.85 threshold, so this real, confirmed TTS error was never flagged. This is
the length-dilution problem named as a caveat from the start, now confirmed
with a real case, not hypothetical.

Fix direction (discussed, not yet built): a second, word-targeted script
that doesn't rely on whole-sentence similarity at all. `example_breakdown`
(populated by `generate_breakdown.py`) already has a per-token `reading`
column, context-resolved per sentence (e.g. id=2211 seq=0: surface=伯父,
reading=おじ) — no need to re-tokenize `jp` ourselves or do offset math.
Plan:
- For each example, find its `example_breakdown` row matching the headword
  (match by surface for nouns; for inflecting words like verbs, map via
  SudachiPy `dictionary_form()` on the breakdown row's surface, e.g.
  引っ越して → 引っ越す, to match `words.headword`).
  Note: ground truth must come from `examples.jp_reading`/`example_breakdown`
  (per-sentence, context-correct), **not** a fixed `words.reading_llm`
  lookup — a word can legitimately have different valid readings in
  different sentences (heteronyms), so a per-word fixed reading would
  itself generate false positives.
- Take that breakdown row's `reading` as the localized ground truth.
- Check it survives as a substring in the already-collected `stt_reading`
  for that row (reuse `check_audio_readings.py`'s existing data, no new STT
  calls needed) — localized, not diluted by sentence length.
- Coverage caveat: `example_breakdown` is only populated where
  `breakdown_status = 'done'` — at last check 5179/10000 done, 3 errored,
  4818 still pending (`generate_breakdown.py --workers 1` running). The
  word-targeted check can only cover rows with breakdown done.

## Known unfixed flaw: weak/slurred mora invisible to STT-based QC

id=165 (後で連絡します。/ あとでれんらくします。): human listening flags the
final く in らく as barely audible/slurred. Confirmed independently with two
other TTS engines (Gemini TTS, ElevenLabs) producing the same weak く —
not a Qwen3-TTS-specific bug.

`check_audio_readings.py` does not catch this: both biased and unbiased
Whisper passes transcribe it as れんらく/連絡 with sim=1.0 — ASR's language
model fills in the expected word from context even when the acoustic signal
for one mora is weak, so a correct-reading transcript is not proof the audio
was clearly articulated. `ffmpeg silencedetect` also didn't cleanly confirm
it (no full silence on the mora, consistent with "weak" rather than
"dropped" — and a kana-input regeneration test showed a similarly-placed gap,
inconclusive either way).

This class of defect — under-articulation/weak phonemes that STT
reconstructs from context rather than truly hearing — is structurally
outside what a reading-comparison QC script can catch. Detecting it
reliably would need per-mora energy/duration analysis (forced alignment)
against reference recordings, not implemented. Logged as a known gap, not
queued for a fix.

## TODO

- **HIGH PRIORITY — pick up here next session:** build the word-targeted
  check described above once both background jobs below have more
  coverage (or are done):
  - `check_audio_readings.py --workers 4` (PID may differ next session) —
    sentence-level STT check, was at 5556/10000 done as of last check.
  - `generate_breakdown.py --workers 1` — was at 5179/10000 done (+3
    error) as of last check; needed for `example_breakdown` coverage.
  - Both are detached (`nohup ... & disown`), survive across sessions on
    their own — just re-check `pgrep -af check_audio_readings` /
    `pgrep -af generate_breakdown.py` and the `*_status` column counts in
    `vocab.db` to see where they left off.
  - Once `check_audio_readings.py` finishes, also pull the full
    `stt_mismatch = 1` list for manual review before building anything
    further on top of it.
- Run both biased and unbiased transcription per clip, derive a reading from
  each, and take `max(sim_biased, sim_unbiased)` as the final score — only
  flag a clip if *neither* pass reproduces the target reading. Fixes the
  jukujikun case (id=107) without reintroducing the punctuation/segmentation
  bug (the first fix) that the unbiased-only approach had. Costs 2x the STT
  calls per clip (run time roughly doubles). Not implemented yet — do this
  before trusting the flagged list for jukujikun-heavy vocabulary.
- After this full pass settles, re-run `check_audio_readings.py` scoped to
  `stt_mismatch = 1` one more time (e.g. add a `--recheck` flag, or just
  re-flip those rows to `stt_check_status = 'pending'`) to confirm flags
  hold up to a second independent STT pass before treating the flagged list
  as final — Whisper's own non-determinism (or the dual-pass fix above)
  could still clear some of them.

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
