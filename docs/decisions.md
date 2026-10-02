# Decisions and conventions

Every choice that changes a number, with its reason. Items marked **[Jensen]**
are ones to raise at the next check-in.

## Data

**D1. AMI references: BUT/pyannote setup, `only_words`, Full-corpus-ASR.**
Pinned at upstream commit `67c2d53`. 136/18/16 meetings. References are
derived from the AMI manual word annotations v1.6.2 with adjacent
same-speaker words merged. `word_and_vocalsounds` is prepared and reported in
the overlap table for comparison only; upstream notes annotator
inconsistency in vocal sounds. Files are copied byte-for-byte and checksummed
in `data/ami/MANIFEST.json`.

Validation on copy: every listed meeting has an RTTM and UEM naming that
meeting; dev and test have 0.0 s of reference outside the UEM. Train has
5.75 speaker-seconds outside, all in TS3007c, whose UEM ends at 2420.0 s while
the reference runs to 2430.16 s. Left as published (outside-UEM time is never
scored, and train is not scored).

**D2. One reference for both microphone conditions.** IHM-Mix and SDM share
the same RTTM and UEM; only the audio differs. Upstream UEMs span each whole
recording. Whether SDM audio length matches the UEM exactly is checked when
audio is downloaded (step 5), not assumed.

**D3. Supreme Court: raw Oyez turns, not hand-corrected.** The reference is
built exactly as `lausanne/src/eval/fetch_oyez.py` built it. It contains no
overlap and no silence: all 248 turns tile 0–62.4 min with zero gaps. It
therefore cannot support the overlap-conditioned decomposition, and DER
against it structurally has zero false alarm and charges any correctly
detected pause as missed speech. **[Jensen]** Either an overlap-annotated
version is produced, or this set is used for attribution only. Its UEM is
the span of the reference turns, since no audio has been fetched.

## Overlap statistics (step 2)

**S1. Overlap is defined on the reference, after per-speaker union.** A
speaker whose own segments overlap is one speaker. Times are computed by
exact interval sweep inside the UEM, no frame grid.

**S2. Two overlap fractions are reported.** `overlap_time_frac` = time with
>= 2 speakers / time with >= 1 speaker (the usual "X% overlap").
`overlap_spk_time_frac` = reference speaker-time inside overlap / all
reference speaker-time. DER's denominator is speaker-time, so the second is
the share of what DER scores that lies in overlap.

**S3. Aggregates are pooled (time-weighted) with a percentile bootstrap over
meetings.** The pooled ratio matches how DER is aggregated over a corpus. The
unweighted mean of per-meeting ratios is reported beside it. 10,000
resamples, 95%, fixed seed in `configs/stats/overlap.yaml`. A single-file
dataset gets no interval (NaN), never a fabricated one.

## Flags for Jensen (literature vs. brief)

**F1. Reporting conventions.** The brief says DER is "almost always" reported
with a 0.25 s collar and overlap excluded. Landini et al. (2022) define three
AMI setups in use (Forgiving: collar 0.25, no overlap; Fair: collar 0.25,
overlap scored; Full: collar 0, overlap scored), and pyannote's 3.1 benchmark
reports Full. Safer framing: the same system is reported under incompatible
conventions, and the forgiving one hides most of the error.

**F2. The 40.4-point result.** In lausanne (manuscript Table 3, from
`src/eval/bias.py`), 0.404 is the *anonymised* condition (all labels set to
one value). The *permuted* condition, the one DER is invariant to, shifted
0.036, below the 0.05 threshold. The control was exactly 0. The result shows
the note selector uses speaker identity as structure; it does not show that a
label permutation moved the notes.

**F3. Fixed-mapping metric and anonymous labels.** Majority-overlap mapping on
single-speaker regions is still an estimated mapping, so a global relabel is
absorbed by it as it is by DER and cpWER. What it does detect is inconsistency
between how a speaker is labelled in single-speaker vs overlapped speech, and
word-level misattribution. Prior work to cite: WDER (Shafey et al. 2019),
SA-WER (Kanda et al. 2020, uses known identities), MeetEval tcpWER/ORC-WER.

**F4. pyannote version.** lausanne ran pyannote.audio >= 4.0 with
`speaker-diarization-community-1` by default (3.1 as fallback), and the
manuscript does not state which produced 7.92%. This repo uses 3.1 per the
brief.

**F5. "AMI has roughly 15–20% overlap."** Measured (run `2eaaa1a309`):
`only_words` gives 13.4% (train), 14.1% (dev), 14.6% (test) of speech time;
15–20% matches `word_and_vocalsounds` (15.7–16.5%), not the primary
annotation. Quote the measured figure with its annotation.

**F6. AMI test overlap is bimodal.** The four EN2002 meetings sit at
24–30% overlap; TS3003a–c and IS1009c sit at 4–5%. The test CI is wide
([10.1, 19.1]%) for this reason, and any test-set DER aggregate will be
driven disproportionately by EN2002. Per-meeting results must be read
alongside the pooled number.

## Scorer (step 3)

**C1. md-eval semantics, reproduced exactly.** See `third_party/README.md`
for the source lines. In particular the speaker mapping is computed before
collars and overlap exclusion, so the mapping does not change between the
four cells of the collar x overlap grid. pyannote.metrics and dscore make
the same choice for the mapping, but pyannote.metrics merges and handles
collars differently; md-eval is the reference here because the brief names it.

**C2. 1 ms frames, exact on a millisecond grid.** All RTTMs this repo writes
are at ms precision, so every boundary is a frame edge. The scorer counts
boundaries that are not on the grid and the validation requires zero.

**C3. Pass criterion pre-registered** in `configs/validation/scorer.yaml`
(<= 0.001 s per component per file, committed before the comparison ran).

**C4. Self-overlapping speakers are an error, not merged.** md-eval aborts on
them; silently merging would make our scorer accept inputs the reference
rejects.

**C5. Validation run 1 failed the pre-registered criterion; kept on record.**
`results/scorer_validation_run1_failed/` (config 1e77d1128d): all 980
comparisons were within 0.000375 s of md-eval, but the criterion also requires
zero off-grid boundaries and 21 were reported. Cause: 21 of 34 dev+test UEMs
end at sample precision (e.g. EN2002a 2142.709375 s), which the scorer snapped
to the nearest ms; every non-zero difference equals the dropped part of a
final frame. Fix: frames cut by a UEM boundary are weighted by the fraction
inside the UEM, so the scorer is exact. The tolerance was not changed.
