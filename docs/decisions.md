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

## Decomposition (step 4)

**R1. Regions are defined by the reference speaker count, before collars.**
nonspeech (0), single (1), overlap (>= 2). A hypothesis that puts two
speakers where the reference has one is charged as false alarm in the
*single* region; system-side overlap does not define a region. Collars and
overlap exclusion then remove frames within each region, so the regions
partition exactly the frames total DER scores; every run checks that the
three regions sum to the total within 1e-6 s or aborts.

**R2. The 2x2x2 grid has structurally empty cells.** With overlap excluded,
the overlap region scores nothing; it is reported as NA, never as 0%. False
alarm in non-speech has no within-region rate (no reference speaker time);
it is reported in seconds and as a contribution to total DER.

**R3. Three ways to normalise a region's error, all reported:**
within-region rate (error in overlap / speaker time in overlap: how bad the
system is there), share of total error (error in overlap / all error: how
much of the error overlap causes), and contribution to DER (error in overlap
/ all scored speaker time: these sum to DER across regions).

**R4. Differences between conventions use a paired bootstrap.** The same
resampled meetings are used for both terms, so "error concealed by the
forgiving convention" (full DER minus forgiving DER) gets a proper interval.

**F7. The 0.25 s collar removes about half of all overlap from scoring
[Jensen].** Reference-only property (identical for every system), AMI test,
`only_words`: 50.4% [40.7, 56.4] of overlap speaker time survives the collar
(run c52c63421a, `results/decomp_synthetic/headline.csv`,
`overlap_time_kept_by_collar`). So the "Fair" convention, which nominally
scores overlap, scores only half of it.

**F8. "Excluding overlap" does not by itself lower DER.** It removes error
and denominator together. It conceals error only to the extent a system is
worse in overlap than outside it. Synthetic check: `swap` (uniform confusion)
loses nothing (-0.02 pp [-0.70, 0.66]); `exclusive` (perfect except in
overlap) goes from 14.55% to 0.00%. The headline claim should be stated per
system with this paired interval, not as a general offset.

## Real system (step 5)

**P1. Main results use the pyannote authors' published AMI test outputs.**
`results/diarise/pyannote31_published/` (model card "reproducible_research",
revision 84fd259). Our scorer gives 18.80% (IHM-Mix) and 22.45% (SDM) under
Full, matching the model card's 18.8% and 22.4%; per meeting, our components
agree with the authors' pyannote.metrics `.eval` within 0.042 s (0.005 DER
points). md-eval agrees with our scorer on all 132 real-output comparisons
(run 1c8169bddf).

**P2. Our own run is a reproduction check, not the source of the numbers.**
pyannote.audio 3.3.2 on CPU (env/constraints-pyannote31-cpu.txt) runs at
RTF 0.78 on 4 cores, about 14 h for both test conditions. On IS1009a (SDM)
our output scores 21.93% vs the published 23.01%, and the two outputs differ
on 3.4% of speaker time (mostly cluster labels). Close, not identical:
different library version and hardware. Remaining meetings run in the
background and are added as they finish.

**P3. Fixing the mapping on single-speaker frames changes almost nothing
[Jensen].** Mapping changed in 1 of 32 file/condition pairs (TS3003a,
IHM-Mix); pooled DER moves by 0.002 points (IHM) and 0.000 (SDM), and DER
inside overlap by at most 0.006 points (`results/mapping_comparison/`). For
this system, the permutation-invariant and fixed-mapping scores coincide at
frame level, so the brief's expected headline gap does not appear here. The
overlap error is missed speech (25.0% IHM, 27.5% SDM of overlap speaker time)
and confusion (13.9%, 16.1%), not inconsistent labelling. Step 6 should
therefore measure what happens to the MISSED speaker's words (credited to
the speaker who was detected, or lost), rather than mapping instability.

## Attribution with reference words (step 6a)

**A1. Reference words come from the AMI manual annotations 1.6.2** (NXT, CC BY
4.0, sha256 pinned in configs/data/ami_words.yaml): `<w>` elements without
punctuation, speaker ids via corpusResources/meetings.xml. 99.998% of test
word midpoints fall inside their own speaker's only_words RTTM turns.

**A2. Word -> speaker = the system speaker overlapping the word most**, as the
brief specifies; no nearest-turn fallback (a word with no system speech is
reported as unassigned). A word is "in overlap" when >= 2 reference speakers
are active at its midpoint. Ties in overlap duration go to the label sorting
first, which is what any one-speaker-per-word rule must do in some way.

**A3. The one-speaker-per-word rule is itself a large error source [Jensen].**
With the reference RTTM as the "system", 48.1% [46.2, 49.9] of overlapped
words are credited to the other talker: a coin flip. This bounds every
timestamp-based word attribution (WhisperX-style) on AMI at about 11.8% of
all words before any diarisation error (run 837370d52e).

**A4. The brief's metric B is more lenient than the permutation-invariant
score, not stricter [Jensen].** Majority mapping on single-speaker frames is
many-to-one, so it absorbs clusters that split one person (the "unmapped"
outcome under the one-to-one mapping: 3.2% of words IHM, 5.1% SDM). Overall
attribution error is 1.4 points (IHM) and 2.4 points (SDM) LOWER under
metric B; in overlap 4.0 and 5.8 points lower. Reported as measured; the
claim that a fixed mapping reveals hidden error is not supported here.

## Attribution with real ASR (step 6b)

**W1. ASR model deviates from the brief [Jensen].** large-v3 int8 on this
machine's 4 CPU cores runs at 1.61x real time (about 15 h per condition);
large-v3-turbo runs at 0.32-0.40x. The CPU run uses large-v3-turbo; on the
8 GB GPU the config switches to large-v3 int8_float16 in one line.

**W2. Word timestamps come from faster-whisper's cross-attention alignment**
(the method of openai-whisper and whisper-timestamped), not WhisperX forced
alignment. The brief names WhisperX or whisper-timestamped; WhisperX is not
used yet.

**W3. Text normalisation.** Whisper's EnglishTextNormalizer, applied per word
to reference and output alike (fillers removed, contractions expanded).
Per-word application does not join number words; it affects every compared
system equally.

**W4. cpWER is computed in-house and checked against MeetEval 0.4.3** on every
file (errors and reference length must match exactly, else the run aborts).
The in-house version is needed to assign each error to a region: deletions
and substitutions to the region of the reference word, insertions to the
region at the output word's midpoint.

**A5. The coin-flip result is robust to the tie-break.** Overlapped-word
attribution error with perfect diarisation: 48.1% (label order, default),
52.1% (credit the speaker whose turn started first), 38.6% (credit the one
whose turn started last). pyannote 3.1: 54.1-59.7% across the three rules
and both conditions (runs 837370d52e, 7e0f98a7a4, c80a92aca6).
