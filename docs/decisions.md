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
background and are added as they finish; `scripts/reproduction_check.py`
(`configs/experiments/reproduction_pyannote31.yaml`) scores whichever
meetings are done. With IS1009a and IS1009b (SDM; run 5152011043): pooled
Full DER 16.84% ours vs 16.81% published, and the outputs differ on 2.4% of
the published speaker time, 2.0 points of it speaker confusion (cluster
labels), 0.2 missed and 0.2 false alarm. With IS1009c added (3 meetings):
14.81% ours vs 14.70% published, outputs differing on 1.95% of speaker time
(1.6 points confusion). Timing is not clean from IS1009b on:
the run shared the 4 cores with other jobs (and was paused while the
alignment of X8 finished), so wall times in its timing.csv are upper bounds.

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
With the reference RTTM as the "system", 47.2% [45.2, 49.0] of overlapped
words are credited to the other talker: a coin flip. This bounds every
timestamp-based word attribution (WhisperX-style) on AMI at about 11.6% of
all words before any diarisation error (run 8095f4fd05; see A6 for the
earlier 48.1% / 11.8%).

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
alignment. The brief names WhisperX or whisper-timestamped. WhisperX forced
alignment was later run on all 16 headset-mix meetings as a check (X8); it
changes each part of the decomposition by less than half a point over all
words and by at most 1.2 points in overlap (the diarisation part there goes
from 1.0 to 2.2 points), so the main results keep Whisper's own timestamps.

**W3. Text normalisation.** Whisper's EnglishTextNormalizer, applied per word
to reference and output alike (fillers removed, contractions expanded).
Per-word application does not join number words; it affects every compared
system equally.

**W4. cpWER is computed in-house and checked against MeetEval 0.4.3** on every
file (errors and reference length must match exactly, else the run aborts).
The in-house version is needed to assign each error to a region: deletions
and substitutions to the region of the reference word, insertions to the
region at the output word's midpoint. In 6b the "single" region means
"outside overlap": an insertion whose midpoint falls in silence is counted
there too (`RegionLookup` returns overlap for >= 2 active speakers, single
otherwise).

**W5. ASR run 1 failed; decoding guards restored for run 2.** Run 1 set a
single `temperature: 0.0` for reproducibility. That switched off Whisper's
temperature fallback, which re-decodes a window whose output is too
repetitive (compression ratio > 2.4), and with `condition_on_previous_text`
true a loop carried from window to window: TS3003b came out as "Uh." 830
times over 27 minutes, TS3003a had two "Um." loops and no output after
1476 s. Caught by inspecting the outputs before any result was reported; run 1
is kept in `results/asr/whisper_turbo_cpu_run1_failed/` and is not used. Run 2
uses the library defaults (temperatures 0.0-1.0 in steps of 0.2,
compression-ratio 2.4, log-prob -1.0, no-speech 0.6) plus
`condition_on_previous_text: false` (WhisperX's default), with the fallback
sampler seeded. The settings were chosen from those defaults, not tuned on
test WER; the only check on test data was that TS3003b no longer loops.
`transcribe.py` now records the longest same-word run per meeting
(`longest_repeat` in timing.csv) so a loop cannot pass unnoticed.

**W6. AMI acronyms are joined before normalisation.** AMI spells acronyms
letter by letter (`L_C_D_`, `T_V_`, `X_M_L_`); Whisper writes `LCD`. Left
alone, each acronym is two or three reference tokens against one output
token. On the test set that is 377 words, 889 tokens (1.0 % of reference
tokens). `join_spelled` removes the underscores before Whisper's normaliser,
for reference and output alike. Effect on headset-mix oracle-diarisation
cpWER: 32.07 % -> 31.47 %. Multi-word numbers ("twenty five" vs "25") are not
joined (per-word normalisation, W3); about 0.1 % of output tokens are affected.

**W7. Extra 6b variant: one-to-one fixed mapping [Jensen].** Not in the
brief. System speakers are mapped once, one-to-one, by maximum overlap on
single-speaker frames; extra system speakers count as insertions. It separates
the two things metric B changes relative to cpWER: (i) no search over
permutations, (ii) many-to-one instead of one-to-one. By construction it is
never below cpWER (the script aborts if it is). Headset-mix result: it equals
cpWER to +0.01 points [+0.00, +0.04], while the many-to-one metric B is 2.09
points lower [-3.20, -1.05] (5.84 lower in overlap). So the leniency of metric
B comes entirely from many-to-one mapping, which forgives over-clustering:
pyannote 3.1 outputs 4-9 speakers for meetings with 3-4 people (96 system
speakers for 63 people over the 16 test meetings, headset mix; 34 of them
are mapped to a person who already has one). Fixing the mapping without
a permutation search changes nothing. Paired cpWER contrasts (pyannote minus
oracle diarisation, same ASR words) are in `contrasts.csv`.

**A5. The coin-flip result is robust to the tie-break.** Overlapped-word
attribution error with perfect diarisation: 47.2% (label order, default),
52.1% (credit the speaker whose turn started first), 38.6% (credit the one
whose turn started last). pyannote 3.1: 54.1-59.7% across the three rules
and both conditions (runs 8095f4fd05, 7e0f98a7a4, c80a92aca6).

**A6. Correction: the default tie-break run was stale.** The first 6a run
(837370d52e) used exact floating-point equality to detect ties in overlap
duration, so two speakers covering a word equally could be split by rounding
noise. The tie-break commit (d5d85cf) changed this to a 1e-9 s tolerance,
and the two tie-break variant runs used it, but the default run was not
redone. Rerun with the current code (8095f4fd05, which also adds the
one-to-one variant, W7): oracle-diarisation overlap attribution error 48.12%
-> 47.22%, all words 11.83% -> 11.61%; every pyannote number moves by less
than 0.01 points. The three tie-break rules are now on the same code.

## Extra analyses (beyond the brief's steps; none replaces a step)

**X1. Overlap detection by pyannote 3.1.** Speaker-count confusion between
reference and system, seconds of UEM time, no collar
(`scripts/overlap_detection.py`). Headset mix: the system shows two speakers
for 63.7% [59.4, 66.6] of the reference overlap time (recall) and 86.4% of
its own two-speaker time is real overlap (precision); 35.3% of overlap time
is shown as one speaker. Distant mic: recall 58.7%, precision 85.0%. By
design the segmentation model outputs at most 2 speakers per frame and 3 per
10 s chunk (model card, huggingface.co/pyannote/segmentation-3.0:
`max_speakers_per_chunk, max_speakers_per_frame = 3, 2`); 600 s of the
3,827 s of reference overlap (headset mix) has 3 or more people talking and
can at best be shown as 2.

**X2. Where speaker-attributed word errors come from.** Three nested scores
on the same Whisper words (`scripts/wer_decomposition.py`): DI-cpWER
(MeetEval's diarisation-invariant cpWER, greedy; every output word may go to
whichever person minimises the error), so it is transcription only; cpWER
with the word rule on the reference turns (perfect diarisation); cpWER with
pyannote's turns. Differences give the word-assignment part and the
diarisation part, paired per meeting. MeetEval's greedy search starts from
the cpWER assignment of the hypothesis labels; started from a single label
it stalls far above cpWER (52% vs 26% on IS1009a), so it is run from the
perfect-diarisation and the pyannote labelling and the lower result kept
(both are upper bounds on the exact DI-cpWER, so the transcription part may
be slightly overstated and the assignment part understated). The two starts
agree closely: 24.23% vs 24.45% over the headset-mix test set, at most 40
errors apart in any meeting. Region-split
counts are checked against MeetEval's totals on every file. tcpWER
(MeetEval) is reported at a 5 s collar, the collar used to rank CHiME-8 DASR
(chimechallenge.org/challenges/chime8/task1) and in the MeetEval paper
(arXiv:2307.11394). Headset mix, pyannote: 33.4% cpWER = 24.2 transcription
+ 7.3 word assignment + 2.0 diarisation; in overlap 78.1 = 54.5 + 22.6 + 1.0.
The word-assignment part exceeds the diarisation part in 16 of 16 meetings.
Distant mic: 39.8 = 29.9 + 6.3 + 3.5; in overlap 81.6 = 59.2 + 18.5 + 3.9;
word assignment exceeds diarisation in 13 of 16 meetings (sign test p=0.02).
tcpWER (5 s) is 1.1 (headset) and 1.2 (distant) points above cpWER.

**X3. Which overlapped words are lost.** Role (whose current turn started
first), turn length (1-2 words vs longer) and relative level, 10 log10 of
the speaker's own headset energy over the summed headset energy of the other
active speakers during the word (`scripts/overlap_words.py`). Headset
channels per speaker come from AMI's meetings.xml; the 63 headset files of
the test meetings are downloaded outside the repository
(`scripts/download_ami_headsets.py`). Crosstalk limits the measure: on
single-speaker words the same ratio (against all other headsets) has median
+20.7 dB (10th-90th percentile 6.2 to 29.2 dB), so levels above about +20 dB
are not resolved. Outcomes: deletion under the DI-cpWER alignment (per
token) and attribution with perfect diarisation (per word, as step 6a). In
overlap, 76% of the words in 1-2-word turns are backchannel-type words
(yeah 834, mm-hmm 322, okay 241, mm 170, yes 103, hmm 101, right 88, ... of
2,832); Whisper's normaliser removes the fillers among them (mm-hmm, mm,
hmm, um, uh), so the ASR outcome covers the rest. The level effect is not
a by-product of backchannels or turn order: it holds within turns longer
than two words (deletion 64% below -10 dB, 32% above +10 dB) and for the
first and the second talker separately (59% to 25%; 70% to 42%); the
`length|level` and `role|level` groups in aggregate.csv give the intervals.
Distant mic (Whisper on the far-field channel, level still from the
headsets): deletion 63.9% below -10 dB and 43.0% above +10 dB, 16.2% outside
overlap; the below-0 vs above-0 difference holds in 14 of 16 meetings.

**X4. Per-meeting consistency.** For each headline effect, the number of
the 16 test meetings in which it holds and an exact two-sided sign test
(`scripts/consistency.py`, `results/consistency/claims.csv`).

**X5. Supreme Court set through the full pipeline.** Audio: the Oyez
delivery mp3 of the 2 December 2019 argument in 2019/18-280 (sha256
pinned in configs/data/scotus_oyez.yaml), converted with ffmpeg to 16 kHz
mono WAV under audio_root. Reference: the existing raw Oyez turns (no
overlap, no silence between turns, not hand-corrected; unchanged, and
re-parsed from today's Oyez JSON they are identical) and reference words
from the Oyez text blocks with pseudo timings (each word an equal share of
its block; one block with stop 0 takes its turn's times). DER is therefore
for orientation only and the overlap split does not apply; the point is
speaker attribution with 10 speakers, where a merged pair of people cannot be
undone by a many-to-one mapping. Same pyannote and Whisper settings as AMI.
Result (run 30d658beeb): pyannote finds 10 clusters, one per person (each
94-98% one person); DER 9.93% Full (missed 7.43, false alarm 1.36,
confusion 1.14) and 8.34% Fair, md-eval-checked; the reference marks all
3,743 s as speech while the system marks 3,517 s, so the missed speech
mostly reflects the reference. DI-cpWER 5.14%, cpWER 7.46% with the Oyez
turns and 7.79% with pyannote; both fixed mappings equal cpWER, because
with one cluster per person there is nothing for many-to-one to absorb.

**X6. Effect of the turbo deviation (W1).** large-v3, the brief's model,
with exactly the turbo run's decoding settings, on CPU for a subset of test
meetings (IS1009a, ES2004a; headset mix), compared with turbo on the same
meetings. A subset check only: it bounds the size of the effect, it does not
replace a full large-v3 run. Result (run 4d07a47080,
`results/wer_decomposition_large_v3_subset/`; 4,647 reference words, 1,102
in overlap; pooled over the two meetings, turbo -> large-v3):

| measure | turbo | large-v3 |
|---|---|---|
| DI-cpWER (transcription) | 21.67 | 20.79 |
| cpWER, oracle diarisation | 27.61 | 27.61 |
| cpWER, pyannote 3.1 | 29.93 | 29.57 |
| word-assignment part | 5.94 | 6.82 |
| diarisation part | 2.32 | 1.96 |
| DI-cpWER in overlap | 51.00 | 48.55 |
| word-assignment part in overlap | 17.70 | 18.60 |

The equal oracle cpWER is a coincidence of totals (1,283 errors each); per
meeting the two models differ in opposite directions: on IS1009a large-v3 is
worse (cpWER pyannote 30.27 vs 29.91, oracle 27.21 vs 25.98), on ES2004a
better (29.06 vs 29.95, oracle 27.90 vs 28.79). Over all words no
per-meeting difference exceeds 2.2 points (largest: ES2004a DI-cpWER, 23.74
vs 21.61); inside overlap, with 473 and 629 words per meeting, differences
reach 4.0 points (ES2004a DI-cpWER, 51.83 vs 47.85) and again go both ways
(pyannote cpWER in overlap: IS1009a +2.5, ES2004a -2.9). The shape of the
decomposition is the same for both models: word assignment costs more than
diarisation, most of all in overlap.
With n = 2 the bootstrap intervals in `aggregate.csv` mean little; the claim
is only that the turbo deviation does not change the picture on these two
meetings. large-v3 ran at RTF 0.82-0.86 on 4 CPU cores against turbo's
0.19-0.20 on the same meetings; the large-v3 run shared the CPU with the
alignment job (X8), so its time is an upper bound.

**X7. Which talker pyannote shows when it shows one.** Sampled every 0.1 s
inside two-speaker reference overlap; system speakers mapped to people with
the optimal mapping; "louder" = higher headset energy in a 0.1 s window
(`scripts/overlap_dominance.py`). When pyannote shows exactly one of the two
talkers, it is the louder one 71.2% [68.8, 73.3] of the time on the headset
mix and 66.3% [62.3, 69.8] on the distant mic. Whisper also loses the quieter
talker (X3), so diarisation and transcription fail on the same person, which
is consistent with diarisation adding only 1.0 point [-0.7, 2.1] of cpWER in
overlap (X2).

**X8. WhisperX forced alignment (the brief's alignment step, W2).** The
words of the turbo run are re-timed with WhisperX's wav2vec2 CTC alignment
(whisperx 3.8.6, `WAV2VEC2_ASR_BASE_960H`; `scripts/align_whisperx.py`);
text and word order are unchanged (checked). Only WhisperX's alignment
module is used, installed without its dependencies in its own environment,
because the full package pins torch 2.8 and pyannote.audio 4. WhisperX aligns
inside Whisper's segments; the word files do not keep faster-whisper's
segments, so words are regrouped at gaps longer than 1 s (segments at most
30 s), which is how the alignment windows are formed here. NLTK's
sentence data (punkt_tab) was fetched from the NLTK data repository by hand,
since NLTK's downloader refuses to run through this environment's proxy.
The decomposition (X2) is then recomputed on both timings
(`configs/experiments/wer_decomposition_wxalign.yaml`). A first run aligned
all segments of a meeting in one call and stopped at EN2002b, where WhisperX
failed on a one-word segment ("Cool.": "backtrack failed, resorting to
original") and returned one word fewer; the script now aligns one segment at
a time and keeps the original timings of a segment WhisperX cannot align,
counting such segments (`n_segments_failed` in timing.csv). All 16 meetings
were rerun this way: 2,451 of 2,453 segments aligned (68,879 words); the two
that failed are one-word segments in EN2002b and EN2002c.

Result (run c1e3610f8a, `results/wer_decomposition_wxalign/`; differences are
WhisperX minus Whisper's own timestamps, paired bootstrap over the 16
meetings, `contrasts.csv`; headset mix, pyannote 3.1):

| | all words | single-speaker | overlap |
|---|---|---|---|
| word-assignment part | -0.13 [-0.40, 0.09] | -0.29 [-0.54, -0.06] | +0.36 [-0.08, 0.69] |
| diarisation part | +0.43 [0.22, 0.65] | +0.18 [0.05, 0.32] | +1.19 [0.60, 1.95] |
| cpWER, pyannote | +0.24 [0.04, 0.44] | -0.09 [-0.26, 0.06] | +1.27 [0.69, 1.99] |

With WhisperX timings the split is 24.1 + 7.2 + 2.4 = 33.7 (Whisper's
timings: 24.2 + 7.3 + 2.0 = 33.4). Better word timing helps the word rule
only outside overlap (0.3 points); in overlap, where its cost sits (22.6
points), it does nothing, which shows that cost comes from giving each word
to one speaker, not from imprecise word times. With pyannote, WhisperX timing
is slightly worse overall (+0.24; worse in 8 meetings, better in 6, equal in
2) and clearly worse in overlap (+1.27, worse in 14 of 16); the diarisation
part grows in 14 of 16. Why is not established here; one possibility is that
Whisper's own, smoother word times sit better inside pyannote's turns. Either
way the deviation from the brief's alignment step (W2) does not change any
finding.

## Second system and second reference

**S1. NeMo streaming Sortformer, the brief's optional NeMo system.**
`nvidia/diar_streaming_sortformer_4spk-v2.1` (NeMo 2.7.3, CPU, Hugging Face
revision cd03eee), an end-to-end model with up to 4 speakers per frame
(AMI test meetings have 3-4 people), run as published: the model card's
30.4 s streaming setting, the model's default post-processing, nothing tuned
on AMI (`configs/systems/sortformer_v21.yaml`). Chosen over NeMo's older
clustering + MSDD pipeline as NeMo's current diariser; flag if the brief meant
MSDD. Frame probabilities are saved, so NVIDIA's published post-processing for
this model family (tuned on DIHARD III dev; none exists for AMI) is applied as
a sensitivity variant: it changes speaker time by about 2%, lowers Full DER
by 1.1-1.5 points overall (0.7-1.6 by region) and changes cpWER by less than 0.5. RTF 0.03-0.04 on 4 cores; 4 speakers found in
every meeting. Not run on the Supreme Court argument (10 speakers > 4).

Against the brief's reference (only_words), Full DER is 29.42% [26.09, 32.46]
(headset mix) and 32.05% (distant mic), against pyannote's 18.80% and 22.45%;
nearly all of the gap is missed speech outside overlap (20.5% of
single-speaker time vs 3.8%). Speaker confusion is lower than pyannote's
(2.8% vs 5.7% of speaker time; in overlap 2.8% vs 13.9%). Sortformer detects
less overlap than pyannote (recall 42.6% vs 63.7%, precision 89.4% vs 86.4%),
so it is not more overlap-aware on AMI in practice. Its missed speech
concentrates in reference words longer than 0.8 s (51.9% of their time vs
10.7% for pyannote), which AMI's manual word times stretch over pauses; see
S2.

With the true words (step 6a), Sortformer credits 45.3% [43.5, 47.3] of
overlapped words to the wrong person on the headset mix (pyannote 57.4%,
perfect diarisation 47.2%) and 47.3% on the distant mic (pyannote 59.7%). The
fixed many-to-one mapping (metric B) is lenient only by 0.1 points [0.0,
0.4] (headset) and 0.9 [0.0, 2.1] (distant) for Sortformer, against 1.4
[0.7, 2.2] and 2.4 [1.3, 3.5] for pyannote: a system
that cannot over-cluster leaves little to forgive, as on the Supreme Court
argument (X5). With Whisper's words (step 6b), cpWER is 32.66% (headset) and
39.74% (distant) against pyannote's 33.44% and 39.79%: no difference overall
(paired -0.78 [-4.55, 4.55] and -0.05 [-4.32, 5.25]), but 10-11 points lower
in overlap (67.71% vs 78.06%, paired -10.35 [-13.48, -5.67]; distant 70.31%
vs 81.65%, -11.34 [-13.84, -7.22]) and 2-4 points higher outside it. Metric B
minus cpWER is +0.24 [-0.09, 0.89] and -0.60 [-2.14, 0.49] for Sortformer
(pyannote -2.09 and -2.93).

Per meeting (`results/compare_sortformer_pyannote/contrasts.csv`, exact sign
test): Sortformer's cpWER in overlap is lower in 14 of 16 meetings (headset,
p = 0.004) and 13 of 16 (distant, p = 0.02); its speaker confusion in overlap
is lower in 14 of 16 on both; its DER against the brief's reference is higher
in all 16 on both. Overall cpWER is lower in 12 and 10 of 16 (not
significant). Why it does better in overlap (X7 rerun,
`results/overlap_dominance_sortformer_v21/`): when it shows one of two
overlapping talkers it is the louder one 87.2% [85.0, 89.1] of the time
(headset; distant 84.5% [80.9, 87.6]; pyannote 71.2% and 66.3%), and it shows
both only 40.6% of the time (pyannote 60.1%). Whisper keeps the louder talker
(X3), so Sortformer's one speaker is usually the one whose words are in the
transcript. It also shows three or more speakers for 14.4% of the reference's
three-or-more-speaker time (pyannote, capped at two: 0.0%).

**S2. The reference's boundary convention [Jensen].** The brief's AMI
reference (only_words, BUT/pyannote setup) takes speech time from AMI's
manual word times, which are loose. Horiguchi et al. (IEEE ASRU 2025,
arXiv:2507.09226) show that such loose boundaries change DER and that models
learn the looseness of their training references; they publish
forced-alignment (MFA) labels for the same AMI only_words setup, which is
what Sortformer is trained and evaluated on. Those labels (CC BY 4.0;
`data/ami/forced_alignment/`, `configs/data/ami_fa.yaml`) are used here as a
second reference, for checks only: same 16 meetings, same words, same UEM.
They have 22% less speaker time (23,931 s vs 30,714 s) and 44% less
overlapped speaker time (4,628 s, 19.3% of speaker time, vs 8,296 s, 27.0%);
three-or-more-speaker time falls from 6.0% to 3.1% of speaker time.

Against them the ranking of the two systems flips (Full DER, headset / distant):
Sortformer 15.94% [13.49, 18.89] / 19.54%, pyannote 36.22% [31.97, 40.32] /
38.41%. Sortformer's 15.94% matches its model card (15.90% headset; 17.80%
distant), which checks this run. pyannote's error against them is mostly
false alarm in the pauses the forced alignment excludes (24.8% of speaker
time); with the 0.25 s collar it falls to 14.85% (Sortformer 6.94%). What
this changes in the findings:

- Finding "overlap carries most of the DER" (P1, F-series) holds for the
  brief's reference only: against forced alignment, 19.3% of pyannote's error
  is in overlap, its share of the time. DER per second stays far higher in
  overlap under both references (pyannote 38.9% vs 8.2% only_words; 36.2% vs
  10.9% forced alignment).
- The coin flip of the word rule (A3) is robust: with forced-alignment turns
  as perfect diarisation, 43.3% [41.9, 44.5] of overlapped words are credited
  to the wrong person (47.2% with the only_words turns;
  `results/attr_oracle_fa_reference/`).
- The split of cpWER into word assignment and diarisation (X2) depends on
  which turns define perfect diarisation. With forced-alignment turns,
  cpWER is 29.63% (headset) and 34.91% (distant) instead of 31.47% and
  36.25%, and 67.89% in overlap instead of 77.05%. The headset-mix split
  becomes 23.8 + 5.8 + 3.8 (transcription, word assignment, diarisation)
  instead of 23.8 + 7.7 + 2.0 in the same run, and 52.8 + 15.1 + 10.2 in
  overlap; on the distant mic 29.5 + 5.4 + 4.9. So word assignment still
  costs more than diarisation, clearly on the headset mix and narrowly on the
  distant mic, but pyannote's cost in overlap is about 10 points against
  tight turns, not 1. Sortformer reaches the tight-turn level in overlap
  (67.71% vs 67.89% headset; 70.31% vs 70.07% distant): its advantage there
  comes from tight speech activity, not from showing more overlap. (DI-cpWER
  in this run is the lowest over five labellings, 23.81% headset, slightly
  below the pyannote-only run's 24.19%.)
- Main results stay on the brief's reference, as planned; whether to move
  them to the forced-alignment labels is a question for Jensen. Word-level
  scores (cpWER, the attribution measures) depend much less on the convention
  than DER: across the two references the cpWER ranking of the systems does
  not change in any region. With the forced-alignment labels defining the
  regions too (`results/wer_decomposition_fa_regions/`; 15,237 overlapped
  reference words instead of 22,105), Sortformer's cpWER in overlap is still
  lower than pyannote's: 71.21% vs 78.23% (paired -7.02 [-9.24, -3.83];
  distant 72.26% vs 81.75%, -9.48 [-11.75, -5.99]), and lower than with the
  forced-alignment turns themselves (73.35%; part -2.15 [-3.19, -0.69]),
  consistent with Sortformer showing the dominant talker, the one Whisper
  transcribes.
