# diarisation

Overlap-conditioned diarisation error and non-permutation-invariant speaker
attribution. Remote phase of a research internship with Jesper Rindom Jensen,
Audio Analysis Lab, Aalborg University (October–November 2026).

## Status

| Step | What | State |
|---|---|---|
| 1 | Repo layout | done |
| 2 | AMI + Supreme Court reference prep, per-meeting overlap table | done (`results/overlap_stats/`) |
| 3 | Frame-level scorer validated against md-eval.pl | done (`results/scorer_validation/`) |
| 4 | Overlap-conditioned decomposition | done on synthetic systems (`results/decomp_synthetic/`) |
| 5 | pyannote 3.1 baseline, full grid on AMI test | done on published outputs (`results/decomp_pyannote31/`); own CPU reproduction running, 3/16 SDM meetings scored (`results/reproduction_pyannote31/`) |
| 6a | Word attribution with true words, fixed vs optimal mapping | done (`results/attr_oracle_pyannote31/`) |
| 6b | Whisper ASR, cpWER vs fixed mappings by region (MeetEval-checked) | done (`results/attr_asr_whisper_turbo/`) |

Extra analyses (`docs/decisions.md` X1-X8, S1-S2; none replaces a step):

| | What | Where |
|---|---|---|
| X1 | Overlap detection by pyannote (speaker-count confusion) | `results/overlap_detection_pyannote31/` |
| X2 | Word error = transcription + word assignment + diarisation (DI-cpWER, cpWER, tcpWER) | `results/wer_decomposition_whisper_turbo/` |
| X3 | Which overlapped words are lost: role, turn length, level from headsets | `results/overlap_words_whisper_turbo/` |
| X4 | Per-meeting consistency of every headline effect (sign test) | `results/consistency/` |
| X5 | Supreme Court argument through the full pipeline | `results/scotus_pyannote31_whisper/` |
| X6 | large-v3 vs large-v3-turbo on a subset | `results/wer_decomposition_large_v3_subset/` |
| X7 | Which talker pyannote shows in two-speaker overlap | `results/overlap_dominance_pyannote31/` |
| X8 | WhisperX forced alignment of the Whisper words (the brief's alignment step) | `results/wer_decomposition_wxalign/` |
| S1 | NeMo streaming Sortformer, the brief's optional second system, against pyannote 3.1 | `results/compare_sortformer_pyannote/`, `results/decomp_sortformer_v21/`, `results/wer_decomposition_sortformer_v21/` |
| S2 | Robustness to the reference's boundary convention: forced-alignment AMI labels (Horiguchi et al., ASRU 2025) | `results/decomp_fa_reference/`, `results/attr_oracle_fa_reference/`, `results/wer_decomposition_fa_regions/` |

## Quick start

```bash
pip install -e ".[dev]"
cp configs/paths.example.yaml configs/paths.yaml   # edit audio_root, cache_root
make all          # tests -> reference prep -> overlap table -> scorer validation
```

Audio is only needed from step 5: `make audio-ihm audio-sdm`. It is written to
`audio_root`, outside the repo, and `*.wav` is git-ignored everywhere.

## Layout

```
configs/      one YAML per dataset and per experiment; no numbers in code
src/diarisation/
  io/         RTTM, UEM readers/writers
  intervals.py   exact interval arithmetic (no frame grid)
  data/       reference preparation (AMI, Supreme Court)
  stats/      overlap statistics, bootstrap over files
  runlog.py   config hash, git commit, versions, seed, wall time
scripts/      one entry point per Makefile target
data/         committed references: RTTM, UEM, lists, MANIFEST with sha256
results/<experiment>/   per_file.csv -> aggregate.csv, plus run.json
docs/decisions.md       every convention choice and why
report/       LaTeX progress report (jensen_report.tex/.pdf); `make report` redraws its figures and builds it
```

Aggregates are always computed from `per_file.csv`, never separately, so any
number in `aggregate.csv` can be checked by hand from the per-file rows.

## Data

- **AMI**: references from
  [pyannote/AMI-diarization-setup](https://github.com/pyannote/AMI-diarization-setup)
  at a pinned commit (Full-corpus-ASR partition, `only_words` primary,
  `word_and_vocalsounds` for the appendix). Cite Landini et al., *VBx*, CSL 2022.
- **Supreme Court (raw Oyez)**: case 2019/18-280, reference built from Oyez
  per-turn timings, CC-BY-NC. This is not a hand-corrected reference; see
  `docs/decisions.md`.
