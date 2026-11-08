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
| 5 | pyannote 3.1 baseline, full grid on AMI test | done on published outputs (`results/decomp_pyannote31/`); own CPU run in progress |
| 6 | ASR, word alignment, cpWER and fixed-mapping attribution | not started |

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
