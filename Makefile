# Each target is one step of the pipeline. `make all` goes from prepared
# references to final tables; audio download is separate because it is large.
PY ?= python3

.PHONY: all test prepare prepare-ami prepare-scotus overlap-stats validate-scorer decompose-synthetic diarise validate-scorer-real decompose-pyannote31 compare-mappings prepare-words attribution-oracle attribution-asr audio-ihm audio-sdm audio-headsets audio-scotus transcribe asr-diagnostics overlap-detection wer-decomposition overlap-words consistency scotus reproduction-check sortformer sortformer-eval extra

all: test prepare overlap-stats validate-scorer decompose-synthetic validate-scorer-real decompose-pyannote31 compare-mappings attribution-oracle attribution-asr asr-diagnostics overlap-detection wer-decomposition consistency

test:
	$(PY) -m pytest -q

prepare: prepare-ami prepare-scotus prepare-words

prepare-ami:
	$(PY) scripts/prepare_ami.py

prepare-scotus:
	$(PY) scripts/prepare_scotus_oyez.py

overlap-stats:
	$(PY) scripts/overlap_stats.py

# Audio (not needed until step 5). Lands under audio_root in configs/paths.yaml.
audio-ihm:
	$(PY) scripts/download_ami_audio.py --condition ihm_mix --partitions dev test

audio-sdm:
	$(PY) scripts/download_ami_audio.py --condition sdm --partitions dev test

validate-scorer:
	$(PY) scripts/validate_scorer.py

decompose-synthetic:
	$(PY) scripts/decompose.py --config configs/experiments/decomp_synthetic.yaml

# Step 5. HF_TOKEN must be set (or injected by a proxy) and the model terms accepted.
diarise:
	$(PY) scripts/diarise.py --config configs/systems/pyannote31.yaml

# Scores whichever meetings of our own CPU run are finished (decisions P2).
reproduction-check:
	$(PY) scripts/reproduction_check.py

validate-scorer-real:
	$(PY) scripts/validate_scorer.py --config configs/validation/scorer_real.yaml

decompose-pyannote31:
	$(PY) scripts/decompose.py --config configs/experiments/decomp_pyannote31.yaml

compare-mappings:
	$(PY) scripts/decompose.py --config configs/experiments/decomp_pyannote31_singlemap.yaml
	$(PY) scripts/compare_mappings.py

prepare-words:
	$(PY) scripts/prepare_ami_words.py

attribution-oracle:
	$(PY) scripts/attribution_oracle.py

attribution-asr:
	$(PY) scripts/attribution_asr.py

# Step 6b ASR (CPU, hours; needs audio). Outputs are committed, so the scoring
# targets below run without audio.
transcribe:
	$(PY) scripts/transcribe.py --config configs/systems/whisper_turbo_cpu.yaml

asr-diagnostics:
	$(PY) scripts/asr_diagnostics.py

# Extra analyses (docs/decisions.md X1-X6)
overlap-detection:
	$(PY) scripts/overlap_detection.py

wer-decomposition:
	$(PY) scripts/wer_decomposition.py

consistency:
	$(PY) scripts/consistency.py

# needs the individual headset channels (audio-headsets)
audio-headsets:
	$(PY) scripts/download_ami_headsets.py --partitions test

overlap-words:
	$(PY) scripts/overlap_words.py

# Supreme Court set through the pipeline (audio-scotus first; diarise and
# transcribe run on CPU)
audio-scotus:
	$(PY) scripts/prepare_scotus_audio.py

scotus:
	$(PY) scripts/diarise.py --config configs/systems/pyannote31.yaml --dataset configs/data/scotus_oyez.yaml --conditions oyez --partitions all
	$(PY) scripts/transcribe.py --config configs/systems/whisper_turbo_cpu.yaml --dataset configs/data/scotus_oyez.yaml --conditions oyez --partitions all
	$(PY) scripts/scotus_eval.py

# NeMo streaming Sortformer, the optional second diariser (decisions S1).
# Diarisation runs in venv-nemo (env/requirements-nemo-cpu.txt): PY_NEMO.
PY_NEMO ?= $(PY)
sortformer:
	$(PY_NEMO) scripts/diarise.py --config configs/systems/sortformer_v21.yaml
	$(PY_NEMO) scripts/sortformer_postprocess.py --config configs/systems/sortformer_v21_dihard3pp.yaml

sortformer-eval:
	$(PY) scripts/decompose.py --config configs/experiments/decomp_sortformer_v21.yaml
	$(PY) scripts/overlap_detection.py --config configs/experiments/overlap_detection_sortformer_v21.yaml
	$(PY) scripts/attribution_oracle.py --config configs/experiments/attr_oracle_sortformer_v21.yaml
	$(PY) scripts/attribution_asr.py --config configs/experiments/attr_asr_sortformer_v21.yaml
	$(PY) scripts/wer_decomposition.py --config configs/experiments/wer_decomposition_sortformer_v21.yaml
	$(PY) scripts/overlap_dominance.py --config configs/experiments/overlap_dominance_sortformer_v21.yaml
	$(PY) scripts/miss_by_word.py
	$(PY) scripts/compare_systems.py

extra: overlap-detection wer-decomposition overlap-words consistency
