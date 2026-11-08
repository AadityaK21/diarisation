# Each target is one step of the pipeline. `make all` goes from prepared
# references to final tables; audio download is separate because it is large.
PY ?= python3

.PHONY: all test prepare prepare-ami prepare-scotus overlap-stats validate-scorer decompose-synthetic diarise validate-scorer-real decompose-pyannote31 compare-mappings prepare-words attribution-oracle attribution-asr audio-ihm audio-sdm

all: test prepare overlap-stats validate-scorer decompose-synthetic validate-scorer-real decompose-pyannote31 compare-mappings attribution-oracle

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
