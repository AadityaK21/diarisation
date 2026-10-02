# Each target is one step of the pipeline. `make all` goes from prepared
# references to final tables; audio download is separate because it is large.
PY ?= python3

.PHONY: all test prepare prepare-ami prepare-scotus overlap-stats validate-scorer decompose-synthetic audio-ihm audio-sdm

all: test prepare overlap-stats validate-scorer decompose-synthetic

test:
	$(PY) -m pytest -q

prepare: prepare-ami prepare-scotus

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
