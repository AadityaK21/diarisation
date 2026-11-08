# pyannote 3.1 outputs as published by its authors

These RTTMs are NOT produced by this repository. They are the AMI test-set
outputs of `pyannote/speaker-diarization-3.1` that the pyannote authors
publish on the model card ("reproducible_research/"), split one file per
meeting, unchanged otherwise. The `.eval` files are the authors' own
per-file scores (pyannote.metrics, "Full" setup: no collar, overlap scored).

- Source: https://huggingface.co/pyannote/speaker-diarization-3.1/tree/main/reproducible_research
- Model repository revision: 84fd25912480287da0247647c3d2b4853cb3ee5d (fetched 2026-11-08)
- sha256 of the original files:
  - AMI.SpeakerDiarization.Benchmark.test.rttm      11a7c7e8913fd20f9bb8f74411474f1c88b56490bdd422cf9564d8830c9e3050
  - AMI-SDM.SpeakerDiarization.Benchmark.test.rttm  88dcf21e6b0469eeefd8b6fb87a563977fd3d98f5c56715c64a64fd579451233
  - AMI.SpeakerDiarization.Benchmark.test.eval      ac46e261261e77918e3c05e64705822eb4513b78f7a1fec372d8ed5c3c6efd02
  - AMI-SDM.SpeakerDiarization.Benchmark.test.eval  5ee47e65376643a32c9435c4b9e207a66db0502ed63932af9c07c0db894ade4f
- `ihm_mix/` = their "AMI (headset mix, only_words)"; `sdm/` = "AMI (array1, channel 1, only_words)".

Our own run of the same pipeline lives in `../pyannote31/` and is compared
against these as a reproduction check.
