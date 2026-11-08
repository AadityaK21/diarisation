# Forced-alignment AMI diarisation reference (robustness check only)

Test-set RTTMs from Horiguchi, Tawara, Ashihara, Ando and Delcroix,
"Can We Really Repurpose Multi-Speaker ASR Corpus for Speaker Diarization?",
IEEE ASRU 2025 (arXiv:2507.09226), repository
https://github.com/nttcslab-sp/diar-forced-alignment at commit
9527b7c64846fb38316a610f32e9d3466bd6d8b7, directory `AMI/test/`.

The labels were made with the Montreal Forced Aligner on the AMI only_words
setup (full-corpus ASR partition), so they cover the same words as
`data/ami/only_words/` with tight word boundaries. AMI labels are licensed
CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/); source corpus:
https://groups.inf.ed.ac.uk/ami/corpus/.

Changed here: speaker labels `<meeting>.<letter>` are renamed to AMI global
participant ids (corpusResources/meetings.xml), as in the only_words
reference; times are unchanged. Produced by `scripts/prepare_ami_fa_reference.py`
(`configs/data/ami_fa.yaml`); `MANIFEST.json` has the totals.

The brief's reference is `only_words`; this one is used only to check how
much the findings depend on the reference's boundary convention
(docs/decisions.md S2).
