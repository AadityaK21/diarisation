# third_party

`md-eval-22.pl` — NIST RT diarisation scorer, used only as the reference
implementation the scorer in `src/diarisation/scoring/` is validated against.

- Source: https://github.com/nryant/dscore/blob/e02f949ac6592279300a2c33d03daf9e0c12fd27/scorelib/md-eval-22.pl
- sha256: 872aa955cbc3d57e6d4fd6fe2e699791739c1cb716cc89105bb4717415b9400d
- This copy prints component times with `%f`, which is what makes a
  sub-millisecond comparison possible.

Semantics our scorer reproduces (read from the source, line numbers in this copy):
- speaker mapping: optimal bipartite match maximising ref/sys overlap time,
  computed on the evaluation UEM *before* collars or overlap exclusion (l. 1890–1907);
- collar: no-score zone of +/- collar around every reference SPEAKER segment
  start and end, with no merging of adjacent same-speaker segments (l. 2034–2069);
- `-1`: removes regions where >= 2 reference SPEAKER tokens are active (l. 2072–2130);
- a speaker overlapping themselves in either file is a fatal error (l. 586–597).
