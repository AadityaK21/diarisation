# Failed ASR run 1 (kept as a record, not used for any result)

Config: `configs/systems/whisper_turbo_cpu.yaml` as of commit 2c245eb, which
set `temperature: 0.0` (a single temperature). That disables faster-whisper's
temperature fallback, the built-in guard that re-decodes a 30 s window whose
output is too repetitive (compression ratio > 2.4). With
`condition_on_previous_text` left at its default (true) a loop in one window
is fed into the next one.

What went wrong (found by inspecting the 13 finished headset-mix meetings
before scoring them):

| meeting | failure |
|---|---|
| TS3003b | "Uh." repeated 830 times from 562 s to 2171 s (27 min); 1828 words output vs 4836 reference words |
| TS3003a | "Um." repeated 75 + 59 times (1167-1476 s), no output after 1476 s |

On the 13 meetings, oracle-diarisation cpWER was 33.1 %, with 76.7 %
deletions in single-speaker speech on TS3003b alone. These numbers measure the
loop, not the task, so they are not reported anywhere.

The run was stopped at 13/16 headset-mix meetings (SDM not started). Run 2
restores the library's default temperature fallback and sets
`condition_on_previous_text: false`; see docs/decisions.md W5.
