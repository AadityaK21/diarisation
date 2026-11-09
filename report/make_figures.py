"""Figures for report/jensen_report.tex, drawn from the committed result CSVs.

Run from anywhere with the diarisation environment (it has matplotlib):

    /home/user/venv-diar/bin/python report/make_figures.py

Writes report/figures/fig1..fig5 as vector PDFs sized to the LaTeX text
width (16 cm), so the figures are included at 100% and their text matches
the body size. Every printed number is formatted from the unrounded value in
the CSV, never from an already-rounded one, and the numbers the report's
text and captions state about the figures are asserted at the end."""

import csv
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"
OUT = Path(__file__).resolve().parent / "figures"

W = 6.3  # inches: \textwidth of the A4 layout with 2.5 cm margins

# Reference palette of the dataviz guidelines: ink, secondary ink, muted,
# gridline; slots 1 (blue) and 2 (orange) pass the palette validator.
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE = "#2a78d6", "#eb6834"
GREY, GREY_LIGHT, GREY_DARK = "#b4b2a9", "#e9e8e2", "#3b3a37"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "text.color": INK,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "axes.linewidth": 0.7,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "legend.frameon": False, "pdf.fonttype": 42,
})

STATED = []  # (what, value as printed, value the report states)


def fmt(x, nd):
    """Round half up in decimal arithmetic from the full-precision value."""
    q = Decimal(1) if nd == 0 else Decimal(10) ** -nd
    return str(Decimal(repr(float(x))).quantize(q, ROUND_HALF_UP))


def rows(path):
    with open(RES / path, newline="") as fh:
        return list(csv.DictReader(fh))


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.03,
                metadata={"CreationDate": None, "Creator": None, "Producer": None})
    plt.close(fig)
    print("wrote", (OUT / name).relative_to(REPO))


def strip(ax, keep=("bottom",)):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def fig1():
    """Per meeting, DER outside and inside overlap (pyannote 3.1, distant mic)."""
    full = lambda r: float(r["collar"]) == 0 and r["exclude_overlap"] == "0"  # noqa: E731
    der = {}
    for r in rows("decomp_pyannote31/per_file.csv"):
        if r["system"] == "pyannote31_sdm" and full(r) and r["region"] in ("single", "overlap"):
            err = float(r["missed_s"]) + float(r["false_alarm_s"]) + float(r["confusion_s"])
            der[(r["uri"], r["region"])] = 100 * err / float(r["scored_speaker_s"])
    share = {r["uri"]: 100 * float(r["overlap_time_frac"]) for r in rows("overlap_stats/per_file.csv")
             if r["annotation"] == "only_words" and r["partition"] == "test"}
    pooled = {r["region"]: 100 * float(r["der"]) for r in rows("decomp_pyannote31/aggregate.csv")
              if r["system"] == "pyannote31_sdm" and full(r) and r["region"] in ("single", "overlap")}
    uris = sorted(share, key=share.get)
    assert len(uris) == 16 and all((u, "overlap") in der for u in uris)
    ratios = [der[(u, "overlap")] / der[(u, "single")] for u in uris]
    STATED.append(("fig1 ratio range", f"{fmt(min(ratios), 1)} to {fmt(max(ratios), 1)}", "2.4 to 7.7"))
    STATED.append(("fig1 pooled single / overlap", f"{fmt(pooled['single'], 1)} / {fmt(pooled['overlap'], 1)}", "11.1 / 43.6"))

    n = len(uris)
    height = 0.22 * n + 1.0
    fig, ax = plt.subplots(figsize=(W, height))
    fig.subplots_adjust(left=0.19, right=0.985, top=1 - 0.32 / height, bottom=0.5 / height)
    ys = list(range(n - 1, -1, -1))  # lowest overlap share at the top
    for y, u in zip(ys, uris):
        s, o = der[(u, "single")], der[(u, "overlap")]
        ax.plot([s, o], [y, y], color=GREY_LIGHT, lw=2.4, solid_capstyle="round", zorder=1)
        ax.scatter([s], [y], s=22, color=GREY, linewidths=0, zorder=3)
        ax.scatter([o], [y], s=22, color=BLUE, linewidths=0, zorder=3)
    for region in ("single", "overlap"):
        v = pooled[region]
        ax.axvline(v, color=MUTED, lw=0.7, ls=(0, (3, 3)), zorder=0)
        ax.text(v + 0.5, n - 0.35, f"all meetings {fmt(v, 1)}%", color=INK2, fontsize=8, va="bottom")
    trans = blended_transform_factory(fig.transFigure, ax.transData)  # label columns left of the axes
    for y, u in zip(ys, uris):
        ax.text(0.003, y, u, transform=trans, va="center")
        ax.text(0.172, y, f"{fmt(share[u], 1)}%", transform=trans, ha="right", va="center", color=INK2)
    ax.text(0.003, n - 0.35, "Meeting", transform=trans, va="bottom", fontsize=8, color=INK2)
    ax.text(0.172, n - 0.35, "Overlap", transform=trans, ha="right", va="bottom", fontsize=8, color=INK2)
    ax.set_ylim(-0.7, n + 0.3)
    ax.set_yticks([])
    ax.set_xlim(0, 60)
    ax.set_xticks(range(0, 61, 10), [f"{v}%" for v in range(0, 61, 10)])
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    strip(ax)
    ax.set_xlabel("DER within the region (% of that region's reference speaker time)")
    handles = [Line2D([], [], marker="o", ls="", color=GREY, ms=5, label="single-speaker speech"),
               Line2D([], [], marker="o", ls="", color=BLUE, ms=5, label="inside overlap")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=2,
               handletextpad=0.2, columnspacing=1.6, borderaxespad=0.1)
    save(fig, "fig1_der_by_meeting.pdf")


def fig2():
    """Outcome of each overlapped true word under the one-speaker-per-word rule."""
    pick = {r["system"]: r for r in rows("attr_oracle_pyannote31/aggregate.csv")
            if r["region"] == "overlap" and r["mapping"] == "optimal"}
    systems = [("oracle_diarisation", "Perfect diarisation"),
               ("pyannote31_ihm_mix", "pyannote 3.1, headset mix"),
               ("pyannote31_sdm", "pyannote 3.1, distant mic")]
    cats = [("correct", "credited to the right speaker", GREY_LIGHT, INK),
            ("rule", "wrong: the word rule picked the other talker", BLUE, "white"),
            ("missed", "wrong: the system missed the speaker", ORANGE, INK),
            ("other", "wrong: extra cluster, silent speaker or none", GREY, INK)]
    fig, ax = plt.subplots(figsize=(W, 2.15))
    fig.subplots_adjust(left=0.002, right=0.998)
    for i, (key, name) in enumerate(systems):
        r = pick[key]
        err, rule = 100 * float(r["attribution_error"]), 100 * float(r["other_active_rule"])
        missed = 100 * float(r["other_active_missed"])
        part = {"correct": 100 - err, "rule": rule, "missed": missed, "other": err - rule - missed}
        y = len(systems) - 1 - i
        ax.text(0, y + 0.33, name, fontweight="bold", va="bottom")
        acc = 0.0
        for c, _, colour, text_colour in cats:
            v = part[c]
            ax.barh(y, v, left=acc, height=0.5, color=colour, edgecolor="white", linewidth=1.2)
            if v >= 4:
                ax.text(acc + v / 2, y, f"{fmt(v, 1)}%", ha="center", va="center", color=text_colour,
                        fontweight="bold" if c == "rule" else "normal")
            acc += v
        STATED.append((f"fig2 {key} correct/rule/missed/other",
                       "/".join(fmt(part[c], 1) for c, *_ in cats), None))
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.4, len(systems) - 0.2)
    ax.set_xticks([])
    ax.set_yticks([])
    strip(ax, keep=())
    # two columns fill column by column; this order reads row by row in bar order
    legend_order = [cats[0], cats[2], cats[1], cats[3]]
    ax.legend(handles=[Patch(facecolor=c, edgecolor="none", label=lab) for _, lab, c, _ in legend_order],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, handlelength=1.0,
              columnspacing=1.6, borderaxespad=0.1)
    save(fig, "fig2_word_outcomes.pdf")


def fig3():
    """pyannote's cpWER split into transcription, word assignment and diarisation."""
    t = {(r["condition"], r["region"], r["statistic"]): 100 * float(r["point"])
         for r in rows("wer_decomposition_whisper_turbo/aggregate.csv")}
    regions = [("all", "All words"), ("single", "Outside overlap"), ("overlap", "In overlap")]
    mics = [("ihm_mix", "Headset mix"), ("sdm", "Distant mic")]
    parts = [("part_transcription", "transcription", GREY, INK),
             ("part_word_rule", "word assignment", BLUE, "white"),
             ("part_diarisation", "diarisation", ORANGE, INK)]
    height = 2.85
    fig, ax = plt.subplots(figsize=(W, height))
    fig.subplots_adjust(left=0.335, right=0.985, top=1 - 0.3 / height, bottom=0.02)
    trans = blended_transform_factory(fig.transFigure, ax.transData)  # label columns left of the axes
    y = 0.0
    totals = []
    for reg, reg_name in regions:
        for j, (mic, mic_name) in enumerate(mics):
            vals = [t[(mic, reg, p)] for p, *_ in parts]
            total = t[(mic, reg, "cpwer_pyannote")]
            assert abs(sum(vals) - total) < 1e-3, (mic, reg)  # parts are stored to 6 decimals
            acc = 0.0
            for v, (_, _, colour, text_colour) in zip(vals, parts):
                ax.barh(-y, v, left=acc, height=0.62, color=colour, edgecolor="white", linewidth=1.2)
                if v >= 4.5:
                    ax.text(acc + v / 2, -y, fmt(v, 1), ha="center", va="center", color=text_colour,
                            fontweight="bold" if colour == BLUE else "normal")
                acc += v
            ax.text(total + 0.8, -y, f"{fmt(total, 1)}%", va="center", fontweight="bold")
            totals.append(fmt(total, 1))
            if j == 0:
                ax.text(0.003, -y, reg_name, transform=trans, va="center", fontweight="bold")
            ax.text(0.322, -y, mic_name, transform=trans, ha="right", va="center", color=INK2)
            y += 1
        y += 0.55
    STATED.append(("fig3 totals", " ".join(totals), "33.4 39.8 18.8 26.1 78.1 81.6"))
    ax.set_xlim(0, 90)
    ax.set_ylim(-(y - 0.55) + 0.45, 0.55)
    ax.set_xticks([])
    ax.set_yticks([])
    strip(ax, keep=())
    fig.legend(handles=[Patch(facecolor=c, edgecolor="none", label=lab) for _, lab, c, _ in parts],
               loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=3, handlelength=1.0,
               columnspacing=1.8, borderaxespad=0.1)
    save(fig, "fig3_cpwer_parts.pdf")


def fig4():
    """Share of overlapped words Whisper deletes, by the talker's relative level."""
    bands = [("[-inf,-10)", "below −10"), ("[-10,-5)", "−10 to −5"), ("[-5,0)", "−5 to 0"),
             ("[0,5)", "0 to 5"), ("[5,10)", "5 to 10"), ("[10,inf)", "above 10")]
    mics = [("ihm_mix", "Headset mix", GREY_DARK, -0.11), ("sdm", "Distant mic", GREY, 0.11)]
    got = {(r["condition"], r["group"]): r for r in rows("overlap_words_whisper_turbo/aggregate.csv")
           if r["analysis"] == "asr" and r["group_type"] == "level"}
    base = {r["condition"]: 100 * float(r["point"]) for r in rows("wer_decomposition_whisper_turbo/aggregate.csv")
            if r["region"] == "single" and r["statistic"] == "di_del"}
    fig, ax = plt.subplots(figsize=(W, 2.7))
    fig.subplots_adjust(left=0.075, right=0.995)
    ends = []
    for mic, name, colour, dx in mics:
        for i, (g, _) in enumerate(bands):
            r = got[(mic, g)]
            v, lo, hi = (100 * float(r[k]) for k in ("rate_bad", "lo", "hi"))
            ax.vlines(i + dx, lo, hi, color=colour, lw=1.4, zorder=2)
            ax.scatter([i + dx], [v], s=26, color=colour, linewidths=0, zorder=3)
            if i in (0, len(bands) - 1):
                left = mic == "ihm_mix"
                ax.text(i + dx + (-0.09 if left else 0.09), v, f"{fmt(v, 0)}%", ha="right" if left else "left",
                        va="center", fontweight="bold")
                ends.append(fmt(v, 0))
    for (mic, name, colour, _), va, dy in zip(mics, ("top", "bottom"), (-1.2, 1.2)):
        ax.axhline(base[mic], color=colour, lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax.text(len(bands) - 0.55, base[mic] + dy, f"outside overlap, {name.lower()}: {fmt(base[mic], 0)}% deleted",
                ha="right", va=va, color=INK2, fontsize=8)
    STATED.append(("fig4 end labels (headset first, last; distant first, last)", " ".join(ends), "65 34 64 43"))
    STATED.append(("fig4 baselines headset / distant", f"{fmt(base['ihm_mix'], 0)} / {fmt(base['sdm'], 0)}", "11 / 16"))
    ax.set_xlim(-0.55, len(bands) - 0.45)
    ax.set_ylim(0, 80)
    ax.set_xticks(range(len(bands)), [lab for _, lab in bands])
    ax.set_yticks(range(0, 81, 20), [f"{v}%" for v in range(0, 81, 20)])
    ax.tick_params(axis="both", length=0)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    strip(ax)
    ax.set_xlabel("talker level minus the other active talkers (dB, from the individual headsets)")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=c, ms=5, label=n) for _, n, c, _ in mics],
              loc="lower left", bbox_to_anchor=(-0.01, 1.0), ncol=2, handletextpad=0.2, columnspacing=1.6)
    save(fig, "fig4_deletion_by_level.pdf")


def fig5():
    """DER of both systems against the two references of the same meetings."""
    def der(path):
        return {r["system"]: tuple(100 * float(r[k]) for k in ("der", "der_lo", "der_hi")) for r in rows(path)
                if float(r["collar"]) == 0 and r["exclude_overlap"] == "0" and r["region"] == "all"}
    ow, fa = der("decomp_sortformer_v21/aggregate.csv"), der("decomp_fa_reference/aggregate.csv")
    systems = [("pyannote31", "pyannote 3.1", GREY), ("sortformer_v21", "Sortformer", BLUE)]
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.6), sharey=True)
    printed = []
    for ax, (mic, title) in zip(axes, (("ihm_mix", "Headset mix"), ("sdm", "Distant mic"))):
        for sy, name, colour in systems:
            pts = [ow[f"{sy}_{mic}"], fa[f"{sy}_{mic}"]]
            ax.plot([0, 1], [pts[0][0], pts[1][0]], color=colour, lw=2, zorder=2)
            for j, (v, lo, hi) in enumerate(pts):
                ax.vlines(j, lo, hi, color=colour, lw=1.3, alpha=0.55, zorder=1)
                ax.scatter([j], [v], s=28, color=colour, linewidths=0, zorder=3)
                ax.text(j - 0.08 if j == 0 else j + 0.08, v, fmt(v, 1), ha="right" if j == 0 else "left",
                        va="center")
                printed.append(fmt(v, 1))
        ax.set_title(title, fontsize=9, fontweight="bold", color=INK)
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(0, 45)
        ax.set_xticks([0, 1], ["AMI word times", "forced alignment"])
        ax.set_yticks([])
        ax.tick_params(axis="x", length=0)
        strip(ax)
    STATED.append(("fig5 values (headset pya ow fa, sortf ow fa; distant ...)", " ".join(printed),
                   "18.8 36.2 29.4 15.9 22.4 38.4 32.0 19.5"))
    fig.legend(handles=[Line2D([], [], marker="o", color=c, lw=2, ms=5, label=n) for _, n, c in systems],
               loc="lower left", bbox_to_anchor=(0.08, 0.98), ncol=2, handletextpad=0.4, columnspacing=1.6)
    fig.subplots_adjust(left=0.02, right=0.98, wspace=0.12)
    save(fig, "fig5_reference_flip.pdf")


if __name__ == "__main__":
    for make in (fig1, fig2, fig3, fig4, fig5):
        make()
    bad = 0
    for what, got, want in STATED:
        ok = want is None or got == want
        bad += not ok
        print(f"{'ok ' if ok else 'BAD'} {what}: {got}" + ("" if want is None or ok else f" (report says {want})"))
    raise SystemExit(1 if bad else 0)
