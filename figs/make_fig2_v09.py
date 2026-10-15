#!/usr/bin/env python3
"""Figure 2 (Preprint v0.9): sensitivity and specificity of ten models under two option orders.

Values are derived from the finished result CSVs with the same loading logic as
25_tables_v09.py (size-matched set, 4.5-14.3 mm, rep == '1'), then asserted against
the numerators printed in the manuscript's Table 2. Any mismatch aborts before plotting.

Outputs: figs/out/fig2_v09_option_order.png (600 dpi) and .pdf (vector).
"""
import csv, math, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figs", "out")
R = lambda p: list(csv.DictReader(open(os.path.join(ROOT, p))))

OPEN = [("Q4", "Qwen3-VL 4B"), ("Q8", "Qwen3-VL 8B"), ("Q25", "Qwen2.5-VL 7B"),
        ("LS7", "Lingshu-7B*"), ("GM3", "Gemma 3 4B"), ("M1", "MedGemma 1 4B*"),
        ("M15", "MedGemma 1.5 4B*")]
COMM = [("GPT56", "GPT-5.6"), ("GM31P", "Gemini 3.1 Pro"), ("FAB5", "Claude Fable 5")]
N_NOD, N_VES = 39, 54

# Manuscript Table 2: key -> ((TP, TN) nodule first, (TP, TN) vessel first)
TABLE2 = {
    "Q4": ((31, 40), (1, 54)), "Q8": ((0, 54), (1, 54)), "Q25": ((22, 33), (36, 16)),
    "LS7": ((7, 54), (26, 48)), "GM3": ((0, 54), (22, 30)), "M1": ((9, 54), (28, 37)),
    "M15": ((0, 54), (31, 31)), "GPT56": ((9, 54), (17, 54)), "GM31P": ((33, 40), (32, 42)),
    "FAB5": ((33, 35), (33, 36)),
}


def load():
    canon = {l.strip() for l in open(os.path.join(ROOT, "manifest/subset_canon104.txt")) if l.strip()}
    op0 = R("results/lidc_roi_results.csv")
    NEW = os.environ.get("FIG2_DATA") == "v011"
    op = R("results/run22_prompt_open.csv") if NEW else op0
    fr = R("results/run22_prompt_frontier.csv") if NEW else R("results/frontier_results.csv")
    ref = {r["patch_id"]: r for r in op0 if r["run_tag"] == "Q4_LIDC_N9" and r["patch_id"] in canon}
    sm = {p for p in ref if 4.5 <= float(ref[p]["size_mm"]) <= 14.3}
    truth = {p: ref[p]["truth"] for p in sm}
    assert sum(t == "nodule" for t in truth.values()) == N_NOD, "nodule count != 39"
    assert sum(t == "vessel" for t in truth.values()) == N_VES, "vessel count != 54"
    res = {}
    for rows, models in ((op, OPEN), (fr, COMM)):
        for key, _ in models:
            pair = []
            for suf in (("_LIDC_N9nd", "_LIDC_N9ndswap") if NEW else ("_LIDC_N9", "_LIDC_N9swap")):  # nodule first, vessel first
                a = {r["patch_id"]: r["answer"] for r in rows
                     if r["run_tag"] == key + suf and r["rep"] == "1" and r["patch_id"] in sm}
                assert len(a) == 93, (key, suf, len(a))
                tp = sum(1 for p in a if truth[p] == "nodule" and a[p] == "nodule")
                tn = sum(1 for p in a if truth[p] == "vessel" and a[p] == "vessel")
                pair.append((tp, tn))
            res[key] = tuple(pair)
    return res


def check(res):
    bad = [(k, res[k], TABLE2[k]) for k in TABLE2 if res[k] != TABLE2[k]]
    if bad:
        for k, got, exp in bad:
            print(f"MISMATCH {k}: computed {got} vs Table 2 {exp}", file=sys.stderr)
        sys.exit("Aborting: computed numerators do not match manuscript Table 2.")
    print("All 40 numerators match manuscript Table 2.")


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, 100 * max(0.0, c - h), 100 * min(1.0, c + h)


C_STD, C_REV = "#2a6db0", "#c2410c"   # nodule first (filled), vessel first (open)
INK, MUTED, GRID = "#1a1a1a", "#555555", "#dcdcdc"
DY = 0.16                             # vertical offset of the two orders within a row


def main():
    res = load()
    if os.environ.get("FIG2_DATA") != "v011": check(res)

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6,
    })
    # rows: group header, 7 open-weight, gap, group header, 3 commercial
    ypos, headers, y = {}, [], 0.0
    for title, models in (("Open-weight models", OPEN), ("Commercial models", COMM)):
        headers.append((y, title)); y += 1
        for key, _ in models:
            ypos[key] = y; y += 1
        y += 0.7
    ymax = y - 0.7

    fig, axes = plt.subplots(1, 2, figsize=(180 / 25.4, 108 / 25.4), sharey=True)
    fig.subplots_adjust(left=0.185, right=0.985, top=0.90, bottom=0.115, wspace=0.07)

    for ax, idx, n, xlabel, letter in ((axes[0], 0, N_NOD, "Sensitivity (%)", "A"),
                                       (axes[1], 1, N_VES, "Specificity (%)", "B")):
        for key, _ in OPEN + COMM:
            y0 = ypos[key]
            pts = []
            for o, (col, dy) in enumerate(((C_STD, -DY), (C_REV, +DY))):
                p, lo, hi = wilson(res[key][o][idx], n)
                pts.append((p, y0 + dy))
                ax.plot([lo, hi], [y0 + dy, y0 + dy], color=col, lw=0.8, alpha=0.75,
                        solid_capstyle="butt", zorder=2)
            ax.plot([pts[0][0], pts[1][0]], [pts[0][1], pts[1][1]], color="#8a8a8a", lw=0.7, zorder=3)
            ax.plot(*pts[0], marker="o", ms=5.2, mfc=C_STD, mec=C_STD, mew=1.0, ls="none",
                    zorder=5, clip_on=False)
            ax.plot(*pts[1], marker="o", ms=5.2, mfc="white", mec=C_REV, mew=1.3, ls="none",
                    zorder=5, clip_on=False)
        ax.set_xlim(-4, 104)
        ax.set_ylim(ymax - 0.35, -0.45)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel(xlabel, color=INK, labelpad=3)
        ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
        ax.yaxis.grid(False)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(MUTED)
        ax.tick_params(axis="x", colors=MUTED, length=2.5, width=0.6, labelcolor=INK)
        ax.tick_params(axis="y", length=0)
        ax.text(0.0, 1.035, letter, transform=ax.transAxes, fontsize=11, fontweight="bold",
                color=INK, ha="left", va="bottom")

    ax = axes[0]
    ax.set_yticks([ypos[k] for k, _ in OPEN + COMM])
    ax.set_yticklabels([nm for _, nm in OPEN + COMM], color=INK)
    ax.tick_params(axis="y", pad=4)
    lab_x = -0.42  # axes fraction, left edge of the label column
    for yh, title in headers:
        ax.text(lab_x, yh + 0.1, title, transform=ax.get_yaxis_transform(), fontsize=8,
                fontweight="bold", color=INK, ha="left", va="center")

    handles = [
        Line2D([0], [0], marker="o", ms=5.2, mfc=C_STD, mec=C_STD, color=C_STD, lw=0.8,
               label="Nodule-first option order"),
        Line2D([0], [0], marker="o", ms=5.2, mfc="white", mec=C_REV, mew=1.3, color=C_REV, lw=0.8,
               label="Vessel-first option order"),
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.985, 0.995), ncol=2,
               frameon=False, fontsize=8, handlelength=2.4, columnspacing=1.8, borderaxespad=0)

    os.makedirs(OUT, exist_ok=True)
    base = os.path.join(OUT, "fig2_v011_option_order" if os.environ.get("FIG2_DATA") == "v011" else "fig2_v09_option_order")
    fig.savefig(base + ".png", dpi=600)
    fig.savefig(base + ".pdf")
    print("wrote", base + ".png", "and .pdf")


if __name__ == "__main__":
    main()
