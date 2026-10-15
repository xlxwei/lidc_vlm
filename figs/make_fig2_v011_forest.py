#!/usr/bin/env python3
"""Figure 2 (v0.11, forest version): sensitivity (A), specificity (B), and Youden's J (C) of the ten models
under the two option orders. Sensitivity and specificity with Wilson 95% CIs from the finished CSVs (loader of
make_fig2_v09.py, asserted against Table 2); J with patient-level bootstrap 95% CIs read from
results/v011_effect_size.txt (output of 34_effect_size_v011.py) and checked against the counts.
Outputs: figs/out/fig2_v011_forest.png (600 dpi) and .pdf."""
import os, re, sys
os.environ["FIG2_DATA"] = "v011"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_fig2_v09 as b
from make_fig2_v011_roc import TABLE2_V011
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def load_j():
    txt = open(os.path.join(b.ROOT, "results", "v011_effect_size.txt")).read()
    out = {}
    for m in re.finditer(r"^(.+?)\s+(NF|VF) J ([+-]\d\.\d\d) \(([+-]\d\.\d\d) to ([+-]\d\.\d\d)\)", txt, re.M):
        out[(m[1].strip(), m[2])] = tuple(float(x) for x in m.groups()[2:])
    return out

def main():
    res = b.load()
    bad = [k for k in TABLE2_V011 if res[k] != TABLE2_V011[k]]
    if bad: sys.exit(f"Aborting: Table 2 mismatch {bad}")
    J = load_j(); names = dict(b.OPEN + b.COMM)
    for k, nm in b.OPEN + b.COMM:
        for o, (tp, tn) in zip(("NF", "VF"), res[k]):
            j = tp / b.N_NOD + tn / b.N_VES - 1
            assert abs(J[(nm.replace("*", ""), o)][0] - round(j, 2)) < 0.006, (k, o, j)
    print("Counts match Table 2; J values match the counts.")
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "font.size": 8, "pdf.fonttype": 42, "axes.linewidth": 0.6})
    ypos, headers, y = {}, [], 0.0
    for title, models in (("Open-weight models", b.OPEN), ("Commercial models", b.COMM)):
        headers.append((y, title)); y += 1
        for key, _ in models: ypos[key] = y; y += 1
        y += 0.7
    ymax = y - 0.7
    fig, axes = plt.subplots(1, 3, figsize=(180 / 25.4, 108 / 25.4), sharey=True,
                             gridspec_kw={"width_ratios": [1, 1, 1]})
    fig.subplots_adjust(left=0.165, right=0.99, top=0.90, bottom=0.115, wspace=0.08)
    spec = [("Sensitivity (%)", "A"), ("Specificity (%)", "B"), ("Youden index", "C")]
    for i, (ax, (xlabel, letter)) in enumerate(zip(axes, spec)):
        for key, nm in b.OPEN + b.COMM:
            y0 = ypos[key]; pts = []
            for o, (col, dy, oo) in enumerate(((b.C_STD, -b.DY, "NF"), (b.C_REV, +b.DY, "VF"))):
                tp, tn = res[key][o]
                if i == 0: p, lo, hi = b.wilson(tp, b.N_NOD)
                elif i == 1: p, lo, hi = b.wilson(tn, b.N_VES)
                else: p, lo, hi = J[(nm.replace("*", ""), oo)]
                pts.append((p, y0 + dy))
                ax.plot([lo, hi], [y0 + dy, y0 + dy], color=col, lw=0.8, alpha=0.75, solid_capstyle="butt", zorder=2)
            ax.plot([pts[0][0], pts[1][0]], [pts[0][1], pts[1][1]], color="#8a8a8a", lw=0.7, zorder=3)
            ax.plot(*pts[0], marker="o", ms=5.0, mfc=b.C_STD, mec=b.C_STD, ls="none", zorder=5, clip_on=False)
            ax.plot(*pts[1], marker="o", ms=5.0, mfc="white", mec=b.C_REV, mew=1.3, ls="none", zorder=5, clip_on=False)
        if i < 2:
            ax.set_xlim(-4, 104); ax.set_xticks([0, 25, 50, 75, 100])
        else:
            ax.set_xlim(-0.24, 1.04); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_xticklabels(["0", "0.25", "0.50", "0.75", "1.00"])
            ax.axvline(0, color="#555555", lw=0.8, ls=(0, (3, 2)), zorder=1)
        ax.set_ylim(ymax - 0.35, -0.45)
        ax.set_xlabel(xlabel, color=b.INK, labelpad=3)
        ax.xaxis.grid(True, color=b.GRID, lw=0.6); ax.set_axisbelow(True)
        for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(b.MUTED)
        ax.tick_params(axis="x", colors=b.MUTED, length=2.5, width=0.6, labelcolor=b.INK)
        ax.tick_params(axis="y", length=0)
        ax.text(0.0, 1.035, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", color=b.INK, ha="left", va="bottom")
    ax = axes[0]
    ax.set_yticks([ypos[k] for k, _ in b.OPEN + b.COMM]); ax.set_yticklabels([nm for _, nm in b.OPEN + b.COMM], color=b.INK)
    for yh, title in headers:
        ax.text(-0.62, yh + 0.1, title, transform=ax.get_yaxis_transform(), fontsize=8, fontweight="bold", color=b.INK, ha="left", va="center")
    h = [Line2D([0], [0], marker="o", ms=5, mfc=b.C_STD, mec=b.C_STD, color=b.C_STD, lw=0.8, label="Nodule-first option order"),
         Line2D([0], [0], marker="o", ms=5, mfc="white", mec=b.C_REV, mew=1.3, color=b.C_REV, lw=0.8, label="Vessel-first option order")]
    fig.legend(handles=h, loc="upper right", bbox_to_anchor=(0.99, 0.995), ncol=2, frameon=False, fontsize=8, handlelength=2.4, columnspacing=1.8, borderaxespad=0)
    out = os.path.join(b.OUT, "fig2_v011_forest")
    fig.savefig(out + ".png", dpi=600); fig.savefig(out + ".pdf"); print("wrote", out + ".png")

if __name__ == "__main__":
    main()
