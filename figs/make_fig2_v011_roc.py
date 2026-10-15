#!/usr/bin/env python3
"""Figure 2 (Preprint v0.11, ROC-space version): each model's two option orders as points in
ROC space (x = 1 - specificity, y = sensitivity), joined by an arrow from nodule-first to
vessel-first. The diagonal is chance (Youden's J = 0); J is the vertical distance above it.

Counts come from the finished CSVs with the same loader as make_fig2_v09.py (FIG2_DATA=v011)
and are asserted against manuscript Table 2 (v0.11) before plotting.
Outputs: figs/out/fig2_v011_roc.png (600 dpi) and .pdf.
"""
import os, sys
os.environ["FIG2_DATA"] = "v011"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_fig2_v09 as base
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

TABLE2_V011 = {
    "Q4": ((32, 42), (1, 54)), "Q8": ((0, 54), (5, 54)), "Q25": ((22, 32), (38, 14)),
    "LS7": ((7, 54), (26, 48)), "GM3": ((0, 54), (22, 29)), "M1": ((9, 53), (30, 37)),
    "M15": ((0, 54), (29, 36)), "GPT56": ((14, 54), (17, 54)), "GM31P": ((33, 39), (33, 38)),
    "FAB5": ((34, 37), (31, 34)),
}

def main():
    res = base.load()
    bad = [k for k in TABLE2_V011 if res[k] != TABLE2_V011[k]]
    if bad: sys.exit(f"Aborting: mismatch with Table 2 for {bad}: {[res[k] for k in bad]}")
    print("All 40 numerators match manuscript Table 2 (v0.11).")
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "font.size": 7.5, "pdf.fonttype": 42, "axes.linewidth": 0.6})
    models = base.OPEN + base.COMM
    fig, axes = plt.subplots(2, 5, figsize=(180 / 25.4, 84 / 25.4))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.86, bottom=0.12, wspace=0.18, hspace=0.42)
    for ax, (key, name) in zip(axes.flat, models):
        pts = [(100 - 100 * tn / base.N_VES, 100 * tp / base.N_NOD) for tp, tn in res[key]]
        ax.plot([0, 100], [0, 100], color="#9a9a9a", lw=0.7, ls=(0, (3, 2)), zorder=1)
        if pts[0] != pts[1]:
            ax.annotate("", xy=pts[1], xytext=pts[0], zorder=3,
                        arrowprops=dict(arrowstyle="-|>", color="#6b6b6b", lw=0.8, mutation_scale=7,
                                        shrinkA=3.5, shrinkB=3.8))
        ax.plot(*pts[0], marker="o", ms=4.6, mfc=base.C_STD, mec=base.C_STD, ls="none", zorder=5, clip_on=False)
        ax.plot(*pts[1], marker="o", ms=4.6, mfc="white", mec=base.C_REV, mew=1.2, ls="none", zorder=6, clip_on=False)
        J = [(tp / base.N_NOD + tn / base.N_VES - 1) for tp, tn in res[key]]
        ax.text(97, 4, f"J {J[0]:.2f} / {J[1]:.2f}", ha="right", va="bottom", fontsize=6.3, color=base.MUTED)
        ax.set_xlim(-3, 103); ax.set_ylim(-3, 103); ax.set_aspect("equal")
        ax.set_xticks([0, 50, 100]); ax.set_yticks([0, 50, 100])
        ax.set_title(name, fontsize=7.5, color=base.INK, pad=3)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        for s in ("bottom", "left"): ax.spines[s].set_color(base.MUTED)
        ax.tick_params(colors=base.MUTED, labelcolor=base.INK, length=2, width=0.5, labelsize=6.5)
        if key not in ("Q4", "M1"): ax.set_yticklabels([])
    for ax in axes[1]: ax.set_xlabel("100 − specificity (%)", fontsize=7, color=base.INK, labelpad=2)
    for ax in axes[:, 0]: ax.set_ylabel("Sensitivity (%)", fontsize=7, color=base.INK, labelpad=2)
    h = [Line2D([0], [0], marker="o", ms=4.6, mfc=base.C_STD, mec=base.C_STD, ls="none", label="Nodule-first order"),
         Line2D([0], [0], marker="o", ms=4.6, mfc="white", mec=base.C_REV, mew=1.2, ls="none", label="Vessel-first order"),
         Line2D([0], [0], color="#9a9a9a", lw=0.7, ls=(0, (3, 2)), label="Chance (J = 0)")]
    fig.legend(handles=h, loc="upper right", bbox_to_anchor=(0.99, 0.995), ncol=3, frameon=False, fontsize=7,
               columnspacing=1.4, handletextpad=0.4, borderaxespad=0)
    out = os.path.join(base.OUT, "fig2_v011_roc")
    fig.savefig(out + ".png", dpi=600); fig.savefig(out + ".pdf"); print("wrote", out + ".png")

if __name__ == "__main__":
    main()
