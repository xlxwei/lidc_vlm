#!/usr/bin/env python3
"""Figure 1 (v0.11): all nine slices given to the models for one nodule and one vessel, each as a 3 x 3 grid
(slices 1-9, superior to inferior). An arrow on the central slice marks the structure at the centre; the
arrow was added for the figure only and was not in the images given to the models.
Outputs: figs/out/fig1_v011_nine_slices.png (600 dpi) and .pdf."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(ROOT, "patches", "roi_patches_gray_N9")
if not os.path.isdir(P): P = os.path.join(ROOT, "images", "ct")  # layout of the public repository
CASES = [("nod_LIDC-IDRI-0752_n01", "a", "Solid nodule, 8.1 mm"), ("foil_0068_089", "b", "Vessel cross-section, 9.7 mm")]
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "pdf.fonttype": 42})
fig = plt.figure(figsize=(180 / 25.4, 96 / 25.4))
outer = fig.add_gridspec(1, 2, left=0.01, right=0.99, top=0.90, bottom=0.01, wspace=0.05)
for col, (pid, letter, title) in enumerate(CASES):
    inner = outer[col].subgridspec(3, 3, wspace=0.03, hspace=0.03)
    for k in range(9):
        ax = fig.add_subplot(inner[k // 3, k % 3])
        im = Image.open(os.path.join(P, pid, f"roi_{k + 1}.png")).convert("L")
        assert im.size == (448, 448), im.size
        ax.imshow(im, cmap="gray", vmin=0, vmax=255, interpolation="lanczos")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
        ax.text(10, 22, str(k + 1), color="white", fontsize=7, fontweight="bold", va="top", ha="left")
        if k == 4:
            ax.annotate("", xy=(224 + 24, 224 + 24), xytext=(224 + 110, 224 + 110),
                        arrowprops=dict(arrowstyle="-|>", color="#ffd400", lw=1.4, mutation_scale=9))
    fig.text(0.01 + col * 0.495, 0.965, f"({letter}) {title}", fontsize=9, fontweight="bold", ha="left", va="center")
out = os.path.join(ROOT, "figs", "out", "fig1_v011_nine_slices")
fig.savefig(out + ".png", dpi=600); fig.savefig(out + ".pdf"); print("wrote", out + ".png")
