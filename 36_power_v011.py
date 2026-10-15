#!/usr/bin/env python3
"""[v0.11 M1] Fisher exact test 的檢定力模擬：39 nodules、54 vessels，sensitivity = specificity = (1 + J) / 2。
回報兩個顯著水準下達到 80% 檢定力的最小 Youden's J：α = .05（未校正）與 α = .05/7（七個 open-weight 模型 Holm 校正中最嚴格的一階）。
固定種子、每個 J 5000 次。輸出 results/v011_power.txt。"""
import os
import numpy as np
from scipy.stats import fisher_exact
ROOT = os.path.dirname(os.path.abspath(__file__))
N_NOD, N_VES, REPS, SEED = 39, 54, 5000, 20261016
ALPHAS = {"0.05": 0.05, "0.05/7": 0.05 / 7}
rng = np.random.default_rng(SEED)
grid = np.round(np.arange(0.20, 0.501, 0.01), 2)
lines = [f"Fisher exact test power; {N_NOD} nodules, {N_VES} vessels; sensitivity = specificity = (1 + J)/2; {REPS} replicates per J; seed {SEED}", "J\t" + "\t".join(f"power@{k}" for k in ALPHAS)]
first = {}
for J in grid:
    p = (1 + J) / 2
    tp = rng.binomial(N_NOD, p, REPS); tn = rng.binomial(N_VES, p, REPS)
    pv = np.array([fisher_exact([[a, N_NOD - a], [N_VES - b, b]])[1] for a, b in zip(tp, tn)])
    pw = {k: float((pv < a).mean()) for k, a in ALPHAS.items()}
    lines.append(f"{J:.2f}\t" + "\t".join(f"{pw[k]:.3f}" for k in ALPHAS))
    for k in ALPHAS:
        if k not in first and pw[k] >= 0.80: first[k] = J
lines.append("smallest J with power >= 0.80: " + "; ".join(f"alpha {k}: {first.get(k)}" for k in ALPHAS))
out = "\n".join(lines) + "\n"
open(os.path.join(ROOT, "results", "v011_power.txt"), "w").write(out)
print(out)
