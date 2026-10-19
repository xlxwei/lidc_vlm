#!/usr/bin/env python3
"""Figure 1 (AJR, 2026-10-19): selection of the nodules and vessel cross-sections.
Every count is recomputed from the manifest CSVs and checked against the numbers in the manuscript.
Style follows the /make-figures flow-diagram pipeline (generate_flow_diagram.R STYLE_HEADER: black
outline, white fill, Arial 11 pt); Graphviz dot is called directly because R is not installed.
Outputs: figs/out/fig1_ajr_flow.pdf (vector), .png (300 dpi), _600.png (600 dpi), .dot."""
import csv, os, re, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = lambda f: list(csv.DictReader(open(os.path.join(ROOT, "manifest", f))))
OUT = os.path.join(ROOT, "figs", "out", "fig1_ajr_flow")

# Nodules (01_build_candidate_pool.py, 02_select_mini_batch.py)
cand = M("candidate_nodules.csv")                          # >=3 of 4 readers, consensus diameter >=3 mm
merged = [r for r in cand if int(r["n_readers"]) > 4]
kept = [r for r in cand if int(r["n_readers"]) <= 4]
spread = [r for r in kept if float(r["diam_spread_mm"]) > 5.0]
elig = [r for r in kept if float(r["diam_spread_mm"]) <= 5.0]
solid = [r for r in elig if r["texture_label"] == "solid"]
sel = M("selected_nodules.csv")
assert all(r["texture_label"] == "solid" for r in sel)
# Vessels (06_find_foils.py, 08_select_foils.py, foil_adjudication.csv)
fcand = M("foil_candidates.csv")
fsel = M("foils_for_adjudication.csv")
adj = {r["patch_id"]: r["verdict"] for r in M("foil_adjudication.csv") if r["verdict"]}
vmax = max(float(r["diam_mm"]) for r in fsel)
big = sorted(float(r["diam_med_mm"]) for r in sel if float(r["diam_med_mm"]) > vmax)
# Final sample (subset_v010_93.txt)
sub = [l.strip() for l in open(os.path.join(ROOT, "manifest", "subset_v010_93.txt")) if l.strip()]
nod = [x for x in sub if x.startswith("nod_")]; ves = [x for x in sub if x.startswith("foil_")]
pn = {re.search(r"IDRI-(\d+)", x).group(1) for x in nod}; pv = {x.split("_")[1] for x in ves}
assert all(adj.get(x) == "vessel" for x in ves)

n = dict(scans_all=1018, scans=218,  # 01_build_candidate_pool.py: 218/1018 scans with 1.25-mm spacing and thickness
         cand=len(cand), cand_pt=len({r["patient_id"] for r in cand}), merged=len(merged), spread=len(spread),
         elig=len(elig), solid=len(solid), sel=len(sel), sel_pt=len({r["patient_id"] for r in sel}),
         fcand=len(fcand), fscans=len({r["exam_id"] for r in fcand}), fsel=len(fsel),
         fconf=sum(adj.get(r["cand_id"]) == "vessel" for r in fsel),
         big=len(big), nod=len(nod), nod_pt=len(pn), ves=len(ves), ves_pt=len(pv), both=len(pn & pv), pts=len(pn | pv))
assert sorted(ves) == sorted(r["cand_id"] for r in fsel)
EXPECT = dict(cand=331, cand_pt=156, merged=1, spread=26, elig=304, solid=222, sel=50, sel_pt=50, fcand=225,
              fscans=48, fsel=54, fconf=54, big=11, nod=39, nod_pt=39, ves=54, ves_pt=35, both=26, pts=48)
for k, v in EXPECT.items():
    assert n[k] == v, (k, n[k], v)
assert n["cand"] - n["merged"] - n["spread"] == n["elig"] and n["sel"] - n["big"] == n["nod"]
assert len(sub) == n["nod"] + n["ves"] == 93 and abs(vmax - 14.3) < 1e-9

STYLE = '''  graph [layout=dot, rankdir=TB, fontname="Arial", splines=ortho, nodesep=0.45, ranksep=0.55]
  node  [shape=box, style="rounded,filled", fillcolor=white, color=black, fontname="Arial", fontsize=11,
         penwidth=1.2, margin="0.20,0.12"]
  edge  [color=black, arrowhead=normal, arrowsize=0.75, penwidth=1.0]'''
L = lambda s: s.replace('"', '\\"')
nodes = {
    "a": f"LIDC-IDRI\\n{n['scans_all']} CT scans",
    "b": f"{n['scans']} scans with 1.25-mm slice spacing",
    "n1": f"{n['cand']} nodules ≥ 3 mm marked by at least\\nthree of four radiologists ({n['cand_pt']} patients)",
    "x1": (f"Excluded (n = {n['merged'] + n['spread']})\\l"
           f"• Marks merged from more than four readers (n = {n['merged']})\\l"
           f"• Diameters differing by more than 5.0 mm\\l   between readers (n = {n['spread']})\\l"),
    "n2": f"{n['elig']} nodules, {n['solid']} solid",
    "n3": f"{n['sel']} solid nodules chosen by a fixed rule\\nstratified by diameter ({n['sel_pt']} patients)",
    "x2": (f"Excluded (n = {n['big']})\\l• Larger than the largest vessel\\l"
           f"   ({big[0]:.1f}–{big[-1]:.1f} mm)\\l"),
    "n4": f"{n['nod']} nodules ({n['nod_pt']} patients)",
    "v1": f"{n['fcand']} vessel candidates located automatically\\nin {n['fscans']} of the {n['sel']} scans with these nodules",
    "v2": f"{n['fsel']} chosen by a fixed rule stratified by\\ndiameter, no more than two per scan",
    "v3": f"{n['fconf']} confirmed as vessel cross-sections\\nby a radiologist ({n['ves_pt']} patients)",
    "f": (f"{n['nod'] + n['ves']} structures: {n['nod']} nodules and {n['ves']} vessels, 4.5–{vmax:.1f} mm\\n"
          f"{n['pts']} patients; {n['both']} contributed both"),
}
dot = ["digraph flow {", STYLE]
for k, v in nodes.items():
    extra = ', penwidth=1.8' if k == "f" else ''
    dot.append(f'  {k} [label="{L(v)}"{extra}]')
for k in ("p1", "p2"):
    dot.append(f'  {k} [shape=point, width=0, height=0, label=""]')
dot += ["  a -> b", "  b -> n1", "  n1 -> p1 [arrowhead=none]", "  p1 -> n2", "  p1 -> x1",
        "  n2 -> n3", "  n3 -> v1", "  n3 -> p2 [arrowhead=none]", "  p2 -> n4", "  p2 -> x2",
        "  v1 -> v2", "  v2 -> v3", "  n4 -> f", "  v3 -> f",
        "  { rank=same; p1; x1 }", "  { rank=same; p2; x2 }", "  { rank=same; n4; v3 }", "}"]
src = "\n".join(dot)
open(OUT + ".dot", "w").write(src)
subprocess.run(["dot", "-Tpdf", "-o", OUT + ".pdf"], input=src, text=True, check=True)
subprocess.run(["dot", "-Tpng", "-Gdpi=300", "-o", OUT + ".png"], input=src, text=True, check=True)
subprocess.run(["dot", "-Tpng", "-Gdpi=600", "-o", OUT + "_600.png"], input=src, text=True, check=True)
print({k: n[k] for k in EXPECT}, "largest vessel", vmax, "excluded nodules", big[0], "-", big[-1])
