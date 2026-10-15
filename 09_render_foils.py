#!/usr/bin/env python3
"""渲染中選的 54 個 foil patch（N=9 灰階），並併出與院內同格式的題目中繼檔。

先渲染、後裁決——與院內流程相同：裁決決定哪些「算」血管（進分析），
不決定哪些被渲染。Wei 判為 nodule／other 的候選仍留在 CSV 裡（標記後排除），
不從資料裡刪掉，維持稽核軌跡。

輸出：
  patches/roi_patches_gray_N9/foil_xxx/roi_1..9.png
  manifest/patch_meta.csv  ← 結節＋foil 合併，欄位對齊院內 `patches.csv`，
                              供 run_roi_batch.py 的 --meta 直接使用
"""
import csv
import os
from collections import defaultdict

import numpy as np

import lidc_compat  # noqa: F401
import pylidc as pl
import lidc_render as R

HERE = os.path.dirname(os.path.abspath(__file__))
MAN = os.path.join(HERE, "manifest")
OUT = os.path.join(HERE, "patches", "roi_patches_gray_N9")
NSLICE = 9

# 院內 patches.csv 的欄位順序——對齊才能共用下游腳本
FIELDS = ["patch_id", "exam_id", "truth", "source", "size_mm", "stratum", "lung_rads",
          "x", "y", "z", "spacing_xy", "half_px", "compactness", "dist_from_gt_mm",
          "center_hu", "ring_hu", "render_ok", "note",
          "n_readers", "texture", "diam_spread_mm", "mask_offset_px",
          "air_frac", "series_uid"]


def main():
    foils = list(csv.DictReader(open(os.path.join(MAN, "foils_for_adjudication.csv"))))
    by_exam = defaultdict(list)
    for r in foils:
        by_exam[r["exam_id"]].append(r)

    rows = []
    for ex, rs in sorted(by_exam.items()):
        scan = pl.query(pl.Scan).filter(
            pl.Scan.series_instance_uid == rs[0]["series_uid"]).first()
        vol = scan.to_volume(verbose=False).transpose(1, 0, 2).astype(np.int32)
        for r in rs:
            cx, cy, cz, half = (int(r["x"]), int(r["y"]), int(r["z"]),
                                int(r["half_px"]))
            imgs = R.render_patch(vol, cx, cy, cz, half, NSLICE)
            ok, note = (1, "ok") if imgs is not None else (0, "z_out_of_range")
            chu = rhu = float("nan")
            if imgs is not None:
                chu, rhu = R.center_ring_hu(vol, cx, cy, cz, half)
                R.save_patch(imgs, os.path.join(OUT, r["cand_id"]))
            rows.append(dict(
                patch_id=r["cand_id"], exam_id=ex, truth="vessel_unconfirmed",
                source="tubular_autodetect", size_mm=r["diam_mm"], stratum="",
                lung_rads="", x=cx, y=cy, z=cz, spacing_xy=r["spacing_xy"],
                half_px=half, compactness=r["compactness"],
                dist_from_gt_mm=r["dist_ann_mm"],
                center_hu=round(chu, 1) if chu == chu else "",
                ring_hu=round(rhu, 1) if rhu == rhu else "",
                render_ok=ok, note=note, n_readers="", texture="",
                diam_spread_mm="", mask_offset_px="",
                air_frac=r["air_frac"], series_uid=r["series_uid"]))
        del vol
    print(f"foil 渲染 {sum(r['render_ok'] for r in rows)}/{len(rows)}")

    # 併入結節
    nod = list(csv.DictReader(open(os.path.join(MAN, "nodule_patches.csv"))))
    for r in nod:
        r.setdefault("air_frac", "")
    allrows = [{k: r.get(k, "") for k in FIELDS} for r in nod + rows]
    path = os.path.join(MAN, "patch_meta.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(allrows)
    n_ok = sum(int(r["render_ok"]) for r in allrows)
    print(f"合併題目中繼檔 {n_ok}/{len(allrows)} 可用 → {path}")
    nn = sum(1 for r in allrows if r["truth"] == "nodule" and int(r["render_ok"]))
    print(f"  結節 {nn}、foil {n_ok - nn}")


if __name__ == "__main__":
    main()
