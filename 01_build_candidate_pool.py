#!/usr/bin/env python3
"""LIDC 候選池：從 pylidc 標註 DB 取出符合凍結協定的結節，**不下載任何影像**。

為什麼先做這一步：LIDC-IDRI 全集 133 GB。先在 26 MB 的標註 DB 上選完題，
再只下載中選的那幾十個 series（估 3–6 GB）。下載量差 20 倍以上。

## 選案條件與理由

1. **slice_spacing == slice_thickness == 1.25 mm**（218/1018 scan 合格）
   這一條看似武斷，其實是凍結協定逼出來的**唯一解**：
   run 9–11 的凍結 prompt（sha `b046abfad45a`）字面寫死
   "9 consecutive axial slices (superior to inferior, 1.25 mm spacing)"。
   LIDC 的 slice_spacing 從 0.45 到 5.0 mm 都有；餵原生 spacing 會讓 prompt
   對多數題目變成假話，改 prompt 則 sha 變動、不再是複製。
   選 1.25 mm 原生子集＝**零內插、prompt 字面為真、幾何與院內 pilot 完全一致**
   （院內 manifest 的 spacing_z 亦為 1.25）。
   代價：犧牲 z 方向異質性。保留的異質性見 §3——in-plane 0.51–0.96 mm（1.9 倍）、
   機型與重建核照收，那些不違反凍結協定。

2. **≥3/4 判讀者共識、≥3 mm**（LUNA16 慣例，Setio 2017）
   pylidc 的 `cluster_annotations()` 以輪廓距離分群，把四位判讀者各自的
   noduleID 對到同一個生理結節上——**LIDC 的 noduleID 只在單一 readingSession
   內唯一，跨 session 不可直接比對**，這是自己寫 XML parser 最常踩的坑。

3. **共識徑＝四位判讀者 `diameter` 的中位數**
   pylidc 的 `diameter` ＝該標註所有輪廓中最大的「單一 axial 切面內」點對距離（mm），
   即臨床慣用的最長軸徑，與院內 `index_size_mm`（Wei 量的最長軸徑）同定義。
   取中位數而非平均：對單一判讀者的離群測量穩健。逐位讀者的值一併記錄，
   spread 大的結節在 QA 時優先抽查。

輸出 `manifest/candidate_nodules.csv`：一列一顆共識結節，含四位讀者的徑與型態評分。
本步驟不選題、不排除、不抽樣——選題在 02。這裡只建立可稽核的母體。
"""
import csv
import os
import statistics as st

import lidc_compat  # noqa: F401  必須在 pylidc 之前
import pylidc as pl

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manifest")
TARGET_SPACING = 1.25
MIN_READERS = 3          # ≥3/4 共識
MIN_DIAM_MM = 3.0        # LUNA16 下限

# LIDC texture 評分 1–5：1=非實質(GGO) … 5=實質。pylidc 的 Annotation.Texture 給字串。
TEXTURE_LABEL = {1: "GGO", 2: "GGO/part-solid", 3: "part-solid",
                 4: "part-solid/solid", 5: "solid"}


def scan_eligible(s):
    return (abs(s.slice_spacing - TARGET_SPACING) < 1e-6
            and abs(s.slice_thickness - TARGET_SPACING) < 1e-6)


def main():
    os.makedirs(OUT, exist_ok=True)
    scans = [s for s in pl.query(pl.Scan).all() if scan_eligible(s)]
    print(f"1.25mm 合格 scan：{len(scans)}/1018")

    rows = []
    n_clusters = n_lt3reader = n_lt3mm = 0
    for s in scans:
        for ci, cluster in enumerate(s.cluster_annotations(), 1):
            n_clusters += 1
            if len(cluster) < MIN_READERS:
                n_lt3reader += 1
                continue
            diams = [a.diameter for a in cluster]
            d_med = st.median(diams)
            if d_med < MIN_DIAM_MM:
                n_lt3mm += 1
                continue
            # 共識座標：四位讀者 centroid 的平均（ijk 體素索引；i=row, j=col, k=slice）
            cen = [sum(a.centroid[k] for a in cluster) / len(cluster) for k in range(3)]
            tex = [a.texture for a in cluster]
            tex_med = int(round(st.median(tex)))
            rows.append(dict(
                nodule_uid=f"{s.patient_id}_n{ci:02d}",
                patient_id=s.patient_id,
                series_uid=s.series_instance_uid,
                n_readers=len(cluster),
                diam_med_mm=round(d_med, 2),
                diam_min_mm=round(min(diams), 2),
                diam_max_mm=round(max(diams), 2),
                diam_spread_mm=round(max(diams) - min(diams), 2),
                diam_per_reader=";".join(f"{d:.2f}" for d in sorted(diams)),
                cen_i=round(cen[0], 2), cen_j=round(cen[1], 2), cen_k=round(cen[2], 2),
                texture_med=tex_med,
                texture_label=TEXTURE_LABEL[tex_med],
                texture_per_reader=";".join(str(t) for t in sorted(tex)),
                subtlety_med=int(round(st.median([a.subtlety for a in cluster]))),
                malignancy_med=int(round(st.median([a.malignancy for a in cluster]))),
                calcification_med=int(round(st.median([a.calcification for a in cluster]))),
                spiculation_med=int(round(st.median([a.spiculation for a in cluster]))),
                pixel_spacing=s.pixel_spacing,
                slice_spacing=s.slice_spacing,
                slice_thickness=s.slice_thickness,
            ))

    rows.sort(key=lambda r: (r["patient_id"], -r["diam_med_mm"]))
    path = os.path.join(OUT, "candidate_nodules.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"分群共 {n_clusters} 個 → 排除 <{MIN_READERS} 位讀者 {n_lt3reader}、"
          f"共識徑 <{MIN_DIAM_MM}mm {n_lt3mm}")
    print(f"候選結節：{len(rows)} 顆，來自 {len({r['patient_id'] for r in rows})} 位病人")
    print(f"→ {path}")

    # 尺寸分層點名（任務書 §2 的四層）——先看母體夠不夠，不夠要提早講
    bands = [("<6", 0, 6), ("6-8", 6, 8), ("8-15", 8, 15), (">=15", 15, 1e9)]
    print("\n候選池尺寸分布（共識中位徑）：")
    for name, lo, hi in bands:
        sel = [r for r in rows if lo <= r["diam_med_mm"] < hi]
        pats = len({r["patient_id"] for r in sel})
        print(f"  {name:>5} mm : {len(sel):4d} 顆 / {pats:3d} 位病人")
    print("\n型態分布：", {k: sum(1 for r in rows if r["texture_label"] == k)
                          for k in sorted({r["texture_label"] for r in rows})})


if __name__ == "__main__":
    main()
