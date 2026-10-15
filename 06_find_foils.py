#!/usr/bin/env python3
"""LIDC 血管 foil 候選搜尋 —— 判準沿用院內 `find_large_foils.py`（距離轉換法）。

為什麼用 large 版的距離轉換、不用 vessel 版的連通元件：原作者已實測過連通元件法
對大血管無效（大血管與分支在 HU>−150 下連成樹狀，被 D_MAX／compactness 篩掉，
六個 exam 只出一個候選）。距離轉換的內切半徑＝局部粗細，與連通性無關，
在整個尺寸範圍都成立，所以這裡用它涵蓋 4.5–15mm 全段，而不是分兩支程式。

## 與院內版的三個差異（其餘判準逐項照抄）
1. **尺寸帶 4.5–15.0mm**（院內 large 版是 8–25mm，因為它只是補洞）。
   理由＝run 10 的硬規則：陰性題必須覆蓋陽性題的尺寸範圍。本批結節 4.8–29.8mm，
   所以 foil 要從 4.5mm 一路鋪到解剖上做得到的上限。
2. **排除距離對「該掃描的全部 ≥3mm 標註」計算，不只一顆 index 結節**。
   院內沙盒每個 exam 只有一顆 index 結節；LIDC 一次掃描平均 3.7 個標註，
   其中包含只有 1–2 位讀者認可的結節——那些沒進我們的陽性題，但**同樣不能當陰性題**。
   用 `scan.annotations`（全部讀者、全部 ≥3mm 標記）算距離，比院內版嚴格。
   未排除的兩類：<3mm 標記（尺寸上不可能長成 ≥4.5mm 的候選）與
   non-nodule 標記（那是放射科醫師已裁決「不是結節」的東西，本來就適合當陰性題）。
3. z 連續性檢查的 slice 距離改由 mm 換算（本批 pz 恆為 1.25mm，
   Z_CONT=8 ⇒ ±10mm，與院內完全相同；寫成換算式是為了將來 spacing 變動時不會靜默錯掉）。

**機器只出候選；是不是血管一律由 Wei 裁決。** 輸出三連圖 QA 頁（z−8 / z / z+8）。
"""
import csv
import os
import sys

import numpy as np
from scipy import ndimage as ndi
from PIL import Image, ImageDraw

import lidc_compat  # noqa: F401
import pylidc as pl
import lidc_render as R

HERE = os.path.dirname(os.path.abspath(__file__))
MAN = os.path.join(HERE, "manifest")
OUTCSV = os.path.join(MAN, "foil_candidates.csv")
SHEET = os.path.join(HERE, "qa", "qa_sheet_foils.png")

MIN_DIST_ANN_MM = 30.0
EDGE_MM = 6.0
Z_CONT_MM = 10.0
D_MIN, D_MAX = 4.5, 15.0
COMPACT_MIN = 0.65
TOP_K = 4
SEP_MM = 20.0
AIR_FRAC_MIN = 0.45
FOV_MM = 64.0
TILE, WL, WW = 168, -600, 1500


def lung_mask(vol):
    """vol [x,y,z] HU → 肺遮罩（含肺內血管，已填洞）。照抄院內版。"""
    air = vol < -400
    lab, n = ndi.label(air)
    if n == 0:
        return np.zeros_like(air)
    border = set(np.unique(np.concatenate([
        lab[0, :, :].ravel(), lab[-1, :, :].ravel(),
        lab[:, 0, :].ravel(), lab[:, -1, :].ravel()])))
    border.discard(0)
    sizes = ndi.sum(air, lab, range(1, n + 1))
    keep = [i + 1 for i, s in enumerate(sizes) if s > 150000 and (i + 1) not in border]
    if not keep:
        keep = [int(np.argmax(sizes)) + 1]
    m = np.isin(lab, keep)
    for k in range(m.shape[2]):
        m[:, :, k] = ndi.binary_fill_holes(m[:, :, k])
    return m


def scan_one(scan, pid):
    v = scan.to_volume(verbose=False)
    vol = v.transpose(1, 0, 2).astype(np.int32)        # → [x, y, z]
    nx, ny, nz = vol.shape
    px, pz = scan.pixel_spacing, scan.slice_spacing
    z_cont = int(round(Z_CONT_MM / pz))

    # 全部讀者的 ≥3mm 標註中心（pylidc [i,j,k] = [y,x,z] → 轉成 [x,y,z]）
    ann_xyz = np.array([[a.centroid[1], a.centroid[0], a.centroid[2]]
                        for a in scan.annotations], float)

    lm = lung_mask(vol)
    zs = np.nonzero(lm.any(axis=(0, 1)))[0]
    if zs.size == 0:
        del vol, lm
        return []
    er = int(round(EDGE_MM / px))
    struct = np.ones((2 * er + 1, 2 * er + 1), bool)
    half_fov = int(round(FOV_MM / 2 / px))
    found = []
    for z in range(zs.min() + z_cont + 2, zs.max() - z_cont - 2, 2):
        inner = ndi.binary_erosion(lm[:, :, z], struct)
        if not inner.any():
            continue
        tissue = (vol[:, :, z] > -150) & inner
        if not tissue.any():
            continue
        dt = ndi.distance_transform_edt(tissue) * px
        peak = (dt >= D_MIN / 2) & (dt == ndi.maximum_filter(dt, size=5))
        if not peak.any():
            continue
        for cx, cy in zip(*np.nonzero(peak)):
            cx, cy = int(cx), int(cy)
            r_in = float(dt[cx, cy])
            d_eq = 2 * r_in
            if not (D_MIN <= d_eq <= D_MAX):
                continue
            rr = int(round(2 * r_in / px))
            y0_, y1_ = max(cx - rr, 0), min(cx + rr + 1, tissue.shape[0])
            x0_, x1_ = max(cy - rr, 0), min(cy + rr + 1, tissue.shape[1])
            win = tissue[y0_:y1_, x0_:x1_]
            gy, gx = np.ogrid[y0_ - cx:y1_ - cx, x0_ - cy:x1_ - cy]
            dist = np.hypot(gy, gx) * px
            ann = (dist > r_in) & (dist <= 2 * r_in)
            if not ann.any():
                continue
            compact = 1.0 - float(win[ann].mean())
            if compact < COMPACT_MIN:
                continue
            if not (half_fov <= cx < nx - half_fov and half_fov <= cy < ny - half_fov):
                continue
            # 離「任何」標註都要 ≥30mm
            d_ann = 1e9
            if len(ann_xyz):
                d_ann = float(np.min(np.linalg.norm(
                    (ann_xyz - np.array([cx, cy, z])) * np.array([px, px, pz]),
                    axis=1)))
            if d_ann < MIN_DIST_ANN_MM:
                continue
            r = int(round(6.0 / px))
            if any(vol[cx - r:cx + r + 1, cy - r:cy + r + 1, z + dz].max() <= -150
                   for dz in (-z_cont, z_cont)):
                continue
            air_ok = True
            for dz in (-z_cont, 0, z_cont):
                w = vol[cx - half_fov:cx + half_fov, cy - half_fov:cy + half_fov, z + dz]
                if w.size == 0 or (w < -400).mean() < AIR_FRAC_MIN:
                    air_ok = False
                    break
            if not air_ok:
                continue
            found.append(dict(exam_id=pid, x=cx, y=cy, z=z, diam_mm=round(d_eq, 1),
                              compactness=round(compact, 3),
                              dist_ann_mm=round(d_ann, 1),
                              center_hu=int(vol[cx, cy, z]), spacing_xy=px,
                              half_px=half_fov, z_cont=z_cont,
                              series_uid=scan.series_instance_uid))
    del vol, lm
    # 每個 exam 取候選，彼此間隔 ≥20mm（避免同一條血管重複入選）。
    #
    # **依 compactness 全域排序是錯的**（第一版這樣寫，照抄院內 large 版）：
    # 院內那支程式的尺寸帶是 8–25mm，帶內排 compactness 沒問題；本版把帶放寬到
    # 4.5–15mm 之後，compactness 與管徑負相關——小血管斷面本來就比較圓——
    # 於是排序把大 foil 全部擠掉。實測 6 個 exam：判準本身找得到 14.3mm 的候選，
    # 但 TOP_K 取完只剩 4.7–5.9mm。那正是 run 9→10 翻盤的那個缺陷，只是換個成因。
    # 改成**先分尺寸帶、帶內再排 compactness**，每帶各取名額。
    kept = []
    for lo, hi, k_band in ((4.5, 6.0, 2), (6.0, 8.0, 2), (8.0, D_MAX, TOP_K)):
        band = sorted((c for c in found if lo <= c["diam_mm"] < hi),
                      key=lambda r: -r["compactness"])
        got = 0
        for c in band:
            if got >= k_band:
                break
            if all(np.linalg.norm(np.array([c["x"] - m["x"], c["y"] - m["y"],
                                            c["z"] - m["z"]])
                                  * np.array([px, px, pz])) >= SEP_MM for m in kept):
                kept.append(c)
                got += 1
    return kept


def make_sheet(rows, scans_by_pid):
    """三連圖：z−8 / z / z+8。管狀結構三張都在，結節會在兩側消失或縮小。"""
    cols = 4
    nrow = (len(rows) + cols - 1) // cols
    W, H = cols * TILE * 3, nrow * (TILE + 18)
    sheet = Image.new("RGB", (W, H), (10, 10, 10))
    d = ImageDraw.Draw(sheet)
    lo, hi = WL - WW / 2, WL + WW / 2
    by_exam = {}
    for i, r in enumerate(rows):
        by_exam.setdefault(r["exam_id"], []).append((i, r))
    for ex, items in sorted(by_exam.items()):
        vol = scans_by_pid[ex].to_volume(verbose=False).transpose(1, 0, 2).astype(np.int32)
        for i, r in items:
            cx, cy, cz, half = int(r["x"]), int(r["y"]), int(r["z"]), int(r["half_px"])
            zc = int(r["z_cont"])
            n = 2 * half + 1
            x0 = (i % cols) * TILE * 3
            y0 = (i // cols) * (TILE + 18)
            for j, dz in enumerate((-zc, 0, zc)):
                sub = R.crop(vol[:, :, cz + dz].T, cy - half, cx - half, n)
                g = (np.clip((sub - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
                im = Image.fromarray(g).convert("RGB").resize((TILE, TILE), Image.BILINEAR)
                dd = ImageDraw.Draw(im)
                c = TILE // 2
                col = (0, 230, 0) if dz == 0 else (90, 140, 90)
                dd.line((c, c - 14, c, c - 6), fill=col); dd.line((c, c + 6, c, c + 14), fill=col)
                dd.line((c - 14, c, c - 6, c), fill=col); dd.line((c + 6, c, c + 14, c), fill=col)
                dd.text((3, 3), f"{dz:+d}" if dz else "ctr", fill=(200, 200, 90))
                sheet.paste(im, (x0 + j * TILE, y0))
            d.text((x0 + 3, y0 + TILE + 3),
                   f"{r['cand_id']}  {r['diam_mm']}mm  cmp{r['compactness']}  HU{r['center_hu']}",
                   fill=(215, 215, 215))
        del vol
    os.makedirs(os.path.dirname(SHEET), exist_ok=True)
    sheet.save(SHEET)
    print("wrote", SHEET, sheet.size)


def main():
    done = {r["patient_id"] for r in
            csv.DictReader(open(os.path.join(MAN, "download_log.csv")))
            if r["http_status"] == "200"}
    sel = [r for r in csv.DictReader(open(os.path.join(MAN, "selected_nodules.csv")))
           if r["patient_id"] in done]
    if len(sys.argv) > 1:
        sel = sel[:int(sys.argv[1])]
    rows, scans_by_pid = [], {}
    for i, r in enumerate(sel, 1):
        scan = pl.query(pl.Scan).filter(
            pl.Scan.series_instance_uid == r["series_uid"]).first()
        scans_by_pid[r["patient_id"]] = scan
        got = scan_one(scan, r["patient_id"])
        rows.extend(got)
        print(f"[{i}/{len(sel)}] {r['patient_id']}: {len(got)} 個候選  "
              f"{' '.join(f'{g['diam_mm']}mm' for g in got)}", flush=True)
    for k, r in enumerate(rows, 1):
        r["cand_id"] = f"foil_{r['exam_id'].replace('LIDC-IDRI-','')}_{k:03d}"
    if not rows:
        print("no candidates")
        return
    with open(OUTCSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cand_id"] + [k for k in rows[0] if k != "cand_id"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUTCSV}  n={len(rows)}")

    import collections
    print("大小分布:", dict(sorted(collections.Counter(
        int(r["diam_mm"] // 1) for r in rows).items())))
    # run 10 硬規則的自動檢查
    nod = [float(r["diam_med_mm"]) for r in sel]
    fmax = max(r["diam_mm"] for r in rows)
    uncov = [d for d in nod if d > fmax]
    print(f"\n★ 尺寸涵蓋檢查（run 10 硬規則）")
    print(f"   結節 {min(nod):.1f}–{max(nod):.1f}mm ／ foil {min(r['diam_mm'] for r in rows):.1f}–{fmax:.1f}mm")
    print(f"   無陰性對照的結節：{len(uncov)}/{len(nod)} 顆（>{fmax:.1f}mm）"
          f" → 主分析須排除，寫入 protocol")
    make_sheet(rows, scans_by_pid)


if __name__ == "__main__":
    main()
