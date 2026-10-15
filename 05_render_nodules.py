#!/usr/bin/env python3
"""渲染 50 顆 LIDC 結節的 N=9 灰階 patch，並逐顆做軸序與品質自動檢查。

渲染呼叫 `lidc_render.py`，該模組已通過 04 的逐位元閘門（1116/1116 與院內凍結
patch 相同），所以這裡的幾何／視窗／內插與 run 9–11 一致，不再重述。

## 軸序：pylidc → 院內慣例
pylidc 的體積是 `[row, col, slice]`＝`[y, x, z]`；院內 `ct_hu.nii.gz` 是 `[x, y, z]`。
故 `vol = pylidc_vol.transpose(1, 0, 2)`，且 `cx = cen_j`、`cy = cen_i`。

**這件事不靠推理背書，靠每一顆的實測**：把該結節的共識分割遮罩走完全相同的
crop 路徑，量遮罩重心離影像中心幾個像素。軸序對 → 0–2 px；軸序錯 → 遮罩根本
不在視野內。實測 17/17 顆轉置後偏移 0.5–1.9 px、不轉置 0/17 在視野內。
`mask_offset_px` 進 CSV，超過 `MAX_OFFSET_PX` 的一律 render_ok=0，不進批次。

## 為什麼還要 center_hu／ring_hu
沿用院內 `patches.csv` 的同名欄位，讓兩個資料集的 QC 欄可以直接並排。
另外它是第二道獨立的健全性檢查：結節中心的 3mm 核心 HU 應高於 12–20mm 環狀背景。

## air_frac：與類別相關的干擾變項，兩類都要量
foil 判準本來就要求 64mm FOV 內肺實質（HU<−400）佔比 ≥0.45，**結節沒有這個限制**
——胸膜下結節的視野裡本來就有一大片胸壁。實測 LIDC 14/50、院內 20/77 的結節
低於該門檻，而 foil 是 0/54 與 0/45。模型只要學會「視野裡有大片胸壁就答 nodule」
就能拿到看似鑑別力的分數。
故**兩類都用完全相同的量法記錄**，供 dry-run §9-D 預登記的敏感度分析
（兩類都限 ≥0.45 再算一次尺寸配對 Fisher）使用。
"""
import csv
import os

import numpy as np

import lidc_compat  # noqa: F401
import pylidc as pl
import lidc_render as R

HERE = os.path.dirname(os.path.abspath(__file__))
MAN = os.path.join(HERE, "manifest")
OUT = os.path.join(HERE, "patches", "roi_patches_gray_N9")
NSLICE = 9
MAX_OFFSET_PX = 10.0
Z_CONT = 8               # ±8 slice = ±10mm @1.25mm，與 foil 判準同


def air_fraction(vol, cx, cy, cz, half):
    """64mm FOV 內 HU<-400 的佔比，三切面取最小——與 06 的 foil 判準逐字同法。"""
    vals = []
    for dz in (-Z_CONT, 0, Z_CONT):
        z = cz + dz
        if not (0 <= z < vol.shape[2]):
            continue
        w = vol[cx - half:cx + half, cy - half:cy + half, z]
        if w.size:
            vals.append((w < -400).mean())
    return round(min(vals), 3) if vals else ""


def mask_of(scan, sel_row):
    """取離共識中心最近的那位讀者的分割遮罩（整卷布林，pylidc 座標系）。"""
    anns = [a for c in scan.cluster_annotations(verbose=False) for a in c]
    ci, cj, ck = (float(sel_row["cen_i"]), float(sel_row["cen_j"]),
                  float(sel_row["cen_k"]))
    a = min(anns, key=lambda a: np.hypot(a.centroid[0] - ci, a.centroid[1] - cj)
            + abs(a.centroid[2] - ck))
    M = np.zeros(scan.to_volume(verbose=False).shape, bool)
    M[a.bbox()] = a.boolean_mask()
    return M


def mask_offset(M, cx, cy, cz, half):
    n = 2 * half + 1
    sub = R.crop(M[:, :, cz].T.astype(np.int32), cy - half, cx - half, n)
    if sub.sum() == 0:
        return None
    yy, xx = np.nonzero(sub)
    return float(np.hypot(yy.mean() - half, xx.mean() - half))


def main():
    sel = list(csv.DictReader(open(os.path.join(MAN, "selected_nodules.csv"))))
    done = {r["patient_id"] for r in
            csv.DictReader(open(os.path.join(MAN, "download_log.csv")))
            if r["http_status"] == "200"}
    todo = [r for r in sel if r["patient_id"] in done]
    print(f"已下載 {len(todo)}/{len(sel)} 顆的影像")

    idx = []
    for i, r in enumerate(todo, 1):
        scan = pl.query(pl.Scan).filter(
            pl.Scan.series_instance_uid == r["series_uid"]).first()
        v = scan.to_volume(verbose=False)
        vol = v.transpose(1, 0, 2).astype(np.int32)     # → 院內慣例 [x, y, z]
        nz = vol.shape[2]
        half = R.half_px(float(r["pixel_spacing"]))
        cx, cy, cz = (int(round(float(r["cen_j"]))), int(round(float(r["cen_i"]))),
                      int(round(float(r["cen_k"]))))

        off = mask_offset(mask_of(scan, r).transpose(1, 0, 2), cx, cy, cz, half)
        imgs = R.render_patch(vol, cx, cy, cz, half, NSLICE)

        note, ok = "ok", 1
        if imgs is None:
            note, ok = f"z_out_of_range(z={cz},nz={nz})", 0
        elif off is None:
            note, ok = "mask_not_in_fov", 0
        elif off > MAX_OFFSET_PX:
            note, ok = f"mask_offset_{off:.1f}px", 0
        chu = rhu = float("nan")
        if imgs is not None:
            chu, rhu = R.center_ring_hu(vol, cx, cy, cz, half)
            # 邊界補值：patch 有沒有超出影像範圍
            n = 2 * half + 1
            pad = (max(0, half - cx) + max(0, half - cy)
                   + max(0, cx + half + 1 - vol.shape[0])
                   + max(0, cy + half + 1 - vol.shape[1]))
            if pad and ok:
                note = f"padded_{pad}px"
            if ok:
                R.save_patch(imgs, os.path.join(OUT, f"nod_{r['nodule_uid']}"))

        idx.append(dict(
            patch_id=f"nod_{r['nodule_uid']}", exam_id=r["patient_id"],
            truth="nodule", source="lidc_consensus",
            size_mm=r["diam_med_mm"], stratum=r["band"], lung_rads="",
            x=cx, y=cy, z=cz, spacing_xy=r["pixel_spacing"], half_px=half,
            compactness="", dist_from_gt_mm="",
            center_hu=round(chu, 1) if chu == chu else "",
            ring_hu=round(rhu, 1) if rhu == rhu else "",
            render_ok=ok, note=note,
            n_readers=r["n_readers"], texture=r["texture_label"],
            diam_spread_mm=r["diam_spread_mm"],
            mask_offset_px="" if off is None else round(off, 2),
            air_frac=air_fraction(vol, cx, cy, cz, half),
            n_slices_vol=nz, series_uid=r["series_uid"]))
        print(f"[{i}/{len(todo)}] {r['nodule_uid']:<22} d={float(r['diam_med_mm']):5.1f}mm "
              f"off={('—' if off is None else f'{off:4.1f}px')}  ctr{chu:7.0f} ring{rhu:7.0f}  {note}")
        del v, vol

    path = os.path.join(MAN, "nodule_patches.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(idx[0].keys()))
        w.writeheader()
        w.writerows(idx)
    okn = sum(r["render_ok"] for r in idx)
    offs = [r["mask_offset_px"] for r in idx if r["mask_offset_px"] != ""]
    print(f"\n渲染成功 {okn}/{len(idx)} → {OUT}")
    if offs:
        print(f"遮罩偏移：中位 {sorted(offs)[len(offs)//2]:.1f}px、最大 {max(offs):.1f}px "
              f"（門檻 {MAX_OFFSET_PX}px）")
    print(f"→ {path}")


if __name__ == "__main__":
    main()
