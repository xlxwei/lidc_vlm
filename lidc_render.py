#!/usr/bin/env python3
"""凍結渲染路徑（run 9–11）的可重用版本 —— 邏輯逐行照抄，只把資料來源抽成參數。

原始碼：`heatmap_v1/scripts/make_roi_patches_multiN.py`（N=9 灰階臂）
        ＋ `make_roi_patches.py`（half_px 與 crop 規則）

**這個檔案不准有任何「順手改進」。** 幾何、視窗、內插、越界補值、dtype 全部維持原樣，
因為 run 9–11 的所有數字都綁在這條路徑上；改一個 bilinear 參數就不是複製了。
唯一的差異是 `vol` 從哪裡來（院內 nii.gz vs LIDC DICOM）。

正確性由 `04_verify_render_equivalence.py` 保證：把院內體積餵進本模組，
輸出必須與凍結的 `roi_patches_gray_N9/` **逐位元相同**。不通過就一張 LIDC 圖都不寫
（run 11 的前置硬閘門做法，沿用）。

軸序約定（與院內 ct_hu.nii.gz 相同，不是 pylidc 的）：
    vol[i, j, k]  i→x（DICOM column）、j→y（DICOM row）、k→z（slice）
    取軸切面時 vol[:, :, z].T ⇒ 列=y、行=x
pylidc 的 centroid 是 [row, col, slice] = [y, x, z]，與此**相反**——轉換在 05 做，
且有 center_hu 自動檢查把關（軸序錯 → 中心落在肺實質 ~−800 HU 而非軟組織）。
"""
import numpy as np
from PIL import Image

WL, WW = -600, 1500          # 肺窗灰階
SIZE = 448
FOV_MM = 64.0


def half_px(spacing_xy):
    """64mm FOV 的半徑（像素）。照抄 make_roi_patches.py。"""
    return int(round((FOV_MM / 2) / spacing_xy))


def crop(sl, r0, c0, n):
    """越界處以 -1000 HU（空氣）補邊——保持目標置中，勝過把目標推離中心。"""
    out = np.full((n, n), -1000, dtype=sl.dtype)
    rs, re_ = max(r0, 0), min(r0 + n, sl.shape[0])
    cs, ce = max(c0, 0), min(c0 + n, sl.shape[1])
    if rs < re_ and cs < ce:
        out[rs - r0:re_ - r0, cs - c0:ce - c0] = sl[rs:re_, cs:ce]
    return out


def render_patch(vol, cx, cy, cz, half, n_slices=9):
    """回傳 n_slices 張 448×448 RGB uint8 陣列（灰階複製三通道）。

    z 邊界規則照抄 multiN 版：z0 = cz - N//2，超出範圍就整個 patch 跳過
    （不補空氣幀——那會偽造「無連續性」訊號，比缺資料更糟）。
    回傳 None 代表跳過。
    """
    lo, hi = WL - WW / 2, WL + WW / 2
    nz = vol.shape[2]
    z0 = cz - n_slices // 2
    if z0 < 0 or z0 + n_slices > nz:
        return None
    n = 2 * half + 1
    out = []
    for i in range(n_slices):
        sub = crop(vol[:, :, z0 + i].T, cy - half, cx - half, n)
        g = (np.clip((sub - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
        im = Image.fromarray(np.stack([g, g, g], -1)).resize(
            (SIZE, SIZE), Image.BILINEAR)
        out.append(np.asarray(im))
    return out


def save_patch(imgs, outdir):
    import os
    os.makedirs(outdir, exist_ok=True)
    for i, a in enumerate(imgs, 1):
        Image.fromarray(a).save(os.path.join(outdir, f"roi_{i}.png"))


def center_ring_hu(vol, cx, cy, cz, half):
    """QC：中心 3mm 核心 HU vs 12–20mm 環狀背景 HU。照抄 make_roi_patches.py。

    這一對數字同時是**軸序的自動檢查**：結節中心該是軟組織（> −400 HU）。
    若 x/y 弄反，中心會落在隨機肺實質（~−800 HU），這個欄位會立刻叫出來。
    """
    n = 2 * half + 1
    sl0 = vol[:, :, cz].T
    core = crop(sl0, cy - half, cx - half, n)
    yy, xx = np.mgrid[0:n, 0:n]
    rr = np.hypot(yy - half, xx - half)
    px = FOV_MM / n
    center_hu = float(core[rr * px <= 3.0].mean())
    ring = core[(rr * px >= 12.0) & (rr * px <= 20.0)]
    return center_hu, (float(ring.mean()) if ring.size else float("nan"))
