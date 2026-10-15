#!/usr/bin/env python3
"""從 225 個 foil 候選挑出要給 Wei 裁決的一組，並出 QA 頁。

225 個全給 Wei 看不切實際（任務書給的人工預算是 ~50 個、30–45 分鐘）。
挑選規則全決定性，跑前寫死：

## 配額：依「結節在該帶的顆數」配，大帶給滿
本批結節分布 4.8–6mm ×10、6–8mm ×15、8–15mm ×15。
foil 配額 15 / 20 / 25 ＝ 每帶都比結節多，讓尺寸配對 Fisher 在每個帶都有分母。
**大帶（≥8mm）候選只有 49 個、≥10mm 只有 9 個**——那是解剖上的稀缺資源，
run 9→10 翻盤就是因為缺它，所以這一帶寧可多取。

## 帶內：徑向等距取樣 ＋ 每 exam 上限
等距取樣同 02 的理由（避免全擠在某個徑）。每 exam 上限 2 個，
理由：同一次掃描的兩段血管不是獨立樣本（同機器、同對比、同呼吸相）。

## 一併輸出 air_frac（肺實質佔比）
foil 判準本來就要求 ≥0.45，結節沒有這個限制——實測院內 20/77、LIDC 14/50
的結節低於 0.45。這是**與類別相關的干擾變項**，必須逐題記錄，
供預登記的敏感度分析（兩類都限 ≥0.45 再算一次）使用。見 dry-run §9-D。
"""
import csv
import os
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw

import lidc_compat  # noqa: F401
import pylidc as pl
import lidc_render as R

HERE = os.path.dirname(os.path.abspath(__file__))
MAN = os.path.join(HERE, "manifest")
SHEET = os.path.join(HERE, "qa", "qa_sheet_foils_selected.png")
BANDS = [(4.5, 6.0, 15), (6.0, 8.0, 20), (8.0, 15.0, 25)]
MAX_PER_EXAM = 2
TILE, WL, WW = 168, -600, 1500
COLS = 4


def spread_pick(pool, k, used, cap):
    pool = sorted(pool, key=lambda r: float(r["diam_mm"]))
    if not pool or k <= 0:
        return []
    out, taken = [], set()
    for q in range(k):
        target = (q + 0.5) * len(pool) / k
        for i in sorted(range(len(pool)), key=lambda i: abs(i + 0.5 - target)):
            if i in taken:
                continue
            r = pool[i]
            if used[r["exam_id"]] >= cap:
                continue
            taken.add(i)
            used[r["exam_id"]] += 1
            out.append(r)
            break
    return out


def main():
    cands = list(csv.DictReader(open(os.path.join(MAN, "foil_candidates.csv"))))
    print(f"候選 {len(cands)} 個")
    used = defaultdict(int)
    picked = []
    for lo, hi, quota in BANDS:
        pool = [r for r in cands if lo <= float(r["diam_mm"]) < hi]
        got = spread_pick(pool, quota, used, MAX_PER_EXAM)
        picked.extend(got)
        ds = [float(r["diam_mm"]) for r in got]
        print(f"  {lo}–{hi}mm: 候選 {len(pool):3d} → 取 {len(got):2d} "
              f"({min(ds):.1f}–{max(ds):.1f}mm)" if ds else f"  {lo}–{hi}mm: 無")

    # 量 air_frac（與 06 的判準同法）並附上，供敏感度分析
    by_exam = defaultdict(list)
    for r in picked:
        by_exam[r["exam_id"]].append(r)
    for ex, rs in sorted(by_exam.items()):
        scan = pl.query(pl.Scan).filter(
            pl.Scan.series_instance_uid == rs[0]["series_uid"]).first()
        vol = scan.to_volume(verbose=False).transpose(1, 0, 2).astype(np.int32)
        for r in rs:
            cx, cy, cz = int(r["x"]), int(r["y"]), int(r["z"])
            half, zc = int(r["half_px"]), int(r["z_cont"])
            vals = []
            for dz in (-zc, 0, zc):
                w = vol[cx - half:cx + half, cy - half:cy + half, cz + dz]
                if w.size:
                    vals.append((w < -400).mean())
            r["air_frac"] = round(min(vals), 3) if vals else ""
        del vol

    picked.sort(key=lambda r: float(r["diam_mm"]))
    out = os.path.join(MAN, "foils_for_adjudication.csv")
    fields = list(picked[0].keys())
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(picked)

    # 給 Wei 填的裁決檔（空 verdict 欄）
    adj = os.path.join(MAN, "foil_adjudication.csv")
    if not os.path.exists(adj):
        with open(adj, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["patch_id", "verdict", "note"])
            w.writerow(["# verdict 欄填 vessel / nodule / other；other＝不確定或非血管非結節",
                        "", ""])
            for r in picked:
                w.writerow([r["cand_id"], "", ""])
        print(f"→ {adj}（待 Wei 填 verdict）")

    print(f"\n選出 {len(picked)} 個 / {len(by_exam)} 個 exam → {out}")
    ds = [float(r["diam_mm"]) for r in picked]
    print(f"尺寸 {min(ds):.1f}–{max(ds):.1f}mm")
    af = [float(r["air_frac"]) for r in picked if r["air_frac"] != ""]
    print(f"air_frac 中位 {np.median(af):.2f}、最小 {min(af):.2f}")
    make_sheet(picked)


def make_sheet(rows):
    nrow = (len(rows) + COLS - 1) // COLS
    sheet = Image.new("RGB", (COLS * TILE * 3, nrow * (TILE + 30)), (10, 10, 10))
    d = ImageDraw.Draw(sheet)
    lo, hi = WL - WW / 2, WL + WW / 2
    idx = {r["cand_id"]: i for i, r in enumerate(rows)}
    by_exam = defaultdict(list)
    for r in rows:
        by_exam[r["exam_id"]].append(r)
    for ex, rs in sorted(by_exam.items()):
        scan = pl.query(pl.Scan).filter(
            pl.Scan.series_instance_uid == rs[0]["series_uid"]).first()
        vol = scan.to_volume(verbose=False).transpose(1, 0, 2).astype(np.int32)
        for r in rs:
            i = idx[r["cand_id"]]
            cx, cy, cz = int(r["x"]), int(r["y"]), int(r["z"])
            half, zc = int(r["half_px"]), int(r["z_cont"])
            n = 2 * half + 1
            x0, y0 = (i % COLS) * TILE * 3, (i // COLS) * (TILE + 30)
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
            d.text((x0 + 3, y0 + TILE + 2),
                   f"{r['cand_id']}  {r['diam_mm']}mm  cmp{r['compactness']}",
                   fill=(215, 215, 215))
            d.text((x0 + 3, y0 + TILE + 15),
                   f"HU{r['center_hu']}  air{r['air_frac']}  離標註{r['dist_ann_mm']}mm",
                   fill=(150, 150, 150))
        del vol
    os.makedirs(os.path.dirname(SHEET), exist_ok=True)
    sheet.save(SHEET)
    print("wrote", SHEET, sheet.size)


if __name__ == "__main__":
    main()
