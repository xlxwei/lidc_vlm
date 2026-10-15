#!/usr/bin/env python3
"""結節 QA 頁：給 Wei 抽查 LIDC 共識標註的品質（任務書 §2 的第二個人工節點）。

每顆結節出三連圖（z−4 / z / z+4，即 N=9 patch 的首／中／末張），中心畫十字，
標註列出：共識中位徑、逐位讀者的徑、共識人數、texture 評分、遮罩偏移。

**為什麼三連圖而不是單張**：抽查要看的是「這顆到底是不是結節」，
而結節與血管最省事的區分就是 z 方向連續性——與 foil QA 頁同一個判準，
兩張表用同樣的視覺語言，Wei 可以用同一套眼睛看。

抽查對象預設為全部 50 顆（任務書只要求 20 顆；出全部不花額外時間，
Wei 要看幾顆自己決定）。排序依徑，方便一眼掃過尺寸梯度。
"""
import csv
import os

import numpy as np
from PIL import Image, ImageDraw

import lidc_compat  # noqa: F401
import pylidc as pl
import lidc_render as R

HERE = os.path.dirname(os.path.abspath(__file__))
MAN = os.path.join(HERE, "manifest")
PATCH = os.path.join(HERE, "patches", "roi_patches_gray_N9")
SHEET = os.path.join(HERE, "qa", "qa_sheet_nodules.png")
TILE = 168
COLS = 4


def main():
    rows = [r for r in csv.DictReader(open(os.path.join(MAN, "nodule_patches.csv")))
            if r["render_ok"] == "1"]
    sel = {r["nodule_uid"]: r for r in
           csv.DictReader(open(os.path.join(MAN, "selected_nodules.csv")))}
    rows.sort(key=lambda r: float(r["size_mm"]))
    print(f"出 {len(rows)} 顆")

    nrow = (len(rows) + COLS - 1) // COLS
    sheet = Image.new("RGB", (COLS * TILE * 3, nrow * (TILE + 30)), (10, 10, 10))
    d = ImageDraw.Draw(sheet)

    for i, r in enumerate(rows):
        pid = r["patch_id"]
        x0 = (i % COLS) * TILE * 3
        y0 = (i // COLS) * (TILE + 30)
        # 直接用已渲染好的 patch（保證看到的就是模型看到的那九張）
        for j, k in enumerate((1, 5, 9)):
            p = os.path.join(PATCH, pid, f"roi_{k}.png")
            im = Image.open(p).convert("RGB").resize((TILE, TILE), Image.BILINEAR)
            dd = ImageDraw.Draw(im)
            c = TILE // 2
            col = (0, 230, 0) if k == 5 else (90, 140, 90)
            dd.line((c, c - 14, c, c - 6), fill=col); dd.line((c, c + 6, c, c + 14), fill=col)
            dd.line((c - 14, c, c - 6, c), fill=col); dd.line((c + 6, c, c + 14, c), fill=col)
            dd.text((3, 3), {1: "-4", 5: "ctr", 9: "+4"}[k], fill=(200, 200, 90))
            sheet.paste(im, (x0 + j * TILE, y0))
        s = sel[pid.replace("nod_", "")]
        d.text((x0 + 3, y0 + TILE + 2),
               f"{pid.replace('nod_LIDC-IDRI-','')}  {r['size_mm']}mm  "
               f"{r['n_readers']}/4  {r['texture']}",
               fill=(215, 215, 215))
        d.text((x0 + 3, y0 + TILE + 15),
               f"讀者徑 {s['diam_per_reader']}  off{r['mask_offset_px']}px  "
               f"ctr{r['center_hu']}/ring{r['ring_hu']}",
               fill=(150, 150, 150))

    os.makedirs(os.path.dirname(SHEET), exist_ok=True)
    sheet.save(SHEET)
    print("wrote", SHEET, sheet.size)


if __name__ == "__main__":
    main()
