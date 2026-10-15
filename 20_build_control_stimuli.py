#!/usr/bin/env python3
"""run 19-A：無資訊基線刺激生成（blank／noise）。任務書 `36`，dry-run `37`。

目的：回答紅隊「常數應答 vs 純文字先驗不可分」。同凍結 prompt、同渲染規格，
但影像不含任何解剖資訊——模型若照樣給出與真影像臂相同的常數答案，
該常數**相容於**純文字先驗；若分布明顯不同，則影像確有進入判斷。

**凍結紀律**：
  - 不動 patches/ 底下任何東西；新影像一律寫進 patches_control/。
  - 影像規格由 lidc_render.save_patch 寫出（與 104 題**同一支存檔函式**），
    448×448 RGB uint8 PNG，9 張／刺激。灰階複製三通道，與真影像同慣例。
  - 決定論：seed 寫死在檔名裡，重跑必須逐位元相同（--verify 會檢查）。

刺激設計（寫死，不看數字前先定）：
  blank      uniform 128 灰 ×1 個刺激（9 張全同）
             ——128 在凍結肺窗（WL −600 / WW 1500）下 ≈ −597 HU，即視窗中點，
               不是「空氣黑」也不是「軟組織白」，是最中性的灰。
  noise_sNN  高斯噪聲 μ=128 σ=40，seed 1–20 ×20 個刺激
             ——σ=40 灰階 ≈ 235 HU；每個刺激的 9 張各自獨立取樣
               （rng = default_rng([seed, slice_index])），因為題目說的是
               「9 張連續切面」，切面之間本來就不該逐像素相同。

第三種刺激 `noimg`（完全不附影像）不在本檔——它沒有影像可生成，
由 21_run_control_logprob.py 以 llama-server 送出（llama-mtmd-cli 不帶 --image
會掉進互動模式而卡死，實測見 `37` §3.3）。

用法：
  20_build_control_stimuli.py            生成 ＋ 寫 manifest ＋ 寫 sha256
  20_build_control_stimuli.py --verify   只重算並與現有檔比對（不覆寫）
"""
import argparse, csv, hashlib, os, sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lidc_render as R                      # noqa: E402  ← 同一支 save_patch

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "patches_control")
META = os.path.join(ROOT, "manifest", "patch_meta_control.csv")
SHAS = os.path.join(ROOT, "patches_control_sha256.txt")
NSLICES = 9
BLANK_VAL = 128
NOISE_MU, NOISE_SIGMA, NOISE_SEEDS = 128.0, 40.0, range(1, 21)


def blank_imgs():
    a = np.full((R.SIZE, R.SIZE), BLANK_VAL, dtype=np.uint8)
    return [np.stack([a, a, a], -1) for _ in range(NSLICES)]


def noise_imgs(seed):
    out = []
    for k in range(1, NSLICES + 1):
        rng = np.random.default_rng([seed, k])
        g = np.clip(np.rint(rng.normal(NOISE_MU, NOISE_SIGMA, (R.SIZE, R.SIZE))),
                    0, 255).astype(np.uint8)
        out.append(np.stack([g, g, g], -1))
    return out


def stimuli():
    yield "ctl_blank", blank_imgs()
    for s in NOISE_SEEDS:
        yield f"ctl_noise_s{s:02d}", noise_imgs(s)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()

    rows, shalines, nfile = [], [], 0
    for pid, imgs in stimuli():
        d = os.path.join(OUT, pid)
        if a.verify:
            for i, arr in enumerate(imgs, 1):
                p = os.path.join(d, f"roi_{i}.png")
                if not os.path.exists(p):
                    sys.exit(f"缺檔：{p}")
                if np.array(Image.open(p)).tobytes() != arr.tobytes():
                    sys.exit(f"!! 不可重現：{p} 與重算結果不同")
                nfile += 1
        else:
            R.save_patch(imgs, d)
            for i in range(1, NSLICES + 1):
                p = os.path.join(d, f"roi_{i}.png")
                shalines.append(f"{sha256(p)}  {os.path.relpath(p, ROOT)}")
                nfile += 1
        kind = "blank" if pid.endswith("blank") else "noise"
        rows.append(dict(patch_id=pid, exam_id="CONTROL", truth=f"control_{kind}",
                         source="synthetic", size_mm="", stratum="control", lung_rads="",
                         center_hu="", ring_hu="", render_ok="1", air_frac="",
                         note=f"{kind} stimulus; no anatomy; see 20_build_control_stimuli.py"))

    if a.verify:
        print(f"重現性檢查：{nfile} 張逐位元相同 ✓")
        return

    with open(META, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    open(SHAS, "w").write("\n".join(shalines) + "\n")
    print(f"寫出 {len(rows)} 個刺激 / {nfile} 張 PNG -> {OUT}")
    print(f"manifest -> {META}")
    print(f"sha256   -> {SHAS}")


if __name__ == "__main__":
    main()
