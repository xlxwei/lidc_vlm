#!/usr/bin/env python3
"""mini 批選題：從候選池抽 50 顆結節，配額與規則照任務書 `14` §2。

配額：<6mm×10、6–8mm×15、8–15mm×15、≥15mm×10（共識中位徑分層）。

## 排除（先於排序）
- `n_readers > 4`：分群把兩顆鄰近結節併在一起了，GT 不明確（全庫僅 1 顆）。
- `diam_spread_mm > SPREAD_MAX`：判讀者間徑差過大。**排除理由不是「難題不要」，
  而是主分析的自變項就是尺寸**——尺寸配對 Fisher 是主指標，GT 徑不可靠會直接
  污染分帶。母體 spread 中位 1.57mm、p90 4.81mm，門檻設 5.0mm 只砍掉尾巴。
  被砍掉的顆數寫進日誌，好讓 reviewer 自己判斷這個門檻有沒有偷偷挑軟柿子。

## 品質分層（band 內，全決定性、無隨機）
tier 0 = solid ＋ 4/4 共識；tier 1 = solid ＋ 3/4；tier 2 = 其餘（依 texture 由實質到 GGO）。
先用 tier 0 填，不夠才往下拿。

## band 內再做「徑向等距取樣」——**這一條是後來補的，不是原設計**
第一版只按品質排序，結果 <6mm band 選出來的十顆全擠在 5.3–6.0mm：
band 名義是 3–6mm，實際只測到最上緣。那等於沒有測到小結節。
改法沿用院內 `make_roi_patches.py` 對 foil 的既有做法（「依直徑排序後等距取樣：
讓 foil 的大小分布鋪開，而不是全部擠在 7mm」）——同樣的理由對陽性題一樣成立：
**尺寸配對 Fisher 是主指標，陽性題的徑分布決定這個分析解析得出什麼**。
band 內取 k 顆＝在該 tier 依徑排序後取 k 個等距分位，遇病人上限則往鄰近徑順延。

## 每病人上限 2 顆
兩輪貪婪：第一輪每位病人只取 1 顆，四個 band 都填過一次後才開第二輪。
理由：50 顆若擠在少數病人身上，題目之間就不獨立（同一台機器、同一次呼吸、
同一組血管）。先鋪開病人再補足配額，能在同樣配額下拿到最多的獨立掃描。

輸出 manifest/selected_nodules.csv（中選）＋ manifest/reserve_nodules.csv（同排序的備取，
供下載後發現 N=9 出界或渲染異常時**按序**替補——替補規則先寫死，避免事後挑題）。
"""
import csv
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
MAN = os.path.join(HERE, "manifest")

QUOTA = [("lt6", 3.0, 6.0, 10), ("6-8", 6.0, 8.0, 15),
         ("8-15", 8.0, 15.0, 15), ("ge15", 15.0, 1e9, 10)]
MAX_PER_PATIENT = 2
SPREAD_MAX = 5.0
RESERVE_PER_BAND = 8

TEXTURE_RANK = {"solid": 0, "part-solid/solid": 1, "part-solid": 2,
                "GGO/part-solid": 3, "GGO": 4}


def tier_of(r):
    """0 = solid，1+ = 非實質（依 texture 由實質到 GGO）。

    **不把 4/4 排在 3/4 前面**（第一版曾這樣做，是錯的）：任務書的門檻就是 ≥3/4，
    再加一層 4/4 偏好會偷偷縮小可選範圍——實測後果是 <6mm band 的四顆最小結節
    （4.72／4.77mm，都是 solid 3/4）永遠選不到，band 底部被自己的排序砍掉。
    尺寸涵蓋是主指標的自變項，比多一位讀者同意重要。
    n_readers 改為記錄欄位（共識強度）而非選題條件。
    """
    return TEXTURE_RANK[r["texture_label"]]


def sort_key(r):
    return (tier_of(r), float(r["diam_spread_mm"]), r["nodule_uid"])


def spread_pick(pool, k, used, cap):
    """在 pool（同 band）內依徑等距取 k 顆，遵守每病人上限。

    做法：把依徑排序後的名單切成 k 個分位，每個分位取最靠近該分位中心、
    且病人額度還夠的一顆。取不到就讓該分位落空，由呼叫端決定要不要放寬 tier。
    純決定性：同樣輸入永遠同樣輸出。
    """
    pool = sorted(pool, key=lambda r: float(r["diam_med_mm"]))
    if not pool or k <= 0:
        return []
    out, taken = [], set()
    for q in range(k):
        target = (q + 0.5) * len(pool) / k
        order = sorted(range(len(pool)), key=lambda i: abs(i + 0.5 - target))
        for i in order:
            if i in taken:
                continue
            r = pool[i]
            if used[r["patient_id"]] >= cap:
                continue
            taken.add(i)
            used[r["patient_id"]] += 1
            out.append(r)
            break
    return out


def band_of(d):
    for name, lo, hi in [(q[0], q[1], q[2]) for q in QUOTA]:
        if lo <= d < hi:
            return name
    return None


def main():
    rows = list(csv.DictReader(open(os.path.join(MAN, "candidate_nodules.csv"))))
    n0 = len(rows)

    drop_cluster = [r for r in rows if int(r["n_readers"]) > 4]
    rows = [r for r in rows if int(r["n_readers"]) <= 4]
    drop_spread = [r for r in rows if float(r["diam_spread_mm"]) > SPREAD_MAX]
    rows = [r for r in rows if float(r["diam_spread_mm"]) <= SPREAD_MAX]
    print(f"候選 {n0} → 排除分群可疑(n>4) {len(drop_cluster)}、"
          f"徑差>{SPREAD_MAX}mm {len(drop_spread)} → 可選 {len(rows)}")

    by_band = defaultdict(list)
    for r in rows:
        b = band_of(float(r["diam_med_mm"]))
        if b:
            by_band[b].append(r)
    for b in by_band:
        by_band[b].sort(key=sort_key)

    used = defaultdict(int)
    picked = {name: [] for name, _, _, _ in QUOTA}

    # 兩輪病人上限 × 逐 tier 放寬；每次都用徑向等距取樣填剩下的配額
    for cap in (1, MAX_PER_PATIENT):
        for name, _, _, quota in QUOTA:
            for tier in sorted({tier_of(r) for r in by_band[name]}):
                need = quota - len(picked[name])
                if need <= 0:
                    break
                have = {r["nodule_uid"] for r in picked[name]}
                pool = [r for r in by_band[name]
                        if tier_of(r) == tier and r["nodule_uid"] not in have]
                picked[name].extend(spread_pick(pool, need, used, cap))

    selected = [r for name, _, _, _ in QUOTA for r in picked[name]]
    for r in selected:
        r["band"] = band_of(float(r["diam_med_mm"]))

    # 備取：同排序、扣掉中選者，每 band 取前 N（替補時病人上限仍要現場檢查）
    sel_ids = {r["nodule_uid"] for r in selected}
    reserve = []
    for name, _, _, _ in QUOTA:
        cnt = 0
        for r in by_band[name]:
            if r["nodule_uid"] in sel_ids:
                continue
            r["band"] = name
            reserve.append(r)
            cnt += 1
            if cnt >= RESERVE_PER_BAND:
                break

    fields = list(selected[0].keys())
    for path, data in [("selected_nodules.csv", selected),
                       ("reserve_nodules.csv", reserve)]:
        with open(os.path.join(MAN, path), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(data)

    print(f"\n中選 {len(selected)} 顆 / {len({r['patient_id'] for r in selected})} 位病人"
          f"（備取 {len(reserve)}）")
    print(f"{'band':<8}{'配額':<6}{'實選':<6}{'徑範圍(mm)':<16}{'型態'}")
    print("-" * 72)
    for name, _, _, quota in QUOTA:
        p = picked[name]
        ds = [float(r["diam_med_mm"]) for r in p]
        tex = {}
        for r in p:
            tex[r["texture_label"]] = tex.get(r["texture_label"], 0) + 1
        rng = f"{min(ds):.1f}–{max(ds):.1f}" if ds else "—"
        flag = "" if len(p) == quota else "  ← 配額未滿！"
        print(f"{name:<8}{quota:<6}{len(p):<6}{rng:<16}"
              f"{', '.join(f'{k}×{v}' for k, v in sorted(tex.items()))}{flag}")

    dist = sorted(used.items(), key=lambda x: -x[1])
    print(f"\n每病人顆數：{sum(1 for _, v in dist if v == 1)} 位 1 顆、"
          f"{sum(1 for _, v in dist if v == 2)} 位 2 顆")
    series = {r["series_uid"] for r in selected}
    print(f"需下載 series：{len(series)}")
    with open(os.path.join(MAN, "series_to_download.txt"), "w") as f:
        f.write("\n".join(sorted(series)) + "\n")
    print(f"→ {MAN}/selected_nodules.csv, reserve_nodules.csv, series_to_download.txt")


if __name__ == "__main__":
    main()
