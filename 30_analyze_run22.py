#!/usr/bin/env python3
"""[run 22] prompt 敏感度分析（vault `62`）。只讀收尾 CSV，不產生新模型輸出。

四組 prompt：orig（主實驗 v1／v1swap，含 "low-dose"）、nd（v5nd：只拿掉 "low-dose"）、
term（v6term：再換選項名詞）、form（v7form：再換問句句型）。各兩種順序 NF／VF。
定義沿用 28_reanalysis_v10.py 與手稿 59：Wilson 95% CI；Fisher exact（discrimination）與
exact McNemar（順序效應）皆 Holm 校正，open-weight 七個一組、commercial 三個一組，各順序分開。

新 prompt 的答案取自回覆的第一個字母詞（與手稿「first word」判定一致）。第一個字不是
Nodule／Vessel 的回覆不自動歸類：記為未歸類、列在第 0 節、從分母剔除並標註。
1 主實驗的答案沿用 `answer` 欄（v0.10 已對 2,010 筆逐一核對第一個字）。
"""
import csv, os, re, math, collections
from scipy.stats import fisher_exact, binomtest

ROOT = os.path.dirname(os.path.abspath(__file__))
R = lambda p: list(csv.DictReader(open(os.path.join(ROOT, p)))) if os.path.exists(os.path.join(ROOT, p)) else []
SM = sorted(l.strip() for l in open(os.path.join(ROOT, "manifest/subset_v010_93.txt")) if l.strip())
assert len(SM) == 93
op, fr = R("results/lidc_roi_results.csv"), R("results/frontier_results.csv")
nop, nfr = R("results/run22_prompt_open.csv"), R("results/run22_prompt_frontier.csv")
truth = {r["patch_id"]: r["truth"] for r in op if r["run_tag"] == "Q4_LIDC_N9" and r["patch_id"] in SM}
assert len(truth) == 93 and collections.Counter(truth.values()) == {"nodule": 39, "vessel": 54}

OPEN = [("Q4", "Qwen3-VL 4B"), ("Q8", "Qwen3-VL 8B"), ("Q25", "Qwen2.5-VL 7B"), ("LS7", "Lingshu-7B"),
        ("GM3", "Gemma 3 4B"), ("M1", "MedGemma 1 4B"), ("M15", "MedGemma 1.5 4B")]
COM = [("GPT56", "GPT-5.6"), ("GM31P", "Gemini 3.1 Pro"), ("FAB5", "Claude Fable 5")]
NAME = dict(OPEN + COM)
SETS = [("orig", "原 prompt（low-dose）", "", "swap"), ("nd", "v5nd（拿掉 low-dose）", "nd", "ndswap"),
        ("term", "v6term（換選項名詞）", "term", "termswap"), ("form", "v7form（換句型）", "form", "formswap")]
SHA = {("orig", "NF"): "b046abfad45a", ("orig", "VF"): "b3a242d23748",
       ("nd", "NF"): "958c37a4c6de", ("nd", "VF"): "666ec3e5d79b",
       ("term", "NF"): "6d27680b9715", ("term", "VF"): "faf80a500d7c",
       ("form", "NF"): "a4dfa6ee5610", ("form", "VF"): "e258cc3fbe22"}
EXPECT_SNAP = {"GPT56": "gpt-5.6-sol", "GM31P": "gemini-3.1-pro-preview", "FAB5": "claude-fable-5"}


def wil(k, n, z=1.96):
    if n == 0:
        return "—"
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f"{k}/{n} ({100*p:.0f}; {100*max(0,c-h):.0f}–{100*min(1,c+h):.0f})"


def holm(ps):
    idx = sorted(range(len(ps)), key=lambda i: ps[i]); out = [0] * len(ps); m = len(ps); run = 0
    for r, i in enumerate(idx):
        run = max(run, min(1, (m - r) * ps[i])); out[i] = run
    return out


fp = lambda p: "—" if p is None else ("<.001" if p < .001 else (">.99" if p > .99 else (f"{p:.3f}" if p < .01 else f"{p:.2f}")).replace("0.", "."))


def first_word(s):
    m = re.search(r"[A-Za-z]+", s or "")
    return m.group(0) if m else ""


# ---- 載入答案 -------------------------------------------------------------------
odd, mism, shabad, incomplete, snapbad = [], [], [], [], []
A = {}          # (set, model, order) -> {pid: 'nodule'|'vessel'|None}
for sk, slab, sufN, sufV in SETS:
    for k, _ in OPEN + COM:
        is_open = k in dict(OPEN)
        for o, suf in (("NF", sufN), ("VF", sufV)):
            tag = f"{k}_LIDC_N9{suf}"
            src = (op if is_open else fr) if sk == "orig" else (nop if is_open else nfr)
            rows = {r["patch_id"]: r for r in src if r["run_tag"] == tag and r["rep"] == "1" and r["patch_id"] in truth}
            if not is_open and sk in ("term", "form"):
                continue                      # commercial 只跑 v5nd 一組（任務書 `62`）
            if len(rows) < 93:
                if rows or sk != "orig":
                    incomplete.append(f"{tag}: {len(rows)}/93")
                if len(rows) == 0:
                    continue
            a = {}
            for p, r in rows.items():
                if r["prompt_sha1"] != SHA[(sk, o)]:
                    shabad.append(f"{tag} {p} {r['prompt_sha1']}")
                if not is_open and sk != "orig" and r.get("model_snapshot") != EXPECT_SNAP[k]:
                    snapbad.append(f"{tag} {p} {r.get('model_snapshot')}")
                if sk == "orig":
                    a[p] = r["answer"] or None
                    continue
                fw = first_word(r["raw_output"])
                if fw.lower() in ("nodule", "vessel"):
                    a[p] = fw.lower()
                    if r["answer"] and r["answer"] != a[p]:
                        mism.append(f"{tag} {p}: 第一個字 {fw}，parse()={r['answer']}")
                else:
                    a[p] = None
                    odd.append(f"{tag} {p} [{truth[p]}] parse={r['answer'] or r['parse_status']} | {r['raw_output'][:300]}")
            A[(sk, k, o)] = a

print("run 22 prompt 敏感度分析　（只讀收尾 CSV；第一個字判定；Holm：open 七個一組、commercial 三個一組、各順序分開）")
print(f"新檔列數：open {len(nop)}，commercial {len(nfr)}")
print("\n== 0 品管 ==")
print(f"未完成的臂：{len(incomplete)}" + ("".join("\n  " + x for x in incomplete)))
print(f"prompt_sha1 不符：{len(shabad)}" + ("".join("\n  " + x for x in shabad[:20])))
print(f"commercial 模型識別碼不符：{len(snapbad)}" + ("".join("\n  " + x for x in snapbad[:20])))
print(f"第一個字與 parse() 不一致：{len(mism)}" + ("".join("\n  " + x for x in mism)))
print(f"第一個字不是 Nodule／Vessel（未歸類，全部列出）：{len(odd)}" + ("".join("\n  " + x for x in odd)))
uncls = collections.Counter()
for (sk, k, o), a in A.items():
    for p, v in a.items():
        if v is None:
            uncls[(sk, k, o)] += 1


# ---- 1 Table 2 -------------------------------------------------------------------
def table2(sk):
    res = {}
    for tier in (OPEN, COM):
        for o in ("NF", "VF"):
            rows = []
            for k, _ in tier:
                a = A.get((sk, k, o))
                if a is None:
                    continue
                nn = sum(1 for p in a if truth[p] == "nodule" and a[p]); nv = sum(1 for p in a if truth[p] == "vessel" and a[p])
                tp = sum(1 for p in a if truth[p] == "nodule" and a[p] == "nodule")
                tn = sum(1 for p in a if truth[p] == "vessel" and a[p] == "vessel")
                pf = fisher_exact([[tp, nn - tp], [nv - tn, tn]])[1]
                rows.append([k, tp, nn, tn, nv, pf])
            for r, h in zip(rows, holm([x[5] for x in rows])):
                res[(r[0], o)] = r[1:] + [h]
        flips = []
        for k, _ in tier:
            a, b = A.get((sk, k, "NF")), A.get((sk, k, "VF"))
            if a is None or b is None:
                continue
            both = [p for p in SM if a.get(p) and b.get(p)]
            n2v = sum(a[p] == "nodule" and b[p] == "vessel" for p in both)
            v2n = sum(a[p] == "vessel" and b[p] == "nodule" for p in both)
            flips.append([k, n2v, v2n, len(both), binomtest(min(n2v, v2n), n2v + v2n, .5).pvalue if n2v + v2n else 1.0])
        for f, h in zip(flips, holm([x[4] for x in flips])):
            res[(f[0], "flip")] = f[1:] + [h]
    return res


T = {sk: table2(sk) for sk, *_ in SETS}
print("\n== 1 每組 prompt 的 Table 2 ==")
for sk, slab, *_ in SETS:
    print(f"\n-- {slab} --")
    print(f"{'Model':16s} {'order':5s} {'Sensitivity':20s} {'Specificity':22s} {'Accuracy':20s} {'p disc':6s} | changed (N→V, V→N)  p order")
    for k, name in OPEN + COM:
        if k == "GPT56":
            print("  " + "-" * 60)
        for o in ("NF", "VF"):
            t = T[sk].get((k, o))
            if t is None:
                print(f"{name:16s} {o:5s} 無資料"); continue
            tp, nn, tn, nv, _, h = t
            u = uncls[(sk, k, o)]
            fl = ""
            if o == "NF" and (k, "flip") in T[sk]:
                n2v, v2n, nb, _, hf = T[sk][(k, "flip")]
                fl = f"{n2v+v2n} ({n2v}, {v2n}){'' if nb == 93 else f' [配對 {nb}]'}  {fp(hf)}"
            print(f"{name:16s} {o:5s} {wil(tp,nn):20s} {wil(tn,nv):22s} {wil(tp+tn,nn+nv):20s} {fp(h):6s} | {fl}"
                  + (f"   未歸類 {u}" if u else ""))


# ---- 2 原 prompt vs v5nd 逐題 ----------------------------------------------------
def diff(a, b, S):
    both = [p for p in S if a.get(p) and b.get(p)]
    x = sum(a[p] == "nodule" and b[p] == "vessel" for p in both)
    y = sum(a[p] == "vessel" and b[p] == "nodule" for p in both)
    return x, y, len(both)


print("\n== 2 原 prompt 與新 prompt 的逐題比較（同模型、同順序）==")
print("格式：不同題數 (原N→新V, 原V→新N) / 可配對題數。open-weight temperature 0、決定性，差異＝措辭的效果。")
for k, name in OPEN + COM:
    cells = []
    for sk in ("nd", "term", "form"):
        for o in ("NF", "VF"):
            a, b = A.get(("orig", k, o)), A.get((sk, k, o))
            if a is None or b is None:
                cells.append(f"{sk}-{o}: —"); continue
            x, y, n = diff(a, b, SM)
            cells.append(f"{sk}-{o}: {x+y} ({x}, {y})" + ("" if n == 93 else f"/{n}"))
    print(f"{name:16s} " + "  ".join(cells))

print("\n-- commercial：重測改答 vs 拿掉 low-dose 的改答（同 44 題，nodule-first）--")
for k, name in COM:
    r2 = {r["patch_id"]: r["answer"] for r in fr if r["run_tag"] == f"{k}_LIDC_N9" and r["rep"] == "2" and r["patch_id"] in truth}
    S44 = sorted(r2)
    a = A[("orig", k, "NF")]
    rx, ry, rn = diff(a, r2, S44)
    line = f"{name:16s} n={len(S44)} 重測改答 {rx+ry} ({rx}, {ry})"
    b = A.get(("nd", k, "NF"))
    if b is not None:
        x, y, n = diff(a, b, S44)
        X, Y, N = diff(a, b, SM)
        line += f" | 原→v5nd 改答 {x+y} ({x}, {y}) /{n}；全 93 題 {X+Y} ({X}, {Y}) /{N}"
        bv, av = A.get(("nd", k, "VF")), A.get(("orig", k, "VF"))
        if bv is not None:
            X, Y, N = diff(av, bv, SM)
            line += f"；vessel-first 全 93 題 {X+Y} ({X}, {Y}) /{N}"
    else:
        line += " | v5nd 無資料"
    print(line)

# ---- 3 主要發現是否重現 -----------------------------------------------------------
print("\n== 3 主要發現是否重現 ==")
print("discrimination＝Fisher Holm p<.05；大量改答＝改答 ≥25 題；方向一致＝少數方向為 0 題（手稿：all in one direction）")
ALLV = ["Q8", "GM3", "M15"]
for sk, slab, *_ in SETS:
    t = T[sk]
    disc = {tier: [NAME[k] for k, _ in tl if (k, "NF") in t and t[(k, "NF")][5] < .05] for tier, tl in (("open", OPEN), ("com", COM))}
    discV = {tier: [NAME[k] for k, _ in tl if (k, "VF") in t and t[(k, "VF")][5] < .05] for tier, tl in (("open", OPEN), ("com", COM))}
    big = [f"{NAME[k]} {t[(k,'flip')][0]+t[(k,'flip')][1]} ({t[(k,'flip')][0]}, {t[(k,'flip')][1]})"
           + ("" if min(t[(k, 'flip')][0], t[(k, 'flip')][1]) == 0 else " 雙向")
           for k, _ in OPEN if (k, "flip") in t and t[(k, "flip")][0] + t[(k, "flip")][1] >= 25]
    allv = []
    for k in ALLV:
        a = A.get((sk, k, "NF"))
        if a is None:
            allv.append(f"{NAME[k]} 無資料"); continue
        nv = sum(v == "vessel" for v in a.values())
        allv.append(f"{NAME[k]} {nv}/{len(a)} vessel")
    anyallv = [NAME[k] for k, _ in OPEN if A.get((sk, k, "NF")) and all(v == "vessel" for v in A[(sk, k, "NF")].values())]
    print(f"\n-- {slab} --")
    print(f"  nodule-first discrimination：open {len(disc['open'])}/7 {disc['open']}；commercial {len(disc['com'])} {disc['com']}")
    print(f"  vessel-first discrimination：open {len(discV['open'])}/7 {discV['open']}；commercial {len(discV['com'])} {discV['com']}")
    print(f"  改答 ≥25 題的 open-weight：{len(big)}/7 {big}")
    print(f"  原本全答 vessel 的三個（nodule-first）：{'；'.join(allv)}")
    print(f"  本組 nodule-first 全答 vessel 的 open-weight：{anyallv}")

# ---- 4 彙總表 --------------------------------------------------------------------
print("\n== 4 彙總：兩種順序之間改答題數 (N→V, V→N)；* = McNemar Holm p<.05 ==")
print(f"{'Model':16s}" + "".join(f"{lab.split('（')[0]:>20s}" for _, lab, *_ in SETS))
for k, name in OPEN + COM:
    cells = []
    for sk, *_ in SETS:
        f = T[sk].get((k, "flip"))
        cells.append("—" if f is None else f"{f[0]+f[1]} ({f[0]}, {f[1]}){'*' if f[4] < .05 else ''}")
    print(f"{name:16s}" + "".join(f"{c:>20s}" for c in cells))

# ---- 附：accuracy 與 commercial in_tokens --------------------------------------
print("\n== 附 A accuracy（答對題數／可歸類題數）==")
for k, name in OPEN + COM:
    cells = []
    for sk, *_ in SETS:
        for o in ("NF", "VF"):
            t = T[sk].get((k, o))
            cells.append("—" if t is None else f"{t[0]+t[2]}/{t[1]+t[3]}")
    print(f"{name:16s} " + "  ".join(f"{sk}-{o}:{c}" for (sk, o), c in zip([(s[0], o) for s in SETS for o in ("NF", "VF")], cells)))
print("\n== 附 B commercial in_tokens 與花費 ==")
for k, name in COM:
    for suf in ("nd", "ndswap"):
        x = [int(r["in_tokens"]) for r in nfr if r["run_tag"] == f"{k}_LIDC_N9{suf}" and r["in_tokens"]]
        c = sum(float(r["cost_usd"] or 0) for r in nfr if r["run_tag"] == f"{k}_LIDC_N9{suf}")
        fails = sum(r["parse_status"] == "api_fail" for r in nfr if r["run_tag"] == f"{k}_LIDC_N9{suf}")
        hard = sum(r["refusal_hard"] == "1" for r in nfr if r["run_tag"] == f"{k}_LIDC_N9{suf}")
        if x:
            print(f"{name:16s} {suf:7s} n={len(x)} in_tokens {min(x)}–{max(x)}  ${c:.2f}  api_fail {fails}  hard refusal {hard}")
        else:
            print(f"{name:16s} {suf:7s} 無資料")
tot = collections.defaultdict(float)
for r in nfr:
    tot[r["vendor"]] += float(r["cost_usd"] or 0)
print("本次花費（依廠商）：" + "，".join(f"{v} ${c:.2f}" for v, c in sorted(tot.items())) + f"；合計 ${sum(tot.values()):.2f}")
