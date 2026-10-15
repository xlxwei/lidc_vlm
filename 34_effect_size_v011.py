#!/usr/bin/env python3
"""[v0.11] 主要分析改為新 prompt（v5nd／v5ndswap，run 22）；重測與 noise／no image 來自 run 23，只讀收尾 CSV，不產生新模型輸出。
A 順序效應：改答數＋方向＋exact McNemar＋Holm（十模型一族／層內）
B 病人叢集：病人數、重疊；每位病人只留一個 vessel 的敏感度分析（全部組合或 2000 次隨機抽）
C accuracy 的 Wilson CI
D base vs 醫療微調：同一結構的答對與否做 exact McNemar
E commercial 重測（同順序 50 題）vs 跨順序（同 50 題）
F commercial in_tokens（九張影像是否進入）"""
import csv, os, re, math, random, collections, itertools
from scipy.stats import fisher_exact, binomtest
ROOT=os.path.dirname(os.path.abspath(__file__)); R=lambda p: list(csv.DictReader(open(os.path.join(ROOT,p))))
canon={l.strip() for l in open(os.path.join(ROOT,'manifest/subset_canon104.txt')) if l.strip()}
op0=R('results/lidc_roi_results.csv')
op=R('results/run22_prompt_open.csv'); fr=R('results/run22_prompt_frontier.csv')+R('results/run23_retest_frontier.csv')
ref={r['patch_id']:r for r in op0 if r['run_tag']=='Q4_LIDC_N9' and r['patch_id'] in canon}
SM=sorted(p for p in ref if 4.5<=float(ref[p]['size_mm'])<=14.3); assert len(SM)==93
truth={p:ref[p]['truth'] for p in SM}
_ex={r['patch_id']:r['exam_id'] for r in R('manifest/patch_meta.csv')}
pat=lambda p: _ex[p]
def wil(k,n,z=1.96):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return f"{k}/{n} ({100*p:.0f}; {100*max(0,c-h):.0f}–{100*min(1,c+h):.0f})"
def holm(ps):
    idx=sorted(range(len(ps)),key=lambda i:ps[i]); out=[0]*len(ps); m=len(ps); run=0
    for r,i in enumerate(idx): run=max(run,min(1,(m-r)*ps[i])); out[i]=run
    return out
fp=lambda p: "<.001" if p<.001 else (">.99" if p>.99 else (f"{p:.3f}" if p<.01 else f"{p:.2f}")).replace("0.",".")
OPEN=[('Q4','Qwen3-VL 4B'),('Q8','Qwen3-VL 8B'),('Q25','Qwen2.5-VL 7B'),('LS7','Lingshu-7B'),('GM3','Gemma 3 4B'),('M1','MedGemma 1 4B'),('M15','MedGemma 1.5 4B')]
COM=[('GPT56','GPT-5.6'),('GM31P','Gemini 3.1 Pro'),('FAB5','Claude Fable 5')]
def ans(k,order,rep='1'):
    rows=op if k in dict(OPEN) else fr
    a={r['patch_id']:r['answer'] for r in rows if r['run_tag']==k+('_LIDC_N9nd' if order=='NF' else '_LIDC_N9ndswap') and r['rep']==rep and r['patch_id'] in truth}
    return a
A={(k,o):ans(k,o) for k,_ in OPEN+COM for o in ('NF','VF')}
for v in A.values(): assert len(v)==93 and set(v.values())<={'nodule','vessel'},collections.Counter(v.values())


# [v0.11 M2] Youden's J 與 balanced accuracy，95% CI 用病人層級 cluster bootstrap（2000 次）；
# 基座 vs 微調的正確率差同樣用 cluster bootstrap。只讀收尾 CSV。
pts=sorted({pat(p) for p in SM}); byp=collections.defaultdict(list)
for p in SM: byp[pat(p)].append(p)
rng=random.Random(20261014); B=2000
boots=[[q for _ in pts for q in byp[rng.choice(pts)]] for _ in range(B)]
def J(a,S):
    nod=[p for p in S if truth[p]=='nodule']; ves=[p for p in S if truth[p]=='vessel']
    return sum(a[p]=='nodule' for p in nod)/len(nod)+sum(a[p]=='vessel' for p in ves)/len(ves)-1
pct=lambda xs:(sorted(xs)[int(.025*len(xs))],sorted(xs)[int(.975*len(xs))-1])
print("== Youden's J (balanced accuracy = (J+1)/2), patient-cluster bootstrap 95% CI ==")
for k,name in OPEN+COM:
    for o in ('NF','VF'):
        a=A[(k,o)]; j=J(a,SM); lo,hi=pct([J(a,b) for b in boots])
        print(f"{name:16s} {o} J {j:+.2f} ({lo:+.2f} to {hi:+.2f})  BA {100*(j+1)/2:.0f}% ({100*(lo+1)/2:.0f}–{100*(hi+1)/2:.0f})")
print("\n== accuracy difference fine-tune minus base, patient-cluster bootstrap 95% CI (percentage points) ==")
acc=lambda a,S: sum(a[p]==truth[p] for p in S)/len(S)
for base,ft in [('GM3','M1'),('GM3','M15'),('Q25','LS7')]:
    for o in ('NF','VF'):
        d=acc(A[(ft,o)],SM)-acc(A[(base,o)],SM); ds=[acc(A[(ft,o)],b)-acc(A[(base,o)],b) for b in boots]; lo,hi=pct(ds)
        print(f"{dict(OPEN)[ft]:16s} - {dict(OPEN)[base]:14s} {o}: {100*d:+.0f} ({100*lo:+.0f} to {100*hi:+.0f})")
