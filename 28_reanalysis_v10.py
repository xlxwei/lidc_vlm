#!/usr/bin/env python3
"""[v0.10] 回應外部審閱的重算，只讀收尾 CSV，不產生新模型輸出。
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
op=R('results/lidc_roi_results.csv'); fr=R('results/frontier_results.csv')
ref={r['patch_id']:r for r in op if r['run_tag']=='Q4_LIDC_N9' and r['patch_id'] in canon}
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
    a={r['patch_id']:r['answer'] for r in rows if r['run_tag']==k+('_LIDC_N9' if order=='NF' else '_LIDC_N9swap') and r['rep']==rep and r['patch_id'] in truth}
    return a
A={(k,o):ans(k,o) for k,_ in OPEN+COM for o in ('NF','VF')}
for v in A.values(): assert len(v)==93 and set(v.values())<={'nodule','vessel'},collections.Counter(v.values())

print("== B0 病人 ==")
pn={pat(p) for p in SM if truth[p]=='nodule'}; pv=collections.Counter(pat(p) for p in SM if truth[p]=='vessel')
print(f"nodule patients {len(pn)}, vessel patients {len(pv)}, overlap {len(pn&set(pv))}, unique total {len(pn|set(pv))}; vessels/patient {dict(collections.Counter(pv.values()))}")

print("\n== A 順序效應 ==")
rowsA=[]
for k,name in OPEN+COM:
    a,b=A[(k,'NF')],A[(k,'VF')]
    n2v=sum(a[p]=='nodule' and b[p]=='vessel' for p in SM); v2n=sum(a[p]=='vessel' and b[p]=='nodule' for p in SM)
    rowsA.append([k,name,n2v,v2n,binomtest(min(n2v,v2n),n2v+v2n,.5).pvalue if n2v+v2n else 1.0])
h10=holm([r[4] for r in rowsA]); ht=holm([r[4] for r in rowsA[:7]])+holm([r[4] for r in rowsA[7:]])
for r,a10,at in zip(rowsA,h10,ht):
    print(f"{r[1]:16s} changed {wil(r[2]+r[3],93):22s} N→V {r[2]:2d}  V→N {r[3]:2d}  McNemar {fp(r[4]):6s} Holm10 {fp(a10):6s} Holm-tier {fp(at)}")

print("\n== C accuracy CI + 原 Table 2（核對）==")
def table(vsel=None, quiet=False):
    S=[p for p in SM if truth[p]=='nodule']+[p for p in SM if truth[p]=='vessel' and (vsel is None or p in vsel)]
    out={}
    for tier in (OPEN,COM):
        for o in ('NF','VF'):
            res=[]
            for k,name in tier:
                a=A[(k,o)]; tp=sum(a[p]=='nodule' for p in S if truth[p]=='nodule'); tn=sum(a[p]=='vessel' for p in S if truth[p]=='vessel')
                nn=sum(truth[p]=='nodule' for p in S); nv=len(S)-nn
                res.append((k,name,tp,nn,tn,nv,fisher_exact([[tp,nn-tp],[nv-tn,tn]])[1]))
            for r,h in zip(res,holm([x[6] for x in res])): out[(r[0],o)]=r+(h,)
    return out
T=table()
for k,name in OPEN+COM:
    for o in ('NF','VF'):
        _,_,tp,nn,tn,nv,p,h=T[(k,o)]; print(f"{name:16s} {o} sens {wil(tp,nn):20s} spec {wil(tn,nv):22s} acc {wil(tp+tn,nn+nv):20s} Holm {fp(h)}")

print("\n== B 每位病人一個 vessel（35 個）==")
multi=[[p for p in SM if truth[p]=='vessel' and pat(p)==q] for q in pv]
ncomb=math.prod(len(m) for m in multi); print("組合數",ncomb)
random.seed(20261006); draws=[{random.choice(m) for m in multi} for _ in range(2000)]
first={sorted(m)[0] for m in multi}; T1=table(first)
agree=collections.Counter(); specr=collections.defaultdict(list)
for d in draws:
    Td=table(d)
    for key in T:
        agree[key]+= (Td[key][7]<.05)==(T[key][7]<.05); specr[key].append(Td[key][4])
for k,name in OPEN+COM:
    for o in ('NF','VF'):
        _,_,tp,nn,tn,nv,p,h=T1[(k,o)]; s=specr[(k,o)]
        print(f"{name:16s} {o} spec(first) {wil(tn,nv):22s} Holm {fp(h):6s} | 全93題 Holm {fp(T[(k,o)][7]):6s} | 2000 抽：spec 範圍 {min(s)}–{max(s)}/35，結論同全樣本 {100*agree[(k,o)]/2000:.1f}%")

print("\n== D base vs 醫療微調（答對與否，exact McNemar）==")
for base,ft in [('GM3','M1'),('GM3','M15'),('Q25','LS7')]:
    for o in ('NF','VF'):
        cb={p:A[(base,o)][p]==truth[p] for p in SM}; cf={p:A[(ft,o)][p]==truth[p] for p in SM}
        b=sum(cf[p] and not cb[p] for p in SM); c=sum(cb[p] and not cf[p] for p in SM)
        print(f"{dict(OPEN)[ft]:16s} vs {dict(OPEN)[base]:14s} {o}: acc {sum(cf.values())}/93 vs {sum(cb.values())}/93; 微調對而基座錯 {b}，基座對而微調錯 {c}；McNemar {fp(binomtest(min(b,c),b+c,.5).pvalue if b+c else 1)}")

print("\n== E commercial：同順序重測 vs 跨順序（同 50 題）==")
for k,name in COM:
    r2=ans(k,'NF','2'); S50=sorted(r2); a,b=A[(k,'NF')],A[(k,'VF')]
    rep=sum(a[p]!=r2[p] for p in S50); cro=sum(a[p]!=b[p] for p in S50)
    rn=sum(a[p]=='nodule' and r2[p]=='vessel' for p in S50); rv=rep-rn
    cn=sum(a[p]=='nodule' and b[p]=='vessel' for p in S50); cv=cro-cn
    print(f"{name:16s} n={len(S50)} (nod {sum(truth[p]=='nodule' for p in S50)}) 重測改答 {wil(rep,len(S50))} [N→V {rn}, V→N {rv}] | 跨順序改答 {wil(cro,len(S50))} [N→V {cn}, V→N {cv}]")

print("\n== F commercial in_tokens ==")
for k,name in COM:
    for o,suf in (('NF','_LIDC_N9'),('VF','_LIDC_N9swap')):
        t=[int(r['in_tokens']) for r in fr if r['run_tag']==k+suf and r['rep']=='1' and r['patch_id'] in truth and r['in_tokens']]
        print(f"{name:16s} {o} in_tokens n={len(t)} min {min(t)} max {max(t)}")
