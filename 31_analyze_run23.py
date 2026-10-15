#!/usr/bin/env python3
"""run 23 分析：新 prompt（v5nd／v5ndswap）下的 noise、no image、commercial 重測。只讀收尾 CSV。
commercial 的 noise／no-image 回覆不自動判定，只給「初判」欄；正式判定由 Wei 完成後另存 CSV。"""
import csv, os, re, collections
ROOT=os.path.dirname(os.path.abspath(__file__)); R=lambda p: list(csv.DictReader(open(os.path.join(ROOT,p)))) if os.path.exists(os.path.join(ROOT,p)) else []
OPEN=[('Q4','Qwen3-VL 4B'),('Q8','Qwen3-VL 8B'),('Q25','Qwen2.5-VL 7B'),('LS7','Lingshu-7B'),('GM3','Gemma 3 4B'),('M1','MedGemma 1 4B'),('M15','MedGemma 1.5 4B')]
COM=[('GPT56','GPT-5.6'),('GM31P','Gemini 3.1 Pro'),('FAB5','Claude Fable 5')]
A=R('results/run23_noise_open.csv'); B=R('results/run23_noimg_open.csv'); C=R('results/run23_control_frontier.csv'); E=R('results/run23_retest_frontier.csv')
oldA=R('results/control_stimuli_results.csv'); oldB=R('results/control_noimg_fulltext.csv'); oldC=R('results/frontier_control_results.csv')
r22=R('results/run22_prompt_frontier.csv'); fr=R('results/frontier_results.csv')
hvC={(r['run_tag'],r['patch_id']):r['hand_verdict'] for r in R('results/run20_noise_hand_verdicts.csv')}
def first(t):
    m=re.match(r"[\s\*\#\-–—:>\"'“”‘’\(\[]*([A-Za-z]+)",t or ""); return m.group(1).lower() if m else ""
def prelim(t):
    """commercial 回覆初判（非正式）：run 20 的規則。拒答＝沒有以兩個選項之一開頭且含拒答／無影像字樣。"""
    f=first(t)
    if f in('nodule','vessel'): return f
    if re.search(r"can.t|cannot|unable|no image|not (been )?(provided|attached|received)|don.t see|didn.t receive|not able|insufficient",t or "",re.I): return 'declined?'
    return '?'
print("run 23 分析（新 prompt v5nd／v5ndswap）。列數：noise open",len(A),"| noimg open",len(B),"| control frontier",len(C),"| retest frontier",len(E))
print("\n== 0 稽核：第一個字 ==")
bad=[r for r in A+E if first(r['raw_output']) not in('nodule','vessel') or first(r['raw_output'])!=(r['answer'] or '').lower()]
print(f"A+E 共 {len(A)+len(E)} 筆，第一個字不是 Nodule/Vessel 或與 parser 不一致：{len(bad)}")
for r in bad: print("  ",r['run_tag'],r['patch_id'],'|',' '.join((r['raw_output'] or '').split())[:160])
bad=[r for r in B if first(r['raw_output']) not in('nodule','vessel')]
print(f"B 共 {len(B)} 筆，第一個字不是 Nodule/Vessel：{len(bad)}")
sha=collections.Counter(r['prompt_sha1'] for r in A+B+C+E); print("prompt_sha1:",dict(sha))
print("\n== 1 Open-weight noise（20 組）與 no image：新 prompt vs 舊 prompt ==")
for k,name in OPEN:
    for suf,osuf,nsuf,onsuf in [('nd','N9','N9nd','N9'),('ndswap','N9swap','N9ndswap','N9swap')]:
        new=collections.Counter(r['answer'] for r in A if r['run_tag']==f'{k}_CTL_N9{suf}')
        old=collections.Counter(r['answer'] for r in oldA if r['run_tag']==f'{k}_CTL_{osuf}' and r['patch_id'].startswith('ctl_noise'))
        nb=[first(r['raw_output']) for r in B if r['run_tag']==f'{k}_CTL_noimgfull_{nsuf}']; ob=[first(r['raw_output']) for r in oldB if r['run_tag']==f'{k}_CTL_noimgfull_{onsuf}']
        noise_stmt=sum(1 for r in A if r['run_tag']==f'{k}_CTL_N9{suf}' and re.search(r'nois',r['raw_output'] or '',re.I))
        print(f"{name:16s} {'NF' if suf=='nd' else 'VF'} noise new {dict(new)} | old {dict(old)} | 含nois字樣 {noise_stmt} | no image new {nb} old {ob}")
print("\n== 2 Commercial noise（20 組）與 no image（每序 10 次）：初判，非正式 ==")
for k,name in COM:
    for suf in ('nd','ndswap'):
        rows=[r for r in C if r['run_tag']==f'{k}_LIDC_CTL{suf}']; c=collections.Counter(prelim(r['raw_output']) for r in rows)
        old=collections.Counter(hvC.get((f"{k}_LIDC_CTL{'swap' if suf=='ndswap' else ''}",r['patch_id'])) for r in oldC if r['run_tag']==f"{k}_LIDC_CTL{'swap' if suf=='ndswap' else ''}" and r['patch_id'].startswith('ctl_noise'))
        ns=sum(1 for r in rows if re.search(r'nois',r['raw_output'] or '',re.I))
        ni=[r for r in C if r['run_tag']==f'{k}_LIDC_CTLnoimg{suf}']; cn=collections.Counter(prelim(r['raw_output']) for r in ni)
        print(f"{name:16s} {'NF' if suf=='nd' else 'VF'} noise n={len(rows)} 初判 {dict(c)} | 舊（Wei 判定）{dict(old)} | 含nois字樣 {ns} | no image n={len(ni)} 初判 {dict(cn)}")
print("\n== 3 Commercial 重測 44 題（新 prompt）：rep1 = run 22，rep2 = run 23 ==")
pm={r['patch_id']:r for r in R('manifest/patch_meta.csv')}
S44=[l.strip() for l in open(os.path.join(ROOT,'manifest/subset_retest44.txt')) if l.strip()]
for k,name in COM:
    out=[]
    for suf in ('nd','ndswap'):
        a={r['patch_id']:r['answer'] for r in r22 if r['run_tag']==f'{k}_LIDC_N9{suf}' and r['rep']=='1'}
        b={r['patch_id']:r['answer'] for r in E if r['run_tag']==f'{k}_LIDC_N9{suf}' and r['rep']=='2'}
        S=[p for p in S44 if p in a and p in b]; same=sum(a[p]==b[p] for p in S)
        nod=[p for p in S if pm[p]['truth']=='nodule']; ves=[p for p in S if pm[p]['truth']!='nodule']
        out.append(f"{'NF' if suf=='nd' else 'VF'} 相同 {same}/{len(S)}（nodule {sum(a[p]==b[p] for p in nod)}/{len(nod)}，vessel {sum(a[p]==b[p] for p in ves)}/{len(ves)}）")
    a={r['patch_id']:r['answer'] for r in r22 if r['run_tag']==f'{k}_LIDC_N9nd' and r['rep']=='1'}; b={r['patch_id']:r['answer'] for r in r22 if r['run_tag']==f'{k}_LIDC_N9ndswap' and r['rep']=='1'}
    cross=sum(a[p]==b[p] for p in S44 if p in a and p in b)
    old=[r for r in fr if r['rep']=='2' and r['run_tag']==f'{k}_LIDC_N9' and r['patch_id'] in S44]; o1={r['patch_id']:r['answer'] for r in fr if r['rep']=='1' and r['run_tag']==f'{k}_LIDC_N9'}
    oldsame=sum(o1[r['patch_id']]==r['answer'] for r in old)
    print(f"{name:16s} "+" | ".join(out)+f" | 兩序之間相同 {cross}/44 | 舊 prompt NF 重測相同 {oldsame}/{len(old)}")
print("\n== 4 費用 ==")
for k,name in COM:
    c=sum(float(r['cost_usd'] or 0) for r in C+E if r['run_tag'].startswith(k)); print(f"{name:16s} ${c:.2f}")
print(f"合計 ${sum(float(r['cost_usd'] or 0) for r in C+E):.2f}")
