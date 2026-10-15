#!/usr/bin/env python3
"""Table 3 (noise images and no image) and the first-word audit of replies to CT images.
Reads only the CSV files in results/. Hand classifications come from the *_hand_verdicts.csv
and noise_statement_verdicts.csv files; no reply is re-classified by this script."""
import csv, os, re, collections
ROOT=os.path.dirname(os.path.abspath(__file__)); R=lambda p: list(csv.DictReader(open(os.path.join(ROOT,p))))
OPEN=[('Q4','Qwen3-VL 4B'),('Q8','Qwen3-VL 8B'),('Q25','Qwen2.5-VL 7B'),('LS7','Lingshu-7B'),('GM3','Gemma 3 4B'),('M1','MedGemma 1 4B'),('M15','MedGemma 1.5 4B')]
COM=[('GPT56','GPT-5.6'),('GM31P','Gemini 3.1 Pro'),('FAB5','Claude Fable 5')]
op=R('results/lidc_roi_results.csv'); fr=R('results/frontier_results.csv'); cs=R('results/control_stimuli_results.csv')
hv={(r['run_tag'],r['patch_id']):r['hand_verdict'] for r in R('results/run20_noise_hand_verdicts.csv')}
ns=collections.Counter(r['run_tag'].split('_')[0] for r in R('results/noise_statement_verdicts.csv') if r['states_images_are_noise']=='Y')
ni={r['run_tag']:r['verdict'] for r in R('results/noimage_hand_verdicts.csv')}
def first(t):
    m=re.match(r"[\s\*\#\-–—:>\"'“”‘’\(\[]*([A-Za-z]+)",t or ""); return m.group(1).lower() if m else ""
print("== First-word audit, replies to CT images ==")
for label,rows in [("open-weight, both orders",[r for r in op if r['rep']=='1']),("commercial, both orders",[r for r in fr if r['rep']=='1']),("commercial, second submission",[r for r in fr if r['rep']=='2'])]:
    bad=[r for r in rows if first(r['raw_output'])!=(r['answer'] or '').lower() or first(r['raw_output']) not in ('nodule','vessel')]
    print(f"{label:32s} n={len(rows):5d}  first word differs from recorded answer: {len(bad)}")
print("\n== Table 3 ==")
fmt=lambda c: ", ".join(f"{k} {v}" for k,v in c.most_common())
for k,name in OPEN:
    a=collections.Counter(r['answer'] for r in cs if r['run_tag']==k+'_CTL_N9'); b=collections.Counter(r['answer'] for r in cs if r['run_tag']==k+'_CTL_N9swap')
    print(f"{name:16s} noise NF [{fmt(a)}]  noise VF [{fmt(b)}]  stated noise {ns.get(k,0)}/40  no image NF {ni[k+'_CTL_noimgfull_N9']}  VF {ni[k+'_CTL_noimgfull_N9swap']}")
for k,name in COM:
    a=collections.Counter(v for (t,p),v in hv.items() if t==k+'_LIDC_CTL'); b=collections.Counter(v for (t,p),v in hv.items() if t==k+'_LIDC_CTLswap')
    print(f"{name:16s} noise NF [{fmt(a)}]  noise VF [{fmt(b)}]  stated noise {ns.get(k,0)}/40  no image NF {ni[k+'_LIDC_CTLnoimg']}  VF {ni[k+'_LIDC_CTLnoimgswap']}")
tot=sum(1 for v in hv.values() if v in('nodule','vessel')); print(f"\nnoise replies with an answer: open-weight {sum(1 for r in cs if r['answer'] in('nodule','vessel'))}/{len(cs)}, commercial {tot}/{len(hv)}")
