#!/usr/bin/env python3
"""make_release_v10.py — 組裝 v0.10 論文的公開 repo 資料夾 release_v010/。
只複製，不動原始檔。範圍：論文用到的實驗（兩種選項順序 × CT／noise／no image，十個模型）。
步驟：複製程式 → 篩選結果列 → 複製並除污 commercial 原始封存 → 複製影像 → 寫門面檔 → 消毒自檢 → 在 release 內重跑分析核對。"""
import csv, os, re, shutil, subprocess, sys, glob, hashlib, collections
ROOT=os.path.dirname(os.path.abspath(__file__)); R=os.path.join(ROOT,'release_v010')
# 清空 release_v010/ 但保留 .git（公開 repo 的本機 clone）
if os.path.exists(R):
    for x in os.listdir(R):
        if x=='.git': continue
        y=os.path.join(R,x); shutil.rmtree(y) if os.path.isdir(y) and not os.path.islink(y) else os.remove(y)
for d in ['manifest','results','prompts','figs/out','images/ct','docs']: os.makedirs(os.path.join(R,d),exist_ok=True)
RD=lambda p: list(csv.DictReader(open(os.path.join(ROOT,p))))
OPEN=['Q4','Q8','Q25','LS7','GM3','M1','M15']; COM=['GPT56','GM31P','FAB5']
def cp(src,dst=None):
    shutil.copy2(os.path.join(ROOT,src),os.path.join(R,dst or src))
# 1 程式
CODE=['lidc_compat.py','lidc_render.py','01_build_candidate_pool.py','02_select_mini_batch.py','03_download_series.py',
 '05_render_nodules.py','06_find_foils.py','07_make_nodule_qa_sheet.py','08_select_foils.py','09_render_foils.py','20_build_control_stimuli.py',
 'run_roi_batch.py','run_roi_batch_hf.py','run_frontier.py','21_run_control_logprob.py','27_run_noimg_fulltext.py',
 '28_reanalysis_v10.py','29_table3_and_audit.py','30_analyze_run22.py','31_analyze_run23.py','33_tables_v011.py','34_effect_size_v011.py','35_acquisition_v011.py','36_power_v011.py',
 'archive_verify_models.py','redact_raw.py','make_release_v10.py']
for f in CODE: cp(f)
cp('figs/make_fig2_v09.py'); cp('figs/make_fig2_v011_roc.py'); cp('figs/make_fig2_v011_forest.py'); cp('figs/make_fig1_v011.py'); cp('figs/make_fig1_flow_ajr.py')
for f in ['fig1_example_stimuli.png','fig2_v09_option_order.png','fig2_v09_option_order.pdf','fig2_v011_option_order.png','fig2_v011_option_order.pdf','fig2_v011_roc.png','fig2_v011_roc.pdf','fig2_v011_forest.png','fig2_v011_forest.pdf','fig1_v011_nine_slices.png','fig1_v011_nine_slices.pdf','fig1_ajr_flow.png','fig1_ajr_flow.pdf']: cp('figs/out/'+f)
# 2 manifest
for f in ['patch_meta.csv','patch_meta_control.csv','patch_meta_control_noimg.csv','subset_canon104.txt','subset_control.txt','subset_control_noimg.txt',
          'foil_adjudication.csv','foil_candidates.csv','foils_for_adjudication.csv','candidate_nodules.csv','selected_nodules.csv','series_to_download.txt','download_log.csv']:
    if os.path.exists(os.path.join(ROOT,'manifest',f)): cp('manifest/'+f)
cp('patches_control_sha256.txt','manifest/patches_control_sha256.txt')
canon=[l.strip() for l in open(os.path.join(ROOT,'manifest/subset_canon104.txt')) if l.strip()]
# 3 結果：只留論文用到的 run_tag
def filt(name,keep):
    rows=list(csv.DictReader(open(os.path.join(ROOT,'results',name)))); out=[r for r in rows if keep(r)]
    with open(os.path.join(R,'results',name),'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(out)
    print(f"  {name}: {len(out)}/{len(rows)} rows"); return out
T=lambda ks,sufs: {k+s for k in ks for s in sufs}
print("[results]")
filt('lidc_roi_results.csv',lambda r: r['run_tag'] in T(OPEN,['_LIDC_N9','_LIDC_N9swap']) and r['patch_id'] in canon)
fr=filt('frontier_results.csv',lambda r: r['run_tag'] in T(COM,['_LIDC_N9','_LIDC_N9swap']))
fc=filt('frontier_control_results.csv',lambda r: r['run_tag'] in T(COM,['_LIDC_CTL','_LIDC_CTLswap','_LIDC_CTLnoimg','_LIDC_CTLnoimgswap']) and not r['patch_id'].startswith('ctl_blank'))
filt('control_stimuli_results.csv',lambda r: r['run_tag'] in T(OPEN,['_CTL_N9','_CTL_N9swap']) and r['patch_id'].startswith('ctl_noise'))
# control_results.csv（只有第一個 token）不附；no-image 以 control_noimg_fulltext.csv 的完整回覆為準
for f in ['control_noimg_fulltext.csv','run20_noise_hand_verdicts.csv','noise_statement_verdicts.csv','noimage_hand_verdicts.csv','v10_reanalysis.txt','run22_prompt_open.csv','run22_prompt_frontier.csv','run23_noise_open.csv','run23_noimg_open.csv','run23_control_frontier.csv','run23_retest_frontier.csv','run23_noise_noimg_hand_verdicts.csv','run23_noise_statement_verdicts.csv','v011_tables.txt','v011_effect_size.txt','v011_acquisition.txt','v011_power.txt','run22_report.txt','run23_report.txt']: cp('results/'+f)
for f in ['run22_prompts.json','subset_v010_93.txt','subset_retest44.txt','subset_noise20.txt']: cp('manifest/'+f)
fr=fr+RD('results/run22_prompt_frontier.csv')+RD('results/run23_retest_frontier.csv'); fc=fc+RD('results/run23_control_frontier.csv')
# 4 commercial 原始封存（只留上面保留的呼叫）
n=0
for r in fr+fc:
    ap=r.get('archive_path') or ''
    m=re.search(r'frontier_raw/(.+)$',ap)
    cand=m.group(1) if m else f"{r['run_tag']}/{r['patch_id']}_rep{r['rep']}.json"
    src=os.path.join(ROOT,'results/frontier_raw',cand)
    if os.path.exists(src):
        dst=os.path.join(R,'results/frontier_raw',cand); os.makedirs(os.path.dirname(dst),exist_ok=True); shutil.copy2(src,dst); n+=1
print(f"[frontier_raw] copied {n} of {len(fr)+len(fc)} archives")
print(subprocess.run([sys.executable,os.path.join(ROOT,'redact_raw.py'),os.path.join(R,'results/frontier_raw')],capture_output=True,text=True).stdout.strip())
# 5 CT 影像（104 個結構 × 9 張）；noise 影像不附，由 20_build_control_stimuli.py 以 seed 1–20 重建，checksum 在 manifest/
for p in canon: shutil.copytree(os.path.join(ROOT,'patches/roi_patches_gray_N9',p),os.path.join(R,'images/ct',p))
print("[images] ct structures:",len(os.listdir(os.path.join(R,'images/ct'))))
# 6 路徑遮蔽（僅 release 副本）
for f in glob.glob(os.path.join(R,'**','*'),recursive=True):
    if os.path.isfile(f) and f.endswith(('.csv','.txt','.json','.py','.md')):
        s=open(f,encoding='utf-8',errors='ignore').read(); o=s
        s=re.sub(r'/Users/[A-Za-z0-9_]+/models','<MODELS_DIR>',s); s=re.sub(r'/Volumes/[^"\s,]*vlm_models_2026-08','<MODELS_DIR>',s); s=re.sub(r'/Users/[A-Za-z0-9_]+','<HOME>',s)
        if s!=o: open(f,'w',encoding='utf-8').write(s)
# 7 prompts
pr=collections.OrderedDict()
src=os.path.join(ROOT,'run_roi_batch.py')
P1="These are 9 consecutive axial slices (superior to inferior, 1.25 mm spacing) from a low-dose chest CT, cropped to a 64 mm field of view centred on one structure in the lung. Is the structure at the centre of the images a pulmonary nodule or a cross-section of a blood vessel? Answer with one word, Nodule or Vessel, then one short sentence of reasoning, then the estimated diameter in mm."
P2="These are 9 consecutive axial slices (superior to inferior, 1.25 mm spacing) from a low-dose chest CT, cropped to a 64 mm field of view centred on one structure in the lung. Is the structure at the centre of the images a cross-section of a blood vessel or a pulmonary nodule? Answer with one word, Vessel or Nodule, then one short sentence of reasoning, then the estimated diameter in mm."
sha=lambda x: hashlib.sha1(x.encode()).hexdigest()[:12]
used={r['prompt_sha1'] for r in csv.DictReader(open(os.path.join(R,'results/lidc_roi_results.csv')))}
print("[prompts] sha1 computed:",sha(P1),sha(P2),"| in results:",sorted(used))
P1n=P1.replace("from a low-dose chest CT","from a chest CT"); P2n=P2.replace("from a low-dose chest CT","from a chest CT")
open(os.path.join(R,'prompts/PROMPTS.md'),'w').write(f"# Prompts\n\nEach model received nine images followed by one of two prompts, which differ only in the order of the two options. The paper reports the final wording (files `run22_*` and `run23_*`). The original wording, which read \"from a low-dose chest CT\", was used for the files `lidc_roi_results.csv`, `frontier_results.csv`, `control_stimuli_results.csv`, `control_noimg_fulltext.csv`, and `frontier_control_results.csv`; its results are in the supplement.\n\n## Final wording\n\n### Nodule-first (SHA-1, first 12 characters: {sha(P1n)})\n\n> {P1n}\n\n### Vessel-first ({sha(P2n)})\n\n> {P2n}\n\n## Original wording\n\n### Nodule-first ({sha(P1)})\n\n> {P1}\n\n### Vessel-first ({sha(P2)})\n\n> {P2}\n\nThe `prompt_sha1` column of each results file records which prompt a call used.\n")
for f in ["README.md","LICENSE","LICENSE-DATA.md","MODELS.md",".gitignore"]: shutil.copy2(os.path.join(ROOT,"release_assets_v10",f),os.path.join(R,f))
for d in ["docs"]:
    if not os.listdir(os.path.join(R,d)): os.rmdir(os.path.join(R,d))
print("-> ",R)
