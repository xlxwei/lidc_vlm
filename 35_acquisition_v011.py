#!/usr/bin/env python3
"""[v0.11 m4] 48 個 scan 的影像參數（廠牌、機型、kernel、管電流、kVp、切片厚度）。讀 series_uid 對應的第一張 DICOM header。"""
import csv, os, glob, collections, statistics
import pydicom
ROOT=os.path.dirname(os.path.abspath(__file__))
canon=set(l.strip() for l in open(os.path.join(ROOT,'manifest/subset_v010_93.txt')) if l.strip())
meta=[r for r in csv.DictReader(open(os.path.join(ROOT,'manifest/patch_meta.csv'))) if r['patch_id'] in canon]
assert len(meta)==93, len(meta)
series={r['exam_id']:r['series_uid'] for r in meta}
idx={}
for d in glob.glob(os.path.join(ROOT,'data/LIDC-IDRI/*/*/*')):
    fs=sorted(glob.glob(os.path.join(d,'*.dcm')))
    if not fs: continue
    h=pydicom.dcmread(fs[0],stop_before_pixels=True)
    if h.SeriesInstanceUID in series.values(): idx[h.SeriesInstanceUID]=(fs,h)
rows=[]
for ex,uid in sorted(series.items()):
    fs,h=idx[uid]; mas=[]
    for f in fs[::max(1,len(fs)//10)]:
        g=pydicom.dcmread(f,stop_before_pixels=True)
        if 'XRayTubeCurrent' in g: mas.append(float(g.XRayTubeCurrent))
    rows.append(dict(exam=ex,man=str(h.get('Manufacturer','')),model=str(h.get('ManufacturerModelName','')),kernel=str(h.get('ConvolutionKernel','')),
        kvp=float(h.get('KVP',0) or 0),thick=float(h.get('SliceThickness',0) or 0),ma=statistics.median(mas) if mas else None))
print("n scans",len(rows))
for k in ('man','model','kernel','kvp','thick'): print(k,dict(collections.Counter(r[k] for r in rows)))
ma=[r['ma'] for r in rows if r['ma'] is not None]
print("mA median",statistics.median(ma),"IQR",statistics.quantiles(ma,n=4),"range",min(ma),max(ma),">=200:",sum(m>=200 for m in ma),"of",len(ma))
# Table 1：依結構（39 nodules、54 vessels）計算機型、kernel、管電流與直徑
info={r['exam']:r for r in rows}
for lab,keep in (('nodules',lambda t: t=='nodule'),('vessels',lambda t: t!='nodule')):
    ms=[m for m in meta if keep(m['truth'])]; sc=[info[m['exam_id']] for m in ms]
    sz=[float(m['size_mm']) for m in ms]; mas=[s['ma'] for s in sc]
    print(f"[{lab}] n {len(ms)} | model",dict(collections.Counter(s['model'] for s in sc)),"| kernel",dict(collections.Counter(s['kernel'] for s in sc)))
    print(f"[{lab}] mA median {statistics.median(mas)} IQR {statistics.quantiles(mas,n=4)} | diameter median {statistics.median(sz):.2f} IQR {[round(q,2) for q in statistics.quantiles(sz,n=4)]} range {min(sz)}–{max(sz)}")
    print(f"[{lab}] diameter bins 4.5–5.9/6.0–7.9/8.0–9.9/10.0–11.9/12.0–14.3:",[sum(lo<=s<hi for s in sz) for lo,hi in ((4.5,6),(6,8),(8,10),(10,12),(12,14.31))])
