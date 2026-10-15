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
