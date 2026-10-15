#!/usr/bin/env python3
"""redact_raw.py <dir> — 對 release 副本的 frontier_raw 做金鑰除污（原始檔不動）。
把 sk-…／AIza…／Bearer token／x-api-key 值全部換成 <REDACTED>；改動計數輸出。"""
import glob,re,sys,os
pats=[r'sk-[A-Za-z0-9]{8,}', r'AIza[A-Za-z0-9_-]{10,}',
      r'([Bb]earer\s+)[A-Za-z0-9._-]{12,}', r'("x-api-key"\s*:\s*")[^"]{8,}(")']
n=0
for f in glob.glob(os.path.join(sys.argv[1],"**","*.json"),recursive=True):
    s=open(f,encoding="utf-8",errors="ignore").read(); o=s
    s=re.sub(pats[0],"<REDACTED>",s); s=re.sub(pats[1],"<REDACTED>",s)
    s=re.sub(pats[2],r"\1<REDACTED>",s); s=re.sub(pats[3],r"\1<REDACTED>\2",s)
    if s!=o: open(f,"w",encoding="utf-8").write(s); n+=1
print(f"[redact_raw] {n} 檔已除污")
