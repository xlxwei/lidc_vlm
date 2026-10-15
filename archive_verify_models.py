#!/usr/bin/env python3
"""archive_verify_models.py — 權重歸檔前後的雜湊驗證。

用法：
    python3 archive_verify_models.py /Volumes/<行動硬碟>/lidc_models

做什麼：
  1. 從 results/*.csv 蒐集本研究實際使用的 model_sha256／mmproj_sha256（證物清單）。
  2. 對指定資料夾內所有 *.gguf／*.safetensors 檔算 sha256。
  3. 報告：每個證物雜湊「找到／缺少」；資料夾中多出的檔案另列。
  全部找到才會 exit 0——**驗證通過前不要刪本機原檔**。

注意：Lingshu-7B 走 transformers（safetensors 多分片），results CSV 記的是
合併摘要值（in_safetensors 佔位者除外）；分片模式下請整個 HF snapshot 資料夾
一起歸檔（含 config／tokenizer），本腳本會對資料夾內每個 *.safetensors 分片
逐一算雜湊列出，供人工比對 HF revision。
"""
import csv, hashlib, glob, os, sys

ROOT = os.path.dirname(os.path.abspath(__file__))

def expected():
    exp = {}
    for f in glob.glob(os.path.join(ROOT, "results", "*.csv")):
        try:
            for r in csv.DictReader(open(f)):
                for col in ("model_sha256", "mmproj_sha256"):
                    v = (r.get(col) or "").strip()
                    if v and v != "in_safetensors" and len(v) >= 32:
                        tag = (r.get("run_tag") or "?").split("_")[0]
                        exp.setdefault(v, set()).add(f"{tag}:{col}")
        except Exception:
            continue
    return exp

def sha256(p, buf=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while True:
            b = fh.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    target = sys.argv[1]
    exp = expected()
    print(f"[證物清單] {len(exp)} 個不重複雜湊（來自 results/*.csv）")
    files = [p for pat in ("**/*.gguf", "**/*.safetensors")
             for p in glob.glob(os.path.join(target, pat), recursive=True)]
    print(f"[掃描] {target}: {len(files)} 個權重檔，開始算 sha256（大檔要幾分鐘）…")
    found = {}
    for p in sorted(files):
        d = sha256(p)
        found[d] = p
        mark = "✔ 證物" if d in exp else "  （清單外）"
        print(f"  {d[:16]}…  {os.path.relpath(p, target)}  {mark}")
    missing = {d: tags for d, tags in exp.items() if d not in found}
    print()
    if missing:
        print(f"!! 缺少 {len(missing)} 個證物雜湊：")
        for d, tags in sorted(missing.items()):
            print(f"   {d[:16]}…  用於 {', '.join(sorted(tags))}")
        print("→ 驗證未通過，本機原檔不可刪。")
        sys.exit(1)
    print("✔ 全部證物雜湊皆在歸檔資料夾中。可安全刪除本機副本（建議仍留第二份）。")

if __name__ == "__main__":
    main()
