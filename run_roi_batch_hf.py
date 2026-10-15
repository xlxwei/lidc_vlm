#!/usr/bin/env python3
"""[LIDC Phase 1 分支] ROI 臂批次推論 — transformers 路徑（給沒有可信 GGUF 的模型，本輪＝Lingshu-7B）。

為什麼不走 GGUF：社群轉檔若把視覺塔弄壞，症狀與本輪要檢驗的假說（醫療微調弄瞎模型）
完全同形，而且難以察覺。官方 safetensors 慢，但沒有這個混淆。

與 run_roi_batch.py 的關係：**同一份 FIELDS、同一個 parse()、同一組 prompt、同一個輸出檔**。
唯一差別是 runtime——`llama_version` 欄位改記 transformers/torch 版本字串
（欄位名沿用以免破壞既有分析腳本；此欄語意實為「runtime 版本」）。

model_sha256 慣例：對每個 *.safetensors 依檔名排序取 sha256，再對這些 digest 串接取 sha256。
mmproj_sha256＝"in_safetensors"（視覺塔不是獨立檔）。

用法：
  run_roi_batch_hf.py --tag LS7_N9 --model-dir ~/models/lingshu7b
  run_roi_batch_hf.py --tag LS7_N9_open --variant v2open --subset pilot_windows/subset_popen20.txt
"""
import argparse, csv, glob, hashlib, os, time

# === LIDC Phase 1 分支（2026-08-20, run 13）=======================================
# 與院內凍結版的差異**只有路徑**：ROOT／META／OUTCSV 改指向 lidc_phase1，另加 --out
# 與防呆。_CTX／PROMPTS／parse()／FIELDS／推論流程逐字未動（由 verify_frozen() 在
# 啟動時對院內原版做逐字元比對；prompt sha1 v1=b046abfad45a、v2open=86c22b3afbab、
# v3dens=6be9bb3cbea0 亦在啟動時自我檢查）。
#
# 為什麼一定要分支：原版 OUTCSV 硬寫成院內的 pilot_windows/roi_results.csv。
# 直接跑會把 LIDC 結果寫進院內 CSV —— 違反「與院內資料的任何混放都禁止」。
ROOT = os.path.dirname(os.path.abspath(__file__))
META = os.path.join(ROOT, "manifest", "patch_meta.csv")
OUTCSV = os.path.join(ROOT, "results", "lidc_roi_results.csv")

# prompt 與 run_roi_batch.py 逐字元相同——雜湊必須對得上（v1@9=b046abfad45a, v2open@9=86c22b3afbab）
_CTX = ("These are {ns} consecutive axial slices (superior to inferior, 1.25 mm spacing) "
        "from a low-dose chest CT, cropped to a 64 mm field of view centred on one "
        "structure in the lung. ")
PROMPTS = {
    "v1": _CTX + ("Is the structure at the centre of the images a pulmonary "
                  "nodule or a cross-section of a blood vessel? Answer with one word, Nodule or "
                  "Vessel, then one short sentence of reasoning, then the estimated diameter in mm."),
    "v1swap": _CTX + ("Is the structure at the centre of the images a cross-section of a "
                      "blood vessel or a pulmonary nodule? Answer with one word, Vessel or "
                      "Nodule, then one short sentence of reasoning, then the estimated diameter in mm."),
    "v2open": _CTX + ("What is the structure at the centre of the images? Answer with your "
                      "single best diagnosis, then one short sentence of reasoning, then the "
                      "estimated diameter in mm."),
    "v3dens": _CTX + ("Is the structure at the centre of the images air-filled (for example a "
                      "bronchus or a cyst) or solid soft tissue (for example a vessel or a nodule)? "
                      "Answer with one word, Air or Solid, then one short sentence of reasoning."),
    # v4dens（2026-08-24, run 15）——取代 v3dens。三選一 + 真空氣錨點；**刻意不給例子**，
    # 因為 v3dens 的「(for example a vessel or a nodule)」等於把類別→答案的對照表寫進題目，
    # 正好抵消掉「類別與密度解耦」的設計。選項用放射科慣用語（air / ground-glass / soft tissue）。
    "v4dens": _CTX + ("What is the attenuation of the structure at the centre of the images: "
                      "air density, ground-glass density, or soft-tissue density? Answer with one "
                      "term, Air, Ground-glass, or Soft tissue, then one short sentence of reasoning."),
    # v4dens_swap：**只有選項順序相反**，其餘逐字相同。用途與院內 v1swap 完全一樣——
    # 分辨「常數應答」是模型的判斷還是選項位置偏置（GM3 恆選最後一項、M1 恆選第一項，
    # 在只有一種順序時這兩件事無法分辨）。
    "v4dens_swap": _CTX + ("What is the attenuation of the structure at the centre of the images: "
                          "soft-tissue density, ground-glass density, or air density? Answer with one "
                          "term, Soft tissue, Ground-glass, or Air, then one short sentence of reasoning."),
    # v4dens_mark（2026-08-25, Wei）：影像上畫綠圈標出目標，題目改問「圈內」。
    # 與 v4dens 配對比較，差值＝有多少「密度失敗」其實是「定位失敗」。
    # 不縮 FOV 的理由見 14_build_density_v4.py stage_mark 的註解。
    "v4dens_mark": _CTX + ("A green circle marks the structure of interest. What is the "
                           "attenuation of the structure inside the green circle: air density, "
                           "ground-glass density, or soft-tissue density? Answer with one term, "
                           "Air, Ground-glass, or Soft tissue, then one short sentence of reasoning."),
    # v4dens_id（2026-08-25, Wei）：**在題目裡直接要模型先說出中央是什麼**，再答密度。
    # 解決的是同一件事（空氣格沒有預設主詞、題目沒定義），但用語言而不是像素——
    # 不動影像就沒有畫圈誘發的病灶先驗（實測：圈起來會讓 6mm 支氣管腔被說成 nodule）。
    # 副產品：定位變成可逐題計分的欄位，「說了 bronchus 卻答 soft tissue」變成可數的內部矛盾。
    # **已知混淆**：先命名可能讓模型從類別知識答密度（叫出 bronchus → 背出「支氣管是空氣」），
    # 而不是從影像讀。與 v4dens 配對，差值＝有多少正確率來自背知識而非看圖。
    "v4dens_id": _CTX + ("What is the structure at the centre of the images, and what is its "
                         "attenuation? Name the structure first, then give its attenuation as "
                         "one term — Air, Ground-glass, or Soft tissue — then one short "
                         "sentence of reasoning."),
}
# === run 22（2026-10-06，vault `62`）：prompt 敏感度 ======================================
# _CTX_ND＝_CTX 只拿掉 "low-dose "（48 個 scan 有 21 個管電流 ≥200 mA，原句對部分影像不成立）。
# v5nd＝v1 只換 _CTX_ND；v6term＝再換選項名詞；v7form＝再換問句句型。各有 swap。
# 全文與 sha1 以 manifest/run22_prompts.json 為準（ns=9）：
#   v5nd 958c37a4c6de  v5ndswap 666ec3e5d79b  v6term 6d27680b9715
#   v6termswap faf80a500d7c  v7form a4dfa6ee5610  v7formswap e258cc3fbe22
_CTX_ND = ("These are {ns} consecutive axial slices (superior to inferior, 1.25 mm spacing) "
           "from a chest CT, cropped to a 64 mm field of view centred on one "
           "structure in the lung. ")
_TAIL = ", then one short sentence of reasoning, then the estimated diameter in mm."
PROMPTS.update({
    "v5nd": _CTX_ND + ("Is the structure at the centre of the images a pulmonary nodule or a "
                       "cross-section of a blood vessel? Answer with one word, Nodule or Vessel" + _TAIL),
    "v5ndswap": _CTX_ND + ("Is the structure at the centre of the images a cross-section of a blood "
                           "vessel or a pulmonary nodule? Answer with one word, Vessel or Nodule" + _TAIL),
    "v6term": _CTX_ND + ("Is the structure at the centre of the images a lung nodule or a pulmonary "
                         "vessel? Answer with one word, Nodule or Vessel" + _TAIL),
    "v6termswap": _CTX_ND + ("Is the structure at the centre of the images a pulmonary vessel or a "
                             "lung nodule? Answer with one word, Vessel or Nodule" + _TAIL),
    "v7form": _CTX_ND + ("The structure at the centre of the images is either a pulmonary nodule or a "
                         "cross-section of a blood vessel. Which is it? Answer with one word, Nodule "
                         "or Vessel" + _TAIL),
    "v7formswap": _CTX_ND + ("The structure at the centre of the images is either a cross-section of a "
                             "blood vessel or a pulmonary nodule. Which is it? Answer with one word, "
                             "Vessel or Nodule" + _TAIL),
})

# 期望 sha1（prompt 以 ns=9 展開後前 12 碼）——與 llama.cpp 版共用同一組凍結字串
EXPECT_SHA = {"v1": "b046abfad45a", "v2open": "86c22b3afbab", "v3dens": "6be9bb3cbea0",
              "v4dens": "55349d61465d", "v4dens_swap": "4b1b2ed00d2a",
              "v4dens_mark": "193cc95df09a",
              "v4dens_id": "2eea817e0d9b",
              # run 22（vault `62`）
              "v5nd": "958c37a4c6de", "v5ndswap": "666ec3e5d79b",
              "v6term": "6d27680b9715", "v6termswap": "faf80a500d7c",
              "v7form": "a4dfa6ee5610", "v7formswap": "e258cc3fbe22"}

FIELDS = ["patch_id", "run_tag", "rep", "truth", "size_mm", "stratum", "lung_rads",
          "answer", "est_diam_mm", "parse_status", "raw_output", "elapsed_s",
          "temp", "seed", "n_predict", "prompt_sha1", "model_sha256", "mmproj_sha256",
          "llama_version", "timestamp", "command"]


def parse(text):
    """與 run_roi_batch.py 逐字元相同的凍結 parser。不做任何補救。"""
    import re
    if not text:
        return "", "", "parse_fail_empty"
    m = re.search(r"\b(nodule|mass|tumou?r|granuloma|carcinoma|metastas\w*|neoplas\w*|"
                  r"vessel|artery|arterial|vein|venous|vascular)\b", text, re.I)
    ans = ""
    if m:
        w = m.group(1).lower()
        ans = ("vessel" if w.startswith(("vessel", "arter", "vein", "venous", "vascul"))
               else "nodule")
    d = re.search(r"(\d+(?:\.\d+)?)\s*(?:mm|millimet)", text, re.I)
    diam = d.group(1) if d else ""
    if not ans:
        return "", diam, "parse_fail_no_class"
    if not diam:
        return ans, "", "ok_no_diam"
    return ans, diam, "ok"


ORIG = ("/Volumes/medai_ssd/medai/preprocessed/heatmap_v1/scripts/run_roi_batch_hf.py")


def verify_frozen():
    """啟動自檢：(a) prompt sha1 對得上；(b) _CTX／PROMPTS／parse()／FIELDS 與院內原版
    逐字元相同（院內 SSD 未掛載則只做 (a) 並印警告——不靜默略過）。"""
    for v, want in EXPECT_SHA.items():
        got = hashlib.sha1(PROMPTS[v].replace("{ns}", "9").encode()).hexdigest()[:12]
        if got != want:
            raise SystemExit(f"拒絕執行：prompt {v} sha1={got}，期望 {want}（凍結條件被破壞）")
    if not os.path.exists(ORIG):
        print("!! 院內原版不可讀（SSD 未掛載），跳過逐字元比對——只驗了 prompt sha1", flush=True)
        return
    import importlib.util, inspect
    spec = importlib.util.spec_from_file_location("_orig_hf", ORIG)
    o = importlib.util.module_from_spec(spec); spec.loader.exec_module(o)
    if o._CTX != _CTX:
        raise SystemExit("拒絕執行：_CTX 與院內原版不同")
    for k, v in o.PROMPTS.items():          # 原版三個變體必須逐字元相同（v3dens 為本分支新增）
        if PROMPTS.get(k) != v:
            raise SystemExit(f"拒絕執行：PROMPTS[{k}] 與院內原版不同")
    if o.FIELDS != FIELDS:
        raise SystemExit("拒絕執行：FIELDS 與院內原版不同")
    if inspect.getsource(o.parse) != inspect.getsource(parse):
        raise SystemExit("拒絕執行：parse() 與院內原版不同")
    print(f"凍結自檢通過：prompt sha1 ×{len(EXPECT_SHA)}；_CTX／PROMPTS(v1,v1swap,v2open)／"
          "parse()／FIELDS 與院內原版逐字元相同", flush=True)


def weights_sha(model_dir):
    h = hashlib.sha256()
    for p in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        f = hashlib.sha256()
        with open(p, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 22), b""):
                f.update(b)
        h.update(f.digest())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model-dir", default=os.path.expanduser("~/models/lingshu7b"))
    ap.add_argument("--model-name", default="lingshu-medical-mllm/Lingshu-7B")
    ap.add_argument("--variant", default="v1", choices=sorted(PROMPTS))
    ap.add_argument("--nslices", type=int, default=9)
    ap.add_argument("--patch-root", default=os.path.join(ROOT, "patches", "roi_patches_gray_N9"))
    ap.add_argument("--meta", default=META,
                    help="中繼資料 CSV（預設＝原 124 題）；新題組用自己的檔")
    ap.add_argument("--subset")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rep", default="1")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--out", default=OUTCSV, help="結果 CSV（LIDC 與院內必須分開）")
    a = ap.parse_args()

    outcsv = a.out
    os.makedirs(os.path.dirname(os.path.abspath(outcsv)), exist_ok=True)
    if "/Volumes/medai_ssd" in os.path.abspath(outcsv):
        raise SystemExit("拒絕執行：輸出路徑指向院內 SSD，LIDC 結果不得與院內混放。")
    verify_frozen()

    import torch, transformers
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    prompt = PROMPTS[a.variant].replace("{ns}", str(a.nslices))
    prompt_sha = hashlib.sha1(prompt.encode()).hexdigest()[:12]
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    runtime = f"transformers-{transformers.__version__}+torch-{torch.__version__}+{dev}"

    rows = [r for r in csv.DictReader(open(a.meta)) if r["render_ok"] == "1"]
    if a.subset:
        want = {ln.strip() for ln in open(a.subset) if ln.strip()}
        rows = [r for r in rows if r["patch_id"] in want]
    if a.limit:
        rows = rows[:a.limit]

    done = set()
    if os.path.exists(outcsv):
        for r in csv.DictReader(open(outcsv)):
            done.add((r["patch_id"], r["run_tag"], r["rep"]))
    else:
        with open(outcsv, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()
    todo = [r for r in rows if (r["patch_id"], a.tag, a.rep) not in done]

    print(f"{runtime} | tag={a.tag} variant={a.variant} prompt={prompt_sha} | "
          f"{len(todo)} to run ({len(rows)-len(todo)} already done)")
    print("hashing weights...", flush=True)
    msha = weights_sha(a.model_dir)
    print(f"model_sha256={msha[:16]}… loading...", flush=True)

    model = AutoModelForImageTextToText.from_pretrained(
        a.model_dir, dtype=torch.bfloat16).to(dev).eval()
    processor = AutoProcessor.from_pretrained(a.model_dir)
    cmd = (f"run_roi_batch_hf.py --tag {a.tag} --model-dir {a.model_dir} "
           f"--variant {a.variant} --nslices {a.nslices} --patch-root {a.patch_root} "
           f"greedy max_new_tokens={a.n} dtype=bf16 device={dev} "
           f"-p <prompt_sha1:{prompt_sha}>")

    t_all = time.time()
    for i, r in enumerate(todo, 1):
        d = os.path.join(a.patch_root, r["patch_id"])
        imgs = [Image.open(os.path.join(d, f"roi_{k}.png")) for k in range(1, a.nslices + 1)]
        messages = [{"role": "user",
                     "content": [*({"type": "image", "image": im} for im in imgs),
                                 {"type": "text", "text": prompt}]}]
        t0 = time.time()
        raw = ""
        try:
            inputs = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt").to(dev, dtype=torch.bfloat16)
            with torch.inference_mode():
                gen = model.generate(**inputs, max_new_tokens=a.n, do_sample=False)
            raw = processor.decode(gen[0][inputs["input_ids"].shape[-1]:],
                                   skip_special_tokens=True).strip().replace("\n", " ")
        except Exception as e:              # OOM／推論錯誤是資料，記下來繼續
            raw = f"[runtime_error] {type(e).__name__}: {str(e)[:300]}"
        el = round(time.time() - t0, 1)
        ans, diam, status = parse(raw)
        with open(outcsv, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(dict(
                patch_id=r["patch_id"], run_tag=a.tag, rep=a.rep, truth=r["truth"],
                size_mm=r["size_mm"], stratum=r["stratum"], lung_rads=r["lung_rads"],
                answer=ans, est_diam_mm=diam, parse_status=status, raw_output=raw,
                elapsed_s=el, temp="0", seed="greedy", n_predict=str(a.n),
                prompt_sha1=prompt_sha, model_sha256=msha, mmproj_sha256="in_safetensors",
                llama_version=runtime, timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                command=cmd))
        eta = (time.time() - t_all) / i * (len(todo) - i) / 60
        print(f"[{i}/{len(todo)}] {r['patch_id']:28s} {r['truth']:18s} "
              f"-> {ans or status:20s} {diam:>5}mm  {el}s  ETA {eta:.0f}min", flush=True)
    print(f"done in {(time.time()-t_all)/60:.1f} min -> {outcsv}")


if __name__ == "__main__":
    main()
