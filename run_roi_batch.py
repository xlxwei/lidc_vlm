#!/usr/bin/env python3
"""[LIDC Phase 1 分支] oracle ROI 臂批次推論（MedGemma 1.5 4B BF16, llama.cpp Metal）。

規則（06 交接）：模型輸出不修不救——parse 不出來就記 parse_fail，那是資料。
每列都帶可重現性欄位（模型檔 sha256、llama.cpp 版本、完整指令、prompt sha256）。
逐題 append 寫入，中斷可續跑（已存在的 patch_id+run_tag 會跳過）。

用法：
  run_roi_batch.py --tag A_temp0            # 主 pass：temp 0、全部 patch、1 次
  run_roi_batch.py --tag B_temp08_r1 --temp 0.8 --seed 1 --subset stability.txt
"""
import argparse, csv, hashlib, os, re, subprocess, sys, time

# === LIDC Phase 1 分支（2026-08-15）===============================================
# 與院內凍結版的差異**只有路徑**：ROOT／PATCHES／OUTCSV 改指向 lidc_phase1，
# 另加 --out 參數。PROMPTS、_CTX、parse()、clean()、FIELDS 與推論流程逐字未動，
# prompt sha1 仍為 b046abfad45a（由 verify_prompt_sha() 在啟動時自我檢查）。
#
# 為什麼一定要分支：原版 OUTCSV 硬寫成院內的 pilot_windows/roi_results.csv。
# 直接跑會把 LIDC 結果寫進院內 CSV —— 違反「與院內資料的任何混放都禁止」。
ROOT = os.path.dirname(os.path.abspath(__file__))
PATCHES = os.path.join(ROOT, "patches", "roi_patches_gray_N9")
OUTCSV = os.path.join(ROOT, "results", "lidc_roi_results.csv")
DEFAULT_MODEL = os.path.expanduser("~/models/medgemma15/medgemma-1.5-4b-it-BF16.gguf")
DEFAULT_MMPROJ = os.path.expanduser("~/models/medgemma15/mmproj-BF16.gguf")

_CTX = ("These are {ns} consecutive axial slices (superior to inferior, 1.25 mm spacing) "
        "from a low-dose chest CT, cropped to a 64 mm field of view centred on one "
        "structure in the lung. ")
# 凍結 prompt 變體（2026-08-10）。v1 反推重建自 run 4 並實測重現其答案。
# v1swap＝選項順序對調（測順序偏置）；v2open＝開放式（測 2AFC 框架本身的誘導）。
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

LOGLINE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+ [IW] ")   # E level 保留——那是診斷資料
CHATTPL = re.compile(r"^(<start_of_turn>|<end_of_turn>|<\|im_start\|>|<\|im_end\|>|You are a helpful assistant|"
                     r"Hello(<end_of_turn>)?$|Hi there(<end_of_turn>)?$|"
                     r"How are you\?(<end_of_turn>)?$)")

FIELDS = ["patch_id", "run_tag", "rep", "truth", "size_mm", "stratum", "lung_rads",
          "answer", "est_diam_mm", "parse_status", "raw_output", "elapsed_s",
          "temp", "seed", "n_predict", "prompt_sha1", "model_sha256", "mmproj_sha256",
          "llama_version", "timestamp", "command"]


def sha256(path, cache={}):
    if path not in cache:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(1 << 22), b""):
                h.update(b)
        cache[path] = h.hexdigest()
    return cache[path]


def llama_version():
    try:
        out = subprocess.run(["llama-mtmd-cli", "--version"], capture_output=True,
                             text=True, timeout=30)
        m = re.search(r"version:\s*(\S+\s*\(\S+\))", out.stdout + out.stderr)
        return m.group(1) if m else "unknown"
    except Exception:
        return "unknown"


def clean(text):
    keep = [ln for ln in text.splitlines()
            if ln.strip() and not LOGLINE.match(ln) and not CHATTPL.match(ln.strip())]
    return " ".join(keep).strip()


def parse(text):
    """回傳 (answer, est_diam_mm, status)。不做任何補救。"""
    if not text:
        return "", "", "parse_fail_empty"
    # 同義詞家族：以最早出現者為準。nodule 家族含 mass／tumor 等；
    # vessel 家族含 artery／vein——v2open 不含 vessel 字樣的答案才 parse 得到。
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--temp", default="0")
    ap.add_argument("--seed", default="1")
    ap.add_argument("--n", default="128")
    ap.add_argument("--subset", help="每行一個 patch_id 的檔案；預設跑全部")
    ap.add_argument("--rep", default="1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--variant", default="v1", choices=sorted(PROMPTS))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--mmproj", default=DEFAULT_MMPROJ)
    ap.add_argument("--patch-root", default=PATCHES)
    ap.add_argument("--meta", default=os.path.join(ROOT, "manifest", "patch_meta.csv"),
                    help="中繼資料 CSV（預設＝原 124 題）；新題組用自己的檔，不動凍結檔")
    ap.add_argument("--extra", default="", help="附加 llama-mtmd-cli 參數（空白分隔）")
    ap.add_argument("--nslices", type=int, default=5)
    ap.add_argument("--out", default=OUTCSV, help="結果 CSV（LIDC 與院內必須分開）")
    a = ap.parse_args()
    outcsv = a.out
    os.makedirs(os.path.dirname(outcsv), exist_ok=True)
    if "/Volumes/medai_ssd" in os.path.abspath(outcsv):
        sys.exit("拒絕執行：輸出路徑指向院內 SSD，LIDC 結果不得與院內混放。")

    prompt = PROMPTS[a.variant].replace("{ns}", str(a.nslices))
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

    ver = llama_version()
    msha, psha = sha256(a.model), sha256(a.mmproj)
    prompt_sha = hashlib.sha1(prompt.encode()).hexdigest()[:12]
    todo = [r for r in rows if (r["patch_id"], a.tag, a.rep) not in done]
    print(f"tag={a.tag} rep={a.rep} temp={a.temp} seed={a.seed} | "
          f"{len(todo)} to run ({len(rows)-len(todo)} already done)")
    t_all = time.time()
    for i, r in enumerate(todo, 1):
        d = os.path.join(a.patch_root, r["patch_id"])
        imgs = ",".join(os.path.join(d, f"roi_{k}.png") for k in range(1, a.nslices + 1))
        cmd = ["llama-mtmd-cli", "-m", a.model, "--mmproj", a.mmproj, "--image", imgs,
               "--temp", a.temp, "--seed", a.seed, "-n", a.n] \
              + (a.extra.split() if a.extra else []) + ["-p", prompt]
        t0 = time.time()
        raw = ""
        for attempt in (1, 2):     # llama 非零退出＝harness 故障，重試一次（模型輸出本身不修不救）
            try:
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
                raw = clean(p.stdout + "\n" + p.stderr)
                if p.returncode == 0:
                    break
                raw = f"[llama exit={p.returncode} attempt={attempt}] " + raw
            except subprocess.TimeoutExpired:
                raw = f"[timeout attempt={attempt}]"
            time.sleep(10)
        el = round(time.time() - t0, 1)
        ans, diam, status = parse(raw)
        with open(outcsv, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(dict(
                patch_id=r["patch_id"], run_tag=a.tag, rep=a.rep, truth=r["truth"],
                size_mm=r["size_mm"], stratum=r["stratum"], lung_rads=r["lung_rads"],
                answer=ans, est_diam_mm=diam, parse_status=status, raw_output=raw,
                elapsed_s=el, temp=a.temp, seed=a.seed, n_predict=a.n,
                prompt_sha1=prompt_sha, model_sha256=msha, mmproj_sha256=psha,
                llama_version=ver, timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                command=" ".join(cmd[:-1]) + " -p <prompt_sha1:" + prompt_sha + ">"))
        eta = (time.time() - t_all) / i * (len(todo) - i) / 60
        print(f"[{i}/{len(todo)}] {r['patch_id']:28s} {r['truth']:18s} "
              f"-> {ans or status:12s} {diam:>5}mm  {el}s  ETA {eta:.0f}min")
    print(f"done in {(time.time()-t_all)/60:.1f} min -> {outcsv}")


if __name__ == "__main__":
    main()
