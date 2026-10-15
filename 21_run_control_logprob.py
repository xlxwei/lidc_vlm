#!/usr/bin/env python3
"""run 19：llama-server 路徑的控制臂 runner。任務書 `36`，dry-run `37`。

三個用途，共用同一條 harness：
  probe    dry-run 探測：每模型少量題目，量出「答案首 token」的類別 token、
           覆蓋率、log-odds、與凍結 CSV 的 argmax 一致率、逐題耗時（→ ETA）。
  logprob  R19-B 全量：canonical 104 × v1／v1swap × 7 模型，出 log-odds(nodule)。
  noimg    R19-A 第三種刺激：payload 完全不附影像（llama-mtmd-cli 不帶 --image
           會掉進互動模式卡死，只能走 server）。

**為什麼是 llama-server 而不是凍結的 llama-mtmd-cli**：
  mtmd-cli 只吐文字，拿不到 logits。server 的 /v1/chat/completions 帶
  logprobs＋top_logprobs 才能把 2AFC 從 argmax 變連續量。
  代價＝換了推論入口，所以**每一次跑都必須同時記錄 argmax 並與凍結 CSV 對帳**，
  一致率是本臂的品管數字，寫進報表（`37` §3.2 實測：Q4 逐字元重現凍結輸出）。

**凍結紀律**：prompt 直接 import run_roi_batch.PROMPTS（同一個物件，非複製）；
結果一律寫 results/control_results.csv，**不碰** results/lidc_roi_results.csv
（4,010 的對帳數字不能被本臂動到）。

用法：
  21_run_control_logprob.py probe   --models Q4,M15
  21_run_control_logprob.py logprob --models all
  21_run_control_logprob.py noimg   --models all
"""
import argparse, base64, csv, hashlib, importlib.util, json, math, os
import subprocess, sys, time, urllib.error, urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTCSV = os.path.join(ROOT, "results", "control_results.csv")
CANON = os.path.join(ROOT, "manifest", "subset_canon104.txt")
META = os.path.join(ROOT, "manifest", "patch_meta.csv")
METAC = os.path.join(ROOT, "manifest", "patch_meta_control.csv")
FROZEN_CSV = os.path.join(ROOT, "results", "lidc_roi_results.csv")
PORT = 8099
TOPK = 100
NSLICES = 9


def _load(mod, path):
    spec = importlib.util.spec_from_file_location(mod, os.path.join(ROOT, path))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


RB = _load("rb", "run_roi_batch.py")          # 凍結 PROMPTS／parse()／sha256
PROMPTS = RB.PROMPTS

# (prefix, 顯示名, gguf 子路徑 或 None＝transformers)
MODELS = [("Q4", "Qwen3-VL 4B", "qwen3vl4b/Qwen3-VL-4B-Instruct-BF16.gguf"),
          ("Q8", "Qwen3-VL 8B", "qwen3vl8b/Qwen3-VL-8B-Instruct-BF16.gguf"),
          ("Q25", "Qwen2.5-VL 7B", "qwen25vl7b/Qwen2.5-VL-7B-Instruct-BF16.gguf"),
          ("GM3", "Gemma 3 4B", "gemma3_4b/gemma-3-4b-it-BF16.gguf"),
          ("M1", "MedGemma 1 4B", "medgemma1_4b/medgemma-4b-it-BF16.gguf"),
          ("M15", "MedGemma 1.5 4B", "medgemma15/medgemma-1.5-4b-it-BF16.gguf"),
          ("LS7", "Lingshu-7B", None)]
ORDERS = [("v1", "N9"), ("v1swap", "N9swap")]

FIELDS = ["patch_id", "run_tag", "rep", "stimulus", "variant", "order_label",
          "answer", "argmax_token", "argmax_logprob", "p_nodule", "p_vessel",
          "logodds_nodule", "covered", "class_mass", "nod_tokens", "ves_tokens",
          "raw_first_token", "elapsed_s", "harness", "top_k", "prompt_sha1",
          "model_sha256", "engine", "timestamp"]


# ---------------------------------------------------------------- token 分類
def token_class(tok):
    """凍結規則（`37` §4 寫死，跑之前定）：去頭尾空白後，**大小寫不敏感地**
    當作 nodule／vessel 的前綴才算數。'N'/'No'/'Nod'/'nodule' → nodule；
    'V'/'Ves'/'vessel' → vessel；'Node'／'Vessels' 等非前綴 → 不計。
    空字串（純空白 token）不計。"""
    s = tok.strip().lower()
    if not s:
        return None
    if "nodule".startswith(s):
        return "nodule"
    if "vessel".startswith(s):
        return "vessel"
    return None


def score_top(tops):
    """tops = [(token, logprob)] → (p_nod, p_ves, nod_toks, ves_toks, class_mass)"""
    pn = pv = 0.0
    nt, vt = [], []
    for tok, lp in tops:
        c = token_class(tok)
        if c == "nodule":
            pn += math.exp(lp); nt.append(tok)
        elif c == "vessel":
            pv += math.exp(lp); vt.append(tok)
    return pn, pv, nt, vt, pn + pv


def logodds(pn, pv):
    if pn <= 0 and pv <= 0:
        return float("nan")
    return math.log(max(pn, 1e-300)) - math.log(max(pv, 1e-300))


# ---------------------------------------------------------------- llama-server
class Server:
    def __init__(self, gguf, port=PORT):
        self.gguf = os.path.expanduser(f"~/models/{gguf}")
        self.mmproj = os.path.join(os.path.dirname(self.gguf), "mmproj-BF16.gguf")
        self.port = port
        self.proc = None

    def __enter__(self):
        log = open(os.path.join(ROOT, "logs_run19_server.txt"), "a")
        log.write(f"\n=== {time.strftime('%F %T')} start {self.gguf}\n"); log.flush()
        self.proc = subprocess.Popen(
            ["llama-server", "-m", self.gguf, "--mmproj", self.mmproj, "-c", "6144",
             "--port", str(self.port), "--host", "127.0.0.1", "-np", "1"],
            stdout=log, stderr=log)
        for _ in range(180):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2)
                return self
            except Exception:
                time.sleep(2)
        self.__exit__(None, None, None)
        raise SystemExit(f"llama-server 起不來：{self.gguf}")

    def __exit__(self, *exc):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except Exception:
                self.proc.kill()
        time.sleep(2)

    def ask(self, images, prompt, max_tokens=1):
        content = [{"type": "image_url",
                    "image_url": {"url": "data:image/png;base64," +
                                  base64.b64encode(open(p, "rb").read()).decode()}}
                   for p in images]
        content.append({"type": "text", "text": prompt})
        body = {"messages": [{"role": "user", "content": content}], "temperature": 0,
                "max_tokens": max_tokens, "seed": 1, "logprobs": True,
                "top_logprobs": TOPK}
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=900))
        ch = r["choices"][0]
        tk = ch["logprobs"]["content"][0]
        tops = [(t["token"], t["logprob"]) for t in tk["top_logprobs"]]
        return ch["message"]["content"], tk["token"], tk["logprob"], tops


# ---------------------------------------------------------------- transformers (LS7)
class HFScorer:
    def __init__(self, model_dir=None):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        d = model_dir or os.path.expanduser("~/models/lingshu7b")
        self.dev = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = AutoModelForImageTextToText.from_pretrained(
            d, dtype=torch.bfloat16).to(self.dev).eval()
        self.proc = AutoProcessor.from_pretrained(d)
        self.tok = self.proc.tokenizer
        self.sha = "hf_lingshu7b"

    def __enter__(self): return self
    def __exit__(self, *e): pass

    def ask(self, images, prompt, max_tokens=1):
        from PIL import Image
        torch = self.torch
        imgs = [Image.open(p) for p in images]
        content = [*({"type": "image", "image": im} for im in imgs),
                   {"type": "text", "text": prompt}]
        inputs = self.proc.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt").to(
                self.dev, dtype=torch.bfloat16)
        with torch.inference_mode():
            out = self.model(**inputs)
        logits = out.logits[0, -1, :].float()
        lp = torch.log_softmax(logits, dim=-1)
        vals, idx = torch.topk(lp, TOPK)
        tops = [(self.tok.decode([int(i)]), float(v)) for v, i in zip(vals, idx)]
        return "", tops[0][0], tops[0][1], tops


# ---------------------------------------------------------------- 題目
def imgs_for(pid, root):
    return [os.path.join(root, pid, f"roi_{k}.png") for k in range(1, NSLICES + 1)]


def frozen_answers():
    out = {}
    for r in csv.DictReader(open(FROZEN_CSV)):
        out[(r["run_tag"], r["patch_id"])] = r["answer"]
    return out


def load_canon():
    return [l.strip() for l in open(CANON) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["probe", "logprob", "noimg", "blankchk"])
    ap.add_argument("--models", default="all")
    ap.add_argument("--n", type=int, default=4, help="probe 模式每序題數")
    ap.add_argument("--out", default=OUTCSV)
    a = ap.parse_args()

    want = None if a.models == "all" else set(a.models.split(","))
    models = [m for m in MODELS if want is None or m[0] in want]
    meta = {r["patch_id"]: r for r in csv.DictReader(open(META))}
    canon = load_canon()
    froz = frozen_answers()

    if a.mode == "probe":
        nod = [p for p in canon if meta[p]["truth"] == "nodule"][:a.n // 2]
        ves = [p for p in canon if meta[p]["truth"] != "nodule"][:a.n - a.n // 2]
        items = [(p, "patches/roi_patches_gray_N9", p) for p in nod + ves]
    elif a.mode == "logprob":
        items = [(p, "patches/roi_patches_gray_N9", p) for p in canon]
    elif a.mode == "blankchk":                # blank 的 server 交叉檢查（Wei 核可，`37` §3.3）
        items = [("ctl_blank", "patches_control", None)]
    else:                                     # noimg
        items = [("ctl_noimg", None, None)]

    if not os.path.exists(a.out):
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()
    done = {(r["patch_id"], r["run_tag"]) for r in csv.DictReader(open(a.out))}

    for pref, name, gguf in models:
        ctx = HFScorer() if gguf is None else Server(gguf)
        engine = ("transformers" if gguf is None else "llama-server b10330")
        t_model = time.time()
        with ctx as S:
            msha = getattr(S, "sha", None) or RB.sha256(S.gguf)[:12]
            # ★ 題目在外圈、序在內圈：同一題的 9 張圖連著問兩次，
            #   llama-server 的 mtmd embedding cache 命中，第二序幾乎免費
            #   （probe 實測 4.18s → 0.24s）。cache 只需容納一題，不靠它容納 104 題。
            stat = {(v, sfx): dict(agree=0, tot=0, lat=[]) for v, sfx in ORDERS}
            for pid, root, froz_pid in items:
              for variant, suffix in ORDERS:
                prompt = PROMPTS[variant].replace("{ns}", str(NSLICES))
                psha = hashlib.sha1(prompt.encode()).hexdigest()[:12]
                tag = f"{pref}_CTL_{a.mode}_{suffix}"
                st = stat[(variant, suffix)]
                if True:
                    if (pid, tag) in done:
                        continue
                    paths = [] if root is None else imgs_for(pid, os.path.join(ROOT, root))
                    t0 = time.time()
                    try:
                        _txt, tok, tlp, tops = S.ask(paths, prompt)
                        err = ""
                    except Exception as e:
                        body = e.read().decode()[:200] if hasattr(e, "read") else ""
                        err = f"{type(e).__name__}: {str(e)[:150]} {body}"
                        tok, tlp, tops = "", float("nan"), []
                    el = round(time.time() - t0, 2); st["lat"].append(el)
                    pn, pv, nt, vt, mass = score_top(tops)
                    amax = token_class(tok) or ""
                    if froz_pid is not None:
                        fz = froz.get((f"{pref}_LIDC_{suffix}", froz_pid), "")
                        if fz and amax:
                            st["tot"] += 1; st["agree"] += (amax == fz)
                    with open(a.out, "a", newline="") as f:
                        csv.DictWriter(f, fieldnames=FIELDS).writerow(dict(
                            patch_id=pid, run_tag=tag, rep="1",
                            stimulus=("real" if root and "gray_N9" in root else
                                      ("noimg" if root is None else
                                       ("blankchk" if a.mode == "blankchk" else "control"))),
                            variant=variant, order_label=suffix, answer=amax,
                            argmax_token=tok, argmax_logprob=f"{tlp:.6f}",
                            p_nodule=f"{pn:.6g}", p_vessel=f"{pv:.6g}",
                            logodds_nodule=f"{logodds(pn, pv):.6g}",
                            covered=int(mass > 0), class_mass=f"{mass:.6g}",
                            nod_tokens="|".join(nt[:6]), ves_tokens="|".join(vt[:6]),
                            raw_first_token=repr(tok) + (" ERR:" + err if err else ""),
                            elapsed_s=el, harness=engine, top_k=TOPK, prompt_sha1=psha,
                            model_sha256=msha, engine=engine,
                            timestamp=time.strftime("%FT%T")))
            for (variant, suffix), st in stat.items():
                lat = st["lat"]
                med = sorted(lat)[len(lat) // 2] if lat else 0
                print(f"  {name:<18}{variant:<8}n={len(lat):3d} 中位 {med:6.2f}s"
                      f"  argmax 與凍結一致 {st['agree']}/{st['tot']}"
                      f"  ETA(104題) {med*104/60:5.1f}min", flush=True)
        print(f"  {name} 全部完成 {(time.time()-t_model)/60:.1f} min", flush=True)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
