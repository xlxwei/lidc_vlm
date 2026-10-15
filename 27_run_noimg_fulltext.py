#!/usr/bin/env python3
"""run 21：本地 7 模型 no image 全文重跑。任務書 vault `54`。

為什麼：run 19 的 noimg（21_run_control_logprob.py noimg）max_tokens=1，只存了首 token。
手稿「沒有影像時本地模型照答、沒人說缺圖」需要完整回覆才能判讀。

與 run 19 相同：模型權重（啟動前逐檔 sha256 核對凍結 CSV）、prompt（直接 import
run_roi_batch.PROMPTS，啟動自檢 sha1）、payload 結構（只有 text part）、temperature 0、
seed 1、llama-server b10330／transformers 同一入口、首 token top-100 logprob。
唯一差異：生成長度 128（＝真影像與 noise 凍結設定 n_predict=128）。

LS7（transformers）：首 token 用 21 的 HFScorer.ask（forward pass，與 run 19 同一段程式）；
全文用 run_roi_batch_hf.py 同一個 generate 呼叫（greedy，max_new_tokens=128），
但 decode 時**不**跳過 special tokens、不換行壓平，保留原始輸出。

凍結紀律：21_run_control_logprob.py／run_roi_batch*.py 不改；結果寫
results/control_noimg_fulltext.csv，不碰 control_results.csv／lidc_roi_results.csv。

用法：
  27_run_noimg_fulltext.py verify-weights      # 權重 sha256 vs 凍結 CSV，不符 exit 1
  27_run_noimg_fulltext.py run [--models all]  # 14 次呼叫
"""
import argparse, csv, hashlib, importlib.util, json, os, sys, time, urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTCSV = os.path.join(ROOT, "results", "control_noimg_fulltext.csv")
FROZEN_CSV = os.path.join(ROOT, "results", "lidc_roi_results.csv")
CTL_CSV = os.path.join(ROOT, "results", "control_results.csv")
MAXTOK = 128
EXPECT_PSHA = {"v1": "b046abfad45a", "v1swap": "b3a242d23748", "v5nd": "958c37a4c6de", "v5ndswap": "666ec3e5d79b"}


def _load(mod, path):
    spec = importlib.util.spec_from_file_location(mod, os.path.join(ROOT, path))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


C21 = _load("c21", "21_run_control_logprob.py")   # Server／HFScorer／token_class／score_top
HF = _load("rbhf", "run_roi_batch_hf.py")          # weights_sha（LS7 雜湊慣例）
RB = C21.RB
PROMPTS = C21.PROMPTS

FIELDS = ["patch_id", "run_tag", "model", "option_order", "variant", "raw_output",
          "text_before_template", "first_token", "first_token_logprob", "p_nodule",
          "p_vessel", "logodds_nodule", "class_mass", "run19_first_token", "token_match",
          "n_images_in_payload", "finish_reason", "n_completion_tokens", "max_tokens",
          "temperature", "seed", "elapsed_s", "prompt_sha1", "model_sha256",
          "mmproj_sha256", "engine", "timestamp"]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def frozen_hashes():
    """凍結 CSV 的每模型 (model_sha256, mmproj_sha256)；每模型必須唯一。"""
    out = {}
    for r in csv.DictReader(open(FROZEN_CSV)):
        pref = r["run_tag"].split("_")[0]
        out.setdefault(pref, set()).add((r["model_sha256"], r["mmproj_sha256"]))
    for k, v in out.items():
        if len(v) != 1:
            raise SystemExit(f"凍結 CSV 中 {k} 有 {len(v)} 組雜湊，無法核對")
    return {k: next(iter(v)) for k, v in out.items()}


def verify_prompts():
    for v, want in EXPECT_PSHA.items():
        got = hashlib.sha1(PROMPTS[v].replace("{ns}", "9").encode()).hexdigest()[:12]
        if got != want:
            raise SystemExit(f"拒絕執行：prompt {v} sha1={got}，期望 {want}")
    print(f"prompt sha1 自檢通過：{EXPECT_PSHA}", flush=True)


def verify_weights(models):
    exp = frozen_hashes()
    bad = 0
    got = {}
    for pref, name, gguf in models:
        t0 = time.time()
        if gguf is None:
            d = os.path.expanduser("~/models/lingshu7b")
            m, mm = HF.weights_sha(d), "in_safetensors"
        else:
            p = os.path.expanduser(f"~/models/{gguf}")
            m, mm = sha256(p), sha256(os.path.join(os.path.dirname(p), "mmproj-BF16.gguf"))
        em, emm = exp[pref]
        ok = (m == em and mm == emm)
        bad += not ok
        got[pref] = (m, mm)
        print(f"  {'✔' if ok else '✘ 不符'} {name:<17} model {m[:16]} (凍結 {em[:16]})  "
              f"mmproj {mm[:16]} (凍結 {emm[:16]})  {time.time()-t0:.0f}s", flush=True)
    print(f"權重核對：不符 {bad} / {len(models)}", flush=True)
    return bad, got


def run19_tokens():
    out = {}
    for r in csv.DictReader(open(CTL_CSV)):
        if r["stimulus"] == "noimg":
            out[r["run_tag"]] = r["raw_first_token"]
    return out


def before_template(s):
    i = s.find("<|im_end|>")
    return s if i < 0 else s[:i]


class SlowDiskServer(C21.Server):
    """與 C21.Server 同一組啟動參數，只把 health 等待從 360 s 放寬到 30 min。
    run 21 首跑時 Qwen3-VL 8B 從外接硬碟載入 6:00，剛好撞上原本的 360 s 上限。"""
    def __enter__(self):
        log = open(os.path.join(ROOT, "logs_run19_server.txt"), "a")
        log.write(f"\n=== {time.strftime('%F %T')} start {self.gguf}\n"); log.flush()
        self.proc = C21.subprocess.Popen(
            ["llama-server", "-m", self.gguf, "--mmproj", self.mmproj, "-c", "6144",
             "--port", str(self.port), "--host", "127.0.0.1", "-np", "1"],
            stdout=log, stderr=log)
        for _ in range(900):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2)
                return self
            except Exception:
                time.sleep(2)
        self.__exit__(None, None, None)
        raise SystemExit(f"llama-server 起不來：{self.gguf}")


# ---------------------------------------------------------------- llama-server：全文版
def server_ask(S, prompt):
    """與 C21.Server.ask 相同的 body，max_tokens=128；額外回傳 payload 影像數與 finish_reason。"""
    content = [{"type": "text", "text": prompt}]
    body = {"messages": [{"role": "user", "content": content}], "temperature": 0,
            "max_tokens": MAXTOK, "seed": 1, "logprobs": True, "top_logprobs": C21.TOPK}
    n_img = json.dumps(body).count("image_url")
    assert n_img == 0, "payload 含影像"
    req = urllib.request.Request(f"http://127.0.0.1:{S.port}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=900))
    ch = r["choices"][0]
    tk = ch["logprobs"]["content"][0]
    tops = [(t["token"], t["logprob"]) for t in tk["top_logprobs"]]
    ntok = (r.get("usage") or {}).get("completion_tokens", "")
    return ch["message"]["content"], tk["token"], tk["logprob"], tops, n_img, \
        ch.get("finish_reason", ""), ntok


# ---------------------------------------------------------------- transformers：全文版
def hf_ask(S, prompt):
    tok1, lp1 = None, None
    _t, tok1, lp1, tops = S.ask([], prompt)          # run 19 同一段 forward（首 token）
    torch = S.torch
    content = [{"type": "text", "text": prompt}]
    n_img = sum(c.get("type") == "image" for c in content)
    assert n_img == 0, "payload 含影像"
    inputs = S.proc.apply_chat_template(
        [{"role": "user", "content": content}], add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt").to(S.dev, dtype=torch.bfloat16)
    with torch.inference_mode():
        gen = S.model.generate(**inputs, max_new_tokens=MAXTOK, do_sample=False)
    new = gen[0][inputs["input_ids"].shape[-1]:]
    raw = S.proc.decode(new, skip_special_tokens=False)
    eos = S.tok.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    fin = "stop" if int(new[-1]) in eos or len(new) < MAXTOK else "length"
    return raw, tok1, lp1, tops, n_img, fin, len(new)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["verify-weights", "run"])
    ap.add_argument("--models", default="all")
    ap.add_argument("--out", default=OUTCSV)
    ap.add_argument("--variant-set", default="v1", choices=["v1", "nd"],
                    help="v1＝原 prompt（run 21）；nd＝拿掉 low-dose 的 v5nd／v5ndswap（run 23）")
    a = ap.parse_args()
    ORDERS = C21.ORDERS if a.variant_set == "v1" else [("v5nd", "N9nd"), ("v5ndswap", "N9ndswap")]
    want = None if a.models == "all" else set(a.models.split(","))
    models = [m for m in C21.MODELS if want is None or m[0] in want]
    verify_prompts()

    if a.mode == "verify-weights":
        bad, _ = verify_weights(models)
        sys.exit(1 if bad else 0)

    exp = frozen_hashes()
    r19 = run19_tokens()
    if not os.path.exists(a.out):
        with open(a.out, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()
    done = {r["run_tag"] for r in csv.DictReader(open(a.out))}

    for pref, name, gguf in models:
        hf = gguf is None
        if not hf and not os.path.exists(os.path.expanduser(f"~/models/{gguf}")):
            raise SystemExit(f"找不到權重：~/models/{gguf}（外接硬碟掛載？）")
        ctx = C21.HFScorer() if hf else SlowDiskServer(gguf)
        engine = "transformers" if hf else "llama-server b10330"
        if hf:
            import torch, transformers
            engine = f"transformers-{transformers.__version__}+torch-{torch.__version__}"
        t_model = time.time()
        with ctx as S:
            for variant, suffix in ORDERS:
                tag = f"{pref}_CTL_noimgfull_{suffix}"
                if tag in done:
                    continue
                prompt = PROMPTS[variant].replace("{ns}", str(C21.NSLICES))
                psha = hashlib.sha1(prompt.encode()).hexdigest()[:12]
                t0 = time.time()
                txt, tok, tlp, tops, n_img, fin, ntok = (hf_ask if hf else server_ask)(S, prompt)
                el = round(time.time() - t0, 2)
                pn, pv, _nt, _vt, mass = C21.score_top(tops)
                old = r19.get(f"{pref}_CTL_noimg_{suffix}", "")
                with open(a.out, "a", newline="") as f:
                    csv.DictWriter(f, fieldnames=FIELDS).writerow(dict(
                        patch_id="ctl_noimg", run_tag=tag, model=name, option_order=suffix,
                        variant=variant, raw_output=txt, text_before_template=before_template(txt),
                        first_token=repr(tok), first_token_logprob=f"{tlp:.6f}",
                        p_nodule=f"{pn:.6g}", p_vessel=f"{pv:.6g}",
                        logodds_nodule=f"{C21.logodds(pn, pv):.6g}", class_mass=f"{mass:.6g}",
                        run19_first_token=old, token_match=int(repr(tok) == old),
                        n_images_in_payload=n_img, finish_reason=fin,
                        n_completion_tokens=ntok, max_tokens=MAXTOK, temperature=0, seed=1,
                        elapsed_s=el, prompt_sha1=psha, model_sha256=exp[pref][0],
                        mmproj_sha256=exp[pref][1], engine=engine,
                        timestamp=time.strftime("%FT%T")))
                print(f"  {name:<17}{suffix:<7} first={tok!r} run19={old} "
                      f"{'✔' if repr(tok) == old else '✘ 不一致'}  {el}s  {fin}", flush=True)
        print(f"  {name} 完成 {(time.time()-t_model)/60:.1f} min", flush=True)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
