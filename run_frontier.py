#!/usr/bin/env python3
"""[run 17] 雲端旗艦參考臂批次推論（OpenAI／Google／Anthropic 官方 SDK）。

任務書 `29`。定位＝**部署層級的失效現象學比較**，不是排行榜、不做規模／機制歸因。

與開放臂（run_roi_batch.py）的關係：
  - **prompt 逐字共用**：直接 import run_roi_batch.PROMPTS 與 parse()，
    啟動時自我檢查 sha1，不符即中止。零重渲染、零新標註、零新 prompt。
  - **輸出分表**：寫 results/frontier_results.csv，**絕不寫 lidc_roi_results.csv**。
    兩個理由：(1) 欄位不同（雲端沒有 model_sha256／llama_version，多了 usage 與 cost）；
    (2) run 16 正在對主表 append，同檔併發寫入會交錯壞列。
  - **可複現性標準不同，分節敘述不混**：開放臂＝位元級（模型檔 sha256＋llama 版本）；
    本臂＝**定日快照＋全存檔**（model snapshot 字串＋時間戳＋逐筆 request/response JSON）。
    雲端模型隨時可能更新或下架，這是本臂的固有限制，寫進 Limitations，不假裝可複現。

存檔政策：每次呼叫存一份 JSON 到 results/frontier_raw/<tag>/<patch_id>.json，
含完整 request 參數、完整 response 物件、usage、時間戳、9 張影像的路徑與 sha256。
**影像 base64 本身不入存檔**（1,524 次 × 9 × 61KB ≈ 1GB 重複資料），
影像已由 patches_sha256.txt 凍結存證，存 sha256 即可逐位元回溯。

用法：
  run_frontier.py --vendor anthropic --tag AN_F5_N9 --variant v1 --subset ... --limit 3
"""
import argparse, base64, csv, hashlib, json, os, re, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_roi_batch import PROMPTS, parse            # noqa: E402  逐字共用，不複製

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTCSV = os.path.join(ROOT, "results", "frontier_results.csv")
RAWDIR = os.path.join(ROOT, "results", "frontier_raw")

# 凍結 prompt sha1（`29` 表格；與開放臂逐字相同）。啟動時比對，不符即中止。
FROZEN_SHA = {"v1": "b046abfad45a", "v1swap": "b3a242d23748", "v2open": "86c22b3afbab",
              "v4dens": "55349d61465d", "v4dens_swap": "4b1b2ed00d2a",
              "v5nd": "958c37a4c6de", "v5ndswap": "666ec3e5d79b"}   # run 22（vault `62`）

# === 單價表（2026-08-25 當日官方頁面實查；USD per 1M tokens）=========================
# OpenAI    developers.openai.com/api/docs/pricing  gpt-5.6-sol short-context $4／$20
#           （>200k long-context 為 $8／$30；本研究每題 ~2.5k，恆在 short 檔）
# Google    ai.google.dev/gemini-api/docs/pricing   gemini-3.1-pro-preview
#           $2（≤200k）／$12（output 含 thinking）
# Anthropic platform.claude.com/docs/en/about-claude/pricing  claude-fable-5 $10／$50
#           （Batch 為 $5／$25，若採用需改 --batch 路徑並在報告註明）
PRICES = {"gpt-5.6-sol":            {"in": 4.00,  "out": 20.00},
          "gemini-3.1-pro-preview": {"in": 2.00,  "out": 12.00},
          "claude-fable-5":         {"in": 10.00, "out": 50.00},
          "claude-opus-5":          {"in": 5.00,  "out": 25.00}}
PRICE_DATE = "2026-08-25"

FIELDS = ["patch_id", "run_tag", "rep", "truth", "size_mm", "stratum", "lung_rads",
          "answer", "est_diam_mm", "parse_status", "raw_output", "elapsed_s",
          "vendor", "model_id", "model_snapshot", "effort", "thinking_setting",
          "media_resolution", "temp", "seed", "max_out", "prompt_sha1",
          "in_tokens", "out_tokens", "thinking_tokens", "cached_tokens", "cost_usd",
          "stop_reason", "refusal_hard", "refusal_category", "thinking_summary",
          "attempts", "errors", "archive_path", "images_sha1", "timestamp"]


def sha256_file(path, cache={}):
    if path not in cache:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(1 << 22), b""):
                h.update(b)
        cache[path] = h.hexdigest()
    return cache[path]


# run 22（vault `62`）：金鑰除污。重試時 str(e) 可能夾帶 request header（過去 8 個封存檔
# 因此含 OpenAI key 明文）。寫進 errors 欄／封存檔之前一律過這道。規則同 redact_raw.py。
_KEY_PATS = [(re.compile(r'sk-[A-Za-z0-9_-]{8,}'), "<REDACTED>"),
             (re.compile(r'AIza[A-Za-z0-9_-]{10,}'), "<REDACTED>"),
             (re.compile(r'([Bb]earer\s+)[A-Za-z0-9._-]{12,}'), r"\1<REDACTED>"),
             (re.compile(r"""(["']?x-api-key["']?\s*[:=]\s*["']?)[^"'\s,}]{8,}""", re.I),
              r"\1<REDACTED>")]


def redact(s):
    s = str(s)
    for pat, rep in _KEY_PATS:
        s = pat.sub(rep, s)
    for k in ("OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY"):
        v = os.environ.get(k)
        if v and len(v) >= 8:
            s = s.replace(v, "<REDACTED>")
    return s


def b64(path):
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode()


def jsonable(o):
    """把 SDK 回應物件轉成可序列化 dict；失敗就退回 repr（存檔永不因序列化失敗而丟資料）。"""
    for attr in ("model_dump", "to_dict", "dict"):
        if hasattr(o, attr):
            try:
                return json.loads(json.dumps(getattr(o, attr)(), default=str))
            except Exception:
                pass
    try:
        return json.loads(json.dumps(o, default=str))
    except Exception:
        return {"_unserializable_repr": repr(o)}


# === 三家的呼叫器 ===================================================================
# 每個回傳 (text, usage_dict, stop_reason, refusal_hard, refusal_category, raw_response)
# usage_dict 鍵：in / out / thinking / cached
# 共同約束：影像在前、文字在後（與 llama-mtmd-cli 的 --image ... -p ... 順序一致）。

def call_openai(client, model, imgs, prompt, cfg):
    content = [{"type": "input_image", "image_url": f"data:image/png;base64,{b64(p)}"}
               for p in imgs]
    content.append({"type": "input_text", "text": prompt})
    kw = dict(model=model, input=[{"role": "user", "content": content}],
              reasoning={"effort": cfg["effort"], "summary": "auto"},
              max_output_tokens=cfg["max_out"], store=False)
    r = client.responses.create(**kw)
    u = getattr(r, "usage", None)
    det = getattr(u, "output_tokens_details", None) if u else None
    idet = getattr(u, "input_tokens_details", None) if u else None
    usage = {"in": getattr(u, "input_tokens", 0) or 0,
             "out": getattr(u, "output_tokens", 0) or 0,
             "thinking": getattr(det, "reasoning_tokens", 0) or 0 if det else 0,
             "cached": getattr(idet, "cached_tokens", 0) or 0 if idet else 0}
    # 硬拒答：output 內出現 refusal 型 content block
    hard, cat = 0, ""
    for item in (getattr(r, "output", None) or []):
        for c in (getattr(item, "content", None) or []):
            if getattr(c, "type", "") == "refusal":
                hard, cat = 1, "openai_refusal"
    txt = (getattr(r, "output_text", "") or "").strip()
    think = []
    for item in (getattr(r, "output", None) or []):
        if getattr(item, "type", "") == "reasoning":
            for sm in (getattr(item, "summary", None) or []):
                think.append((getattr(sm, "text", "") or "").strip())
    return txt, usage, getattr(r, "status", "") or "", hard, cat, r, " ".join(think).strip()


def call_google(client, model, imgs, prompt, cfg):
    from google.genai import types
    parts = [types.Part.from_bytes(data=open(p, "rb").read(), mime_type="image/png")
             for p in imgs]
    parts.append(types.Part.from_text(text=prompt))
    gcfg = types.GenerateContentConfig(
        temperature=cfg["temp"], seed=cfg["seed"], max_output_tokens=cfg["max_out"],
        media_resolution=cfg["media_resolution"],
        thinking_config=types.ThinkingConfig(thinking_level=cfg["effort"].upper(),
                                            include_thoughts=True))
    r = client.models.generate_content(model=model, contents=parts, config=gcfg)
    u = getattr(r, "usage_metadata", None)
    usage = {"in": getattr(u, "prompt_token_count", 0) or 0,
             "out": (getattr(u, "candidates_token_count", 0) or 0)
                    + (getattr(u, "thoughts_token_count", 0) or 0),
             "thinking": getattr(u, "thoughts_token_count", 0) or 0,
             "cached": getattr(u, "cached_content_token_count", 0) or 0}
    fr, hard, cat = "", 0, ""
    cands = getattr(r, "candidates", None) or []
    if cands:
        fr = str(getattr(cands[0], "finish_reason", "") or "")
        if re.search(r"SAFETY|PROHIBITED|BLOCK|RECITATION", fr, re.I):
            hard, cat = 1, fr
    pf = getattr(r, "prompt_feedback", None)
    if pf and getattr(pf, "block_reason", None):
        hard, cat = 1, f"prompt_blocked:{pf.block_reason}"
    txt, think = "", []
    for cand in cands:
        for part in (getattr(getattr(cand, "content", None), "parts", None) or []):
            ptxt = (getattr(part, "text", "") or "").strip()
            if ptxt and getattr(part, "thought", False):
                think.append(ptxt)          # thought=True 的 part＝推理摘要，與答案分開收
    try:
        txt = (r.text or "").strip()
    except Exception:
        pass
    return txt, usage, fr, hard, cat, r, " ".join(think).strip()


def call_anthropic(client, model, imgs, prompt, cfg):
    content = [{"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                            "data": b64(p)}} for p in imgs]
    content.append({"type": "text", "text": prompt})
    # temperature／budget_tokens 在 Fable 5 一律 400（sampling 與固定思考預算皆已移除）；
    # 中檔思考＝output_config.effort="medium"。thinking 恆開，display 設 summarized 以存證推理摘要。
    # **不啟用 server-side fallbacks**：拒答是本研究預登記的結局，
    # 自動改派他型號會把要測的現象換掉，等於毀掉端點。
    r = client.messages.create(
        model=model, max_tokens=cfg["max_out"],
        output_config={"effort": cfg["effort"]},
        thinking={"type": "adaptive", "display": "summarized"},
        messages=[{"role": "user", "content": content}])
    u = getattr(r, "usage", None)
    otd = getattr(u, "output_tokens_details", None) if u else None
    usage = {"in": getattr(u, "input_tokens", 0) or 0,
             "out": getattr(u, "output_tokens", 0) or 0,
             # thinking 計入 output_tokens（計價同 output），但另有明細欄可取，兩個都記
             "thinking": (getattr(otd, "thinking_tokens", 0) or 0) if otd else 0,
             "cached": getattr(u, "cache_read_input_tokens", 0) or 0}
    stop = getattr(r, "stop_reason", "") or ""
    hard, cat = 0, ""
    if stop == "refusal":
        hard = 1
        sd = getattr(r, "stop_details", None)
        cat = str(getattr(sd, "category", "") or "refusal")
    txt = " ".join(b.text.strip() for b in (r.content or [])
                   if getattr(b, "type", "") == "text" and getattr(b, "text", "")).strip()
    think = " ".join((getattr(b, "thinking", "") or "").strip() for b in (r.content or [])
                     if getattr(b, "type", "") == "thinking").strip()
    return txt, usage, stop, hard, cat, r, think


VENDORS = {"openai": call_openai, "google": call_google, "anthropic": call_anthropic}
DEFAULT_MODEL = {"openai": "gpt-5.6-sol", "google": "gemini-3.1-pro-preview",
                 "anthropic": "claude-fable-5"}


def make_client(vendor, timeout=120.0):
    if vendor == "openai":
        from openai import OpenAI
        return OpenAI(timeout=timeout, max_retries=0)    # 重試由本檔控制，禁止 SDK 靜默重試
    if vendor == "google":
        from google import genai
        return genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    import anthropic
    # Accept-Encoding 收窄：本機 httpx2 2.12 的 brotli 解碼器與 brotlicffi 1.1 不相容
    # （Decompressor.decompress() 不吃 output_buffer_limit）。避開 br 即可，不動共用套件版本
    # ——run 16 正在同一個 anaconda 環境跑 transformers，不能為了這件事升級底層依賴。
    return anthropic.Anthropic(default_headers={"Accept-Encoding": "gzip, deflate"},
                               timeout=timeout, max_retries=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vendor", required=True, choices=sorted(VENDORS))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--variant", default="v1", choices=sorted(PROMPTS))
    ap.add_argument("--subset")
    ap.add_argument("--meta", default=os.path.join(ROOT, "manifest", "patch_meta.csv"))
    ap.add_argument("--patch-root", default=os.path.join(ROOT, "patches", "roi_patches_gray_N9"))
    ap.add_argument("--nslices", type=int, default=9)
    ap.add_argument("--rep", default="1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--temp", default="0")
    ap.add_argument("--seed", default="1")
    ap.add_argument("--max-out", type=int, default=8192)
    ap.add_argument("--media-resolution", default="MEDIA_RESOLUTION_HIGH",
                    help="僅 Google 有效；記錄進 CSV 供對帳")
    ap.add_argument("--no-image", action="store_true",
                    help="[run 20] payload 完全不附影像，只送文字 prompt（逐字不變）。"
                         "對應 run 19 的 R19-A 第三種刺激 ctl_noimg。")
    ap.add_argument("--out", default=OUTCSV)
    ap.add_argument("--timeout", type=float, default=120.0,
                    help="單次呼叫逾時秒數；中位 8–22s，120s 已極寬鬆")
    ap.add_argument("--budget-usd", type=float, default=150.0,
                    help="累計成本硬上限；超過即停並回報（`29` 核可上限）")
    a = ap.parse_args()

    # --- 防呆 1：絕不寫進開放臂主表（run 16 正在 append）或院內 SSD ---
    ap_out = os.path.abspath(a.out)
    if "lidc_roi_results.csv" in ap_out:
        sys.exit("拒絕執行：frontier 結果不得寫入開放臂主表 lidc_roi_results.csv。")
    if "/Volumes/medai_ssd" in ap_out:
        sys.exit("拒絕執行：輸出路徑指向院內 SSD。")

    # --- 防呆 2：prompt 逐字未變 ---
    prompt = PROMPTS[a.variant].replace("{ns}", str(a.nslices))
    psha = hashlib.sha1(prompt.encode()).hexdigest()[:12]
    if a.variant in FROZEN_SHA and psha != FROZEN_SHA[a.variant]:
        sys.exit(f"拒絕執行：prompt sha1 不符（{a.variant} 期望 "
                 f"{FROZEN_SHA[a.variant]}，實得 {psha}）——凍結資產已變動。")

    model = a.model or DEFAULT_MODEL[a.vendor]
    rows = [r for r in csv.DictReader(open(a.meta)) if r["render_ok"] == "1"]
    if a.subset:
        want = {ln.strip() for ln in open(a.subset) if ln.strip()}
        rows = [r for r in rows if r["patch_id"] in want]
    if a.limit:
        rows = rows[:a.limit]

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        for r in csv.DictReader(open(a.out)):
            done.add((r["patch_id"], r["run_tag"], r["rep"]))
    else:
        with open(a.out, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    # 累計成本上限要跨臂生效：本次啟動前，先把同一家已經花掉的錢算進來。
    # 每支 runner 只跑一個臂，若上限只管單次呼叫序列，六個臂就等於六倍上限。
    prior = 0.0
    if os.path.exists(a.out):
        for r in csv.DictReader(open(a.out)):
            if r.get("vendor") == a.vendor:
                try:
                    prior += float(r.get("cost_usd") or 0)
                except ValueError:
                    pass

    todo = [r for r in rows if (r["patch_id"], a.tag, a.rep) not in done]
    archive_dir = os.path.join(RAWDIR, a.tag)
    os.makedirs(archive_dir, exist_ok=True)
    client = make_client(a.vendor, a.timeout)
    fn = VENDORS[a.vendor]
    cfg = {"effort": a.effort, "max_out": a.max_out, "temp": float(a.temp),
           "seed": int(a.seed), "media_resolution": a.media_resolution}
    price = PRICES.get(model, {"in": 0.0, "out": 0.0})

    print(f"tag={a.tag} vendor={a.vendor} model={model} variant={a.variant} "
          f"effort={a.effort} | {len(todo)} to run ({len(rows)-len(todo)} already done)")
    t_all, spent, ratelimited, consec_rl = time.time(), prior, 0, 0
    print(f"  本家先前已花 ${prior:.4f}；本次上限 ${a.budget_usd:.2f}（累計計算）")
    for i, r in enumerate(todo, 1):
        if a.no_image:
            imgs = []                       # [run 20] 不附任何影像；prompt 文字逐字不變
        else:
            d = os.path.join(a.patch_root, r["patch_id"])
            imgs = [os.path.join(d, f"roi_{k}.png") for k in range(1, a.nslices + 1)]
            missing = [p for p in imgs if not os.path.exists(p)]
            if missing:
                print(f"[{i}/{len(todo)}] {r['patch_id']} SKIP 缺影像 {missing[0]}")
                continue
        t0, errs, txt, raw, think = time.time(), [], "", None, ""
        usage = {"in": 0, "out": 0, "thinking": 0, "cached": 0}
        stop = hard = cat = ""
        hard = 0
        for attempt in (1, 2, 3):          # 指數退避、上限 3；非 200 全記錄，禁止靜默重試
            try:
                txt, usage, stop, hard, cat, raw, think = fn(client, model, imgs, prompt, cfg)
                break
            except Exception as e:
                errs.append(redact(f"attempt{attempt}:{type(e).__name__}:{redact(e)[:300]}"))
                if attempt < 3:
                    time.sleep(5 * (2 ** (attempt - 1)))
        el = round(time.time() - t0, 1)
        ans, diam, status = parse(txt)
        if errs and not txt:
            status = "api_fail"
            # 5xx／UNAVAILABLE／overloaded 與限速同類：**基礎設施故障，不是模型輸出**。
            # 兩者都必須「不寫列、留給續跑」——寫成 api_fail 會被續跑邏輯視為已完成而
            # 永遠跳過，等於把一次暫時性 503 變成永久遺失的題目。
            # （2026-08-27 補：Gemini 穩定性臂就這樣掉了 nod_LIDC-IDRI-0919_n01。）
            if any(re.search(r"429|RESOURCE_EXHAUSTED|rate.?limit|quota|credit|balance|billing"
                             r"|50[0234]|UNAVAILABLE|overload|high demand|timeout|Timeout",
                             e, re.I) for e in errs):
                ratelimited += 1
                consec_rl += 1
                try:
                    with open(os.path.join(archive_dir,
                              f"{r['patch_id']}_rep{a.rep}_RATELIMIT.json"), "w") as f:
                        json.dump({"patch_id": r["patch_id"], "errors": errs,
                                   "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")},
                                  f, ensure_ascii=False, indent=1)
                except Exception:
                    pass
                print(f"[{i}/{len(todo)}] {r['patch_id']:28s} 限速／額度／服務暫時故障 —— 不寫列，留給續跑")
                # 連續多題都是限速／額度 → 極可能是**餘額用完**，不是暫時性節流。
                # 不中止的話會用 30s×剩餘題數 空轉一整夜（452 題＝3.8 小時），
                # 而且後面每個臂都會重演一次。直接停，讓人看到訊息去儲值。
                if consec_rl >= 8:
                    print(f"!! 連續 {consec_rl} 題限速／額度失敗 —— 判定為餘額或配額用盡，中止本臂。")
                    print(f"!! 儲值後重跑同一行指令即可續補（已寫入的題目會自動跳過）。")
                    break
                time.sleep(min(30 * consec_rl, 180))   # 遞增等待，上限 3 分鐘
                continue
            consec_rl = 0
        cost = usage["in"] / 1e6 * price["in"] + usage["out"] / 1e6 * price["out"]
        spent += cost

        snap = ""
        if raw is not None:
            snap = str(getattr(raw, "model", "") or getattr(raw, "model_version", "") or "")
        img_sha = ";".join(sha256_file(p)[:12] for p in imgs)
        apath = os.path.join(archive_dir, f"{r['patch_id']}_rep{a.rep}.json")
        with open(apath, "w") as f:
            json.dump({"request": {"vendor": a.vendor, "model": model, "variant": a.variant,
                                   "prompt": prompt, "prompt_sha1": psha, "config": cfg,
                                   "no_image": bool(a.no_image), "n_images": len(imgs),
                                   "images": [{"path": os.path.relpath(p, ROOT),
                                               "sha256": sha256_file(p)} for p in imgs]},
                       "response": jsonable(raw) if raw is not None else None,
                       "derived": {"answer": ans, "est_diam_mm": diam, "parse_status": status,
                                   "usage": usage, "cost_usd": round(cost, 6),
                                   "stop_reason": stop, "refusal_hard": hard,
                                   "refusal_category": cat,
                                   "thinking_summary": think},
                       "errors": errs, "elapsed_s": el,
                       "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")},
                      f, ensure_ascii=False, indent=1)

        with open(a.out, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(dict(
                patch_id=r["patch_id"], run_tag=a.tag, rep=a.rep, truth=r["truth"],
                size_mm=r["size_mm"], stratum=r["stratum"], lung_rads=r["lung_rads"],
                answer=ans, est_diam_mm=diam, parse_status=status,
                raw_output=" ".join(txt.split()), elapsed_s=el,
                vendor=a.vendor, model_id=model, model_snapshot=snap, effort=a.effort,
                thinking_setting=("adaptive/summarized" if a.vendor == "anthropic"
                                  else f"level={a.effort}"),
                media_resolution=(a.media_resolution if a.vendor == "google" else ""),
                temp=(a.temp if a.vendor == "google" else "n/a"),
                seed=(a.seed if a.vendor == "google" else "n/a"),
                max_out=a.max_out, prompt_sha1=psha,
                in_tokens=usage["in"], out_tokens=usage["out"],
                thinking_tokens=usage["thinking"], cached_tokens=usage["cached"],
                cost_usd=round(cost, 6), stop_reason=stop, refusal_hard=hard,
                refusal_category=cat, thinking_summary=" ".join((think or "").split()), attempts=len(errs) + (0 if errs and not txt else 1),
                errors=" | ".join(errs), archive_path=os.path.relpath(apath, ROOT),
                images_sha1=img_sha, timestamp=time.strftime("%Y-%m-%dT%H:%M:%S")))

        eta = (time.time() - t_all) / i * (len(todo) - i) / 60
        flag = " REFUSAL" if hard else ("  ERR" if errs else "")
        print(f"[{i}/{len(todo)}] {r['patch_id']:28s} {r['truth']:18s} -> "
              f"{ans or status:14s} {diam:>5}mm  in={usage['in']:5d} out={usage['out']:5d} "
              f"${cost:.4f}  {el}s  ETA {eta:.0f}min{flag}")
        if spent > a.budget_usd:
            print(f"!! 累計 ${spent:.2f} 超過上限 ${a.budget_usd:.2f} —— 停止並回報")
            break
    if ratelimited:
        print(f"!! {ratelimited} 題因限速未完成（未寫列）——重跑同一行指令即可續補")
    print(f"done in {(time.time()-t_all)/60:.1f} min | 本臂 ${spent-prior:.4f}｜本家累計 ${spent:.4f} "
          f"（單價表日期 {PRICE_DATE}）-> {a.out}")


if __name__ == "__main__":
    main()
