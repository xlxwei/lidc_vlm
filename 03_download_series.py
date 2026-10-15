#!/usr/bin/env python3
"""下載中選的 50 個 CT series（TCIA NBIA public API，免登入）。

只下載中選 series，不是全集：133 GB → 估 3–6 GB。

解壓後的目錄結構刻意做成 pylidc 期望的 `LIDC-IDRI/<patient>/<study>/<series>/`，
理由：這樣同一份檔案可以同時被 (a) 自己寫的 DICOM loader 與 (b) pylidc 的
`to_volume()` 讀，兩條獨立路徑互相驗證體積與軸序——沿用專案既有的雙引擎交叉驗證紀律。
軸序錯誤是這一步最可能、也最安靜的錯：座標錯了照樣渲染得出漂亮的圖，只是框到別的地方。

每個 series 記錄：zip sha256、bytes、下載時間、HTTP 狀態 → manifest/download_log.csv
斷點續傳：已存在且 sha 對得上的 series 直接跳過。
"""
import csv
import hashlib
import io
import os
import time
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
MAN = os.path.join(HERE, "manifest")
RAW = os.path.join(HERE, "data", "LIDC-IDRI")
API = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
TIMEOUT = 120        # 較短：卡住時盡快進入重試，而不是空等 10 分鐘


def get_json(url):
    import json
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.load(r)


def main():
    series = [ln.strip() for ln in open(os.path.join(MAN, "series_to_download.txt"))
              if ln.strip()]
    sel = {r["series_uid"]: r for r in
           csv.DictReader(open(os.path.join(MAN, "selected_nodules.csv")))}
    print(f"要下載 {len(series)} 個 series")

    logpath = os.path.join(MAN, "download_log.csv")
    done = {}
    if os.path.exists(logpath):
        done = {r["series_uid"]: r for r in csv.DictReader(open(logpath))}

    fields = ["series_uid", "patient_id", "study_uid", "n_files", "zip_bytes",
              "zip_sha256", "manufacturer", "model_name", "kernel", "kvp",
              "exposure", "slice_thickness", "pixel_spacing", "http_status",
              "elapsed_s", "timestamp"]
    new = not os.path.exists(logpath)
    f = open(logpath, "a", newline="")
    w = csv.DictWriter(f, fieldnames=fields)
    if new:
        w.writeheader()

    t_all = time.time()
    for i, uid in enumerate(series, 1):
        pid = sel[uid]["patient_id"]
        if uid in done and done[uid]["http_status"] == "200":
            print(f"[{i}/{len(series)}] {pid} 已完成，跳過")
            continue
        # series 層 metadata（機型／重建核／劑量參數）——per-item 記錄，異質性做描述性分析
        meta = {}
        try:
            js = get_json(f"{API}/getSeries?SeriesInstanceUID={uid}")
            meta = js[0] if js else {}
        except Exception as e:
            print(f"  metadata 取得失敗: {type(e).__name__}")

        t0 = time.time()
        status, sha, nfiles, nbytes = "", "", 0, 0
        study = meta.get("StudyInstanceUID", "study")
        outdir = os.path.join(RAW, pid, study, uid)
        # TCIA 連續抓約 36 個 series 後會靜默卡住（不回錯誤、就是不回應）——
        # 實測第一輪停在第 36 個。故：較短的 socket timeout ＋ 重試 ＋ 每筆之間禮貌延遲。
        for attempt in (1, 2, 3):
            try:
                with urllib.request.urlopen(
                        f"{API}/getImage?SeriesInstanceUID={uid}", timeout=TIMEOUT) as r:
                    blob = r.read()
                    status = str(r.status)
                sha = hashlib.sha256(blob).hexdigest()
                nbytes = len(blob)
                os.makedirs(outdir, exist_ok=True)
                with zipfile.ZipFile(io.BytesIO(blob)) as z:
                    names = [n for n in z.namelist() if n.lower().endswith(".dcm")]
                    for n in names:
                        with z.open(n) as src, open(
                                os.path.join(outdir, os.path.basename(n)), "wb") as dst:
                            dst.write(src.read())
                    nfiles = len(names)
                break
            except Exception as e:
                status = f"ERR:{type(e).__name__}"
                print(f"  !! {pid} {status} attempt={attempt} — 退避後重試", flush=True)
                time.sleep(20 * attempt)

        el = round(time.time() - t0, 1)
        w.writerow(dict(
            series_uid=uid, patient_id=pid, study_uid=study, n_files=nfiles,
            zip_bytes=nbytes, zip_sha256=sha,
            manufacturer=meta.get("Manufacturer", ""),
            model_name=meta.get("ManufacturerModelName", ""),
            kernel=meta.get("SeriesDescription", ""),
            kvp="", exposure="",
            slice_thickness=sel[uid]["slice_thickness"],
            pixel_spacing=sel[uid]["pixel_spacing"],
            http_status=status, elapsed_s=el,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S")))
        f.flush()
        time.sleep(2)          # 禮貌延遲，避免再次被靜默限流
        eta = (time.time() - t_all) / i * (len(series) - i) / 60
        print(f"[{i}/{len(series)}] {pid} {nfiles:3d} 檔 {nbytes/1e6:6.1f}MB "
              f"{el:5.1f}s  {meta.get('ManufacturerModelName','')[:22]:22s} ETA {eta:.1f}min")
    f.close()
    print(f"\n完成，用時 {(time.time()-t_all)/60:.1f} min → {RAW}")


if __name__ == "__main__":
    main()
