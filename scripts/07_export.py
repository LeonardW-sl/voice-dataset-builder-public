"""Apply quality gates and export wavs from the ORIGINAL mp3, never from separated audio."""
import argparse, collections, json, os, re, subprocess, sys
import numpy as np
import soundfile as sf

ap = argparse.ArgumentParser()
ap.add_argument("mp3")                     # 原始录音
ap.add_argument("aed_json")
ap.add_argument("asr_json")
ap.add_argument("-o", "--outdir", default="out/dataset")
ap.add_argument("--config", default="config/default.yaml")
ap.add_argument("--emb-json", default=None)
ap.add_argument("--sep-json", default=None)
a = ap.parse_args()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_config, dump_json

cfg = load_config(a.config)
aed = json.load(open(a.aed_json, encoding="utf-8"))
asr = json.load(open(a.asr_json, encoding="utf-8"))
emb = json.load(open(a.emb_json, encoding="utf-8")) if a.emb_json else []
sep = json.load(open(a.sep_json, encoding="utf-8")) if a.sep_json else []
emb_by = {(round(x["start"], 2), round(x["end"], 2)): x for x in emb}
sep_by = {(round(x["start"], 2), round(x["end"], 2)): x for x in sep}

def ov(s1, e1, s2, e2): return max(0.0, min(e1, e2) - max(s1, s2))
segs = sorted(
    [(k, s, e) for k, ivs in aed["event2timestamps"].items() for s, e in ivs],
    key=lambda x: x[1])

os.makedirs(os.path.join(a.outdir, "wavs"), exist_ok=True)
kept, drops = [], collections.Counter()
for kind, st, en in segs:
    d = en - st
    if kind != "speech": drops["not_speech"] += 1; continue
    if d < cfg["seg"]["min_dur"]: drops["too_short"] += 1; continue
    if d > cfg["seg"]["max_dur"]: drops["too_long"] += 1; continue
    key = (round(st, 2), round(en, 2))
    v = emb_by.get(key)
    if v and v.get("sim") is not None and v["sim"] < cfg["embed"]["min_sim"]:
        drops["low_similarity"] += 1; continue
    v = sep_by.get(key)
    if v and v["v2a_db"] < cfg["quality"]["min_v2a_db"]:
        drops["low_v2a"] += 1; continue
    hits = [x for x in asr["segments"] if ov(st, en, x["start"], x["end"]) > 0.2]
    if not hits: drops["no_text"] += 1; continue
    text = "".join(x["text"] for x in hits).strip()
    cover = sum(ov(st, en, x["start"], x["end"]) for x in hits) / d
    conf = float(np.mean([x["confidence"] for x in hits]))  # 0-1, higher is better
    nchar = len(re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", text))
    if cover < cfg["asr"]["min_cover"]: drops["low_cover"] += 1; continue
    if conf < cfg["asr"]["min_confidence"]: drops["low_confidence"] += 1; continue
    if nchar < cfg["asr"]["min_chars"]: drops["too_few_chars"] += 1; continue
    if nchar / d > cfg["asr"]["max_cps"]: drops["too_fast"] += 1; continue
    if re.search(r"(.{6,})\1", text): drops["repeated"] += 1; continue
    stem = re.sub(r"[^A-Za-z0-9]+", "_", os.path.basename(a.mp3).split(".")[0]).strip("_")
    name = "%s_%09d.wav" % (stem, int(st * 1000))
    dst = os.path.join(a.outdir, "wavs", name)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "%.3f" % st, "-t", "%.3f" % d,
                    "-i", a.mp3, "-ac", "1", "-ar", str(cfg["export"]["sample_rate"]),
                    "-c:a", "pcm_s16le", dst], check=True)
    x, _ = sf.read(dst, dtype="float32")
    if np.mean(np.abs(x) >= 0.999) > cfg["quality"]["max_clip_ratio"]:
        os.remove(dst); drops["clipped"] += 1; continue
    kept.append({"wav": "wavs/" + name, "text": text, "dur": round(d, 2),
                 "cover": round(cover, 3), "confidence": round(float(conf), 3)})

with open(os.path.join(a.outdir, "metadata.csv"), "w", encoding="utf-8") as f:
    for r in kept:
        f.write("%s|%s|%s|%s\n" % (r["wav"], cfg["export"]["speaker"], cfg["export"]["lang"], r["text"]))
dump_json({"segments": kept, "drops": dict(drops)}, os.path.join(a.outdir, "manifest.json"))
print("kept %d segs (%.1f min); drops: %s" %
      (len(kept), sum(r["dur"] for r in kept) / 60, dict(drops)))
