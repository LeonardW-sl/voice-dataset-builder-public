"""Score V2A (vocal-to-accompaniment dB) with Demucs. SCORING ONLY — never export separated audio."""
import argparse, json, os, sys, time
import numpy as np
import soundfile as sf
import torch
import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model

ap = argparse.ArgumentParser()
ap.add_argument("wav")             # 16 kHz mono
ap.add_argument("aed_json")
ap.add_argument("-o", "--out", default=None)
ap.add_argument("--model", default="htdemucs")
ap.add_argument("--min-dur", type=float, default=1.0)
a = ap.parse_args()

aed = json.load(open(a.aed_json, encoding="utf-8"))
segs = [s for s in aed["event2timestamps"].get("speech", []) if s[1] - s[0] >= a.min_dur]
data, sr = sf.read(a.wav, dtype="float32")
assert sr == 16000, f"expect 16k wav, got {sr}"

print("loading demucs %s ..." % a.model, flush=True)
model = get_model(a.model); model.eval()
t0 = time.time()
rows = []
for i, (st, en) in enumerate(segs):
    x = torch.from_numpy(data[int(st * sr):int(en * sr)])
    x = torchaudio.functional.resample(x, sr, 44100)
    with torch.no_grad():
        out = apply_model(model, x[None, None].repeat(1, 2, 1), split=True, overlap=0.1)[0]
    e = out.pow(2).mean(dim=(1, 2)).sqrt()
    vi = model.sources.index("vocals")
    voc = float(e[vi]); acc = float(e.sum() - e[vi])
    db = 20 * np.log10(max(voc, 1e-8) / max(acc, 1e-8))
    rows.append({"start": round(st, 3), "end": round(en, 3), "v2a_db": round(float(db), 2)})
    if (i + 1) % 20 == 0:
        print("  %d/%d, %.0fs" % (i + 1, len(segs), time.time() - t0), flush=True)

dst = a.out or os.path.splitext(a.wav)[0] + ".sep.json"
json.dump(rows, open(dst, "w"), ensure_ascii=False)
vals = [r["v2a_db"] for r in rows]
print("scored %d segs in %.0fs; V2A dB p10/p50/p90 = %s -> %s" %
      (len(rows), time.time() - t0, *np.percentile(vals, [10, 50, 90]).round(1)))
print("->", dst)
