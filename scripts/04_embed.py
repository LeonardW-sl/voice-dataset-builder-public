"""CAM++ speaker verification: optional reference, else reveal cluster structure."""
import argparse, json, os, sys
import numpy as np
import soundfile as sf
from funasr import AutoModel

ap = argparse.ArgumentParser()
ap.add_argument("wav")                     # 16 kHz mono
ap.add_argument("aed_json")
ap.add_argument("-o", "--out", default=None)
ap.add_argument("--model", default="iic/speech_campplus_sv_zh-cn_16k-common")
ap.add_argument("--ref-wav", default=None, help="30-60 s clean reference of the anchor speaker")
ap.add_argument("--min-dur", type=float, default=1.0)
a = ap.parse_args()

aed = json.load(open(a.aed_json, encoding="utf-8"))
segs = [s for s in aed["event2timestamps"].get("speech", []) if s[1] - s[0] >= a.min_dur]
data, sr = sf.read(a.wav, dtype="float32"); assert sr == 16000

model = AutoModel(model=a.model, device="cpu", disable_update=True)
def emb(chunk):
    r = model.generate(input=chunk, embedding=True)
    v = np.array(r[0]["spk_embedding"]).reshape(-1)
    return v / np.linalg.norm(v)
ref = None
if a.ref_wav:
    rd, rs = sf.read(a.ref_wav, dtype="float32")
    ref = emb(rd[:rs * 60] if rs > 16000 else rd)
    print("reference loaded: %.1fs" % (len(rd) / rs))
E = np.array([emb(data[int(s * sr):int(e * sr)]) for s, e in segs])
S = E @ E.T; iu = S[np.triu_indices(len(E), 1)]
print("segments %d; pairwise cosine p10/p50/p90 = %s" % (len(E), np.percentile(iu, [10, 50, 90]).round(3)))
if ref is not None:
    sim = E @ ref
    print("similarity to reference p10/p50/p90 = %s" % np.percentile(sim, [10, 50, 90]).round(3))
rows = [{"start": round(s, 3), "end": round(e, 3),
         "sim": round(float(E[i] @ ref), 4) if ref is not None else None} for i, (s, e) in enumerate(segs)]
dst = a.out or os.path.splitext(a.wav)[0] + ".emb.json"
json.dump(rows, open(dst, "w"), ensure_ascii=False)
print("->", dst)
