"""Multi-label AED (speech / singing / music) with FireRedVAD."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_config

from fireredvad import FireRedAed, FireRedAedConfig

ap = argparse.ArgumentParser()
ap.add_argument("wav")
ap.add_argument("-o", "--out", default=None)
ap.add_argument("--config", default="config/default.yaml")
ap.add_argument("--model-dir", default=None)
a = ap.parse_args()
cfg = load_config(a.config)["aed"]
default_dir = cfg.pop("model_dir")
model_dir = a.model_dir or default_dir
if not os.path.isabs(model_dir):
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), model_dir)

t0 = time.time()
model = FireRedAed.from_pretrained(model_dir, config=FireRedAedConfig(use_gpu=False, **cfg))
res = model.detect(a.wav)
if isinstance(res, tuple):
    res = res[0]
dst = a.out or os.path.splitext(a.wav)[0] + ".aed.json"
with open(dst, "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False)
print("dur %.1fs, elapsed %.1fs -> %s" % (res["dur"], time.time() - t0, dst))
for k in ("speech", "singing", "music"):
    v = res["event2timestamps"].get(k, [])
    print("  %-7s %3d segs, %6.1fs (ratio %.3f)" % (k, len(v), sum(e - s for s, e in v), res["event2ratio"].get(k, 0)))
