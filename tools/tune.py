"""Plot / print distributions of V2A and similarity scores to pick thresholds."""
import argparse, json, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("json", nargs="+", help="sep.json or emb.json")
ap.add_argument("--bins", type=int, default=20)
a = ap.parse_args()
vals = []
for path in a.json:
    rows = json.load(open(path, encoding="utf-8"))
    for r in rows:
        if r.get("v2a_db") is not None:
            vals.append(("v2a_db", r["v2a_db"]))
        elif r.get("sim") is not None:
            vals.append(("sim", r["sim"]))
for key in ("v2a_db", "sim"):
    v = [x for k, x in vals if k == key]
    if not v:
        continue
    v = np.array(v)
    hist, edges = np.histogram(v, bins=a.bins)
    print(f"\n== {key} (n={len(v)}) p10/p50/p90 = {np.percentile(v,[10,50,90]).round(2)}")
    for h, e0, e1 in zip(hist, edges[:-1], edges[1:]):
        print(f"{e0:8.2f} .. {e1:8.2f}  {'#' * int(h / max(hist.max(), 1) * 50)} {h}")
