"""Build human review sheets: 30 stratified kept segments + 10 dropped segments."""
import argparse, collections, json, os, random

ap = argparse.ArgumentParser()
ap.add_argument("manifest_json")            # out/<stem>/manifest.json
ap.add_argument("--outdir", default=None)
ap.add_argument("--n", type=int, default=30)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
m = json.load(open(a.manifest_json, encoding="utf-8"))
segs = sorted(m["segments"], key=lambda x: x.get("v2a_db", 0))
outdir = a.outdir or os.path.dirname(a.manifest_json)
random.seed(a.seed)

def sheet(rows, path, header):
    lines = [header]
    for r in rows:
        lines.append(",".join([r["wav"], str(r["dur"]), r["text"].replace(",", "，"), "", "", ""]))
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("->", path, len(rows), "rows")

n = min(a.n, len(segs))
if n:
    lo = segs[: n // 3]; mid = segs[len(segs) // 2 - n // 6: len(segs) // 2 + n // 6]; hi = segs[-n // 3:]
    sheet(lo + mid + hi, os.path.join(outdir, "review.csv"), "音频,时长s,机器文字,①多人在说,②是主播,③文字对")
print("drops:", dict(collections.Counter(json.load(open(a.manifest_json))["drops"])))
