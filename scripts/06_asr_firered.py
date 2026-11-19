"""Server-side ASR: call FireRedASR2S CLI, convert its jsonl to our segment format.

Prerequisites: clone https://github.com/FireRedTeam/FireRedASR2S, install its
requirements and model weights, put its bin/ on PATH (or pass --cli).
"""
import argparse, json, os, subprocess

ap = argparse.ArgumentParser()
ap.add_argument("wav")                        # 16 kHz mono
ap.add_argument("--cli", default="fireredasr2s-cli")
ap.add_argument("--model-dir", default="pretrained_models/FireRedASR2-AED")
ap.add_argument("-o", "--out", default=None)
a = ap.parse_args()

os.makedirs("out/_firered_tmp", exist_ok=True)
subprocess.run([a.cli, "--wav_paths", a.wav, "--outdir", "out/_firered_tmp",
                "--model_dir", a.model_dir], check=True)
src = os.path.join("out/_firered_tmp", "result.jsonl")
out = []
with open(src, encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        for s in row.get("sentences", []):
            out.append({"start": round(s["start_ms"] / 1000, 3),
                        "end": round(s["end_ms"] / 1000, 3),
                        "text": s["text"].strip(),
                        "confidence": round(float(s.get("asr_confidence", 0)), 3)})
out.sort(key=lambda x: x["start"])
dst = a.out or os.path.splitext(a.wav)[0] + ".asr.json"
json.dump({"duration": row.get("dur_s"), "engine": "FireRedASR2-AED", "segments": out},
          open(dst, "w"), ensure_ascii=False)
print("%d segs -> %s" % (len(out), dst))
