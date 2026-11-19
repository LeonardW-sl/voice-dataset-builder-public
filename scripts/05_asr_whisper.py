"""CPU baseline ASR with faster-whisper. For pipeline validation only, NOT final text."""
import argparse, json, math, os, time
import soundfile as sf
from faster_whisper import WhisperModel

ap = argparse.ArgumentParser()
ap.add_argument("wav")
ap.add_argument("--model", default="small")
ap.add_argument("--compute", default="int8")
ap.add_argument("-o", "--out", default=None)
a = ap.parse_args()
t0 = time.time()
audio, sr = sf.read(a.wav, dtype="float32")
assert sr == 16000
m = WhisperModel(a.model, device="cpu", compute_type=a.compute, cpu_threads=6)
segs, info = m.transcribe(audio, language="zh", beam_size=5, vad_filter=True,
                          vad_parameters={"min_silence_duration_ms": 300},
                          word_timestamps=False, condition_on_previous_text=False)
out = [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip(),
        "confidence": round(math.exp(s.avg_logprob), 3)} for s in segs]
dst = a.out or os.path.splitext(a.wav)[0] + ".asr.json"
json.dump({"duration": info.duration, "engine": "faster-whisper:" + a.model, "segments": out},
          open(dst, "w"), ensure_ascii=False)
print("%d segs in %.0fs (RTF %.2f) -> %s" % (len(out), time.time() - t0, (time.time() - t0) / info.duration, dst))
