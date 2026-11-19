"""Measure 10 s of audio: RMS, clipping, speech ratio (helper for checking a source)."""
import argparse, numpy as np, soundfile as sf
from silero_vad import load_silero_vad, get_speech_timestamps
import torch

ap = argparse.ArgumentParser(); ap.add_argument("wav"); ap.add_argument("--sec", type=int, default=10)
a = ap.parse_args()
d, sr = sf.read(a.wav, dtype="float32")
x = d[:a.sec * sr]
m = load_silero_vad()
ts = get_speech_timestamps(torch.from_numpy(x), m, sampling_rate=sr, return_seconds=True)
print(f"sr={sr} rms={np.sqrt((x*x).mean()):.4f} clipped={np.mean(np.abs(x)>=0.999):.4%} "
      f"speech={sum(t['end']-t['start'] for t in ts):.1f}s/{a.sec}s")
