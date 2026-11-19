"""Decode an mp3 to 16 kHz mono wav (ffmpeg must be on PATH)."""
import argparse, subprocess, sys

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("out_wav")
a = ap.parse_args()
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", a.src,
                "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", a.out_wav], check=True)
print("->", a.out_wav)
