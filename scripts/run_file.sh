#!/usr/bin/env bash
# 单文件全流程，已完成的步骤自动跳过（断电重跑不重复计算）。
# 用法: bash scripts/run_file.sh <录音.mp3> [outdir]
# 可选: STEP_SEP=1 额外跑 Demucs 打分（CPU 约 1x 实时，很慢）
set -euo pipefail
MP3="$1"; OUT="${2:-out/run}"; PY="${PY:-python}"
BASE="$(basename "${MP3%.*}")"; STEM="$(printf '%s' "$BASE" | sed 's/[^A-Za-z0-9]\+/_/g; s/^_//; s/_$//')"
W="$OUT/work/$STEM"; mkdir -p "$W"
CFG="${CFG:-config/default.yaml}"

step() { echo "[step] $*"; }

step "decode $STEM"
[ -f "$W/a.wav" ] || $PY scripts/00_decode.py "$MP3" "$W/a.wav"

step "aed"
[ -f "$W/a.aed.json" ] || $PY scripts/02_aed.py "$W/a.wav" -o "$W/a.aed.json" --config "$CFG"

step "speaker embedding"
[ -f "$W/a.emb.json" ] || $PY scripts/04_embed.py "$W/a.wav" "$W/a.aed.json" -o "$W/a.emb.json" \
    $( [ -n "${REF_WAV:-}" ] && echo "--ref-wav $REF_WAV" )

if [ "${STEP_SEP:-0}" = "1" ]; then
  step "demucs V2A scoring (slow on CPU)"
  [ -f "$W/a.sep.json" ] || $PY scripts/03_sep_score.py "$W/a.wav" "$W/a.aed.json" -o "$W/a.sep.json"
fi

step "asr"
[ -f "$W/a.asr.json" ] || $PY "${ASR_SCRIPT:-scripts/05_asr_whisper.py}" "$W/a.wav" -o "$W/a.asr.json"

step "export"
$PY scripts/07_export.py "$MP3" "$W/a.aed.json" "$W/a.asr.json" -o "$OUT/$STEM" --config "$CFG" \
    $( [ -f "$W/a.emb.json" ] && echo "--emb-json $W/a.emb.json" ) \
    $( [ -f "$W/a.sep.json" ] && echo "--sep-json $W/a.sep.json" )
echo "[done] $OUT/$STEM"
