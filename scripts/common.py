"""Shared helpers: config loading and small audio utilities."""
import json
import os

import numpy as np

DEFAULTS = {
    "aed": {
        "model_dir": "pretrained_models/FireRedVAD/AED",
        "speech_threshold": 0.5,
        "min_event_frame": 5,       # 10 ms per frame
        "min_silence_frame": 15,
        "merge_silence_frame": 20,
        "extend_speech_frame": 5,
        "chunk_max_frame": 1000000,
    },
    "seg": {"min_dur": 1.0, "max_dur": 15.0},
    "embed": {"model": "iic/speech_campplus_sv_zh-cn_16k-common", "device": "cpu", "min_sim": 0.0},
    "asr": {"min_confidence": 0.35, "min_cover": 0.7, "min_chars": 4, "max_cps": 7.0},
    "quality": {"min_v2a_db": 20.0, "max_clip_ratio": 0.01},
    "export": {"sample_rate": 32000, "speaker": "speaker", "lang": "ZH"},
}


def load_config(path=None):
    """Load YAML config merged over DEFAULTS (one level deep)."""
    import copy

    cfg = copy.deepcopy(DEFAULTS)
    if path and os.path.exists(path):
        import yaml

        with open(path, encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        for section, values in user.items():
            if isinstance(values, dict) and isinstance(cfg.get(section), dict):
                cfg[section].update(values)
            else:
                cfg[section] = values
    return cfg


def merge_intervals(intervals, gap):
    """Merge [start, end] pairs whose gap is below `gap` seconds."""
    if not intervals:
        return []
    out = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def clip_ratio(x):
    """Fraction of samples at or beyond full scale."""
    x = np.asarray(x, dtype=np.float32)
    return float(np.mean(np.abs(x) >= 0.999)) if x.size else 0.0


def dump_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)