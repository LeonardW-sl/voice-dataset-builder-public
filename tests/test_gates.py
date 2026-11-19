"""Gate logic tests: the rules that decide keep/drop."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import pytest
from common import merge_intervals, clip_ratio
import numpy as np


def test_merge_close_intervals():
    assert merge_intervals([(0, 1), (1.1, 2), (3, 4)], gap=0.2) == [(0.0, 2.0), (3.0, 4.0)]


def test_merge_keeps_wide_gap():
    assert merge_intervals([(0, 1), (2, 3)], gap=0.1) == [(0.0, 1.0), (2.0, 3.0)]


def test_merge_empty():
    assert merge_intervals([], gap=0.1) == []


def test_clip_ratio_detects_full_scale():
    x = np.array([0.0, 0.5, 1.0, -1.0], dtype=np.float32)
    assert clip_ratio(x) == pytest.approx(0.5)


def test_clip_ratio_clean():
    assert clip_ratio(np.zeros(10, dtype=np.float32)) == 0.0
