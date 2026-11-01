"""T13 source_stops (clipped-source brightness offset) synthetic checks."""
import numpy as np
from filmlab import rawio


def scene():
    rgb = np.full((64, 64, 3), 0.05)
    rgb[28:36, 28:36] = 1.0                              # clipped lamp
    clip = np.zeros((64, 64)); clip[28:36, 28:36] = 1.0
    return rgb, clip


def recon(offset):
    rgb, clip = scene()
    return rawio.reconstruct_highlights(rgb, clip, np.random.default_rng(7), None, offset)


def test_zero_offset_is_unchanged():
    rgb, clip = scene()
    a = rawio.reconstruct_highlights(rgb, clip, np.random.default_rng(7), None)
    np.testing.assert_array_equal(a, recon(0.0))


def test_offset_is_monotone_and_stays_above_clip():
    peaks = [recon(o)[32, 32].sum() for o in (-3.0, -1.0, 0.0, 1.0, 3.0)]
    assert all(b >= a for a, b in zip(peaks, peaks[1:]))          # 负偏移可能都触到 0.05 档下限
    assert peaks[2] < peaks[3] < peaks[4]
    low = recon(-10.0)
    rgb, _ = scene()
    assert low[28:36, 28:36].sum(-1).min() >= rgb[28:36, 28:36].sum(-1).min() - 1e-12


def test_shift_stops_bounds():
    assert rawio._shift_stops(3.0, 0.0) == 3.0
    assert rawio._shift_stops(3.0, -5.0) == 0.05
    assert rawio._shift_stops(7.5, 2.0) == 9.5
