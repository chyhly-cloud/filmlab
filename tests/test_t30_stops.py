"""T30: 截断块档数按"高出截断电平的档数"统一取值 (rawio.HIGHLIGHT_RECON = "stops", 代际 t30_stops / t30_stops_f)。"""
import numpy as np
import pytest

from filmlab import generation, rawio


def _scene():
    rgb = np.full((96, 96, 3), 0.03)
    clip = np.zeros((96, 96))
    for y, x in ((20, 20), (20, 60), (60, 30), (70, 70)):          # 4 盏灯 (灯具类)
        clip[y:y + 5, x:x + 5] = 1.0
    clip[45, 5:40] = 1.0                                           # 细线
    rgb[clip > 0] = 1.0
    return rgb, clip


def _run(mode, l_clip, seed=3):
    rgb, clip = _scene()
    old = rawio.HIGHLIGHT_RECON
    rawio.HIGHLIGHT_RECON, rawio.STOPS_LOG = mode, []
    try:
        out = rawio.reconstruct_highlights(rgb, clip, np.random.default_rng(seed), l_clip, 0.0)
        return out, rawio.STOPS_LOG
    finally:
        rawio.HIGHLIGHT_RECON, rawio.STOPS_LOG = old, None


def test_stops_independent_of_l_clip():
    ref, log = _run("stops", None)
    for lc in (3.24, 19.2, 1200.0):
        out, lg = _run("stops", lc)
        np.testing.assert_array_equal(out, ref)
        assert [b["stops"] for b in lg] == [b["stops"] for b in log]
    # 对照: 原先验随 L_clip 变
    a, _ = _run("prior", 19.2)
    b, _ = _run("prior", 1200.0)
    assert not np.array_equal(a, b)


def test_stops_ranges_and_relative_offsets():
    _, log = _run("stops", 50.0)
    lamps = [b["stops"] for b in log if b["cls"] == "lamp"]
    lines = [b["stops"] for b in log if b["cls"] == "line"]
    assert len(lamps) == 4 and len(lines) == 1
    assert all(rawio.STOPS_PRIOR[0] <= s <= rawio.STOPS_PRIOR[1] for s in lamps)
    lo, hi = rawio._stops_range(rawio.LINE_PRIOR[0])
    assert lo <= lines[0] <= hi
    # 偏移 = cd/m² 范围之比 (下限对下限, 上限对上限)
    assert lo - rawio.STOPS_PRIOR[0] == pytest.approx(np.log2(1e2 / 2e3))
    assert hi - rawio.STOPS_PRIOR[1] == pytest.approx(np.log2(2e3 / 3e4))
    glo, ghi = rawio._stops_range(rawio.HIGHLIGHT_PRIOR[1][1])
    assert (glo - rawio.STOPS_PRIOR[0], ghi - rawio.STOPS_PRIOR[1]) == pytest.approx((-1.0, np.log2(1e4 / 3e4)))


def test_prior_mode_unchanged_by_log():
    rgb, clip = _scene()
    a = rawio.reconstruct_highlights(rgb, clip, np.random.default_rng(3), 19.2, 0.0)
    b, _ = _run("prior", 19.2)
    np.testing.assert_array_equal(a, b)


def test_generations():
    prev = generation.active()
    try:
        generation.use("t30_stops")
        assert rawio.HIGHLIGHT_RECON == "stops" and rawio.METERING == "session" and rawio.RECON_COLOUR == "ring"
        assert generation.GENERATIONS["t30_stops"] == generation.GENERATIONS["t22_bxyz"]
        generation.use("t30_stops_f")
        assert rawio.HIGHLIGHT_RECON == "stops"
        assert generation.GENERATIONS["t30_stops_f"] == generation.GENERATIONS["t26_fwd"]
        generation.use("t22_bxyz")
        assert rawio.HIGHLIGHT_RECON == "prior"
    finally:
        generation.use(prev)
