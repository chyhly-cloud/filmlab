"""T37: 截断块亮度按实测分布抽样 (rawio.HIGHLIGHT_RECON = "measured", 代际 t37_base)。"""
import numpy as np
import pytest

from filmlab import generation, rawio


def _scene():
    rgb = np.full((120, 120, 3), 0.03)
    clip = np.zeros((120, 120))
    for y, x, s in ((10, 10, 2), (20, 60, 5), (60, 30, 9), (80, 80, 14)):     # 灯具类, 不同尺寸
        clip[y:y + s, x:x + s] = 1.0
    clip[45, 5:40] = 1.0                                                     # 细线
    clip[100:103, 100:110] = 0.5                                             # 部分截断
    rgb[clip > 0] = 1.0
    return rgb, clip


def _run(mode, l_clip=None, seed=5):
    rgb, clip = _scene()
    old = rawio.HIGHLIGHT_RECON
    rawio.HIGHLIGHT_RECON, rawio.STOPS_LOG = mode, []
    try:
        out = rawio.reconstruct_highlights(rgb, clip, np.random.default_rng(seed), l_clip, 0.0)
        return out, rawio.STOPS_LOG
    finally:
        rawio.HIGHLIGHT_RECON, rawio.STOPS_LOG = old, None


def test_table_shape():
    assert set(rawio.MEASURED_FLUX) == {"lamp", "line", "large"}
    for c, tabs in rawio.MEASURED_FLUX.items():
        assert len(tabs) == len(rawio.MEASURED_SIZE_UM) + 1
        for t in tabs:
            assert len(t) == len(rawio.MEASURED_Q)
            assert np.all(np.diff(t) >= 0) and t[0] >= 0


def test_block_flux_equals_sampled_flux():
    """每块的 f 加权 2^(stops·dome) 平均 (STOPS_LOG 的 flux) = 该类、该尺寸档分布在本块随机数 u 处的分位。"""
    _, log = _run("measured")
    us = np.random.default_rng(5).uniform(size=len(log))         # 每块一次, 与 stops 模式同一随机数序列
    rgb, _ = _scene()
    um = 36000.0 / max(rgb.shape[:2])
    for b, u in zip(log, us):
        want = rawio._measured_flux(b["cls"], b["r_eq"] * um, u)
        assert b["flux"] == pytest.approx(max(want, 0.0), abs=2e-3) or b["stops"] == 0.05


def test_same_uniform_as_stops_and_independent_of_l_clip():
    _, ls = _run("stops")
    _, lm = _run("measured")
    assert [b["area"] for b in ls] == [b["area"] for b in lm] and [b["cls"] for b in ls] == [b["cls"] for b in lm]
    ref, _ = _run("measured", None)
    for lc in (3.24, 1200.0):
        out, _ = _run("measured", lc)
        np.testing.assert_array_equal(out, ref)


def test_solver_monotone_and_point_source():
    f = np.ones(1); dome = np.ones(1)
    assert rawio._measured_stops(3.3, dome, f) == pytest.approx(3.3, abs=1e-6)   # 单像素块: s = flux
    d = np.linspace(0.1, 1.0, 50); f = np.ones(50)
    s1, s2 = (rawio._measured_stops(x, d ** rawio.MEASURED_DOME, f) for x in (2.0, 4.0))
    assert s2 > s1 > 2.0


def test_other_modes_unchanged():
    """stops / prior 模式不受 MEASURED_DOME 影响。"""
    a, _ = _run("stops")
    old = rawio.MEASURED_DOME
    rawio.MEASURED_DOME = 2.5
    try:
        b, _ = _run("stops")
        c, _ = _run("prior", 19.2)
        rawio.MEASURED_DOME = old
        d, _ = _run("prior", 19.2)
    finally:
        rawio.MEASURED_DOME = old
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(c, d)


def test_generation():
    prev = generation.active()
    try:
        generation.use("t37_base")
        assert rawio.HIGHLIGHT_RECON == "measured" and rawio.METERING == "session"
        assert generation.GENERATIONS["t37_base"] == generation.GENERATIONS["t31_base"]
        assert generation.MASK_PLACEMENT["t37_base"] == "layers"
        generation.use("t31_base")
        assert rawio.HIGHLIGHT_RECON == "stops"
    finally:
        generation.use(prev)
