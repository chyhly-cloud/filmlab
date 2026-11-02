"""T38: 截断块亮度按观测量条件化的实测分布 (rawio.HIGHLIGHT_RECON = "measured_cond", 代际 t38_base)。"""
import numpy as np
import pytest

from filmlab import generation, rawio


def _scene(bg_right=0.003):
    """左半暗背景、右半亮背景 (亮面), 各放两盏同样的灯。"""
    rgb = np.full((120, 160, 3), 0.003)
    rgb[:, 80:] = bg_right
    clip = np.zeros((120, 160))
    for y, x in ((30, 20), (80, 40), (30, 110), (80, 130)):
        clip[y:y + 4, x:x + 4] = 1.0
    rgb[clip > 0] = 1.0
    return rgb, clip


def _run(mode, rgb, clip, seed=7):
    old = rawio.HIGHLIGHT_RECON
    rawio.HIGHLIGHT_RECON, rawio.STOPS_LOG = mode, []
    try:
        out = rawio.reconstruct_highlights(rgb, clip, np.random.default_rng(seed), None, 0.0)
        return out, rawio.STOPS_LOG
    finally:
        rawio.HIGHLIGHT_RECON, rawio.STOPS_LOG = old, None


def test_table_shape():
    T = rawio.MEASURED_COND
    nb = len(T["edges"]) + 1
    assert np.all(np.diff(T["edges"]) > 0)
    for c in ("lamp", "line", "large"):
        assert len(T["flux"][c]) == 3 and len(T["flux_all"][c]) == 3
        for k in range(3):
            assert len(T["flux"][c][k]) == nb
            for t in T["flux"][c][k] + [T["flux_all"][c][k]]:
                assert len(t) == len(rawio.MEASURED_Q) and np.all(np.diff(t) >= 0)


def test_block_obs_background():
    rgb, clip = _scene(bg_right=0.3)
    _, log = _run("measured_cond", rgb, clip)
    dark = [b["obs"]["bg"] for b in log if b["cx"] < 80]
    bright = [b["obs"]["bg"] for b in log if b["cx"] >= 80]
    assert len(dark) == 2 and len(bright) == 2
    assert max(dark) < -6 and min(bright) == pytest.approx(np.log2(0.3), abs=0.05)
    for b in log:
        assert b["obs"]["fmean"] == pytest.approx(1.0) and b["obs"]["r"] == pytest.approx(np.sqrt(16 / np.pi))


def test_flux_equals_conditional_quantile():
    rgb, clip = _scene(bg_right=0.3)
    _, log = _run("measured_cond", rgb, clip)
    us = np.random.default_rng(7).uniform(size=len(log))
    um = 36000.0 / max(rgb.shape[:2])
    for b, u in zip(log, us):
        want = rawio._cond_flux(b["cls"], b["r_eq"] * um, b["obs"], u)
        assert b["flux"] == pytest.approx(max(want, 0.0), abs=2e-3) or b["stops"] == 0.05


def test_same_uniform_and_classes_as_measured():
    rgb, clip = _scene()
    _, a = _run("measured", rgb, clip)
    _, b = _run("measured_cond", rgb, clip)
    assert [x["area"] for x in a] == [x["area"] for x in b] and [x["cls"] for x in a] == [x["cls"] for x in b]


def test_condition_changes_distribution():
    """同一随机数, 暗背景与亮背景的同一盏灯抽到的 flux 由各自观测档决定。"""
    T = rawio.MEASURED_COND
    lo, hi = T["edges"][0] - 1.0, T["edges"][-1] + 1.0
    a = rawio._cond_flux("lamp", 30.0, {T["var"]: lo}, 0.5)
    b = rawio._cond_flux("lamp", 30.0, {T["var"]: hi}, 0.5)
    assert a != b
    assert rawio._cond_flux("lamp", 30.0, {T["var"]: float("nan")}, 0.5) == pytest.approx(np.interp(0.5, rawio.MEASURED_Q, T["flux_all"]["lamp"][0]))


def test_other_modes_unchanged():
    rgb, clip = _scene()
    a, _ = _run("measured", rgb, clip)
    old = rawio.MEASURED_COND
    rawio.MEASURED_COND = None
    try:
        b, _ = _run("measured", rgb, clip)
        c, _ = _run("stops", rgb, clip)
    finally:
        rawio.MEASURED_COND = old
    d, _ = _run("stops", rgb, clip)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(c, d)


def test_generation():
    prev = generation.active()
    try:
        generation.use("t38_base")
        assert rawio.HIGHLIGHT_RECON == "measured_cond" and rawio.METERING == "session"
        assert generation.GENERATIONS["t38_base"] == generation.GENERATIONS["t31_base"]
        assert generation.MASK_PLACEMENT["t38_base"] == "layers"
    finally:
        generation.use(prev)
