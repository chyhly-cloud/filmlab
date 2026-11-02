"""T39: 截断块亮度按 块外背景 × 传感器三通道饱和比例 条件化的实测分布 (rawio.HIGHLIGHT_RECON = "measured_cond2", 代际 t39_base)。"""
import numpy as np
import pytest

from filmlab import generation, rawio


def _scene():
    rgb = np.full((120, 160, 3), 0.003)
    clip = np.zeros((120, 160))
    for y, x in ((30, 20), (80, 40), (30, 110), (80, 130)):          # 4 盏同样大小的灯
        clip[y:y + 4, x:x + 4] = 1.0
    rgb[clip > 0] = 1.0
    # 左边两盏三通道全饱和, 右边两盏只一个通道饱和
    f_any = (clip > 0).astype(np.float32)
    f_all = f_any.copy(); f_all[:, 80:] = 0.0
    f_nch = np.where(np.arange(160)[None, :] < 80, 3.0, 1.0).astype(np.float32) * f_any
    return rgb, clip, (f_any, f_all, f_nch)


def _run(mode, sat=None, seed=11):
    rgb, clip, s = _scene()
    old = rawio.HIGHLIGHT_RECON
    rawio.HIGHLIGHT_RECON, rawio.STOPS_LOG = mode, []
    try:
        out = rawio.reconstruct_highlights(rgb, clip, np.random.default_rng(seed), None, 0.0, sat=s if sat is None else sat)
        return out, rawio.STOPS_LOG
    finally:
        rawio.HIGHLIGHT_RECON, rawio.STOPS_LOG = old, None


def test_table_shape():
    T = rawio.MEASURED_COND2
    nb, nn = len(T["edges"]["bg"]) + 1, len(T["edges"]["nsat3"]) + 1
    for c in ("lamp", "line", "large"):
        for k in range(3):
            assert len(T["flux"][c][k]) == nb and all(len(r) == nn for r in T["flux"][c][k])
            for t in [t for r in T["flux"][c][k] for t in r] + T["flux_ns"][c][k] + [T["flux_all"][c][k]]:
                assert len(t) == len(rawio.MEASURED_Q) and np.all(np.diff(t) >= 0)


def test_block_obs_nsat():
    _, log = _run("measured_cond2")
    left = [b["obs"] for b in log if b["cx"] < 80]
    right = [b["obs"] for b in log if b["cx"] >= 80]
    assert all(o["nsat3"] == pytest.approx(1.0) and o["nch"] == pytest.approx(3.0) for o in left)
    assert all(o["nsat3"] == pytest.approx(0.0) and o["nch"] == pytest.approx(1.0) for o in right)


def test_flux_equals_conditional_quantile_and_monotone_in_nsat():
    _, log = _run("measured_cond2")
    us = np.random.default_rng(11).uniform(size=len(log))
    um = 36000.0 / 160
    for b, u in zip(log, us):
        want = rawio._cond2_flux(b["cls"], b["r_eq"] * um, b["obs"], u)
        assert b["flux"] == pytest.approx(max(want, 0.0), abs=2e-3) or b["stops"] == 0.05
    # 同一 u、同一背景: 三通道全饱和的块的 flux 分位不低于只一个通道饱和的块
    T = rawio.MEASURED_COND2
    for u in (0.25, 0.5, 0.75):
        lo = rawio._cond2_flux("lamp", 30.0, {"bg": -5.0, "nsat3": 0.0}, u)
        hi = rawio._cond2_flux("lamp", 30.0, {"bg": -5.0, "nsat3": 1.0}, u)
        assert hi > lo
    assert rawio._cond2_flux("lamp", 30.0, {"bg": float("nan"), "nsat3": 1.0}, 0.5) == pytest.approx(
        np.interp(0.5, rawio.MEASURED_Q, T["flux_ns"]["lamp"][0][-1]))


def test_no_sat_info_means_nsat_zero():
    rgb, clip, _ = _scene()
    old = rawio.HIGHLIGHT_RECON
    rawio.HIGHLIGHT_RECON, rawio.STOPS_LOG = "measured_cond2", []
    try:
        rawio.reconstruct_highlights(rgb, clip, np.random.default_rng(1), None, 0.0)
        assert all(b["obs"]["nsat3"] == 0.0 for b in rawio.STOPS_LOG)
    finally:
        rawio.HIGHLIGHT_RECON, rawio.STOPS_LOG = old, None


def test_same_uniform_and_classes_as_measured():
    _, a = _run("measured")
    _, b = _run("measured_cond2")
    assert [x["area"] for x in a] == [x["area"] for x in b] and [x["cls"] for x in a] == [x["cls"] for x in b]


def test_seed_offset_default_and_other_modes():
    assert rawio.SEED_OFFSET == 0
    rgb, clip, s = _scene()
    a = rawio.reconstruct_highlights(rgb, clip, np.random.default_rng(3), None, 0.0)
    b = rawio.reconstruct_highlights(rgb, clip, np.random.default_rng(3), None, 0.0, sat=s)   # prior 模式忽略 sat
    np.testing.assert_array_equal(a, b)


def test_generation():
    prev = generation.active()
    try:
        generation.use("t39_base")
        assert rawio.HIGHLIGHT_RECON == "measured_cond2" and rawio.METERING == "session" and rawio.CLIP_DETECT == "channel_max"
        assert generation.GENERATIONS["t39_base"] == generation.GENERATIONS["t31_base"]
        assert generation.MASK_PLACEMENT["t39_base"] == "layers"
    finally:
        generation.use(prev)


def test_corr_mode_marginal_and_sharing():
    """t39_corr: 同一帧同一条件格共享 z_组 (确定性), ρ = 1 时同格的块分位相同; 不同帧不同。"""
    obs = {"bg": -5.0, "nsat3": 1.0}
    r1 = np.random.default_rng(1234)
    a = rawio._cond2_corr_u(r1, "lamp", 30.0, obs, 0.3)
    b = rawio._cond2_corr_u(np.random.default_rng(1234), "lamp", 30.0, obs, 0.3)
    assert a == b
    old = dict(rawio.MEASURED_COND2_ICC)
    try:
        rawio.MEASURED_COND2_ICC["lamp"] = 1.0 - 1e-12
        x = rawio._cond2_corr_u(r1, "lamp", 30.0, obs, 0.1)
        y = rawio._cond2_corr_u(r1, "lamp", 30.0, obs, 0.9)
        assert x == pytest.approx(y, abs=1e-4)
        rawio.MEASURED_COND2_ICC["lamp"] = 0.0
        assert rawio._cond2_corr_u(r1, "lamp", 30.0, obs, 0.37) == pytest.approx(0.37)
    finally:
        rawio.MEASURED_COND2_ICC.update(old)
    # 边缘分布不变: 对很多帧, u' 仍近似均匀
    us = [rawio._cond2_corr_u(np.random.default_rng(s), "lamp", 30.0, obs, float(u))
          for s, u in zip(range(2000), np.random.default_rng(0).uniform(size=2000))]
    assert abs(np.mean(us) - 0.5) < 0.03 and abs(np.var(us) - 1 / 12) < 0.01
