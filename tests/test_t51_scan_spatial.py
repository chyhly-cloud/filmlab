"""T51: 冲印店扫描的空间传递 (filmlab.scanning.spatial; 修正标志 scan_spatial; 代际 t51_base = t46_base + scan_spatial,
t51_bw = t50_bw + scan_spatial; docs/tasks/T51_LAB_SCANNER_SPATIAL.md)。
现有代际不变; 新阶段是器件参数 (扫描光学 σ、传感器孔径、亮度 USM、色度降噪), 各项直流增益为 1 (均匀区与大面积颜色不变)。"""
from dataclasses import replace

import numpy as np
import pytest

from filmlab import engine, generation
from filmlab.engine import Engine, Params
from filmlab.recipes import recipe
from filmlab.scanning import spatial


@pytest.fixture(autouse=True)
def _restore_generation():
    prev = generation.active()
    yield
    generation.use(prev)


def test_generations():
    G = generation.GENERATIONS
    assert G["t51_base"] == ("D65", G["t46_base"][1] + ("scan_spatial",), {})
    assert G["t51_bw"] == ("D65", G["t50_bw"][1] + ("scan_spatial",), {})
    for g, (_, corr, _) in G.items():
        if g not in ("t51_base", "t51_bw", "t56_base", "t56_bw"):           # t56_* = t51_* + scan_mtf (T56)
            assert "scan_spatial" not in corr, g
    generation.use("t51_bw")
    assert engine.SCAN_SPATIAL_LAB == "noritsu_rpl"
    generation.use("t51_base")
    assert engine.SCAN_SPATIAL_LAB == "frontier_epfl"
    generation.use("t50_bw")
    assert engine.SCAN_SPATIAL_LAB is None


def test_measured_labs():
    for k in ("frontier_epfl", "noritsu_rpl"):
        lab = spatial.LABS[k]
        assert 0 < lab.optics_px < 2 and 5 < lab.pitch_um < 25
        assert lab.usm_amount >= 0 and lab.chroma_sigma_px >= 0
    # Frontier (EPFL) 的直边与颗粒都没有锐化; Noritsu (RPL) 有 (T51 第 3、4 节)
    assert spatial.LABS["frontier_epfl"].usm_amount == 0.0
    assert spatial.LABS["noritsu_rpl"].usm_amount > 0.3


def _random_image(seed=0, shape=(64, 80)):
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.05, 0.6, (1, 1, 3))
    return np.clip(base * (1 + 0.15 * rng.standard_normal((*shape, 3))), 1e-4, None)


@pytest.mark.parametrize("key", ["frontier_epfl", "noritsu_rpl"])
def test_uniform_field_unchanged(key):
    """各项直流增益为 1: 均匀色块 (含彩色) 不变。"""
    rgb = np.ones((40, 50, 3)) * np.array([0.30, 0.12, 0.05])
    out = spatial.apply(rgb, spatial.LABS[key], 11.9)
    np.testing.assert_allclose(out, rgb, rtol=1e-9, atol=1e-12)


def test_identity_when_all_off():
    lab = spatial.LabSpatial("off", 11.9, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    rgb = _random_image()
    np.testing.assert_allclose(spatial.apply(rgb, lab, 11.9), rgb, rtol=1e-10, atol=1e-13)


def test_chroma_nr_keeps_luma_and_mono_has_no_chroma():
    """色度降噪不改编码域 BT.601 亮度; 黑白输出三通道相同。"""
    lab = spatial.LabSpatial("c", 11.9, 0.0, 1.0, 0.0, 0.0, 2.5, 0.0)
    rgb = _random_image(1)
    Y = lambda x: spatial.encode(x) @ spatial._YCC[0]
    np.testing.assert_allclose(Y(spatial.apply(rgb, lab, 11.9)), Y(rgb), atol=1e-9)
    m = spatial.apply(rgb, spatial.LABS["noritsu_rpl"], 11.9, mono=True)
    assert np.all(m[..., 0] == m[..., 1]) and np.all(m[..., 1] == m[..., 2])


def test_usm_transfer_matches_formula():
    """小幅度正弦 (编码域) 经 USM 的增益 = 1 + a·(1 − exp(−2π²σ²f²))。"""
    a, s, f = 0.8, 0.9, 0.2
    x = np.arange(400)
    Y = 0.5 + 0.01 * np.cos(2 * np.pi * f * x)[None, :] * np.ones((8, 1))
    out = spatial.usm(Y, s, a)
    g = np.ptp(out[4, 100:300]) / np.ptp(Y[4, 100:300])
    assert abs(g - (1 + a * (1 - np.exp(-2 * np.pi ** 2 * s ** 2 * f ** 2)))) < 0.01


def test_chroma_threshold_preserves_edges_and_smooths_grain():
    """保边色度降噪: 大的色度台阶几乎不被展宽; 小幅色度起伏被平滑。"""
    rng = np.random.default_rng(3)
    C = np.zeros((32, 200)); C[:, 100:] = 60 / 255.0                   # 60 码值的 Cb 台阶
    noise = 1.5 / 255.0 * rng.standard_normal(C.shape)
    out = spatial.chroma_nr(C + noise, 2.5, threshold=8.0)
    lin = spatial.chroma_nr(C + noise, 2.5, threshold=0.0)
    edge = lambda z: np.abs(np.diff(z[:, 90:110].mean(0))).max()
    assert edge(out) > 0.8 * edge(C) and edge(lin) < 0.5 * edge(C)
    assert (out - C)[:, 20:80].std() < 0.35 * noise[:, 20:80].std()


def _render(g, lab=None, grain=0.0, fw=2.0):
    generation.use(g)
    kw = {**recipe("cinestill_800t"), "timing": "frame", "roll": None, "frame": 0, "grain": grain, "frame_width_mm": fw}
    if lab is not None:
        kw["scan_spatial"] = lab
    rng = np.random.default_rng(5)
    img = np.full((60, 80, 3), 0.05) + rng.uniform(0, 0.2, (60, 80, 3))
    img[20:40, 30:50] = (0.6, 0.3, 0.1)
    return Engine(Params(**kw)).render(img)


def test_engine_stage_equals_spatial_apply_on_t46_output(monkeypatch):
    """t51_base 的输出 = t46_base 的输出经 spatial.apply (扫描光学 σ 取同一值时)。"""
    fw = 0.8                                                            # 10 µm/像素: SCAN_PSF_UM / 像素 ≥ 0.3, 两条路径都加光学
    lab = replace(spatial.LABS["frontier_epfl"], optics_px=engine.SCAN_PSF_UM / (fw * 1000 / 80))
    monkeypatch.setitem(spatial.LABS, "test_lab", lab)
    a = _render("t46_base", fw=fw)
    b = _render("t51_base", "test_lab", fw=fw)
    np.testing.assert_allclose(b, spatial.apply(a, lab, fw * 1000 / 80), rtol=1e-7, atol=1e-10)


def test_params_choose_lab():
    generation.use("t51_base")
    e = Engine(Params(**{**recipe("cinestill_800t"), "roll": None, "grain": 0, "scan_spatial": "noritsu_rpl"}))
    assert e.scan_spatial_lab() is spatial.LABS["noritsu_rpl"]
    e = Engine(Params(**{**recipe("cinestill_800t"), "roll": None, "grain": 0}))
    assert e.scan_spatial_lab() is spatial.LABS["frontier_epfl"]
    generation.use("t46_base")
    assert Engine(Params(**{**recipe("cinestill_800t"), "roll": None, "grain": 0})).scan_spatial_lab() is None
