"""T56: 冲印店空间传递的高斯按连续传递 (修正标志 scan_mtf; 代际 t56_base = t51_base + scan_mtf, t56_bw = t51_bw + scan_mtf;
docs/tasks/T56_SCAN_MTF_CONTINUOUS.md)。T51 拟合 σ_o / σ_u 时用的是连续传递 e^{−2π²σ²f²}; 输出像素上的取样高斯核
对亚像素 σ 近似恒等。现有代际不变 (默认 continuous=False 走原来的 gaussian_filter)。"""
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from filmlab import engine, generation
from filmlab.engine import Engine, Params
from filmlab.recipes import recipe
from filmlab.scanning import spatial


@pytest.fixture(autouse=True)
def _restore_generation():
    prev = generation.active()
    yield
    generation.use(prev)


def _gain(fn, f, n=512):
    """小幅度余弦 (沿第 1 轴) 经 fn 的增益, 只取远离边界的中段。"""
    x = np.arange(n)
    Y = 0.5 + 0.01 * np.cos(2 * np.pi * f * x)[None, :] * np.ones((8, 1))
    out = fn(Y)
    return np.ptp(out[4, n // 4: 3 * n // 4]) / np.ptp(Y[4, n // 4: 3 * n // 4])


@pytest.mark.parametrize("s,f", [(0.35, 0.45), (0.35, 0.25), (0.68, 0.4), (1.75, 0.15)])
def test_gauss_mtf_is_continuous_transfer(s, f):
    g = _gain(lambda Y: spatial.gauss_mtf(Y, s), f)
    assert abs(g - np.exp(-2 * np.pi ** 2 * s ** 2 * f ** 2)) < 2e-3


def test_sampled_kernel_is_near_identity_for_subpixel_sigma():
    """问题本身: σ = 0.35 像素的取样核在 0.45 周/像素处传递约 0.94, 连续传递 0.61。"""
    g = _gain(lambda Y: gaussian_filter(Y, 0.35, mode="reflect"), 0.45)
    assert g > 0.9 and np.exp(-2 * np.pi ** 2 * 0.35 ** 2 * 0.45 ** 2) < 0.65


def test_gauss_mtf_equals_full_reflect_padding():
    """DCT-II 实现 = 整幅反射填充 (d c b a | a b c d) 后 FFT 相乘再裁回。"""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((23, 30, 3))
    s = 0.6
    p = np.pad(a, ((23, 23), (30, 30), (0, 0)), mode="symmetric")[:46, :60]   # 一个周期 (2H, 2W)
    fy, fx = np.fft.fftfreq(46)[:, None], np.fft.fftfreq(60)[None, :]
    H = np.exp(-2 * np.pi ** 2 * s ** 2 * (fy ** 2 + fx ** 2))[..., None]
    ref = np.real(np.fft.ifft2(np.fft.fft2(p, axes=(0, 1)) * H, axes=(0, 1)))
    ref = np.roll(ref, (-23, -30), (0, 1))[:23, :30]
    np.testing.assert_allclose(spatial.gauss_mtf(a, s), ref, atol=1e-12)


def test_gauss_mtf_matches_sampled_kernel_when_well_sampled():
    rng = np.random.default_rng(1)
    a = gaussian_filter(rng.standard_normal((64, 64)), 1.0)
    np.testing.assert_allclose(spatial.gauss_mtf(a, 3.0), gaussian_filter(a, 3.0, mode="reflect", truncate=8.0), atol=2e-4)


@pytest.mark.parametrize("key", ["frontier_epfl", "noritsu_rpl"])
def test_uniform_field_unchanged_continuous(key):
    rgb = np.ones((40, 50, 3)) * np.array([0.30, 0.12, 0.05])
    np.testing.assert_allclose(spatial.apply(rgb, spatial.LABS[key], 11.9, continuous=True), rgb, rtol=1e-9, atol=1e-12)
    m = spatial.apply(rgb, spatial.LABS[key], 11.9, mono=True, continuous=True)
    assert np.all(m[..., 0] == m[..., 1]) and np.all(m[..., 1] == m[..., 2])


def test_usm_continuous_matches_t51_formula_at_high_frequency():
    """T51 的 U(f) = 1 + a·(1 − e^{−2π²σ_u²f²}) 在 0.45 周/像素: 连续实现吻合; 取样核偏弱 (Noritsu a = 1.03, σ_u = 0.68)。"""
    a, s, f = 1.03, 0.68, 0.45
    U = 1 + a * (1 - np.exp(-2 * np.pi ** 2 * s ** 2 * f ** 2))
    assert abs(_gain(lambda Y: spatial.usm(Y, s, a, continuous=True), f) - U) < 5e-3
    assert U - _gain(lambda Y: spatial.usm(Y, s, a), f) > 0.05


def test_default_is_unchanged():
    """continuous 默认 False: 与 scipy 取样核逐位相同 (现有代际不变)。"""
    rng = np.random.default_rng(2)
    rgb = np.clip(0.2 + 0.05 * rng.standard_normal((30, 40, 3)), 1e-4, None)
    lab = spatial.LABS["noritsu_rpl"]
    v = spatial.encode(rgb) @ spatial._YCC.T
    Y = v[..., 0] + lab.usm_amount * (v[..., 0] - gaussian_filter(v[..., 0], lab.usm_sigma_px, mode="reflect"))
    v[..., 0] = Y
    np.testing.assert_array_equal(spatial.apply(rgb, lab, 11.9), spatial.decode(v @ spatial._YCC_INV.T))


def test_generations():
    G = generation.GENERATIONS
    assert G["t56_base"] == ("D65", G["t51_base"][1] + ("scan_mtf",), {})
    assert G["t56_bw"] == ("D65", G["t51_bw"][1] + ("scan_mtf",), {})
    for g, (_, corr, _) in G.items():
        if g not in ("t56_base", "t56_bw"):
            assert "scan_mtf" not in corr, g
    generation.use("t56_bw")
    assert engine.SCAN_SPATIAL_LAB == "noritsu_rpl"
    generation.use("t56_base")
    assert engine.SCAN_SPATIAL_LAB == "frontier_epfl"


def _engine(g, **kw):
    generation.use(g)
    return Engine(Params(**{**recipe("kodak_trix_400"), "timing": "gray", "roll": None, "frame": 0, "grain": 0.0,
                            "frame_width_mm": 1.28, **kw}))


def test_engine_scan_psf_uses_continuous_transfer():
    rng = np.random.default_rng(4)
    Dn = 0.8 + 0.1 * rng.standard_normal((32, 40, 3))
    Dn[3, 5, 1] = np.nan
    sp = spatial.LABS["noritsu_rpl"].optics_px
    e56, e51 = _engine("t56_bw"), _engine("t51_bw")
    e56.px_um = e51.px_um = 20.0
    T = 10.0 ** -np.nan_to_num(Dn, nan=0.0)
    ref56 = np.where(np.isfinite(Dn), -np.log10(spatial.gauss_mtf(T, sp)), Dn)
    ref51 = np.where(np.isfinite(Dn), -np.log10(gaussian_filter(T, (sp, sp, 0))), Dn)
    np.testing.assert_allclose(e56._scan_psf(Dn.copy()), ref56, rtol=1e-12, equal_nan=True)
    np.testing.assert_array_equal(e51._scan_psf(Dn.copy()), ref51)


def test_engine_uniform_grain_free_same_as_t51():
    """颗粒关的均匀场: 各项直流增益为 1, t56 与 t51 相同 (连续传递只改空间细节; 输出里约 1e-5 的缓变 → 差约 1e-6)。"""
    img = np.full((48, 64, 3), 0.18)
    a = _engine("t56_bw").render(img.copy())
    b = _engine("t51_bw").render(img.copy())
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-9)


def test_engine_pixel_grain_lower_than_t51():
    """颗粒开: 连续光学把像素级 (高频) 颗粒压低, 低频颗粒基本不变。"""
    img = np.full((96, 96, 3), 0.18)
    out = {g: _engine(g, grain=1.0).render(img.copy())[..., 1] for g in ("t51_bw", "t56_bw")}
    hp = {g: (o - gaussian_filter(o, 1.0, mode="reflect")).std() for g, o in out.items()}
    lo = {g: (gaussian_filter(o, 3.0, mode="reflect")).std() for g, o in out.items()}
    assert hp["t56_bw"] < 0.85 * hp["t51_bw"]
    assert 0.85 < lo["t56_bw"] / lo["t51_bw"] < 1.15
