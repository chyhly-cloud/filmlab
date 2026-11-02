"""T41: 乳剂正面散射按 Kodak 5219 官方 MTF (emulsion_mtf) 与拍摄端镜头 PSF 近场 (camera_glare_t41)。
现有代际不变; 新项只在 t41_* 启用。"""
import numpy as np
import pytest

from filmlab import engine, film, generation, rawio
from filmlab.engine import Engine, Params
from filmlab.optics import glare_kernel, lens_glare
from filmlab.recipes import recipe


@pytest.fixture(autouse=True)
def _restore_generation():
    prev = generation.active()
    yield
    generation.use(prev)


def _eng(g):
    generation.use(g)
    return Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "roll": None, "frame": 0, "grain": 0}))


def test_generations():
    base = generation.GENERATIONS["t31_base"][1]
    assert generation.GENERATIONS["t41_emul"][1] == base + ("emulsion_mtf",)
    assert generation.GENERATIONS["t41_glare"][1] == base + ("camera_glare",)
    assert generation.GENERATIONS["t41_base"][1] == base + ("emulsion_mtf", "camera_glare", "camera_glare_t41")
    for g in ("t41_emul", "t41_glare", "t41_base"):
        generation.use(g)
        assert (rawio.METERING, rawio.HIGHLIGHT_RECON, rawio.RECON_COLOUR, rawio.APERTURE_RANGE, film.MASK_PLACEMENT,
                rawio.STOPS_PRIOR) == ("session", "stops", "ring", None, "layers", (6.0, 8.0))
    for g, corr in generation.GENERATIONS.items():
        if not g.startswith("t41_"):
            assert "emulsion_mtf" not in corr[1] and "camera_glare_t41" not in corr[1]


def _mtf_exp(a, f):
    """二维指数 PSF 的解析 MTF (f: c/mm)。"""
    return (1 + (2 * np.pi * a * f / 1000) ** 2) ** -1.5


def test_emulsion_psf_matches_kodak_mtf():
    # Kodak H-1-5219 第 3 页 MTF (除以低频平台 1.03), 50 c/mm: R 57%, G 50%, B 31%
    for a, want in zip(film.EMULSION_PSF_A_UM, (0.573, 0.505, 0.313)):
        assert abs(_mtf_exp(a, 50) - want) < 0.05
    # 数值核与解析 MTF 一致 (1 µm 像素, 中心行的一维傅里叶变换 = 线扩散函数的 MTF)
    H = np.zeros((257, 257, 3)); H[128, 128] = 1.0
    out = film.emulsion_scatter_kodak(H, 1.0)
    assert np.allclose(out.sum((0, 1)), 1.0)
    lsf = out.sum(0)
    F = np.abs(np.fft.rfft(np.fft.ifftshift(lsf, axes=0), axis=0))
    f = np.fft.rfftfreq(257, d=1.0) * 1000
    for c, a in enumerate(film.EMULSION_PSF_A_UM):
        i = np.argmin(np.abs(f - 50))
        assert abs(F[i, c] - _mtf_exp(a, f[i])) < 0.02


def test_emulsion_psf_has_no_far_field():
    # r > 25 µm 的能量 ≤ 0.6% (蓝层 0.51%, 红绿 < 0.03%): 乳剂正面散射填不了灯芯外 25–200 µm
    H = np.zeros((201, 201, 3)); H[100, 100] = 1.0
    out = film.emulsion_scatter_kodak(H, 1.0)
    yy, xx = np.mgrid[-100:101, -100:101]
    far = out[np.hypot(yy, xx) > 25].sum(0)
    assert np.all(far < 6e-3) and np.all(far[:2] < 3e-4)


def test_existing_scatter_unchanged():
    rng = np.random.default_rng(0)
    H = rng.uniform(0, 1, (40, 40, 3))
    assert np.array_equal(film.emulsion_scatter(H, 22.5), H)          # 原设定在 22.5 µm/像素时整项跳过
    assert "emulsion_mtf" not in _eng("t31_base").p.corrections


def test_glare_t41_parameters_separate():
    assert engine.CAMERA_GLARE["eta"] == 0.0                            # T20 / T22 研究代际的参数不变
    assert engine.CAMERA_GLARE_T41["eta"] > 0
    e = _eng("t41_base")
    assert "camera_glare" in e.p.corrections and "camera_glare_t41" in e.p.corrections


def test_glare_energy_conserving():
    rgb = np.full((121, 121, 3), 0.01)
    assert np.allclose(lens_glare(rgb, 100.0, **engine.CAMERA_GLARE_T41), rgb)     # 均匀画面不变 (灰卡定光不受影响)
    rgb[60, 60] = 100.0
    out = lens_glare(rgb, 100.0, **engine.CAMERA_GLARE_T41)
    assert np.allclose(out.sum((0, 1)), rgb.sum((0, 1)), rtol=2e-2)
    assert np.isclose(out[60, 60, 0], (1 - engine.CAMERA_GLARE_T41["eta"]) * 100.0, rtol=0.05)
    k = glare_kernel(22.5, engine.CAMERA_GLARE_T41["r0_um"], engine.CAMERA_GLARE_T41["beta"])
    assert np.isclose(k.sum(), 1.0)
