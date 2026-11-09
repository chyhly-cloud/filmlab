"""物理核心: 显式渲染的平均 = 闭式解; 颗粒度接口; 颗粒粗细的标度; 快速路径的二阶统计。"""
import numpy as np
import pytest

import silvergrain as sg
from silvergrain import dye, silver
from silvergrain.render import render


@pytest.fixture(scope="module")
def trix():
    return sg.derive("kodak_trix_400")


def test_explicit_mean_equals_closed_form(trix):
    H = np.full((96, 96), 10 / 400)
    r = render(trix, exposure=H, px_um=12.0, seed=2, method="explicit")
    T = 10.0 ** -r.density
    assert -np.log10(T.mean()) == pytest.approx(float(r.mean[0, 0]), abs=0.004)
    assert r.density.mean() > r.mean.mean()                      # −log⟨T⟩: 像素对数的平均更高 (T53 2.6)


def test_render_from_density_and_format(trix):
    d = np.full((64, 64), 0.8)
    r = render(trix, density=d, format=64 * 12 / 1000, seed=1)
    assert r.px_um == pytest.approx(12.0) and r.method == "explicit"
    assert np.allclose(trix.mean_density(r.exposure), 0.8, atol=1e-3)


def test_granularity_api(trix):
    s = sg.granularity(trix, 1.0, 48)
    assert 0.010 < s < 0.020                                     # T47: Kodak 48 µm 预测 14.6 (×1000)
    c = sg.granularity(sg.derive("kodak_portra_400"), 0.5)
    assert c.shape == (3,) and np.all(c > 0)
    # Selwyn 面积律 (染料: 白噪声换算)
    c2 = sg.granularity(sg.derive("kodak_portra_400"), 0.5, aperture_um=96)
    assert np.allclose(c2, c / 2)


def test_grain_scale_selwyn(trix):
    H = 10 / 400
    s1 = silver.silver_granularity(trix.spec, H, [96.0], adjacent=False)[0]
    s2 = silver.silver_granularity(sg.derive("kodak_trix_400", grain_scale=2.0).spec, H, [96.0], adjacent=False)[0]
    assert s2 / s1 / 2 == pytest.approx(1.0, abs=0.01)
    lh = np.linspace(-3, 0, 31)
    assert np.max(np.abs(silver.silver_curve(trix.spec, lh) - silver.silver_curve(sg.derive("kodak_trix_400", grain_scale=2.0).spec, lh))) < 0.01


def test_dye_render_mean_and_sigma():
    em = sg.derive("kodak_portra_400")
    H = np.full((128, 128, 3), 10 / 400)
    r = render(em, exposure=H, px_um=12.0, seed=1)
    assert np.allclose(r.density.mean((0, 1)), r.mean[0, 0], atol=0.003)
    sd = r.density.std((0, 1))
    assert np.all(sd > 0) and np.all(sd < 0.2)


def test_closed_form_matches_explicit(trix):
    H = np.full((128, 128), 10 / 400)
    a = render(trix, exposure=H, px_um=12.0, seed=3, method="explicit").density[8:-8, 8:-8]
    b = render(trix, exposure=H, px_um=12.0, seed=3, method="closed_form").density[8:-8, 8:-8]
    assert b.std() / a.std() == pytest.approx(1.0, abs=0.08)
    assert b.mean() == pytest.approx(a.mean(), abs=0.003)


def test_closed_form_varying_exposure_continuity(trix):
    H = 10 / 400 * 2.0 ** np.linspace(-3, 3, 256)[None, :].repeat(64, 0)
    r = render(trix, exposure=H, px_um=8.0, seed=4, method="closed_form")
    f = r.density - r.mean
    assert np.all(np.isfinite(r.density))
    sd = f[:, 16:240].reshape(64, 7, 32).std((0, 2))
    assert np.all(np.diff(sd) > -0.03)                            # 颗粒随曝光平滑变化 (无档间跳变)


def test_dye_scale_and_reversal_paths():
    p = sg.derive("fujifilm_provia_100f")
    assert p.curve.design == "reversal" and p.developer.key == "E-6"
    H = np.full((32, 32, 3), 10 / 100)
    r = render(p, exposure=H, px_um=10.0, seed=1)
    assert np.all(np.isfinite(r.density))
