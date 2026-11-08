"""T53 颗粒粗细 k (docs/tasks/T53_GRAIN_COARSENESS.md): 守恒量、闭式解的空间放大、接口与默认值不变。"""
import numpy as np
import pytest

from filmlab import generation, grain, materials
from filmlab.engine import FILM_FORMATS, Engine, Params, crop_to_format
from filmlab.recipes import recipe, wratten


@pytest.fixture(autouse=True)
def _gen():
    prev = generation.active()
    generation.use("t51_bw")
    yield
    generation.use(prev)


def _trix():
    return materials.silver_spec(materials.bw_silver_film("kodak_trix_400"))


def test_range_and_identity():
    sp = _trix()
    assert grain.scale_silver(sp, 1.0) is sp
    for bad in (0.4, 3.5, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            grain.scale_silver(sp, bad)
    assert grain.GRAIN_SCALE_RANGE == (0.5, 3.0)


def test_3d_silver_is_exact_magnification():
    """三维晶体: 丝团是 k = 1 的整体放大 (A ∝ k²), 每颗晶体的光子数不变, 特性曲线不变, 涂布银 ∝ k。"""
    sp = _trix()
    b1, A1 = grain.silver_bins(sp), grain.silver_library(sp)["A_spec"]
    H = 0.8 / 400 * 10.0 ** np.linspace(-0.5, 2.3, 12)
    for k in (0.5, 2.0, 3.0):
        sk = grain.scale_silver(sp, k)
        bk = grain.silver_bins(sk)
        np.testing.assert_allclose(grain.silver_library(sk)["A_spec"], k * k * A1, rtol=1e-9)
        np.testing.assert_allclose(bk["n"], b1["n"] / k ** 2, rtol=1e-9)        # 每 µm² 晶体数 ∝ k⁻²
        np.testing.assert_allclose(grain.silver_develop(sk, H, bk["a"]), grain.silver_develop(sp, H, b1["a"]), atol=1e-12)
        np.testing.assert_allclose(grain.silver_density(sk, H), grain.silver_density(sp, H), atol=1e-12)
        np.testing.assert_allclose(np.sum(sk.coat_g_m2) / np.sum(sp.coat_g_m2), k, rtol=1e-9)
        assert abs(grain.silver_budget(sk)["covering_power"] * k / grain.silver_budget(sp)["covering_power"] - 1) < 1e-9


def test_selwyn_and_structure_scale_with_k():
    """闭式协方差是空间放大: σ_k(d) = σ_1(d/k); 大孔径 σ√A ∝ k。"""
    sp = _trix()
    H0 = 0.8 / 400 * 10.0 ** 1.2
    sk = grain.scale_silver(sp, 2.0)
    s1 = grain.silver_granularity(sp, H0, (12.0, 24.0, 48.0), adjacent=False)
    s2 = grain.silver_granularity(sk, H0, (24.0, 48.0, 96.0), adjacent=False)
    np.testing.assert_allclose(s2, s1, rtol=0.02)
    s96 = grain.silver_granularity(sp, H0, (96.0,), adjacent=False)
    assert abs(s2[2] / s96[0] / 2.0 - 1) < 0.05


def test_cluster_radius_scales():
    sp = materials.silver_spec(materials.bw_silver_film("kodak_trix_400@rodinal"))
    sk = grain.scale_silver(sp, 1.5)
    assert sp.cluster_mu > 0 and sk.cluster_mu == sp.cluster_mu
    assert abs(sk.cluster_um / sp.cluster_um - 1.5) < 1e-12
    H = 0.8 / 400 * 10.0 ** np.linspace(-0.3, 2.0, 8)
    np.testing.assert_allclose(grain.silver_density(sk, H), grain.silver_density(sp, H), atol=1e-9)


def test_tgrain_silver_mass_and_curve():
    """T 颗粒 (厚度不变): 丝径不变、丝团重新生成; 覆盖截面守恒 → 曲线差 < 0.01 D, 涂布银约不变。"""
    sp = materials.silver_spec(materials.bw_silver_film("kodak_tmax_400"))
    assert sp.thickness_um == materials.GRAIN_T_UM
    H = 0.8 / sp.ei * 10.0 ** np.linspace(-0.3, 2.3, 10)
    for k in (0.5, 2.0):
        sk = grain.scale_silver(sp, k)
        assert sk.thickness_um == sp.thickness_um and sk.filament_um == sp.filament_um
        assert np.abs(grain.silver_density(sk, H) - grain.silver_density(sp, H)).max() < 0.01
        assert abs(np.sum(sk.coat_g_m2) / np.sum(sp.coat_g_m2) - 1) < 0.1


def test_dye_path_scaling():
    """染料云 (T 颗粒): 每亚群密度不变 → 曲线不变; 48 µm σ ∝ k; 云 σ ∝ k; 染料量守恒 (n ∝ k⁻²)。"""
    e = materials.emulsion("kodak_portra_400")
    sp = materials.grain_spec_t46(e, "C-41")
    H = np.logspace(-5, 0, 60)
    for k in (0.5, 2.0, 3.0):
        sk = grain.scale_grain(sp, k)
        for c in range(3):
            D1, s1 = grain.layer_curves(sp, c, H)
            Dk, sk_ = grain.layer_curves(sk, c, H)
            np.testing.assert_allclose(Dk, D1, atol=1e-10)
            np.testing.assert_allclose(sk_, k * s1, rtol=1e-9)
            np.testing.assert_allclose(grain.cloud_sigma_um(sk, c), k * grain.cloud_sigma_um(sp, c), rtol=1e-12)
            n1 = [n for n, *_ in grain._nodes(sp, c)]; nk = [n for n, *_ in grain._nodes(sk, c)]
            np.testing.assert_allclose(nk, np.asarray(n1) / k ** 2, rtol=1e-9)


def test_3d_kappa_path():
    """三维晶体的 κ 模型 (T46 黑白): κ ×1/k (覆盖力 ∝ 1/√a), 每晶体积分密度 ∝ k², 曲线不变, σ ∝ k。"""
    sp = materials.grain_spec_t46(materials.bw_film(), "BW-D76")
    assert sp.thickness_um is None
    sk = grain.scale_grain(sp, 2.0)
    H = np.logspace(-4, 0, 30)
    D1, s1 = grain.layer_curves(sp, 1, H)
    Dk, s2 = grain.layer_curves(sk, 1, H)
    np.testing.assert_allclose(Dk, D1, atol=1e-10)
    np.testing.assert_allclose(s2, 2.0 * s1, rtol=1e-9)


def test_explicit_render_mean_matches_closed_form():
    sp = grain.scale_silver(_trix(), 2.0)
    H0 = 0.8 / 400 * 10.0 ** 1.2
    T = grain.render_silver(sp, np.full((160, 160), H0), 2.0, seed=5, workers=1)[16:-16, 16:-16]
    assert abs(T.mean() / 10.0 ** -float(grain.silver_density(sp, H0)) - 1) < 0.02


def _render(rec, **kw):
    img = np.full((160, 160, 3), 0.18)
    p = Params(**{**recipe(rec), "timing": "gray", "roll": None, "scene_lights": None, "frame_width_mm": 160 * 36 / 3032,
                  **kw})
    return Engine(p).render(img)


def test_engine_default_and_explicit_one_identical():
    a = _render("kodak_trix_400")
    b = _render("kodak_trix_400", grain_scale=1.0)
    c = _render("kodak_trix_400+g1")
    assert np.array_equal(a, b) and np.array_equal(a, c)


def test_engine_grain_scale_coarser_same_tone():
    from filmlab.scanning.spatial import encode
    y = {k: encode(_render("kodak_trix_400", grain_scale=k)).mean(-1)[8:-8, 8:-8] * 255 for k in (1.0, 2.0)}
    assert 1.6 < y[2.0].std() / y[1.0].std() < 2.4
    assert abs(y[2.0].mean() - y[1.0].mean()) < 3.0                  # 特性曲线不变 (平均码值只差逐像素非线性的几个码值)


def test_engine_rejects():
    with pytest.raises(ValueError):
        Engine(Params(film="kodak_portra_400", grain_scale=5.0))
    with pytest.raises(KeyError):
        Engine(Params(film="kodak_portra_400", film_format="8x10"))
    generation.use("current")                                       # 没有银盐颗粒路径的代际
    with pytest.raises(ValueError):
        Engine(Params(film="kodak_portra_400", grain_scale=2.0))


def test_film_format_param():
    img = np.full((60, 90, 3), 0.18)
    e = Engine(Params(**{**recipe("kodak_trix_400"), "grain": 0, "film_format": "half", "roll": None}))
    e.negative(img)
    assert abs(e.px_um - 24000.0 / 90) < 1e-9
    assert abs(e.scan_spatial_lab().pitch_um - 12.0 * 24 / 36) < 1e-12
    e35 = Engine(Params(**{**recipe("kodak_trix_400"), "film_format": "35mm"}))
    assert e35.frame_long_mm() == 36.0 and e35.scan_spatial_lab().pitch_um == 12.0
    assert Engine(Params(**recipe("kodak_trix_400"))).frame_long_mm() == 36.0
    assert recipe("kodak_trix_400+fhalf")["film_format"] == "half"
    c = crop_to_format(np.zeros((200, 300, 3)), "half")
    assert c.shape[:2] == (200, 267)
    assert crop_to_format(np.zeros((300, 200, 3)), "6x6").shape[:2] == (200, 200)
    assert set(FILM_FORMATS) >= {"35mm", "half", "110", "645", "6x6"}


def test_recipe_suffixes():
    r = recipe("kodak_trix_400+w25")
    assert np.array_equal(r["lens_filter"], wratten("25")) and "grain_scale" not in r
    r = recipe("kodak_trix_400@rodinal+w25+g2")
    assert r["grain_scale"] == 2.0 and np.array_equal(r["lens_filter"], wratten("25"))
    assert recipe("kodak_trix_400+g1.5")["grain_scale"] == 1.5
    for bad, err in (("kodak_trix_400+g4", ValueError), ("kodak_trix_400+g0.3", ValueError), ("kodak_trix_400+gx", ValueError),
                     ("kodak_trix_400+fbig", KeyError), ("kodak_trix_400+q1", KeyError)):
        with pytest.raises(err):
            recipe(bad)
