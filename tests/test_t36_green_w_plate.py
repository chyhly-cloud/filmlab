"""T36: 按层深度的 w 逐层计算 (halo_layered_perlayer) 与哑光压片板的朗伯漫反射 (halo_plate_lambert)。
现有代际不变; 新项只在 t36_phys / t36_base 启用。"""
import numpy as np
import pytest

from filmlab import engine, film, generation, materials, rawio, variation
from filmlab.engine import Engine, Params
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
    c34, c36, c36b = (generation.GENERATIONS[g][1] for g in ("t34_phys", "t36_phys", "t36_base"))
    assert c36 == c34 + ("halo_layered_perlayer",)
    assert c36b == c36 + ("halo_plate", "halo_plate_lambert")
    for g in ("t36_phys", "t36_base"):
        assert generation.GENERATIONS[g][0] == "D65" and generation.GENERATIONS[g][2] == {}
        generation.use(g)
        assert (rawio.METERING, rawio.HIGHLIGHT_RECON, rawio.RECON_COLOUR, rawio.APERTURE_RANGE, film.MASK_PLACEMENT) == \
            ("session", "stops", "ring", None, "layers")
    for g, corr in generation.GENERATIONS.items():
        if g not in ("t36_phys", "t36_base"):
            assert "halo_layered_perlayer" not in corr[1] and "halo_plate_lambert" not in corr[1]


def test_per_layer_weights():
    generation.use("t31_base")
    st, v = materials.emulsion("kodak_vision3_500t").stock, variation.sample(None, 0)
    wR, wG, wB = film.scatter_above_layers(st, v)
    assert wR == film.scatter_above_red(st, v)
    assert 0.5 < wG < wR and wB == 0.0
    e34, e36 = _eng("t34_phys"), _eng("t36_phys")
    w34 = e34._halo_oblique_ws()
    assert w34[0] == w34[1] == w34[2] == wR                     # t34_phys 不变
    assert e36._halo_oblique_ws() == (wR, wG, 0.0)
    t34, t36 = e34._halo_total_multi(), e36._halo_total_multi()
    assert t36[0] == t34[0] and t36[1] > t34[1] and t36[2] > t34[2]


def test_existing_generations_have_no_diffuse_plate():
    for g in ("current", "next", "next_lab_halo", "next_lab_halo_plate", "t31_base", "t34_phys", "t36_phys"):
        assert not _eng(g)._plate_diffuse()
    e = _eng("next_lab_halo_plate")
    assert e._halo_extra()[1] == engine.PLATE_REFLECTANCE           # T15 镜面研究项不变
    e = _eng("t36_base")
    assert e._plate_diffuse() and e._halo_extra()[1] == engine.PLATE_REFLECTANCE_LAMBERT


def test_lambert_plate_zero_is_bitwise_and_specular_unchanged():
    a = film.halation_kernel(20.0, forward=4.0, oblique_density=0.5, oblique_w=0.7)
    np.testing.assert_array_equal(film.halation_kernel(20.0, forward=4.0, oblique_density=0.5, oblique_w=0.7,
                                                       plate=0.0, plate_diffuse=True), a)
    np.testing.assert_array_equal(film.halation_kernel(20.0, plate=0.2, plate_diffuse=False), film.halation_kernel(20.0, plate=0.2))


def test_lambert_plate_energy_and_support():
    n, t, m, rho = 1.5, 130.0, 4.0, 0.05
    th = np.linspace(1e-4, np.radians(89.5), 4000); dth = np.gradient(th)
    ang = np.sin(th) * np.cos(th) ** m; ang /= (ang * dth).sum()
    R = film._fresnel_unpolarized(th, n); tc = np.arcsin(1 / n)
    ta = np.linspace(1e-6, np.pi / 2 - 1e-6, 20000)        # 空气侧朗伯入射的半球反射率 (独立积分)
    ci = np.cos(ta); ct = np.sqrt(1 - (np.sin(ta) / n) ** 2)
    Rd = (0.5 * (((ci - n * ct) / (ci + n * ct)) ** 2 + ((ct - n * ci) / (ct + n * ci)) ** 2) * 2 * np.sin(ta) * ci * np.gradient(ta)).sum()
    expect = (ang * dth * (1 - R))[th < tc].sum() * rho * (1 - Rd) / (1 - rho * Rd)
    k0 = film.halation_kernel(5.0, base_um=t, forward=m, r_max_um=600.0)
    kd = film.halation_kernel(5.0, base_um=t, forward=m, r_max_um=600.0, plate=rho, plate_diffuse=True) - k0
    assert abs(kd.sum() - expect) < 1e-3 * expect
    h = kd.shape[0] // 2
    rr = np.hypot(*np.mgrid[-h:h + 1, -h:h + 1]) * 5.0
    assert kd[rr > 2 * t * np.tan(tc) + 5.0].sum() < 1e-6 * kd.sum()     # 只落在 2·t·tanθc 以内
    # 漫反射: 落点 |a + b| 方位随机, 比镜面 (2·t·tanθ) 更靠内
    ks = film.halation_kernel(5.0, base_um=t, forward=m, r_max_um=600.0, plate=rho) - k0
    assert (kd * rr).sum() / kd.sum() < (ks * rr).sum() / ks.sum()
