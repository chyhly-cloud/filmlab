"""T34: 研究候选 t34_phys = t31_base + 光晕链三个无自由参数的物理项 (halo_oblique, halo_layered, halo_return); 不含 halo_plate。
其余设置 (定光、灯亮度先验、遮罩放置、蓝层映射) 与 t31_base 相同; t31_base 不变。"""
import numpy as np

from filmlab import engine, film, generation, materials, rawio, variation


def test_t34_is_t31_base_plus_three_terms():
    wb, corr, rev = generation.GENERATIONS["t34_phys"]
    wb0, corr0, rev0 = generation.GENERATIONS["t31_base"]
    assert wb == wb0 and rev == rev0 == {}
    assert corr == corr0 + ("halo_oblique", "halo_layered", "halo_return")
    assert "halo_plate" not in corr and "halo_forward" in corr
    assert "halo_oblique" not in corr0 and "halo_return" not in corr0
    try:
        for g in ("t31_base", "t34_phys"):
            generation.use(g)
            assert (rawio.METERING, rawio.HIGHLIGHT_RECON, rawio.RECON_COLOUR, rawio.APERTURE_RANGE, film.MASK_PLACEMENT) == \
                ("session", "stops", "ring", None, "layers")
    finally:
        generation.use("current")


def test_three_terms_lower_red_total_and_narrow_kernel():
    """m = 4: 下行斜程 + 按层 降低核总量; 返回斜光程提高红层; 三项合起来红层核总量 < 原核, 外圈占比更小。"""
    try:
        generation.use("t31_base")
        st = materials.emulsion("kodak_vision3_500t").stock
        v = variation.sample(None, 0)
        from dataclasses import replace
        v = replace(v, forward=4.0)
        rho = film.bounce_retention(st, v)[0]
        D, w, g = film.oblique_densities(st, v)[0], film.scatter_above_red(st, v), film.return_gain(st, v)[0]
        k0 = film.layer_halation_kernel(22.5, v, rho)
        k_ret = film.layer_halation_kernel(22.5, v, rho, return_gain=g)
        k_all = film.layer_halation_kernel(22.5, v, rho, D, w, g)
        assert k_ret.sum() > k0.sum() > k_all.sum()
        n = k0.shape[0] // 2
        r = np.hypot(*np.mgrid[-n:n + 1, -n:n + 1]) * 22.5
        outer = lambda k: k[r > 400].sum() / k.sum()
        assert outer(k_all) < outer(k0)
    finally:
        generation.use("current")
