"""T50: 显影液 → 显影晶体成团 (Thomas 过程) 的闭式颗粒解, 与显式渲染互验; 显影液档案与新代际。"""
import numpy as np
import pytest

from filmlab import generation, grain, materials


def _trix():
    generation.use("t49_bw")
    return materials.silver_spec(materials.trix_film())


def test_cluster_limit_is_poisson():
    sp = _trix()
    H = 10 ** -1.2
    tiny = sp.with_(cluster_mu=1e-7, cluster_um=2.0, cluster_mode="developed")
    assert float(grain.silver_density(tiny, H)) == pytest.approx(float(grain.silver_density(sp, H)), rel=1e-6)
    a = grain.silver_granularity(sp, H, [10.0, 40.0], 0.0, adjacent=False)
    b = grain.silver_granularity(tiny, H, [10.0, 40.0], 0.0, adjacent=False)
    assert np.allclose(a, b, rtol=2e-3)


def test_crystal_mode_default_unchanged():
    sp = _trix()
    assert sp.cluster_mode == "crystal" and sp.cluster_mu == 0.0


@pytest.mark.parametrize("mu,sig,logH", [(3.0, 2.0, -0.8), (4.0, 1.0, -1.6)])
def test_cluster_closed_form_matches_explicit_render(mu, sig, logH):
    """显式渲染 (显影晶体成团) 的平均透射率与孔径方差 = 闭式解 (去掉 4σ 以上的边框: 视场外的团中心不生成)。"""
    sp = _trix().with_(cluster_mu=mu, cluster_um=sig, cluster_mode="developed")
    H = 10 ** logH
    ET = 10 ** -float(grain.silver_density(sp, H))
    _, C, h = grain.silver_covariance(sp, H)
    vcf = grain.window_variance(C, h, [8.0], 0.0, adjacent=False)[0]
    T = np.stack([grain.render_silver(sp, np.full((224, 224), H), 2.0, s, sub=16, count_sub=4) for s in range(3)])[:, 16:-16, 16:-16]
    m = T.shape[1] // 4
    v = float(T[:, :m * 4, :m * 4].reshape(3, m, 4, m, 4).mean((2, 4)).var())
    assert T.mean() / ET == pytest.approx(1.0, abs=0.03)
    assert v / vcf == pytest.approx(1.0, abs=0.12)


def test_cluster_raises_granularity_and_keeps_curve_fit():
    import json
    from pathlib import Path
    sp = _trix()
    x = np.linspace(-2.5, 0.5, 13)
    net = grain.silver_curve(sp, x) - grain.silver_curve(sp, [-14.0])[0]
    H = 10 ** x
    m = dict(materials.GRAIN_T47_MORPH, cluster_mu=3.0, cluster_um=2.0, cluster_mode="developed")
    U = dict(materials.grain_universal())
    s2 = grain.silver_derive(sp.ei, U, materials.GRAIN_T47_ETA3D_SCALE, H, net, sp.base_density + float(grain.silver_density(sp, 0.0)), m)
    fit = grain.silver_curve(s2, x) - grain.silver_curve(s2, [-14.0])[0]
    assert np.max(np.abs(fit - net)) < 0.1
    Hm = 10 ** -1.0
    assert grain.silver_granularity(s2, Hm, [200.0], 0.0, adjacent=False)[0] > 1.2 * grain.silver_granularity(sp, Hm, [200.0], 0.0, adjacent=False)[0]


def test_t50_generation_and_plain_recipes_unchanged():
    from filmlab.recipes import recipe
    assert generation.GENERATIONS["t50_bw"] == generation.GENERATIONS["t49_bw"]
    assert generation.BW_PROFILES["t50_bw"] == "t50"
    out = {}
    for g in ("t49_bw", "t50_bw"):
        generation.use(g)
        out[g] = {k: recipe(k)["film"] for k in ("kodak_trix_400", "kodak_tmax_100", "ilford_hp5_plus")}
    for k in out["t49_bw"]:
        a, b = out["t49_bw"][k].stock, out["t50_bw"][k].stock
        assert np.array_equal(a.log_sensitivity, b.log_sensitivity) and np.array_equal(a.density_curves, b.density_curves)


def test_developer_variants():
    from filmlab.recipes import recipe
    d = materials.bw_t50_developers()
    assert d["developers"]["d76"]["cluster_mu"] == 0 and d["developers"]["xtol"]["cluster_mu"] == 0
    assert set(d["developers"]) == {"d76", "xtol", "rodinal", "fx39"}          # SD2525 / Wehner-Alpha / Shadowmax 只报告
    generation.use("t50_bw")
    base = materials.silver_spec(materials.trix_film())
    sp = {k: materials.silver_spec(recipe(f"kodak_trix_400@{k}")["film"]) for k in ("d76", "rodinal", "xtol", "fx39", "d76_ei1600")}
    for k, s in sp.items():
        assert s.coat_g_m2 == base.coat_g_m2 and s.ecd_um == base.ecd_um         # 片不变, 只改显影
    assert sp["d76"] == base
    assert sp["rodinal"].cluster_mode == "developed" and sp["rodinal"].cluster_mu > 0
    g = lambda s: grain.silver_granularity(s, 10 ** -1.0, [200.0], 0.0, adjacent=False)[0]
    assert g(sp["rodinal"]) > 1.2 * g(sp["d76"])
    assert sp["d76_ei1600"].photons_r < 4.0 and recipe("kodak_trix_400@d76_ei1600")["exposure_ev"] == -2.0
    assert d["push"]["kodak_trix_400"]["d76_ei1600"]["verified"] is False
    generation.use("t49_bw")
    with pytest.raises(KeyError):
        recipe("kodak_trix_400@rodinal")
