"""T31: 有色成色剂遮罩在叠层中的放置 (film.MASK_PLACEMENT; 代际 t31_mask)。默认 (None) 逐位不变。"""
import numpy as np
import pytest

from filmlab import film, generation, stocks, variation


def _stack():
    st = stocks.load_stock("kodak_vision3_500t")
    v = variation.sample(None, 0, None)
    return st, v, film.emulsion_stack(st, v.layer_absorbance, v.yellow_filter)


def _halo_gr(st, v, light="BB3200"):
    from filmlab import spectral
    E = spectral.illuminant(light)
    T, d, b = film.emulsion_stack(st, v.layer_absorbance, v.yellow_filter)
    S = 10.0 ** st.log_sensitivity
    h = (E * T) @ b * (E @ S) / (E @ d)
    return h[1] / h[0]


def test_default_none_for_existing_generations():
    try:
        for g in generation.GENERATIONS:
            generation.use(g)
            assert film.MASK_PLACEMENT == ("layers" if g == "t31_mask" else None)
    finally:
        generation.use("current")
    assert film.MASK_PLACEMENT is None


def test_layers_only_changes_halo_terms_and_is_redder():
    st, v, (T0, d0, b0) = _stack()
    g0 = _halo_gr(st, v)
    film.MASK_PLACEMENT = "layers"
    try:
        T1, d1, b1 = film.emulsion_stack(st, v.layer_absorbance, v.yellow_filter)
        g1 = _halo_gr(st, v)
    finally:
        film.MASK_PLACEMENT = None
    np.testing.assert_array_equal(T0, T1)                 # 到达片基的透射率不变 (遮罩总量不变)
    np.testing.assert_allclose(d1[:, 2], d0[:, 2])        # 蓝层直接曝光不变
    assert abs(g0 - 0.105) < 0.005                         # T24: 3200K 光晕源 G/R ≈ 0.107
    assert g1 < g0 * 0.75                                  # 绿光多经过半次遮罩 (550 nm 0.44 D) → 更红
    # 默认路径逐位不变
    np.testing.assert_array_equal(film.emulsion_stack(st, v.layer_absorbance, v.yellow_filter)[2], b0)


def test_unknown_placement_rejected():
    st, v, _ = _stack()
    film.MASK_PLACEMENT = "bogus"
    try:
        with pytest.raises(ValueError):
            film.emulsion_stack(st, v.layer_absorbance, v.yellow_filter)
    finally:
        film.MASK_PLACEMENT = None
