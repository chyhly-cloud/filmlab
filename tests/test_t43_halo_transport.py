"""T43: 输运一致的片基光晕 (halo_transport) 与 t43_* 代际。现有代际不变; 新项只在 t43_* 启用。"""
import numpy as np
import pytest

from filmlab import film, generation, rawio, spectral
from filmlab.engine import Engine, Params
from filmlab.recipes import recipe


@pytest.fixture(autouse=True)
def _restore_generation():
    prev = generation.active()
    yield
    generation.use(prev)


def _eng(g, **kw):
    generation.use(g)
    return Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "roll": None, "frame": 0, "grain": 0, **kw}))


def test_generations():
    base = generation.GENERATIONS["t31_base"][1]
    assert generation.GENERATIONS["t43_base"][1] == base + ("halo_transport",)
    assert generation.GENERATIONS["t43_tr"][1] == base + ("halo_transport",)
    for g, hr in (("t43_base", "measured_cond2"), ("t43_tr", "stops")):
        generation.use(g)
        assert (rawio.METERING, rawio.HIGHLIGHT_RECON, rawio.RECON_COLOUR, rawio.APERTURE_RANGE, film.MASK_PLACEMENT) == \
            ("session", hr, "ring", None, "layers")
    for g, corr in generation.GENERATIONS.items():
        if not g.startswith("t43_"):
            assert "halo_transport" not in corr[1]


def test_table_and_kernel():
    ratio, edges, prof = film._halo_transport_table()
    assert ratio.shape == (81, 3) and (ratio >= 0).all()
    for c in range(3):
        k = film.transport_kernel(22.5, c)
        assert k.shape[0] % 2 == 1 and np.isclose(k.sum(), film.transport_total(c))
        assert np.allclose(k, k[::-1, ::-1])                       # 径向对称
    assert 0.99 < film.transport_total(0) <= 1.0


def _weighted(g):
    """3200K 下 光晕 / 直接曝光 (红、绿层): 输运表 与 原链 (halation_response × 多次反射核总量)。"""
    e = _eng(g)
    st, v = e.e.stock, e.v
    E = spectral.illuminant(st.reference_illuminant)
    S = 10.0 ** st.log_sensitivity
    T_down, direct, back = film.emulsion_stack(st, v.layer_absorbance, v.yellow_filter)
    unit = (E @ S) / (E @ direct)
    old = (E[:, None] * T_down[:, None] * back * v.diffuse_fraction * unit).sum(0) * e._halo_total_multi_cache_free()
    ratio = film._halo_transport_table()[0]
    new = (E[:, None] * S * ratio).sum(0) / (E @ S) * np.array([film.transport_total(c) for c in range(3)])
    return old / (E @ S), new


def test_transport_vs_original_chain(monkeypatch):
    from filmlab import engine as eng
    monkeypatch.setattr(eng.Engine, "_halo_total_multi_cache_free",
                        lambda self: np.array([film.layer_halation_kernel(10.0, self.v, r, 0.0, 0.0, None, 0.0, False).sum()
                                               for r in film.bounce_retention(self.e.stock, self.v)]), raising=False)
    old, new = _weighted("t31_base")
    # 红层 0.093 → 0.10 (×1.0–1.25); 绿层 ×1.2–1.45; 都远小于实测亮度下补足光晕所需的 ×2.5–3 (T43 第 4 节)
    assert 0.085 < old[0] < 0.10 and 0.09 < new[0] < 0.115
    assert 1.0 < new[0] / old[0] < 1.25 and 1.2 < new[1] / old[1] < 1.45


def test_tail_longer_than_cos4_kernel():
    e = _eng("t31_base")
    k_old = film.layer_halation_kernel(22.5, e.v, film.bounce_retention(e.e.stock, e.v)[0])
    k_new = film.transport_kernel(22.5, 0)
    def frac(k, r0):
        n = k.shape[0] // 2; yy, xx = np.mgrid[-n:n + 1, -n:n + 1] * 22.5
        return k[np.hypot(yy, xx) > r0].sum() / k.sum()
    assert frac(k_new, 700) > 1.5 * frac(k_old, 700)


def _img():
    rng = np.random.default_rng(1)
    img = np.full((96, 128, 3), 0.01) + rng.uniform(0, 0.02, (96, 128, 3))
    img[46:51, 62:67] = 300.0 * np.array((1, .7, .4))
    return img


def test_render_only_halo_changes():
    img = _img()
    a = _eng("t31_base").render(img)
    b = _eng("t43_tr").render(img)
    assert not np.array_equal(a, b)
    # 带 remjet (无片基光晕): 修正不起作用, 逐位相同
    from filmlab import materials
    ra = _eng("t31_base", film=materials.emulsion("kodak_vision3_500t")).render(img)
    rb = _eng("t43_tr", film=materials.emulsion("kodak_vision3_500t")).render(img)
    assert np.array_equal(ra, rb)
    # 远离灯的暗部 (该小图 281 µm/像素, 灯的 6 mm 尾部仍覆盖全图) 变化很小
    assert np.abs(b[5:15, 5:15] / a[5:15, 5:15] - 1).max() < 0.05
