"""T15: return-path oblique absorption and pressure-plate reflection in the halation kernel."""
import numpy as np
import pytest
from filmlab import film, generation
from filmlab.engine import Engine, Params
from filmlab.recipes import recipe


@pytest.fixture(autouse=True)
def _restore_generation():
    prev = generation.active()
    yield
    generation.use(prev)


def _v():
    generation.use("next_lab_halo")
    eng = Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "roll": None, "frame": 0, "grain": 0}))
    return eng.e.stock, eng.v


def test_defaults_bitwise():
    a = film.halation_kernel(20.0, oblique_density=0.5, oblique_w=0.7)
    b = film._halation_kernel_base(20.0, 130.0, 1.5, 1.5, 1500.0, 0.5, 0.7)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(film.halation_kernel(20.0, plate=0.0, return_gain=None), film.halation_kernel(20.0))


def test_return_gain_is_one_at_normal_and_red_grows():
    st, v = _v()
    g = film.return_gain(st, v)
    th = np.radians([0.0, 30.0, 45.0, 60.0, 75.0])
    for c in range(3):
        assert abs(g[c](np.array([1e-6]))[0] - 1.0) < 1e-6
    r = g[0](th)
    assert np.all(np.diff(r) > 0)                  # 红层: 斜光程越长吸收越多
    assert g[2](th)[-1] < 1.0                       # 蓝层在最上面: 先被红/绿层吃掉, 斜穿反而更少


def test_plate_only_adds_inside_critical_ring():
    k0 = film.halation_kernel(10.0, r_max_um=600.0)
    k1 = film.halation_kernel(10.0, r_max_um=600.0, plate=0.5)
    n = k0.shape[0] // 2
    yy, xx = np.mgrid[-n:n + 1, -n:n + 1] * 10.0
    rr = np.hypot(yy, xx)
    inner, outer = rr < 200, (rr > 260) & (rr < 600)
    assert k1[inner].sum() > 3 * k0[inner].sum()
    np.testing.assert_allclose(k1[outer].sum(), k0[outer].sum(), rtol=0.02)   # 栅格归一的微小差


def test_production_generations_unchanged():
    for g in ("current", "next", "next_lab_halo"):
        generation.use(g)
        eng = Engine(Params(**{**recipe("cinestill_800t"), "timing": "frame", "roll": None, "frame": 0, "grain": 0}))
        assert eng._halo_extra() == ((None, None, None), 0.0)
