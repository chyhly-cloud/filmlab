"""T11 next_lab_halo (halo_layered) synthetic checks; no scene data."""
import numpy as np
import pytest
from filmlab import film, generation, variation
from filmlab.engine import Engine, Params
from filmlab.recipes import recipe

TH = np.linspace(1e-4, np.radians(89.5), 4000)


def t03_uniform(theta, density):
    # Frozen T03 formula (uniform-depth average).
    x = density * (1.0 / np.cos(theta) - 1.0) * np.log(10.0)
    return np.divide(-np.expm1(-x), x, out=np.ones_like(x), where=x != 0)


@pytest.fixture
def restore():
    old = generation.active()
    yield
    generation.use(old)


@pytest.mark.parametrize('d', [0.0, 0.53, 1.1, 3.0])
def test_zero_weight_is_t03_bitwise(d):
    np.testing.assert_array_equal(film.oblique_attenuation(TH, d), t03_uniform(TH, d))
    np.testing.assert_array_equal(film.oblique_attenuation(TH, d, 0.0), t03_uniform(TH, d))


def test_layered_attenuates_more_than_uniform():
    for d in (0.53, 1.1):
        u, l = t03_uniform(TH, d), film.oblique_attenuation(TH, d, 0.765)
        assert np.all(l <= u + 1e-15) and l[-1] < u[-1]
        np.testing.assert_allclose(l, 0.765 * 10.0 ** (-d * (1 / np.cos(TH) - 1)) + 0.235 * u, rtol=0, atol=1e-15)


def test_scatter_weight_from_stack(restore):
    generation.use('next_lab_halo')
    e = Engine(Params(**{**recipe('cinestill_800t'), 'roll': None, 'grain': 0, 'timing': 'frame'}))
    w = e._halo_oblique_ws()
    assert w[0] == w[1] == w[2] and 0.5 < w[0] < 0.95
    assert w[0] == film.scatter_above_red(e.e.stock, e.v)


def test_tail_energy_reduced(restore):
    generation.use('next_lab_halo')
    e = Engine(Params(**{**recipe('cinestill_800t'), 'roll': None, 'grain': 0, 'timing': 'frame'}))
    d, w = e._halo_oblique_densities()[0], e._halo_oblique_ws()[0]
    k_u = film.layer_halation_kernel(25., e.v, 0.0, d)
    k_l = film.layer_halation_kernel(25., e.v, 0.0, d, w)
    yy, xx = np.indices(k_l.shape)
    r = np.hypot(yy - k_l.shape[0] // 2, xx - k_l.shape[1] // 2) * 25
    assert k_l[r > 700].sum() / k_l.sum() < k_u[r > 700].sum() / k_u.sum()
    assert k_l.sum() < k_u.sum()


def test_other_generations_have_no_layer_weight(restore):
    for g in ('next', 'next_lab', 'next_oblique'):
        generation.use(g)
        e = Engine(Params(**{**recipe('cinestill_800t'), 'roll': None, 'grain': 0, 'timing': 'frame'}))
        assert e._halo_oblique_ws() == (0.0, 0.0, 0.0)
