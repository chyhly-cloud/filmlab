"""T03 synthetic checks; threshold failures deliberately remain visible."""
import numpy as np
import pytest
from filmlab import film, generation
from filmlab.engine import Engine, Params
from filmlab.recipes import recipe


@pytest.fixture
def engine():
    old = generation.active()
    generation.use('next_oblique')
    yield Engine(Params(**{**recipe('cinestill_800t'), 'roll': None, 'grain': 0}))
    generation.use(old)


def original_kernel(px_um, base_um=130., forward=1.5, r_max_um=1500.):
    # Frozen pre-T03 algorithm, including its raster integration correction.
    th = np.linspace(1e-4, np.radians(89.5), 4000)
    r = 2.0*base_um*np.tan(th)
    ang = np.sin(th)*np.cos(th)**forward
    ang /= (ang*np.gradient(th)).sum()
    w = film._fresnel_unpolarized(th, 1.5)*ang*np.gradient(th)
    n = int(np.ceil(r_max_um/px_um))
    yy, xx = np.mgrid[-n:n+1, -n:n+1]*px_um
    rr = np.hypot(yy, xx)
    edges = np.append(r, r[-1]*1.01)
    area = np.pi*(edges[1:]**2-edges[:-1]**2)
    k = np.interp(rr, r, w/area, right=0.)*px_um**2
    return k*(w[r<=r_max_um].sum()/max(k.sum(), 1e-12))


@pytest.mark.parametrize('px', [10., 100., 562.5])
def test_zero_density_bitwise_original(px):
    np.testing.assert_array_equal(film.halation_kernel(px, oblique_density=0), original_kernel(px))
    rho = .08
    expected = original_kernel(px, r_max_um=6000.)
    for n in range(2, 5):
        expected += rho**(n-1)*original_kernel(px, base_um=130*n, r_max_um=6000.)
    np.testing.assert_array_equal(film.multi_bounce_kernel(px,130,1.5,rho,oblique_density=0), expected)


def test_formula_and_limit():
    theta = np.array([0., .1, .8, 1.4])
    np.testing.assert_array_equal(film.oblique_attenuation(theta,0), np.ones(4))
    u = 1/np.cos(theta[1:])-1
    np.testing.assert_allclose(film.oblique_attenuation(theta,.53)[1:],
        (1-10**(-.53*u))/(.53*u*np.log(10)), rtol=1e-12)
    assert film.oblique_attenuation(theta,.53)[0] == 1


def test_density_definition(engine):
    down,_,_ = film.emulsion_stack(engine.e.stock,engine.v.layer_absorbance,engine.v.yellow_filter)
    s = 10**engine.e.stock.log_sensitivity
    np.testing.assert_array_equal(film.oblique_densities(engine.e.stock,engine.v),
        -np.log10((down[:,None]*s).sum(0)/s.sum(0)))


def test_energy_monotone_no_renormalization(engine):
    for rho in film.bounce_retention(engine.e.stock,engine.v):
        kernels = [film.layer_halation_kernel(40,engine.v,rho,d) for d in (0,.2,.53,1.,3.)]
        for a,b in zip(kernels,kernels[1:]):
            assert np.all(b <= a)
            assert b.sum() < a.sum()


def test_gray_total_uses_same_layer_builder(engine):
    rho = film.bounce_retention(engine.e.stock,engine.v)
    ds = engine._halo_oblique_densities()
    np.testing.assert_array_equal(engine._halo_total_multi(),
        [film.layer_halation_kernel(10,engine.v,rho[c],ds[c]).sum() for c in range(3)])
    np.testing.assert_array_equal(engine._halo_total(),
        [film.layer_halation_kernel(10,engine.v,oblique_density=d).sum() for d in ds])


def test_red_energy_over_700_requirement(engine):
    k = film.layer_halation_kernel(10,engine.v,film.bounce_retention(engine.e.stock,engine.v)[0],engine._halo_oblique_densities()[0])
    yy,xx = np.indices(k.shape)
    r = np.hypot(yy-k.shape[0]//2,xx-k.shape[1]//2)*10
    assert k[r>700].sum()/k.sum() < .05


def test_3200k_gray_requirement(engine):
    from tools.eval.t03_kernel_diagnostic import gray_chroma
    assert gray_chroma() < .5
