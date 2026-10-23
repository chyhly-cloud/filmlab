"""Analytic and limiting cases for the independent transport verifier."""
import numpy as np
import pytest

from tools.eval.photon_transport import Layer, angle_return, fresnel, simulate, spin


def assert_closed(result):
    assert abs(result['ledger']['closure_error']) < 2e-12
    assert all(v >= 0 for k, v in result['ledger'].items() if k != 'closure_error')
    np.testing.assert_allclose(result['radial_absorbed'].sum(1)+result['radial_overflow'],
                               result['absorbed'], atol=2e-12)
    np.testing.assert_allclose(result['radial_returned'].sum(1)+result['returned_overflow'],
                               result['returned_absorbed'], atol=2e-12)
    assert np.all(result['returned_absorbed'] <= result['absorbed']+1e-12)


def test_fresnel_normal_tir_and_reciprocity():
    r, ct = fresnel(1., 1., 1.5)
    assert float(r) == pytest.approx(.04)
    assert float(ct) == 1
    assert float(fresnel(.5, 1.5, 1.)[0]) == 1
    ci = np.array([.4, .7, 1.])
    r, ct = fresnel(ci, 1., 1.5)
    back, _ = fresnel(ct, 1.5, 1.)
    np.testing.assert_allclose(r, back, atol=1e-14)


@pytest.mark.parametrize('g', [0., .8, -.5])
def test_hg_first_moment_and_unit_directions(g):
    directions = np.tile([0., 0., 1.], (60_000, 1))
    drawn = spin(directions, g, np.random.default_rng(4))
    np.testing.assert_allclose(np.linalg.norm(drawn, axis=1), 1, atol=1e-14)
    assert drawn[:, 2].mean() == pytest.approx(g, abs=.01)
    # Arbitrary incoming direction: relative scattering angle has the same mean.
    directions[:] = np.array([1., 2., 3.])/np.sqrt(14)
    drawn = spin(directions, g, np.random.default_rng(5))
    assert (drawn*directions).sum(1).mean() == pytest.approx(g, abs=.01)


def test_beer_lambert_multilayer_and_oblique_path():
    ci = .6
    result = simulate([Layer('a', 2, .1, 0, n=1), Layer('b', 3, .01, 0, n=1)],
                      photons=80_000, seed=2, incident_cosine=ci)
    ta, tb = np.exp(-.2/ci), np.exp(-.03/ci)
    expected = np.array([1-ta, ta*(1-tb), ta*tb])
    observed = np.r_[result['absorbed'], result['ledger']['bottom_escape']]
    assert np.all(np.abs(observed-expected) < 6*np.sqrt(expected*(1-expected)/80_000))
    assert_closed(result)


def test_transparent_slab_infinite_fresnel_series():
    result = simulate([Layer('glass', 130, 0, 0)], photons=60_000, seed=5)
    r = .04
    assert result['ledger']['specular_reflection'] == pytest.approx(r)
    total_r = result['ledger']['specular_reflection']+result['ledger']['top_escape']
    assert total_r == pytest.approx(2*r/(1+r), abs=.005)
    assert result['ledger']['bottom_escape'] == pytest.approx((1-r)/(1+r), abs=.005)
    assert_closed(result)


def test_conservative_scattering_and_absorbing_scattering_close():
    for mua in (0., .1):
        result = simulate([Layer('turbid', 10, mua, 1, g=.6)], photons=5000, seed=2)
        assert_closed(result)
        assert result['ledger']['event_residual'] < 1e-5
        if mua == 0:
            assert result['absorbed'].sum() == 0


def test_no_scattering_normal_beam_has_no_broad_return():
    result = simulate([Layer('emulsion', 5, .05, 0), Layer('base', 130, 0, 0)],
                      photons=20_000, seed=7)
    assert result['returned_absorbed'].sum() > 0
    assert result['radial_returned'][:, 1:].sum() == 0
    assert_closed(result)


def test_index_matched_rear_has_no_rear_reflected_absorption():
    result = simulate([Layer('emulsion', 5, .04, .2, g=.7), Layer('base', 130, 0, 0)],
                      photons=10_000, seed=3, n_bottom=1.5)
    assert result['returned_absorbed'].sum() == 0
    assert_closed(result)


def test_radial_and_event_truncations_are_reported_not_renormalized():
    result = simulate([Layer('emulsion', 20, .02, .2), Layer('base', 130, 0, 0)],
                      photons=10_000, seed=9, radial_edges=[0., .001], max_events=3)
    assert result['radial_overflow'].sum() > 0
    assert result['ledger']['event_residual'] > 0
    assert_closed(result)


def test_geometrical_unit_scaling_preserves_energy():
    a = simulate([Layer('a', 5, .02, .2, g=.6)], photons=3000, seed=1)
    b = simulate([Layer('a', 50, .002, .02, g=.6)], photons=3000, seed=1,
                 radial_edges=a['radial_edges']*10)
    np.testing.assert_allclose(a['absorbed'], b['absorbed'], atol=1e-13)
    np.testing.assert_allclose(a['radial_absorbed'], b['radial_absorbed'], atol=1e-13)
    assert_closed(a); assert_closed(b)


def test_angle_absorption_is_conservative_but_need_not_narrow_tail():
    def tail(a):
        return a['energy'][a['radius'] > 800].sum()/a['energy'].sum()
    for passive in (0., .4):
        for angled in (False, True):
            a = angle_return(passive, .1, angle_dependent=angled)
            assert abs(a['ledger']['closure_error']) < 1e-12
    # Longer active-layer paths alone can capture MORE of the far-angle light.
    assert tail(angle_return(0., .1)) > tail(angle_return(0., .1, angle_dependent=False))
    # A preceding passive absorber selectively removes grazing returned rays.
    assert tail(angle_return(.4, .1)) < tail(angle_return(.4, .1, angle_dependent=False))


def test_invalid_material_parameters_rejected():
    with pytest.raises(ValueError):
        Layer('bad', 5, -.1, 0)
    with pytest.raises(ValueError):
        Layer('bad', 5, 0, .1, g=1.)
    with pytest.raises(ValueError):
        simulate([Layer('a', 1, 0, 0)], radial_edges=[1, 2])
    with pytest.raises(ValueError):
        angle_return(float('nan'), .1)
    with pytest.raises(ValueError):
        fresnel(.5, 1., float('nan'))
