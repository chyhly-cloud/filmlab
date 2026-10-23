import numpy as np
import pytest
from scipy.signal import fftconvolve

from tools.eval.source_geometry_bounds import adjoint_response, ratio_bounds


def test_open_frame_adjoint_matches_two_distinct_source_placements():
    weights = np.zeros((9, 9)); weights[1, 2] = .3; weights[5, 7] = .7
    kernel = np.array([[.1, .2, .0], [.05, .4, .1], [.02, .01, .12]])
    source = np.zeros_like(weights); source[0, 2] = 3.; source[6, 6] = 5.
    response = adjoint_response(weights, kernel)
    np.testing.assert_allclose(np.sum(weights * fftconvolve(source, kernel, mode='same')),
                               np.sum(source * response), atol=1e-15)


def test_fractional_bounds_cover_arbitrary_mixtures_and_vertices_attain_them():
    near = np.array([.2, .5, .4]); far = np.array([.3, .2, .6])
    floor = np.array([1., 2., .5]); total = 10.
    bound = ratio_bounds(near, far, floor=floor, total=total)
    remainder = total-floor.sum()
    for mixture in np.random.default_rng(4).dirichlet(np.ones(3), 200):
        source = floor + remainder*mixture
        value = source @ far / (source @ near)
        assert bound['minimum']-1e-14 <= value <= bound['maximum']+1e-14
    for label in ['minimum', 'maximum']:
        source = floor.copy(); source[bound[label+'_source_index']] += remainder
        assert source @ far / (source @ near) == pytest.approx(bound[label])
    unchanged = ratio_bounds(near, far, floor=floor, total=floor.sum())
    assert unchanged['minimum'] == pytest.approx(floor @ far / (floor @ near))
    assert unchanged['minimum'] == unchanged['maximum']


def test_bounds_preserve_source_scale_and_reject_infeasible_total():
    a = ratio_bounds([1., 2.], [.4, 1.2], floor=[2., 1.], total=8.)
    b = ratio_bounds([1., 2.], [.4, 1.2], floor=[20., 10.], total=80.)
    assert a['minimum'] == pytest.approx(b['minimum'])
    assert a['maximum'] == pytest.approx(b['maximum'])
    with pytest.raises(ValueError, match='total'):
        ratio_bounds([1., 2.], [.4, 1.2], floor=[2., 1.], total=2.)
