import numpy as np

from tools.eval.spectral_library_coverage import mixture_distance


def test_fixed_y_hull_is_not_free_amplitude_cone():
    basis = np.eye(2)
    target = np.array([1., 1.])
    assert not mixture_distance(basis, target, convex=True)['inside_at_declared_tolerance']
    free = mixture_distance(basis, target, convex=False)
    assert free['inside_at_declared_tolerance']
    assert abs(free['coefficient_sum'] - 2) < 1e-12


def test_full_linear_rank_does_not_cover_nonnegative_ratio_outside_endpoints():
    basis = np.array([[1., 1.], [2., 1.]])
    assert np.linalg.matrix_rank(basis) == 2
    result = mixture_distance(basis, np.array([3., 1.]), convex=False)
    assert not result['inside_at_declared_tolerance']
    assert result['minimum_scaled_Linf'] > .1
