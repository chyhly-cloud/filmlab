import numpy as np

from tools.eval.inverse_registration_audit import (
    average_with_aperture_support, common_samples, predeclared_shifts,
)


def test_integer_target_shift_of_affine_linear_rgb_has_exact_known_change():
    yy, xx = np.indices((90, 90))
    real = np.stack([.001*yy+.002*xx, .003*yy, .001*xx], -1)
    points = np.array([[40, 40], [50, 30]])
    keep, targets, predictions = common_samples(real, {'same': real}, points, 2., [(0, 0), (4, -8)])
    assert keep.all()
    np.testing.assert_allclose(targets[1]-targets[0], [[-.012, .012, -.008]]*2, atol=1e-15)
    np.testing.assert_allclose(predictions['same'], targets[0], atol=1e-15)


def test_all_paths_and_shifts_share_strict_finite_aperture_support():
    real = np.ones((90, 90, 3))*.2
    candidate = real.copy()
    candidate[42, 40] = np.nan
    points = np.array([[40, 40], [20, 20], [10, 10]])
    keep, _, _ = common_samples(real, {'baseline': real, 'candidate': candidate},
                                points, 1., [(0, 0), (-8, -8)])
    np.testing.assert_array_equal(keep, [False, True, False])


def test_constant_field_averaging_does_not_renormalize_missing_apertures():
    field = np.ones((50, 50, 3))*.3
    field[25, 25] = np.nan
    mean, support = average_with_aperture_support(field, 1.)
    assert not support[25, 25] and not support[25, 21]
    assert support[15, 15]
    np.testing.assert_allclose(mean[15, 15], .3, atol=1e-15)


def test_predeclared_grid_keeps_all_17_controls_including_opposites():
    shifts = predeclared_shifts()
    assert len(shifts) == len(set(shifts)) == 17
    assert shifts[0] == (0, 0)
    assert all((-dy, -dx) in shifts for dy, dx in shifts)
