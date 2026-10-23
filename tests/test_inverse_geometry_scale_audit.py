import numpy as np

from tools.eval.inverse_geometry_scale_audit import project, homography_jacobian, metric_summary


def test_affine_translation_does_not_change_local_metric_and_rotation_preserves_circle():
    theta = .31
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    matrix = np.eye(3)
    matrix[:2, :2] = 2*rotation
    matrix[:2, 2] = [300., -20.]
    summary = metric_summary(matrix, [17., 9.], 10., 5.)
    np.testing.assert_allclose(summary['nominal_physical_singular_values'], [1., 1.], atol=1e-15)
    np.testing.assert_allclose(summary['mapped_nominal_circle_radii_um']['235'], [235., 235.], atol=1e-12)


def test_projective_jacobian_matches_independent_finite_differences():
    matrix = np.array([[2., .01, 10.], [-.02, 2.1, -5.], [1e-4, -2e-4, 1.]])
    point = np.array([43., 69.])
    h = 1e-3
    finite = np.stack([(project(matrix, point+h*unit)-project(matrix, point-h*unit))/(2*h)
                       for unit in np.eye(2)], -1)
    np.testing.assert_allclose(homography_jacobian(matrix, point), finite, atol=2e-11)


def test_nonisometric_warp_turns_isotropic_covariance_into_known_ellipse():
    # This is the linear convolution-coordinate identity, not a fitted kernel.
    matrix = np.diag([3., 2., 1.])
    summary = metric_summary(matrix, [0., 0.], 2., 1.)
    np.testing.assert_allclose(summary['nominal_physical_singular_values'], [1.5, 1.])
    physical = np.asarray(summary['nominal_physical_Jacobian'])
    covariance = physical@np.eye(2)@physical.T
    np.testing.assert_allclose(covariance, [[2.25, 0.], [0., 1.]])
