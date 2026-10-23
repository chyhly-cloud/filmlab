"""Independent standards for the conditional-kernel numerical audit only."""
import numpy as np

from tools.eval.transport_contract_review import (
    angular_disk_mass, convolution_ledger, pixel_integral,
    radial_disk_mass, reflected_fraction,
)


def test_polar_jacobian_preserves_angular_probability_at_three_apertures():
    # Separate theta- and radius-space quadratures; no kernel normalization.
    for radius in (100., 300., 1500.):
        np.testing.assert_allclose(radial_disk_mass(radius), angular_disk_mass(radius),
                                   atol=1e-10, rtol=0)


def test_index_matched_back_boundary_has_no_reflected_energy():
    np.testing.assert_allclose(reflected_fraction(np.array([1., .8, .3]), 1.),
                               0., atol=1e-28)


def test_unnormalized_pixel_area_quadrature_matches_angular_disk_energy():
    kernel, _ = pixel_integral(62.5, order=16)
    # Deliberately no post-normalization to force the spatial/angle agreement.
    assert abs(kernel.sum() - angular_disk_mass(1500.)) < 1e-4
    assert np.all(kernel >= 0)
    n = kernel.shape[0] // 2
    yy, xx = np.mgrid[-n:n + 1, -n:n + 1] * 62.5
    wholly_outside = np.hypot(xx, yy) > 1500 + np.sqrt(2) * 62.5 / 2
    assert np.all(kernel[wholly_outside] == 0)


def test_open_frame_ledger_retains_escaped_energy_instead_of_renormalizing():
    kernel = np.zeros((45, 45))
    kernel[22, 22] = .3
    kernel[0, 0] = .2
    # The 41x41 frame keeps the centred deposit and excludes the far one.
    row = convolution_ledger(kernel)
    np.testing.assert_allclose([row['retained_frame_sum'], row['outside_frame_sum']],
                               [.3, .2], atol=1e-15)
    assert abs(row['closure_error']) < 1e-15
