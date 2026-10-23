import numpy as np
import pytest

from tools.eval.catalogue_source_inverse import cone_halfspaces, spectral_witness


def test_projective_cone_contains_all_mixtures_and_excludes_known_ratio_violation():
    # Six explicit points span the 5D section; pivot is uniformly one.
    rays = np.ones((6, 6))
    other = [0, 2, 3, 4, 5]
    for row, column in enumerate(other, 1):
        rays[row, column] = 2.
    A = cone_halfspaces(rays)
    random = np.random.default_rng(31).uniform(size=(100, len(rays))) @ rays
    assert (A @ random.T).max() < 1e-12
    # Every generator has R/direct-G >= 1; this point violates that ratio.
    impossible = np.ones(6); impossible[0] = .5
    assert (A @ impossible).max() > .1
    assert np.max(A @ rays.T) < 1e-12


def test_one_spectrum_witness_retains_response_equalities_and_camera_bounds():
    responses = np.array([[1., 2., 3., 4., 5., 6.],
                          [2., 1., 4., 3., 6., 5.]])
    camera = np.array([[.5, 1., 2.], [2., .5, 1.]])
    moment = np.array([2., 3.]) @ responses
    result = spectral_witness(responses, moment, camera, [6., 3., 6.])
    assert result.success
    np.testing.assert_allclose(result.x, [2., 3.], atol=1e-10)
    np.testing.assert_allclose(result.x @ responses, moment)
    assert not spectral_witness(responses, moment, camera, [100., 100., 100.]).success


def test_cone_rejects_a_pivot_that_cannot_define_a_finite_section():
    with pytest.raises(ValueError):
        cone_halfspaces(np.eye(6), pivot=1)


def test_same_response_can_have_distinct_input_y_extrema():
    # Two Y-normalized spectra can have different material response per unit Y.
    rays = np.array([np.ones(6), 2*np.ones(6)])
    moments = 4*np.ones(6)
    minimum = spectral_witness(rays, moments)
    maximum = spectral_witness(rays, moments, maximize_input_Y=True)
    assert minimum.success and maximum.success
    assert minimum.x.sum() == pytest.approx(2.)
    assert maximum.x.sum() == pytest.approx(4.)
