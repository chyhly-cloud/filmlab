"""Analytic small spectral systems validate the diagnostic LP, not 800T."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/eval'))
from spectral_exposure_bounds import exposure_bounds


def test_unique_spectrum_recovers_exact_exposure():
    rgb = np.array([.2, .4, .8])
    sensitivity = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    r = exposure_bounds(rgb, np.eye(3), sensitivity)
    assert r['success']
    np.testing.assert_allclose(r['H_layer_min_max'], np.repeat((rgb @ sensitivity)[:, None], 2, axis=1))


def test_metameric_extrema_known_cap_and_radiometric_scaling():
    camera = np.array([[1., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    sensitivity = np.array([[1., 1., 1.], [3., 2., 4.], [0., 1., 0.], [0., 0., 1.]])
    rgb = np.array([1., .5, .25])
    loose = exposure_bounds(rgb, camera, sensitivity)
    upper = np.array([1., .2, 1., 1.])
    tight = exposure_bounds(rgb, camera, sensitivity, upper)
    np.testing.assert_allclose(loose['H_layer_min_max'], [[1, 3], [1.5, 2.5], [1.25, 4.25]])
    np.testing.assert_allclose(tight['H_layer_min_max'], [[1, 1.4], [1.5, 1.7], [1.25, 1.85]])
    doubled = exposure_bounds(2 * rgb, camera, sensitivity, 2 * upper)
    np.testing.assert_allclose(doubled['H_layer_min_max'], 2 * np.array(tight['H_layer_min_max']))


def test_infeasible_camera_signal_is_not_returned_as_a_spectrum():
    r = exposure_bounds(np.array([-.1, .3, .4]), np.eye(3), np.eye(3))
    assert not r['success']
