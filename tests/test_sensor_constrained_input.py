"""Censored observations remain measurements, estimates remain explicitly unknown."""
import numpy as np
import pytest

from filmlab.rawio import XYZ_TO_SRGB
from tools.eval.sensor_constrained_input import constrain_camera, from_measurements


def test_float_matrix_direction_and_highlights_are_not_clipped():
    camera = np.array([[[1.3, .2, .1], [.2, .5, .8]]])
    matrix = np.array([[.5, .1, .05], [.2, .7, .1], [.1, .05, 1.1]])
    result = from_measurements(camera, np.zeros_like(camera, bool), matrix)
    expected = np.einsum("...c,cd,ed->...e", camera, matrix, XYZ_TO_SRGB)
    np.testing.assert_allclose(result.rgb, expected, atol=1e-15)
    assert np.any(result.rgb > 1)
    np.testing.assert_array_equal(result.camera, camera)
    # Purely linear: no hidden exposure/tone/normalization step.
    doubled = from_measurements(2*camera, np.zeros_like(camera, bool), matrix)
    np.testing.assert_allclose(doubled.rgb, 2*result.rgb)


def test_floor_retains_negative_noise_and_censored_values_without_claiming_recovery():
    camera = np.array([[[-.002, .1, .3], [1., .8, .4]]])
    sat = np.array([[[False, False, False], [True, False, False]]])
    result = from_measurements(camera, sat, np.eye(3))
    np.testing.assert_array_equal(result.camera, camera)
    np.testing.assert_array_equal(result.unknown_mask, sat)
    assert not result.estimated_mask.any()
    assert result.metadata['counts']['observed_camera_channels_below_zero'] == 1
    assert result.metadata['unsaturated_camera_channels_max_change'] == 0
    assert "not a rigorous bound" in " ".join(result.metadata['limits'])


def test_ratio_only_changes_missing_channels_and_keeps_fully_lost_cell_unknown():
    camera = np.broadcast_to([.8, .2, .1], (25, 25, 3)).copy()
    sat = np.zeros_like(camera, bool)
    camera[12, 12] = [1., .5, .25]  # Nearby colour predicts missing R=2.
    sat[12, 12, 0] = True
    camera[7, 7] = [1., 1., 1.]
    sat[7, 7] = True
    before = camera.copy()
    result = from_measurements(camera, sat, np.eye(3), mode='local_ratio')
    assert result.camera[12, 12, 0] == pytest.approx(2.)
    np.testing.assert_array_equal(result.camera[~sat], before[~sat])
    np.testing.assert_array_equal(result.camera[7, 7], before[7, 7])
    np.testing.assert_array_equal(result.unknown_mask, sat)  # Estimates do not become observations.
    assert result.estimated_mask.sum() == 1
    assert result.metadata['unsaturated_camera_channels_max_change'] == 0
    np.testing.assert_array_equal(camera, before)
    again = from_measurements(camera, sat, np.eye(3), mode='local_ratio')
    np.testing.assert_array_equal(result.camera, again.camera)


def test_missing_channel_is_never_reduced_and_no_neighbours_means_no_estimate():
    camera = np.broadcast_to([.1, .2, .3], (15, 15, 3)).copy()
    sat = np.zeros_like(camera, bool)
    camera[7, 7, 0] = 1.
    sat[7, 7, 0] = True
    constrained, estimated = constrain_camera(camera, sat, mode='local_ratio')
    np.testing.assert_array_equal(constrained, camera)
    assert not estimated.any()
    sat[:] = True
    sat[7, 7, 1:] = False
    constrained, estimated = constrain_camera(camera, sat, mode='local_ratio')
    np.testing.assert_array_equal(constrained, camera)
    assert not estimated.any()


def test_bad_shapes_and_modes_are_rejected():
    with pytest.raises(ValueError, match='mode'):
        constrain_camera(np.zeros((2, 2, 3)), np.zeros((2, 2, 3)), mode='guess')
    with pytest.raises(ValueError, match='shapes'):
        constrain_camera(np.zeros((2, 3)), np.zeros((2, 3)))
    with pytest.raises(ValueError, match='matrix_xyz'):
        from_measurements(np.zeros((2, 2, 3)), np.zeros((2, 2, 3)), np.eye(2))
