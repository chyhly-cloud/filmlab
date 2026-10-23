"""Sensor censoring, rank deficiency and independent predictions stay distinct."""
import numpy as np
import pytest

from tools.eval.inverse_radiance_checks import (
    censored_sensor_residual, sensor_consistency, spectral_rank_report,
    heldout_residual_report, near_equivalent_fit_report, fixed_spectrum_amplitude_bounds,
)


def test_censored_camera_samples_have_no_penalty_above_lower_bound():
    observation = np.array([[1., .2, .1], [1., 1., 1.]])
    saturated = np.array([[True, False, False], [True, True, True]])
    prediction = np.array([[8., .22, .08], [1., 10., 100.]])
    residual = censored_sensor_residual(prediction, observation, saturated, .01)
    np.testing.assert_allclose(residual, [[0., 2., -2.], [0., 0., 0.]])
    report = sensor_consistency(prediction, observation, saturated, .01)
    assert report['available_channels'] == 2
    assert report['censored_lower_bound_violation_fraction'] == 0
    assert report['upper_radiance_bound_provided'] is False
    prediction[0, 0] = .9
    assert censored_sensor_residual(prediction, observation, saturated, .01)[0, 0] == pytest.approx(-10.)


def test_sensor_residual_is_invariant_to_common_linear_signal_unit_change():
    a = np.array([[1., -.001, .12]])
    b = np.array([[2., -.002, .115]])
    sat = np.array([[True, False, False]])
    sigma = np.array([.01, .002, .004])
    np.testing.assert_allclose(censored_sensor_residual(b, a, sat, sigma),
                               censored_sensor_residual(7*b, 7*a, sat, 7*sigma))
    with pytest.raises(ValueError, match='noise_sigma'):
        censored_sensor_residual(b, a, sat, 0.)


def test_additional_same_sensor_exposure_cannot_remove_spectral_nullity():
    basis = np.eye(4)
    camera = np.array([[1., 0, 0], [0, 1., 0], [0, 0, 1.], [.4, .3, .3]])
    a = spectral_rank_report(basis, camera)
    # Concatenating channels from a second exposure does not add independent spectral information.
    b = spectral_rank_report(basis, np.concatenate([camera, 2*camera], axis=1))
    assert a['rank'] == b['rank'] == 3
    assert a['coefficient_nullity'] == b['coefficient_nullity'] == 1


def test_holdout_can_reject_a_perfect_training_fit():
    target = np.zeros((2, 3))
    baseline = np.ones_like(target)
    fitted = np.array([[0., 0, 0], [2., 2, 2]])
    report = heldout_residual_report(baseline, fitted, target, [True, False], [False, True])
    assert report['train']['fitted_channel_RMSE'] == 0
    assert report['heldout']['fitted_channel_RMSE'] == 2
    assert report['train_improves_but_holdout_worsens']
    with pytest.raises(ValueError, match='disjoint'):
        heldout_residual_report(baseline, fitted, target, [True, False], [True, True])


def test_near_equal_final_losses_do_not_imply_a_single_source_spectrum():
    coefficients = np.array([[1., 0.], [0., 3.], [0., 100.]])
    losses = np.array([1., 1.001, 2.])
    basis = np.array([[1., 0, 0], [0., 0, 1.]])
    report = near_equivalent_fit_report(coefficients, losses, accepted_loss_ceiling=1.01, basis_spectra=basis)
    assert report['accepted_trials'] == 2
    assert report['discrete_spectral_sum_range'] == [1., 3.]
    assert report['spectrum_min'] == [0., 0., 0.]
    assert report['spectrum_max'] == [1., 0., 3.]
    empty = near_equivalent_fit_report(coefficients, losses, accepted_loss_ceiling=.1)
    assert empty['accepted_trials'] == 0


def test_fixed_shape_available_channels_bound_amplitude_and_censored_ones_only_lower_it():
    # R has clipped at 1; known G/B constrain the common amplitude near 2.
    response = np.array([1., .25, .125])
    observed = np.array([1., .5, .25])
    sat = np.array([True, False, False])
    bounds = fixed_spectrum_amplitude_bounds(response, observed, sat, .01, tolerance_sigma=1.)
    assert bounds['feasible']
    assert bounds['amplitude_lower'] == pytest.approx(1.96)
    assert bounds['amplitude_upper'] == pytest.approx(2.04)
    all_lost = fixed_spectrum_amplitude_bounds(response, np.ones(3), np.ones(3, bool), .01, tolerance_sigma=1.)
    assert all_lost['amplitude_lower'] == pytest.approx(7.92)
    assert all_lost['amplitude_upper'] is None
    assert all_lost['unbounded_above']


def test_fixed_shape_can_be_rejected_without_fitting_its_amplitude():
    incompatible = fixed_spectrum_amplitude_bounds([1., 1.], [1., .1], [False, False], .01)
    assert not incompatible['feasible']
    zero_response = fixed_spectrum_amplitude_bounds([0.], [1.], [True], .01)
    assert not zero_response['feasible']
    assert zero_response['incompatible_zero_response_channels'] == 1
    with pytest.raises(ValueError, match='nonnegative'):
        fixed_spectrum_amplitude_bounds([-1.], [.1], [False], .01)
