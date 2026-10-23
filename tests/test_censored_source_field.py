import numpy as np
import pytest

from tools.eval.censored_source_field import linear_proposal, field_summary


def test_fixed_total_linear_step_recovers_known_zero_sum_signal():
    s = np.array([1., 1., 1.])
    target = np.array([1.4, .7, .9])
    result = linear_proposal(np.eye(3), s-target, s, np.full(3,.1), .5)
    assert result.success
    np.testing.assert_allclose(s+result.x, target, atol=1e-6)
    assert abs(result.x.sum()) < 1e-12


def test_censor_floor_and_trust_box_hold_when_target_is_infeasible():
    s = np.array([1., 1.])
    result = linear_proposal(np.eye(2), s-np.array([4., -2.]), s,
                             np.array([.9, .8]), .15)
    assert result.success
    np.testing.assert_allclose(result.x, [.15, -.15], atol=1e-7)
    assert np.all(s+result.x >= [.9, .8])
    assert abs(result.x.sum()) < 1e-12


def test_infeasible_initial_source_is_rejected():
    with pytest.raises(ValueError):
        linear_proposal(np.eye(2), np.ones(2), np.array([.5,1.]), np.ones(2), .5)


def test_shape_summary_distinguishes_two_cell_uniform_from_point_source():
    positions = np.array([[0.,0.],[1.,0.]])
    uniform = field_summary(np.ones(2), positions, np.zeros(2))
    point = field_summary(np.array([2.,0.]), positions, np.zeros(2))
    assert uniform['effective_cells'] == pytest.approx(2.)
    assert uniform['peak_to_mean'] == pytest.approx(1.)
    assert uniform['rms_radius_native_pixels'] == pytest.approx(.5)
    assert point['effective_cells'] == pytest.approx(1.)
    assert point['maximum_single_cell_fraction'] == pytest.approx(1.)
