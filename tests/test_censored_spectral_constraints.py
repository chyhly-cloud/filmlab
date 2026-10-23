"""Hand-solvable censored mixtures; no production rendering is involved."""

import numpy as np
import pytest

from tools.eval.censored_spectral_constraints import (
    fixed_spectrum_amplitude_interval,
    minimum_relative_tolerance,
    observation_bounds,
    solve_censored_constraints,
)


def test_two_basis_unique_mixture_and_coupled_signed_query():
    # R=c0, G=c1, B=c0+c1: measurements determine (2,1).
    camera = [[1., 0., 1.], [0., 1., 1.]]
    answer = solve_censored_constraints(camera, [2., 1., 3.], [False]*3,
                                        responses={"direct": [3., 2.],
                                                   "return": [-1., 4.]})
    assert answer["feasible"]
    np.testing.assert_allclose(answer["coefficients"], [2., 1.])
    for name, expected in [("direct", 8.), ("return", 2.)]:
        bound = answer["response_bounds"][name]
        assert bound["minimum"] == pytest.approx(expected)
        assert bound["maximum"] == pytest.approx(expected)
        np.testing.assert_allclose(bound["minimum_coefficients"], [2., 1.])
        np.testing.assert_allclose(bound["maximum_coefficients"], [2., 1.])


def test_three_basis_metamer_segment_has_hand_derived_extrema_and_witnesses():
    # Measurements yield c=(2-t,3-t,t), 0<=t<=2, so sum(c)=5-t
    # and (-c0+2*c2)=-2+3t. No particular point is the true spectrum.
    camera = np.array([[1., 0., 1.], [0., 1., 1.], [1., 1., 2.]])
    observed = np.array([2., 3., 5.])
    queries = {"inputY": np.ones(3), "red_return": np.array([-1., 0., 2.])}
    answer = solve_censored_constraints(camera, observed, [False]*3, responses=queries)
    expected = {"inputY": (3., 5.), "red_return": (-2., 4.)}
    for name, (lo, hi) in expected.items():
        bounds = answer["response_bounds"][name]
        assert bounds["minimum"] == pytest.approx(lo)
        assert bounds["maximum"] == pytest.approx(hi)
        for side in ("minimum", "maximum"):
            c = bounds[side+"_coefficients"]
            assert np.all(c >= 0.)
            np.testing.assert_allclose(c @ camera, observed)
            assert c @ queries[name] == pytest.approx(bounds[side])
    # The red-return maximum is not the inputY maximum; do not splice their
    # component extrema into a single alleged spectrum/exposure.
    np.testing.assert_allclose(answer["response_bounds"]["red_return"]["maximum_coefficients"], [0., 1., 2.])
    np.testing.assert_allclose(answer["response_bounds"]["inputY"]["maximum_coefficients"], [2., 3., 0.])


def test_saturation_is_lower_limit_not_equality_and_conflict_is_reported():
    camera = [[1., 0., 1.], [0., 1., 1.]]
    mask = [False, False, True]
    answer = solve_censored_constraints(camera, [2., 3., 4.], mask)
    assert answer["feasible"]
    np.testing.assert_allclose(answer["predicted_camera"], [2., 3., 5.])
    conflict = solve_censored_constraints(camera, [2., 3., 6.], mask)
    assert not conflict["feasible"]
    assert conflict["coefficients"] is None
    assert conflict["response_bounds"] == {}


def test_tolerances_are_explicit_and_never_relax_a_saturated_lower_limit():
    lower, upper = observation_bounds([4., -2., 1.], [False, False, True],
                                     relative_tolerance=[.1, .2, .9],
                                     absolute_tolerance=[.2, .1, .9])
    np.testing.assert_allclose(lower, [3.4, -2.5, 1.])
    np.testing.assert_allclose(upper[:2], [4.6, -1.5])
    assert upper[2] == np.inf
    # Large available-channel tolerances cannot rescue a zero-response
    # saturated channel. Its positive lower bound remains a hard constraint.
    conflict = solve_censored_constraints([[1., 0., 0.]], [0., 0., 1.],
                                          [False, False, True],
                                          relative_tolerance=10., absolute_tolerance=10.)
    assert not conflict["feasible"]


def test_all_saturated_positive_response_has_no_finite_upper_bound():
    answer = solve_censored_constraints(np.eye(3), [1., 2., 3.], [True]*3,
                                        responses={"energy": np.ones(3),
                                                   "zero": np.zeros(3),
                                                   "negative": -np.ones(3)})
    assert answer["feasible"] and answer["all_saturated"]
    assert not answer["has_measured_upper_bound"]
    energy = answer["response_bounds"]["energy"]
    assert energy["minimum"] == pytest.approx(6.)
    assert energy["maximum"] == np.inf
    assert energy["maximum_coefficients"] is None
    assert answer["response_bounds"]["zero"]["minimum"] == 0.
    assert answer["response_bounds"]["zero"]["maximum"] == 0.
    assert answer["response_bounds"]["negative"]["minimum"] == -np.inf
    assert answer["response_bounds"]["negative"]["maximum"] == pytest.approx(-6.)


def test_signed_camera_entries_do_not_allow_negative_spectral_coefficients():
    answer = solve_censored_constraints([[1., -1., 0.], [0., 1., 1.]],
                                        [2., -1., 1.], [False]*3,
                                        responses={"signed": [-2., 1.]})
    np.testing.assert_allclose(answer["coefficients"], [2., 1.])
    assert answer["response_bounds"]["signed"]["minimum"] == pytest.approx(-3.)
    assert not solve_censored_constraints([[1., 1., 1.]], [-1., -1., -1.],
                                           [False]*3)["feasible"]


def test_signed_all_saturated_constraints_do_not_force_a_false_unbounded_report():
    # Signed C and signed observations can imply c<=2 despite no observation
    # upper limits. Preserve this algebraic distinction from positive sensors.
    answer = solve_censored_constraints([[1., -1., 0.]], [-1., -2., 0.], [True]*3,
                                        responses={"coefficient": [1.]})
    assert answer["feasible"] and not answer["has_measured_upper_bound"]
    assert answer["response_bounds"]["coefficient"]["minimum"] == 0.
    assert answer["response_bounds"]["coefficient"]["maximum"] == pytest.approx(2.)


def test_camera_invisible_basis_can_leave_return_unbounded_even_without_saturation():
    answer = solve_censored_constraints([[1., 1., 1.], [0., 0., 0.]],
                                        [1., 1., 1.], [False]*3,
                                        responses={"return": [1., 1.]})
    assert answer["feasible"]
    assert answer["response_bounds"]["return"]["minimum"] == pytest.approx(1.)
    assert answer["response_bounds"]["return"]["maximum"] == np.inf


def test_fixed_spectrum_intersection_including_partial_censoring():
    answer = fixed_spectrum_amplitude_interval([2., 1., .5], [4., 2., 1.],
                                               [False, False, True],
                                               relative_tolerance=[.1, .2, 0.],
                                               absolute_tolerance=[.2, .1, 0.])
    # Available intervals [1.7,2.3] and [1.5,2.5], saturated lower a>=2.
    assert answer["feasible"]
    assert answer["minimum"] == pytest.approx(2.)
    assert answer["maximum"] == pytest.approx(2.3)
    conflict = fixed_spectrum_amplitude_interval([2., 1., .5], [4., 2., 1.5],
                                                 [False, False, True])
    assert not conflict["feasible"]
    assert conflict["minimum"] is None and conflict["maximum"] is None
    unbounded = fixed_spectrum_amplitude_interval([2., 1., .5], [4., 2., 1.], [True]*3)
    assert unbounded["minimum"] == pytest.approx(2.)
    assert unbounded["maximum"] == np.inf


def test_fixed_spectrum_zero_and_signed_responses():
    answer = fixed_spectrum_amplitude_interval([-1., 2., 0.], [-2., 4., 0.], [False]*3)
    assert answer["feasible"]
    assert answer["minimum"] == answer["maximum"] == 2.
    assert not fixed_spectrum_amplitude_interval([1., 1., 0.], [1., 1., .1],
                                                  [False]*3)["feasible"]
    assert not fixed_spectrum_amplitude_interval([1., 1., 0.], [1., 1., .1],
                                                  [False, False, True])["feasible"]


def test_minimum_tolerance_has_hand_derived_value_with_hard_saturation():
    # c0~2, c1~3 and c0+c1>=6. Both available channels must rise by 20%.
    answer = minimum_relative_tolerance([[1., 0., 1.], [0., 1., 1.]],
                                        [2., 3., 6.], [False, False, True])
    assert answer["feasible"]
    assert answer["minimum_relative_tolerance"] == pytest.approx(.2)
    np.testing.assert_allclose(answer["coefficients"], [2.4, 3.6])
    assert answer["predicted_camera"][2] >= 6.-1e-12
    # An independently known feasible observation requires no relaxation.
    exact = minimum_relative_tolerance([[1., 0., 1.], [0., 1., 1.]],
                                       [2., 3., 4.], [False, False, True])
    assert exact["minimum_relative_tolerance"] == pytest.approx(0.)


def test_minimum_tolerance_dark_floor_and_impossible_saturated_zero_response():
    dark = minimum_relative_tolerance([[0., 0., 0.]], [0., .001, 0.], [False]*3)
    assert dark["minimum_relative_tolerance"] == pytest.approx(.2)
    np.testing.assert_allclose(dark["normalization"], [.005]*3)
    conflict = minimum_relative_tolerance([[1., 1., 0.]], [1., 1., .1],
                                          [False, False, True])
    assert not conflict["feasible"]
    assert conflict["minimum_relative_tolerance"] == np.inf


@pytest.mark.parametrize("bad_camera,bad_observed,bad_mask,kwargs", [
    ([[np.nan, 1., 1.]], [1., 1., 1.], [False]*3, {}),
    ([[1., 1., 1.]], [1., np.inf, 1.], [False]*3, {}),
    ([[1., 1., 1.]], [1., 1., 1.], [False]*3, {"relative_tolerance": np.nan}),
    ([[1., 1., 1.]], [1., 1., 1.], [False]*3, {"absolute_tolerance": -.1}),
    ([[1., 1., 1.]], [1., 1., 1.], [False]*3, {"responses": {"bad": [np.inf]}}),
    ([[1., 1., 1.]], [1., 1., 1.], [0., 0., np.nan], {}),
    ([[1., 1.]], [1., 1., 1.], [False]*3, {}),
    ([], [1., 1., 1.], [False]*3, {}),
])
def test_invalid_inputs_are_rejected_before_solving(bad_camera, bad_observed, bad_mask, kwargs):
    with pytest.raises(ValueError):
        solve_censored_constraints(bad_camera, bad_observed, bad_mask, **kwargs)


def test_other_public_entrypoints_also_reject_nonfinite_inputs():
    with pytest.raises(ValueError):
        fixed_spectrum_amplitude_interval([1., 1., np.inf], [1., 1., 1.], [False]*3)
    with pytest.raises(ValueError):
        minimum_relative_tolerance([[1., np.nan, 1.]], [1., 1., 1.], [False]*3)
