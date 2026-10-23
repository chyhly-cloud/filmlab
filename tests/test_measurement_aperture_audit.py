import numpy as np
import pytest
from scipy import sparse

from tools.eval.measurement_aperture_audit import (
    native_support_overlap, radial_footprint_summary, geometric_axis_guard,
)


def test_exact_native_overlap_retains_tiny_positive_weights_and_reports_rows():
    # Two separate output measurements can share one native sample. Even a
    # small finite tail counts for a claim of strict support independence.
    operator = sparse.csr_matrix([[1.-1e-12, 0., 1e-12, 0.],
                                  [0., .7, .3, 0.], [0., 0., 0., 1.]])
    answer, columns = native_support_overlap(operator, [0], [1, 2])
    np.testing.assert_array_equal(columns, [2])
    assert not answer['strictly_disjoint']
    assert answer['train_rows_touching_shared_columns'] == 1
    assert answer['heldout_rows_touching_shared_columns'] == 1
    assert answer['maximum_train_row_weight_on_shared_columns'] == pytest.approx(1e-12)
    assert answer['maximum_heldout_row_weight_on_shared_columns'] == pytest.approx(.3)
    disjoint, _ = native_support_overlap(operator, [0], [2])
    assert disjoint['strictly_disjoint']


def test_distinct_zero_radius_output_apertures_still_overlap_after_bilinear_warp():
    from tools.eval.inverse_source_radiance import observation_operator
    # Output x=20 samples native x=10; output x=21 samples x=10 and11.
    # Gaussian sigma .1 rounds its finite support radius to zero.
    operator = observation_operator((30, 30), np.diag([2., 2., 1.]),
                                    np.array([[20, 20], [20, 21]]), .1)
    answer, columns = native_support_overlap(operator, [0], [1])
    np.testing.assert_array_equal(columns, [10*30+10])
    assert answer['maximum_train_row_weight_on_shared_columns'] == 1.
    assert answer['maximum_heldout_row_weight_on_shared_columns'] == .5


def test_radial_labels_are_not_claims_that_entire_apertures_are_in_one_zone():
    operator = sparse.csr_matrix([[.75, .25, 0.], [0., .4, .6]])
    # Centre labels are 290 and310, while actual native measurements straddle300.
    distances = np.array([[290., 310.]])
    points = np.array([[0, 0], [0, 1]])
    native_distance = np.array([280., 305., 320.])
    near = radial_footprint_summary(operator, [0], points, distances, native_distance,
                                    centre_side='near')
    outer = radial_footprint_summary(operator, [1], points, distances, native_distance,
                                     centre_side='outer')
    assert near['rows_with_nonzero_weight_across_boundary'] == 1
    assert near['mean_opposite_side_row_weight'] == .25
    assert outer['rows_with_nonzero_weight_across_boundary'] == 0
    assert outer['mean_opposite_side_row_weight'] == 0.
    assert near['native_footprint_distance_global_min_max'] == [280., 305.]


def test_geometric_guard_accounts_for_native_neighbour_beyond_gaussian_radius():
    # A native pixel spans four output pixels. Fractional output positions can
    # draw the opposite native neighbour three output pixels away.
    answer = geometric_axis_guard((8, 8), np.diag([4., 4., 1.]), 1., output_shape=(24, 24))
    assert answer['gaussian_square_half_width_output_pixels'] == 4
    np.testing.assert_allclose(answer['maximum_bilinear_native_centre_displacement_xy_output_pixels'], [3., 3.])
    assert answer['conservative_axis_guard_output_pixels'] == 8


def test_invalid_or_incomplete_measurements_are_not_silently_accepted():
    with pytest.raises(ValueError):
        native_support_overlap(sparse.csr_matrix([[.9, 0.], [0., 1.]]), [0], [1])
    with pytest.raises(ValueError):
        native_support_overlap(sparse.eye(2), [0, 0], [1])
