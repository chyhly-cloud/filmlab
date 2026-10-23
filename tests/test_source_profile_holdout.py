import numpy as np
import pytest

from tools.eval.source_profile_holdout import shape_metrics


def test_shape_metrics_are_invariant_to_separate_positive_scalars():
    a = np.array([[1.,2.],[3.,4.]])
    b = np.array([[2.,1.],[2.,5.]])
    mask = np.ones((2,2),bool)
    left, right = shape_metrics(a,b,mask), shape_metrics(9*a,3*b,mask)
    for key in ['unit_sum_total_variation','centroid_distance_pixels','peak_to_mean_ratio_estimate_over_reference']:
        assert left[key] == pytest.approx(right[key])


def test_centroid_shift_and_total_variation_of_point_signals():
    a = np.zeros((3,4)); b = a.copy()
    a[1,2], b[1,1] = 3.,2.
    r=shape_metrics(a,b,np.ones(a.shape,bool))
    assert r['unit_sum_total_variation'] == pytest.approx(1.)
    assert r['centroid_distance_pixels'] == pytest.approx(1.)
    assert r['centroid_estimate_minus_reference_yx_pixels'] == pytest.approx([0.,1.])


def test_unsupported_values_do_not_enter_and_negative_supported_signal_rejected():
    mask=np.array([[True,False],[True,False]])
    a=np.array([[1.,np.nan],[2.,-100.]])
    assert shape_metrics(a,a,mask)['unit_sum_total_variation']==0.
    a[0,0]=-1.
    with pytest.raises(ValueError):shape_metrics(a,a,mask)
