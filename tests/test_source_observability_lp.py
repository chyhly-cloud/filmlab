import numpy as np
import pytest

from tools.eval.source_observability_lp import extreme_Y


def test_two_exact_responses_allow_different_nonnegative_total_input_extrema():
    responses=np.array([[1.,1.],[2.,1.],[1.,2.]])
    camera=np.ones((3,3))
    low=extreme_Y(responses,[0,1],[3.,3.],camera,np.ones(3))
    high=extreme_Y(responses,[0,1],[3.,3.],camera,np.ones(3),maximize=True)
    assert low.success and high.success
    assert low.x.sum()==pytest.approx(2.)
    assert high.x.sum()==pytest.approx(3.)
    np.testing.assert_allclose(low.x@responses,[3.,3.])
    np.testing.assert_allclose(high.x@responses,[3.,3.])


def test_hard_camera_lower_bound_can_reject_an_apparent_response_witness():
    responses=np.array([[1.,1.],[2.,1.],[1.,2.]])
    assert not extreme_Y(responses,[0,1],[3.,3.],np.ones((3,3)),np.full(3,4.)).success
