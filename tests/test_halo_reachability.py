import numpy as np
import pytest
from tools.eval.halo_reachability import add_exposure,feasible
from tools.eval.halo_joint_transport import marginal_product,relative_tv


def test_return_is_added_in_linear_exposure_not_log_exposure():
    q=np.log10(np.array([1.,10.,100.]))
    # Equal optical additions do NOT give equal EV/logH increments.
    result=add_exposure(q,[1.,1.,1.])
    np.testing.assert_allclose(10**result,[2.,11.,101.])
    assert np.all(result>=q)
    with pytest.raises(ValueError):add_exposure(q,[1.,0.,-.1])


def test_luminance_cannot_be_traded_for_good_hue_in_feasibility():
    target=np.array([82.,10.,12.])
    assert not feasible([84.,10.,12.],target)
    assert not feasible([82.,10.,15.],target)
    assert feasible([82.5,10.5,12.5],target)


def test_layerwise_separation_preserves_energy_marginals_and_overflow():
    # Last bin can be overflow: it is never discarded or redistributed.
    joint=np.array([[[3.,0.,1.],[0.,1.,0.]],[[0.,2.,1.],[0.,0.,0.]]])
    approx=marginal_product(joint)
    np.testing.assert_allclose(approx.sum(-1),joint.sum(-1))
    np.testing.assert_allclose(approx.sum(0),joint.sum(0))
    assert relative_tv(joint,approx)>0


def test_shared_spectral_shape_is_an_exact_separable_control():
    tensor=np.array([1.,2.,3.])[:,None,None]*np.array([1.,4.,.5])[None,:,None]*np.array([.1,.2,.7])[None,None,:]
    np.testing.assert_allclose(marginal_product(tensor),tensor,atol=1e-15)
