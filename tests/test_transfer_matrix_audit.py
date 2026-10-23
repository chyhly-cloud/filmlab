import numpy as np
from tools.eval.transfer_matrix_audit import jacobian,sensor_density_jacobian


def test_detector_jacobian_preserves_neutral_density_and_matches_spectral_difference():
    rng=np.random.default_rng(9);density=rng.uniform(.1,2,(4,81));response=rng.uniform(.1,1,(81,3))
    trans=10**(-density);j=sensor_density_jacobian(trans,response)
    # Any spectrally neutral density increment must appear in all detector bands.
    np.testing.assert_allclose(j.sum(-1),1,atol=3e-15)
    direction=rng.normal(size=(4,81));eps=1e-5
    actual=(-np.log10(10**(-(density+eps*direction))@response)+np.log10(10**(-(density-eps*direction))@response))/(2*eps)
    expected=np.einsum('ncl,nl->nc',j,direction)
    np.testing.assert_allclose(actual,expected,atol=3e-10)


def test_local_jacobian_has_output_by_input_orientation():
    x=np.array([[.2,-.1,.4],[.5,.1,-.2]])
    matrix=np.array([[.2,.1,-.3],[.5,-.2,.1],[-.1,.3,.6]])
    expected=np.exp(x@matrix)[:,:,None]*matrix.T[None,:,:]
    np.testing.assert_allclose(jacobian(lambda a:np.exp(a@matrix),x,1e-5),expected,atol=3e-11)
