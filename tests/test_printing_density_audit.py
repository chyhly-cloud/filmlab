import numpy as np
import pytest
from tools.eval.printing_density_audit import exact_printing_density


def test_base_zero_and_neutral_absorber_adds_exact_density():
    t0=np.linspace(.05,.9,81);s=np.column_stack([np.linspace(.1,1,81),np.linspace(1,.1,81),np.ones(81)])
    np.testing.assert_allclose(exact_printing_density(t0,t0,s),0,atol=1e-15)
    for d in [.1,1,3]:np.testing.assert_allclose(exact_printing_density(t0*10**(-d),t0,s),d,atol=1e-14)


def test_spectral_integration_is_linear_before_log_not_density_averaging():
    t0=np.ones(2);s=np.ones((2,1));t=np.array([1,.01])
    np.testing.assert_allclose(exact_printing_density(t,t0,s),-np.log10(.505))
    assert not np.isclose(exact_printing_density(t,t0,s)[0],1.)
    np.testing.assert_allclose(exact_printing_density(t,t0,s*17),exact_printing_density(t,t0,s))


def test_zero_response_is_explicit_failure():
    with pytest.raises(ValueError,match='positive'):exact_printing_density(np.zeros(81),np.ones(81),np.ones((81,3)))


def test_neutral_matrix_constraint_and_constrained_minimum():
    from tools.eval.printing_density_audit import neutral_preserving_matrix
    rng=np.random.default_rng(123);x=rng.uniform(0,2,(100,3));y=rng.uniform(0,2,(100,3))
    p=neutral_preserving_matrix(x,y)
    np.testing.assert_allclose(p.sum(0),1,atol=3e-16)
    # Residual is orthogonal to every admissible zero-column-sum update.
    basis=np.array([[1.,0.],[0.,1.],[-1.,-1.]])
    np.testing.assert_allclose((x@basis).T@(x@p-y),0,atol=1e-12)
    neutral=np.full((4,3),.7)
    np.testing.assert_allclose(neutral@p,neutral,atol=1e-15)
