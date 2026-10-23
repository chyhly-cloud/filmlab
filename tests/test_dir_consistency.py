import numpy as np
import pytest
from scipy.optimize import root
from filmlab import chemistry,generation
from filmlab.engine import Engine,Params
from filmlab.recipes import recipe
from tools.eval.self_consistent_dir import solve


@pytest.fixture
def emulsion():
    previous=generation.active();generation.use('next')
    try:yield Engine(Params(**{**recipe('cinestill_800t'),'roll':None})).e
    finally:generation.use(previous)


def test_closure_agrees_with_independent_root_solver_and_relaxation(emulsion):
    e=emulsion;S=chemistry.kinetic_curve(e.S_nom,1.1,e.log_exposure)
    A=chemistry._dir_matrix(e.dir_amount,True);C=chemistry._intrinsic_curve(e,S,A)
    q=np.array([[-1,.2,.4],[.5,-.3,1.1],[1.2,1.4,.7]])
    d=solve(e,q,1.1,e.dir_amount,0)
    for qi,di in zip(q,d):
        def residual(x):return x-np.minimum(chemistry._curves_at(qi-x/e.halide_cap@A,e.log_exposure,C),e.halide_cap)
        result=root(residual,di+.01)
        assert result.success
        np.testing.assert_allclose(di,result.x,atol=2e-8)
    np.testing.assert_allclose(d,solve(e,q,1.1,e.dir_amount,0,relaxation=.25),atol=3e-8)


def test_no_inhibition_limit_and_failed_convergence_are_explicit(emulsion):
    q=np.array([[.2,-.5,1.]])
    np.testing.assert_array_equal(solve(emulsion,q,1.1,0,0),chemistry.develop_image(emulsion,q,1.1,0,0))
    with pytest.raises(RuntimeError,match='did not converge'):
        solve(emulsion,q,1.1,1,0,max_iterations=1,tolerance=1e-14)


def test_spatial_closure_preserves_finite_nonnegative_bounded_silver(emulsion):
    q=np.zeros((9,9,3));q[4,4]=[2.,-.5,.2]
    d=solve(emulsion,q,1.1,1,1)
    assert np.isfinite(d).all()
    assert (d>=0).all() and (d<=emulsion.halide_cap).all()
    # Channel-symmetric spatial geometry must not acquire a directional artefact.
    np.testing.assert_allclose(d,d[::-1,::-1],atol=1e-14)
