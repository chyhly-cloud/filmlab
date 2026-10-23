"""Depth need is a model-identifiability question, not an appearance score."""
import numpy as np
import pytest
from scipy.integrate import quad
from tools.eval.depth_sufficiency_audit import absorption_bins,integrated_response
from tools.eval.photon_transport import Layer,simulate


def test_equal_absorption_front_back_does_not_change_constant_yield():
    edges=np.linspace(0,3,49)
    a=absorption_bins(edges,.05);b=absorption_bins(edges,.05,back=True)
    np.testing.assert_allclose(a,b[::-1],atol=2e-16)
    assert not np.allclose(a,b,rtol=1e-3)
    assert a.sum()==pytest.approx(b.sum(),abs=2e-16)
    f=integrated_response(3,.05,.5,.5);r=integrated_response(3,.05,.5,.5,back=True)
    assert f['effective_latent']==r['effective_latent']
    assert f['total_absorbed']==r['total_absorbed']
    assert f['first_depth_moment']!=r['first_depth_moment']


def test_stipulated_gradient_changes_yield_but_weighted_integral_still_suffices():
    d=3.;a=.05;gradient=[]
    for back in [False,True]:
        exact=integrated_response(d,a,.2,.8,back=back)
        numerical=quad(lambda z:a*np.exp(-a*(d-z if back else z))*(.2+.6*z/d),0,d)[0]
        assert exact['effective_latent']==pytest.approx(numerical,abs=2e-15)
        gradient.append(exact['effective_latent'])
    assert gradient[1]>gradient[0]
    # Given q(z), all depth data can be compressed into this SAME scalar H_eff.
    # The counterexample concerns unweighted absorption, not effective-H records.
    assert sum(gradient)==pytest.approx((-np.expm1(-a*d))*(.2+.8),abs=2e-15)
    with pytest.raises(ValueError,match='yield'):
        integrated_response(3,.05,-.1,.8)


def test_independent_mc_depth_histogram_matches_exact_slab():
    count=25000;n=24;edges=np.linspace(0,3,n+1)
    layers=[Layer('identical_numeric_depth_box',3/n,.05,0,n=1.5) for _ in range(n)]
    result=simulate(layers,photons=count,seed=903,n_top=1.5,n_bottom=1.5,max_events=100)
    exact=absorption_bins(edges,.05);sigma=np.sqrt(exact*(1-exact)/count)
    assert np.all(abs(result['absorbed']-exact)<6*sigma+1/count)
    assert abs(result['ledger']['closure_error'])<1e-13
    assert result['ledger']['event_residual']==0
