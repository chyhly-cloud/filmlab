"""Independent analytic checks for angle/layer absorption, not 800T fitting."""
import numpy as np
import pytest
from tools.eval.angular_colour_audit import deposit_segment,base_return,analytic_ballistic
from tools.eval.photon_transport import Layer,simulate


def test_finite_absorption_segment_matches_beer_lambert_and_radial_integrals():
    edges=np.array([0.,1.,2.,3.,20.]);h=np.zeros(4)
    mu=np.array([.6]);tangent=np.sqrt(1-mu[0]**2)/mu[0]
    out,overflow=deposit_segment(h,edges,np.array([.7]),.2,3,mu,np.array([0.]))
    assert out[0]==pytest.approx(.7*np.exp(-1))
    assert h[0]==pytest.approx(.7*(-np.expm1(-.2*(1/tangent)/mu[0])))
    assert h.sum()+overflow+out[0]==pytest.approx(.7,abs=1e-15)
    # Pure normal incidence deposits at one radius; overflow stays in its ledger.
    h=np.zeros(4);o,t=deposit_segment(h,edges,np.array([1.]),.1,2,np.array([1.]),np.array([21.]))
    assert h.sum()==0
    assert o[0]+t==pytest.approx(1)


def test_multilayer_order_changes_return_allocation_without_energy_creation():
    layers=[Layer('lower',3,.04,0),Layer('middle',3,.04,0),Layer('upper',3,.04,0)]
    r=base_return(layers,count=2000)
    assert abs(r['ledger']['closure_error'])<1e-13
    np.testing.assert_allclose(r['radial_absorbed'].sum(1)+r['overflow'],r['absorbed'],atol=1e-14)
    edges=r['radial_edges'];h=r['radial_absorbed']
    near=h[:,edges[:-1]<300].sum(1);far=h[:,edges[:-1]>=800].sum(1)
    assert far[0]/far.sum()>near[0]/near.sum()
    # The effect is path ordering, not a colour-specific multiplier.
    assert r['absorbed'][0]>r['absorbed'][1]>r['absorbed'][2]


def test_exact_two_pass_geometry_matches_mc_layer_and_radius_counts():
    layers=[Layer('top',3,.02,0),Layer('lower',3,.04,0),Layer('base',130,0,0)]
    mu=.6;edges=np.arange(0,1001,25.)
    a=analytic_ballistic(layers,mu,edges)
    b=simulate(layers,photons=40000,seed=137,n_top=1.5,n_bottom=1,incident_cosine=mu,radial_edges=edges)
    assert abs(a['ledger']['closure_error'])<1e-13
    expected=a['radial_returned'];sigma=np.sqrt(expected*(1-expected)/40000)
    assert np.all(abs(b['radial_returned']-expected)<6*sigma+1/40000)
    # An incorrect assignment at base-front instead of actual absorption depth
    # would shift event bins in this oblique finite-thickness configuration.
    assert np.count_nonzero(expected)>1
