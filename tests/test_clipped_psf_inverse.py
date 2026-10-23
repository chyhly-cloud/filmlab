import numpy as np
import pytest
from scipy.integrate import quad
from tools.eval.clipped_psf_inverse import profile,integrated_signal,censored_residual,fit_channel


def test_finite_energy_integrals_match_independent_radial_quadrature():
    for kind,beta in [('gaussian',2.),('moffat',1.5),('moffat',3.)]:
        numeric=quad(lambda r:2*np.pi*r*3*profile(kind,r*r,2,beta),0,np.inf)[0]
        assert numeric==pytest.approx(integrated_signal(kind,3,2,beta),rel=1e-9)
    with pytest.raises(ValueError):integrated_signal('moffat',1,2,1.)


def test_clipped_samples_supply_only_lower_constraints():
    r=censored_residual([2.,.5,1.],[1.,1.,.9],[True,True,False])
    assert r[0]==0 and r[1]<0 and r[2]>0


def test_synthetic_gaussian_core_can_be_recovered_from_its_unclipped_wings():
    y,x=np.indices((41,41));true=.01+3*np.exp(-((x-20)**2+(y-20)**2)/(2*3**2))
    sat=true>=.8;observed=np.minimum(true,.8)
    estimate,report,train,test=fit_channel(observed,sat,7,'gaussian',100.,starts=4,max_nfev=250)
    assert not np.any(train&test)
    assert np.max(abs(estimate-true))<.002
    assert report['heldout_asinh_RMSE']<.002


def test_hard_parameterization_obeys_all_declared_core_censoring_bounds():
    from tools.eval.clipped_psf_hard_inverse import hard_prediction
    y,x=np.indices((21,21));obs=np.ones((21,21));source=(x-10)**2+(y-10)**2<=4**2;sat=source.copy()
    for kind,beta in [('gaussian',2.),('moffat',1.5)]:
        p=[10.2,9.8,-12.,np.log(2),.01,beta]
        pred,amin,amplitude=hard_prediction(kind,p,y,x,obs,sat,source)
        assert np.min(pred[source]-obs[source])>=-1e-12
        assert amplitude>amin


def test_hard_fit_recovers_known_gaussian_without_seeing_hidden_core():
    from tools.eval.clipped_psf_hard_inverse import hard_fit
    y,x=np.indices((41,41));truth=.01+3*np.exp(-((x-20)**2+(y-20)**2)/(2*3**2));sat=truth>.8;obs=np.minimum(truth,.8);source=(x-20)**2+(y-20)**2<=7**2
    pred,info=hard_fit(obs,sat,source,7,'gaussian',100.,starts=4,max_nfev=300)
    assert np.max(abs(pred-truth))<1e-5
    assert info['best']['minimum_censoring_margin']>=-1e-9


def test_disk_gaussian_convolution_preserves_disk_integrated_flux():
    radius,sigma,amplitude=3.,1.,2.
    numeric=quad(lambda r:2*np.pi*r*amplitude*profile('disk',r*r,sigma,radius),0,30,epsabs=1e-9)[0]
    assert numeric==pytest.approx(np.pi*radius**2*amplitude,rel=1e-9)
    assert profile('disk',0,sigma,radius)==pytest.approx(1-np.exp(-radius**2/(2*sigma**2)),rel=1e-10)
