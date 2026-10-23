import numpy as np
from tools.eval.radiance_censor_holdout import censor,local_regression,reference_fields,score


def test_censor_keeps_measured_channels_and_only_gives_lower_bounds():
    truth=np.array([[[.2,1.2,3.],[-.01,.3,.8]]]);obs,sat=censor(truth,1.)
    np.testing.assert_array_equal(obs[~sat],truth[~sat])
    assert np.all(truth[sat]>=obs[sat])
    assert sat.sum()==2 and obs.min()<0


def test_regression_never_uses_hidden_source_targets_and_preserves_measurements():
    yy,xx=np.indices((41,41));amp=.01+.0008*(xx+yy)
    signal=amp[...,None]*np.array([4.,1.,.5]);core=(xx-20)**2+(yy-20)**2<36
    hidden=signal.copy();hidden[core&(signal[...,0]>=.15),0]=10. # Target values cannot leak through cap.
    a,s=censor(signal,.15);b,t=censor(hidden,.15)
    # Only score core cells actually red-censored in both copies.
    assert np.array_equal(a[core],b[core])
    p,lo,hi,_=local_regression(a,s,core);q,_,_,_=local_regression(b,t,core)
    np.testing.assert_array_equal(p,q)
    np.testing.assert_array_equal(p[~s],a[~s])
    assert np.max(abs(p[core,0]-signal[core,0]))<1e-12


def test_full_censoring_has_no_finite_upper_or_claimed_recovery():
    obs=np.ones((12,12,3))*.1;sat=np.ones_like(obs,bool);core=np.ones((12,12),bool)
    p,lo,hi,fit=local_regression(obs,sat,core)
    np.testing.assert_array_equal(p,obs);assert np.isinf(hi).all();assert not fit


def test_two_view_reference_rejects_disagreement_or_saturation():
    a=np.ones((4,4,3))*.1;b=a.copy();b[0,0]=.2
    safe=np.ones((4,4),bool);unsafe=safe.copy();unsafe[1,1]=False
    lamp={'views':['long','short'],'alternatives':[{'local_geometry':{'pass':True}}]}
    data={'0_camera':a,'1_camera':b,'0_sat':np.zeros_like(a,bool),'1_sat':np.zeros_like(a,bool),'0_safe':safe,'1_safe':unsafe}
    _,support,_,_=reference_fields(lamp,data)
    assert not support[0,0] and not support[1,1] and support.sum()==14
