import numpy as np
from tools.eval.dye_contribution_units import ev_to_log10_exposure,nonlinear_counterexample,shapley_corner_allocation,corner_marginal_ranges


def test_ev_is_log2_but_effective_q_is_log10():
    np.testing.assert_allclose(10**ev_to_log10_exposure(np.array([.1,.5,1,2])),2**np.array([.1,.5,1,2]),atol=1e-15)
    assert not np.isclose(ev_to_log10_exposure(.5),.5)


def test_monotone_function_has_path_dependent_allocation_but_fixed_total():
    x=nonlinear_counterexample([1,1,1],[2,2,2])
    assert x['endpoint_change']==4
    contributions=np.array([p['contributions'] for p in x['paths']])
    np.testing.assert_allclose(contributions.sum(-1),4)
    assert np.ptp(contributions[:,0])==1
    np.testing.assert_allclose(x['shapley'],[1.5,1.5,1])
    for i,r in enumerate(x['marginal_ranges']):assert r['minimum']<=x['shapley'][i]<=r['maximum']


def test_mixed_sign_endpoints_keep_same_total_and_shapley_efficiency():
    x=nonlinear_counterexample([1,2,1],[2,1,.5])
    np.testing.assert_allclose([p['total'] for p in x['paths']],-.5)
    np.testing.assert_allclose(x['shapley'],[1.5,-1.5,-.5])
    np.testing.assert_allclose(sum(x['shapley']),x['endpoint_change'])
