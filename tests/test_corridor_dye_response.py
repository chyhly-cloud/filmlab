import numpy as np
from tools.eval.corridor_dye_response import shapley_three,error_stats


def test_shapley_is_exact_for_linear_three_dye_mapping():
    delta=np.array([.2,-.04,-.01]);m=np.array([[1.,2.,3.],[.2,-1.,4.],[2.,.5,-2.]])
    values=np.stack([(delta*np.array([(mask>>c)&1 for c in range(3)]))@m.T for mask in range(8)])
    terms,orders=shapley_three(values)
    np.testing.assert_allclose(terms,m*delta[None,:],atol=1e-15)
    np.testing.assert_allclose(terms.sum(-1),values[7]-values[0],atol=1e-15)
    np.testing.assert_allclose(orders,np.broadcast_to(terms,(6,3,3)),atol=1e-15)


def test_nonlinear_attribution_has_exact_total_but_depends_on_order():
    a=np.array([1.,2.,0.]);b=np.array([3.,4.,0.]);values=[]
    for mask in range(8):
        x=a+(b-a)*np.array([(mask>>i)&1 for i in range(3)])
        values.append([x[0]*x[1]])
    terms,orders=shapley_three(values)
    np.testing.assert_allclose(terms,[[6.,4.,0.]],atol=1e-15)
    np.testing.assert_allclose(terms.sum(-1),np.array(values[7])-values[0],atol=1e-15)
    assert orders[:,0,0].min()==4 and orders[:,0,0].max()==8
    # The baseline local derivative times the finite increment is not exact.
    assert (a[1]*(b[0]-a[0])+a[0]*(b[1]-a[1])) != terms.sum()
