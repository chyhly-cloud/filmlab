import numpy as np
from tools.eval.corridor_cache_geometry import CacheGeometry,bilinear_operator,recover_native


def test_sparse_bilinear_operator_matches_actual_projective_warp():
    shape=(11,13);yy,xx=np.indices(shape)
    field=np.stack([xx+2*yy,np.sin(xx)*np.cos(yy),xx%2+yy%3],axis=-1).astype(float)
    matrix=np.array([[2.7,.13,2.3],[-.08,2.9,1.7],[.0004,-.0008,1.]])
    geo=CacheGeometry(shape,matrix,output_shape=(32,38));a,rows=bilinear_operator(shape,matrix,geo.output_shape)
    actual=geo.warp(field).reshape(-1,3)[rows]
    np.testing.assert_allclose(a@field.reshape(-1,3),actual,atol=2e-13)


def test_inverse_recovers_overdetermined_native_field_with_forward_validation():
    shape=(6,7);matrix=np.array([[4.,0,2.],[0,4.,2.],[0,0,1.]])
    a,rows=bilinear_operator(shape,matrix,(27,31));rng=np.random.default_rng(3)
    field=rng.normal(size=(*shape,3));observed=a@field.reshape(-1,3)
    recovered,active,info=recover_native(a,observed,shape)
    assert active.all()
    np.testing.assert_allclose(recovered,field,atol=1e-10)
    assert max(r['max_reprojection_error'] for r in info)<1e-10


def test_unobserved_pixels_are_nan_and_not_silently_extrapolated():
    shape=(12,12);matrix=np.eye(3)
    a,rows=bilinear_operator(shape,matrix,(4,4));field=np.arange(144.).reshape(shape)
    recovered,active,info=recover_native(a,(a@field.ravel()),shape)
    assert active.sum()==16
    assert np.isnan(recovered[~active]).all()
    np.testing.assert_allclose(recovered[...,0][active],field[active],atol=1e-10)
