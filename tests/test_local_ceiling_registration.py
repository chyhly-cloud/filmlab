import numpy as np
from scipy.ndimage import gaussian_filter,shift
from skimage.transform import AffineTransform

from tools.eval.local_ceiling_registration import structure,patch_match,estimate


def test_dark_ridge_structure_is_invariant_to_positive_scalar_exposure():
    rng=np.random.default_rng(8)
    rgb=np.exp(gaussian_filter(rng.normal(size=(80,80,3)),1))*.1
    np.testing.assert_allclose(structure(rgb),structure(rgb*4),atol=2e-15)


def test_patch_matching_direction_uses_source_to_target_coordinates():
    rng=np.random.default_rng(9)
    raw=gaussian_filter(rng.normal(size=(128,128)),1)
    target=shift(raw,(-2,3),order=1)
    point,score,_=patch_match(raw,target,[64,64])
    np.testing.assert_array_equal(point,[62,67])
    assert score>.99


def test_validation_cannot_change_fitted_affine():
    xx,yy=np.meshgrid([50.,180.,330.,460.],[50.,180.,330.,460.])
    points=np.c_[xx.ravel(),yy.ravel()]
    train=((points[:,0]//128+points[:,1]//128).astype(int)%2)==0
    truth=AffineTransform(scale=(1.003,.997),rotation=.002,translation=(5.,-3.))
    target=truth(points)
    model,record=estimate(points,target,train)
    assert record['adopt_for_diagnostic']
    np.testing.assert_allclose(model.params,truth.params,atol=1e-12)
    corrupted=target.copy();corrupted[~train]+=np.array([7.,-7.])
    other,rejected=estimate(points,corrupted,train)
    np.testing.assert_allclose(other.params,model.params,atol=1e-12)
    assert not rejected['adopt_for_diagnostic']
