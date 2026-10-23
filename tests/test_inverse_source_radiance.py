import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
from tools.eval.inverse_source_radiance import shared_spectral_response,replace_source,sector_split,strict_full_cfa,observation_operator


def test_one_nonnegative_spectrum_must_drive_both_exposure_paths_additively():
    spectra=np.array([[1.,0.,2.,0.],[0.,3.,0.,1.]])
    direct=np.array([[1.,2.,0.],[2.,1.,0.],[0.,0.,4.],[1.,1.,1.]])
    returned=np.array([[.01,0.,0.],[0.,.02,0.],[.03,0.,.01],[0.,0.,.04]])
    a=shared_spectral_response([2.,0.],spectra,direct,returned)
    b=shared_spectral_response([0.,3.],spectra,direct,returned)
    combined=shared_spectral_response([2.,3.],spectra,direct,returned)
    np.testing.assert_allclose(combined[0],a[0]+b[0])
    np.testing.assert_allclose(combined[1],a[1]+b[1])
    # The two endpoint vectors cannot choose independent coefficients.
    np.testing.assert_allclose(combined[0],np.array([2.,9.,4.,3.])@direct)
    np.testing.assert_allclose(combined[1],np.array([2.,9.,4.,3.])@returned)
    with pytest.raises(ValueError):shared_spectral_response([-1.,1.],spectra,direct,returned)


def test_source_subtract_add_closes_in_linear_exposure_and_preserves_background():
    background=np.array([[[1.,2.,3.]]]);basis=np.array([[[[.1,.2,.3]]],[[[.3,.2,.1]]]])
    c=np.array([2.,4.]);total=replace_source(background,basis,c)
    np.testing.assert_allclose(total-background,np.array([[[1.4,1.2,1.]]]))
    np.testing.assert_array_equal(replace_source(background,basis,[0.,0.]),background)
    np.testing.assert_allclose(total-(total-background),background)
    with pytest.raises(ValueError):replace_source(background,basis,[-1.,1.])


def test_spatial_split_holds_out_sectors_and_outer_radius_without_target_selection():
    yy,xx=np.indices((121,121));distance=np.hypot(yy-60,xx-60)*10
    train,test,outer=sector_split(distance,np.ones((121,121),bool),(60,60))
    assert train.any() and test.any() and outer.any()
    assert not np.any(train&test) and not np.any((train|test)&outer)
    assert np.all((distance[train]>=50)&(distance[train]<=300))
    assert np.all(distance[outer]>300)


def test_strict_cfa_support_requires_both_green_samples_to_be_flagged():
    class Raw:pass
    raw=Raw();raw.black_level_per_channel=[0]*4
    raw.raw_colors_visible=np.tile([[0,1],[3,2]],(60,60))
    raw.raw_image_visible=np.random.default_rng(4).integers(0,2000,(120,120))
    raw.raw_image_visible[80:]=4095
    raw.raw_image_visible[100,101]=1000  # one measured green remains in this 2x2 cell
    raw.sizes=type('Sizes',(),{'flip':0})()
    strict,threshold=strict_full_cfa(raw)
    assert np.isfinite(threshold).all()
    assert strict[49,49] and not strict[50,50]


def test_composed_observation_preserves_constant_and_matches_gaussian_bilinear_field():
    from skimage.transform import warp,ProjectiveTransform
    shape=(30,30);matrix=np.array([[4.,0.,180.],[0.,4.,180.],[0.,0.,1.]])
    points=np.array([[220,220],[225,230],[240,240]]);sigma=2.
    P=observation_operator(shape,matrix,points,sigma)
    np.testing.assert_allclose(P@np.ones(np.prod(shape)),1.,atol=1e-12)
    field=np.random.default_rng(11).uniform(size=shape)
    mapped=warp(field,ProjectiveTransform(matrix).inverse,output_shape=(512,512),order=1,cval=np.nan)
    reference=gaussian_filter(mapped,sigma)[tuple(points.T)]
    np.testing.assert_allclose(P@field.ravel(),reference,atol=1e-12)


def test_sector_aperture_guard_keeps_training_and_heldout_target_footprints_disjoint():
    yy,xx=np.indices((121,121));distance=np.hypot(yy-60,xx-60)*6
    train,test,_=sector_split(distance,np.ones((121,121),bool),(60,60),aperture_radius_pixels=10)
    assert train.any() and test.any()
    train_footprint=__import__('scipy').ndimage.maximum_filter(train,size=21)
    test_footprint=__import__('scipy').ndimage.maximum_filter(test,size=21)
    assert not np.any(train_footprint&test_footprint)
