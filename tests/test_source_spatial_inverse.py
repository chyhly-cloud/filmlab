import numpy as np
import pytest
from tools.eval.source_spatial_inverse import spatial_templates,physical_profile,spatial_responses


def example():
    yy,xx=np.indices((21,23));mask=((yy-10)/4)**2+((xx-11)/7)**2<=1
    legacy=np.where(mask,.4+np.exp(-((yy-10)**2+(xx-11)**2)/20),0)
    return mask,legacy


def test_templates_are_positive_mean_one_and_do_not_relocate_known_pixels():
    mask,legacy=example();templates,meta=spatial_templates(mask,legacy)
    assert templates.shape==(5,21,23)
    np.testing.assert_allclose(templates[:,mask].mean(1),1,atol=1e-15)
    assert np.all(templates[:,mask]>0)
    assert np.all(templates[:,~mask]==0)
    np.testing.assert_allclose(templates[0,mask],legacy[mask]/legacy[mask].mean())
    assert meta['principal_axis_yx'][1]>0


def test_nonnegative_coefficients_have_explicit_total_signal_and_stay_on_support():
    mask,legacy=example();templates,_=spatial_templates(mask,legacy);c=np.array([2.,3.,4.,5.,6.])
    total=physical_profile(c,templates)
    assert np.all(total[~mask]==0)
    assert total[mask].mean()==pytest.approx(c.sum())
    np.testing.assert_allclose(physical_profile(c,templates),physical_profile(c/2,templates)*2)
    with pytest.raises(ValueError):physical_profile([-1]*5,templates)
    with pytest.raises(ValueError):physical_profile([np.nan]*5,templates)


def test_long_axis_templates_are_opposite_gradients_not_new_halo_kernels():
    mask,legacy=example();templates,_=spatial_templates(mask,legacy)
    np.testing.assert_allclose((templates[3]+templates[4])/2,templates[1],atol=1e-15)
    yy,xx=np.indices(mask.shape)
    c1=(templates[3]*xx).sum()/templates[3].sum();c2=(templates[4]*xx).sum()/templates[4].sum()
    assert c1>c2


def test_bad_or_nonpositive_source_support_is_rejected():
    with pytest.raises(ValueError):spatial_templates(np.zeros((3,3),bool),np.ones((3,3)))
    with pytest.raises(ValueError):spatial_templates(np.ones((3,3),bool),np.zeros((3,3)))
