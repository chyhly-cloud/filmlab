import numpy as np
from tools.eval.local_background_contrast import relative_logs,contrast_lab,summary,circular_average


def test_channel_ratio_contrast_cancels_independent_global_linear_gains():
    rgb=np.random.default_rng(4).uniform(.05,.4,(12,12,3));bg=np.zeros((12,12),bool);bg[:3,:3]=True
    first=relative_logs(rgb,bg);second=relative_logs(rgb*np.array([.8,1.2,.9]),bg)
    np.testing.assert_allclose(first[...,1:],second[...,1:],atol=2e-15)
    np.testing.assert_allclose(first[...,0],relative_logs(rgb*1.3,bg)[...,0],atol=2e-15)


def test_lab_subtraction_cancels_additive_offsets_only():
    lab=np.random.default_rng(5).normal(size=(8,8,3));bg=np.zeros((8,8),bool);bg[:2,:2]=True
    np.testing.assert_allclose(contrast_lab(lab,bg),contrast_lab(lab+[3.,10.,-6.],bg),atol=3e-15)


def test_clipped_reference_does_not_gain_an_invented_red_floor():
    rgb=np.full((8,8,3),.2);bg=np.zeros((8,8),bool);bg[:2,:2]=True;rgb[:2,:2,0]=0
    result=relative_logs(rgb,bg)
    assert np.isnan(result[...,1]).all()
    assert np.isfinite(result[...,0]).all() and np.isfinite(result[...,2]).all()


def test_neutral_ceiling_response_has_no_defined_hue():
    rgb=np.full((8,8,3),.2);lab=np.zeros_like(rgb);lab[...,0]=50
    bg=np.zeros((8,8),bool);bg[:2,:2]=True;target=~bg
    result=summary(lab,rgb,bg,target)
    assert result['background_hue_deg'] is None
    assert result['circular_mean_hue_shift_deg_when_defined'] is None
    assert result['target_chromatic_fraction']==0


def test_orange_opposite_cyan_does_not_average_wraparound_to_zero():
    angle,length=circular_average(np.array([179.,-179.]))
    assert abs(abs(angle)-180)<1e-12 and length>.99
