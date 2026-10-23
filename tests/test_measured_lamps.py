import numpy as np
from scipy.ndimage import gaussian_filter, shift

from tools.eval.measure_lamp_brackets import align_lamp, pair_stability
from tools.eval.measured_source_chain import select_patches


def test_local_geometry_recovers_translation_with_fixed_exposure_ratio():
    rng=np.random.default_rng(5)
    a=gaussian_filter(rng.uniform(.02,.4,(240,240,3)),(1,1,0))
    # A broad patterned background constrains geometry, not the lamp centre.
    yy,xx=np.mgrid[:240,:240];a*= (1+.35*np.sin(xx/9)+.25*np.cos(yy/11))[...,None]
    a[117:124,117:124]=.9
    b=shift(a,(1.4,-2.2,0),order=1,mode='nearest')/4
    sat=np.zeros(a.shape,bool);sat[117:124,117:124]=True
    result=align_lamp(a,sat,b,np.zeros_like(sat),np.eye(3),4,
                      {'center':[120,120],'radius':12})
    np.testing.assert_allclose(result['geometry']['translation_xy'],[-2.2,1.4],atol=.12)
    assert result['geometry']['pass']


def test_temporal_check_reports_changed_radiance_without_rescaling_it():
    a=np.full((32,32,3),.3);sat=np.zeros(a.shape,bool);mask=np.ones(a.shape[:2],bool)
    report=pair_stability({'camera':a,'sat':sat},{'camera':a*2,'sat':sat},mask)
    for channel in report:
        assert channel['pixels']==1024
        assert channel['median_stops']==1


def test_full_saturated_core_requires_two_agreeing_alternate_exposures(tmp_path):
    def setup(second):
        a=np.ones((4,4,3));mask=np.ones((4,4),bool);sat=np.ones_like(a,bool)
        np.savez(tmp_path/'scene_lamp_measurements.npz',mask=mask,
                 **{'0_camera':a,'0_sat':sat,'1_camera':a*2,'1_safe':mask,
                    '2_camera':a*second,'2_safe':mask})
        return {'region':{'scene':'scene','name':'lamp','kind':'visible_luminaire','center':[2,2],'half':2},
                'views':['primary','short1','short2'],'primary_any_saturated':16,'primary_all_saturated':16,
                'comparisons':[],'alternatives':[{'file':'short1','scale':2,'local_geometry':{'pass':True}},
                                                 {'file':'short2','scale':4,'local_geometry':{'pass':True}}]}
    patches,evidence=select_patches([setup(2.01)],tmp_path)
    assert sum(p.trusted.sum() for p in patches)==16
    assert evidence[0]['replaced_all_saturated_pixels']==16
    patches,evidence=select_patches([setup(4.)],tmp_path)
    assert not patches
    assert evidence[0]['replaced_pixels']==0
