import numpy as np
from tools.eval.halo_residual_direction import vector_metrics,circular_abs_angle


def test_positive_projection_can_still_make_target_residual_worse():
    E=np.array([[1.,0.,0.],[1.,0.,0.],[1.,0.,0.]])
    V=np.array([[3.,0.,0.],[.5,0.,0.],[-.5,0.,0.]])
    m=vector_metrics(E,V,target_floor=0,movement_floor=0)
    np.testing.assert_array_equal(m['towards'],[True,True,False])
    np.testing.assert_array_equal(m['improved'],[False,True,False])
    np.testing.assert_allclose(m['squared_improvement'],2*m['dot']-m['movement']**2,atol=1e-15)


def test_brightness_and_colour_can_move_in_opposite_quality_directions():
    E=np.array([[-2.,10.,5.]])  # No-halo is too bright but lacks a and b.
    V=np.array([[1.,5.,2.]])
    assert not vector_metrics(E[:,:1],V[:,:1],target_floor=0,movement_floor=0)['improved'][0]
    assert vector_metrics(E[:,1:],V[:,1:],target_floor=0,movement_floor=0)['improved'][0]


def test_hue_wrap_and_zero_vectors_do_not_fake_direction():
    np.testing.assert_allclose(circular_abs_angle(np.array([179.,1.]),np.array([-179.,359.])),[2.,2.])
    m=vector_metrics(np.zeros((2,3)),np.array([[1.,0.,0.],[0.,0.,0.]]))
    assert not m['valid'].any() and np.isnan(m['cosine']).all()
