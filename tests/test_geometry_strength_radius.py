import numpy as np
import pytest

from tools.eval.geometry_strength_radius import ray_bounds


def test_ray_bounds_enforce_camera_floor_and_the_same_numeric_coordinate_box():
    m=np.array([2.,4.,0.])
    scale=np.array([1.,1.,3.])
    camera=np.array([4.,8.,2.])
    required=np.array([2.,2.,2.])
    low,high=ray_bounds(m,scale,camera,required,upper_z=16.)
    assert low==pytest.approx(1.)
    assert high==pytest.approx(4.)
    assert np.all(low*camera>=required)
    assert np.max(high*m/scale)==pytest.approx(16.)


def test_ray_bounds_reject_infeasible_camera_floor():
    with pytest.raises(ValueError):
        ray_bounds(np.ones(3),np.ones(3),np.ones(3),np.full(3,20.),upper_z=16.)
