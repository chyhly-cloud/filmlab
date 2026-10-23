"""Local comparisons must retain their own geometry and pixel-centre convention."""
import numpy as np
from skimage.transform import ProjectiveTransform

from tools.eval.light_acceptance import crop_warp, pixel_scaling


def test_two_crops_remain_distinct_after_both_transforms_exist():
    image = np.broadcast_to(np.arange(1024, dtype=float), (600,1024)).copy()
    crops = [crop_warp(ProjectiveTransform(np.array([[1,0,-x],[0,1,0],[0,0,1.]]))) for x in (0,100)]
    assert np.array_equal(crops[0](image), image[:512,:512])
    assert np.array_equal(crops[1](image), image[:512,100:612])


def test_downsampled_pixel_centres_match_the_mean_of_native_samples():
    # Native samples 0 and 1 average at .5, the centre of output sample 0;
    # native samples 2 and 3 average at 2.5, the centre of output sample 1.
    H = pixel_scaling(.5,.5)
    points = np.array([[.5,.5,1],[2.5,2.5,1]])
    assert np.array_equal((points @ H.T)[:,:2], [[0,0],[1,1]])
