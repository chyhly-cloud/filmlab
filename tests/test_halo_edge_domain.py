import numpy as np
import pytest
from filmlab.spectral import srgb_encode
from tools.eval.halo_edge import linear_input, profile, report


def test_same_linear_image_and_encoded_scan_have_identical_profiles():
    image = np.broadcast_to([.6,.4,.2], (40,40,3)).copy()
    distances = np.full((40,40), 175.)
    mask = np.ones((40,40), bool)
    sim = linear_input(image, 'linear_srgb')
    real = linear_input(srgb_encode(image), 'srgb')
    np.testing.assert_allclose(sim, real, atol=2e-15)
    a, b = profile(sim, distances, mask), profile(real, distances, mask)
    np.testing.assert_allclose(a,b,atol=1e-13,equal_nan=True)
    # Inject the historical extra EOTF: this MUST break the identity contract.
    broken = profile(linear_input(image,'srgb'), distances, mask)
    assert abs(broken[3,1]-a[3,1]) > .4


def test_profile_encoding_is_explicit():
    with pytest.raises(ValueError,match='Encoding'):
        linear_input(np.ones((2,2,3)), 'auto')


def test_legacy_cache_cannot_be_reported_as_corrected_measurement():
    with pytest.raises(ValueError,match='Legacy halo_edge cache'):
        report({'old_scene':{'real':[],'sim':[]}})
