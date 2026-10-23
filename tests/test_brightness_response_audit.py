"""Protect the brightness-domain and stage-attribution audit, not image aesthetics."""
import numpy as np
from tools.eval.brightness_response_audit import brightness_bin, conversion_info, lab


def test_bins_use_linear_luminance_and_keep_invalid_pixels_out():
    from filmlab.spectral import srgb_encode
    gray=np.array([0,.004,.006,.049,.051,.39,.41,.84,.86,1.])
    rgb=np.repeat(gray[:,None],3,axis=-1)
    np.testing.assert_array_equal(brightness_bin(rgb),[0,0,1,1,2,2,3,3,4,4])
    assert brightness_bin(np.full(3,np.nan)) == -1
    # This catches feeding encoded scan values to the linear classifier.
    assert brightness_bin(np.full(3,.04)) != brightness_bin(srgb_encode(np.full(3,.04)))
    assert abs(lab(np.full(3,.18))[0]-49.4961076)<1e-5


def test_density_conversion_matrix_closes_before_brightness_attribution():
    from filmlab import generation
    from filmlab.engine import Engine,Params
    from filmlab.recipes import recipe
    previous=generation.active()
    try:
        generation.use('next')
        scanner=Engine(Params(**recipe('cinestill_800t'))).scanner
        matrix=np.array(conversion_info(scanner)['combined_matrix'])
        d=np.array([[.2,.4,.8],[1.2,.1,.3]])
        delta=np.array([[.03,.01,.002],[.001,.01,.05]])
        np.testing.assert_allclose(scanner.conversion(d+delta)-scanner.conversion(d),delta@matrix,atol=1e-14)
    finally:
        generation.use(previous)
