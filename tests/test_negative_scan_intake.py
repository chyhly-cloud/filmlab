import sys
from pathlib import Path
import numpy as np
import tifffile
from PIL import ImageCms

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/eval'))
from negative_scan_intake import inspect_tiff


def test_uint16_rgb_does_not_authorize_density_or_claim_linearity(tmp_path):
    p = tmp_path / 'synthetic16.tif'
    a = np.zeros((4, 5, 3), dtype=np.uint16)
    a[0, 0] = 65535
    tifffile.imwrite(p, a, photometric='rgb', resolution=(1408, 1408), resolutionunit='INCH')
    r = inspect_tiff(p)
    assert r['single_page_RGB_uint16_container']
    assert not r['linear_encoding_verified'] and not r['density_analysis_ready']
    assert r['raw_code_statistics'][0]['code_max_fraction'] == .05
    assert r['metadata']['XResolution'] == (1408, 1)
    assert np.array_equal(tifffile.imread(p), a)


def test_eight_bit_and_grayscale_not_accepted_as_rgb16(tmp_path):
    for name, a, photo in [('rgb8', np.zeros((4, 5, 3), np.uint8), 'rgb'),
                           ('gray16', np.zeros((4, 5), np.uint16), 'minisblack')]:
        p = tmp_path / (name + '.tif')
        tifffile.imwrite(p, a, photometric=photo)
        assert not inspect_tiff(p)['single_page_RGB_uint16_container']


def test_gamma_table_metadata_is_recorded_without_linear_assumption(tmp_path):
    p = tmp_path / 'transfer_table.tif'
    lut = np.arange(256, dtype=np.uint16) * 257
    tifffile.imwrite(p, np.zeros((4, 5, 3), np.uint8), photometric='rgb',
                     extratags=[(301, 'H', 768, tuple(np.tile(lut, 3)), False)])
    r = inspect_tiff(p)
    assert r['metadata']['TransferFunction']['present']
    assert not r['linear_encoding_verified']


def test_srgb_profile_records_declared_encoding_without_certifying_capture(tmp_path):
    p = tmp_path / 'tagged16.tif'
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    tifffile.imwrite(p, np.ones((4, 5, 3), np.uint16), photometric='rgb',
                     extratags=[(34675, 'B', len(profile), profile, False)])
    r = inspect_tiff(p)
    declared = r['metadata']['InterColorProfile']['declared_profile']
    assert 'sRGB' in declared['description']
    assert len(declared['declared_transfer_curves']) == 3
    assert not declared['actual_scan_encoding_verified']
    assert not r['density_analysis_ready']
