from unittest.mock import patch

import numpy as np
import pytest
import rawpy

from tools.eval.raw_hdr_decode_audit import camera_signature, decode_hdr_float


class DigitalRaw:
    black_level_per_channel = [0, 0, 0, 0]
    white_level = 65535
    color_desc = b'RGBG'
    raw_pattern = np.array([[0, 1], [3, 2]])
    rgb_xyz_matrix = np.eye(4, 3)
    daylight_whitebalance = [2., 1., 1.5, 1.]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def postprocess(self, **kw):
        assert kw['output_color'] == rawpy.ColorSpace.raw
        assert kw['highlight_mode'] == rawpy.HighlightMode.Ignore
        assert kw['adjust_maximum_thr'] == 0
        assert kw['bright'] == 1
        assert kw['user_wb'] == [1, 1, 1, 1]
        return np.array([[[52428, 13107, 13107]]], dtype=np.uint16)


def calibration(raw):
    return {'camera_signature': camera_signature(raw),
            'colour_calibration_daylight_WB': raw.daylight_whitebalance,
            'low_camera_affine_matrix': np.eye(3).tolist(), 'low_camera_affine_offset': [0, 0, 0],
            'fixed_float_camera_to_XYZ': [[2, 0, 0], [0, 1, 0], [0, -2, 1]],
            'fixed_float_XYZ_offset': [0, 0, 0]}


def test_optin_helper_preserves_hdr_and_negative_float_colour_values():
    raw = DigitalRaw()
    with patch('rawpy.imread', return_value=raw):
        decoded = decode_hdr_float('digital.raw', calibration(raw))
    np.testing.assert_allclose(decoded['camera'], [[[.8, .2, .2]]])
    np.testing.assert_allclose(decoded['XYZ'], [[[1.6, -.2, .2]]])
    assert decoded['linear_RGB'].max() > 1


def test_helper_rejects_incompatible_low_probe_calibration():
    raw = DigitalRaw()
    c = calibration(raw)
    c['colour_calibration_daylight_WB'] = [1, 1, 1, 1]
    with patch('rawpy.imread', return_value=raw), pytest.raises(ValueError, match='WB'):
        decode_hdr_float('digital.raw', c)
