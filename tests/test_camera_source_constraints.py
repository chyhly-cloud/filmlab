import numpy as np

from tools.eval.camera_source_constraints import analyse


def fixture_case():
    yy, xx = np.indices((32, 32))
    amplitude = .1 + .01 * (xx + yy)
    shape = np.array([.6, 1., .4])
    camera = amplitude[..., None] * shape
    sat = np.zeros_like(camera, bool)
    core = (xx - 16) ** 2 + (yy - 16) ** 2 <= 14 ** 2
    lamp = {'region': {'scene': 'test', 'name': 'known', 'kind': 'hypothetical',
                       'center': [16, 16], 'half': 16, 'radius': 14},
            'views': ['primary'], 'alternatives': []}
    return lamp, {'0_camera': camera, '0_sat': sat, 'mask': core}


def test_known_fixed_colour_is_preserved_and_full_censoring_stays_unresolved():
    lamp, data = fixture_case()
    original = data['0_camera'].copy()
    data['0_sat'][15:17, 15:17] = True
    data['0_sat'][17:19, 17:19, 1] = True
    row, arrays = analyse(lamp, data)
    assert row['heldout_measured_all_pair_summary']['p90_absolute_EV'] < 1e-12
    assert not arrays['predicted_missing_mask'][15:17, 15:17].any()
    np.testing.assert_array_equal(arrays['predicted_only_missing'][15:17, 15:17], original[15:17, 15:17])
    np.testing.assert_allclose(arrays['predicted_only_missing'], original, atol=1e-15)
    assert row['all_RGB_flag_cells_have_no_measured_amplitude'] == 4


def test_spatial_colour_shift_is_detected_without_training_on_heldout_blocks():
    lamp, data = fixture_case()
    yy, xx = np.indices(data['mask'].shape)
    checker = (xx // 4 + yy // 4) % 2 == 1
    data['0_camera'][..., 0][checker] *= 2
    row, _ = analyse(lamp, data)
    for fold in row['folds']:
        # Each fold's independent R/G prediction must expose the one-stop
        # between-block shift, not fit it away using heldout samples.
        metric = fold['heldout_measured_channel_prediction']['R_from_G']
        assert abs(metric['median_absolute_EV'] - 1) < 1e-12


def test_no_core_colour_support_does_not_borrow_measured_background():
    lamp, data = fixture_case()
    data['0_sat'][data['mask']] = True
    row, arrays = analyse(lamp, data)
    assert all(f['status'].startswith('unsupported') for f in row['folds'])
    assert not arrays['predicted_missing_mask'].any()
