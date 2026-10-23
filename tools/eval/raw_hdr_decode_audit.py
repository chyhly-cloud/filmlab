"""Known digital CFA probes for a reversible HDR RAW decode contract.

No production code changes, scene fitting, denoising or highlight inference.
Four low probes define each fixed affine matrix; declared high probes and an
analytic cube corner test it. Gain is calibrated only from low probes.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

import numpy as np
import rawpy

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from filmlab.rawio import XYZ_TO_SRGB

CALIBRATION = [[.0625, .0625, .0625], [.125, .0625, .0625],
               [.0625, .125, .0625], [.0625, .0625, .125]]
VALIDATION = [[.125, .125, .125], [.8, .2, .2], [.2, .8, .2], [.8, .8, .8]]
MODES = [('clip_1', rawpy.HighlightMode.Clip, 1.),
         ('ignore_1', rawpy.HighlightMode.Ignore, 1.),
         ('clip_eighth', rawpy.HighlightMode.Clip, .125),
         ('ignore_eighth', rawpy.HighlightMode.Ignore, .125)]


def decode_probe(path, levels, mode, bright, freeze_white=False, neutral_camera=False):
    with rawpy.imread(str(path)) as raw:
        data = raw.raw_image_visible
        colors = raw.raw_colors_visible
        black = np.asarray(raw.black_level_per_channel, float)
        white = float(raw.white_level)
        desc = raw.color_desc.decode()
        actual = []
        maximum_code = 0
        for name, signal in zip('RGB', levels):
            signals = []
            for c, ch in enumerate(desc):
                if ch != name:
                    continue
                value = round(black[c] + signal * (white - black[c]))
                data[colors == c] = value
                maximum_code = max(maximum_code, value)
                signals.append((value - black[c]) / (white - black[c]))
            actual.append(np.mean(signals))
        wb = list(raw.daylight_whitebalance)
        wb = wb[:3] + [wb[1] if wb[3] <= 0 else wb[3]]
        if neutral_camera:
            wb = [1., 1., 1., 1.]
        extra = {'adjust_maximum_thr': 0} if freeze_white else {}
        xyz = raw.postprocess(half_size=True, no_auto_bright=True, gamma=(1, 1),
                              output_bps=16, output_color=rawpy.ColorSpace.raw if neutral_camera else rawpy.ColorSpace.XYZ,
                              user_wb=wb, use_camera_wb=False, use_auto_wb=False,
                              highlight_mode=mode, bright=bright, **extra)
        # Interior excludes sensor borders, where LibRaw uses special handling.
        interior = xyz[24:-24, 24:-24]
        pixels = interior.reshape(-1, 3)
        med = np.median(pixels, axis=0) / 65535
        return {'requested_camera': list(levels), 'actual_camera': actual,
                'output_median': med.tolist(),
                'output_space': 'neutral camera' if neutral_camera else 'XYZ',
                **({'output_median_camera': med.tolist()} if neutral_camera else {'output_median_XYZ': med.tolist()}),
                'interior_output_min_codes': pixels.min(0).tolist(),
                'interior_output_max_codes': pixels.max(0).tolist(),
                'output_at_65535_channels': (pixels == 65535).any(0).tolist(),
                'maximum_written_CFA_code': maximum_code,
                'declared_white_code': white,
                'all_written_codes_strictly_below_white': bool(maximum_code < white),
                'daylight_WB': wb,
                'black': black.tolist()}


def fit_affine(rows):
    X = np.column_stack([[r['actual_camera'] for r in rows], np.ones(len(rows))])
    Y = np.asarray([r['output_median'] for r in rows])
    coefficient = np.linalg.solve(X, Y)
    return coefficient[:3], coefficient[3]


def add_prediction(row, matrix, offset, space='XYZ'):
    pred = np.asarray(row['actual_camera']) @ matrix + offset
    observed = np.asarray(row['output_median'])
    row['low_matrix_prediction_' + space] = pred.tolist()
    row['residual_' + space] = (observed - pred).tolist()
    row['max_absolute_residual_output_codes'] = float(np.max(abs(observed - pred)) * 65535)


def run(path, out, freeze_white=False):
    results = {'scope': __doc__, 'raw': str(path),
               'raw_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
               'rawpy_version': rawpy.__version__, 'libraw_version': list(rawpy.libraw_version),
               'predeclared_calibration': CALIBRATION, 'predeclared_validation': VALIDATION,
               'common': {'half_size': True, 'gamma': [1, 1], 'output': 'XYZ16',
                          'no_auto_bright': True, 'white_balance': 'fixed daylight',
                          'adjust_maximum_thr': 0 if freeze_white else 'LibRaw default (.75)'},
               'contract_tolerance': {'validation_absolute_codes': 64,
                    'meaning': 'Conservative fixed numerical budget for low-step matrix calibration and integer processing, not a physical fit.'},
               'limits': ['Synthetic constant patches are known not saturated by construction; pile-up detectors must not interpret them as a sensor plateau.',
                          'HDR decoding cannot recover genuinely saturated CFA samples.',
                          'Probe-derived gain is numerical normalization, not scene exposure compensation.',
                          'Cube bound is for the measured nonnegative camera domain [0,1]^3, not arbitrary extrapolated radiance.',
                          'The matrix and gain are camera/WB/LibRaw settings specific.'], 'modes': {}}
    for name, mode, bright in MODES:
        calibration = [decode_probe(path, p, mode, bright, freeze_white) for p in CALIBRATION]
        matrix, offset = fit_affine(calibration)
        validation = [decode_probe(path, p, mode, bright, freeze_white) for p in VALIDATION]
        for row in validation:
            add_prediction(row, matrix, offset)
        # For row-vector camera, each XYZ maximum is sum positive coefficients
        # along that column, plus the affine offset.
        cube_max = np.maximum(matrix, 0).sum(0) + offset
        cube_min = np.minimum(matrix, 0).sum(0) + offset
        index = int(np.argmax(cube_max))
        corner = (matrix[:, index] > 0).astype(float)
        # One analytic worst corner only, not a parameter sweep. Exact 1 can be
        # physically saturated; use the largest code still below source white.
        with rawpy.imread(str(path)) as raw:
            black = np.asarray(raw.black_level_per_channel, float)
            white = float(raw.white_level)
            margin = 16 / np.min(white - black)
        safe_corner = np.maximum(corner * (1 - margin), margin)
        corner_row = decode_probe(path, safe_corner.tolist(), mode, bright, freeze_white)
        add_prediction(corner_row, matrix, offset)
        row = {'highlight_mode': mode.name, 'bright': bright, 'calibration': calibration,
               'low_matrix_camera_row_to_XYZ': matrix.tolist(), 'low_affine_offset_XYZ': offset.tolist(),
               'validation': validation, 'analytic_camera_cube_max_XYZ': cube_max.tolist(),
               'analytic_camera_cube_min_XYZ': cube_min.tolist(),
               'analytic_maximum_corner': corner.tolist(), 'safe_maximum_corner_probe': corner_row,
               'all_high_tests_within_numerical_budget': all(r['max_absolute_residual_output_codes'] <= 64 for r in validation + [corner_row]),
               'all_high_tests_below_output_ceiling': not any(any(r['output_at_65535_channels']) for r in validation + [corner_row]),
               'analytic_whole_cube_below_XYZ16_ceiling': bool(np.max(cube_max) < 1)}
        results['modes'][name] = row
        print(name, 'cube maxima', cube_max, 'residual codes', [round(r['max_absolute_residual_output_codes'], 2) for r in validation], flush=True)
    base = np.asarray(results['modes']['clip_1']['low_matrix_camera_row_to_XYZ'])
    for name, row in results['modes'].items():
        matrix = np.asarray(row['low_matrix_camera_row_to_XYZ'])
        gain = float(np.sum(matrix * base) / np.sum(matrix * matrix))
        supported = abs(base) > .01
        ratios = base[supported] / matrix[supported]
        row['calibrated_low_matrix_gain_to_clip_1'] = gain
        row['coefficient_gain_min_max'] = [float(ratios.min()), float(ratios.max())]
        row['coefficient_max_relative_gain_deviation'] = float(np.max(abs(ratios / gain - 1)))
        row['gain_per_XYZ_output_column'] = (np.sum(matrix * base, axis=0) / np.sum(matrix * matrix, axis=0)).tolist()
        row['gain_per_camera_input_row'] = (np.sum(matrix * base, axis=1) / np.sum(matrix * matrix, axis=1)).tolist()
        row['lifted_low_matrix_camera_to_XYZ'] = (gain * matrix).tolist()
        row['lifted_low_matrix_camera_to_linear_RGB'] = (gain * matrix @ XYZ_TO_SRGB.T).tolist()
        row['cube_max_XYZ_after_float_gain'] = (gain * np.asarray(row['analytic_camera_cube_max_XYZ'])).tolist()
    (out / 'results.json').write_text(json.dumps(results, indent=2, allow_nan=False) + '\n')
    return results


def camera_signature(raw):
    return {'black': list(raw.black_level_per_channel), 'white': int(raw.white_level),
            'color_desc': raw.color_desc.decode(), 'raw_pattern': raw.raw_pattern.tolist(),
            'rgb_xyz_matrix': raw.rgb_xyz_matrix.tolist()}


def decode_hdr_float(path, calibration):
    """Eval-only explicit opt-in: neutral camera16 transport -> fixed float XYZ.

    Calibration must be established with digital probes for this camera/WB.
    No denoising, reconstruction, tone curve, output clamp or metering is applied.
    Saturation masks must still be obtained from RAW before calling this helper.
    """
    with rawpy.imread(str(path)) as raw:
        if camera_signature(raw) != calibration['camera_signature']:
            raise ValueError('Digital camera normalization/matrix calibration does not match this RAW')
        if not np.allclose(raw.daylight_whitebalance[:3], calibration['colour_calibration_daylight_WB'][:3], rtol=1e-6):
            raise ValueError('Digital colour calibration WB does not match this RAW')
        cam16 = raw.postprocess(half_size=True, no_auto_bright=True, gamma=(1, 1),
                               output_bps=16, output_color=rawpy.ColorSpace.raw,
                               user_wb=[1, 1, 1, 1], use_camera_wb=False, use_auto_wb=False,
                               highlight_mode=rawpy.HighlightMode.Ignore, bright=1., adjust_maximum_thr=0)
    matrix = np.asarray(calibration['low_camera_affine_matrix'])
    offset = np.asarray(calibration['low_camera_affine_offset'])
    camera = (cam16.astype(float) / 65535 - offset) @ np.linalg.inv(matrix)
    xyz = camera @ np.asarray(calibration['fixed_float_camera_to_XYZ']) + calibration['fixed_float_XYZ_offset']
    return {'camera': camera, 'XYZ': xyz, 'linear_RGB': xyz @ XYZ_TO_SRGB.T}


def actual_lamp_controls(path, calibration, directory, out):
    decoded = decode_hdr_float(path, calibration)
    (out / 'actual').mkdir(parents=True, exist_ok=True)
    with rawpy.imread(str(path)) as raw:
        source_max = int(raw.raw_image_visible.max())
        source_white = int(raw.white_level)
        wb = list(raw.daylight_whitebalance)
        wb = wb[:3] + [wb[1] if wb[3] <= 0 else wb[3]]
        old = raw.postprocess(half_size=True, no_auto_bright=True, gamma=(1, 1),
                             output_bps=16, output_color=rawpy.ColorSpace.XYZ,
                             user_wb=wb, use_camera_wb=False, use_auto_wb=False).astype(float) / 65535
    matrix = np.asarray(calibration['fixed_float_camera_to_XYZ'])
    offset = np.asarray(calibration['fixed_float_XYZ_offset'])
    rows = []
    for name in ('ceiling_lamp', 'middle_ceiling'):
        with np.load(directory / (name + '.npz')) as a:
            x0, y0, x1, y1 = a['source_bounds']
            sl = (slice(y0, y1), slice(x0, x1))
            camera = a['camera']
            measured = ~a['saturation'].any(-1)
            body = a['body_mask'] & measured
            reference = camera @ matrix + offset
            low = measured & np.isfinite(camera).all(-1) & (camera.min(-1) > .005) & (camera.max(-1) < .1) & (reference.min(-1) > 0)
            nrgb = decoded['linear_RGB'][sl]
            np.savez_compressed(out / 'actual' / (name + '.npz'), hdr_rgb=nrgb, source_bounds=a['source_bounds'])
            old_camera_proxy = (old[sl] - offset) @ np.linalg.inv(matrix)
            row = {'name': name, 'only_fully_measured_body_cells': int(body.sum()),
                   'old_XYZ16_any_channel_ceiling_cells': int((body & (old[sl] >= 1).any(-1)).sum()),
                   'source_measured_camera_red_range': [float(camera[..., 0][body].min()), float(camera[..., 0][body].max())],
                   'old_inverse_XYZ_camera_red_proxy_range': [float(old_camera_proxy[..., 0][body].min()), float(old_camera_proxy[..., 0][body].max())],
                   'new_camera_red_range': [float(decoded['camera'][sl][..., 0][body].min()), float(decoded['camera'][sl][..., 0][body].max())],
                   'new_float_RGB_body_peak': float(nrgb[body].max()),
                   'old_RGB_body_peak': float((old[sl] @ XYZ_TO_SRGB.T)[body].max()),
                   'groups': {}}
            for label, use in [('fully_measured_body', body), ('low_signal_measured_control', low)]:
                row['groups'][label] = {'cells': int(use.sum()),
                    'old_XYZ_vs_Bayer_float_median_abs': float(np.median(abs(old[sl][use] - reference[use]))) if use.any() else None,
                    'old_XYZ_vs_Bayer_float_max_abs': float(np.max(abs(old[sl][use] - reference[use]))) if use.any() else None,
                    'new_XYZ_vs_Bayer_float_median_abs': float(np.median(abs(decoded['XYZ'][sl][use] - reference[use]))) if use.any() else None,
                    'new_XYZ_vs_Bayer_float_max_abs': float(np.max(abs(decoded['XYZ'][sl][use] - reference[use]))) if use.any() else None,
                    'new_camera_vs_Bayer_max_abs': float(np.max(abs(decoded['camera'][sl][use] - camera[use]))) if use.any() else None}
            rows.append(row)
    report = {'scope': 'Numeric decode comparison only; no film rendering or parameter selection.',
              'source_raw_max_code': source_max, 'source_raw_declared_white': source_white,
              'fully_saturated_RAW_not_evaluated': True, 'regions': rows}
    (out / 'actual-lamp-controls.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def run_neutral_camera_contract(path, out, baseline_path, minimal=False):
    """Keep signed colour transforms in float, outside XYZ16 output clipping."""
    calibration = [decode_probe(path, p, rawpy.HighlightMode.Ignore, 1., True, True) for p in CALIBRATION]
    matrix, offset = fit_affine(calibration)
    validation = [VALIDATION[1], VALIDATION[3]] if minimal else VALIDATION
    tests = [decode_probe(path, p, rawpy.HighlightMode.Ignore, 1., True, True) for p in validation]
    # Independent, predeclared signed-XYZ edge and near-white camera corner.
    with rawpy.imread(str(path)) as raw:
        margin = 16 / np.min(float(raw.white_level) - np.asarray(raw.black_level_per_channel, float))
        signature = camera_signature(raw)
    for p in ([] if minimal else ([1 - margin, 1 - margin, margin], [1 - margin] * 3)):
        tests.append(decode_probe(path, p, rawpy.HighlightMode.Ignore, 1., True, True))
    for row in tests:
        add_prediction(row, matrix, offset, 'camera')
    baseline = json.loads(baseline_path.read_text())
    if baseline['raw'] != str(path):
        colour_cal = [decode_probe(path, p, rawpy.HighlightMode.Clip, 1., True) for p in CALIBRATION]
        colour_matrix, colour_offset = fit_affine(colour_cal)
        baseline = {'raw': str(path), 'modes': {'clip_1': {
            'low_matrix_camera_row_to_XYZ': colour_matrix.tolist(),
            'low_affine_offset_XYZ': colour_offset.tolist(), 'calibration': colour_cal}}}
        baseline_path = out / 'camera-specific-low-colour-baseline.json'
        baseline_path.write_text(json.dumps(baseline, indent=2, allow_nan=False) + '\n')
    to_xyz = np.asarray(baseline['modes']['clip_1']['low_matrix_camera_row_to_XYZ'])
    xyz_offset = np.asarray(baseline['modes']['clip_1']['low_affine_offset_XYZ'])
    for row in tests:
        measured = (np.asarray(row['output_median']) - offset) @ np.linalg.inv(matrix)
        xyz = measured @ to_xyz + xyz_offset
        row['inverted_input_camera'] = measured.tolist()
        row['input_camera_max_absolute_error'] = float(np.max(abs(measured - row['actual_camera'])))
        row['signed_float_XYZ'] = xyz.tolist()
        row['signed_float_linear_RGB'] = (xyz @ XYZ_TO_SRGB.T).tolist()
    report = {'scope': 'Neutral WB camera16 numeric transport followed by fixed float transform; no production changes.',
              'raw': str(path), 'settings': {'highlight_mode': 'Ignore', 'bright': 1.,
                  'adjust_maximum_thr': 0, 'user_wb': [1, 1, 1, 1], 'output_color': 'raw',
                  'gamma': [1, 1], 'output_bps': 16, 'half_size': True, 'no_auto_bright': True},
              'calibration': calibration, 'low_camera_affine_matrix': matrix.tolist(),
              'camera_signature': signature,
              'low_camera_affine_offset': offset.tolist(), 'validation': tests,
              'baseline_colour_matrix_source': str(baseline_path),
              'colour_calibration_daylight_WB': baseline['modes']['clip_1']['calibration'][0]['daylight_WB'],
              'fixed_float_camera_to_XYZ': to_xyz.tolist(), 'fixed_float_XYZ_offset': xyz_offset.tolist(),
              'all_tests_within_64_output_codes': all(r['max_absolute_residual_output_codes'] <= 64 for r in tests),
              'all_tests_below_output_ceiling': not any(any(r['output_at_65535_channels']) for r in tests),
              'limit': 'Does not infer saturated sensor values; camera16 quantization remains. Negative float XYZ is preserved, not declared physical radiance.'}
    (out / 'neutral-camera.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('neutral camera', report['all_tests_within_64_output_codes'], [r['max_absolute_residual_output_codes'] for r in tests], flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw', type=Path, default=Path('/workspace/filmlab-data/epfl/data/digital/P1480512.RW2'))
    p.add_argument('--out', type=Path, default=Path('/workspace/filmlab-audit/raw-hdr-decode'))
    p.add_argument('--freeze-white', action='store_true', help='Disable LibRaw scene-maximum white-level adjustment.')
    p.add_argument('--neutral-camera-only', action='store_true')
    p.add_argument('--baseline', type=Path, default=Path('/workspace/filmlab-audit/raw-hdr-decode-fixed-white/results.json'))
    p.add_argument('--minimal-validation', action='store_true', help='Only red-strong and gray-high validation; for cross-camera control.')
    p.add_argument('--real-lamp-controls', action='store_true')
    args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    if args.neutral_camera_only:
        report = run_neutral_camera_contract(args.raw, args.out, args.baseline, args.minimal_validation)
        if args.real_lamp_controls:
            actual_lamp_controls(args.raw, report, Path('/workspace/filmlab-audit/lamp-edge-input-audit'), args.out)
    else:
        run(args.raw, args.out, args.freeze_white)
    print('seconds', time.time() - start, flush=True)


if __name__ == '__main__':
    main()
