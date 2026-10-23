"""RAW-only test of the fixed-camera-chromaticity extrapolation assumption.

Train exclusively on measured primary RGB inside the pre-existing source/core
mask; never on film, scan, short-exposure hidden truth or surrounding surfaces.
Four-pixel checkerboard blocks separate train and validation. An ROI is not a
verified pure emitter: Bayer sampling, mixing, optics and geometry remain limits.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / 'tools/eval')]
from radiance_censor_holdout import reference_fields


def errors(values):
    a = np.asarray(values, float)
    a = a[np.isfinite(a)]
    return {'count': int(a.size),
            'median_signed_EV': float(np.median(a)) if a.size else None,
            'median_absolute_EV': float(np.median(abs(a))) if a.size else None,
            'p90_absolute_EV': float(np.quantile(abs(a), .9)) if a.size else None,
            'within_quarter_EV_fraction': float(np.mean(abs(a) <= .25)) if a.size else None}


def fixed_shape(camera, train):
    """Equal-pixel robust log colour ratios; no fitted brightness or intercept."""
    ratio = np.log2(camera[train] / camera[train, 1, None])
    return 2 ** np.median(ratio, axis=0)


def all_pair_errors(camera, shape, use):
    a = camera[use]
    return {f'{"RGB"[i]}_from_{"RGB"[j]}':
            errors(np.log2((a[:, j] * shape[i] / shape[j]) / a[:, i]))
            for i in range(3) for j in range(3) if i != j}


def relative_shape_error(camera, shape, use):
    a = camera[use]
    return {'log_R_over_G_error': errors(np.log2(shape[0] / (a[:, 0] / a[:, 1]))),
            'log_B_over_G_error': errors(np.log2(shape[2] / (a[:, 2] / a[:, 1])))}


def analyse(lamp, data):
    camera = data['0_camera'].astype(float)
    sat = data['0_sat'].astype(bool)
    core = data['mask'].astype(bool)
    signal = np.isfinite(camera).all(-1) & (camera.min(-1) > .005)
    safe = ~sat.any(-1) & signal
    yy, xx = np.indices(core.shape)
    # Anchored to native half-size coordinates, not dependent on crop origin.
    half = lamp['region']['half']
    x, y = lamp['region']['center']
    origin = lamp['region'].get('origin_xy', [x - half, y - half])
    xx = xx + origin[0]
    yy = yy + origin[1]
    checker = ((xx // 4 + yy // 4) % 2).astype(bool)
    radius = np.hypot(xx - x, yy - y)
    r0 = lamp['region']['radius']
    # Background is a separate diagnostic, never training support.
    ring = (radius > r0 + 8) & (radius <= r0 + 20) & safe
    truth, support, _, pairs = reference_fields(lamp, data)
    missing = core & sat.any(-1)
    partial = missing & ~sat.all(-1)
    full = core & sat.all(-1)
    row = {'name': lamp['region']['scene'] + '_' + lamp['region']['name'],
           'kind': lamp['region']['kind'], 'radius_halfsize_camera_cells': r0,
           'coverage': {'core_cells': int(core.sum()), 'RGB_measured_core_cells': int((core & safe).sum()),
                        'primary_partial_RGB_flag_cells': int(partial.sum()),
                        'primary_all_RGB_flag_cells': int(full.sum()),
                        'independent_truth_missing_cells': int((missing & support).sum()),
                        'truth_supported_partial_cells': int((partial & support).sum()),
                        'truth_supported_all_RGB_flag_cells': int((full & support).sum())},
           'independent_reference_pairs': pairs, 'folds': []}
    validation = []
    hidden_errors = []
    predicted = camera.copy()
    resolved = np.zeros_like(sat)
    for heldout in (False, True):
        train = core & safe & (checker != heldout)
        test = core & safe & (checker == heldout)
        entry = {'heldout_checker_value': bool(heldout), 'train_core_cells': int(train.sum()),
                 'heldout_measured_core_cells': int(test.sum())}
        if train.sum() < 16 or test.sum() < 16:
            entry['status'] = 'unsupported: fewer than 16 train or heldout measured core cells'
            row['folds'].append(entry)
            continue
        shape = fixed_shape(camera, train)
        entry.update(status='supported', fixed_camera_shape_R_G_B=shape.tolist(),
                     heldout_measured_channel_prediction=all_pair_errors(camera, shape, test),
                     heldout_measured_shape_error=relative_shape_error(camera, shape, test),
                     background_ring_shape_error=relative_shape_error(camera, shape, ring),
                     inner_core_shape_error=relative_shape_error(camera, shape, test & (radius <= .5 * r0)),
                     outer_core_shape_error=relative_shape_error(camera, shape, test & (radius > .5 * r0)))
        for i in range(3):
            for j in range(3):
                if i != j:
                    validation.extend(np.log2((camera[..., j][test] * shape[i] / shape[j]) / camera[..., i][test]))
        active = partial & (checker == heldout)
        available = ~sat & np.isfinite(camera) & (camera > .005)
        amp_channels = np.where(available, camera / shape, np.nan)
        # Only pixels with a genuinely measured channel can determine amplitude.
        active &= available.any(-1)
        amp = np.zeros(core.shape)
        amp[active] = np.nanmedian(amp_channels[active], axis=-1)
        pred = amp[..., None] * shape
        use = active[..., None] & sat
        predicted[use] = pred[use]
        resolved |= use
        # A saturated sample is a lower bound. A prediction below it violates
        # the observation; report this without clamping away the contradiction.
        lower_gap = np.log2(np.maximum(pred[use], 1e-15) / camera[use])
        both = active & (available.sum(-1) >= 2)
        spread = np.log2(np.nanmax(amp_channels[both], axis=-1) /
                         np.nanmin(amp_channels[both], axis=-1))
        entry['partial_channel_censor_constraints'] = {
            'predicted_missing_channels': int(use.sum()),
            'below_recorded_saturation_lower_bound_over_0_05_EV': int((lower_gap < -.05).sum()),
            'below_bound_fraction': float(np.mean(lower_gap < -.05)) if lower_gap.size else None,
            'available_channel_amplitude_spread_EV': errors(spread)}
        scoring = use & support[..., None]
        ev = np.log2(np.maximum(pred[scoring], 1e-15) / truth[scoring])
        hidden_errors.extend(ev)
        entry['independent_original_missing_channel_error'] = errors(ev)
        # Hidden truth can test direction in fully clipped cells, but cannot
        # be used to set their amplitude or train the chromaticity.
        entry['independent_hidden_shape_error'] = relative_shape_error(
            truth, shape, missing & support & (checker == heldout))
        row['folds'].append(entry)
    row['heldout_measured_all_pair_summary'] = errors(validation)
    row['independent_partial_missing_summary'] = errors(hidden_errors)
    row['unresolved_original_missing_channels'] = int((core[..., None] & sat & ~resolved).sum())
    row['all_RGB_flag_cells_have_no_measured_amplitude'] = int(full.sum())
    return row, {'primary_camera': camera, 'primary_saturation': sat, 'core_mask': core,
                 'checker': checker, 'predicted_only_missing': predicted,
                 'predicted_missing_mask': resolved, 'reference_camera': truth,
                 'independent_reference_support': support}


def corridor_controls(directory):
    """Explicit geometry masks from a separate RAW input audit; no scan usage."""
    rows = []
    for name in ('ceiling_lamp', 'middle_ceiling'):
        path = directory / (name + '.npz')
        with np.load(path) as a:
            # Read only the raw measurements and predeclared geometry masks.
            camera, sat = a['camera'], a['saturation']
            bounds = a['source_bounds']
            for key in ('component_mask', 'body_mask'):
                mask = a[key]
                yy, xx = np.where(mask)
                x, y = np.mean(xx) + bounds[0], np.mean(yy) + bounds[1]
                radius = np.max(np.hypot(xx + bounds[0] - x, yy + bounds[1] - y))
                lamp = {'region': {'scene': 'R1-02-3', 'name': name + '_' + key,
                                  'kind': 'conditional_geometry_mask_not_pure_emitter',
                                  'center': [x, y], 'origin_xy': bounds[:2].tolist(),
                                  'half': 0, 'radius': float(radius)},
                        'alternatives': [], 'views': ['primary']}
                row, _ = analyse(lamp, {'0_camera': camera, '0_sat': sat, 'mask': mask})
                row['mask_provenance'] = str(path)
                row['input_cache_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
                row['no_independent_missing_channel_truth'] = True
                row['mask_limit'] = ('Saturated component has no fully measured RGB training.'
                                     if key == 'component_mask' else
                                     'Explicit lamp-strip polygon includes optically mixed boundary; not certified pure emission.')
                rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--measurements', type=Path, default=Path('/workspace/filmlab-audit/measured-sources'))
    parser.add_argument('--out', type=Path, default=Path('/workspace/filmlab-audit/camera-source-constraints'))
    parser.add_argument('--corridor', type=Path, default=Path('/workspace/filmlab-audit/lamp-edge-input-audit'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((args.measurements / 'lamp-measurements.json').read_text())
    report = {'scope': __doc__, 'predeclared_design': {
        'signal_min_camera_units': .005, 'checker_block_size': 4, 'min_train_and_test_cells': 16,
        'chromaticity': 'median log2(R/G), median log2(B/G), G=1',
        'train_region': 'primary measured RGB intersect cache[mask] core only',
        'heldout': 'both checker directions; measured channels predict other measured channels',
        'missing_amplitude': 'median available camera channel / fixed shape',
        'censor_violation_tolerance_EV': .05,
        'forbidden_selection': 'no film/scan truth, no short-exposure truth for fitting, no spectrum-library ranking'},
        'limits': ['Fixed camera chromaticity tests this extrapolation model; it does not prove a fixed emitted spectrum.',
                   'A fixed emitter spectrum plus chromatic spatial transfer can still yield spatially varying camera ratios.',
                   'Core masks are pre-existing source ROIs, not verified pure emitting surfaces.',
                   'Bayer subpixels are not colocated; chromatic PSF, mixed surfaces and registration can break the test.',
                   'Green is the mean of two CFA greens while its saturation flag is an OR; partial-green cells need caution.',
                   'Unknown sensor SPD prevents recovering a unique physical source spectrum.',
                   'All-channel saturated pixels have no measured amplitude, even if source shape is known.',
                   'Empirical errors are not physical uncertainty or credible intervals.'], 'regions': []}
    for lamp in metadata['lamps']:
        name = lamp['region']['scene'] + '_' + lamp['region']['name']
        path = args.measurements / (name + '_measurements.npz')
        with np.load(path) as data:
            row, arrays = analyse(lamp, data)
        row['input_cache_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        report['regions'].append(row)
        if lamp['region']['name'] in ('roof_opening_1', 'roof_opening_2'):
            np.savez_compressed(args.out / (name + '.npz'), **arrays)
        print(name, json.dumps({'measured': row['heldout_measured_all_pair_summary'],
                               'hidden_partial': row['independent_partial_missing_summary']}), flush=True)
    report['corridor_geometry_controls'] = corridor_controls(args.corridor) if args.corridor.exists() else []
    report['corridor_geometry_controls_status'] = 'included' if report['corridor_geometry_controls'] else 'input absent; unsupported'
    (args.out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
