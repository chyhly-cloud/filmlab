"""Measurement-constrained lamp-edge spectral witnesses, with a fixed backend.

No scan samples choose coefficients. Available camera channels are equalities;
saturated channels are hard lower bounds. Only zero-tolerance feasible cells in
an explicit lamp-component mask are replaced. A ten-spectrum/CIE-decoder proxy
is not measured camera spectral sensitivity or recovered lamp SPD. Three LP
witnesses retain the SAME coefficients for direct and returning exposure.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / 'tools/eval')]
from censored_spectral_constraints import (
    solve_censored_constraints, fixed_spectrum_amplitude_interval,
    minimum_relative_tolerance,
)
from inverse_source_radiance import engine_for, observation_operator
from brightness_response_audit import lab

MODES = ('minimum_input_Y', 'minimum_return_R', 'maximum_return_R')


def transport(engine, direct, returning):
    from filmlab import film
    rho = film.bounce_retention(engine.e.stock, engine.v)
    return (film.emulsion_scatter(direct, engine.px_um)
            + film.halation(np.zeros_like(returning), returning,
                            engine.px_um, engine.v, rho))


def run(name, edge_root, root, out):
    import colour
    from PIL import Image, ImageDraw, ImageFont
    from skimage.transform import ProjectiveTransform
    from light_acceptance import crop_warp
    from filmlab.spectral import srgb_encode

    out.mkdir(parents=True, exist_ok=True)
    source = np.load(root / (name + '_source.npz'))
    previous = np.load(root / (name + '_fit.npz'))
    edge = np.load(edge_root / (name + '.npz'))
    prep = json.loads((root / 'prepared.json').read_text())
    ref = json.loads((root / (name + '_fit.json')).read_text())
    engine, meta = engine_for()
    camera_response = source['camera_response']
    direct_response, return_response = source['direct_response'], source['return_response']
    shape = source['support'].shape
    assert edge['camera'].shape == (*shape, 3)
    assert np.array_equal(edge['strict_support'], source['support'])
    # Membership is fixed by the RAW saturation component, not target colour.
    requested = edge['component_mask'].astype(bool) & ~source['support']
    saturation = edge['saturation'].astype(bool)
    if np.any(requested & saturation.all(-1)):
        # Two-green ANY flags can leave one measured CFA sample. The half-size
        # RGB proxy still has no available channel: do not invent a finite SPD.
        requested &= ~saturation.all(-1)
    fields = {mode: np.zeros((*shape, len(camera_response))) for mode in MODES}
    accepted = np.zeros(shape, bool)
    records = []
    rtol_values = [0., 2**.05-1, 2**.15-1]
    fl2 = prep['names'].index('FL2')
    responses = {'input_Y': np.ones(len(camera_response))}
    for c, label in enumerate('RGB'):
        responses['direct_' + label] = direct_response[:, c]
        responses['return_' + label] = return_response[:, c]
    for y, x in np.argwhere(requested):
        observed, sat = edge['camera'][y, x], saturation[y, x]
        solved = solve_censored_constraints(camera_response, observed, sat,
                                            responses=responses)
        consistency = minimum_relative_tolerance(camera_response, observed, sat)
        fixed = [fixed_spectrum_amplitude_interval(
            camera_response[fl2], observed, sat, relative_tolerance=t)
            for t in rtol_values]
        row = {'yx': [int(y), int(x)], 'observed_camera': observed.tolist(),
               'saturated': sat.tolist(), 'exact_proxy_feasible': solved['feasible'],
               'minimum_normalized_available_residual': float(consistency['minimum_relative_tolerance']),
               'fixed_FL2_feasible_at_declared_tolerances': [v['feasible'] for v in fixed]}
        if solved['feasible']:
            accepted[y, x] = True
            bounds = solved['response_bounds']
            for mode, key, side in [('minimum_input_Y', 'input_Y', 'minimum'),
                                    ('minimum_return_R', 'return_R', 'minimum'),
                                    ('maximum_return_R', 'return_R', 'maximum')]:
                coefficients = bounds[key][side + '_coefficients']
                if coefficients is None:
                    raise ValueError('Available-channel edge unexpectedly lacks a finite witness')
                prediction = coefficients @ camera_response
                if (np.max(abs(prediction[~sat] - observed[~sat]), initial=0.) > 1e-7
                        or np.min(prediction[sat] - observed[sat], initial=0.) < -1e-7):
                    raise ValueError('LP witness violates declared exact channel constraints')
                fields[mode][y, x] = coefficients
            row['linear_response_bounds'] = {
                key: {side: float(value[side]) for side in ['minimum', 'maximum']}
                for key, value in bounds.items()}
        records.append(row)
    if not accepted.any():
        raise ValueError('No zero-tolerance feasible edge cells; report proxy failure before rendering')
    old_direct = edge['old_direct'] * accepted[..., None]
    old_return = edge['old_halo_source'] * accepted[..., None]
    removed = transport(engine, old_direct, old_return)
    core_H = source['background'] + np.einsum(
        'k,khwc->hwc', previous['coefficients'], source['basis'], optimize=True)
    background = core_H - removed
    if background[source['safe']].min() < -1e-7:
        raise ValueError('Edge subtraction does not preserve a nonnegative background')
    points = previous['points']
    P = observation_operator(shape, source['matrix'], points, 25 / meta['scan_um_per_pixel'])
    warp = crop_warp(ProjectiveTransform(source['matrix']))

    def render(H):
        q = engine._log_exposure(np.maximum(H, 1e-12))
        state = engine._develop(q, blur_px=12 / engine.px_um)
        state.dye *= source['factor'][..., None]
        rgb = engine.scanner.to_rgb(engine.scanner.normalized_density(engine._transmittance(state)))
        return warp(rgb), lab(P @ rgb.reshape(-1, 3)), q

    baseline_image, baseline_lab, _ = render(source['total_local'])
    core_image, core_lab, _ = render(core_H)
    replay = {'old_baseline_Lab': float(np.max(abs(baseline_lab - previous['baseline_lab']))),
              'core_inverse_Lab': float(np.max(abs(core_lab - previous['fitted_lab'])))}
    if max(replay.values()) > 2e-4:
        raise ValueError('Existing fixed-state replay failed: ' + str(replay))
    nt, nv = ref['baseline']['train']['samples'], ref['baseline']['heldout']['samples']
    regions = {'original_train_diagnostic_only': slice(0, nt),
               'original_inner_heldout': slice(nt, nt+nv), 'original_outer': slice(nt+nv, None)}
    target = previous['target_lab']

    def stats(prediction):
        return {label: {'samples': len(prediction[s]),
                        'Lab_RMSE': float(np.sqrt(np.mean((prediction[s]-target[s])**2))),
                        'median_deltaE00': float(np.median(colour.delta_E(prediction[s], target[s], method='CIE 2000'))),
                        'mean_model_minus_real_Lab': (prediction[s]-target[s]).mean(0).tolist()}
                for label, s in regions.items()}

    images = {'real': source['real'], 'old_baseline': baseline_image, 'core_inverse': core_image}
    result = {'name': name, 'coefficient_selection': 'Camera constraints and named LP objectives only; no film/scan sample selects coefficients or mask.',
              'scope': __doc__, 'exact_feasible_cells': int(accepted.sum()),
              'requested_component_non_strict_cells': int((edge['component_mask'] & ~source['support']).sum()),
              'requested_with_available_RGB_channel_cells': int(requested.sum()),
              'all_RGB_flagged_non_strict_cells_not_replaced': int((edge['component_mask'] & ~source['support'] & saturation.all(-1)).sum()),
              'infeasible_cells_retained_legacy_path': int((requested & ~accepted).sum()),
              'measurement_constrained_available_channels': int((accepted[..., None] & ~saturation).sum()),
              'still_estimated_saturated_channels_on_replaced_cells': int((accepted[..., None] & saturation).sum()),
              'FL2_relative_tolerance_values': rtol_values, 'camera_response': camera_response.tolist(),
              'records': records, 'replay': replay,
              'old_baseline': stats(baseline_lab), 'core_inverse': stats(core_lab), 'candidates': {},
              'limits': ['Ten spectra and CIE/inverse-decoder camera proxy, not measured G81 spectral sensitivity.',
                         'Core inverse is inherited from the previous same-scan fit; edge witnesses do not refit it.',
                         'Only saturated-component edge cells with at least one available RGB channel and exact feasible proxy are changed.',
                         'No brightness threshold expands the source mask into unsaturated lamp covers/background.',
                         'All witness spectra and unknown saturated channel values remain estimates.',
                         'Per-cell spectral mixtures allow mixed outgoing light; they do not identify a uniform intrinsic lamp SPD.',
                         'Finite library bounds are not physical confidence bounds or bounds over every real spectrum.',
                         'Same-frame fixed split is a prediction check; physical radius still uses uncalibrated nominal frame widths.',
                         'Red-return extrema are for modeled exposure, not extrema of final RGB or actual halation.']}
    # Constraining a sensor integral does not turn an inferred spectrum into a
    # measurement; retain channel-level provenance alongside every witness.
    output_arrays = {'accepted': accepted, 'requested': requested,
                     'observed_camera': edge['camera'], 'saturation': saturation,
                     'measurement_constrained_channels': accepted[..., None] & ~saturation,
                     'still_estimated_channels': accepted[..., None] & saturation,
                     'strict_core_still_estimated': source['support'],
                     'retained_legacy_component_cells': edge['component_mask'] & ~source['support'] & ~accepted,
                     'baseline': baseline_image,
                     'core_inverse': core_image, 'real': source['real'], 'points': points,
                     'baseline_lab': baseline_lab, 'core_inverse_lab': core_lab}
    for mode, coefficients in fields.items():
        direct = coefficients @ direct_response
        returning = coefficients @ return_response
        H = background + transport(engine, direct, returning)
        image, prediction, q = render(H)
        images[mode] = image
        result['candidates'][mode] = {'metrics': stats(prediction),
            'edge_total_input_Y': float(coefficients.sum()),
            'new_edge_direct_sum': direct.sum((0, 1)).tolist(),
            'old_edge_direct_sum': old_direct.sum((0, 1)).tolist(),
            'new_edge_return_source_sum': returning.sum((0, 1)).tolist(),
            'old_edge_return_source_sum': old_return.sum((0, 1)).tolist(),
            'maximum_available_camera_residual': float(np.max(abs(
                (coefficients @ camera_response - edge['camera'])[accepted[..., None] & ~saturation]), initial=0.))}
        output_arrays[mode] = image
        output_arrays[mode + '_lab'] = prediction
        output_arrays[mode + '_coefficients'] = coefficients
        output_arrays[mode + '_camera_proxy'] = coefficients @ camera_response
        output_arrays[mode + '_q'] = q
    (out / (name + '.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    np.savez_compressed(out / (name + '.npz'), **output_arrays)
    titles = [('real', '真实800T扫描'), ('old_baseline', '旧研究基准'), ('core_inverse', '仅失测核心反演'),
              ('minimum_input_Y', '边缘·最小输入Y'), ('minimum_return_R', '边缘·最小红层返光'),
              ('maximum_return_R', '边缘·最大红层返光')]
    font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 21)
    sheet = Image.new('RGB', (1536, 1160), (18, 18, 18)); draw = ImageDraw.Draw(sheet)
    for index, (key, title) in enumerate(titles):
        x, y = (index % 3) * 512, (index // 3) * 580
        picture = Image.fromarray(np.rint(srgb_encode(np.nan_to_num(images[key])) * 255).astype('uint8'))
        sheet.paste(picture, (x, y+60)); draw.text((x+10, y+12), title, font=font, fill='white')
    sheet.save(out / (name + '_comparison.jpg'), quality=96)
    print(json.dumps({key: result[key] for key in ['name', 'exact_feasible_cells', 'requested_with_available_RGB_channel_cells', 'replay', 'candidates']}, ensure_ascii=False), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--region', choices=['ceiling_lamp', 'middle_ceiling'], required=True)
    parser.add_argument('--edge-root', type=Path, default=Path('/workspace/filmlab-audit/lamp-edge-input-audit'))
    parser.add_argument('--root', type=Path, default=Path('/workspace/filmlab-audit/inverse-source-radiance'))
    parser.add_argument('--out', type=Path, default=Path('/workspace/filmlab-audit/lamp-edge-spectral-candidate'))
    args = parser.parse_args()
    run(args.region, args.edge_root, args.root, args.out)
