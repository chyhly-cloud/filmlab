"""Conditional metameric film-exposure bounds; never fit target photographs.

Exact ideal-camera RGB constraints with nonnegative radiance, optionally
bounded by a KNOWN illuminant and reflectance <=1. The film sensitivities
are the stored model, not independent measured CineStill 800T truth.
"""
from pathlib import Path
import json
import sys
import numpy as np
from scipy.optimize import linprog

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / 'tools/eval')]
from spectral_admissibility import nonnegative_spectrum


def exposure_bounds(rgb, camera, sensitivity, upper=None):
    scale = max(float(np.max(abs(rgb))), 1e-12)
    target = np.asarray(rgb) / scale
    bounds = (0, None) if upper is None else list(zip(np.zeros(len(camera)), upper / scale))
    extrema = []
    worst = 0.
    for j in range(3):
        values = []
        for sign in (1., -1.):
            sol = linprog(sign * sensitivity[:, j], A_eq=camera.T, b_eq=target,
                          bounds=bounds, method='highs',
                          options={'primal_feasibility_tolerance': 1e-9, 'dual_feasibility_tolerance': 1e-9})
            if not sol.success:
                return {'success': False, 'message': sol.message}
            worst = max(worst, float(np.max(abs(sol.x @ camera - target))))
            values.append(float(sol.x @ sensitivity[:, j] * scale))
        extrema.append(values)
    return {'success': True, 'H_layer_min_max': extrema, 'max_relative_RGB_closure': worst,
            'layer_interval_width_EV': [float(np.log2(hi / lo)) if lo > 0 else None for lo, hi in extrema],
            'limit': 'Per-layer extrema generally require DIFFERENT spectra; this box is not a joint reachable exposure set.'}


def main():
    import colour
    from filmlab import generation, lights, spectral, film
    from filmlab.materials import emulsion
    generation.use('next_sensor')
    out = Path('/workspace/filmlab-audit/exposure-floor-audit')
    camera = lights.camera_rgb(np.eye(81))
    stock = emulsion('kodak_vision3_500t').stock
    sensitivity = 10 ** stock.log_sensitivity
    cc = colour.SDS_COLOURCHECKERS['ColorChecker N Ohta']
    names = list(cc)
    reflectance = np.array([s.copy().align(colour.SpectralShape(380, 780, 5)).values for s in cc.values()])
    report = {'scope': __doc__, 'controlled': [], 'real_input_samples': [],
              'selection': 'All 24 patches under each of two fixed illuminants; real samples reused verbatim from prior input-only selection. No target scan RGB used.'}
    for name in ('BB3200', 'FL11'):
        illum = spectral.illuminant(name) if name == 'BB3200' else lights._sd(name)
        illum /= float(lights.camera_rgb(illum) @ np.array([.2126, .7152, .0722]))
        rgb = lights.camera_rgb(reflectance * illum)
        truth = (reflectance * illum) @ sensitivity
        old = film.exposure_response(stock, illum, method='linear_wp')(rgb)
        for j, patch in enumerate(names):
            row = {'light': name, 'patch': patch, 'old_has_nonpositive_layer': bool((old[j] <= 0).any()),
                   'conditional_reference_H': truth[j].tolist(), 'old_unfloored_H': old[j].tolist()}
            for label, upper in [('unknown_source_nonnegative_radiance', None), ('known_source_reflectance_0_to_1', illum)]:
                bound = exposure_bounds(rgb[j], camera, sensitivity, upper)
                assert bound['success'], (name, patch, bound)
                lo, hi = np.array(bound['H_layer_min_max']).T
                assert np.all(truth[j] >= lo - 1e-7) and np.all(truth[j] <= hi + 1e-7)
                row[label] = bound
            checks, candidate = nonnegative_spectrum(rgb[j], camera, smooth=True)
            row['unmeasured_smoothness_prior_checks'] = checks
            row['smooth_prior_layer_EV_error'] = None if candidate is None else np.log2((candidate @ sensitivity) / truth[j]).tolist()
            # Same predefined minimum-curvature prior, now on reflectance
            # under a genuinely known source. Do not infer this source from
            # the target film scan, and do not apply it to the real samples.
            from scipy.optimize import minimize, Bounds, LinearConstraint
            scale = max(float(np.max(abs(rgb[j]))), 1e-12)
            A = (camera * illum[:, None]).T / scale
            target = rgb[j] / scale
            seed = linprog(np.zeros(len(illum)), A_eq=A, b_eq=target,
                           bounds=(0, 1), method='highs')
            assert seed.success, (name, patch, seed.message)
            difference = np.diff(np.eye(len(illum)), n=2, axis=0)
            Q = difference.T @ difference
            sol = minimize(lambda r: float(r @ Q @ r), seed.x,
                           jac=lambda r: 2 * Q @ r, method='SLSQP',
                           bounds=Bounds(0, 1), constraints=[LinearConstraint(A, target, target)],
                           options={'ftol': 1e-11, 'maxiter': 1500})
            closure = float(np.max(abs(A @ sol.x - target)))
            success = bool(sol.success and closure < 1e-7)
            row['known_source_smooth_reflectance'] = {
                'success': success, 'message': str(sol.message), 'relative_RGB_closure': closure,
                'layer_EV_error': np.log2(((sol.x * illum) @ sensitivity) / truth[j]).tolist() if success else None}
            report['controlled'].append(row)
    prior = json.loads((out / 'spectral_admissibility.json').read_text())
    for scene in prior['real_input_samples']:
        rows = []
        for sample in scene['samples']:
            bound = exposure_bounds(np.array(sample['input_RGB']), camera, sensitivity)
            assert bound['success'], (scene['scene'], sample['yx'], bound)
            rows.append({'yx': sample['yx'], **bound})
        report['real_input_samples'].append({'scene': scene['scene'], 'samples': rows})
    controlled = report['controlled']
    summary = {'controlled_cases': len(controlled), 'cases_old_has_nonpositive_layer': sum(r['old_has_nonpositive_layer'] for r in controlled),
               'all_reference_H_inside_both_bounds': True,
               'max_LP_RGB_closure': max(r[k]['max_relative_RGB_closure'] for r in controlled for k in ['unknown_source_nonnegative_radiance', 'known_source_reflectance_0_to_1'])}
    for label, rows in [('old_nonpositive', [r for r in controlled if r['old_has_nonpositive_layer']]),
                        ('old_positive_controls', [r for r in controlled if not r['old_has_nonpositive_layer']])]:
        e = np.abs(np.array([r['smooth_prior_layer_EV_error'] for r in rows if r['smooth_prior_layer_EV_error'] is not None]))
        summary[label] = {'cases': len(rows), 'smooth_successes': sum(r['smooth_prior_layer_EV_error'] is not None for r in rows),
                          'smooth_abs_EV_p50_p95_max': np.percentile(e, [50, 95, 100]).tolist()}
        bounded = [r['known_source_smooth_reflectance']['layer_EV_error'] for r in rows
                   if r['known_source_smooth_reflectance']['success']]
        summary[label]['known_source_smooth_successes'] = len(bounded)
        summary[label]['known_source_smooth_abs_EV_p50_p95_max'] = np.percentile(np.abs(bounded), [50, 95, 100]).tolist() if bounded else None
        if label == 'old_positive_controls':
            old_errors = np.abs([np.log2(np.array(r['old_unfloored_H']) / r['conditional_reference_H']) for r in rows])
            summary[label]['old_abs_EV_p50_p95_max'] = np.percentile(old_errors, [50, 95, 100]).tolist()
    for k in ['unknown_source_nonnegative_radiance', 'known_source_reflectance_0_to_1']:
        summary[k + '_layer_width_EV_p50_p95_max'] = np.percentile(np.array([r[k]['layer_interval_width_EV'] for r in controlled]), [50, 95, 100]).tolist()
    for scene in report['real_input_samples']:
        summary[scene['scene']] = {'samples': len(scene['samples']), 'layer_width_EV_p50_p95_max': np.percentile(np.array([r['layer_interval_width_EV'] for r in scene['samples']]), [50, 95, 100]).tolist()}
    report['summary'] = summary
    (out / 'metameric_exposure_bounds.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
