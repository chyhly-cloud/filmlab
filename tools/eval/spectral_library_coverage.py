"""Predeclared public lamp-library coverage in fixed FilmLab 6D response space.

No scene, RAW, film scan, camera SSF or material coefficient is fitted/read.
Use all bundled colour illuminants/light sources with full 380..780nm coverage,
plus all four already-shipped blackbodies. No synthetic delta wavelengths.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from filmlab import generation, lights
from filmlab.engine import Engine, Params
from filmlab.incident import spectral_responses
from filmlab.recipes import recipe

LUMA = np.array([.2126, .7152, .0722])
WAVELENGTHS = np.arange(380, 781, 5)
OLD_NAMES = ['D65', '2700K 白炽 / 卤素', '3300K 白炽 / 卤素', 'FL2', 'FL11',
             'LED-B1', 'LED-B3', 'LED-B5', 'HP1', 'Metal Halide']
MEMBERSHIP_TOLERANCE = 1e-7


def mixture_distance(basis, target, *, convex):
    """Chebyshev distance in already scaled response coordinates.

    Same single nonnegative coefficient vector drives all six signals.
    Convex mode fixes input Y by sum(c)=1; cone mode permits free amplitude.
    """
    b = np.asarray(basis, float)
    y = np.asarray(target, float)
    constraints = np.vstack([np.column_stack([b.T, -np.ones(y.size)]),
                             np.column_stack([-b.T, -np.ones(y.size)])])
    objective = np.r_[np.zeros(len(b)), 1.]
    eq = np.r_[np.ones(len(b)), 0.][None] if convex else None
    fit = linprog(objective, A_ub=constraints, b_ub=np.r_[y, -y],
                  A_eq=eq, b_eq=np.array([1.]) if convex else None,
                  bounds=[(0, None)] * (len(b) + 1), method='highs')
    if not fit.success:
        raise RuntimeError(fit.message)
    coefficient = fit.x[:-1]
    distance = float(np.max(abs(coefficient @ b - y)))
    return {'minimum_scaled_Linf': distance, 'LP_objective': float(fit.fun),
            'inside_at_declared_tolerance': bool(distance <= MEMBERSHIP_TOLERANCE),
            'coefficient_sum': float(coefficient.sum()), 'coefficients': coefficient.tolist()}


def rank_report(points):
    singular = np.linalg.svd(points, compute_uv=False)
    affine = np.linalg.svd(points - points.mean(0), compute_uv=False)
    return {'linear_rank': int(np.count_nonzero(singular > singular[0] * 1e-10)),
            'affine_rank': int(np.count_nonzero(affine > affine[0] * 1e-10)),
            'linear_singular_values': singular.tolist(), 'affine_singular_values': affine.tolist(),
            'relative_rank_threshold': 1e-10}


def library():
    import colour
    from colour.colorimetry.datasets.light_sources import sds as ls
    entries, spectra, excluded = [], [], []
    groupings = [(n, getattr(ls, n)) for n in ('SDS_LIGHT_SOURCES_RIT', 'SDS_LIGHT_SOURCES_NIST_TRADITIONAL',
                  'SDS_LIGHT_SOURCES_NIST_LED', 'SDS_LIGHT_SOURCES_NIST_PHILIPS', 'SDS_LIGHT_SOURCES_COMMON')]
    for category, data in [('standard_illuminant', colour.SDS_ILLUMINANTS),
                           ('named_light_source', colour.SDS_LIGHT_SOURCES)]:
        for name, sd in data.items():
            bounds = [float(sd.wavelengths[0]), float(sd.wavelengths[-1])]
            item = {'name': name, 'category': category, 'source_coverage_nm': bounds,
                    'source_samples': int(len(sd.wavelengths)),
                    'source_group': ('CIE_or_ISO_reference' if category == 'standard_illuminant' else
                                     next(n for n, d in groupings if name in d)),
                    'is_current_project_member': any(name == n for c in lights.classes() for n, _ in c.members)}
            if bounds[0] > 380 or bounds[1] < 780:
                excluded.append({**item, 'reason': 'No full engine wavelength coverage; no zero/end-point extrapolation allowed.'})
                continue
            values = np.interp(WAVELENGTHS, sd.wavelengths, sd.values)
            if np.any(values < 0) or not np.isfinite(values).all():
                raise ValueError('Public source is not finite nonnegative: ' + name)
            entries.append(item)
            spectra.append(values)
    # All existing project blackbodies, not a temperature search.
    for c in lights.classes():
        for name, values in c.members:
            if '白炽 / 卤素' not in name:
                continue
            entries.append({'name': name, 'category': 'project_theoretical_blackbody',
                            'source_group': 'existing_project_Planck_model', 'source_coverage_nm': [380, 780],
                            'source_samples': 81, 'is_current_project_member': True})
            spectra.append(values)
    spectra = np.asarray(spectra, float)
    y = lights.camera_rgb(spectra) @ LUMA
    if np.any(y <= 0):
        raise ValueError('Input Y must be positive.')
    spectra = spectra / y[:, None]
    # Verify all original ten spectra retain the exact pre-existing library.
    members = {n: s for c in lights.classes() for n, s in c.members}
    old = np.array([members[n] for n in OLD_NAMES])
    old /= (lights.camera_rgb(old) @ LUMA)[:, None]
    indices = [next(i for i, e in enumerate(entries) if e['name'] == name) for name in OLD_NAMES]
    reproduction = float(np.max(abs(spectra[indices] - old)))
    if reproduction > 1e-10:
        raise ValueError('Linear resampling changed the original library: ' + str(reproduction))
    for entry, sp in zip(entries, spectra):
        entry['normalized_spectrum_sha256'] = hashlib.sha256(np.round(sp, 12).tobytes()).hexdigest()
    return entries, spectra, excluded, indices, reproduction


def ratio_arrays(response):
    d, r = response[:, :3], response[:, 3:]
    return {'return_R_over_G': r[:, 0] / r[:, 1],
            'direct_R_over_return_R': d[:, 0] / r[:, 0],
            'direct_G_over_return_G': d[:, 1] / r[:, 1],
            'direct_B_over_return_B': d[:, 2] / r[:, 2],
            'sum_direct_over_sum_return': d.sum(1) / r.sum(1)}


def run(out):
    import colour
    normalization_wb = lights.WB_REFERENCE
    entries, spectra, excluded, indices, reproduction = library()
    normalization_error = float(np.max(abs(lights.camera_rgb(spectra) @ LUMA - 1)))
    generation.use('next_sensor')
    engine = Engine(Params(**{**recipe('cinestill_800t'), 'grain': 0, 'output_conditioning': None}))
    direct, returning = spectral_responses(engine)
    response = np.column_stack([spectra @ direct, spectra @ returning]) * 2 ** engine.p.exposure_ev
    old = response[indices]
    scale = old.max(0)
    standardized, old_scaled = response / scale, old / scale
    hull_old, hull_all = ConvexHull(old_scaled), ConvexHull(standardized)
    rows = []
    for i, entry in enumerate(entries):
        rows.append({**entry, 'response6': response[i].tolist(),
                     'old_convex_hull': mixture_distance(old_scaled, standardized[i], convex=True),
                     'old_nonnegative_cone': mixture_distance(old_scaled, standardized[i], convex=False)})
    ratio_results = {}
    for name, values in ratio_arrays(response).items():
        lower, upper = float(values[indices].min()), float(values[indices].max())
        outside = (values < lower * (1 - MEMBERSHIP_TOLERANCE)) | (values > upper * (1 + MEMBERSHIP_TOLERANCE))
        ratio_results[name] = {'original10_min_max': [lower, upper],
                              'original_min_source': OLD_NAMES[int(np.argmin(values[indices]))],
                              'original_max_source': OLD_NAMES[int(np.argmax(values[indices]))],
                              'expanded_min_max': [float(values.min()), float(values.max())],
                              'expanded_min_source': entries[int(np.argmin(values))]['name'],
                              'expanded_max_source': entries[int(np.argmax(values))]['name'],
                              'outside_original_range': [{**entries[i], 'value': float(values[i])} for i in np.where(outside)[0]]}
    categories = sorted({e['category'] for e in entries})
    counts = {category: {'members': sum(e['category'] == category for e in entries),
                        'outside_old_hull': sum(r['category'] == category and not r['old_convex_hull']['inside_at_declared_tolerance'] for r in rows),
                        'outside_old_cone': sum(r['category'] == category and not r['old_nonnegative_cone']['inside_at_declared_tolerance'] for r in rows)}
              for category in categories}
    report = {'scope': __doc__, 'colour_version': colour.__version__,
              'universe_declared_before_response_scoring': 'All bundled colour SDS_ILLUMINANTS and SDS_LIGHT_SOURCES with complete 380..780nm coverage, plus all existing project blackbodies.',
              'interpolation': 'Piecewise linear, nonnegative, no extrapolation or zero-padding; incomplete sources excluded.',
              'normalization': 'Each spectrum divided by lights.camera_rgb(spectrum) @ [.2126,.7152,.0722] before generation.use(next_sensor); see normalization_WB_REFERENCE.',
              'normalization_WB_REFERENCE': normalization_wb,
              'initialization_order': ['Capture initial WB_REFERENCE and normalize the spectral library',
                                       'generation.use(next_sensor)', 'Construct nominal Engine with default roll/frame',
                                       'Integrate fixed nominal direct/return responses'],
              'response_instance_scope': 'Nominal instance only, not corridor scene response6. Re-normalize shapes and re-integrate with the actual scene engine before using them there.',
              'engine': {'generation': 'next_sensor', 'recipe': 'cinestill_800t', 'exposure_ev': float(engine.p.exposure_ev),
                         'WB_REFERENCE_after_generation': lights.WB_REFERENCE, 'roll': int(engine.p.roll), 'frame': int(engine.p.frame),
                         'lens_filter': engine.p.lens_filter},
              'response_axes': ['direct_R', 'direct_G', 'direct_B', 'return_R', 'return_G', 'return_B'],
              'original10_names': OLD_NAMES, 'original10_indices': indices,
              'original10_linear_resampling_reproduction_max': reproduction,
              'coordinate_scale_for_geometry': scale.tolist(), 'membership_tolerance_scaled_Linf': MEMBERSHIP_TOLERANCE,
              'normalized_Y_max_error': normalization_error,
              'post_generation_WB_Y_deviation': float(np.max(abs(lights.camera_rgb(spectra) @ LUMA - 1))),
              'source_counts': counts, 'retained_count': len(entries), 'excluded': excluded,
              'unique_normalized_shape_count_rounded12': len({e['normalized_spectrum_sha256'] for e in entries}),
              'original10_rank': rank_report(old_scaled), 'expanded_rank': rank_report(standardized),
              'convex_hulls': {'original_vertices': len(hull_old.vertices), 'expanded_vertices': len(hull_all.vertices),
                               'original_scaled_6D_volume': float(hull_old.volume),
                               'expanded_scaled_6D_volume': float(hull_all.volume),
                               'expanded_over_original_volume': float(hull_all.volume / hull_old.volume)},
              'ratios': ratio_results, 'sources': rows,
              'limits': ['These are fixed current-model exposure responses, not independently measured 800T material responses.',
                         'Standard illuminants, named public lamp tables and theoretical Planck spectra are not equally strong evidence for a specific real lamp.',
                         'No source is selected using real scans; no actual lamp is identified by this audit.',
                         'A linear or affine rank of six does not imply convex hull or nonnegative cone completeness.',
                         'Cone membership allows free input-Y amplitude; convex membership fixes Y=1.',
                         'Five-nanometre binning and finite source coverage do not prove completeness for all physical spectra.',
                         'No new SSF, material, layer, scanner or output parameter is fitted.']}
    np.savez_compressed(out / 'normalized_spectra.npz', names=np.array([e['name'] for e in entries]),
                        categories=np.array([e['category'] for e in entries]), spectra=spectra,
                        wavelengths_nm=WAVELENGTHS, response6=response,
                        original10_indices=indices, coordinate_scale=scale,
                        direct_response_per_wavelength=direct, return_response_per_wavelength=returning,
                        original_hull_equations_scaled=hull_old.equations,
                        expanded_hull_equations_scaled=hull_all.equations)
    (out / 'results.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({'source_counts': counts, 'ratios': {k: {a: v[a] for a in ('original10_min_max', 'expanded_min_max')} for k, v in ratio_results.items()},
                      'hull_volume_ratio': report['convex_hulls']['expanded_over_original_volume']}, ensure_ascii=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('/workspace/filmlab-audit/spectral-library-coverage'))
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True); run(args.out)


if __name__ == '__main__':
    main()
