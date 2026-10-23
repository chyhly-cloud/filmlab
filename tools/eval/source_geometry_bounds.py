"""Conditional bounds on returned-exposure ratios for unknown source geometry.

No image fitting. For a fixed nonnegative source support and fixed kernel,
far/near return is a weighted average of per-source-pixel ratios. Bounds below
allow arbitrary nonnegative spatial redistribution, rather than a few templates.
They concern modeled linear layer exposure, never RGB or measured real halation.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
from scipy.signal import fftconvolve

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / 'tools/eval')]


def adjoint_response(receiver_weights, kernel):
    """Response per source cell, with the same zero-padded open frame."""
    weights, k = np.asarray(receiver_weights, float), np.asarray(kernel, float)
    if weights.ndim != 2 or k.ndim != 2 or any(n % 2 == 0 for n in k.shape):
        raise ValueError('2D weights and odd kernel dimensions required')
    if not np.isfinite(weights).all() or not np.isfinite(k).all():
        raise ValueError('finite weights and kernel required')
    if np.any(weights < 0) or np.any(k < 0):
        raise ValueError('nonnegative weights and kernel required')
    result = fftconvolve(weights, k[::-1, ::-1], mode='same')
    if result.min() < -1e-12 * max(1., float(result.max())):
        raise ValueError('negative adjoint response beyond roundoff')
    return np.maximum(result, 0.)


def ratio_bounds(near, far, *, floor=None, total=None):
    """Exact finite-support extrema, with optional lower floors and total flux.

    All source-cell near responses must be positive. With no total specified,
    these are unrestricted nonnegative-source bounds. With floors l and fixed
    sum M, each extremum is attained at l+(M-sum(l))*e_j. No fitted source,
    finite amplitude upper bound or photon calibration is inferred here.
    """
    n, f = np.asarray(near, float), np.asarray(far, float)
    if n.ndim != 1 or n.size == 0 or f.shape != n.shape:
        raise ValueError('matching nonempty source response vectors required')
    if not np.isfinite(n).all() or not np.isfinite(f).all() or np.any(n <= 0) or np.any(f < 0):
        raise ValueError('positive near and nonnegative far responses required')
    if total is None:
        if floor is not None:
            raise ValueError('floor-only unbounded problem must be stated separately')
        values = f / n
        remainder = None
    else:
        l = np.zeros_like(n) if floor is None else np.asarray(floor, float)
        if l.shape != n.shape or not np.isfinite(l).all() or np.any(l < 0):
            raise ValueError('finite nonnegative matching floors required')
        if not np.isfinite(total) or total < l.sum() or total <= 0:
            raise ValueError('positive total must satisfy all source floors')
        remainder = float(total - l.sum())
        values = (l @ f + remainder * f) / (l @ n + remainder * n)
    lo, hi = int(np.argmin(values)), int(np.argmax(values))
    return {'minimum': float(values[lo]), 'maximum': float(values[hi]),
            'minimum_source_index': lo, 'maximum_source_index': hi,
            'fixed_total_remainder': remainder}


def run(root, out):
    from filmlab import film
    from inverse_source_radiance import engine_for, observation_operator
    e, meta = engine_for()
    report = {'scope': __doc__, 'regions': [], 'limits': [
        'Fixed existing conditional kernel, source support and source spectrum; not actual800T material bounds.',
        'Linear modeled returned layer exposure is not final RGB and is not an isolated measured real halo.',
        'The most extreme allowed source maps need not describe a realizable lamp or independently measured camera PSF.',
        'Camera floors use the existing CIE/inverse-decoder proxy, not measured G81 spectral sensitivity.',
        'A narrow modeled geometry range cannot by itself identify a scanner, material or background error.',
    ]}
    for name in ['ceiling_lamp', 'middle_ceiling']:
        a = np.load(root / (name + '_source.npz'))
        fitted = np.load(root / (name + '_fit.npz'))
        info = json.loads((root / (name + '_fit.json')).read_text())
        points = fitted['points']
        near_mask = (fitted['train'] | fitted['heldout'])[tuple(points.T)]
        far_mask = fitted['outer'][tuple(points.T)]
        sigma = 25 / meta['scan_um_per_pixel']
        P = observation_operator(a['support'].shape, a['matrix'], points, sigma)
        w_near = np.asarray(P.T @ (near_mask / near_mask.sum())).reshape(a['support'].shape)
        w_far = np.asarray(P.T @ (far_mask / far_mask.sum())).reshape(a['support'].shape)
        coefficients = fitted['coefficients']
        amplitude = float(coefficients.sum())
        camera = coefficients @ a['camera_response'] / amplitude
        if np.any(camera <= 0):
            raise ValueError('Camera proxy must be positive for source floor interpretation')
        source_floor = np.max(a['source_camera'] / camera[None, :], axis=1)
        source = amplitude * a['source_profile']
        if np.any(source < source_floor):
            raise ValueError('Frozen inverse candidate violates exact camera-proxy floor')
        support = a['support']
        rho = film.bounce_retention(e.e.stock, e.v)
        row = {'name': name, 'source_cells': int(support.sum()),
               'source_total_input_Y': float(source.sum()),
               'camera_proxy_floor_total': float(source_floor.sum()), 'layers': {}}
        for c, channel in enumerate('RGB'):
            kernel = (film.multi_bounce_kernel(e.px_um, e.v.base_um, e.v.forward, rho[c])
                      if rho[c] > .01 else film.halation_kernel(e.px_um, e.v.base_um, forward=e.v.forward))
            n = adjoint_response(w_near, kernel)[support]
            f = adjoint_response(w_far, kernel)[support]
            original = float((source @ f) / (source @ n))
            total_bound = ratio_bounds(n, f, floor=source_floor, total=source.sum())
            total_bound['relative_to_frozen_profile'] = [total_bound[k] / original for k in ['minimum', 'maximum']]
            row['layers'][channel] = {
                'frozen_profile_far_over_near': original,
                'arbitrary_nonnegative_profile': ratio_bounds(n, f),
                'same_total_with_camera_proxy_floors': total_bound,
            }
        report['regions'].append(row)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/workspace/filmlab-audit/inverse-source-radiance'))
    p.add_argument('--out', type=Path, default=Path('/workspace/filmlab-audit/source-geometry-bounds'))
    args = p.parse_args()
    result = run(args.root, args.out)
    print(json.dumps(result['regions'], indent=2))
